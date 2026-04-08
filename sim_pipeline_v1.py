#!/usr/bin/env python3
import argparse
import json
import queue
import signal
import threading
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from trt_backend import TensorRTBackend

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid image folder: {folder}")
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise ValueError(f"No images found in folder: {folder}")
    return images


def circular_twenty(paths: List[Path]) -> List[Path]:
    if len(paths) >= 20:
        return paths[:20]
    return [paths[i % len(paths)] for i in range(20)]


def letterbox(image: np.ndarray, target_h: int, target_w: int, pad_value: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    r = min(target_h / float(h), target_w / float(w))
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (target_h - nh) // 2
    bottom = target_h - nh - top
    left = (target_w - nw) // 2
    right = target_w - nw - left
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )


class TrtModelWorker:
    def __init__(self, engine_path: str, device: str, batch_size: int):
        # print('self.device_id', self.device_id)
        self.device = device
        # print('self.device', self.device)
        self.batch_size = max(1, int(batch_size))
        self.model = TensorRTBackend()
        self.model.load_model(engine_path, self.device)
        self.input_shape = self.model.bindings["images"].shape
        print('self.input_shape', self.input_shape)
        self.input_h = int(self.input_shape[2])
        self.input_w = int(self.input_shape[3])

        configured_batch = int(self.input_shape[0])
        self.max_batch = configured_batch if configured_batch > 0 else self.batch_size
        self.effective_batch = min(self.batch_size, self.max_batch)

    def infer(self, images: List[np.ndarray]) -> None:
        if not images:
            return

        pre = []
        for img in images:
            lb = letterbox(img, self.input_h, self.input_w)
            rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
            pre.append(chw)

        batch = np.stack(pre, axis=0)
        t = torch.from_numpy(batch).to(self.device)
        t = t.half() if self.model.fp16 else t.float()

        with torch.no_grad():
            _ = self.model(t)
        torch.cuda.synchronize(self.device)


def source_loop(seed_paths: List[Path], out_q: queue.Queue, fps: float, stop_event: threading.Event) -> None:
    idx = 0
    interval = 0.0 if fps <= 0 else 1.0 / fps
    next_deadline = time.perf_counter()

    while not stop_event.is_set():
        img = cv2.imread(str(seed_paths[idx]))
        if img is not None:
            while not stop_event.is_set():
                try:
                    out_q.put(img, timeout=0.2)
                    break
                except queue.Full:
                    continue

        idx = (idx + 1) % len(seed_paths)
        if interval > 0:
            next_deadline += interval
            now = time.perf_counter()
            if next_deadline > now:
                time.sleep(next_deadline - now)
            else:
                next_deadline = now


def worker_loop(model: TrtModelWorker, in_q: queue.Queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        batch = []
        for _ in range(model.effective_batch):
            try:
                item = in_q.get(timeout=0.2)
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            continue

        try:
            model.infer(batch)
        except Exception as e:
            print(f"[worker] inference error on device {model.device_id}: {e}")
            stop_event.set()
        finally:
            for _ in batch:
                in_q.task_done()


def logger_loop(
    det_q: queue.Queue,
    seg_q: queue.Queue,
    log_q: queue.Queue,
    log_jsonl: Path,
    stop_event: threading.Event,
) -> None:
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with log_jsonl.open("a", encoding="utf-8") as f:
        while not stop_event.is_set():
            rec = {
                "det_buffer_size": det_q.qsize(),
                "seg_buffer_size": seg_q.qsize(),
            }
            line = json.dumps(rec)
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()

            while not stop_event.is_set():
                try:
                    log_q.put(rec, timeout=0.2)
                    break
                except queue.Full:
                    continue

            for _ in range(10):
                if stop_event.is_set():
                    break
                time.sleep(0.1)


def main() -> None:
    ap = argparse.ArgumentParser("TensorRT simulation pipeline v1 (single worker per task)")
    ap.add_argument("--det-folder", type=str, required=True)
    ap.add_argument("--seg-folder", type=str, required=True)
    ap.add_argument("--det-engine", type=str, required=True)
    ap.add_argument("--seg-engine", type=str, required=True)
    ap.add_argument("--det-batch-size", type=int, default=1)
    ap.add_argument("--seg-batch-size", type=int, default=1)
    ap.add_argument("--det-source-fps", type=float, default=10.0)
    ap.add_argument("--seg-source-fps", type=float, default=10.0)
    ap.add_argument("--det-buffer-capacity", type=int, default=64)
    ap.add_argument("--seg-buffer-capacity", type=int, default=64)
    ap.add_argument("--log-buffer-capacity", type=int, default=256)
    ap.add_argument("--log-jsonl", type=str, default="output/sim_v1_log.jsonl")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    det_seed = circular_twenty(list_images(Path(args.det_folder)))
    seg_seed = circular_twenty(list_images(Path(args.seg_folder)))

    det_q: queue.Queue = queue.Queue(maxsize=max(1, args.det_buffer_capacity))
    seg_q: queue.Queue = queue.Queue(maxsize=max(1, args.seg_buffer_capacity))
    log_q: queue.Queue = queue.Queue(maxsize=max(1, args.log_buffer_capacity))

    stop_event = threading.Event()

    def handle_stop(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    # det_device = args.device
    # seg_device = args.device
    device = f"cuda:{args.device}"

    # x = torch.rand((1, 3, 480, 640)).to(f"cuda:{device}")
    torch.cuda.set_device(device)


    det_model = TrtModelWorker(args.det_engine, device, args.det_batch_size)
    seg_model = TrtModelWorker(args.seg_engine, device, args.seg_batch_size)

    threads = [
        threading.Thread(target=source_loop, args=(det_seed, det_q, args.det_source_fps, stop_event), daemon=True),
        threading.Thread(target=source_loop, args=(seg_seed, seg_q, args.seg_source_fps, stop_event), daemon=True),
        threading.Thread(target=worker_loop, args=(det_model, det_q, stop_event), daemon=True),
        threading.Thread(target=worker_loop, args=(seg_model, seg_q, stop_event), daemon=True),
        threading.Thread(target=logger_loop, args=(det_q, seg_q, log_q, Path(args.log_jsonl), stop_event), daemon=True),
    ]

    for t in threads:
        t.start()

    while not stop_event.is_set():
        time.sleep(0.2)


if __name__ == "__main__":
    main()
