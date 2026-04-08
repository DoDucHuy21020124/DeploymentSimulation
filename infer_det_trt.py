import torch
import torch.nn as nn
import tensorrt as trt
from collections import OrderedDict, namedtuple
import numpy as np
from pathlib import Path
import cv2
from typing import List, Optional, Tuple, Union
import os

from trt_backend import TensorRTBackend

class TensorRTBackend(nn.Module):
    """NVIDIA TensorRT inference backend for GPU-accelerated deployment.

    Loads and runs inference with NVIDIA TensorRT serialized engines (.engine files). Supports both TensorRT 7-9 and
    TensorRT 10+ APIs, dynamic input shapes, FP16 precision, and DLA core offloading.
    """

    def load_model(self, weight: Union[str, Path], device: str) -> None:
        """Load an NVIDIA TensorRT engine from a serialized .engine file.

        Args:
            weight (str | Path): Path to the .engine file with optional embedded metadata.
        """

        # if self.device.type == "cpu":
        #     self.device = torch.device("cuda:0")
        device = torch.device(device)

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        # Read engine file
        with open(weight, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        try:
            self.context = engine.create_execution_context()
        except Exception as e:
            print("TensorRT model exported with a different version than expected\n")
            raise e

        # Setup bindings
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False
        self.is_trt10 = not hasattr(engine, "num_bindings")
        num = range(engine.num_io_tensors) if self.is_trt10 else range(engine.num_bindings)

        for i in num:
            if self.is_trt10:
                name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(engine.get_tensor_shape(name))
                profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = engine.get_binding_name(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                shape = tuple(engine.get_binding_shape(i))
                profile_shape = tuple(engine.get_profile_shape(0, i)[1]) if is_input else None

            if is_input:
                if -1 in shape:
                    self.dynamic = True
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape = (
                tuple(self.context.get_tensor_shape(name))
                if self.is_trt10
                else tuple(self.context.get_binding_shape(i))
            )
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            # im = torch.from_numpy(np.empty(shape, dtype=dtype))
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.model = engine

    def forward(self, im: torch.Tensor) -> List[torch.Tensor]:
        """Run NVIDIA TensorRT inference with dynamic shape handling.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format on the CUDA device.

        Returns:
            (list[torch.Tensor]): Model predictions as a list of tensors on the CUDA device.
        """
        if self.dynamic and im.shape != self.bindings["images"].shape:
            if self.is_trt10:
                self.context.set_input_shape("images", im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
            else:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"

        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]


def letterbox(
    image: np.ndarray,
    target_shape: Optional[Tuple[int, int]] = None,
    pad_value: int = 0,
    stride: int = 32,
    auto: bool = False,
) -> Tuple[np.ndarray, float, Tuple[int, int, int, int], Tuple[int, int]]:
    """Resize+pad image while keeping aspect ratio (letterbox).

    Args:
        image (np.ndarray): Input image in HWC BGR format (OpenCV default).
        target_shape (tuple[int, int] | None): Output (height, width). If None, use model input shape.
        pad_value (int): Padding color value.
        stride (int): Stride-aligned padding when auto=True.
        auto (bool): If True, adjust padding to be stride-multiple.

    Returns:
        tuple[np.ndarray, float, tuple[int, int, int, int], tuple[int, int]]:
            - Preprocessed letterboxed image in HWC BGR format.
            - Resize ratio.
            - Padding (left, top, right, bottom).
            - Resized shape before padding (resized_h, resized_w).
    """
    if image is None:
        raise ValueError("image is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC BGR image with 3 channels, got shape={image.shape}")

    new_h, new_w = int(target_shape[0]), int(target_shape[1])
    h, w = image.shape[:2]

    ratio = min(new_h / h, new_w / w)
    resized_w, resized_h = int(round(w * ratio)), int(round(h * ratio))

    dw, dh = new_w - resized_w, new_h - resized_h
    if auto:
        dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2

    if (w, h) != (resized_w, resized_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )

    return image, ratio, (left, top, right, bottom), (resized_h, resized_w)


def scale_boxes_to_original(
    boxes: np.ndarray, ratio: float, pad: Tuple[int, int, int, int], orig_shape: Tuple[int, int]
) -> np.ndarray:
    left, top, _, _ = pad
    h0, w0 = orig_shape
    b = boxes.copy()
    b[:, [0, 2]] -= left
    b[:, [1, 3]] -= top
    b /= ratio
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w0 - 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h0 - 1)
    return b


def xywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def postprocess_detection_output(
    pred: np.ndarray,
    input_hw: Tuple[int, int],
    conf_thr: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inp_h, inp_w = input_hw

    if pred.ndim == 3:
        pred = pred[0]

    if pred.ndim != 2:
        raise RuntimeError(f"Unsupported detection output shape: {pred.shape}")

    # Common YOLO TensorRT raw output: (84, 8400) -> transpose to (8400, 84)
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 128 and pred.shape[1] > 512:
        pred = pred.T

    # Decoded/NMS-export format: (N, >=6) as [x1, y1, x2, y2, conf, cls, ...]
    if pred.shape[1] >= 6 and pred.shape[1] <= 16:
        det = pred[pred[:, 4] > conf_thr]
        if det.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        boxes = det[:, :4].astype(np.float32)
        scores = det[:, 4].astype(np.float32)
        cls_ids = det[:, 5].astype(np.int32)

        # If boxes look normalized, convert to input pixels
        if boxes.max() <= 1.5:
            boxes[:, [0, 2]] *= inp_w
            boxes[:, [1, 3]] *= inp_h

        return boxes, scores, cls_ids

    # Raw YOLO format: (N, 4 + num_classes)
    if pred.shape[1] <= 5:
        raise RuntimeError(f"Detection tensor has too few channels: {pred.shape}")

    boxes = xywh_to_xyxy(pred[:, :4].astype(np.float32))
    cls_scores = pred[:, 4:].astype(np.float32)
    cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    valid = scores > conf_thr
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]
    if boxes.shape[0] == 0:
        return boxes, scores, cls_ids

    return boxes, scores, cls_ids


if __name__ == '__main__':
    device = 'cuda:1'
    weight = 'weights/yolov8n.engine'
    image_file_path = 'data/ship/ship1.jpg'
    output_folder_path = './output/detection'
    os.makedirs(output_folder_path, exist_ok=True)
    image_name = os.path.basename(image_file_path)
    output_image_path = os.path.join(output_folder_path, image_name)

    image = cv2.imread(image_file_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_file_path}")

    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_h, input_w = 480, 640
    image_resized, ratio, pad, _ = letterbox(image_rgb, target_shape=(input_h, input_w))

    image_x = torch.from_numpy(np.ascontiguousarray(image_resized.transpose(2, 0, 1))).float() / 255.0
    image_x = image_x.to(device).unsqueeze(0)

    model = TensorRTBackend()
    model.load_model(weight, device)

    outputs = model(image_x)
    if len(outputs) == 0:
        raise RuntimeError("No outputs from TensorRT detection engine.")

    det_out = outputs[0].detach().float().cpu().numpy()
    boxes_inp, scores, cls_ids = postprocess_detection_output(
        det_out,
        input_hw=(input_h, input_w),
        conf_thr=0.25,
    )

    vis = image.copy()
    if boxes_inp.shape[0] > 0:
        boxes_orig = scale_boxes_to_original(boxes_inp, ratio, pad, (orig_h, orig_w)).astype(np.int32)
        rng = np.random.default_rng(42)
        class_colors = {}

        for box, score, cls_id in zip(boxes_orig, scores, cls_ids):
            if cls_id not in class_colors:
                class_colors[cls_id] = tuple(int(v) for v in rng.integers(64, 256, size=3))
            color = class_colors[cls_id]

            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"cls {cls_id} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(0, y1 - th - 6)
            cv2.rectangle(vis, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
            cv2.putText(vis, label, (x1 + 4, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_image_path, vis)
    print(f"Detections: {boxes_inp.shape[0]}")
    print(f"Saved visualization to: {output_image_path}")
