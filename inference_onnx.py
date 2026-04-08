import argparse
import json
import os, os.path as osp
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add kapao/ to path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import time

import onnx
import onnxruntime as ort

def load_image(path: str, img_size: int):
    im = cv2.imread(path)  # BGR
    assert im is not None, 'Image Not Found ' + path
    h0, w0 = im.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

def save_image(output_folder: str, image_name: str, image: np.array):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(image_path, image)

@torch.no_grad()
def inference(opt):
    
    # device = select_device(opt.device)

    # Load model
    # model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # gs = max(int(model.stride.max()), 32)
    gs = 64
    imgsz = check_img_size(opt.imgsz, s=gs)  # check image size
    
    model = ort.InferenceSession(opt.weights)


    # Half
    # half = opt.half
    # half &= device.type != 'cpu'  # half precision only supported on CUDA
    # if half:
    #     print("\n<<< Evaluating with half precision >>>\n")
    #     model.half()
    # else:
    #     print("\n<<< Evaluating WITHOUT half precision >>>\n")

    # Configure
    # model.eval()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='yolov5s6.pt')
    parser.add_argument('--image_path', help='image file path')
    parser.add_argument('--output_folder', default='.', help='saving folder')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1536, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--num_offsets', type=int, default=2, help='number of offsets')
    parser.add_argument('--num_lmks', type=int, default=5, help='number of landmarks')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--conf_thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--conf_thres_part', type=float, default=0.01, help='confidence threshold for part')
    parser.add_argument('--iou_thres_part', type=float, default=0.5, help='NMS IoU threshold for part')
    parser.add_argument('--th_match', type=float, default=0.96, help='cut off for matching')
    parser.add_argument('--show_score', action='store_true', help='show confidence score on the image')
    parser.add_argument('--show_lmks', action='store_true', help='show landmarks on the image')
    parser.add_argument('--single_cls', action='store_true', help='only use single class')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    opt = parser.parse_args()
    return opt


def main(opt):
    set_logging()
    inference(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)