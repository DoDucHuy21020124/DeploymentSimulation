import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import argparse
import yaml
import os

import onnx
from ultralytics import YOLO

def convert(opt):
    
    bs, weight, onnx_path, imgsz = \
        opt.batch_size, opt.weight, opt.output, opt.imgsz

    imgsz = (480, 640)
    
    model = YOLO(weight)
    model = model.model.fuse()

    # model.onnx_dynamic = True

    model.eval()
    # model.model[-1].export = True
    # imgsz = check_img_size(imgsz, s=stride)  # check image size
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    inp = torch.rand(bs, 3, imgsz[0], imgsz[1])
    inp /= 255
    if len(inp.shape) == 3:
        # inp = inp[None]  # expand for batch dim
        # inp = torch.cat([inp, inp], axis=0)
        inp = inp.repeat(bs, 1, 1, 1)
    
    # breakpoint()
    pred = model(inp)
    print(len(pred))
    # print(pred[0], pred[1])
    print(pred[0].shape)
    print(pred[1].keys())
    print(pred[1]['boxes'].shape)
    print(pred[1]['scores'].shape)
    # print(pred[1]['feats'].shape)
    for feat in pred[1]['feats']:
        print(feat.shape)
    # print(pred.shape)
    torch.onnx.export(
        model = model, 
        args = inp,
        f = onnx_path, 
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=16,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
    )
    
    onnx_model = onnx.load_model(onnx_path)
    onnx.checker.check_model(onnx_model)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='yolov5s6.pt')
    parser.add_argument('--output', '-o', default='./output', type=str)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1536, help='inference size (pixels)')

    opt = parser.parse_args()
    return opt

def run():
    args = parse_opt()
    # assert '.onnx' in opt.onnx_path, 'Output file must in the .onnx format'

    os.makedirs(args.output, exist_ok=True)
    basename = os.path.basename(args.weight)
    basename = os.path.splitext(basename)[0]
    output_file_name = basename + '.onnx'
    args.output = os.path.join(args.output, output_file_name)
    convert(args)

if __name__ == "__main__":
    run()