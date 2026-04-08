import time
from collections import namedtuple, OrderedDict

import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

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

class TRTEngine:
    def __init__(self, engine_file: str, batch_size: int):
        self.batch_size = batch_size
        self.engine_file = engine_file

        self.init()
    
    def init(self):
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        with open(self.engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            serialized_engine = f.read()
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
        self.context = engine.create_execution_context()
                
        self.host_inputs  = []
        self.cuda_inputs  = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.imgsz = None
        self.outsz = None
        self.dtype = None
        
        for binding in engine:
            bsize = engine.get_tensor_shape(binding)
            # bsize = tuple([self.batch_size] + list(bsize[1:]))
            btype = engine.get_tensor_dtype(binding)
            
            host_mem = cuda.pagelocked_empty(trt.volume(bsize), trt.nptype(btype))
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
                            
            if engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                self.imgsz = bsize
                self.dtype = trt.nptype(btype)
                print(f"BBBBBSSSSSIZZZEEEE input = {trt.volume(bsize)}")
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                self.outsz = bsize
                print(f"BBBBBSSSSSIZZZEEEE output = {trt.volume(bsize)}")
                
                
        logger.info(f"Binding memory: {self.bindings}")
        logger.info(f"Binding inputs: {[self.host_inputs[i].shape for i in range(len(self.host_inputs))]}")
        logger.info(f"Binding outputs: {[self.host_outputs[i].shape for i in range(len(self.host_outputs))]}")
        logger.info(f"Binding input size: {self.imgsz} - output size: {self.outsz}")
                
    def execute_trt(self, img):
        np.copyto(self.host_inputs[0], img.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0])
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh_async(self.host_outputs[-1], self.cuda_outputs[-1])
        out_shape = [self.batch_size] + list(self.outsz[1:])
        return torch.from_numpy(np.reshape(self.host_outputs[-1], out_shape))

    def infer(self, image):
        y = self.execute_trt(image)
        return y

    def warmup(self):
        imgsz = [self.batch_size] + list(self.imgsz[1:])
        image = np.zeros(imgsz)
        y = self.execute_trt(image)
        return y
    

if __name__ == "__main__":
    weight = '/mnt/huydd/code/counting/Human-Detector/yolov8n.engine'
    max_batch_size = 64

    model = TRTEngine(weight, max_batch_size)
    model.warmup()

    x = np.random.random((64, 3, 480, 640))
    y = model(x)
    print(y)