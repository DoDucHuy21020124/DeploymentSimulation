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

def save_image(output_folder: str, image_name: str, image: np.array):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(image_path, image)

class Processor:
    def __init__(self, engine_file: str, cfg: dict):
        self.cfg = cfg
        self.engine_file = engine_file
        self.version = self.cfg["version"]
        self.func_implement(self.version, "init")
        self.path = "./logs"
    
    def func_implement(self, version, func_name, *args, **kwargs):
        if version == "v1":
            func = getattr(self, f"{func_name}_v1")
        else:
            func = getattr(self, f"{func_name}_v2")
        
        return func(*args, **kwargs)
    
    def init_v1(self):
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        # self.runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            serialized_engine = f.read()
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
        # deserialize the engine from a memory buffer
        # self.engine = self.runtime.deserialize_cuda_engine(self.serialized_engine)
        self.context = engine.create_execution_context()
        self.device = torch.device('cuda')

        # Create a stream in which to copy inputs/outputs and run inference
        # self.stream = cuda.Stream()

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
                
        self.bindings = OrderedDict()
        self.output_names = []
        self.input_names = []
        
        self.imgsz, self.outsz = None, None
        
        try:
            for binding in engine:
                bname = binding
                bsize = engine.get_tensor_shape(binding)
                btype = engine.get_tensor_dtype(binding)
                if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    self.input_names.append(bname)
                    self.dtype = trt.nptype(btype)
                    self.imgsz = bsize
                else:
                    self.output_names.append(bname)
                    self.outsz = bsize
                    
                im = torch.from_numpy(np.empty(bsize, dtype=trt.nptype(btype))).to(self.device)
                logger.info(f"bname = {bname} - bsize = {bsize} - btype = {btype}")
                self.bindings[bname] = Binding(bname, trt.nptype(btype), bsize, im, int(im.data_ptr()))
            
            self.binding_addrs = OrderedDict((k, v.ptr) for k, v in self.bindings.items())
            
            logger.info(f"Binding address: {self.binding_addrs}")
            logger.info(f"Binding input size = {self.imgsz} - output size = {self.outsz}")
        
        except Exception as e:
            logger.error(e)
            breakpoint()

    def execute_trt_v1(self, im):
        for x in self.input_names:
            assert im.shape == self.bindings[x].shape, f"input size {im.shape} != model size {self.bindings[x].shape}"
            self.binding_addrs[x] = int(im.data_ptr())

        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in self.output_names]
        return y[0]
    
    def init_v2(self):
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
        self.imgsbz = None
        self.outsz = None
        self.dtype = None
        
        for binding in engine:
            bname = binding
            bsize = engine.get_tensor_shape(binding)
            btype = engine.get_tensor_dtype(binding)
            
            host_mem = cuda.pagelocked_empty(trt.volume(bsize) * self.cfg['batch_size'], trt.nptype(btype))
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
                print('bsize', bsize)
                print('self.outsz', self.outsz)
                print(f"BBBBBSSSSSIZZZEEEE output = {trt.volume(bsize)}")
                
                
        logger.info(f"Binding memory: {self.bindings}")
        logger.info(f"Binding inputs: {[self.host_inputs[i].shape for i in range(len(self.host_inputs))]}")
        logger.info(f"Binding outputs: {[self.host_outputs[i].shape for i in range(len(self.host_outputs))]}")
        logger.info(f"Binding input size: {self.imgsz} - output size: {self.outsz}")
                
    def execute_trt_v2(self, img):
        np.copyto(self.host_inputs[0], img.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0])
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0])
        # breakpoint()
        # for val in self.host_outputs[0]:
        #     print(val)
        # np.savetxt(self.path + "/" + "raw_output.txt", np.array(self.host_outputs[0]), fmt="%f")
        # print_out(self.host_outputs[0].reshape(self.cfg['batch_size'], self.outsz[1], 20), "chw", "raw_reshape.txt")
        # print_out(self.host_outputs[0].reshape(self.cfg['batch_size'], self.outsz[1], 20)[..., 4:5], "chw", os.path.join(self.path, "raw_conf.txt"))
        
        # print(self.host_outputs)
        # print(len(self.host_outputs))
        # print(self.host_outputs[0].shape)
        return torch.from_numpy(np.reshape(self.host_outputs[0], (self.cfg['batch_size'], self.outsz[1], self.outsz[2])))

    def infer(self, im_path):
        imgs, paths, shapes = self.preprocess_image(im_path, self.imgsz[3])
        y = self.func_implement(self.version, "execute_trt", imgs)
        # print('y.shape', y.shape)
        y = self.post_process(y, imgs, paths, shapes)
        return y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        img = self.func_implement(self.version, "emptyimg", imgsz)
        y = self.func_implement(self.version, "execute_trt", img)
    

    def imgtype_v1(self, img):
        return torch.from_numpy(img).to(self.device)
    
    def imgtype_v2(self, img):
        return img

    def emptyimg_v1(self, imgsz):
        return torch.empty(*imgsz, dtype=torch.float, device=self.device)
    
    def emptyimg_v2(self, imgsz):
        return np.empty(imgsz, dtype=np.float32)

def print_out(img, channels="hwc", save="img.txt"):
    log = ""
    
    if channels == "hwc":
        h, w, c = img.shape
        for i in range(h):
            for j in range(w):
                log += " ".join(map(str, img[i, j].tolist())) + "\n"
            log += "\n"
            
    if channels == "chw":
        c, h, w = img.shape
        for i in range(c):
            for j in range(h):    
                log += " ".join(map(str, img[i, j, :].tolist())) + "\n"
            log += "\n"
            
    if channels == "bchw":
        b, c, h, w = img.shape
        for l in range(b):
            for i in range(c):
                for j in range(h):    
                    log += " ".join(map(lambda t: "{:.6f}".format(round(t, 6)), img[l, i, j, :].tolist())) + "\n"
                log += "\n"
            
    with open(save, "w") as f:
        f.write(log)
    
    
    
def run(opt):
    processor = Processor(opt.weights.replace(".pt", ".engine").replace(".pth", ".engine"), cfg={
        'conf_thres': opt.conf_thres,  # the larger conf threshold for filtering body detection proposals
        'iou_thres': opt.iou_thres, # the smaller iou threshold for filtering body detection proposals
        'conf_thres_part': opt.conf_thres_part, # the larger conf threshold for filtering body-part detection proposals
        'iou_thres_part': opt.iou_thres_part,  # the smaller iou threshold for filtering body-part detection proposals
        'match_iou_thres': opt.th_match,  # whether a body-part in matched with one body bbox
        'single_cls': opt.single_cls,  # whether only use single class
        'visual': opt.visual,
        'nc': opt.num_classes,
        'num_offsets': opt.num_offsets,
        'num_lmks': opt.num_lmks,
        "version": "v2",
        'batch_size': 1,
        'output_folder': opt.output_folder
    })
    
    # processor.warmup()

    start = time.time()
    cnt = 0
    if opt.visual and not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    if os.path.isfile(opt.source_dir):
        cnt = 1
        pred = processor.infer(opt.source_dir)
        # processor.visual(pred, opt.source_dir)

    else:
        from tqdm import tqdm
        cnt = len(os.listdir(opt.source_dir))
        # for source_dir in tqdm(os.listdir(opt.source_dir)):
        #     source = os.path.join(opt.source_dir, source_dir)
        pred = processor.infer(opt.source_dir)
        # processor.visual(pred, opt.source_dir, opt.output_folder)
            
    end = time.time()
    logger.info(f'execution time: {end - start} - fps = {cnt / (end - start)}')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='./test_imgs/NVRDataset/108_11_54_45_to_11_55_30_0512202_image-3.jpeg', help='dataset.yaml path')
    parser.add_argument('--output_folder', default='.', help='saving folder')
    parser.add_argument('--weights', default='./weights/yolov5s6.pt')
    parser.add_argument('--num_offsets', type=int, default=2, help='number of offsets')
    parser.add_argument('--num_lmks', type=int, default=5, help='number of landmarks')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--conf-thres-part', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres-part', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--th_match', type=float, default=0.96, help='cut off for matching')
    parser.add_argument('--show_score', action='store_true', help='show confidence score on the image')
    parser.add_argument('--show_lmks', action='store_true', help='show landmarks on the image')
    parser.add_argument('--single_cls', action='store_true', help='only use single class')
    parser.add_argument('--visual', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    run(opt)