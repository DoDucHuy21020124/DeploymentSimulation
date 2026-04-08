import torch
import torch.nn as nn
import tensorrt as trt
from collections import OrderedDict, namedtuple
import numpy as np
from pathlib import Path
import cv2
from typing import List, Optional, Tuple, Union

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
                # Use OPT profile shape for initial allocation to avoid reserving memory for MAX profile upfront.
                # Tensor shapes are still adjusted dynamically in forward() when needed.
                # profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[1]) if is_input else None
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

if __name__ == "__main__":
    # Example usage
    engine_path = "weights/yolov8n.engine"
    x = torch.randn(1, 3, 480, 640).to("cuda:1")

    device_id = 1
    batch_size = 4
    model = TensorRTBackend()
    model.load_model(engine_path, "cuda:1")

    # # Dummy input tensor
    # input_tensor = torch.randn(batch_size, 3, 224, 224).to(f"cuda:{device_id}")
    # outputs = model(input_tensor)
    # print(outputs)