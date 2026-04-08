import tensorrt as trt
import sys
import argparse

print(trt.__version__)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
print(EXPLICIT_BATCH)

def build_engine(onnx_file_path, output_engine_path, max_batch_size, imgsz):   
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()

    # config.max_workspace_size = (1<<32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32) # 1 MiB
    config.set_flag(trt.BuilderFlag.FP16)
    config.default_device_type = trt.DeviceType.GPU

    min_shape = (1, 3, imgsz, imgsz)
    opt_shape = (1, 3, imgsz, imgsz)
    max_shape = (max_batch_size, 3, imgsz, imgsz)


    profile = builder.create_optimization_profile()
    profile.set_shape('input', min_shape, opt_shape, max_shape)    # random nubmers for min. opt. max batch
    config.add_optimization_profile(profile)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    serialized_engine = builder.build_serialized_network(network, config)

    with open(output_engine_path, 'wb') as f:
        f.write(serialized_engine)

    print('Completed creating Engine')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-weights', type=str, help='onnx-weights path')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--max-batch-size', type=int, default=1)
    opt = parser.parse_args()
    return opt
    
if __name__ == "__main__":
    opt = parse_opt()
    build_engine(opt.onnx_weights, opt.onnx_weights.replace(".onnx", ".engine"), opt.max_batch_size, opt.imgsz)