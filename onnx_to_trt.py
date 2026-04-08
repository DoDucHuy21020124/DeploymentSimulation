import tensorrt as trt
import sys
import argparse

print(trt.__version__)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
print(EXPLICIT_BATCH)

def build_engine_v1(onnx_file_path, output_engine_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        # logger.info('Beginning ONNX file parsing')
        print('Beginning ONNX file parsing')
        success = parser.parse(model.read())
        if not success:
            for error in range(parser.num_errors):
                logger.info("ERROR ", parser.get_error(error))
    # logger.info('Completed parsing of ONNX file')
    print('Completed parsing of ONNX file')

    # building an Engine by TensorRT 7
    """
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 1
    # use FP16 model if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
        logger.info('Building an engine...')
        engine = builder.build_cuda_engine(network)
        context = engine.create_execution_context()
        logger.info('Completed creating Engine')
        with open(output_engine_path, 'wb') as f:
            f.write(engine.serialize())
        logger.info('Completed save Engine')
    """

    # building an Engine by TensorRT 8
    # logger.info('Building an engine...')
    print('Building an engine...')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 MiB

    # use FP16
    config.set_flag(trt.BuilderFlag.FP16)
    # use INT8
    # config.set_flag(trt.BuilderFlag.INT8)

    serialized_engine = builder.build_serialized_network(network, config)
    # engine = builder.build_engine(network, config)
    with open(output_engine_path, 'wb') as f:
        f.write(serialized_engine)
    # logger.info('Completed creating Engine')
    print('Completed creating Engine')
    
def build_engine_v2(onnx_file_path, output_engine_path, max_batch_size, imgsz):   
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()

    # config.max_workspace_size = (1<<32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32) # 1 MiB
    config.set_flag(trt.BuilderFlag.FP16)
    config.default_device_type = trt.DeviceType.GPU

    min_shape = (max_batch_size, 3, imgsz, imgsz)
    opt_shape = (max_batch_size, 3, imgsz, imgsz)
    max_shape = (max_batch_size, 3, imgsz, imgsz)


    profile = builder.create_optimization_profile()
    profile.set_shape('input', min_shape, opt_shape, max_shape)    # random nubmers for min. opt. max batch
    config.add_optimization_profile(profile)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    # engine = builder.build_engine(network, config)
    # buf = engine.serialize()
    # with open(output_engine_path, 'wb') as f:
    #     f.write(buf)

    serialized_engine = builder.build_serialized_network(network, config)
    # engine = builder.build_engine(network, config)
    with open(output_engine_path, 'wb') as f:
        f.write(serialized_engine)
    # logger.info('Completed creating Engine')
    print('Completed creating Engine')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-weights', type=str, default='output.engine', help='dataset.yaml path')
    parser.add_argument('--imgsz', type=int, default=1536)
    parser.add_argument('--max-batch-size', type=int, default=1)
    opt = parser.parse_args()
    return opt
    
if __name__ == "__main__":
    opt = parse_opt()
    # build_engine_v1(opt.onnx_weights, opt.onnx_weights.replace(".onnx", "_v1.engine"))
    build_engine_v2(opt.onnx_weights, opt.onnx_weights.replace(".onnx", "_v2.engine"), opt.max_batch_size, opt.imgsz)