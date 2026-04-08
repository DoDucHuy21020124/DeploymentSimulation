# Simulation Pipeline v2

This project includes two implementations of a simulation pipeline: one in C++ and one in Python. Both versions perform object detection and segmentation using TensorRT inference engines.

## Prerequisites

- CUDA (default: `/usr/local/cuda`)
- TensorRT 10.13.0.35 (default: `/mnt/huydd/code/counting/TensorRT-10.13.0.35`)
- OpenCV 4
- PyTorch (for Python version)
- CMake and g++ (for C++ compilation)

## Running the Simulation Pipeline

### Option 1: C++ Implementation

Run the C++ version using:

```bash
./run_sim_pipeline_v2_cpp.sh
```

**What it does:**
- Compiles `sim_pipeline_v2.cpp` to produce a binary
- Executes the compiled binary with configured parameters
- Processes detection and segmentation tasks using TensorRT engines

### Option 2: Python Implementation

Run the Python version using:

```bash
./run_sim_pipeline_v2_py.sh
```

**What it does:**
- Directly executes `sim_pipeline_v2.py` using Python 3
- Processes detection and segmentation tasks using TensorRT engines
- Generally supports higher batch sizes than C++ (default 64 vs 1)

## Configuration Options

Both scripts support the following environment variables. Set them before running to customize behavior:

### Paths
- `CUDA_ROOT` - CUDA installation path (default: `/usr/local/cuda`)
- `TRT_ROOT` - TensorRT installation path (default: `/mnt/huydd/code/counting/TensorRT-10.13.0.35`)
- `DET_FOLDER` - Detection input folder (default: `$APP_DIR/data/person`)
- `SEG_FOLDER` - Segmentation input folder (default: `$APP_DIR/data/ship`)
- `DET_ENGINE` - Detection model engine file (default: `$APP_DIR/weights/yolov8n.engine`)
- `SEG_ENGINE` - Segmentation model engine file (default: `$APP_DIR/weights/yolov8n-seg.engine`)
- `LOG_JSONL` - Output log file path (default: `$APP_DIR/output/sim_v2_*_log.jsonl`)

### Processing Parameters
- `NUM_DET_WORKERS` - Number of detection worker threads (default: `4`)
- `NUM_SEG_WORKERS` - Number of segmentation worker threads (default: `4`)
- `GPU_IDS` - Comma-separated GPU IDs to use (default: `1,2,3,5`)
- `DET_BATCH_SIZE` - Detection batch size (C++ default: `1`, Python default: `64`)
- `SEG_BATCH_SIZE` - Segmentation batch size (C++ default: `1`, Python default: `64`)
- `DET_SOURCE_FPS` - Detection source FPS (default: `20`)
- `SEG_SOURCE_FPS` - Segmentation source FPS (default: `20`)
- `DET_BUFFER_CAPACITY` - Detection buffer size (default: `256`)
- `SEG_BUFFER_CAPACITY` - Segmentation buffer size (default: `256`)
- `LOG_BUFFER_CAPACITY` - Log buffer size (default: `256`)

## Usage Examples

### Run with default settings
```bash
./run_sim_pipeline_v2_cpp.sh
```
