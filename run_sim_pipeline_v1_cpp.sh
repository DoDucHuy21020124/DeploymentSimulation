#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRT_ROOT="${TRT_ROOT:-/mnt/huydd/code/counting/TensorRT-10.13.0.35}"
BIN_NAME="sim_pipeline_v1"

DET_FOLDER="${DET_FOLDER:-$APP_DIR/data/person}"
SEG_FOLDER="${SEG_FOLDER:-$APP_DIR/data/ship}"
DET_ENGINE="${DET_ENGINE:-$APP_DIR/weights/yolov8n.engine}"
SEG_ENGINE="${SEG_ENGINE:-$APP_DIR/weights/yolov8n-seg.engine}"

DET_BATCH_SIZE="${DET_BATCH_SIZE:-1}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-1}"
DET_SOURCE_FPS="${DET_SOURCE_FPS:-20}"
SEG_SOURCE_FPS="${SEG_SOURCE_FPS:-20}"
DET_BUFFER_CAPACITY="${DET_BUFFER_CAPACITY:-256}"
SEG_BUFFER_CAPACITY="${SEG_BUFFER_CAPACITY:-256}"
LOG_BUFFER_CAPACITY="${LOG_BUFFER_CAPACITY:-256}"
LOG_JSONL="${LOG_JSONL:-$APP_DIR/output/sim_v1_cpp_log.jsonl}"
DEVICE="${DEVICE:-1}"

cd "$APP_DIR"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"

g++ -std=c++17 -O2 sim_pipeline_v1.cpp -o "$BIN_NAME" \
  $(pkg-config --cflags --libs opencv4) \
  -I"$TRT_ROOT/include" -I"$CUDA_ROOT/include" \
  -L"$TRT_ROOT/lib" -L"$CUDA_ROOT/lib64" \
  -Wl,-rpath,"$TRT_ROOT/lib" -Wl,-rpath,"$CUDA_ROOT/lib64" \
  -lnvinfer -lnvinfer_plugin -lcudart -pthread

"./$BIN_NAME" \
  --det-folder "$DET_FOLDER" \
  --seg-folder "$SEG_FOLDER" \
  --det-engine "$DET_ENGINE" \
  --seg-engine "$SEG_ENGINE" \
  --det-batch-size "$DET_BATCH_SIZE" \
  --seg-batch-size "$SEG_BATCH_SIZE" \
  --det-source-fps "$DET_SOURCE_FPS" \
  --seg-source-fps "$SEG_SOURCE_FPS" \
  --det-buffer-capacity "$DET_BUFFER_CAPACITY" \
  --seg-buffer-capacity "$SEG_BUFFER_CAPACITY" \
  --log-buffer-capacity "$LOG_BUFFER_CAPACITY" \
  --log-jsonl "$LOG_JSONL" \
  --device "$DEVICE"
