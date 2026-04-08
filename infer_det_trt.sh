#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="/usr/local/cuda"
APP_DIR="/mnt/huydd/code/counting/converter"
TRT_ROOT="/mnt/huydd/code/counting/TensorRT-10.13.0.35"
ENGINE_PATH="weights/yolov8n.engine"
INPUT_IMAGE="data/ship/ship1.jpg"
OUTPUT_DIR="output/detection_cpp"
# DEVICE_ID="${DEVICE_ID:-0}"
DEVICE_ID=3
BIN_NAME="infer_det_trt"

cd "$APP_DIR"

export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"

build_binary() {
  g++ -std=c++17 -O2 infer_det_trt.cpp -o "$BIN_NAME" \
    $(pkg-config --cflags --libs opencv4) \
    -I"$TRT_ROOT/include" -I"$CUDA_ROOT/include" \
    -L"$TRT_ROOT/lib" -L"$CUDA_ROOT/lib64" \
    -Wl,-rpath,"$TRT_ROOT/lib" -Wl,-rpath,"$CUDA_ROOT/lib64" \
    -lnvinfer -lnvinfer_plugin -lcudart
}

if [[ -f "$BIN_NAME" ]]; then
  read -r -p "[build] $BIN_NAME exists. Rebuild? [y/N]: " ans
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    build_binary
  else
    echo "[build] Reusing existing $BIN_NAME"
  fi
else
  echo "[build] $BIN_NAME not found -> building"
  build_binary
fi

"./$BIN_NAME" \
  "$ENGINE_PATH" \
  "$INPUT_IMAGE" \
  "$OUTPUT_DIR" 0.4 1 "$DEVICE_ID"
