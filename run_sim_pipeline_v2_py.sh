#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DET_FOLDER="${DET_FOLDER:-$APP_DIR/data/person}"
SEG_FOLDER="${SEG_FOLDER:-$APP_DIR/data/ship}"
DET_ENGINE="${DET_ENGINE:-$APP_DIR/weights/yolov8n.engine}"
SEG_ENGINE="${SEG_ENGINE:-$APP_DIR/weights/yolov8n-seg.engine}"

NUM_DET_WORKERS="${NUM_DET_WORKERS:-4}"
NUM_SEG_WORKERS="${NUM_SEG_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-1,2,3,5}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-64}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-64}"
DET_SOURCE_FPS="${DET_SOURCE_FPS:-20}"
SEG_SOURCE_FPS="${SEG_SOURCE_FPS:-20}"
DET_BUFFER_CAPACITY="${DET_BUFFER_CAPACITY:-256}"
SEG_BUFFER_CAPACITY="${SEG_BUFFER_CAPACITY:-256}"
LOG_BUFFER_CAPACITY="${LOG_BUFFER_CAPACITY:-256}"
LOG_JSONL="${LOG_JSONL:-$APP_DIR/output/sim_v2_py_log.jsonl}"

cd "$APP_DIR"

python3 sim_pipeline_v2.py \
  --det-folder "$DET_FOLDER" \
  --seg-folder "$SEG_FOLDER" \
  --det-engine "$DET_ENGINE" \
  --seg-engine "$SEG_ENGINE" \
  --num-det-workers "$NUM_DET_WORKERS" \
  --num-seg-workers "$NUM_SEG_WORKERS" \
  --gpu-ids "$GPU_IDS" \
  --det-batch-size "$DET_BATCH_SIZE" \
  --seg-batch-size "$SEG_BATCH_SIZE" \
  --det-source-fps "$DET_SOURCE_FPS" \
  --seg-source-fps "$SEG_SOURCE_FPS" \
  --det-buffer-capacity "$DET_BUFFER_CAPACITY" \
  --seg-buffer-capacity "$SEG_BUFFER_CAPACITY" \
  --log-buffer-capacity "$LOG_BUFFER_CAPACITY" \
  --log-jsonl "$LOG_JSONL"
