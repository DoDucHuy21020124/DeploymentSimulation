#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DET_FOLDER="${DET_FOLDER:-$APP_DIR/data/person}"
SEG_FOLDER="${SEG_FOLDER:-$APP_DIR/data/ship}"
DET_ENGINE="${DET_ENGINE:-$APP_DIR/weights/yolov8n.engine}"
SEG_ENGINE="${SEG_ENGINE:-$APP_DIR/weights/yolov8n-seg.engine}"

DET_BATCH_SIZE="${DET_BATCH_SIZE:-64}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-64}"
DET_SOURCE_FPS="${DET_SOURCE_FPS:-20}"
SEG_SOURCE_FPS="${SEG_SOURCE_FPS:-20}"
DET_BUFFER_CAPACITY="${DET_BUFFER_CAPACITY:-256}"
SEG_BUFFER_CAPACITY="${SEG_BUFFER_CAPACITY:-256}"
LOG_BUFFER_CAPACITY="${LOG_BUFFER_CAPACITY:-256}"
LOG_JSONL="${LOG_JSONL:-$APP_DIR/output/sim_v1_py_log.jsonl}"
DEVICE="${DEVICE:-1}"
# DET_DEVICE="${DET_DEVICE:-$DEVICE}"  # Default to same as DEVICE if not set
# SEG_DEVICE="${SEG_DEVICE:-$DEVICE}"  # Default to same as DEVICE if not set
# DEVICE=1

cd "$APP_DIR"

python3 sim_pipeline_v1.py \
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
  --device "$DEVICE" \
  # --det-device "$DET_DEVICE" \
  # --seg-device "$SEG_DEVICE"
