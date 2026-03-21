#!/bin/bash
# Launch SSFA webapp
# Usage: ./launch.sh [GPU_ID] [extra app args...]

GPU_ID=${1:-0}
shift || true

export CUDA_VISIBLE_DEVICES=$GPU_ID

cd "$(dirname "$0")"

python app.py \
    --device 0 \
    --bg-ckpt-dir ../output/BG_output \
    --seg-ckpt-dir ../output/SEG_output \
    --port 7860 \
    "$@"
