#!/bin/bash
set -euo pipefail
cd /home/src/dllm
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=logs/train_pro6000_continuation.log
CKPT=data/checkpoints/pro6000-continuation.pt
RESUME=datasets/checkpoints/memory-transformer-5090-continuation.pt
if [ -f "$CKPT" ]; then RESUME="$CKPT"; fi
setsid nohup .venv-pro6000/bin/torchrun --standalone --nproc_per_node=2 train.py \
  --no-fsdp \
  --resume "$RESUME" \
  --resume-repartition \
  --checkpoint "$CKPT" \
  --dataset-dir data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --sequence-length 2048 \
  --batch-size 82 --gradient-accumulation 1 \
  --precision amp --grad-checkpoint --ce-chunk-rows 2048 \
  --steps 25000 --schedule-steps 25000 \
  --lr 1.5e-4 --warmup-steps 300 --min-lr-ratio 0.1 \
  --router-weight 0.01 --max-grad-norm 1.0 --seed 42 \
  --save-steps 250 --save-interval-hours 1.0 --log-steps 1 \
  --max-runtime-hours 48 \
  >> "$LOG" 2>&1 < /dev/null &
echo "$!" > temp/pro6000.pid
echo "launched pid=$! log=$LOG checkpoint=$CKPT"
