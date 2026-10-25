#!/bin/bash
set -euo pipefail
cd /home/src/dllm
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=checkpoints/logs/pretrain_1epoch_v2.log
CKPT=checkpoints/pretrain_1epoch_v2.pt
# fresh initialization: no --resume anywhere
setsid nohup .venv/bin/torchrun --standalone --nproc_per_node=2 train.py \
  --no-fsdp \
  --dataset-dir data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --sequence-length 2048 \
  --batch-size 82 --gradient-accumulation 1 \
  --precision amp --grad-checkpoint --ce-chunk-rows 2048 \
  --steps 59374 --schedule-steps 59374 \
  --lr 3e-4 --warmup-steps 1000 --min-lr-ratio 0.1 \
  --router-weight 0.01 --max-grad-norm 1.0 --seed 42 \
  --save-steps 250 --save-interval-hours 1.0 --log-steps 1 \
  --checkpoint "$CKPT" \
  >> "$LOG" 2>&1 < /dev/null &
echo "$!" > temp/pretrain_1epoch_v2.pid
echo "launched pid=$(cat temp/pretrain_1epoch_v2.pid) log=$LOG checkpoint=$CKPT"
