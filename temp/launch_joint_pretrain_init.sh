#!/bin/bash
set -euo pipefail
cd /home/src/dllm
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=logs/joint_pretrain_init_1epoch.log
CKPT=checkpoints/joint_pretrain_init_1epoch.pt
BASE=".venv-pro6000/bin/torchrun --standalone --nproc_per_node=2 train.py \
  --train-exit-joint \
  --dataset-dir data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --sequence-length 2048 \
  --batch-size 160 --gradient-accumulation 1 \
  --precision amp --grad-checkpoint --ce-chunk-rows 4096 \
  --steps 30429 --lr 1.5e-4 --warmup-steps 600 --min-lr-ratio 0.1 \
  --exit-joint-readout-lr 1e-4 \
  --exit-hard-min-depth 2 --exit-min-depth 3 \
  --loss-weight-prior "0.70,0.06,0.06,0.12,0.06" \
  --save-steps 50 --save-interval-hours 1.0 --log-steps 1 --profile-timing \
  --dispatch-mode sorted --use-gqa \
  --checkpoint $CKPT \
  --seed 42"
if [ -f "$CKPT" ]; then
  CMD="$BASE --resume $CKPT"
else
  CMD="$BASE --init-weights-from checkpoints/pretrain_1epoch_v2.pt"
  if [ -f "$LOG" ]; then
    mv "$LOG" "${LOG%.log}.before_pretrain_init_$(date +%Y%m%d_%H%M%S).log"
  fi
fi
setsid nohup bash -c "$CMD" >> "$LOG" 2>&1 < /dev/null &
echo "$!" > temp/joint_pretrain_init.pid
echo "launched pid=$(cat temp/joint_pretrain_init.pid) log=$LOG checkpoint=$CKPT"
