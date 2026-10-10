#!/bin/bash
set -euo pipefail
cd /home/src/dllm
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=logs/joint_scratch_1epoch.log
CKPT=data/checkpoints/joint_scratch_1epoch.pt
CMD=".venv-pro6000/bin/torchrun --standalone --nproc_per_node=2 train.py \
  --train-exit-joint \
  --dataset-dir data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --sequence-length 2048 \
  --batch-size 160 --gradient-accumulation 1 \
  --precision amp --grad-checkpoint --ce-chunk-rows 4096 \
  --steps 30429 --lr 1.5e-4 --warmup-steps 600 --min-lr-ratio 0.1 \
  --exit-joint-readout-lr 1e-4 \
  --exit-hard-min-depth 2 --exit-min-depth 3 \
  --save-steps 50 --save-interval-hours 1.0 --log-steps 1 --profile-timing \
  --dispatch-mode sorted --use-gqa \
  --checkpoint data/checkpoints/joint_scratch_1epoch.pt \
  --seed 42"
if [ -f "$CKPT" ]; then
  CMD="$CMD --resume $CKPT"
elif [ -f "$LOG" ]; then
  mv "$LOG" "${LOG%.log}.before_scratch_$(date +%Y%m%d_%H%M%S).log"
fi
setsid nohup bash -c "$CMD" >> "$LOG" 2>&1 < /dev/null &
echo "$!" > temp/joint_1epoch.pid
echo "launched pid=$(cat temp/joint_1epoch.pid) log=$LOG checkpoint=$CKPT"
