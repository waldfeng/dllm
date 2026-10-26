#!/bin/bash
# Launch the joint exit training in the background, resuming from the latest
# checkpoint.  Usage: launch_joint.sh [batch_size]
cd /home/src/dllm
B=${1:-14}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup \
  .venv/bin/torchrun --standalone --nproc_per_node=2 train.py \
  --train-exit-joint \
  --resume data/checkpoints/joint-exit-v1.pt \
  --checkpoint data/checkpoints/joint-exit-v1.pt \
  --dataset-dir data/tokens_v2 \
  --sequence-length 2048 \
  --batch-size "$B" --gradient-accumulation 2 \
  --steps 20000 --schedule-steps 20000 --warmup-steps 100 \
  --lr 1e-4 --exit-min-depth 6 \
  --exit-joint-readout-lr 1.25e-5 \
  --log-steps 10 --save-steps 250 --save-interval-hours 0.5 \
  --max-runtime-hours 48 \
  >> logs/train_joint_exit_41k.log 2>&1 < /dev/null &
echo "launched batch=$B pid=$!"
