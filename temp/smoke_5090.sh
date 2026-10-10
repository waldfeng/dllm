#!/bin/bash
# End-to-end smoke of the single-GPU 5090 continuation path: 3 fresh steps
# (warm start from the continuation checkpoint) + 1 resume step, then
# cleanup.  Uses a throwaway checkpoint file.
# 注意：必须在 GPU 空闲时运行（峰值 ~28.5GiB，与在线训练互斥）。
cd /home/src/dllm
set -e
SMOKE_CKPT=/home/datasets/dllm/data/checkpoints/.smoke-5090.pt
rm -f "$SMOKE_CKPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== smoke phase 1: warm start, 3 steps"
.venv/bin/python train.py \
  --init-weights-from /home/datasets/dllm/datasets/checkpoints/memory-transformer-5090-continuation.pt \
  --dataset-dir /home/datasets/dllm/data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --sequence-length 2048 --batch-size 11 --gradient-accumulation 3 \
  --precision amp --grad-checkpoint --ce-chunk-rows 2048 \
  --steps 3 --schedule-steps 25000 \
  --lr 1.5e-4 --warmup-steps 300 --min-lr-ratio 0.1 \
  --save-steps 3 --log-steps 1 --no-final-save \
  --checkpoint "$SMOKE_CKPT"

echo "=== smoke phase 2: resume, 1 more step"
.venv/bin/python train.py \
  --resume "$SMOKE_CKPT" \
  --dataset-dir /home/datasets/dllm/data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --sequence-length 2048 --batch-size 11 --gradient-accumulation 3 \
  --precision amp --grad-checkpoint --ce-chunk-rows 2048 \
  --steps 4 --schedule-steps 25000 \
  --lr 1.5e-4 --warmup-steps 300 --min-lr-ratio 0.1 \
  --save-steps 100 --log-steps 1 \
  --checkpoint "$SMOKE_CKPT"
rm -f "$SMOKE_CKPT"
echo "=== smoke OK"
