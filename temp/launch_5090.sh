#!/bin/bash
# Single-GPU RTX 5090 continuation of the memory-transformer pretraining.
# Warm start from the latest surviving checkpoint (router03, step 39004) on
# the relocated tokens_v2 stream; AMP (fp32 master + bf16 autocast), per-cycle
# activation checkpointing, chunked fp32 CE.  B11 x GA3 x seq2048 = 67,584
# tokens/step (~ the original 2xGPU x B2 x GA4 x seq4096 = 65,536), peak
# ~28.5GiB / 31.4GiB (90.9%).
cd /home/src/dllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup .venv/bin/python train.py \
  --init-weights-from /home/datasets/dllm/datasets/checkpoints/memory-transformer-24h-router03.pt \
  --dataset-dir /home/datasets/dllm/data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --sequence-length 2048 --batch-size 11 --gradient-accumulation 3 \
  --precision amp --grad-checkpoint --ce-chunk-rows 2048 \
  --steps 25000 --schedule-steps 25000 \
  --lr 1.5e-4 --warmup-steps 300 --min-lr-ratio 0.1 \
  --router-weight 0.01 --max-grad-norm 1.0 --seed 42 \
  --checkpoint /home/datasets/dllm/datasets/checkpoints/memory-transformer-5090-continuation.pt \
  --save-steps 2000 --save-interval-hours 1.0 --log-steps 2 \
  >> logs/train_5090_continuation.log 2>&1 < /dev/null &
echo "launched pid=$! log=logs/train_5090_continuation.log"
