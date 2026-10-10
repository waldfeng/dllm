#!/bin/bash
# 错误恢复：从最新 continuation ckpt resume，继续 25000 步训练计划。
# 超参/数据流与 temp/launch_5090.sh 完全一致（step/优化器/schedule/数据位置由 ckpt 恢复）。
cd /home/src/dllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup .venv/bin/python train.py \
  --resume /home/datasets/dllm/datasets/checkpoints/memory-transformer-5090-continuation.pt \
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
echo "resumed pid=$! log=logs/train_5090_continuation.log"
