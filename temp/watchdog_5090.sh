#!/bin/bash
# 前台看门狗：每60秒检查训练主进程与显存，并打印一行状态（进度以 step 计）。
# 唤醒条件（打印 ALERT 后以码 1 退出，返回控制给助手）：
#   1) 训练主进程退出  2) 显存占用 < 10GiB
#   3) nvidia-smi 无响应  4) 日志超过 20 分钟无新 step 行（疑似卡死）
# 说明：训练日志每 2 步(~66s)写一行(--log-steps 2)，与看门狗频率一致。
# 日志行仅含：step/loss/lr + 路由健康（rl_min/rl_max=24 层路由负载损失
# min/max；rb_dead/rb_max/rb_min/rb_ent=24 个 memory block 的 dead 数/
# 负载 max/min/归一化熵）+ 早退（eavg/emax=逐样本早退层数均值/最大值
# 的 no-grad 探针，exit/ro/md=机制状态/读出头/最小早退深度）。
# 处理完错误后：temp/resume_5090.sh 重启训练，并再次前台运行本看门狗。
LOG=/home/src/dllm/logs/train_5090_continuation.log
DATASET=tokens_v2
CKPT=memory-transformer-5090-continuation.pt
BATCH="B11 x GA3 x seq2048 = 67584 tokens/step"
RATE="~1.8 steps/min (~33s/step)"
STEP_SEC=33.0

start_ts=$(date +%s)
last_step=0
last_change=$(date +%s)
while true; do
  if ! pgrep -f "python train.py" >/dev/null 2>&1; then
    echo "ALERT: 训练主进程已退出，可能出错了，请求检查！日志尾部:"
    tail -15 "$LOG"
    exit 1
  fi
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$mem" ]; then
    echo "ALERT: nvidia-smi 无响应，可能出错了，请求检查！日志尾部:"
    tail -15 "$LOG"
    exit 1
  fi
  if [ "$mem" -lt 10240 ]; then
    echo "ALERT: 显存仅 ${mem}MiB (<10GiB)，训练可能异常，可能出错了，请求检查！日志尾部:"
    tail -15 "$LOG"
    exit 1
  fi
  now=$(date +%s)
  mins=$(( (now - start_ts) / 60 ))
  line=$(grep "^step=" "$LOG" | tail -1)
  cur=$(printf '%s' "$line" | sed -n 's/^step=\([0-9]*\).*/\1/p')
  [ -z "$cur" ] && cur=0
  if [ "$cur" != "$last_step" ]; then last_change=$now; fi
  stale=$(( now - last_change ))
  if [ "$stale" -gt 1200 ]; then
    echo "ALERT: 日志已 ${stale}s 无新 step 行（>20min，疑似卡死），可能出错了，请求检查！日志尾部:"
    tail -15 "$LOG"
    exit 1
  fi
  delta=$((cur - last_step)); last_step=$cur
  expect=$(awk -v m="$mins" -v s="$STEP_SEC" 'BEGIN{printf "%.1f", m*60/s}')
  echo "[watchdog ${mins}min] step=$cur (本次+${delta}, 累计预期≈${expect} steps, ${RATE}) mem=${mem}MiB 日志更新于${stale}s前 dataset=${DATASET} batch=${BATCH} ckpt=${CKPT}"
  echo "  ${line}"
  sleep 60
done
