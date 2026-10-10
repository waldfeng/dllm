#!/bin/bash
# One 30-minute watchdog cycle for the joint exit training run.
# Sleeps, then inspects health; repairs by resuming from the latest checkpoint
# (dropping batch size if the crash was OOM). Prints one status line either way.
# Exit codes: 0 = healthy or repaired, 2 = fatal, do not re-arm.

LOG=/home/src/dllm/logs/train_joint_exit_41k.log
CKPT=/home/src/dllm/data/checkpoints/joint-exit-v1.pt
FIXLOG=/home/src/dllm/logs/watchdog_fixes.log
MAX_B=9
FIXES_THIS_CYCLE=0

cd /home/src/dllm

stamp() { date "+%F %T"; }

launch() {  # $1 = batch size
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup \
    .venv/bin/torchrun --standalone --nproc_per_node=2 train.py \
    --train-exit-joint \
    --resume "$CKPT" --checkpoint "$CKPT" \
    --dataset-dir data/tokens_v2 \
    --sequence-length 2048 \
    --batch-size "$1" --gradient-accumulation 2 \
    --steps 41000 --schedule-steps 41000 --warmup-steps 100 \
    --lr 1e-4 --exit-min-depth 6 \
    --loss-weight-prior "0.0,0.15,0.15,0.10,0.45,0.15" \
    --exit-joint-readout-lr 1.25e-5 \
    --log-steps 1 --save-steps 250 --save-interval-hours 0.5 \
    --max-runtime-hours 48 \
    >> "$LOG" 2>&1 < /dev/null &
  echo "[$(stamp)] watchdog: relaunched training (batch=$1, resume from $CKPT)" >> "$FIXLOG"
}

sleep 1800

# ------- wake up: gather status -------
LAST_STEP=$(grep -oE "^joint step=[0-9]+" "$LOG" | tail -1 | cut -d= -f2)
LAST_LINE=$(grep "^joint step" "$LOG" | tail -1)
LAST_ERR=$(grep -cE "OutOfMemoryError|NCCL.*timeout|ChildFailedError|SIGTERM|non-finite" "$LOG")

if pgrep -f "train.py" > /dev/null; then
  ALIVE=1
else
  ALIVE=0
fi

# fresh log? (1 step ≈ 4.5 s; 15 min staleness = hang)
if [ -n "$(find "$LOG" -mmin -15)" ]; then
  FRESH=1
else
  FRESH=0
fi

echo "[$(stamp)] status: alive=$ALIVE fresh=$FRESH last_step=$LAST_STEP recent_errs_total=$LAST_ERR"
echo "[$(stamp)] last: $LAST_LINE"

# ------- repair logic (max 2 repair attempts per cycle) -------
while [ "$FIXES_THIS_CYCLE" -lt 2 ]; do
  NEED_FIX=0
  REASON=""

  if [ "$ALIVE" -eq 0 ]; then
    NEED_FIX=1
    REASON="process dead"
  elif [ "$FRESH" -eq 0 ]; then
    NEED_FIX=1
    REASON="log stale >15min (hang)"
  elif [ "$(grep -c "OutOfMemoryError" "$LOG")" -gt 0 ]; then
    NEED_FIX=1
    REASON="OOM in log"
  elif [ "$(grep -c "non-finite.*backbone\|NaN detected" "$LOG")" -gt 0 ]; then
    NEED_FIX=1
    REASON="non-finite grads in log"
  fi

  if [ "$NEED_FIX" -eq 0 ]; then
    break
  fi

  echo "[$(stamp)] watchdog: problem detected ($REASON), attempting repair $((FIXES_THIS_CYCLE+1))/2" | tee -a "$FIXLOG"

  # kill everything left over
  pkill -9 -f "train.py" 2>/dev/null
  pkill -9 -f "torchrun" 2>/dev/null
  sleep 10

  # pick batch size: if the tail of the log shows OOM, drop to B=8
  B="$MAX_B"
  if [ "$REASON" = "OOM in log" ] || tail -200 "$LOG" | grep -q "OutOfMemoryError"; then
    B=8
    echo "[$(stamp)] watchdog: OOM seen, relaunching with batch-size 8" | tee -a "$FIXLOG"
  fi

  launch "$B"
  sleep 420   # give it ~7 min to load 13GB weights + write first step line

  if grep "^joint step" "$LOG" | tail -1 | grep -qE "step=$LAST_STEP|step=$((LAST_STEP+1))" && pgrep -f "train.py" > /dev/null; then
    NEWLAST=$(grep "^joint step" "$LOG" | tail -1)
    echo "[$(stamp)] watchdog: repair OK -> $NEWLAST" | tee -a "$FIXLOG"
    FIXES_THIS_CYCLE=$((FIXES_THIS_CYCLE+1))
    ALIVE=1
    FRESH=1
    LAST_STEP=$(grep -oE "^joint step=[0-9]+" "$LOG" | tail -1 | cut -d= -f2)
    LAST_ERR=0
    break   # repaired; wait for next cycle to judge stability
  else
    echo "[$(stamp)] watchdog: repair attempt $((FIXES_THIS_CYCLE+1)) did not produce progress" | tee -a "$FIXLOG"
    FIXES_THIS_CYCLE=$((FIXES_THIS_CYCLE+1))
    ALIVE=$(pgrep -f "train.py" > /dev/null && echo 1 || echo 0)
    LAST_ERR=$(grep -cE "OutOfMemoryError|ChildFailedError" "$LOG")
  fi
done

if [ "$ALIVE" -eq 1 ] && [ "$FRESH" -eq 1 ]; then
  echo "[$(stamp)] watchdog: healthy, re-arming for next cycle"
  exit 0
fi

echo "[$(stamp)] watchdog: STILL BROKEN after 2 repair attempts, giving up" | tee -a "$FIXLOG"
exit 2
