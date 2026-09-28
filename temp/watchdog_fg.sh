#!/bin/bash
# FOREGROUND watchdog: streams every logged training step (log-steps=10),
# enforces >=85% GPU memory utilization (nvidia-smi), repairs the run if it
# dies/hangs/under-utilizes, then EXITS so the caller wakes up and re-arms.
# Exit codes: 7 = wake caller (repaired or adjustment needed), 8 = fatal.
cd /home/src/dllm
LOG=logs/train_joint_exit_41k.log
stamp() { date "+%F %T"; }
gpu_mem_pct() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk -v t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) \
    'BEGIN{m=999} {p=$1/t*100; if(p<m)m=p} END{print int(m)}'
}
off=$(stat -c %s "$LOG")
last_step_time=$(date +%s)
last_healthy=$(date +%s)

# startup grace: model load + first step can take ~8 min
GRACE=${WATCHDOG_GRACE:-540}

while true; do
  new=$(stat -c %s "$LOG")
  if [ "$new" -gt "$off" ]; then
    tail -c +"$((off + 1))" "$LOG" | grep "^joint step"
    off=$new
    last_step_time=$(date +%s)
    # memory gate: only after grace period and only when training is mid-run
    now=$(date +%s)
    if [ $((now - last_healthy)) -gt "$GRACE" ]; then
      pct=$(gpu_mem_pct)
      if [ "$pct" -lt 85 ]; then
        echo "[$(stamp)] ALERT: GPU mem utilization ${pct}% < 85% (run needs a bigger batch)"
        grep "^joint step" "$LOG" | tail -1
        exit 7
      fi
    fi
  fi
  if ! pgrep -f "train.py" > /dev/null; then
    now=$(date +%s)
    if [ $((now - last_healthy)) -lt "$GRACE" ]; then
      echo "[$(stamp)] still starting up (grace), process vanished pre-first-step: relaunching"
      bash temp/launch_joint.sh 20
      sleep "$GRACE"
      last_healthy=$(date +%s)
      continue
    fi
    echo "[$(stamp)] ALERT: trainer process gone"
    grep "^joint step" "$LOG" | tail -1
    reason="unknown"; b=20
    if tail -120 "$LOG" | grep -q "OutOfMemoryError"; then b=16; reason=oom; fi
    echo "[$(stamp)] repairing: relaunch batch=$b reason=$reason"
    bash temp/launch_joint.sh "$b"
    for i in $(seq 1 80); do
      sleep 7
      if [ "$(stat -c %s "$LOG")" -gt "$off" ] && pgrep -f "train.py" > /dev/null; then
        sleep 30
        echo "[$(stamp)] RECOVERED (batch=$b)"
        grep "^joint step" "$LOG" | tail -1
        exit 7
      fi
      pgrep -f "train.py" > /dev/null || break
    done
    echo "[$(stamp)] FATAL: relaunch produced no progress in ~9min"
    exit 8
  fi
  now=$(date +%s)
  if [ $((now - last_step_time)) -gt 420 ]; then
    echo "[$(stamp)] ALERT: no new logged step for >7min while process alive (hang/oom-skip loop)"
    grep "^joint step" "$LOG" | tail -1
    echo "[$(stamp)] hang-repair: killing and resuming"
    pkill -9 -f "torchrun" 2>/dev/null
    sleep 5
    bash temp/launch_joint.sh 20
    sleep 420
    echo "[$(stamp)] hang-repair done"
    grep "^joint step" "$LOG" | tail -1
    exit 7
  fi
  sleep 5
done
