#!/bin/bash
# FOREGROUND watchdog for the joint exit training run.
# Streams every logged step line, enforces >=85% GPU memory (torch-side mem=
# field, not nvidia-smi which over-reports due to allocator cache), and on
# crash/hang/OOM-storm STOPS and wakes the operator (exit codes below).
# Policy (operator decision): an OOM storm is a CONFIG problem — never
# auto-restart; crash without storm signature gets ONE resume attempt.
# Exit codes: 7 = repaired-and-recovered or utilization below floor (re-arm),
#             8 = OOM storm / fatal, operator must fix before relaunching.
cd /home/src/dllm
LOG=logs/train_joint_exit_41k.log
stamp() { date "+%F %T"; }
off=$(stat -c %s "$LOG")
last_step_time=$(date +%s)
GRACE=${WATCHDOG_GRACE:-540}
storm_report() {
  echo "[$(stamp)] ===== OOM-STORM / FATAL REPORT ====="
  grep "OOM-STORM" "$LOG" | tail -3
  grep "^joint step" "$LOG" | tail -5
  tail -30 "$LOG" | grep -vE "^joint step" | tail -12
  echo "[$(stamp)] training stopped; fix config (batch size) before relaunch"
}
while true; do
  new=$(stat -c %s "$LOG")
  if [ "$new" -gt "$off" ]; then
    tail -c +"$((off + 1))" "$LOG" | grep -E "^joint step|OOM-STORM|max_runtime_stopped"
    off=$new
    last_step_time=$(date +%s)
    # OOM storm circuit breaker fired inside the trainer (exit 42)
    if tail -40 "$LOG" | grep -q "OOM-STORM"; then
      echo "[$(stamp)] ALERT: trainer aborted itself on OOM storm (exit 42)"
      storm_report
      pkill -9 -f "torchrun" 2>/dev/null
      exit 8
    fi
    # empty-stats signature: two consecutive dead windows (no CE at all)
    if [ "$(tail -2 "$LOG" | grep -c 'lossB_ce=0.00000 .*E\[t\]=0.00')" -eq 2 ]; then
      echo "[$(stamp)] ALERT: two consecutive steps with empty stats (every micro OOM-skipped)"
      storm_report
      pkill -9 -f "torchrun" 2>/dev/null
      exit 8
    fi
    # memory floor: torch-side utilization from the mem= field
    last_mem=$(grep "^joint step" "$LOG" | tail -1 | grep -oE "mem=[0-9.]+/[0-9]+GiB\(([0-9]+)%\)" | grep -oE "\([0-9]+%\)" | tr -d '(%)')
    if [ -n "$last_mem" ] && [ "$last_mem" -lt 85 ] && [ "$(grep -c "^joint step" "$LOG")" -gt 3 ]; then
      echo "[$(stamp)] ALERT: torch-side GPU memory ${last_mem}% < 85% floor"
      grep "^joint step" "$LOG" | tail -1
      exit 7
    fi
  fi
  if ! pgrep -f "train.py" > /dev/null; then
    now=$(date +%s)
    if [ $((now - last_step_time)) -lt "$GRACE" ] && [ "$(grep -c '^joint step' "$LOG")" -le 3 ]; then
      sleep 30   # still within startup grace; torchrun may just be loading weights
      continue
    fi
    echo "[$(stamp)] ALERT: trainer process gone"
    if tail -60 "$LOG" | grep -qE "OOM-STORM|exitcode  : 42"; then
      storm_report
      exit 8
    fi
    # non-storm crash (e.g. transient NCCL): one silent resume attempt
    echo "[$(stamp)] non-storm crash: one resume attempt from latest checkpoint"
    bash temp/launch_joint.sh 14
    for i in $(seq 1 80); do
      sleep 7
      if [ "$(stat -c %s "$LOG")" -gt "$off" ] && pgrep -f "train.py" > /dev/null; then
        sleep 30
        echo "[$(stamp)] RECOVERED (resumed from checkpoint)"
        grep "^joint step" "$LOG" | tail -1
        exit 7
      fi
      pgrep -f "train.py" > /dev/null || break
    done
    echo "[$(stamp)] FATAL: resume attempt made no progress in ~9min"
    storm_report
    exit 8
  fi
  now=$(date +%s)
  if [ $((now - last_step_time)) -gt 420 ]; then
    echo "[$(stamp)] ALERT: no new logged step for >7min while process alive (hang)"
    grep "^joint step" "$LOG" | tail -1
    storm_report
    pkill -9 -f "torchrun" 2>/dev/null
    exit 8
  fi
  sleep 5
done
