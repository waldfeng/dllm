#!/bin/bash
set -euo pipefail
cd /home/src/dllm
LOG=logs/train_pro6000_continuation.log
GRACE=${WATCHDOG_GRACE:-900}
FLOOR=${WATCHDOG_MEMORY_FLOOR:-90}
start=$(date +%s)
last=$(date +%s)
offset=0
echo "[$(date '+%F %T')] watchdog started log=$LOG memory_floor=${FLOOR}%"
while true; do
  now=$(date +%s)
  if [ -f "$LOG" ]; then
    size=$(stat -c %s "$LOG")
    if [ "$size" -gt "$offset" ]; then
      if tail -c +$((offset + 1)) "$LOG" | grep -qE 'Traceback|OutOfMemoryError|FloatingPointError|NCCL.*(error|timeout)|non-finite'; then
        echo "ALERT training exception: returning control for repair"
        tail -60 "$LOG"
        exit 8
      fi
      tail -c +$((offset + 1)) "$LOG" | grep -E '^(start |resume_data|step=|gpu_memory|checkpoint_saved|max_runtime_stopped)' || true
      offset=$size
      last=$now
    fi
  fi
  if ! kill -0 "$(cat temp/pro6000.pid)" 2>/dev/null; then
    if grep -qE 'checkpoint_saved final_step=25000|max_runtime_stopped' "$LOG"; then
      echo "training completed cleanly"; exit 0
    fi
    if [ $((now - start)) -lt "$GRACE" ]; then sleep 10; continue; fi
    echo "[$(date '+%F %T')] ALERT trainer process exited"
    tail -40 "$LOG" || true
    exit 8
  fi
  if [ $((now - last)) -gt 900 ]; then
    echo "[$(date '+%F %T')] ALERT no log progress for 15 minutes"
    tail -40 "$LOG" || true
    exit 8
  fi
  if [ -f "$LOG" ]; then
    mem=$(grep '^step=' "$LOG" | tail -1 | grep -oE 'peak_gib=[0-9.]+' | cut -d= -f2 || true)
    if [ -n "$mem" ]; then
      pct=$(awk -v m="$mem" 'BEGIN {printf "%d", (m/94.97)*100}')
      if [ "$pct" -lt "$FLOOR" ]; then
        echo "[$(date '+%F %T')] ALERT peak GPU memory ${pct}% below ${FLOOR}%"
        tail -5 "$LOG"
        exit 7
      fi
    fi
  fi
  sleep 10
done
