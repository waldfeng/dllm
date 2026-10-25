#!/bin/bash
set -uo pipefail
cd /home/src/dllm
LOG=checkpoints/logs/pretrain_1epoch_v2.log
PIDFILE=temp/pretrain_1epoch_v2.pid
STATUS=temp/pause_status_v2.txt
MARKER="checkpoint_saved step=2750"
echo "[$(date '+%F %T')] waiter START waiting for: $MARKER" > "$STATUS"
while true; do
  if grep -q "$MARKER" "$LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] marker found; letting save+barrier settle 30s" >> "$STATUS"
    sleep 30
    PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
    echo "[$(date '+%F %T')] SIGTERM trainer pid=$PID" >> "$STATUS"
    if [ -n "$PID" ]; then kill -TERM "$PID" 2>/dev/null; fi
    for i in $(seq 1 24); do
      sleep 5
      if ! kill -0 "$PID" 2>/dev/null; then break; fi
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo "[$(date '+%F %T')] still alive; SIGKILL" >> "$STATUS"
      kill -9 "$PID" 2>/dev/null
      sleep 10
    fi
    sleep 5
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
    echo "[$(date '+%F %T')] trainer stopped; GPU MiB: $MEM" >> "$STATUS"
    echo "PAUSED" >> "$STATUS"
    exit 0
  fi
  sleep 60
done
