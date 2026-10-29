#!/bin/bash
cd /home/src/dllm
while :; do
  bash temp/watchdog_pretrain_init_fg.sh
  code=$?
  echo "[$(date '+%F %T')] loop: watchdog exit=$code"
  if [ "$code" -eq 7 ]; then
    continue
  fi
  echo "[$(date '+%F %T')] loop STOPPED (exit=$code) -> agent must inspect, fix, and restart watchdog_loop"
  exit "$code"
done
