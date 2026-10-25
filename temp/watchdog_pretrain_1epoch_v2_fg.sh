#!/bin/bash
set -uo pipefail
cd /home/src/dllm
LOG=checkpoints/logs/pretrain_1epoch_v2.log
CKPT=checkpoints/pretrain_1epoch_v2.pt
PIDFILE=temp/pretrain_1epoch_v2.pid
TARGET_STEPS=59374
TOTAL_MIB=97887
FLOOR_MIB=$(( TOTAL_MIB * 85 / 100 ))
STALE=900
HOURLY=3600
POLL=10
STEP_REPORT=100

report() {
  python3 - "$LOG" "$TARGET_STEPS" "$PIDFILE" <<'PYEOF'
import sys, time, os
from pathlib import Path
log = Path(sys.argv[1])
target = int(sys.argv[2])
try:
    lines = log.read_text(errors='ignore').splitlines()
except OSError:
    sys.exit(0)
def fields(l):
    d = {}
    for tok in l.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            try: d[k] = float(v)
            except ValueError: pass
    return d
steps = [fields(l) for l in lines if l.startswith('step=')]
if not steps:
    print('steps_logged=0')
    sys.exit(0)
n = len(steps)
f, p, last = steps[0], steps[max(0, n - 101)], steps[-1]
g = lambda d, k, alt=0.0: d.get(k, alt)
last_step = int(g(last, 'step'))
birth = 0
try:
    birth = os.stat(sys.argv[3]).st_mtime
except OSError:
    pass
eta_h = ''
if birth and last_step > 0:
    rate = (time.time() - birth) / last_step
    eta_h = f' eta_h={(target - last_step) * rate / 3600.0:.1f}'
print(f"steps_logged={n} step={last_step}/{target}{eta_h}")
print(f"loss: init={g(f,'loss'):.3f} -100={g(p,'loss'):.4f} now={g(last,'loss'):.4f}")
print(f"lr={g(last,'lr'):.2e} tokens_per_s={g(last,'tokens_per_s'):.0f} "
      f"peak={g(last,'peak_gib'):.1f}GiB alloc={g(last,'alloc_gib'):.1f} reserved={g(last,'reserved_gib'):.1f}")
print(f"grad_health: rl_min={g(last,'rl_min'):.3f} rl_max={g(last,'rl_max'):.3f} "
      f"rb_dead={int(g(last,'rb_dead'))}/24 rb_max={g(last,'rb_max'):.3f}")
PYEOF
}

offset=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
last_change=$(date +%s)
start=$(date +%s)
next_hourly=$(( start + HOURLY ))
last_report_bucket=-1
if [ -f "$LOG" ] && tail -40 "$LOG" | grep -qE 'Traceback|OutOfMemoryError|FloatingPointError|OOM-STORM|non-finite|RuntimeError|ValueError|AssertionError|ChildFailedError|CUDA error'; then
  echo "[$(date '+%F %T')] ALERT log already contains errors at startup -> wake up for repair"
  tail -60 "$LOG"
  exit 8
fi
if ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  if [ -f "$LOG" ] && grep -qE "checkpoint_saved final_step=$TARGET_STEPS" "$LOG"; then
    echo "[$(date '+%F %T')] DONE training completed ($TARGET_STEPS steps = 1 epoch)"
    report
    exit 0
  fi
  echo "[$(date '+%F %T')] ALERT trainer already dead at startup -> wake up for repair"
  tail -60 "$LOG" 2>/dev/null || true
  exit 8
fi
echo "[$(date '+%F %T')] watchdog START log=$LOG floor=${FLOOR_MIB}MiB(85%) stall=${STALE}s poll=${POLL}s hourly_exit=${HOURLY}s"
while true; do
  now=$(date +%s)
  latest_step=$(grep -E '^step=' "$LOG" 2>/dev/null | tail -1)
  step_n=$(printf '%s' "$latest_step" | sed -n 's/^step=\([0-9]*\).*/\1/p')
  if [ -n "$step_n" ]; then
    bucket=$(( step_n / STEP_REPORT ))
    if [ "$last_report_bucket" -lt 0 ]; then
      last_report_bucket=$bucket
    elif [ "$bucket" -gt "$last_report_bucket" ]; then
      echo "[$(date '+%F %T')] STEP CHECK-IN step=$step_n (every ${STEP_REPORT} steps)"
      echo "$latest_step"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
      report
      last_report_bucket=$bucket
    fi
  fi
  if [ -f "$LOG" ]; then
    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$size" -gt "$offset" ]; then
      new=$(tail -c +$((offset + 1)) "$LOG")
      if printf '%s' "$new" | grep -qE 'Traceback|OutOfMemoryError|FloatingPointError|OOM-STORM|non-finite|RuntimeError|ValueError|AssertionError|ChildFailedError|CUDA error|loss=nan|loss=inf'; then
        echo "[$(date '+%F %T')] ALERT training exception -> wake up for repair"
        printf '%s' "$new" | grep -nE 'Traceback|OutOfMemoryError|FloatingPointError|OOM-STORM|non-finite|RuntimeError|ValueError|AssertionError|ChildFailedError|CUDA error|loss=nan|loss=inf' | head -5
        report
        tail -60 "$LOG"
        exit 8
      fi
      offset=$size
      last_change=$now
    fi
  fi
  if ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)"; then
    if [ -f "$LOG" ] && grep -qE "checkpoint_saved final_step=$TARGET_STEPS" "$LOG"; then
      echo "[$(date '+%F %T')] DONE training completed ($TARGET_STEPS steps = 1 epoch)"
      report
      exit 0
    fi
    echo "[$(date '+%F %T')] ALERT trainer process exited before completion -> wake up for repair"
    report
    tail -60 "$LOG" 2>/dev/null || true
    exit 8
  fi
  if [ $((now - last_change)) -gt "$STALE" ]; then
    echo "[$(date '+%F %T')] ALERT no log progress for 15 minutes -> wake up for repair"
    report
    tail -40 "$LOG" 2>/dev/null || true
    exit 8
  fi
  while IFS= read -r line; do
    idx=${line%%,*}
    used=$(printf '%s' "$line" | cut -d, -f2 | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$FLOOR_MIB" ]; then
      echo "[$(date '+%F %T')] ALERT GPU${idx} memory ${used}MiB < ${FLOOR_MIB}MiB (85%) -> wake up for repair"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
      report
      tail -20 "$LOG" 2>/dev/null || true
      exit 8
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if [ "$now" -ge "$next_hourly" ]; then
    echo "[$(date '+%F %T')] HOURLY CHECK-IN (routine)"
    echo "$latest_step"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU /'
    report
    exit 7
  fi
  sleep "$POLL"
done
