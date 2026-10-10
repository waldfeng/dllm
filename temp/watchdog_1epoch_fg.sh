#!/bin/bash
set -uo pipefail
cd /home/src/dllm
LOG=logs/joint_scratch_1epoch.log
CKPT=data/checkpoints/joint_scratch_1epoch.pt
PIDFILE=temp/joint_1epoch.pid
TOTAL_MIB=97887
FLOOR_MIB=$(( TOTAL_MIB * 20 / 100 ))
STALE=900
HOURLY=3600
POLL=10
STEP_REPORT=10

report() {
  python3 - "$LOG" <<'PYEOF'
import sys, re, collections, json
from pathlib import Path
try:
    lines = open(sys.argv[1], errors='ignore').read().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('joint_start')]
    if starts:
        lines = lines[starts[-1]:]
except OSError:
    sys.exit(0)
def fields(l):
    d = {}
    for tok in l.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            try: d[k] = float(v.removesuffix('GiB'))
            except ValueError: pass
    return d
steps = [fields(l) for l in lines if l.startswith('joint step=')]
if steps:
    n = len(steps)
    f, p, last = steps[0], steps[max(0, n - 101)], steps[-1]
    g = lambda d, k, alt=0.0: d.get(k, alt)
    print(f"steps_logged={n} elapsed_h={g(last,'elapsed_h'):.2f} step_s={(g(last,'elapsed_h')*3600/max(n,1)):.1f}")
    print(f"B_CE: init={g(f,'lossB_ce'):.3f} -100steps={g(p,'lossB_ce'):.3f} now={g(last,'lossB_ce'):.3f}")
    print(f"lossA: init={g(f,'lossA'):.4f} now={g(last,'lossA'):.4f}")
    print(f"depth: init={g(f,'depth'):.2f} -100steps={g(p,'depth'):.2f} now={g(last,'depth'):.2f} E[t]={g(last,'E[t]'):.2f}")
    print(f"grad={g(last,'grad'):.3f} lrB={g(last,'lrB'):.2e} lrA={g(last,'lrA'):.2e} peak={g(last,'peak'):.1f}GiB oom={int(g(last,'oom'))} skipped={int(g(last,'skipped'))}")
a = [fields(l) for l in lines if l.startswith('joint_A') and 'rank=0' in l]
if a:
    print(f"A1={a[-1].get('A1',0):.4f} A2={a[-1].get('A2',0):.4f} entropy={a[-1].get('entropy',0):.6f}")
sel = []
for rank in (0, 1):
    path = Path('data/checkpoints/joint_scratch_1epoch.pt').with_suffix(f'.rank{rank}.diagnostics.jsonl')
    if path.exists():
        with path.open() as stream:
            rows = collections.deque(stream, maxlen=100)
        for line in rows:
            try:
                sel.extend(json.loads(line).get('selected', []))
            except json.JSONDecodeError:
                pass
if sel:
    c = collections.Counter(sel)
    print('depth_hist(recent %d samples, both GPUs):' % len(sel), ' '.join(f'{d}:{k}' for d, k in sorted(c.items())))
PYEOF
}

offset=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
last_change=$(date +%s)
start=$(date +%s)
next_hourly=$(( start + HOURLY ))
last_report_bucket=-1
if [ -f "$LOG" ] && tail -40 "$LOG" | grep -qE 'Traceback|OutOfMemoryError|FloatingPointError|OOM-STORM|non-finite|RuntimeError|ValueError|AssertionError'; then
  echo "[$(date '+%F %T')] ALERT log already contains errors at startup -> wake up for repair"
  tail -60 "$LOG"
  exit 8
fi
if ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  if [ -f "$LOG" ] && grep -qE 'joint_saved step=30429|max_runtime_stopped' "$LOG"; then
    echo "[$(date '+%F %T')] DONE training completed"
    report
    exit 0
  fi
  echo "[$(date '+%F %T')] ALERT trainer already dead at startup -> wake up for repair"
  tail -60 "$LOG" 2>/dev/null || true
  exit 8
fi
echo "[$(date '+%F %T')] watchdog START log=$LOG floor=${FLOOR_MIB}MiB stall=${STALE}s poll=${POLL}s hourly_exit=${HOURLY}s"
while true; do
  now=$(date +%s)
  latest_step=$(grep -E '^joint step=' "$LOG" 2>/dev/null | tail -1)
  step_n=$(printf '%s' "$latest_step" | sed -n 's/^joint step=\([0-9]*\).*/\1/p')
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
  oom_n=$(printf '%s' "$latest_step" | sed -n 's/.* oom=\([0-9]*\).*/\1/p')
  skip_n=$(printf '%s' "$latest_step" | sed -n 's/.* skipped=\([0-9]*\).*/\1/p')
  if [ -n "$oom_n" ] && [ "$oom_n" -gt 0 ]; then
    echo "[$(date '+%F %T')] ALERT OOM detected (oom=$oom_n) -> wake up for repair"
    report
    echo "$latest_step"
    tail -60 "$LOG"
    exit 8
  fi
  if [ -n "$skip_n" ] && [ "$skip_n" -gt 0 ]; then
    echo "[$(date '+%F %T')] ALERT skipped microbatch (skipped=$skip_n) -> wake up for repair"
    report
    echo "$latest_step"
    tail -60 "$LOG"
    exit 8
  fi
  if [ -f "$LOG" ]; then
    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$size" -gt "$offset" ]; then
      new=$(tail -c +$((offset + 1)) "$LOG")
      if printf '%s' "$new" | grep -qE 'Traceback|OutOfMemoryError|FloatingPointError|OOM-STORM|non-finite|RuntimeError|ValueError|AssertionError'; then
        echo "[$(date '+%F %T')] ALERT training exception -> wake up for repair"
        printf '%s' "$new" | grep -nE 'Traceback|OutOfMemoryError|FloatingPointError|OOM-STORM|non-finite|RuntimeError|ValueError|AssertionError' | head -5
        report
        tail -60 "$LOG"
        exit 8
      fi
      offset=$size
      last_change=$now
    fi
  fi
  if ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    if [ -f "$LOG" ] && grep -qE 'joint_saved step=30429|max_runtime_stopped' "$LOG"; then
      echo "[$(date '+%F %T')] DONE training completed"
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
      echo "[$(date '+%F %T')] ALERT GPU${idx} memory ${used}MiB < ${FLOOR_MIB}MiB (20%) -> wake up for repair"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
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
