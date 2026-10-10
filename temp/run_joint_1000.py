"""Launch isolated 1000-step experiments and monitor them in foreground."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['launch', 'watch', 'summary'])
parser.add_argument('--seconds', type=int, default=3600)
args = parser.parse_args()
name = 'joint_scratch_1000_bounded_hard2_guard3_aligned'
log = ROOT / 'logs' / (name + '.log')
checkpoint = ROOT / 'data/checkpoints' / (name + '.pt')
state = ROOT / 'temp' / (name + '.state.json')

if args.mode == 'launch':
    if state.exists() or checkpoint.exists():
        raise SystemExit('Experiment already exists; inspect before relaunching')
    cmd = [str(ROOT / '.venv-pro6000/bin/torchrun'), '--standalone', '--nproc_per_node=2',
           'train.py', '--train-exit-joint', '--dataset-dir', 'data/tokens_v2',
           '--tokenizer', 'tokenizer/qwen', '--sequence-length', '2048',
           '--batch-size', '2', '--gradient-accumulation', '1', '--precision', 'amp',
           '--steps', '1000', '--lr', '1.5e-4', '--warmup-steps', '100',
           '--exit-joint-readout-lr', '1e-4', '--log-steps', '1', '--save-steps', '250',
           '--exit-hard-min-depth', '2', '--exit-min-depth', '3',
           '--joint-selection-entropy-beta', '0.05', '--grad-checkpoint',
           '--checkpoint', str(checkpoint)]
    env = dict(os.environ, PYTORCH_ALLOC_CONF='expandable_segments:True')
    with log.open('x') as output:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    state.write_text(json.dumps({'pid':proc.pid, 'cmd':cmd, 'started':time.time()}, indent=2))
    print('Started',name,'pid',proc.pid,flush=True)
elif args.mode == 'watch':
    info = json.loads(state.read_text()); start=time.monotonic(); last_size=-1; last_change=time.monotonic()
    while time.monotonic()-start < args.seconds:
        text=log.read_text(); size=log.stat().st_size
        if size!=last_size:
            last_change=time.monotonic(); last_size=size
        summaries=[line for line in text.splitlines() if line.startswith(('joint step=', 'joint_saved','joint_start','OOM-STORM'))]
        if summaries: print(summaries[-1],flush=True)
        for rank in (0,1):
            diagnostics=checkpoint.with_suffix(f'.rank{rank}.diagnostics.jsonl')
            if diagnostics.exists():
                lines=diagnostics.read_text().splitlines()
                if lines:
                    row=json.loads(lines[-1])
                    print(f"GPU{rank} micro={row['micro']} A1={row.get('loss_a1')} A2={row.get('loss_a2')} "
                          f"CE_B={row['sequence_ce']:.5f} depths={row['selected']} peak={row['peak_gib']:.2f}GiB",flush=True)
        if any(x in text for x in ('Traceback', 'FloatingPointError', 'OOM-STORM', 'OutOfMemoryError', 'non-finite')) or any(
            __import__('re').search(r'\b(?:oom|skipped)=[1-9]', line) for line in summaries):
            print(text[-6000:],flush=True); raise SystemExit('ALERT: training error')
        if 'joint_saved step=1000 ' in text:
            print('COMPLETE',flush=True); break
        try:
            os.kill(info['pid'],0)
        except ProcessLookupError:
            raise SystemExit('ALERT: process exited before final save')
        if time.monotonic()-last_change>900: raise SystemExit('ALERT: log stalled')
        time.sleep(30)
else:
    from collections import Counter
    all_rows=[]
    for rank in (0,1):
        path=checkpoint.with_suffix(f'.rank{rank}.diagnostics.jsonl')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        print('rank',rank,'records',len(rows),'depth_histogram',dict(Counter(d for r in rows for d in r['selected'])))
        print('router_gradient_steps',[sum(r['router_grad'][i] is not None and r['router_grad'][i]>0 for r in rows) for i in range(24)])
        all_rows+=rows
    for lo in range(0,1000,100):
        group=[r for r in all_rows if lo<r['micro']<=lo+100]
        if group:
            depths=[d for r in group for d in r['selected']]
            print('steps',lo+1,lo+100,'mean_depth',round(sum(depths)/len(depths),3),
                  'max_depth',max(depths),'sequence_ce',round(sum(r['sequence_ce'] for r in group)/len(group),4),
                  'depth_histogram',dict(sorted(Counter(depths).items())),
                  'deep_16plus_pct',round(100*sum(d>=16 for d in depths)/len(depths),2),
                  'A1',round(sum(r['loss_a1'] for r in group)/len(group),4),
                  'A2',round(sum(r['loss_a2'] for r in group)/len(group),4))
