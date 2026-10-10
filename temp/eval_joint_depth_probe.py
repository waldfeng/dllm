"""Matched unseen stream windows; not a separately curated validation corpus."""
import json
from pathlib import Path
import torch
from memory_transformer import MemoryTransformerConfig, MemoryTransformerForCausalLM, ensure_exit_readout
from pretrain_data import TokenStore, token_batches

torch.set_num_threads(1)
store = TokenStore(Path('data/tokens_v2'))
# Original trials consumed the first 120 samples of this exact permutation.
windows = list(token_batches(store, sequence_length=2048, batch_size=1,
                            steps=16, rank=0, world_size=2, start_step=1000, seed=42))
results = {}
for name in ('joint-scratch-2048-probe', 'joint-scratch-depth-ramp-probe'):
    payload = torch.load(f'data/checkpoints/{name}.pt', map_location='cpu', weights_only=False)
    model = MemoryTransformerForCausalLM(MemoryTransformerConfig(**payload['config']))
    ensure_exit_readout(model, rank=payload['exit_readout_rank'])
    model.load_state_dict(payload['model'])
    model.apply_exit_selection(payload['exit_loss_weights'].tolist(), payload['exit_temperature'])
    model.to('cuda').eval()
    del payload
    ces = {str(t): [] for t in (8,16,24)}
    depths = []
    exit_ce = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for inputs, labels in windows:
            ids = torch.from_numpy(inputs).cuda()
            target = int(labels[0, -1])
            rollout = model.joint_rollout(ids)
            for depth in (8,16,24):
                logits = model.exit_readout.logits_at(depth-1, rollout['states'][depth-1], model.lm_head.weight)
                ces[str(depth)].append(float(torch.nn.functional.cross_entropy(logits.float(), torch.tensor([target],device='cuda'))))
            out = model(ids, early_exit=True)
            depths.append(int(out['exit_depth']))
            exit_ce.append(float(torch.nn.functional.cross_entropy(out['logits'][:,-1].float(), torch.tensor([target],device='cuda'))))
            del out, rollout
    results[name] = {'windows':len(windows), 'last_token_ce':{t:sum(v)/len(v) for t,v in ces.items()},
                     'online_exit_ce':sum(exit_ce)/len(exit_ce), 'online_exit_depths':depths}
    print(name, results[name], flush=True)
    del model
    torch.cuda.empty_cache()
Path('logs/joint_depth_probe_evaluation.json').write_text(json.dumps(results,indent=2))
