"""Matched real A/B updates; isolated per mode/depth/device processes."""
import argparse
import json
import time
from pathlib import Path
import torch
from memory_transformer import MemoryTransformerConfig, MemoryTransformerForCausalLM, ensure_exit_readout
from pretrain_data import TokenStore, token_batches
from train import assemble_exit_terms

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['recompute', 'reuse'], required=True)
p.add_argument('--depth', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(42)
payload = torch.load('data/checkpoints/joint_scratch_1000_sigmoid_natural.pt', map_location='cpu', weights_only=False)
m = MemoryTransformerForCausalLM(MemoryTransformerConfig(**payload['config']))
ensure_exit_readout(m, rank=payload['exit_readout_rank'])
m.load_state_dict(payload['model'])
raw = torch.nn.Parameter(payload['loss_weights_raw'].cuda())
del payload
m.cuda().train()
rp = list(m.exit_readout.parameters())
bp = [v for n,v in m.named_parameters() if not n.startswith('exit_readout.')]
oa = torch.optim.AdamW(rp+[raw], lr=1e-4)
ob = torch.optim.AdamW(bp, lr=1.5e-4)
args = argparse.Namespace(exit_min_depth=8, exit_tau_confidence=0.9, exit_tau_margin=0.3, exit_tau_delta=0.05)
prior = torch.tensor([.65,.05,.05,.05,.15,.05], device='cuda')
store = TokenStore(Path('data/tokens_v2'))
windows = list(token_batches(store, sequence_length=2048, batch_size=1, steps=4, seed=42, world_size=2, rank=0, start_step=2000))
measurements = []
for step,(x,y) in enumerate(windows):
    oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    ids=torch.from_numpy(x).cuda(); labels=torch.from_numpy(y).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if a.mode=='recompute':
            rollout=m.joint_rollout(ids)
            cache={k:v.clone() for k,v in rollout.items()}
            del rollout
            saved=None
        else:
            # Keep the graph only for sequence states/router losses. All A-phase
            # observations and base-head products are explicitly detached.
            h=m.embed_tokens(ids); saved=[]; losses=[]
            states=[];deltas=[];prev=None
            for layer in m.layers:
                h,indices,_,prob=layer(h,m.memory)
                losses.append(m.load_balance_loss(indices,prob,None))
                normed=m.norm(h);saved.append(normed)
                detached=normed.detach().float()
                states.append(normed[:,-1].detach())
                deltas.append(detached.new_zeros(ids.size(0)) if prev is None else
                              ((detached-prev).norm(dim=-1)/detached.norm(dim=-1).clamp_min(1e-6)).mean(-1))
                prev=detached
            with torch.no_grad():
                stacked=torch.stack(states)
                cache={'states':stacked,'deltas':torch.stack(deltas),'base_logits':torch.nn.functional.linear(stacked,m.lm_head.weight)}
            # These tensors are no longer needed beyond the retained graphs.
            del h,normed,detached,prev,states,deltas,indices,prob,stacked
    terms,_,_=assemble_exit_terms(m,cache,labels[:,-1],args,m.config.num_layers,base_logits=cache['base_logits'])
    weighted=raw.softmax(0)*terms.mean((1,2))
    la=weighted.sum()+torch.sigmoid(torch.nn.functional.cross_entropy(weighted.log()[None],prior[None]))
    la.backward()
    assert all(v.grad is None for v in bp), 'A-phase leaked to backbone'
    torch.nn.utils.clip_grad_norm_(rp+[raw],1.)
    oa.step();oa.zero_grad(set_to_none=True)
    del terms,weighted,cache
    for v in rp:v.requires_grad_(False)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if a.mode=='recompute':
            forward=m.forward_until(ids,num_cycles=a.depth)
            hidden=forward['final_normed'];router=forward['router_losses'][:a.depth].mean()
        else:
            hidden=saved[a.depth-1];router=torch.stack(losses[:a.depth]).mean()
        rows=hidden[:,:-1].reshape(-1,m.config.hidden_size)
        targets=labels[:,1:].reshape(-1)
        def ce_fn(hh, yy):
            logits=m.exit_readout.logits_at(a.depth-1,hh,m.lm_head.weight)
            return torch.nn.functional.cross_entropy(logits.float(),yy)
        ce=torch.utils.checkpoint.checkpoint(ce_fn,rows,targets,use_reentrant=False)
        lb=ce+.01*router
    lb.backward()
    assert all(v.grad is None for v in rp), 'B-phase updated readout'
    grad=torch.nn.utils.clip_grad_norm_(bp,1.)
    ob.step()
    torch.cuda.synchronize()
    measurements.append({'step':step,'loss_a':float(la.detach()),'loss_b':float(lb.detach()),
                         'grad_norm':float(grad),'seconds':time.perf_counter()-start,
                         'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                         'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
    for v in rp:v.requires_grad_(True)
    del la,lb,ce,router,hidden,rows,targets
    if a.mode=='reuse': del saved,losses
    else: del forward
    oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
    print(a.mode,a.depth,measurements[-1],flush=True)
a.output.write_text(json.dumps({'mode':a.mode,'depth':a.depth,'device':torch.cuda.get_device_name(0),
                               'measurements':measurements},indent=2))
