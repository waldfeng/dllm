"""Probe the production dtype flow: autocast rollout, print dtypes in the
legacy routed_linear, and check whether scatter_add survives the mix."""
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, "/home/src/dllm")

import train as T
from pretrain_data import TokenStore, token_batches
from memory_transformer import SharedMemory

CKPT = "data/checkpoints/joint_scratch_1epoch.pt"
sys.argv = [
    "probe", "--train-exit-joint", "--dataset-dir", "data/tokens_v2",
    "--tokenizer", "tokenizer/qwen", "--sequence-length", "2048",
    "--batch-size", "8", "--steps", "1", "--precision", "amp",
    "--grad-checkpoint", "--ce-chunk-rows", "4096", "--lr", "1.5e-4",
    "--warmup-steps", "600", "--exit-joint-readout-lr", "1e-4",
    "--exit-hard-min-depth", "2", "--exit-min-depth", "3",
    "--save-steps", "0", "--log-steps", "1", "--checkpoint", CKPT, "--seed", "42",
]
args = T.parse_args()
T.seed_everything(args.seed, 0)
store = TokenStore(args.dataset_dir)
vocab_size = int(store.manifest["vocab_size"])
config = T.build_config(args, vocab_size)
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
model = T.MemoryTransformerForCausalLM(config)
if any(n.startswith("exit_readout.") for n in payload.get("model", {})):
    T.ensure_exit_readout(model, rank=int(payload.get("exit_readout_rank", args.exit_readout_rank)))
model.load_state_dict(payload["model"])
model.to("cuda")
model.eval()
del payload
torch.cuda.empty_cache()
iterator = token_batches(store, sequence_length=2048, batch_size=8, steps=1,
                         rank=0, world_size=1, start_step=0, seed=42)
inputs, _ = next(iter(iterator))
input_tensor = torch.from_numpy(inputs).to("cuda")

printed = {"n": 0}
orig_rl = SharedMemory.routed_linear

def probe_rl(x, bank, indices, weights):
    x_flat = x.reshape(-1, x.shape[-1])
    weight_flat = weights.reshape(-1, weights.shape[-1])
    output = x_flat.new_zeros((x_flat.shape[0], bank.shape[-1]))
    for block_id in range(bank.shape[0]):
        locations = SharedMemory._locations(indices, block_id)
        if locations.numel() == 0:
            continue
        token_ids, slot_ids = locations.unbind(dim=1)
        block_input = x_flat.index_select(0, token_ids)
        block_output = F.linear(block_input, bank[block_id].transpose(0, 1))
        if printed["n"] < 3:
            print(f"    x={x.dtype} weights={weights.dtype} bank={bank.dtype} "
                  f"block_input={block_input.dtype} block_output={block_output.dtype} "
                  f"output_buf={output.dtype}", flush=True)
            printed["n"] += 1
        block_output = block_output * weight_flat[token_ids, slot_ids].to(
            block_output.dtype).unsqueeze(-1)
        scatter_ids = token_ids.unsqueeze(-1).expand(-1, output.shape[-1])
        if printed["n"] < 6:
            print(f"    after-mul block_output={block_output.dtype} "
                  f"scatter_add self={output.dtype} src={block_output.dtype}", flush=True)
            printed["n"] += 1
        output = output.scatter_add(0, scatter_ids, block_output)
    return output.reshape(*x.shape[:-1], bank.shape[-1])

SharedMemory.routed_linear = staticmethod(probe_rl)

print("== legacy rollout under autocast (production flow, B=8) ==", flush=True)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        out = model.joint_rollout(input_tensor)
print("legacy rollout OK; states dtype:", out["states"].dtype, flush=True)
