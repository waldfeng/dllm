# Dynamic Router Prototype

This is a fixed-capacity PyTorch prototype for a block-level dynamic router. It
intentionally avoids adding `nn.Module` objects during training: growth activates
a preallocated slot, so the optimizer, checkpoints, and DDP parameter set remain
stable.

## Design

- Inputs and outputs are blocks shaped `[B, 8, d_model]` by default.
- The default preallocated capacity is `max_modules=1024`; slots are activated
  incrementally, while the full parameter set is allocated from construction.
- The C4 trainer defaults to `d_model=256` and `n_heads=8` (32 dimensions per
  head). With `max_modules=1024`, this is approximately half the parameter
  budget of the previous `d_model=363` configuration.
- C4 preparation uses the fast `Qwen/Qwen2.5-0.5B-Instruct` tokenizer by
  default, with vocabulary length `151665`, EOS ID `151645`, and PAD ID `151643`.
- After token embedding or hidden-state input, the model adds a learned block
  position embedding `[8, d_model]`; the resulting `x_with_pos` is the shared
  input to each module's Q projection and GRU path.
- Each module has an independent recurrent GRU state shaped `[B, modules, d_model]`.
- The GRU state is passed between calls through `aux["memory_state"]` and detached by `train_step`.
- There are no permanent fixed edges: every active module can route to up to three
  distinct active modules or the output module at each forward pass. The input module
  is the sole external entry and the output module is the sole final exit.
- Multiple inputs to one module at the same depth are averaged before that module
  runs. A module may be visited again at a later depth and receives its updated state.
- Top-3 routing is hard in the forward pass and uses a softmax straight-through
  gradient for route logits. Only paths that reach output, or reach max depth and
  are then forced to output, are valid for training.
- During cooldown, the newly activated module runs in parallel with the triggering
  source on the same input; its residual output is weighted by an alpha that decays
  linearly from one to zero, after which normal routing resumes.
- During evaluation, the current implementation still uses the same candidate
  route semantics but uses deterministic argmax routing and evaluates only the
  selected candidate; inference is sparse over the active module graph.
- Each module projects raw `x` to Q, runs the same raw `x` through a standard
  GRU, and projects the GRU sequence to K and V. Attention is bidirectional and
  non-causal, with residuals around both attention and FFN.
- Training uses sequential block prediction within each text sample: an input
  block `[x0...x7]` produces eight logits trained against `[x8...x15]`, then
  `[x8...x15]` is fed as the next input for that sample. The GRU state is reset
  between text samples, and padding targets are masked out.
- Dataset preparation stores text boundaries in the mmap-able `samples.offsets`
  uint64 sidecar rather than a large JSON length array.
- The differentiable route cost uses the expected survival length.
- Growth monitors normalized time differences of route-logit gradients, a cheap
  second-order proxy rather than a full Hessian. A source must remain below the
  configurable velocity threshold for the configured patience before growth.

## Minimal use

The C4 trainer reads the prepared dataset from `datasets/c4_qwen2.5` by
default. It resumes from
`datasets/checkpoints/qwen10b-epoch1-b8-c32-top1-batched-w1e-4.pt`, writes the
updated checkpoint to `datasets/checkpoints/`, and writes normal and NaN/Inf
logs to `datasets/logs/`. Override these locations with `--dataset-dir`,
`--resume-checkpoint`, `--checkpoint`, `--log-file`, or `--nan-log-file` when
needed.

Install and run through the project virtual environment with the Tsinghua
package index:

```bash
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
.venv/bin/python -m pytest
uv run --no-sync train-c4 --device cuda
```

```python
import torch
from dynamic_router import DynamicRouterLM, GrowthController, train_step

model = DynamicRouterLM(d_model=128, vocab_size=32000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
controller = GrowthController()
memory = None

for x, y in loader:
    logits, aux = model(x, memory_state=memory, tau=1.0)
    memory = aux["memory_state"].detach()  # truncated BPTT between blocks
    # Or use train_step(model, optimizer, x, y, controller) for one-step training.
```

`train_step` returns a detached memory state for the next block. This is
truncated BPTT with truncation length 1: the next block receives the previous
hidden state value, but the previous block's computation graph is detached.
The C4 smoke
trainer lays out each batch as fixed lanes over independent text samples so
memory is continuous within each lane. A lane is zeroed when its sample ends;
text samples never cross boundaries. `route_lengths`
counts route decisions, including the transition to output; `expected_length`
is the differentiable soft survival cost accumulated along the sampled execution
path. The two are deliberately kept separate because sampled integer paths do
not have ordinary gradients. Growth only activates a new slot; it never edits
an edge.
