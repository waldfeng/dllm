# Memory Transformer Design

## Requirement Boundary

The supplied Intern-S2-Mobius paper describes a globally shared FFN knowledge
memory. Layer-local FFN weights are removed from individual Transformer layers,
placed in one shared memory, and accessed by the Self-Attention reasoner. The
paper also describes repeatedly applying a shared reasoner to a hidden-state
carrier.

The paper does **not** specify routing Q/K/V/O projections, 24 blocks, or
top-3 routing. Those are explicit requirements of this implementation.

## Model Topology

`SharedMemory` owns four banks, each with exactly 24 addressable blocks:

- FFN blocks: SwiGLU gate, up, and down matrices stay paired by block;
- Q blocks: query projection matrices;
- KV blocks: key and value projection matrices;
- O blocks: attention output projection matrices.

The model has exactly 24 reasoner cycles. Each cycle has its own RMSNorm and
four-head router, then accesses the same global Memory banks. Every router uses
differentiable top-3 selection from the 24 blocks. K and V use the same selected
indices and weights from the KV route head.

The selected matrices are dispatched by block. The implementation never creates
a `[batch, sequence, top_k, input, output]` copy of a bank. FFN dispatch computes
each selected block's complete gate/up/down path before mixing the block outputs,
so the three FFN matrices remain semantically paired.

## Routing Objective

The causal loss is combined with a Switch-style importance/load objective:

```text
total_loss = causal_cross_entropy + 0.01 * router_loss
```

The importance term uses the full softmax probability over all 24 blocks, so
unselected logits receive an auxiliary gradient. The load term uses the actual
hard top-3 block IDs. Both are computed independently for each of the four route
heads and then averaged.

## Initialization

Every memory bank is initialized one two-dimensional matrix at a time. All
trainable projection matrices, including FFN gate/up/down, Q/K/V/O, routers, and
the language-model head, use Xavier uniform initialization. SiLU and RMSNorm are
not trainable linear layers: SiLU has no parameters to initialize and RMSNorm
starts with a unit scale. Applying an initializer to the complete three-dimensional
bank would incorrectly treat the block axis as a convolution dimension and make
the weights much too small.

## Default Scale

```text
vocab 151,665
hidden 1,536
intermediate 4,096
reasoner cycles 24
attention heads 16
KV heads 8
memory blocks 24
top-k 3
```

The exact default parameter count is `1,092,383,232` (1.09B), including two
untied embedding matrices, the four memory banks, 24 routers, all cycle norms,
and the final norm. It remains inside the requested 1B-1.5B range.

## Early Exit And Calibrated Readout

After each reasoner cycle the model evaluates a weighted halting cost on the
last token position; the exit depth is the argmax of the depth-selection
distribution over the whole loop (batch-level):

```text
l_t = pi_conf   * max(0, tau_c - p_t)   # top-1 probability deficit
    + pi_margin * max(0, tau_m - m_t)   # m_t = p1 - p2 probability margin
    + pi_delta  * max(0, d_t - tau_d)   # normalized state change (mask-averaged)
    + pi_speed  * t / num_depths        # speed pressure
    + pi_guard  * max(0, min_depth - t) / num_depths
rho = softmax(-l_t / temperature)
exit depth = argmax_t rho_t             # equivalently the smallest l_t
```

Online the loop tracks the running minimum of `l_t` and halts at the first
depth whose cost rises above it; depths below `min_depth` are never exit
candidates, and the shortlist cross-entropy term is dropped because it has no
ground truth at inference (its annealed prior mass is zero, so the training
and inference depth-selection distributions coincide).

`CalibratedExitReadout` is an optional per-depth correction on the frozen
`lm_head`: `logits_t = s_t * (W h + B_t (A_t h) + c_t)` with rank-16 LoRA
(`A_t`, `B_t`), a full-vocabulary bias `c_t`, and a per-depth temperature
`s_t`.  `B_t` and `c_t` start at zero and `s_t` at one, so an unfitted readout
is bit-identical to the shared head.  Only readout parameters train in the
continued-training mode (`train.py --train-exit-readout`); the backbone is
frozen and the rollout runs under `no_grad`.

The readout objective sums eight terms, every one minimized and normalized
into [0, 1]: six per-depth terms (sigmoid of shortlist CE, confidence deficit
`max(0, tau_c - p_t)`, probability-margin deficit, state-delta deficit,
speed `t/24`, min-depth guard `max(0, 8 - t)/24`) combined by a learnable
six-way weight distribution `pi = softmax(theta)`; term seven aligns
`softmax(weighted terms)` with the fixed prior
`[0.65, 0.05, 0.05, 0.05, 0.15, 0.05]` through a sigmoid (the prior's CE mass
anneals to 0.0 so the training depth-selection distribution is identical to
the inference-time one, because the CE term does not exist without a
ground-truth token); term eight sharpens the depth-selection distribution
`rho = softmax(-l_t / T)` and normalizes its entropy by `log(depths)`.  The
shortlist CE uses the top-64 teacher tokens plus the true token, so per-depth
supervision costs almost nothing.

The trained six-way weights and the final temperature are saved inside the
exit checkpoint (`exit_loss_weights`, `exit_temperature`); `test.py` restores
them and enables the depth-selection halting automatically for two-stage
checkpoints.

```bash
torchrun --standalone --nproc_per_node=2 train.py ...   # pretraining (unchanged)
python train.py --train-exit-readout --resume <base.pt> \
  --checkpoint datasets/checkpoints/exit-readout.pt --steps 2000 \
  --exit-readout-rank 16 --device auto
python test.py --checkpoint datasets/checkpoints/exit-readout.pt \
  --benchmark-tokens 128
```

Disable the early exit with `--no-early-exit` (the default configuration runs
the full 24 cycles and is bit-identical to the unmodified model).  Design
references: CALM (arXiv 2207.07061), Huginn deep supervision (arXiv
2502.05171), post-hoc readout diagnostics (arXiv 2607.20519), FPRM
fixed-point halting (arXiv 2606.18206), RecurTrace (arXiv 2609.03379).

## Data And Training

1. Install dependencies with the required Tsinghua mirror:

```bash
uv venv .venv --python 3.10
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --python .venv -e ".[dev]"
```

2. Download selected source shards if they are not already present:

```bash
.venv/bin/python scripts/download_pretrain_datasets.py \
  --output datasets \
  --workers 16
```

3. Tokenize every JSONL/JSONL.ZST document and require at least 20B tokens:

```bash
.venv/bin/python scripts/prepare_pretrain_data.py \
  --input-dir datasets \
  --output-dir datasets/tokens \
  --tokenizer tokenizer/qwen \
  --local-files-only \
  --min-tokens 20000000000
```

The preprocessing step appends the Qwen EOS token to each document, writes
uint32 mmap shards, and records the exact token count in
`datasets/tokens/manifest.json`. The training loader reads windows directly
from these shards and assigns disjoint batches to ranks.

The two-GPU training entry point uses bf16, FSDP, gradient accumulation,
cosine decay with warmup, gradient clipping, atomic checkpoint saves, and
checkpoint resume. The production 12-hour run uses the largest batch that
was safe in a real forward/backward/Adam benchmark, then backs off the
effective token batch by one eighth to leave GPU headroom:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --dataset-dir datasets/tokens \
  --tokenizer tokenizer/qwen \
  --steps 900 \
  --sequence-length 4608 \
  --batch-size 1 \
  --gradient-accumulation 7 \
  --lr 2.5e-4 \
  --warmup-steps 60 \
  --min-lr-ratio 0.1 \
  --precision bf16 \
  --save-steps 0 \
  --save-interval-hours 1 \
  --checkpoint datasets/checkpoints/memory-transformer-12h.pt \
  --log-steps 10
```

This configuration has an effective global batch of 64,512 tokens per
optimizer step. The measured throughput is about 1,348 tokens/second across
both GPUs, giving about 12 hours for 900 steps. The checkpoint path is
atomically overwritten once per wall-clock hour; training output should be
redirected to a persistent log file such as `logs/train_12h.log`.

The reference implementation is correctness-oriented. The block dispatch is
memory-safe but not a fused production kernel; benchmark throughput before
committing to a 20B-token run.
