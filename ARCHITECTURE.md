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

At inference each sample runs independently. From cycle `d_min` on, the loop
halts at the first cycle whose last-position readout satisfies all three
thresholds; otherwise it runs to the full depth `L` and stops there
(``exit_halted`` stays false at full depth):

```text
halt at cycle t  iff  t >= d_min
                  and p_t >= tau_p          # top-1 probability
                  and m_t >= tau_m          # m_t = p1 - p2 probability margin
                  and delta_t <= tau_delta  # normalized state change
```

Depths below `d_min` are never exit candidates. Defaults: `d_min=2`,
`tau_p=0.6`, `tau_m=0.3`, `tau_delta=0.2`; the training-side delta scale
(`--exit-tau-delta 0.05`) is a separate parameter.

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
exit checkpoint (`exit_loss_weights`, `exit_temperature`) for diagnostics
only; the inference halting rule consumes just the thresholds above.  The
joint two-phase mode (`train.py --train-exit-joint`) selects per-sample
depths with the fixed six-term weights and updates the readout and the
backbone with two disjoint losses in one micro-batch.  `test.py` enables the
threshold-conjunction halting automatically for two-stage checkpoints.

```bash
torchrun --standalone --nproc_per_node=2 train.py ...   # pretraining (unchanged)
python train.py --train-exit-readout --resume <base.pt> \
  --checkpoint checkpoints/exit-readout.pt --steps 2000 \
  --exit-readout-rank 16 --device auto
python test.py --checkpoint checkpoints/exit-readout.pt \
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
  --output data/text_v2 \
  --workers 16
```

3. Tokenize every JSONL/JSONL.ZST document and require at least 20B tokens:

```bash
.venv/bin/python scripts/prepare_pretrain_data.py \
  --input-dir data/text_v2 \
  --output-dir data/tokens_v2 \
  --tokenizer tokenizer/qwen \
  --local-files-only \
  --min-tokens 20000000000
```

The preprocessing step appends the Qwen EOS token to each document, writes
uint32 mmap shards, and records the exact token count in
`data/tokens_v2/manifest.json`. The training loader reads windows directly
from these shards and assigns disjoint batches to ranks.

The two-GPU training entry point uses bf16, FSDP, gradient accumulation,
cosine decay with warmup, gradient clipping, atomic checkpoint saves, and
checkpoint resume. The production 12-hour run uses the largest batch that
was safe in a real forward/backward/Adam benchmark, then backs off the
effective token batch by one eighth to leave GPU headroom:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --dataset-dir data/tokens_v2 \
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
  --checkpoint checkpoints/memory-transformer-12h.pt \
  --log-steps 10
```

This configuration has an effective global batch of 64,512 tokens per
optimizer step. The measured throughput is about 1,348 tokens/second across
both GPUs, giving about 12 hours for 900 steps. The checkpoint path is
atomically overwritten once per wall-clock hour (all three training paths:
pretraining, joint, and legacy exit-readout); training output should be
redirected to a persistent log file such as `checkpoints/logs/train_12h.log`.

The reference implementation is correctness-oriented. The block dispatch is
memory-safe but not a fused production kernel; benchmark throughput before
committing to a 20B-token run.
