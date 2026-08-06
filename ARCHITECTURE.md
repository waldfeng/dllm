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

Every memory bank is initialized one two-dimensional matrix at a time. Q/K/V/O
matrices use Xavier uniform initialization. FFN gate/up/down matrices use Kaiming
uniform initialization. Applying an initializer to the complete three-dimensional
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

## Data And Training

1. Install dependencies with the required Tsinghua mirror:

```bash
uv venv .venv --python 3.10
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --python .venv -e ".[dev]"
```

2. Download selected source shards if they are not already present:

```bash
.venv/bin/python scripts/download_pretrain_datasets.py --output datasets
```

3. Tokenize every JSONL/JSONL.ZST document and require at least 20B tokens:

```bash
.venv/bin/python scripts/prepare_pretrain_data.py \
  --input-dir datasets \
  --output-dir datasets/tokens \
  --tokenizer temp/qwen-tokenizer \
  --local-files-only \
  --min-tokens 20000000000
```

The preprocessing step appends the Qwen EOS token to each document, writes
uint32 mmap shards, and records the exact token count in
`datasets/tokens/manifest.json`. The training loader reads windows directly
from these shards and assigns disjoint batches to ranks.

The two-GPU training entry point uses bf16, FSDP, gradient accumulation,
cosine decay with warmup, gradient clipping, atomic checkpoint saves, and
checkpoint resume:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --dataset-dir datasets/tokens \
  --tokenizer temp/qwen-tokenizer \
  --steps 100000 \
  --sequence-length 2048 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --precision bf16 \
  --checkpoint datasets/checkpoints/memory-transformer.pt
```

The reference implementation is correctness-oriented. The block dispatch is
memory-safe but not a fused production kernel; benchmark throughput before
committing to a 20B-token run.
