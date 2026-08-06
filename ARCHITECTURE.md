# Memory Transformer Design

## What the paper establishes

The supplied Intern-S2-Mobius paper describes FFN weights as a knowledge-vector
pool. It removes the layer-wise FFN binding, horizontally concatenates the FFN
knowledge vectors into one globally shared Memory, and gives every Self-Attention
reasoner access to that memory. At larger scale it partitions the memory into
blocks and sparsely activates blocks in an MoE-like manner. Hidden states are the
recurrent cache/carrier, so the same reasoner stack can be applied repeatedly for
latent refinement.

The paper does **not** specify routing Q/K/V/O projections, 24 blocks, or top-3
routing. Those are requirements of this implementation, not claims about the
paper's exact model.

## Implemented extension

`SharedMemory` has four address spaces:

- FFN blocks: gated-up projection and down projection
- Q blocks: query projection
- KV blocks: key and value projections share one route head
- O blocks: output projection

`MemoryReasoner.route` produces four independent distributions over 24 blocks.
Each distribution uses differentiable top-3 selection and softmax weights. K and V
consume the same selected block IDs and weights, as requested. The reasoner then
performs standard causal scaled dot-product attention and a SwiGLU FFN, with RMSNorm,
RoPE, GQA, causal language-model cross-entropy, and a router auxiliary loss.

`recurrent_steps > 1` reapplies the shared reasoner stack to the hidden state. This
is the practical looped latent-refinement analogue of the paper's recurrent cache;
it does not generate intermediate tokens.

## Initialization

All linear operators use Xavier uniform initialization: Q/K/V/O Memory projections,
the four-head router, and the language-model output head. The gated FFN Memory
projections (`gate`, `up`, and `down`) use Kaiming uniform initialization because
they feed the nonlinear SwiGLU path. Embeddings remain on a separate normal
initialization path.

## Default scale

The default configuration is:

```text
vocab 151,665, hidden 1,536, intermediate 4,096
layers 16, attention heads 16, KV heads 8, memory blocks 24, top-k 3
```

The configuration estimate is `1,089,650,688` parameters (1.09B), within the
requested 1B-1.5B range. This estimate includes two embedding matrices and the
four memory banks. Actual allocation should use bf16/fp16 and sharding or CPU/NVMe
offload for practical training; the current reference implementation is intended
for correctness and experimentation, not a fused production kernel.

## Training recipe

Use the fixed `Qwen/Qwen2.5-0.5B` tokenizer and causal labels shifted by one token.
Do not substitute another tokenizer: its vocabulary and token/byte ratio would
change both the model output head and the estimated training-token budget.
Optimize:

```text
total_loss = causal_cross_entropy + 0.01 * router_loss
```

The supplied `train.py` is a minimal smoke-training entry point. A production run
should add packed streaming data, bf16 autocast, gradient accumulation, distributed
FSDP/ZeRO, activation checkpointing, gradient clipping, checkpoint/resume, and a
capacity-aware dispatch kernel. For a more faithful Switch/MoE balance objective,
the router can additionally log raw top-k selection frequency per head and add the
standard importance-times-load auxiliary term.
