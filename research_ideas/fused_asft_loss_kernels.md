# Fused ASFT Weighted Loss Kernels — Research Idea

## Overview

During training of the GUI-Libra replication (Gemma 3 4B + LoRA, ASFT loss), the primary GPU memory
bottleneck was the cross-entropy computation in `ASFTTrainer.compute_loss`. Even with dynamic padding
reducing sequence lengths from 2048 to ~400–500 tokens, the standard PyTorch cross-entropy must
materialize a full `(batch × seq_len, vocab_size)` logit tensor before computing the loss.

For Gemma 3 with a 256K vocabulary, a single example at 450 tokens produces:

```
(1 × 450, 256000) × 2 bytes (bfloat16) ≈ 230 MB
```

A fused kernel approach eliminates this tensor entirely by computing the loss in chunks, never
staging the full logit matrix in GPU SRAM or HBM.

---

## Current Training Pipeline (Baseline)

```
ASFTTrainer.compute_loss:
  1. model(**inputs)                    → logits: (batch, seq_len, vocab)
  2. shift_logits = logits[..., :-1]   → allocates (batch, seq_len-1, vocab)
  3. CrossEntropyLoss(reduction='none') → allocates (batch × seq_len, vocab) peak
  4. loss * shift_loss_weights          → element-wise multiply
  5. (loss * mask).sum() / mask.sum()   → scalar
```

Steps 3–4 are separate GPU kernel launches. The full vocabulary projection exists in memory during
both forward and backward passes, and again during gradient checkpointing recomputation.

---

## Approach 1: Liger Kernel (Drop-In, Production Ready)

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is LinkedIn's open-source library of
Triton-based fused kernels for LLM training. The key primitive is `FusedLinearCrossEntropyLoss`,
which fuses the `lm_head` linear projection with the softmax and cross-entropy reduction into a
single chunked Triton kernel.

### What it does

- Processes tokens in chunks of ~128 at a time
- Computes `logits_chunk = hidden_states_chunk @ lm_head.weight.T` in the kernel
- Immediately computes cross-entropy on the chunk and accumulates the scalar loss
- **The full (seq_len, vocab) tensor is never written to HBM**
- Peak memory for logits drops from `O(seq_len × vocab)` to `O(chunk_size × vocab)`

### Integration

```python
# pip install liger-kernel

from liger_kernel.transformers import apply_liger_kernel_to_gemma3

# Call before loading the model — patches the model class in-place
apply_liger_kernel_to_gemma3(
    rope=True,
    cross_entropy=True,
    fused_linear_cross_entropy=True,
    rms_norm=True,
    swiglu=True,
)

# Everything else stays the same
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)
```

### Limitation for ASFT

Liger's `FusedLinearCrossEntropyLoss` does not natively accept a per-token `loss_weights` tensor.
After applying Liger, the ASFT weighting step would still be a separate element-wise multiply on the
pre-computed per-token loss vector:

```python
# With Liger: still two steps
per_token_loss = liger_fused_ce(logits, labels)   # fused, no HBM spike
weighted_loss = per_token_loss * shift_loss_weights  # separate kernel
```

This is still a major improvement — the vocabulary projection OOM spike is eliminated — but the
weight multiply remains an extra step.

### Expected gains on L4 (24 GB VRAM)

| Metric | Baseline (dynamic padding) | With Liger |
|--------|---------------------------|------------|
| Peak logit tensor memory | ~230 MB / example | ~46 MB / example (chunk_size=90) |
| Throughput | ~5 steps/min | ~6–7 steps/min (est.) |
| Max batch size (before OOM) | 1 | 2–3 (est.) |
| Implementation effort | — | ~10 lines |

---

## Approach 2: Custom ASFT-Weighted Triton Kernel

A purpose-built Triton kernel fuses all five steps of the current loss computation into one:

```
lm_head projection + softmax + cross_entropy + loss_weight multiply + masked mean
```

### Why ASFT weights are special

The ASFT `loss_weights` tensor is **structured-sparse**: a prefix of tokens (user turn + reasoning)
has weight `1.0`, then the action suffix gets weight `action_weight` (default `3.0`), and padding
positions have weight `0.0`. This structure can be exploited:

- The transition index (where weight changes from 1.0 to 3.0) is known per-sequence
- Tokens before the transition contribute at full weight without a multiply
- Tokens at/after the transition are simply accumulated into a separate register and scaled at the
  end of the kernel

### Kernel design sketch

```triton
@triton.jit
def asft_fused_cross_entropy_kernel(
    hidden_ptr,         # (batch, seq_len, hidden_dim)
    weight_ptr,         # lm_head weight: (vocab_size, hidden_dim)
    labels_ptr,         # (batch, seq_len)
    action_starts_ptr,  # (batch,) — per-sequence token index where action begins
    action_weight,      # scalar float, e.g. 3.0
    loss_out_ptr,       # scalar output
    CHUNK_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    ...
):
    # Each program instance handles one (batch_idx, token_chunk) tile
    # 1. Load hidden_states chunk: (CHUNK_SIZE, hidden_dim)
    # 2. Compute logits chunk = hidden_chunk @ weight.T: (CHUNK_SIZE, VOCAB_SIZE)
    #    Using tl.dot with accumulation in fp32
    # 3. Numerically stable log-softmax on the chunk (online softmax trick)
    # 4. Look up label for each token → NLL
    # 5. Check token_idx against action_starts[batch_idx]:
    #    - if token_idx < action_start: weight = 1.0
    #    - else: weight = action_weight
    #    - if label == -100: weight = 0.0  (padding)
    # 6. loss_accumulator += nll * weight
    # 7. count_accumulator += (label != -100)
    # Atomic add to loss_out_ptr at end
```

The key technique enabling this is the **online softmax** (from the Flash Attention / online
normalization literature): computing `log-softmax` and NLL over the vocabulary dimension without
holding the full softmax distribution in SRAM. Only the running max and sum-of-exp need to be
tracked.

### Memory comparison

```
Baseline:
  (450, 256000) × 2 bytes = 230 MB  — allocated for logits
  (450, 256000) × 2 bytes = 230 MB  — allocated again during backward
  Total peak addition: ~460 MB per example

Custom kernel:
  (CHUNK_SIZE=128, 256000) × 2 bytes = 65 MB  — one chunk at a time
  Gradient of lm_head.weight: (256000, hidden_dim) — same either way
  Total peak addition: ~65 MB (constant regardless of seq_len)
```

### Implementation considerations

1. **Backward pass**: Triton's `@triton.jit` is forward-only. The backward kernel must be written
   separately (or use `torch.autograd.Function` with a custom backward that calls a second kernel).
   This is the majority of the implementation complexity.

2. **Gradient of `lm_head.weight`**: The weight gradient accumulates across all tokens. Chunking
   makes this a sequence of `chunk_size × hidden_dim @ vocab_size` matmul accumulations — manageable
   but requires careful fp32 accumulation to avoid precision loss.

3. **Fusion boundary**: The kernel fuses from the last hidden state through to the scalar loss. The
   upstream gradient (d_loss / d_hidden) flows back into the transformer layers normally.

4. **Gemma 3 specifics**: The `lm_head` in Gemma 3 is tied to the embedding matrix. The kernel must
   handle the tied weight correctly — the gradient accumulates into the shared embedding/lm_head
   weight.

### Estimated implementation effort

| Component | Effort |
|-----------|--------|
| Forward kernel (chunked NLL + ASFT weights) | ~150–200 lines of Triton |
| Backward kernel (d_hidden gradient) | ~200–250 lines of Triton |
| Weight gradient accumulation | ~100 lines |
| `torch.autograd.Function` wrapper | ~50 lines |
| Unit tests vs. reference PyTorch | ~100 lines |
| **Total** | ~600–700 lines |

### Expected gains vs. Liger

| Property | Liger | Custom ASFT Kernel |
|----------|-------|--------------------|
| Eliminates logit HBM spike | Yes | Yes |
| ASFT weights fused into kernel | No (extra kernel) | Yes |
| Exploits weight sparsity (1.0/3.0/0.0) | No | Yes |
| Action boundary computation per-sequence | Not applicable | Can be passed as input |
| Implementation effort | ~10 lines | ~700 lines |
| Throughput gain over baseline | ~20–30% | ~25–35% (marginal over Liger) |
| Memory gain over baseline | ~3–4x | Same as Liger |

The throughput delta between Liger and a custom kernel is modest (~5–10%) because the weight
multiply is a cheap element-wise op. The primary value of a custom kernel is cleanliness
(single launch, no intermediate allocation) and the ability to exploit the structured sparsity
of ASFT weights at very high batch sizes — where the element-wise multiply kernel overhead becomes
non-negligible.

---

## Recommendation

For the current training run (batch_size=1, L4 GPU, ~450 token sequences):

1. **Use Liger Kernel first.** The 10-line integration cost is extremely low and it eliminates the
   vocabulary projection memory spike entirely. This is the pragmatic choice.

2. **Build the custom ASFT kernel** if: (a) batch size is scaled to 4+ and the loss weighting step
   shows up in profiling, or (b) the research direction moves toward longer sequences (>1024 tokens)
   where the fused forward/backward memory saving matters more.

3. **Profile before building.** On the L4 with dynamic padding, the actual bottleneck may be the
   image processing in the DataLoader (CPU-side JPEG decode + `image_processor` resizing) rather
   than the GPU loss kernel. Use `torch.profiler` or `nsys` to confirm before investing in a custom
   kernel.

---

## References

- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — LinkedIn's Triton LLM kernel library
- [Liger Paper (arXiv:2410.10989)](https://arxiv.org/abs/2410.10989) — benchmarks and design
- [Online Softmax / Flash Attention](https://arxiv.org/abs/2205.14135) — the chunked normalization technique enabling fused CE
- [Triton Language](https://triton-lang.org/) — used to implement all kernels above
- [GUI-Libra Paper](https://arxiv.org/abs/2406.10935) — ASFT loss formulation this kernel would accelerate
