# RL Kernel Optimizations for GUI Agent Training — Research Idea

## Overview

The GUI-Libra pipeline supports two training modes: SFT (Supervised Fine-Tuning with ASFT loss)
and RL (Reinforcement Learning). The computational profile of RL training differs fundamentally
from SFT, which changes where kernel-level optimizations deliver value and which bottlenecks
dominate.

This document covers:
1. The two RL paradigms (offline vs. online) and how they map to this project
2. GPU kernel opportunities unique to RL (fused log-prob gather, chunked KL divergence)
3. Architectural choices that affect memory and throughput more than any kernel (LoRA weight sharing,
   GRPO vs. PPO)
4. The real bottleneck in online RL for GUI tasks: rollout generation

---

## RL Paradigm: Offline vs. Online

### Offline RL (likely what `DATASET_ID_RL` represents)

The project's `DATASET_ID_RL` points to a pre-collected dataset of (screenshot, goal, action,
reward) tuples — reward-labeled trajectories assembled before training begins. The training loop
looks like reward-weighted supervised learning:

```
loss = CrossEntropy(logits, labels) * reward_weight(example)
```

This is structurally identical to the ASFT weighted loss from the SFT phase. The reward scalar
plays the same role as the `action_weight` multiplier. All kernel opportunities described in
[fused_asft_loss_kernels.md](fused_asft_loss_kernels.md) apply directly, with one difference:
the weight tensor is now a **continuous scalar per example** (not the structured-sparse 1.0/3.0
binary used in ASFT), so the sparsity exploitation described for the custom ASFT kernel doesn't
apply — but the chunked fused CE kernel eliminates the same HBM spike.

### Online RL (PPO / GRPO with live GUI environment)

Online RL involves the model actually interacting with a GUI:

```
[Observe screenshot] → [Generate action] → [Execute in simulator]
→ [Capture new screenshot] → [Compute reward] → [Update policy]
```

This introduces a fundamentally different computational profile. The GPU is not the primary
bottleneck — the environment interaction is. However, the **policy update step** (which runs on
GPU) has unique kernel opportunities not present in SFT.

---

## GPU Kernel Opportunities in Online RL

### 1. Fused Log-Probability Gather Kernel — Highest Value

PPO and GRPO do not use cross-entropy loss directly. They compute the probability ratio between
the current (updated) policy and the old (reference) policy:

```
ratio = exp(log_prob_new(action | state) - log_prob_old(action | state))
```

Both `log_prob_new` and `log_prob_old` require a full forward pass through the vocabulary
projection. The standard approach:

```
Step 1:  logits = hidden_states @ lm_head.T    → (seq_len, 256K)  ← HBM spike
Step 2:  log_probs = log_softmax(logits)        → same shape
Step 3:  gathered = log_probs[labels]            → (seq_len,)       ← only this is kept
```

Steps 1–3 run **twice** per training step (once for new policy, once for old/reference policy),
causing two separate `(seq_len, 256K)` allocations. For Gemma 3 at 450 tokens, each allocation
is ~230 MB — 460 MB total, compared to the single spike in SFT.

**Fused log-prob gather kernel approach:**
- Project hidden states to logits in chunks of `CHUNK_SIZE` tokens (e.g., 128)
- Compute `log_softmax` on each chunk using the online normalization trick (running max + sum-of-exp)
- Immediately gather `log_probs[label_token]` for the relevant token
- Accumulate only the gathered scalar per token into a `(seq_len,)` buffer
- **The full `(seq_len, vocab)` tensor never lands in HBM**

Peak memory drops from `O(seq_len × vocab)` to `O(chunk_size × vocab)` — constant with respect
to sequence length.

This kernel is the direct RL analog of the fused CE kernel for SFT, and it matters more here
because it runs twice per step rather than once.

---

### 2. Chunked KL Divergence Kernel — Eliminates the Worst-Case Spike

The KL penalty in PPO bounds the policy from drifting too far from the reference:

```
KL(π_new || π_ref) = Σ_v softmax(logits_new)[v] * (log_softmax(logits_new)[v] - log_softmax(logits_ref)[v])
```

Computing this exactly requires materializing **two** full `(seq_len, vocab)` tensors simultaneously:
one for `π_new`, one for `π_ref`. At 450 tokens and 256K vocab in bfloat16, that is ~460 MB — and
both must coexist during the KL computation, bringing peak pressure to ~920 MB for this single
operation.

**Two approaches:**

**Approximate KL** (used by most RL libraries in practice):
```python
kl ≈ log_prob_new - log_prob_ref   # per token, scalar subtraction
```
This requires only the gathered log-probs — no kernel needed, zero overhead. The approximation
is `KL(p||q) ≈ log(p/q)` for the greedy token, which underestimates the true KL but is
sufficient as a regularization signal and is what TRL's PPO implementation uses by default.

**Exact chunked KL kernel** (custom Triton):
- Process vocabulary in chunks; maintain running sums `Σ p*log(p)` and `Σ p*log(q)` per token
- Uses the online softmax trick for both distributions simultaneously
- Returns only the `(seq_len,)` KL vector
- Peak memory: `O(chunk_size × vocab)` instead of `O(2 × seq_len × vocab)`
- Applicable when exact KL is required (e.g., during evaluation, or strict PPO implementations)

Effort: ~600–800 lines of Triton. No existing library implements this for the exact-KL case.

---

### 3. LoRA Reference Weight Sharing — Zero Kernel Cost, 8 GB Saved

In PPO and GRPO, a frozen reference policy is needed to compute `log_prob_ref` and enforce the
KL constraint. Without optimization, this means keeping two full copies of Gemma 3 4B in memory:

```
Reference policy (frozen):  ~8 GB (bfloat16)
Active policy (trainable):  ~8 GB (bfloat16) + LoRA adapter weights
Total:                       ~16+ GB on a 24 GB L4
```

With LoRA, the reference policy **is the base model without adapters**. The active policy is the
same base model with LoRA deltas applied. They share all frozen weights:

```
Base model weights:   shared, one copy in memory     → ~8 GB
LoRA adapter weights: small, only for active policy  → ~50–200 MB (r=16)
```

TRL's `create_reference_model` with `is_trainable=False` implements this. The reference forward
pass simply skips LoRA application. No custom kernel required — this is an architectural choice
that frees ~8 GB on the L4, enabling larger batches or higher LoRA rank.

---

## Algorithmic Choice: GRPO vs. PPO

This affects memory and throughput before any kernel is written.

### PPO (Proximal Policy Optimization)

- Requires a **value (critic) head** — typically a linear layer on top of the LM hidden states
- Critic estimates expected future reward per token; used for advantage estimation (GAE)
- Extra memory: small (one linear layer), but extra forward/backward pass through the critic
- Requires batched rollouts with stored advantages, old log-probs, and value estimates

### GRPO (Group Relative Policy Optimization — DeepSeek-R1)

- **No value/critic head** — advantage is computed relative to other completions in a group
- For each prompt, generate K completions; normalize rewards within the group:
  ```
  advantage_i = (reward_i - mean(rewards)) / std(rewards)
  ```
- The policy update uses these group-normalized advantages as weights (similar to ASFT weights)
- Memory-efficient: no second model component to maintain
- Particularly well-suited to GUI tasks where reward is **trajectory-level binary** (did the
  action complete the goal?), not per-token

**GRPO and kernel implications:**

GRPO runs K forward passes per prompt (K completions). The log-prob gather kernel runs K times
per prompt, making the per-kernel-call memory savings (eliminating the vocab projection HBM spike)
multiply by K. At K=8, avoiding the spike 8× per step is materially more important than in PPO.

GRPO also produces a per-example reward weight that is **continuous** (the normalized advantage),
making the loss update equivalent to reward-weighted CE — directly compatible with a fused
weighted CE kernel (same design as the ASFT kernel, but with continuous weights).

| Property | PPO | GRPO |
|----------|-----|------|
| Value/critic head needed | Yes | No |
| Rollout samples per prompt | 1 | K (e.g., 8) |
| Advantage estimation | GAE (requires value model) | Group normalization |
| Loss form | Clipped surrogate + value loss | Reward-weighted CE |
| Suited for sparse/binary rewards | Poorly | Well |
| Kernel opportunity | Log-prob gather × 2 | Log-prob gather × (K+1) |

---

## The Real Bottleneck: Rollout Generation

For **online RL** with a live GUI environment, the GPU loss kernel is not the primary bottleneck.
The environment interaction loop is:

```
1. Model generates action (autoregressive, GPU-bound)     ~1–3 sec
2. Parse action (click/scroll/type) (CPU, trivial)        ~1 ms
3. Execute action in simulator (browser, CPU/IO-bound)    ~500ms–2 sec
4. Capture screenshot (CPU/IO-bound)                      ~100–500 ms
5. Compute reward (CPU, rule-based for GUI tasks)         ~1 ms
6. Store transition in replay buffer                      ~1 ms
                                                   Total: ~2–6 sec per step
```

At 12 seconds per training step (from observed training logs), with K=8 rollouts for GRPO:
rollout collection alone takes 16–48 seconds, dwarfing the ~1 second policy update.

### Kernel-level tools for rollout acceleration

**Speculative decoding** — the highest-value technique for generation speed:
- A small draft model (e.g., Gemma 3 1B) generates candidate token sequences speculatively
- The full 4B model verifies K tokens in parallel rather than autoregressively
- Accepted tokens advance the generation without extra wall-clock time
- Typical speedup: 2–3× for tasks with predictable action formats (GUI actions are short and
  repetitive — click(x,y), scroll(dx,dy), type("text") — making speculative decoding highly
  effective here)

**Flash Attention / PagedAttention** for the decode phase:
- Flash Attention is likely already active via PyTorch SDPA; no action needed
- PagedAttention (vLLM-style) manages KV cache in non-contiguous pages, enabling higher
  batch throughput during generation — relevant if generating K completions in parallel

**Async rollout collection** — architectural, not a kernel:
- While the GPU runs the policy update for batch N, K CPU workers collect rollouts for batch N+1
- Requires a separate rollout buffer process and careful synchronization
- Can hide most of the environment latency behind the GPU training step
- This is the standard production approach in large-scale RL (e.g., AlphaCode, RLHF at scale)

---

## Summary: Prioritized Kernel and Optimization Roadmap for RL

| Priority | Optimization | RL Type | Effort | Gain |
|----------|-------------|---------|--------|------|
| 1 | LoRA reference weight sharing | Online | ~0 (TRL built-in) | ~8 GB VRAM freed |
| 2 | GRPO instead of PPO | Online | Algorithmic | No critic head, binary reward friendly |
| 3 | Approximate KL (`log_prob_new - log_prob_ref`) | Online | ~0 (already in TRL) | Eliminates full-distribution materialization |
| 4 | Fused log-prob gather kernel | Online | ~400 lines Triton | Eliminates 2× (or K+1×) HBM spike |
| 5 | Fused reward-weighted CE (GRPO loss) | Online | ~300 lines Triton | Same as ASFT kernel; continuous weights |
| 6 | Speculative decoding for rollouts | Online | Medium (requires draft model) | 2–3× generation speed |
| 7 | Async rollout collection | Online | High (infrastructure) | Hides environment latency |
| 8 | Chunked exact KL kernel | Online | ~700 lines Triton | Eliminates 2× HBM spike for exact KL case |
| 9 | Fused reward-weighted CE | Offline | Same as SFT | Identical to ASFT kernel |

For the current L4-based training setup, **priorities 1–3 are zero-effort and should be
implemented first if moving to online RL**. The fused log-prob gather kernel (priority 4) is
the highest-value custom Triton work. Speculative decoding (priority 6) addresses the rollout
bottleneck that kernels cannot touch.

---

## References

- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [GRPO / DeepSeek-R1 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [TRL Library — PPO and GRPO Trainers](https://github.com/huggingface/trl)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — for fused CE; log-prob gather pending
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) — KV cache management for generation
- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- [GUI-Libra Paper](https://arxiv.org/abs/2406.10935) — RL formulation this applies to
- [SFT Kernel Notes](fused_asft_loss_kernels.md) — companion document for the SFT phase
