# DeepSeek & Qwen features for nanochat — candidate roadmap

This document surveys architectural and training features from the **DeepSeek**
(V3 → V3.2 → V4) and **Qwen** (Qwen3 → Qwen3-Next → Qwen 3.6) lineages and assesses
each one for adoption into nanochat. nanochat is a lean, single-node, educational
full-stack LLM; recommendations are weighed against that goal (hackability,
compute-efficiency, keeping the FA3 fast path), **not** frontier-scale serving.

Each feature is tagged:

- ✅ **Take** — good value-to-effort, fits nanochat.
- ⚠️ **Take with caveats** — worthwhile but invasive or with real trade-offs.
- ⛔ **Skip** — cost outweighs benefit at nanochat's scale/mission.
- ✔️ **Already present** — the repo already implements this (in spirit).
- 🚧 **Done** — implemented on this branch.

> Sourcing note: some 2026 V4/3.6 details come from secondary/marketing sources.
> Recommendations are anchored to the *published, reproducible* techniques in the
> DeepSeek-V3/V3.2 and Qwen3/Qwen3-Next lineages (MLA, MoE, aux-loss-free balancing,
> MTP, DSA, Gated DeltaNet, YaRN), not the unverified novel claims.

---

## 0. What nanochat already has

Several "modern" features from these model families are already in the repo, so they
should not be double-counted as new work:

| Feature | Origin | Where in repo |
|---|---|---|
| QK-norm | Qwen3 | `gpt.py` (`q,k = norm(q),norm(k)`) |
| FP8 training | DeepSeek-V3 | `fp8.py` (tensorwise scaling, `_scaled_mm`) |
| Sliding-window attention | Gemma/Qwen | `window_pattern="SSSL"` |
| Logit softcap | Gemini | `softcap=15` in `gpt.py` |
| Residual scaling / initial-embedding blend (≈ Hyper-Connections lite) | modded-nanogpt | `resid_lambdas`, `x0_lambdas` |
| RoPE precomputed for 10× context (enables YaRN) | — | `_precompute_rotary_embeddings` |
| GQA infrastructure | Llama/DeepSeek | `n_kv_head` (currently `== n_head`) |
| Muon optimizer (+ NorMuon, Polar Express, cautious WD) | modded-nanogpt | `optim.py` |

---

## 1. Implemented on this branch

### 🚧 Multi-head Latent Attention (MLA) — DeepSeek-V2/V3
Low-rank KV cache: cache a compressed latent `c_KV` + a shared decoupled-RoPE key,
reconstruct full multi-head K/V on the fly. **The best quality-per-cache-byte option**
because it compresses without dropping head expressivity (unlike GQA/FP8-KV).

- Opt-in via `--use-mla`; presets `--mla-preset {4x,8x}` (4.4× / 8.0× smaller KV at d20 dims).
- Keeps per-head dims == `head_dim` so `c_q`/`c_proj` and the FA3 fast path are unchanged.
- Files: `gpt.py`, `engine.py` (`MLAKVCache`), `base_train.py`; tests in `tests/test_mla.py`;
  CPU smoke-train in `dev/mla_smoke_train.py`.
- **Known limitation / follow-up:** value-embeddings are disabled on MLA layers because
  their per-token V contribution can't be reconstructed from a latent-only cache. Folding
  them into the latent (a learned per-token vector added to `c_KV`, optionally input-gated)
  would restore the feature while staying cacheable.
- **Further follow-up:** the *absorbed-projection* MLA inference trick (fold `kv_up` into
  `c_q`/`c_proj` to skip decompression each decode step) for faster long-context decode.

### 🚧 Multi-Token Prediction (MTP) + speculative decoding — DeepSeek-V3 / Qwen3-Next
Auxiliary heads predict tokens `t+2, t+3, …` to sharpen the training signal, and double as a
draft model for **greedy self-speculative decoding** (which nanochat previously lacked).

- Opt-in via `--n-mtp N` (+ `--mtp-weight`); composes with MLA and FP8.
- MTP modules are **position-wise** (proj + relu² MLP, no attention), so a single-token draft
  at inference is numerically identical to the training-time computation — no separate MTP KV
  cache needed. Embedding (`wte`) and output head (`lm_head`) are shared with the main model;
  the new matmuls auto-join the Muon group.
- `GPT.generate_speculative` drafts `n_mtp` tokens per step and verifies them in one main
  forward, accepting the longest greedy-matching prefix (+1 bonus when the whole draft matches),
  with KV-cache rollback. Output is **token-for-token identical to greedy `generate`** — MTP
  quality only affects speed, never correctness (verified in `tests/test_mtp.py`).
- Also fixed a latent gap: the smear feature's prefill path now smears a block's first token
  from the previous committed token when mid-sequence (`kv_cache` pos > 0), required for
  speculative-verify parity. (The pre-existing Engine only ever prefilled at pos 0, so this
  path was never exercised before.)
- Files: `gpt.py` (MTP heads, `_mtp_loss`, `generate_speculative`), `base_train.py`;
  tests in `tests/test_mtp.py`.
- **Follow-ups:** temperature/rejection-sampling speculative decoding (current path is greedy);
  wiring speculative decode into `Engine.generate`'s batched tool-use loop; transformer-block
  MTP modules (with their own KV cache) for higher draft acceptance.

---

## 2. KV-cache reduction (other options)

These compose with each other and with MLA; MLA is the quality-per-byte winner, but
these are cheaper and stay on the FA3 fast path.

### ✅ GQA — turn the dial that already exists
`c_k`/`c_v` already project to `n_kv_head`, and the cache is sized by it — but the config
ships `n_kv_head == n_head`, so no savings are realized today. Setting `n_kv_head` to e.g. 2
gives a ~5× KV-cache cut with **zero new code** (just a retune). Trades some quality for
bytes. It also shrinks the `value_embeds` tables. Highest value-to-effort cache win.

### ✅ Window-aware / ring-buffer KV cache
The `SSSL` pattern means ~75% of layers only attend to the last ~512 tokens, but the cache
allocates full `seq_len` for **every** layer. Give windowed layers a ring buffer of `W (+margin)`
slots. **Engine-only change** (`KVCache` per-layer sizing + wrap-around writes); the model is
untouched. ~2× at 2048 ctx, and the win *grows* with context length. Subtlety: correct
wrap-around with FA3's in-place `flash_attn_with_kvcache` update.

### ✅ FP8 (or int8) KV cache
Store `k_cache`/`v_cache` (or the MLA latent) in `float8_e4m3`; quantize on write, dequant on
read. 2×, composes with everything. `fp8.py` already has the scaling primitives. Small quality
cost, so keep the MLA latent in bf16 if pure quality-per-byte is the goal.

### ⛔ Cross-layer KV sharing (CLA / YOCO)
Share KV across adjacent layers for a linear reduction. Not from the DeepSeek/Qwen lineage
and interacts awkwardly with the per-layer window pattern; MLA + the above already cover the
need. Skip unless specifically studying cross-layer sharing.

---

## 3. Mixture-of-Experts (the biggest architectural lever)

### ⚠️ DeepSeekMoE + auxiliary-loss-free load balancing
Replace the dense `MLP` with fine-grained routed experts + a shared expert, and balance load
with DeepSeek-V3's **bias-based, aux-loss-free** controller (no gradient-based balancing loss —
elegant and cheap to teach). This is the single most educationally important frontier feature
(every model here is MoE).

**Costs specific to this repo (be honest):**
- **Scaling laws break.** nanochat's charm is the single `--depth` dial auto-deriving width,
  heads, LR, and Chinchilla tokens:params (`base_train.py`). MoE decouples parameters from
  FLOPs, so those derivations need separate active-vs-total accounting.
- **Optimizer.** Muon handles expert matrices fine, but the router + aux-free bias controller
  live outside that regime (AdamW / manual).
- **Parallelism.** DDP-only, no expert parallelism — every GPU holds every expert, so it's
  memory-bound with no sharding relief.

Recommendation: worth doing as a **clearly feature-flagged variant**, not the new default.
Start with few experts (e.g. 8 routed + 1 shared, top-2) so it stays single-GPU trainable.

### ⚠️ Shared-expert isolation / fine-grained experts
The DeepSeekMoE refinements (a subset of always-on shared experts + many small experts) come
"for free" once MoE exists; include them in the MoE variant rather than as separate work.

---

## 4. Training-signal & optimization

### 🚧 Multi-Token Prediction (MTP) — DeepSeek-V3 & Qwen3-Next — **implemented, see §1**
Auxiliary heads predicting tokens *t+2, t+3…* alongside the main next-token head. Improves the
base-training signal and yields a free draft model for speculative decoding. Landed on this
branch — see §1 for the design and the position-wise (attention-free) module choice.

### ✔️/✅ Optimizer & precision
Muon (+ NorMuon, Polar Express, cautious weight decay) and FP8 training are already present.
No action beyond confirming MLA + `--fp8` compose (the `kv_down`/`kv_up` projections satisfy
the FP8 dim filter).

---

## 5. Attention & positional variants

### 🚧 Gated attention (output gating) — Qwen3-Next — **implemented**
A per-element sigmoid gate on the attention output (before the output projection), computed
from the block input. Opt-in via `--use-gated-attn`; composes with MHA, GQA, MLA, and MTP
speculative decoding (all verified in `tests/test_gated_attn.py`). Applied in both attention
paths in `gpt.py` just before `c_proj`; the gate matrix auto-joins the Muon group.

### 🚧 YaRN / RoPE context extension — Qwen3 — **implemented**
NTK-by-parts frequency interpolation in `_rope_inv_freq`: high-frequency dims are kept
(extrapolated), low-frequency dims are interpolated toward `inv_freq / s`, with a linear ramp
between. Opt-in via `--rope-scaling s` / `--rope-original-seq-len`; composes with MLA (acts on
`qk_rope_head_dim`). Tests in `tests/test_yarn.py`. **Note:** the YaRN attention-temperature
(`mscale`) term is intentionally omitted — RoPE is norm-preserving and nanochat's QK-norm
renormalizes q/k afterward, so `mscale` would have no effect here.

### ⛔ Gated DeltaNet / linear-attention hybrid — Qwen3-Next / Qwen 3.6 core
Interleave linear-attention (constant-memory recurrent state) layers with full attention. It's
a full attention rewrite (chunked scan kernels), incompatible with FA3's `flash_attn_with_kvcache`
fast path, and its benefit (constant-memory decode) only materializes at very long context. For
a 2048-token educational model it's enormous complexity for negative practical payoff. Skip
unless linear attention itself is the lesson or context grows to tens of thousands of tokens.

### ⛔ DeepSeek Sparse Attention (DSA / CSA / HCA + "lightning indexer") — V3.2/V4
Exists to make **128K–1M-token** context tractable via a learned token-selection indexer.
nanochat's context is 2048 — there is essentially nothing to accelerate, and it adds a whole
indexer subsystem. Skip.

---

## 6. Context length & memory

### ⛔ 1M-token context
Depends on the sparse-attention and linear-attention machinery above plus memory offload. Out
of scope for a single-node educational repo. (YaRN in §5 covers modest extension cheaply.)

### ⛔ Engram Conditional Memory (V4) / persistent external memory
A novel, poorly-specified persistent memory store; really a systems/serving feature orthogonal
to nanochat's "train a small model end-to-end" mission. A research bet, not a stable port. Skip.

### ⛔ Manifold-Constrained Hyper-Connections (V4)
Replaces residual connections with learned manifold mappings. Novel/marketing-described with no
public reference impl, and nanochat already captures the cheap, proven part via
`resid_lambdas`/`x0_lambdas`. Skip.

---

## 7. Data & recipe (not architecture)

### ✅ Hybrid thinking mode — DeepSeek-V4 & Qwen 3.6
"Thinking / non-thinking selectable by a control token" is an **SFT/RL data recipe**, not a
model change. It slots into nanochat's existing SFT stage (`chat_sft.py`, the special-token
machinery) — add reasoning-trace data gated behind a control token. Zero architecture risk; the
cost is data.

### ⛔ 201-language multilingual scale — Qwen 3.6
A data-and-scale story. A 40–300M educational model won't benefit; it dilutes the tokenizer and
training signal. Skip.

---

## 8. Inference / serving

### 🚧 Speculative decoding — **implemented, see §1**
Pairs naturally with MTP: the MTP heads propose multiple tokens and the main model verifies in
one forward pass. Landed as `GPT.generate_speculative` (greedy, exact-parity). Follow-ups:
temperature sampling and integration into the batched tool-use loop — see §1.

### ⚠️ Inference-time KV quantization / paged cache
Beyond FP8-KV (§2), paged/block KV management enables larger batch serving. Useful but more of a
systems feature; lower priority than the modeling items for an educational repo.

---

## 9. Suggested sequencing

A pragmatic order that front-loads low-risk, high-value wins and defers the invasive change:

1. **Window-aware KV cache** — engine-only, composes with MLA for more cache savings. *(next)*
2. **GQA + FP8-KV** — cheap composable cache cuts (if not relying solely on MLA).
3. **MoE (aux-loss-free)** — the big lesson; land as a feature-flagged variant.
4. **Hybrid-thinking data recipe** — SFT-stage, data-bound.

Already landed: **MLA**, **MTP + speculative decoding**, **gated attention**, and **YaRN** (§1).

Explicitly parked: **Gated DeltaNet, DSA/sparse attention, 1M context, Engram memory,
Manifold Hyper-Connections, 201-language scale** — see rationale above.

---

## Sources

- DeepSeek-V3 technical report — MLA, DeepSeekMoE, aux-loss-free balancing, MTP, FP8.
- DeepSeek-V3.2-Exp — DeepSeek Sparse Attention (DSA), lightning indexer.
- Qwen3 / Qwen3-Next — QK-norm, YaRN, Gated DeltaNet, gated attention, ultra-sparse MoE, MTP.
- 2026 model overviews (secondary): DeepSeek V4 (Tech Jacks, Morph), Qwen 3.6 (Qwen blog,
  GitHub QwenLM/Qwen3.6). Treated as directional, not authoritative, for unverified claims
  (Engram memory, Manifold-Constrained Hyper-Connections, CSA/HCA specifics).
