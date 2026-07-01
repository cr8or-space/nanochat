# Tiny reasoning model — build plan (math-first, ~150M, distilled CoT)

Direction plan for adapting nanochat into a small, capable **reasoning** model.
Companion to `deepseek_qwen_features.md` (the architecture-feature roadmap). **No code
yet** — this is the written plan to build against. Work continues on branch `current`.

## Decisions locked (with the user)
- **Size:** small, ~100–200M params. Target ~150M.
- **Primary signal:** distill long chain-of-thought from a strong teacher → SFT, then
  RL with verifiable rewards (RLVR). Distillation is the biggest single lever.
- **Scope:** specialized on **math first** (highest-signal, easiest-to-verify rewards);
  prove the data+RL pipeline before broadening. This also defers the shared-base /
  shared-vocab question — a math-only vocab is fine for now.

## Guiding principle
Data + RL dominate; architecture is last and higher-variance. Do **not** invest in clever
architecture before the distillation + RLVR pipeline is real and moving the eval numbers.

---

## 1. Model spec (~150M) — corrected from a first-principles param count
nanochat derives everything from `--depth` (`base_train.py`): `model_dim = depth × 64`,
`n_head = model_dim / head_dim`, `n_layer = depth`, plus LR / Chinchilla-token derivations
tuned at the **d12 reference** and muP-transferred.

**The dominant param term at tiny scale is the value-embedding tables (ResFormer), not the
token embeddings.** Measured `num_scaling_params()` breakdown at **d12** (`n_embd=768`, 12L/12H):

| config | total | value_embeds | wte | lm_head | transformer | mtp |
|---|--:|--:|--:|--:|--:|--:|
| vocab 32K, baseline | **286M** | 151M | 25M | 25M | 85M | – |
| vocab 32K, +MLA+MTP+gated | **147M** | 0 | 25M | 25M | 91M | 6M |
| vocab 12K, baseline | **160M** | 57M | 9M | 9M | 85M | – |
| vocab 12K, +MTP+gated | **173M** | 57M | 9M | 9M | 92M | 6M |

Key facts this surfaces:
- Value-embeds = `~6 layers × vocab × (n_kv_head·head_dim)` → **151M at 32K vocab** — they dwarf
  everything and scale with vocab. This is the #1 param lever at tiny scale.
- **MLA disables value-embeds** (they can't be reconstructed from a latent-only cache), which
  *also* removes that 151M tax → d12 drops 286M → **147M**. So for the tiny model MLA is doubly
  attractive: cheap long-CoT KV **and** it reclaims the value-embed budget.
- The **token budget decouples from total params**: `base_train`'s scaling params =
  `transformer_matrices + lm_head` (~110–116M at d12), *not* total. So the Chinchilla horizon
  (~20×) is ~2.2–2.3B tokens regardless of value-embeds/vocab. Trimming vocab mainly cuts *total*
  (embeddings); tying removes `lm_head` from both total and the token budget.

- **Recommended baseline:** **d12 + `--use-mla`** ≈ **147M** at the stock 32K vocab — hits the
  target cleanly and reclaims the value-embed budget. Then layer in trimmed vocab / tying as
  independent wins. Confirm exact size with a `num_scaling_params()` dry run per config.
- **Landed features to enable:** `--use-mla` (see above), `--use-gated-attn` (cheap stability),
  `--n-mtp 1` (sharper signal + faster CoT drafting). `--rope-scaling` (YaRN) held in reserve for
  the long-CoT phase. GQA not needed at this size.
- **Open modeling decision:** value-embeds are a *training-signal* feature (ResFormer value
  residual). Disabling them (via MLA, or independently) trades some quality for a large param
  saving — very attractive at tiny scale, but worth an A/B.

## 2. Tokenizer (do first — cheap, high-leverage, low-risk)
- **Single-digit numbers:** change `SPLIT_PATTERN` at `tokenizer.py:30` from `\p{N}{1,2}` to
  `\p{N}`. Big win for arithmetic/reasoning; there's precedent (line 71 already documents a
  deliberate `{1,3}→{1,2}` narrowing).
- **Trim vocab:** ~12–16K technical tokens instead of 32K. At this scale untied embeddings are
  `2 × vocab × n_embd` — at 32K/768 that's ~50M, i.e. a third of a 150M budget. Trimming
  reallocates budget into depth.
- **Weight tying:** nanochat unties wte/lm_head by default. In the tiny + small-vocab regime,
  tying halves embedding cost — add an opt-in config flag and A/B it. (Interacts with the
  separate Muon/AdamW param groups + the `setup_optimizer` count assert — must stay satisfied.)
- Train the BPE (`scripts/tok_train.py`, rustbpe) on the **technical/math corpus**, not generic web.

## 3. Data pipeline (the dominant lever — most of the real work)
- **Stage A — pretrain:** textbook-quality technical corpus (open textbooks, arXiv math,
  math/code StackExchange, curated code, filtered math web). Dedup + quality-filter hard.
  Budget ~Chinchilla (`target_param_data_ratio`, ~20× params ≈ ~3B tokens at 150M).
- **Stage B — mid-train (reasoning-dense):** worked solutions, proofs, code-with-tests — shift
  the distribution toward step-by-step reasoning before SFT.
- **Stage C — SFT on distilled long-CoT (the big lever):** generate CoT traces with a strong
  teacher over math problem banks (GSM8K, MATH, competition sets, synthetic templates).
  **Reject-sample by verifier** — keep only traces whose final answer checks out. Format behind
  a thinking control token (the hybrid-thinking recipe slots into `chat_sft.py`'s special-token
  machinery), so thinking/non-thinking is selectable at inference.

## 4. RLVR — how a tiny model punches above its weight
- Build on the existing GRPO scaffold (`scripts/chat_rl.py`, GSM8K; `execution.py` runs code).
- **Expand verifiers:** exact numeric match (single-digit tokenization makes this cleaner),
  symbolic equivalence (sympy), and unit-tested code via `execution.py`.
- Reward-shape for planning/backtracking; add mild CoT-length control. Curriculum easy→hard.

## 5. Tool-augmented reasoning
- The Engine already has a Python/calculator tool loop (`<|python_start|>` … state machine).
  Train SFT traces that *call the tool* so the model offloads arithmetic precision and uses code
  execution as a planning substrate. Reinforce tool-use in RLVR.

## 6. Architecture experiments (last, optional, higher-variance)
Only after data+RL are moving the numbers. Candidates for serial reasoning depth:
looped / recurrent-depth transformers (Universal-Transformer, Geiping latent-recurrent-depth),
latent reasoning (Coconut), adaptive compute. The landed features already lean this way
(MTP, YaRN, MLA, gated attn).

## 7. Evaluation
- Use `tasks/` (gsm8k present; **add a MATH task** — not currently in the repo). Track pass@1,
  CoT length, tool-use rate, and `val_bpb` on held-out technical text. Keep a fixed eval set from
  day one so pipeline changes are measurable.

## 8. Sequencing / milestones
1. **Tokenizer:** single-digit `\p{N}`, ~12–16K technical BPE, tying flag. Dry-run param count → confirm ~150M at chosen depth.
2. **Pretrain baseline:** assemble Stage-A corpus; train d12–14; sanity evals (val_bpb, gsm8k pass@1 from base).
3. **Distillation:** teacher CoT generation + verifier reject-sampling → SFT set.
4. **SFT:** train on distilled CoT; measure gsm8k / MATH pass@1 and CoT quality.
5. **RLVR:** expand verifiers; GRPO; curriculum. Re-measure.
6. **(Optional) architecture:** only if 2–5 plateau below target.

## Risks / notes
- **Teacher access/cost** is a dependency for the primary (distillation) path — budget for it.
- The hard part is **data assembly + verifier coverage**, not the modeling changes.
- Keep every change opt-in and on `current`; preserve the `--depth` scaling story where possible
  (vocab/tying changes touch it — re-verify the token-budget derivation after trimming vocab).
- **Deprioritized under this goal:** MoE (memory-bound single-GPU, breaks the `--depth` dial,
  tiny active params — wrong lever for a tiny model), and the parked items in the features doc.
