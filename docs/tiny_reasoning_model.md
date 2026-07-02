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

## 2. Data pipeline (the dominant lever AND the critical-path dependency)
**Grounded in the repo:** `base_train.py` pretrains on **ClimbMix-400B**, a generic shuffled-web
mix — *no* math/technical corpus by default. The path is **hardcoded** (`dataset.py:22-27`:
`BASE_URL`, `MAX_SHARD=6542`, dir `base_data_climbmix`); there is **no `--data` CLI arg**. Data is
**parquet with a `'text'` column, tokenized on the fly** by the BOS-aligned best-fit loader
(`dataloader.py:74-161`); the last shard is val (`dataloader.py:38`). Crucially, the **tokenizer
trains on the *same* parquet corpus** (`tok_train.py:11,35` via `parquets_iter_batched`).

⇒ **Assembling a math/technical parquet corpus (rows with a `'text'` column) is milestone 0** —
it feeds *both* the BPE and pretrain. The loader is corpus-agnostic, so injection = point
`dataset.py:22-27` at math parquet shards (or build a mixed shard set). Stages:
- **Stage A — pretrain corpus:** textbook-quality technical/math/code (open-web-math, proof-pile,
  arXiv math, math/code StackExchange, curated textbooks/code). Dedup + quality-filter hard, write
  to parquet `'text'` shards. **Horizon reality-check:** d12+MLA scaling params ≈116M × ~20
  (`target_param_data_ratio`) ≈ **~2.3B tokens** — a few-B-token curated math corpus is feasible
  from open sources, so the budget is realistic (not web-scale).
- **Stage B — mid-train (reasoning-dense):** worked solutions, proofs, code-with-tests — shift the
  distribution toward step-by-step reasoning before SFT.
- **Stage C — SFT on distilled long-CoT (the big lever):** generate CoT with a strong teacher over
  math banks (GSM8K, MATH, competition, synthetic). **Reject-sample by verifier** (keep only
  correct-final-answer traces). Format behind a thinking control token — the hybrid-thinking recipe
  slots into `chat_sft.py`'s special-token machinery; the SFT mixture lives at `chat_sft.py:166-179`.

## 3. Tokenizer (cheap, high-leverage — but depends on §2's corpus existing first)
- **Single-digit numbers:** change `SPLIT_PATTERN` at `tokenizer.py:30` from `\p{N}{1,2}` to
  `\p{N}`. Confirmed a **genuine one-liner** — nothing else hardcodes 2-digit assumptions (grep
  found only the pattern itself + a cosmetic banner regex + a `d\d+` model-tag parser); GSM8K
  answer-extraction works on decoded text so it's unaffected. **Must retrain** the tokenizer (the
  pattern is frozen into the saved `tokenizer.pkl`, `tok_train.py:56-58`). Precedent: `tokenizer.py:71`
  documents the earlier deliberate `{1,3}→{1,2}` narrowing.
- **Trim vocab:** set `tok_train.py` `--vocab-size` (default 32768, `tok_train.py:19`) to ~12–16K.
  At tiny scale this mostly cuts *total* params (embeddings + value-embeds, which scale with vocab —
  see §1); the Chinchilla horizon barely moves (keys off `lm_head`, not `wte`).
- **Weight tying (opt-in `tie_embeddings` flag):** halves embedding cost. Touch points (all in
  `gpt.py`): add the config field (`GPTConfig` ~29-62); share `lm_head.weight` with `wte`
  (construction ~278-281); collapse the two `normal_` inits (~330-331); and **fix two param-count
  asserts that would double-count a shared tensor** — `num_scaling_params` (~526) and
  `setup_optimizer` (~550) — plus reconcile the two different AdamW LRs those groups use
  (`unembedding_lr` vs `embedding_lr`, ~559-560). `get_scaling_params` (base_train `283-287`) also
  changes meaning under tying (re-verify the horizon).
- **Order:** assemble §2 corpus → edit `SPLIT_PATTERN` + set vocab → `tok_train` **on the math
  corpus** → then pretrain. (Tokenizer trains on whatever `dataset.py` points at.)

## 4. RLVR — how a tiny model punches above its weight
- **Current state (grounded):** `chat_rl.py` is GRPO-simplified-to-REINFORCE, **hardwired to
  `GSM8K`** (`chat_rl.py:28,80-81`); reward is `train_task.reward(...)` → 0/1 (`gsm8k.py:110-117`)
  via **exact string match** on the `#### <number>` marker (`GSM_RE`, `gsm8k.py:22-34,87-108`) — no
  tolerance, no sympy. A **code sandbox already exists** (`execution.py:134-211`, subprocess +
  timeout + memory cap), used by HumanEval (`humaneval.py:79-97`).
- **A verifier is just a `Task`** (`tasks/common.py:10-51`): implement `eval_type`, `num_examples`,
  `get_example` (returns a messages conversation), `evaluate`, and **`reward`** (required for RL).
- **Expand verifiers:** (a) MATH-style task with **sympy** symbolic equivalence in `evaluate`/`reward`
  (parse `\boxed{}`); (b) **unit-tested code** by reusing `execute_code` from `execution.py`;
  (c) numeric-with-tolerance (single-digit tokenization makes exact match cleaner too). Generalize
  `chat_rl.py`'s hardwired task selection to train against a mixture.
- Reward-shape for planning/backtracking; add mild CoT-length control; curriculum easy→hard.

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
- `tasks/` today has **only GSM8K for math** (MMLU/ARC are multiple-choice; HumanEval is code).
  **Add `tasks/math.py`** (competition math, sympy-checked) subclassing `Task`, then register it in
  the `chat_eval.py` factory + `all_tasks` (`chat_eval.py:162,204-211`), optionally the SFT mixture
  (`chat_sft.py:166-179`) and RL (`chat_rl.py`).
- Track pass@1 (GSM8K + MATH), CoT length, tool-use rate, and `val_bpb` on held-out technical text
  (`token_bytes.pt` machinery, `tok_train.py:79-91`). Freeze a fixed eval set from day one so every
  pipeline change is measurable.

## 8. Sequencing / milestones
0. **Data corpus (milestone 0)** — ✅ *sourced & wired this session.* Corpus = **FineMath-4plus**
   (`HuggingFaceTB/finemath`, config `finemath-4plus`): 64 parquet shards, 18GB, **9.57B tokens /
   6.7M docs** (`text` + `token_count` columns) — ~4× the d12 ~2.3B horizon. Downloaded to `$HF_HOME`
   and symlinked into `~/.cache/nanochat/finemath4plus/` (last shard auto-selected as val: train
   9.42B / val 0.15B). Wired via a new `NANOCHAT_BASE_DATA_DIR` env override in `dataset.py` (ClimbMix
   default untouched). Reproduce: `snapshot_download("HuggingFaceTB/finemath", allow_patterns=
   "finemath-4plus/*.parquet")` → symlink the shards → `export NANOCHAT_BASE_DATA_DIR=…`.
   **Refinement (optional, pre-pretrain):** FineMath shards aren't globally shuffled like ClimbMix;
   the loader reads shards in order, so consider shuffling shard order (or a loader shuffle buffer).
1. **Tokenizer + model knobs** — ✅ *done this session.* `--single-digit-numbers`, `--tie-embeddings`
   (both param-count asserts fixed), the `num_scaling_params` MTP crash fix, AND a **16,384-vocab
   single-digit BPE trained on FineMath** (`~/.cache/nanochat-math/tokenizer/`; general 32K tokenizer
   left intact). At half the vocab it matches the 32K tokenizer's on-math compression (3.27 vs 3.29
   bytes/tok). **Config locked** (d12, vocab 16384): the target **d12 + MLA + tie + MTP1 + gated ≈
   110M total / 103.6M scaling → ~2.07B-token horizon** (vs 9.42B corpus ⇒ ~4.5 epochs headroom).
2. **Pretrain baseline** — ✅ *done this session.* Trained d12 + MLA + tie + MTP1 + gated (~110M),
   `--window-pattern L` (SDPA has no efficient sliding-window on this no-FA3 Blackwell), 2,353 steps /
   **1.23B tokens** (~1.9h, 183K tok/s, peak 38GB). **Val bpb 12.09 → 0.997** (clean monotonic
   convergence). Checkpoint: `~/.cache/nanochat-math/base_checkpoints/finemath-d12/` (step 2353).
   Qualitative check: "derivative of x^2 is" → "2x" (correct); base-model repetition/web-artifacts as
   expected pre-SFT. Shards used unshuffled (fine here). To train longer, raise `--target-param-data-ratio`.
3. **Distillation (milestone 3)** — ✅ *done this session.* Teacher = **open R1 traces**
   (`open-r1/OpenR1-Math-220k`, `default`/curated config, 93,733 problems / 193,767 R1 generations)
   — decided with the user (free, already answer-verified, R1-quality; no API cost). Built
   `scripts/prepare_math_cot.py`: reject-samples to correct-final-answer traces (OpenR1's
   `correctness_math_verify` flag as primary signal; **our sympy grader `tasks/math.answers_equal`
   rescues flag-False traces** — 1,844 rescued, honoring the verifier step), length-filters to the
   base model's 2048-token context, and keeps the **shortest** correct trace per problem (most
   learnable for ~110M, packs more/batch). The literal `<think>…</think>` tags in the R1 traces
   **are** the thinking-control format — no tokenizer/embedding surgery. Output = CustomJSON-format
   JSONL at `~/.cache/nanochat-math/math_cot/{train,val}.jsonl` (**8,324 train / 500 val**). Length
   is the dominant filter (only ~9% of problems yield a ≤2048-tok trace; p50=1621, p90=1964).
   Reproduce: `NANOCHAT_BASE_DIR=…nanochat-math … .venv/bin/python -m scripts.prepare_math_cot
   --config default --max-seq-len 2048 --max-traces 60000 --val-size 500`.
   **Lever if evals want longer reasoning:** raise `--max-seq-len` (+ YaRN, held in reserve) to
   admit more of the R1 trace-length distribution — the current 2048 cap discards the long tail.
4. **SFT (milestone 4):** wired the CoT set into `scripts/chat_sft.py` via a new `--math-cot-epochs`
   (default 3, `CustomJSON` on the prepared JSONL, added to both train + val mixtures; auto-skips if
   the file is absent). Train from `finemath-d12` base; measure GSM8K/MATH pass@1 (already registered
   in `chat_eval.py`) + CoT quality. **In progress this session.**
5. **RLVR:** ✅ MATH/sympy verifier landed (`tasks/math.py`, `extract_boxed`/`answers_equal`
   reusable as an RL reward). **Pending:** add code verifier (reuse `execution.py`), generalize
   `chat_rl.py`'s hardwired GSM8K, GRPO, curriculum.
6. **(Optional) architecture:** only if 2–5 plateau below target.

## Risks / notes
- **Teacher access/cost** is a dependency for the primary (distillation) path — budget for it.
- The hard part is **data assembly + verifier coverage**, not the modeling changes.
- Keep every change opt-in and on `current`; preserve the `--depth` scaling story where possible
  (vocab/tying changes touch it — re-verify the token-budget derivation after trimming vocab).
- **Deprioritized under this goal:** MoE (memory-bound single-GPU, breaks the `--depth` dial,
  tiny active params — wrong lever for a tiny model), and the parked items in the features doc.
