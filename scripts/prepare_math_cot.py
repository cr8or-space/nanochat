"""
Build a distilled long-CoT SFT set for the math model from an open R1-trace dataset
(default: open-r1/OpenR1-Math-220k). Milestone 3 of docs/tiny_reasoning_model.md.

Pipeline (per problem):
  1. Take the R1 `generations` (each already formatted as `<think>...</think>\n\nsolution`
     ending in a \boxed{...} final answer). The literal <think>/</think> tags ARE our
     "thinking control token" format -- no tokenizer/embedding surgery needed; the model
     learns them as ordinary text tokens.
  2. Reject-sample to correct-final-answer traces. A generation is kept if EITHER the
     dataset's own `correctness_math_verify[i]` is True OR our sympy grader
     (tasks/math.answers_equal, reused as the RL verifier) agrees. This uses our verifier
     as the plan intends while leaning on OpenR1's robust math_verify for the messy,
     multi-answer ground truths where our grader would over-reject.
  3. Length-filter: render the full conversation with the *math* tokenizer and keep it only
     if it fits within --max-seq-len (the base model's context). Among a problem's surviving
     correct traces, keep the SHORTEST (most learnable for a ~110M model, packs more
     problems per batch).

Output: JSONL in the CustomJSON conversation format (a list of {role, content} messages),
so scripts/chat_sft.py can consume it via tasks/customjson.py with no new Task class:
  [{"role":"user","content": problem}, {"role":"assistant","content": "<think>...</think>\n\n...\\boxed{ans}"}]

Run (from repo root, with the math base dir + FineMath data dir + HF token exported):
  NANOCHAT_BASE_DIR=/home/jbennett/.cache/nanochat-math PYTHONPATH=. \
    .venv/bin/python -m scripts.prepare_math_cot --max-traces 60000
"""

import os
import json
import argparse

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from tasks.math import extract_boxed, answers_equal


def parse_args():
    p = argparse.ArgumentParser(description="Build distilled long-CoT SFT set from open R1 traces")
    p.add_argument("--dataset", type=str, default="open-r1/OpenR1-Math-220k",
                   help="HF dataset id of R1 traces")
    p.add_argument("--config", type=str, default="default", help="dataset config/subset")
    p.add_argument("--split", type=str, default="train", help="dataset split")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="drop any conversation whose rendered length exceeds this (base model context)")
    p.add_argument("--max-traces", type=int, default=60000,
                   help="cap on total kept traces (bounds SFT time); -1 = no cap")
    p.add_argument("--val-size", type=int, default=500, help="held-out traces for the SFT val mixture")
    p.add_argument("--char-prefilter", type=int, default=12000,
                   help="skip generations longer than this many chars before tokenizing (cheap gate)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output dir (default: <base_dir>/math_cot)")
    p.add_argument("--limit-rows", type=int, default=-1,
                   help="only scan the first N dataset rows (for smoke tests); -1 = all")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = get_base_dir()
    out_dir = args.out_dir or os.path.join(base_dir, "math_cot")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    print(f"Loaded math tokenizer from {tokenizer_dir} (vocab {tokenizer.get_vocab_size()})")

    from datasets import load_dataset
    print(f"Loading {args.dataset} [{args.config}] split={args.split} ...")
    ds = load_dataset(args.dataset, args.config, split=args.split)
    n_rows = len(ds) if args.limit_rows < 0 else min(args.limit_rows, len(ds))
    print(f"Scanning {n_rows:,} problems (of {len(ds):,})")

    # Counters for a transparent report of where traces are lost.
    stats = {
        "rows": 0, "no_gen": 0, "gen_total": 0,
        "kept_correct": 0, "correct_flag": 0, "grader_rescued": 0,
        "dropped_wrong": 0, "dropped_too_long_chars": 0, "dropped_too_long_tokens": 0,
        "problems_with_trace": 0,
    }

    kept = []  # list of (n_tokens, conversation_messages)
    for idx in range(n_rows):
        row = ds[idx]
        stats["rows"] += 1
        problem = row.get("problem") or ""
        ref = row.get("answer") or ""
        gens = row.get("generations") or []
        flags = row.get("correctness_math_verify") or []
        if not problem or not gens:
            stats["no_gen"] += 1
            continue

        best = None  # (n_tokens, messages) shortest correct trace that fits
        for i, gen in enumerate(gens):
            stats["gen_total"] += 1
            if not gen:
                continue
            flag_ok = bool(flags[i]) if i < len(flags) else False
            # Reject-sample to correct traces. OpenR1's math_verify flag is the primary signal;
            # our sympy grader (tasks/math.answers_equal) is only invoked to RESCUE flag-False
            # traces -- this both honors "verify with our grader" and avoids running sympy on the
            # ~70% already flagged correct (a large speedup, same keep decision).
            if flag_ok:
                stats["correct_flag"] += 1
            else:
                pred = extract_boxed(gen)
                if pred is not None and ref and answers_equal(pred, ref):
                    stats["grader_rescued"] += 1
                else:
                    stats["dropped_wrong"] += 1
                    continue

            # cheap char pre-filter before the (heavier) exact tokenization
            if len(problem) + len(gen) > args.char_prefilter:
                stats["dropped_too_long_chars"] += 1
                continue
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": gen},
            ]
            ids, _ = tokenizer.render_conversation({"messages": messages}, max_tokens=10**9)
            n_tok = len(ids)
            if n_tok > args.max_seq_len:
                stats["dropped_too_long_tokens"] += 1
                continue
            if best is None or n_tok < best[0]:
                best = (n_tok, messages)

        if best is not None:
            kept.append(best)
            stats["kept_correct"] += 1
            stats["problems_with_trace"] += 1

        if (idx + 1) % 5000 == 0:
            print(f"  [{idx+1:,}/{n_rows:,}] kept={len(kept):,} "
                  f"dropped_wrong={stats['dropped_wrong']:,} "
                  f"too_long_tok={stats['dropped_too_long_tokens']:,}")
        if args.max_traces > 0 and len(kept) >= args.max_traces + args.val_size:
            print(f"Reached cap of {args.max_traces + args.val_size:,} traces, stopping scan early.")
            break

    print(f"\nKept {len(kept):,} correct traces that fit {args.max_seq_len} tokens.")
    if not kept:
        raise SystemExit("No traces kept -- check dataset schema / filters.")

    # Token length distribution (helps decide whether to raise seq len later).
    lens = sorted(t for t, _ in kept)
    def pct(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))]
    print(f"Token lengths: min={lens[0]} p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={lens[-1]}")

    # Deterministic split: last --val-size go to val. (Dataset was scanned in order; that's fine
    # -- the SFT TaskMixture reshuffles everything with a fixed seed anyway.)
    val = kept[-args.val_size:] if args.val_size > 0 else []
    train = kept[:-args.val_size] if args.val_size > 0 else kept
    if args.max_traces > 0:
        train = train[:args.max_traces]

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for _, messages in items:
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    print(f"Wrote {len(train):,} train -> {train_path}")
    print(f"Wrote {len(val):,} val   -> {val_path}")

    # Persist a small report next to the data.
    stats["train"] = len(train)
    stats["val"] = len(val)
    stats["token_len_p50"] = pct(0.5)
    stats["token_len_p90"] = pct(0.9)
    stats["token_len_max"] = lens[-1]
    with open(os.path.join(out_dir, "prepare_report.json"), "w") as f:
        json.dump({"args": vars(args), "stats": stats}, f, indent=2)
    print("\nReport:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
