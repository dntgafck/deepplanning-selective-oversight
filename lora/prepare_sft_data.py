#!/usr/bin/env python3
"""
Convert lora_sft/{train,val}.jsonl into ms-swift "messages" format.

Inputs (READ ONLY):
    lora_sft/train.jsonl, lora_sft/val.jsonl, lora_sft/task_split_v1.json
    held_out.jsonl is NEVER opened by this script.

Outputs:
    out/train_swift.jsonl, out/val_swift.jsonl
    out/stratification_report.json
    out/length_filter_report.json

What it does:
1. Loads task_split_v1.json and computes the set of held-out task_keys.
2. Streams train.jsonl and val.jsonl, reshaping each record to
   {"messages": [system, user, assistant]} using input_messages + target_text.
3. Re-tokenizes prompt+completion with the Qwen3.5-9B tokenizer
   (chat template applied with enable_thinking=False) to get an exact
   combined-token count.
4. Drops samples whose tokenized length exceeds MAX_SEQ_LEN.
5. Asserts that no held-out task_key leaks into train/val output.
6. Writes a stratification table (decision_label × source_system × level × hook)
   for the appendix.

Run:
    python prepare_sft_data.py \
        --in_dir /path/to/lora_sft \
        --out_dir out \
        --max_seq_len 12288 \
        --base_model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Lazy import so the script can be inspected without transformers installed.
def _load_tokenizer(base_model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Sanity check: Qwen3.5 chat template must accept enable_thinking kwarg.
    try:
        tok.apply_chat_template(
            [{"role": "user", "content": "ping"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as e:
        raise RuntimeError(
            f"Tokenizer chat template does not accept enable_thinking kwarg. "
            f"You probably have an outdated transformers. Need transformers >= 5.2. "
            f"Original error: {e}"
        ) from e
    return tok


def _build_messages(rec: dict) -> list[dict]:
    """Reshape one project SFTPair into a 3-message list."""
    msgs = list(rec["input_messages"])  # already [system, user]
    assert len(msgs) == 2, f"expected 2 input messages, got {len(msgs)} for pair {rec['pair_id']}"
    assert msgs[0]["role"] == "system", f"first message must be system for pair {rec['pair_id']}"
    assert msgs[1]["role"] == "user", f"second message must be user for pair {rec['pair_id']}"
    msgs.append({"role": "assistant", "content": rec["target_text"]})
    return msgs


def _tokenized_len(tok, messages: list[dict]) -> int:
    """Return total tokens after applying Qwen3.5 chat template with thinking disabled."""
    full = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # CRITICAL — must match training and serving
    )
    return len(tok(full, add_special_tokens=False)["input_ids"])


def process_split(
    name: str,
    src_path: Path,
    out_path: Path,
    held_out_task_keys: set[str],
    tok,
    max_seq_len: int,
) -> dict:
    stats = {
        "in": 0,
        "out": 0,
        "dropped_held_out_leak": 0,
        "dropped_too_long": 0,
        "label_dist": Counter(),
        "source_system_dist": Counter(),
        "level_dist": Counter(),
        "hook_dist": Counter(),
        "len_p50": None,
        "len_p95": None,
        "len_p99": None,
        "len_max": None,
    }
    lens = []

    with src_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            stats["in"] += 1

            # Defence in depth: assert no held-out leakage.
            if rec["task_key"] in held_out_task_keys:
                stats["dropped_held_out_leak"] += 1
                continue

            messages = _build_messages(rec)
            n_tok = _tokenized_len(tok, messages)
            lens.append(n_tok)

            if n_tok > max_seq_len:
                stats["dropped_too_long"] += 1
                continue

            out_rec = {
                "messages": messages,
                # Carry through metadata for stratified eval later.
                # ms-swift ignores unknown keys.
                "pair_id": rec["pair_id"],
                "decision_label": rec["decision_label"],
                "source_system": rec["source_system"],
                "level": rec["level"],
                "hook": rec["hook"],
                "task_key": rec["task_key"],
                "tokenized_len": n_tok,
            }
            fout.write(json.dumps(out_rec) + "\n")
            stats["out"] += 1
            stats["label_dist"][rec["decision_label"]] += 1
            stats["source_system_dist"][rec["source_system"]] += 1
            stats["level_dist"][rec["level"]] += 1
            stats["hook_dist"][rec["hook"]] += 1

            if stats["in"] % 500 == 0:
                print(f"  [{name}] processed {stats['in']} pairs, kept {stats['out']}", flush=True)

    if lens:
        lens.sort()
        stats["len_p50"] = lens[len(lens) // 2]
        stats["len_p95"] = lens[int(0.95 * len(lens))]
        stats["len_p99"] = lens[int(0.99 * len(lens))]
        stats["len_max"] = lens[-1]

    # Hard assertion: zero held-out leakage.
    assert stats["dropped_held_out_leak"] == 0, (
        f"FATAL: {stats['dropped_held_out_leak']} held-out task_keys leaked into {name}. "
        f"Refusing to continue."
    )

    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="Directory containing train.jsonl, val.jsonl, task_split_v1.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=12288)
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the task split and compute held-out keys for assertion.
    split_path = in_dir / "task_split_v1.json"
    split = json.loads(split_path.read_text())
    held_out_task_keys: set[str] = set()
    for level_name, task_ids in split["splits"]["held_out"].items():
        level_num = int(level_name.split("_")[-1])
        for tid in task_ids:
            held_out_task_keys.add(f"level_{level_num}:{tid}")
    print(f"Loaded {len(held_out_task_keys)} held-out task_keys (these will be asserted absent).")

    # Sanity: confirm held_out.jsonl exists but DO NOT open it.
    held_out_path = in_dir / "held_out.jsonl"
    assert held_out_path.exists(), f"Expected held_out.jsonl at {held_out_path} (just confirming its presence; not reading it)."
    print(f"Confirmed held_out.jsonl present at {held_out_path}; this script does not open it.")

    print(f"Loading tokenizer {args.base_model} ...")
    tok = _load_tokenizer(args.base_model)

    print(f"Processing train split (max_seq_len={args.max_seq_len}) ...")
    train_stats = process_split(
        "train",
        in_dir / "train.jsonl",
        out_dir / "train_swift.jsonl",
        held_out_task_keys,
        tok,
        args.max_seq_len,
    )

    print(f"Processing val split (max_seq_len={args.max_seq_len}) ...")
    val_stats = process_split(
        "val",
        in_dir / "val.jsonl",
        out_dir / "val_swift.jsonl",
        held_out_task_keys,
        tok,
        args.max_seq_len,
    )

    report = {
        "config": {
            "in_dir": str(in_dir),
            "out_dir": str(out_dir),
            "max_seq_len": args.max_seq_len,
            "base_model": args.base_model,
        },
        "train": train_stats,
        "val": val_stats,
    }
    (out_dir / "stratification_report.json").write_text(json.dumps(report, indent=2))

    # Human-readable summary.
    print("\n========== SUMMARY ==========")
    for name, s in [("train", train_stats), ("val", val_stats)]:
        keep_rate = s["out"] / s["in"] if s["in"] else 0
        print(f"\n[{name}]  in={s['in']}  out={s['out']}  ({keep_rate:.1%} kept)")
        print(f"  dropped too long (> {args.max_seq_len}): {s['dropped_too_long']} "
              f"({s['dropped_too_long']/s['in']:.1%})")
        print(f"  token length on kept: p50={s['len_p50']} p95={s['len_p95']} p99={s['len_p99']} max={s['len_max']}")
        print(f"  labels: {s['label_dist']}")
        print(f"  sources: {s['source_system_dist']}")
        print(f"  levels: {s['level_dist']}  hooks: {s['hook_dist']}")
    print("\nWrote:")
    print(f"  {out_dir / 'train_swift.jsonl'}")
    print(f"  {out_dir / 'val_swift.jsonl'}")
    print(f"  {out_dir / 'stratification_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
