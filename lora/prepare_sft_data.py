#!/usr/bin/env python3
"""
Convert lora_sft/{train,val}.jsonl into the messages format expected by
train_lora.py.

Configured for Qwen/Qwen2.5-7B-Instruct.

Inputs (READ ONLY):
    lora_sft/train.jsonl, lora_sft/val.jsonl, lora_sft/task_split_v1.json
    held_out.jsonl is NEVER opened by this script.

Outputs:
    out/train_swift.jsonl, out/val_swift.jsonl
    out/stratification_report.json

(File names start with "swift_" for legacy compatibility with carry-over scripts;
the data format is plain JSONL with "messages" arrays — works for any TRL/PEFT trainer.)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_tokenizer(base_model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


def _build_messages(rec: dict) -> list[dict]:
    msgs = list(rec["input_messages"])
    assert len(msgs) == 2, f"expected 2 input messages, got {len(msgs)} for pair {rec['pair_id']}"
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    msgs.append({"role": "assistant", "content": rec["target_text"]})
    return msgs


def _tokenized_len(tok, messages: list[dict]) -> int:
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tok(full, add_special_tokens=False)["input_ids"])


def process_split(name, src_path, out_path, held_out_task_keys, tok, max_seq_len):
    stats = {
        "in": 0, "out": 0, "dropped_held_out_leak": 0, "dropped_too_long": 0,
        "label_dist": Counter(), "source_system_dist": Counter(),
        "level_dist": Counter(), "hook_dist": Counter(),
        "len_p50": None, "len_p95": None, "len_p99": None, "len_max": None,
    }
    lens = []
    with src_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            stats["in"] += 1
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
                print(f"  [{name}] processed {stats['in']} kept {stats['out']}", flush=True)
    if lens:
        lens.sort()
        stats["len_p50"] = lens[len(lens) // 2]
        stats["len_p95"] = lens[int(0.95 * len(lens))]
        stats["len_p99"] = lens[int(0.99 * len(lens))]
        stats["len_max"] = lens[-1]
    assert stats["dropped_held_out_leak"] == 0, f"FATAL: held-out leak into {name}"
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=12288)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads((in_dir / "task_split_v1.json").read_text())
    held_out_task_keys = set()
    for level_name, task_ids in split["splits"]["held_out"].items():
        level_num = int(level_name.split("_")[-1])
        for tid in task_ids:
            held_out_task_keys.add(f"level_{level_num}:{tid}")
    print(f"Loaded {len(held_out_task_keys)} held-out task_keys (assertion only).")

    held_out_path = in_dir / "held_out.jsonl"
    assert held_out_path.exists(), f"Expected held_out.jsonl at {held_out_path} (presence-check only)."
    print(f"Confirmed held_out.jsonl present at {held_out_path}; not opened here.")

    tok = _load_tokenizer(args.base_model)

    train_stats = process_split("train", in_dir / "train.jsonl", out_dir / "train_swift.jsonl",
                                 held_out_task_keys, tok, args.max_seq_len)
    val_stats = process_split("val", in_dir / "val.jsonl", out_dir / "val_swift.jsonl",
                               held_out_task_keys, tok, args.max_seq_len)

    report = {
        "config": {"in_dir": str(in_dir), "out_dir": str(out_dir),
                   "max_seq_len": args.max_seq_len, "base_model": args.base_model},
        "train": train_stats, "val": val_stats,
    }
    (out_dir / "stratification_report.json").write_text(json.dumps(report, indent=2))

    print("\n========== SUMMARY ==========")
    for name, s in [("train", train_stats), ("val", val_stats)]:
        kr = s["out"] / s["in"] if s["in"] else 0
        print(f"\n[{name}] in={s['in']} out={s['out']} ({kr:.1%} kept)")
        print(f"  too long: {s['dropped_too_long']} ({s['dropped_too_long']/s['in']:.1%})")
        print(f"  lens: p50={s['len_p50']} p95={s['len_p95']} p99={s['len_p99']} max={s['len_max']}")
        print(f"  labels: {s['label_dist']}")
        print(f"  sources: {s['source_system_dist']}")
        print(f"  levels: {s['level_dist']} hooks: {s['hook_dist']}")


if __name__ == "__main__":
    sys.exit(main())
