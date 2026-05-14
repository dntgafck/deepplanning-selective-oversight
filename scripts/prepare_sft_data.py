#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import fire


DEFAULT_MAX_SEQ_LEN = 16384
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _load_tokenizer(base_model: str) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


def _build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = list(record["input_messages"])
    pair_id = record.get("pair_id", "<unknown>")
    if len(messages) != 2:
        raise ValueError(f"expected 2 input messages, got {len(messages)} for {pair_id}")
    if messages[0].get("role") != "system":
        raise ValueError(f"first message must be system for {pair_id}")
    if messages[1].get("role") != "user":
        raise ValueError(f"second message must be user for {pair_id}")
    messages.append({"role": "assistant", "content": record["target_text"]})
    return messages


def _tokenized_len(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _held_out_task_keys(split: dict[str, Any]) -> set[str]:
    task_keys: set[str] = set()
    for level_name, task_ids in split["splits"]["held_out"].items():
        level_num = int(str(level_name).split("_")[-1])
        for task_id in task_ids:
            task_keys.add(f"level_{level_num}:{task_id}")
    return task_keys


def _length_stats(lengths: list[int]) -> dict[str, int | None]:
    if not lengths:
        return {"len_p50": None, "len_p95": None, "len_p99": None, "len_max": None}
    values = sorted(lengths)

    def percentile_index(percentile: float) -> int:
        return min(len(values) - 1, int(percentile * len(values)))

    return {
        "len_p50": values[percentile_index(0.50)],
        "len_p95": values[percentile_index(0.95)],
        "len_p99": values[percentile_index(0.99)],
        "len_max": values[-1],
    }


def _counter_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def process_split(
    name: str,
    src_path: Path,
    out_path: Path,
    held_out_task_keys: set[str],
    tokenizer: Any,
    max_seq_len: int,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "in": 0,
        "out": 0,
        "dropped_held_out_leak": 0,
        "dropped_too_long": 0,
        "label_dist": Counter(),
        "source_system_dist": Counter(),
        "level_dist": Counter(),
        "hook_dist": Counter(),
    }
    kept_lengths: list[int] = []
    all_lengths: list[int] = []

    with src_path.open(encoding="utf-8") as input_file, out_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            stats["in"] += 1

            if record["task_key"] in held_out_task_keys:
                stats["dropped_held_out_leak"] += 1
                continue

            messages = _build_messages(record)
            token_count = _tokenized_len(tokenizer, messages)
            all_lengths.append(token_count)

            if token_count > max_seq_len:
                stats["dropped_too_long"] += 1
                continue

            output_record = {
                "messages": messages,
                "pair_id": record["pair_id"],
                "decision_label": record["decision_label"],
                "source_system": record["source_system"],
                "level": record["level"],
                "hook": record["hook"],
                "task_key": record["task_key"],
                "tokenized_len": token_count,
            }
            output_file.write(
                json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            stats["out"] += 1
            kept_lengths.append(token_count)
            stats["label_dist"][record["decision_label"]] += 1
            stats["source_system_dist"][record["source_system"]] += 1
            stats["level_dist"][str(record["level"])] += 1
            stats["hook_dist"][record["hook"]] += 1

            if stats["in"] % 500 == 0:
                print(
                    f"  [{name}] processed {stats['in']} pairs, kept {stats['out']}",
                    flush=True,
                )

    if stats["dropped_held_out_leak"]:
        raise RuntimeError(
            f"FATAL: {stats['dropped_held_out_leak']} held-out task_keys leaked "
            f"into {name}; refusing to continue."
        )

    return _counter_stats(
        stats
        | _length_stats(kept_lengths)
        | {
            "all_len_max": max(all_lengths) if all_lengths else None,
            "source_path": str(src_path),
            "output_path": str(out_path),
        }
    )


def _report_config(in_dir: Path, out_dir: Path, max_seq_len: int, base_model: str) -> dict:
    return {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "max_seq_len": max_seq_len,
        "base_model": base_model,
    }


def main(
    in_dir: str,
    out_dir: str,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    base_model: str = DEFAULT_BASE_MODEL,
) -> None:
    """Prepare LoRA SFT train/val JSONL files for ms-swift."""
    input_dir = Path(in_dir)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = input_dir / "task_split_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    held_out_keys = _held_out_task_keys(split)
    print(
        f"Loaded {len(held_out_keys)} held-out task_keys "
        "(these will be asserted absent)."
    )

    held_out_path = input_dir / "held_out.jsonl"
    if not held_out_path.exists():
        raise FileNotFoundError(
            f"Expected held_out.jsonl at {held_out_path}; this script only "
            "confirms its presence and does not read it."
        )
    print(
        f"Confirmed held_out.jsonl present at {held_out_path}; "
        "this script does not open it."
    )

    print(f"Loading tokenizer {base_model} ...")
    tokenizer = _load_tokenizer(base_model)

    print(f"Processing train split (max_seq_len={max_seq_len}) ...")
    train_stats = process_split(
        "train",
        input_dir / "train.jsonl",
        output_dir / "train_swift.jsonl",
        held_out_keys,
        tokenizer,
        max_seq_len,
    )

    print(f"Processing val split (max_seq_len={max_seq_len}) ...")
    val_stats = process_split(
        "val",
        input_dir / "val.jsonl",
        output_dir / "val_swift.jsonl",
        held_out_keys,
        tokenizer,
        max_seq_len,
    )

    config = _report_config(input_dir, output_dir, max_seq_len, base_model)
    stratification_report = {"config": config, "train": train_stats, "val": val_stats}
    _write_json(output_dir / "stratification_report.json", stratification_report)
    _write_json(
        output_dir / "length_filter_report.json",
        {
            "config": config,
            "train": {
                "in": train_stats["in"],
                "out": train_stats["out"],
                "dropped_too_long": train_stats["dropped_too_long"],
                "kept_len_p50": train_stats["len_p50"],
                "kept_len_p95": train_stats["len_p95"],
                "kept_len_p99": train_stats["len_p99"],
                "kept_len_max": train_stats["len_max"],
                "all_len_max": train_stats["all_len_max"],
            },
            "val": {
                "in": val_stats["in"],
                "out": val_stats["out"],
                "dropped_too_long": val_stats["dropped_too_long"],
                "kept_len_p50": val_stats["len_p50"],
                "kept_len_p95": val_stats["len_p95"],
                "kept_len_p99": val_stats["len_p99"],
                "kept_len_max": val_stats["len_max"],
                "all_len_max": val_stats["all_len_max"],
            },
        },
    )

    print("\n========== SUMMARY ==========")
    for split_name, stats in (("train", train_stats), ("val", val_stats)):
        keep_rate = stats["out"] / stats["in"] if stats["in"] else 0
        drop_rate = stats["dropped_too_long"] / stats["in"] if stats["in"] else 0
        print(f"\n[{split_name}]  in={stats['in']}  out={stats['out']}  ({keep_rate:.1%} kept)")
        print(
            f"  dropped too long (> {max_seq_len}): {stats['dropped_too_long']} "
            f"({drop_rate:.1%})"
        )
        print(
            "  token length on kept: "
            f"p50={stats['len_p50']} p95={stats['len_p95']} "
            f"p99={stats['len_p99']} max={stats['len_max']}"
        )
        print(f"  labels: {stats['label_dist']}")
        print(f"  sources: {stats['source_system_dist']}")
        print(f"  levels: {stats['level_dist']}  hooks: {stats['hook_dist']}")

    print("\nWrote:")
    print(f"  {output_dir / 'train_swift.jsonl'}")
    print(f"  {output_dir / 'val_swift.jsonl'}")
    print(f"  {output_dir / 'stratification_report.json'}")
    print(f"  {output_dir / 'length_filter_report.json'}")


if __name__ == "__main__":
    fire.Fire(main)
