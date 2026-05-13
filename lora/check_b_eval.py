#!/usr/bin/env python3
"""
Check B eval: decision-token accuracy and parse success on 100 val pairs.

Loads the pilot LoRA adapter onto Qwen/Qwen3.5-9B (bf16, single GPU),
samples N val pairs deterministically, generates the assistant turn with
greedy decoding and enable_thinking=False, and reports:
   - parse success rate (>= 95% to pass)
   - decision-token accuracy vs teacher label (>= 70% to pass)

This is the offline pre-flight gate. End-to-end deployment evaluation
(against held_out.jsonl) is a separate run.

Run:
    python check_b_eval.py \
        --adapter out/pilot_lora/checkpoint-<step> \
        --val out/val_swift.jsonl \
        --n 100 \
        --base_model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DECISION_RE = re.compile(r'"action"\s*:\s*"([^"]+)"')


def extract_action(text: str) -> str | None:
    """Try strict JSON parse first, then a permissive regex on first 'action' field."""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "action" in payload:
            return str(payload["action"])
    except json.JSONDecodeError:
        pass
    m = DECISION_RE.search(text)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to PEFT adapter directory")
    parser.add_argument("--val", required=True, help="Path to val_swift.jsonl")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=900)  # p99 of target_text on train is ~660
    parser.add_argument("--accuracy_threshold", type=float, default=0.70)
    parser.add_argument("--parse_threshold", type=float, default=0.95)
    args = parser.parse_args()

    # 1. Load val and subsample.
    with open(args.val) as f:
        val = [json.loads(line) for line in f]
    rng = random.Random(args.seed)
    rng.shuffle(val)
    val = val[:args.n]
    print(f"Loaded {len(val)} val pairs (seed={args.seed}).")

    # 2. Load model + adapter.
    print(f"Loading base {args.base_model} + adapter {args.adapter} ...")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # 3. Generate and score.
    results = []
    n_parse_ok = 0
    n_action_match = 0
    action_confusion = Counter()
    for i, rec in enumerate(val):
        # Build the prompt: system + user, with assistant turn left open.
        prompt_msgs = rec["messages"][:2]
        teacher_action = extract_action(rec["messages"][2]["content"]) or rec.get("decision_label")

        prompt_text = tok.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        # Strip any leading whitespace / stray template artifacts.
        gen_text = gen_text.strip()

        pred_action = extract_action(gen_text)
        parse_ok = pred_action is not None
        action_match = (pred_action == teacher_action) if parse_ok else False
        n_parse_ok += int(parse_ok)
        n_action_match += int(action_match)
        action_confusion[(teacher_action, pred_action)] += 1
        results.append({
            "pair_id": rec["pair_id"],
            "teacher_action": teacher_action,
            "pred_action": pred_action,
            "parse_ok": parse_ok,
            "action_match": action_match,
            "gen_text_preview": gen_text[:200],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(val)}] parse_ok={n_parse_ok} match={n_action_match}", flush=True)

    parse_rate = n_parse_ok / len(val)
    acc = n_action_match / len(val)
    pass_parse = parse_rate >= args.parse_threshold
    pass_acc = acc >= args.accuracy_threshold

    print("\n========== CHECK B ==========")
    print(f"Parse success: {n_parse_ok}/{len(val)} = {parse_rate:.2%} (>= {args.parse_threshold:.0%} required)  "
          f"{'PASS' if pass_parse else 'FAIL'}")
    print(f"Decision accuracy: {n_action_match}/{len(val)} = {acc:.2%} (>= {args.accuracy_threshold:.0%} required)  "
          f"{'PASS' if pass_acc else 'FAIL'}")
    print("Per (teacher, pred) confusion:")
    for k, v in sorted(action_confusion.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Save full report.
    out_path = Path(args.adapter).parent / "check_b_report.json"
    out_path.write_text(json.dumps({
        "n": len(val),
        "parse_rate": parse_rate,
        "decision_accuracy": acc,
        "passed": pass_parse and pass_acc,
        "passed_parse": pass_parse,
        "passed_accuracy": pass_acc,
        "results": results,
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in action_confusion.items()},
    }, indent=2))
    print(f"\nWrote {out_path}")

    sys.exit(0 if (pass_parse and pass_acc) else 1)


if __name__ == "__main__":
    main()
