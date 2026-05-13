#!/usr/bin/env python3
"""
Check A: round-trip parse sanity on the prepared training data.

Acts on:
    out/train_swift.jsonl, out/val_swift.jsonl

Does NOT touch held_out.jsonl.

For each pair, attempts to parse the assistant turn as JSON and validate
the expected v1.4 overseer schema. Pass = >= 99% of pairs parse cleanly
per `experimental_protocol_v3.md` Gate 3.

Run:
    python check_a_parse_sanity.py --in_dir out
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# v1.4 overseer schema, per c2_lite_design_memo_v1_4.md.
# Each hook has its own action space:
#   pre_tool / post_tool : action in {APPROVE_AND_CONTINUE, APPROVE_WITH_NUDGE, HARD_BLOCK}
#   final                : action in {approve, run_verification}
# In addition, "provide_guidance" and "correct_observation" appear as
# decision_labels in the data; they are mapped to the action field directly
# in the structured payload.
ALLOWED_ACTIONS = {
    "approve",
    "run_verification",
    "provide_guidance",
    "correct_observation",
    "APPROVE_AND_CONTINUE",
    "APPROVE_WITH_NUDGE",
    "HARD_BLOCK",
}


def check_payload(payload) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "top-level not a dict"
    if "action" not in payload:
        return False, "missing 'action' field"
    if payload["action"] not in ALLOWED_ACTIONS:
        return False, f"unknown action: {payload['action']!r}"
    return True, None


def check_split(path: Path) -> dict:
    stats = {
        "total": 0,
        "parsed_ok": 0,
        "parse_fail": 0,
        "schema_fail": 0,
        "fail_reasons": Counter(),
        "action_dist": Counter(),
    }
    with path.open() as f:
        for line in f:
            stats["total"] += 1
            rec = json.loads(line)
            assistant_turn = rec["messages"][-1]["content"]
            try:
                payload = json.loads(assistant_turn)
            except json.JSONDecodeError as e:
                stats["parse_fail"] += 1
                stats["fail_reasons"][f"json_decode:{type(e).__name__}"] += 1
                continue
            ok, reason = check_payload(payload)
            if not ok:
                stats["schema_fail"] += 1
                stats["fail_reasons"][f"schema:{reason}"] += 1
                continue
            stats["parsed_ok"] += 1
            stats["action_dist"][payload["action"]] += 1
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    all_pass = True
    report = {}
    for name in ["train", "val"]:
        path = in_dir / f"{name}_swift.jsonl"
        s = check_split(path)
        pass_rate = s["parsed_ok"] / s["total"] if s["total"] else 0
        ok = pass_rate >= args.threshold
        all_pass &= ok
        print(f"\n[{name}]")
        print(f"  total={s['total']}  parsed_ok={s['parsed_ok']}  ({pass_rate:.3%})")
        print(f"  parse_fail={s['parse_fail']}  schema_fail={s['schema_fail']}")
        if s["fail_reasons"]:
            print(f"  fail_reasons: {dict(s['fail_reasons'])}")
        print(f"  action_dist: {dict(s['action_dist'])}")
        print(f"  CHECK A {'PASS' if ok else 'FAIL'} (threshold {args.threshold:.0%})")
        report[name] = {
            "total": s["total"],
            "parsed_ok": s["parsed_ok"],
            "parse_fail": s["parse_fail"],
            "schema_fail": s["schema_fail"],
            "pass_rate": pass_rate,
            "passed": ok,
            "fail_reasons": dict(s["fail_reasons"]),
            "action_dist": dict(s["action_dist"]),
        }

    out_path = in_dir / "check_a_report.json"
    out_path.write_text(json.dumps({"threshold": args.threshold, "all_pass": all_pass, "splits": report}, indent=2))
    print(f"\nWrote {out_path}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()