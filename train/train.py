#!/usr/bin/env python3
"""
LoRA SFT training for Qwen2.5-7B-Instruct overseer.

Uses TRL's SFTTrainer + PEFT. Single script for both pilot (Check B gate)
and headline runs — select via --mode pilot or --mode headline.

What this replaces:
  - run_check_b_pilot.sh + run_headline_training.sh + their swift sft commands

What it does NOT do:
  - Touch held_out.jsonl (NEVER opened by this script)
  - Apply enable_thinking=False (Qwen2.5 has no thinking mode)
  - Use ms-swift or modelscope (TRL+PEFT only)

Usage:
    python train_lora.py --mode pilot                 # ~30-45 min on H100
    python train_lora.py --mode headline              # ~3-4h on H100

    # Custom paths:
    python train_lora.py --mode headline \\
        --train_path out/train_swift.jsonl \\
        --val_path out/val_swift.jsonl \\
        --output_dir out/headline_lora

W&B logging is on by default; export WANDB_API_KEY or run `wandb login` first.
Set --report_to none to disable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ============================================================================
# Configuration profiles for pilot vs headline.
# ============================================================================

@dataclass
class TrainProfile:
    name: str
    num_train_epochs: float
    eval_steps: int
    save_steps: int
    save_strategy: str
    eval_subsample: Optional[int]   # subsample val to this size for in-loop eval (None = use all)
    train_subsample: Optional[int]  # subsample train to this size (None = use all)
    run_name: str
    wandb_project: str


PROFILES = {
    "pilot": TrainProfile(
        name="pilot",
        num_train_epochs=1.0,
        eval_steps=25,
        save_steps=999999,         # save once at end via save_strategy=epoch
        save_strategy="epoch",
        eval_subsample=200,        # 200 pairs for in-loop eval — keeps pilot under 1h total
        train_subsample=1000,
        run_name="pilot-qwen25-r16-1ep-1k",
        wandb_project="overseer-lora-pilot",
    ),
    "headline": TrainProfile(
        name="headline",
        num_train_epochs=3.0,
        eval_steps=200,
        save_steps=200,
        save_strategy="steps",
        eval_subsample=300,        # 300 pairs for in-loop eval (full val saved for offline check_b_eval)
        train_subsample=None,
        run_name="headline-qwen25-r16-3ep",
        wandb_project="overseer-lora-headline",
    ),
}


# ============================================================================
# Data loading. Each line of the jsonl is one SFTPair, format:
#   {"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}], "pair_id":..., ...}
# Other keys are metadata (decision_label, source_system, level, hook, task_key, tokenized_len)
# and are ignored at training time.
# ============================================================================

def load_jsonl(path: Path, subsample: Optional[int], seed: int) -> Dataset:
    with path.open() as f:
        records = [json.loads(line) for line in f]
    if subsample is not None and subsample < len(records):
        rng = random.Random(seed)
        rng.shuffle(records)
        records = records[:subsample]
    # SFTTrainer accepts datasets with a "messages" column directly.
    # Strip metadata to make the dataset smaller in memory.
    dataset_records = [{"messages": r["messages"]} for r in records]
    return Dataset.from_list(dataset_records)


# ============================================================================
# Main.
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "headline"], required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_path", default="out/train_swift.jsonl")
    parser.add_argument("--val_path", default="out/val_swift.jsonl")
    parser.add_argument("--output_dir", default=None,
                        help="Default: out/pilot_lora or out/headline_lora based on --mode")
    parser.add_argument("--max_seq_length", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    # LoRA hyperparameters — match the locked design memo §8.2.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Training hyperparameters.
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    # Attention implementation — flash_attention_2 if available, sdpa otherwise.
    parser.add_argument("--attn_impl", default=None,
                        help="Default: auto-detect. flash_attention_2 if importable, else sdpa.")
    args = parser.parse_args()

    profile = PROFILES[args.mode]
    if args.output_dir is None:
        args.output_dir = f"out/{profile.name}_lora"

    # ------------------------------------------------------------------------
    # Auto-detect attention implementation.
    # ------------------------------------------------------------------------
    if args.attn_impl is None:
        try:
            import flash_attn  # noqa: F401
            args.attn_impl = "flash_attention_2"
            print(f"[setup] flash-attn is importable; using {args.attn_impl}")
        except ImportError:
            args.attn_impl = "sdpa"
            print(f"[setup] flash-attn not available; falling back to {args.attn_impl}")

    # ------------------------------------------------------------------------
    # W&B environment.
    # ------------------------------------------------------------------------
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", profile.wandb_project)

    # ------------------------------------------------------------------------
    # Tokenizer and datasets.
    # ------------------------------------------------------------------------
    print(f"[data] loading tokenizer {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    # Ensure pad_token is set and DISTINCT from eos so the model still emits eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Common Qwen footgun: pad and eos identical means model never emits eos at gen time.
    # Workaround: use a different special token as pad if available.
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Look for an unused special token to use as pad.
        if "<|fim_pad|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|fim_pad|>"
            print(f"[data] separated pad_token from eos_token: pad={tokenizer.pad_token!r}")

    print(f"[data] loading train from {args.train_path}")
    train_ds = load_jsonl(Path(args.train_path), profile.train_subsample, args.seed)
    print(f"[data]   train size after subsample: {len(train_ds)}")

    print(f"[data] loading val from {args.val_path}")
    val_ds = load_jsonl(Path(args.val_path), profile.eval_subsample, args.seed)
    print(f"[data]   val size after subsample: {len(val_ds)}")

    # ------------------------------------------------------------------------
    # LoRA config.
    # ------------------------------------------------------------------------
    # For Qwen2.5 (standard dense LM), all-linear maps to these 7 leaves per layer.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ------------------------------------------------------------------------
    # SFT config. assistant_only_loss=True is the critical bit: TRL applies
    # its patched Qwen2.5 chat template that has explicit {% generation %}
    # markers, then masks the prompt tokens (loss only on assistant turn).
    # ------------------------------------------------------------------------
    sft_config = SFTConfig(
        # Output and runtime
        output_dir=args.output_dir,
        run_name=profile.run_name,
        seed=args.seed,
        report_to=args.report_to if args.report_to != "none" else "none",
        # Optimization
        num_train_epochs=profile.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Eval / save
        eval_strategy="steps",
        eval_steps=profile.eval_steps,
        save_strategy=profile.save_strategy,
        save_steps=profile.save_steps,
        save_total_limit=4,
        load_best_model_at_end=(profile.save_strategy == "steps"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        # Dataset processing
        max_length=args.max_seq_length,
        dataset_num_proc=8,
        # The headline of this switch: TRL masks the prompt, computes loss
        # only on the assistant turn, using the ChatML "{% generation %}"
        # markers from its patched Qwen2.5 template.
        assistant_only_loss=True,
        # Packing off — keeps each pair isolated, no cross-contamination.
        packing=False,
        # Bucketing by length for the long-tail efficiency win.
        group_by_length=True,
        # Be explicit so we never accidentally enable any of these.
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------------
    # Model load + trainer.
    # ------------------------------------------------------------------------
    # SFTTrainer handles model loading when passed a string. Pass kwargs via
    # model_init_kwargs (TRL >= 0.15).
    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": args.attn_impl,
        "trust_remote_code": True,
    }
    sft_config.model_init_kwargs = model_init_kwargs

    print(f"[train] starting {profile.name} run")
    print(f"[train]   base model: {args.base_model}")
    print(f"[train]   attn impl: {args.attn_impl}")
    print(f"[train]   max_length: {args.max_seq_length}")
    print(f"[train]   effective batch: {args.per_device_batch_size * args.gradient_accumulation_steps}")
    print(f"[train]   epochs: {profile.num_train_epochs}")
    print(f"[train]   output_dir: {args.output_dir}")

    trainer = SFTTrainer(
        model=args.base_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # ------------------------------------------------------------------------
    # Save final adapter. For headline, load_best_model_at_end means the
    # in-memory model is already the best checkpoint. For pilot, save at end.
    # ------------------------------------------------------------------------
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[train] final adapter saved to {final_dir}")

    print("\n========== DONE ==========")
    print(f"Adapter: {final_dir}")
    print(f"Next:")
    if profile.name == "pilot":
        print(f"  python check_b_eval.py --adapter {final_dir} --val out/val_swift.jsonl --n 100")
    else:
        print(f"  python check_b_eval.py --adapter {final_dir} --val out/val_swift.jsonl --n 500")
        print(f"  ADAPTER_PATH={final_dir} bash serve_vllm.sh")


if __name__ == "__main__":
    sys.exit(main())