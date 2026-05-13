#!/usr/bin/env bash
# Check B: pilot LoRA training as Gate 4 per experimental_protocol_v3.md.
#
# Trains a small LoRA (r=16, 1 epoch, 1000 random pairs sampled from
# train_swift.jsonl). Pass criteria are checked separately by
# check_b_eval.py against 100 val pairs:
#   (i) decision-token accuracy >= 70%
#   (ii) parse success >= 95%
#
# Wall-clock target: <= 1h on H100 80GB.

set -euo pipefail

OUT_DIR=${OUT_DIR:-out}
PILOT_TRAIN=${PILOT_TRAIN:-${OUT_DIR}/train_swift_pilot1k.jsonl}
VAL=${VAL:-${OUT_DIR}/val_swift.jsonl}
PILOT_OUT=${PILOT_OUT:-${OUT_DIR}/pilot_lora}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-overseer-lora-pilot}

# 1. Subsample 1000 pairs (deterministic).
python - <<PY
import json, random
random.seed(${SEED})
src = "${OUT_DIR}/train_swift.jsonl"
dst = "${PILOT_TRAIN}"
with open(src) as f:
    lines = f.readlines()
random.shuffle(lines)
with open(dst, "w") as f:
    f.writelines(lines[:1000])
print(f"Wrote {dst} with {len(lines[:1000])} pairs (seed={${SEED}})")
PY

# 2. Train pilot LoRA.
CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model Qwen/Qwen3.5-9B \
  --train_type lora \
  --dataset "${PILOT_TRAIN}" \
  --val_dataset "${VAL}" \
  --dataset_num_proc 8 \
  --enable_thinking false \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --gradient_checkpointing true \
  --max_length 12288 \
  --group_by_length true \
  --freeze_vit true \
  --freeze_aligner true \
  --output_dir "${PILOT_OUT}" \
  --save_strategy epoch \
  --eval_strategy steps \
  --eval_steps 25 \
  --logging_steps 5 \
  --save_total_limit 1 \
  --seed ${SEED} \
  --report_to wandb \
  --run_name pilot-r16-1ep-1k

echo "Pilot training complete: ${PILOT_OUT}"
echo "Next: python check_b_eval.py --adapter ${PILOT_OUT}/checkpoint-* --val ${VAL} --n 100"
