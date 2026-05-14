#!/usr/bin/env bash
# Full headline LoRA training — Qwen2.5-7B-Instruct + ms-swift + all-linear + bf16.
#
# Prerequisites:
#   - Check A passed on out/train_swift.jsonl and out/val_swift.jsonl
#   - Check B passed on the pilot adapter
#
# Wall-clock target: ~3-4h on H100 80GB at max_length=8192.

set -euo pipefail

OUT_DIR=${OUT_DIR:-out}
TRAIN=${TRAIN:-${OUT_DIR}/train_swift.jsonl}
VAL=${VAL:-${OUT_DIR}/val_swift.jsonl}
ADAPTER_OUT=${ADAPTER_OUT:-${OUT_DIR}/headline_lora}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-overseer-lora-headline}
RUN_NAME=${RUN_NAME:-qwen25-7b-overseer-r16-3ep}

CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tuner_type lora \
  --dataset "${TRAIN}" \
  --val_dataset "${VAL}" \
  --dataset_num_proc 8 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --weight_decay 0.0 \
  --gradient_checkpointing true \
  --max_length 8192 \
  --group_by_length true \
  --output_dir "${ADAPTER_OUT}" \
  --save_strategy steps \
  --save_steps 200 \
  --eval_strategy steps \
  --eval_steps 200 \
  --logging_steps 10 \
  --save_total_limit 4 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --seed ${SEED} \
  --report_to wandb \
  --run_name "${RUN_NAME}"

echo "Headline training complete: ${ADAPTER_OUT}"
echo "Final adapter at: ${ADAPTER_OUT}/checkpoint-best (load_best_model_at_end=true)"
echo ""
echo "Next steps:"
echo "  1. Validate adapter on val with: python check_b_eval.py --adapter ${ADAPTER_OUT}/checkpoint-best --val ${VAL} --n 500"
echo "  2. Serve via vLLM (see serve_vllm.sh)"
echo "  3. Run end-to-end C2-lora and C2-deepseek-lora against held_out 24"
