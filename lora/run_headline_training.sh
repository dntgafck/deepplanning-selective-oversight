#!/usr/bin/env bash
# Full headline LoRA training — Qwen3.5-9B + ms-swift + all-linear + bf16.
#
# Prerequisites:
#   - Check A passed on out/train_swift.jsonl and out/val_swift.jsonl
#   - Check B passed on the pilot adapter
#
# Wall-clock target: 12-18h on H100 80GB at max_length=12288.

set -euo pipefail

OUT_DIR=${OUT_DIR:-out}
TRAIN=${TRAIN:-${OUT_DIR}/train_swift.jsonl}
VAL=${VAL:-${OUT_DIR}/val_swift.jsonl}
ADAPTER_OUT=${ADAPTER_OUT:-${OUT_DIR}/headline_lora}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-overseer-lora-headline}
RUN_NAME=${RUN_NAME:-qwen35-9b-overseer-r16-3ep}

CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model Qwen/Qwen3.5-9B \
  --train_type lora \
  --dataset "${TRAIN}" \
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
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --weight_decay 0.0 \
  --gradient_checkpointing true \
  --max_length 12288 \
  --group_by_length true \
  --freeze_vit true \
  --freeze_aligner true \
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
echo "Final adapter is at: ${ADAPTER_OUT}/checkpoint-best (load_best_model_at_end=true)"
echo ""
echo "Next steps:"
echo "  1. Validate adapter on val set with a fresh pass through check_b_eval.py"
echo "  2. Serve via vLLM (see serve_vllm.sh)"
echo "  3. Run end-to-end C2-lora and C2-deepseek-lora against held_out 24"
