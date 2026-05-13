#!/usr/bin/env bash
# vLLM serving for the trained LoRA adapter.
#
# Known caveat (per the May 2026 LoRA training memo, vLLM issue #38085):
# vLLM emits a warning of the form
#   "LoRA module 'language_model.model.layers.X.linear_attn.in_proj_a' in
#    adapter '...' is not in the model's supported LoRA target modules
#    [conv1d, down_proj, gate_up_proj, in_proj_ba, in_proj_qkv, in_proj_z,
#    linear_fc1, linear_fc2, o_proj, out_proj, proj, qkv, qkv_proj]."
# This is cosmetic: vLLM uses fused internal names; for the Qwen3.5 DENSE
# path it packs the unfused HF leaves into these fused slots and the adapter
# loads. Confirmed in vLLM issue #40005. Document this in the thesis writeup.
#
# Adapter served by name "overseer". Base model addressed as "Qwen/Qwen3.5-9B".

set -euo pipefail

ADAPTER_PATH=${ADAPTER_PATH:?must set ADAPTER_PATH to the trained adapter directory}
PORT=${PORT:-8000}
MAX_LORAS=${MAX_LORAS:-2}
MAX_LORA_RANK=${MAX_LORA_RANK:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_UTIL=${GPU_UTIL:-0.90}

vllm serve Qwen/Qwen3.5-9B \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --max-loras "${MAX_LORAS}" \
  --max-cpu-loras 4 \
  --lora-modules overseer="${ADAPTER_PATH}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --language-model-only \
  --port "${PORT}"

# To address the adapter at request time:
#   curl http://localhost:8000/v1/chat/completions \
#        -H 'Content-Type: application/json' -d '{
#          "model": "overseer",
#          "messages": [...],
#          "temperature": 0.0,
#          "max_tokens": 900
#        }'
#
# To address the bare base model from the same server:
#   "model": "Qwen/Qwen3.5-9B"
