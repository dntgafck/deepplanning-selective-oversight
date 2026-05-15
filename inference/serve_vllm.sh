#!/usr/bin/env bash
# vLLM serving for Qwen2.5-7B-Instruct + LoRA overseer adapter.
#
# Prerequisites:
#   - setup_serve.sh has been run; venv at .venv is set up with torch 2.8 + vllm
#   - Adapter directory exists on the pod (local path, not HF Hub for this script)
#
# Target hardware: L40S 48GB. Memory footprint:
#   - Base Qwen2.5-7B bf16: ~14 GB
#   - LoRA adapter r=16:    ~150 MB
#   - KV cache:             rest of GPU (up to ~30 GB on L40S)
#
# Same vLLM instance serves BOTH:
#   - "Qwen/Qwen2.5-7B-Instruct"  → bare base model (executor in C2-lora)
#   - "overseer"                    → base + adapter (overseer in C2-lora and C2-deepseek-lora)
# Client picks via the "model" field in the request body.

set -euo pipefail

# ============================================================================
# Config — override via env vars.
# ============================================================================
ADAPTER_PATH=${ADAPTER_PATH:?must set ADAPTER_PATH (e.g. /workspace/headline_lora/checkpoint-best)}
VLLM_API_KEY=${VLLM_API_KEY:?must set VLLM_API_KEY (use a long random string)}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

# Sized for L40S 48GB. For A100 80GB you can push max_model_len higher or
# raise gpu-memory-utilization.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}     # matches training seq length
MAX_LORA_RANK=${MAX_LORA_RANK:-16}        # must match adapter's r
MAX_LORAS=${MAX_LORAS:-2}                 # serve one + headroom for hot-swap
MAX_CPU_LORAS=${MAX_CPU_LORAS:-4}
GPU_UTIL=${GPU_UTIL:-0.90}
DTYPE=${DTYPE:-bfloat16}


VENV_DIR=${VENV_DIR:-.venv}

# ============================================================================
# Activate venv if not already.
# ============================================================================
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d "${VENV_DIR}" ]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    else
        echo "FATAL: no venv active and ${VENV_DIR} doesn't exist. Run setup_serve.sh first."
        exit 1
    fi
fi

# ============================================================================
# Sanity checks before burning startup time on model download.
# ============================================================================
echo "==> Pre-flight checks ..."

# Adapter directory exists and looks like a PEFT adapter.
if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "FATAL: ADAPTER_PATH does not exist: ${ADAPTER_PATH}"
    exit 1
fi
if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
    echo "FATAL: ${ADAPTER_PATH}/adapter_config.json not found."
    echo "       The path must point at a PEFT adapter directory (one that contains"
    echo "       adapter_config.json and adapter_model.safetensors)."
    exit 1
fi
echo "  ✓ adapter at ${ADAPTER_PATH}"
ADAPTER_SIZE=$(du -sh "${ADAPTER_PATH}" | cut -f1)
echo "  ✓ adapter size: ${ADAPTER_SIZE}"

# Verify the adapter's declared rank matches MAX_LORA_RANK.
DECLARED_RANK=$(python -c "import json; print(json.load(open('${ADAPTER_PATH}/adapter_config.json')).get('r', 'unknown'))")
echo "  ✓ adapter r: ${DECLARED_RANK}"
if [ "${DECLARED_RANK}" != "${MAX_LORA_RANK}" ]; then
    echo "  WARNING: adapter r=${DECLARED_RANK} but MAX_LORA_RANK=${MAX_LORA_RANK}."
    echo "           Bump MAX_LORA_RANK to >= ${DECLARED_RANK} or vllm will reject the adapter."
fi

# GPU memory check.
python - <<'PY'
import torch
free, total = torch.cuda.mem_get_info(0)
free_gb, total_gb = free / 1e9, total / 1e9
print(f"  ✓ gpu: {torch.cuda.get_device_name(0)}  ({total_gb:.0f} GB total, {free_gb:.0f} GB free)")
if total_gb < 24:
    print(f"  WARNING: {total_gb:.0f} GB may be too small for Qwen2.5-7B bf16 + KV cache. Consider L40S 48GB or larger.")
PY

# vLLM importable.
python -c "import vllm; print(f'  ✓ vllm {vllm.__version__}')"

echo "==> Pre-flight OK. Starting vllm serve ..."
echo ""

# ============================================================================
# Launch vLLM.
# ============================================================================
exec vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --enable-chunked-prefill --enable-prefix-caching \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --max-loras "${MAX_LORAS}" \
  --max-cpu-loras "${MAX_CPU_LORAS}" \
  --lora-modules "overseer=${ADAPTER_PATH}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --dtype "${DTYPE}" \
  --api_key "${VLLM_API_KEY}"
  --host "${HOST}" \
  --port "${PORT}"

# ============================================================================
# After vLLM starts (you'll see "Uvicorn running on ..."), test from the same
# pod or from your laptop (if HOST=0.0.0.0 and the port is exposed):
#
# 1. Bare base (for executor in C2-lora):
#    curl http://localhost:${PORT}/v1/chat/completions \
#      -H 'Content-Type: application/json' \
#      -d '{
#        "model": "Qwen/Qwen2.5-7B-Instruct",
#        "messages": [{"role": "user", "content": "ping"}],
#        "temperature": 0.0,
#        "max_tokens": 50
#      }'
#
# 2. Adapter (overseer):
#    curl http://localhost:${PORT}/v1/chat/completions \
#      -H 'Content-Type: application/json' \
#      -d '{
#        "model": "overseer",
#        "messages": [
#          {"role": "system", "content": "<your overseer system prompt>"},
#          {"role": "user", "content": "<a real overseer input from val_swift.jsonl>"}
#        ],
#        "temperature": 0.0,
#        "max_tokens": 900
#      }'
# ============================================================================
