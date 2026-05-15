#!/usr/bin/env bash
# Fresh-pod setup for vLLM serving of Qwen2.5-7B-Instruct + LoRA adapter.
#
# Target hardware: A100 SXM 80GB on RunPod (or L40S 48GB / RTX 6000).
# Container: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   — or anything with CUDA 12.8+ visible.
#
# This is a SEPARATE setup from the training pod. The training pod's torch
# (2.8.0+cu128) doesn't have to match the serving pod's torch — the LoRA
# adapter is just bf16 safetensors that vllm loads with its own torch.
# So we let vllm pull whatever torch it needs.

set -euo pipefail

# ============================================================================
# 0. Config.
# ============================================================================
VENV_DIR=${VENV_DIR:-.venv}

# ============================================================================
# 1. Verify host CUDA + Python.
# ============================================================================
CONTAINER_PYTHON=${CONTAINER_PYTHON:-$(which python3)}
echo "==> Container python: ${CONTAINER_PYTHON}"
"${CONTAINER_PYTHON}" --version

# ============================================================================
# 2. Install uv if missing.
# ============================================================================
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
echo "==> uv version: $(uv --version)"

# ============================================================================
# 3. Fresh venv.
# ============================================================================
if [ -d "${VENV_DIR}" ]; then
    echo "==> Removing previous ${VENV_DIR} for a clean install ..."
    rm -rf "${VENV_DIR}"
fi
echo "==> Creating venv at ${VENV_DIR} using ${CONTAINER_PYTHON} ..."
uv venv "${VENV_DIR}" --python "${CONTAINER_PYTHON}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ============================================================================
# 4. Install vllm and let it pull whatever torch it needs.
#    vllm pulls in transformers and peft as deps, so we don't list those.
# ============================================================================
echo "==> Installing vllm (will pull torch, transformers, peft as deps) ..."
uv pip install --torch-backend=cu128 "vllm>=0.10,<0.12" hf-transfer

# ============================================================================
# 5. Sanity check: torch sees CUDA, vllm imports.
# ============================================================================
python - <<'PY'
import torch, vllm
print(f"torch:   {torch.__version__}")
print(f"cuda:    {torch.version.cuda}")
print(f"vllm:    {vllm.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "FATAL: torch can't see CUDA"
print(f"gpu:     {torch.cuda.get_device_name(0)}")
print(f"vram:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"bf16:    {torch.cuda.is_bf16_supported()}")
assert torch.cuda.is_bf16_supported(), "FATAL: GPU doesn't support bf16"
PY

# ============================================================================
# 6. Optional HF login (for private adapter download; Qwen2.5 base is public).
# ============================================================================
echo ""
echo "==> Optional HF login. Press Ctrl+D to skip."
hf auth login || echo "  (skipped HF login)"

# ============================================================================
# 7. Persist env vars.
# ============================================================================
ACTIVATE="${VENV_DIR}/bin/activate"
if ! grep -q HF_HUB_ENABLE_HF_TRANSFER "${ACTIVATE}"; then
    echo "" >> "${ACTIVATE}"
    echo "# Added by setup_serve.sh" >> "${ACTIVATE}"
    echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> "${ACTIVATE}"
fi
if ! grep -q '.local/bin' "${ACTIVATE}"; then
    echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${ACTIVATE}"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1

# ============================================================================
# 8. Final report.
# ============================================================================
echo ""
echo "============================================================"
echo "Serve env setup complete. Venv at ${VENV_DIR}."
echo ""
echo "Versions:"
uv pip show torch vllm transformers peft 2>/dev/null | grep -E '^(Name|Version):' | paste - - | column -t
echo ""
echo "Get your adapter onto this pod, then run serve_vllm.sh:"
echo "  (from local) scp -P <PORT> ~/Downloads/headline_lora_*.tar.zst root@<POD_IP>:/workspace/"
echo "  (on pod)    cd /workspace && tar --use-compress-program=unzstd -xf headline_lora_*.tar.zst"
echo "  (on pod)    ADAPTER_PATH=/workspace/headline_lora/checkpoint-best bash serve_vllm.sh"
echo "============================================================"