#!/usr/bin/env bash
# Fresh-pod setup for vLLM serving of Qwen2.5-7B-Instruct + LoRA adapter.
#
# Target hardware: L40S 48GB on RunPod (or A100 40GB / 80GB; H100 also works).
# Container: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (matches training).
#
# This is a SEPARATE setup from the training pod. Spin up a smaller, cheaper
# GPU specifically for serving; H100/H200 is overkill for 7B + LoRA inference.
#
# What this does:
#   1. Verifies container's preinstalled torch+CUDA
#   2. Installs uv
#   3. Creates a venv and installs vllm + matching transformers/peft
#   4. Sanity-checks the install
#
# Adapter source: a local directory on the pod (mount your volume, or scp
# the adapter onto /workspace before running this script).

set -euo pipefail

# ============================================================================
# 0. Config.
# ============================================================================
VENV_DIR=${VENV_DIR:-.venv}

# These pins MUST match the training pod's torch ABI to use the same
# adapter cleanly. Container should be runpod/pytorch:*-cu1281-torch280.
TORCH_VERSION=2.8.0
TORCH_INDEX=https://download.pytorch.org/whl/cu128

# ============================================================================
# 1. Verify host CUDA + Python.
# ============================================================================
CONTAINER_PYTHON=${CONTAINER_PYTHON:-$(which python3)}
echo "==> Container python: ${CONTAINER_PYTHON}"
"${CONTAINER_PYTHON}" --version

"${CONTAINER_PYTHON}" - <<'PY' || true
try:
    import torch
    print(f"[diagnostic] container torch: {torch.__version__}, cuda: {torch.version.cuda}")
except ImportError:
    print("[diagnostic] no torch in container Python (will install fresh)")
PY

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
# 4. Pin torch first to match training pod's ABI.
# ============================================================================
echo "==> Installing torch ${TORCH_VERSION} from ${TORCH_INDEX} ..."
uv pip install \
    "torch==${TORCH_VERSION}" torchvision torchaudio \
    --index-url "${TORCH_INDEX}"

python - <<'PY'
import torch
print(f"venv torch: {torch.__version__}")
print(f"venv cuda:  {torch.version.cuda}")
assert torch.cuda.is_available(), "FATAL: torch can't see CUDA"
assert torch.version.cuda.startswith("12.8"), (
    f"FATAL: expected cu128 torch but got cu{torch.version.cuda}"
)
print(f"gpu: {torch.cuda.get_device_name(0)}")
print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

# ============================================================================
# 5. Install vLLM + the rest. We let uv resolve a vllm version compatible
# with torch 2.8.0+cu128. Recent vllm (0.10+) supports this combo.
# transformers/peft come in for adapter loading; huggingface-hub for the
# initial model download.
# ============================================================================
echo "==> Installing vllm + serving deps ..."
uv pip install \
    "vllm" \
    "transformers" \
    "peft>=0.14" \
    "huggingface-hub>=0.30" \
    "hf-transfer>=0.1"

# Sanity: torch version should be unchanged after vllm install.
python - <<'PY'
import torch
assert torch.version.cuda.startswith("12.8"), (
    f"FATAL: torch was upgraded during dep install — now cu{torch.version.cuda}. "
    f"vllm pulled in a different torch build. Pin vllm to a version compatible with torch 2.8.0+cu128."
)
print(f"[verify] torch still at {torch.__version__} after vllm install — good.")

import vllm
print(f"[verify] vllm: {vllm.__version__}")
PY

# ============================================================================
# 6. HF login (in case adapter download from Hub is needed later, or if base
# model gating ever changes).
# ============================================================================
echo ""
echo "==> Optional HF login (Qwen2.5 base doesn't require auth; useful for"
echo "    pulling adapters from private Hub repos). Press Ctrl+D to skip."
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
echo "Next: get your adapter onto this pod, then run serve_vllm.sh"
echo "  Either scp the adapter from your local backup:"
echo "    (from local) scp -P <PORT> -r ~/Downloads/headline_lora root@<POD_IP>:/workspace/"
echo "  Or extract from a tar.zst:"
echo "    cd /workspace && tar --use-compress-program=unzstd -xf headline_lora_*.tar.zst"
echo ""
echo "Then:"
echo "  ADAPTER_PATH=/workspace/headline_lora/checkpoint-best bash serve_vllm.sh"
echo "============================================================"
