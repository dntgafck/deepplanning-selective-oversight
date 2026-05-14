#!/usr/bin/env bash
# Fresh-pod setup for Qwen2.5-7B-Instruct LoRA training.
#
# Run this once on a freshly-rented H100 pod (RunPod CUDA 13 template).
# Idempotent: re-runs are safe.
#
# What this does:
#   1. System-level prereqs (git, curl, build tools — usually present on RunPod)
#   2. Python venv (NOT pixi — fresh pod, no cross-platform need)
#   3. Pinned pip installs in dependency order so resolver can't fight itself
#   4. Sanity check: torch + cuda + bf16 + flash-attn availability
#
# Usage:
#   cd /workspace                       # or wherever you keep things on the pod
#   git clone <your_project> && cd <your_project>
#   bash setup_pod.sh
#   source .venv/bin/activate
#
# Time: ~5-10 min depending on network speed.

set -euo pipefail

# ============================================================================
# 0. Config — edit if you want a different python or venv path.
# ============================================================================
PYTHON_BIN=${PYTHON_BIN:-python3.11}
VENV_DIR=${VENV_DIR:-.venv}

# Pinned versions. All carried from the May 2026 research memo; loosened where
# the memo was over-pinned for the actual ecosystem state.
TORCH_VERSION=2.11.0
TORCH_INDEX=https://download.pytorch.org/whl/cu130

# ============================================================================
# 1. Sanity check the system.
# ============================================================================
echo "==> nvidia-smi:"
nvidia-smi | head -10 || { echo "FATAL: nvidia-smi failed. Wrong template?"; exit 1; }

echo "==> python:"
which "${PYTHON_BIN}" || { echo "FATAL: ${PYTHON_BIN} not on PATH. Install or set PYTHON_BIN."; exit 1; }
"${PYTHON_BIN}" --version

# ============================================================================
# 2. Create venv and bootstrap pip.
# ============================================================================
if [ ! -d "${VENV_DIR}" ]; then
    echo "==> Creating venv at ${VENV_DIR} ..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# ============================================================================
# 3. Install torch from the cu130 wheel index FIRST.
# Reason: flash-attn (if we install it later) must build against the same
# torch ABI. Installing torch first locks the ABI; installing flash-attn
# afterwards picks up the right wheel automatically.
# ============================================================================
echo "==> Installing torch ${TORCH_VERSION} from ${TORCH_INDEX} ..."
pip install \
    "torch==${TORCH_VERSION}" "torchvision" "torchaudio" \
    --index-url "${TORCH_INDEX}"

# Verify torch + CUDA.
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not visible to torch — check driver/template."
print(f"CUDA version (torch): {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
PY

# ============================================================================
# 4. Install the rest of the training stack from PyPI.
# Pinning policy: floor versions known to work, no upper bounds unless an
# upstream conflict is known. pip will resolve to current latest within
# these floors.
# ============================================================================
echo "==> Installing TRL+PEFT training stack ..."
pip install \
    "transformers>=4.50" \
    "peft>=0.14" \
    "trl>=0.15" \
    "accelerate>=1.0" \
    "datasets>=3.0" \
    "bitsandbytes>=0.45" \
    "huggingface-hub>=0.30" \
    "hf-transfer>=0.1" \
    "wandb>=0.18" \
    "einops"

# ============================================================================
# 5. Try to install flash-attn. NON-FATAL if it fails — SDPA fallback works.
# Build path: requires ninja + a working C++ toolchain. RunPod CUDA 13
# templates usually have these. If not, this step prints a warning and
# the training script falls back to attn_implementation="sdpa".
# ============================================================================
echo "==> Attempting flash-attn install (non-fatal) ..."
pip install ninja
if pip install flash-attn --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log; then
    if python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" 2>/dev/null; then
        echo "==> flash-attn installed and importable."
        echo "    Use --attn_impl flash_attention_2 in train_lora.py."
    else
        echo "==> flash-attn package installed but import failed. Falling back to SDPA."
    fi
else
    echo "==> flash-attn install FAILED (see /tmp/flash_attn_install.log)."
    echo "    No problem — training will use SDPA (default in train_lora.py)."
fi

# ============================================================================
# 6. HuggingFace login (interactive).
# ============================================================================
echo ""
echo "==> One-time HF login (paste your token from https://huggingface.co/settings/tokens):"
echo "    Or skip and use anonymous access (Qwen2.5 doesn't require auth)."
echo "    Press Ctrl+D to skip."
huggingface-cli login || echo "  (skipped HF login)"

# ============================================================================
# 7. W&B login (interactive).
# ============================================================================
echo ""
echo "==> W&B login (paste your key from https://wandb.ai/authorize):"
echo "    Press Ctrl+D to skip."
wandb login || echo "  (skipped wandb login; training will fail at report_to=wandb unless WANDB_API_KEY is set)"

# ============================================================================
# 8. Speed up HF downloads.
# ============================================================================
echo ""
echo "==> Adding HF_HUB_ENABLE_HF_TRANSFER=1 to venv activation ..."
ACTIVATE="${VENV_DIR}/bin/activate"
if ! grep -q HF_HUB_ENABLE_HF_TRANSFER "${ACTIVATE}"; then
    echo "" >> "${ACTIVATE}"
    echo "# Enabled by setup_pod.sh — parallel HF downloads" >> "${ACTIVATE}"
    echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> "${ACTIVATE}"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1

# ============================================================================
# 9. Final report.
# ============================================================================
echo ""
echo "============================================================"
echo "Setup complete. Activated venv at ${VENV_DIR}."
echo ""
echo "Versions in this env:"
pip show torch transformers peft trl accelerate 2>/dev/null | grep -E '^(Name|Version):' | paste - - | column -t
echo ""
echo "Next:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python prepare_sft_data.py --in_dir <data> --out_dir out --max_seq_len 12288"
echo "  python check_a_parse_sanity.py --in_dir out"
echo "  python train_lora.py --mode pilot   # ~30 min on H100"
echo "  python check_b_eval.py --adapter out/pilot_lora --val out/val_swift.jsonl --n 100"
echo "  python train_lora.py --mode headline   # ~3-4h on H100"
echo "============================================================"
