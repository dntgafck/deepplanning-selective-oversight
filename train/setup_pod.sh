#!/usr/bin/env bash
# Fresh-pod setup for Qwen2.5-7B-Instruct LoRA training.
#
# Designed for the runpod/pytorch container image (preinstalled torch + CUDA).
# Tested with:  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Any image with torch >= 2.4 and CUDA visible to torch should also work.
#
# How it handles PEP 668 (Ubuntu 24.04+):
#   Ubuntu's system Python is "externally managed" so uv/pip refuse to install
#   into it. Solution: create a venv with --system-site-packages so we INHERIT
#   the preinstalled torch from the container without reinstalling it, then
#   install our training stack into the venv (where pip is allowed to write).
#
# What this does:
#   1. Verify the preinstalled torch+CUDA stack (using the container's Python)
#   2. Install uv if missing
#   3. Create a venv that inherits torch from the container
#   4. uv-install the training stack into the venv
#   5. Try flash-attn (non-fatal)
#   6. Configure HF and W&B
#
# Speed: ~1-3 minutes total.

set -euo pipefail

# ============================================================================
# 0. Config.
# ============================================================================
VENV_DIR=${VENV_DIR:-.venv}

# ============================================================================
# 1. Verify the preinstalled torch+CUDA stack (against container Python).
# ============================================================================
# Find the container's Python. On runpod/pytorch images this is typically
# /usr/bin/python3 with torch installed system-wide.
CONTAINER_PYTHON=${CONTAINER_PYTHON:-$(which python3)}
echo "==> Container python: ${CONTAINER_PYTHON}"

"${CONTAINER_PYTHON}" - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("FATAL: torch is not installed in this image's system Python.")
    print("       Use a runpod/pytorch image with torch preinstalled.")
    sys.exit(1)

print(f"python: {sys.version.split()[0]}")
print(f"torch:  {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "FATAL: torch can't see CUDA"
print(f"cuda (torch): {torch.version.cuda}")
print(f"gpu: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) < (2, 4):
    print(f"FATAL: torch {torch.__version__} is too old. Need >= 2.4 for TRL+PEFT.")
    sys.exit(1)
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
# 3. Create a venv that inherits the container's site-packages.
# This is the magic: --system-site-packages means the venv can see and use
# the container's torch without reinstalling it.
# ============================================================================
if [ ! -d "${VENV_DIR}" ]; then
    echo "==> Creating venv at ${VENV_DIR} with inherited site-packages ..."
    uv venv "${VENV_DIR}" --python "${CONTAINER_PYTHON}" --system-site-packages
else
    # If the user already created a venv (without --system-site-packages),
    # remove it and recreate. Otherwise torch is invisible inside.
    if ! "${VENV_DIR}/bin/python" -c "import torch" 2>/dev/null; then
        echo "==> Existing venv lacks torch — recreating with --system-site-packages ..."
        rm -rf "${VENV_DIR}"
        uv venv "${VENV_DIR}" --python "${CONTAINER_PYTHON}" --system-site-packages
    else
        echo "==> Reusing existing venv at ${VENV_DIR} (has torch)."
    fi
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Verify torch is visible from inside the venv.
python -c "import torch; print(f'venv sees torch {torch.__version__}, cuda={torch.cuda.is_available()}')"

# ============================================================================
# 4. Install the training stack INTO the venv (not --system; the venv is
# where pip is allowed to write under PEP 668).
# ============================================================================
echo "==> Installing TRL+PEFT training stack via uv ..."
uv pip install \
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
# 5. flash-attn — try prebuilt wheel, fall back to source build, fall back to SDPA.
# ============================================================================
echo "==> Attempting flash-attn install (non-fatal) ..."
uv pip install ninja
if uv pip install flash-attn --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log; then
    if python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" 2>/dev/null; then
        echo "==> flash-attn installed and importable."
        echo "    train_lora.py will auto-detect and use flash_attention_2."
    else
        echo "==> flash-attn package installed but import failed. SDPA fallback active."
    fi
else
    echo "==> flash-attn install FAILED (see /tmp/flash_attn_install.log)."
    echo "    Training will use SDPA (default in train_lora.py)."
fi

# ============================================================================
# 6. HuggingFace login (interactive).
# ============================================================================
echo ""
echo "==> One-time HF login (paste your token from https://huggingface.co/settings/tokens):"
echo "    Or skip and use anonymous access (Qwen2.5 doesn't require auth)."
echo "    Press Ctrl+D to skip."
hf auth login || echo "  (skipped HF login)"

# ============================================================================
# 7. W&B login (interactive).
# ============================================================================
echo ""
echo "==> W&B login (paste your key from https://wandb.ai/authorize):"
echo "    Press Ctrl+D to skip."
wandb login || echo "  (skipped wandb login; training will fail at report_to=wandb unless WANDB_API_KEY is set)"

# ============================================================================
# 8. Persist env vars and uv PATH into the venv activation script.
# ============================================================================
ACTIVATE="${VENV_DIR}/bin/activate"
if ! grep -q HF_HUB_ENABLE_HF_TRANSFER "${ACTIVATE}"; then
    echo "" >> "${ACTIVATE}"
    echo "# Added by setup_pod.sh — parallel HF downloads" >> "${ACTIVATE}"
    echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> "${ACTIVATE}"
fi
if ! grep -q '.local/bin' "${ACTIVATE}"; then
    echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${ACTIVATE}"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1

# ============================================================================
# 9. Final report.
# ============================================================================
echo ""
echo "============================================================"
echo "Setup complete. Venv at ${VENV_DIR} (inherits container's torch)."
echo ""
echo "Versions:"
uv pip show torch transformers peft trl accelerate 2>/dev/null | grep -E '^(Name|Version):' | paste - - | column -t
echo ""
echo "Next (in this shell — venv is already active):"
echo "  python prepare_sft_data.py --in_dir <data> --out_dir out --max_seq_len 12288"
echo "  python check_a_parse_sanity.py --in_dir out"
echo "  python train_lora.py --mode pilot     # ~30 min on H100"
echo "  python check_b_eval.py --adapter out/pilot_lora/final --val out/val_swift.jsonl --n 100"
echo "  python train_lora.py --mode headline  # ~3-4h on H100"
echo ""
echo "For new shells on this pod:  source ${VENV_DIR}/bin/activate"
echo "============================================================"