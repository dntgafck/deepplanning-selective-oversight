#!/usr/bin/env bash
# Fresh-pod setup for Qwen2.5-7B-Instruct LoRA training.
#
# Designed for the runpod/pytorch container image. Tested with:
#   runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#
# Design choices, written down so future-me doesn't undo them:
#
# - Ubuntu 24.04's system Python is PEP-668 "externally managed" — uv pip
#   --system fails without --break-system-packages. We use a venv instead.
#
# - The venv does NOT inherit site-packages (`--system-site-packages` was
#   tried; uv's resolver still installed a fresh torch from PyPI's default
#   index, which is currently a cu130 build. That cu130 torch then
#   shadowed the container's cu128 torch, causing flash-attn build failures
#   when it found cu128 nvcc but cu130 torch headers.)
#
# - Instead, we install torch==2.8.0 explicitly from the cu128 wheel index
#   as the FIRST step. This reinstalls torch into the venv (~1-2 min,
#   ~3 GB) but the result is fully deterministic. Subsequent installs see
#   torch already at 2.8.0+cu128 in the venv and won't try to upgrade it.
#
# Speed: ~3-5 min on a typical pod.

set -euo pipefail

# ============================================================================
# 0. Config.
# ============================================================================
VENV_DIR=${VENV_DIR:-.venv}

# These pins MUST match the container's torch build. For the
# runpod/pytorch:cu1281-torch280-ubuntu2404 image: torch 2.8.0 + cu128.
# If you switch to a different container template, update these and re-run.
TORCH_VERSION=2.8.0
TORCH_INDEX=https://download.pytorch.org/whl/cu128

# ============================================================================
# 1. Verify the host has CUDA visible and find the container's Python.
# ============================================================================
CONTAINER_PYTHON=${CONTAINER_PYTHON:-$(which python3)}
echo "==> Container python: ${CONTAINER_PYTHON}"
"${CONTAINER_PYTHON}" --version

# Show what torch the container has, for diagnostic purposes only.
# (We'll install our own copy in the venv regardless.)
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
# 3. Create a fresh venv (no inheritance — we want a deterministic env).
# If a previous .venv exists, wipe it.
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
# 4. Install torch FIRST, explicitly pinned to cu128 wheels.
# This is the load-bearing step. Subsequent uv pip installs will see torch
# already present at the right version and will not silently upgrade it.
# ============================================================================
echo "==> Installing torch ${TORCH_VERSION} from ${TORCH_INDEX} ..."
uv pip install \
    "torch==${TORCH_VERSION}" torchvision torchaudio \
    --index-url "${TORCH_INDEX}"

# Verify what we got.
python - <<'PY'
import torch
print(f"venv torch: {torch.__version__}")
print(f"venv cuda:  {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "FATAL: torch can't see CUDA"
assert torch.version.cuda.startswith("12.8"), (
    f"FATAL: expected cu128 torch but got cu{torch.version.cuda}. "
    f"Resolution went wrong; check the --index-url."
)
print(f"gpu: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
PY

# ============================================================================
# 5. Install the training stack on top.
# ============================================================================
echo "==> Installing TRL+PEFT training stack ..."
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

# Sanity check: torch version should still be cu128 after other installs.
python - <<'PY'
import torch
assert torch.version.cuda.startswith("12.8"), (
    f"FATAL: torch was upgraded during dep install — now cu{torch.version.cuda}. "
    f"One of the deps requires a different torch. Pin it below."
)
print(f"[verify] torch still at {torch.__version__} after stack install — good.")
PY

# ============================================================================
# 6. flash-attn — try prebuilt wheel, fall back to source build, fall back to SDPA.
# ============================================================================
echo "==> Attempting flash-attn install (non-fatal) ..."
uv pip install ninja wheel
if uv pip install flash-attn --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log; then
    if python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" 2>/dev/null; then
        echo "==> flash-attn installed and importable."
        echo "    train_lora.py will auto-detect and use flash_attention_2."
    else
        echo "==> flash-attn package installed but import failed. SDPA fallback active."
    fi
else
    echo "==> flash-attn install FAILED (see /tmp/flash_attn_install.log)."
    echo "    Training will use SDPA (default in train_lora.py). ~25% slower at seq 12288,"
    echo "    but acceptable. Continue."
fi

# ============================================================================
# 7. HuggingFace login (interactive).
# ============================================================================
echo ""
echo "==> One-time HF login (paste your token from https://huggingface.co/settings/tokens):"
echo "    Or skip and use anonymous access (Qwen2.5 doesn't require auth)."
echo "    Press Ctrl+D to skip."
hf auth login || echo "  (skipped HF login)"

# ============================================================================
# 8. W&B login (interactive).
# ============================================================================
echo ""
echo "==> W&B login (paste your key from https://wandb.ai/authorize):"
echo "    Press Ctrl+D to skip."
wandb login || echo "  (skipped wandb login; training will fail at report_to=wandb unless WANDB_API_KEY is set)"

# ============================================================================
# 9. Persist env vars into venv activation.
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
# 10. Final report.
# ============================================================================
echo ""
echo "============================================================"
echo "Setup complete. Venv at ${VENV_DIR}."
echo ""
echo "Versions:"
uv pip show torch transformers peft trl accelerate 2>/dev/null | grep -E '^(Name|Version):' | paste - - | column -t
echo ""
echo "Venv is active in this shell. For new shells:  source ${VENV_DIR}/bin/activate"
echo ""
echo "Next:"
echo "  python prepare_sft_data.py --in_dir <data> --out_dir out --max_seq_len 12288"
echo "  python check_a_parse_sanity.py --in_dir out"
echo "  python train_lora.py --mode pilot     # ~30 min on H100"
echo "  python check_b_eval.py --adapter out/pilot_lora/final --val out/val_swift.jsonl --n 100"
echo "  python train_lora.py --mode headline  # ~3-4h on H100"
echo "============================================================"
