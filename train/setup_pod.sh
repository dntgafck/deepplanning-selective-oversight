#!/usr/bin/env bash
# Fresh-pod setup for Qwen2.5-7B-Instruct LoRA training.
#
# Designed for the runpod/pytorch container image (preinstalled torch + CUDA).
# Tested with:  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Any image with torch >= 2.4 and CUDA visible to torch should also work.
#
# What this does:
#   1. Verify the preinstalled torch+CUDA stack
#   2. Install uv (fast pip replacement)
#   3. uv-install the training stack into the container's system Python
#      (no venv — the container IS the isolation boundary)
#   4. Try flash-attn (non-fatal)
#   5. Configure HF and W&B
#
# Speed: ~1-3 minutes total (vs ~5-10 min when torch is reinstalled from scratch).

set -euo pipefail

# ============================================================================
# 1. Verify the preinstalled torch+CUDA stack.
# ============================================================================
echo "==> Verifying preinstalled stack ..."

nvidia-smi | head -10 || { echo "FATAL: nvidia-smi failed"; exit 1; }

python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("FATAL: torch is not installed in this image.")
    print("       Either use a runpod/pytorch image, or revert to the venv+install version of setup_pod.sh.")
    sys.exit(1)

print(f"python: {sys.version.split()[0]}")
print(f"torch:  {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "FATAL: torch can't see CUDA"
print(f"cuda (torch): {torch.version.cuda}")
print(f"gpu: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

# Minimum torch version check for TRL+PEFT compatibility.
major, minor = map(int, torch.__version__.split(".")[:2])
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
# 3. Install the training stack on top of the container's Python.
# --system tells uv to use the active Python (system Python in a runpod
# container) rather than refusing without a venv.
# ============================================================================
echo "==> Installing TRL+PEFT training stack via uv ..."
uv pip install --system \
    "transformers" \
    "peft" \
    "trl" \
    "accelerate" \
    "datasets" \
    "bitsandbytes" \
    "huggingface-hub" \
    "hf-transfer" \
    "wandb" \
    "einops"

# ============================================================================
# 4. flash-attn — try prebuilt wheel, fall back to source build, fall back to SDPA.
# ============================================================================
echo "==> Attempting flash-attn install (non-fatal) ..."
uv pip install --system ninja
if uv pip install --system flash-attn --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log; then
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
# 5. HuggingFace login (interactive).
# ============================================================================
echo ""
echo "==> One-time HF login (paste your token from https://huggingface.co/settings/tokens):"
echo "    Or skip and use anonymous access (Qwen2.5 doesn't require auth)."
echo "    Press Ctrl+D to skip."
hf auth login || echo "  (skipped HF login)"

# ============================================================================
# 6. W&B login (interactive).
# ============================================================================
echo ""
echo "==> W&B login (paste your key from https://wandb.ai/authorize):"
echo "    Press Ctrl+D to skip."
wandb login || echo "  (skipped wandb login; training will fail at report_to=wandb unless WANDB_API_KEY is set)"

# ============================================================================
# 7. Persist env vars for parallel HF downloads and the uv PATH addition.
# Append to ~/.bashrc so a new shell on the same pod inherits them.
# ============================================================================
echo ""
echo "==> Persisting env vars to ~/.bashrc ..."
BASHRC="${HOME}/.bashrc"
touch "${BASHRC}"

if ! grep -q HF_HUB_ENABLE_HF_TRANSFER "${BASHRC}"; then
    echo "" >> "${BASHRC}"
    echo "# Added by setup_pod.sh — parallel HF downloads" >> "${BASHRC}"
    echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> "${BASHRC}"
fi

if ! grep -q '.local/bin' "${BASHRC}"; then
    echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${BASHRC}"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

# ============================================================================
# 8. Final report.
# ============================================================================
echo ""
echo "============================================================"
echo "Setup complete. No venv — running on the container's system Python."
echo ""
echo "Versions:"
uv pip show torch transformers peft trl accelerate 2>/dev/null | grep -E '^(Name|Version):' | paste - - | column -t
echo ""
echo "Next:"
echo "  python prepare_sft_data.py --in_dir <data> --out_dir out --max_seq_len 12288"
echo "  python check_a_parse_sanity.py --in_dir out"
echo "  python train_lora.py --mode pilot     # ~30 min on H100"
echo "  python check_b_eval.py --adapter out/pilot_lora/final --val out/val_swift.jsonl --n 100"
echo "  python train_lora.py --mode headline  # ~3-4h on H100"
echo "============================================================"