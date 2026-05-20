#!/bin/bash
# ============================================================
# setup_paper1.sh
# One-time setup for ALIVE-ED Paper 1 development environment
#
# Run:  bash data_pipeline/setup_paper1.sh
# From: ~/islamm11/alive/backend
# ============================================================

set -e   # exit on any error
BACKEND_DIR="$HOME/islamm11/alive/backend"

echo ""
echo "============================================================"
echo "  ALIVE-ED Paper 1 — Environment Setup"
echo "============================================================"
echo ""

# ── 1. Create directory structure ────────────────────────────
echo "[1/6] Creating directories..."
mkdir -p data/qa_raw
mkdir -p data/finetune
mkdir -p data/finetune/prepared
mkdir -p configs
mkdir -p evaluation
mkdir -p logs
mkdir -p figures
echo "  ✓ Directories created"

# ── 2. Install eval dependencies in sadtalker env ────────────
echo ""
echo "[2/6] Installing evaluation dependencies in sadtalker env..."
# These go into your existing app env so eval_llm.py can use
# the same Python that runs app.py
conda run -n sadtalker pip install \
    bert-score \
    rouge-score \
    sentence-transformers \
    faiss-gpu \
    --quiet
echo "  ✓ bert-score, rouge-score, sentence-transformers, faiss-gpu installed"

# ── 3. Create finetune conda env ─────────────────────────────
echo ""
echo "[3/6] Creating 'finetune' conda environment (for axolotl)..."
if conda env list | grep -q "^finetune "; then
    echo "  ℹ  'finetune' env already exists — skipping creation"
else
    conda create -n finetune python=3.10 -y
    echo "  ✓ 'finetune' env created"
fi

echo "  Installing PyTorch + axolotl in finetune env..."
conda run -n finetune pip install \
    torch==2.1.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
conda run -n finetune pip install packaging ninja --quiet
conda run -n finetune pip install flash-attn --no-build-isolation --quiet
conda run -n finetune pip install axolotl --quiet
echo "  ✓ axolotl installed"

# ── 4. Verify axolotl ────────────────────────────────────────
echo ""
echo "[4/6] Verifying axolotl install..."
conda run -n finetune python -c "import axolotl; print(f'  ✓ axolotl {axolotl.__version__}')"

# ── 5. Verify vLLM is reachable ──────────────────────────────
echo ""
echo "[5/6] Checking vLLM endpoint..."
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "  ✓ vLLM is running at http://127.0.0.1:8000"
else
    echo "  ⚠  vLLM is NOT running — start Terminal A first:"
    echo "     conda activate vllm_deploy"
    echo "     CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \\"
    echo "       --host 127.0.0.1 --port 8000 --max-model-len 4096"
fi

# ── 6. Run data audit ────────────────────────────────────────
echo ""
echo "[6/6] Running data audit..."
conda run -n sadtalker python data_pipeline/verify_data.py

echo ""
echo "============================================================"
echo "  Setup complete! Next steps:"
echo "  1. Review audit output above"
echo "  2. Make sure vLLM is running (Terminal A)"
echo "  3. Run:  python data_pipeline/generate_qa.py"
echo "============================================================"
echo ""
