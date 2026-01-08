#!/bin/bash
# Setup script for Magima Kids RunPod Pipeline
# Run this on any fresh pod to get up and running
#
# Usage:
#   cd /workspace
#   git clone https://github.com/magimakids/magima-kids-runpod.git
#   cd magima-kids-runpod
#   ./scripts/setup-runpod.sh

set -e

REPO_DIR="${REPO_DIR:-/workspace/magima-kids-runpod}"
MODEL_DIR="$REPO_DIR/models"

echo "=== Setting up Magima Kids LTX-2 Pipeline ==="
echo "Repository: $REPO_DIR"
echo ""

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv already installed"
fi

cd "$REPO_DIR"

# Create venv and install deps
echo "[2/5] Setting up Python environment..."
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install huggingface-hub transformers accelerate bitsandbytes

# Create models directory
mkdir -p "$MODEL_DIR"

# Clone LTX-2 repo for inference code
if [ ! -d "$MODEL_DIR/LTX-2" ]; then
    echo "[3/5] Cloning LTX-2 repository..."
    git clone https://github.com/Lightricks/LTX-2.git "$MODEL_DIR/LTX-2"
    cd "$MODEL_DIR/LTX-2"
    uv pip install -e .
    cd "$REPO_DIR"
else
    echo "[3/5] LTX-2 repo already present"
fi

# Download full model from HuggingFace
echo "[4/5] Downloading LTX-2 full model (43GB)..."
python -c "
from huggingface_hub import hf_hub_download
import os

model_dir = '$MODEL_DIR'

# Download full model (not distilled, not FP8)
print('Downloading ltx-2-19b-dev.safetensors (43GB)...')
hf_hub_download(
    repo_id='Lightricks/LTX-2',
    filename='ltx-2-19b-dev.safetensors',
    local_dir=model_dir,
    local_dir_use_symlinks=False
)
print('Full model downloaded!')
"

# Download Gemma for prompt enhancement
echo "[5/5] Downloading Gemma for prompt enhancement (7.4GB)..."
python -c "
from huggingface_hub import snapshot_download
import os

model_dir = '$MODEL_DIR'

print('Downloading unsloth/gemma-3-12b-it-bnb-4bit...')
snapshot_download(
    repo_id='unsloth/gemma-3-12b-it-bnb-4bit',
    local_dir=os.path.join(model_dir, 'gemma'),
    local_dir_use_symlinks=False
)
print('Gemma downloaded!')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Models downloaded:"
echo "  - ltx-2-19b-dev.safetensors (43GB) - Full quality model"
echo "  - gemma-3-12b-it-bnb-4bit (7.4GB) - Prompt enhancement"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo ""
echo "  # Run warmup (recommended first time)"
echo "  python scripts/warmup.py"
echo ""
echo "  # Generate video with default preset (1080p, 5sec)"
echo "  python -m src.pipelines.generate -p 'A colorful letter A bouncing happily'"
echo ""
echo "  # Fast test"
echo "  python -m src.pipelines.generate -p 'test' --preset fast-test"
echo ""
