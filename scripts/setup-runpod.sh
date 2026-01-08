#!/bin/bash
# Setup script for Magima Kids RunPod Pipeline
# Run this on any fresh pod to get up and running

set -e

REPO_DIR="/workspace/magima-kids-runpod"

echo "=== Setting up Magima Kids Pipeline ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_DIR"

# Create venv and install deps
echo "Setting up Python environment..."
uv venv
source .venv/bin/activate
uv sync

# Clone LTX-Video if not present
if [ ! -d "models/LTX-Video" ]; then
    echo "Cloning LTX-Video..."
    mkdir -p models
    git clone https://github.com/Lightricks/LTX-Video.git models/LTX-Video
fi

# Download model weights from HuggingFace
echo "Downloading model weights..."
python -c "
from huggingface_hub import snapshot_download

print('Downloading LTX-Video model weights...')
snapshot_download(
    repo_id='Lightricks/LTX-Video',
    local_dir='./models/ltx-video-weights',
    local_dir_use_symlinks=False
)
print('Models downloaded!')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Run the API server:"
echo "  python -m src.api.server"
echo ""
echo "Or generate directly:"
echo "  python -m src.pipelines.generate --prompt 'your prompt'"
