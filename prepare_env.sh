#!/bin/bash

set -e
set -o pipefail

# Set up environment variables
add_to_bashrc() {
    local line="$1"
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}

add_to_bashrc "export UV_CACHE_DIR=/workspace/.cache/uv"
add_to_bashrc "export HF_HOME=/workspace/.cache/huggingface"
add_to_bashrc "export HF_HUB_ENABLE_HF_TRANSFER=1"
add_to_bashrc "source /workspace/.env"

source ~/.bashrc

# Install system dependencies
apt-get update
apt-get install -y tmux nvtop htop entr build-essential

# Install python dependencies
curl -LsSf https://astral.sh/uv/0.4.6/install.sh | sh

cd /workspace/best-hn

/root/.cargo/bin/uv sync
/root/.cargo/bin/uv pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
