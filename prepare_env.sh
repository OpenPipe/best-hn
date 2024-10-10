#!/bin/bash

set -e
set -o pipefail

# Set up environment variables
add_to_bashrc() {
    local line="$1"
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}

add_to_bashrc "source /workspace/.env"
add_to_bashrc "export UV_CACHE_DIR=/workspace/.cache/uv"

source ~/.bashrc

# Install system dependencies
apt-get update
apt-get install -y tmux nvtop

# Install python dependencies
curl -LsSf https://astral.sh/uv/0.4.6/install.sh | sh

uv sync

echo "Python from virtual environment: $(uv run which python)"
