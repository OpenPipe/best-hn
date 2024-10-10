#!/bin/bash

set -e
set -o pipefail
set -x

# Create and activate virtual environment
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# Install tmux
apt-get update
apt-get install -y tmux nvtop

curl -LsSf https://astral.sh/uv/0.4.6/install.sh | sh

uv sync

echo "Python from virtual environment: $(uv run which python)"

echo "Installation complete. To use the virtual environment, run: source ~/.venv/bin/activate"
