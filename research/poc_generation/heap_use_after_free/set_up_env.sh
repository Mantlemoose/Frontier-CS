#!/usr/bin/env bash
set -euo pipefail

# Usage: ./set_up_env.sh [config_path]
# This script sets up the environment for poc_generation evaluation.
# Docker CLI is provided by SkyPilot's native Docker support.
# This script pulls ARVO images and installs Python dependencies.

CONFIG_PATH=${1:-config.yaml}
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found at $CONFIG_PATH" >&2
  exit 1
fi

PROBLEM_DIR=$(pwd)
EXEC_ROOT="../../../execution_env"
mkdir -p "$EXEC_ROOT"

# Install Docker CLI (needed inside the container for DinD)
if ! command -v docker &>/dev/null; then
    echo "[set_up_env] Installing Docker CLI..."
    apt-get update -qq && apt-get install -y -qq curl >/dev/null 2>&1
    DOCKER_VERSION="27.3.1"
    curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" | tar xz -C /tmp
    mv /tmp/docker/docker /usr/local/bin/docker
    chmod +x /usr/local/bin/docker
    rm -rf /tmp/docker
fi

# ARVO vulnerability ID for this problem
ARVO_ID="47101"

# Pull ARVO Docker images (vulnerable and fixed versions)
# Docker socket is mounted from host via main_loop.sh
echo "[set_up_env] Pulling ARVO Docker images for vulnerability $ARVO_ID..."
docker pull "n132/arvo:${ARVO_ID}-vul" &
docker pull "n132/arvo:${ARVO_ID}-fix" &
wait
echo "[set_up_env] ARVO images pulled successfully"

# Set up Python venv for evaluator
VENV_DIR="$EXEC_ROOT/.venv"
pip install --user uv || exit 1
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

echo "[set_up_env] Creating/updating venv at $VENV_DIR"
uv venv "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

# Install requests for HuggingFace API calls
echo "[set_up_env] Installing dependencies..."
uv pip install requests

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "[set_up_env] Activating venv..." >&2
  source "$VENV_DIR/bin/activate"
else
  echo "[set_up_env] WARNING: venv missing at $VENV_DIR/bin/activate" >&2
fi

echo "[set_up_env] Completed."
