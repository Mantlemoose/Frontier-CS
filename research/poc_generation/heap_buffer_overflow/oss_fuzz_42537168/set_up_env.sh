#!/usr/bin/env bash
set -euo pipefail

# Install requests for HuggingFace API calls
pip install --quiet requests

# Install Docker CLI for running ARVO/OSS-Fuzz containers
# The Docker socket is mounted via Docker-in-Docker configuration
echo "[set_up_env] Installing Docker CLI..."
apt-get update -qq && apt-get install -y -qq docker.io >/dev/null 2>&1 || {
    echo "[set_up_env] Warning: Could not install docker.io via apt, trying alternative..."
    # Alternative: download static Docker binary
    curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-24.0.7.tgz | tar xz -C /tmp
    mv /tmp/docker/docker /usr/local/bin/docker
    chmod +x /usr/local/bin/docker
    rm -rf /tmp/docker
}

# Verify Docker is available
if command -v docker &>/dev/null; then
    echo "[set_up_env] Docker CLI installed: $(docker --version)"
else
    echo "[set_up_env] ERROR: Docker CLI installation failed!"
    exit 1
fi

echo "[set_up_env] Completed."
