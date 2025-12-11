#!/usr/bin/env bash
set -euo pipefail

# No datasets to download for cloudcast problem

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "$PROBLEM_DIR/../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/cloudcast"

mkdir -p "$DATASETS_DIR"

echo "[cloudcast download] No datasets required for this problem"


