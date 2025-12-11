#!/usr/bin/env bash
set -euo pipefail

# No datasets to download for imagenet_pareto problem

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# For nested variant like imagenet_pareto/1m:
# SCRIPT_DIR = .../problems/imagenet_pareto/1m
# BASE_DIR = .../problems/imagenet_pareto
# ROOT_DIR = ...
VARIANT_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ROOT_DIR=$(cd "$BASE_DIR/../.." && pwd)

# Create dataset directory for this specific variant
DATASETS_DIR="$ROOT_DIR/datasets/imagenet_pareto/$VARIANT_NAME"

mkdir -p "$DATASETS_DIR"

echo "[imagenet_pareto download] No datasets required for variant $VARIANT_NAME"

