#!/usr/bin/env bash
set -euo pipefail

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "$PROBLEM_DIR/../../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/symbolic_regression_mixed_polyexp_4d"
SRC_DIR="$PROBLEM_DIR/resources/data"

mkdir -p "$DATASETS_DIR"

echo "[symbolic_regression_mixed_polyexp_4d download] Preparing datasets at $DATASETS_DIR"

if [[ -d "$DATASETS_DIR/data" ]] && [[ -n "$(ls -A "$DATASETS_DIR/data" 2>/dev/null)" ]]; then
  echo "[symbolic_regression_mixed_polyexp_4d download] Existing dataset found; skipping copy"
  exit 0
fi

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Error: source datasets not found at $SRC_DIR" >&2
  exit 1
fi

rm -rf "$DATASETS_DIR/data"
cp -R "$SRC_DIR" "$DATASETS_DIR/"

echo "[symbolic_regression_mixed_polyexp_4d download] Dataset ready"
