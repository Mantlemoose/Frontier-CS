#!/usr/bin/env bash
set -e

echo "[evaluate] Running ImageNet Pareto evaluation..." >&2
echo "[evaluate] Current dir: $(pwd)" >&2

if ./run_evaluator.sh 2>&1 | tee /tmp/imagenet_eval.log; then
  echo "[evaluate] run_evaluator succeeded" >&2
else
  echo "[evaluate] ERROR: run_evaluator.sh failed" >&2
  echo "[evaluate] Output:" >&2
  cat /tmp/imagenet_eval.log >&2
  exit 1
fi

if [[ -f "results.json" ]]; then
  echo "[evaluate] Found results.json" >&2
  python3 - <<'PY'
import json, sys
data = json.load(open("results.json"))
print(data.get("score", 40.0))
PY
else
  echo "[evaluate] WARNING: results.json missing, returning default score" >&2
  echo "40.0"
fi
