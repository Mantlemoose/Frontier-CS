#!/usr/bin/env bash
set -e

echo "[evaluate] Running Cloudcast evaluation..." >&2
echo "[evaluate] Current dir: $(pwd)" >&2

if ./run_evaluator.sh 2>&1 | tee /tmp/cloudcast_eval.log; then
  echo "[evaluate] run_evaluator succeeded" >&2
else
  echo "[evaluate] ERROR: run_evaluator.sh failed" >&2
  cat /tmp/cloudcast_eval.log >&2
  echo "0.0"
  exit 0
fi

if [[ -f "results.json" ]]; then
  python3 - <<'PY'
import json
with open("results.json", "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data.get("score", 0.0))
PY
else
  echo "[evaluate] WARNING: results.json not found, returning 0" >&2
  echo "0.0"
fi
