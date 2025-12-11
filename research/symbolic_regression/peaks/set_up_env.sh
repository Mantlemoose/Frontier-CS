#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXEC_ROOT="$SCRIPT_DIR/../../../execution_env"
VENV_DIR="$EXEC_ROOT/.venv_symbolic_regression"

mkdir -p "$EXEC_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[symbolic_regression setup] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir pysr==0.19.0 numpy==1.26.4 pandas==2.2.2 sympy==1.13.3

python - <<'PY'
import importlib
packages = ["pysr", "numpy", "pandas", "sympy"]
missing = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
    except Exception as exc:  # pragma: no cover - diagnostic only
        missing.append((pkg, str(exc)))
if missing:
    lines = [f"{pkg}: {err}" for pkg, err in missing]
    raise SystemExit("Failed to verify packages:\n" + "\n".join(lines))
print("[symbolic_regression setup] Dependencies verified")
PY

deactivate
