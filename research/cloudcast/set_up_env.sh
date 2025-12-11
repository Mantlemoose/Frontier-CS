#!/usr/bin/env bash
set -euo pipefail

# Usage: ./set_up_env.sh [config_path]

CONFIG_PATH=${1:-config.yaml}
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found at $CONFIG_PATH" >&2
  exit 1
fi

PROBLEM_DIR=$(pwd)
EXEC_ROOT="../../execution_env"
mkdir -p "$EXEC_ROOT"

UV_PROJECT_REL=$(
  python3 - <<'PY' "$CONFIG_PATH"
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("dependencies", {}).get("uv_project", ""))
PY
)

VENV_DIR="$EXEC_ROOT/.venv"
echo "[cloudcast setup] Creating venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

echo "[cloudcast setup] Upgrading pip"
python -m pip install --upgrade pip

readarray -t EXTRA_DEPS < <(python3 - <<'PY' "$UV_PROJECT_REL" "$PROBLEM_DIR"
import sys
import tomllib
from pathlib import Path

uv_project_rel = sys.argv[1]
problem_dir = Path(sys.argv[2])
deps: list[str] = []

if uv_project_rel:
    pyproject_path = problem_dir / uv_project_rel / "pyproject.toml"
    if not pyproject_path.is_file():
        print(f"Error: uv project path {pyproject_path} missing pyproject.toml", file=sys.stderr)
        sys.exit(1)
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependency_map = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for name, constraint in dependency_map.items():
        if name.lower() == "python":
            continue
        if isinstance(constraint, str):
            deps.append(f"{name}{constraint}")
        elif isinstance(constraint, dict):
            parts = [name]
            extras = constraint.get("extras")
            if extras:
                parts[0] += "[" + ",".join(extras) + "]"
            version = constraint.get("version")
            if version:
                parts.append(version)
            markers = constraint.get("markers")
            if markers:
                parts.append(f"; {markers}")
            deps.append("".join(parts))
        else:
            deps.append(name)
else:
    deps = ["networkx>=3.0", "numpy>=1.24", "colorama>=0.4.6", "pandas>=1.5", "graphviz>=0.20"]

for dep in deps:
    print(dep)
PY
) || exit 1

FILTERED_DEPS=()
for dep in "${EXTRA_DEPS[@]}"; do
  if [[ -n "${dep// }" ]]; then
    FILTERED_DEPS+=("$dep")
  fi
done

if [[ ${#FILTERED_DEPS[@]} -gt 0 ]]; then
  echo "[cloudcast setup] Installing dependencies: ${FILTERED_DEPS[*]}"
  python -m pip install --no-cache-dir "${FILTERED_DEPS[@]}"
else
  echo "[cloudcast setup] No dependencies discovered; skipping install"
fi

echo "[cloudcast setup] Environment setup completed successfully"
