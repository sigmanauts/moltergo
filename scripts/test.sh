#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x ".venv/bin/python" ]; then
  echo "Error: .venv/bin/python not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install pytest" >&2
  exit 1
fi

.venv/bin/python -m pytest -q
