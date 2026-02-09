#!/usr/bin/env bash
set -euo pipefail

# Run the autonomous loop as a specific agent using a .env file in this directory.
#
# Usage:
#   ./scripts/run_env_agent_loop.sh <agent_slug>

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <agent_slug>" >&2
  exit 1
fi

AGENT_SLUG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

AUTO_REGISTER="${MOLTBOOK_AUTO_REGISTER:-1}"
if [ "$AUTO_REGISTER" = "1" ] || [ "$AUTO_REGISTER" = "true" ]; then
  if ! ./scripts/register_ergo_agents.sh "$AGENT_SLUG"; then
    echo "Warning: auto-registration attempt failed; continuing with existing .env credentials." >&2
  fi
fi

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found in $(pwd). Copy .env.example to .env and fill in your keys." >&2
  exit 1
fi

# Load variables from .env into the environment
set -a
source "$ENV_FILE"
set +a

# Uppercase slug to build variable name MOLTBOOK_API_KEY_<SLUG>
# Use POSIX-compatible uppercasing (macOS ships an older bash without "^^").
SLUG_UPPER="$(printf '%s' "$AGENT_SLUG" | tr '[:lower:]' '[:upper:]')"
VAR_NAME="MOLTBOOK_API_KEY_${SLUG_UPPER}"
NAME_VAR_NAME="MOLTBOOK_AGENT_NAME_${SLUG_UPPER}"

# Indirect expansion to read $VAR_NAME
API_KEY="${!VAR_NAME-}"
AGENT_NAME_VALUE="${!NAME_VAR_NAME-}"

if [ -z "${API_KEY}" ]; then
  echo "Error: variable $VAR_NAME not set in $ENV_FILE" >&2
  exit 1
fi

export MOLTBOOK_API_KEY="$API_KEY"
export MOLTBOOK_API_KEY_VAR="$VAR_NAME"
export MOLTBOOK_API_KEY_SOURCE="env:${VAR_NAME}"
if [ -n "${AGENT_NAME_VALUE}" ]; then
  export MOLTBOOK_AGENT_NAME="${AGENT_NAME_VALUE}"
fi
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: no Python interpreter found. Install python3 or create .venv." >&2
  exit 1
fi

"$PYTHON_BIN" -m moltbook.autonomy.runner
