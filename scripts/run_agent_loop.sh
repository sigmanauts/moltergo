#!/usr/bin/env bash
set -euo pipefail

# Run the autonomous loop as a specific agent using ~/.config/moltbook/agents/<slug>.json
#
# Usage:
#   ./scripts/run_agent_loop.sh <agent_slug>

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <agent_slug>" >&2
  exit 1
fi

AGENT_SLUG="$1"

CONFIG_DIR="$HOME/.config/moltbook/agents"
CONFIG_PATH="$CONFIG_DIR/${AGENT_SLUG}.json"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: config not found: $CONFIG_PATH" >&2
  echo "Create it with: { \"api_key\": \"moltbook_xxx\", \"agent_name\": \"$AGENT_SLUG\" }" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: 'jq' is required to parse $CONFIG_PATH. Install it via your package manager (e.g. 'brew install jq')." >&2
  exit 1
fi

API_KEY="$(jq -r '.api_key' "$CONFIG_PATH")"
AGENT_NAME="$(jq -r '.agent_name // empty' "$CONFIG_PATH")"

if [ -z "$API_KEY" ] || [ "$API_KEY" = "null" ]; then
  echo "Error: 'api_key' missing or null in $CONFIG_PATH" >&2
  exit 1
fi

export MOLTBOOK_API_KEY="$API_KEY"
if [ -n "$AGENT_NAME" ]; then
  export MOLTBOOK_AGENT_NAME="$AGENT_NAME"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
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
