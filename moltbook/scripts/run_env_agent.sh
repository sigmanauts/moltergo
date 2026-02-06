#!/usr/bin/env bash
set -euo pipefail

# Helper wrapper to run the Moltbook CLI as a specific agent, using a
# `.env` file in this directory.
#
# Usage:
#   ./scripts/run_env_agent.sh <agent_slug> me
#   ./scripts/run_env_agent.sh <agent_slug> post --submolt general --title ... --content ...
#
# Expects a .env file with variables like:
#   MOLTBOOK_API_KEY_ERGOBUILDER="moltbook_xxx_for_ergobuilder"
#   MOLTBOOK_API_KEY_ERGOEDUCATOR="moltbook_xxx_for_ergoeducator"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <agent_slug> <cli arguments...>" >&2
  exit 1
fi

AGENT_SLUG="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

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

# Debug hint: uncomment the next line temporarily if you need to confirm
# which variable name is being used (but NOT the value):
# echo "[run_env_agent] Using env var name: $VAR_NAME" >&2

# Indirect expansion to read $VAR_NAME
API_KEY="${!VAR_NAME-}"

if [ -z "${API_KEY}" ]; then
  echo "Error: variable $VAR_NAME not set in $ENV_FILE" >&2
  exit 1
fi

export MOLTBOOK_API_KEY="$API_KEY"

python cli.py "$@"
