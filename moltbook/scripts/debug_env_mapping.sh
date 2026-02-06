#!/usr/bin/env bash
set -euo pipefail

# Debug helper: show how run_env_agent.sh maps a slug to an env var and
# whether a key is present (without printing the key itself).
#
# Usage:
#   ./scripts/debug_env_mapping.sh <agent_slug>
#
# Example:
#   ./scripts/debug_env_mapping.sh ergo_builder

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <agent_slug>" >&2
  exit 1
fi

AGENT_SLUG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found in $(pwd)." >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

SLUG_UPPER="$(printf '%s' "$AGENT_SLUG" | tr '[:lower:]' '[:upper:]')"
VAR_NAME="MOLTBOOK_API_KEY_${SLUG_UPPER}"

VAL="${!VAR_NAME-}"

echo "Agent slug:        $AGENT_SLUG"
echo "Env var name:      $VAR_NAME"
if [ -z "$VAL" ]; then
  echo "Status:            NOT SET"
else
  echo "Status:            SET"
  echo "Approx key length: ${#VAL} characters"
fi
