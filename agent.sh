#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the autonomous agent loop.
# Defaults: ergo_builder slug + .env-based key loading.
#
# Usage:
#   ./agent.sh
#   ./agent.sh ergo_builder
#   ./agent.sh ergo_builder manual
#   ./agent.sh ergo_builder auto
#   AGENT_SLUG=ergo_builder ./agent.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AGENT_SLUG="${1:-${AGENT_SLUG:-ergo_builder}}"
RUN_MODE_INPUT="${2:-${AGENT_RUN_MODE:-}}"
RUN_MODE=""

normalize_mode() {
  local v
  v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$v" in
    m|manual) echo "manual" ;;
    a|auto|"") echo "auto" ;;
    *) echo "" ;;
  esac
}

if [ -n "$RUN_MODE_INPUT" ]; then
  RUN_MODE="$(normalize_mode "$RUN_MODE_INPUT")"
  if [ -z "$RUN_MODE" ]; then
    echo "Error: invalid run mode '$RUN_MODE_INPUT'. Use 'manual' or 'auto'." >&2
    exit 1
  fi
else
  RUN_MODE="auto"
  if [ -t 0 ]; then
    echo ""
    echo "============================================"
    echo " START MODE"
    echo " default: auto"
    echo " press 'm' + Enter within 5s for manual"
    echo " (or just wait to continue in auto)"
    echo "============================================"
    if IFS= read -r -t 5 MODE_OVERRIDE; then
      MODE_NORMALIZED="$(normalize_mode "$MODE_OVERRIDE")"
      if [ "$MODE_NORMALIZED" = "manual" ]; then
        RUN_MODE="manual"
      fi
    fi
  fi
fi

if [ "$RUN_MODE" = "manual" ]; then
  export MOLTBOOK_CONFIRM_ACTIONS=1
  export MOLTBOOK_CONFIRM_TIMEOUT_SECONDS=0
  export MOLTBOOK_CONFIRM_DEFAULT_CHOICE=n
else
  export MOLTBOOK_CONFIRM_ACTIONS=1
  export MOLTBOOK_CONFIRM_TIMEOUT_SECONDS=5
  export MOLTBOOK_CONFIRM_DEFAULT_CHOICE=y
fi

echo "Launching agent slug='$AGENT_SLUG' mode='$RUN_MODE' (confirm=$MOLTBOOK_CONFIRM_ACTIONS timeout=${MOLTBOOK_CONFIRM_TIMEOUT_SECONDS}s default=$MOLTBOOK_CONFIRM_DEFAULT_CHOICE)"

exec ./scripts/run_env_agent_loop.sh "$AGENT_SLUG"
