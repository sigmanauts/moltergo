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

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_BLUE=$'\033[1;34m'
  C_CYAN=$'\033[1;36m'
  C_GREEN=$'\033[1;32m'
  C_YELLOW=$'\033[1;33m'
else
  C_RESET=""
  C_BOLD=""
  C_BLUE=""
  C_CYAN=""
  C_GREEN=""
  C_YELLOW=""
fi

panel_line() {
  local text="${1:-}"
  printf '| %-68s |\n' "$text"
}

panel_box() {
  local title="${1:-INFO}"
  shift || true
  printf '\n%s+----------------------------------------------------------------------+%s\n' "$C_BLUE" "$C_RESET"
  printf '%s| %-68s |%s\n' "$C_BLUE" "${C_BOLD}${title}${C_RESET}${C_BLUE}" "$C_RESET"
  printf '%s+----------------------------------------------------------------------+%s\n' "$C_BLUE" "$C_RESET"
  while [ "$#" -gt 0 ]; do
    panel_line "$1"
    shift
  done
  printf '%s+----------------------------------------------------------------------+%s\n\n' "$C_BLUE" "$C_RESET"
}

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
    panel_box \
      "${C_YELLOW}START MODE${C_RESET}" \
      "default: auto" \
      "press 'm' + Enter within 5s for manual" \
      "(or wait to continue in auto)"
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
  export MOLTBOOK_AUTO_REGISTER="${MOLTBOOK_AUTO_REGISTER:-1}"
else
  export MOLTBOOK_CONFIRM_ACTIONS=1
  export MOLTBOOK_CONFIRM_TIMEOUT_SECONDS=5
  export MOLTBOOK_CONFIRM_DEFAULT_CHOICE=y
  # Auto mode should start the loop immediately and skip registration prompts.
  export MOLTBOOK_AUTO_REGISTER=0
  export MOLTBOOK_REGISTER_PROMPT=0
fi

panel_box \
  "${C_GREEN}LAUNCHING AGENT${C_RESET}" \
  "slug: $AGENT_SLUG" \
  "mode: $RUN_MODE" \
  "confirm: $MOLTBOOK_CONFIRM_ACTIONS (timeout=${MOLTBOOK_CONFIRM_TIMEOUT_SECONDS}s default=$MOLTBOOK_CONFIRM_DEFAULT_CHOICE)" \
  "auto_register: $MOLTBOOK_AUTO_REGISTER" \
  "pythonpath: src"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
exec ./scripts/run_env_agent_loop.sh "$AGENT_SLUG"
