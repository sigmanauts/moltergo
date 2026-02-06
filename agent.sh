#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the autonomous agent loop.
# Defaults: ergo_builder slug + .env-based key loading.
#
# Usage:
#   ./agent.sh
#   ./agent.sh ergo_builder
#   AGENT_SLUG=ergo_builder ./agent.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AGENT_SLUG="${1:-${AGENT_SLUG:-ergo_builder}}"

exec ./scripts/run_env_agent_loop.sh "$AGENT_SLUG"
