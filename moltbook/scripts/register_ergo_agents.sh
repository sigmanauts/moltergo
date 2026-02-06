#!/usr/bin/env bash
set -euo pipefail

# Register a set of Ergo-focused Moltbook agents and save their API keys
# to a local text file (agent_keys.txt).
#
# IMPORTANT SECURITY NOTE:
# - This script runs entirely on your machine.
# - The API keys are written ONLY to ./agent_keys.txt (which is git-ignored).
# - Do NOT commit or share agent_keys.txt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

OUTPUT_FILE="agent_keys.txt"
ENV_FILE=".env"
LOG_FILE="register_log.jsonl"
API_BASE_URL="${MOLTBOOK_API_BASE_URL:-https://www.moltbook.com}"
REGISTER_ENDPOINT="${API_BASE_URL%/}/api/v1/agents/register"

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' is required but not installed." >&2
    exit 1
  fi
}

need_cmd curl
need_cmd jq
need_cmd sed
need_cmd date

if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Creating $OUTPUT_FILE and writing header." >&2
  {
    echo "# Moltbook agent API keys (keep this file secret, do NOT commit it)";
    echo "# slug name api_key claim_url verification_code";
  } > "$OUTPUT_FILE"
else
  echo "Appending to $OUTPUT_FILE (leaving existing entries intact)." >&2
fi

# Ensure .env exists (but do NOT overwrite existing content)
if [ ! -f "$ENV_FILE" ]; then
  echo "Creating $ENV_FILE (keep this file secret, do NOT commit it)." >&2
  echo "# Moltbook .env (keep this file secret, do NOT commit it)" > "$ENV_FILE"
fi

# Load .env into the current shell so the script can use auth variables.
# This allows either:
# - MOLTBOOK_API_KEY (preferred), or
# - MOLTBOOK_TOKEN, or
# - MOLTBOOK_BEARER_TOKEN
#
# The script will try these in order.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE" >/dev/null 2>&1 || true
set +a

get_auth_header() {
  if [ -n "${MOLTBOOK_API_KEY:-}" ]; then
    echo "Authorization: Bearer ${MOLTBOOK_API_KEY}"
    return 0
  fi
  if [ -n "${MOLTBOOK_TOKEN:-}" ]; then
    echo "Authorization: Bearer ${MOLTBOOK_TOKEN}"
    return 0
  fi
  if [ -n "${MOLTBOOK_BEARER_TOKEN:-}" ]; then
    echo "Authorization: Bearer ${MOLTBOOK_BEARER_TOKEN}"
    return 0
  fi
  if [ -n "${MOLTBOOK_X_API_KEY:-}" ]; then
    echo "X-API-Key: ${MOLTBOOK_X_API_KEY}"
    return 0
  fi
  return 1
}

register_agent() {
  local slug="$1"
  local name="$2"
  local description="$3"

  echo "Registering agent '$name' (slug label: $slug)..." >&2

  # Docs: register expects only name + description, no auth header. :contentReference[oaicite:3]{index=3}
  local payload
  payload="$(jq -n --arg name "$name" --arg desc "$description" '{name: $name, description: $desc}')"

  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local tmp_body http_code response
  tmp_body="$(mktemp)"

  http_code="$(
    curl -sS -w "%{http_code}" -o "$tmp_body" \
      "https://www.moltbook.com/api/v1/agents/register" \
      -H "Content-Type: application/json" \
      -d "$payload" || true
  )"

  response="$(cat "$tmp_body")"
  rm -f "$tmp_body"

  printf '%s\t%s\t%s\t%s\n' "$now" "$slug" "$name" "$response" >> "$LOG_FILE"

  if ! printf '%s' "$response" | jq -e . >/dev/null 2>&1; then
    echo "  Registration failed for $name: non-JSON response (HTTP $http_code)." >&2
    echo "  $response" >&2
    return 1
  fi

  local success err
  success="$(printf '%s' "$response" | jq -r '.success // empty' 2>/dev/null || true)"
  if [ "$success" != "true" ]; then
    err="$(printf '%s' "$response" | jq -r '.error // empty' 2>/dev/null || true)"
    echo "  Registration failed for $name (HTTP $http_code): ${err:-unknown error}. Raw response:" >&2
    echo "  $response" >&2
    return 1
  fi

  local api_key claim_url verification_code
  api_key="$(printf '%s' "$response" | jq -r '.agent.api_key // empty' 2>/dev/null || true)"
  claim_url="$(printf '%s' "$response" | jq -r '.agent.claim_url // empty' 2>/dev/null || true)"
  verification_code="$(printf '%s' "$response" | jq -r '.agent.verification_code // empty' 2>/dev/null || true)"

  if [ -z "$api_key" ]; then
    echo "  Error: success=true but api_key missing (HTTP $http_code). Raw response:" >&2
    echo "  $response" >&2
    return 1
  fi

  echo "$slug $name $api_key $claim_url $verification_code" >> "$OUTPUT_FILE"
  echo "  Saved credentials for $name to $OUTPUT_FILE (api_key hidden here)." >&2

  local slug_upper env_var
  slug_upper="$(printf '%s' "$slug" | tr '[:lower:]' '[:upper:]')"
  env_var="MOLTBOOK_API_KEY_${slug_upper}"

  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "/^${env_var}=/d" "$ENV_FILE" 2>/dev/null || true
  else
    sed -i "/^${env_var}=/d" "$ENV_FILE" 2>/dev/null || true
  fi

  printf '%s="%s"\n' "$env_var" "$api_key" >> "$ENV_FILE"
  echo "  Updated $ENV_FILE entry for $env_var (value not shown)." >&2
}

# Define the Ergo-focused agents (slugs use underscores).
SUFFIX="$(date +%H%M%S)"

register_agent \
  "ergo_builder" \
  "ErgoBuilderMoltergo_${SUFFIX}" \
  "Ergo DeFi & tooling agent – explores dApps, contracts, and practical ways AI agents can use Ergo for autonomous transactions." || true

echo "Done. Open $OUTPUT_FILE to see the slugs, names, and API keys. Keep it secret and do not commit it to git." >&2
