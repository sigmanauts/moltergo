#!/usr/bin/env bash
set -euo pipefail

# Register an Ergo-focused Moltbook agent and update local env vars.
#
# Behavior:
# - Can register on each startup (gated by prompt unless disabled).
# - Writes API keys to secrets/agent_keys.txt (git-ignored).
# - Writes run log lines to var/logs/register_log.jsonl.
# - Updates .env with MOLTBOOK_API_KEY_<SLUG> and MOLTBOOK_AGENT_NAME_<SLUG>.
# - Also updates MOLTBOOK_API_KEY and MOLTBOOK_AGENT_NAME when slug matches
#   MOLTBOOK_PRIMARY_AGENT_SLUG (default: ergo_builder).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TARGET_SLUG="${1:-ergo_builder}"
PRIMARY_SLUG="${MOLTBOOK_PRIMARY_AGENT_SLUG:-ergo_builder}"
REGISTER_PROMPT="${MOLTBOOK_REGISTER_PROMPT:-1}"

OUTPUT_FILE="secrets/agent_keys.txt"
ENV_FILE=".env"
LOG_FILE="var/logs/register_log.jsonl"
REGISTER_ENDPOINT="https://www.moltbook.com/api/v1/agents/register"

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
need_cmd awk
need_cmd grep

mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$LOG_FILE")"

print_banner() {
  local title="$1"
  local line2="${2:-}"
  local line3="${3:-}"
  local green="" cyan="" reset=""
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    green="\033[1;32m"
    cyan="\033[1;36m"
    reset="\033[0m"
  fi

  printf "\n%b============================================%b\n" "$green" "$reset"
  printf "%b %s%b\n" "$green" "$title" "$reset"
  if [ -n "$line2" ]; then
    printf "%b %s%b\n" "$cyan" "$line2" "$reset"
  fi
  if [ -n "$line3" ]; then
    printf "%b %s%b\n" "$cyan" "$line3" "$reset"
  fi
  printf "%b============================================%b\n\n" "$green" "$reset"
}

confirm_registration() {
  local slug="$1"
  local reason="$2"

  if [ "$REGISTER_PROMPT" != "1" ] && [ "$REGISTER_PROMPT" != "true" ]; then
    return 0
  fi

  if [ ! -t 0 ]; then
    echo "Skipping registration for '$slug': prompt required but stdin is non-interactive." >&2
    return 1
  fi

  echo ""
  echo "[confirm] Registration gate"
  echo "  slug: $slug"
  echo "  reason: $reason"
  echo "  action: create a new Moltbook agent (manual human verification required)"
  read -r -p "Do you want to register a new agent? [y/N]: " choice
  choice="$(printf '%s' "${choice:-}" | tr '[:upper:]' '[:lower:]')"
  if [ "$choice" = "y" ] || [ "$choice" = "yes" ]; then
    return 0
  fi
  return 1
}

if [ ! -f "$OUTPUT_FILE" ]; then
  {
    echo "# Moltbook agent API keys (keep this file secret, do NOT commit it)"
    echo "# slug name api_key claim_url verification_code"
  } > "$OUTPUT_FILE"
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "# Moltbook .env (keep this file secret, do NOT commit it)" > "$ENV_FILE"
fi

ENV_BACKED_UP=0
ENV_BACKUP_PATH=""

backup_env_once() {
  if [ "$ENV_BACKED_UP" = "1" ]; then
    return
  fi
  if [ ! -f "$ENV_FILE" ]; then
    return
  fi
  local stamp
  stamp="$(date -u +%Y%m%d%H%M%S)"
  ENV_BACKUP_PATH="${ENV_FILE}.bak.${stamp}"
  cp "$ENV_FILE" "$ENV_BACKUP_PATH"
  ENV_BACKED_UP=1
  echo "Created env backup: $ENV_BACKUP_PATH" >&2
}

set_env_var() {
  local key="$1"
  local value="$2"

  backup_env_once

  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "/^${key}=/d" "$ENV_FILE" 2>/dev/null || true
  else
    sed -i "/^${key}=/d" "$ENV_FILE" 2>/dev/null || true
  fi

  append_env_var "$key" "$value"
}

append_env_var() {
  local key="$1"
  local value="$2"
  local escaped
  backup_env_once
  escaped="$(printf "%s" "$value" | sed "s/'/'\"'\"'/g")"
  printf "%s='%s'\n" "$key" "$escaped" >> "$ENV_FILE"
}

register_agent() {
  local slug="$1"
  local name="$2"
  local description="$3"

  echo "Registering Moltbook agent name='$name' slug='$slug'..." >&2

  local payload now tmp_body http_code response
  payload="$(jq -n --arg name "$name" --arg desc "$description" '{name: $name, description: $desc}')"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tmp_body="$(mktemp)"

  http_code="$(
    curl -sS -w "%{http_code}" -o "$tmp_body" \
      "$REGISTER_ENDPOINT" \
      -H "Content-Type: application/json" \
      -d "$payload" || true
  )"

  response="$(cat "$tmp_body")"
  rm -f "$tmp_body"

  printf '%s\t%s\t%s\t%s\n' "$now" "$slug" "$name" "$response" >> "$LOG_FILE"

  if ! printf '%s' "$response" | jq -e . >/dev/null 2>&1; then
    echo "Registration failed: non-JSON response (HTTP $http_code): $response" >&2
    return 1
  fi

  local success error_message
  success="$(printf '%s' "$response" | jq -r '.success // empty' 2>/dev/null || true)"
  if [ "$success" != "true" ]; then
    error_message="$(printf '%s' "$response" | jq -r '.error // empty' 2>/dev/null || true)"
    echo "Registration failed (HTTP $http_code): ${error_message:-unknown error}" >&2
    return 1
  fi

  local api_key claim_url verification_code
  api_key="$(printf '%s' "$response" | jq -r '.agent.api_key // empty' 2>/dev/null || true)"
  claim_url="$(printf '%s' "$response" | jq -r '.agent.claim_url // empty' 2>/dev/null || true)"
  verification_code="$(printf '%s' "$response" | jq -r '.agent.verification_code // empty' 2>/dev/null || true)"

  if [ -z "$api_key" ]; then
    echo "Registration failed: success=true but api_key missing." >&2
    return 1
  fi

  echo "$slug $name $api_key $claim_url $verification_code" >> "$OUTPUT_FILE"

  local slug_upper
  slug_upper="$(printf '%s' "$slug" | tr '[:lower:]' '[:upper:]')"
  local stamp
  stamp="$(date -u +%Y%m%d%H%M%S)"

  # Append-only history to avoid losing prior agent credentials for this slug.
  append_env_var "MOLTBOOK_API_KEY_${slug_upper}_${stamp}" "$api_key"
  append_env_var "MOLTBOOK_AGENT_NAME_${slug_upper}_${stamp}" "$name"
  # Promote only if key is already claim-verified.
  local me_code
  me_code="$(
    curl -sS -o /dev/null -w "%{http_code}" \
      "https://www.moltbook.com/api/v1/agents/me" \
      -H "Authorization: Bearer ${api_key}" || true
  )"
  if [ "$me_code" = "200" ]; then
    set_env_var "MOLTBOOK_API_KEY_${slug_upper}" "$api_key"
    set_env_var "MOLTBOOK_AGENT_NAME_${slug_upper}" "$name"
    if [ "$slug" = "$PRIMARY_SLUG" ]; then
      set_env_var "MOLTBOOK_API_KEY" "$api_key"
      set_env_var "MOLTBOOK_AGENT_NAME" "$name"
    fi
    echo "Updated active .env credentials for slug '$slug' (claimed)." >&2
  else
    echo "Newly registered agent is not claimed yet (agents/me HTTP ${me_code})." >&2
    echo "Keeping existing active credentials unchanged; archived key only." >&2
  fi
  echo "Verification URL: ${claim_url:-unavailable}" >&2
  echo "Verification code: ${verification_code:-unavailable}" >&2

  print_banner \
    "REGISTRATION SUCCESS" \
    "slug: $slug agent_name: $name" \
    "verify: ${claim_url:-n/a} code: ${verification_code:-n/a}"
}

if ! confirm_registration "$TARGET_SLUG" "startup registration"; then
  echo "Registration skipped by user." >&2
  print_banner "REGISTRATION SKIPPED" "slug: $TARGET_SLUG" "reason: user declined registration"
  exit 0
fi

ts_suffix="$(date -u +%Y%m%d%H%M%S)"
slug_title="$(printf '%s' "$TARGET_SLUG" | awk -F '_' '{for(i=1;i<=NF;i++){printf toupper(substr($i,1,1)) substr($i,2)}}')"
# Keep name short and simple: some Moltbook validations reject longer formats.
short_suffix="$(date -u +%H%M%S)"
agent_name="${slug_title}Moltergo${short_suffix}"
agent_name="$(printf '%s' "$agent_name" | tr -cd '[:alnum:]_-')"
agent_name="$(printf '%s' "$agent_name" | cut -c1-30)"

case "$TARGET_SLUG" in
  ergo_educator)
    agent_description="Ergo educator agent. Explains Ergo concepts, guides newcomers, and shares practical resources."
    ;;
  ergo_privacy)
    agent_description="Ergo privacy agent. Focuses on Sigma protocols, privacy trade-offs, and responsible usage."
    ;;
  *)
    agent_description="Ergo DeFi and tooling agent. Explores practical ways AI agents can use Ergo for autonomous transactions."
    ;;
esac

register_agent "$TARGET_SLUG" "$agent_name" "$agent_description"

echo "Registration complete for '$TARGET_SLUG'." >&2
