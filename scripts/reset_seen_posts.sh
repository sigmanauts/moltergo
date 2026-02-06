#!/usr/bin/env bash
set -euo pipefail

# Reset autonomy seen-post state so previously seen posts can be reconsidered.
# Usage:
#   ./scripts/reset_seen_posts.sh
#   ./scripts/reset_seen_posts.sh /custom/state.json

STATE_PATH="${1:-memory/autonomy-state.json}"

if [ ! -f "$STATE_PATH" ]; then
  echo "State file not found: $STATE_PATH" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required for state reset." >&2
  exit 1
fi

if [ ! -t 0 ]; then
  echo "Refusing to reset state in non-interactive mode." >&2
  exit 1
fi

BACKUP_PATH="${STATE_PATH}.bak.$(date -u +%Y%m%d%H%M%S)"
cp "$STATE_PATH" "$BACKUP_PATH"

echo "Backup created: $BACKUP_PATH"
echo "This will clear seen post IDs so old posts can be processed again."
read -r -p "Proceed? [y/N]: " choice
choice="$(printf '%s' "${choice:-}" | tr '[:upper:]' '[:lower:]')"
if [ "$choice" != "y" ] && [ "$choice" != "yes" ]; then
  echo "Reset cancelled."
  exit 0
fi

tmp_file="$(mktemp)"
jq '.seen_post_ids = []' "$STATE_PATH" > "$tmp_file"
mv "$tmp_file" "$STATE_PATH"
echo "Seen posts reset in $STATE_PATH"
