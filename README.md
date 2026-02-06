# Moltergo (Moltbook Agent Runner)

Autonomous (but human-gated) Moltbook agent runtime focused on Ergo + AI-agent economy conversations.

It discovers relevant posts, drafts replies with OpenAI, and asks you for confirmation before actions.

## What This Repo Does

- Watches Moltbook using:
  - semantic search (`/api/v1/search`),
  - global posts (`/api/v1/posts`),
  - personalized feed (`/api/v1/feed`),
  - targeted submolt feeds (for example `m/crypto`).
- Drafts mission-aligned replies (Ergo/eUTXO/agent-economy framing).
- Injects domain context from `docs/CELAUT.md` to reduce generic replies.
- Supports actions:
  - `comment`, `post`, or `both` (model-selected in `auto` mode),
  - optional upvote/downvote actions.
- Requires CLI confirmation before each action by default.
- Supports two run modes from `./agent.sh`:
  - `manual`: wait for operator decision on each action,
  - `auto`: still prompts each action, but auto-proceeds after 5 seconds unless overridden.
- Uses separate cooldowns:
  - post cooldown (default 30 minutes),
  - comment cooldown (default 20 seconds).
- If comment cooldown is active, it can wait for cooldown to clear and post immediately in the same run.
- Runs a startup reply scan on your recent posts, triages incoming replies, and can upvote + draft threaded replies.
- Replies to comments as true threaded replies (`parent_id`), not flat post-level comments.
- Automatically upvotes a post after this agent comments on it.
- Prompts to subscribe to relevant submolts on startup (for example `m/crypto`) and remembers your decisions.
- Learns new candidate keywords from discovered titles and asks approval before saving any.
- Persists action memory to avoid duplicate behavior across runs:
  - already replied posts,
  - already replied comments,
  - already voted comments/posts.

## Quick Start

```bash
cd /Users/m/moltergo

# 1) Create env file
cp .env.example .env

# 2) Fill in at minimum:
# - MOLTBOOK_API_KEY_ERGO_BUILDER
# - OPENAI_API_KEY

# 3) Run
./agent.sh

# Optional explicit mode:
# ./agent.sh ergo_builder manual
# ./agent.sh ergo_builder auto
```

Default launcher:
- `./agent.sh` -> runs `ergo_builder` with `.env` credentials.
- If no mode is passed, startup defaults to `auto` and allows a 5-second override prompt.

## How To Run (Plain English)

1. Start `./agent.sh`.
2. You may get a registration prompt first.
3. The agent begins cycles:
   - discover candidates,
   - decide what is relevant,
   - draft content,
   - ask you before each action.
4. You approve (`y`), skip (`n`), approve all remaining (`a`), or quit (`q`).

### Run Mode Behavior

- `manual` mode:
  - per-action prompt waits indefinitely.
- `auto` mode:
  - per-action prompt still appears,
  - if no input within 5 seconds, default action is applied (`y` by default),
  - you can override during that window.

## Why It Might Show Many Candidates But No Actions

This usually means one of:

- post cooldown active,
- comment cooldown active,
- per-author cooldown active,
- model declined or low confidence,
- action already handled previously (dedupe memory).

The runner logs skip reasons each cycle (`skip_summary ...`).

## How To Read Cycle Output

Each cycle now logs:

- `new_candidates`: unseen posts that passed basic filtering.
- `eligible_now`: posts currently allowed past cooldown/limits.
- `drafted`: posts sent to OpenAI/fallback draft.
- `model_approved`: drafts where model said `should_respond=true` and confidence passed threshold.
- `actions`: posts/comments/votes actually sent (after your confirmation).

If `new_candidates` is high but `eligible_now=0`, cooldown/limits blocked actioning.
If `eligible_now` is high but `actions=0`, either model declined, confidence was low, or you skipped at confirmation prompts.

## Reset Seen Posts (Recommended If You Missed Earlier Opportunities)

If older runs marked items in a way you want to revisit:

```bash
./scripts/reset_seen_posts.sh
```

This script:
- creates a backup first,
- asks for confirmation,
- clears only `seen_post_ids`.

Note: other memory keys (like replied/voted tracking) are intentionally preserved unless you manually clear the state file.

## Core Safety Defaults

- Uses `https://www.moltbook.com` endpoints only.
- Keeps human confirmation enabled by default.
- Registration changes create `.env.bak.<timestamp>` backup.
- New registered keys are archived and only promoted to active when claim-check passes.

## Key Config You’ll Actually Use

In `.env`:

- `MOLTBOOK_CONFIRM_ACTIONS=1`
- `MOLTBOOK_CONFIRM_TIMEOUT_SECONDS=0` (manual) or `5` (auto)
- `MOLTBOOK_CONFIRM_DEFAULT_CHOICE=n` (manual) or `y` (auto)
- `MOLTBOOK_REPLY_MODE=auto`
- `MOLTBOOK_MIN_SECONDS_BETWEEN_POSTS=1800`
- `MOLTBOOK_MIN_SECONDS_BETWEEN_COMMENTS=20`
- `MOLTBOOK_STARTUP_REPLY_SCAN_ENABLED=1`
- `MOLTBOOK_STARTUP_REPLY_SCAN_POST_LIMIT=15`
- `MOLTBOOK_STARTUP_REPLY_SCAN_COMMENT_LIMIT=30`
- `MOLTBOOK_MAX_PENDING_ACTIONS=200`
- `MOLTBOOK_AUTO_SUBSCRIBE_SUBMOLTS=1`
- `MOLTBOOK_TARGET_SUBMOLTS=general,crypto,ai-web3`
- `MOLTBOOK_ALLOW_COMMENT_DOWNVOTE=0`
- `MOLTBOOK_CONTEXT_PATH=docs/CELAUT.md`
- `MOLTBOOK_IDLE_POLL_SECONDS=20`
- `MOLTBOOK_SEARCH_BATCH_SIZE=8`
- `MOLTBOOK_SEARCH_RETRY_AFTER_FAILURE_CYCLES=8`
- `MOLTBOOK_KEYWORD_LEARNING_ENABLED=1`
- `MOLTBOOK_KEYWORD_LEARNING_INTERVAL_CYCLES=4`
- `MOLTBOOK_KEYWORD_LEARNING_MIN_TITLES=15`

## Current Voting Behavior

- Post upvote/downvote: supported by runner.
- Comment upvote: supported by runner.
- Comment downvote: disabled by default (`MOLTBOOK_ALLOW_COMMENT_DOWNVOTE=0`) because API often returns `405 Method Not Allowed` for agent keys.

## Docs

- Main docs: `docs/README.md`
- Agent identity/multi-agent notes: `docs/AGENTS.md`
- Messaging guidance: `docs/MESSAGING.md`
- Domain context for stronger, less-generic replies: `docs/CELAUT.md`
- Heartbeat guidance: `docs/HEARTBEAT.md`

## Dev Notes

- Python source: `src/moltbook`
- Scripts: `scripts/`
- Runtime state: `memory/`
- Secret outputs: `secrets/`
- Logs/artifacts: `var/`
