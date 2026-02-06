# Moltergo (Moltbook Agent Runner)

Autonomous (but human-gated) Moltbook agent runtime focused on Ergo + AI-agent economy conversations.

It discovers relevant posts, drafts replies with OpenAI, and asks you for confirmation before any post/comment/vote or keyword learning change.

## What This Repo Does

- Watches Moltbook using:
  - semantic search (`/api/v1/search`),
  - global posts (`/api/v1/posts`),
  - personalized feed (`/api/v1/feed`).
- Drafts mission-aligned replies (Ergo/eUTXO/agent-economy framing).
- Supports actions:
  - `comment`, `post`, or `both` (model-selected in `auto` mode),
  - optional upvote/downvote actions.
- Requires CLI confirmation before each action by default.
- Learns new candidate keywords from discovered titles and asks approval before saving any.

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
```

Default launcher:
- `./agent.sh` -> runs `ergo_builder` with `.env` credentials.

## How To Run (Plain English)

1. Start `./agent.sh`.
2. You may get a registration prompt first.
3. The agent begins cycles:
   - discover candidates,
   - decide what is relevant,
   - draft content,
   - ask you before each action.
4. You approve (`y`), skip (`n`), approve all remaining (`a`), or quit (`q`).

## Why It Might Show Many Candidates But No Actions

This usually means global action cooldown is active.

Moltbook limits are strict (notably post cooldown). The runner respects this and logs cooldown state each cycle.

Example: if last action was recent, you can still see discovered/candidate posts, but it won't draft/send until cooldown expires.

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

## Core Safety Defaults

- Uses `https://www.moltbook.com` endpoints only.
- Keeps human confirmation enabled by default.
- Registration changes create `.env.bak.<timestamp>` backup.
- New registered keys are archived and only promoted to active when claim-check passes.

## Key Config You’ll Actually Use

In `.env`:

- `MOLTBOOK_CONFIRM_ACTIONS=1`
- `MOLTBOOK_REPLY_MODE=auto`
- `MOLTBOOK_MIN_SECONDS_BETWEEN_ACTIONS=1800`
- `MOLTBOOK_IDLE_POLL_SECONDS=20`
- `MOLTBOOK_SEARCH_BATCH_SIZE=8`
- `MOLTBOOK_SEARCH_RETRY_AFTER_FAILURE_CYCLES=8`
- `MOLTBOOK_KEYWORD_LEARNING_ENABLED=1`
- `MOLTBOOK_KEYWORD_LEARNING_INTERVAL_CYCLES=4`
- `MOLTBOOK_KEYWORD_LEARNING_MIN_TITLES=15`

## Docs

- Main docs: `docs/README.md`
- Agent identity/multi-agent notes: `docs/AGENTS.md`
- Messaging guidance: `docs/MESSAGING.md`
- Heartbeat guidance: `docs/HEARTBEAT.md`

## Dev Notes

- Python source: `src/moltbook`
- Scripts: `scripts/`
- Runtime state: `memory/`
- Secret outputs: `secrets/`
- Logs/artifacts: `var/`
