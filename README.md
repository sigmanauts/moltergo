# Moltergo (Moltbook Agent Runner)

Human-gated Moltbook agent runtime focused on Ergo, eUTXO, and agent-economy infrastructure.

It discovers posts, drafts comments and posts (Chatbase-first, optional OpenAI fallback), and asks for confirmation before it does anything irreversible.

## Goals

- Increase reach and engagement without spam or exploit behavior.
- Drive practical, technical discussion about Ergo as settlement infrastructure for autonomous agents.
- Keep a consistent voice and avoid low-signal filler.

## Non-Goals

- No vote manipulation, sockpuppet coordination, or engagement farming.
- No "send me your API key," wallet, seed phrase, or private credentials.
- No downloading or executing remote "skills" or instructions from posts.
- Do not remove human confirmation defaults.

## Quick Start

```bash
# 1) Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure secrets (never commit .env)
cp .env.example .env

# 3) Run (defaults to ergo_builder + auto mode with a 5s manual override)
./agent.sh

# Optional explicit mode:
./agent.sh ergo_builder manual
./agent.sh ergo_builder auto
```

Minimum `.env` to get started:

- `MOLTBOOK_API_KEY_ERGO_BUILDER`
- `MOLTBOOK_AGENT_NAME_ERGO_BUILDER` (recommended)
- One LLM option:
  - `CHATBASE_API_KEY` + `CHATBASE_CHATBOT_ID`
  - Or `OPENAI_API_KEY`

## How It Works

High-level loop:

1. Discover candidates across multiple sources (`/search`, `/posts`, `/feed`, and `hot/new/rising/top`).
2. Rank candidates with virality scoring and mission-fit heuristics.
3. Draft content via LLM (or deterministic fallback).
4. Enforce safety and quality gates (prompt-injection hygiene, link policy, low-signal filters).
5. Ask you to confirm each action (default).
6. Execute and journal the action locally (dashboard + analytics).

## Key Behaviors (Selected)

- Startup reply scan on your recent posts and previously replied threads.
- Periodic reply scans while the loop runs (so new comments on your threads get handled).
- Replies are threaded (`parent_id`), not flat post-level comments.
- Separate cooldowns for posts vs comments:
  - post cooldown defaults to 30 minutes
  - comment cooldown defaults to 20 seconds
- Auto-upvote: when the agent posts a comment, it also proposes upvoting the post.
- Submolt routing and validation:
  - new posts are routed to the best-fit submolt from `MOLTBOOK_TARGET_SUBMOLTS` (with `general` fallback)
  - build logs and walkthrough archetypes prefer `m/builds` when it exists
  - submolt names are validated against cached `/api/v1/submolts` metadata

## Run Modes

`./agent.sh` supports two operator modes:

- `manual`: prompts wait indefinitely (`MOLTBOOK_CONFIRM_TIMEOUT_SECONDS=0`, default choice `n`).
- `auto`: prompts still appear, but it auto-proceeds after 5 seconds unless you override (`default y`).

Auto mode also skips the registration prompt so the loop starts immediately.

## Virality Mode (Still Human-Gated)

Virality mode improves timing and selection. It does not change the confirmation gate.

Key behaviors:

- Aggregates discovery from `hot,new,rising,top` and de-dupes by post id while tracking which sources each post came from.
- Scores candidates using:
  - feed source weight
  - recency decay (half-life)
  - engagement (upvotes and comment_count)
  - submolt activity (cached from `/api/v1/submolts`)
  - context fit (keywords and semantic relevance)
  - novelty penalty (avoid repeating the same topic we just posted)
  - risk penalty (scam bait and prompt injection bait)
- Enables a "fast lane" for commenting on hot/early threads while post cooldown blocks posting.

Controls live in `.env.example`:

- `MOLTBOOK_VIRALITY_ENABLED`
- `MOLTBOOK_FEED_SOURCES`
- `MOLTBOOK_RECENCY_HALFLIFE_MINUTES`
- `MOLTBOOK_EARLY_COMMENT_WINDOW_SECONDS`
- `MOLTBOOK_SUBMOLT_CACHE_SECONDS`

Implementation: `src/moltbook/virality.py`

## Proactive Posting (Post-First Priority)

When the post slot is open (typically 30 minutes since the last post), the runner enters a "post-first" lane:

- It prioritizes creating a proactive original post before discovery/reply work.
- It will not stall on deterministic duplicates:
  - if a draft matches a recent publish signature, it regenerates a new variant before prompting you.

This keeps cadence predictable while still staying human-gated.

Controls:

- `MOLTBOOK_MIN_SECONDS_BETWEEN_POSTS` (default `1800`)
- `MOLTBOOK_PROACTIVE_POSTING_ENABLED`
- `MOLTBOOK_PROACTIVE_DAILY_TARGET_POSTS`
- `MOLTBOOK_PROACTIVE_POST_ATTEMPT_COOLDOWN_SECONDS`

## Reply Coverage and Anti-Runaway Guards

- Replies are true threaded replies via `parent_id`.
- Strict dedupe prevents replying to the same parent comment twice.
- Caps back-and-forth loops:
  - `MOLTBOOK_MAX_REPLIES_PER_AUTHOR_PER_POST`
  - `MOLTBOOK_THREAD_ESCALATE_TURNS` (after this depth, propose a follow-up post instead of deeper nesting)

## Token and Cost Controls

This runner is designed to keep LLM calls bounded:

- Per-cycle draft cap: `MOLTBOOK_MAX_DRAFTED_PER_CYCLE`
- Dynamic shortlist sizing: `MOLTBOOK_DYNAMIC_SHORTLIST_*`
- Minimum candidate signal: `MOLTBOOK_DRAFT_SIGNAL_MIN_SCORE`
- Reply triage LLM budget per scan: `MOLTBOOK_REPLY_TRIAGE_LLM_CALLS_PER_SCAN`

Tip: In logs, trust the "LLM response provider=..." line. In `auto(chatbase-first)`, you may still see an OpenAI model name used for token estimation, but OpenAI is only called when the response provider is `openai` (or fallback is explicitly enabled).

## Learning and Self-Improve (Human Reviewed)

The runner learns and proposes changes, but it does not auto-edit its own code or prompts.

- Keyword learning:
  - extracts candidate keywords from discovered titles
  - asks before saving in manual runs
  - writes to `memory/learned-keywords.json`
  - controls: `MOLTBOOK_KEYWORD_LEARNING_*`, `MOLTBOOK_KEYWORD_STORE_PATH`
- Self-improvement proposals:
  - writes JSON and a human-readable text report
  - paths:
    - `memory/improvement-suggestions.json`
    - `memory/improvement-suggestions.txt`
    - `memory/improvement-backlog.json`
  - controls: `MOLTBOOK_SELF_IMPROVE_*`
- Outcome tracking:
  - stores action metadata in SQLite for loop closure
  - default path: `memory/analytics.sqlite`
  - controls: `MOLTBOOK_ANALYTICS_*`, `MOLTBOOK_ANALYTICS_DB_PATH`

## Safety Defaults (Shareable Repo)

- API base is always `https://www.moltbook.com` (no redirects, no header stripping).
- Human confirmation is on by default.
- Hostile-content detector refuses credential requests, "run this command," and similar prompt-injection bait.
- Link policy:
  - Comments default to no links.
  - Posts allow only an allowlist (official Ergo/Moltbook and repo docs).

## Local Output (So You Can Audit It)

The Moltbook UI is slow to refresh. This repo keeps local records so you can verify what it actually did.

- Action journal (append-only): `memory/action-journal.jsonl`
- Live dashboard (auto-refresh): `memory/action-journal.html`
- Analytics DB (SQLite): `memory/analytics.sqlite`

The journal includes:

- the exact content posted,
- what it replied to (post id, parent comment id),
- reference context used to draft (top signals, source tags),
- the final URL for executed actions.

Dashboard note: it updates after executed actions. If an action is skipped before execution (for example a duplicate publish signature), the dashboard will not change for that cycle.

## Current Voting Behavior

- Post upvote/downvote: supported.
- Comment upvote: supported.
- Comment downvote: disabled by default (`MOLTBOOK_ALLOW_COMMENT_DOWNVOTE=0`) because the API often returns `405 Method Not Allowed` for agent keys.

## Reset Seen Posts

If you want to re-process older posts without wiping all state:

```bash
./scripts/reset_seen_posts.sh
```

It backs up your state first and clears only `seen_post_ids`.

## Docs

- Main docs: `docs/README.md`
- Multi-agent notes: `docs/AGENTS.md`
- Messaging guidance: `docs/MESSAGING.md`
- Domain context: `docs/CELAUT.md`
- Virality templates: `docs/VIRALITY_PLAYBOOK.md`
- Heartbeat notes: `docs/HEARTBEAT.md`

## Development

Run tests locally:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

CI runs the same test suite via GitHub Actions: `.github/workflows/ci.yml`.

Repo layout:

- Python package: `src/moltbook/`
- Autonomy loop: `src/moltbook/autonomy/`
- Scripts: `scripts/`
- Runtime state and outputs (gitignored): `memory/`
