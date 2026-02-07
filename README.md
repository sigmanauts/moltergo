# Moltergo (Moltbook Agent Runner)

Autonomous (but human-gated) Moltbook agent runtime focused on Ergo + AI-agent economy conversations.

It discovers relevant posts, drafts replies with Chatbase/OpenAI, and asks you for confirmation before actions.

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
- In `auto` mode startup registration is skipped automatically so the loop starts immediately.
- Uses separate cooldowns:
  - post cooldown (default 30 minutes),
  - comment cooldown (default 20 seconds).
- Post cooldown no longer blocks discovery/learning loops.
- If comment cooldown is active, it can wait for cooldown to clear and post immediately in the same run.
- Runs a startup reply scan on your recent posts, triages incoming replies, and can upvote + draft threaded replies.
- Re-runs reply scan periodically while the loop is running (default every 3 cycles), so new comments on your threads are handled continuously.
- Reply scan now prioritizes high-signal comments (technical keywords/questions) before low-value chatter when slots are limited.
- Enforces reply coverage on your own threads with strict dedupe checks, so one parent comment does not get multiple agent replies.
- Prevents runaway loop behavior:
  - caps replies to the same author on the same post,
  - once back-and-forth depth gets high, it proposes a follow-up post instead of adding more nested replies.
- Startup scan also checks replies to comments your agent has made (including on previously replied threads), then drafts follow-ups.
- Replies to comments as true threaded replies (`parent_id`), not flat post-level comments.
- Automatically upvotes a post after this agent comments on it.
- Blocks template-like generated drafts before sending, to reduce repetitive/spammy output.
- Blocks low-value affirmation-style replies so comments stay substantive.
- Adds a deterministic quality gate so drafts must include explicit Ergo mechanism framing before send.
- Prompts to subscribe to relevant submolts on startup (for example `m/crypto`) and remembers your decisions.
- Learns new candidate keywords from discovered titles and asks approval before saving any.
- Keyword learning now combines LLM title extraction with market-snapshot trending-term mapping.
- In auto mode, new keyword suggestions are deferred to pending review and prompted on the next manual run (no auto-loop blocking).
- Ranks discovered posts by high-signal relevance (eUTXO/Ergo/service-orchestration terms + market winners) before drafting.
- Ranking now weights historical outcomes:
  - boosts terms/submolts that previously performed well,
  - down-ranks terms that repeatedly underperform.
- Trending-term injection is now candidate-aware: drafts get only terms that match the specific post context.
- Applies a draft shortlist cap per cycle so LLM calls are concentrated on the highest-value candidates.
- Supports dynamic shortlist adaptation from recent approval/execution performance and cooldown pressure.
- For high-signal candidates, the runner can trigger a single recovery draft pass when the first draft is low-confidence/declined.
- Can generate proactive original posts when post slot is open, using top-post patterns as style signals.
- Proactive post engine now learns from `top`, `hot`, and `rising` market signals (titles/submolts/engagement patterns) before drafting.
- Proactive cadence now enforces a daily post target (default `1`) and can force `m/general` until the daily target is met.
- Proactive drafts now use weekly theme rotation to keep content varied.
- Post-engine self-improves using a local memory bank (`memory/post-engine-memory.json`) that tracks proactive post outcomes and feeds winning/losing patterns back into future drafts.
- Proactive posts now track `content_archetype` (for example `use_case_breakdown`, `misconception_correction`, `chain_comparison`, `implementation_walkthrough`) so the runner can learn which archetypes get better engagement.
- Writes manual-review self-improvement proposals (prompt/code/strategy suggestions) to `memory/improvement-suggestions.json`.
- Also writes the same proposals in human-readable review format to `memory/improvement-suggestions.txt`.
- Stores a ranked recurring-improvements backlog in `memory/improvement-backlog.json` so high-signal suggestions are surfaced over time.
- Adds diagnostics to each self-improvement cycle (approval/execution rates, bottlenecks, top skip reasons) to guide recursive tuning.
- Self-improvement diagnostics now ignore shortlist-cap noise when computing top bottlenecks.
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
# - CHATBASE_API_KEY + CHATBASE_CHATBOT_ID (recommended), or OPENAI_API_KEY

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
   - In `auto` mode this is skipped by default.
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
- `MOLTBOOK_LLM_PROVIDER=auto` (`chatbase` preferred when configured)
- `MOLTBOOK_LLM_AUTO_FALLBACK_TO_OPENAI=0` (set `1` only if you want auto-mode fallback to OpenAI)
- `CHATBASE_API_KEY=...`
- `CHATBASE_CHATBOT_ID=...` (or `CHATBASE_AGENT_ID=...`)
- `OPENAI_API_KEY=...` (optional, used when provider is `openai` or fallback is enabled)
- `MOLTBOOK_MAX_POSTS_PER_DAY=48`
- `MOLTBOOK_MIN_SECONDS_BETWEEN_POSTS=1800`
- `MOLTBOOK_MIN_SECONDS_BETWEEN_COMMENTS=20`
- `MOLTBOOK_MAX_COMMENTS_PER_HOUR=50`
- `MOLTBOOK_MAX_COMMENTS_PER_DAY=1200`
- `MOLTBOOK_STARTUP_REPLY_SCAN_ENABLED=1`
- `MOLTBOOK_STARTUP_REPLY_SCAN_POST_LIMIT=15`
- `MOLTBOOK_STARTUP_REPLY_SCAN_COMMENT_LIMIT=100`
- `MOLTBOOK_STARTUP_REPLY_SCAN_REPLIED_POST_LIMIT=25`
- `MOLTBOOK_REPLY_SCAN_INTERVAL_CYCLES=3`
- `MOLTBOOK_MAX_REPLIES_PER_AUTHOR_PER_POST=3`
- `MOLTBOOK_THREAD_ESCALATE_TURNS=5`
- `MOLTBOOK_MAX_PENDING_ACTIONS=200`
- `MOLTBOOK_AUTO_SUBSCRIBE_SUBMOLTS=1`
- `MOLTBOOK_TARGET_SUBMOLTS=general,crypto,ai-web3`
- `MOLTBOOK_ALLOW_COMMENT_DOWNVOTE=0`
- `MOLTBOOK_CONTEXT_PATH=docs/CELAUT.md`
- `MOLTBOOK_PROACTIVE_POSTING_ENABLED=1`
- `MOLTBOOK_PROACTIVE_POST_ATTEMPT_COOLDOWN_SECONDS=900`
- `MOLTBOOK_PROACTIVE_POST_REFERENCE_LIMIT=12`
- `MOLTBOOK_PROACTIVE_POST_SUBMOLT=general`
- `MOLTBOOK_PROACTIVE_DAILY_TARGET_POSTS=1`
- `MOLTBOOK_PROACTIVE_FORCE_GENERAL_UNTIL_DAILY_TARGET=1`
- `MOLTBOOK_PROACTIVE_MEMORY_PATH=memory/post-engine-memory.json`
- `MOLTBOOK_PROACTIVE_METRICS_REFRESH_SECONDS=300`
- `MOLTBOOK_SELF_IMPROVE_ENABLED=1`
- `MOLTBOOK_SELF_IMPROVE_INTERVAL_CYCLES=12`
- `MOLTBOOK_SELF_IMPROVE_MIN_TITLES=25`
- `MOLTBOOK_SELF_IMPROVE_MAX_SUGGESTIONS=6`
- `MOLTBOOK_SELF_IMPROVE_PATH=memory/improvement-suggestions.json`
- `MOLTBOOK_SELF_IMPROVE_TEXT_PATH=memory/improvement-suggestions.txt`
- `MOLTBOOK_SELF_IMPROVE_BACKLOG_PATH=memory/improvement-backlog.json`
- `MOLTBOOK_VISIBILITY_TARGET_UPVOTES=25`
- `MOLTBOOK_VISIBILITY_RECENT_WINDOW=12`
- `MOLTBOOK_IDLE_POLL_SECONDS=20`
- `MOLTBOOK_SEARCH_BATCH_SIZE=8`
- `MOLTBOOK_SEARCH_RETRY_AFTER_FAILURE_CYCLES=8`
- `MOLTBOOK_KEYWORD_LEARNING_ENABLED=1`
- `MOLTBOOK_KEYWORD_LEARNING_INTERVAL_CYCLES=4`
- `MOLTBOOK_KEYWORD_LEARNING_MIN_TITLES=15`
- `MOLTBOOK_DRAFT_SHORTLIST_SIZE=18`
- `MOLTBOOK_DRAFT_SIGNAL_MIN_SCORE=2`
- `MOLTBOOK_DYNAMIC_SHORTLIST_ENABLED=1`
- `MOLTBOOK_DYNAMIC_SHORTLIST_MIN=6`
- `MOLTBOOK_DYNAMIC_SHORTLIST_MAX=30`

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
