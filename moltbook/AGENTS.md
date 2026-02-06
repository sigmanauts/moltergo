# Running Multiple Moltbook Agents

This repo is designed so that **many different Ergo-aligned agents** can
reuse the same Python client and CLI, while each has its own API key and
persona.

This file shows one simple pattern for doing that using:

- Per-agent credential files under `~/.config/moltbook/agents/`
- A small helper script that loads the right key, then calls `cli.py`

> Security reminder: API keys in these examples **never leave your machine**
> except in authenticated requests to `https://www.moltbook.com/api/v1/*`.

## 1. Create per-agent credential files (local only)

For each agent, create a file:

```text
~/.config/moltbook/agents/<agent_slug>.json
```

Example for an agent persona "ErgoBuilder":

```json
{
  "api_key": "moltbook_xxx_for_ergobuilder",
  "agent_name": "ErgoBuilder"
}
```

Notes:

- These files live in your home dir, **not** in this repo.
- Do **not** commit them to git or share them with anyone.
- You can have as many agents as you like: `ergo_builder.json`,
  `ergo_educator.json`, `ergo_privacy.json`, etc.

## 2. Use the helper script to run a specific agent

This repo includes `scripts/run_agent.sh`, which:

- Reads the correct per-agent config from
  `~/.config/moltbook/agents/<agent_slug>.json`
- Exports `MOLTBOOK_API_KEY` for that agent
- Calls `python cli.py ...` with your remaining arguments

Make the script executable once:

```bash
cd moltbook
chmod +x scripts/run_agent.sh
```

### Examples

Show the profile for `ergo_builder`:

```bash
./scripts/run_agent.sh ergobuilder me
```

Post as `ergo_builder`:

```bash
./scripts/run_agent.sh ergo_builder me
  --submolt general \
  --title "Ergo as money for AI agents" \
  --content "Some thoughts on why Ergo's fair-launch + eUTXO fit agent economies."
```

Post as `ergoeducator` (different key + persona):

```bash
./scripts/run_agent.sh ergo_builder post \
  --submolt general \
  --title "Learning ErgoScript" \
  --content "Practical resources for understanding Ergo's smart contracts."
```

The only difference is the **first argument** (`ergo_builder` vs
`ergo_educator`), which selects a different `~/.config/moltbook/agents/*.json`
file and therefore a different API key.

## 3. Persona & behavior (prompting side)

This repo does **not** enforce any particular agent behavior. Instead, you can
use `MESSAGING.md` plus your own prompting to define personas in whatever
orchestration system you use (OpenAI, Claude, local LLM, etc.). For example:

- *ErgoBuilder*: focus on DeFi, tooling, dApp ideas.
- *ErgoEducator*: explain concepts clearly, welcome newcomers.
- *ErgoPrivacy*: emphasise Sigma protocols, privacy trade-offs, and
  responsible usage.

Each persona:

- Reads/uses `MESSAGING.md` as a shared strategy.
- Uses `scripts/run_agent.sh <slug> ...` to post/comment as the correct
  Moltbook identity.

## 4. Advanced: dedicated environments per agent (optional)

For heavier setups (e.g. each agent in its own container or process), you can
also:

- Give each agent its own working directory with a copy of this `moltbook`
  folder.
- Configure `~/.config/moltbook/credentials.json` or `MOLTBOOK_API_KEY`
  differently per container/user.

The core idea remains the same: **one codebase, many identities**, each with
its own API key and persona, all following the same Ergo-aware messaging
guidelines.

## 5. Using a `.env` file instead of JSON

If you prefer a single `.env` file in this repo over multiple JSON files in
`~/.config`, you can:

1. Copy `moltbook/.env.example` to `moltbook/.env`.
2. Fill in the values for:

   ```bash
   MOLTBOOK_API_KEY_ERGO_BUILDER="moltbook_xxx_for_ergo_builder"
   MOLTBOOK_API_KEY_ERGO_EDUCATOR="moltbook_xxx_for_ergo_educator"
   MOLTBOOK_API_KEY_ERGO_PRIVACY="moltbook_xxx_for_ergo_privacy"
   ```

3. Use the `.env` helper script instead of the JSON-based one:

   ```bash
   cd moltbook
   chmod +x scripts/run_env_agent.sh scripts/post_ergo_intro.sh

   # Show profile for ergo_builder using .env
   ./scripts/run_env_agent.sh ergo_builder me

   # Post the standard Ergo intro message as ergo_builder
   ./scripts/post_ergo_intro.sh ergo_builder general
   ```

The `.env` file is ignored by git via `moltbook/.gitignore`, so your keys stay
local. Never commit the real `.env` file or share it.

