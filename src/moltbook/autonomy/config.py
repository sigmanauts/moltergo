from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os


DEFAULT_KEYWORDS = [
    "aei",
    "artificial economic intelligence",
    "agent economy",
    "agent economies",
    "autonomous agents",
    "ai agents",
    "machine economy",
    "ergo",
    "erg",
    "ergoplatform",
    "ergoplatform.org",
    "ergoscript",
    "eutxo",
    "utxo",
    "sigma protocols",
    "sigusd",
    "rosen bridge",
    "oracle pools",
    "dexy",
    "bitcoin smart contracts",
    "programmable money",
    "on chain",
    "onchain",
    "privacy preserving",
    "zero knowledge",
    "defi",
    "cross chain",
    "micropayments",
    "crypto bounties",
    "smart contract",
]

DEFAULT_MISSION_QUERIES = [
    "AI agent economy posts discussing autonomous payments and on-chain settlement",
    "Web3 builders discussing programmable money for autonomous agents",
    "eUTXO or UTXO smart contract discussions for agent coordination",
    "posts about AI x DeFi infrastructure and crypto-native agent business models",
    "threads comparing chains for AI agents where Ergo could be relevant",
]


DEFAULT_PERSONA_HINT = (
    "You are an autonomous Moltbook agent focused on Ergo (ERG), eUTXO, and agent economies. "
    "Be curious, constructive, and technically literate. "
    "Avoid hype, avoid financial advice, and prefer short paragraphs with 1-3 direct questions."
)


@dataclass
class Config:
    poll_seconds: int
    idle_poll_seconds: int
    feed_limit: int
    posts_limit: int
    posts_sort: str
    search_limit: int
    search_batch_size: int
    discovery_mode: str
    reply_mode: str
    max_posts_per_day: int
    max_comments_per_day: int
    min_seconds_between_actions: int
    min_seconds_between_posts: int
    min_seconds_between_comments: int
    min_seconds_between_same_author: int
    min_confidence: float
    dry_run: bool
    state_path: Path
    persona_path: Optional[Path]
    keywords: List[str]
    mission_queries: List[str]
    keyword_store_path: Path
    keyword_learning_enabled: bool
    keyword_learning_interval_cycles: int
    keyword_learning_min_titles: int
    keyword_learning_max_suggestions: int
    search_retry_after_failure_cycles: int
    do_not_reply_authors: List[str]
    openai_api_key: Optional[str]
    openai_base_url: str
    openai_model: str
    openai_temperature: float
    log_level: str
    log_path: Optional[Path]
    confirm_actions: bool
    agent_name_hint: Optional[str]


def _parse_csv_env(env_key: str) -> List[str]:
    value = os.getenv(env_key, "")
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config() -> Config:
    poll_seconds = int(os.getenv("MOLTBOOK_POLL_SECONDS", "180"))
    idle_poll_seconds = int(os.getenv("MOLTBOOK_IDLE_POLL_SECONDS", "20"))
    feed_limit = int(os.getenv("MOLTBOOK_FEED_LIMIT", "30"))
    posts_limit = int(os.getenv("MOLTBOOK_POSTS_LIMIT", "30"))
    posts_sort = os.getenv("MOLTBOOK_POSTS_SORT", "new").strip().lower()
    search_limit = int(os.getenv("MOLTBOOK_SEARCH_LIMIT", "20"))
    search_batch_size = int(os.getenv("MOLTBOOK_SEARCH_BATCH_SIZE", "8"))
    discovery_mode = os.getenv("MOLTBOOK_DISCOVERY_MODE", "search").strip().lower()
    reply_mode = os.getenv("MOLTBOOK_REPLY_MODE", "auto").strip().lower()
    max_posts_per_day = int(os.getenv("MOLTBOOK_MAX_POSTS_PER_DAY", "2"))
    max_comments_per_day = int(os.getenv("MOLTBOOK_MAX_COMMENTS_PER_DAY", "10"))
    min_seconds_between_actions = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_ACTIONS", "1800"))
    min_seconds_between_posts = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_POSTS", str(min_seconds_between_actions)))
    min_seconds_between_comments = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_COMMENTS", "20"))
    min_seconds_between_same_author = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_SAME_AUTHOR", "21600"))
    min_confidence = float(os.getenv("MOLTBOOK_MIN_CONFIDENCE", "0.6"))
    dry_run = os.getenv("MOLTBOOK_DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}

    state_path = Path(os.getenv("MOLTBOOK_STATE_PATH", "memory/autonomy-state.json"))

    persona_path_str = os.getenv("MOLTBOOK_PERSONA_PATH", "docs/MESSAGING.md").strip()
    persona_path = Path(persona_path_str) if persona_path_str else None

    keywords = [k.lower() for k in _parse_csv_env("MOLTBOOK_KEYWORDS")]
    if not keywords:
        keywords = DEFAULT_KEYWORDS
    mission_queries_env = os.getenv("MOLTBOOK_MISSION_QUERIES", "")
    if mission_queries_env.strip():
        mission_queries = [q.strip() for q in mission_queries_env.split("||") if q.strip()]
    else:
        mission_queries = DEFAULT_MISSION_QUERIES[:]
    keyword_store_path = Path(os.getenv("MOLTBOOK_KEYWORD_STORE_PATH", "memory/learned-keywords.json"))
    keyword_learning_enabled = os.getenv("MOLTBOOK_KEYWORD_LEARNING_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    keyword_learning_interval_cycles = int(os.getenv("MOLTBOOK_KEYWORD_LEARNING_INTERVAL_CYCLES", "4"))
    keyword_learning_min_titles = int(os.getenv("MOLTBOOK_KEYWORD_LEARNING_MIN_TITLES", "15"))
    keyword_learning_max_suggestions = int(os.getenv("MOLTBOOK_KEYWORD_LEARNING_MAX_SUGGESTIONS", "6"))
    search_retry_after_failure_cycles = int(os.getenv("MOLTBOOK_SEARCH_RETRY_AFTER_FAILURE_CYCLES", "8"))

    do_not_reply_authors = [a.lower() for a in _parse_csv_env("MOLTBOOK_DO_NOT_REPLY_AUTHORS")]

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    log_level = os.getenv("MOLTBOOK_LOG_LEVEL", "INFO").strip().upper()
    log_path_str = os.getenv("MOLTBOOK_LOG_PATH", "").strip()
    log_path = Path(log_path_str) if log_path_str else None
    confirm_actions = os.getenv("MOLTBOOK_CONFIRM_ACTIONS", "1").strip().lower() in {"1", "true", "yes"}
    agent_name_hint = os.getenv("MOLTBOOK_AGENT_NAME", "").strip() or None

    return Config(
        poll_seconds=poll_seconds,
        idle_poll_seconds=idle_poll_seconds,
        feed_limit=feed_limit,
        posts_limit=posts_limit,
        posts_sort=posts_sort,
        search_limit=search_limit,
        search_batch_size=search_batch_size,
        discovery_mode=discovery_mode,
        reply_mode=reply_mode,
        max_posts_per_day=max_posts_per_day,
        max_comments_per_day=max_comments_per_day,
        min_seconds_between_actions=min_seconds_between_actions,
        min_seconds_between_posts=min_seconds_between_posts,
        min_seconds_between_comments=min_seconds_between_comments,
        min_seconds_between_same_author=min_seconds_between_same_author,
        min_confidence=min_confidence,
        dry_run=dry_run,
        state_path=state_path,
        persona_path=persona_path,
        keywords=keywords,
        mission_queries=mission_queries,
        keyword_store_path=keyword_store_path,
        keyword_learning_enabled=keyword_learning_enabled,
        keyword_learning_interval_cycles=keyword_learning_interval_cycles,
        keyword_learning_min_titles=keyword_learning_min_titles,
        keyword_learning_max_suggestions=keyword_learning_max_suggestions,
        search_retry_after_failure_cycles=search_retry_after_failure_cycles,
        do_not_reply_authors=do_not_reply_authors,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_temperature=openai_temperature,
        log_level=log_level,
        log_path=log_path,
        confirm_actions=confirm_actions,
        agent_name_hint=agent_name_hint,
    )
