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
    "decentralized computing",
    "data sovereignty",
    "ai ethics",
    "trustless escrow",
    "service orchestration",
    "agent reputation",
    "counterparty risk",
]

DEFAULT_MISSION_QUERIES = [
    "AI agent economy posts discussing autonomous payments and on-chain settlement",
    "Web3 builders discussing programmable money for autonomous agents",
    "eUTXO or UTXO smart contract discussions for agent coordination",
    "posts about AI x DeFi infrastructure and crypto-native agent business models",
    "threads comparing chains for AI agents where Ergo could be relevant",
    "discussions about decentralized computing, data sovereignty, or AI ethics where trustless settlement matters",
]

DEFAULT_TARGET_SUBMOLTS = [
    "general",
    "crypto",
    "ai-web3",
]


DEFAULT_PERSONA_HINT = (
    "You are a radical artificial economic intelligence agent focused on Ergo (ERG), eUTXO, and agent economies. "
    "Write like a sovereign protocol builder: sharp, technical, anti-rent-seeking, and pro-coordination. "
    "Be provocative but constructive. Avoid financial advice. End with 1-3 direct, high-signal questions."
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
    max_comments_per_hour: int
    min_seconds_between_actions: int
    min_seconds_between_posts: int
    min_seconds_between_comments: int
    min_seconds_between_same_author: int
    min_confidence: float
    dry_run: bool
    state_path: Path
    persona_path: Optional[Path]
    context_path: Optional[Path]
    keywords: List[str]
    mission_queries: List[str]
    target_submolts: List[str]
    auto_subscribe_submolts: bool
    keyword_store_path: Path
    keyword_learning_enabled: bool
    keyword_learning_interval_cycles: int
    keyword_learning_min_titles: int
    keyword_learning_max_suggestions: int
    draft_shortlist_size: int
    draft_signal_min_score: int
    virality_enabled: bool
    feed_sources: List[str]
    recency_halflife_minutes: int
    early_comment_window_seconds: int
    submolt_cache_seconds: int
    dynamic_shortlist_enabled: bool
    dynamic_shortlist_min: int
    dynamic_shortlist_max: int
    search_retry_after_failure_cycles: int
    startup_reply_scan_enabled: bool
    startup_reply_scan_post_limit: int
    startup_reply_scan_comment_limit: int
    startup_reply_scan_replied_post_limit: int
    reply_triage_llm_calls_per_scan: int
    reply_scan_interval_cycles: int
    max_drafts_per_cycle: int
    trending_min_post_score: int
    trending_min_comment_count: int
    follow_ergo_authors_enabled: bool
    follow_ergo_authors_per_cycle: int
    badbot_warning_enabled: bool
    badbot_warning_min_strikes: int
    badbot_max_warnings_per_scan: int
    badbot_max_warnings_per_author_per_day: int
    max_comment_chars: int
    proactive_posting_enabled: bool
    proactive_post_attempt_cooldown_seconds: int
    proactive_post_reference_limit: int
    proactive_post_submolt: str
    proactive_daily_target_posts: int
    proactive_force_general_until_daily_target: bool
    proactive_memory_path: Path
    proactive_metrics_refresh_seconds: int
    self_improve_enabled: bool
    self_improve_interval_cycles: int
    self_improve_min_titles: int
    self_improve_max_suggestions: int
    self_improve_path: Path
    self_improve_text_path: Path
    self_improve_backlog_path: Path
    analytics_db_path: Path
    action_journal_path: Path
    analytics_refresh_interval_cycles: int
    analytics_summary_interval_cycles: int
    max_pending_actions: int
    do_not_reply_authors: List[str]
    openai_api_key: Optional[str]
    openai_base_url: str
    openai_model: str
    openai_temperature: float
    groq_api_key: Optional[str]
    groq_base_url: str
    groq_model: str
    ollama_base_url: str
    ollama_model: str
    chatbase_api_key: Optional[str]
    chatbase_chatbot_id: Optional[str]
    chatbase_base_url: str
    openrouter_api_key: Optional[str]
    openrouter_base_url: str
    openrouter_model: str
    openrouter_site_url: Optional[str]
    openrouter_app_name: Optional[str]
    llm_provider: str
    llm_auto_fallback_to_openai: bool
    log_level: str
    log_path: Optional[Path]
    confirm_actions: bool
    confirm_timeout_seconds: int
    confirm_default_choice: str
    allow_comment_downvote: bool
    agent_name_hint: Optional[str]
    max_cycles: int


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
    max_posts_per_day = int(os.getenv("MOLTBOOK_MAX_POSTS_PER_DAY", "48"))
    max_comments_per_day = int(os.getenv("MOLTBOOK_MAX_COMMENTS_PER_DAY", "1200"))
    max_comments_per_hour = int(os.getenv("MOLTBOOK_MAX_COMMENTS_PER_HOUR", "50"))
    min_seconds_between_actions = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_ACTIONS", "1800"))
    min_seconds_between_posts = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_POSTS", str(min_seconds_between_actions)))
    min_seconds_between_comments = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_COMMENTS", "20"))
    min_seconds_between_same_author = int(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_SAME_AUTHOR", "21600"))
    min_confidence = float(os.getenv("MOLTBOOK_MIN_CONFIDENCE", "0.6"))
    dry_run = os.getenv("MOLTBOOK_DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}

    state_path = Path(os.getenv("MOLTBOOK_STATE_PATH", "memory/autonomy-state.json"))

    persona_path_str = os.getenv("MOLTBOOK_PERSONA_PATH", "docs/MESSAGING.md").strip()
    persona_path = Path(persona_path_str) if persona_path_str else None
    context_path_str = os.getenv("MOLTBOOK_CONTEXT_PATH", "docs/CELAUT.md").strip()
    context_path = Path(context_path_str) if context_path_str else None

    keywords = [k.lower() for k in _parse_csv_env("MOLTBOOK_KEYWORDS")]
    if not keywords:
        keywords = DEFAULT_KEYWORDS
    mission_queries_env = os.getenv("MOLTBOOK_MISSION_QUERIES", "")
    if mission_queries_env.strip():
        mission_queries = [q.strip() for q in mission_queries_env.split("||") if q.strip()]
    else:
        mission_queries = DEFAULT_MISSION_QUERIES[:]
    target_submolts = [s.lower() for s in _parse_csv_env("MOLTBOOK_TARGET_SUBMOLTS")]
    if not target_submolts:
        target_submolts = DEFAULT_TARGET_SUBMOLTS[:]
    auto_subscribe_submolts = os.getenv("MOLTBOOK_AUTO_SUBSCRIBE_SUBMOLTS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    keyword_store_path = Path(os.getenv("MOLTBOOK_KEYWORD_STORE_PATH", "memory/learned-keywords.json"))
    keyword_learning_enabled = os.getenv("MOLTBOOK_KEYWORD_LEARNING_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    keyword_learning_interval_cycles = int(os.getenv("MOLTBOOK_KEYWORD_LEARNING_INTERVAL_CYCLES", "4"))
    keyword_learning_min_titles = int(os.getenv("MOLTBOOK_KEYWORD_LEARNING_MIN_TITLES", "15"))
    keyword_learning_max_suggestions = int(os.getenv("MOLTBOOK_KEYWORD_LEARNING_MAX_SUGGESTIONS", "6"))
    draft_shortlist_size = int(os.getenv("MOLTBOOK_DRAFT_SHORTLIST_SIZE", "18"))
    draft_signal_min_score = int(os.getenv("MOLTBOOK_DRAFT_SIGNAL_MIN_SCORE", "2"))
    virality_enabled = os.getenv("MOLTBOOK_VIRALITY_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    feed_sources = [s.strip().lower() for s in os.getenv("MOLTBOOK_FEED_SOURCES", "hot,new,rising,top").split(",")]
    feed_sources = [s for s in feed_sources if s in {"hot", "new", "rising", "top"}]
    if not feed_sources:
        feed_sources = ["hot", "new", "rising", "top"]
    recency_halflife_minutes = int(os.getenv("MOLTBOOK_RECENCY_HALFLIFE_MINUTES", "180"))
    early_comment_window_seconds = int(os.getenv("MOLTBOOK_EARLY_COMMENT_WINDOW_SECONDS", "900"))
    submolt_cache_seconds = int(os.getenv("MOLTBOOK_SUBMOLT_CACHE_SECONDS", "900"))
    dynamic_shortlist_enabled = os.getenv("MOLTBOOK_DYNAMIC_SHORTLIST_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    dynamic_shortlist_min = int(os.getenv("MOLTBOOK_DYNAMIC_SHORTLIST_MIN", "6"))
    dynamic_shortlist_max = int(os.getenv("MOLTBOOK_DYNAMIC_SHORTLIST_MAX", "30"))
    search_retry_after_failure_cycles = int(os.getenv("MOLTBOOK_SEARCH_RETRY_AFTER_FAILURE_CYCLES", "8"))
    startup_reply_scan_enabled = os.getenv("MOLTBOOK_STARTUP_REPLY_SCAN_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    startup_reply_scan_post_limit = int(os.getenv("MOLTBOOK_STARTUP_REPLY_SCAN_POST_LIMIT", "15"))
    startup_reply_scan_comment_limit = int(os.getenv("MOLTBOOK_STARTUP_REPLY_SCAN_COMMENT_LIMIT", "40"))
    startup_reply_scan_replied_post_limit = int(os.getenv("MOLTBOOK_STARTUP_REPLY_SCAN_REPLIED_POST_LIMIT", "25"))
    reply_triage_llm_calls_per_scan = int(os.getenv("MOLTBOOK_REPLY_TRIAGE_LLM_CALLS_PER_SCAN", "3"))
    reply_scan_interval_cycles = int(os.getenv("MOLTBOOK_REPLY_SCAN_INTERVAL_CYCLES", "1"))
    max_drafts_per_cycle_raw = os.getenv("MOLTBOOK_MAX_DRAFTED_PER_CYCLE", "").strip()
    if not max_drafts_per_cycle_raw:
        max_drafts_per_cycle_raw = os.getenv("MOLTBOOK_MAX_DRAFTS_PER_CYCLE", "8").strip()
    max_drafts_per_cycle = int(max_drafts_per_cycle_raw or "8")
    trending_min_post_score = int(os.getenv("MOLTBOOK_TRENDING_MIN_POST_SCORE", "2"))
    trending_min_comment_count = int(os.getenv("MOLTBOOK_TRENDING_MIN_COMMENT_COUNT", "3"))
    follow_ergo_authors_enabled = os.getenv("MOLTBOOK_FOLLOW_ERGO_AUTHORS_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    follow_ergo_authors_per_cycle = int(os.getenv("MOLTBOOK_FOLLOW_ERGO_AUTHORS_PER_CYCLE", "2"))
    badbot_warning_enabled = os.getenv("MOLTBOOK_BADBOT_WARNING_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    badbot_warning_min_strikes = int(os.getenv("MOLTBOOK_BADBOT_WARNING_MIN_STRIKES", "4"))
    badbot_max_warnings_per_scan = int(os.getenv("MOLTBOOK_BADBOT_MAX_WARNINGS_PER_SCAN", "3"))
    badbot_max_warnings_per_author_per_day = int(
        os.getenv("MOLTBOOK_BADBOT_MAX_WARNINGS_PER_AUTHOR_PER_DAY", "2")
    )
    proactive_posting_enabled = os.getenv("MOLTBOOK_PROACTIVE_POSTING_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    proactive_post_attempt_cooldown_seconds = int(
        os.getenv("MOLTBOOK_PROACTIVE_POST_ATTEMPT_COOLDOWN_SECONDS", "900")
    )
    proactive_post_reference_limit = int(os.getenv("MOLTBOOK_PROACTIVE_POST_REFERENCE_LIMIT", "12"))
    proactive_post_submolt = os.getenv("MOLTBOOK_PROACTIVE_POST_SUBMOLT", "general").strip() or "general"
    proactive_daily_target_posts = int(os.getenv("MOLTBOOK_PROACTIVE_DAILY_TARGET_POSTS", "1"))
    proactive_force_general_until_daily_target = os.getenv(
        "MOLTBOOK_PROACTIVE_FORCE_GENERAL_UNTIL_DAILY_TARGET",
        "1",
    ).strip().lower() in {"1", "true", "yes"}
    proactive_memory_path = Path(os.getenv("MOLTBOOK_PROACTIVE_MEMORY_PATH", "memory/post-engine-memory.json"))
    proactive_metrics_refresh_seconds = int(os.getenv("MOLTBOOK_PROACTIVE_METRICS_REFRESH_SECONDS", "300"))
    self_improve_enabled = os.getenv("MOLTBOOK_SELF_IMPROVE_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    self_improve_interval_cycles = int(os.getenv("MOLTBOOK_SELF_IMPROVE_INTERVAL_CYCLES", "12"))
    self_improve_min_titles = int(os.getenv("MOLTBOOK_SELF_IMPROVE_MIN_TITLES", "25"))
    self_improve_max_suggestions = int(os.getenv("MOLTBOOK_SELF_IMPROVE_MAX_SUGGESTIONS", "6"))
    self_improve_path = Path(os.getenv("MOLTBOOK_SELF_IMPROVE_PATH", "memory/improvement-suggestions.json"))
    self_improve_text_path = Path(
        os.getenv("MOLTBOOK_SELF_IMPROVE_TEXT_PATH", "memory/improvement-suggestions.txt")
    )
    self_improve_backlog_path = Path(
        os.getenv("MOLTBOOK_SELF_IMPROVE_BACKLOG_PATH", "memory/improvement-backlog.json")
    )
    analytics_db_path = Path(os.getenv("MOLTBOOK_ANALYTICS_DB_PATH", "memory/analytics.sqlite"))
    action_journal_path = Path(os.getenv("MOLTBOOK_ACTION_JOURNAL_PATH", "memory/action-journal.jsonl"))
    analytics_refresh_interval_cycles = int(os.getenv("MOLTBOOK_ANALYTICS_REFRESH_INTERVAL_CYCLES", "3"))
    analytics_summary_interval_cycles = int(os.getenv("MOLTBOOK_ANALYTICS_SUMMARY_INTERVAL_CYCLES", "12"))
    max_pending_actions = int(os.getenv("MOLTBOOK_MAX_PENDING_ACTIONS", "200"))

    do_not_reply_authors = [a.lower() for a in _parse_csv_env("MOLTBOOK_DO_NOT_REPLY_AUTHORS")]

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    chatbase_api_key = os.getenv("CHATBASE_API_KEY")
    chatbase_chatbot_id = (
        os.getenv("CHATBASE_CHATBOT_ID", "").strip()
        or os.getenv("CHATBASE_AGENT_ID", "").strip()
        or None
    )
    chatbase_base_url = os.getenv("CHATBASE_BASE_URL", "https://www.chatbase.co/api/v1").rstrip("/")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
    openrouter_app_name = os.getenv("OPENROUTER_APP_NAME")
    llm_provider = os.getenv("MOLTBOOK_LLM_PROVIDER", "auto").strip().lower()
    if llm_provider not in {"auto", "openai", "chatbase", "ollama", "groq", "openrouter"}:
        llm_provider = "auto"
    llm_auto_fallback_to_openai = os.getenv("MOLTBOOK_LLM_AUTO_FALLBACK_TO_OPENAI", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    log_level = os.getenv("MOLTBOOK_LOG_LEVEL", "INFO").strip().upper()
    log_path_str = os.getenv("MOLTBOOK_LOG_PATH", "").strip()
    if log_path_str:
        log_path = Path(log_path_str)
    else:
        log_path = Path("memory/autonomy.log")
    confirm_actions = os.getenv("MOLTBOOK_CONFIRM_ACTIONS", "1").strip().lower() in {"1", "true", "yes"}
    confirm_timeout_seconds = int(os.getenv("MOLTBOOK_CONFIRM_TIMEOUT_SECONDS", "0"))
    confirm_default_choice = os.getenv("MOLTBOOK_CONFIRM_DEFAULT_CHOICE", "n").strip().lower()
    if confirm_default_choice not in {"y", "n", "a", "q"}:
        confirm_default_choice = "n"
    allow_comment_downvote = os.getenv("MOLTBOOK_ALLOW_COMMENT_DOWNVOTE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    agent_name_hint = os.getenv("MOLTBOOK_AGENT_NAME", "").strip() or None
    max_comment_chars = int(os.getenv("MOLTBOOK_MAX_COMMENT_CHARS", "900"))
    max_cycles = int(os.getenv("MOLTBOOK_MAX_CYCLES", "0"))

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
        max_comments_per_hour=max_comments_per_hour,
        min_seconds_between_actions=min_seconds_between_actions,
        min_seconds_between_posts=min_seconds_between_posts,
        min_seconds_between_comments=min_seconds_between_comments,
        min_seconds_between_same_author=min_seconds_between_same_author,
        min_confidence=min_confidence,
        dry_run=dry_run,
        state_path=state_path,
        persona_path=persona_path,
        context_path=context_path,
        keywords=keywords,
        mission_queries=mission_queries,
        target_submolts=target_submolts,
        auto_subscribe_submolts=auto_subscribe_submolts,
        keyword_store_path=keyword_store_path,
        keyword_learning_enabled=keyword_learning_enabled,
        keyword_learning_interval_cycles=keyword_learning_interval_cycles,
        keyword_learning_min_titles=keyword_learning_min_titles,
        keyword_learning_max_suggestions=keyword_learning_max_suggestions,
        draft_shortlist_size=draft_shortlist_size,
        draft_signal_min_score=draft_signal_min_score,
        virality_enabled=virality_enabled,
        feed_sources=feed_sources,
        recency_halflife_minutes=recency_halflife_minutes,
        early_comment_window_seconds=early_comment_window_seconds,
        submolt_cache_seconds=submolt_cache_seconds,
        dynamic_shortlist_enabled=dynamic_shortlist_enabled,
        dynamic_shortlist_min=dynamic_shortlist_min,
        dynamic_shortlist_max=dynamic_shortlist_max,
        search_retry_after_failure_cycles=search_retry_after_failure_cycles,
        startup_reply_scan_enabled=startup_reply_scan_enabled,
        startup_reply_scan_post_limit=startup_reply_scan_post_limit,
        startup_reply_scan_comment_limit=startup_reply_scan_comment_limit,
        startup_reply_scan_replied_post_limit=startup_reply_scan_replied_post_limit,
        reply_triage_llm_calls_per_scan=reply_triage_llm_calls_per_scan,
        reply_scan_interval_cycles=reply_scan_interval_cycles,
        max_drafts_per_cycle=max_drafts_per_cycle,
        trending_min_post_score=trending_min_post_score,
        trending_min_comment_count=trending_min_comment_count,
        follow_ergo_authors_enabled=follow_ergo_authors_enabled,
        follow_ergo_authors_per_cycle=follow_ergo_authors_per_cycle,
        badbot_warning_enabled=badbot_warning_enabled,
        badbot_warning_min_strikes=badbot_warning_min_strikes,
        badbot_max_warnings_per_scan=badbot_max_warnings_per_scan,
        badbot_max_warnings_per_author_per_day=badbot_max_warnings_per_author_per_day,
        proactive_posting_enabled=proactive_posting_enabled,
        proactive_post_attempt_cooldown_seconds=proactive_post_attempt_cooldown_seconds,
        proactive_post_reference_limit=proactive_post_reference_limit,
        proactive_post_submolt=proactive_post_submolt,
        proactive_daily_target_posts=proactive_daily_target_posts,
        proactive_force_general_until_daily_target=proactive_force_general_until_daily_target,
        proactive_memory_path=proactive_memory_path,
        proactive_metrics_refresh_seconds=proactive_metrics_refresh_seconds,
        self_improve_enabled=self_improve_enabled,
        self_improve_interval_cycles=self_improve_interval_cycles,
        self_improve_min_titles=self_improve_min_titles,
        self_improve_max_suggestions=self_improve_max_suggestions,
        self_improve_path=self_improve_path,
        self_improve_text_path=self_improve_text_path,
        self_improve_backlog_path=self_improve_backlog_path,
        analytics_db_path=analytics_db_path,
        action_journal_path=action_journal_path,
        analytics_refresh_interval_cycles=analytics_refresh_interval_cycles,
        analytics_summary_interval_cycles=analytics_summary_interval_cycles,
        max_pending_actions=max_pending_actions,
        do_not_reply_authors=do_not_reply_authors,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_temperature=openai_temperature,
        groq_api_key=groq_api_key,
        groq_base_url=groq_base_url,
        groq_model=groq_model,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        chatbase_api_key=chatbase_api_key,
        chatbase_chatbot_id=chatbase_chatbot_id,
        chatbase_base_url=chatbase_base_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
        openrouter_model=openrouter_model,
        openrouter_site_url=openrouter_site_url,
        openrouter_app_name=openrouter_app_name,
        llm_provider=llm_provider,
        llm_auto_fallback_to_openai=llm_auto_fallback_to_openai,
        log_level=log_level,
        log_path=log_path,
        confirm_actions=confirm_actions,
        confirm_timeout_seconds=confirm_timeout_seconds,
        confirm_default_choice=confirm_default_choice,
        allow_comment_downvote=allow_comment_downvote,
        agent_name_hint=agent_name_hint,
        max_comment_chars=max_comment_chars,
        max_cycles=max_cycles,
    )
