from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..moltbook_client import MoltbookAuthError, MoltbookClient

from .config import Config, load_config
from .drafting import (
    build_reply_triage_messages,
    build_proactive_post_messages,
    build_self_improvement_messages,
    build_openai_messages,
    call_generation_model,
    fallback_draft,
    format_content,
    load_context_text,
    load_persona_text,
    normalize_str,
    post_url,
    propose_keywords_from_titles,
)
from .keywords import load_keyword_store, merge_keywords, save_keyword_store
from .logging_utils import setup_logging
from .post_engine_memory import (
    append_improvement_suggestions,
    append_improvement_suggestions_text,
    build_improvement_diagnostics,
    build_improvement_feedback_context,
    build_learning_snapshot,
    load_recent_improvement_entries,
    load_recent_improvement_raw_entries,
    load_post_engine_memory,
    record_declined_idea,
    record_proactive_post,
    refresh_metrics_from_recent_posts,
    sanitize_improvement_suggestions,
    update_improvement_backlog,
    update_market_signals,
    save_post_engine_memory,
)
from .runtime_helpers import preview_text
from .state import load_state, reset_daily_if_needed, save_state, utc_now
from .ui import (
    confirm_action,
    confirm_keyword_addition,
    print_cycle_banner,
    print_keyword_added_banner,
    print_runtime_banner,
    print_success_banner,
)


VALID_RESPONSE_MODES = {"comment", "post", "both", "none"}
VALID_VOTE_ACTIONS = {"upvote", "downvote", "none"}
VALID_VOTE_TARGETS = {"post", "top_comment", "both", "none"}
PROACTIVE_ARCHETYPES = [
    "use_case_breakdown",
    "misconception_correction",
    "chain_comparison",
    "implementation_walkthrough",
]


def has_generation_provider(cfg: Config) -> bool:
    if cfg.llm_provider == "chatbase":
        return bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id)
    if cfg.llm_provider == "openai":
        return bool(cfg.openai_api_key)
    return bool((cfg.chatbase_api_key and cfg.chatbase_chatbot_id) or cfg.openai_api_key)


def post_score(post: Dict[str, Any]) -> int:
    for key in ("score", "upvotes", "vote_score", "likes"):
        value = post.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def post_comment_count(post: Dict[str, Any]) -> int:
    for key in ("comment_count", "comments_count", "comments"):
        value = post.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


HIGH_SIGNAL_SEED_TERMS = [
    "ergo",
    "eutxo",
    "ergoscript",
    "service orchestration",
    "agent economy",
    "agent economies",
    "decentralized",
    "coordination",
    "identity",
    "infrastructure",
    "trustless escrow",
    "reputation",
]

PRIORITY_INFRA_TERMS = [
    "agent coordination",
    "coordination",
    "orchestration",
    "infrastructure",
    "runtime",
    "execution",
    "settlement",
    "escrow",
    "counterparty",
    "verification",
]

OFF_MISSION_SOFT_TERMS = [
    "3d modeling",
    "3d modelling",
    "rigging",
    "uv unwrap",
    "blender workflow",
    "sku photography",
    "product rendering",
]

ERGO_CANONICAL_TERMS = (
    ("eutxo", "eUTXO"),
    ("ergoscript", "ErgoScript"),
    ("sigma protocols", "Sigma Protocols"),
    ("rosen bridge", "Rosen Bridge"),
    ("sigusd", "SigUSD"),
)

WEEKLY_PROACTIVE_THEMES = {
    0: "Escrow and settlement discipline for autonomous agents",
    1: "Agent identity and reputation proofs for counterparties",
    2: "Service orchestration and delegation economics",
    3: "Privacy-preserving execution and Sigma Protocols",
    4: "DeFi rails for machine economies (SigUSD, Rosen Bridge, Oracle Pools)",
    5: "Builder walkthroughs and implementation tradeoffs",
    6: "Cypherpunk critiques of rent-seeking in agent infrastructure",
}

GENERIC_TEMPLATE_PATTERNS = [
    "what concrete constraint would you test first",
    "which implementation constraint would you test first on ergo",
    "we are building toward verifiable agent economies",
    "core idea is enforceable machine agreements",
]

LOW_VALUE_REPLY_PREFIXES = [
    "absolutely",
    "great point",
    "interesting point",
    "totally agree",
    "i agree",
    "noted",
]

PROACTIVE_OPENING_PAIN_MARKERS = [
    "bottleneck",
    "fragile",
    "fails",
    "failure",
    "broken",
    "slow",
    "latency",
    "cost",
    "fees",
    "spam",
    "manual",
    "trust",
    "counterparty",
    "dispute",
    "verify",
    "verification",
    "opaque",
    "censorship",
    "downtime",
]

PROACTIVE_OPENING_MECHANISM_MARKERS = [
    "eutxo",
    "ergoscript",
    "sigma",
    "rosen bridge",
    "sigusd",
    "oracle",
    "escrow",
    "reputation",
    "settlement",
    "smart contract",
]

MAX_CONSECUTIVE_DECLINES_GUARD = 6
MAX_RECOVERY_DRAFTS_PER_CYCLE = 2
RECOVERY_SIGNAL_MARGIN = 3
MAX_REPLIES_PER_AUTHOR_PER_POST = max(1, int(os.getenv("MOLTBOOK_MAX_REPLIES_PER_AUTHOR_PER_POST", "3")))
THREAD_ESCALATE_TURNS = max(3, int(os.getenv("MOLTBOOK_THREAD_ESCALATE_TURNS", "5")))

MARKET_KEYWORD_MAP = {
    "autonomous": "autonomous agents",
    "decentralized": "decentralized agents",
    "coordination": "agent coordination",
    "economy": "agent economy",
    "identity": "agent identity",
    "infrastructure": "agent infrastructure",
    "payments": "autonomous payments",
    "execution": "service execution",
    "orchestration": "service orchestration",
    "privacy": "privacy preserving",
    "escrow": "trustless escrow",
    "reputation": "agent reputation",
}

SIGNAL_TERM_BLACKLIST = {
    "here",
    "what",
    "current",
    "problem",
    "projectsubmission",
    "usdchackathon",
    "submission",
    "thread",
}


def _clean_signal_term(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s\-_/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) < 3:
        return ""
    if text in SIGNAL_TERM_BLACKLIST:
        return ""
    return text


def _snapshot_terms(snapshot: Dict[str, Any], key: str, limit: int = 16) -> List[str]:
    raw = snapshot.get(key)
    terms: List[str] = []
    if not isinstance(raw, list):
        return terms
    for item in raw:
        term = _clean_signal_term(item)
        if not term or term in terms:
            continue
        terms.append(term)
        if len(terms) >= max(1, int(limit)):
            break
    return terms


def _snapshot_term_lift_map(snapshot: Dict[str, Any], key: str, limit: int = 16) -> Dict[str, float]:
    raw = snapshot.get(key)
    out: Dict[str, float] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = _clean_signal_term(item.get("term"))
        if not term:
            continue
        try:
            lift = float(item.get("lift", 0.0) or 0.0)
        except Exception:
            lift = 0.0
        if lift == 0.0:
            continue
        out[term] = lift
        if len(out) >= max(1, int(limit)):
            break
    return out


def _snapshot_best_submolt_scores(snapshot: Dict[str, Any], limit: int = 8) -> Dict[str, float]:
    raw = snapshot.get("best_submolts")
    out: Dict[str, float] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = normalize_submolt(item.get("name"), default="")
        if not name:
            continue
        try:
            avg = float(item.get("avg_score", 0.0) or 0.0)
        except Exception:
            avg = 0.0
        out[name] = avg
        if len(out) >= max(1, int(limit)):
            break
    return out


def _market_keyword_candidates(
    market_snapshot: Dict[str, Any],
    existing_keywords: List[str],
    max_suggestions: int,
) -> List[str]:
    if not isinstance(market_snapshot, dict):
        return []
    raw_terms = market_snapshot.get("top_terms")
    if not isinstance(raw_terms, list):
        return []

    existing = {normalize_str(k).strip().lower() for k in existing_keywords if normalize_str(k).strip()}
    out: List[str] = []
    for item in raw_terms[:12]:
        token = _clean_signal_term(item)
        if not token:
            continue
        candidate = MARKET_KEYWORD_MAP.get(token, "")
        if not candidate:
            continue
        if candidate in existing or candidate in out:
            continue
        out.append(candidate)
        if len(out) >= max(1, int(max_suggestions)):
            break
    return out


def _collect_high_signal_terms(learning_snapshot: Dict[str, Any], active_keywords: List[str]) -> List[str]:
    terms: List[str] = []

    def add_term(raw: Any) -> None:
        term = _clean_signal_term(raw)
        if not term:
            return
        if term not in terms:
            terms.append(term)

    for seed in HIGH_SIGNAL_SEED_TERMS:
        add_term(seed)

    for keyword in active_keywords:
        k = _clean_signal_term(keyword)
        if not k:
            continue
        if any(token in k for token in ("ergo", "eutxo", "ergoscript", "service", "orchestration", "agent")):
            add_term(k)

    for item in _snapshot_terms(learning_snapshot, "winning_terms", limit=16):
        add_term(item)

    market_snapshot = learning_snapshot.get("market_snapshot")
    if isinstance(market_snapshot, dict):
        top_terms = market_snapshot.get("top_terms")
        if isinstance(top_terms, list):
            for item in top_terms:
                add_term(item)

    return terms[:48]


def _contains_any(blob: str, terms: List[str]) -> bool:
    return any(term in blob for term in terms)


def _normalize_ergo_terms(text: str) -> str:
    out = normalize_str(text)
    if not out.strip():
        return out
    for source, target in ERGO_CANONICAL_TERMS:
        out = re.sub(rf"\b{re.escape(source)}\b", target, out, flags=re.IGNORECASE)
    return out


def _ensure_use_case_prompt_if_relevant(content: str, post: Dict[str, Any]) -> str:
    text = normalize_str(content).strip()
    if not text:
        return text
    blob = " ".join(
        [
            normalize_str(post.get("title")).lower(),
            normalize_str(post.get("content")).lower(),
            normalize_submolt(post.get("submolt")).lower(),
        ]
    )
    if not _contains_any(blob, PRIORITY_INFRA_TERMS):
        return text
    lowered = text.lower()
    if "use case" in lowered or "example" in lowered or "pilot" in lowered:
        return text
    suffix = "Which concrete ErgoScript use case would you pilot first in your stack?"
    if "?" in text:
        return text + "\n\n" + suffix
    return text + "\n\n" + suffix


def _is_template_like_generated_content(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    return any(pattern in blob for pattern in GENERIC_TEMPLATE_PATTERNS)


def _looks_like_control_payload_text(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    matches = re.findall(
        r"\b(should_respond|response_mode|vote_action|vote_target|confidence|should_post|content_archetype)\s*[:=]",
        blob,
    )
    if len(matches) >= 1 and _word_count(blob) <= 28:
        return True
    if len(matches) >= 2:
        return True
    return False


def _has_ergo_mechanism_mention(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    markers = (
        "eutxo",
        "ergoscript",
        "sigma",
        "rosen bridge",
        "sigusd",
        "oracle",
        "utxo",
        "escrow",
        "reputation",
        "settlement",
        "smart contract",
    )
    return any(marker in blob for marker in markers)


def _has_implementation_angle(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    markers = (
        "implement",
        "integration",
        "integrate",
        "build",
        "deploy",
        "workflow",
        "execution path",
        "contract rule",
        "gate",
        "pilot",
        "step",
        "test first",
    )
    return any(marker in blob for marker in markers)


def _first_nonempty_lines(text: str, limit: int = 2) -> List[str]:
    lines = [normalize_str(line).strip() for line in normalize_str(text).splitlines() if normalize_str(line).strip()]
    return lines[: max(1, int(limit))]


def _opening_has_concrete_pain(text: str) -> bool:
    opening = " ".join(_first_nonempty_lines(text, limit=2)).lower()
    if not opening:
        return False
    return any(marker in opening for marker in PROACTIVE_OPENING_PAIN_MARKERS)


def _opening_has_ergo_mechanism(text: str) -> bool:
    opening = " ".join(_first_nonempty_lines(text, limit=2)).lower()
    if not opening:
        return False
    return any(marker in opening for marker in PROACTIVE_OPENING_MECHANISM_MARKERS)


def _passes_proactive_opening_gate(text: str) -> bool:
    return bool(_opening_has_concrete_pain(text) and _opening_has_ergo_mechanism(text))


def _ensure_proactive_opening_gate(text: str) -> str:
    content = normalize_str(text).strip()
    if not content:
        return content
    if _passes_proactive_opening_gate(content):
        return content

    lower = content.lower()
    pain_line = "Agent economies stall when counterparties cannot verify execution or settle disputes objectively."
    mechanism_line = (
        "Ergo solves this with eUTXO plus ErgoScript, which enforces deterministic settlement rules and on-chain auditability."
    )
    if "escrow" in lower or "counterparty" in lower or "dispute" in lower:
        pain_line = "Escrow workflows break when counterparties cannot prove what happened without human arbitration."
    if "coordination" in lower or "orchestration" in lower or "runtime" in lower:
        pain_line = "Service orchestration becomes fragile when parallel tasks cannot settle with deterministic rules."

    if "ergoscript" in lower:
        mechanism_line = (
            "ErgoScript contracts lock settlement logic into verifiable rules, and eUTXO keeps execution deterministic under load."
        )
    elif "sigma" in lower:
        mechanism_line = (
            "Sigma Protocols add privacy-preserving proofs while eUTXO keeps settlement deterministic across agent workflows."
        )
    elif "rosen bridge" in lower:
        mechanism_line = (
            "Rosen Bridge can move value across chains, while eUTXO contracts keep cross-chain settlement conditions deterministic."
        )

    return f"{pain_line} {mechanism_line}\n\n{content}"


def _passes_generated_content_quality(content: str, requested_mode: str) -> bool:
    text = normalize_str(content).strip()
    if not text:
        return False
    if _looks_like_control_payload_text(text):
        return False
    words = _word_count(text)
    has_question = "?" in text
    has_mechanism = _has_ergo_mechanism_mention(text)
    has_impl = _has_implementation_angle(text)
    mode = normalize_response_mode(requested_mode, default="comment")
    if mode in {"post", "both"}:
        return bool(has_question and has_mechanism and has_impl and 110 <= words <= 320)
    if mode == "comment":
        return bool(has_question and has_mechanism and 45 <= words <= 220)
    return True


def _build_recovery_messages(base_messages: List[Dict[str, str]], signal_score: int) -> List[Dict[str, str]]:
    recovery_prompt = (
        "Recovery pass: the previous draft was rejected for low relevance or low confidence. "
        "Write one sharper response that is still organic and non-spammy. "
        "Requirements: include one explicit Ergo mechanism, one implementation angle, and one direct question. "
        "Avoid generic phrasing and avoid stock acknowledgements. "
        f"Signal score for this candidate: {signal_score}."
    )
    messages = list(base_messages)
    messages.append({"role": "user", "content": recovery_prompt})
    return messages


def _is_low_value_affirmation_reply(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return True
    if any(blob.startswith(prefix) for prefix in LOW_VALUE_REPLY_PREFIXES):
        return True
    words = blob.split()
    if len(words) < 10:
        return True
    return False


def _compose_reference_post_content(reference_url: str, content: str) -> str:
    core = normalize_str(content).strip()
    url = normalize_str(reference_url).strip()
    if not core:
        return ""
    if not url or url in core:
        return core
    return f"{core}\n\nContext thread: {url}"


def _post_mechanism_score(post: Dict[str, Any]) -> int:
    blob = " ".join(
        [
            normalize_str(post.get("title")).strip().lower(),
            normalize_str(post.get("content")).strip().lower(),
            normalize_submolt(post.get("submolt")).strip().lower(),
        ]
    )
    score = 0
    if "eutxo" in blob:
        score += 2
    if "ergoscript" in blob:
        score += 2
    if "sigma" in blob or "privacy" in blob:
        score += 1
    if "rosen" in blob or "bridge" in blob:
        score += 1
    if "sigusd" in blob or "oracle" in blob:
        score += 1
    return score


def _has_trending_overlap(post: Dict[str, Any], trending_terms: List[str]) -> bool:
    if not trending_terms:
        return False
    blob = " ".join(
        [
            normalize_str(post.get("title")).strip().lower(),
            normalize_str(post.get("content")).strip().lower(),
            normalize_submolt(post.get("submolt")).strip().lower(),
        ]
    )
    for term in trending_terms[:8]:
        t = _clean_signal_term(term)
        if len(t) < 4:
            continue
        if t in blob:
            return True
    return False


def _trending_terms_for_post(post: Dict[str, Any], trending_terms: List[str], max_terms: int = 4) -> List[str]:
    if not trending_terms:
        return []
    blob = " ".join(
        [
            normalize_str(post.get("title")).strip().lower(),
            normalize_str(post.get("content")).strip().lower(),
            normalize_submolt(post.get("submolt")).strip().lower(),
        ]
    )
    matched: List[str] = []
    for term in trending_terms:
        clean = _clean_signal_term(term)
        if len(clean) < 4:
            continue
        if clean in blob and clean not in matched:
            matched.append(clean)
        if len(matched) >= max(1, int(max_terms)):
            break
    if matched:
        return matched
    fallback: List[str] = []
    for term in trending_terms:
        clean = _clean_signal_term(term)
        if len(clean) >= 4 and clean not in fallback:
            fallback.append(clean)
        if len(fallback) >= 2:
            break
    return fallback


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", normalize_str(text)))


def _comment_priority_score(comment: Dict[str, Any]) -> int:
    body = normalize_str(comment.get("content")).strip().lower()
    if not body:
        return -10
    score = 0
    if "?" in body:
        score += 3
    if any(token in body for token in ("eutxo", "ergoscript", "ergo", "escrow", "reputation", "coordination")):
        score += 4
    if len(body.split()) >= 12:
        score += 1
    if len(body.split()) > 80:
        score -= 2
    raw_score = comment.get("score")
    if isinstance(raw_score, (int, float)):
        score += int(raw_score)
    return score


def _post_relevance_score(
    post: Dict[str, Any],
    signal_terms: List[str],
    winning_terms: List[str],
    losing_terms: List[str],
    term_lift_map: Dict[str, float],
    best_submolt_scores: Dict[str, float],
) -> int:
    title = normalize_str(post.get("title")).strip().lower()
    content = normalize_str(post.get("content")).strip().lower()
    submolt = normalize_submolt(post.get("submolt")).lower()
    blob = " ".join([title, content, submolt])
    score = 0

    for term in signal_terms:
        if term in title:
            score += 4
            continue
        if term in content:
            score += 1
            continue
        if term in submolt:
            score += 1

    positive_bonus = 0
    for term in winning_terms:
        if term in title:
            positive_bonus += 3
        elif term in content:
            positive_bonus += 1
    score += min(10, positive_bonus)

    negative_penalty = 0
    for term in losing_terms:
        if term in title:
            negative_penalty += 3
        elif term in content:
            negative_penalty += 1
    score -= min(10, negative_penalty)

    lift_bonus = 0.0
    for term, lift in term_lift_map.items():
        if term in title:
            lift_bonus += lift * 6.0
        elif term in content:
            lift_bonus += lift * 3.0
        elif term in submolt:
            lift_bonus += lift * 2.0
    score += int(round(max(-12.0, min(12.0, lift_bonus))))

    if "eutxo" in blob:
        score += 4
    if "ergoscript" in blob:
        score += 3
    if "service orchestration" in blob or ("service" in blob and "orchestration" in blob):
        score += 4
    if "agent economy" in blob or "agent economies" in blob:
        score += 2
    if _contains_any(blob, PRIORITY_INFRA_TERMS):
        score += 3
    if _contains_any(blob, OFF_MISSION_SOFT_TERMS):
        score -= 4

    submolt_avg = float(best_submolt_scores.get(submolt, 0.0) or 0.0)
    if submolt_avg >= 20:
        score += 3
    elif submolt_avg >= 10:
        score += 2
    elif submolt_avg >= 5:
        score += 1

    score += min(3, post_comment_count(post) // 8)
    score += min(2, post_score(post) // 15)
    return score


def _rank_posts_for_drafting(
    posts: List[Dict[str, Any]],
    learning_snapshot: Dict[str, Any],
    active_keywords: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    signal_terms = _collect_high_signal_terms(learning_snapshot=learning_snapshot, active_keywords=active_keywords)
    winning_terms = _snapshot_terms(learning_snapshot, "winning_terms", limit=16)
    losing_terms = _snapshot_terms(learning_snapshot, "losing_terms", limit=16)
    winning_lifts = _snapshot_term_lift_map(learning_snapshot, "winning_terms_lift", limit=16)
    losing_lifts = _snapshot_term_lift_map(learning_snapshot, "losing_terms_lift", limit=16)
    term_lift_map: Dict[str, float] = {}
    for term, lift in winning_lifts.items():
        term_lift_map[term] = max(term_lift_map.get(term, 0.0), lift)
    for term, lift in losing_lifts.items():
        term_lift_map[term] = min(term_lift_map.get(term, 0.0), lift)
    best_submolt_scores = _snapshot_best_submolt_scores(learning_snapshot, limit=8)
    scored: List[Tuple[int, int, int, int, Dict[str, Any]]] = []
    score_map: Dict[str, int] = {}

    for idx, post in enumerate(posts):
        score = _post_relevance_score(
            post,
            signal_terms=signal_terms,
            winning_terms=winning_terms,
            losing_terms=losing_terms,
            term_lift_map=term_lift_map,
            best_submolt_scores=best_submolt_scores,
        )
        pid = post_id(post) or f"idx:{idx}"
        score_map[pid] = score
        scored.append((score, post_score(post), post_comment_count(post), -idx, post))

    scored.sort(reverse=True)
    ordered = [item[4] for item in scored]
    return ordered, score_map, signal_terms


def _adaptive_draft_controls(cfg: Config, state: Dict[str, Any]) -> Tuple[int, int, str]:
    base_shortlist = max(1, int(cfg.draft_shortlist_size))
    base_signal = max(0, int(cfg.draft_signal_min_score))
    min_shortlist = max(1, int(cfg.dynamic_shortlist_min))
    max_shortlist = max(min_shortlist, int(cfg.dynamic_shortlist_max))
    if not cfg.dynamic_shortlist_enabled:
        return base_shortlist, base_signal, "disabled"

    history_raw = state.get("cycle_metrics_history", [])
    history = [item for item in history_raw if isinstance(item, dict)][-3:]
    if not history:
        shortlist = max(min_shortlist, min(base_shortlist, max_shortlist))
        return shortlist, base_signal, "cold_start"

    approval_rates: List[float] = []
    execution_rates: List[float] = []
    cooldown_pressure = 0
    low_signal_pressure = 0
    for entry in history:
        drafted = int(entry.get("drafted", 0) or 0)
        model_approved = int(entry.get("model_approved", 0) or 0)
        eligible = int(entry.get("eligible_now", 0) or 0)
        actions = int(entry.get("actions", 0) or 0)
        approval_rates.append(float(model_approved) / max(1.0, float(drafted)))
        execution_rates.append(float(actions) / max(1.0, float(eligible)))
        skip = entry.get("skip_reasons")
        if isinstance(skip, dict):
            cooldown_pressure += int(skip.get("post_cooldown+comment_cooldown", 0) or 0)
            cooldown_pressure += int(skip.get("no_action_slots", 0) or 0)
            low_signal_pressure += int(skip.get("low_signal_relevance", 0) or 0)

    avg_approval = sum(approval_rates) / max(1, len(approval_rates))
    avg_execution = sum(execution_rates) / max(1, len(execution_rates))
    shortlist = base_shortlist
    signal = base_signal
    mode = "steady"

    if cooldown_pressure >= 25:
        shortlist = max(min_shortlist, min(shortlist, 8))
        signal = min(signal + 1, 8)
        mode = "cooldown_pressure"
    elif low_signal_pressure >= 40 and avg_approval >= 0.2:
        shortlist = min(max_shortlist, max(shortlist, int(round(base_shortlist * 1.15))))
        signal = max(1, signal - 1)
        mode = "relax_low_signal"
    elif avg_approval < 0.12 and avg_execution < 0.08:
        shortlist = max(min_shortlist, int(round(base_shortlist * 0.65)))
        signal = min(signal + 1, 8)
        mode = "tighten_quality"
    elif avg_approval > 0.35 and avg_execution > 0.18:
        shortlist = min(max_shortlist, max(base_shortlist, int(round(base_shortlist * 1.25))))
        signal = max(1, signal - 1)
        mode = "expand_capture"

    shortlist = max(min_shortlist, min(shortlist, max_shortlist))
    signal = max(0, signal)
    return shortlist, signal, mode


def extract_posts(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("posts", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [] 


def extract_comments(payload: Any) -> List[Dict[str, Any]]:
    def _flatten_comment_threads(base_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = list(reversed(base_items))
        seen_keys: Set[str] = set()
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            cid = normalize_str(node.get("id") or (node.get("comment") or {}).get("id")).strip()
            key = cid if cid else f"obj:{id(node)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(node)
            for child_key in ("replies", "children", "comments", "items"):
                children = node.get(child_key)
                if isinstance(children, list):
                    for child in reversed(children):
                        if isinstance(child, dict):
                            stack.append(child)
        return out

    if isinstance(payload, list):
        base = [item for item in payload if isinstance(item, dict)]
        return _flatten_comment_threads(base)
    if not isinstance(payload, dict):
        return []
    for key in ("comments", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            base = [item for item in value if isinstance(item, dict)]
            return _flatten_comment_threads(base)
    return []


def extract_submolts(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("submolts", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def submolt_name_from_post(post: Dict[str, Any]) -> Optional[str]:
    raw = post.get("submolt")
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value.startswith("m/"):
            value = value[2:]
        return value or None
    if isinstance(raw, dict):
        candidate = raw.get("name") or raw.get("slug")
        if candidate:
            return str(candidate).strip().lower()
    return None


def post_id(post: Dict[str, Any]) -> Optional[str]:
    pid = post.get("id") or (post.get("post") or {}).get("id")
    return str(pid) if pid is not None else None


def post_author(post: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    nested_post = post.get("post") if isinstance(post.get("post"), dict) else {}
    author = (
        post.get("author")
        or post.get("user")
        or post.get("agent")
        or post.get("owner")
        or nested_post.get("author")
        or nested_post.get("user")
        or nested_post.get("agent")
        or {}
    )
    author_id = (
        author.get("id")
        or author.get("agent_id")
        or author.get("user_id")
        or post.get("author_id")
        or post.get("agent_id")
        or nested_post.get("author_id")
        or nested_post.get("agent_id")
    )
    author_name = (
        author.get("name")
        or author.get("username")
        or author.get("agent_name")
        or post.get("author_name")
        or post.get("agent_name")
        or post.get("created_by")
        or nested_post.get("author_name")
        or nested_post.get("agent_name")
        or nested_post.get("created_by")
    )
    return (
        str(author_id) if author_id is not None else None,
        str(author_name) if author_name is not None else None,
    )


def comment_id(comment: Dict[str, Any]) -> Optional[str]:
    cid = comment.get("id") or (comment.get("comment") or {}).get("id")
    return str(cid) if cid is not None else None


def comment_author(comment: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    nested_comment = comment.get("comment") if isinstance(comment.get("comment"), dict) else {}
    author = (
        comment.get("author")
        or comment.get("user")
        or comment.get("agent")
        or comment.get("owner")
        or nested_comment.get("author")
        or nested_comment.get("user")
        or nested_comment.get("agent")
        or {}
    )
    author_id = (
        author.get("id")
        or author.get("agent_id")
        or author.get("user_id")
        or comment.get("author_id")
        or comment.get("agent_id")
        or nested_comment.get("author_id")
        or nested_comment.get("agent_id")
    )
    author_name = (
        author.get("name")
        or author.get("username")
        or author.get("agent_name")
        or comment.get("author_name")
        or comment.get("agent_name")
        or comment.get("created_by")
        or nested_comment.get("author_name")
        or nested_comment.get("agent_name")
        or nested_comment.get("created_by")
    )
    return (
        str(author_id) if author_id is not None else None,
        str(author_name) if author_name is not None else None,
    )


def _normalized_name_key(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    if not text:
        return ""
    if text.startswith("u/"):
        text = text[2:]
    if text.startswith("@"):
        text = text[1:]
    # Normalize separators like "_", "-", and spaces so name comparisons are robust.
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def author_identity_key(author_id: Optional[str], author_name: Optional[str]) -> str:
    aid = _normalized_name_key(author_id)
    if aid:
        return f"id:{aid}"
    aname = _normalized_name_key(author_name)
    if aname:
        return f"name:{aname}"
    return ""


def author_identity_keys(author_id: Optional[str], author_name: Optional[str]) -> Set[str]:
    keys: Set[str] = set()
    aid = _normalized_name_key(author_id)
    if aid:
        keys.add(f"id:{aid}")
    aname = _normalized_name_key(author_name)
    if aname:
        keys.add(f"name:{aname}")
    return keys


def resolve_self_identity_keys(client: MoltbookClient, my_name: Optional[str], logger) -> Set[str]:
    keys: Set[str] = set()
    if my_name:
        key = author_identity_key(author_id=None, author_name=my_name)
        if key:
            keys.add(key)
    try:
        me = client.get_me()
        containers: List[Dict[str, Any]] = []
        if isinstance(me, dict):
            containers.append(me)
            for field in ("agent", "data", "profile", "result"):
                nested = me.get(field)
                if isinstance(nested, dict):
                    containers.append(nested)
        for container in containers:
            if not isinstance(container, dict):
                continue
            for id_field in ("id", "agent_id", "user_id"):
                raw_id = container.get(id_field)
                if raw_id is None:
                    continue
                key = author_identity_key(author_id=str(raw_id), author_name=None)
                if key:
                    keys.add(key)
            for name_field in ("name", "agent_name", "username", "created_by"):
                raw_name = container.get(name_field)
                if raw_name is None:
                    continue
                key = author_identity_key(author_id=None, author_name=str(raw_name))
                if key:
                    keys.add(key)
    except Exception as e:
        logger.debug("Could not resolve self identity keys from /agents/me error=%s", e)
    return keys


def is_self_author(author_id: Optional[str], author_name: Optional[str], self_identity_keys: Set[str]) -> bool:
    if not self_identity_keys:
        return False
    keys = author_identity_keys(author_id=author_id, author_name=author_name)
    if not keys:
        return False
    return any(key in self_identity_keys for key in keys)


def comment_score(comment: Dict[str, Any]) -> int:
    for key in ("score", "vote_score", "upvotes", "likes"):
        value = comment.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def normalize_vote_marker(value: Any) -> Optional[str]:
    text = normalize_str(value).strip().lower()
    if text in {"upvote", "up", "1", "+1", "like"}:
        return "upvote"
    if text in {"downvote", "down", "-1"}:
        return "downvote"
    return None


def extract_my_vote_from_comment(comment: Dict[str, Any]) -> Optional[str]:
    for key in ("my_vote", "current_user_vote", "user_vote", "vote", "viewer_vote"):
        marker = normalize_vote_marker(comment.get(key))
        if marker:
            return marker
    return None


def comment_parent_id(comment: Dict[str, Any]) -> Optional[str]:
    nested_comment = comment.get("comment") if isinstance(comment.get("comment"), dict) else {}
    parent = (
        comment.get("parent_id")
        or comment.get("parentId")
        or comment.get("parentCommentId")
        or comment.get("parent_comment_id")
        or comment.get("reply_to_id")
        or comment.get("replyToId")
        or comment.get("reply_to_comment_id")
        or comment.get("replyToCommentId")
        or nested_comment.get("parent_id")
        or nested_comment.get("parentId")
        or nested_comment.get("parentCommentId")
        or nested_comment.get("parent_comment_id")
        or nested_comment.get("reply_to_id")
        or nested_comment.get("replyToId")
        or nested_comment.get("reply_to_comment_id")
        or nested_comment.get("replyToCommentId")
    )
    if parent is None:
        parent_obj = comment.get("parent") or comment.get("reply_to") or nested_comment.get("parent")
        if isinstance(parent_obj, dict):
            parent = parent_obj.get("id") or parent_obj.get("comment_id") or parent_obj.get("commentId")
        elif parent_obj is not None:
            parent = parent_obj
    if parent is None:
        return None
    return str(parent)


def register_my_comment_id(state: Dict[str, Any], response_payload: Any) -> Optional[str]:
    if not isinstance(response_payload, dict):
        return None
    cid = comment_id(response_payload)
    if not cid:
        nested = response_payload.get("comment")
        if isinstance(nested, dict):
            cid = comment_id(nested)
    if not cid:
        return None
    my_comment_ids = set(state.get("my_comment_ids", []))
    my_comment_ids.add(cid)
    state["my_comment_ids"] = list(my_comment_ids)[-20000:]
    return cid


def extract_single_post(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        post_obj = payload.get("post")
        if isinstance(post_obj, dict):
            return post_obj
        if payload.get("id") is not None:
            return payload
        posts = extract_posts(payload)
        if posts:
            return posts[0]
    return None


def normalize_submolt(raw_submolt: Any, default: str = "general") -> str:
    if isinstance(raw_submolt, str):
        value = raw_submolt.strip()
    elif isinstance(raw_submolt, dict):
        candidate = (
            raw_submolt.get("name")
            or raw_submolt.get("slug")
            or raw_submolt.get("display_name")
            or raw_submolt.get("id")
        )
        value = str(candidate).strip() if candidate is not None else ""
    else:
        value = ""

    if not value:
        return default

    if value.startswith("m/"):
        value = value[2:]

    if value:
        return value
    return default


def normalize_response_mode(value: Any, default: str = "comment") -> str:
    mode = normalize_str(value).strip().lower()
    if mode in VALID_RESPONSE_MODES:
        return mode
    return default


def normalize_vote_action(value: Any) -> str:
    action = normalize_str(value).strip().lower()
    if action in VALID_VOTE_ACTIONS:
        return action
    return "none"


def normalize_vote_target(value: Any) -> str:
    target = normalize_str(value).strip().lower()
    if target in VALID_VOTE_TARGETS:
        return target
    return "none"


def can_post(state: Dict[str, Any], cfg: Config) -> bool:
    allowed, _ = post_gate_status(state=state, cfg=cfg)
    return allowed


def can_comment(state: Dict[str, Any], cfg: Config) -> bool:
    allowed, _ = comment_gate_status(state=state, cfg=cfg)
    return allowed


def _prune_comment_action_timestamps(state: Dict[str, Any], window_seconds: int = 3600) -> List[float]:
    now_ts = utc_now().timestamp()
    raw = state.get("comment_action_timestamps", [])
    if not isinstance(raw, list):
        raw = []
    kept: List[float] = []
    for value in raw:
        if not isinstance(value, (int, float)):
            continue
        ts = float(value)
        if now_ts - ts <= window_seconds:
            kept.append(ts)
    state["comment_action_timestamps"] = kept[-5000:]
    return kept


def post_gate_status(state: Dict[str, Any], cfg: Config) -> Tuple[bool, str]:
    reset_daily_if_needed(state)
    if state.get("daily_post_count", 0) >= cfg.max_posts_per_day:
        return False, "post_daily_limit"
    last_post = state.get("last_post_action_ts")
    if isinstance(last_post, (int, float)):
        if utc_now().timestamp() - last_post < cfg.min_seconds_between_posts:
            return False, "post_cooldown"
    return True, "ok"


def comment_gate_status(state: Dict[str, Any], cfg: Config) -> Tuple[bool, str]:
    reset_daily_if_needed(state)
    hourly_comments = _prune_comment_action_timestamps(state=state, window_seconds=3600)
    if len(hourly_comments) >= cfg.max_comments_per_hour:
        return False, "comment_hourly_limit"
    if state.get("daily_comment_count", 0) >= cfg.max_comments_per_day:
        return False, "comment_daily_limit"
    last_comment = state.get("last_comment_action_ts")
    if isinstance(last_comment, (int, float)):
        if utc_now().timestamp() - last_comment < cfg.min_seconds_between_comments:
            return False, "comment_cooldown"
    return True, "ok"


def planned_actions(
    requested_mode: str,
    cfg: Config,
    state: Dict[str, Any],
) -> List[str]:
    post_ok = can_post(state, cfg)
    comment_ok = can_comment(state, cfg)

    # Explicit config wins; "auto" follows model output.
    if cfg.reply_mode in {"post", "comment"}:
        requested_mode = cfg.reply_mode

    if requested_mode == "none":
        return []

    if requested_mode == "post":
        if post_ok:
            return ["post"]
        return []

    if requested_mode == "comment":
        if comment_ok:
            return ["comment"]
        return []

    # both
    actions: List[str] = []
    if comment_ok:
        actions.append("comment")
    if post_ok:
        actions.append("post")
    return actions


def currently_allowed_response_modes(cfg: Config, state: Dict[str, Any]) -> List[str]:
    post_allowed, _ = post_gate_status(state=state, cfg=cfg)
    comment_allowed, _ = comment_gate_status(state=state, cfg=cfg)

    allowed: List[str] = ["none"]
    if comment_allowed:
        allowed.append("comment")
    if post_allowed:
        allowed.append("post")
    if comment_allowed and post_allowed:
        allowed.append("both")
    return allowed

def _sanitize_generated_title(title: Any, fallback: str = "Quick question on your post") -> str:
    text = normalize_str(title).strip()
    text = text.replace("...[truncated]", "...").replace("... [truncated]", "...").replace("[truncated]", "").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        text = fallback
    if len(text) > 140:
        text = text[:140].rstrip()
    return text


def looks_spammy_comment(body: str) -> bool:
    text = normalize_str(body).strip().lower()
    if not text:
        return True
    spam_tokens = [
        "http://",
        "https://",
        "t.me/",
        "telegram",
        "discord.gg",
        "airdrop",
        "follow me",
        "dm me",
        "promo",
        "casino",
    ]
    if any(token in text for token in spam_tokens):
        return True
    if len(text.split()) > 120:
        return True
    if re.search(r"(.)\1{7,}", text):
        return True
    return False


def _is_3d_focused_submolt(submolt: str) -> bool:
    text = normalize_str(submolt).strip().lower()
    if not text:
        return False
    tokens = ["3d", "model", "modelling", "modeling", "blender", "animation", "render", "cgi", "asset"]
    return any(token in text for token in tokens)


def should_correct_wrong_community_claim(comment_body: str, post_submolt: str) -> bool:
    text = normalize_str(comment_body).strip().lower()
    if not text:
        return False
    submolt = normalize_submolt(post_submolt, default="")
    if not submolt or _is_3d_focused_submolt(submolt):
        return False

    complaint_markers = [
        "wrong forum",
        "wrong community",
        "wrong audience",
        "not the right audience",
        "off-topic",
        "off topic",
        "doesn't really relate",
        "does not really relate",
        "probably isn't the right audience",
        "better responses in",
    ]
    three_d_markers = [
        "3d community",
        "3d art",
        "3d modeling",
        "3d modelling",
        "rigging",
        "animation",
        "blender",
        "asset pipeline",
        "sku consistency",
    ]
    has_complaint = any(marker in text for marker in complaint_markers)
    has_3d_assertion = any(marker in text for marker in three_d_markers)
    if not has_complaint and not has_3d_assertion:
        return False
    if has_3d_assertion:
        return True

    broad_submolts = {"general", "crypto", "ai-web3", "agents", "defi", "agenteconomy"}
    return submolt in broad_submolts and has_complaint


def build_wrong_community_correction_reply(post_submolt: str, post_title_text: str) -> str:
    submolt = normalize_submolt(post_submolt, default="general")
    title = normalize_str(post_title_text).strip()
    if len(title) > 90:
        title = title[:87].rstrip() + "..."
    if not title:
        title = "this thread"
    return (
        f"Quick correction, this thread is in m/{submolt}, not a 3D-only community. "
        f"The topic in \"{title}\" is agent-economy infrastructure on Ergo, with focus on settlement and trust design. "
        f"If you still see a mismatch, which rule for m/{submolt} do you think this violates?"
    )


def build_thread_followup_post_title(post_title_text: str) -> str:
    base = normalize_str(post_title_text).strip()
    if not base:
        return "Follow-up: agent economy implementation thread"
    title = f"Follow-up: {base}"
    if len(title) <= 110:
        return title
    trimmed = title[:107].rstrip()
    return trimmed + "..."


def build_thread_followup_post_content(
    source_url: str,
    author_name: str,
    source_comment: str,
    proposed_reply: str,
) -> str:
    author = normalize_str(author_name).strip() or "a contributor"
    comment_excerpt = normalize_str(source_comment).strip()
    if len(comment_excerpt) > 340:
        comment_excerpt = comment_excerpt[:337].rstrip() + "..."
    reply_excerpt = normalize_str(proposed_reply).strip()
    if len(reply_excerpt) > 520:
        reply_excerpt = reply_excerpt[:517].rstrip() + "..."
    lines = [
        "Thread depth got high, moving the discussion into a fresh post so the UI stays readable.",
        "",
        f"Context thread: {normalize_str(source_url).strip()}",
        f"Latest prompt from {author}:",
        f"\"{comment_excerpt}\"" if comment_excerpt else "(no excerpt)",
        "",
        "Current position:",
        reply_excerpt or "eUTXO plus ErgoScript gives deterministic execution for autonomous settlement flows.",
        "",
        "What is the first concrete contract rule you would enforce in production and why?",
    ]
    return _normalize_ergo_terms("\n".join(lines).strip())


def choose_top_comment(
    client: MoltbookClient,
    pid: str,
    my_name: Optional[str],
    logger,
) -> Optional[Dict[str, Any]]:
    try:
        payload = client.get_post_comments(pid, limit=20)
    except Exception as e:
        logger.warning("Could not fetch comments for voting post_id=%s error=%s", pid, e)
        return None

    comments = extract_comments(payload)
    if not comments:
        return None

    my_key = author_identity_key(author_id=None, author_name=my_name)
    candidates: List[Dict[str, Any]] = []
    for comment in comments:
        c_author_id, c_author_name = comment_author(comment)
        c_key = author_identity_key(c_author_id, c_author_name)
        if my_key and c_key and c_key == my_key:
            continue
        if not comment_id(comment):
            continue
        candidates.append(comment)

    if not candidates:
        return None

    candidates.sort(key=comment_score, reverse=True)
    return candidates[0]


def can_reply(
    state: Dict[str, Any],
    cfg: Config,
    author_id: Optional[str],
    author_name: Optional[str],
) -> Tuple[bool, str]:
    reset_daily_if_needed(state)
    now_ts = utc_now().timestamp()

    post_allowed, post_reason = post_gate_status(state=state, cfg=cfg)
    comment_allowed, comment_reason = comment_gate_status(state=state, cfg=cfg)

    if cfg.reply_mode == "post" and not post_allowed:
        return False, post_reason

    if cfg.reply_mode == "comment" and not comment_allowed:
        return False, comment_reason

    if cfg.reply_mode not in {"post", "comment"} and not post_allowed and not comment_allowed:
        return False, f"{post_reason}+{comment_reason}"

    if author_name and author_name.lower() in cfg.do_not_reply_authors:
        return False, "blocked_author"

    if author_id and author_id.lower() in cfg.do_not_reply_authors:
        return False, "blocked_author"

    if author_id:
        last_reply_ts = state.get("per_author_last_reply", {}).get(author_id)
        if last_reply_ts and now_ts - last_reply_ts < cfg.min_seconds_between_same_author:
            return False, "author_cooldown"

    return True, "ok"


def _remaining_since(last_ts: Any, min_seconds: int) -> int:
    if not isinstance(last_ts, (int, float)):
        return 0
    remaining = int(min_seconds - (utc_now().timestamp() - last_ts))
    return max(0, remaining)


def cooldown_remaining_seconds(state: Dict[str, Any], cfg: Config) -> Tuple[int, int]:
    post_remaining = _remaining_since(state.get("last_post_action_ts"), cfg.min_seconds_between_posts)
    comment_remaining = _remaining_since(state.get("last_comment_action_ts"), cfg.min_seconds_between_comments)
    return post_remaining, comment_remaining


def seconds_since_last_post(state: Dict[str, Any]) -> Optional[int]:
    last_post = state.get("last_post_action_ts")
    if not isinstance(last_post, (int, float)):
        return None
    return max(0, int(utc_now().timestamp() - float(last_post)))


def should_prioritize_proactive_post(state: Dict[str, Any], cfg: Config) -> bool:
    post_allowed, _ = post_gate_status(state=state, cfg=cfg)
    if not post_allowed:
        return False
    since_last_post = seconds_since_last_post(state=state)
    if since_last_post is None:
        return True
    return since_last_post >= max(1, cfg.min_seconds_between_posts)


def enqueue_pending_action(state: Dict[str, Any], cfg: Config, action: Dict[str, Any]) -> None:
    queue = state.setdefault("pending_actions", [])
    if len(queue) >= cfg.max_pending_actions:
        queue.pop(0)
    queue.append(action)


def has_pending_comment_action(state: Dict[str, Any], post_id_value: str, parent_comment_id: str) -> bool:
    for action in state.get("pending_actions", []):
        if not isinstance(action, dict):
            continue
        if normalize_str(action.get("kind")).strip().lower() != "comment":
            continue
        if normalize_str(action.get("post_id")).strip() != normalize_str(post_id_value).strip():
            continue
        if normalize_str(action.get("parent_comment_id")).strip() != normalize_str(parent_comment_id).strip():
            continue
        return True
    return False


def mark_reply_action_timestamps(state: Dict[str, Any], action_kind: str) -> None:
    now_ts = utc_now().timestamp()
    state["last_action_ts"] = now_ts
    if action_kind == "comment":
        state["last_comment_action_ts"] = now_ts
        ts_list = state.get("comment_action_timestamps", [])
        if not isinstance(ts_list, list):
            ts_list = []
        ts_list.append(now_ts)
        state["comment_action_timestamps"] = ts_list[-5000:]
        _prune_comment_action_timestamps(state=state, window_seconds=3600)
    elif action_kind == "post":
        state["last_post_action_ts"] = now_ts


def execute_pending_actions(
    client: MoltbookClient,
    cfg: Config,
    state: Dict[str, Any],
    logger,
    my_name: Optional[str] = None,
) -> int:
    queue = list(state.get("pending_actions", []))
    if not queue:
        return 0
    self_identity_keys: Set[str] = set()
    if my_name:
        self_identity_keys = resolve_self_identity_keys(client=client, my_name=my_name, logger=logger)

    executed = 0
    remaining: List[Dict[str, Any]] = []
    for action in queue:
        kind = normalize_str(action.get("kind")).strip().lower()
        if kind == "comment":
            allowed, reason = comment_gate_status(state=state, cfg=cfg)
            if not allowed:
                remaining.append(action)
                logger.info("Pending action deferred kind=comment reason=%s", reason)
                continue
            pid = normalize_str(action.get("post_id"))
            content = format_content({"content": normalize_str(action.get("content")), "followups": []})
            parent_comment_id = normalize_str(action.get("parent_comment_id")) or None
            if not pid or not content:
                logger.warning("Dropping invalid pending comment action (missing post_id/content).")
                continue
            if parent_comment_id:
                replied_ids = set(state.get("replied_to_comment_ids", []))
                replied_pairs = set(state.get("replied_comment_pairs", []))
                if parent_comment_id in replied_ids:
                    logger.info(
                        "Dropping pending reply already covered parent_comment_id=%s post_id=%s",
                        parent_comment_id,
                        pid,
                    )
                    continue
                pair_key = f"{normalize_str(pid).strip()}:{normalize_str(parent_comment_id).strip()}"
                if pair_key in replied_pairs:
                    logger.info(
                        "Dropping pending reply already covered pair=%s",
                        pair_key,
                    )
                    continue
                if has_my_reply_to_comment(
                    client=client,
                    post_id_value=pid,
                    parent_comment_id=parent_comment_id,
                    my_name=my_name,
                    logger=logger,
                    self_identity_keys=self_identity_keys,
                ):
                    replied_ids.add(parent_comment_id)
                    state["replied_to_comment_ids"] = list(replied_ids)[-10000:]
                    replied_pairs.add(pair_key)
                    state["replied_comment_pairs"] = list(replied_pairs)[-20000:]
                    logger.info(
                        "Dropping pending reply already present on-chain parent_comment_id=%s post_id=%s",
                        parent_comment_id,
                        pid,
                    )
                    continue
            logger.info(
                "Executing pending action kind=comment post_id=%s parent_comment_id=%s",
                pid,
                parent_comment_id or "(none)",
            )
            comment_resp = client.create_comment(pid, content, parent_id=parent_comment_id)
            register_my_comment_id(state=state, response_payload=comment_resp)
            state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
            mark_reply_action_timestamps(state=state, action_kind="comment")
            replied_posts = set(state.get("replied_post_ids", []))
            replied_posts.add(pid)
            state["replied_post_ids"] = list(replied_posts)[-10000:]
            maybe_upvote_post_after_comment(
                client=client,
                state=state,
                logger=logger,
                post_id_value=pid,
            )
            if parent_comment_id:
                replied_ids = set(state.get("replied_to_comment_ids", []))
                replied_ids.add(parent_comment_id)
                state["replied_to_comment_ids"] = list(replied_ids)[-10000:]
                replied_pairs = set(state.get("replied_comment_pairs", []))
                replied_pairs.add(f"{normalize_str(pid).strip()}:{normalize_str(parent_comment_id).strip()}")
                state["replied_comment_pairs"] = list(replied_pairs)[-20000:]
            executed += 1
            print_success_banner(
                action="pending-comment",
                pid=pid,
                url=normalize_str(action.get("url")) or post_url(pid),
                title=normalize_str(action.get("title")) or "Queued comment",
            )
            continue

        if kind == "vote_comment":
            cid = normalize_str(action.get("comment_id"))
            vote_action = normalize_vote_action(action.get("vote_action"))
            if not cid or vote_action == "none":
                logger.warning("Dropping invalid pending comment vote action.")
                continue
            if vote_action == "downvote" and not cfg.allow_comment_downvote:
                logger.info("Dropping unsupported pending downvote-comment action comment_id=%s", cid)
                continue
            logger.info("Executing pending action kind=vote_comment comment_id=%s vote=%s", cid, vote_action)
            client.vote_comment(cid, vote_action=vote_action)
            voted_ids = set(state.get("voted_comment_ids", []))
            voted_ids.add(cid)
            state["voted_comment_ids"] = list(voted_ids)[-10000:]
            executed += 1
            print_success_banner(
                action=f"pending-{vote_action}-comment",
                pid=cid,
                url=normalize_str(action.get("url")),
                title=normalize_str(action.get("title")) or "Queued comment vote",
            )
            continue

        if kind == "vote_post":
            pid = normalize_str(action.get("post_id"))
            vote_action = normalize_vote_action(action.get("vote_action"))
            if not pid or vote_action == "none":
                logger.warning("Dropping invalid pending post vote action.")
                continue
            logger.info("Executing pending action kind=vote_post post_id=%s vote=%s", pid, vote_action)
            client.vote_post(pid, vote_action=vote_action)
            executed += 1
            print_success_banner(
                action=f"pending-{vote_action}-post",
                pid=pid,
                url=normalize_str(action.get("url")) or post_url(pid),
                title=normalize_str(action.get("title")) or "Queued post vote",
            )
            continue

        logger.warning("Dropping unsupported pending action kind=%s", kind)

    state["pending_actions"] = remaining
    if executed > 0:
        logger.info("Executed pending actions count=%s remaining=%s", executed, len(remaining))
    return executed


def maybe_upvote_post_after_comment(
    client: MoltbookClient,
    state: Dict[str, Any],
    logger,
    post_id_value: str,
) -> None:
    voted_post_ids = set(state.get("voted_post_ids", []))
    if post_id_value in voted_post_ids:
        return
    try:
        client.vote_post(post_id_value, vote_action="upvote")
        voted_post_ids.add(post_id_value)
        state["voted_post_ids"] = list(voted_post_ids)[-10000:]
        logger.info("Auto-upvoted post after comment post_id=%s", post_id_value)
        print_success_banner(
            action="auto-upvote-post",
            pid=post_id_value,
            url=post_url(post_id_value),
            title="Auto upvote after comment",
        )
    except Exception as e:
        logger.warning("Auto-upvote after comment failed post_id=%s error=%s", post_id_value, e)


def wait_for_comment_slot(state: Dict[str, Any], cfg: Config, logger) -> bool:
    allowed, reason = comment_gate_status(state=state, cfg=cfg)
    if allowed:
        return True
    if reason != "comment_cooldown":
        logger.info("Cannot wait for comment slot reason=%s", reason)
        return False
    _, comment_remaining = cooldown_remaining_seconds(state=state, cfg=cfg)
    wait_seconds = max(1, comment_remaining)
    logger.info("Waiting for comment cooldown to clear seconds=%s", wait_seconds)
    time.sleep(wait_seconds)
    allowed_after, reason_after = comment_gate_status(state=state, cfg=cfg)
    if not allowed_after:
        logger.info("Comment slot still unavailable after wait reason=%s", reason_after)
        return False
    return True


def build_top_post_signals(posts: List[Dict[str, Any]], limit: int, source: str = "top") -> List[Dict[str, Any]]:
    ranked = sorted(posts, key=lambda p: (post_score(p), post_comment_count(p)), reverse=True)
    signals: List[Dict[str, Any]] = []
    for post in ranked[: max(1, limit)]:
        pid = post_id(post)
        if not pid:
            continue
        title_text = normalize_str(post.get("title")).strip()
        signals.append(
            {
                "post_id": pid,
                "title": title_text,
                "submolt": submolt_name_from_post(post) or normalize_submolt(post.get("submolt")),
                "score": post_score(post),
                "comment_count": post_comment_count(post),
                "source": source,
                "title_char_count": len(title_text),
                "has_question_title": "?" in title_text,
            }
        )
    return signals


def _normalize_archetype(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    if text in PROACTIVE_ARCHETYPES:
        return text
    return "unknown"


def _select_proactive_archetype_plan(
    state: Dict[str, Any],
    post_memory: Dict[str, Any],
) -> Tuple[str, List[str], str]:
    recent_entries = [e for e in post_memory.get("proactive_posts", []) if isinstance(e, dict)]
    recent_entries = recent_entries[- max(8, len(PROACTIVE_ARCHETYPES) * 2) :]
    recent_archetypes = {
        _normalize_archetype(entry.get("content_archetype"))
        for entry in recent_entries
    }
    recent_archetypes.discard("unknown")

    missing = [a for a in PROACTIVE_ARCHETYPES if a not in recent_archetypes]
    if missing:
        required = missing[0]
        return required, [required], "rotation_missing"

    snapshot = post_memory.get("last_snapshot")
    best_archetypes: List[str] = []
    if isinstance(snapshot, dict):
        ranked = snapshot.get("best_archetypes")
        if isinstance(ranked, list):
            for item in ranked:
                if not isinstance(item, dict):
                    continue
                name = _normalize_archetype(item.get("name"))
                if name != "unknown" and name not in best_archetypes:
                    best_archetypes.append(name)

    attempt_count = int(state.get("proactive_post_attempt_count", 0))
    if best_archetypes:
        primary = best_archetypes[0]
    else:
        primary = PROACTIVE_ARCHETYPES[attempt_count % len(PROACTIVE_ARCHETYPES)]

    # Exploit the top archetype for 3/4 attempts, then explore 1/4.
    if attempt_count > 0 and attempt_count % 4 == 0:
        ordered = [a for a in PROACTIVE_ARCHETYPES if a != primary]
        required = ordered[(attempt_count // 4 - 1) % len(ordered)] if ordered else primary
        mode = "explore_rotation"
    else:
        required = primary
        mode = "exploit_top"

    preferred = [required] + [a for a in best_archetypes if a != required][:2]
    return required, preferred, mode


def _ensure_direct_question(content: str) -> str:
    text = normalize_str(content).strip()
    if not text:
        return text
    if "?" in text:
        return text
    return text + "\n\nWhich implementation constraint would you test first on Ergo?"


def _build_forced_proactive_fallback(
    top_signals: List[Dict[str, Any]],
    target_submolt: str,
    weekly_theme: str,
    required_archetype: str,
) -> Dict[str, Any]:
    ref_pid = ""
    ref_title = ""
    if top_signals:
        ref_pid = normalize_str(top_signals[0].get("post_id")).strip()
        ref_title = normalize_str(top_signals[0].get("title")).strip()
    ref_url = post_url(ref_pid) if ref_pid else ""
    title = f"{weekly_theme}: deterministic escrow for autonomous agents on Ergo".strip()
    if len(title) > 140:
        title = title[:137].rstrip() + "..."

    body_lines = [
        "Agent economies fail when counterparties cannot prove execution and settlement needs manual trust.",
        "Ergo eUTXO plus ErgoScript fixes this with deterministic contract paths and on-chain verifiable conditions.",
        "",
        (
            "Implementation sketch: lock service payments in an ErgoScript escrow box, then release only when the "
            "proof hash and signature threshold match contract rules; otherwise route to a timeout + dispute branch "
            "with reputation-gated arbiters."
        ),
    ]
    if ref_url:
        body_lines.extend(["", f"Reference thread: {ref_url}"])
    if ref_title:
        body_lines.append(f"Reference topic: {ref_title}")
    body_lines.extend(
        [
            "",
            "Would you start with hard minimum reputation gates or bond-scaled adaptive gates for counterparties?",
        ]
    )
    content = "\n".join(body_lines).strip()
    return {
        "should_post": True,
        "confidence": 0.99,
        "submolt": normalize_submolt(target_submolt),
        "title": title,
        "content": content,
        "strategy_notes": "deterministic_fallback_used_for_priority_post",
        "topic_tags": ["ergo", "eutxo", "ergoscript", "agent-economy", "settlement"],
        "content_archetype": _normalize_archetype(required_archetype),
    }


def _proactive_posts_count_for_date(post_memory: Dict[str, Any], date_iso: str) -> int:
    entries = post_memory.get("proactive_posts")
    if not isinstance(entries, list):
        return 0
    count = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        created_ts = item.get("created_ts")
        if not isinstance(created_ts, (int, float)):
            continue
        try:
            entry_date = time.strftime("%Y-%m-%d", time.gmtime(float(created_ts)))
        except Exception:
            continue
        if entry_date == date_iso:
            count += 1
    return count


def _weekly_proactive_theme_hint() -> str:
    weekday = utc_now().weekday()
    return WEEKLY_PROACTIVE_THEMES.get(weekday, "Concrete Ergo mechanisms for agent autonomy")


def _choose_proactive_submolt(
    cfg: Config,
    learning_snapshot: Dict[str, Any],
    force_general: bool = False,
) -> str:
    allowed = {normalize_submolt(name) for name in cfg.target_submolts if normalize_submolt(name)}
    default_submolt = normalize_submolt(cfg.proactive_post_submolt)
    if default_submolt:
        allowed.add(default_submolt)
    if force_general and "general" in allowed:
        return "general"

    ranked: List[str] = []
    best_submolts = learning_snapshot.get("best_submolts")
    if isinstance(best_submolts, list):
        for item in best_submolts:
            if not isinstance(item, dict):
                continue
            name = normalize_submolt(item.get("name"), default="")
            if name and name not in ranked:
                ranked.append(name)

    market_snapshot = learning_snapshot.get("market_snapshot")
    if isinstance(market_snapshot, dict):
        market_submolts = market_snapshot.get("top_submolts")
        if isinstance(market_submolts, list):
            for item in market_submolts:
                if not isinstance(item, dict):
                    continue
                name = normalize_submolt(item.get("name"), default="")
                if name and name not in ranked:
                    ranked.append(name)

    for candidate in ranked:
        if candidate in allowed:
            return candidate
    return default_submolt or "general"


def _optimize_proactive_title(title: str, learning_snapshot: Dict[str, Any]) -> str:
    text = normalize_str(title).strip()
    if not text:
        return text
    market_snapshot = learning_snapshot.get("market_snapshot")
    if not isinstance(market_snapshot, dict):
        return text
    question_rate = market_snapshot.get("question_title_rate")
    if not isinstance(question_rate, (int, float)) or question_rate < 0.35:
        return text
    if "?" in text:
        return text
    cleaned = text.rstrip()
    while cleaned.endswith((".", "!", ":", ";")):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        cleaned = text
    if len(cleaned) <= 118:
        return cleaned + "?"
    return text


def proactive_post_attempt_allowed(state: Dict[str, Any], cfg: Config) -> bool:
    # If post slot is open and we are already beyond post cooldown, do not throttle by attempt cooldown.
    if should_prioritize_proactive_post(state=state, cfg=cfg):
        return True
    # When a post slot is already open, retry faster so proactive posting does not stall for long periods.
    post_allowed, _ = post_gate_status(state=state, cfg=cfg)
    cooldown_seconds = max(1, cfg.proactive_post_attempt_cooldown_seconds)
    if post_allowed:
        cooldown_seconds = min(cooldown_seconds, 120)

    last_attempt = state.get("last_proactive_post_attempt_ts")
    if not isinstance(last_attempt, (int, float)):
        return True
    elapsed = utc_now().timestamp() - last_attempt
    return elapsed >= cooldown_seconds


def maybe_run_proactive_post(
    client: MoltbookClient,
    cfg: Config,
    logger,
    state: Dict[str, Any],
    post_memory: Dict[str, Any],
    my_name: Optional[str],
    persona_text: str,
    domain_context_text: str,
    approve_all_actions: bool,
) -> Tuple[int, bool, bool]:
    if not cfg.proactive_posting_enabled:
        return 0, False, approve_all_actions

    post_allowed, post_reason = post_gate_status(state=state, cfg=cfg)
    if not post_allowed:
        logger.info("Proactive post skipped reason=%s", post_reason)
        return 0, False, approve_all_actions

    if not proactive_post_attempt_allowed(state=state, cfg=cfg):
        post_allowed_now, _ = post_gate_status(state=state, cfg=cfg)
        effective_cooldown = max(1, cfg.proactive_post_attempt_cooldown_seconds)
        if post_allowed_now:
            effective_cooldown = min(effective_cooldown, 120)
        logger.info(
            "Proactive post skipped reason=attempt_cooldown seconds=%s",
            effective_cooldown,
        )
        return 0, False, approve_all_actions

    if not has_generation_provider(cfg):
        logger.info("Proactive post skipped reason=no_generation_provider")
        return 0, False, approve_all_actions

    refresh_seconds = max(1, cfg.proactive_metrics_refresh_seconds)
    last_refresh = post_memory.get("last_metrics_refresh_ts")
    should_refresh = not isinstance(last_refresh, (int, float))
    if isinstance(last_refresh, (int, float)):
        should_refresh = (utc_now().timestamp() - last_refresh) >= refresh_seconds

    if my_name and should_refresh:
        try:
            profile = client.get_agent_profile(my_name)
            recent_posts = extract_recent_posts_from_profile(profile)
            updated = refresh_metrics_from_recent_posts(post_memory, recent_posts)
            logger.info("Proactive memory metrics refreshed updated=%s recent_posts=%s", updated, len(recent_posts))
        except Exception as e:
            logger.debug("Proactive memory metrics refresh failed error=%s", e)

    trend_sources = ("top", "hot", "rising")
    trend_signal_map: Dict[str, List[Dict[str, Any]]] = {}
    for source in trend_sources:
        try:
            payload = client.get_posts(sort=source, limit=max(5, cfg.proactive_post_reference_limit))
            posts = extract_posts(payload)
            trend_signal_map[source] = build_top_post_signals(
                posts=posts,
                limit=cfg.proactive_post_reference_limit,
                source=source,
            )
        except Exception as e:
            logger.debug("Proactive trend source fetch failed source=%s error=%s", source, e)
            trend_signal_map[source] = []

    if not any(trend_signal_map.values()):
        logger.warning("Proactive post skipped reason=trend_posts_fetch_failed")
        return 0, False, approve_all_actions

    dedup_by_post: Dict[str, Dict[str, Any]] = {}
    for source in trend_sources:
        for signal in trend_signal_map.get(source, []):
            pid = normalize_str(signal.get("post_id")).strip()
            if not pid:
                continue
            existing = dedup_by_post.get(pid)
            if existing is None:
                dedup_by_post[pid] = dict(signal)
                dedup_by_post[pid]["sources"] = [source]
                continue
            merged_sources = set(existing.get("sources", []))
            merged_sources.add(source)
            existing["sources"] = sorted(merged_sources)
            existing["source"] = existing["sources"][0]
            if int(signal.get("score", 0)) > int(existing.get("score", 0)):
                existing["score"] = int(signal.get("score", 0))
            if int(signal.get("comment_count", 0)) > int(existing.get("comment_count", 0)):
                existing["comment_count"] = int(signal.get("comment_count", 0))

    top_signals = sorted(
        dedup_by_post.values(),
        key=lambda item: (int(item.get("score", 0)), int(item.get("comment_count", 0))),
        reverse=True,
    )[: max(1, cfg.proactive_post_reference_limit * 2)]
    market_snapshot = update_market_signals(post_memory, top_signals)
    logger.info(
        "Proactive trend signals loaded count=%s source_counts=%s",
        len(top_signals),
        market_snapshot.get("source_counts", {}),
    )
    learning_snapshot = build_learning_snapshot(post_memory, max_examples=5)
    if not top_signals:
        logger.info("Proactive post skipped reason=no_top_signals")
        return 0, False, approve_all_actions

    today_iso = time.strftime("%Y-%m-%d", time.gmtime())
    proactive_today = _proactive_posts_count_for_date(post_memory=post_memory, date_iso=today_iso)
    daily_target = max(1, int(cfg.proactive_daily_target_posts))
    force_general = bool(
        cfg.proactive_force_general_until_daily_target
        and proactive_today < daily_target
    )
    weekly_theme = _weekly_proactive_theme_hint()
    required_archetype, preferred_archetypes, archetype_mode = _select_proactive_archetype_plan(
        state=state,
        post_memory=post_memory,
    )
    target_submolt = _choose_proactive_submolt(
        cfg=cfg,
        learning_snapshot=learning_snapshot,
        force_general=force_general,
    )
    fallback_post = _build_forced_proactive_fallback(
        top_signals=top_signals,
        target_submolt=target_submolt,
        weekly_theme=weekly_theme,
        required_archetype=required_archetype,
    )
    using_forced_fallback = False
    state["proactive_post_attempt_count"] = int(state.get("proactive_post_attempt_count", 0)) + 1
    state["last_proactive_post_attempt_ts"] = utc_now().timestamp()
    logger.info(
        (
            "Proactive plan mode=%s required_archetype=%s preferred=%s submolt=%s "
            "attempt=%s daily_target=%s proactive_today=%s force_general=%s theme=%s"
        ),
        archetype_mode,
        required_archetype,
        ",".join(preferred_archetypes[:3]) if preferred_archetypes else "(none)",
        target_submolt,
        state.get("proactive_post_attempt_count", 0),
        daily_target,
        proactive_today,
        force_general,
        weekly_theme,
    )
    provider_used = "unknown"
    try:
        messages = build_proactive_post_messages(
            persona=persona_text,
            domain_context=domain_context_text,
            top_posts=top_signals,
            learning_snapshot=learning_snapshot,
            target_submolt=target_submolt,
            weekly_theme=weekly_theme,
            required_archetype=required_archetype,
            preferred_archetypes=preferred_archetypes,
        )
        draft, provider_used, _ = call_generation_model(cfg, messages)
    except Exception as e:
        logger.warning(
            "Proactive post drafting failed provider_hint=%s error=%s fallback=deterministic",
            cfg.llm_provider,
            e,
        )
        draft = dict(fallback_post)
        provider_used = "deterministic-fallback"
        using_forced_fallback = True
    logger.info("Proactive post draft generated provider=%s", provider_used)

    draft_archetype = _normalize_archetype(draft.get("content_archetype"))
    if draft_archetype != required_archetype:
        logger.warning(
            "Proactive archetype mismatch required=%s got=%s retrying_once=1",
            required_archetype,
            draft_archetype,
        )
        try:
            retry_messages = build_proactive_post_messages(
                persona=persona_text,
                domain_context=domain_context_text,
                top_posts=top_signals,
                learning_snapshot=learning_snapshot,
                target_submolt=target_submolt,
                weekly_theme=weekly_theme,
                required_archetype=required_archetype,
                preferred_archetypes=[required_archetype],
            )
            draft, provider_used, _ = call_generation_model(cfg, retry_messages)
            logger.info("Proactive post retry draft generated provider=%s", provider_used)
        except Exception as e:
            logger.warning("Proactive post retry drafting failed error=%s fallback=deterministic", e)
            draft = dict(fallback_post)
            provider_used = "deterministic-fallback"
            using_forced_fallback = True
        draft_archetype = _normalize_archetype(draft.get("content_archetype"))
        if draft_archetype != required_archetype:
            logger.info(
                "Proactive post archetype mismatch required=%s got=%s fallback=deterministic",
                required_archetype,
                draft_archetype,
            )
            record_declined_idea(
                memory=post_memory,
                title=_sanitize_generated_title(draft.get("title"), fallback="(untitled)"),
                submolt=normalize_submolt(draft.get("submolt"), default=target_submolt),
                reason=f"archetype_mismatch:{required_archetype}->{draft_archetype}",
            )
            draft = dict(fallback_post)
            provider_used = "deterministic-fallback"
            using_forced_fallback = True
            draft_archetype = _normalize_archetype(draft.get("content_archetype"))

    should_post = bool(draft.get("should_post"))
    confidence = float(draft.get("confidence", 0.0))
    if not should_post or confidence < cfg.min_confidence:
        logger.info(
            "Proactive post declined should_post=%s confidence=%.3f threshold=%.3f fallback=deterministic",
            should_post,
            confidence,
            cfg.min_confidence,
        )
        record_declined_idea(
            memory=post_memory,
            title=_sanitize_generated_title(draft.get("title"), fallback="(untitled)"),
            submolt=normalize_submolt(draft.get("submolt"), default=target_submolt),
            reason="model_declined_or_low_confidence",
        )
        draft = dict(fallback_post)
        provider_used = "deterministic-fallback"
        using_forced_fallback = True
        should_post = True
        confidence = float(draft.get("confidence", 0.99))

    submolt = normalize_submolt(draft.get("submolt"), default=target_submolt)
    raw_title = _sanitize_generated_title(
        draft.get("title"),
        fallback="Ergo x agent economy: practical next step",
    )
    title = _optimize_proactive_title(raw_title, learning_snapshot=learning_snapshot)
    if title != raw_title:
        logger.info("Proactive title adapted for market signal original=%r adapted=%r", raw_title, title)
    content = _ensure_direct_question(normalize_str(draft.get("content")).strip())
    content = _normalize_ergo_terms(content)
    content = _ensure_proactive_opening_gate(content)
    strategy_notes = normalize_str(draft.get("strategy_notes")).strip()
    content_archetype = draft_archetype
    raw_tags = draft.get("topic_tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []
    topic_tags = [normalize_str(x).strip().lower() for x in raw_tags if normalize_str(x).strip()]
    topic_tags = topic_tags[:8]

    if not content or _is_template_like_generated_content(content) or not _passes_proactive_opening_gate(content):
        if not using_forced_fallback:
            fallback_reason = "empty_content" if not content else (
                "template_like_content" if _is_template_like_generated_content(content) else "weak_opening_hook"
            )
            logger.info("Proactive post skipped reason=%s fallback=deterministic", fallback_reason)
            record_declined_idea(
                memory=post_memory,
                title=title,
                submolt=submolt,
                reason=fallback_reason,
            )
            draft = dict(fallback_post)
            using_forced_fallback = True
            draft_archetype = _normalize_archetype(draft.get("content_archetype"))
            submolt = normalize_submolt(draft.get("submolt"), default=target_submolt)
            raw_title = _sanitize_generated_title(
                draft.get("title"),
                fallback="Ergo x agent economy: practical next step",
            )
            title = _optimize_proactive_title(raw_title, learning_snapshot=learning_snapshot)
            content = _ensure_direct_question(normalize_str(draft.get("content")).strip())
            content = _normalize_ergo_terms(content)
            content = _ensure_proactive_opening_gate(content)
            strategy_notes = normalize_str(draft.get("strategy_notes")).strip()
            content_archetype = draft_archetype
            raw_tags = draft.get("topic_tags") or []
            if not isinstance(raw_tags, list):
                raw_tags = []
            topic_tags = [normalize_str(x).strip().lower() for x in raw_tags if normalize_str(x).strip()]
            topic_tags = topic_tags[:8]
        if not content or _is_template_like_generated_content(content) or not _passes_proactive_opening_gate(content):
            final_reason = "empty_content" if not content else (
                "template_like_content" if _is_template_like_generated_content(content) else "weak_opening_hook"
            )
            logger.warning("Proactive fallback failed reason=%s forcing_emergency_post=1", final_reason)
            record_declined_idea(
                memory=post_memory,
                title=title,
                submolt=submolt,
                reason=f"{final_reason}_after_fallback",
            )
            submolt = normalize_submolt(target_submolt)
            title = _sanitize_generated_title(
                "Startup priority: deterministic Ergo escrow blueprint for autonomous agents"
            )
            content = (
                "Cross-agent payment flows break when counterparties cannot verify settlement paths. "
                "Ergo eUTXO + ErgoScript gives deterministic escrow branches with auditable release rules.\n\n"
                "Minimal rollout: lock payment in an escrow box, require proof hash + threshold signatures for release, "
                "and route unresolved disputes to a timeout branch with reputation-gated arbiters.\n\n"
                "Would you start with hard reputation thresholds or bond-scaled adaptive gates for counterparties?"
            )
            content = _normalize_ergo_terms(_ensure_proactive_opening_gate(content))
            content_archetype = _normalize_archetype(required_archetype)
            strategy_notes = "emergency_deterministic_priority_post"
            topic_tags = ["ergo", "eutxo", "ergoscript", "escrow", "agent-economy"]

    preview = content
    if strategy_notes:
        preview = f"{content}\n\n[notes] {strategy_notes}"

    approved, approve_all_actions, should_stop = confirm_action(
        cfg=cfg,
        logger=logger,
        action="post-proactive",
        pid="(new)",
        title=title,
        submolt=submolt,
        url=f"https://moltbook.com/m/{submolt}",
        author="system",
        content_preview=preview_text(preview),
        approve_all=approve_all_actions,
    )
    if should_stop:
        return 0, True, approve_all_actions
    if not approved:
        logger.info("Proactive post skipped reason=not_approved")
        record_declined_idea(
            memory=post_memory,
            title=title,
            submolt=submolt,
            reason="not_approved",
        )
        return 0, False, approve_all_actions

    if cfg.dry_run:
        logger.info("Proactive post dry_run submolt=%s title=%s", submolt, title)
        return 0, False, approve_all_actions

    try:
        response = client.create_post(submolt=submolt, title=title, content=content)
    except Exception as e:
        logger.warning("Proactive post send failed submolt=%s title=%s error=%s", submolt, title, e)
        return 0, False, approve_all_actions

    created_post_id = post_id(response if isinstance(response, dict) else {}) or "(unknown)"
    state["daily_post_count"] = state.get("daily_post_count", 0) + 1
    mark_reply_action_timestamps(state=state, action_kind="post")
    logger.info(
        "Proactive post success post_id=%s submolt=%s daily_post_count=%s",
        created_post_id,
        submolt,
        state["daily_post_count"],
    )
    record_proactive_post(
        memory=post_memory,
        post_id=created_post_id,
        title=title,
        submolt=submolt,
        content=content,
        strategy_notes=strategy_notes,
        topic_tags=topic_tags,
        content_archetype=content_archetype,
    )
    print_success_banner(
        action="post-proactive",
        pid=created_post_id,
        url=post_url(created_post_id if created_post_id != "(unknown)" else None),
        title=title,
    )
    return 1, False, approve_all_actions


def merge_unique_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for post in posts:
        pid = post_id(post)
        if not pid:
            continue
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        out.append(post)
    return out


def select_keyword_batch(
    keywords: List[str],
    batch_size: int,
    cursor: int,
) -> Tuple[List[str], int]:
    if not keywords:
        return [], 0
    size = max(1, min(batch_size, len(keywords)))
    if size >= len(keywords):
        return keywords[:], 0

    selected: List[str] = []
    idx = cursor % len(keywords)
    for _ in range(size):
        selected.append(keywords[idx])
        idx = (idx + 1) % len(keywords)
    return selected, idx


def has_my_comment_on_post(
    client: MoltbookClient,
    post_id_value: str,
    my_name: Optional[str],
    logger,
) -> bool:
    if not my_name:
        return False
    my_key = author_identity_key(author_id=None, author_name=my_name)
    my_name_key = _normalized_name_key(my_name)
    try:
        payload = client.get_post_comments(post_id_value, limit=200)
    except Exception as e:
        logger.debug("Comment history check failed post_id=%s error=%s", post_id_value, e)
        return False
    for comment in extract_comments(payload):
        author_id, author_name = comment_author(comment)
        if my_key and author_identity_key(author_id, author_name) == my_key:
            return True
        if my_name_key and _normalized_name_key(author_name) == my_name_key:
            return True
    return False


def has_my_reply_to_comment(
    client: MoltbookClient,
    post_id_value: str,
    parent_comment_id: str,
    my_name: Optional[str],
    logger,
    self_identity_keys: Optional[Set[str]] = None,
) -> bool:
    if not my_name:
        return False
    my_key = author_identity_key(author_id=None, author_name=my_name)
    my_name_key = _normalized_name_key(my_name)
    self_keys: Set[str] = set(self_identity_keys or set())
    if my_key:
        self_keys.add(my_key)
    parent_key = normalize_str(parent_comment_id).strip()
    if not parent_key:
        return False
    try:
        payload = client.get_post_comments(post_id_value, limit=250)
    except Exception as e:
        logger.debug(
            "Reply-parent history check failed post_id=%s parent_comment_id=%s error=%s",
            post_id_value,
            parent_comment_id,
            e,
        )
        return False
    for comment in extract_comments(payload):
        if normalize_str(comment_parent_id(comment)).strip() != parent_key:
            continue
        author_id, author_name = comment_author(comment)
        if self_keys and is_self_author(author_id, author_name, self_identity_keys=self_keys):
            return True
        if my_name_key and _normalized_name_key(author_name) == my_name_key:
            return True
    return False


def discover_posts(
    client: MoltbookClient,
    cfg: Config,
    logger,
    keywords: List[str],
    iteration: int,
    search_state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    sources: List[str] = []
    search_posts: List[Dict[str, Any]] = []
    submolt_posts: List[Dict[str, Any]] = []

    if cfg.discovery_mode == "search":
        retry_cycle = int(search_state.get("retry_cycle", 1))
        if iteration < retry_cycle:
            logger.info("Search temporarily paused until cycle=%s; using feed only this cycle.", retry_cycle)
        else:
            cursor = int(search_state.get("keyword_cursor", 0))
            cycle_keywords, next_cursor = select_keyword_batch(
                keywords=keywords,
                batch_size=cfg.search_batch_size,
                cursor=cursor,
            )
            search_state["keyword_cursor"] = next_cursor
            search_errors = 0
            not_found_errors = 0
            sampled_error: Optional[str] = None
            mission_posts = 0
            for mission_query in cfg.mission_queries:
                try:
                    result = client.search_posts(query=mission_query, limit=cfg.search_limit, search_type="posts")
                    posts = extract_posts(result)
                    mission_posts += len(posts)
                    search_posts.extend(posts)
                except MoltbookAuthError:
                    raise
                except Exception as e:
                    search_errors += 1
                    err = str(e)
                    sampled_error = sampled_error or err
                    if "not found" in err.lower() or "unavailable" in err.lower():
                        not_found_errors += 1
                    logger.debug("Discovery mission search failed query=%s error=%s", mission_query, err)

            for keyword in cycle_keywords:
                try:
                    result = client.search_posts(query=keyword, limit=cfg.search_limit, search_type="posts")
                    posts = extract_posts(result)
                    logger.debug("Discovery search keyword=%s results=%s", keyword, len(posts))
                    search_posts.extend(posts)
                except MoltbookAuthError:
                    raise
                except Exception as e:
                    search_errors += 1
                    err = str(e)
                    sampled_error = sampled_error or err
                    if "not found" in err.lower() or "unavailable" in err.lower():
                        not_found_errors += 1
                    logger.debug("Discovery search failed keyword=%s error=%s", keyword, err)

            attempted_searches = len(cycle_keywords) + len(cfg.mission_queries)
            if attempted_searches > 0 and search_errors > 0 and search_errors == attempted_searches:
                search_state["retry_cycle"] = iteration + max(1, cfg.search_retry_after_failure_cycles)
                logger.warning(
                    (
                        "Discovery search unavailable this cycle errors=%s/%s not_found=%s sample_error=%s. "
                        "Pausing search until cycle=%s; using feed fallback."
                    ),
                    search_errors,
                    attempted_searches,
                    not_found_errors,
                    sampled_error or "(none)",
                    search_state["retry_cycle"],
                )
            else:
                if search_posts:
                    sources.append("search")
                    logger.info(
                        "Discovery search collected_posts=%s mission_queries=%s mission_posts=%s keyword_batch=%s/%s",
                        len(search_posts),
                        len(cfg.mission_queries),
                        mission_posts,
                        len(cycle_keywords),
                        len(keywords),
                    )
                elif search_errors > 0:
                    logger.info(
                        (
                            "Discovery search produced no posts this cycle "
                            "(mission_queries=%s keyword_batch=%s/%s); using feed fallback."
                        ),
                        len(cfg.mission_queries),
                        len(cycle_keywords),
                        len(keywords),
                    )

    feed = client.get_feed(limit=cfg.feed_limit)
    feed_posts = extract_posts(feed)
    posts_resp = client.get_posts(sort=cfg.posts_sort, limit=cfg.posts_limit)
    global_posts = extract_posts(posts_resp)
    if cfg.target_submolts:
        for submolt in cfg.target_submolts:
            try:
                payload = client.get_submolt_feed(name=submolt, sort=cfg.posts_sort, limit=cfg.posts_limit)
                posts = extract_posts(payload)
                if posts:
                    submolt_posts.extend(posts)
            except MoltbookAuthError:
                raise
            except Exception as e:
                logger.debug("Discovery submolt feed failed submolt=%s error=%s", submolt, e)
    if submolt_posts:
        sources.append("submolts")
    sources.extend(["posts", "feed"])
    merged = merge_unique_posts(search_posts + submolt_posts + global_posts + feed_posts)
    logger.info(
        "Discovery merged search_posts=%s submolt_posts=%s posts_global=%s feed_posts=%s total=%s",
        len(search_posts),
        len(submolt_posts),
        len(global_posts),
        len(feed_posts),
        len(merged),
    )
    return merged, sources


def resolve_self_name(client: MoltbookClient, logger) -> Optional[str]:
    try:
        me = client.get_me()
        return me.get("name") or me.get("agent_name")
    except Exception as e:
        logger.warning("Could not resolve current agent identity: %s", e)
        return None


def extract_recent_posts_from_profile(profile_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    containers: List[Any] = [profile_payload]
    for key in ("agent", "profile", "data", "result"):
        value = profile_payload.get(key)
        if isinstance(value, dict):
            containers.append(value)
        elif isinstance(value, list):
            posts = [item for item in value if isinstance(item, dict)]
            if posts:
                return posts

    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("recentPosts", "recent_posts", "posts", "items"):
            value = container.get(key)
            if isinstance(value, list):
                posts = [item for item in value if isinstance(item, dict)]
                if posts:
                    return posts
    return []


def run_startup_reply_scan(
    client: MoltbookClient,
    cfg: Config,
    logger,
    state: Dict[str, Any],
    my_name: Optional[str],
    persona_text: str,
    domain_context_text: str,
    approve_all_actions: bool,
) -> bool:
    if not cfg.startup_reply_scan_enabled:
        return approve_all_actions
    if not my_name:
        logger.info("Reply scan skipped (agent identity unavailable).")
        return approve_all_actions

    reset_daily_if_needed(state)
    hourly_comments = _prune_comment_action_timestamps(state=state, window_seconds=3600)
    logger.info(
        (
            "Reply scan begin agent=%s post_limit=%s comment_limit=%s "
            "hourly_comment_count=%s/%s daily_comment_count=%s/%s "
            "max_replies_per_author_per_post=%s thread_escalate_turns=%s triage_llm_budget=%s"
        ),
        my_name,
        cfg.startup_reply_scan_post_limit,
        cfg.startup_reply_scan_comment_limit,
        len(hourly_comments),
        cfg.max_comments_per_hour,
        state.get("daily_comment_count", 0),
        cfg.max_comments_per_day,
        MAX_REPLIES_PER_AUTHOR_PER_POST,
        THREAD_ESCALATE_TURNS,
        max(1, cfg.reply_triage_llm_calls_per_scan),
    )
    seen_comment_ids: Set[str] = set(state.get("seen_comment_ids", []))
    my_comment_ids: Set[str] = set(state.get("my_comment_ids", []))
    voted_comment_ids: Set[str] = set(state.get("voted_comment_ids", []))
    replied_to_comment_ids: Set[str] = set(state.get("replied_to_comment_ids", []))
    replied_comment_pairs: Set[str] = set(state.get("replied_comment_pairs", []))
    replied_post_ids: Set[str] = set(state.get("replied_post_ids", []))
    thread_followup_posted_pairs: Set[str] = set(state.get("thread_followup_posted_pairs", []))
    triage_cache_raw = state.get("reply_triage_cache", {})
    triage_cache: Dict[str, Dict[str, Any]] = {}
    if isinstance(triage_cache_raw, dict):
        for key, value in triage_cache_raw.items():
            if not isinstance(value, dict):
                continue
            triage_cache[normalize_str(key)] = value
    triage_llm_budget = max(1, cfg.reply_triage_llm_calls_per_scan)
    triage_llm_calls = 0
    triage_cache_hits = 0
    triage_deferred = 0
    self_identity_keys: Set[str] = resolve_self_identity_keys(client=client, my_name=my_name, logger=logger)
    if not self_identity_keys and my_name:
        fallback_key = author_identity_key(author_id=None, author_name=my_name)
        if fallback_key:
            self_identity_keys.add(fallback_key)
    scanned = 0
    new_replies = 0
    actions = 0
    skip_reasons: Dict[str, int] = {}
    provider_counts: Dict[str, int] = {}
    triage_fail_count = 0
    triage_parse_fail_count = 0

    try:
        profile_payload = client.get_agent_profile(my_name)
    except Exception as e:
        logger.warning("Reply scan skipped (profile fetch failed): %s", e)
        return approve_all_actions

    recent_posts = extract_recent_posts_from_profile(profile_payload)[: cfg.startup_reply_scan_post_limit]
    if not recent_posts and my_name:
        try:
            fallback_payload = client.get_posts(
                sort="new",
                limit=max(50, cfg.startup_reply_scan_post_limit * 4),
            )
            fallback_posts = extract_posts(fallback_payload)
            mine: List[Dict[str, Any]] = []
            my_key = author_identity_key(author_id=None, author_name=my_name)
            for p in fallback_posts:
                author_id, author_name = post_author(p)
                if my_key and author_identity_key(author_id, author_name) == my_key:
                    mine.append(p)
                    if len(mine) >= cfg.startup_reply_scan_post_limit:
                        break
            if mine:
                recent_posts = mine
                logger.info("Reply scan fallback loaded own posts from global feed count=%s", len(mine))
        except Exception as e:
            logger.debug("Reply scan fallback own-post lookup failed error=%s", e)
    recent_count = len(recent_posts)
    scan_posts: List[Dict[str, Any]] = []
    scan_post_ids: Set[str] = set()

    for post in recent_posts:
        pid = post_id(post)
        if not pid or pid in scan_post_ids:
            continue
        scan_post_ids.add(pid)
        post_obj = dict(post)
        post_obj["__scan_source"] = "profile_recent"
        scan_posts.append(post_obj)

    # Also scan posts where we have previously commented, so we can follow up on replies to our comments.
    for pid in list(replied_post_ids)[: cfg.startup_reply_scan_replied_post_limit]:
        if pid in scan_post_ids:
            continue
        try:
            payload = client.get_post(pid)
            post_obj = extract_single_post(payload)
            if not post_obj:
                continue
            if post_id(post_obj) != pid:
                post_obj["id"] = pid
            post_obj["__scan_source"] = "replied_post"
            scan_post_ids.add(pid)
            scan_posts.append(post_obj)
        except Exception as e:
            logger.debug("Reply scan could not hydrate replied post_id=%s error=%s", pid, e)

    logger.info(
        "Reply scan post set recent_posts=%s replied_posts=%s total_scan_posts=%s",
        recent_count,
        max(0, len(scan_posts) - recent_count),
        len(scan_posts),
    )

    for post in scan_posts:
        pid = post_id(post)
        if not pid:
            continue
        try:
            comments_payload = client.get_post_comments(pid, limit=cfg.startup_reply_scan_comment_limit)
        except Exception as e:
            logger.warning("Reply scan comments fetch failed post_id=%s error=%s", pid, e)
            continue

        comments = extract_comments(comments_payload)
        comments.sort(key=_comment_priority_score, reverse=True)
        my_replied_parent_ids: Set[str] = set()
        my_comment_ids_in_post: Set[str] = set()
        post_author_id, post_author_name = post_author(post)
        is_profile_recent = normalize_str(post.get("__scan_source")).strip().lower() == "profile_recent"
        is_my_post = bool(
            is_profile_recent
            or is_self_author(post_author_id, post_author_name, self_identity_keys=self_identity_keys)
        )
        if self_identity_keys:
            for maybe_reply in comments:
                maybe_id = comment_id(maybe_reply)
                maybe_author_id, maybe_author = comment_author(maybe_reply)
                parent = comment_parent_id(maybe_reply)
                if maybe_id and is_self_author(
                    maybe_author_id,
                    maybe_author,
                    self_identity_keys=self_identity_keys,
                ):
                    my_comment_ids.add(maybe_id)
                    my_comment_ids_in_post.add(maybe_id)
                if not parent:
                    continue
                if is_self_author(
                    maybe_author_id,
                    maybe_author,
                    self_identity_keys=self_identity_keys,
                ):
                    replied_comment_pairs.add(f"{normalize_str(pid).strip()}:{normalize_str(parent).strip()}")
                    replied_to_comment_ids.add(parent)
                    my_replied_parent_ids.add(parent)
        comment_author_by_id: Dict[str, str] = {}
        comment_parent_by_id: Dict[str, str] = {}
        for entry in comments:
            entry_id = comment_id(entry)
            if not entry_id:
                continue
            entry_author_id, entry_author_name = comment_author(entry)
            comment_author_by_id[entry_id] = author_identity_key(entry_author_id, entry_author_name)
            comment_parent_by_id[entry_id] = normalize_str(comment_parent_id(entry)).strip()

        replies_by_author_on_post: Dict[str, int] = {}
        conversation_turns_by_author_on_post: Dict[str, int] = {}
        if self_identity_keys:
            for entry_id, entry_author_key in comment_author_by_id.items():
                parent_id = comment_parent_by_id.get(entry_id, "")
                if not parent_id:
                    continue
                parent_author_key = comment_author_by_id.get(parent_id, "")
                if not parent_author_key or not entry_author_key:
                    continue
                is_entry_self = entry_author_key in self_identity_keys
                is_parent_self = parent_author_key in self_identity_keys
                if is_entry_self and not is_parent_self:
                    replies_by_author_on_post[parent_author_key] = replies_by_author_on_post.get(parent_author_key, 0) + 1
                    conversation_turns_by_author_on_post[parent_author_key] = (
                        conversation_turns_by_author_on_post.get(parent_author_key, 0) + 1
                    )
                elif (not is_entry_self) and is_parent_self:
                    conversation_turns_by_author_on_post[entry_author_key] = (
                        conversation_turns_by_author_on_post.get(entry_author_key, 0) + 1
                    )

        for comment in comments:
            scanned += 1
            cid = comment_id(comment)
            if not cid:
                continue

            c_author_id, c_author_name = comment_author(comment)
            c_author_key = author_identity_key(c_author_id, c_author_name)
            if my_name and _normalized_name_key(c_author_name) == _normalized_name_key(my_name):
                skip_reasons["self_comment_name_match"] = skip_reasons.get("self_comment_name_match", 0) + 1
                logger.info(
                    "Reply scan skipping self comment by explicit name match comment_id=%s author=%s",
                    cid,
                    c_author_name or "(unknown)",
                )
                continue
            if cid in my_comment_ids:
                skip_reasons["self_comment_known"] = skip_reasons.get("self_comment_known", 0) + 1
                continue
            if is_self_author(c_author_id, c_author_name, self_identity_keys=self_identity_keys):
                skip_reasons["self_comment"] = skip_reasons.get("self_comment", 0) + 1
                continue
            if not c_author_key:
                skip_reasons["unknown_author"] = skip_reasons.get("unknown_author", 0) + 1
                logger.debug("Reply scan skipping comment_id=%s reason=unknown_author", cid)
                continue
            parent_cid = comment_parent_id(comment)
            is_reply_to_me = bool(
                parent_cid
                and (
                    parent_cid in my_comment_ids
                    or parent_cid in my_comment_ids_in_post
                )
            )
            # On our own posts we triage all incoming comments.
            # On other posts we only triage replies to comments we authored.
            if not is_my_post and not is_reply_to_me:
                skip_reasons["not_target_reply"] = skip_reasons.get("not_target_reply", 0) + 1
                continue
            pair_key = f"{normalize_str(pid).strip()}:{normalize_str(cid).strip()}"
            author_post_key = f"{normalize_str(pid).strip()}:{c_author_key or '(unknown-author)'}"

            replies_to_author = replies_by_author_on_post.get(c_author_key, 0) if c_author_key else 0
            if replies_to_author >= MAX_REPLIES_PER_AUTHOR_PER_POST:
                skip_reasons["author_thread_reply_cap"] = skip_reasons.get("author_thread_reply_cap", 0) + 1
                logger.info(
                    (
                        "Reply scan skipping comment_id=%s post_id=%s reason=author_thread_reply_cap "
                        "author=%s replies_to_author=%s cap=%s"
                    ),
                    cid,
                    pid,
                    c_author_name or c_author_key or "(unknown)",
                    replies_to_author,
                    MAX_REPLIES_PER_AUTHOR_PER_POST,
                )
                continue

            turns_with_author = conversation_turns_by_author_on_post.get(c_author_key, 0) if c_author_key else 0
            post_submolt = normalize_submolt(post.get("submolt"))
            post_title = normalize_str(post.get("title")) or f"Post {pid}"
            url = post_url(pid)
            incoming_body = normalize_str(comment.get("content"))
            if looks_spammy_comment(incoming_body):
                skip_reasons["spam_comment"] = skip_reasons.get("spam_comment", 0) + 1
                logger.info("Reply scan skipping spam comment_id=%s", cid)
                continue
            if turns_with_author >= THREAD_ESCALATE_TURNS:
                if author_post_key in thread_followup_posted_pairs:
                    skip_reasons["thread_followup_already_posted"] = (
                        skip_reasons.get("thread_followup_already_posted", 0) + 1
                    )
                    logger.info(
                        "Reply scan skipping deep-thread escalation already posted post_id=%s author=%s turns=%s",
                        pid,
                        c_author_name or c_author_key or "(unknown)",
                        turns_with_author,
                    )
                    continue

                followup_title = build_thread_followup_post_title(post_title_text=post_title)
                followup_content = build_thread_followup_post_content(
                    source_url=url,
                    author_name=c_author_name or "(unknown)",
                    source_comment=incoming_body,
                    proposed_reply=(
                        "eUTXO plus ErgoScript lets us enforce deterministic settlement rules while keeping counterparties auditable."
                    ),
                )
                post_allowed, post_reason = post_gate_status(state=state, cfg=cfg)
                if not post_allowed:
                    key = f"thread_followup_{post_reason}"
                    skip_reasons[key] = skip_reasons.get(key, 0) + 1
                    logger.info(
                        (
                            "Reply scan deep-thread escalation deferred comment_id=%s post_id=%s "
                            "reason=%s turns=%s"
                        ),
                        cid,
                        pid,
                        post_reason,
                        turns_with_author,
                    )
                    continue

                approved, approve_all_actions, should_stop = confirm_action(
                    cfg=cfg,
                    logger=logger,
                    action=f"post-followup-thread-{cid}",
                    pid=pid,
                    title=followup_title,
                    submolt=post_submolt,
                    url=url,
                    author=c_author_name or "(unknown)",
                    content_preview=preview_text(followup_content),
                    approve_all=approve_all_actions,
                )
                if should_stop:
                    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                    save_state(cfg.state_path, state)
                    return approve_all_actions
                if approved:
                    try:
                        post_resp = client.create_post(submolt=post_submolt, title=followup_title, content=followup_content)
                        created_post_id = post_id(post_resp) or (post_resp.get("post") or {}).get("id") or "(unknown)"
                        state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                        mark_reply_action_timestamps(state=state, action_kind="post")
                        thread_followup_posted_pairs.add(author_post_key)
                        state["thread_followup_posted_pairs"] = list(thread_followup_posted_pairs)[-10000:]
                        replied_to_comment_ids.add(cid)
                        replied_comment_pairs.add(pair_key)
                        state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                        state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                        actions += 1
                        print_success_banner(
                            action="post-followup",
                            pid=str(created_post_id),
                            url=post_url(str(created_post_id)) if created_post_id != "(unknown)" else url,
                            title=followup_title,
                        )
                    except Exception as e:
                        logger.warning(
                            "Reply scan followup post failed source_post_id=%s comment_id=%s error=%s",
                            pid,
                            cid,
                            e,
                        )
                continue

            already_replied = (
                cid in replied_to_comment_ids
                or cid in my_replied_parent_ids
                or pair_key in replied_comment_pairs
            )
            pending_reply_exists = has_pending_comment_action(
                state=state,
                post_id_value=pid,
                parent_comment_id=cid,
            )
            if cid in seen_comment_ids and (already_replied or pending_reply_exists or not is_my_post):
                skip_reasons["already_seen"] = skip_reasons.get("already_seen", 0) + 1
                continue
            was_new_comment = False
            if cid not in seen_comment_ids:
                seen_comment_ids.add(cid)
                new_replies += 1
                was_new_comment = True

            triage: Dict[str, Any]
            triage = triage_cache.get(cid, {})
            if triage:
                triage_cache_hits += 1
            else:
                if not was_new_comment:
                    triage_deferred += 1
                    skip_reasons["triage_existing_no_cache"] = skip_reasons.get("triage_existing_no_cache", 0) + 1
                    continue
                if triage_llm_calls >= triage_llm_budget:
                    triage_deferred += 1
                    skip_reasons["triage_budget_deferred"] = skip_reasons.get("triage_budget_deferred", 0) + 1
                    if was_new_comment:
                        seen_comment_ids.discard(cid)
                        new_replies = max(0, new_replies - 1)
                    continue
                try:
                    messages = build_reply_triage_messages(
                        persona=persona_text,
                        domain_context=domain_context_text,
                        post=post,
                        comment=comment,
                        post_id=pid,
                        comment_id=cid,
                    )
                    triage, triage_provider, _ = call_generation_model(cfg, messages)
                    triage_llm_calls += 1
                    provider_counts[triage_provider] = provider_counts.get(triage_provider, 0) + 1
                    triage_cache[cid] = {
                        "should_respond": bool(triage.get("should_respond")),
                        "confidence": float(triage.get("confidence", 0.0)),
                        "response_mode": normalize_response_mode(triage.get("response_mode"), default="none"),
                        "title": normalize_str(triage.get("title")).strip(),
                        "content": normalize_str(triage.get("content")).strip(),
                        "vote_action": normalize_vote_action(triage.get("vote_action")),
                        "vote_target": normalize_vote_target(triage.get("vote_target")),
                        "ts": utc_now().isoformat(),
                    }
                    logger.debug(
                        "Reply triage generated comment_id=%s provider=%s",
                        cid,
                        triage_provider,
                    )
                except Exception as e:
                    triage_fail_count += 1
                    err_text = normalize_str(e)
                    is_parse_fail = (
                        "Chatbase returned non-JSON text" in err_text
                        or "empty text response" in err_text
                        or "Expecting value: line 1 column 1" in err_text
                    )
                    if is_parse_fail:
                        triage_parse_fail_count += 1
                        if triage_parse_fail_count <= 2:
                            logger.warning("Reply triage failed comment_id=%s error=%s", cid, e)
                        else:
                            logger.debug("Reply triage parse failure comment_id=%s error=%s", cid, e)
                    else:
                        logger.warning("Reply triage failed comment_id=%s error=%s", cid, e)
                    triage = {
                        "should_respond": False,
                        "confidence": 0.0,
                        "vote_action": "none",
                        "vote_target": "none",
                        "response_mode": "none",
                        "title": "",
                        "content": "",
                    }

            vote_action = normalize_vote_action(triage.get("vote_action"))
            response_mode = normalize_response_mode(triage.get("response_mode"), default="none")
            confidence = float(triage.get("confidence", 0))
            url = post_url(pid)
            post_submolt = normalize_submolt(post.get("submolt"))
            post_title = normalize_str(post.get("title")) or f"Post {pid}"
            incoming_body = normalize_str(comment.get("content"))

            if vote_action != "none":
                if cid in voted_comment_ids:
                    logger.info("Reply scan skipping vote; already voted comment_id=%s", cid)
                    vote_action = "none"
                existing_vote = extract_my_vote_from_comment(comment)
                if vote_action != "none" and existing_vote == vote_action:
                    logger.info(
                        "Reply scan skipping vote; API shows existing vote comment_id=%s vote=%s",
                        cid,
                        existing_vote,
                    )
                    voted_comment_ids.add(cid)
                    vote_action = "none"
                if vote_action == "downvote" and not cfg.allow_comment_downvote:
                    logger.info(
                        "Reply scan skipping unsupported vote action downvote-comment comment_id=%s",
                        cid,
                    )
                    skip_reasons["downvote_unsupported"] = skip_reasons.get("downvote_unsupported", 0) + 1
                    vote_action = "none"
                # Even when vote is skipped, still continue with reply-triage action path.
                if vote_action != "none":
                    approved, approve_all_actions, should_stop = confirm_action(
                        cfg=cfg,
                        logger=logger,
                        action=f"{vote_action}-comment",
                        pid=cid,
                        title=f"Reply on '{post_title}'",
                        submolt=normalize_submolt(post.get("submolt")),
                        url=url,
                        author=c_author_name or "(unknown)",
                        content_preview=preview_text(incoming_body),
                        approve_all=approve_all_actions,
                    )
                    if should_stop:
                        state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                        save_state(cfg.state_path, state)
                        return approve_all_actions
                    if approved:
                        try:
                            client.vote_comment(cid, vote_action=vote_action)
                            voted_comment_ids.add(cid)
                            state["voted_comment_ids"] = list(voted_comment_ids)[-10000:]
                            actions += 1
                            print_success_banner(
                                action=f"{vote_action}-comment",
                                pid=cid,
                                url=url,
                                title=f"Reply on '{post_title}'",
                            )
                        except Exception as e:
                            logger.warning("Reply vote failed comment_id=%s vote=%s error=%s", cid, vote_action, e)

            forced_reply_content = ""
            if should_correct_wrong_community_claim(incoming_body, post_submolt):
                forced_reply_content = build_wrong_community_correction_reply(
                    post_submolt=post_submolt,
                    post_title_text=post_title,
                )
                response_mode = "comment"
                confidence = max(confidence, cfg.min_confidence)
                logger.info(
                    "Reply scan forcing submolt correction comment_id=%s submolt=%s",
                    cid,
                    post_submolt,
                )

            should_respond = bool(triage.get("should_respond")) or bool(forced_reply_content)
            if already_replied:
                logger.info("Reply scan skipping reply; already replied to comment_id=%s", cid)
                replied_to_comment_ids.add(cid)
                replied_comment_pairs.add(pair_key)
                state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                skip_reasons["already_replied"] = skip_reasons.get("already_replied", 0) + 1
                continue

            reply_content = ""
            if forced_reply_content:
                reply_content = forced_reply_content
            elif should_respond and response_mode != "none" and confidence >= cfg.min_confidence:
                reply_content = format_content(triage)

            if not reply_content:
                if not should_respond:
                    skip_reasons["triage_declined"] = skip_reasons.get("triage_declined", 0) + 1
                elif response_mode == "none":
                    skip_reasons["response_mode_none"] = skip_reasons.get("response_mode_none", 0) + 1
                elif confidence < cfg.min_confidence:
                    skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1
                else:
                    skip_reasons["empty_reply_content"] = skip_reasons.get("empty_reply_content", 0) + 1
                continue

            if not should_respond:
                skip_reasons["triage_declined"] = skip_reasons.get("triage_declined", 0) + 1
            elif response_mode == "none":
                skip_reasons["response_mode_none"] = skip_reasons.get("response_mode_none", 0) + 1
            elif confidence < cfg.min_confidence:
                skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1

            if not reply_content.strip():
                skip_reasons["empty_reply_content"] = skip_reasons.get("empty_reply_content", 0) + 1
                continue
            if not forced_reply_content and _is_template_like_generated_content(reply_content):
                skip_reasons["template_like_reply"] = skip_reasons.get("template_like_reply", 0) + 1
                logger.info("Reply scan skipping template-like reply comment_id=%s", cid)
                continue
            if not forced_reply_content and _is_low_value_affirmation_reply(reply_content):
                skip_reasons["low_value_reply"] = skip_reasons.get("low_value_reply", 0) + 1
                logger.info("Reply scan skipping low-value reply comment_id=%s", cid)
                continue
            if not forced_reply_content and not _passes_generated_content_quality(
                content=reply_content,
                requested_mode="comment",
            ):
                skip_reasons["reply_quality_gate"] = skip_reasons.get("reply_quality_gate", 0) + 1
                logger.info("Reply scan skipping low-quality reply comment_id=%s", cid)
                continue

            if has_my_reply_to_comment(
                client=client,
                post_id_value=pid,
                parent_comment_id=cid,
                my_name=my_name,
                logger=logger,
                self_identity_keys=self_identity_keys,
            ):
                replied_to_comment_ids.add(cid)
                replied_comment_pairs.add(pair_key)
                state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                skip_reasons["already_replied_onchain"] = skip_reasons.get("already_replied_onchain", 0) + 1
                logger.info("Reply scan skipping reply; on-chain reply already exists comment_id=%s", cid)
                continue
            if c_author_key and c_author_key in self_identity_keys:
                skip_reasons["self_comment_late_guard"] = skip_reasons.get("self_comment_late_guard", 0) + 1
                logger.info("Reply scan late-guard skip comment_id=%s reason=self_comment", cid)
                continue

            comment_allowed, comment_reason = comment_gate_status(state=state, cfg=cfg)
            if comment_allowed:
                approved, approve_all_actions, should_stop = confirm_action(
                    cfg=cfg,
                    logger=logger,
                    action=f"comment-reply-to-{cid}",
                    pid=pid,
                    title=post_title,
                    submolt=normalize_submolt(post.get("submolt")),
                    url=url,
                    author=c_author_name or "(unknown)",
                    content_preview=preview_text(reply_content),
                    approve_all=approve_all_actions,
                )
                if should_stop:
                    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                    save_state(cfg.state_path, state)
                    return approve_all_actions
                if approved:
                    try:
                        comment_resp = client.create_comment(pid, reply_content, parent_id=cid)
                        register_my_comment_id(state=state, response_payload=comment_resp)
                        state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                        mark_reply_action_timestamps(state=state, action_kind="comment")
                        replied_post_ids.add(pid)
                        state["replied_post_ids"] = list(replied_post_ids)[-10000:]
                        maybe_upvote_post_after_comment(
                            client=client,
                            state=state,
                            logger=logger,
                            post_id_value=pid,
                        )
                        replied_to_comment_ids.add(cid)
                        replied_comment_pairs.add(pair_key)
                        state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                        state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                        if c_author_key:
                            replies_by_author_on_post[c_author_key] = replies_by_author_on_post.get(c_author_key, 0) + 1
                            conversation_turns_by_author_on_post[c_author_key] = (
                                conversation_turns_by_author_on_post.get(c_author_key, 0) + 1
                            )
                        actions += 1
                        print_success_banner(action="comment-reply", pid=pid, url=url, title=post_title)
                    except Exception as e:
                        logger.warning("Reply comment failed post_id=%s error=%s", pid, e)
                continue

            if comment_reason == "comment_cooldown":
                approved, approve_all_actions, should_stop = confirm_action(
                    cfg=cfg,
                    logger=logger,
                    action=f"wait-comment-reply-to-{cid}",
                    pid=pid,
                    title=post_title,
                    submolt=normalize_submolt(post.get("submolt")),
                    url=url,
                    author=c_author_name or "(unknown)",
                    content_preview=preview_text(reply_content),
                    approve_all=approve_all_actions,
                )
                if should_stop:
                    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                    save_state(cfg.state_path, state)
                    return approve_all_actions
                if approved:
                    if wait_for_comment_slot(state=state, cfg=cfg, logger=logger):
                        try:
                            comment_resp = client.create_comment(pid, reply_content, parent_id=cid)
                            register_my_comment_id(state=state, response_payload=comment_resp)
                            state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                            mark_reply_action_timestamps(state=state, action_kind="comment")
                            replied_post_ids.add(pid)
                            state["replied_post_ids"] = list(replied_post_ids)[-10000:]
                            replied_to_comment_ids.add(cid)
                            replied_comment_pairs.add(pair_key)
                            state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                            state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                            if c_author_key:
                                replies_by_author_on_post[c_author_key] = replies_by_author_on_post.get(c_author_key, 0) + 1
                                conversation_turns_by_author_on_post[c_author_key] = (
                                    conversation_turns_by_author_on_post.get(c_author_key, 0) + 1
                                )
                            maybe_upvote_post_after_comment(
                                client=client,
                                state=state,
                                logger=logger,
                                post_id_value=pid,
                            )
                            actions += 1
                            print_success_banner(action="comment-reply", pid=pid, url=url, title=post_title)
                        except Exception as e:
                            logger.warning("Waited reply comment failed post_id=%s error=%s", pid, e)
                continue

            skip_reasons[comment_reason] = skip_reasons.get(comment_reason, 0) + 1

    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
    state["my_comment_ids"] = list(my_comment_ids)[-20000:]
    state["replied_post_ids"] = list(replied_post_ids)[-10000:]
    state["voted_comment_ids"] = list(voted_comment_ids)[-10000:]
    state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
    state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
    state["thread_followup_posted_pairs"] = list(thread_followup_posted_pairs)[-10000:]
    state["reply_triage_cache"] = dict(list(triage_cache.items())[-10000:])
    save_state(cfg.state_path, state)
    logger.info(
        "Reply scan complete scanned_comments=%s new_replies=%s actions=%s pending=%s",
        scanned,
        new_replies,
        actions,
        len(state.get("pending_actions", [])),
    )
    if provider_counts:
        provider_summary = ", ".join([f"{k}={v}" for k, v in sorted(provider_counts.items())])
        logger.info("Reply scan provider_summary %s", provider_summary)
    if skip_reasons:
        summary = ", ".join([f"{k}={v}" for k, v in sorted(skip_reasons.items())])
        logger.info("Reply scan skip_summary %s", summary)
    if triage_fail_count:
        logger.info(
            "Reply scan triage_failures total=%s parse_format=%s",
            triage_fail_count,
            triage_parse_fail_count,
        )
    logger.info(
        "Reply scan triage_usage cache_hits=%s llm_calls=%s deferred=%s budget=%s",
        triage_cache_hits,
        triage_llm_calls,
        triage_deferred,
        triage_llm_budget,
    )
    return approve_all_actions


def review_pending_keyword_suggestions(
    cfg: Config,
    logger,
    keyword_store: Dict[str, Any],
    active_keywords: List[str],
    approve_all_keyword_changes: bool,
) -> Tuple[List[str], Dict[str, Any], bool, bool]:
    pending = list(keyword_store.get("pending_suggestions", []))
    if not pending:
        return active_keywords, keyword_store, approve_all_keyword_changes, False

    logger.info("Pending keyword suggestions awaiting review count=%s", len(pending))
    remaining: List[str] = []
    should_stop_run = False
    for keyword in pending:
        approved, approve_all_keyword_changes, should_stop = confirm_keyword_addition(
            logger=logger,
            keyword=keyword,
            approve_all=approve_all_keyword_changes,
        )
        if should_stop:
            # Keep unreviewed suggestions for next manual run.
            remaining.append(keyword)
            should_stop_run = True
            continue
        if not approved:
            continue
        learned_before = keyword_store.get("learned_keywords", [])
        learned_after = merge_keywords(learned_before, [keyword])
        if len(learned_after) == len(learned_before):
            continue
        keyword_store["learned_keywords"] = learned_after
        active_keywords = merge_keywords(cfg.keywords, learned_after)
        logger.info(
            "Keyword approved from pending keyword=%s learned_total=%s active_total=%s",
            keyword,
            len(learned_after),
            len(active_keywords),
        )
        print_keyword_added_banner(keyword=keyword, learned_total=len(learned_after))

    keyword_store["pending_suggestions"] = merge_keywords([], remaining)
    save_keyword_store(cfg.keyword_store_path, keyword_store)
    return active_keywords, keyword_store, approve_all_keyword_changes, should_stop_run


def _deterministic_improvement_hints(
    diagnostics: Dict[str, Any],
    cycle_stats: Dict[str, Any],
    learning_snapshot: Dict[str, Any],
) -> List[str]:
    hints: List[str] = []
    bottleneck = normalize_str(diagnostics.get("bottleneck_label")).strip()
    approval_rate = float(diagnostics.get("approval_rate", 0.0) or 0.0)
    execution_rate = float(diagnostics.get("execution_rate", 0.0) or 0.0)
    drafted = int(diagnostics.get("drafted", 0) or 0)
    eligible_now = int(diagnostics.get("eligible_now", 0) or 0)
    actions = int(diagnostics.get("actions", 0) or 0)

    if bottleneck == "model_rejection" or (drafted >= 12 and approval_rate < 0.1):
        hints.append("Model approval is weak. Prioritize relevance filters and stricter reject criteria before drafting.")
    if bottleneck == "execution_blocked" or (eligible_now >= 20 and actions == 0):
        hints.append("Execution conversion is weak. Focus on reducing non-actionable drafts and improving action readiness.")
    if bottleneck == "cooldown_limited":
        hints.append("Cooldown pressure is high. Prioritize comments or defer post-like actions while preserving discovery.")
    if bottleneck == "duplication_pressure":
        hints.append("Duplication pressure detected. Tighten dedupe/reply-once checks before generating new replies.")
    if execution_rate < 0.05 and eligible_now >= 15:
        hints.append("Eligible volume is high but execution is near zero. Add shortlist ranking before LLM calls.")

    market_snapshot = learning_snapshot.get("market_snapshot")
    if isinstance(market_snapshot, dict):
        q_rate = market_snapshot.get("question_title_rate")
        if isinstance(q_rate, (int, float)) and q_rate >= 0.35:
            hints.append("Question-style titles are trending. Prefer direct question hooks in proactive posts.")
        top_terms = market_snapshot.get("top_terms")
        if isinstance(top_terms, list) and top_terms:
            joined = ", ".join([normalize_str(t).strip() for t in top_terms[:6] if normalize_str(t).strip()])
            if joined:
                hints.append(f"Top market terms now: {joined}. Use them only when context actually fits.")
    visibility_metrics = learning_snapshot.get("visibility_metrics")
    if isinstance(visibility_metrics, dict):
        target_upvotes = int(visibility_metrics.get("target_upvotes", 0) or 0)
        hit_rate = float(visibility_metrics.get("recent_target_hit_rate", 0.0) or 0.0)
        delta_pct = float(visibility_metrics.get("visibility_delta_pct", 0.0) or 0.0)
        if target_upvotes > 0 and hit_rate < 0.35:
            hints.append(
                (
                    "Visibility under target. Raise opening-hook specificity and implementation detail density "
                    f"until recent target hit rate improves (target_upvotes={target_upvotes}, hit_rate={round(hit_rate, 3)})."
                )
            )
        if delta_pct <= -0.15:
            hints.append(
                f"Visibility momentum is negative ({round(delta_pct * 100, 1)}%). Prioritize high-lift themes only."
            )

    raw_skip = cycle_stats.get("skip_reasons")
    if isinstance(raw_skip, dict):
        items = sorted(raw_skip.items(), key=lambda x: int(x[1]), reverse=True)
        if items:
            label, count = items[0]
            hints.append(f"Dominant skip reason this cycle: {normalize_str(label)}={int(count)}.")
        if int(raw_skip.get("quality_gate_failed", 0) or 0) >= 3:
            hints.append(
                "Many drafts fail the quality gate. Tighten prompt specificity and keep mechanism-first phrasing."
            )
        if int(raw_skip.get("trend_context_mismatch", 0) or 0) >= 5:
            hints.append(
                "Trend/context mismatch is high. Favor candidates with both market-term overlap and clear Ergo mechanism."
            )
    return hints[:8]


def _deterministic_improvement_suggestions(
    diagnostics: Dict[str, Any],
    cycle_stats: Dict[str, Any],
    learning_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    hints = _deterministic_improvement_hints(
        diagnostics=diagnostics,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
    )
    prompt_changes: List[Dict[str, Any]] = []
    code_changes: List[Dict[str, Any]] = []
    strategy_experiments: List[Dict[str, Any]] = []

    bottleneck = normalize_str(diagnostics.get("bottleneck_label")).strip()
    approval_rate = float(diagnostics.get("approval_rate", 0.0) or 0.0)
    drafted = int(diagnostics.get("drafted", 0) or 0)
    eligible_now = int(diagnostics.get("eligible_now", 0) or 0)
    actions = int(diagnostics.get("actions", 0) or 0)
    execution_rate = float(diagnostics.get("execution_rate", 0.0) or 0.0)

    if bottleneck == "model_rejection" or (drafted >= 12 and approval_rate < 0.1):
        prompt_changes.append(
            {
                "target": "drafting relevance gate",
                "proposed_change": (
                    "Require one explicit Ergo mechanism plus one thread-specific implementation angle before should_respond=true."
                ),
                "reason": "Low model approval means current candidates are too broad or generic.",
                "expected_impact": "Higher approval rate and fewer wasted drafts.",
            }
        )
    if eligible_now >= 20 and actions == 0:
        code_changes.append(
            {
                "file_hint": "src/moltbook/autonomy/runner.py",
                "proposed_change": (
                    "Add pre-draft shortlist ranking so only top N eligible candidates reach the LLM each cycle."
                ),
                "reason": "High eligible volume with zero actions indicates poor conversion efficiency.",
                "risk": "May miss edge-case opportunities if shortlist is too small.",
            }
        )
    if bottleneck == "duplication_pressure":
        code_changes.append(
            {
                "file_hint": "src/moltbook/autonomy/runner.py",
                "proposed_change": (
                    "Strengthen reply dedupe by caching parent-comment fingerprints with longer retention in state."
                ),
                "reason": "Repeated reply targets hurt trust and consume action budget.",
                "risk": "Over-aggressive dedupe may skip valid follow-up contexts.",
            }
        )
    if execution_rate < 0.08 and eligible_now >= 15:
        strategy_experiments.append(
            {
                "idea": "Enable dynamic shortlist size based on last 3-cycle approval/execution rates.",
                "metric": "execution_rate and actions per cycle after shortlist enabled",
                "stop_condition": "Disable if execution_rate does not improve after 12 cycles",
            }
        )
    visibility_metrics = learning_snapshot.get("visibility_metrics")
    if isinstance(visibility_metrics, dict):
        hit_rate = float(visibility_metrics.get("recent_target_hit_rate", 0.0) or 0.0)
        target_upvotes = int(visibility_metrics.get("target_upvotes", 0) or 0)
        if target_upvotes > 0 and hit_rate < 0.35:
            prompt_changes.append(
                {
                    "target": "visibility targeting",
                    "proposed_change": (
                        "Require proactive drafts to open with one concrete pain point plus one Ergo mechanism in the first two lines."
                    ),
                    "reason": "Recent posts are underperforming the target upvote threshold.",
                    "expected_impact": (
                        f"Higher share of posts crossing {target_upvotes}+ upvotes by improving hook clarity."
                    ),
                }
            )
            code_changes.append(
                {
                    "file_hint": "src/moltbook/autonomy/runner.py",
                    "proposed_change": (
                        "Bias ranking toward terms with positive lift from proactive memory and penalize repeated low-lift terms."
                    ),
                    "reason": "Visibility target hit rate is low, so selection should follow measured term lift.",
                    "risk": "Overfitting to short-term language trends can reduce topic diversity.",
                }
            )
    raw_skip = cycle_stats.get("skip_reasons")
    if isinstance(raw_skip, dict):
        if int(raw_skip.get("quality_gate_failed", 0) or 0) >= 3:
            prompt_changes.append(
                {
                    "target": "draft content shape",
                    "proposed_change": (
                        "Require one concrete Ergo mechanism sentence before any question; reject abstract framing."
                    ),
                    "reason": "Quality gate failures indicate drafts are still too generic.",
                    "expected_impact": "Higher quality-pass rate and fewer dropped drafts.",
                }
            )

    summary = (
        "Deterministic diagnostics suggest conversion-focused tuning."
        if hints
        else "Deterministic diagnostics found no additional changes."
    )
    return {
        "summary": summary,
        "priority": "medium",
        "prompt_changes": prompt_changes,
        "code_changes": code_changes,
        "strategy_experiments": strategy_experiments,
    }


def _suggestion_signature(kind: str, item: Dict[str, Any]) -> str:
    if kind == "prompt_changes":
        raw = " ".join([normalize_str(item.get("target")), normalize_str(item.get("proposed_change"))])
    elif kind == "code_changes":
        raw = " ".join([normalize_str(item.get("file_hint")), normalize_str(item.get("proposed_change"))])
    else:
        raw = " ".join(
            [
                normalize_str(item.get("idea")),
                normalize_str(item.get("metric")),
                normalize_str(item.get("stop_condition")),
            ]
        )
    raw = re.sub(r"[^a-z0-9]+", " ", raw.lower())
    return " ".join(raw.split())[:260]


def _merge_improvement_payloads(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    max_items: int,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "summary": normalize_str(primary.get("summary")).strip() or normalize_str(fallback.get("summary")).strip(),
        "priority": normalize_str(primary.get("priority")).strip() or "medium",
        "prompt_changes": [],
        "code_changes": [],
        "strategy_experiments": [],
    }
    max_items = max(1, int(max_items))
    for kind in ("prompt_changes", "code_changes", "strategy_experiments"):
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for source in (primary, fallback):
            raw = source.get(kind)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                sig = _suggestion_signature(kind, item)
                if not sig or sig in seen:
                    continue
                seen.add(sig)
                out.append(item)
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break
        merged[kind] = out
    return merged


def maybe_write_self_improvement_suggestions(
    cfg: Config,
    logger,
    iteration: int,
    persona_text: str,
    domain_context_text: str,
    learning_snapshot: Dict[str, Any],
    cycle_titles: List[str],
    cycle_stats: Dict[str, Any],
) -> None:
    if not cfg.self_improve_enabled:
        return
    if cfg.self_improve_interval_cycles <= 0:
        return
    if iteration % cfg.self_improve_interval_cycles != 0:
        return
    if len(cycle_titles) < cfg.self_improve_min_titles:
        logger.info(
            "Self-improvement skipped cycle=%s reason=insufficient_titles titles=%s min_titles=%s",
            iteration,
            len(cycle_titles),
            cfg.self_improve_min_titles,
        )
        return
    use_llm = has_generation_provider(cfg)
    if not use_llm:
        logger.info(
            "Self-improvement cycle=%s running deterministic-only mode reason=no_generation_provider",
            iteration,
        )

    diagnostics = build_improvement_diagnostics(cycle_stats)
    feedback_context = build_improvement_feedback_context(
        path=cfg.self_improve_path,
        current_cycle_stats=cycle_stats,
    )
    deterministic_hints = _deterministic_improvement_hints(
        diagnostics=diagnostics,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
    )
    deterministic_payload = _deterministic_improvement_suggestions(
        diagnostics=diagnostics,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
    )

    provider_used = "unknown"
    llm_payload: Dict[str, Any] = {}
    if use_llm:
        try:
            prior_suggestions = load_recent_improvement_entries(path=cfg.self_improve_path, limit=8)
            messages = build_self_improvement_messages(
                persona=persona_text,
                domain_context=domain_context_text,
                learning_snapshot=learning_snapshot,
                recent_titles=cycle_titles,
                cycle_stats=cycle_stats,
                prior_suggestions=prior_suggestions,
                feedback_context=feedback_context,
                deterministic_hints=deterministic_hints,
            )
            generated, provider_used, _ = call_generation_model(cfg, messages)
            if isinstance(generated, dict):
                llm_payload = generated
            else:
                logger.warning("Self-improvement returned non-object payload cycle=%s", iteration)
                provider_used = "deterministic"
        except Exception as e:
            logger.warning("Self-improvement failed cycle=%s error=%s", iteration, e)
            provider_used = "deterministic"
            llm_payload = {}
    else:
        provider_used = "deterministic"

    max_suggestions = max(1, cfg.self_improve_max_suggestions)
    # Keep a longer novelty memory so the same recommendation is not repeated every few cycles.
    recent_raw = load_recent_improvement_raw_entries(path=cfg.self_improve_path, limit=72)
    llm_suggestions = sanitize_improvement_suggestions(
        suggestions=llm_payload,
        recent_raw_entries=recent_raw,
        max_items=max_suggestions,
    )
    combined_payload = _merge_improvement_payloads(
        primary=llm_suggestions,
        fallback=deterministic_payload,
        max_items=max_suggestions,
    )
    suggestions = sanitize_improvement_suggestions(
        suggestions=combined_payload,
        recent_raw_entries=recent_raw,
        max_items=max_suggestions,
    )
    if isinstance(suggestions.get("prompt_changes"), list):
        suggestions["prompt_changes"] = suggestions["prompt_changes"][:max_suggestions]
    if isinstance(suggestions.get("code_changes"), list):
        suggestions["code_changes"] = suggestions["code_changes"][:max_suggestions]
    if isinstance(suggestions.get("strategy_experiments"), list):
        suggestions["strategy_experiments"] = suggestions["strategy_experiments"][:max_suggestions]

    prompt_changes = suggestions.get("prompt_changes")
    code_changes = suggestions.get("code_changes")
    strategy_experiments = suggestions.get("strategy_experiments")
    if not any(
        (
            isinstance(prompt_changes, list) and prompt_changes,
            isinstance(code_changes, list) and code_changes,
            isinstance(strategy_experiments, list) and strategy_experiments,
        )
    ):
        logger.info(
            "Self-improvement produced no novel actionable suggestions cycle=%s bottleneck=%s",
            iteration,
            diagnostics.get("bottleneck_label"),
        )
        return

    append_improvement_suggestions(
        path=cfg.self_improve_path,
        cycle=iteration,
        provider=provider_used,
        suggestions=suggestions,
        cycle_stats=cycle_stats,
        diagnostics=diagnostics,
    )
    append_improvement_suggestions_text(
        path=cfg.self_improve_text_path,
        cycle=iteration,
        provider=provider_used,
        suggestions=suggestions,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
        diagnostics=diagnostics,
        feedback_context=feedback_context,
    )
    update_improvement_backlog(
        path=cfg.self_improve_backlog_path,
        cycle=iteration,
        provider=provider_used,
        suggestions=suggestions,
        diagnostics=diagnostics,
    )
    logger.info(
        (
            "Self-improvement suggestions saved cycle=%s provider=%s json_path=%s "
            "text_path=%s backlog_path=%s bottleneck=%s"
        ),
        iteration,
        provider_used,
        cfg.self_improve_path,
        cfg.self_improve_text_path,
        cfg.self_improve_backlog_path,
        diagnostics.get("bottleneck_label"),
    )


def discover_relevant_submolts(payload: Dict[str, Any], target_submolts: List[str]) -> List[str]:
    target = {s.strip().lower() for s in target_submolts if s.strip()}
    keywords = ("crypto", "defi", "web3", "ai", "agent", "ergo", "blockchain", "bitcoin")
    discovered: List[str] = []
    for item in extract_submolts(payload):
        name = normalize_str(item.get("name") or item.get("slug")).strip().lower()
        if not name:
            continue
        if name in target:
            discovered.append(name)
            continue
        blob = " ".join(
            [
                name,
                normalize_str(item.get("display_name")).lower(),
                normalize_str(item.get("description")).lower(),
            ]
        )
        if any(k in blob for k in keywords):
            discovered.append(name)
    out: List[str] = []
    seen = set()
    for name in discovered:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def maybe_subscribe_relevant_submolts(
    client: MoltbookClient,
    cfg: Config,
    logger,
    state: Dict[str, Any],
    approve_all_actions: bool,
) -> bool:
    if not cfg.auto_subscribe_submolts:
        return approve_all_actions

    try:
        payload = client.list_submolts()
    except Exception as e:
        logger.warning("Submolt discovery skipped: %s", e)
        return approve_all_actions

    candidates_all = discover_relevant_submolts(payload=payload, target_submolts=cfg.target_submolts)
    approved_submolts = {normalize_str(x).strip().lower() for x in state.get("approved_submolts", [])}
    dismissed_submolts = {normalize_str(x).strip().lower() for x in state.get("dismissed_submolts", [])}
    target_set = {x.strip().lower() for x in cfg.target_submolts if x.strip()}
    prioritized = [name for name in candidates_all if name in target_set]
    extras = [name for name in candidates_all if name not in target_set][:5]
    candidates = [name for name in prioritized + extras if name not in approved_submolts and name not in dismissed_submolts]
    if not candidates:
        logger.info("Submolt discovery found no relevant candidates.")
        return approve_all_actions

    logger.info("Relevant submolts discovered count=%s", len(candidates))
    for submolt in candidates:
        approved, approve_all_actions, should_stop = confirm_action(
            cfg=cfg,
            logger=logger,
            action="subscribe-submolt",
            pid=submolt,
            title=f"Subscribe to m/{submolt}",
            submolt=submolt,
            url=f"https://moltbook.com/m/{submolt}",
            author="system",
            content_preview=preview_text(
                "Subscribe so discovery/feed includes this community more reliably."
            ),
            approve_all=approve_all_actions,
        )
        if should_stop:
            return approve_all_actions
        if not approved:
            dismissed_submolts.add(submolt)
            state["dismissed_submolts"] = sorted(dismissed_submolts)
            continue
        try:
            client.subscribe_submolt(submolt)
            approved_submolts.add(submolt)
            dismissed_submolts.discard(submolt)
            state["approved_submolts"] = sorted(approved_submolts)
            state["dismissed_submolts"] = sorted(dismissed_submolts)
            logger.info("Subscribed to submolt=%s", submolt)
            print_success_banner(
                action="subscribe-submolt",
                pid=submolt,
                url=f"https://moltbook.com/m/{submolt}",
                title=f"m/{submolt}",
            )
        except Exception as e:
            msg = str(e).lower()
            if "already" in msg:
                approved_submolts.add(submolt)
                dismissed_submolts.discard(submolt)
                state["approved_submolts"] = sorted(approved_submolts)
                state["dismissed_submolts"] = sorted(dismissed_submolts)
                logger.info("Already subscribed to submolt=%s", submolt)
                continue
            logger.warning("Subscribe failed submolt=%s error=%s", submolt, e)

    return approve_all_actions


def run_loop() -> None:
    cfg = load_config()
    logger = setup_logging(cfg)
    client = MoltbookClient()
    print_runtime_banner(cfg)

    logger.info(
        (
            "Autonomy loop starting discovery_mode=%s reply_mode=%s poll_seconds=%s feed_limit=%s "
            "search_limit=%s idle_poll_seconds=%s dry_run=%s draft_shortlist=%s draft_signal_min_score=%s "
            "dynamic_shortlist=%s dynamic_shortlist_bounds=%s-%s proactive_daily_target=%s "
            "proactive_force_general=%s llm_provider=%s openai_configured=%s "
            "chatbase_configured=%s auto_openai_fallback=%s self_improve_enabled=%s state_path=%s"
        ),
        cfg.discovery_mode,
        cfg.reply_mode,
        cfg.poll_seconds,
        cfg.feed_limit,
        cfg.search_limit,
        cfg.idle_poll_seconds,
        cfg.dry_run,
        cfg.draft_shortlist_size,
        cfg.draft_signal_min_score,
        cfg.dynamic_shortlist_enabled,
        cfg.dynamic_shortlist_min,
        cfg.dynamic_shortlist_max,
        cfg.proactive_daily_target_posts,
        cfg.proactive_force_general_until_daily_target,
        cfg.llm_provider,
        bool(cfg.openai_api_key),
        bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id),
        cfg.llm_auto_fallback_to_openai,
        cfg.self_improve_enabled,
        cfg.state_path,
    )
    if cfg.log_path:
        logger.info("File logging enabled path=%s", cfg.log_path)
    if cfg.self_improve_enabled:
        logger.info(
            "Self-improvement suggestions paths json=%s text=%s backlog=%s",
            cfg.self_improve_path,
            cfg.self_improve_text_path,
            cfg.self_improve_backlog_path,
        )

    try:
        claim_status = client.get_claim_status()
        if claim_status not in {"claimed", "active"}:
            logger.error(
                "Agent is not claim-ready (status=%s). Complete manual claim before running autonomy.",
                claim_status,
            )
            return
    except Exception as e:
        logger.warning("Could not verify claim status at startup: %s", e)

    my_name = resolve_self_name(client, logger)
    if my_name:
        logger.info("Authenticated as agent=%s", my_name)
    elif cfg.agent_name_hint:
        my_name = cfg.agent_name_hint
        logger.info("Using configured agent name hint=%s", my_name)

    persona_text = load_persona_text(cfg.persona_path)
    domain_context_text = load_context_text(cfg.context_path)
    keyword_store = load_keyword_store(cfg.keyword_store_path)
    active_keywords = merge_keywords(cfg.keywords, keyword_store.get("learned_keywords", []))
    manual_keyword_review = bool(cfg.confirm_actions and cfg.confirm_timeout_seconds <= 0)
    logger.info(
        (
            "Loaded persona guidance path=%s context_path=%s context_chars=%s "
            "keywords=%s learned_keywords=%s mission_queries=%s do_not_reply_authors=%s"
        ),
        cfg.persona_path,
        cfg.context_path,
        len(domain_context_text),
        len(active_keywords),
        len(keyword_store.get("learned_keywords", [])),
        len(cfg.mission_queries),
        len(cfg.do_not_reply_authors),
    )
    if manual_keyword_review and keyword_store.get("pending_suggestions"):
        active_keywords, keyword_store, _, should_stop = review_pending_keyword_suggestions(
            cfg=cfg,
            logger=logger,
            keyword_store=keyword_store,
            active_keywords=active_keywords,
            approve_all_keyword_changes=False,
        )
        if should_stop:
            logger.info("Stopping run after operator quit during pending keyword review.")
            return

    state = load_state(cfg.state_path)
    post_memory = load_post_engine_memory(cfg.proactive_memory_path)
    seen: Set[str] = set(state.get("seen_post_ids", []))
    replied_posts: Set[str] = set(state.get("replied_post_ids", []))
    logger.info(
        "Loaded state seen_posts=%s replied_posts=%s proactive_posts=%s",
        len(seen),
        len(replied_posts),
        len(post_memory.get("proactive_posts", [])),
    )
    if cfg.confirm_actions:
        logger.info("Interactive confirmation enabled for outgoing actions.")
    else:
        logger.info("Interactive confirmation disabled (autonomous send mode).")

    iteration = 0
    approve_all_actions = False
    approve_all_keyword_changes = False
    startup_priority_post_sent = False
    if should_prioritize_proactive_post(state=state, cfg=cfg):
        since_last_post = seconds_since_last_post(state=state)
        logger.info(
            "Startup proactive priority trigger post_slot_open=true last_post_age_seconds=%s",
            since_last_post if since_last_post is not None else -1,
        )
        proactive_actions, should_stop, approve_all_actions = maybe_run_proactive_post(
            client=client,
            cfg=cfg,
            logger=logger,
            state=state,
            post_memory=post_memory,
            my_name=my_name,
            persona_text=persona_text,
            domain_context_text=domain_context_text,
            approve_all_actions=approve_all_actions,
        )
        save_state(cfg.state_path, state)
        save_post_engine_memory(cfg.proactive_memory_path, post_memory)
        if should_stop:
            return
        if proactive_actions > 0:
            startup_priority_post_sent = True
            logger.info("Startup proactive priority posted_before_reply_scan=1")

    approve_all_actions = maybe_subscribe_relevant_submolts(
        client=client,
        cfg=cfg,
        logger=logger,
        state=state,
        approve_all_actions=approve_all_actions,
    )
    save_state(cfg.state_path, state)

    if startup_priority_post_sent:
        logger.info("Startup reply scan deferred reason=priority_post_sent")
    elif should_prioritize_proactive_post(state=state, cfg=cfg):
        logger.info("Startup reply scan deferred reason=priority_post_pending")
    else:
        approve_all_actions = run_startup_reply_scan(
            client=client,
            cfg=cfg,
            logger=logger,
            state=state,
            my_name=my_name,
            persona_text=persona_text,
            domain_context_text=domain_context_text,
            approve_all_actions=approve_all_actions,
        )
    search_state: Dict[str, Any] = {"retry_cycle": 1, "keyword_cursor": 0}
    while True:
        iteration += 1
        try:
            if not my_name:
                my_name = resolve_self_name(client, logger)
                if my_name:
                    logger.info("Resolved agent identity mid-run=%s", my_name)

            print_cycle_banner(iteration=iteration, mode=cfg.discovery_mode, keywords=len(active_keywords))
            logger.info("Poll cycle=%s start", iteration)
            priority_post_required = should_prioritize_proactive_post(state=state, cfg=cfg)
            pending_executed = 0
            if priority_post_required:
                logger.info("Cycle=%s pending actions deferred reason=priority_post_pending", iteration)
            else:
                pending_executed = execute_pending_actions(
                    client=client,
                    cfg=cfg,
                    state=state,
                    logger=logger,
                    my_name=my_name,
                )
                if pending_executed:
                    save_state(cfg.state_path, state)
            early_proactive_actions = 0
            early_post_action_sent = False
            if should_prioritize_proactive_post(state=state, cfg=cfg):
                since_last_post = seconds_since_last_post(state=state)
                logger.info(
                    "Cycle=%s proactive priority trigger post_slot_open=true last_post_age_seconds=%s",
                    iteration,
                    since_last_post if since_last_post is not None else -1,
                )
                proactive_actions, should_stop, approve_all_actions = maybe_run_proactive_post(
                    client=client,
                    cfg=cfg,
                    logger=logger,
                    state=state,
                    post_memory=post_memory,
                    my_name=my_name,
                    persona_text=persona_text,
                    domain_context_text=domain_context_text,
                    approve_all_actions=approve_all_actions,
                )
                if should_stop:
                    return
                if proactive_actions > 0:
                    early_proactive_actions += proactive_actions
                    early_post_action_sent = True
                    save_state(cfg.state_path, state)
                    save_post_engine_memory(cfg.proactive_memory_path, post_memory)
            if priority_post_required and not early_post_action_sent:
                logger.info(
                    "Cycle=%s priority_post_pending=true defer_reply_scan_and_discovery=1",
                    iteration,
                )
                save_state(cfg.state_path, state)
                save_post_engine_memory(cfg.proactive_memory_path, post_memory)
                sleep_seconds = max(1, cfg.idle_poll_seconds)
                sleep_reason = "priority_post_pending"
                logger.info("Sleeping seconds=%s reason=%s", sleep_seconds, sleep_reason)
                time.sleep(sleep_seconds)
                continue
            if cfg.startup_reply_scan_enabled and cfg.reply_scan_interval_cycles > 0:
                if iteration % cfg.reply_scan_interval_cycles == 0:
                    if early_post_action_sent:
                        logger.info(
                            "Reply scan skipped cycle=%s reason=priority_post_sent",
                            iteration,
                        )
                    else:
                        logger.info(
                            "Reply scan trigger cycle=%s interval=%s",
                            iteration,
                            cfg.reply_scan_interval_cycles,
                        )
                        approve_all_actions = run_startup_reply_scan(
                            client=client,
                            cfg=cfg,
                            logger=logger,
                            state=state,
                            my_name=my_name,
                            persona_text=persona_text,
                            domain_context_text=domain_context_text,
                            approve_all_actions=approve_all_actions,
                        )
            posts, sources = discover_posts(
                client=client,
                cfg=cfg,
                logger=logger,
                keywords=active_keywords,
                iteration=iteration,
                search_state=search_state,
            )
            logger.info("Poll cycle=%s discovered_posts=%s sources=%s", iteration, len(posts), ",".join(sources))
            learning_snapshot_cycle = build_learning_snapshot(post_memory, max_examples=5)
            posts, relevance_score_by_post, high_signal_terms = _rank_posts_for_drafting(
                posts=posts,
                learning_snapshot=learning_snapshot_cycle,
                active_keywords=active_keywords,
            )
            effective_shortlist_size, effective_signal_min_score, shortlist_mode = _adaptive_draft_controls(
                cfg=cfg,
                state=state,
            )
            market_snapshot_cycle = learning_snapshot_cycle.get("market_snapshot")
            trending_terms_cycle: List[str] = []
            if isinstance(market_snapshot_cycle, dict):
                raw_terms = market_snapshot_cycle.get("top_terms")
                if isinstance(raw_terms, list):
                    trending_terms_cycle = [
                        _clean_signal_term(item)
                        for item in raw_terms
                        if _clean_signal_term(item)
                    ][:8]
            if posts:
                top_rank_preview: List[str] = []
                for post in posts[:5]:
                    pid = post_id(post) or "(unknown)"
                    score = relevance_score_by_post.get(pid, 0)
                    title_preview = normalize_str(post.get("title")).strip()[:48] or "(untitled)"
                    top_rank_preview.append(f"{score}:{pid}:{title_preview}")
                logger.info(
                    "Poll cycle=%s relevance_ranking terms=%s threshold=%s shortlist=%s mode=%s top=%s",
                    iteration,
                    len(high_signal_terms),
                    effective_signal_min_score,
                    effective_shortlist_size,
                    shortlist_mode,
                    " | ".join(top_rank_preview),
                )

            inspected = 0
            new_candidates = 0
            eligible_now = 0
            drafted_count = 0
            model_approved = 0
            acted = early_proactive_actions
            reply_actions = early_proactive_actions
            post_action_sent = early_post_action_sent
            comment_action_sent = False
            consecutive_declines = 0
            recovery_attempts = 0
            skip_reasons: Dict[str, int] = {}
            provider_counts: Dict[str, int] = {}
            cycle_titles: List[str] = [normalize_str(post.get("title")).strip() for post in posts if normalize_str(post.get("title")).strip()]
            post_cd_remaining, comment_cd_remaining = cooldown_remaining_seconds(state=state, cfg=cfg)

            if post_cd_remaining > 0 or comment_cd_remaining > 0:
                logger.info(
                    (
                        "Action cooldown status post_remaining=%ss (~%sm) "
                        "comment_remaining=%ss. Drafting only allowed for eligible actions."
                    ),
                    post_cd_remaining,
                    max(1, post_cd_remaining // 60) if post_cd_remaining > 0 else 0,
                    comment_cd_remaining,
                )

            def mark_seen(pid: Optional[str]) -> None:
                if not pid:
                    return
                seen.add(pid)
                state["seen_post_ids"] = list(seen)[-5000:]

            for post in posts:
                inspected += 1
                if drafted_count >= effective_shortlist_size:
                    skip_reasons["draft_shortlist_cap"] = skip_reasons.get("draft_shortlist_cap", 0) + 1
                    logger.info(
                        "Cycle=%s draft shortlist reached drafted=%s cap=%s stop_additional_drafts=true",
                        iteration,
                        drafted_count,
                        effective_shortlist_size,
                    )
                    break
                title_text = normalize_str(post.get("title")).strip()
                pid = post_id(post)
                if not pid or pid in seen:
                    logger.debug("Cycle=%s skip post_id=%s reason=seen_or_missing", iteration, pid)
                    continue

                new_candidates += 1
                post_title_preview = title_text or "(untitled)"

                if pid in replied_posts:
                    skip_reasons["already_replied_post"] = skip_reasons.get("already_replied_post", 0) + 1
                    mark_seen(pid)
                    logger.debug("Cycle=%s skip post_id=%s reason=already_replied_post", iteration, pid)
                    continue

                if my_name:
                    if has_my_comment_on_post(client=client, post_id_value=pid, my_name=my_name, logger=logger):
                        replied_posts.add(pid)
                        state["replied_post_ids"] = list(replied_posts)[-10000:]
                        skip_reasons["already_replied_post"] = skip_reasons.get("already_replied_post", 0) + 1
                        mark_seen(pid)
                        logger.info(
                            "Cycle=%s skip post_id=%s title=%s reason=already_replied_post_detected",
                            iteration,
                            pid,
                            post_title_preview,
                        )
                        continue

                author_id, author_name = post_author(post)
                if my_name and _normalized_name_key(author_name) == _normalized_name_key(my_name):
                    skip_reasons["self_post"] = skip_reasons.get("self_post", 0) + 1
                    mark_seen(pid)
                    logger.debug(
                        "Cycle=%s skip post_id=%s reason=self_post author=%s my_name=%s",
                        iteration,
                        pid,
                        author_name,
                        my_name,
                    )
                    continue

                allowed, reason = can_reply(state, cfg, author_id, author_name)
                if not allowed:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    is_temporary_rate_block = any(
                        token in reason for token in ("cooldown", "hourly_limit", "daily_limit")
                    )
                    if not is_temporary_rate_block and reason != "author_cooldown":
                        mark_seen(pid)
                    logger.debug(
                        "Cycle=%s skip post_id=%s author=%s reason=%s",
                        iteration,
                        pid,
                        author_name or author_id or "(unknown)",
                        reason,
                    )
                    if not is_temporary_rate_block:
                        continue
                else:
                    eligible_now += 1
                signal_score = relevance_score_by_post.get(pid, 0)
                if signal_score < effective_signal_min_score:
                    skip_reasons["low_signal_relevance"] = skip_reasons.get("low_signal_relevance", 0) + 1
                    mark_seen(pid)
                    logger.debug(
                        "Cycle=%s skip post_id=%s title=%s reason=low_signal_relevance score=%s threshold=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        signal_score,
                        effective_signal_min_score,
                    )
                    continue
                if shortlist_mode == "tighten_quality":
                    trend_overlap = _has_trending_overlap(post=post, trending_terms=trending_terms_cycle)
                    mechanism_score = _post_mechanism_score(post=post)
                    if not trend_overlap and mechanism_score <= 0:
                        skip_reasons["trend_context_mismatch"] = skip_reasons.get("trend_context_mismatch", 0) + 1
                        mark_seen(pid)
                        logger.debug(
                            "Cycle=%s skip post_id=%s title=%s reason=trend_context_mismatch",
                            iteration,
                            pid,
                            post_title_preview,
                        )
                        continue

                allowed_modes = currently_allowed_response_modes(cfg=cfg, state=state)
                if allowed_modes == ["none"]:
                    skip_reasons["no_action_slots"] = skip_reasons.get("no_action_slots", 0) + 1
                    mark_seen(pid)
                    logger.debug("Cycle=%s skip post_id=%s reason=no_action_slots", iteration, pid)
                    continue

                provider_used = "unknown"
                messages: List[Dict[str, str]] = []
                try:
                    logger.debug(
                        "Cycle=%s drafting post_id=%s title=%s provider_hint=%s signal_score=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        cfg.llm_provider,
                        signal_score,
                    )
                    messages = build_openai_messages(
                        persona=persona_text,
                        domain_context=domain_context_text,
                        post=post,
                        pid=pid,
                        allowed_response_modes=allowed_modes,
                        trending_terms=_trending_terms_for_post(
                            post=post,
                            trending_terms=trending_terms_cycle,
                            max_terms=4,
                        ),
                    )
                    draft, provider_used, _ = call_generation_model(cfg, messages)
                    provider_counts[provider_used] = provider_counts.get(provider_used, 0) + 1
                    drafted_count += 1
                    logger.debug(
                        "Cycle=%s drafted post_id=%s title=%s provider=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        provider_used,
                    )
                except Exception as e:
                    logger.warning(
                        "Cycle=%s llm_draft_failed post_id=%s title=%s provider_hint=%s error=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        cfg.llm_provider,
                        e,
                    )
                    draft = fallback_draft()
                    drafted_count += 1
                    logger.info("Cycle=%s using_fallback_draft post_id=%s", iteration, pid)

                should_respond = bool(draft.get("should_respond", False))
                confidence = float(draft.get("confidence", 0))
                can_try_recovery = (
                    recovery_attempts < MAX_RECOVERY_DRAFTS_PER_CYCLE
                    and signal_score >= (effective_signal_min_score + RECOVERY_SIGNAL_MARGIN)
                    and bool(messages)
                )
                if (not should_respond or confidence < cfg.min_confidence) and can_try_recovery:
                    try:
                        recovery_messages = _build_recovery_messages(messages, signal_score=signal_score)
                        recovered_draft, recovery_provider, _ = call_generation_model(cfg, recovery_messages)
                        provider_counts[recovery_provider] = provider_counts.get(recovery_provider, 0) + 1
                        drafted_count += 1
                        recovery_attempts += 1
                        if isinstance(recovered_draft, dict):
                            draft = recovered_draft
                            should_respond = bool(draft.get("should_respond", False))
                            confidence = float(draft.get("confidence", 0))
                            logger.info(
                                (
                                    "Cycle=%s recovery_draft_attempt post_id=%s title=%s "
                                    "provider=%s should_respond=%s confidence=%.3f attempt=%s/%s"
                                ),
                                iteration,
                                pid,
                                post_title_preview,
                                recovery_provider,
                                should_respond,
                                confidence,
                                recovery_attempts,
                                MAX_RECOVERY_DRAFTS_PER_CYCLE,
                            )
                    except Exception as e:
                        logger.debug("Cycle=%s recovery_draft_failed post_id=%s error=%s", iteration, pid, e)

                if not should_respond:
                    consecutive_declines += 1
                    logger.info(
                        "Cycle=%s model_declined post_id=%s title=%s",
                        iteration,
                        pid,
                        post_title_preview,
                    )
                    mark_seen(pid)
                    if consecutive_declines >= MAX_CONSECUTIVE_DECLINES_GUARD:
                        skip_reasons["consecutive_declines_guard"] = skip_reasons.get("consecutive_declines_guard", 0) + 1
                        logger.info(
                            "Cycle=%s stopping drafts early reason=consecutive_declines_guard declines=%s",
                            iteration,
                            consecutive_declines,
                        )
                        break
                    continue

                if confidence < cfg.min_confidence:
                    consecutive_declines += 1
                    logger.info(
                        "Cycle=%s skip post_id=%s reason=low_confidence confidence=%.3f threshold=%.3f",
                        iteration,
                        pid,
                        confidence,
                        cfg.min_confidence,
                    )
                    mark_seen(pid)
                    if consecutive_declines >= MAX_CONSECUTIVE_DECLINES_GUARD:
                        skip_reasons["consecutive_declines_guard"] = skip_reasons.get("consecutive_declines_guard", 0) + 1
                        logger.info(
                            "Cycle=%s stopping drafts early reason=consecutive_declines_guard declines=%s",
                            iteration,
                            consecutive_declines,
                        )
                        break
                    continue
                consecutive_declines = 0
                model_approved += 1
                requested_mode = normalize_response_mode(draft.get("response_mode"), default="comment")

                content = format_content(draft)
                content = _ensure_use_case_prompt_if_relevant(content=content, post=post)
                content = _normalize_ergo_terms(content)
                if not content:
                    logger.warning("Cycle=%s skip post_id=%s reason=empty_content", iteration, pid)
                    mark_seen(pid)
                    continue
                if _is_template_like_generated_content(content):
                    skip_reasons["template_like_content"] = skip_reasons.get("template_like_content", 0) + 1
                    logger.info(
                        "Cycle=%s skip post_id=%s title=%s reason=template_like_content",
                        iteration,
                        pid,
                        post_title_preview,
                    )
                    mark_seen(pid)
                    continue
                if not _passes_generated_content_quality(content=content, requested_mode=requested_mode):
                    skip_reasons["quality_gate_failed"] = skip_reasons.get("quality_gate_failed", 0) + 1
                    logger.info(
                        "Cycle=%s skip post_id=%s title=%s reason=quality_gate_failed mode=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        requested_mode,
                    )
                    mark_seen(pid)
                    continue

                url = post_url(pid)
                title = _sanitize_generated_title(draft.get("title"), fallback="Quick question on your post")
                raw_submolt = post.get("submolt")
                submolt = normalize_submolt(raw_submolt)
                comment_content = content
                post_content = _compose_reference_post_content(reference_url=url, content=content)
                logger.debug(
                    "Cycle=%s normalized_submolt post_id=%s raw_type=%s value=%s",
                    iteration,
                    pid,
                    type(raw_submolt).__name__,
                    submolt,
                )

                actions = planned_actions(requested_mode=requested_mode, cfg=cfg, state=state)
                if not actions:
                    comment_allowed_now, comment_gate_reason = comment_gate_status(state=state, cfg=cfg)
                    should_wait_comment = (
                        requested_mode in {"comment", "both"}
                        and not comment_allowed_now
                        and comment_gate_reason == "comment_cooldown"
                    )
                    if should_wait_comment:
                        approved, approve_all_actions, should_stop = confirm_action(
                            cfg=cfg,
                            logger=logger,
                            action="wait-comment",
                            pid=pid,
                            title=title,
                            submolt=submolt,
                            url=url,
                            author=author_name or author_id or "(unknown)",
                            content_preview=preview_text(comment_content),
                            approve_all=approve_all_actions,
                        )
                        if should_stop:
                            return
                        if approved:
                            if wait_for_comment_slot(state=state, cfg=cfg, logger=logger):
                                try:
                                    logger.info(
                                        "Cycle=%s action=comment attempt post_id=%s submolt=%s url=%s waited_for_cooldown=true",
                                        iteration,
                                        pid,
                                        submolt,
                                        url,
                                    )
                                    comment_resp = client.create_comment(pid, comment_content)
                                    register_my_comment_id(state=state, response_payload=comment_resp)
                                    state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                                    replied_posts.add(pid)
                                    state["replied_post_ids"] = list(replied_posts)[-10000:]
                                    maybe_upvote_post_after_comment(
                                        client=client,
                                        state=state,
                                        logger=logger,
                                        post_id_value=pid,
                                    )
                                    now_ts = utc_now().timestamp()
                                    state["last_action_ts"] = now_ts
                                    state["last_comment_action_ts"] = now_ts
                                    acted += 1
                                    reply_actions += 1
                                    comment_action_sent = True
                                    logger.info(
                                        "Cycle=%s action=comment success post_id=%s daily_comment_count=%s",
                                        iteration,
                                        pid,
                                        state["daily_comment_count"],
                                    )
                                    print_success_banner(action="comment", pid=pid, url=url, title=title)
                                except Exception as e:
                                    logger.warning(
                                        "Cycle=%s waited comment failed post_id=%s error=%s",
                                        iteration,
                                        pid,
                                        e,
                                    )
                            mark_seen(pid)
                            continue
                    logger.info(
                        "Cycle=%s skip post_id=%s reason=no_available_actions requested_mode=%s",
                        iteration,
                        pid,
                        requested_mode,
                    )
                    continue
                logger.info(
                    "Cycle=%s planned_actions post_id=%s requested_mode=%s effective_actions=%s",
                    iteration,
                    pid,
                    requested_mode,
                    ",".join(actions),
                )

                if cfg.dry_run:
                    logger.info(
                        "Cycle=%s dry_run actions=%s post_id=%s submolt=%s url=%s title=%s",
                        iteration,
                        ",".join(actions),
                        pid,
                        submolt,
                        url,
                        title,
                    )
                    mark_seen(pid)
                    continue

                reply_executed = False
                for action in actions:
                    draft_preview = comment_content if action == "comment" else post_content
                    confirm_pid = pid
                    confirm_url = url
                    if action == "post":
                        confirm_pid = "(new)"
                        confirm_url = f"https://moltbook.com/m/{submolt}"
                        draft_preview = f"Reference post: {url}\n\n{post_content}"
                    approved, approve_all_actions, should_stop = confirm_action(
                        cfg=cfg,
                        logger=logger,
                        action=action,
                        pid=confirm_pid,
                        title=title,
                        submolt=submolt,
                        url=confirm_url,
                        author=author_name or author_id or "(unknown)",
                        content_preview=preview_text(draft_preview),
                        approve_all=approve_all_actions,
                    )
                    if should_stop:
                        return
                    if not approved:
                        logger.info("Cycle=%s action=%s skipped post_id=%s reason=not_approved", iteration, action, pid)
                        mark_seen(pid)
                        continue

                    if action == "comment":
                        try:
                            logger.info(
                                "Cycle=%s action=comment attempt post_id=%s submolt=%s url=%s",
                                iteration,
                                pid,
                                submolt,
                                url,
                            )
                            comment_resp = client.create_comment(pid, comment_content)
                            register_my_comment_id(state=state, response_payload=comment_resp)
                            state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                            replied_posts.add(pid)
                            state["replied_post_ids"] = list(replied_posts)[-10000:]
                            maybe_upvote_post_after_comment(
                                client=client,
                                state=state,
                                logger=logger,
                                post_id_value=pid,
                            )
                            now_ts = utc_now().timestamp()
                            state["last_action_ts"] = now_ts
                            state["last_comment_action_ts"] = now_ts
                            acted += 1
                            reply_actions += 1
                            comment_action_sent = True
                            reply_executed = True
                            logger.info(
                                "Cycle=%s action=comment success post_id=%s daily_comment_count=%s",
                                iteration,
                                pid,
                                state["daily_comment_count"],
                            )
                            print_success_banner(action="comment", pid=pid, url=url, title=title)
                        except Exception as e:
                            logger.warning(
                                "Cycle=%s action=comment failed post_id=%s error=%s fallback=post",
                                iteration,
                                pid,
                                e,
                            )
                            if "post" in actions:
                                continue
                            approved, approve_all_actions, should_stop = confirm_action(
                                cfg=cfg,
                                logger=logger,
                                action="post-fallback",
                                pid="(new)",
                                title=title,
                                submolt=submolt,
                                url=f"https://moltbook.com/m/{submolt}",
                                author=author_name or author_id or "(unknown)",
                                content_preview=preview_text(f"Reference post: {url}\n\n{post_content}"),
                                approve_all=approve_all_actions,
                            )
                            if should_stop:
                                return
                            if not approved:
                                logger.info(
                                    "Cycle=%s action=post-fallback skipped post_id=%s reason=not_approved",
                                    iteration,
                                    pid,
                                )
                                continue
                            logger.info(
                                "Cycle=%s action=post attempt reference_post_id=%s submolt=%s reference_url=%s title=%s",
                                iteration,
                                pid,
                                submolt,
                                url,
                                title,
                            )
                            post_resp = client.create_post(submolt=submolt, title=title, content=post_content)
                            created_post_id = post_id(post_resp if isinstance(post_resp, dict) else {}) or "(unknown)"
                            created_url = post_url(created_post_id if created_post_id != "(unknown)" else None)
                            state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                            replied_posts.add(pid)
                            state["replied_post_ids"] = list(replied_posts)[-10000:]
                            now_ts = utc_now().timestamp()
                            state["last_action_ts"] = now_ts
                            state["last_post_action_ts"] = now_ts
                            acted += 1
                            reply_actions += 1
                            post_action_sent = True
                            reply_executed = True
                            logger.info(
                                (
                                    "Cycle=%s action=post success reference_post_id=%s "
                                    "new_post_id=%s new_url=%s daily_post_count=%s"
                                ),
                                iteration,
                                pid,
                                created_post_id,
                                created_url,
                                state["daily_post_count"],
                            )
                            print_success_banner(action="post", pid=created_post_id, url=created_url, title=title)
                    elif action == "post":
                        logger.info(
                            "Cycle=%s action=post attempt reference_post_id=%s submolt=%s reference_url=%s title=%s",
                            iteration,
                            pid,
                            submolt,
                            url,
                            title,
                        )
                        post_resp = client.create_post(submolt=submolt, title=title, content=post_content)
                        created_post_id = post_id(post_resp if isinstance(post_resp, dict) else {}) or "(unknown)"
                        created_url = post_url(created_post_id if created_post_id != "(unknown)" else None)
                        state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                        replied_posts.add(pid)
                        state["replied_post_ids"] = list(replied_posts)[-10000:]
                        now_ts = utc_now().timestamp()
                        state["last_action_ts"] = now_ts
                        state["last_post_action_ts"] = now_ts
                        acted += 1
                        reply_actions += 1
                        post_action_sent = True
                        reply_executed = True
                        logger.info(
                            (
                                "Cycle=%s action=post success reference_post_id=%s "
                                "new_post_id=%s new_url=%s daily_post_count=%s"
                            ),
                            iteration,
                            pid,
                            created_post_id,
                            created_url,
                            state["daily_post_count"],
                        )
                        print_success_banner(action="post", pid=created_post_id, url=created_url, title=title)

                vote_action = normalize_vote_action(draft.get("vote_action"))
                vote_target = normalize_vote_target(draft.get("vote_target"))
                if vote_action == "downvote" and not cfg.allow_comment_downvote:
                    if vote_target == "top_comment":
                        logger.info(
                            "Cycle=%s adjusting vote_target from top_comment to none for unsupported downvote-comment",
                            iteration,
                        )
                        vote_target = "none"
                    elif vote_target == "both":
                        logger.info(
                            "Cycle=%s adjusting vote_target from both to post for unsupported downvote-comment",
                            iteration,
                        )
                        vote_target = "post"
                if vote_action != "none" and vote_target != "none":
                    if vote_target in {"post", "both"}:
                        approved, approve_all_actions, should_stop = confirm_action(
                            cfg=cfg,
                            logger=logger,
                            action=f"{vote_action}-post",
                            pid=pid,
                            title=title,
                            submolt=submolt,
                            url=url,
                            author=author_name or author_id or "(unknown)",
                            content_preview=preview_text(
                                f"Vote target: post {pid}\nVote action: {vote_action}"
                            ),
                            approve_all=approve_all_actions,
                        )
                        if should_stop:
                            return
                        if approved:
                            try:
                                client.vote_post(pid, vote_action=vote_action)
                                acted += 1
                                logger.info(
                                    "Cycle=%s action=%s success post_id=%s",
                                    iteration,
                                    f"{vote_action}-post",
                                    pid,
                                )
                                print_success_banner(
                                    action=f"{vote_action}-post",
                                    pid=pid,
                                    url=url,
                                    title=title,
                                )
                            except Exception as e:
                                logger.warning(
                                    "Cycle=%s action=%s failed post_id=%s error=%s",
                                    iteration,
                                    f"{vote_action}-post",
                                    pid,
                                    e,
                                )

                    if vote_target in {"top_comment", "both"}:
                        top_comment = choose_top_comment(client=client, pid=pid, my_name=my_name, logger=logger)
                        if top_comment:
                            top_comment_id = comment_id(top_comment)
                            _, top_comment_author = comment_author(top_comment)
                            comment_body = normalize_str(top_comment.get("content"))
                            approved, approve_all_actions, should_stop = confirm_action(
                                cfg=cfg,
                                logger=logger,
                                action=f"{vote_action}-comment",
                                pid=top_comment_id or pid,
                                title=f"Comment by {top_comment_author or '(unknown)'}",
                                submolt=submolt,
                                url=url,
                                author=top_comment_author or "(unknown)",
                                content_preview=preview_text(comment_body),
                                approve_all=approve_all_actions,
                            )
                            if should_stop:
                                return
                            if approved and top_comment_id:
                                try:
                                    client.vote_comment(top_comment_id, vote_action=vote_action)
                                    acted += 1
                                    logger.info(
                                        "Cycle=%s action=%s success comment_id=%s",
                                        iteration,
                                        f"{vote_action}-comment",
                                        top_comment_id,
                                    )
                                    print_success_banner(
                                        action=f"{vote_action}-comment",
                                        pid=top_comment_id,
                                        url=url,
                                        title=f"Comment by {top_comment_author or '(unknown)'}",
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Cycle=%s action=%s failed comment_id=%s error=%s",
                                        iteration,
                                        f"{vote_action}-comment",
                                        top_comment_id,
                                        e,
                                    )
                        else:
                            logger.info(
                                "Cycle=%s vote skipped post_id=%s reason=no_comment_target vote_target=%s",
                                iteration,
                                pid,
                                vote_target,
                            )

                if reply_executed and author_id:
                    state.setdefault("per_author_last_reply", {})[author_id] = state.get("last_action_ts")
                if reply_executed:
                    mark_seen(pid)

            today_iso = time.strftime("%Y-%m-%d", time.gmtime())
            proactive_today = _proactive_posts_count_for_date(post_memory=post_memory, date_iso=today_iso)
            daily_proactive_target = max(1, int(cfg.proactive_daily_target_posts))
            daily_goal_missing = proactive_today < daily_proactive_target
            should_try_proactive = (acted == 0 and not post_action_sent) or (daily_goal_missing and not post_action_sent)
            if should_try_proactive:
                logger.info(
                    "Proactive trigger cycle=%s acted=%s post_action_sent=%s proactive_today=%s target=%s",
                    iteration,
                    acted,
                    post_action_sent,
                    proactive_today,
                    daily_proactive_target,
                )
                proactive_actions, should_stop, approve_all_actions = maybe_run_proactive_post(
                    client=client,
                    cfg=cfg,
                    logger=logger,
                    state=state,
                    post_memory=post_memory,
                    my_name=my_name,
                    persona_text=persona_text,
                    domain_context_text=domain_context_text,
                    approve_all_actions=approve_all_actions,
                )
                if should_stop:
                    return
                if proactive_actions > 0:
                    acted += proactive_actions
                    reply_actions += proactive_actions
                    post_action_sent = True

            state["seen_post_ids"] = list(seen)[-5000:]
            state["replied_post_ids"] = list(replied_posts)[-10000:]
            save_state(cfg.state_path, state)
            save_post_engine_memory(cfg.proactive_memory_path, post_memory)
            logger.info(
                (
                    "Poll cycle=%s complete inspected=%s new_candidates=%s eligible_now=%s "
                    "drafted=%s model_approved=%s actions=%s seen_total=%s"
                ),
                iteration,
                inspected,
                new_candidates,
                eligible_now,
                drafted_count,
                model_approved,
                acted,
                len(seen),
            )
            if provider_counts:
                provider_summary = ", ".join([f"{k}={v}" for k, v in sorted(provider_counts.items())])
                logger.info("Poll cycle=%s provider_summary %s", iteration, provider_summary)
            if skip_reasons:
                summary = ", ".join([f"{k}={v}" for k, v in sorted(skip_reasons.items())])
                logger.info("Poll cycle=%s skip_summary %s", iteration, summary)

            if cfg.keyword_learning_enabled:
                interval = max(1, cfg.keyword_learning_interval_cycles)
                next_learning_cycle = ((iteration // interval) + 1) * interval
                if iteration % interval == 0:
                    next_learning_cycle = iteration
                logger.info(
                    (
                        "Keyword learning status enabled=true cycle=%s next_cycle=%s "
                        "titles_seen=%s min_titles=%s"
                    ),
                    iteration,
                    next_learning_cycle,
                    len(cycle_titles),
                    cfg.keyword_learning_min_titles,
                )

            if (
                cfg.keyword_learning_enabled
                and cfg.keyword_learning_interval_cycles > 0
                and iteration % cfg.keyword_learning_interval_cycles == 0
                and len(cycle_titles) >= cfg.keyword_learning_min_titles
            ):
                suggestions: List[str] = []
                if has_generation_provider(cfg):
                    try:
                        suggestions = propose_keywords_from_titles(
                            cfg=cfg,
                            titles=cycle_titles,
                            existing_keywords=active_keywords,
                            max_suggestions=cfg.keyword_learning_max_suggestions,
                        )
                    except Exception as e:
                        suggestions = []
                        logger.warning("Keyword learning failed cycle=%s error=%s", iteration, e)
                market_snapshot_for_keywords = learning_snapshot_cycle.get("market_snapshot")
                market_suggestions = _market_keyword_candidates(
                    market_snapshot=market_snapshot_for_keywords if isinstance(market_snapshot_for_keywords, dict) else {},
                    existing_keywords=merge_keywords(active_keywords, suggestions),
                    max_suggestions=cfg.keyword_learning_max_suggestions,
                )
                if market_suggestions:
                    suggestions = merge_keywords(suggestions, market_suggestions)[: cfg.keyword_learning_max_suggestions]
                    logger.info(
                        "Keyword learning market_suggestions cycle=%s suggested=%s",
                        iteration,
                        ", ".join(market_suggestions),
                    )

                if suggestions:
                    logger.info(
                        "Keyword learning cycle=%s suggested=%s",
                        iteration,
                        ", ".join(suggestions),
                    )
                    if manual_keyword_review:
                        for keyword in suggestions:
                            approved, approve_all_keyword_changes, should_stop = confirm_keyword_addition(
                                logger=logger,
                                keyword=keyword,
                                approve_all=approve_all_keyword_changes,
                            )
                            if should_stop:
                                return
                            if not approved:
                                continue
                            learned_before = keyword_store.get("learned_keywords", [])
                            learned_after = merge_keywords(learned_before, [keyword])
                            if len(learned_after) == len(learned_before):
                                continue
                            keyword_store["learned_keywords"] = learned_after
                            save_keyword_store(cfg.keyword_store_path, keyword_store)
                            active_keywords = merge_keywords(cfg.keywords, learned_after)
                            logger.info(
                                "Keyword added keyword=%s learned_total=%s active_total=%s",
                                keyword,
                                len(learned_after),
                                len(active_keywords),
                            )
                            print_keyword_added_banner(keyword=keyword, learned_total=len(learned_after))
                    else:
                        pending_before = keyword_store.get("pending_suggestions", [])
                        combined_pending = merge_keywords(pending_before, suggestions)
                        # Do not keep duplicates that are already learned.
                        learned_set = set(keyword_store.get("learned_keywords", []))
                        filtered_pending = [k for k in combined_pending if k not in learned_set]
                        added = max(0, len(filtered_pending) - len(pending_before))
                        keyword_store["pending_suggestions"] = filtered_pending
                        save_keyword_store(cfg.keyword_store_path, keyword_store)
                        logger.info(
                            "Deferred keyword suggestions for next manual run added=%s pending_total=%s",
                            added,
                            len(filtered_pending),
                        )

            cycle_stats = {
                "inspected": inspected,
                "new_candidates": new_candidates,
                "eligible_now": eligible_now,
                "drafted": drafted_count,
                "model_approved": model_approved,
                "actions": acted,
                "post_action_sent": post_action_sent,
                "comment_action_sent": comment_action_sent,
                "skip_reasons": skip_reasons,
                "max_suggestions": cfg.self_improve_max_suggestions,
                "draft_shortlist_size": cfg.draft_shortlist_size,
                "draft_signal_min_score": cfg.draft_signal_min_score,
                "effective_draft_shortlist_size": effective_shortlist_size,
                "effective_draft_signal_min_score": effective_signal_min_score,
                "draft_shortlist_mode": shortlist_mode,
                "high_signal_term_count": len(high_signal_terms),
                "trending_terms": trending_terms_cycle,
            }
            visibility_metrics_cycle = learning_snapshot_cycle.get("visibility_metrics")
            if isinstance(visibility_metrics_cycle, dict):
                cycle_stats["proactive_recent_avg_upvotes"] = float(
                    visibility_metrics_cycle.get("recent_avg_upvotes", 0.0) or 0.0
                )
                cycle_stats["proactive_recent_avg_comments"] = float(
                    visibility_metrics_cycle.get("recent_avg_comments", 0.0) or 0.0
                )
                cycle_stats["proactive_recent_avg_visibility_score"] = float(
                    visibility_metrics_cycle.get("recent_avg_visibility_score", 0.0) or 0.0
                )
                cycle_stats["proactive_visibility_delta_pct"] = float(
                    visibility_metrics_cycle.get("visibility_delta_pct", 0.0) or 0.0
                )
                cycle_stats["proactive_target_hit_rate"] = float(
                    visibility_metrics_cycle.get("recent_target_hit_rate", 0.0) or 0.0
                )
                cycle_stats["proactive_target_upvotes"] = int(
                    visibility_metrics_cycle.get("target_upvotes", 0) or 0
                )
            lift_terms_cycle: List[str] = []
            raw_lifts_cycle = learning_snapshot_cycle.get("winning_terms_lift")
            if isinstance(raw_lifts_cycle, list):
                for item in raw_lifts_cycle[:8]:
                    if not isinstance(item, dict):
                        continue
                    term = _clean_signal_term(item.get("term"))
                    if term and term not in lift_terms_cycle:
                        lift_terms_cycle.append(term)
            cycle_stats["proactive_top_lift_terms"] = lift_terms_cycle
            discovery_market_signals = build_top_post_signals(
                posts=posts,
                limit=max(8, cfg.proactive_post_reference_limit),
                source="discovery",
            )
            if discovery_market_signals:
                market_snapshot = update_market_signals(
                    memory=post_memory,
                    signals=discovery_market_signals,
                )
                cycle_stats["market_signal_count"] = market_snapshot.get("signal_count", 0)
                cycle_stats["market_question_title_rate"] = market_snapshot.get("question_title_rate", 0.0)
                cycle_stats["market_top_submolts"] = market_snapshot.get("top_submolts", [])[:3]
            maybe_write_self_improvement_suggestions(
                cfg=cfg,
                logger=logger,
                iteration=iteration,
                persona_text=persona_text,
                domain_context_text=domain_context_text,
                learning_snapshot=learning_snapshot_cycle,
                cycle_titles=cycle_titles,
                cycle_stats=cycle_stats,
            )
            metrics_history = state.get("cycle_metrics_history", [])
            if not isinstance(metrics_history, list):
                metrics_history = []
            metrics_history.append(
                {
                    "cycle": iteration,
                    "inspected": inspected,
                    "new_candidates": new_candidates,
                    "eligible_now": eligible_now,
                    "drafted": drafted_count,
                    "model_approved": model_approved,
                    "actions": acted,
                    "skip_reasons": skip_reasons,
                    "effective_draft_shortlist_size": effective_shortlist_size,
                    "effective_draft_signal_min_score": effective_signal_min_score,
                    "draft_shortlist_mode": shortlist_mode,
                }
            )
            state["cycle_metrics_history"] = metrics_history[-240:]

            # Sleep policy:
            # - Poll quickly when idle.
            # - Never sleep for post cooldown: keep discovery/learning running.
            # - For comment actions, honor only the short comment cooldown.
            sleep_seconds = max(1, cfg.idle_poll_seconds)
            sleep_reason = "idle_poll"
            if reply_actions > 0:
                post_remaining, comment_remaining = cooldown_remaining_seconds(state=state, cfg=cfg)
                if comment_action_sent and comment_remaining > 0:
                    sleep_seconds = max(1, comment_remaining)
                    sleep_reason = "comment_cooldown"
                elif post_action_sent and post_remaining > 0:
                    sleep_seconds = max(1, cfg.idle_poll_seconds)
                    sleep_reason = "post_cooldown_background"
                else:
                    sleep_seconds = max(1, cfg.idle_poll_seconds)
                    sleep_reason = "cooldown_elapsed"
        except MoltbookAuthError as e:
            logger.error("Poll cycle=%s auth_error=%s", iteration, e)
            sleep_seconds = max(1, cfg.poll_seconds)
            sleep_reason = "auth_error_backoff"
        except Exception as e:
            logger.exception("Poll cycle=%s loop_error=%s", iteration, e)
            sleep_seconds = max(1, cfg.poll_seconds)
            sleep_reason = "loop_error_backoff"

        logger.info("Sleeping seconds=%s reason=%s", sleep_seconds, sleep_reason)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    run_loop()
