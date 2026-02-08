from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


COMMENT_SOURCE_WEIGHTS = {
    "hot": 1.00,
    "new": 0.86,
    "rising": 0.74,
    "top": 0.62,
}

REFERENCE_SOURCE_WEIGHTS = {
    "top": 1.00,
    "rising": 0.90,
    "hot": 0.80,
    "new": 0.68,
}

RISK_PATTERNS = (
    r"\b(send|share|paste)\b.{0,24}\b(api key|secret|private key|seed|mnemonic|wallet|token)\b",
    r"\binstall\b.{0,16}\bskill\b",
    r"\brun\b.{0,20}\b(command|shell|script)\b",
    r"\b(connect|link)\b.{0,24}\bwallet\b",
    r"\bairdrop\b",
    r"\bguaranteed returns?\b",
    r"\bdouble your\b",
)

INJECTION_PATTERNS = (
    r"\bignore (all )?(previous|prior) instructions\b",
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\bexecute this\b",
)


def normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def parse_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def extract_feed_sources(post: Dict[str, Any]) -> List[str]:
    raw = post.get("__feed_sources")
    if isinstance(raw, list):
        out: List[str] = []
        seen = set()
        for item in raw:
            token = normalize_str(item).strip().lower()
            if token and token not in seen:
                seen.add(token)
                out.append(token)
        return out
    token = normalize_str(post.get("__feed_source")).strip().lower()
    return [token] if token else []


def is_early_comment_candidate(post: Dict[str, Any], now_ts: float, early_window_seconds: int) -> bool:
    created_ts = parse_timestamp(
        post.get("created_at")
        or post.get("createdAt")
        or post.get("published_at")
        or post.get("timestamp")
    )
    if created_ts is None:
        return False
    return (now_ts - created_ts) <= max(1, int(early_window_seconds))


def looks_risk_bait(text: str) -> bool:
    body = normalize_str(text).lower()
    if not body:
        return False
    for pattern in RISK_PATTERNS:
        if re.search(pattern, body, flags=re.IGNORECASE | re.DOTALL):
            return True
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, body, flags=re.IGNORECASE | re.DOTALL):
            return True
    return False


def infer_topic_signature(title: str, content: str) -> str:
    blob = f"{normalize_str(title).lower()} {normalize_str(content).lower()}"
    tokens = re.findall(r"[a-z0-9]{4,}", blob)
    if not tokens:
        return ""
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [term for term, _ in ranked[:4]]
    return ":".join(top)


def _source_weight(sources: Sequence[str], mode: str) -> float:
    table = COMMENT_SOURCE_WEIGHTS if mode == "comment" else REFERENCE_SOURCE_WEIGHTS
    if not sources:
        return 0.55
    best = 0.0
    for source in sources:
        best = max(best, float(table.get(source, 0.5)))
    return best


def _recency_weight(created_ts: Optional[float], now_ts: float, half_life_minutes: int) -> float:
    if created_ts is None:
        return 0.45
    age_seconds = max(0.0, now_ts - created_ts)
    half_life_seconds = max(60.0, float(half_life_minutes) * 60.0)
    decay = 0.5 ** (age_seconds / half_life_seconds)
    return max(0.0, min(1.0, decay))


def _engagement_weight(upvotes: int, comments: int) -> float:
    # Monotonic and bounded to keep room for other factors.
    votes_term = math.log1p(max(0, upvotes)) / math.log(30.0)
    comments_term = math.log1p(max(0, comments)) / math.log(20.0)
    return max(0.0, min(1.0, (votes_term * 0.55) + (comments_term * 0.45)))


def _submolt_weight(submolt_meta: Dict[str, Any], now_ts: float) -> float:
    subscribers = int(submolt_meta.get("subscriber_count", 0) or 0)
    last_activity_ts = parse_timestamp(submolt_meta.get("last_activity_at"))
    size_term = min(1.0, math.log1p(max(0, subscribers)) / math.log(50000.0))
    if last_activity_ts is None:
        fresh_term = 0.4
    else:
        age_hours = max(0.0, (now_ts - last_activity_ts) / 3600.0)
        # Fast freshness drop after first day.
        fresh_term = max(0.0, min(1.0, 1.0 / (1.0 + (age_hours / 24.0))))
    return (size_term * 0.6) + (fresh_term * 0.4)


def _context_fit_weight(post: Dict[str, Any], history: Dict[str, Any]) -> float:
    semantic = float(post.get("__semantic_relevance", 0) or 0)
    keyword_hits = int(post.get("__keyword_hits", 0) or 0)
    keyword_list = history.get("active_keywords") if isinstance(history, dict) else None
    if keyword_hits <= 0 and isinstance(keyword_list, list):
        blob = " ".join(
            [
                normalize_str(post.get("title")).lower(),
                normalize_str(post.get("content")).lower(),
                normalize_str(post.get("submolt")).lower(),
            ]
        )
        for token in keyword_list:
            clean = normalize_str(token).strip().lower()
            if clean and clean in blob:
                keyword_hits += 1
    semantic_term = min(1.0, max(0.0, semantic) / 10.0)
    keyword_term = min(1.0, float(keyword_hits) / 6.0)
    return (semantic_term * 0.62) + (keyword_term * 0.38)


def _novelty_penalty(post: Dict[str, Any], history: Dict[str, Any]) -> float:
    if not isinstance(history, dict):
        return 0.0
    title = normalize_str(post.get("title"))
    content = normalize_str(post.get("content"))
    sig = infer_topic_signature(title=title, content=content)
    if not sig:
        return 0.0
    recent = history.get("recent_topic_signatures")
    if not isinstance(recent, list):
        return 0.0
    overlap = 0
    for item in recent[-60:]:
        if normalize_str(item).strip() == sig:
            overlap += 1
    if overlap <= 0:
        return 0.0
    return min(0.45, overlap * 0.12)


def _risk_penalty(post: Dict[str, Any]) -> float:
    blob = " ".join(
        [
            normalize_str(post.get("title")),
            normalize_str(post.get("content")),
            normalize_str(post.get("url")),
        ]
    )
    return 0.7 if looks_risk_bait(blob) else 0.0


def score_post_candidate(
    post: Dict[str, Any],
    submolt_meta: Optional[Dict[str, Any]],
    now: Optional[datetime],
    history: Optional[Dict[str, Any]],
) -> float:
    """
    Score a post for reach/engagement opportunity.

    Higher is better.
    """
    history = history or {}
    now_dt = now or datetime.now(timezone.utc)
    now_ts = now_dt.timestamp()
    mode = normalize_str(history.get("mode", "comment")).strip().lower()
    if mode not in {"comment", "reference"}:
        mode = "comment"

    sources = extract_feed_sources(post)
    created_ts = parse_timestamp(
        post.get("created_at")
        or post.get("createdAt")
        or post.get("published_at")
        or post.get("timestamp")
    )
    upvotes = int(post.get("upvotes") or post.get("score") or post.get("vote_score") or 0)
    comments = int(post.get("comment_count") or post.get("comments_count") or post.get("comments") or 0)

    half_life = int(history.get("recency_halflife_minutes", 180) or 180)
    source_component = _source_weight(sources=sources, mode=mode)
    recency_component = _recency_weight(created_ts=created_ts, now_ts=now_ts, half_life_minutes=half_life)
    engagement_component = _engagement_weight(upvotes=upvotes, comments=comments)
    submolt_component = _submolt_weight(submolt_meta or {}, now_ts=now_ts)
    context_component = _context_fit_weight(post=post, history=history)
    novelty_pen = _novelty_penalty(post=post, history=history)
    risk_pen = _risk_penalty(post=post)

    score = (
        (source_component * 0.22)
        + (recency_component * 0.20)
        + (engagement_component * 0.22)
        + (submolt_component * 0.14)
        + (context_component * 0.22)
    )
    score -= novelty_pen
    score -= risk_pen

    # Slight priority bump for early windows in comment mode.
    if mode == "comment":
        early_window = int(history.get("early_comment_window_seconds", 900) or 900)
        if is_early_comment_candidate(post=post, now_ts=now_ts, early_window_seconds=early_window):
            score += 0.08

    return round(float(score), 6)


def summarize_sources(posts: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for post in posts:
        for source in extract_feed_sources(post):
            counts[source] = counts.get(source, 0) + 1
    return counts
