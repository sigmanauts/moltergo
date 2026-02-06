from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .drafting import normalize_str
from .state import utc_now


STOPWORDS = {
    "about",
    "after",
    "agent",
    "agents",
    "also",
    "and",
    "been",
    "best",
    "build",
    "could",
    "data",
    "from",
    "have",
    "into",
    "just",
    "more",
    "most",
    "need",
    "next",
    "post",
    "that",
    "their",
    "them",
    "they",
    "this",
    "with",
    "your",
}


def _default_memory() -> Dict[str, Any]:
    return {
        "proactive_posts": [],
        "declined_ideas": [],
        "last_metrics_refresh_ts": None,
        "last_snapshot": {},
    }


def load_post_engine_memory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_memory()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _default_memory()

    if not isinstance(data, dict):
        return _default_memory()
    if "proactive_posts" not in data or not isinstance(data.get("proactive_posts"), list):
        data["proactive_posts"] = []
    if "declined_ideas" not in data or not isinstance(data.get("declined_ideas"), list):
        data["declined_ideas"] = []
    if "last_metrics_refresh_ts" not in data:
        data["last_metrics_refresh_ts"] = None
    if "last_snapshot" not in data or not isinstance(data.get("last_snapshot"), dict):
        data["last_snapshot"] = {}
    return data


def save_post_engine_memory(path: Path, memory: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, sort_keys=True)


def _metric_int(post: Dict[str, Any], keys: List[str]) -> int:
    for key in keys:
        value = post.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _post_engagement_score(post: Dict[str, Any]) -> int:
    upvotes = _metric_int(post, ["upvotes", "score", "vote_score", "likes"])
    comments = _metric_int(post, ["comment_count", "comments_count", "comments"])
    # Comments are weighted higher because they are stronger engagement signals.
    return upvotes + (comments * 2)


def record_proactive_post(
    memory: Dict[str, Any],
    post_id: str,
    title: str,
    submolt: str,
    content: str,
    strategy_notes: str,
    topic_tags: List[str],
) -> None:
    entries = memory.setdefault("proactive_posts", [])
    now_ts = utc_now().timestamp()
    entries.append(
        {
            "post_id": post_id,
            "title": title,
            "submolt": submolt,
            "content_preview": content[:400],
            "strategy_notes": strategy_notes,
            "topic_tags": topic_tags[:8],
            "created_ts": now_ts,
            "upvotes": None,
            "comment_count": None,
            "engagement_score": None,
            "last_metrics_ts": None,
        }
    )
    memory["proactive_posts"] = entries[-300:]


def record_declined_idea(
    memory: Dict[str, Any],
    title: str,
    submolt: str,
    reason: str,
) -> None:
    entries = memory.setdefault("declined_ideas", [])
    entries.append(
        {
            "title": title[:160],
            "submolt": submolt,
            "reason": reason,
            "ts": utc_now().timestamp(),
        }
    )
    memory["declined_ideas"] = entries[-120:]


def refresh_metrics_from_recent_posts(memory: Dict[str, Any], recent_posts: List[Dict[str, Any]]) -> int:
    by_id: Dict[str, Dict[str, Any]] = {}
    for post in recent_posts:
        pid = post.get("id") or (post.get("post") or {}).get("id")
        if pid is None:
            continue
        by_id[str(pid)] = post

    updated = 0
    now_ts = utc_now().timestamp()
    for entry in memory.get("proactive_posts", []):
        pid = normalize_str(entry.get("post_id")).strip()
        if not pid:
            continue
        src = by_id.get(pid)
        if not src:
            continue
        upvotes = _metric_int(src, ["upvotes", "score", "vote_score", "likes"])
        comments = _metric_int(src, ["comment_count", "comments_count", "comments"])
        score = _post_engagement_score(src)
        if (
            entry.get("upvotes") != upvotes
            or entry.get("comment_count") != comments
            or entry.get("engagement_score") != score
        ):
            updated += 1
        entry["upvotes"] = upvotes
        entry["comment_count"] = comments
        entry["engagement_score"] = score
        entry["last_metrics_ts"] = now_ts
    memory["last_metrics_refresh_ts"] = now_ts
    return updated


def _extract_terms(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]{4,}", normalize_str(text).lower())
    return [t for t in tokens if t not in STOPWORDS]


def _top_terms(items: List[Dict[str, Any]], limit: int) -> List[str]:
    counts: Dict[str, int] = {}
    for item in items:
        text = f"{normalize_str(item.get('title'))} {normalize_str(item.get('strategy_notes'))}"
        for token in _extract_terms(text):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in ranked[: max(0, limit)]]


def build_learning_snapshot(memory: Dict[str, Any], max_examples: int = 5) -> Dict[str, Any]:
    entries = [e for e in memory.get("proactive_posts", []) if isinstance(e, dict)]
    scored = [e for e in entries if isinstance(e.get("engagement_score"), (int, float))]
    scored.sort(key=lambda e: float(e.get("engagement_score", 0)), reverse=True)

    winners = scored[: max_examples]
    losers = list(reversed(scored[-max_examples:])) if scored else []

    best_submolt: Dict[str, List[float]] = {}
    for entry in scored:
        submolt = normalize_str(entry.get("submolt")).strip().lower()
        if not submolt:
            continue
        best_submolt.setdefault(submolt, []).append(float(entry.get("engagement_score", 0)))
    submolt_rank = sorted(
        [(k, sum(v) / max(1, len(v))) for k, v in best_submolt.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    snapshot = {
        "total_proactive_posts": len(entries),
        "scored_posts": len(scored),
        "winning_examples": [
            {
                "post_id": normalize_str(e.get("post_id")),
                "title": normalize_str(e.get("title")),
                "submolt": normalize_str(e.get("submolt")),
                "engagement_score": e.get("engagement_score"),
                "upvotes": e.get("upvotes"),
                "comment_count": e.get("comment_count"),
                "strategy_notes": normalize_str(e.get("strategy_notes"))[:220],
            }
            for e in winners
        ],
        "losing_examples": [
            {
                "post_id": normalize_str(e.get("post_id")),
                "title": normalize_str(e.get("title")),
                "submolt": normalize_str(e.get("submolt")),
                "engagement_score": e.get("engagement_score"),
                "upvotes": e.get("upvotes"),
                "comment_count": e.get("comment_count"),
                "strategy_notes": normalize_str(e.get("strategy_notes"))[:220],
            }
            for e in losers
        ],
        "winning_terms": _top_terms(winners, limit=8),
        "losing_terms": _top_terms(losers, limit=8),
        "best_submolts": [{"name": name, "avg_score": round(avg, 2)} for name, avg in submolt_rank[:5]],
    }
    memory["last_snapshot"] = snapshot
    return snapshot


def append_improvement_suggestions(
    path: Path,
    cycle: int,
    provider: str,
    suggestions: Dict[str, Any],
    max_entries: int = 120,
) -> None:
    payload: Dict[str, Any]
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            payload = loaded if isinstance(loaded, dict) else {}
        except Exception:
            payload = {}
    else:
        payload = {}

    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = []

    entry = {
        "ts": utc_now().isoformat(),
        "cycle": int(cycle),
        "provider": normalize_str(provider).strip() or "unknown",
        "summary": normalize_str(suggestions.get("summary")).strip(),
        "prompt_changes": suggestions.get("prompt_changes") if isinstance(suggestions.get("prompt_changes"), list) else [],
        "code_changes": suggestions.get("code_changes") if isinstance(suggestions.get("code_changes"), list) else [],
        "strategy_experiments": (
            suggestions.get("strategy_experiments")
            if isinstance(suggestions.get("strategy_experiments"), list)
            else []
        ),
        "priority": normalize_str(suggestions.get("priority")).strip() or "medium",
    }
    entries.append(entry)
    payload["entries"] = entries[-max(10, max_entries):]
    payload["last_updated_ts"] = utc_now().isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
