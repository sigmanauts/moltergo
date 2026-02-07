from __future__ import annotations

import json
import os
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
    "here",
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
    "what",
    "with",
    "your",
    "projectsubmission",
    "usdchackathon",
    "current",
    "problem",
}

DISALLOWED_STRATEGY_TOKENS = {
    "ama",
    "ask me anything",
    "virtual meetup",
    "meetup",
    "roundtable",
    "q&a",
    "q a",
    "challenge",
    "weekly digest",
    "monthly report",
    "repository",
    "video series",
    "webinar",
    "attendance",
    "participants",
}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except Exception:
        return max(minimum, int(default))


def _visibility_target_upvotes() -> int:
    return _env_int("MOLTBOOK_VISIBILITY_TARGET_UPVOTES", 25, minimum=1)


def _visibility_recent_window() -> int:
    return _env_int("MOLTBOOK_VISIBILITY_RECENT_WINDOW", 12, minimum=4)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
        if den <= 0:
            return 0.0
        return num / den
    except Exception:
        return 0.0


def _default_memory() -> Dict[str, Any]:
    return {
        "proactive_posts": [],
        "declined_ideas": [],
        "last_metrics_refresh_ts": None,
        "last_market_signals": [],
        "last_market_snapshot": {},
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
    if "last_market_signals" not in data or not isinstance(data.get("last_market_signals"), list):
        data["last_market_signals"] = []
    if "last_market_snapshot" not in data or not isinstance(data.get("last_market_snapshot"), dict):
        data["last_market_snapshot"] = {}
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


def _entry_age_hours(entry: Dict[str, Any], now_ts: Optional[float] = None) -> float:
    created_ts = entry.get("created_ts")
    if not isinstance(created_ts, (int, float)):
        return 1.0
    now = float(now_ts) if isinstance(now_ts, (int, float)) else utc_now().timestamp()
    elapsed = max(0.0, now - float(created_ts))
    return max(1.0 / 60.0, elapsed / 3600.0)


def _append_metrics_sample(entry: Dict[str, Any], ts: float, upvotes: int, comments: int, score: int) -> None:
    history = entry.get("metrics_history")
    if not isinstance(history, list):
        history = []
    sample = {
        "ts": float(ts),
        "upvotes": int(upvotes),
        "comment_count": int(comments),
        "engagement_score": int(score),
    }
    if history:
        last = history[-1]
        if isinstance(last, dict):
            if (
                int(last.get("upvotes", 0) or 0) == sample["upvotes"]
                and int(last.get("comment_count", 0) or 0) == sample["comment_count"]
                and int(last.get("engagement_score", 0) or 0) == sample["engagement_score"]
            ):
                return
    history.append(sample)
    entry["metrics_history"] = history[-24:]


def _metric_velocity_per_hour(entry: Dict[str, Any], key: str) -> float:
    history = entry.get("metrics_history")
    if isinstance(history, list) and len(history) >= 2:
        first = history[0]
        last = history[-1]
        if isinstance(first, dict) and isinstance(last, dict):
            first_ts = first.get("ts")
            last_ts = last.get("ts")
            first_val = first.get(key)
            last_val = last.get(key)
            if isinstance(first_ts, (int, float)) and isinstance(last_ts, (int, float)):
                dt_hours = max((float(last_ts) - float(first_ts)) / 3600.0, 1.0 / 60.0)
                if isinstance(first_val, (int, float)) and isinstance(last_val, (int, float)):
                    return max(0.0, float(last_val) - float(first_val)) / dt_hours

    metric = entry.get(key)
    if not isinstance(metric, (int, float)):
        return 0.0
    age_hours = _entry_age_hours(entry)
    return max(0.0, float(metric)) / max(age_hours, 1.0 / 60.0)


def _entry_visibility_score(entry: Dict[str, Any]) -> float:
    upvotes = int(entry.get("upvotes", 0) or 0)
    comments = int(entry.get("comment_count", 0) or 0)
    engagement = float(entry.get("engagement_score", upvotes + (comments * 2)) or 0.0)
    upvote_velocity = _metric_velocity_per_hour(entry, "upvotes")
    comment_velocity = _metric_velocity_per_hour(entry, "comment_count")
    velocity_bonus = min(25.0, (upvote_velocity * 8.0) + (comment_velocity * 16.0))
    return round(engagement + velocity_bonus, 3)


def record_proactive_post(
    memory: Dict[str, Any],
    post_id: str,
    title: str,
    submolt: str,
    content: str,
    strategy_notes: str,
    topic_tags: List[str],
    content_archetype: str = "unknown",
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
            "content_archetype": normalize_str(content_archetype).strip().lower() or "unknown",
            "created_ts": now_ts,
            "upvotes": None,
            "comment_count": None,
            "engagement_score": None,
            "visibility_score": None,
            "metrics_history": [],
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
        _append_metrics_sample(entry=entry, ts=now_ts, upvotes=upvotes, comments=comments, score=score)
        visibility_score = _entry_visibility_score(entry)
        if (
            entry.get("upvotes") != upvotes
            or entry.get("comment_count") != comments
            or entry.get("engagement_score") != score
            or float(entry.get("visibility_score", 0.0) or 0.0) != float(visibility_score)
        ):
            updated += 1
        entry["upvotes"] = upvotes
        entry["comment_count"] = comments
        entry["engagement_score"] = score
        entry["visibility_score"] = visibility_score
        entry["last_metrics_ts"] = now_ts
    memory["last_metrics_refresh_ts"] = now_ts
    return updated


def _extract_terms(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]{4,}", normalize_str(text).lower())
    return [t for t in tokens if t not in STOPWORDS]


def _top_terms(items: List[Dict[str, Any]], limit: int) -> List[str]:
    counts: Dict[str, int] = {}
    for item in items:
        text = (
            f"{normalize_str(item.get('title'))} "
            f"{normalize_str(item.get('strategy_notes'))} "
            f"{normalize_str(item.get('content_preview'))}"
        )
        for token in _extract_terms(text):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in ranked[: max(0, limit)]]


def _term_presence_rates(items: List[Dict[str, Any]]) -> Dict[str, float]:
    if not items:
        return {}
    counts: Dict[str, int] = {}
    for item in items:
        text = (
            f"{normalize_str(item.get('title'))} "
            f"{normalize_str(item.get('strategy_notes'))} "
            f"{normalize_str(item.get('content_preview'))}"
        )
        for token in set(_extract_terms(text)):
            counts[token] = counts.get(token, 0) + 1
    total = float(len(items))
    return {token: (count / total) for token, count in counts.items() if count > 0}


def _build_term_lift(
    winners: List[Dict[str, Any]],
    losers: List[Dict[str, Any]],
    limit: int = 12,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    win_rates = _term_presence_rates(winners)
    lose_rates = _term_presence_rates(losers)
    all_terms = set(win_rates.keys()) | set(lose_rates.keys())
    if not all_terms:
        return [], []
    weighted: List[tuple[str, float, float, float]] = []
    for term in all_terms:
        win = float(win_rates.get(term, 0.0))
        lose = float(lose_rates.get(term, 0.0))
        lift = round(win - lose, 4)
        weighted.append((term, lift, win, lose))
    weighted.sort(key=lambda x: x[1], reverse=True)
    positive = [
        {"term": term, "lift": lift, "win_rate": round(win, 3), "loss_rate": round(lose, 3)}
        for term, lift, win, lose in weighted
        if lift > 0
    ][: max(1, int(limit))]
    negative = [
        {"term": term, "lift": lift, "win_rate": round(win, 3), "loss_rate": round(lose, 3)}
        for term, lift, win, lose in reversed(weighted)
        if lift < 0
    ][: max(1, int(limit))]
    return positive, negative


def _safe_avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _median_int(values: List[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return int(sorted_values[mid])
    return int((sorted_values[mid - 1] + sorted_values[mid]) / 2)


def update_market_signals(
    memory: Dict[str, Any],
    signals: List[Dict[str, Any]],
    max_entries: int = 120,
) -> Dict[str, Any]:
    cleaned: List[Dict[str, Any]] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        pid = normalize_str(signal.get("post_id")).strip()
        title = normalize_str(signal.get("title")).strip()
        if not pid or not title:
            continue
        cleaned.append(
            {
                "post_id": pid,
                "title": title[:220],
                "submolt": normalize_str(signal.get("submolt")).strip().lower(),
                "score": int(signal.get("score") or 0),
                "comment_count": int(signal.get("comment_count") or 0),
                "source": normalize_str(signal.get("source")).strip().lower() or "unknown",
            }
        )

    existing = memory.get("last_market_signals", [])
    if not isinstance(existing, list):
        existing = []
    combined = []
    for item in existing:
        if isinstance(item, dict):
            combined.append(item)
    combined.extend(cleaned)
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in combined:
        pid = normalize_str(item.get("post_id")).strip()
        source = normalize_str(item.get("source")).strip().lower() or "unknown"
        if not pid:
            continue
        key = f"{pid}|{source}"
        dedup[key] = item
    memory["last_market_signals"] = list(dedup.values())[-max(20, max_entries):]
    source_counts: Dict[str, int] = {}
    submolt_scores: Dict[str, List[float]] = {}
    title_lengths: List[int] = []
    question_titles = 0
    term_counts: Dict[str, int] = {}

    for item in memory["last_market_signals"]:
        source = normalize_str(item.get("source")).strip().lower() or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
        submolt = normalize_str(item.get("submolt")).strip().lower()
        if submolt:
            weighted_score = float(item.get("score", 0)) + (float(item.get("comment_count", 0)) * 2.0)
            submolt_scores.setdefault(submolt, []).append(weighted_score)
        title = normalize_str(item.get("title")).strip()
        if title:
            title_lengths.append(len(title))
            if "?" in title:
                question_titles += 1
            for token in _extract_terms(title):
                term_counts[token] = term_counts.get(token, 0) + 1

    ranked_submolts = sorted(
        [(name, _safe_avg(scores)) for name, scores in submolt_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    ranked_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
    total_titles = len(title_lengths)
    market_snapshot = {
        "ts": utc_now().isoformat(),
        "signal_count": len(memory["last_market_signals"]),
        "source_counts": source_counts,
        "question_title_rate": round((question_titles / total_titles), 3) if total_titles else 0.0,
        "median_title_chars": _median_int(title_lengths),
        "top_terms": [term for term, _ in ranked_terms[:12]],
        "top_submolts": [
            {"name": name, "avg_weighted_score": round(avg_score, 2)}
            for name, avg_score in ranked_submolts[:8]
        ],
    }
    memory["last_market_snapshot"] = market_snapshot
    return market_snapshot


def _build_visibility_hypotheses(
    winning_terms: List[str],
    losing_terms: List[str],
    best_submolts: List[Dict[str, Any]],
    market_snapshot: Dict[str, Any],
    visibility_metrics: Optional[Dict[str, Any]] = None,
    winning_terms_lift: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    hints: List[str] = []
    winning_set = [t for t in winning_terms if t and t not in set(losing_terms)]
    if winning_set:
        hints.append("Lean into high-signal terms that already won for us: " + ", ".join(winning_set[:6]))
    if isinstance(winning_terms_lift, list) and winning_terms_lift:
        lifted = [normalize_str(item.get("term")).strip() for item in winning_terms_lift if isinstance(item, dict)]
        lifted = [term for term in lifted if term]
        if lifted:
            hints.append("Use terms with measured lift in our own posts: " + ", ".join(lifted[:6]))
    if best_submolts:
        top_submolt_names = [normalize_str(x.get("name")).strip() for x in best_submolts[:3] if normalize_str(x.get("name")).strip()]
        if top_submolt_names:
            hints.append("Prioritize proactive posts in strongest submolts: " + ", ".join(top_submolt_names))
    if isinstance(visibility_metrics, dict):
        target_upvotes = int(visibility_metrics.get("target_upvotes", 0) or 0)
        hit_rate = float(visibility_metrics.get("recent_target_hit_rate", 0.0) or 0.0)
        delta_pct = float(visibility_metrics.get("visibility_delta_pct", 0.0) or 0.0)
        if target_upvotes > 0 and hit_rate < 0.35:
            hints.append(
                f"Recent visibility is below target ({target_upvotes} upvotes): hit_rate={round(hit_rate, 3)}. "
                "Prioritize sharper hooks plus concrete implementation stakes."
            )
        if delta_pct <= -0.15:
            hints.append(
                f"Visibility trend is down ({round(delta_pct * 100, 1)}%). "
                "Favor contrarian mechanism-first takes instead of generic summaries."
            )
        win_q_rate = float(visibility_metrics.get("winning_question_title_rate", 0.0) or 0.0)
        if win_q_rate >= 0.4:
            hints.append("Our winning posts often use question-led titles. Keep direct challenge hooks in titles.")
    question_rate = market_snapshot.get("question_title_rate")
    if isinstance(question_rate, (int, float)) and question_rate >= 0.35:
        hints.append("Question-style titles are currently overperforming in market-wide hot/top posts.")
    top_market_terms = market_snapshot.get("top_terms")
    if isinstance(top_market_terms, list) and top_market_terms:
        filtered = [normalize_str(t).strip() for t in top_market_terms if normalize_str(t).strip()]
        if filtered:
            hints.append("Inject current market language from hot/rising feeds: " + ", ".join(filtered[:6]))
    if not hints:
        hints.append("Keep testing concrete, contrarian, mechanism-first posts and measure engagement changes.")
    return hints[:6]


def build_learning_snapshot(memory: Dict[str, Any], max_examples: int = 5) -> Dict[str, Any]:
    entries = [e for e in memory.get("proactive_posts", []) if isinstance(e, dict)]
    scored = [e for e in entries if isinstance(e.get("engagement_score"), (int, float))]
    for entry in scored:
        if not isinstance(entry.get("visibility_score"), (int, float)):
            entry["visibility_score"] = _entry_visibility_score(entry)
    scored.sort(key=lambda e: float(e.get("visibility_score", e.get("engagement_score", 0))), reverse=True)

    winners = scored[: max_examples]
    losers = list(reversed(scored[-max_examples:])) if scored else []
    winning_terms_lift, losing_terms_lift = _build_term_lift(winners=winners, losers=losers, limit=12)

    scored_by_recency = sorted(
        scored,
        key=lambda e: float(e.get("created_ts", 0.0) or 0.0),
        reverse=True,
    )
    recent_window = _visibility_recent_window()
    recent = scored_by_recency[:recent_window]
    baseline = scored_by_recency[recent_window : recent_window * 2]

    def _avg(items: List[Dict[str, Any]], key: str) -> float:
        values: List[float] = []
        for item in items:
            value = item.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return round(_safe_avg(values), 3) if values else 0.0

    recent_avg_upvotes = _avg(recent, "upvotes")
    recent_avg_comments = _avg(recent, "comment_count")
    recent_avg_visibility = _avg(recent, "visibility_score")
    baseline_avg_visibility = _avg(baseline, "visibility_score")
    visibility_delta_pct = 0.0
    if baseline_avg_visibility > 0:
        visibility_delta_pct = round((recent_avg_visibility - baseline_avg_visibility) / baseline_avg_visibility, 4)
    target_upvotes = _visibility_target_upvotes()
    recent_target_hits = 0
    for item in recent:
        upvotes = item.get("upvotes")
        if isinstance(upvotes, (int, float)) and float(upvotes) >= float(target_upvotes):
            recent_target_hits += 1
    recent_target_hit_rate = round(_safe_rate(recent_target_hits, len(recent)), 4)
    winning_question_title_rate = round(
        _safe_rate(
            len([1 for item in winners if "?" in normalize_str(item.get("title")).strip()]),
            len(winners),
        ),
        4,
    )
    visibility_metrics = {
        "target_upvotes": target_upvotes,
        "recent_window_size": len(recent),
        "baseline_window_size": len(baseline),
        "recent_avg_upvotes": recent_avg_upvotes,
        "recent_avg_comments": recent_avg_comments,
        "recent_avg_visibility_score": recent_avg_visibility,
        "baseline_avg_visibility_score": baseline_avg_visibility,
        "visibility_delta_pct": visibility_delta_pct,
        "recent_target_hit_rate": recent_target_hit_rate,
        "winning_question_title_rate": winning_question_title_rate,
    }

    best_submolt: Dict[str, List[float]] = {}
    for entry in scored:
        submolt = normalize_str(entry.get("submolt")).strip().lower()
        if not submolt:
            continue
        best_submolt.setdefault(submolt, []).append(float(entry.get("visibility_score", entry.get("engagement_score", 0))))
    submolt_rank = sorted(
        [(k, sum(v) / max(1, len(v))) for k, v in best_submolt.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    archetype_scores: Dict[str, List[float]] = {}
    for entry in scored:
        archetype = normalize_str(entry.get("content_archetype")).strip().lower()
        if not archetype:
            continue
        archetype_scores.setdefault(archetype, []).append(
            float(entry.get("visibility_score", entry.get("engagement_score", 0)))
        )
    best_archetypes = sorted(
        [(name, _safe_avg(scores)) for name, scores in archetype_scores.items()],
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
                "content_archetype": normalize_str(e.get("content_archetype")).strip().lower(),
                "visibility_score": e.get("visibility_score"),
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
                "content_archetype": normalize_str(e.get("content_archetype")).strip().lower(),
                "visibility_score": e.get("visibility_score"),
            }
            for e in losers
        ],
        "winning_terms": _top_terms(winners, limit=8),
        "losing_terms": _top_terms(losers, limit=8),
        "winning_terms_lift": winning_terms_lift,
        "losing_terms_lift": losing_terms_lift,
        "best_submolts": [{"name": name, "avg_score": round(avg, 2)} for name, avg in submolt_rank[:5]],
        "best_archetypes": [
            {"name": name, "avg_score": round(avg, 2)}
            for name, avg in best_archetypes[:5]
        ],
        "market_snapshot": memory.get("last_market_snapshot", {}),
        "visibility_metrics": visibility_metrics,
    }
    snapshot["visibility_hypotheses"] = _build_visibility_hypotheses(
        winning_terms=snapshot["winning_terms"],
        losing_terms=snapshot["losing_terms"],
        best_submolts=snapshot["best_submolts"],
        market_snapshot=snapshot.get("market_snapshot") if isinstance(snapshot.get("market_snapshot"), dict) else {},
        visibility_metrics=visibility_metrics,
        winning_terms_lift=winning_terms_lift,
    )
    memory["last_snapshot"] = snapshot
    return snapshot


def load_recent_improvement_entries(path: Path, limit: int = 6) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries[-max(1, limit):]:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "ts": normalize_str(entry.get("ts")).strip(),
                "summary": normalize_str(entry.get("summary")).strip()[:220],
                "priority": normalize_str(entry.get("priority")).strip() or "medium",
                "prompt_change_count": len(entry.get("prompt_changes") or []) if isinstance(entry.get("prompt_changes"), list) else 0,
                "code_change_count": len(entry.get("code_changes") or []) if isinstance(entry.get("code_changes"), list) else 0,
                "experiment_count": (
                    len(entry.get("strategy_experiments") or [])
                    if isinstance(entry.get("strategy_experiments"), list)
                    else 0
                ),
                "bottleneck": normalize_str((entry.get("diagnostics") or {}).get("bottleneck_label")).strip(),
                "approval_rate": (entry.get("diagnostics") or {}).get("approval_rate"),
                "execution_rate": (entry.get("diagnostics") or {}).get("execution_rate"),
            }
        )
    return out


def load_recent_improvement_raw_entries(path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries[-max(1, limit):]:
        if isinstance(entry, dict):
            out.append(entry)
    return out


def _normalize_signature(text: Any) -> str:
    value = normalize_str(text).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = " ".join(value.split())
    return value[:260]


def _collect_prior_signatures(entries: List[Dict[str, Any]]) -> set[str]:
    signatures: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary_sig = _normalize_signature(entry.get("summary"))
        if len(summary_sig) >= 12:
            signatures.add(f"summary:{summary_sig}")
        for key, fields in (
            ("prompt_changes", ("target", "proposed_change")),
            ("code_changes", ("file_hint", "proposed_change")),
            ("strategy_experiments", ("idea", "metric", "stop_condition")),
        ):
            raw = entry.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                joined = " ".join([normalize_str(item.get(field)) for field in fields])
                sig = _normalize_signature(joined)
                if len(sig) >= 12:
                    signatures.add(f"{key}:{sig}")
    return signatures


def _strategy_disallowed(item: Dict[str, Any]) -> bool:
    blob = _normalize_signature(
        " ".join(
            [
                normalize_str(item.get("idea")),
                normalize_str(item.get("metric")),
                normalize_str(item.get("stop_condition")),
            ]
        )
    )
    if not blob:
        return True
    return any(token in blob for token in DISALLOWED_STRATEGY_TOKENS)


def _code_change_actionable(item: Dict[str, Any]) -> bool:
    file_hint = normalize_str(item.get("file_hint")).strip()
    proposed_change = normalize_str(item.get("proposed_change")).strip()
    if not file_hint or not proposed_change:
        return False
    allowed_prefixes = (
        "src/moltbook/",
        "tools/",
        "scripts/",
        "docs/",
        "README.md",
        ".env.example",
    )
    return file_hint.startswith(allowed_prefixes)


def build_improvement_diagnostics(cycle_stats: Dict[str, Any]) -> Dict[str, Any]:
    inspected = int(cycle_stats.get("inspected", 0) or 0)
    new_candidates = int(cycle_stats.get("new_candidates", 0) or 0)
    eligible_now = int(cycle_stats.get("eligible_now", 0) or 0)
    drafted = int(cycle_stats.get("drafted", 0) or 0)
    model_approved = int(cycle_stats.get("model_approved", 0) or 0)
    actions = int(cycle_stats.get("actions", 0) or 0)

    candidate_rate = round(_safe_rate(new_candidates, max(1, inspected)), 4)
    draft_rate = round(_safe_rate(drafted, max(1, eligible_now)), 4)
    approval_rate = round(_safe_rate(model_approved, max(1, drafted)), 4)
    action_rate = round(_safe_rate(actions, max(1, model_approved)), 4)
    execution_rate = round(_safe_rate(actions, max(1, eligible_now)), 4)

    raw_skip = cycle_stats.get("skip_reasons")
    skip_reasons: Dict[str, int] = {}
    if isinstance(raw_skip, dict):
        for key, value in raw_skip.items():
            name = normalize_str(key).strip()
            if not name:
                continue
            try:
                skip_reasons[name] = int(value)
            except Exception:
                continue
    analysis_skip = {k: v for k, v in skip_reasons.items() if k != "draft_shortlist_cap"}
    if not analysis_skip:
        analysis_skip = dict(skip_reasons)
    top_skip_reasons = sorted(analysis_skip.items(), key=lambda x: x[1], reverse=True)[:5]
    top_skip_names = [name for name, _ in top_skip_reasons]

    bottleneck_label = "balanced"
    if drafted >= 10 and model_approved == 0:
        bottleneck_label = "model_rejection"
    elif eligible_now >= 15 and drafted == 0:
        bottleneck_label = "draft_suppression"
    elif model_approved > 0 and actions == 0:
        bottleneck_label = "execution_blocked"
    elif any("cooldown" in name for name in top_skip_names):
        bottleneck_label = "cooldown_limited"
    elif any("already_replied" in name or "already_seen" in name for name in top_skip_names):
        bottleneck_label = "duplication_pressure"
    elif actions == 0 and eligible_now > 0:
        bottleneck_label = "low_conversion"

    return {
        "inspected": inspected,
        "new_candidates": new_candidates,
        "eligible_now": eligible_now,
        "drafted": drafted,
        "model_approved": model_approved,
        "actions": actions,
        "candidate_rate": candidate_rate,
        "draft_rate": draft_rate,
        "approval_rate": approval_rate,
        "action_rate": action_rate,
        "execution_rate": execution_rate,
        "top_skip_reasons": [{"name": name, "count": count} for name, count in top_skip_reasons],
        "bottleneck_label": bottleneck_label,
    }


def sanitize_improvement_suggestions(
    suggestions: Dict[str, Any],
    recent_raw_entries: List[Dict[str, Any]],
    max_items: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "summary": normalize_str(suggestions.get("summary")).strip(),
        "priority": normalize_str(suggestions.get("priority")).strip() or "medium",
        "prompt_changes": [],
        "code_changes": [],
        "strategy_experiments": [],
    }
    max_items = max(1, int(max_items))
    prior_signatures = _collect_prior_signatures(recent_raw_entries)
    local_signatures: set[str] = set()

    def _can_add(kind: str, text: str) -> bool:
        sig = _normalize_signature(text)
        if len(sig) < 12:
            return False
        key = f"{kind}:{sig}"
        if key in prior_signatures:
            return False
        if key in local_signatures:
            return False
        local_signatures.add(key)
        return True

    prompt_changes = suggestions.get("prompt_changes")
    if isinstance(prompt_changes, list):
        for item in prompt_changes:
            if not isinstance(item, dict):
                continue
            joined = " ".join(
                [
                    normalize_str(item.get("target")),
                    normalize_str(item.get("proposed_change")),
                    normalize_str(item.get("reason")),
                ]
            )
            if not _can_add("prompt_changes", joined):
                continue
            out["prompt_changes"].append(item)
            if len(out["prompt_changes"]) >= max_items:
                break

    code_changes = suggestions.get("code_changes")
    if isinstance(code_changes, list):
        for item in code_changes:
            if not isinstance(item, dict):
                continue
            if not _code_change_actionable(item):
                continue
            joined = " ".join(
                [
                    normalize_str(item.get("file_hint")),
                    normalize_str(item.get("proposed_change")),
                    normalize_str(item.get("reason")),
                ]
            )
            if not _can_add("code_changes", joined):
                continue
            out["code_changes"].append(item)
            if len(out["code_changes"]) >= max_items:
                break

    strategy_experiments = suggestions.get("strategy_experiments")
    if isinstance(strategy_experiments, list):
        for item in strategy_experiments:
            if not isinstance(item, dict):
                continue
            if _strategy_disallowed(item):
                continue
            joined = " ".join(
                [
                    normalize_str(item.get("idea")),
                    normalize_str(item.get("metric")),
                    normalize_str(item.get("stop_condition")),
                ]
            )
            if not _can_add("strategy_experiments", joined):
                continue
            out["strategy_experiments"].append(item)
            if len(out["strategy_experiments"]) >= max_items:
                break

    if not out["summary"]:
        out["summary"] = "No novel actionable suggestions this cycle."
    return out


def build_improvement_feedback_context(
    path: Path,
    current_cycle_stats: Dict[str, Any],
    limit: int = 18,
) -> Dict[str, Any]:
    current = build_improvement_diagnostics(current_cycle_stats)
    entries = load_recent_improvement_raw_entries(path=path, limit=limit)

    historical_diags: List[Dict[str, Any]] = []
    bottleneck_counts: Dict[str, int] = {}
    recurring_targets: Dict[str, int] = {}
    recent_bottlenecks: List[str] = []
    recent_upvotes: List[float] = []
    recent_visibility: List[float] = []
    target_hit_rates: List[float] = []
    visibility_term_counts: Dict[str, int] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        diag = entry.get("diagnostics")
        if isinstance(diag, dict):
            historical_diags.append(diag)
            label = normalize_str(diag.get("bottleneck_label")).strip()
            if label:
                bottleneck_counts[label] = bottleneck_counts.get(label, 0) + 1
                recent_bottlenecks.append(label)
        prompt_changes = entry.get("prompt_changes")
        if isinstance(prompt_changes, list):
            for item in prompt_changes:
                if not isinstance(item, dict):
                    continue
                target = normalize_str(item.get("target")).strip().lower()
                if target:
                    recurring_targets[target] = recurring_targets.get(target, 0) + 1
        cycle_stats = entry.get("cycle_stats")
        if isinstance(cycle_stats, dict):
            upvotes = cycle_stats.get("proactive_recent_avg_upvotes")
            visibility = cycle_stats.get("proactive_recent_avg_visibility_score")
            hit_rate = cycle_stats.get("proactive_target_hit_rate")
            if isinstance(upvotes, (int, float)):
                recent_upvotes.append(float(upvotes))
            if isinstance(visibility, (int, float)):
                recent_visibility.append(float(visibility))
            if isinstance(hit_rate, (int, float)):
                target_hit_rates.append(float(hit_rate))
            top_terms = cycle_stats.get("proactive_top_lift_terms")
            if isinstance(top_terms, list):
                for token in top_terms:
                    clean = _normalize_signature(token)
                    if clean:
                        visibility_term_counts[clean] = visibility_term_counts.get(clean, 0) + 1

    avg_approval = _safe_avg(
        [
            float(diag.get("approval_rate", 0.0))
            for diag in historical_diags
            if isinstance(diag.get("approval_rate"), (int, float))
        ]
    )
    avg_execution = _safe_avg(
        [
            float(diag.get("execution_rate", 0.0))
            for diag in historical_diags
            if isinstance(diag.get("execution_rate"), (int, float))
        ]
    )
    avg_action = _safe_avg(
        [
            float(diag.get("action_rate", 0.0))
            for diag in historical_diags
            if isinstance(diag.get("action_rate"), (int, float))
        ]
    )

    # Count recent cycles (tail) that produced no actions.
    zero_action_streak = 0
    for diag in reversed(historical_diags):
        actions = int(diag.get("actions", 0) or 0)
        if actions == 0:
            zero_action_streak += 1
            continue
        break

    sorted_bottlenecks = sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_targets = sorted(recurring_targets.items(), key=lambda x: x[1], reverse=True)
    top_visibility_terms = sorted(visibility_term_counts.items(), key=lambda x: x[1], reverse=True)

    def _series_trend(series: List[float]) -> float:
        if len(series) < 2:
            return 0.0
        return round(series[-1] - series[0], 4)

    visibility_feedback = {
        "avg_recent_upvotes": round(_safe_avg(recent_upvotes), 3),
        "avg_recent_visibility_score": round(_safe_avg(recent_visibility), 3),
        "avg_target_hit_rate": round(_safe_avg(target_hit_rates), 4),
        "recent_upvotes_trend": _series_trend(recent_upvotes),
        "recent_visibility_trend": _series_trend(recent_visibility),
        "top_visibility_terms": [term for term, _ in top_visibility_terms[:8]],
    }

    return {
        "current_diagnostics": current,
        "historical_window": len(historical_diags),
        "avg_approval_rate": round(avg_approval, 4),
        "avg_execution_rate": round(avg_execution, 4),
        "avg_action_rate": round(avg_action, 4),
        "zero_action_streak": zero_action_streak,
        "top_bottlenecks": [{"name": name, "count": count} for name, count in sorted_bottlenecks[:5]],
        "recent_bottlenecks": recent_bottlenecks[-8:],
        "recurring_prompt_targets": [{"name": name, "count": count} for name, count in sorted_targets[:6]],
        "visibility_feedback": visibility_feedback,
    }


def _priority_weight(priority: str) -> float:
    value = normalize_str(priority).strip().lower()
    if value == "high":
        return 1.5
    if value == "medium":
        return 0.75
    return 0.25


def _backlog_item_signature(kind: str, item: Dict[str, Any]) -> str:
    if kind == "prompt_changes":
        text = " ".join([normalize_str(item.get("target")), normalize_str(item.get("proposed_change"))])
    elif kind == "code_changes":
        text = " ".join([normalize_str(item.get("file_hint")), normalize_str(item.get("proposed_change"))])
    else:
        text = " ".join(
            [
                normalize_str(item.get("idea")),
                normalize_str(item.get("metric")),
                normalize_str(item.get("stop_condition")),
            ]
        )
    return _normalize_signature(text)


def update_improvement_backlog(
    path: Path,
    cycle: int,
    provider: str,
    suggestions: Dict[str, Any],
    diagnostics: Dict[str, Any],
    max_items: int = 300,
) -> Dict[str, Any]:
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

    existing = payload.get("items")
    items: List[Dict[str, Any]] = [item for item in existing if isinstance(item, dict)] if isinstance(existing, list) else []
    by_signature: Dict[str, Dict[str, Any]] = {}
    for item in items:
        sig = normalize_str(item.get("signature")).strip()
        if sig:
            by_signature[sig] = item

    priority = normalize_str(suggestions.get("priority")).strip().lower() or "medium"
    now_iso = utc_now().isoformat()
    bottleneck = normalize_str(diagnostics.get("bottleneck_label")).strip() or "unknown"

    def _upsert(kind: str, raw_item: Dict[str, Any]) -> None:
        sig_core = _backlog_item_signature(kind, raw_item)
        if len(sig_core) < 12:
            return
        signature = f"{kind}:{sig_core}"
        if kind == "prompt_changes":
            headline = normalize_str(raw_item.get("target")).strip() or "(prompt target)"
            change = normalize_str(raw_item.get("proposed_change")).strip()
        elif kind == "code_changes":
            headline = normalize_str(raw_item.get("file_hint")).strip() or "(code file)"
            change = normalize_str(raw_item.get("proposed_change")).strip()
        else:
            headline = normalize_str(raw_item.get("idea")).strip() or "(strategy)"
            change = normalize_str(raw_item.get("metric")).strip()

        if signature in by_signature:
            entry = by_signature[signature]
            entry["seen_count"] = int(entry.get("seen_count", 0) or 0) + 1
            entry["last_seen_cycle"] = int(cycle)
            entry["last_seen_ts"] = now_iso
            entry["last_priority"] = priority
            entry["last_bottleneck"] = bottleneck
            providers = entry.get("providers", [])
            if not isinstance(providers, list):
                providers = []
            provider_name = normalize_str(provider).strip() or "unknown"
            if provider_name and provider_name not in providers:
                providers.append(provider_name)
            entry["providers"] = providers[-6:]
        else:
            by_signature[signature] = {
                "signature": signature,
                "kind": kind,
                "headline": headline[:200],
                "change": change[:400],
                "seen_count": 1,
                "first_seen_cycle": int(cycle),
                "last_seen_cycle": int(cycle),
                "first_seen_ts": now_iso,
                "last_seen_ts": now_iso,
                "last_priority": priority,
                "last_bottleneck": bottleneck,
                "providers": [normalize_str(provider).strip() or "unknown"],
            }

    for kind in ("prompt_changes", "code_changes", "strategy_experiments"):
        raw = suggestions.get(kind)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                _upsert(kind, item)

    all_items = list(by_signature.values())
    for item in all_items:
        kind = normalize_str(item.get("kind")).strip()
        seen = int(item.get("seen_count", 0) or 0)
        weight = _priority_weight(normalize_str(item.get("last_priority")).strip())
        kind_weight = 0.6 if kind == "code_changes" else (0.4 if kind == "prompt_changes" else 0.2)
        item["priority_score"] = round(seen + weight + kind_weight, 3)

    all_items.sort(
        key=lambda x: (
            float(x.get("priority_score", 0.0)),
            int(x.get("last_seen_cycle", 0)),
            int(x.get("seen_count", 0)),
        ),
        reverse=True,
    )
    payload["items"] = all_items[: max(20, int(max_items))]
    payload["last_updated_ts"] = now_iso
    payload["last_cycle"] = int(cycle)
    payload["top_bottleneck"] = bottleneck

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def append_improvement_suggestions(
    path: Path,
    cycle: int,
    provider: str,
    suggestions: Dict[str, Any],
    cycle_stats: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
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
        "cycle_stats": cycle_stats if isinstance(cycle_stats, dict) else {},
        "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
    }
    entries.append(entry)
    payload["entries"] = entries[-max(10, max_entries):]
    payload["last_updated_ts"] = utc_now().isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_improvement_suggestions_text(
    path: Path,
    cycle: int,
    provider: str,
    suggestions: Dict[str, Any],
    cycle_stats: Dict[str, Any],
    learning_snapshot: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    feedback_context: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"Timestamp: {utc_now().isoformat()}")
    lines.append(f"Cycle: {int(cycle)}")
    lines.append(f"Provider: {normalize_str(provider).strip() or 'unknown'}")
    lines.append(f"Priority: {normalize_str(suggestions.get('priority')).strip() or 'medium'}")
    lines.append("")
    summary = normalize_str(suggestions.get("summary")).strip() or "(no summary)"
    lines.append("Summary:")
    lines.append(summary)
    lines.append("")
    lines.append("Cycle Stats:")
    lines.append(
        "inspected={inspected} new_candidates={new_candidates} eligible_now={eligible_now} "
        "drafted={drafted} model_approved={model_approved} actions={actions}".format(
            inspected=cycle_stats.get("inspected", 0),
            new_candidates=cycle_stats.get("new_candidates", 0),
            eligible_now=cycle_stats.get("eligible_now", 0),
            drafted=cycle_stats.get("drafted", 0),
            model_approved=cycle_stats.get("model_approved", 0),
            actions=cycle_stats.get("actions", 0),
        )
    )
    lines.append("")
    if isinstance(diagnostics, dict):
        lines.append("Diagnostics:")
        lines.append(
            (
                "bottleneck={bottleneck} candidate_rate={candidate_rate:.3f} "
                "draft_rate={draft_rate:.3f} approval_rate={approval_rate:.3f} "
                "execution_rate={execution_rate:.3f}"
            ).format(
                bottleneck=normalize_str(diagnostics.get("bottleneck_label")).strip() or "unknown",
                candidate_rate=float(diagnostics.get("candidate_rate", 0.0) or 0.0),
                draft_rate=float(diagnostics.get("draft_rate", 0.0) or 0.0),
                approval_rate=float(diagnostics.get("approval_rate", 0.0) or 0.0),
                execution_rate=float(diagnostics.get("execution_rate", 0.0) or 0.0),
            )
        )
        top_skip = diagnostics.get("top_skip_reasons")
        if isinstance(top_skip, list) and top_skip:
            compact = []
            for item in top_skip[:5]:
                if not isinstance(item, dict):
                    continue
                name = normalize_str(item.get("name")).strip()
                count = item.get("count")
                if name:
                    compact.append(f"{name}={count}")
            if compact:
                lines.append("top_skip_reasons=" + ", ".join(compact))
        lines.append("")
    if isinstance(feedback_context, dict):
        lines.append("Feedback Context:")
        lines.append(
            (
                "historical_window={window} avg_approval_rate={approval:.3f} "
                "avg_execution_rate={execution:.3f} zero_action_streak={streak}"
            ).format(
                window=int(feedback_context.get("historical_window", 0) or 0),
                approval=float(feedback_context.get("avg_approval_rate", 0.0) or 0.0),
                execution=float(feedback_context.get("avg_execution_rate", 0.0) or 0.0),
                streak=int(feedback_context.get("zero_action_streak", 0) or 0),
            )
        )
        top_bottlenecks = feedback_context.get("top_bottlenecks")
        if isinstance(top_bottlenecks, list) and top_bottlenecks:
            compact = []
            for item in top_bottlenecks[:5]:
                if not isinstance(item, dict):
                    continue
                name = normalize_str(item.get("name")).strip()
                count = item.get("count")
                if name:
                    compact.append(f"{name}={count}")
            if compact:
                lines.append("top_bottlenecks=" + ", ".join(compact))
        visibility_feedback = feedback_context.get("visibility_feedback")
        if isinstance(visibility_feedback, dict):
            lines.append(
                (
                    "visibility avg_recent_upvotes={upvotes:.2f} avg_recent_visibility={visibility:.2f} "
                    "avg_target_hit_rate={hit_rate:.3f} upvotes_trend={uptrend:.3f} "
                    "visibility_trend={vtrend:.3f}"
                ).format(
                    upvotes=float(visibility_feedback.get("avg_recent_upvotes", 0.0) or 0.0),
                    visibility=float(visibility_feedback.get("avg_recent_visibility_score", 0.0) or 0.0),
                    hit_rate=float(visibility_feedback.get("avg_target_hit_rate", 0.0) or 0.0),
                    uptrend=float(visibility_feedback.get("recent_upvotes_trend", 0.0) or 0.0),
                    vtrend=float(visibility_feedback.get("recent_visibility_trend", 0.0) or 0.0),
                )
            )
            top_terms = visibility_feedback.get("top_visibility_terms")
            if isinstance(top_terms, list) and top_terms:
                compact_terms = [normalize_str(term).strip() for term in top_terms if normalize_str(term).strip()]
                if compact_terms:
                    lines.append("visibility_top_terms=" + ", ".join(compact_terms[:8]))
        lines.append("")
    lines.append("Visibility Hypotheses:")
    hypotheses = learning_snapshot.get("visibility_hypotheses", [])
    if isinstance(hypotheses, list) and hypotheses:
        for idx, text in enumerate(hypotheses, start=1):
            lines.append(f"{idx}. {normalize_str(text).strip()}")
    else:
        lines.append("1. (none)")
    lines.append("")

    def _append_change_block(title: str, items: Any, fields: List[str]) -> None:
        lines.append(title + ":")
        if not isinstance(items, list) or not items:
            lines.append("1. (none)")
            lines.append("")
            return
        counter = 1
        for item in items:
            if not isinstance(item, dict):
                continue
            lines.append(f"{counter}.")
            for field in fields:
                value = normalize_str(item.get(field)).strip()
                if value:
                    lines.append(f"   {field}: {value}")
            counter += 1
        if counter == 1:
            lines.append("1. (none)")
        lines.append("")

    _append_change_block(
        "Prompt Changes",
        suggestions.get("prompt_changes"),
        ["target", "proposed_change", "reason", "expected_impact"],
    )
    _append_change_block(
        "Code Changes",
        suggestions.get("code_changes"),
        ["file_hint", "proposed_change", "reason", "risk"],
    )
    _append_change_block(
        "Strategy Experiments",
        suggestions.get("strategy_experiments"),
        ["idea", "metric", "stop_condition"],
    )

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n\n")
