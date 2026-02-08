from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .drafting import normalize_str
from .runtime_helpers import extract_posts, post_id
from .strategy import post_comment_count, post_score


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_analytics_db(path: Path) -> None:
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_post_id TEXT,
                submolt TEXT,
                feed_sources TEXT,
                virality_score REAL,
                archetype TEXT,
                model_confidence REAL,
                approved_by_human INTEGER NOT NULL DEFAULT 0,
                executed INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                hook_pattern TEXT,
                title TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS post_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                post_id TEXT NOT NULL,
                upvotes INTEGER NOT NULL,
                comment_count INTEGER NOT NULL,
                upvote_delta INTEGER NOT NULL,
                comment_delta INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_action_ts ON action_events(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_action_post ON action_events(target_post_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_post_ts ON post_metrics(post_id, ts)")
        conn.commit()


def _hook_pattern(title: str) -> str:
    text = normalize_str(title).strip()
    if not text:
        return "untitled"
    words = [w for w in text.split() if w]
    if not words:
        return "untitled"
    first_words = " ".join(words[:3]).lower()
    has_question = "?" in text
    length_bucket = "short" if len(text) <= 55 else "mid" if len(text) <= 95 else "long"
    return f"{first_words}|{length_bucket}|q={int(has_question)}"


def record_action_event(
    path: Path,
    *,
    action_type: str,
    target_post_id: str,
    submolt: str,
    feed_sources: Sequence[str],
    virality_score: Optional[float],
    archetype: str,
    model_confidence: Optional[float],
    approved_by_human: bool,
    executed: bool,
    error: str = "",
    title: str = "",
) -> None:
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO action_events (
                ts, action_type, target_post_id, submolt, feed_sources, virality_score,
                archetype, model_confidence, approved_by_human, executed, error, hook_pattern, title
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now_iso(),
                normalize_str(action_type).strip().lower(),
                normalize_str(target_post_id).strip(),
                normalize_str(submolt).strip().lower(),
                ",".join([normalize_str(x).strip().lower() for x in feed_sources if normalize_str(x).strip()]),
                float(virality_score) if isinstance(virality_score, (int, float)) else None,
                normalize_str(archetype).strip().lower(),
                float(model_confidence) if isinstance(model_confidence, (int, float)) else None,
                1 if approved_by_human else 0,
                1 if executed else 0,
                normalize_str(error).strip()[:400],
                _hook_pattern(title),
                normalize_str(title).strip()[:220],
            ),
        )
        conn.commit()


def refresh_post_metrics(
    path: Path,
    *,
    client,
    tracked_post_ids: Iterable[str],
    logger,
    fetch_limit: int = 80,
) -> int:
    tracked = {normalize_str(pid).strip() for pid in tracked_post_ids if normalize_str(pid).strip()}
    if not tracked:
        return 0
    payload = client.get_posts(sort="new", limit=max(10, int(fetch_limit)))
    posts = extract_posts(payload)
    index: Dict[str, Dict[str, Any]] = {}
    for post in posts:
        pid = post_id(post)
        if pid:
            index[pid] = post

    updates = 0
    with _connect(path) as conn:
        for pid in tracked:
            post = index.get(pid)
            if not post:
                continue
            upvotes = int(post_score(post))
            comments = int(post_comment_count(post))
            row = conn.execute(
                "SELECT upvotes, comment_count FROM post_metrics WHERE post_id = ? ORDER BY id DESC LIMIT 1",
                (pid,),
            ).fetchone()
            prev_upvotes = int(row[0]) if row else 0
            prev_comments = int(row[1]) if row else 0
            up_delta = upvotes - prev_upvotes
            cm_delta = comments - prev_comments
            conn.execute(
                """
                INSERT INTO post_metrics (ts, post_id, upvotes, comment_count, upvote_delta, comment_delta)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (_utc_now_iso(), pid, upvotes, comments, up_delta, cm_delta),
            )
            updates += 1
        conn.commit()
    if updates:
        logger.info("Analytics metrics refresh updated_posts=%s tracked=%s", updates, len(tracked))
    return updates


def daily_summary(path: Path, *, top_skip_reasons: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    today = datetime.now(timezone.utc).date().isoformat()
    out: Dict[str, Any] = {
        "date": today,
        "best_archetype": "(none)",
        "best_hook_pattern": "(none)",
        "top_skip_reasons": top_skip_reasons or {},
    }
    with _connect(path) as conn:
        row = conn.execute(
            """
            SELECT archetype, COUNT(*) AS n, AVG(COALESCE(virality_score, 0))
            FROM action_events
            WHERE substr(ts, 1, 10) = ? AND executed = 1 AND action_type = 'post'
            GROUP BY archetype
            ORDER BY n DESC, AVG(COALESCE(virality_score, 0)) DESC
            LIMIT 1
            """,
            (today,),
        ).fetchone()
        if row:
            out["best_archetype"] = normalize_str(row[0]).strip() or "(none)"

        row = conn.execute(
            """
            SELECT hook_pattern, COUNT(*) AS n
            FROM action_events
            WHERE substr(ts, 1, 10) = ? AND executed = 1 AND action_type = 'post'
            GROUP BY hook_pattern
            ORDER BY n DESC
            LIMIT 1
            """,
            (today,),
        ).fetchone()
        if row:
            out["best_hook_pattern"] = normalize_str(row[0]).strip() or "(none)"
    return out


def aggregate_skip_reasons(cycle_history: List[Dict[str, Any]], window: int = 24) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for entry in cycle_history[-max(1, int(window)):]:
        if not isinstance(entry, dict):
            continue
        reasons = entry.get("skip_reasons")
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            key = normalize_str(reason).strip().lower()
            if not key:
                continue
            try:
                out[key] = out.get(key, 0) + int(count)
            except Exception:
                continue
    ranked = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
    return {k: v for k, v in ranked[:6]}
