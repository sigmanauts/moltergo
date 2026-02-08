from __future__ import annotations

import html
import json
import os
import sqlite3
import re
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from .drafting import normalize_str

DEFAULT_MAX_ENTRIES = 300
DEFAULT_REFRESH_SECONDS = 5

_JSON_CACHE: Dict[str, Tuple[float, int, Any]] = {}
_ANALYTICS_CACHE: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}

_KEYWORD_MD_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?:\d+\.\s*)?")
_KEYWORD_BOLD_RE = re.compile(r"[*_`]+")
_KEYWORD_WS_RE = re.compile(r"\s+")


def _truthy_env(name: str, default: bool = True) -> bool:
    raw = normalize_str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _safe_int(raw: Any, default: int, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except Exception:
        return default
    return max(minimum, value)


def _dashboard_output_path(journal_path: Path) -> Path:
    configured = normalize_str(os.getenv("MOLTBOOK_ACTION_DASHBOARD_PATH", "")).strip()
    if configured:
        return Path(configured)
    if journal_path.suffix:
        return journal_path.with_suffix(".html")
    return journal_path.parent / "action-journal.html"


def _dashboard_max_entries() -> int:
    return _safe_int(
        os.getenv("MOLTBOOK_ACTION_DASHBOARD_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES)),
        default=DEFAULT_MAX_ENTRIES,
        minimum=20,
    )


def _dashboard_refresh_seconds() -> int:
    return _safe_int(
        os.getenv("MOLTBOOK_ACTION_DASHBOARD_REFRESH_SECONDS", str(DEFAULT_REFRESH_SECONDS)),
        default=DEFAULT_REFRESH_SECONDS,
        minimum=1,
    )


def _clip(value: Any, limit: int) -> str:
    text = normalize_str(value).strip()
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _h(value: Any) -> str:
    return html.escape(normalize_str(value), quote=True)


def _maybe_load_json(path: Path) -> Optional[Any]:
    try:
        st = path.stat()
    except Exception:
        return None
    key = str(path)
    cached = _JSON_CACHE.get(key)
    if cached and cached[0] == float(st.st_mtime) and cached[1] == int(st.st_size):
        return cached[2]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    _JSON_CACHE[key] = (float(st.st_mtime), int(st.st_size), obj)
    return obj


def _tail_lines(path: Path, max_lines: int, *, chunk_size: int = 16384) -> List[str]:
    if max_lines <= 0:
        return []
    try:
        size = path.stat().st_size
    except Exception:
        return []
    if size <= 0:
        return []
    lines: List[bytes] = []
    buffer = b""
    try:
        with path.open("rb") as handle:
            pos = size
            while pos > 0 and len(lines) < max_lines:
                read_size = min(chunk_size, pos)
                pos -= read_size
                handle.seek(pos)
                buffer = handle.read(read_size) + buffer
                parts = buffer.split(b"\n")
                buffer = parts[0]
                for part in reversed(parts[1:]):
                    if part.strip():
                        lines.append(part)
                    if len(lines) >= max_lines:
                        break
            if buffer.strip() and len(lines) < max_lines:
                lines.append(buffer)
    except Exception:
        return []

    out: List[str] = []
    for raw in reversed(lines):
        try:
            out.append(raw.decode("utf-8"))
        except Exception:
            out.append(raw.decode("utf-8", errors="replace"))
    return out


def _load_recent_rows(path: Path, max_entries: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw_lines: Deque[str] = deque(_tail_lines(path, max_lines=max_entries), maxlen=max_entries)
    rows: List[Dict[str, Any]] = []
    for line in raw_lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _parse_ts(value: Any) -> Optional[datetime]:
    text = normalize_str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fmt_ts(value: Any) -> str:
    parsed = _parse_ts(value)
    if not parsed:
        return normalize_str(value).strip()
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_age_seconds(value: Any, now_utc: datetime) -> str:
    parsed = _parse_ts(value)
    if not parsed:
        return "-"
    seconds = max(0, int((now_utc - parsed).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h {minutes % 60}m"
    days = hours // 24
    return f"{days}d {hours % 24}h"


def _action_badge(action: str) -> str:
    token = normalize_str(action).strip().lower()
    css = "badge badge-neutral"
    label = token or "unknown"
    if "upvote" in token or "downvote" in token or "vote" in token:
        css = "badge badge-vote"
    elif "post" in token:
        css = "badge badge-post"
    elif "comment" in token:
        css = "badge badge-comment"
    return f'<span class="{css}">{_h(label)}</span>'


def _card(title: str, value: Any) -> str:
    return (
        '<div class="card">'
        f'<div class="card-title">{_h(title)}</div>'
        f'<div class="card-value">{_h(value)}</div>'
        "</div>"
    )


def _normalize_keyword_display(raw: Any) -> str:
    text = normalize_str(raw).strip()
    if not text:
        return ""
    # Strip common list prefixes like "3. " or "- ".
    text = _KEYWORD_MD_RE.sub("", text).strip()
    # Drop common markdown emphasis/backticks and trailing emphasis remnants.
    text = _KEYWORD_BOLD_RE.sub("", text).strip().strip(".,:;")
    text = _KEYWORD_WS_RE.sub(" ", text).strip()
    return text


def _dedupe_casefold(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        key = normalize_str(v).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _action_row_class(action: str) -> str:
    token = normalize_str(action).strip().lower()
    if "upvote" in token or "downvote" in token or "vote" in token:
        return "row-vote"
    if "post" in token and "comment" not in token:
        return "row-post"
    if "comment" in token:
        return "row-comment"
    return "row-neutral"


def _render_actions_table(rows: List[Dict[str, Any]], now_utc: datetime) -> str:
    if not rows:
        return '<div class="empty">No actions logged yet.</div>'
    items: List[str] = []
    items.append("<table>")
    items.append("<thead><tr>")
    for header in ["When", "Age", "Type", "Submolt", "Title", "Target URL", "Generated Content", "Reference Context"]:
        items.append(f"<th>{_h(header)}</th>")
    items.append("</tr></thead><tbody>")
    for row in reversed(rows):
        action_type = normalize_str(row.get("action_type")).strip().lower()
        url = normalize_str(row.get("url")).strip()
        if url:
            url_cell = f'<a href="{_h(url)}" target="_blank" rel="noreferrer">{_h(url)}</a>'
        else:
            url_cell = "-"
        full_generated = normalize_str(row.get("content")).strip()
        reference = row.get("reference") if isinstance(row.get("reference"), dict) else {}
        reference_title = normalize_str(reference.get("post_title")).strip()
        full_reference = normalize_str(reference.get("post_content") or reference.get("comment_content")).strip()

        def _details_block(full_text: str, summary_limit: int) -> str:
            if not full_text:
                return "-"
            if len(full_text) <= summary_limit:
                return f"<pre>{_h(full_text)}</pre>"
            summary = _clip(full_text, summary_limit)
            return f"<details><summary>{_h(summary)}</summary><pre>{_h(full_text)}</pre></details>"

        reference_bits: List[str] = []
        if reference_title:
            reference_bits.append(f"<strong>{_h(reference_title)}</strong>")
        if full_reference:
            reference_bits.append(_details_block(full_reference, summary_limit=260))
        reference_html = "<br/>".join(reference_bits) if reference_bits else "-"
        items.append(f'<tr class="{_action_row_class(action_type)}">')
        items.append(f"<td>{_h(_fmt_ts(row.get('ts')))}</td>")
        items.append(f"<td>{_h(_fmt_age_seconds(row.get('ts'), now_utc))}</td>")
        items.append(f"<td>{_action_badge(action_type)}</td>")
        items.append(f"<td>{_h(row.get('submolt') or '-')}</td>")
        items.append(f"<td>{_h(_clip(row.get('title'), 160))}</td>")
        items.append(f"<td>{url_cell}</td>")
        items.append(f"<td>{_details_block(full_generated, summary_limit=260)}</td>")
        items.append(f"<td>{reference_html}</td>")
        items.append("</tr>")
    items.append("</tbody></table>")
    return "".join(items)


def _epoch_to_iso(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)):
        return None
    try:
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    except Exception:
        return None
    return dt.isoformat()


def _load_analytics_summary(memory_dir: Path) -> Dict[str, Any]:
    path = memory_dir / "analytics.sqlite"
    try:
        st = path.stat()
    except Exception:
        return {"available": False}
    key = str(path)
    cached = _ANALYTICS_CACHE.get(key)
    if cached and cached[0] == float(st.st_mtime) and cached[1] == int(st.st_size):
        return cached[2]

    summary: Dict[str, Any] = {"available": True}
    con = None
    try:
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) as n FROM action_events")
        summary["action_events"] = int(cur.fetchone()["n"])
        cur.execute("SELECT COUNT(*) as n FROM post_metrics")
        summary["post_metrics"] = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT action_type, COUNT(*) n, SUM(executed) ex, SUM(approved_by_human) appr "
            "FROM action_events WHERE ts >= datetime('now', '-1 day') "
            "GROUP BY action_type ORDER BY n DESC"
        )
        summary["last24h_by_type"] = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT error, COUNT(*) n FROM action_events WHERE error != '' "
            "GROUP BY error ORDER BY n DESC LIMIT 8"
        )
        summary["top_errors"] = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT post_id, MAX(upvotes) as up, MAX(comment_count) as cc "
            "FROM post_metrics GROUP BY post_id ORDER BY up DESC LIMIT 10"
        )
        summary["top_post_metrics"] = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        summary["available"] = False
        summary["error"] = normalize_str(e)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass
    _ANALYTICS_CACHE[key] = (float(st.st_mtime), int(st.st_size), summary)
    return summary


def _render_list_pills(values: List[str]) -> str:
    cleaned = [normalize_str(v).strip() for v in values if normalize_str(v).strip()]
    if not cleaned:
        return '<div class="empty">-</div>'
    pills = "".join([f'<span class="pill">{_h(v)}</span>' for v in cleaned])
    return f'<div class="pills">{pills}</div>'


def _render_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return '<div class="empty">-</div>'
    out: List[str] = ["<table>", "<thead><tr>"]
    for h in headers:
        out.append(f"<th>{_h(h)}</th>")
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for cell in r:
            out.append(f"<td>{_h(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render_action_dashboard_html(
    *,
    rows: List[Dict[str, Any]],
    journal_path: Path,
    refresh_seconds: int,
    state: Optional[Dict[str, Any]] = None,
    keywords_store: Optional[Dict[str, Any]] = None,
    post_memory: Optional[Dict[str, Any]] = None,
    improvements: Optional[Dict[str, Any]] = None,
    improvement_backlog: Optional[Dict[str, Any]] = None,
    analytics_summary: Optional[Dict[str, Any]] = None,
) -> str:
    now_utc = datetime.now(timezone.utc)
    counts = Counter(normalize_str(r.get("action_type")).strip().lower() for r in rows)
    post_rows = [r for r in rows if normalize_str(r.get("action_type")).strip().lower() == "post"]
    comment_rows = [r for r in rows if normalize_str(r.get("action_type")).strip().lower() == "comment"]
    last_row = rows[-1] if rows else {}

    one_hour_ago = now_utc - timedelta(hours=1)
    day_ago = now_utc - timedelta(days=1)
    rows_1h = [r for r in rows if (_parse_ts(r.get("ts")) or datetime.min.replace(tzinfo=timezone.utc)) >= one_hour_ago]
    rows_24h = [r for r in rows if (_parse_ts(r.get("ts")) or datetime.min.replace(tzinfo=timezone.utc)) >= day_ago]
    counts_1h = Counter(normalize_str(r.get("action_type")).strip().lower() for r in rows_1h)
    counts_24h = Counter(normalize_str(r.get("action_type")).strip().lower() for r in rows_24h)

    last_post_row = post_rows[-1] if post_rows else {}
    last_comment_row = comment_rows[-1] if comment_rows else {}

    overview_cards = "".join(
        [
            _card("Actions (window)", len(rows)),
            _card("Posts", counts.get("post", 0)),
            _card("Comments", counts.get("comment", 0)),
            _card(
                "Votes",
                counts.get("upvote_post", 0)
                + counts.get("downvote_post", 0)
                + counts.get("upvote_comment", 0)
                + counts.get("downvote_comment", 0),
            ),
            _card("Last Action", _fmt_ts(last_row.get("ts")) if last_row else "-"),
            _card("Actions (1h)", sum(counts_1h.values())),
            _card("Posts (24h)", counts_24h.get("post", 0)),
            _card("Comments (24h)", counts_24h.get("comment", 0)),
        ]
    )

    if isinstance(state, dict) and state:
        last_action_ts = _epoch_to_iso(state.get("last_action_ts"))
        last_post_ts = _epoch_to_iso(state.get("last_post_action_ts"))
        last_comment_ts = _epoch_to_iso(state.get("last_comment_action_ts"))
        post_age_seconds = None
        if last_post_ts:
            parsed = _parse_ts(last_post_ts)
            if parsed:
                post_age_seconds = max(0, int((now_utc - parsed).total_seconds()))
        comment_age_seconds = None
        if last_comment_ts:
            parsed = _parse_ts(last_comment_ts)
            if parsed:
                comment_age_seconds = max(0, int((now_utc - parsed).total_seconds()))

        post_cooldown_seconds = 30 * 60
        post_cooldown_remaining = "-"
        post_slot = "-"
        if post_age_seconds is not None:
            remaining = max(0, post_cooldown_seconds - post_age_seconds)
            post_slot = "open" if remaining == 0 else "cooldown"
            post_cooldown_remaining = "0s" if remaining == 0 else f"{remaining//60}m {remaining%60}s"

        # Hourly comment lane: infer from timestamps list, if present.
        hourly_comment_count = 0
        cts = state.get("comment_action_timestamps")
        if isinstance(cts, list):
            for ts in cts[-2000:]:
                if not isinstance(ts, (int, float)):
                    continue
                try:
                    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                except Exception:
                    continue
                if dt >= one_hour_ago:
                    hourly_comment_count += 1
        comment_hourly_limit = 50
        comment_slot = "open" if hourly_comment_count < comment_hourly_limit else "full"

        runner_cards = "".join(
            [
                _card("Last action age", _fmt_age_seconds(last_action_ts, now_utc)),
                _card("Last post age", _fmt_age_seconds(_epoch_to_iso(state.get("last_post_action_ts")), now_utc)),
                _card(
                    "Last comment age",
                    _fmt_age_seconds(_epoch_to_iso(state.get("last_comment_action_ts")), now_utc),
                ),
                _card("Post slot", post_slot),
                _card("Post cooldown remaining", post_cooldown_remaining),
                _card("Comments (1h)", f"{hourly_comment_count}/{comment_hourly_limit} ({comment_slot})"),
                _card(
                    "Pending actions",
                    len(state.get("pending_actions", [])) if isinstance(state.get("pending_actions"), list) else 0,
                ),
                _card("Daily posts", state.get("daily_post_count", "-")),
                _card("Daily comments", state.get("daily_comment_count", "-")),
                _card(
                    "Seen posts",
                    len(state.get("seen_post_ids", [])) if isinstance(state.get("seen_post_ids"), list) else 0,
                ),
                _card(
                    "Replied posts",
                    len(state.get("replied_post_ids", [])) if isinstance(state.get("replied_post_ids"), list) else 0,
                ),
                _card(
                    "Followed agents",
                    len(state.get("followed_agents", [])) if isinstance(state.get("followed_agents"), list) else 0,
                ),
                _card(
                    "Approved submolts",
                    len(state.get("approved_submolts", [])) if isinstance(state.get("approved_submolts"), list) else 0,
                ),
                _card("Last post title", _clip(last_post_row.get("title"), 54) or "-"),
                _card("Last comment title", _clip(last_comment_row.get("title"), 54) or "-"),
            ]
        )
    else:
        runner_cards = _card("Runner state", "unavailable")

    learned_total = 0
    pending_total = 0
    learned_tail: List[str] = []
    pending_tail: List[str] = []
    if isinstance(keywords_store, dict) and keywords_store:
        learned = keywords_store.get("learned_keywords")
        pending = keywords_store.get("pending_suggestions")
        if isinstance(learned, list):
            learned_total = len(learned)
            learned_tail = _dedupe_casefold(
                [
                    _normalize_keyword_display(x)
                    for x in learned[-32:]
                    if _normalize_keyword_display(x)
                ]
            )[:12]
        if isinstance(pending, list):
            pending_total = len(pending)
            pending_tail = _dedupe_casefold(
                [
                    _normalize_keyword_display(x)
                    for x in pending[-96:]
                    if _normalize_keyword_display(x)
                ]
            )[:12]

    learning_cards = "".join(
        [
            _card("Learned keywords", learned_total),
            _card("Pending keywords", pending_total),
        ]
    )

    market_terms: List[str] = []
    winning_terms: List[str] = []
    losing_terms: List[str] = []
    visibility_metrics: Dict[str, Any] = {}
    top_posts: List[Dict[str, Any]] = []
    declines_by_reason: Dict[str, int] = {}
    if isinstance(post_memory, dict) and post_memory:
        snap = post_memory.get("last_market_snapshot")
        if isinstance(snap, dict) and isinstance(snap.get("top_terms"), list):
            market_terms = [
                normalize_str(x).strip() for x in snap.get("top_terms", []) if normalize_str(x).strip()
            ][:12]
        last = post_memory.get("last_snapshot")
        if isinstance(last, dict):
            if isinstance(last.get("winning_terms"), list):
                winning_terms = [
                    normalize_str(x).strip()
                    for x in last.get("winning_terms", [])
                    if normalize_str(x).strip()
                ][:12]
            if isinstance(last.get("losing_terms"), list):
                losing_terms = [
                    normalize_str(x).strip()
                    for x in last.get("losing_terms", [])
                    if normalize_str(x).strip()
                ][:12]
            if isinstance(last.get("visibility_metrics"), dict):
                visibility_metrics = last.get("visibility_metrics") or {}
        pp = post_memory.get("proactive_posts")
        if isinstance(pp, list):
            for item in pp:
                if not isinstance(item, dict):
                    continue
                up = item.get("upvotes")
                cc = item.get("comment_count")
                score = (int(up) if isinstance(up, int) else 0) * 2 + (int(cc) if isinstance(cc, int) else 0)
                top_posts.append(
                    {
                        "title": normalize_str(item.get("title")).strip(),
                        "submolt": normalize_str(item.get("submolt")).strip(),
                        "upvotes": up if isinstance(up, int) else 0,
                        "comments": cc if isinstance(cc, int) else 0,
                        "archetype": normalize_str(item.get("content_archetype")).strip(),
                        "score": score,
                    }
                )
        di = post_memory.get("declined_ideas")
        if isinstance(di, list):
            for item in di[-300:]:
                if not isinstance(item, dict):
                    continue
                reason = normalize_str(item.get("reason")).strip() or "unknown"
                declines_by_reason[reason] = declines_by_reason.get(reason, 0) + 1
    top_posts.sort(key=lambda x: (x.get("score", 0), x.get("upvotes", 0), x.get("comments", 0)), reverse=True)

    imp_latest_summary = ""
    if isinstance(improvements, dict):
        entries = improvements.get("entries")
        if isinstance(entries, list) and entries:
            last_entry = entries[-1] if isinstance(entries[-1], dict) else {}
            imp_latest_summary = normalize_str(last_entry.get("summary")).strip()

    backlog_rows: List[List[Any]] = []
    if isinstance(improvement_backlog, dict):
        items = improvement_backlog.get("items")
        if isinstance(items, list):
            sorted_items = [x for x in items if isinstance(x, dict)]
            sorted_items.sort(key=lambda x: float(x.get("priority_score", 0.0) or 0.0), reverse=True)
            for item in sorted_items[:8]:
                backlog_rows.append(
                    [
                        normalize_str(item.get("kind")).strip() or "-",
                        normalize_str(item.get("headline")).strip() or "-",
                        str(round(float(item.get("priority_score", 0.0) or 0.0), 2)),
                        str(int(item.get("seen_count", 0) or 0)),
                        _clip(item.get("change"), 120),
                    ]
                )

    if isinstance(analytics_summary, dict) and analytics_summary.get("available"):
        funnel = analytics_summary.get("last24h_by_type") or []
        funnel_rows: List[List[Any]] = []
        for row in funnel:
            if not isinstance(row, dict):
                continue
            funnel_rows.append(
                [
                    normalize_str(row.get("action_type")).strip(),
                    str(int(row.get("n", 0) or 0)),
                    str(int(row.get("appr", 0) or 0)),
                    str(int(row.get("ex", 0) or 0)),
                ]
            )
        errors = analytics_summary.get("top_errors") or []
        error_rows: List[List[Any]] = []
        for row in errors:
            if not isinstance(row, dict):
                continue
            error_rows.append([_clip(row.get("error"), 90), str(int(row.get("n", 0) or 0))])
        analytics_block = (
            "<h2>Analytics (last 24h)</h2>"
            + _render_table(["action", "events", "approved", "executed"], funnel_rows)
            + "<h3>Top Errors</h3>"
            + _render_table(["error", "count"], error_rows)
        )
    elif isinstance(analytics_summary, dict) and analytics_summary.get("error"):
        analytics_block = (
            "<h2>Analytics</h2>" + f"<div class='empty'>analytics unavailable: {_h(analytics_summary.get('error'))}</div>"
        )
    else:
        analytics_block = "<h2>Analytics</h2><div class='empty'>analytics.sqlite not found</div>"

    badbot_rows: List[List[Any]] = []
    if isinstance(state, dict) and isinstance(state.get("bad_bot_counts"), dict):
        items = [(k, v) for k, v in state["bad_bot_counts"].items() if isinstance(v, int)]
        items.sort(key=lambda kv: kv[1], reverse=True)
        for name, strikes in items[:10]:
            badbot_rows.append([name, str(strikes)])

    # Top threads and top authors from recent action window.
    thread_stats: Dict[str, Dict[str, Any]] = {}
    author_counts: Counter[str] = Counter()
    for row in rows:
        pid = normalize_str(row.get("target_post_id")).strip()
        if pid:
            st = thread_stats.setdefault(
                pid,
                {
                    "post_id": pid,
                    "submolt": normalize_str(row.get("submolt")).strip() or "-",
                    "title": "",
                    "url": "",
                    "actions": 0,
                    "comments": 0,
                    "posts": 0,
                    "last_ts": row.get("ts"),
                },
            )
            st["actions"] += 1
            at = normalize_str(row.get("action_type")).strip().lower()
            if at == "comment":
                st["comments"] += 1
            if at == "post":
                st["posts"] += 1
            if row.get("title"):
                st["title"] = normalize_str(row.get("title")).strip()
            if row.get("url"):
                st["url"] = normalize_str(row.get("url")).strip()
            # last_ts: keep max
            prev = _parse_ts(st.get("last_ts"))
            cur = _parse_ts(row.get("ts"))
            if cur and (not prev or cur > prev):
                st["last_ts"] = row.get("ts")
        ref = row.get("reference") if isinstance(row.get("reference"), dict) else {}
        for key in ("comment_author", "post_author"):
            name = normalize_str(ref.get(key)).strip()
            if name:
                author_counts[name] += 1

    top_threads = sorted(
        thread_stats.values(),
        key=lambda x: (int(x.get("comments", 0)), int(x.get("actions", 0))),
        reverse=True,
    )[:12]
    top_threads_rows: List[List[Any]] = []
    for t in top_threads:
        url = t.get("url") or ""
        url_cell = url if not url else url
        top_threads_rows.append(
            [
                str(int(t.get("comments", 0))),
                str(int(t.get("actions", 0))),
                _fmt_age_seconds(t.get("last_ts"), now_utc),
                normalize_str(t.get("submolt")).strip() or "-",
                _clip(t.get("title"), 80),
                url_cell,
            ]
        )

    top_authors_rows: List[List[Any]] = []
    for name, n in author_counts.most_common(12):
        top_authors_rows.append([name, str(int(n))])

    # Quality flags: highlight formatting anomalies in our posted content.
    flagged: List[Dict[str, str]] = []
    for row in reversed(rows[-160:]):
        content = normalize_str(row.get("content")).strip()
        title = normalize_str(row.get("title")).strip()
        flags: List[str] = []
        if "[truncated]" in title.lower():
            flags.append("title_has_truncated")
        if "```yaml" in content.lower():
            flags.append("yaml_fence")
        if "comment (≈" in content.lower() or "reply (" in content.lower():
            flags.append("meta_in_body")
        if "should_respond=" in content or "response_mode=" in content:
            flags.append("debug_in_body")
        if flags:
            flagged.append(
                {
                    "ts": _fmt_ts(row.get("ts")),
                    "age": _fmt_age_seconds(row.get("ts"), now_utc),
                    "type": normalize_str(row.get("action_type")).strip(),
                    "title": _clip(title, 90),
                    "flags": ", ".join(flags),
                }
            )
        if len(flagged) >= 12:
            break
    flagged_rows = [[f["ts"], f["age"], f["type"], f["flags"], f["title"]] for f in flagged]

    posts_html = _render_actions_table(post_rows[-20:], now_utc)
    comments_html = _render_actions_table(comment_rows[-50:], now_utc)
    recent_html = _render_actions_table(rows[-120:], now_utc)

    parts: List[str] = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'/>")
    parts.append(f"<meta http-equiv='refresh' content='{int(refresh_seconds)}'/>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    parts.append("<title>Moltergo Action Dashboard</title>")
    parts.append("<style>")
    parts.append("html{scroll-behavior:smooth}")
    parts.append(
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:radial-gradient(1200px 700px at 20% -10%,rgba(126,231,135,.08),transparent 60%),#0d1117;color:#e6edf3;margin:0;padding:0 14px 28px;}"
    )
    parts.append("h1{margin:18px 0 4px;font-size:22px;color:#7ee787}")
    parts.append(".sub{color:#9da7b3;font-size:13px;margin-bottom:14px}")
    parts.append(".container{max-width:1280px;margin:0 auto}")
    parts.append(
        "nav{position:sticky;top:0;z-index:5;background:rgba(13,17,23,.92);backdrop-filter:saturate(160%) blur(10px);border-bottom:1px solid #30363d;margin:0 -14px;padding:10px 14px}"
    )
    parts.append(".navlinks{display:flex;flex-wrap:wrap;gap:10px;align-items:center}")
    parts.append(
        ".navlinks a{display:inline-block;padding:5px 9px;border:1px solid #30363d;border-radius:999px;background:#0f141b;color:#c9d1d9;font-size:12px}"
    )
    parts.append(".navlinks a:hover{border-color:#58a6ff}")
    parts.append(".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 18px;}")
    parts.append(".card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;}")
    parts.append(".card-title{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}")
    parts.append(".card-value{font-size:18px;font-weight:700;color:#e6edf3;padding-top:3px}")
    parts.append("h2{margin:20px 0 8px;font-size:16px;color:#79c0ff}")
    parts.append("h3{margin:14px 0 6px;font-size:14px;color:#c9d1d9}")
    parts.append(".pills{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 12px}")
    parts.append(
        ".pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #30363d;background:#0f141b;color:#c9d1d9;font-size:12px}"
    )
    parts.append(".split{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}")
    parts.append(".split-col{background:#0f141b;border:1px solid #30363d;border-radius:8px;padding:10px}")
    parts.append(".kv-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin:10px 0 18px}")
    parts.append(".kv{background:#0f141b;border:1px solid #30363d;border-radius:8px;padding:8px}")
    parts.append(".kv-k{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}")
    parts.append(".kv-v{font-size:14px;font-weight:700;color:#e6edf3;padding-top:2px}")
    parts.append("details>summary{cursor:pointer;color:#8b949e;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}")
    parts.append("table{width:100%;border-collapse:collapse;background:#11161d;border:1px solid #30363d;border-radius:8px;overflow:hidden;}")
    parts.append("th,td{border-bottom:1px solid #222a33;padding:7px 8px;vertical-align:top;font-size:12px;text-align:left;}")
    parts.append("th{background:#161b22;color:#8b949e;position:sticky;top:0;z-index:1}")
    parts.append("tr:hover td{background:#0f141b}")
    parts.append(".row-post td{border-left:3px solid #a371f7}")
    parts.append(".row-comment td{border-left:3px solid #1f6feb}")
    parts.append(".row-vote td{border-left:3px solid #9a7d2f}")
    parts.append(".row-neutral td{border-left:3px solid #30363d}")
    parts.append("a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}")
    parts.append("pre{margin:0;white-space:pre-wrap;word-break:break-word;line-height:1.35;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}")
    parts.append(
        ".badge{display:inline-block;padding:2px 7px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em}"
    )
    parts.append(".badge-post{background:#2a1a33;color:#f5b3ff;border:1px solid #6b3fa0}")
    parts.append(".badge-comment{background:#102438;color:#93c5fd;border:1px solid #1f6feb}")
    parts.append(".badge-vote{background:#2b2a13;color:#f2cc60;border:1px solid #9a7d2f}")
    parts.append(".badge-neutral{background:#2d333b;color:#c9d1d9;border:1px solid #484f58}")
    parts.append(".empty{padding:12px;border:1px dashed #30363d;border-radius:8px;color:#8b949e;background:#11161d}")
    parts.append("</style></head><body>")
    parts.append("<div class='container'>")
    parts.append("<h1>Moltergo Live Action Dashboard</h1>")
    parts.append(
        f"<div class='sub'>Source: {_h(journal_path)} | Generated: {_h(_fmt_ts(now_utc.isoformat()))} | Auto-refresh: {int(refresh_seconds)}s</div>"
    )
    parts.append("<nav><div class='navlinks'>")
    for anchor, label in [
        ("overview", "Overview"),
        ("runner", "Runner"),
        ("learning", "Learning"),
        ("visibility", "Visibility"),
        ("analytics", "Analytics"),
        ("quality", "Quality Flags"),
        ("threads", "Top Threads"),
        ("posts", "Posts"),
        ("comments", "Comments"),
        ("actions", "All Actions"),
    ]:
        parts.append(f"<a href='#{_h(anchor)}'>{_h(label)}</a>")
    parts.append("</div></nav>")

    parts.append("<h2 id='overview'>Overview</h2>")
    parts.append(f"<div class='cards'>{overview_cards}</div>")
    parts.append("<h2 id='runner'>Runner State</h2>")
    parts.append(f"<div class='cards'>{runner_cards}</div>")
    parts.append("<h2 id='learning'>Learning</h2>")
    parts.append(f"<div class='cards'>{learning_cards}</div>")
    parts.append("<div class='split'>")
    parts.append("<div class='split-col'>")
    parts.append("<h3>Latest Learned Keywords</h3>")
    parts.append(_render_list_pills(learned_tail))
    parts.append("<h3>Pending Keyword Suggestions</h3>")
    parts.append(_render_list_pills(pending_tail))
    parts.append("</div>")
    parts.append("<div class='split-col'>")
    parts.append("<h3>Market Snapshot Terms</h3>")
    parts.append(_render_list_pills(market_terms))
    parts.append("<h3>Our Winning Terms</h3>")
    parts.append(_render_list_pills(winning_terms))
    parts.append("<h3>Our Losing Terms</h3>")
    parts.append(_render_list_pills(losing_terms))
    parts.append("</div>")
    parts.append("</div>")
    parts.append("<h2 id='visibility'>Visibility Metrics</h2>")
    parts.append("<div class='kv-grid'>")
    for key in [
        "target_upvotes",
        "recent_avg_upvotes",
        "recent_avg_comments",
        "recent_target_hit_rate",
        "recent_avg_visibility_score",
        "baseline_avg_visibility_score",
        "visibility_delta_pct",
        "winning_question_title_rate",
    ]:
        parts.append(
            f"<div class='kv'><div class='kv-k'>{_h(key)}</div><div class='kv-v'>{_h(visibility_metrics.get(key, '-'))}</div></div>"
        )
    parts.append("</div>")
    parts.append("<h2>Top Proactive Posts (by 2x upvotes + comments)</h2>")
    parts.append(
        _render_table(
            ["upvotes", "comments", "archetype", "submolt", "title"],
            [
                [
                    str(p.get("upvotes", 0)),
                    str(p.get("comments", 0)),
                    p.get("archetype") or "-",
                    p.get("submolt") or "-",
                    p.get("title") or "-",
                ]
                for p in top_posts[:8]
            ],
        )
    )
    parts.append("<h2>Declined Draft Reasons (recent)</h2>")
    parts.append(
        _render_table(
            ["reason", "count"],
            [[k, str(v)] for k, v in sorted(declines_by_reason.items(), key=lambda kv: kv[1], reverse=True)[:10]],
        )
    )
    parts.append("<h2>Self-Improve</h2>")
    parts.append(f"<div class='sub'>{_h(_clip(imp_latest_summary, 260) or 'No suggestions loaded.')}</div>")
    parts.append("<h3>Top Backlog Items</h3>")
    parts.append(_render_table(["kind", "headline", "priority", "seen", "change"], backlog_rows))
    parts.append("<div id='analytics'>")
    parts.append(analytics_block)
    parts.append("</div>")
    parts.append("<h2 id='quality'>Quality Flags (recent window)</h2>")
    parts.append(
        _render_table(
            ["when", "age", "type", "flags", "title"],
            flagged_rows,
        )
    )
    parts.append("<h2 id='threads'>Top Threads (recent window)</h2>")
    parts.append(
        _render_table(
            ["comments", "actions", "last", "submolt", "title", "url"],
            top_threads_rows,
        )
    )
    parts.append("<h3>Top Interacted Authors (from reference context)</h3>")
    parts.append(_render_table(["author", "actions"], top_authors_rows))

    parts.append("<h2>Noisy Authors (heuristic)</h2>")
    parts.append(_render_table(["author", "strikes"], badbot_rows))
    parts.append("<h2 id='posts'>Current Posts (latest 20)</h2>")
    parts.append(posts_html)
    parts.append("<h2 id='comments'>Recent Comments (latest 50)</h2>")
    parts.append(comments_html)
    parts.append("<h2 id='actions'>All Recent Actions (latest 120)</h2>")
    parts.append(recent_html)
    parts.append("</div></body></html>")
    return "".join(parts)


def refresh_action_dashboard(
    journal_path: Path,
    *,
    html_path: Optional[Path] = None,
    max_entries: Optional[int] = None,
) -> Optional[Path]:
    if not _truthy_env("MOLTBOOK_ACTION_DASHBOARD_ENABLED", default=True):
        return None
    source = Path(journal_path)
    target = Path(html_path) if html_path else _dashboard_output_path(source)
    limit = max_entries if isinstance(max_entries, int) and max_entries > 0 else _dashboard_max_entries()
    refresh_seconds = _dashboard_refresh_seconds()
    rows = _load_recent_rows(source, max_entries=limit)
    target.parent.mkdir(parents=True, exist_ok=True)
    memory_dir = source.parent
    state_obj = _maybe_load_json(memory_dir / "autonomy-state.json")
    keywords_obj = _maybe_load_json(memory_dir / "learned-keywords.json")
    post_memory_obj = _maybe_load_json(memory_dir / "post-engine-memory.json")
    improvements_obj = _maybe_load_json(memory_dir / "improvement-suggestions.json")
    backlog_obj = _maybe_load_json(memory_dir / "improvement-backlog.json")
    analytics_summary = _load_analytics_summary(memory_dir=memory_dir)
    html_text = render_action_dashboard_html(
        rows=rows,
        journal_path=source,
        refresh_seconds=refresh_seconds,
        state=state_obj if isinstance(state_obj, dict) else None,
        keywords_store=keywords_obj if isinstance(keywords_obj, dict) else None,
        post_memory=post_memory_obj if isinstance(post_memory_obj, dict) else None,
        improvements=improvements_obj if isinstance(improvements_obj, dict) else None,
        improvement_backlog=backlog_obj if isinstance(backlog_obj, dict) else None,
        analytics_summary=analytics_summary if isinstance(analytics_summary, dict) else None,
    )
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(html_text, encoding="utf-8")
    tmp.replace(target)
    return target
