from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .action_dashboard import refresh_action_dashboard
from .drafting import normalize_str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip_text(value: Any, limit: int) -> str:
    text = normalize_str(value).strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _sanitize_reference(reference: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(reference, dict):
        return None
    out: Dict[str, Any] = {}
    for raw_key, raw_value in reference.items():
        key = normalize_str(raw_key).strip()
        if not key:
            continue
        lower_key = key.lower()
        if raw_value is None:
            continue
        if isinstance(raw_value, bool):
            out[key] = raw_value
            continue
        if isinstance(raw_value, (int, float)):
            out[key] = raw_value
            continue
        if isinstance(raw_value, dict):
            nested = _sanitize_reference(raw_value)
            if nested:
                out[key] = nested
            continue
        if isinstance(raw_value, list):
            items = []
            for item in raw_value[:10]:
                if isinstance(item, dict):
                    nested_item = _sanitize_reference(item)
                    if nested_item:
                        items.append(nested_item)
                elif isinstance(item, (int, float, bool)):
                    items.append(item)
                else:
                    limit = 5000 if "content" in lower_key else 400
                    clipped = _clip_text(item, limit)
                    if clipped:
                        items.append(clipped)
            if items:
                out[key] = items
            continue
        limit = 5000 if "content" in lower_key else 400
        clipped = _clip_text(raw_value, limit)
        if clipped:
            out[key] = clipped
    return out or None


def append_action_journal(
    path: Path,
    *,
    action_type: str,
    target_post_id: str,
    submolt: str,
    title: str,
    content: str,
    parent_comment_id: Optional[str] = None,
    reference_post_id: Optional[str] = None,
    url: Optional[str] = None,
    reference: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row: Dict[str, Any] = {
        "ts": _utc_now_iso(),
        "action_type": normalize_str(action_type).strip().lower(),
        "target_post_id": normalize_str(target_post_id).strip(),
        "submolt": normalize_str(submolt).strip().lower(),
        "title": normalize_str(title).strip(),
        "content": normalize_str(content).strip(),
    }
    if parent_comment_id:
        row["parent_comment_id"] = normalize_str(parent_comment_id).strip()
    if reference_post_id:
        row["reference_post_id"] = normalize_str(reference_post_id).strip()
    if url:
        row["url"] = normalize_str(url).strip()
    safe_reference = _sanitize_reference(reference)
    if safe_reference:
        row["reference"] = safe_reference
    if isinstance(meta, dict) and meta:
        row["meta"] = meta
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    try:
        refresh_action_dashboard(path)
    except Exception:
        # Dashboard rendering must never block posting/commenting actions.
        pass
