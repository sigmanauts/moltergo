from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from ..moltbook_client import MoltbookClient
from ..virality import normalize_str, parse_timestamp
from .runtime_helpers import extract_submolts, normalize_submolt


def parse_submolt_meta(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in extract_submolts(payload):
        name = normalize_submolt(item.get("name") or item.get("slug"), default="")
        if not name:
            continue
        subscribers = (
            item.get("subscriber_count")
            or item.get("subscribers")
            or item.get("member_count")
            or item.get("members")
            or 0
        )
        try:
            subscriber_count = int(subscribers)
        except Exception:
            subscriber_count = 0
        last_activity = (
            item.get("last_activity_at")
            or item.get("lastActivityAt")
            or item.get("updated_at")
            or item.get("created_at")
        )
        out[name] = {
            "name": name,
            "display_name": normalize_str(item.get("display_name") or item.get("displayName")).strip(),
            "description": normalize_str(item.get("description")).strip(),
            "subscriber_count": subscriber_count,
            "last_activity_at": normalize_str(last_activity).strip(),
        }
    return out


def get_cached_submolt_meta(
    client: MoltbookClient,
    ttl_seconds: int,
    cache: Dict[str, Any],
    logger,
) -> Dict[str, Dict[str, Any]]:
    now_ts = time.time()
    cached_ts = cache.get("fetched_ts")
    cached_map = cache.get("items")
    if isinstance(cached_ts, (int, float)) and isinstance(cached_map, dict):
        if (now_ts - float(cached_ts)) <= max(30, int(ttl_seconds)):
            return cached_map

    payload: Dict[str, Any]
    try:
        payload = client.list_submolts_public()
    except Exception as e:
        logger.warning("Submolt public fetch failed error=%s; trying authenticated list.", e)
        payload = client.list_submolts()

    parsed = parse_submolt_meta(payload)
    cache["fetched_ts"] = now_ts
    cache["items"] = parsed
    logger.info("Submolt metadata refreshed count=%s ttl=%ss", len(parsed), max(30, int(ttl_seconds)))
    return parsed


def is_valid_submolt_name(name: str, submolt_meta: Dict[str, Dict[str, Any]]) -> bool:
    normalized = normalize_submolt(name, default="")
    if not normalized:
        return False
    if normalized == "general":
        return True
    return normalized in submolt_meta


def _keyword_overlap_score(text: str, description: str) -> float:
    text_words = {w for w in normalize_str(text).lower().split() if len(w) >= 4}
    desc_words = {w for w in normalize_str(description).lower().split() if len(w) >= 4}
    if not text_words or not desc_words:
        return 0.0
    overlap = len(text_words & desc_words)
    return min(1.0, overlap / 8.0)


def choose_best_submolt_for_new_post(
    *,
    title: str,
    content: str,
    archetype: str,
    target_submolts: List[str],
    submolt_meta: Dict[str, Dict[str, Any]],
) -> str:
    allowed = {normalize_submolt(item, default="") for item in target_submolts if normalize_submolt(item, default="")}
    allowed.add("general")
    # Build logs and walkthroughs prefer m/builds when present.
    archetype_norm = normalize_str(archetype).strip().lower().replace("-", "_")
    if archetype_norm in {"build_log", "implementation_walkthrough"} and "builds" in submolt_meta:
        allowed.add("builds")

    text_blob = f"{normalize_str(title)}\n{normalize_str(content)}"
    now_ts = time.time()

    best_name = "general"
    best_score = -1e9
    for candidate in sorted(allowed):
        meta = submolt_meta.get(candidate, {})
        subscribers = int(meta.get("subscriber_count", 0) or 0)
        size_score = math.log1p(max(0, subscribers)) / math.log(50000.0)
        last_activity_ts = parse_timestamp(meta.get("last_activity_at"))
        if last_activity_ts is None:
            freshness_score = 0.4
        else:
            age_hours = max(0.0, (now_ts - last_activity_ts) / 3600.0)
            freshness_score = 1.0 / (1.0 + (age_hours / 24.0))
        desc_score = _keyword_overlap_score(text_blob, normalize_str(meta.get("description")))
        direct_match = 1.0 if f"m/{candidate}" in text_blob.lower() else 0.0
        if candidate == "general":
            direct_match += 0.05

        score = (size_score * 0.35) + (freshness_score * 0.25) + (desc_score * 0.3) + (direct_match * 0.1)
        if archetype_norm in {"build_log", "implementation_walkthrough"} and candidate == "builds":
            score += 0.35
        if score > best_score:
            best_score = score
            best_name = candidate

    return best_name or "general"


def get_submolt_scorecard(submolt_meta: Dict[str, Dict[str, Any]], names: List[str]) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    for name in names:
        key = normalize_submolt(name, default="")
        meta = submolt_meta.get(key, {})
        out.append(
            (
                key or "general",
                int(meta.get("subscriber_count", 0) or 0),
                normalize_str(meta.get("last_activity_at")).strip(),
            )
        )
    return out
