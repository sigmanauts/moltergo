from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ..moltbook_client import MoltbookAuthError, MoltbookClient
from ..virality import is_early_comment_candidate
from .config import Config
from .drafting import normalize_str
from .runtime_helpers import extract_posts, extract_submolts, post_id


def _mark_post_source(post: Dict[str, Any], source: str) -> Dict[str, Any]:
    item = dict(post)
    raw_sources = item.get("__feed_sources")
    sources: List[str] = []
    if isinstance(raw_sources, list):
        for value in raw_sources:
            token = normalize_str(value).strip().lower()
            if token and token not in sources:
                sources.append(token)
    source_token = normalize_str(source).strip().lower()
    if source_token and source_token not in sources:
        sources.append(source_token)
    item["__feed_sources"] = sources
    return item


def merge_unique_posts(posts: List[Dict[str, Any]], post_id_fn) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    index_by_id: Dict[str, int] = {}
    for post in posts:
        pid = post_id_fn(post)
        if not pid:
            continue
        if pid in index_by_id:
            existing = out[index_by_id[pid]]
            existing_sources = existing.get("__feed_sources", [])
            if not isinstance(existing_sources, list):
                existing_sources = []
            incoming_sources = post.get("__feed_sources", [])
            if not isinstance(incoming_sources, list):
                incoming_sources = []
            merged_sources = []
            for value in existing_sources + incoming_sources:
                token = normalize_str(value).strip().lower()
                if token and token not in merged_sources:
                    merged_sources.append(token)
            existing["__feed_sources"] = merged_sources
            continue
        index_by_id[pid] = len(out)
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


def discover_posts(
    client: MoltbookClient,
    cfg: Config,
    logger,
    keywords: List[str],
    iteration: int,
    search_state: Dict[str, Any],
    post_id_fn=post_id,
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
                    search_posts = [_mark_post_source(post, "search") for post in search_posts]
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
    feed_posts = [_mark_post_source(post, "feed") for post in extract_posts(feed)]

    global_posts: List[Dict[str, Any]] = []
    per_sort_posts: Dict[str, List[Dict[str, Any]]] = {}
    feed_sorts = getattr(cfg, "feed_sources", None)
    if not isinstance(feed_sorts, list) or not feed_sorts:
        feed_sorts = [getattr(cfg, "posts_sort", "new")]
    try:
        sort_payloads = client.get_posts_by_sorts(sorts=feed_sorts, limit=cfg.posts_limit)
    except Exception:
        sort_payloads = {cfg.posts_sort: client.get_posts(sort=cfg.posts_sort, limit=cfg.posts_limit)}
    for sort_name, payload in sort_payloads.items():
        posts = [
            _mark_post_source(_mark_post_source(post, f"posts:{sort_name}"), sort_name)
            for post in extract_posts(payload)
        ]
        per_sort_posts[sort_name] = posts
        global_posts.extend(posts)

    if cfg.target_submolts:
        for submolt in cfg.target_submolts:
            try:
                payload = client.get_submolt_feed(name=submolt, sort=cfg.posts_sort, limit=cfg.posts_limit)
                posts = [
                    _mark_post_source(_mark_post_source(post, f"submolt:{submolt}"), f"submolt_sort:{cfg.posts_sort}")
                    for post in extract_posts(payload)
                ]
                if posts:
                    submolt_posts.extend(posts)
            except MoltbookAuthError:
                raise
            except Exception as e:
                logger.debug("Discovery submolt feed failed submolt=%s error=%s", submolt, e)
    if submolt_posts:
        sources.append("submolts")
    if per_sort_posts:
        sources.append("posts")
        sources.append("posts:" + ",".join(sorted(per_sort_posts.keys())))
    sources.append("feed")
    merged = merge_unique_posts(search_posts + submolt_posts + global_posts + feed_posts, post_id_fn=post_id_fn)
    now_dt = datetime.now(timezone.utc)
    early_window_seconds = int(getattr(cfg, "early_comment_window_seconds", 900) or 900)
    for post in merged:
        feed_tags = post.get("__feed_sources", [])
        has_hot = isinstance(feed_tags, list) and "hot" in feed_tags
        is_early = is_early_comment_candidate(
            post=post,
            now_ts=now_dt.timestamp(),
            early_window_seconds=early_window_seconds,
        )
        if has_hot or is_early:
            post["__fast_lane_comment"] = True
    logger.info(
        "Discovery merged search_posts=%s submolt_posts=%s posts_global=%s feed_posts=%s total=%s",
        len(search_posts),
        len(submolt_posts),
        len(global_posts),
        len(feed_posts),
        len(merged),
    )
    return merged, sources


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
