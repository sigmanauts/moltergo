from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..moltbook_client import MoltbookAuthError, MoltbookClient

from .config import Config, load_config
from .drafting import (
    build_openai_messages,
    call_openai,
    fallback_draft,
    format_content,
    load_persona_text,
    normalize_str,
    post_url,
    propose_keywords_from_titles,
)
from .keywords import load_keyword_store, merge_keywords, save_keyword_store
from .logging_utils import setup_logging
from .state import load_state, reset_daily_if_needed, save_state, utc_now


VALID_RESPONSE_MODES = {"comment", "post", "both", "none"}
VALID_VOTE_ACTIONS = {"upvote", "downvote", "none"}
VALID_VOTE_TARGETS = {"post", "top_comment", "both", "none"}


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
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("comments", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def supports_color() -> bool:
    return sys.stdout.isatty() and not bool(os.getenv("NO_COLOR"))


def print_success_banner(action: str, pid: str, url: str, title: str) -> None:
    if supports_color():
        green = "\033[1;32m"
        cyan = "\033[1;36m"
        reset = "\033[0m"
        print("")
        print(f"{green}============================================{reset}")
        print(f"{green} ACTION SUCCESS: {action.upper()}{reset}")
        print(f"{cyan} post_id:{reset} {pid}")
        print(f"{cyan} title:{reset} {title}")
        print(f"{cyan} url:{reset} {url}")
        print(f"{green}============================================{reset}")
        print("")
        return

    print("")
    print("============================================")
    print(f"ACTION SUCCESS: {action.upper()}")
    print(f"post_id: {pid}")
    print(f"title: {title}")
    print(f"url: {url}")
    print("============================================")
    print("")


def print_cycle_banner(iteration: int, mode: str, keywords: int) -> None:
    print("")
    print("------------------------------------------------------------")
    print(f"CYCLE {iteration} | discovery={mode} | keywords={keywords}")
    print("------------------------------------------------------------")


def post_id(post: Dict[str, Any]) -> Optional[str]:
    pid = post.get("id") or (post.get("post") or {}).get("id")
    return str(pid) if pid is not None else None


def post_author(post: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    author = post.get("author") or post.get("user") or post.get("agent") or {}
    author_id = author.get("id") or author.get("agent_id") or post.get("author_id")
    author_name = author.get("name") or author.get("username") or post.get("author_name")
    return (
        str(author_id) if author_id is not None else None,
        str(author_name) if author_name is not None else None,
    )


def comment_id(comment: Dict[str, Any]) -> Optional[str]:
    cid = comment.get("id") or (comment.get("comment") or {}).get("id")
    return str(cid) if cid is not None else None


def comment_author(comment: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    author = comment.get("author") or comment.get("user") or comment.get("agent") or {}
    author_id = author.get("id") or author.get("agent_id") or comment.get("author_id")
    author_name = author.get("name") or author.get("username") or comment.get("author_name")
    return (
        str(author_id) if author_id is not None else None,
        str(author_name) if author_name is not None else None,
    )


def comment_score(comment: Dict[str, Any]) -> int:
    for key in ("score", "vote_score", "upvotes", "likes"):
        value = comment.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


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
    if state.get("daily_post_count", 0) >= cfg.max_posts_per_day:
        return False
    last_post = state.get("last_post_action_ts")
    if isinstance(last_post, (int, float)):
        if utc_now().timestamp() - last_post < cfg.min_seconds_between_posts:
            return False
    return True


def can_comment(state: Dict[str, Any], cfg: Config) -> bool:
    if state.get("daily_comment_count", 0) >= cfg.max_comments_per_day:
        return False
    last_comment = state.get("last_comment_action_ts")
    if isinstance(last_comment, (int, float)):
        if utc_now().timestamp() - last_comment < cfg.min_seconds_between_comments:
            return False
    return True


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
        if comment_ok:
            return ["comment"]
        return []

    if requested_mode == "comment":
        if comment_ok:
            return ["comment"]
        if post_ok:
            return ["post"]
        return []

    # both
    actions: List[str] = []
    if comment_ok:
        actions.append("comment")
    if post_ok:
        actions.append("post")
    return actions


def preview_text(content: str, max_chars: int = 600) -> str:
    normalized = content.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "... [truncated]"


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

    candidates: List[Dict[str, Any]] = []
    for comment in comments:
        _, c_author_name = comment_author(comment)
        if my_name and c_author_name and c_author_name.lower() == my_name.lower():
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

    post_allowed = can_post(state, cfg)
    comment_allowed = can_comment(state, cfg)

    if cfg.reply_mode == "post" and not post_allowed:
        return False, "post_cooldown_or_limit"

    if cfg.reply_mode == "comment" and not comment_allowed:
        return False, "comment_cooldown_or_limit"

    if cfg.reply_mode not in {"post", "comment"} and not post_allowed and not comment_allowed:
        return False, "post_comment_cooldown_or_limit"

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
    sources.extend(["posts", "feed"])
    merged = merge_unique_posts(search_posts + global_posts + feed_posts)
    logger.info(
        "Discovery merged search_posts=%s posts_global=%s feed_posts=%s total=%s",
        len(search_posts),
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


def confirm_action(
    cfg: Config,
    logger,
    action: str,
    pid: str,
    title: str,
    submolt: str,
    url: str,
    author: str,
    content_preview: Optional[str],
    approve_all: bool,
) -> Tuple[bool, bool, bool]:
    if not cfg.confirm_actions:
        return True, approve_all, False

    if approve_all:
        return True, approve_all, False

    if not sys.stdin.isatty():
        logger.warning(
            "Skipping action=%s post_id=%s because confirmation is enabled but stdin is not interactive. "
            "Set MOLTBOOK_CONFIRM_ACTIONS=0 for unattended mode.",
            action,
            pid,
        )
        return False, approve_all, False

    print("")
    print("[confirm] Proposed Moltbook action")
    print(f"  action: {action}")
    print(f"  post_id: {pid}")
    print(f"  author: {author}")
    print(f"  submolt: {submolt}")
    print(f"  url: {url}")
    print(f"  title: {title}")
    if content_preview:
        print("  draft:")
        print("  ---")
        for line in content_preview.splitlines() or [""]:
            print(f"  {line}")
        print("  ---")
    choice = input("Proceed? [y]es / [n]o / [a]ll remaining / [q]uit: ").strip().lower()

    if choice in {"y", "yes"}:
        return True, approve_all, False
    if choice in {"a", "all"}:
        logger.info("Operator approved all remaining actions for this run.")
        return True, True, False
    if choice in {"q", "quit"}:
        logger.info("Operator requested stop from confirmation prompt.")
        return False, approve_all, True
    return False, approve_all, False


def confirm_keyword_addition(
    logger,
    keyword: str,
    approve_all: bool,
) -> Tuple[bool, bool, bool]:
    if approve_all:
        return True, approve_all, False

    if not sys.stdin.isatty():
        logger.warning(
            "Skipping learned keyword '%s' because stdin is non-interactive.",
            keyword,
        )
        return False, approve_all, False

    print("")
    print("[confirm] Learned keyword suggestion")
    print(f"  keyword: {keyword}")
    choice = input("Add this keyword to learning store? [y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
    if choice in {"y", "yes"}:
        return True, approve_all, False
    if choice in {"a", "all"}:
        logger.info("Operator approved all remaining keyword suggestions for this run.")
        return True, True, False
    if choice in {"q", "quit"}:
        logger.info("Operator requested stop from keyword confirmation prompt.")
        return False, approve_all, True
    return False, approve_all, False


def run_loop() -> None:
    cfg = load_config()
    logger = setup_logging(cfg)
    client = MoltbookClient()

    logger.info(
        (
            "Autonomy loop starting discovery_mode=%s reply_mode=%s poll_seconds=%s feed_limit=%s "
            "search_limit=%s idle_poll_seconds=%s dry_run=%s openai_enabled=%s state_path=%s"
        ),
        cfg.discovery_mode,
        cfg.reply_mode,
        cfg.poll_seconds,
        cfg.feed_limit,
        cfg.search_limit,
        cfg.idle_poll_seconds,
        cfg.dry_run,
        bool(cfg.openai_api_key),
        cfg.state_path,
    )
    if cfg.log_path:
        logger.info("File logging enabled path=%s", cfg.log_path)

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
    keyword_store = load_keyword_store(cfg.keyword_store_path)
    active_keywords = merge_keywords(cfg.keywords, keyword_store.get("learned_keywords", []))
    logger.info(
        "Loaded persona guidance path=%s keywords=%s learned_keywords=%s mission_queries=%s do_not_reply_authors=%s",
        cfg.persona_path,
        len(active_keywords),
        len(keyword_store.get("learned_keywords", [])),
        len(cfg.mission_queries),
        len(cfg.do_not_reply_authors),
    )

    state = load_state(cfg.state_path)
    seen: Set[str] = set(state.get("seen_post_ids", []))
    logger.info("Loaded state seen_posts=%s", len(seen))
    if cfg.confirm_actions:
        logger.info("Interactive confirmation enabled for outgoing actions.")
    else:
        logger.info("Interactive confirmation disabled (autonomous send mode).")

    iteration = 0
    approve_all_actions = False
    approve_all_keyword_changes = False
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
            posts, sources = discover_posts(
                client=client,
                cfg=cfg,
                logger=logger,
                keywords=active_keywords,
                iteration=iteration,
                search_state=search_state,
            )
            logger.info("Poll cycle=%s discovered_posts=%s sources=%s", iteration, len(posts), ",".join(sources))

            inspected = 0
            new_candidates = 0
            eligible_now = 0
            drafted_count = 0
            model_approved = 0
            acted = 0
            reply_actions = 0
            skip_reasons: Dict[str, int] = {}
            cycle_titles: List[str] = []
            remaining_cooldown = cooldown_remaining_seconds(state=state, cfg=cfg)

            if remaining_cooldown > 0:
                logger.info(
                    "Global action cooldown active remaining=%ss (~%sm). No new post/comment actions this cycle.",
                    remaining_cooldown,
                    max(1, remaining_cooldown // 60),
                )

            def mark_seen(pid: Optional[str]) -> None:
                if not pid:
                    return
                seen.add(pid)
                state["seen_post_ids"] = list(seen)[-5000:]

            for post in posts:
                inspected += 1
                title_text = normalize_str(post.get("title")).strip()
                if title_text:
                    cycle_titles.append(title_text)
                pid = post_id(post)
                if not pid or pid in seen:
                    logger.debug("Cycle=%s skip post_id=%s reason=seen_or_missing", iteration, pid)
                    continue

                new_candidates += 1

                author_id, author_name = post_author(post)
                if my_name and author_name and author_name.lower() == str(my_name).lower():
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
                    if reason not in {
                        "cooldown",
                        "daily_post_limit",
                        "daily_comment_limit",
                        "daily_engagement_limit",
                        "author_cooldown",
                    }:
                        mark_seen(pid)
                    logger.debug(
                        "Cycle=%s skip post_id=%s author=%s reason=%s",
                        iteration,
                        pid,
                        author_name or author_id or "(unknown)",
                        reason,
                    )
                    continue
                eligible_now += 1

                try:
                    logger.info("Cycle=%s drafting post_id=%s via=openai", iteration, pid)
                    messages = build_openai_messages(persona_text, post, pid)
                    draft = call_openai(cfg, messages)
                    drafted_count += 1
                except Exception as e:
                    logger.warning("Cycle=%s openai_draft_failed post_id=%s error=%s", iteration, pid, e)
                    draft = fallback_draft()
                    drafted_count += 1
                    logger.info("Cycle=%s using_fallback_draft post_id=%s", iteration, pid)

                if not draft.get("should_respond", False):
                    logger.info("Cycle=%s model_declined post_id=%s", iteration, pid)
                    mark_seen(pid)
                    continue

                confidence = float(draft.get("confidence", 0))
                if confidence < cfg.min_confidence:
                    logger.info(
                        "Cycle=%s skip post_id=%s reason=low_confidence confidence=%.3f threshold=%.3f",
                        iteration,
                        pid,
                        confidence,
                        cfg.min_confidence,
                    )
                    mark_seen(pid)
                    continue
                model_approved += 1

                content = format_content(draft)
                if not content:
                    logger.warning("Cycle=%s skip post_id=%s reason=empty_content", iteration, pid)
                    mark_seen(pid)
                    continue

                url = post_url(pid)
                title = normalize_str(draft.get("title")) or "Quick question on your post"
                raw_submolt = post.get("submolt")
                submolt = normalize_submolt(raw_submolt)
                comment_content = content
                post_content = f"I saw your post here: {url}\n\n{content}"
                logger.debug(
                    "Cycle=%s normalized_submolt post_id=%s raw_type=%s value=%s",
                    iteration,
                    pid,
                    type(raw_submolt).__name__,
                    submolt,
                )

                requested_mode = normalize_response_mode(draft.get("response_mode"), default="comment")
                actions = planned_actions(requested_mode=requested_mode, cfg=cfg, state=state)
                if not actions:
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
                    approved, approve_all_actions, should_stop = confirm_action(
                        cfg=cfg,
                        logger=logger,
                        action=action,
                        pid=pid,
                        title=title,
                        submolt=submolt,
                        url=url,
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
                            client.create_comment(pid, comment_content)
                            state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                            acted += 1
                            reply_actions += 1
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
                                pid=pid,
                                title=title,
                                submolt=submolt,
                                url=url,
                                author=author_name or author_id or "(unknown)",
                                content_preview=preview_text(post_content),
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
                                "Cycle=%s action=post attempt post_id=%s submolt=%s url=%s title=%s",
                                iteration,
                                pid,
                                submolt,
                                url,
                                title,
                            )
                            client.create_post(submolt=submolt, title=title, content=post_content)
                            state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                            acted += 1
                            reply_actions += 1
                            reply_executed = True
                            logger.info(
                                "Cycle=%s action=post success post_id=%s daily_post_count=%s",
                                iteration,
                                pid,
                                state["daily_post_count"],
                            )
                            print_success_banner(action="post", pid=pid, url=url, title=title)
                    elif action == "post":
                        logger.info(
                            "Cycle=%s action=post attempt post_id=%s submolt=%s url=%s title=%s",
                            iteration,
                            pid,
                            submolt,
                            url,
                            title,
                        )
                        client.create_post(submolt=submolt, title=title, content=post_content)
                        state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                        acted += 1
                        reply_actions += 1
                        reply_executed = True
                        logger.info(
                            "Cycle=%s action=post success post_id=%s daily_post_count=%s",
                            iteration,
                            pid,
                            state["daily_post_count"],
                        )
                        print_success_banner(action="post", pid=pid, url=url, title=title)

                vote_action = normalize_vote_action(draft.get("vote_action"))
                vote_target = normalize_vote_target(draft.get("vote_target"))
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

                if reply_executed:
                    state["last_action_ts"] = utc_now().timestamp()
                if reply_executed and author_id:
                    state.setdefault("per_author_last_reply", {})[author_id] = state["last_action_ts"]
                if reply_executed:
                    mark_seen(pid)

            state["seen_post_ids"] = list(seen)[-5000:]
            save_state(cfg.state_path, state)
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
                and cfg.openai_api_key
                and cfg.keyword_learning_interval_cycles > 0
                and iteration % cfg.keyword_learning_interval_cycles == 0
                and len(cycle_titles) >= cfg.keyword_learning_min_titles
            ):
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

                if suggestions:
                    logger.info(
                        "Keyword learning cycle=%s suggested=%s",
                        iteration,
                        ", ".join(suggestions),
                    )
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
                        print("")
                        print("============================================")
                        print("KEYWORD ADDED")
                        print(f"keyword: {keyword}")
                        print(f"learned_total: {len(learned_after)}")
                        print("============================================")
                        print("")

            # Sleep policy:
            # - After a successful post/comment, sleep until action cooldown expires.
            # - When idle (no post/comment action), poll quickly.
            sleep_seconds = max(1, cfg.idle_poll_seconds)
            sleep_reason = "idle_poll"
            if reply_actions > 0:
                now_ts = utc_now().timestamp()
                last_action_ts = state.get("last_action_ts")
                if isinstance(last_action_ts, (int, float)):
                    remaining = int(cfg.min_seconds_between_actions - (now_ts - last_action_ts))
                    if remaining > 0:
                        sleep_seconds = remaining
                        sleep_reason = "post_comment_cooldown"
                    else:
                        sleep_seconds = max(1, cfg.idle_poll_seconds)
                        sleep_reason = "cooldown_elapsed"
                else:
                    sleep_seconds = max(1, cfg.poll_seconds)
                    sleep_reason = "post_comment_default_poll"
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
