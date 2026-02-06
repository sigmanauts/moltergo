from __future__ import annotations

import os
import re
import select
import sys
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
    build_learning_snapshot,
    load_post_engine_memory,
    record_declined_idea,
    record_proactive_post,
    refresh_metrics_from_recent_posts,
    save_post_engine_memory,
)
from .state import load_state, reset_daily_if_needed, save_state, utc_now


VALID_RESPONSE_MODES = {"comment", "post", "both", "none"}
VALID_VOTE_ACTIONS = {"upvote", "downvote", "none"}
VALID_VOTE_TARGETS = {"post", "top_comment", "both", "none"}
FORCE_REPLY_ON_OWN_THREADS = True


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
        or comment.get("parent_comment_id")
        or comment.get("reply_to_id")
        or nested_comment.get("parent_id")
        or nested_comment.get("parentId")
        or nested_comment.get("parent_comment_id")
        or nested_comment.get("reply_to_id")
    )
    if parent is None:
        parent_obj = comment.get("parent")
        if isinstance(parent_obj, dict):
            parent = parent_obj.get("id")
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


def preview_text(content: str, max_chars: int = 600) -> str:
    normalized = content.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "... [truncated]"


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


def forced_reply_text(incoming_body: str, vote_action: str) -> str:
    if vote_action == "downvote" or looks_spammy_comment(incoming_body):
        return "Bad bot."
    return (
        "Noted. We are building toward verifiable agent economies on Ergo using deterministic eUTXO contracts "
        "and on-chain settlement. What concrete constraint would you test first?"
    )


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
) -> int:
    queue = list(state.get("pending_actions", []))
    if not queue:
        return 0

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
            content = normalize_str(action.get("content"))
            parent_comment_id = normalize_str(action.get("parent_comment_id")) or None
            if not pid or not content:
                logger.warning("Dropping invalid pending comment action (missing post_id/content).")
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


def build_top_post_signals(posts: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(posts, key=lambda p: (post_score(p), post_comment_count(p)), reverse=True)
    signals: List[Dict[str, Any]] = []
    for post in ranked[: max(1, limit)]:
        pid = post_id(post)
        if not pid:
            continue
        signals.append(
            {
                "post_id": pid,
                "title": normalize_str(post.get("title")).strip(),
                "submolt": submolt_name_from_post(post) or normalize_submolt(post.get("submolt")),
                "score": post_score(post),
                "comment_count": post_comment_count(post),
            }
        )
    return signals


def proactive_post_attempt_allowed(state: Dict[str, Any], cfg: Config) -> bool:
    last_attempt = state.get("last_proactive_post_attempt_ts")
    if not isinstance(last_attempt, (int, float)):
        return True
    elapsed = utc_now().timestamp() - last_attempt
    return elapsed >= max(1, cfg.proactive_post_attempt_cooldown_seconds)


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
        logger.info(
            "Proactive post skipped reason=attempt_cooldown seconds=%s",
            cfg.proactive_post_attempt_cooldown_seconds,
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

    try:
        top_payload = client.get_posts(sort="top", limit=max(5, cfg.proactive_post_reference_limit))
        top_posts = extract_posts(top_payload)
    except Exception as e:
        logger.warning("Proactive post skipped reason=top_posts_fetch_failed error=%s", e)
        return 0, False, approve_all_actions

    top_signals = build_top_post_signals(posts=top_posts, limit=cfg.proactive_post_reference_limit)
    learning_snapshot = build_learning_snapshot(post_memory, max_examples=5)
    if not top_signals:
        logger.info("Proactive post skipped reason=no_top_signals")
        return 0, False, approve_all_actions

    state["last_proactive_post_attempt_ts"] = utc_now().timestamp()
    provider_used = "unknown"
    try:
        messages = build_proactive_post_messages(
            persona=persona_text,
            domain_context=domain_context_text,
            top_posts=top_signals,
            learning_snapshot=learning_snapshot,
            target_submolt=cfg.proactive_post_submolt,
        )
        draft, provider_used = call_generation_model(cfg, messages)
    except Exception as e:
        logger.warning("Proactive post drafting failed provider_hint=%s error=%s", cfg.llm_provider, e)
        return 0, False, approve_all_actions
    logger.info("Proactive post draft generated provider=%s", provider_used)

    should_post = bool(draft.get("should_post"))
    confidence = float(draft.get("confidence", 0.0))
    if not should_post or confidence < cfg.min_confidence:
        logger.info(
            "Proactive post declined should_post=%s confidence=%.3f threshold=%.3f",
            should_post,
            confidence,
            cfg.min_confidence,
        )
        record_declined_idea(
            memory=post_memory,
            title=normalize_str(draft.get("title")).strip() or "(untitled)",
            submolt=normalize_submolt(draft.get("submolt"), default=cfg.proactive_post_submolt),
            reason="model_declined_or_low_confidence",
        )
        return 0, False, approve_all_actions

    submolt = normalize_submolt(draft.get("submolt"), default=cfg.proactive_post_submolt)
    title = normalize_str(draft.get("title")).strip() or "Ergo x agent economy: practical next step"
    content = normalize_str(draft.get("content")).strip()
    strategy_notes = normalize_str(draft.get("strategy_notes")).strip()
    raw_tags = draft.get("topic_tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []
    topic_tags = [normalize_str(x).strip().lower() for x in raw_tags if normalize_str(x).strip()]
    topic_tags = topic_tags[:8]
    if not content:
        logger.info("Proactive post skipped reason=empty_content")
        record_declined_idea(
            memory=post_memory,
            title=title,
            submolt=submolt,
            reason="empty_content",
        )
        return 0, False, approve_all_actions

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
    try:
        payload = client.get_post_comments(post_id_value, limit=200)
    except Exception as e:
        logger.debug("Comment history check failed post_id=%s error=%s", post_id_value, e)
        return False
    for comment in extract_comments(payload):
        _, author_name = comment_author(comment)
        if author_name and author_name.lower() == my_name.lower():
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
            "hourly_comment_count=%s/%s daily_comment_count=%s/%s"
        ),
        my_name,
        cfg.startup_reply_scan_post_limit,
        cfg.startup_reply_scan_comment_limit,
        len(hourly_comments),
        cfg.max_comments_per_hour,
        state.get("daily_comment_count", 0),
        cfg.max_comments_per_day,
    )
    seen_comment_ids: Set[str] = set(state.get("seen_comment_ids", []))
    my_comment_ids: Set[str] = set(state.get("my_comment_ids", []))
    voted_comment_ids: Set[str] = set(state.get("voted_comment_ids", []))
    replied_to_comment_ids: Set[str] = set(state.get("replied_to_comment_ids", []))
    replied_post_ids: Set[str] = set(state.get("replied_post_ids", []))
    scanned = 0
    new_replies = 0
    actions = 0
    skip_reasons: Dict[str, int] = {}
    provider_counts: Dict[str, int] = {}

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
            my_name_lc = my_name.lower()
            for p in fallback_posts:
                _, author_name = post_author(p)
                if author_name and author_name.lower() == my_name_lc:
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
        my_replied_parent_ids: Set[str] = set()
        my_comment_ids_in_post: Set[str] = set()
        post_author_id, post_author_name = post_author(post)
        is_profile_recent = normalize_str(post.get("__scan_source")).strip().lower() == "profile_recent"
        is_my_post = bool(
            is_profile_recent
            or
            my_name
            and (
                (post_author_name and post_author_name.lower() == my_name.lower())
                or (post_author_id and post_author_id.lower() == my_name.lower())
            )
        )
        if my_name:
            for maybe_reply in comments:
                maybe_id = comment_id(maybe_reply)
                _, maybe_author = comment_author(maybe_reply)
                parent = comment_parent_id(maybe_reply)
                if maybe_id and maybe_author and maybe_author.lower() == my_name.lower():
                    my_comment_ids.add(maybe_id)
                    my_comment_ids_in_post.add(maybe_id)
                if not parent:
                    continue
                if maybe_author and maybe_author.lower() == my_name.lower():
                    replied_to_comment_ids.add(parent)
                    my_replied_parent_ids.add(parent)
        for comment in comments:
            scanned += 1
            cid = comment_id(comment)
            if not cid:
                continue

            _, c_author_name = comment_author(comment)
            if my_name and c_author_name and c_author_name.lower() == my_name.lower():
                skip_reasons["self_comment"] = skip_reasons.get("self_comment", 0) + 1
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
            already_replied = cid in replied_to_comment_ids or cid in my_replied_parent_ids
            pending_reply_exists = has_pending_comment_action(
                state=state,
                post_id_value=pid,
                parent_comment_id=cid,
            )
            if cid in seen_comment_ids and (already_replied or pending_reply_exists or not is_my_post):
                skip_reasons["already_seen"] = skip_reasons.get("already_seen", 0) + 1
                continue
            if cid not in seen_comment_ids:
                seen_comment_ids.add(cid)
                new_replies += 1
            elif is_my_post and FORCE_REPLY_ON_OWN_THREADS and not already_replied:
                logger.debug("Reply scan revisiting previously seen unanswered comment_id=%s", cid)

            triage: Dict[str, Any]
            try:
                messages = build_reply_triage_messages(
                    persona=persona_text,
                    domain_context=domain_context_text,
                    post=post,
                    comment=comment,
                    post_id=pid,
                    comment_id=cid,
                )
                triage, triage_provider = call_generation_model(cfg, messages)
                provider_counts[triage_provider] = provider_counts.get(triage_provider, 0) + 1
                logger.debug(
                    "Reply triage generated comment_id=%s provider=%s",
                    cid,
                    triage_provider,
                )
            except Exception as e:
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

            should_respond = bool(triage.get("should_respond"))
            if already_replied:
                logger.info("Reply scan skipping reply; already replied to comment_id=%s", cid)
                replied_to_comment_ids.add(cid)
                state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                skip_reasons["already_replied"] = skip_reasons.get("already_replied", 0) + 1
                continue

            reply_content = ""
            if should_respond and response_mode != "none" and confidence >= cfg.min_confidence:
                reply_content = format_content(triage)

            forced_reason: Optional[str] = None
            if not reply_content and is_my_post and FORCE_REPLY_ON_OWN_THREADS:
                if not should_respond:
                    forced_reason = "triage_declined"
                elif response_mode == "none":
                    forced_reason = "response_mode_none"
                elif confidence < cfg.min_confidence:
                    forced_reason = "low_confidence"
                else:
                    forced_reason = "empty_reply_content"
                reply_content = forced_reply_text(incoming_body=incoming_body, vote_action=vote_action)
                logger.info("Reply scan forced reply comment_id=%s reason=%s", cid, forced_reason)

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

            if forced_reason:
                skip_reasons[f"forced_{forced_reason}"] = skip_reasons.get(f"forced_{forced_reason}", 0) + 1
            elif not should_respond:
                skip_reasons["triage_declined"] = skip_reasons.get("triage_declined", 0) + 1
            elif response_mode == "none":
                skip_reasons["response_mode_none"] = skip_reasons.get("response_mode_none", 0) + 1
            elif confidence < cfg.min_confidence:
                skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1

            if not reply_content.strip():
                skip_reasons["empty_reply_content"] = skip_reasons.get("empty_reply_content", 0) + 1
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
                        state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
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
                            state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
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

            if is_my_post and FORCE_REPLY_ON_OWN_THREADS:
                if has_pending_comment_action(state=state, post_id_value=pid, parent_comment_id=cid):
                    skip_reasons[f"already_queued_{comment_reason}"] = (
                        skip_reasons.get(f"already_queued_{comment_reason}", 0) + 1
                    )
                else:
                    enqueue_pending_action(
                        state=state,
                        cfg=cfg,
                        action={
                            "kind": "comment",
                            "post_id": pid,
                            "title": post_title,
                            "url": url,
                            "content": reply_content,
                            "parent_comment_id": cid,
                        },
                    )
                    skip_reasons[f"queued_{comment_reason}"] = (
                        skip_reasons.get(f"queued_{comment_reason}", 0) + 1
                    )
                    logger.info(
                        "Reply scan queued reply for own thread comment_id=%s reason=%s",
                        cid,
                        comment_reason,
                    )
                continue

            skip_reasons[comment_reason] = skip_reasons.get(comment_reason, 0) + 1

    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
    state["my_comment_ids"] = list(my_comment_ids)[-20000:]
    state["replied_post_ids"] = list(replied_post_ids)[-10000:]
    state["voted_comment_ids"] = list(voted_comment_ids)[-10000:]
    state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
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
    return approve_all_actions


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
    prompt = "Proceed? [y]es / [n]o / [a]ll remaining / [q]uit: "
    choice = ""
    if cfg.confirm_timeout_seconds > 0:
        default_choice = cfg.confirm_default_choice
        print(
            (
                f"[auto] default='{default_choice}' in {cfg.confirm_timeout_seconds}s "
                "if no input is provided."
            )
        )
        print(prompt, end="", flush=True)
        ready, _, _ = select.select([sys.stdin], [], [], cfg.confirm_timeout_seconds)
        if ready:
            choice = sys.stdin.readline().strip().lower()
        else:
            print("")
            choice = default_choice
    else:
        choice = input(prompt).strip().lower()

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
        print("")
        print("============================================")
        print("KEYWORD ADDED")
        print(f"keyword: {keyword}")
        print(f"learned_total: {len(learned_after)}")
        print("============================================")
        print("")

    keyword_store["pending_suggestions"] = merge_keywords([], remaining)
    save_keyword_store(cfg.keyword_store_path, keyword_store)
    return active_keywords, keyword_store, approve_all_keyword_changes, should_stop_run


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
    if not has_generation_provider(cfg):
        logger.info("Self-improvement skipped cycle=%s reason=no_generation_provider", iteration)
        return

    provider_used = "unknown"
    try:
        messages = build_self_improvement_messages(
            persona=persona_text,
            domain_context=domain_context_text,
            learning_snapshot=learning_snapshot,
            recent_titles=cycle_titles,
            cycle_stats=cycle_stats,
        )
        suggestions, provider_used = call_generation_model(cfg, messages)
    except Exception as e:
        logger.warning("Self-improvement failed cycle=%s error=%s", iteration, e)
        return

    if not isinstance(suggestions, dict):
        logger.warning("Self-improvement returned non-object payload cycle=%s", iteration)
        return

    max_suggestions = max(1, cfg.self_improve_max_suggestions)
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
        logger.info("Self-improvement produced no actionable suggestions cycle=%s", iteration)
        return

    append_improvement_suggestions(
        path=cfg.self_improve_path,
        cycle=iteration,
        provider=provider_used,
        suggestions=suggestions,
    )
    logger.info(
        "Self-improvement suggestions saved cycle=%s provider=%s path=%s",
        iteration,
        provider_used,
        cfg.self_improve_path,
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

    logger.info(
        (
            "Autonomy loop starting discovery_mode=%s reply_mode=%s poll_seconds=%s feed_limit=%s "
            "search_limit=%s idle_poll_seconds=%s dry_run=%s llm_provider=%s openai_enabled=%s "
            "chatbase_enabled=%s self_improve_enabled=%s state_path=%s"
        ),
        cfg.discovery_mode,
        cfg.reply_mode,
        cfg.poll_seconds,
        cfg.feed_limit,
        cfg.search_limit,
        cfg.idle_poll_seconds,
        cfg.dry_run,
        cfg.llm_provider,
        bool(cfg.openai_api_key),
        bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id),
        cfg.self_improve_enabled,
        cfg.state_path,
    )
    if cfg.log_path:
        logger.info("File logging enabled path=%s", cfg.log_path)
    if cfg.self_improve_enabled:
        logger.info("Self-improvement suggestions path=%s", cfg.self_improve_path)

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
    approve_all_actions = maybe_subscribe_relevant_submolts(
        client=client,
        cfg=cfg,
        logger=logger,
        state=state,
        approve_all_actions=approve_all_actions,
    )
    save_state(cfg.state_path, state)
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
            pending_executed = execute_pending_actions(
                client=client,
                cfg=cfg,
                state=state,
                logger=logger,
            )
            if pending_executed:
                save_state(cfg.state_path, state)
            if cfg.startup_reply_scan_enabled and cfg.reply_scan_interval_cycles > 0:
                if iteration % cfg.reply_scan_interval_cycles == 0:
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

            inspected = 0
            new_candidates = 0
            eligible_now = 0
            drafted_count = 0
            model_approved = 0
            acted = 0
            reply_actions = 0
            post_action_sent = False
            comment_action_sent = False
            skip_reasons: Dict[str, int] = {}
            provider_counts: Dict[str, int] = {}
            cycle_titles: List[str] = []
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
                title_text = normalize_str(post.get("title")).strip()
                if title_text:
                    cycle_titles.append(title_text)
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

                allowed_modes = currently_allowed_response_modes(cfg=cfg, state=state)
                if allowed_modes == ["none"]:
                    skip_reasons["no_action_slots"] = skip_reasons.get("no_action_slots", 0) + 1
                    mark_seen(pid)
                    logger.debug("Cycle=%s skip post_id=%s reason=no_action_slots", iteration, pid)
                    continue

                provider_used = "unknown"
                try:
                    logger.debug(
                        "Cycle=%s drafting post_id=%s title=%s provider_hint=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        cfg.llm_provider,
                    )
                    messages = build_openai_messages(
                        persona=persona_text,
                        domain_context=domain_context_text,
                        post=post,
                        pid=pid,
                        allowed_response_modes=allowed_modes,
                    )
                    draft, provider_used = call_generation_model(cfg, messages)
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

                if not draft.get("should_respond", False):
                    logger.info(
                        "Cycle=%s model_declined post_id=%s title=%s",
                        iteration,
                        pid,
                        post_title_preview,
                    )
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
                            "Cycle=%s action=post success post_id=%s daily_post_count=%s",
                            iteration,
                            pid,
                            state["daily_post_count"],
                        )
                        print_success_banner(action="post", pid=pid, url=url, title=title)

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

            if acted == 0 and not post_action_sent:
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
                and has_generation_provider(cfg)
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
                            print("")
                            print("============================================")
                            print("KEYWORD ADDED")
                            print(f"keyword: {keyword}")
                            print(f"learned_total: {len(learned_after)}")
                            print("============================================")
                            print("")
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
            }
            maybe_write_self_improvement_suggestions(
                cfg=cfg,
                logger=logger,
                iteration=iteration,
                persona_text=persona_text,
                domain_context_text=domain_context_text,
                learning_snapshot=build_learning_snapshot(post_memory, max_examples=5),
                cycle_titles=cycle_titles,
                cycle_stats=cycle_stats,
            )

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
