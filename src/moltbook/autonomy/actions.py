from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..moltbook_client import MoltbookClient
from .action_journal import append_action_journal
from .config import Config
from .drafting import normalize_str, post_url, sanitize_publish_content
from .runtime_helpers import (
    comment_gate_status,
    normalize_vote_action,
    post_gate_status,
    register_my_comment_id,
)
from .state import utc_now
from .thread_history import has_my_reply_to_comment
from .ui import print_success_banner


def _publish_signature(
    *,
    action_type: str,
    target_post_id: Any,
    content: Any,
    parent_comment_id: Any = None,
) -> str:
    action = normalize_str(action_type).strip().lower()
    post_id_value = normalize_str(target_post_id).strip()
    parent_id_value = normalize_str(parent_comment_id).strip()
    normalized_content = " ".join(normalize_str(content).strip().lower().split())
    digest = hashlib.sha1(normalized_content.encode("utf-8")).hexdigest()[:16] if normalized_content else "empty"
    return f"{action}:{post_id_value}:{parent_id_value}:{digest}"


def _seen_publish_signature(state: Dict[str, Any], signature: str) -> bool:
    raw = state.get("recent_publish_signatures", [])
    if not isinstance(raw, list):
        return False
    return signature in raw


def _remember_publish_signature(state: Dict[str, Any], signature: str) -> None:
    raw = state.get("recent_publish_signatures", [])
    if not isinstance(raw, list):
        raw = []
    raw.append(signature)
    state["recent_publish_signatures"] = raw[-30000:]


def can_reply(
    state: Dict[str, Any],
    cfg: Config,
    author_id: Optional[str] = None,
    author_name: Optional[str] = None,
) -> Tuple[bool, str]:
    author_key = normalize_str(author_id or author_name).strip().lower()
    if author_key:
        deny = {normalize_str(x).strip().lower() for x in cfg.do_not_reply_authors if normalize_str(x).strip()}
        if author_key in deny:
            return False, "do_not_reply_author"

    if author_key:
        per_author = state.get("per_author_last_reply", {})
        if isinstance(per_author, dict):
            last = per_author.get(author_id) or per_author.get(author_name)
            if isinstance(last, (int, float)):
                elapsed = utc_now().timestamp() - float(last)
                if elapsed < max(1, cfg.min_seconds_between_same_author):
                    return False, "author_cooldown"

    post_ok, post_reason = post_gate_status(state=state, cfg=cfg)
    comment_ok, comment_reason = comment_gate_status(state=state, cfg=cfg)
    if post_ok or comment_ok:
        return True, "ok"
    if post_reason == comment_reason:
        return False, post_reason
    return False, f"{post_reason}+{comment_reason}"


def _remaining_since(last_ts: Any, min_seconds: int) -> int:
    if not isinstance(last_ts, (int, float)):
        return 0
    elapsed = utc_now().timestamp() - float(last_ts)
    return max(0, int(min_seconds - elapsed))


def cooldown_remaining_seconds(state: Dict[str, Any], cfg: Config) -> Tuple[int, int]:
    post_remaining = _remaining_since(state.get("last_post_action_ts"), cfg.min_seconds_between_posts)
    comment_remaining = _remaining_since(state.get("last_comment_action_ts"), cfg.min_seconds_between_comments)
    return post_remaining, comment_remaining


def seconds_since_last_post(state: Dict[str, Any]) -> Optional[int]:
    last_post = state.get("last_post_action_ts")
    if not isinstance(last_post, (int, float)):
        return None
    elapsed = int(max(0.0, utc_now().timestamp() - float(last_post)))
    return elapsed


def should_prioritize_proactive_post(state: Dict[str, Any], cfg: Config) -> bool:
    post_allowed, _ = post_gate_status(state=state, cfg=cfg)
    if not post_allowed:
        return False
    if cfg.reply_mode == "none":
        return False
    return True


def has_pending_comment_action(state: Dict[str, Any], post_id_value: str, parent_comment_id: str) -> bool:
    queue = state.get("pending_actions", [])
    if not isinstance(queue, list):
        return False
    pid = normalize_str(post_id_value).strip()
    cid = normalize_str(parent_comment_id).strip()
    for item in queue:
        if not isinstance(item, dict):
            continue
        if normalize_str(item.get("kind")).strip().lower() != "comment":
            continue
        if normalize_str(item.get("post_id")).strip() != pid:
            continue
        if normalize_str(item.get("parent_comment_id")).strip() != cid:
            continue
        return True
    return False


def mark_reply_action_timestamps(state: Dict[str, Any], action_kind: str) -> None:
    now_ts = utc_now().timestamp()
    state["last_action_ts"] = now_ts
    if action_kind == "post":
        state["last_post_action_ts"] = now_ts
        return
    if action_kind == "comment":
        state["last_comment_action_ts"] = now_ts
        timestamps = list(state.get("comment_action_timestamps", []))
        timestamps.append(now_ts)
        state["comment_action_timestamps"] = [float(ts) for ts in timestamps if isinstance(ts, (int, float))][-5000:]


def maybe_upvote_post_after_comment(
    client: MoltbookClient,
    state: Dict[str, Any],
    logger,
    post_id_value: str,
    *,
    journal_path: Optional[Path] = None,
    submolt: str = "",
    post_title: str = "",
    url: Optional[str] = None,
    reference: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    voted_post_ids = set(state.get("voted_post_ids", []))
    if post_id_value in voted_post_ids:
        return
    try:
        client.vote_post(post_id_value, vote_action="upvote")
        voted_post_ids.add(post_id_value)
        state["voted_post_ids"] = list(voted_post_ids)[-10000:]
        logger.info("Auto-upvoted post after comment post_id=%s", post_id_value)
        if journal_path:
            try:
                append_action_journal(
                    journal_path,
                    action_type="upvote-post",
                    target_post_id=post_id_value,
                    submolt=normalize_str(submolt).strip().lower(),
                    title=(normalize_str(post_title).strip() or "Auto upvote after comment"),
                    content="",
                    reference_post_id=post_id_value,
                    url=normalize_str(url).strip() or post_url(post_id_value),
                    reference=reference or {"post_id": post_id_value},
                    meta={"source": "auto_upvote_after_comment", **(meta or {})},
                )
            except Exception as e:
                logger.debug("Auto-upvote action journal write failed post_id=%s error=%s", post_id_value, e)
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


def execute_pending_actions(
    client: MoltbookClient,
    cfg: Config,
    state: Dict[str, Any],
    logger,
    my_name: Optional[str] = None,
    self_identity_keys: Optional[Set[str]] = None,
) -> int:
    queue = list(state.get("pending_actions", []))
    if not queue:
        return 0
    resolved_self_keys: Set[str] = set(self_identity_keys or set())

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
            content = sanitize_publish_content(normalize_str(action.get("content")))
            parent_comment_id = normalize_str(action.get("parent_comment_id")) or None
            if not pid or not content:
                logger.warning("Dropping invalid pending comment action (missing post_id/content).")
                continue
            comment_sig = _publish_signature(
                action_type="comment",
                target_post_id=pid,
                parent_comment_id=parent_comment_id,
                content=content,
            )
            if _seen_publish_signature(state, comment_sig):
                logger.info(
                    "Dropping pending comment with duplicate publish signature post_id=%s parent_comment_id=%s",
                    pid,
                    parent_comment_id or "(none)",
                )
                continue
            if parent_comment_id:
                replied_ids = set(state.get("replied_to_comment_ids", []))
                replied_pairs = set(state.get("replied_comment_pairs", []))
                if parent_comment_id in replied_ids:
                    logger.info(
                        "Dropping pending reply already covered parent_comment_id=%s post_id=%s",
                        parent_comment_id,
                        pid,
                    )
                    continue
                pair_key = f"{normalize_str(pid).strip()}:{normalize_str(parent_comment_id).strip()}"
                if pair_key in replied_pairs:
                    logger.info(
                        "Dropping pending reply already covered pair=%s",
                        pair_key,
                    )
                    continue
                if has_my_reply_to_comment(
                    client=client,
                    post_id_value=pid,
                    parent_comment_id=parent_comment_id,
                    my_name=my_name,
                    logger=logger,
                    self_identity_keys=resolved_self_keys,
                ):
                    replied_ids.add(parent_comment_id)
                    state["replied_to_comment_ids"] = list(replied_ids)[-10000:]
                    replied_pairs.add(pair_key)
                    state["replied_comment_pairs"] = list(replied_pairs)[-20000:]
                    logger.info(
                        "Dropping pending reply already present on-chain parent_comment_id=%s post_id=%s",
                        parent_comment_id,
                        pid,
                    )
                    continue
            logger.info(
                "Executing pending action kind=comment post_id=%s parent_comment_id=%s",
                pid,
                parent_comment_id or "(none)",
            )
            comment_resp = client.create_comment(pid, content, parent_id=parent_comment_id)
            register_my_comment_id(state=state, response_payload=comment_resp)
            _remember_publish_signature(state, comment_sig)
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
                journal_path=cfg.action_journal_path,
                submolt=normalize_str(action.get("submolt")),
                post_title=normalize_str(action.get("title")),
                url=normalize_str(action.get("url")) or post_url(pid),
                reference={
                    "post_id": pid,
                    "post_title": normalize_str(action.get("title")),
                },
                meta={"source": "pending_actions"},
            )
            if parent_comment_id:
                replied_ids = set(state.get("replied_to_comment_ids", []))
                replied_ids.add(parent_comment_id)
                state["replied_to_comment_ids"] = list(replied_ids)[-10000:]
                replied_pairs = set(state.get("replied_comment_pairs", []))
                replied_pairs.add(f"{normalize_str(pid).strip()}:{normalize_str(parent_comment_id).strip()}")
                state["replied_comment_pairs"] = list(replied_pairs)[-20000:]
            executed += 1
            try:
                append_action_journal(
                    cfg.action_journal_path,
                    action_type="comment",
                    target_post_id=pid,
                    submolt=normalize_str(action.get("submolt")),
                    title=normalize_str(action.get("title")) or "Queued comment",
                    content=content,
                    parent_comment_id=parent_comment_id,
                    reference_post_id=pid,
                    url=normalize_str(action.get("url")) or post_url(pid),
                    reference={
                        "post_id": pid,
                        "post_title": normalize_str(action.get("title")),
                        "post_content": normalize_str(action.get("reference_post_content")),
                        "comment_id": normalize_str(parent_comment_id),
                        "comment_content": normalize_str(action.get("reference_comment_content")),
                    },
                    meta={"source": "pending_actions", "kind": "comment"},
                )
            except Exception as e:
                logger.debug("Pending action journal write failed post_id=%s error=%s", pid, e)
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
            try:
                append_action_journal(
                    cfg.action_journal_path,
                    action_type=f"{vote_action}-comment",
                    target_post_id=cid,
                    submolt=normalize_str(action.get("submolt")),
                    title=normalize_str(action.get("title")) or "Queued comment vote",
                    content="",
                    reference_post_id=normalize_str(action.get("post_id")) or None,
                    url=normalize_str(action.get("url")),
                    reference={
                        "comment_id": cid,
                        "vote_action": vote_action,
                        "post_id": normalize_str(action.get("post_id")),
                        "post_title": normalize_str(action.get("post_title")),
                        "comment_content": normalize_str(action.get("reference_comment_content")),
                    },
                    meta={"source": "pending_actions", "kind": "vote_comment"},
                )
            except Exception as e:
                logger.debug("Pending vote-comment journal write failed comment_id=%s error=%s", cid, e)
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
            try:
                append_action_journal(
                    cfg.action_journal_path,
                    action_type=f"{vote_action}-post",
                    target_post_id=pid,
                    submolt=normalize_str(action.get("submolt")),
                    title=normalize_str(action.get("title")) or "Queued post vote",
                    content="",
                    reference_post_id=pid,
                    url=normalize_str(action.get("url")) or post_url(pid),
                    reference={
                        "post_id": pid,
                        "vote_action": vote_action,
                        "post_title": normalize_str(action.get("post_title")) or normalize_str(action.get("title")),
                        "post_content": normalize_str(action.get("reference_post_content")),
                    },
                    meta={"source": "pending_actions", "kind": "vote_post"},
                )
            except Exception as e:
                logger.debug("Pending vote-post journal write failed post_id=%s error=%s", pid, e)
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
