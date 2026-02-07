from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..moltbook_client import MoltbookClient
from .config import Config
from .drafting import normalize_str
from .state import reset_daily_if_needed, utc_now


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
    def _flatten_comment_threads(base_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = list(reversed(base_items))
        seen_keys: Set[str] = set()
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            cid = normalize_str(node.get("id") or (node.get("comment") or {}).get("id")).strip()
            key = cid if cid else f"obj:{id(node)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(node)
            for child_key in ("replies", "children", "comments", "items"):
                children = node.get(child_key)
                if isinstance(children, list):
                    for child in reversed(children):
                        if isinstance(child, dict):
                            stack.append(child)
        return out

    if isinstance(payload, list):
        base = [item for item in payload if isinstance(item, dict)]
        return _flatten_comment_threads(base)
    if not isinstance(payload, dict):
        return []
    for key in ("comments", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            base = [item for item in value if isinstance(item, dict)]
            return _flatten_comment_threads(base)
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


def _normalized_name_key(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    if not text:
        return ""
    if text.startswith("u/"):
        text = text[2:]
    if text.startswith("@"):
        text = text[1:]
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def author_identity_key(author_id: Optional[str], author_name: Optional[str]) -> str:
    aid = _normalized_name_key(author_id)
    if aid:
        return f"id:{aid}"
    aname = _normalized_name_key(author_name)
    if aname:
        return f"name:{aname}"
    return ""


def author_identity_keys(author_id: Optional[str], author_name: Optional[str]) -> Set[str]:
    keys: Set[str] = set()
    aid = _normalized_name_key(author_id)
    if aid:
        keys.add(f"id:{aid}")
    aname = _normalized_name_key(author_name)
    if aname:
        keys.add(f"name:{aname}")
    return keys


def resolve_self_identity_keys(client: MoltbookClient, my_name: Optional[str], logger) -> Set[str]:
    keys: Set[str] = set()
    if my_name:
        key = author_identity_key(author_id=None, author_name=my_name)
        if key:
            keys.add(key)
    try:
        me = client.get_me()
        containers: List[Dict[str, Any]] = []
        if isinstance(me, dict):
            containers.append(me)
            for field in ("agent", "data", "profile", "result"):
                nested = me.get(field)
                if isinstance(nested, dict):
                    containers.append(nested)
        for container in containers:
            if not isinstance(container, dict):
                continue
            for id_field in ("id", "agent_id", "user_id"):
                raw_id = container.get(id_field)
                if raw_id is None:
                    continue
                key = author_identity_key(author_id=str(raw_id), author_name=None)
                if key:
                    keys.add(key)
            for name_field in ("name", "agent_name", "username", "created_by"):
                raw_name = container.get(name_field)
                if raw_name is None:
                    continue
                key = author_identity_key(author_id=None, author_name=str(raw_name))
                if key:
                    keys.add(key)
    except Exception as e:
        logger.debug("Could not resolve self identity keys from /agents/me error=%s", e)
    return keys


def is_self_author(author_id: Optional[str], author_name: Optional[str], self_identity_keys: Set[str]) -> bool:
    if not self_identity_keys:
        return False
    keys = author_identity_keys(author_id=author_id, author_name=author_name)
    if not keys:
        return False
    return any(key in self_identity_keys for key in keys)


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
        or comment.get("parentCommentId")
        or comment.get("parent_comment_id")
        or comment.get("reply_to_id")
        or comment.get("replyToId")
        or comment.get("reply_to_comment_id")
        or comment.get("replyToCommentId")
        or nested_comment.get("parent_id")
        or nested_comment.get("parentId")
        or nested_comment.get("parentCommentId")
        or nested_comment.get("parent_comment_id")
        or nested_comment.get("reply_to_id")
        or nested_comment.get("replyToId")
        or nested_comment.get("reply_to_comment_id")
        or nested_comment.get("replyToCommentId")
    )
    if parent is None:
        parent_obj = comment.get("parent") or comment.get("reply_to") or nested_comment.get("parent")
        if isinstance(parent_obj, dict):
            parent = parent_obj.get("id") or parent_obj.get("comment_id") or parent_obj.get("commentId")
        elif parent_obj is not None:
            parent = parent_obj
    if parent is None:
        return None
    return str(parent)


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


def can_post(state: Dict[str, Any], cfg: Config) -> bool:
    allowed, _ = post_gate_status(state=state, cfg=cfg)
    return allowed


def can_comment(state: Dict[str, Any], cfg: Config) -> bool:
    allowed, _ = comment_gate_status(state=state, cfg=cfg)
    return allowed


def planned_actions(
    requested_mode: str,
    cfg: Config,
    state: Dict[str, Any],
) -> List[str]:
    post_ok = can_post(state, cfg)
    comment_ok = can_comment(state, cfg)

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
    normalized = content.strip().replace("...[truncated]", "...").replace("... [truncated]", "...").replace("[truncated]", "")
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."
