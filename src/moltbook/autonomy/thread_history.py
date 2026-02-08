from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..moltbook_client import MoltbookClient
from .drafting import normalize_str
from .runtime_helpers import (
    _normalized_name_key,
    author_identity_key,
    comment_author,
    comment_parent_id,
    extract_comments,
    is_self_author,
)


def has_my_comment_on_post(
    client: MoltbookClient,
    post_id_value: str,
    my_name: Optional[str],
    logger,
) -> bool:
    if not my_name:
        return False
    my_key = author_identity_key(author_id=None, author_name=my_name)
    my_name_key = _normalized_name_key(my_name)
    try:
        payload = client.get_post_comments(post_id_value, limit=200)
    except Exception as e:
        logger.debug("Comment history check failed post_id=%s error=%s", post_id_value, e)
        return False
    for comment in extract_comments(payload):
        author_id, author_name = comment_author(comment)
        if my_key and author_identity_key(author_id, author_name) == my_key:
            return True
        if my_name_key and _normalized_name_key(author_name) == my_name_key:
            return True
    return False


def has_my_reply_to_comment(
    client: MoltbookClient,
    post_id_value: str,
    parent_comment_id: str,
    my_name: Optional[str],
    logger,
    self_identity_keys: Optional[Set[str]] = None,
) -> bool:
    if not my_name:
        return False
    my_key = author_identity_key(author_id=None, author_name=my_name)
    my_name_key = _normalized_name_key(my_name)
    self_keys: Set[str] = set(self_identity_keys or set())
    if my_key:
        self_keys.add(my_key)
    parent_key = normalize_str(parent_comment_id).strip()
    if not parent_key:
        return False
    try:
        payload = client.get_post_comments(post_id_value, limit=250)
    except Exception as e:
        logger.debug(
            "Reply-parent history check failed post_id=%s parent_comment_id=%s error=%s",
            post_id_value,
            parent_comment_id,
            e,
        )
        return False
    for comment in extract_comments(payload):
        if normalize_str(comment_parent_id(comment)).strip() != parent_key:
            continue
        author_id, author_name = comment_author(comment)
        if self_keys and is_self_author(author_id, author_name, self_identity_keys=self_keys):
            return True
        if my_name_key and _normalized_name_key(author_name) == my_name_key:
            return True
    return False


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
