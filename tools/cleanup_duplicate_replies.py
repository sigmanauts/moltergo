#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

from moltbook.moltbook_client import MoltbookClient


def normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def extract_posts(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("posts", "data", "items", "results", "recentPosts", "recent_posts"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    for key in ("agent", "profile", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = extract_posts(value)
            if nested:
                return nested
    return []


def extract_me(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("agent", "data", "result", "profile"):
        value = payload.get(key)
        if isinstance(value, dict):
            if "name" in value or "agent_name" in value or "id" in value or "agent_id" in value:
                return value
            nested = extract_me(value)
            if nested:
                return nested
    if "name" in payload or "agent_name" in payload or "id" in payload or "agent_id" in payload:
        return payload
    return {}


def extract_comments(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("comments", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def post_id(post: Dict[str, Any]) -> Optional[str]:
    value = post.get("id") or (post.get("post") or {}).get("id")
    return str(value) if value is not None else None


def comment_id(comment: Dict[str, Any]) -> Optional[str]:
    value = comment.get("id") or (comment.get("comment") or {}).get("id")
    return str(value) if value is not None else None


def comment_parent_id(comment: Dict[str, Any]) -> Optional[str]:
    nested = comment.get("comment") if isinstance(comment.get("comment"), dict) else {}
    parent = (
        comment.get("parent_id")
        or comment.get("parentId")
        or comment.get("parent_comment_id")
        or comment.get("reply_to_id")
        or nested.get("parent_id")
        or nested.get("parentId")
        or nested.get("parent_comment_id")
        or nested.get("reply_to_id")
    )
    if parent is None:
        parent_obj = comment.get("parent")
        if isinstance(parent_obj, dict):
            parent = parent_obj.get("id")
    return str(parent) if parent is not None else None


def comment_author(comment: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    nested = comment.get("comment") if isinstance(comment.get("comment"), dict) else {}
    author = (
        comment.get("author")
        or comment.get("user")
        or comment.get("agent")
        or comment.get("owner")
        or nested.get("author")
        or nested.get("user")
        or nested.get("agent")
        or {}
    )
    aid = (
        author.get("id")
        or author.get("agent_id")
        or author.get("user_id")
        or comment.get("author_id")
        or comment.get("agent_id")
        or nested.get("author_id")
        or nested.get("agent_id")
    )
    aname = (
        author.get("name")
        or author.get("username")
        or author.get("agent_name")
        or comment.get("author_name")
        or comment.get("agent_name")
        or comment.get("created_by")
        or nested.get("author_name")
        or nested.get("agent_name")
        or nested.get("created_by")
    )
    return (str(aid) if aid is not None else None, str(aname) if aname is not None else None)


def comment_score(comment: Dict[str, Any]) -> int:
    for key in ("score", "vote_score", "upvotes", "likes"):
        value = comment.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def parse_created_at(comment: Dict[str, Any]) -> datetime:
    value = (
        comment.get("created_at")
        or comment.get("createdAt")
        or comment.get("created")
        or comment.get("timestamp")
    )
    text = normalize_str(value).strip()
    if not text:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def post_url(pid: str) -> str:
    return f"https://moltbook.com/post/{pid}"


@dataclass
class DuplicateCandidate:
    post_id: str
    parent_comment_id: Optional[str]
    reason: str
    keep_comment_id: str
    delete_comment_ids: List[str]


def load_state_post_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload.get("replied_post_ids", [])
    if not isinstance(items, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for value in items:
        text = normalize_str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def choose_keep_and_deletes(items: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    ranked = sorted(
        items,
        key=lambda c: (
            parse_created_at(c),
            normalize_str(comment_id(c)),
        ),
    )
    keep = comment_id(ranked[0])
    deletes = [normalize_str(comment_id(c)) for c in ranked[1:] if comment_id(c)]
    return normalize_str(keep), deletes


def normalize_content_fingerprint(text: Any) -> str:
    value = normalize_str(text).strip().lower()
    value = " ".join(value.split())
    return value


def load_env(dotenv_path: Path) -> None:
    if load_dotenv is None:
        return
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def gather_target_post_ids(
    client: MoltbookClient,
    agent_name: str,
    extra_post_ids: Sequence[str],
    state_path: Path,
    recent_limit: int,
) -> List[str]:
    post_ids: List[str] = []
    seen: set[str] = set()

    for pid in extra_post_ids:
        value = normalize_str(pid).strip()
        if value and value not in seen:
            seen.add(value)
            post_ids.append(value)

    for pid in load_state_post_ids(state_path):
        if pid not in seen:
            seen.add(pid)
            post_ids.append(pid)

    try:
        profile_payload = client.get_agent_profile(agent_name)
        recent_posts = extract_posts(profile_payload)[: max(1, recent_limit)]
        for post in recent_posts:
            pid = post_id(post)
            if pid and pid not in seen:
                seen.add(pid)
                post_ids.append(pid)
    except Exception as exc:
        print(f"Warning: could not fetch recent posts for {agent_name}: {exc}")

    return post_ids


def build_duplicate_candidates(
    client: MoltbookClient,
    my_name: Optional[str],
    my_id: Optional[str],
    post_ids: Sequence[str],
    comment_limit: int,
) -> Tuple[List[DuplicateCandidate], int]:
    candidates: List[DuplicateCandidate] = []
    scanned_comments = 0

    my_name_lc = normalize_str(my_name).strip().lower()
    my_id_lc = normalize_str(my_id).strip().lower()

    for pid in post_ids:
        try:
            payload = client.get_post_comments(pid, limit=comment_limit)
        except Exception as exc:
            print(f"Warning: comments fetch failed for post {pid}: {exc}")
            continue
        comments = extract_comments(payload)
        scanned_comments += len(comments)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        own_comments_in_post: List[Dict[str, Any]] = []
        for comment in comments:
            cid = comment_id(comment)
            if not cid:
                continue
            parent = comment_parent_id(comment)
            if not parent:
                continue
            author_id, author_name = comment_author(comment)
            author_id_lc = normalize_str(author_id).strip().lower()
            author_name_lc = normalize_str(author_name).strip().lower()
            is_me = bool(
                (my_id_lc and author_id_lc and author_id_lc == my_id_lc)
                or (my_name_lc and author_name_lc and author_name_lc == my_name_lc)
            )
            if not is_me:
                continue
            own_comments_in_post.append(comment)
            grouped.setdefault(parent, []).append(comment)

        for parent_id, replies in grouped.items():
            if len(replies) < 2:
                continue
            keep_id, delete_ids = choose_keep_and_deletes(replies)
            if not keep_id or not delete_ids:
                continue
            candidates.append(
                DuplicateCandidate(
                    post_id=pid,
                    parent_comment_id=parent_id,
                    reason="same_parent",
                    keep_comment_id=keep_id,
                    delete_comment_ids=delete_ids,
                )
            )

        # Secondary detection: same author, same post, near-identical content.
        # This catches duplicates when parent metadata is missing/inconsistent.
        by_fingerprint: Dict[str, List[Dict[str, Any]]] = {}
        for comment in own_comments_in_post:
            content = normalize_content_fingerprint(comment.get("content"))
            if len(content) < 40:
                continue
            by_fingerprint.setdefault(content, []).append(comment)

        for fingerprint, same_content_comments in by_fingerprint.items():
            if len(same_content_comments) < 2:
                continue
            # Keep this conservative: only treat as duplicate when created close in time.
            # This avoids deleting intentional repeated phrases across long periods.
            sorted_items = sorted(
                same_content_comments,
                key=lambda c: (
                    parse_created_at(c),
                    normalize_str(comment_id(c)),
                ),
            )
            first_ts = parse_created_at(sorted_items[0]).timestamp()
            last_ts = parse_created_at(sorted_items[-1]).timestamp()
            if (last_ts - first_ts) > (6 * 60 * 60):
                continue

            keep_id, delete_ids = choose_keep_and_deletes(sorted_items)
            if not keep_id or not delete_ids:
                continue
            key_signature = f"{pid}|content|{keep_id}|{','.join(delete_ids)}"
            already = False
            for existing in candidates:
                existing_signature = (
                    f"{existing.post_id}|{existing.reason}|{existing.keep_comment_id}|"
                    f"{','.join(existing.delete_comment_ids)}"
                )
                if existing_signature == key_signature:
                    already = True
                    break
            if already:
                continue
            candidates.append(
                DuplicateCandidate(
                    post_id=pid,
                    parent_comment_id=None,
                    reason="same_content",
                    keep_comment_id=keep_id,
                    delete_comment_ids=delete_ids,
                )
            )

    return candidates, scanned_comments


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    choice = input(f"{prompt} [y/N]: ").strip().lower()
    return choice in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and optionally delete duplicate reply-comments by this agent."
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete duplicates.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts.")
    parser.add_argument(
        "--state-path",
        default="memory/autonomy-state.json",
        help="Path to autonomy state JSON (default: memory/autonomy-state.json).",
    )
    parser.add_argument(
        "--recent-limit",
        type=int,
        default=30,
        help="How many recent own posts to scan from profile (default: 30).",
    )
    parser.add_argument(
        "--comment-limit",
        type=int,
        default=200,
        help="Max comments fetched per post (default: 200).",
    )
    parser.add_argument(
        "--post-id",
        action="append",
        default=[],
        help="Additional post ID to scan (can repeat).",
    )
    parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Path to dotenv file for API key loading (default: .env).",
    )
    args = parser.parse_args()

    load_env(Path(args.dotenv_path))

    client = MoltbookClient()
    me_payload = client.get_me()
    me = extract_me(me_payload)
    my_name = normalize_str(me.get("name") or me.get("agent_name")).strip() or None
    my_id = normalize_str(me.get("id") or me.get("agent_id")).strip() or None
    if not my_name and not my_id:
        print("Error: could not resolve current agent identity from /agents/me")
        return 1

    post_ids = gather_target_post_ids(
        client=client,
        agent_name=my_name or "",
        extra_post_ids=args.post_id,
        state_path=Path(args.state_path),
        recent_limit=args.recent_limit,
    )
    if not post_ids:
        print("No post IDs discovered for duplicate scan.")
        return 0

    duplicates, scanned_comments = build_duplicate_candidates(
        client=client,
        my_name=my_name,
        my_id=my_id,
        post_ids=post_ids,
        comment_limit=args.comment_limit,
    )
    print(
        f"Scan complete: posts={len(post_ids)} comments={scanned_comments} duplicate_groups={len(duplicates)}"
    )
    if not duplicates:
        return 0

    total_delete = sum(len(item.delete_comment_ids) for item in duplicates)
    for item in duplicates:
        print("")
        print(f"post: {item.post_id} ({post_url(item.post_id)})")
        print(f"reason: {item.reason}")
        if item.parent_comment_id:
            print(f"parent_comment_id: {item.parent_comment_id}")
        print(f"keep: {item.keep_comment_id}")
        print(f"delete: {', '.join(item.delete_comment_ids)}")

    if not args.apply:
        print("")
        print("Dry run only. Re-run with --apply to delete duplicates.")
        return 0

    if not confirm(
        f"Delete {total_delete} duplicate replies across {len(duplicates)} groups?",
        assume_yes=args.yes,
    ):
        print("Aborted.")
        return 0

    deleted = 0
    failed = 0
    for item in duplicates:
        for cid in item.delete_comment_ids:
            try:
                client.delete_comment(cid)
                deleted += 1
                print(f"Deleted comment: {cid}")
            except Exception as exc:
                failed += 1
                print(f"Failed delete comment {cid}: {exc}")

    print(f"Done. deleted={deleted} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
