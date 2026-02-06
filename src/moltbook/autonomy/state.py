import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_date_str() -> str:
    return utc_now().date().isoformat()


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "seen_post_ids": [],
            "replied_post_ids": [],
            "voted_post_ids": [],
            "seen_comment_ids": [],
            "my_comment_ids": [],
            "voted_comment_ids": [],
            "replied_to_comment_ids": [],
            "last_action_ts": None,
            "last_post_action_ts": None,
            "last_comment_action_ts": None,
            "comment_action_timestamps": [],
            "per_author_last_reply": {},
            "daily_post_count": 0,
            "daily_comment_count": 0,
            "last_daily_reset": utc_date_str(),
            "last_proactive_post_attempt_ts": None,
            "pending_actions": [],
            "approved_submolts": [],
            "dismissed_submolts": [],
        }
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    # Backward compatibility with older state files.
    if "last_post_action_ts" not in state:
        state["last_post_action_ts"] = state.get("last_action_ts")
    if "last_comment_action_ts" not in state:
        state["last_comment_action_ts"] = state.get("last_action_ts")
    if "comment_action_timestamps" not in state:
        state["comment_action_timestamps"] = []
    if "seen_comment_ids" not in state:
        state["seen_comment_ids"] = []
    if "replied_post_ids" not in state:
        state["replied_post_ids"] = []
    if "voted_post_ids" not in state:
        state["voted_post_ids"] = []
    if "my_comment_ids" not in state:
        state["my_comment_ids"] = []
    if "pending_actions" not in state:
        state["pending_actions"] = []
    if "last_proactive_post_attempt_ts" not in state:
        state["last_proactive_post_attempt_ts"] = None
    if "approved_submolts" not in state:
        state["approved_submolts"] = []
    if "dismissed_submolts" not in state:
        state["dismissed_submolts"] = []
    if "voted_comment_ids" not in state:
        state["voted_comment_ids"] = []
    if "replied_to_comment_ids" not in state:
        state["replied_to_comment_ids"] = []
    return state


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def reset_daily_if_needed(state: Dict[str, Any]) -> None:
    today = utc_date_str()
    if state.get("last_daily_reset") != today:
        state["daily_post_count"] = 0
        state["daily_comment_count"] = 0
        state["last_daily_reset"] = today
