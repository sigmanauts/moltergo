import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests import exceptions as requests_exceptions


MOLTBOOK_BASE_URL = "https://www.moltbook.com/api/v1"
MOLTBOOK_BASE_ENV = "MOLTBOOK_API_BASE"
CREDENTIALS_PATH = Path.home() / ".config" / "moltbook" / "credentials.json"
_MOLTBOOK_ALLOWED_PREFIX = "https://www.moltbook.com/api/v1"


def normalize_sort(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"hot", "new", "rising", "top"}:
        return text
    return "new"


class MoltbookAuthError(Exception):
    pass


@dataclass
class MoltbookCredentials:
    api_key: str
    agent_name: Optional[str] = None
    source: str = "unknown"

    @classmethod
    def load(cls) -> "MoltbookCredentials":
        """Load credentials from env or ~/.config/moltbook/credentials.json.

        Priority:
        1. MOLTBOOK_API_KEY env var
        2. credentials.json file
        """
        if os.getenv("MOLTBOOK_SKIP_AUTH_VALIDATION", "").strip().lower() in {"1", "true", "yes"}:
            return cls(
                api_key=os.getenv("MOLTBOOK_API_KEY", ""),
                agent_name=os.getenv("MOLTBOOK_AGENT_NAME"),
                source=os.getenv("MOLTBOOK_API_KEY_SOURCE", "env:MOLTBOOK_API_KEY"),
            )
        api_key = os.getenv("MOLTBOOK_API_KEY")
        agent_name: Optional[str] = None
        source = os.getenv("MOLTBOOK_API_KEY_SOURCE", "env:MOLTBOOK_API_KEY")

        if not api_key and CREDENTIALS_PATH.exists():
            with CREDENTIALS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            api_key = data.get("api_key")
            agent_name = data.get("agent_name")
            source = f"file:{CREDENTIALS_PATH}"

        if api_key is not None:
            api_key = str(api_key).strip()

        if not api_key:
            raise MoltbookAuthError(
                "Missing Moltbook API key. Set MOLTBOOK_API_KEY or create "
                f"{CREDENTIALS_PATH} with an 'api_key' field."
            )

        return cls(api_key=api_key, agent_name=agent_name, source=source)

    @classmethod
    def load_from_file(cls) -> Optional["MoltbookCredentials"]:
        if not CREDENTIALS_PATH.exists():
            return None
        with CREDENTIALS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        api_key = str(data.get("api_key") or "").strip()
        if not api_key:
            return None
        agent_name = data.get("agent_name")
        return cls(api_key=api_key, agent_name=agent_name, source=f"file:{CREDENTIALS_PATH}")


class MoltbookClient:
    """Minimal Moltbook API client for posting and basic actions.

    SECURITY: This client only ever sends your API key to https://www.moltbook.com.
    Never modify it to talk to other domains with your key.
    """

    def __init__(self, credentials: Optional[MoltbookCredentials] = None):
        self.credentials = credentials or MoltbookCredentials.load()
        env_base = os.getenv(MOLTBOOK_BASE_ENV)
        self.base_url = self._normalize_base_url(env_base or MOLTBOOK_BASE_URL)
        self._search_endpoint: Optional[str] = None
        self._search_endpoint_checked = False

    def _normalize_base_url(self, raw: str) -> str:
        candidate = str(raw).strip().rstrip("/")
        if candidate.startswith(_MOLTBOOK_ALLOWED_PREFIX):
            return candidate
        # Enforce the official API host so auth headers are never sent elsewhere.
        return MOLTBOOK_BASE_URL

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credentials.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.base_url}/{path}"

    def get_me(self) -> Dict[str, Any]:
        try:
            resp = requests.get(
                self._url("agents/me"),
                headers=self._headers,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook for /agents/me. "
                "Check https://www.moltbook.com is reachable from your network "
                "and try again."
            ) from e

        # Provide a clearer message when the agent is not yet claimed
        if resp.status_code == 401:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()

            error = data.get("error") or "Unauthorized"
            hint = data.get("hint") or "Your agent may not be claimed yet."
            raise RuntimeError(f"Moltbook 401: {error}. {hint}")

        resp.raise_for_status()
        return resp.json()

    def get_claim_status(self) -> str:
        try:
            resp = requests.get(
                self._url("agents/status"),
                headers=self._headers,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook for /agents/status."
            ) from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        data = resp.json()
        status = data.get("status")
        if isinstance(status, str):
            return status
        return "unknown"

    def get_agent_profile(self, agent_name: str) -> Dict[str, Any]:
        if not agent_name.strip():
            raise ValueError("agent_name must be provided.")
        try:
            resp = requests.get(
                self._url("agents/profile"),
                headers=self._headers,
                params={"name": agent_name},
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook for /agents/profile."
            ) from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()

    def create_post(
        self,
        submolt: str,
        title: str,
        content: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not content and not url:
            raise ValueError("Either 'content' or 'url' must be provided for a post.")

        payload: Dict[str, Any] = {
            "submolt": submolt,
            "title": title,
        }
        if content:
            payload["content"] = content
        if url:
            payload["url"] = url

        try:
            resp = requests.post(
                self._url("posts"),
                headers=self._headers,
                data=json.dumps(payload),
                timeout=60,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while creating a post on Moltbook. "
                "Your network or Moltbook may be slow or temporarily unavailable."
            ) from e

        # Let caller see rate limits / errors clearly
        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()

    def get_feed(self, limit: int = 30) -> Dict[str, Any]:
        try:
            resp = requests.get(
                self._url("feed"),
                headers=self._headers,
                params={"limit": limit},
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook for /feed. "
                "Check https://www.moltbook.com is reachable from your network "
                "and try again."
            ) from e

        if resp.status_code >= 400:
            if resp.status_code in {401, 403}:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or "Authentication required"
                raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()

    def get_posts(self, sort: str = "new", limit: int = 30, submolt: Optional[str] = None) -> Dict[str, Any]:
        sort = sort.strip().lower()
        if sort not in {"hot", "new", "rising", "top"}:
            sort = "new"
        params: Dict[str, Any] = {"sort": sort, "limit": limit}
        if submolt:
            params["submolt"] = submolt
        try:
            resp = requests.get(
                self._url("posts"),
                headers=self._headers,
                params=params,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook for /posts."
            ) from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()

    def get_posts_by_sorts(
        self,
        sorts: list[str],
        limit: int = 30,
        submolt: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for sort in sorts:
            key = normalize_sort(sort)
            if key in out:
                continue
            out[key] = self.get_posts(sort=key, limit=limit, submolt=submolt)
        return out

    def get_post(self, post_id: str) -> Dict[str, Any]:
        if not post_id.strip():
            raise ValueError("post_id must be provided")
        try:
            resp = requests.get(
                self._url(f"posts/{post_id}"),
                headers=self._headers,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                f"Timed out while contacting Moltbook for /posts/{post_id}."
            ) from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or data.get("message") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()

    def list_submolts(self) -> Dict[str, Any]:
        try:
            resp = requests.get(
                self._url("submolts"),
                headers=self._headers,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook for /submolts."
            ) from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()

    def list_submolts_public(self) -> Dict[str, Any]:
        try:
            resp = requests.get(
                self._url("submolts"),
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook for public /submolts."
            ) from e

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or data.get("message") or resp.text
            raise RuntimeError(f"Moltbook public submolts error {resp.status_code}: {message}")

        return resp.json()

    def subscribe_submolt(self, name: str) -> Dict[str, Any]:
        if not name.strip():
            raise ValueError("submolt name must be provided")
        try:
            resp = requests.post(
                self._url(f"submolts/{name}/subscribe"),
                headers=self._headers,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                f"Timed out while subscribing to submolt '{name}'."
            ) from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or data.get("message") or resp.text
            raise RuntimeError(f"Moltbook subscribe error {resp.status_code}: {message}")

        try:
            return resp.json()
        except Exception:
            return {"ok": True}

    def follow_agent(self, agent_name: str) -> Dict[str, Any]:
        name = agent_name.strip()
        if not name:
            raise ValueError("agent_name must be provided")
        try:
            resp = requests.post(
                self._url(f"agents/{name}/follow"),
                headers=self._headers,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(f"Timed out while following agent '{name}'.") from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or data.get("message") or resp.text
            raise RuntimeError(f"Moltbook follow error {resp.status_code}: {message}")

        try:
            return resp.json()
        except Exception:
            return {"ok": True}

    def get_submolt_feed(self, name: str, sort: str = "new", limit: int = 30) -> Dict[str, Any]:
        if not name.strip():
            raise ValueError("submolt name must be provided")
        candidates = [
            (f"submolts/{name}/feed", {"sort": sort, "limit": limit}),
            ("posts", {"submolt": name, "sort": sort, "limit": limit}),
        ]
        last_error: Optional[str] = None
        for path, params in candidates:
            try:
                resp = requests.get(
                    self._url(path),
                    headers=self._headers,
                    params=params,
                    timeout=30,
                )
            except requests_exceptions.Timeout as e:
                raise RuntimeError(
                    f"Timed out while loading feed for submolt '{name}'."
                ) from e

            if resp.status_code == 404:
                last_error = f"{path}: not found"
                continue
            if resp.status_code in {401, 403}:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or "Authentication required"
                raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")
            if resp.status_code >= 400:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or data.get("message") or resp.text
                last_error = f"{path}: {message}"
                continue

            return resp.json()

        raise RuntimeError(f"Moltbook submolt feed unavailable for '{name}'. Last error: {last_error}")

    def search_posts(self, query: str, limit: int = 20, search_type: str = "posts") -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("Search query must be provided.")

        if not self._search_endpoint_checked:
            self._discover_search_endpoint()

        if not self._search_endpoint:
            raise RuntimeError("Moltbook search endpoint unavailable.")

        params = {"q": query, "limit": limit, "type": search_type}
        try:
            resp = requests.get(
                self._url(self._search_endpoint),
                headers=self._headers,
                params=params,
                timeout=30,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while contacting Moltbook search endpoint. "
                "Check https://www.moltbook.com is reachable from your network "
                "and try again."
            ) from e

        if resp.status_code in {401, 403}:
            try:
                data = resp.json()
            except Exception:
                data = {}
            message = data.get("error") or data.get("hint") or "Authentication required"
            raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

        if resp.status_code == 404:
            # Endpoint may have changed; rediscover once.
            self._search_endpoint = None
            self._search_endpoint_checked = False
            self._discover_search_endpoint()
            if not self._search_endpoint:
                raise RuntimeError("Moltbook search endpoint unavailable.")
            return self.search_posts(query=query, limit=limit)

        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook search error {resp.status_code}: {message}")

        return resp.json()

    def _discover_search_endpoint(self) -> None:
        self._search_endpoint_checked = True
        candidates = ("search", "posts/search")
        last_error: Optional[str] = None

        for path in candidates:
            try:
                resp = requests.get(
                    self._url(path),
                    headers=self._headers,
                    params={"q": "ergo", "limit": 1, "type": "posts"},
                    timeout=30,
                )
            except requests_exceptions.Timeout as e:
                raise RuntimeError(
                    "Timed out while probing Moltbook search endpoint."
                ) from e

            if resp.status_code == 404:
                last_error = f"{path}: not found"
                continue

            if resp.status_code in {401, 403}:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or "Authentication required"
                raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

            if resp.status_code >= 400:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or data.get("message") or resp.text
                last_error = f"{path}: {message}"
                continue

            self._search_endpoint = path
            return

        self._search_endpoint = None
        raise RuntimeError(f"Moltbook search endpoint unavailable. Last error: {last_error}")

    def create_comment(self, post_id: str, content: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        if not content:
            raise ValueError("Comment content must be provided.")

        payload = {"content": content}
        if parent_id:
            payload["parent_id"] = parent_id

        try:
            resp = requests.post(
                self._url(f"posts/{post_id}/comments"),
                headers=self._headers,
                data=json.dumps(payload),
                timeout=60,
            )
        except requests_exceptions.Timeout as e:
            raise RuntimeError(
                "Timed out while creating a comment on Moltbook. "
                "Your network or Moltbook may be slow or temporarily unavailable."
            ) from e

        if resp.status_code >= 400:
            if resp.status_code in {401, 403}:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or "Authentication required"
                raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()

    def delete_comment(self, comment_id: str) -> Dict[str, Any]:
        if not comment_id:
            raise ValueError("comment_id must be provided.")

        candidates = [
            f"comments/{comment_id}",
            f"posts/comments/{comment_id}",
        ]
        last_error: Optional[str] = None

        for path in candidates:
            try:
                resp = requests.delete(
                    self._url(path),
                    headers=self._headers,
                    timeout=30,
                )
            except requests_exceptions.Timeout as e:
                raise RuntimeError(
                    "Timed out while deleting a comment on Moltbook."
                ) from e

            if resp.status_code == 404:
                last_error = f"{path}: not found"
                continue

            if resp.status_code in {401, 403}:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or "Authentication required"
                raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

            if resp.status_code >= 400:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or data.get("message") or resp.text
                last_error = f"{path}: {message}"
                continue

            try:
                return resp.json()
            except Exception:
                return {"ok": True}

        raise RuntimeError(f"Moltbook comment delete failed for '{comment_id}'. Last error: {last_error}")

    def get_post_comments(self, post_id: str, limit: int = 20) -> Dict[str, Any]:
        candidates = [
            (f"posts/{post_id}/comments", {"limit": limit}),
            ("comments", {"post_id": post_id, "limit": limit}),
            ("comments/search", {"post_id": post_id, "limit": limit}),
        ]
        last_error: Optional[str] = None

        for path, params in candidates:
            try:
                resp = requests.get(
                    self._url(path),
                    headers=self._headers,
                    params=params,
                    timeout=30,
                )
            except requests_exceptions.Timeout as e:
                raise RuntimeError(
                    "Timed out while fetching comments from Moltbook."
                ) from e

            if resp.status_code == 404:
                last_error = f"{path}: not found"
                continue

            if resp.status_code in {401, 403}:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or "Authentication required"
                raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

            if resp.status_code >= 400:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or data.get("message") or resp.text
                last_error = f"{path}: {message}"
                continue

            return resp.json()

        raise RuntimeError(f"Moltbook comments unavailable for post '{post_id}'. Last error: {last_error}")

    def vote_post(self, post_id: str, vote_action: str) -> Dict[str, Any]:
        return self._vote_resource(
            resource_path=f"posts/{post_id}",
            resource_name="post",
            resource_id=post_id,
            vote_action=vote_action,
        )

    def vote_comment(self, comment_id: str, vote_action: str) -> Dict[str, Any]:
        return self._vote_resource(
            resource_path=f"comments/{comment_id}",
            resource_name="comment",
            resource_id=comment_id,
            vote_action=vote_action,
        )

    def _vote_resource(
        self,
        resource_path: str,
        resource_name: str,
        resource_id: str,
        vote_action: str,
    ) -> Dict[str, Any]:
        action = vote_action.strip().lower()
        if action not in {"upvote", "downvote"}:
            raise ValueError("vote_action must be 'upvote' or 'downvote'.")

        vote_value = "up" if action == "upvote" else "down"
        candidates = [
            (f"{resource_path}/{action}", None),
            (f"{resource_path}/vote", {"vote": vote_value}),
            (f"{resource_path}/votes", {"vote": vote_value}),
        ]
        last_error: Optional[str] = None

        for path, payload in candidates:
            try:
                resp = requests.post(
                    self._url(path),
                    headers=self._headers,
                    data=json.dumps(payload) if payload is not None else None,
                    timeout=30,
                )
            except requests_exceptions.Timeout as e:
                raise RuntimeError(
                    f"Timed out while sending {action} for {resource_name}."
                ) from e

            if resp.status_code == 404:
                last_error = f"{path}: not found"
                continue

            if resp.status_code in {401, 403}:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                message = data.get("error") or data.get("hint") or "Authentication required"
                raise MoltbookAuthError(f"Moltbook auth error {resp.status_code}: {message}")

            if resp.status_code >= 400:
                try:
                    data = resp.json()
                except Exception:
                    resp.raise_for_status()
                message = data.get("error") or data.get("hint") or resp.text
                last_error = f"{path}: {message}"
                continue

            try:
                return resp.json()
            except Exception:
                return {"ok": True}

        raise RuntimeError(
            f"Moltbook vote failed for {resource_name} '{resource_id}'. Last error: {last_error}"
        )


def load_heartbeat_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"lastMoltbookCheck": None}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_heartbeat_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
