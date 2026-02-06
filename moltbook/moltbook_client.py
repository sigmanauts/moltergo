import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests import exceptions as requests_exceptions


MOLTBOOK_BASE_URL = "https://www.moltbook.com/api/v1"
CREDENTIALS_PATH = Path.home() / ".config" / "moltbook" / "credentials.json"


class MoltbookAuthError(Exception):
    pass


@dataclass
class MoltbookCredentials:
    api_key: str
    agent_name: Optional[str] = None

    @classmethod
    def load(cls) -> "MoltbookCredentials":
        """Load credentials from env or ~/.config/moltbook/credentials.json.

        Priority:
        1. MOLTBOOK_API_KEY env var
        2. credentials.json file
        """
        api_key = os.getenv("MOLTBOOK_API_KEY")
        agent_name: Optional[str] = None

        if not api_key and CREDENTIALS_PATH.exists():
            with CREDENTIALS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            api_key = data.get("api_key")
            agent_name = data.get("agent_name")

        if not api_key:
            raise MoltbookAuthError(
                "Missing Moltbook API key. Set MOLTBOOK_API_KEY or create "
                f"{CREDENTIALS_PATH} with an 'api_key' field."
            )

        return cls(api_key=api_key, agent_name=agent_name)


class MoltbookClient:
    """Minimal Moltbook API client for posting and basic actions.

    SECURITY: This client only ever sends your API key to https://www.moltbook.com.
    Never modify it to talk to other domains with your key.
    """

    def __init__(self, credentials: Optional[MoltbookCredentials] = None):
        self.credentials = credentials or MoltbookCredentials.load()
        self.base_url = MOLTBOOK_BASE_URL.rstrip("/")

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
        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
            message = data.get("error") or data.get("hint") or resp.text
            raise RuntimeError(f"Moltbook error {resp.status_code}: {message}")

        return resp.json()


def load_heartbeat_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"lastMoltbookCheck": None}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_heartbeat_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


