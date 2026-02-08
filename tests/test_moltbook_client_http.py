import unittest
from unittest.mock import patch

from moltbook.moltbook_client import MoltbookClient, MoltbookCredentials


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class MoltbookClientHttpTests(unittest.TestCase):
    @patch("moltbook.moltbook_client.requests.get")
    def test_get_posts_by_sorts_hits_posts_endpoint_per_sort(self, mock_get):
        def _fake_get(url, headers=None, params=None, timeout=30):  # noqa: ARG001
            sort = (params or {}).get("sort", "new")
            return _Resp(
                status_code=200,
                payload={"posts": [{"id": f"{sort}-1", "title": f"{sort} title"}]},
            )

        mock_get.side_effect = _fake_get
        client = MoltbookClient(credentials=MoltbookCredentials(api_key="k"))
        data = client.get_posts_by_sorts(sorts=["hot", "new", "hot"], limit=7)

        self.assertSetEqual(set(data.keys()), {"hot", "new"})
        self.assertEqual(mock_get.call_count, 2)
        called_urls = [call.kwargs.get("url") or call.args[0] for call in mock_get.call_args_list]
        for url in called_urls:
            self.assertTrue(url.startswith("https://www.moltbook.com/api/v1/posts"))

    @patch("moltbook.moltbook_client.requests.get")
    def test_list_submolts_public_uses_public_endpoint_without_auth_header(self, mock_get):
        def _fake_get(url, timeout=30, **kwargs):  # noqa: ARG001
            # Public endpoint call should not pass Authorization headers.
            self.assertNotIn("headers", kwargs)
            self.assertTrue(url.endswith("/submolts"))
            return _Resp(
                status_code=200,
                payload={"submolts": [{"name": "general", "subscriber_count": 100}]},
            )

        mock_get.side_effect = _fake_get
        client = MoltbookClient(credentials=MoltbookCredentials(api_key="k"))
        payload = client.list_submolts_public()

        self.assertIn("submolts", payload)
        self.assertEqual(len(payload["submolts"]), 1)


if __name__ == "__main__":
    unittest.main()
