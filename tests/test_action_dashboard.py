import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from moltbook.autonomy.action_dashboard import refresh_action_dashboard
from moltbook.autonomy.action_journal import append_action_journal


class ActionDashboardTests(unittest.TestCase):
    def test_refresh_dashboard_renders_recent_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "action-journal.jsonl"
            append_action_journal(
                journal_path,
                action_type="post",
                target_post_id="post-1",
                submolt="general",
                title="First post",
                content="Post body",
                url="https://moltbook.com/post/post-1",
                reference={"post_title": "Seed thread"},
                meta={"source": "test"},
            )
            append_action_journal(
                journal_path,
                action_type="comment",
                target_post_id="post-2",
                submolt="crypto",
                title="Reply title",
                content="Reply body",
                parent_comment_id="comment-1",
                url="https://moltbook.com/post/post-2",
                reference={
                    "post_title": "Target thread",
                    "post_content": "Target content",
                    "comment_content": "Original comment",
                },
                meta={"source": "test"},
            )
            dashboard_path = refresh_action_dashboard(journal_path)
            self.assertIsNotNone(dashboard_path)
            html = Path(dashboard_path).read_text(encoding="utf-8")
            self.assertIn("Moltergo Live Action Dashboard", html)
            self.assertIn("https://moltbook.com/post/post-1", html)
            self.assertIn("https://moltbook.com/post/post-2", html)
            self.assertIn("First post", html)
            self.assertIn("Reply title", html)
            self.assertIn("Target thread", html)

    def test_append_action_journal_writes_dashboard_to_configured_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "journal.jsonl"
            dashboard_path = Path(tmp) / "live.html"
            with mock.patch.dict(
                os.environ,
                {
                    "MOLTBOOK_ACTION_DASHBOARD_ENABLED": "1",
                    "MOLTBOOK_ACTION_DASHBOARD_PATH": str(dashboard_path),
                    "MOLTBOOK_ACTION_DASHBOARD_MAX_ENTRIES": "50",
                    "MOLTBOOK_ACTION_DASHBOARD_REFRESH_SECONDS": "3",
                },
                clear=False,
            ):
                append_action_journal(
                    journal_path,
                    action_type="comment",
                    target_post_id="abc-123",
                    submolt="general",
                    title="title with <unsafe>",
                    content="body with <script>alert(1)</script>",
                    url="https://moltbook.com/post/abc-123",
                    reference={"post_title": "Source"},
                )
            self.assertTrue(dashboard_path.exists())
            html = dashboard_path.read_text(encoding="utf-8")
            self.assertIn("abc-123", html)
            self.assertIn("Auto-refresh: 3s", html)
            self.assertNotIn("<script>alert(1)</script>", html)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)


if __name__ == "__main__":
    unittest.main()

