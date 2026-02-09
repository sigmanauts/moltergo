import json
import os
import tempfile
import json
import os
import tempfile
import unittest
from pathlib import Path
from pathlib import Path

from moltbook.autonomy.runner import (
    _unpublishable_reason,
    _zero_action_streak_from_state,
    _is_duplicate_post,
    _recent_post_fingerprints_from_journal,
)


class RunnerLogicTests(unittest.TestCase):
    def test_zero_action_streak_counts_trailing_zero_action_cycles(self):
        state = {
            "cycle_metrics_history": [
                {"cycle": 1, "actions": 2},
                {"cycle": 2, "actions": 0},
                {"cycle": 3, "actions": 0},
                {"cycle": 4, "actions": 0},
            ]
        }
        self.assertEqual(_zero_action_streak_from_state(state), 3)

    def test_zero_action_streak_resets_after_non_zero_action(self):
        state = {
            "cycle_metrics_history": [
                {"cycle": 1, "actions": 0},
                {"cycle": 2, "actions": 0},
                {"cycle": 3, "actions": 1},
                {"cycle": 4, "actions": 0},
            ]
        }
        self.assertEqual(_zero_action_streak_from_state(state), 1)

    def test_unpublishable_reason_detects_payload_wrappers(self):
        self.assertEqual(_unpublishable_reason("```yaml\ncomment: hi\n```"), "markdown_fence_payload")
        self.assertEqual(_unpublishable_reason("response_mode=comment should_respond=true"), "control_flag_payload")
        self.assertEqual(
            _unpublishable_reason("If you want, check my ongoing Ergo build threads/profile for details."),
            "self_promo_bridge_payload",
        )

    def test_duplicate_post_detection_by_fingerprint(self):
        state = {
            "recent_post_fingerprints": [],
            "recent_post_titles": [],
            "recent_topic_signatures": [],
        }
        title = "Ergo escrow for agents"
        content = "Use eUTXO escrow boxes to enforce deterministic settlement."
        dup, reason = _is_duplicate_post(title, content, state)
        self.assertFalse(dup)
        # Second call should be flagged once fingerprint is remembered.
        state["recent_post_fingerprints"].append(
            __import__("hashlib").sha1(
                "ergo escrow for agents\nuse eutxo escrow boxes to enforce deterministic settlement.".encode("utf-8")
            ).hexdigest()[:16]
        )
        dup, reason = _is_duplicate_post(title, content, state)
        self.assertTrue(dup)
        self.assertIn("duplicate", reason)

    def test_recent_post_fingerprints_from_journal(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as handle:
            handle.write(
                json.dumps(
                    {
                        "action_type": "post",
                        "title": "Ergo escrow for agents",
                        "content": "Use eUTXO escrow boxes to enforce deterministic settlement.",
                    }
                )
                + "\n"
            )
            path = handle.name
        try:
            fps, titles = _recent_post_fingerprints_from_journal(Path(path))
            self.assertTrue(fps)
            self.assertIn("Ergo escrow for agents", titles)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
