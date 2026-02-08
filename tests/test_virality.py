import unittest
from datetime import datetime, timedelta, timezone

from moltbook.virality import score_post_candidate


class ViralityTests(unittest.TestCase):
    def test_more_upvotes_increases_score(self):
        now = datetime.now(timezone.utc)
        base = {
            "id": "p1",
            "title": "Ergo eUTXO escrow path",
            "content": "Deterministic release rules for agent settlement",
            "created_at": now.isoformat(),
            "__feed_sources": ["hot"],
        }
        low = dict(base, upvotes=1, comment_count=1)
        high = dict(base, upvotes=25, comment_count=1)
        history = {"mode": "comment", "active_keywords": ["ergo", "eutxo"]}

        low_score = score_post_candidate(low, submolt_meta={}, now=now, history=history)
        high_score = score_post_candidate(high, submolt_meta={}, now=now, history=history)

        self.assertGreater(high_score, low_score)

    def test_older_post_decreases_score(self):
        now = datetime.now(timezone.utc)
        newer = {
            "id": "n",
            "title": "ErgoScript for agent payouts",
            "content": "Use eUTXO branches for dispute control",
            "created_at": now.isoformat(),
            "__feed_sources": ["hot"],
            "upvotes": 10,
            "comment_count": 3,
        }
        older = dict(newer)
        older["id"] = "o"
        older["created_at"] = (now - timedelta(hours=8)).isoformat()

        history = {
            "mode": "comment",
            "active_keywords": ["ergo", "ergoscript"],
            "recency_halflife_minutes": 60,
        }

        newer_score = score_post_candidate(newer, submolt_meta={}, now=now, history=history)
        older_score = score_post_candidate(older, submolt_meta={}, now=now, history=history)

        self.assertGreater(newer_score, older_score)


if __name__ == "__main__":
    unittest.main()
