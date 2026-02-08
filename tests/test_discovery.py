import types
import unittest
from datetime import datetime, timezone

from moltbook.autonomy.discovery import discover_posts


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None


class _Client:
    def get_feed(self, limit=30):
        return {"posts": [{"id": "1", "title": "A"}, {"id": "2", "title": "B"}]}

    def get_posts(self, sort="new", limit=30):
        return {"posts": [{"id": "2", "title": "B dup"}, {"id": "3", "title": "C"}]}

    def get_submolt_feed(self, name, sort="new", limit=30):
        return {"posts": []}

    def search_posts(self, query, limit=20, search_type="posts"):
        return {"posts": []}


class _ClientWithSorts:
    def get_feed(self, limit=30):
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "posts": [
                {"id": "a", "title": "Feed only", "created_at": now_iso},
                {"id": "b", "title": "Shared from feed", "created_at": now_iso},
            ]
        }

    def get_posts_by_sorts(self, sorts, limit=30):
        now_iso = datetime.now(timezone.utc).isoformat()
        out = {}
        for sort in sorts:
            if sort == "hot":
                out[sort] = {
                    "posts": [
                        {"id": "b", "title": "Shared from hot", "created_at": now_iso},
                        {"id": "c", "title": "Hot only", "created_at": now_iso},
                    ]
                }
            else:
                out[sort] = {"posts": [{"id": "d", "title": "New only", "created_at": now_iso}]}
        return out

    def get_posts(self, sort="new", limit=30):
        return {"posts": []}

    def get_submolt_feed(self, name, sort="new", limit=30):
        return {"posts": []}

    def search_posts(self, query, limit=20, search_type="posts"):
        return {"posts": []}


class DiscoverPostsTests(unittest.TestCase):
    def test_discover_posts_works_without_explicit_post_id_fn(self):
        cfg = types.SimpleNamespace(
            discovery_mode="search",
            search_batch_size=4,
            mission_queries=[],
            search_limit=10,
            search_retry_after_failure_cycles=4,
            feed_limit=10,
            posts_sort="new",
            posts_limit=10,
            target_submolts=[],
        )
        posts, sources = discover_posts(
            client=_Client(),
            cfg=cfg,
            logger=_Logger(),
            keywords=["ergo"],
            iteration=1,
            search_state={"retry_cycle": 1, "keyword_cursor": 0},
        )
        ids = [p.get("id") for p in posts]
        self.assertEqual(len(ids), 3)
        self.assertCountEqual(ids, ["1", "2", "3"])
        self.assertIn("posts", sources)
        self.assertIn("feed", sources)

    def test_discover_posts_merges_sources_and_sets_fast_lane(self):
        cfg = types.SimpleNamespace(
            discovery_mode="search",
            search_batch_size=4,
            mission_queries=[],
            search_limit=10,
            search_retry_after_failure_cycles=4,
            feed_limit=10,
            posts_sort="new",
            posts_limit=10,
            feed_sources=["hot", "new"],
            target_submolts=[],
            early_comment_window_seconds=3600,
        )
        posts, _ = discover_posts(
            client=_ClientWithSorts(),
            cfg=cfg,
            logger=_Logger(),
            keywords=["ergo"],
            iteration=1,
            search_state={"retry_cycle": 1, "keyword_cursor": 0},
        )
        ids = {p.get("id"): p for p in posts}
        self.assertSetEqual(set(ids.keys()), {"a", "b", "c", "d"})
        self.assertIn("feed", ids["b"].get("__feed_sources", []))
        self.assertIn("hot", ids["b"].get("__feed_sources", []))
        # Hot-source posts should be fast-lane comment candidates.
        self.assertTrue(bool(ids["b"].get("__fast_lane_comment")))


if __name__ == "__main__":
    unittest.main()
