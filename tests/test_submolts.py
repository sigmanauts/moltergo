import unittest

from moltbook.autonomy.submolts import (
    get_cached_submolt_meta,
    is_valid_submolt_name,
    parse_submolt_meta,
)


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class _Client:
    def __init__(self):
        self.calls = 0

    def list_submolts_public(self):
        self.calls += 1
        return {
            "submolts": [
                {
                    "name": "general",
                    "display_name": "General",
                    "description": "General chat",
                    "subscriber_count": 1000,
                    "last_activity_at": "2026-02-07T10:00:00Z",
                },
                {
                    "slug": "crypto",
                    "display_name": "Crypto",
                    "description": "Crypto topics",
                    "members": 500,
                    "updated_at": "2026-02-07T09:00:00Z",
                },
            ]
        }

    def list_submolts(self):
        raise AssertionError("authenticated fallback should not be used in this test")


class SubmoltTests(unittest.TestCase):
    def test_parse_submolt_meta_normalizes_fields(self):
        parsed = parse_submolt_meta(
            {
                "submolts": [
                    {
                        "slug": "crypto",
                        "display_name": "Crypto",
                        "description": "Crypto topics",
                        "members": 500,
                        "updated_at": "2026-02-07T09:00:00Z",
                    }
                ]
            }
        )
        self.assertIn("crypto", parsed)
        self.assertEqual(parsed["crypto"]["subscriber_count"], 500)
        self.assertEqual(parsed["crypto"]["display_name"], "Crypto")

    def test_get_cached_submolt_meta_uses_ttl_cache(self):
        client = _Client()
        cache = {}
        logger = _Logger()

        first = get_cached_submolt_meta(client=client, ttl_seconds=900, cache=cache, logger=logger)
        second = get_cached_submolt_meta(client=client, ttl_seconds=900, cache=cache, logger=logger)

        self.assertEqual(client.calls, 1)
        self.assertEqual(first, second)
        self.assertTrue(is_valid_submolt_name("general", first))
        self.assertTrue(is_valid_submolt_name("crypto", first))
        self.assertFalse(is_valid_submolt_name("does-not-exist", first))


if __name__ == "__main__":
    unittest.main()
