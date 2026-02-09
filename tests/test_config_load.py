import unittest

from moltbook.autonomy.config import load_config


class ConfigLoadTests(unittest.TestCase):
    def test_load_config_returns_config(self) -> None:
        cfg = load_config()
        self.assertIsNotNone(cfg)

