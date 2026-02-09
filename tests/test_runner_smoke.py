import os
import unittest
import warnings


class RunnerSmokeTests(unittest.TestCase):
    def test_runner_startup_smoke(self) -> None:
        os.environ["MOLTBOOK_DRY_RUN"] = "1"
        os.environ["MOLTBOOK_MAX_CYCLES"] = "1"
        os.environ["MOLTBOOK_CONFIRM_ACTIONS"] = "0"
        os.environ["MOLTBOOK_LOG_LEVEL"] = "INFO"
        os.environ["MOLTBOOK_API_KEY"] = "moltbook_test_key"
        os.environ["MOLTBOOK_SKIP_AUTH_VALIDATION"] = "1"
        os.environ["MOLTBOOK_PROACTIVE_POSTING_ENABLED"] = "0"
        os.environ["MOLTBOOK_OLLAMA_MODEL"] = ""
        os.environ["MOLTBOOK_STARTUP_REPLY_SCAN_ENABLED"] = "0"
        os.environ["MOLTBOOK_REPLY_SCAN_INTERVAL_CYCLES"] = "0"
        os.environ["MOLTBOOK_IDLE_POLL_SECONDS"] = "1"
        from moltbook.autonomy.runner import run_loop
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            run_loop()
