import unittest

from moltbook.autonomy.runner import _unpublishable_reason, _zero_action_streak_from_state


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


if __name__ == "__main__":
    unittest.main()
