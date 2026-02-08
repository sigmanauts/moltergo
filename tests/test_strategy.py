import types
import unittest

from moltbook.autonomy.generation_utils import sanitize_generated_title
from moltbook.autonomy.strategy import (
    _adaptive_draft_controls,
    _ensure_use_case_prompt_if_relevant,
    _post_relevance_score,
)


class StrategyAdaptiveControlsTests(unittest.TestCase):
    def _cfg(self):
        return types.SimpleNamespace(
            draft_shortlist_size=18,
            draft_signal_min_score=2,
            dynamic_shortlist_enabled=True,
            dynamic_shortlist_min=6,
            dynamic_shortlist_max=30,
        )

    def test_streak_recovery_mode_activates(self):
        cfg = self._cfg()
        history = []
        for _ in range(6):
            history.append(
                {
                    "drafted": 6,
                    "model_approved": 0,
                    "eligible_now": 12,
                    "actions": 0,
                    "skip_reasons": {"consecutive_declines_guard": 1},
                }
            )
        shortlist, signal, mode = _adaptive_draft_controls(cfg, {"cycle_metrics_history": history})
        self.assertEqual(mode, "streak_recovery")
        self.assertLessEqual(shortlist, 6)
        self.assertGreaterEqual(signal, 2)

    def test_relax_context_gate_mode_activates(self):
        cfg = self._cfg()
        history = []
        for _ in range(6):
            history.append(
                {
                    "drafted": 5,
                    "model_approved": 0,
                    "eligible_now": 10,
                    "actions": 1,
                    "skip_reasons": {"trend_context_mismatch": 2},
                }
            )
        shortlist, signal, mode = _adaptive_draft_controls(cfg, {"cycle_metrics_history": history})
        self.assertEqual(mode, "relax_context_gate")
        self.assertGreaterEqual(shortlist, cfg.draft_shortlist_size)
        self.assertLessEqual(signal, cfg.draft_signal_min_score)

    def test_conversion_collapse_mode_activates(self):
        cfg = self._cfg()
        history = []
        for _ in range(6):
            history.append(
                {
                    "drafted": 6,
                    "model_approved": 0,
                    "eligible_now": 12,
                    "actions": 0,
                    "skip_reasons": {"consecutive_declines_guard": 1},
                }
            )
        history.extend(
            [
                {
                    "drafted": 5,
                    "model_approved": 0,
                    "eligible_now": 10,
                    "actions": 0,
                    "skip_reasons": {},
                },
                {
                    "drafted": 5,
                    "model_approved": 0,
                    "eligible_now": 10,
                    "actions": 0,
                    "skip_reasons": {},
                },
            ]
        )
        shortlist, signal, mode = _adaptive_draft_controls(cfg, {"cycle_metrics_history": history})
        self.assertEqual(mode, "conversion_collapse")
        self.assertEqual(shortlist, cfg.dynamic_shortlist_min)
        self.assertGreaterEqual(signal, cfg.draft_signal_min_score + 1)


class GenerationTitleSanitizeTests(unittest.TestCase):
    def test_default_fallback_title_is_not_generic(self):
        self.assertEqual(sanitize_generated_title(""), "Ergo implementation question")

    def test_truncation_marker_removed(self):
        title = sanitize_generated_title("Ergo execution path [truncated]")
        self.assertNotIn("truncated", title.lower())


class StrategyUseCasePromptTests(unittest.TestCase):
    def test_use_case_prompt_not_appended_when_question_exists(self):
        content = "Ergo eUTXO can reduce coordination failures in settlement. What dispute rule should be mandatory?"
        post = {
            "title": "Agent coordination and settlement",
            "content": "We need better orchestration and execution guarantees.",
            "submolt": "general",
        }
        out = _ensure_use_case_prompt_if_relevant(content, post)
        self.assertEqual(out, content)

    def test_use_case_prompt_appends_contextual_suffix_when_missing_question(self):
        content = "Escrow flows need deterministic conditions so counterparties cannot stall execution."
        post = {
            "title": "Escrow dispute design",
            "content": "Counterparty settlement and orchestration",
            "submolt": "general",
        }
        out = _ensure_use_case_prompt_if_relevant(content, post)
        self.assertIn("?", out)
        self.assertIn("ergoscript", out.lower())


class StrategyRelevanceScoreTests(unittest.TestCase):
    def test_negative_lift_terms_penalize_generic_market_posts(self):
        generic_post = {
            "title": "Market coordination and economic discussion",
            "content": "Practical market terms and comparative analysis.",
            "submolt": "general",
            "comment_count": 5,
            "upvotes": 8,
        }
        ergo_post = {
            "title": "Market execution with Ergo eUTXO settlement",
            "content": "Use ErgoScript escrow with proof hashes and timeout rules.",
            "submolt": "general",
            "comment_count": 5,
            "upvotes": 8,
        }
        signal_terms = ["market", "execution", "comparison", "practical"]
        winning_terms = ["execution"]
        losing_terms = ["comparison", "market"]
        term_lift_map = {
            "comparison": -0.7,
            "market": -0.6,
            "execution": 0.4,
        }
        best_submolts = {"general": 10.0}

        generic_score = _post_relevance_score(
            generic_post,
            signal_terms=signal_terms,
            winning_terms=winning_terms,
            losing_terms=losing_terms,
            term_lift_map=term_lift_map,
            best_submolt_scores=best_submolts,
        )
        ergo_score = _post_relevance_score(
            ergo_post,
            signal_terms=signal_terms,
            winning_terms=winning_terms,
            losing_terms=losing_terms,
            term_lift_map=term_lift_map,
            best_submolt_scores=best_submolts,
        )
        self.assertGreater(ergo_score, generic_score)


if __name__ == "__main__":
    unittest.main()
