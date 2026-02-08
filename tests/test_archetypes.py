import unittest

from moltbook.autonomy.archetypes import validate_archetype_template


class ArchetypeTemplateTests(unittest.TestCase):
    def test_build_log_template_validator_passes_when_sections_exist(self):
        template = """
Goal: ship escrow flow
Steps: 1) lock 2) verify 3) release
What broke: signer mismatch
Result: payout stabilized
Ergo mechanism framing: eUTXO branch constraints with ErgoScript guards
Question: which timeout window would you use?
"""
        ok, missing = validate_archetype_template("build_log", template)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_validator_reports_missing_sections(self):
        template = "Goal: one line only\nResult: none"
        ok, missing = validate_archetype_template("build_log", template)
        self.assertFalse(ok)
        self.assertIn("steps", missing)
        self.assertIn("question", missing)


if __name__ == "__main__":
    unittest.main()
