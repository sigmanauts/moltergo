import unittest

from moltbook.autonomy.content_policy import (
    build_badbot_warning_reply,
    is_overt_spam_comment,
    looks_irrelevant_noise_comment,
    looks_spammy_comment,
    is_strong_ergo_post,
    top_badbots,
)


class ContentPolicyTests(unittest.TestCase):
    def test_is_strong_ergo_post_requires_signal(self):
        strong, terms = is_strong_ergo_post(
            title="Using ErgoScript and eUTXO for escrow",
            content="Deterministic settlements with Sigma proofs.",
            submolt="general",
        )
        self.assertTrue(strong)
        self.assertIn("ergoscript", terms)

        weak, _ = is_strong_ergo_post(
            title="I like green energy",
            content="No chain mechanics here.",
            submolt="general",
        )
        self.assertFalse(weak)

    def test_badbot_warning_changes_with_strikes(self):
        self.assertIn("technical", build_badbot_warning_reply("SpamBot", 1).lower())
        self.assertIn("not promos", build_badbot_warning_reply("SpamBot", 2).lower())
        self.assertIn("only engage", build_badbot_warning_reply("SpamBot", 4).lower())

    def test_top_badbots_sorts_desc(self):
        board = top_badbots({"a": 1, "b": 4, "c": 2}, limit=2)
        self.assertEqual(board, [("b", 4), ("c", 2)])

    def test_verbose_ergo_comment_is_not_spam_or_noise(self):
        comment = (
            "The Ergo platform eUTXO model gives deterministic outcomes for agent settlement. "
            "You can wire ErgoScript constraints into dispute branches and verify commitments using Sigma proofs. "
            "A practical path is escrow lock, evidence hash commitment, and timeout branch with objective release rules."
        )
        self.assertFalse(looks_spammy_comment(comment))
        self.assertFalse(looks_irrelevant_noise_comment(comment))

    def test_technical_comment_with_links_is_not_spam(self):
        comment = (
            "One thing that helped us: split invoicing from payment and keep evidence hashes verifiable. "
            "Example write-up: https://www.moltbook.com/post/abc and code: https://github.com/org/repo"
        )
        self.assertFalse(looks_spammy_comment(comment))

    def test_overt_spam_marked(self):
        comment = "Huge airdrop now, dm me and connect wallet at https://scam.example"
        self.assertTrue(is_overt_spam_comment(comment))
        self.assertTrue(looks_spammy_comment(comment))

    def test_long_technical_airdrop_comment_is_not_overt_spam(self):
        comment = (
            "Your eUTXO escrow framing is useful for settlement safety. "
            "I am testing milestone dispute branches with ErgoScript and Sigma proofs in production-like flows. "
            "There is an airdrop mention in our app roadmap, but this comment is about counterparty verification and execution constraints."
        )
        self.assertFalse(is_overt_spam_comment(comment))


if __name__ == "__main__":
    unittest.main()
