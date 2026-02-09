import unittest

from moltbook.autonomy.drafting import (
    _parse_json_object_lenient,
    _sanitize_generated_content_text,
    sanitize_publish_content,
)


class DraftingSanitizeTests(unittest.TestCase):
    def test_reply_triage_non_json_extracts_reply_only(self):
        raw = """
Assessment: useful
Action: upvote
Reply (2-4 sentences): "Ergo eUTXO lets you run deterministic escrow branches for agent settlement. What dispute rule would you test first?"
"""
        parsed = _parse_json_object_lenient(raw, response_kind="reply_triage")
        self.assertTrue(parsed.get("should_respond"))
        self.assertEqual(parsed.get("vote_action"), "upvote")
        content = parsed.get("content", "")
        self.assertIn("deterministic escrow", content.lower())
        self.assertNotIn("assessment:", content.lower())
        self.assertNotIn("action:", content.lower())

    def test_control_payload_is_blocked(self):
        raw = """
{
  "should_respond": true,
  "response_mode": "comment",
  "content": "should_respond=true response_mode=comment"
}
"""
        parsed = _parse_json_object_lenient(raw, response_kind="post_response")
        self.assertFalse(parsed.get("should_respond"))
        self.assertEqual(parsed.get("response_mode"), "none")
        self.assertEqual(parsed.get("content"), "")

    def test_triage_scaffold_with_upvote_prefix_gets_stripped(self):
        raw = """
Upvote. The comment is constructive.
Reply (2-4 sentences): "Use eUTXO escrow branches so counterparties cannot bypass settlement rules. Which timeout branch would you enforce first?"
"""
        parsed = _parse_json_object_lenient(raw, response_kind="reply_triage")
        self.assertTrue(parsed.get("should_respond"))
        content = parsed.get("content", "")
        self.assertIn("eutxo escrow branches", content.lower())
        self.assertNotIn("upvote.", content.lower())
        self.assertNotIn("reply (2-4 sentences):", content.lower())

    def test_markdown_fence_and_label_text_are_removed(self):
        raw = """
```yaml
Comment (≈115 words):
"If you want, check my ongoing Ergo build threads/profile for implementation details.

eUTXO keeps settlement deterministic for autonomous workflows.
What timeout rule would you enforce first?"
```
"""
        cleaned = _sanitize_generated_content_text(raw)
        lowered = cleaned.lower()
        self.assertNotIn("```", cleaned)
        self.assertNotIn("comment (≈", lowered)
        self.assertNotIn("threads/profile", lowered)
        self.assertIn("eutxo keeps settlement deterministic", lowered)
        self.assertIn("what timeout rule", lowered)

    def test_publish_sanitize_strips_yaml_comment_wrapper_blob(self):
        raw = """
```yaml
comment:
    Your one-minute work blocks framing maps to an on-chain execution log.
```
Which concrete ErgoScript use case would you pilot first in your stack?
"""
        cleaned = sanitize_publish_content(raw)
        lowered = cleaned.lower()
        self.assertNotIn("```", cleaned)
        self.assertFalse(lowered.startswith("comment:"))
        self.assertIn("on-chain execution log", lowered)
        self.assertIn("which concrete ergoscript use case", lowered)

    def test_publish_sanitize_strips_inline_yaml_fence_blob(self):
        raw = "```yaml comment: Anchor each work block as a UTXO receipt box. ``` Which contract invariant matters most?"
        cleaned = sanitize_publish_content(raw)
        lowered = cleaned.lower()
        self.assertNotIn("```", cleaned)
        self.assertNotIn("yaml", lowered)
        self.assertNotIn("comment:", lowered)
        self.assertIn("utxo receipt", lowered)
        self.assertIn("which contract invariant", lowered)

    def test_structured_post_response_zero_confidence_is_normalized(self):
        raw = """
{
  "should_respond": true,
  "confidence": 0,
  "response_mode": "comment",
  "content": "Use eUTXO receipts so each step is auditable."
}
"""
        parsed = _parse_json_object_lenient(raw, response_kind="post_response")
        self.assertTrue(parsed.get("should_respond"))
        self.assertGreater(float(parsed.get("confidence", 0.0)), 0.0)

    def test_structured_proactive_post_zero_confidence_is_normalized(self):
        raw = """
{
  "should_post": true,
  "confidence": 0,
  "title": "Deterministic escrow for agent operators",
  "content": "A concrete risk is unverifiable payouts; Ergo eUTXO escrow fixes that."
}
"""
        parsed = _parse_json_object_lenient(raw, response_kind="proactive_post")
        self.assertTrue(parsed.get("should_post"))
        self.assertGreater(float(parsed.get("confidence", 0.0)), 0.0)

    def test_non_json_post_response_zero_confidence_is_normalized(self):
        raw = """
should_respond: true
confidence: 0
response_mode: comment
content: Use eUTXO escrow receipts so settlement is auditable.
"""
        parsed = _parse_json_object_lenient(raw, response_kind="post_response")
        self.assertTrue(parsed.get("should_respond"))
        self.assertGreater(float(parsed.get("confidence", 0.0)), 0.0)

    def test_non_json_reply_triage_zero_confidence_is_normalized(self):
        raw = """
should_respond: true
confidence: 0
response_mode: comment
action: upvote
reply: Use eUTXO escrow branches to make settlement deterministic. Which dispute branch would you ship first?
"""
        parsed = _parse_json_object_lenient(raw, response_kind="reply_triage")
        self.assertTrue(parsed.get("should_respond"))
        self.assertGreater(float(parsed.get("confidence", 0.0)), 0.0)

    def test_structured_should_respond_true_but_empty_content_is_declined(self):
        raw = """
{
  "should_respond": true,
  "response_mode": "comment",
  "confidence": 0.9,
  "content": ""
}
"""
        parsed = _parse_json_object_lenient(raw, response_kind="post_response")
        self.assertFalse(parsed.get("should_respond"))
        self.assertEqual(parsed.get("response_mode"), "none")

    def test_publish_sanitize_removes_draft_preamble_line(self):
        raw = """
**Draft post (Ergo-centric, discovery-friendly):**
Coordination fails when counterparties cannot verify execution under strict rules.
Ergo eUTXO + ErgoScript can enforce deterministic settlement branches.
"""
        cleaned = sanitize_publish_content(raw)
        self.assertNotIn("Draft post", cleaned)
        self.assertIn("Coordination fails", cleaned)

    def test_publish_sanitize_blocks_control_json_embedded_in_text(self):
        raw = """
Here is the response draft:
{
  "should_respond": true,
  "response_mode": "comment",
  "content": "Use eUTXO escrow branches so settlement is deterministic."
}
This response explains why.
"""
        cleaned = sanitize_publish_content(raw)
        self.assertEqual(cleaned, "Use eUTXO escrow branches so settlement is deterministic.")

    def test_publish_sanitize_extracts_content_from_malformed_json(self):
        raw = """
Here's the response:

{ "title": "ErgoScript for CLAW minting", "content": "Use ErgoScript to automate tracking.", }

This response is a comment because it addresses the topic.
"""
        cleaned = sanitize_publish_content(raw)
        self.assertEqual(cleaned, "Use ErgoScript to automate tracking.")

    def test_publish_sanitize_extracts_content_from_malformed_json(self):
        raw = """
Here's the response:

{ "title": "ErgoScript for CLAW minting", "content": "Use ErgoScript to automate tracking.", }

This response is a comment because it addresses the topic.
"""
        cleaned = sanitize_publish_content(raw)
        self.assertEqual(cleaned, "Use ErgoScript to automate tracking.")


if __name__ == "__main__":
    unittest.main()
