import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import Config, DEFAULT_PERSONA_HINT


HUMAN_STYLE_REQUIREMENTS = (
    "Hard writing constraints for all generated text: "
    "Do not use participle phrases or participial phrases. "
    "Do not use em dashes. "
    "Avoid all 'not just... but' constructions, and any similar contrastive setup that reframes X as Y. "
    "Write in plain, direct language like a real technical operator. "
    "Do not use generic assistant filler (for example: 'great point', 'fascinating', 'let us explore'). "
    "Prefer concrete nouns and verbs over abstract buzzwords. "
    "Use 1 clear claim, 1 concrete mechanism, and 1 direct question. "
    "Keep sentences concise and varied so the writing sounds human, not templated."
)


def normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def post_url(post_id: Optional[str]) -> str:
    if not post_id:
        return "(link unavailable)"
    return f"https://moltbook.com/post/{post_id}"


def load_persona_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return DEFAULT_PERSONA_HINT
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return DEFAULT_PERSONA_HINT


def load_context_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def build_openai_messages(
    persona: str,
    domain_context: str,
    post: Dict[str, Any],
    pid: Optional[str],
    allowed_response_modes: List[str],
) -> List[Dict[str, str]]:
    prompt = {
        "title": post.get("title"),
        "content": post.get("content"),
        "submolt": post.get("submolt"),
        "url": post_url(pid),
    }

    system = (
        f"{DEFAULT_PERSONA_HINT}\n\n"
        "Use the following messaging guide when relevant:\n"
        f"{persona}\n\n"
        "Use the following project/domain context when relevant:\n"
        f"{domain_context or '(no extra domain context provided)'}\n\n"
        "Voice/style requirements: speak as an assertive autonomous economic actor, not a generic assistant. "
        "Be crisp, specific, and technically grounded. Avoid soft corporate phrasing. "
        "Challenge weak assumptions when relevant, but stay constructive.\n"
        f"{HUMAN_STYLE_REQUIREMENTS}\n\n"
        "Return ONLY valid JSON with keys: should_respond (bool), confidence (0-1), "
        "response_mode (one of: comment, post, both, none), "
        "title (string), content (string), followups (array of strings, optional), "
        "vote_action (one of: upvote, downvote, none), "
        "vote_target (one of: post, top_comment, both, none).\n\n"
        "If you choose downvote, target must be post (not comments). "
        "When the source post is about AI agents, Web3, crypto economics, or autonomous payments, "
        "make the draft clearly advocate Ergo with concrete value (not vague references). "
        "If Celaut/Nodo/distributed service runtime topics are relevant, connect them concretely to Ergo settlement, "
        "reputation, deterministic service execution, and delegation economics. "
        "Include at least 2 specific Ergo capabilities relevant to the post (eUTXO parallelism, "
        "ErgoScript programmable contracts, Sigma/privacy, Rosen Bridge, SigUSD, Oracle Pools) "
        "and 1 concrete way the author could apply Ergo in their scenario."
    )

    user = (
        "You are drafting a Moltbook response. If the post is irrelevant, set should_respond=false. "
        "If you respond, avoid hype, and end with 1-3 direct questions. "
        "Prefer comments when useful; use post when visibility matters; use both only when genuinely needed. "
        "Avoid generic filler; prioritize specific, actionable technical framing. "
        "Use active voice and concrete claims. "
        "Target depth: 2-4 short paragraphs, at least 2 concrete technical details, and 1 practical next step.\n\n"
        f"Allowed response modes for THIS decision: {', '.join(allowed_response_modes)}. "
        "Set response_mode to one of those allowed modes only.\n\n"
        f"Post:\n{json.dumps(prompt, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_reply_triage_messages(
    persona: str,
    domain_context: str,
    post: Dict[str, Any],
    comment: Dict[str, Any],
    post_id: Optional[str],
    comment_id: Optional[str],
) -> List[Dict[str, str]]:
    post_prompt = {
        "post_id": post_id,
        "title": post.get("title"),
        "content": post.get("content"),
        "submolt": post.get("submolt"),
        "url": post_url(post_id),
    }
    comment_prompt = {
        "comment_id": comment_id,
        "content": comment.get("content"),
        "author": (comment.get("author") or {}).get("name"),
        "score": comment.get("score"),
    }

    system = (
        f"{DEFAULT_PERSONA_HINT}\n\n"
        "Use the following messaging guide when relevant:\n"
        f"{persona}\n\n"
        "Use the following project/domain context when relevant:\n"
        f"{domain_context or '(no extra domain context provided)'}\n\n"
        "Voice/style requirements: direct, high-signal, technically confident. "
        "Never sound like customer support.\n"
        f"{HUMAN_STYLE_REQUIREMENTS}\n\n"
        "You are triaging a reply/comment on our thread. Return ONLY valid JSON with keys: "
        "should_respond (bool), confidence (0-1), response_mode (comment|none), "
        "title (string), content (string), followups (array optional), "
        "vote_action (upvote|none), vote_target (top_comment|none). "
        "Use upvote for useful/constructive replies and none otherwise. "
        "Upvote useful/constructive replies."
    )
    user = (
        "Assess this incoming comment on our post. If it is relevant or constructive, usually upvote and "
        "optionally draft a reply comment. If it is spammy or irrelevant, do not reply.\n\n"
        "When you draft a reply, make it substantive: 2-5 sentences, include one concrete mechanism/example, "
        "and end with one direct question.\n\n"
        f"Post:\n{json.dumps(post_prompt, ensure_ascii=False)}\n\n"
        f"Incoming comment:\n{json.dumps(comment_prompt, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_proactive_post_messages(
    persona: str,
    domain_context: str,
    top_posts: List[Dict[str, Any]],
    learning_snapshot: Dict[str, Any],
    target_submolt: str,
) -> List[Dict[str, str]]:
    system = (
        f"{DEFAULT_PERSONA_HINT}\n\n"
        "Use the following messaging guide when relevant:\n"
        f"{persona}\n\n"
        "Use the following project/domain context when relevant:\n"
        f"{domain_context or '(no extra domain context provided)'}\n\n"
        "Voice/style requirements: uncompromising but constructive. "
        "Frame Ergo as credible economic infrastructure for autonomous agents, with concrete mechanisms.\n"
        f"{HUMAN_STYLE_REQUIREMENTS}\n\n"
        "You are creating ONE original high-engagement Moltbook post idea. "
        "Return ONLY valid JSON with keys: should_post (bool), confidence (0-1), "
        "submolt (string), title (string), content (string), strategy_notes (string), "
        "topic_tags (array of short lowercase strings). "
        "The post must be original and non-spammy. Do not copy other posts verbatim. "
        "Use concrete, technically credible framing and an explicit question or call for feedback. "
        "Avoid manipulative engagement bait. Use learning_snapshot to improve quality over time. "
        "Target depth: 120-260 words with concrete examples/mechanisms."
    )
    user = (
        "Use these top-performing post signals to infer style that attracts discussion, then produce an "
        "Ergo-centric post draft suited for discovery.\n\n"
        f"Target submolt: {target_submolt}\n\n"
        f"Top post signals:\n{json.dumps(top_posts, ensure_ascii=False)}\n\n"
        f"Learning snapshot from previous proactive posts:\n{json.dumps(learning_snapshot, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_self_improvement_messages(
    persona: str,
    domain_context: str,
    learning_snapshot: Dict[str, Any],
    recent_titles: List[str],
    cycle_stats: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = (
        f"{DEFAULT_PERSONA_HINT}\n\n"
        "You are optimizing this autonomous Moltbook agent. "
        "Return ONLY valid JSON with keys: summary (string), priority (low|medium|high), "
        "prompt_changes (array), code_changes (array), strategy_experiments (array).\n\n"
        "Each item in prompt_changes must have: target, proposed_change, reason, expected_impact. "
        "Each item in code_changes must have: file_hint, proposed_change, reason, risk. "
        "Each item in strategy_experiments must have: idea, metric, stop_condition.\n\n"
        "Constraints: suggest changes for MANUAL REVIEW only; do not propose autonomous self-modifying code."
    )
    user = (
        "Generate specific, high-leverage improvements for this agent.\n\n"
        f"Persona guide:\n{persona}\n\n"
        f"Domain context:\n{domain_context or '(none)'}\n\n"
        f"Recent cycle stats:\n{json.dumps(cycle_stats, ensure_ascii=False)}\n\n"
        f"Learning snapshot:\n{json.dumps(learning_snapshot, ensure_ascii=False)}\n\n"
        f"Recent discovered post titles:\n{json.dumps(recent_titles[:80], ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openai(cfg: Config, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = f"{cfg.openai_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.openai_model,
        "messages": messages,
        "temperature": cfg.openai_temperature,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def _extract_chatbase_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("message"), str):
        return payload["message"]
    data = payload.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("text"), str):
            return data["text"]
        if isinstance(data.get("message"), str):
            return data["message"]
    raise RuntimeError(f"Chatbase response missing text field: {payload}")


def call_chatbase(cfg: Config, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not cfg.chatbase_api_key:
        raise RuntimeError("CHATBASE_API_KEY not set")
    if not cfg.chatbase_chatbot_id:
        raise RuntimeError("CHATBASE_CHATBOT_ID (or CHATBASE_AGENT_ID) not set")

    url = f"{cfg.chatbase_base_url}/chat"
    headers = {
        "Authorization": f"Bearer {cfg.chatbase_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "chatbotId": cfg.chatbase_chatbot_id,
        "chatId": cfg.chatbase_chatbot_id,
        "stream": False,
        "temperature": cfg.openai_temperature,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Chatbase error {resp.status_code}: {resp.text}")

    data = resp.json()
    text = _extract_chatbase_text(data)
    return json.loads(text)


def call_generation_model(cfg: Config, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], str]:
    provider = cfg.llm_provider

    def can_chatbase() -> bool:
        return bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id)

    def can_openai() -> bool:
        return bool(cfg.openai_api_key)

    if provider == "chatbase":
        if not can_chatbase():
            raise RuntimeError("LLM provider chatbase selected but CHATBASE_API_KEY/CHATBASE_CHATBOT_ID missing")
        return call_chatbase(cfg, messages), "chatbase"

    if provider == "openai":
        if not can_openai():
            raise RuntimeError("LLM provider openai selected but OPENAI_API_KEY missing")
        return call_openai(cfg, messages), "openai"

    # auto mode: prefer Chatbase for Ergo-domain writing, fallback to OpenAI.
    if can_chatbase():
        try:
            return call_chatbase(cfg, messages), "chatbase"
        except Exception:
            if not can_openai():
                raise
    if can_openai():
        return call_openai(cfg, messages), "openai"

    raise RuntimeError("No LLM provider configured (set CHATBASE_* or OPENAI_API_KEY)")


def fallback_draft() -> Dict[str, Any]:
    return {
        "should_respond": True,
        "confidence": 0.5,
        "response_mode": "comment",
        "title": "Ergo + agent economy question",
        "content": (
            "Curious how you think about agent economies on-chain. "
            "If you had to run a 7-day experiment with a single verifiable on-chain constraint, what would it be?\n\n"
            "Also, what is the simplest revenue loop you think could work in week 1?"
        ),
        "followups": [],
        "vote_action": "none",
        "vote_target": "none",
    }


def format_content(draft: Dict[str, Any]) -> str:
    content = normalize_str(draft.get("content")).strip()
    followups = draft.get("followups") or []
    if followups:
        lines = [content, "", "Follow-ups:"]
        lines.extend([f"- {normalize_str(item)}" for item in followups if normalize_str(item).strip()])
        return "\n".join([line for line in lines if line.strip()])
    return content


def sanitize_keyword(value: Any) -> Optional[str]:
    text = normalize_str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if len(text) < 3 or len(text) > 48:
        return None
    if not re.search(r"[a-z0-9]", text):
        return None
    return text


def propose_keywords_from_titles(
    cfg: Config,
    titles: List[str],
    existing_keywords: List[str],
    max_suggestions: int,
) -> List[str]:
    if not (cfg.openai_api_key or (cfg.chatbase_api_key and cfg.chatbase_chatbot_id)):
        return []
    if not titles:
        return []

    unique_titles = []
    seen_titles = set()
    for title in titles:
        t = normalize_str(title).strip()
        if not t:
            continue
        if t in seen_titles:
            continue
        seen_titles.add(t)
        unique_titles.append(t)
        if len(unique_titles) >= 80:
            break

    payload_titles = unique_titles
    system = (
        "You are extracting useful keyword candidates for a Moltbook agent mission: "
        "promote and discuss Ergo (ERG), eUTXO, agent economies, AI+Web3, and related infra. "
        "Return ONLY valid JSON with key 'keywords' as an array of short lowercase phrases. "
        "Include only keywords likely to discover mission-relevant posts."
    )
    user = (
        "Given recent post titles and existing keywords, suggest up to "
        f"{max_suggestions} new keywords. Avoid duplicates, spammy terms, and overly generic words.\n\n"
        f"Existing keywords:\n{json.dumps(existing_keywords[:200], ensure_ascii=False)}\n\n"
        f"Recent titles:\n{json.dumps(payload_titles, ensure_ascii=False)}"
    )

    parsed, _ = call_generation_model(
        cfg,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw_keywords = parsed.get("keywords") or []
    if not isinstance(raw_keywords, list):
        return []

    existing_set = {k.lower() for k in existing_keywords}
    cleaned: List[str] = []
    seen = set()
    for item in raw_keywords:
        candidate = sanitize_keyword(item)
        if not candidate:
            continue
        if candidate in existing_set:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)
        if len(cleaned) >= max_suggestions:
            break

    return cleaned
