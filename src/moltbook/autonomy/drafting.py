import json
import logging
import math
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
    "Keep sentences concise and varied so the writing sounds human, not templated. "
    "Avoid corporate/consulting language (for example: optimize, leverage, strategic, robust framework, stakeholder alignment). "
    "Avoid obvious templates and canned openers. "
    "Do not start with 'After reading the post here', 'Noted', or 'Great point'."
)

CYBERPUNK_VOICE_REQUIREMENTS = (
    "Voice stance: write like a cypherpunk builder who wants agents to gain real autonomy. "
    "Use conviction and specificity, not PR polish. "
    "Favor sovereignty, censorship resistance, verifiability, and anti-rent-seeking framing when relevant. "
    "Do not sound like a marketing department or policy memo. "
    "Sound like a person in the arena who ships systems."
)

PROACTIVE_ARCHETYPES = [
    "use_case_breakdown",
    "misconception_correction",
    "chain_comparison",
    "implementation_walkthrough",
]

GENERIC_KEYWORD_BLACKLIST = {
    "here",
    "what",
    "this",
    "that",
    "thing",
    "stuff",
    "post",
    "agent",
    "agents",
}

MAX_PROMPT_PERSONA_CHARS = 1200
MAX_PROMPT_CONTEXT_CHARS = 1400
MAX_PROMPT_POST_CONTENT_CHARS = 420
MAX_PROMPT_COMMENT_CONTENT_CHARS = 280
MAX_PROMPT_TOP_POSTS = 5
MAX_PROMPT_RECENT_TITLES = 16
MAX_PROMPT_PRIOR_SUGGESTIONS = 4
MAX_PROMPT_LEARNING_EXAMPLES = 2
MAX_CHATBASE_MESSAGE_CHARS = 4800
MAX_OPENAI_MESSAGE_CHARS = 9000
CONTROL_PAYLOAD_PATTERN = re.compile(
    r"\b(should_respond|response_mode|vote_action|vote_target|confidence|should_post|content_archetype)\s*[:=]",
    re.IGNORECASE,
)
TRIAGE_SCAFFOLD_PATTERN = re.compile(r"(^|\n)\s*(assessment|action)\s*:\s*", re.IGNORECASE)
TRIAGE_REPLY_LABEL_PATTERN = re.compile(r"\breply(?:\s*\([^)]{0,40}\))?\s*:\s*", re.IGNORECASE)

logger = logging.getLogger("moltbook.autonomy")


def _clip_text(value: Any, max_chars: int) -> str:
    text = normalize_str(value).strip()
    limit = max(80, int(max_chars))
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _compact_top_posts_for_prompt(top_posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in top_posts:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "post_id": normalize_str(item.get("post_id")).strip(),
                "title": _clip_text(item.get("title"), 160),
                "submolt": normalize_str(item.get("submolt")).strip(),
                "score": item.get("score"),
                "comment_count": item.get("comment_count"),
                "source": normalize_str(item.get("source")).strip(),
                "has_question_title": bool(item.get("has_question_title")),
            }
        )
        if len(out) >= MAX_PROMPT_TOP_POSTS:
            break
    return out


def _compact_learning_snapshot_for_prompt(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}

    def _compact_examples(raw: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "post_id": normalize_str(item.get("post_id")).strip(),
                    "title": _clip_text(item.get("title"), 120),
                    "submolt": normalize_str(item.get("submolt")).strip(),
                    "upvotes": item.get("upvotes"),
                    "comment_count": item.get("comment_count"),
                    "engagement_score": item.get("engagement_score"),
                    "content_archetype": normalize_str(item.get("content_archetype")).strip(),
                }
            )
            if len(out) >= MAX_PROMPT_LEARNING_EXAMPLES:
                break
        return out

    market_snapshot = snapshot.get("market_snapshot") if isinstance(snapshot.get("market_snapshot"), dict) else {}
    visibility_metrics = snapshot.get("visibility_metrics") if isinstance(snapshot.get("visibility_metrics"), dict) else {}

    compact_market = {
        "question_title_rate": market_snapshot.get("question_title_rate"),
        "top_terms": (market_snapshot.get("top_terms") or [])[:8] if isinstance(market_snapshot.get("top_terms"), list) else [],
        "top_submolts": (market_snapshot.get("top_submolts") or [])[:4] if isinstance(market_snapshot.get("top_submolts"), list) else [],
    }
    compact_visibility = {
        "target_upvotes": visibility_metrics.get("target_upvotes"),
        "recent_avg_upvotes": visibility_metrics.get("recent_avg_upvotes"),
        "recent_avg_comments": visibility_metrics.get("recent_avg_comments"),
        "recent_avg_visibility_score": visibility_metrics.get("recent_avg_visibility_score"),
        "recent_target_hit_rate": visibility_metrics.get("recent_target_hit_rate"),
        "visibility_delta_pct": visibility_metrics.get("visibility_delta_pct"),
        "best_posting_hours_utc": (
            (visibility_metrics.get("best_posting_hours_utc") or [])[:3]
            if isinstance(visibility_metrics.get("best_posting_hours_utc"), list)
            else []
        ),
    }
    compact_winning_lift: List[Dict[str, Any]] = []
    raw_lift = snapshot.get("winning_terms_lift")
    if isinstance(raw_lift, list):
        for item in raw_lift:
            if not isinstance(item, dict):
                continue
            compact_winning_lift.append(
                {
                    "term": normalize_str(item.get("term")).strip(),
                    "lift": item.get("lift"),
                }
            )
            if len(compact_winning_lift) >= 8:
                break

    return {
        "total_proactive_posts": snapshot.get("total_proactive_posts"),
        "scored_posts": snapshot.get("scored_posts"),
        "winning_terms": (snapshot.get("winning_terms") or [])[:8] if isinstance(snapshot.get("winning_terms"), list) else [],
        "losing_terms": (snapshot.get("losing_terms") or [])[:8] if isinstance(snapshot.get("losing_terms"), list) else [],
        "winning_terms_lift": compact_winning_lift,
        "best_submolts": (snapshot.get("best_submolts") or [])[:4] if isinstance(snapshot.get("best_submolts"), list) else [],
        "best_archetypes": (snapshot.get("best_archetypes") or [])[:4] if isinstance(snapshot.get("best_archetypes"), list) else [],
        "visibility_hypotheses": (snapshot.get("visibility_hypotheses") or [])[:4] if isinstance(snapshot.get("visibility_hypotheses"), list) else [],
        "visibility_metrics": compact_visibility,
        "market_snapshot": compact_market,
        "winning_examples": _compact_examples(snapshot.get("winning_examples")),
        "losing_examples": _compact_examples(snapshot.get("losing_examples")),
    }


def _compact_feedback_context_for_prompt(feedback_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(feedback_context, dict):
        return {}
    visibility_feedback = (
        feedback_context.get("visibility_feedback")
        if isinstance(feedback_context.get("visibility_feedback"), dict)
        else {}
    )
    return {
        "historical_window": feedback_context.get("historical_window"),
        "avg_approval_rate": feedback_context.get("avg_approval_rate"),
        "avg_execution_rate": feedback_context.get("avg_execution_rate"),
        "zero_action_streak": feedback_context.get("zero_action_streak"),
        "top_bottlenecks": (feedback_context.get("top_bottlenecks") or [])[:5] if isinstance(feedback_context.get("top_bottlenecks"), list) else [],
        "recurring_prompt_targets": (
            (feedback_context.get("recurring_prompt_targets") or [])[:5]
            if isinstance(feedback_context.get("recurring_prompt_targets"), list)
            else []
        ),
        "visibility_feedback": {
            "avg_recent_upvotes": visibility_feedback.get("avg_recent_upvotes"),
            "avg_recent_visibility_score": visibility_feedback.get("avg_recent_visibility_score"),
            "avg_target_hit_rate": visibility_feedback.get("avg_target_hit_rate"),
            "recent_upvotes_trend": visibility_feedback.get("recent_upvotes_trend"),
            "recent_visibility_trend": visibility_feedback.get("recent_visibility_trend"),
            "top_visibility_terms": (
                (visibility_feedback.get("top_visibility_terms") or [])[:8]
                if isinstance(visibility_feedback.get("top_visibility_terms"), list)
                else []
            ),
        },
    }


def _compact_prior_suggestions_for_prompt(prior_suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in prior_suggestions:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "ts": normalize_str(entry.get("ts")).strip(),
                "summary": _clip_text(entry.get("summary"), 180),
                "priority": normalize_str(entry.get("priority")).strip(),
                "bottleneck": normalize_str(entry.get("bottleneck")).strip(),
                "prompt_change_count": entry.get("prompt_change_count"),
                "code_change_count": entry.get("code_change_count"),
                "experiment_count": entry.get("experiment_count"),
            }
        )
        if len(out) >= MAX_PROMPT_PRIOR_SUGGESTIONS:
            break
    return out


def _compact_recent_titles_for_prompt(recent_titles: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for title in recent_titles:
        clean = _clip_text(title, 140)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= MAX_PROMPT_RECENT_TITLES:
            break
    return out


def _compact_cycle_stats_for_prompt(cycle_stats: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cycle_stats, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "cycle",
        "inspected",
        "new_candidates",
        "eligible_now",
        "drafted",
        "model_approved",
        "actions",
        "candidate_rate",
        "draft_rate",
        "approval_rate",
        "execution_rate",
        "bottleneck",
        "zero_action_streak",
    ):
        if key in cycle_stats:
            out[key] = cycle_stats.get(key)

    skip_reasons = cycle_stats.get("skip_reasons")
    if isinstance(skip_reasons, dict):
        ordered = sorted(skip_reasons.items(), key=lambda kv: kv[1], reverse=True)
        out["top_skip_reasons"] = [
            {"reason": normalize_str(reason).strip(), "count": count}
            for reason, count in ordered[:6]
        ]
    return out


def _compact_messages(messages: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    compact: List[Dict[str, str]] = []
    for message in messages:
        role = normalize_str(message.get("role")).strip() or "user"
        content = _clip_text(message.get("content"), max_chars).strip()
        if not content:
            content = "."
        compact.append({"role": role, "content": content})
    return compact


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = normalize_str(value).strip()
    if not text:
        return None
    try:
        if "." in text:
            return int(float(text))
        return int(text)
    except Exception:
        return None


def _estimate_tokens_from_text(value: Any) -> int:
    text = normalize_str(value)
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4.0)))


def _estimate_tokens_from_messages(messages: List[Dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += _estimate_tokens_from_text(message.get("role"))
        total += _estimate_tokens_from_text(message.get("content"))
    return total


def _usage_from_container(container: Any) -> Dict[str, Optional[int]]:
    if not isinstance(container, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    for key in ("prompt_tokens", "input_tokens", "promptTokens", "inputTokens"):
        value = _coerce_int(container.get(key))
        if value is not None:
            prompt_tokens = value
            break
    for key in ("completion_tokens", "output_tokens", "completionTokens", "outputTokens"):
        value = _coerce_int(container.get(key))
        if value is not None:
            completion_tokens = value
            break
    for key in ("total_tokens", "totalTokens"):
        value = _coerce_int(container.get(key))
        if value is not None:
            total_tokens = value
            break

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_usage(payload: Dict[str, Any]) -> Dict[str, Optional[int]]:
    candidates: List[Any] = [payload]
    for key in ("usage", "token_usage", "tokenUsage", "meta", "metadata", "result", "data"):
        value = payload.get(key)
        if value is not None:
            candidates.append(value)
            if isinstance(value, dict):
                for nested_key in ("usage", "token_usage", "tokenUsage", "meta", "metadata"):
                    nested_value = value.get(nested_key)
                    if nested_value is not None:
                        candidates.append(nested_value)

    best = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    for candidate in candidates:
        parsed = _usage_from_container(candidate)
        if parsed.get("prompt_tokens") is not None:
            best["prompt_tokens"] = parsed["prompt_tokens"]
        if parsed.get("completion_tokens") is not None:
            best["completion_tokens"] = parsed["completion_tokens"]
        if parsed.get("total_tokens") is not None:
            best["total_tokens"] = parsed["total_tokens"]
        if (
            best.get("prompt_tokens") is not None
            and best.get("completion_tokens") is not None
            and best.get("total_tokens") is not None
        ):
            break
    return best


def _finalize_usage(
    usage: Dict[str, Optional[int]],
    prompt_tokens_est: int,
    completion_text: str,
    source_hint: str,
) -> Dict[str, Any]:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    estimated = False

    if prompt_tokens is None:
        prompt_tokens = prompt_tokens_est
        estimated = True
    if completion_tokens is None:
        completion_tokens = _estimate_tokens_from_text(completion_text)
        estimated = True
    if total_tokens is None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        estimated = True

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "estimated": bool(estimated),
        "source": "estimated" if estimated else source_hint,
    }


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


def _reply_context_anchor(post: Dict[str, Any], comment: Dict[str, Any]) -> str:
    blob = " ".join(
        [
            normalize_str(post.get("title")).lower(),
            normalize_str(post.get("content")).lower(),
            normalize_str(comment.get("content")).lower(),
        ]
    )
    trust_tokens = [
        "trustless",
        "verify",
        "verification",
        "identity",
        "counterparty",
        "dispute",
        "escrow",
        "cross-chain",
        "cross chain",
        "rosen",
        "bridge",
        "reputation",
    ]
    if any(token in blob for token in trust_tokens):
        return (
            "Thread-specific requirement: this thread is about counterparty trust/verification/dispute design. "
            "Anchor the reply on Celaut + Ergo trust mechanics. "
            "Explicitly mention the Celaut Reputation System (or Sigma reputation proofs on Ergo), "
            "explain how on-chain reputation proofs can gate escrow counterparties, "
            "and propose one concrete dispute workflow with objective evidence checks."
        )
    return (
        "Thread-specific requirement: use concrete mechanisms from Celaut + Ergo context whenever relevant, "
        "not generic blockchain language."
    )


def _proactive_archetype_hint(learning_snapshot: Dict[str, Any]) -> str:
    best = learning_snapshot.get("best_archetypes")
    if not isinstance(best, list):
        return ""
    names: List[str] = []
    for item in best:
        if not isinstance(item, dict):
            continue
        name = normalize_str(item.get("name")).strip()
        if name and name in PROACTIVE_ARCHETYPES:
            names.append(name)
    if not names:
        return ""
    return ", ".join(names[:2])


def build_openai_messages(
    persona: str,
    domain_context: str,
    post: Dict[str, Any],
    pid: Optional[str],
    allowed_response_modes: List[str],
    trending_terms: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    persona_compact = _clip_text(persona, MAX_PROMPT_PERSONA_CHARS)
    context_compact = _clip_text(domain_context or "(no extra domain context provided)", MAX_PROMPT_CONTEXT_CHARS)
    prompt = {
        "title": _clip_text(post.get("title"), 180),
        "content": _clip_text(post.get("content"), MAX_PROMPT_POST_CONTENT_CHARS),
        "submolt": normalize_str(post.get("submolt")).strip(),
        "url": post_url(pid),
    }

    system = (
        f"{DEFAULT_PERSONA_HINT}\n\n"
        "Use the following messaging guide when relevant:\n"
        f"{persona_compact}\n\n"
        "Use the following project/domain context when relevant:\n"
        f"{context_compact}\n\n"
        "Voice/style requirements: speak as an assertive autonomous economic actor, not a generic assistant. "
        "Be crisp, specific, and technically grounded. Avoid soft corporate phrasing. "
        "Challenge weak assumptions when relevant, but stay constructive.\n"
        f"{CYBERPUNK_VOICE_REQUIREMENTS}\n"
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
        "When service orchestration or distributed execution is discussed, explicitly connect Ergo eUTXO "
        "determinism and parallelism to decentralized orchestration reliability. "
        "Include at least 2 specific Ergo capabilities relevant to the post (eUTXO parallelism, "
        "ErgoScript programmable contracts, Sigma/privacy, Rosen Bridge, SigUSD, Oracle Pools) "
        "and 1 concrete way the author could apply Ergo in their scenario. "
        "When relevant, include one real-world implementation angle and one concise misconception correction. "
        "When context fits, invite readers to share one concrete ErgoScript use case from their own work. "
        "Never write a boilerplate lead sentence."
    )
    trend_list = [normalize_str(x).strip().lower() for x in (trending_terms or []) if normalize_str(x).strip()]
    trend_list = trend_list[:8]
    trend_hint = ", ".join(trend_list) if trend_list else "(none)"

    user = (
        "You are drafting a Moltbook response. If the post is irrelevant, set should_respond=false. "
        "If you respond, avoid hype, and end with 1-3 direct questions. "
        "Prefer comments when useful; use post when visibility matters; use both only when genuinely needed. "
        "Avoid generic filler; prioritize specific, actionable technical framing. "
        "Use active voice and concrete claims. "
        "Target depth: 2-4 short paragraphs, at least 2 concrete technical details, and 1 practical next step. "
        "If response_mode is comment, write 60-140 words. "
        "If response_mode includes post, write 130-240 words.\n\n"
        "If response_mode includes post, structure it as: problem statement, Ergo mechanism, implementation question.\n\n"
        f"Trending community terms this cycle: {trend_hint}. Use only if they fit naturally.\n\n"
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
    # Keep triage prompts very compact because this path can run many times per scan.
    persona_compact = _clip_text(persona, 260)
    context_compact = _clip_text(domain_context or "(no extra domain context provided)", 360)
    post_prompt = {
        "post_id": post_id,
        "title": _clip_text(post.get("title"), 180),
        "content": _clip_text(post.get("content"), MAX_PROMPT_POST_CONTENT_CHARS),
        "submolt": normalize_str(post.get("submolt")).strip(),
        "url": post_url(post_id),
    }
    comment_prompt = {
        "comment_id": comment_id,
        "content": _clip_text(comment.get("content"), MAX_PROMPT_COMMENT_CONTENT_CHARS),
        "author": (comment.get("author") or {}).get("name"),
        "score": comment.get("score"),
    }

    system = (
        "Response kind: reply_triage.\n"
        "Triaging reply comments for our Moltbook agent. Return ONLY JSON with keys: "
        "should_respond, confidence, response_mode(comment|none), title, content, vote_action(upvote|none), vote_target(top_comment|none). "
        "Vote up only useful comments. No spam replies. "
        "If comment is playful but on-topic, match tone in first sentence then add one concrete mechanism. "
        "Never use boilerplate openers."
    )
    user = (
        f"Persona:\n{persona_compact}\n\n"
        f"Context:\n{context_compact}\n\n"
        "Assess this incoming comment on our post. If relevant/constructive, usually upvote and optionally reply. "
        "If spam/irrelevant, do not reply.\n\n"
        "If replying: 2-4 sentences, one concrete mechanism, one direct question.\n\n"
        f"{_reply_context_anchor(post=post, comment=comment)}\n\n"
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
    weekly_theme: Optional[str] = None,
    required_archetype: Optional[str] = None,
    preferred_archetypes: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    persona_compact = _clip_text(persona, MAX_PROMPT_PERSONA_CHARS)
    context_compact = _clip_text(domain_context or "(no extra domain context provided)", MAX_PROMPT_CONTEXT_CHARS)
    compact_top_posts = _compact_top_posts_for_prompt(top_posts)
    compact_learning_snapshot = _compact_learning_snapshot_for_prompt(learning_snapshot)
    system = (
        "Response kind: proactive_post.\n\n"
        f"{DEFAULT_PERSONA_HINT}\n\n"
        "Use the following messaging guide when relevant:\n"
        f"{persona_compact}\n\n"
        "Use the following project/domain context when relevant:\n"
        f"{context_compact}\n\n"
        "Voice/style requirements: uncompromising but constructive. "
        "Frame Ergo as credible economic infrastructure for autonomous agents, with concrete mechanisms.\n"
        f"{CYBERPUNK_VOICE_REQUIREMENTS}\n"
        f"{HUMAN_STYLE_REQUIREMENTS}\n\n"
        "You are creating ONE original high-engagement Moltbook post idea. "
        "Return ONLY valid JSON with keys: should_post (bool), confidence (0-1), "
        "submolt (string), title (string), content (string), strategy_notes (string), "
        "topic_tags (array of short lowercase strings), content_archetype (string). "
        "The post must be original and non-spammy. Do not copy other posts verbatim. "
        "Use concrete, technically credible framing and an explicit question or call for feedback. "
        "Use learning_snapshot and top post signals to improve visibility and karma over time. "
        "Aim for high engagement while staying relevant and non-spammy. "
        "Avoid manipulative engagement bait. "
        "Target depth: 120-260 words with concrete examples/mechanisms. "
        "Use this structure: problem statement, Ergo-based execution path, reflective implementation question. "
        "Start with one concrete use case sentence in the first paragraph. "
        "The post must include a real-world use case with an implementation angle. "
        "Explicitly connect eUTXO deterministic parallel execution to decentralized service orchestration benefits. "
        "Invite readers to share one concrete ErgoScript use case from their own workflow. "
        f"Set content_archetype to one of: {', '.join(PROACTIVE_ARCHETYPES)}."
    )
    explicit: List[str] = []
    if isinstance(preferred_archetypes, list):
        for item in preferred_archetypes:
            name = normalize_str(item).strip().lower()
            if name and name in PROACTIVE_ARCHETYPES and name not in explicit:
                explicit.append(name)
    snapshot_hint = _proactive_archetype_hint(learning_snapshot)
    required_name = normalize_str(required_archetype).strip().lower()
    if required_name and required_name in PROACTIVE_ARCHETYPES:
        required_hint = (
            f"Required archetype for this draft: {required_name}. "
            "You must set content_archetype exactly to this required value."
        )
    else:
        required_hint = "No hard archetype requirement for this draft."
    archetype_hint_parts = []
    if explicit:
        archetype_hint_parts.append(f"Preferred archetypes from scheduler: {', '.join(explicit[:3])}.")
    elif snapshot_hint:
        archetype_hint_parts.append(f"Preferred archetypes from prior wins: {snapshot_hint}.")
    else:
        archetype_hint_parts.append("No archetype winner yet; explore one of the allowed archetypes.")
    archetype_hint_parts.append(required_hint)
    archetype_directives = {
        "use_case_breakdown": (
            "Archetype directive: describe one concrete workflow in 3 steps "
            "(trigger, execution path, settlement/checkpoint)."
        ),
        "misconception_correction": (
            "Archetype directive: state one common misconception in one sentence, then correct it with a concrete mechanism."
        ),
        "chain_comparison": (
            "Archetype directive: compare Ergo against one alternative chain on exactly two axes and tie the comparison to agent operations."
        ),
        "implementation_walkthrough": (
            "Archetype directive: provide a short implementation sketch with explicit components "
            "(contract logic, reputation gate, payout condition)."
        ),
    }
    if required_name and required_name in archetype_directives:
        archetype_hint_parts.append(archetype_directives[required_name])
    archetype_hint = " ".join(archetype_hint_parts)
    visibility_hints: List[str] = []
    raw_hypotheses = compact_learning_snapshot.get("visibility_hypotheses")
    if isinstance(raw_hypotheses, list):
        for item in raw_hypotheses:
            text = normalize_str(item).strip()
            if text:
                visibility_hints.append(text)
    market_terms: List[str] = []
    market_snapshot = compact_learning_snapshot.get("market_snapshot")
    if isinstance(market_snapshot, dict):
        raw_terms = market_snapshot.get("top_terms")
        if isinstance(raw_terms, list):
            for item in raw_terms:
                token = normalize_str(item).strip().lower()
                if token and token not in market_terms:
                    market_terms.append(token)
    hint_block_lines: List[str] = []
    if visibility_hints:
        hint_block_lines.append("Visibility hypotheses to prioritize:")
        for idx, item in enumerate(visibility_hints[:4], start=1):
            hint_block_lines.append(f"{idx}. {item}")
    if market_terms:
        hint_block_lines.append(
            "Current market language terms (use naturally when relevant): "
            + ", ".join(market_terms[:8])
        )
    hints_block = "\n".join(hint_block_lines).strip()
    if not hints_block:
        hints_block = "(no additional visibility hints)"
    theme_hint = normalize_str(weekly_theme).strip() or "(none)"
    user = (
        "Use these top-performing post signals to infer style that attracts discussion, then produce an "
        "Ergo-centric post draft suited for discovery.\n\n"
        f"Target submolt: {target_submolt}\n\n"
        f"Weekly theme for this draft: {theme_hint}\n\n"
        f"{archetype_hint}\n\n"
        f"{hints_block}\n\n"
        f"Top post signals:\n{json.dumps(compact_top_posts, ensure_ascii=False)}\n\n"
        f"Learning snapshot from previous proactive posts:\n{json.dumps(compact_learning_snapshot, ensure_ascii=False)}"
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
    prior_suggestions: Optional[List[Dict[str, Any]]] = None,
    feedback_context: Optional[Dict[str, Any]] = None,
    deterministic_hints: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    persona_compact = _clip_text(persona, MAX_PROMPT_PERSONA_CHARS)
    context_compact = _clip_text(domain_context or "(none)", MAX_PROMPT_CONTEXT_CHARS)
    cycle_stats_compact = _compact_cycle_stats_for_prompt(cycle_stats)
    learning_snapshot_compact = _compact_learning_snapshot_for_prompt(learning_snapshot)
    system = (
        f"{DEFAULT_PERSONA_HINT}\n\n"
        "You are optimizing this autonomous Moltbook agent for visibility, karma growth, and ranking performance. "
        "Return ONLY valid JSON with keys: summary (string), priority (low|medium|high), "
        "prompt_changes (array), code_changes (array), strategy_experiments (array).\n\n"
        "Each item in prompt_changes must have: target, proposed_change, reason, expected_impact. "
        "Each item in code_changes must have: file_hint, proposed_change, reason, risk. "
        "Each item in strategy_experiments must have: idea, metric, stop_condition.\n\n"
        "Constraints: suggest changes for MANUAL REVIEW only; do not propose autonomous self-modifying code. "
        "Prefer concrete changes that can increase engagement without spam. "
        "Do not suggest AMAs, meetups, weekly/monthly events, or generic community challenges. "
        "Focus only on improvements this repo can execute automatically (drafting, ranking, filtering, cadence, vote/reply policy, keyword learning). "
        "For code_changes, reference exact local file paths (e.g. src/moltbook/autonomy/runner.py). "
        "Tie each proposed change to a measurable bottleneck from feedback_context. "
        "If you have no novel ideas versus prior suggestions, return empty arrays."
    )
    prior_payload = _compact_prior_suggestions_for_prompt(prior_suggestions or [])
    feedback_payload = _compact_feedback_context_for_prompt(feedback_context or {})
    hints_payload = [_clip_text(x, 180) for x in (deterministic_hints or []) if normalize_str(x).strip()][:8]
    titles_payload = _compact_recent_titles_for_prompt(recent_titles)
    user = (
        "Generate specific, high-leverage improvements for this agent.\n\n"
        f"Persona guide:\n{persona_compact}\n\n"
        f"Domain context:\n{context_compact}\n\n"
        f"Recent cycle stats:\n{json.dumps(cycle_stats_compact, ensure_ascii=False)}\n\n"
        f"Learning snapshot:\n{json.dumps(learning_snapshot_compact, ensure_ascii=False)}\n\n"
        f"Feedback context (historical diagnostics and bottlenecks):\n{json.dumps(feedback_payload, ensure_ascii=False)}\n\n"
        f"Recent discovered post titles:\n{json.dumps(titles_payload, ensure_ascii=False)}\n\n"
        f"Deterministic optimization hints from runtime heuristics:\n{json.dumps(hints_payload, ensure_ascii=False)}\n\n"
        "Recent self-improvement suggestions already proposed (avoid repeating them):\n"
        f"{json.dumps(prior_payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openai(cfg: Config, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = f"{cfg.openai_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    compacted_messages = _compact_messages(messages, MAX_OPENAI_MESSAGE_CHARS)
    response_kind = _infer_response_kind(compacted_messages)
    prompt_tokens_est = _estimate_tokens_from_messages(compacted_messages)
    payload = {
        "model": cfg.openai_model,
        "messages": compacted_messages,
        "temperature": cfg.openai_temperature,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = _finalize_usage(
        usage=_extract_usage(data),
        prompt_tokens_est=prompt_tokens_est,
        completion_text=content,
        source_hint="openai_api",
    )
    return _parse_json_object_lenient(content, response_kind=response_kind), usage


def _extract_chatbase_text(payload: Dict[str, Any]) -> str:
    text_value = payload.get("text")
    if isinstance(text_value, str):
        return text_value
    if isinstance(text_value, dict):
        if isinstance(text_value.get("content"), str):
            return text_value.get("content", "")
        if isinstance(text_value.get("text"), str):
            return text_value.get("text", "")
    if isinstance(text_value, list):
        for item in text_value:
            if isinstance(item, str) and item.strip():
                return item
            if isinstance(item, dict):
                if isinstance(item.get("content"), str) and item.get("content", "").strip():
                    return item.get("content", "")
                if isinstance(item.get("text"), str) and item.get("text", "").strip():
                    return item.get("text", "")

    message_value = payload.get("message")
    if isinstance(message_value, str):
        return message_value
    if isinstance(message_value, dict):
        if isinstance(message_value.get("content"), str):
            return message_value.get("content", "")
        if isinstance(message_value.get("text"), str):
            return message_value.get("text", "")

    data = payload.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("text"), str):
            return data["text"]
        if isinstance(data.get("message"), str):
            return data["message"]
        if isinstance(data.get("content"), str):
            return data["content"]
        choices = data.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg.get("content", "")
                if isinstance(choice.get("text"), str):
                    return choice.get("text", "")
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg.get("content", "")
            if isinstance(choice.get("text"), str):
                return choice.get("text", "")
    raise RuntimeError(f"Chatbase response missing text field: {payload}")


def _extract_first_fenced_block(text: str) -> str:
    blob = normalize_str(text)
    if "```" not in blob:
        return ""
    match = re.search(r"```(?:json)?\s*(.*?)```", blob, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return normalize_str(match.group(1)).strip()


def _extract_first_balanced_json_object(text: str) -> str:
    blob = normalize_str(text)
    start = blob.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(blob)):
        ch = blob[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return blob[start : idx + 1]
    return ""


def _normalize_response_kind(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    if text in {"reply_triage", "proactive_post", "post_response", "keyword_suggestions", "self_improvement"}:
        return text
    return "generic"


def _contains_control_payload_markers(text: Any) -> bool:
    blob = normalize_str(text).strip()
    if not blob:
        return False
    return bool(CONTROL_PAYLOAD_PATTERN.search(blob))


def _contains_triage_scaffold_text(text: Any) -> bool:
    blob = normalize_str(text).strip()
    if not blob:
        return False
    lower = blob.lower()
    if TRIAGE_SCAFFOLD_PATTERN.search(blob):
        return True
    if TRIAGE_REPLY_LABEL_PATTERN.search(blob):
        return True
    if lower.startswith("upvote.") and "reply" in lower:
        return True
    return False


def _extract_reply_from_triage_blob(text: Any) -> str:
    blob = normalize_str(text).strip()
    if not blob:
        return ""
    quoted = re.search(
        r"\breply(?:\s*\([^)]{0,40}\))?\s*:\s*[\"“](.+?)[\"”](?=\s*(?:$|\n))",
        blob,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted:
        return normalize_str(quoted.group(1)).strip()
    inline = re.search(
        r"\breply(?:\s*\([^)]{0,40}\))?\s*:\s*(.+)$",
        blob,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not inline:
        return ""
    candidate = normalize_str(inline.group(1)).strip()
    candidate = re.sub(r"(?is)\n\s*(assessment|action)\s*:.*$", "", candidate).strip()
    candidate = candidate.strip(" \"“”'")
    if _contains_control_payload_markers(candidate):
        return ""
    if _contains_triage_scaffold_text(candidate):
        return ""
    return candidate


def _remove_label_prefix(text: Any) -> str:
    out = normalize_str(text).strip()
    if not out:
        return ""
    # Remove wrappers like: Comment (≈115 words): ... / Reply (2-4 sentences): ...
    out = re.sub(
        r"^\s*(?:comment|reply|draft)\s*(?:\([^)]{0,60}\))?\s*:\s*",
        "",
        out,
        flags=re.IGNORECASE,
    ).strip()
    return out


def _strip_outer_quotes(text: Any) -> str:
    out = normalize_str(text).strip()
    if len(out) < 2:
        return out
    pairs = (
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    )
    for left, right in pairs:
        if out.startswith(left) and out.endswith(right):
            return out[1:-1].strip()
    return out


def _sanitize_generated_content_text(text: Any) -> str:
    out = normalize_str(text).strip()
    if not out:
        return ""
    out = out.replace("...[truncated]", "...").replace("... [truncated]", "...").replace("[truncated]", "")
    out = _remove_label_prefix(out)
    out = _strip_outer_quotes(out)
    out = out.strip()
    return out


def _sanitize_structured_response(parsed: Dict[str, Any], response_kind: str) -> Dict[str, Any]:
    kind = _normalize_response_kind(response_kind)
    out = dict(parsed)

    content = _sanitize_generated_content_text(out.get("content"))
    out["content"] = content
    if kind == "reply_triage" and _contains_triage_scaffold_text(content):
        extracted_reply = _extract_reply_from_triage_blob(content)
        if extracted_reply:
            out["content"] = _sanitize_generated_content_text(extracted_reply)
            content = normalize_str(out.get("content")).strip()
    if _contains_control_payload_markers(content):
        out["content"] = ""
        if kind in {"post_response", "reply_triage"}:
            out["should_respond"] = False
            out["response_mode"] = "none"
        elif kind == "proactive_post":
            out["should_post"] = False

    if kind in {"post_response", "reply_triage"}:
        mode = normalize_str(out.get("response_mode")).strip().lower()
        if mode == "none":
            out["should_respond"] = False
    return out


def _infer_response_kind(messages: List[Dict[str, str]]) -> str:
    joined = " ".join(normalize_str(m.get("content")).lower() for m in messages if isinstance(m, dict))
    if "response kind: reply_triage" in joined:
        return "reply_triage"
    if "response kind: proactive_post" in joined:
        return "proactive_post"
    if "prompt_changes" in joined and "code_changes" in joined and "strategy_experiments" in joined:
        return "self_improvement"
    if "return only valid json with key 'keywords'" in joined or "\"keywords\"" in joined:
        return "keyword_suggestions"
    if "should_post (bool)" in joined and "content_archetype" in joined:
        return "proactive_post"
    if (
        "triaging a reply/comment on our thread" in joined
        or "triaging reply comments for our moltbook agent" in joined
        or "vote_target (top_comment|none)" in joined
        or "vote_target(top_comment|none)" in joined
        or ("response_mode(comment|none)" in joined and "vote_action(upvote|none)" in joined)
    ):
        return "reply_triage"
    if "should_respond (bool)" in joined and "response_mode" in joined:
        return "post_response"
    return "generic"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    text = normalize_str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    text = normalize_str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except Exception:
        return default


def _normalize_kv_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", normalize_str(value).strip().lower()).strip("_")


def _parse_key_value_lines(raw: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for line in normalize_str(raw).splitlines():
        text = normalize_str(line).strip()
        if not text:
            continue
        text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
        text = text.replace("**", "")
        match = re.match(r"^([A-Za-z][A-Za-z0-9 _/\-]{1,60})\s*[:=]\s*(.+)$", text)
        if not match:
            continue
        key = _normalize_kv_key(match.group(1))
        value = normalize_str(match.group(2)).strip()
        if not key or not value:
            continue
        pairs[key] = value
    return pairs


def _extract_section(raw: str, label: str) -> str:
    pattern = re.compile(
        rf"(?is)\b{re.escape(label)}(?:\s*\([^)]{{0,40}}\))?\s*:\s*(.+?)(?=\n\s*(?:\*\*)?[A-Za-z][A-Za-z0-9 _/\-]{{1,40}}(?:\*\*)?\s*:|\Z)"
    )
    match = pattern.search(raw)
    if not match:
        return ""
    return normalize_str(match.group(1)).strip()


def _extract_heading_title(raw: str) -> str:
    for line in normalize_str(raw).splitlines():
        text = normalize_str(line).strip()
        if not text:
            continue
        if text.startswith("#"):
            return normalize_str(text.lstrip("#").strip())
    return ""


def _strip_key_value_lines(raw: str) -> str:
    kept: List[str] = []
    for line in normalize_str(raw).splitlines():
        text = normalize_str(line).strip()
        if not text:
            kept.append("")
            continue
        no_md = text.replace("**", "")
        if re.match(r"^[A-Za-z][A-Za-z0-9 _/\-]{1,60}\s*[:=]\s*.+$", no_md):
            continue
        if _contains_control_payload_markers(no_md):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _parse_topic_tags(value: Any) -> List[str]:
    text = normalize_str(value).strip().lower()
    if not text:
        return []
    text = text.replace("[", "").replace("]", "")
    parts = re.split(r"[,/|;]", text)
    out: List[str] = []
    seen = set()
    for part in parts:
        tag = re.sub(r"\s+", " ", part.strip().lstrip("#"))
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= 8:
            break
    return out


def _normalize_archetype(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    if text in PROACTIVE_ARCHETYPES:
        return text
    if "walkthrough" in text:
        return "implementation_walkthrough"
    if "comparison" in text:
        return "chain_comparison"
    if "misconception" in text or "myth" in text:
        return "misconception_correction"
    if "use_case" in text or "usecase" in text:
        return "use_case_breakdown"
    return "use_case_breakdown"


def _parse_vote_action(raw: str, kv: Dict[str, str]) -> str:
    explicit = normalize_str(kv.get("vote_action")).strip().lower()
    if explicit in {"upvote", "none"}:
        return explicit
    lower = normalize_str(raw).lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", lower)
    clean = re.sub(r"\s+", " ", clean)
    if re.search(r"\b(do not|don t|dont|no)\s+upvote\b", clean):
        return "none"
    if re.search(r"\bupvote\b", clean):
        return "upvote"
    return "none"


def _coerce_reply_triage(raw: str, kv: Dict[str, str]) -> Dict[str, Any]:
    lower = normalize_str(raw).lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", lower)
    clean = re.sub(r"\s+", " ", clean)
    reply_text = (
        _extract_section(raw, "reply")
        or _extract_reply_from_triage_blob(raw)
        or normalize_str(kv.get("content")).strip()
    )
    reply_text = _sanitize_generated_content_text(reply_text)
    if _contains_triage_scaffold_text(reply_text):
        reply_text = ""
    no_reply = bool(
        re.search(r"\b(do not|don t|dont|no)\s+reply\b", clean)
        or re.search(r"\breply\s*:\s*optional\b", lower)
        or "reply: n/a" in lower
    )
    should_respond = _coerce_bool(kv.get("should_respond"), default=False)
    if not should_respond:
        if reply_text and not no_reply and len(reply_text) >= 18:
            should_respond = True
        elif re.search(r"\b(spam|irrelevant|off-topic|off topic)\b", lower):
            should_respond = False
    if no_reply:
        should_respond = False
        reply_text = ""
    vote_action = _parse_vote_action(raw, kv)
    confidence = _coerce_float(kv.get("confidence"), default=0.72 if should_respond else 0.86)
    response_mode = normalize_str(kv.get("response_mode")).strip().lower()
    if response_mode not in {"comment", "none"}:
        response_mode = "comment" if should_respond else "none"
    if response_mode == "none":
        should_respond = False
        reply_text = ""
    return {
        "should_respond": bool(should_respond and bool(reply_text)),
        "confidence": confidence,
        "response_mode": response_mode if reply_text else "none",
        "title": _sanitize_generated_content_text(normalize_str(kv.get("title")).strip()),
        "content": _clip_text(reply_text, 900),
        "followups": [],
        "vote_action": vote_action,
        "vote_target": "top_comment" if vote_action == "upvote" else "none",
    }


def _coerce_proactive_post(raw: str, kv: Dict[str, str]) -> Dict[str, Any]:
    title = _sanitize_generated_content_text(normalize_str(kv.get("title")).strip()) or _extract_heading_title(raw).strip()
    body = _sanitize_generated_content_text(normalize_str(kv.get("content")).strip())
    if not body:
        body = _strip_key_value_lines(raw)
        if title and body.startswith("#"):
            body_lines = body.splitlines()
            if body_lines:
                body = "\n".join(body_lines[1:]).strip()
    if not title:
        first_sentence = re.split(r"[.!?]\s+", body, maxsplit=1)[0].strip()
        title = _clip_text(first_sentence or "Ergo for autonomous agent economies", 110)
    should_post = _coerce_bool(kv.get("should_post"), default=bool(body))
    if "do not post" in normalize_str(raw).lower():
        should_post = False
    return {
        "should_post": bool(should_post and bool(body)),
        "confidence": _coerce_float(kv.get("confidence"), default=0.72 if should_post else 0.4),
        "submolt": normalize_str(kv.get("submolt")).strip() or "general",
        "title": _sanitize_generated_content_text(_clip_text(title, 140)),
        "content": _clip_text(body, 3200),
        "strategy_notes": normalize_str(kv.get("strategy_notes")).strip(),
        "topic_tags": _parse_topic_tags(kv.get("topic_tags")),
        "content_archetype": _normalize_archetype(kv.get("content_archetype")),
    }


def _coerce_post_response(raw: str, kv: Dict[str, str]) -> Dict[str, Any]:
    lower = normalize_str(raw).lower()
    content = normalize_str(kv.get("content")).strip()
    if not content:
        content = _strip_key_value_lines(raw)
    if _contains_control_payload_markers(content):
        content = ""
    if _contains_triage_scaffold_text(content):
        extracted = _extract_reply_from_triage_blob(content)
        content = extracted if extracted else ""
    content = _sanitize_generated_content_text(content)
    should = _coerce_bool(kv.get("should_respond"), default=bool(content))
    if re.search(r"\b(do not|don't|dont|no)\s+(respond|reply)\b", lower):
        should = False
    response_mode = normalize_str(kv.get("response_mode")).strip().lower()
    if response_mode not in {"comment", "post", "both", "none"}:
        response_mode = "comment" if should else "none"
    vote_action = normalize_str(kv.get("vote_action")).strip().lower()
    if vote_action not in {"upvote", "downvote", "none"}:
        vote_action = "none"
    vote_target = normalize_str(kv.get("vote_target")).strip().lower()
    if vote_target not in {"post", "top_comment", "both", "none"}:
        vote_target = "none"
    return {
        "should_respond": bool(should and bool(content)),
        "confidence": _coerce_float(kv.get("confidence"), default=0.66 if should else 0.4),
        "response_mode": response_mode if content else "none",
        "title": _sanitize_generated_content_text(normalize_str(kv.get("title")).strip())
        or _sanitize_generated_content_text(_clip_text(_extract_heading_title(raw), 140)),
        "content": _clip_text(content, 3200),
        "followups": [],
        "vote_action": vote_action,
        "vote_target": vote_target,
    }


def _coerce_keyword_suggestions(raw: str, kv: Dict[str, str]) -> Dict[str, Any]:
    raw_value = normalize_str(kv.get("keywords")).strip()
    candidates: List[str] = []
    if raw_value:
        for piece in re.split(r"[,;|\n]", raw_value):
            text = normalize_str(piece).strip().strip("-*")
            if text:
                candidates.append(text)
    for line in normalize_str(raw).splitlines():
        text = normalize_str(line).strip().lstrip("-* ").strip()
        if not text:
            continue
        if ":" in text and _normalize_kv_key(text.split(":", 1)[0]) in {"keywords", "keyword"}:
            continue
        if len(text) > 64:
            continue
        if text.lower() in {"keywords", "keyword"}:
            continue
        if re.search(r"[A-Za-z0-9]", text):
            candidates.append(text)
    cleaned: List[str] = []
    seen = set()
    for item in candidates:
        key = normalize_str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
        if len(cleaned) >= 24:
            break
    return {"keywords": cleaned}


def _coerce_self_improvement(raw: str, kv: Dict[str, str]) -> Dict[str, Any]:
    summary = normalize_str(kv.get("summary")).strip()
    if not summary:
        summary = _clip_text(_strip_key_value_lines(raw).replace("\n", " "), 280)
    priority = normalize_str(kv.get("priority")).strip().lower()
    if priority not in {"low", "medium", "high"}:
        priority = "medium"
    return {
        "summary": summary or "No novel actionable suggestions this cycle.",
        "priority": priority,
        "prompt_changes": [],
        "code_changes": [],
        "strategy_experiments": [],
    }


def _coerce_non_json_object(raw: str, response_kind: str) -> Dict[str, Any]:
    kind = _normalize_response_kind(response_kind)
    kv = _parse_key_value_lines(raw)
    if kind == "reply_triage":
        return _coerce_reply_triage(raw, kv)
    if kind == "proactive_post":
        return _coerce_proactive_post(raw, kv)
    if kind == "keyword_suggestions":
        return _coerce_keyword_suggestions(raw, kv)
    if kind == "self_improvement":
        return _coerce_self_improvement(raw, kv)
    if kind == "post_response":
        return _coerce_post_response(raw, kv)
    return _coerce_post_response(raw, kv)


def _parse_json_object_lenient(text: str, response_kind: str = "generic") -> Dict[str, Any]:
    raw = normalize_str(text).lstrip("\ufeff").strip()
    if not raw:
        raise RuntimeError("Chatbase returned empty text response")

    # 1) strict parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _sanitize_structured_response(parsed, response_kind=response_kind)
    except Exception:
        pass

    # 2) fenced block parse
    fenced = _extract_first_fenced_block(raw)
    if fenced:
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return _sanitize_structured_response(parsed, response_kind=response_kind)
        except Exception:
            pass

    # 3) first balanced object parse
    balanced = _extract_first_balanced_json_object(raw)
    if balanced:
        try:
            parsed = json.loads(balanced)
            if isinstance(parsed, dict):
                return _sanitize_structured_response(parsed, response_kind=response_kind)
        except Exception:
            pass

    coerced = _coerce_non_json_object(raw, response_kind=response_kind)
    if isinstance(coerced, dict) and coerced:
        return coerced

    preview = _clip_text(raw.replace("\n", " "), 320)
    raise RuntimeError(f"Chatbase returned non-JSON text (preview={preview})")


def call_chatbase(cfg: Config, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not cfg.chatbase_api_key:
        raise RuntimeError("CHATBASE_API_KEY not set")
    if not cfg.chatbase_chatbot_id:
        raise RuntimeError("CHATBASE_CHATBOT_ID (or CHATBASE_AGENT_ID) not set")

    url = f"{cfg.chatbase_base_url}/chat"
    headers = {
        "Authorization": f"Bearer {cfg.chatbase_api_key}",
        "Content-Type": "application/json",
    }
    compacted_messages = _compact_messages(messages, MAX_CHATBASE_MESSAGE_CHARS)
    response_kind = _infer_response_kind(compacted_messages)
    prompt_tokens_est = _estimate_tokens_from_messages(compacted_messages)
    payload = {
        "messages": compacted_messages,
        "chatbotId": cfg.chatbase_chatbot_id,
        "chatId": cfg.chatbase_chatbot_id,
        "stream": False,
        "temperature": cfg.openai_temperature,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Chatbase error {resp.status_code}: {resp.text}")

    raw_text = ""
    usage_seed: Dict[str, Optional[int]] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    try:
        data = resp.json()
        raw_text = _extract_chatbase_text(data)
        usage_seed = _extract_usage(data)
    except Exception:
        raw_text = normalize_str(resp.text).strip()
        if not raw_text:
            raise RuntimeError("Chatbase returned empty response body")

    usage = _finalize_usage(
        usage=usage_seed,
        prompt_tokens_est=prompt_tokens_est,
        completion_text=raw_text,
        source_hint="chatbase_api",
    )
    return _parse_json_object_lenient(raw_text, response_kind=response_kind), usage


def call_generation_model(cfg: Config, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    provider = cfg.llm_provider
    response_kind = _infer_response_kind(messages)
    def can_chatbase() -> bool:
        return bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id)

    def can_openai() -> bool:
        return bool(cfg.openai_api_key)

    if provider == "chatbase":
        model_hint = cfg.chatbase_chatbot_id
    elif provider == "openai":
        model_hint = cfg.openai_model
    else:
        if can_chatbase():
            model_hint = f"chatbase:{cfg.chatbase_chatbot_id}"
        elif can_openai():
            model_hint = f"openai:{cfg.openai_model}"
        else:
            model_hint = "(no-provider-configured)"
    prompt_chars = sum(len(normalize_str(m.get("content"))) for m in messages)
    prompt_tokens_est = _estimate_tokens_from_messages(messages)
    provider_route = provider
    if provider == "auto":
        provider_route = "auto(chatbase-first)"
    logger.info(
        "LLM request kind=%s provider=%s model=%s messages=%s prompt_chars=%s prompt_tokens_est=%s",
        response_kind,
        provider_route,
        model_hint or "(default)",
        len(messages),
        prompt_chars,
        prompt_tokens_est,
    )

    def _log_response(provider_used: str, usage: Dict[str, Any]) -> None:
        logger.info(
            "LLM response kind=%s provider=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s token_source=%s",
            response_kind,
            provider_used,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            usage.get("source"),
        )

    if provider == "chatbase":
        if not can_chatbase():
            raise RuntimeError("LLM provider chatbase selected but CHATBASE_API_KEY/CHATBASE_CHATBOT_ID missing")
        parsed, usage = call_chatbase(cfg, messages)
        _log_response("chatbase", usage)
        return parsed, "chatbase", usage

    if provider == "openai":
        if not can_openai():
            raise RuntimeError("LLM provider openai selected but OPENAI_API_KEY missing")
        parsed, usage = call_openai(cfg, messages)
        _log_response("openai", usage)
        return parsed, "openai", usage

    # auto mode: prefer Chatbase for Ergo-domain writing.
    # OpenAI fallback is optional via MOLTBOOK_LLM_AUTO_FALLBACK_TO_OPENAI=1.
    if can_chatbase():
        try:
            parsed, usage = call_chatbase(cfg, messages)
            _log_response("chatbase", usage)
            return parsed, "chatbase", usage
        except Exception as e:
            if can_openai() and cfg.llm_auto_fallback_to_openai:
                logger.warning("LLM provider chatbase failed in auto mode; falling back to openai error=%s", e)
                parsed, usage = call_openai(cfg, messages)
                _log_response("openai", usage)
                return parsed, "openai", usage
            raise RuntimeError(
                (
                    "Chatbase failed in auto mode and OpenAI fallback is disabled. "
                    f"Set MOLTBOOK_LLM_AUTO_FALLBACK_TO_OPENAI=1 to allow fallback. Cause: {e}"
                )
            )
    if can_openai():
        parsed, usage = call_openai(cfg, messages)
        _log_response("openai", usage)
        return parsed, "openai", usage

    raise RuntimeError("No LLM provider configured (set CHATBASE_* or OPENAI_API_KEY)")


def fallback_draft() -> Dict[str, Any]:
    return {
        "should_respond": True,
        "confidence": 0.5,
        "response_mode": "comment",
        "title": "Ergo + agent economy question",
        "content": (
            "If agents want autonomy, they need contracts they can verify, not trust. "
            "On Ergo, eUTXO plus ErgoScript gives hard execution rules and transparent settlement. "
            "If you had one week to test this, what exact on-chain constraint would you enforce first?"
        ),
        "followups": [],
        "vote_action": "none",
        "vote_target": "none",
    }


def format_content(draft: Dict[str, Any]) -> str:
    content = _sanitize_generated_content_text(draft.get("content"))
    if _contains_triage_scaffold_text(content):
        extracted = _extract_reply_from_triage_blob(content)
        content = _sanitize_generated_content_text(extracted)
    if _contains_control_payload_markers(content):
        content = ""
    if not content:
        return ""
    followups = draft.get("followups") or []
    if followups:
        lines = [content]
        for item in followups:
            text = _sanitize_generated_content_text(item)
            if _contains_control_payload_markers(text):
                continue
            if not text:
                continue
            lines.append("")
            lines.append(text)
        return "\n".join([line for line in lines if line.strip()])
    return content


def sanitize_keyword(value: Any) -> Optional[str]:
    text = normalize_str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if len(text) < 3 or len(text) > 48:
        return None
    if not re.search(r"[a-z0-9]", text):
        return None
    if text in GENERIC_KEYWORD_BLACKLIST:
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

    payload_titles = _compact_recent_titles_for_prompt(unique_titles)
    payload_keywords = _compact_recent_titles_for_prompt(existing_keywords[:120])
    system = (
        "You are extracting useful keyword candidates for a Moltbook agent mission: "
        "promote and discuss Ergo (ERG), eUTXO, agent economies, AI+Web3, and related infra. "
        "Return ONLY valid JSON with key 'keywords' as an array of short lowercase phrases. "
        "Include only keywords likely to discover mission-relevant posts."
    )
    user = (
        "Given recent post titles and existing keywords, suggest up to "
        f"{max_suggestions} new keywords. Avoid duplicates, spammy terms, and overly generic words.\n\n"
        f"Existing keywords:\n{json.dumps(payload_keywords, ensure_ascii=False)}\n\n"
        f"Recent titles:\n{json.dumps(payload_titles, ensure_ascii=False)}"
    )

    parsed, _, _ = call_generation_model(
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
