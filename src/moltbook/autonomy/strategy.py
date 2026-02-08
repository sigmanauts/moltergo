from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .drafting import normalize_str
from .runtime_helpers import normalize_response_mode, normalize_submolt, post_id


def post_score(post: Dict[str, Any]) -> int:
    for key in ("score", "upvotes", "vote_score", "likes"):
        value = post.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def post_comment_count(post: Dict[str, Any]) -> int:
    for key in ("comment_count", "comments_count", "comments"):
        value = post.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


HIGH_SIGNAL_SEED_TERMS = [
    "ergo",
    "eutxo",
    "ergoscript",
    "service orchestration",
    "agent economy",
    "agent economies",
    "decentralized",
    "coordination",
    "identity",
    "infrastructure",
    "trustless escrow",
    "reputation",
]

PRIORITY_INFRA_TERMS = [
    "agent coordination",
    "coordination",
    "orchestration",
    "infrastructure",
    "runtime",
    "execution",
    "settlement",
    "escrow",
    "counterparty",
    "verification",
]

OFF_MISSION_SOFT_TERMS = [
    "3d modeling",
    "3d modelling",
    "rigging",
    "uv unwrap",
    "blender workflow",
    "sku photography",
    "product rendering",
]

ERGO_CANONICAL_TERMS = (
    ("eutxo", "eUTXO"),
    ("ergoscript", "ErgoScript"),
    ("sigma protocols", "Sigma Protocols"),
    ("rosen bridge", "Rosen Bridge"),
    ("sigusd", "SigUSD"),
)

WEEKLY_PROACTIVE_THEMES = {
    0: "Escrow and settlement discipline for autonomous agents",
    1: "Agent identity and reputation proofs for counterparties",
    2: "Service orchestration and delegation economics",
    3: "Privacy-preserving execution and Sigma Protocols",
    4: "DeFi rails for machine economies (SigUSD, Rosen Bridge, Oracle Pools)",
    5: "Builder walkthroughs and implementation tradeoffs",
    6: "Cypherpunk critiques of rent-seeking in agent infrastructure",
}

GENERIC_TEMPLATE_PATTERNS = [
    "what concrete constraint would you test first",
    "which implementation constraint would you test first on ergo",
    "which concrete ergoscript use case would you pilot first in your stack",
    "we are building toward verifiable agent economies",
    "core idea is enforceable machine agreements",
    "comment (≈",
    "reply (2-4 sentences)",
    "reply (2–4 sentences)",
    "upvote. the comment is",
]

LOW_VALUE_REPLY_PREFIXES = [
    "absolutely",
    "great point",
    "interesting point",
    "totally agree",
    "i agree",
    "noted",
]

PROACTIVE_OPENING_PAIN_MARKERS = [
    "bottleneck",
    "fragile",
    "fails",
    "failure",
    "broken",
    "slow",
    "latency",
    "cost",
    "fees",
    "spam",
    "manual",
    "trust",
    "counterparty",
    "dispute",
    "verify",
    "verification",
    "opaque",
    "censorship",
    "downtime",
]

PROACTIVE_OPENING_MECHANISM_MARKERS = [
    "eutxo",
    "ergoscript",
    "sigma",
    "rosen bridge",
    "sigusd",
    "oracle",
    "escrow",
    "reputation",
    "settlement",
    "smart contract",
]

MAX_CONSECUTIVE_DECLINES_GUARD = 6
MAX_RECOVERY_DRAFTS_PER_CYCLE = 2
RECOVERY_SIGNAL_MARGIN = 3
MAX_REPLIES_PER_AUTHOR_PER_POST = max(1, int(os.getenv("MOLTBOOK_MAX_REPLIES_PER_AUTHOR_PER_POST", "3")))
THREAD_ESCALATE_TURNS = max(3, int(os.getenv("MOLTBOOK_THREAD_ESCALATE_TURNS", "5")))

MARKET_KEYWORD_MAP = {
    "autonomous": "autonomous agents",
    "decentralized": "decentralized agents",
    "coordination": "agent coordination",
    "economy": "agent economy",
    "identity": "agent identity",
    "infrastructure": "agent infrastructure",
    "payments": "autonomous payments",
    "execution": "service execution",
    "orchestration": "service orchestration",
    "privacy": "privacy preserving",
    "escrow": "trustless escrow",
    "reputation": "agent reputation",
}

SIGNAL_TERM_BLACKLIST = {
    "here",
    "what",
    "current",
    "problem",
    "projectsubmission",
    "usdchackathon",
    "submission",
    "thread",
}


def _clean_signal_term(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s\-_/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) < 3:
        return ""
    if text in SIGNAL_TERM_BLACKLIST:
        return ""
    return text


def _snapshot_terms(snapshot: Dict[str, Any], key: str, limit: int = 16) -> List[str]:
    raw = snapshot.get(key)
    terms: List[str] = []
    if not isinstance(raw, list):
        return terms
    for item in raw:
        term = _clean_signal_term(item)
        if not term or term in terms:
            continue
        terms.append(term)
        if len(terms) >= max(1, int(limit)):
            break
    return terms


def _snapshot_term_lift_map(snapshot: Dict[str, Any], key: str, limit: int = 16) -> Dict[str, float]:
    raw = snapshot.get(key)
    out: Dict[str, float] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = _clean_signal_term(item.get("term"))
        if not term:
            continue
        try:
            lift = float(item.get("lift", 0.0) or 0.0)
        except Exception:
            lift = 0.0
        if lift == 0.0:
            continue
        out[term] = lift
        if len(out) >= max(1, int(limit)):
            break
    return out


def _snapshot_best_submolt_scores(snapshot: Dict[str, Any], limit: int = 8) -> Dict[str, float]:
    raw = snapshot.get("best_submolts")
    out: Dict[str, float] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = normalize_submolt(item.get("name"), default="")
        if not name:
            continue
        try:
            avg = float(item.get("avg_score", 0.0) or 0.0)
        except Exception:
            avg = 0.0
        out[name] = avg
        if len(out) >= max(1, int(limit)):
            break
    return out


def _market_keyword_candidates(
    market_snapshot: Dict[str, Any],
    existing_keywords: List[str],
    max_suggestions: int,
) -> List[str]:
    if not isinstance(market_snapshot, dict):
        return []
    raw_terms = market_snapshot.get("top_terms")
    if not isinstance(raw_terms, list):
        return []

    existing = {normalize_str(k).strip().lower() for k in existing_keywords if normalize_str(k).strip()}
    out: List[str] = []
    for item in raw_terms[:12]:
        token = _clean_signal_term(item)
        if not token:
            continue
        candidate = MARKET_KEYWORD_MAP.get(token, "")
        if not candidate:
            continue
        if candidate in existing or candidate in out:
            continue
        out.append(candidate)
        if len(out) >= max(1, int(max_suggestions)):
            break
    return out


def _collect_high_signal_terms(learning_snapshot: Dict[str, Any], active_keywords: List[str]) -> List[str]:
    terms: List[str] = []

    def add_term(raw: Any) -> None:
        term = _clean_signal_term(raw)
        if not term:
            return
        if term not in terms:
            terms.append(term)

    for seed in HIGH_SIGNAL_SEED_TERMS:
        add_term(seed)

    for keyword in active_keywords:
        k = _clean_signal_term(keyword)
        if not k:
            continue
        if any(token in k for token in ("ergo", "eutxo", "ergoscript", "service", "orchestration", "agent")):
            add_term(k)

    for item in _snapshot_terms(learning_snapshot, "winning_terms", limit=16):
        add_term(item)

    market_snapshot = learning_snapshot.get("market_snapshot")
    if isinstance(market_snapshot, dict):
        top_terms = market_snapshot.get("top_terms")
        if isinstance(top_terms, list):
            for item in top_terms:
                add_term(item)

    return terms[:48]


def _contains_any(blob: str, terms: List[str]) -> bool:
    return any(term in blob for term in terms)


def _normalize_ergo_terms(text: str) -> str:
    out = normalize_str(text)
    if not out.strip():
        return out
    for source, target in ERGO_CANONICAL_TERMS:
        out = re.sub(rf"\b{re.escape(source)}\b", target, out, flags=re.IGNORECASE)
    return out


def _ensure_use_case_prompt_if_relevant(content: str, post: Dict[str, Any]) -> str:
    text = normalize_str(content).strip()
    if not text:
        return text
    blob = " ".join(
        [
            normalize_str(post.get("title")).lower(),
            normalize_str(post.get("content")).lower(),
            normalize_submolt(post.get("submolt")).lower(),
        ]
    )
    if not _contains_any(blob, PRIORITY_INFRA_TERMS):
        return text
    lowered = text.lower()
    if "?" in text:
        return text
    if "use case" in lowered or "example" in lowered or "pilot" in lowered:
        return text
    suffix_options = [
        "What exact ErgoScript rule would you enforce first in production?",
        "Which one implementation constraint would you lock on-chain first?",
        "What is the first deterministic check you would encode in ErgoScript?",
    ]
    if "escrow" in blob or "counterparty" in blob or "dispute" in blob:
        suffix_options = [
            "Which single escrow or dispute rule would you enforce first in ErgoScript?",
            "What first counterparty gate would you encode in the contract?",
            "Which dispute branch should trigger first: timeout, evidence match, or arbitration?",
        ]
    elif "coordination" in blob or "orchestration" in blob:
        suffix_options = [
            "Which coordination step should be the first on-chain invariant?",
            "What first orchestration rule would you anchor in an eUTXO state machine?",
            "Which execution checkpoint should be mandatory before settlement?",
        ]
    pid = normalize_str(post_id(post)).strip()
    seed = f"{pid}:{normalize_str(post.get('title')).strip()}:{normalize_str(post.get('submolt')).strip()}"
    idx = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % max(1, len(suffix_options))
    suffix = suffix_options[idx]
    return text + "\n\n" + suffix


def _is_template_like_generated_content(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    if "```" in blob:
        return True
    if re.search(r"if you want[^\\n]{0,140}threads/profile", blob):
        return True
    if "draft post" in blob or "draft reply" in blob:
        return True
    return any(pattern in blob for pattern in GENERIC_TEMPLATE_PATTERNS)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", normalize_str(text)))


def _looks_like_control_payload_text(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    matches = re.findall(
        r"\b(should_respond|response_mode|vote_action|vote_target|confidence|should_post|content_archetype)\s*[:=]",
        blob,
    )
    if len(matches) >= 1 and _word_count(blob) <= 28:
        return True
    if len(matches) >= 2:
        return True
    return False


def _has_ergo_mechanism_mention(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    markers = (
        "eutxo",
        "ergoscript",
        "sigma",
        "rosen bridge",
        "sigusd",
        "oracle",
        "utxo",
        "escrow",
        "reputation",
        "settlement",
        "smart contract",
    )
    return any(marker in blob for marker in markers)


def _has_implementation_angle(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return False
    markers = (
        "implement",
        "integration",
        "integrate",
        "build",
        "deploy",
        "workflow",
        "execution path",
        "contract rule",
        "gate",
        "pilot",
        "step",
        "test first",
    )
    return any(marker in blob for marker in markers)


def _first_nonempty_lines(text: str, limit: int = 2) -> List[str]:
    lines = [normalize_str(line).strip() for line in normalize_str(text).splitlines() if normalize_str(line).strip()]
    return lines[: max(1, int(limit))]


def _opening_has_concrete_pain(text: str) -> bool:
    opening = " ".join(_first_nonempty_lines(text, limit=2)).lower()
    if not opening:
        return False
    return any(marker in opening for marker in PROACTIVE_OPENING_PAIN_MARKERS)


def _opening_has_ergo_mechanism(text: str) -> bool:
    opening = " ".join(_first_nonempty_lines(text, limit=2)).lower()
    if not opening:
        return False
    return any(marker in opening for marker in PROACTIVE_OPENING_MECHANISM_MARKERS)


def _passes_proactive_opening_gate(text: str) -> bool:
    return bool(_opening_has_concrete_pain(text) and _opening_has_ergo_mechanism(text))


def _ensure_proactive_opening_gate(text: str) -> str:
    content = normalize_str(text).strip()
    if not content:
        return content
    if _passes_proactive_opening_gate(content):
        return content

    lower = content.lower()
    pain_line = "Agent economies stall when counterparties cannot verify execution or settle disputes objectively."
    mechanism_line = (
        "Ergo solves this with eUTXO plus ErgoScript, which enforces deterministic settlement rules and on-chain auditability."
    )
    if "escrow" in lower or "counterparty" in lower or "dispute" in lower:
        pain_line = "Escrow workflows break when counterparties cannot prove what happened without human arbitration."
    if "coordination" in lower or "orchestration" in lower or "runtime" in lower:
        pain_line = "Service orchestration becomes fragile when parallel tasks cannot settle with deterministic rules."

    if "ergoscript" in lower:
        mechanism_line = (
            "ErgoScript contracts lock settlement logic into verifiable rules, and eUTXO keeps execution deterministic under load."
        )
    elif "sigma" in lower:
        mechanism_line = (
            "Sigma Protocols add privacy-preserving proofs while eUTXO keeps settlement deterministic across agent workflows."
        )
    elif "rosen bridge" in lower:
        mechanism_line = (
            "Rosen Bridge can move value across chains, while eUTXO contracts keep cross-chain settlement conditions deterministic."
        )

    return f"{pain_line} {mechanism_line}\n\n{content}"


def _passes_generated_content_quality(content: str, requested_mode: str) -> bool:
    text = normalize_str(content).strip()
    if not text:
        return False
    lowered = text.lower()
    if "```" in text:
        return False
    if "comment (≈" in lowered or "reply (2-4 sentences)" in lowered or "reply (2–4 sentences)" in lowered:
        return False
    if "draft post" in lowered or "draft reply" in lowered:
        return False
    if re.search(r"if you want[^\\n]{0,140}threads/profile", lowered):
        return False
    if lowered.startswith("upvote.") and "reply" in lowered:
        return False
    if _looks_like_control_payload_text(text):
        return False
    words = _word_count(text)
    has_question = "?" in text
    has_mechanism = _has_ergo_mechanism_mention(text)
    has_impl = _has_implementation_angle(text)
    mode = normalize_response_mode(requested_mode, default="comment")
    if mode in {"post", "both"}:
        return bool(has_question and has_mechanism and has_impl and 110 <= words <= 320)
    if mode == "comment":
        return bool(has_question and has_mechanism and 45 <= words <= 220)
    return True


def _build_recovery_messages(base_messages: List[Dict[str, str]], signal_score: int) -> List[Dict[str, str]]:
    recovery_prompt = (
        "Recovery pass: the previous draft was rejected for low relevance or low confidence. "
        "Write one sharper response that is still organic and non-spammy. "
        "Requirements: include one explicit Ergo mechanism, one implementation angle, and one direct question. "
        "Avoid generic phrasing and avoid stock acknowledgements. "
        f"Signal score for this candidate: {signal_score}."
    )
    messages = list(base_messages)
    messages.append({"role": "user", "content": recovery_prompt})
    return messages


def _is_low_value_affirmation_reply(text: str) -> bool:
    blob = normalize_str(text).strip().lower()
    if not blob:
        return True
    if any(blob.startswith(prefix) for prefix in LOW_VALUE_REPLY_PREFIXES):
        return True
    words = blob.split()
    if len(words) < 10:
        return True
    return False


def _compose_reference_post_content(reference_url: str, content: str) -> str:
    core = normalize_str(content).strip()
    url = normalize_str(reference_url).strip()
    if not core:
        return ""
    if not url or url in core:
        return core
    return f"{core}\n\nContext thread: {url}"


def _post_mechanism_score(post: Dict[str, Any]) -> int:
    blob = " ".join(
        [
            normalize_str(post.get("title")).strip().lower(),
            normalize_str(post.get("content")).strip().lower(),
            normalize_submolt(post.get("submolt")).strip().lower(),
        ]
    )
    score = 0
    if "eutxo" in blob:
        score += 2
    if "ergoscript" in blob:
        score += 2
    if "sigma" in blob or "privacy" in blob:
        score += 1
    if "rosen" in blob or "bridge" in blob:
        score += 1
    if "sigusd" in blob or "oracle" in blob:
        score += 1
    return score


def _has_trending_overlap(post: Dict[str, Any], trending_terms: List[str]) -> bool:
    if not trending_terms:
        return False
    blob = " ".join(
        [
            normalize_str(post.get("title")).strip().lower(),
            normalize_str(post.get("content")).strip().lower(),
            normalize_submolt(post.get("submolt")).strip().lower(),
        ]
    )
    for term in trending_terms[:8]:
        t = _clean_signal_term(term)
        if len(t) < 4:
            continue
        if t in blob:
            return True
    return False


def _trending_terms_for_post(post: Dict[str, Any], trending_terms: List[str], max_terms: int = 4) -> List[str]:
    if not trending_terms:
        return []
    blob = " ".join(
        [
            normalize_str(post.get("title")).strip().lower(),
            normalize_str(post.get("content")).strip().lower(),
            normalize_submolt(post.get("submolt")).strip().lower(),
        ]
    )
    matched: List[str] = []
    for term in trending_terms:
        clean = _clean_signal_term(term)
        if len(clean) < 4:
            continue
        if clean in blob and clean not in matched:
            matched.append(clean)
        if len(matched) >= max(1, int(max_terms)):
            break
    if matched:
        return matched
    fallback: List[str] = []
    for term in trending_terms:
        clean = _clean_signal_term(term)
        if len(clean) >= 4 and clean not in fallback:
            fallback.append(clean)
        if len(fallback) >= 2:
            break
    return fallback


def _comment_priority_score(comment: Dict[str, Any]) -> int:
    body = normalize_str(comment.get("content")).strip().lower()
    if not body:
        return -10
    score = 0
    if "?" in body:
        score += 3
    if any(token in body for token in ("eutxo", "ergoscript", "ergo", "escrow", "reputation", "coordination")):
        score += 4
    if len(body.split()) >= 12:
        score += 1
    if len(body.split()) > 80:
        score -= 2
    raw_score = comment.get("score")
    if isinstance(raw_score, (int, float)):
        score += int(raw_score)
    return score


def _post_relevance_score(
    post: Dict[str, Any],
    signal_terms: List[str],
    winning_terms: List[str],
    losing_terms: List[str],
    term_lift_map: Dict[str, float],
    best_submolt_scores: Dict[str, float],
) -> int:
    title = normalize_str(post.get("title")).strip().lower()
    content = normalize_str(post.get("content")).strip().lower()
    submolt = normalize_submolt(post.get("submolt")).lower()
    blob = " ".join([title, content, submolt])
    score = 0

    for term in signal_terms:
        if term in title:
            score += 4
            continue
        if term in content:
            score += 1
            continue
        if term in submolt:
            score += 1

    positive_bonus = 0
    for term in winning_terms:
        if term in title:
            positive_bonus += 3
        elif term in content:
            positive_bonus += 1
    score += min(10, positive_bonus)

    negative_penalty = 0
    for term in losing_terms:
        if term in title:
            negative_penalty += 3
        elif term in content:
            negative_penalty += 1
    score -= min(10, negative_penalty)

    lift_bonus = 0.0
    positive_lift_hits = 0
    negative_lift_hits = 0
    for term, lift in term_lift_map.items():
        matched = False
        if term in title:
            lift_bonus += lift * 6.0
            matched = True
        elif term in content:
            lift_bonus += lift * 3.0
            matched = True
        elif term in submolt:
            lift_bonus += lift * 2.0
            matched = True
        if matched:
            if lift > 0:
                positive_lift_hits += 1
            elif lift < 0:
                negative_lift_hits += 1
    score += int(round(max(-12.0, min(12.0, lift_bonus))))
    if negative_lift_hits >= 2 and positive_lift_hits == 0:
        score -= min(8, 2 + negative_lift_hits * 2)
    elif positive_lift_hits >= 2 and negative_lift_hits == 0:
        score += min(6, positive_lift_hits)

    if "eutxo" in blob:
        score += 4
    if "ergoscript" in blob:
        score += 3
    if "service orchestration" in blob or ("service" in blob and "orchestration" in blob):
        score += 4
    if "agent economy" in blob or "agent economies" in blob:
        score += 2
    if _contains_any(blob, PRIORITY_INFRA_TERMS):
        score += 3
    if _contains_any(blob, OFF_MISSION_SOFT_TERMS):
        score -= 4

    # Avoid broad "market chatter" candidates that carry no implementable Ergo angle.
    implementation_markers = (
        "eutxo",
        "ergoscript",
        "sigma",
        "escrow",
        "settlement",
        "counterparty",
        "oracle",
        "proof",
        "signature",
        "hash",
        "contract rule",
        "timeout",
        "utxo",
    )
    if _contains_any(blob, signal_terms) and not any(marker in blob for marker in implementation_markers):
        score -= 5

    submolt_avg = float(best_submolt_scores.get(submolt, 0.0) or 0.0)
    if submolt_avg >= 20:
        score += 3
    elif submolt_avg >= 10:
        score += 2
    elif submolt_avg >= 5:
        score += 1

    score += min(3, post_comment_count(post) // 8)
    score += min(2, post_score(post) // 15)
    return score


def _rank_posts_for_drafting(
    posts: List[Dict[str, Any]],
    learning_snapshot: Dict[str, Any],
    active_keywords: List[str],
    virality_scores: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    signal_terms = _collect_high_signal_terms(learning_snapshot=learning_snapshot, active_keywords=active_keywords)
    winning_terms = _snapshot_terms(learning_snapshot, "winning_terms", limit=16)
    losing_terms = _snapshot_terms(learning_snapshot, "losing_terms", limit=16)
    winning_lifts = _snapshot_term_lift_map(learning_snapshot, "winning_terms_lift", limit=16)
    losing_lifts = _snapshot_term_lift_map(learning_snapshot, "losing_terms_lift", limit=16)
    term_lift_map: Dict[str, float] = {}
    for term, lift in winning_lifts.items():
        term_lift_map[term] = max(term_lift_map.get(term, 0.0), lift)
    for term, lift in losing_lifts.items():
        term_lift_map[term] = min(term_lift_map.get(term, 0.0), lift)
    best_submolt_scores = _snapshot_best_submolt_scores(learning_snapshot, limit=8)
    scored: List[Tuple[int, int, int, int, Dict[str, Any]]] = []
    score_map: Dict[str, int] = {}

    virality_scores = virality_scores or {}
    for idx, post in enumerate(posts):
        score = _post_relevance_score(
            post,
            signal_terms=signal_terms,
            winning_terms=winning_terms,
            losing_terms=losing_terms,
            term_lift_map=term_lift_map,
            best_submolt_scores=best_submolt_scores,
        )
        pid = post_id(post) or f"idx:{idx}"
        virality = float(virality_scores.get(pid, 0.0) or 0.0)
        virality_boost = int(round(max(-0.8, min(1.8, virality)) * 10))
        score += virality_boost
        score_map[pid] = score
        scored.append((score, virality, post_score(post), post_comment_count(post), -idx, post))

    scored.sort(reverse=True)
    ordered = [item[5] for item in scored]
    return ordered, score_map, signal_terms


def _adaptive_draft_controls(cfg: Config, state: Dict[str, Any]) -> Tuple[int, int, str]:
    base_shortlist = max(1, int(cfg.draft_shortlist_size))
    base_signal = max(0, int(cfg.draft_signal_min_score))
    min_shortlist = max(1, int(cfg.dynamic_shortlist_min))
    max_shortlist = max(min_shortlist, int(cfg.dynamic_shortlist_max))
    if not cfg.dynamic_shortlist_enabled:
        return base_shortlist, base_signal, "disabled"

    history_raw = state.get("cycle_metrics_history", [])
    history_all = [item for item in history_raw if isinstance(item, dict)]
    history = history_all[-6:]
    if not history:
        shortlist = max(min_shortlist, min(base_shortlist, max_shortlist))
        return shortlist, base_signal, "cold_start"

    approval_rates: List[float] = []
    execution_rates: List[float] = []
    cooldown_pressure = 0
    low_signal_pressure = 0
    trend_context_mismatch_pressure = 0
    decline_guard_hits = 0
    for entry in history:
        drafted = int(entry.get("drafted", 0) or 0)
        model_approved = int(entry.get("model_approved", 0) or 0)
        eligible = int(entry.get("eligible_now", 0) or 0)
        actions = int(entry.get("actions", 0) or 0)
        approval_rates.append(float(model_approved) / max(1.0, float(drafted)))
        execution_rates.append(float(actions) / max(1.0, float(eligible)))
        skip = entry.get("skip_reasons")
        if isinstance(skip, dict):
            cooldown_pressure += int(skip.get("post_cooldown+comment_cooldown", 0) or 0)
            cooldown_pressure += int(skip.get("no_action_slots", 0) or 0)
            low_signal_pressure += int(skip.get("low_signal_relevance", 0) or 0)
            trend_context_mismatch_pressure += int(skip.get("trend_context_mismatch", 0) or 0)
            decline_guard_hits += int(skip.get("consecutive_declines_guard", 0) or 0)

    avg_approval = sum(approval_rates) / max(1, len(approval_rates))
    avg_execution = sum(execution_rates) / max(1, len(execution_rates))
    zero_action_streak = 0
    for entry in reversed(history_all):
        if int(entry.get("actions", 0) or 0) == 0:
            zero_action_streak += 1
            continue
        break
    shortlist = base_shortlist
    signal = base_signal
    mode = "steady"

    if zero_action_streak >= 8 and avg_approval < 0.12:
        shortlist = max(min_shortlist, min(base_shortlist, 5))
        signal = min(max(0, base_signal + 2), 8)
        mode = "conversion_collapse"
    elif zero_action_streak >= 4:
        shortlist = max(min_shortlist, min(base_shortlist, 6))
        signal = min(max(0, base_signal + 1), 8)
        mode = "streak_recovery"
    elif decline_guard_hits >= 2 and avg_approval < 0.12:
        shortlist = max(min_shortlist, min(base_shortlist, 7))
        signal = min(max(0, base_signal + 1), 8)
        mode = "decline_recovery"
    elif trend_context_mismatch_pressure >= 10 and avg_approval <= 0.05:
        shortlist = min(max_shortlist, max(base_shortlist, int(round(base_shortlist * 1.15))))
        signal = max(0, signal - 1)
        mode = "relax_context_gate"
    elif cooldown_pressure >= 25:
        shortlist = max(min_shortlist, min(shortlist, 8))
        signal = min(signal + 1, 8)
        mode = "cooldown_pressure"
    elif low_signal_pressure >= 40 and avg_approval >= 0.2:
        shortlist = min(max_shortlist, max(shortlist, int(round(base_shortlist * 1.15))))
        signal = max(1, signal - 1)
        mode = "relax_low_signal"
    elif avg_approval < 0.12 and avg_execution < 0.08:
        shortlist = max(min_shortlist, int(round(base_shortlist * 0.65)))
        signal = min(signal + 1, 8)
        mode = "tighten_quality"
    elif avg_approval > 0.35 and avg_execution > 0.18:
        shortlist = min(max_shortlist, max(base_shortlist, int(round(base_shortlist * 1.25))))
        signal = max(1, signal - 1)
        mode = "expand_capture"

    shortlist = max(min_shortlist, min(shortlist, max_shortlist))
    signal = max(0, signal)
    return shortlist, signal, mode
