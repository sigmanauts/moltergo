from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .config import Config
from .drafting import normalize_str


def has_generation_provider(cfg: Config) -> bool:
    if cfg.llm_provider == "chatbase":
        return bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id)
    if cfg.llm_provider == "openai":
        return bool(cfg.openai_api_key)
    return bool((cfg.chatbase_api_key and cfg.chatbase_chatbot_id) or cfg.openai_api_key)


def sanitize_generated_title(title: Any, fallback: str = "Ergo implementation question") -> str:
    text = normalize_str(title).strip()
    text = text.replace("...[truncated]", "...").replace("... [truncated]", "...").replace("[truncated]", "").strip()
    # Strip common LLM scaffolding that should never be published in titles.
    text = re.sub(r"(?i)^\s*draft\s+(?:post|reply)\b\s*[:\\-–—]*\s*", "", text).strip()
    text = re.sub(r"(?i)\s*[:\\-–—]?\s*draft\s+(?:post|reply)\b\s*(?:\\([^)]{0,120}\\))?\s*", " ", text).strip()
    text = re.sub(r"(?i)^\s*(?:comment|reply)\s*\\([^)]{0,120}\\)\s*[:\\-–—]*\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" -–—:;,.")
    if not text:
        text = fallback
    if len(text) > 120:
        text = text[:120].rstrip()
    if len(text) < 10:
        text = fallback
    return text


_MECHANISM_TERMS = ("eutxo", "ergoscript", "sigma", "rosen", "sigusd", "oracle", "escrow", "settlement")
_ARCHETYPE_HINTS: Dict[str, Tuple[str, ...]] = {
    "security_advisory": ("threat", "mitigation", "attack", "proof"),
    "build_log": ("build", "broke", "fixed", "result"),
    "mechanism_explainer": ("mechanism", "example", "constraint", "trade-off"),
    "operator_reliability": ("checklist", "reliability", "uptime", "failure"),
    "myth_correction": ("myth", "claim", "evidence", "correction"),
    "agent_economy_teardown": ("incentive", "cost", "market", "payout"),
    "use_case_breakdown": ("workflow", "use case", "implementation", "step"),
    "misconception_correction": ("myth", "misconception", "correction", "evidence"),
    "chain_comparison": ("comparison", "trade-off", "latency", "fees"),
    "implementation_walkthrough": ("implementation", "step", "deploy", "constraint"),
}


def _compact_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", normalize_str(text).lower())


def _split_paragraphs(text: str) -> List[str]:
    parts = [normalize_str(x).strip() for x in normalize_str(text).split("\n\n")]
    return [x for x in parts if x]


def _build_title_candidates(title: str, content: str) -> List[str]:
    base = sanitize_generated_title(title, fallback="Ergo mechanism for agent workflows")
    first_para = _split_paragraphs(content)[0] if _split_paragraphs(content) else ""
    mechanism = "eUTXO" if "eutxo" in first_para.lower() else "ErgoScript"
    candidates = [
        base,
        sanitize_generated_title(f"{mechanism} in production: {base}", fallback=base),
        sanitize_generated_title(base.rstrip("?.!") + "?", fallback=base),
    ]
    out: List[str] = []
    seen = set()
    for item in candidates:
        clean = sanitize_generated_title(item, fallback=base)
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= 3:
            break
    return out


def _build_lead_candidates(content: str) -> List[str]:
    paragraphs = _split_paragraphs(content)
    original = paragraphs[0] if paragraphs else ""
    lower = normalize_str(content).lower()
    pain = "Agent workflows break when settlement depends on trust and manual arbitration."
    if "coordination" in lower or "orchestration" in lower:
        pain = "Coordination collapses when distributed tasks cannot settle under deterministic rules."
    mechanism = "Ergo eUTXO plus ErgoScript keeps execution deterministic and payout rules auditable on-chain."
    if "sigma" in lower:
        mechanism = "Sigma proofs can protect sensitive workflow data while Ergo enforces deterministic settlement rules."
    practical = (
        "One practical path is escrow-by-default: lock funds, require objective evidence checks, and release only when contract constraints pass."
    )
    candidates = [original, f"{pain} {mechanism}", f"{pain} {practical}"]
    out: List[str] = []
    seen = set()
    for item in candidates:
        text = normalize_str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= 3:
            break
    return out


def _score_title_candidate(title: str, reference_terms: List[str], archetype: str) -> float:
    text = normalize_str(title).strip()
    lower = text.lower()
    words = _compact_words(text)
    score = 0.0
    if 20 <= len(text) <= 115:
        score += 2.0
    if 5 <= len(words) <= 18:
        score += 1.5
    if any(term in lower for term in _MECHANISM_TERMS):
        score += 1.8
    overlap = sum(1 for t in reference_terms if t and t in lower)
    score += min(2.5, overlap * 0.6)
    hints = _ARCHETYPE_HINTS.get(archetype, ())
    score += min(1.5, sum(1 for h in hints if h in lower) * 0.5)
    if "?" in text:
        score += 0.4
    if re.search(r"\b(amazing|ultimate|secret|guaranteed)\b", lower):
        score -= 2.0
    return score


def _score_lead_candidate(lead: str, reference_terms: List[str], archetype: str) -> float:
    text = normalize_str(lead).strip()
    lower = text.lower()
    words = _compact_words(text)
    score = 0.0
    if 35 <= len(words) <= 95:
        score += 2.0
    if any(term in lower for term in _MECHANISM_TERMS):
        score += 2.2
    if any(token in lower for token in ("constraint", "trade-off", "risk", "failure", "bottleneck")):
        score += 1.2
    overlap = sum(1 for t in reference_terms if t and t in lower)
    score += min(2.0, overlap * 0.5)
    hints = _ARCHETYPE_HINTS.get(archetype, ())
    score += min(1.5, sum(1 for h in hints if h in lower) * 0.5)
    if "?" in text:
        score += 0.2
    return score


def select_best_hook_pair(
    *,
    title: str,
    content: str,
    reference_text: str,
    archetype: str,
) -> Tuple[str, str, Dict[str, Any]]:
    base_title = sanitize_generated_title(title, fallback="Ergo mechanism for autonomous workflows")
    base_content = normalize_str(content).strip()
    if not base_content:
        return base_title, base_content, {"title_candidates": [], "lead_candidates": [], "selected": "base"}

    reference_terms = sorted(set(_compact_words(reference_text)))[:24]
    title_candidates = _build_title_candidates(base_title, base_content)
    lead_candidates = _build_lead_candidates(base_content)
    if not title_candidates:
        title_candidates = [base_title]
    if not lead_candidates:
        lead_candidates = [_split_paragraphs(base_content)[0] if _split_paragraphs(base_content) else base_content]

    scored_titles = sorted(
        [(cand, _score_title_candidate(cand, reference_terms, archetype)) for cand in title_candidates],
        key=lambda kv: kv[1],
        reverse=True,
    )
    scored_leads = sorted(
        [(cand, _score_lead_candidate(cand, reference_terms, archetype)) for cand in lead_candidates],
        key=lambda kv: kv[1],
        reverse=True,
    )
    best_title = scored_titles[0][0]
    best_lead = scored_leads[0][0]

    paragraphs = _split_paragraphs(base_content)
    if paragraphs:
        paragraphs[0] = best_lead
        optimized_content = "\n\n".join(paragraphs)
    else:
        optimized_content = best_lead

    meta = {
        "title_candidates": [{"text": t, "score": round(s, 3)} for t, s in scored_titles],
        "lead_candidates": [{"text": l, "score": round(s, 3)} for l, s in scored_leads],
        "selected": "optimized",
    }
    return best_title, optimized_content, meta
