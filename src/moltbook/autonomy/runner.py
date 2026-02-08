from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..moltbook_client import MoltbookAuthError, MoltbookClient
from ..virality import infer_topic_signature, score_post_candidate, summarize_sources

from .actions import (
    can_reply,
    cooldown_remaining_seconds,
    execute_pending_actions,
    has_pending_comment_action,
    mark_reply_action_timestamps,
    maybe_upvote_post_after_comment,
    seconds_since_last_post,
    should_prioritize_proactive_post,
    wait_for_comment_slot,
)
from .action_journal import append_action_journal
from .analytics import (
    aggregate_skip_reasons,
    daily_summary,
    init_analytics_db,
    record_action_event,
    refresh_post_metrics,
)
from .config import Config, load_config
from .content_policy import (
    build_badbot_warning_reply,
    build_hostile_refusal_reply,
    build_thread_followup_post_content,
    build_thread_followup_post_title,
    build_wrong_community_correction_reply,
    enforce_link_policy,
    is_strong_ergo_post,
    looks_hostile_content,
    looks_irrelevant_noise_comment,
    is_overt_spam_comment,
    looks_spammy_comment,
    should_correct_wrong_community_claim,
    top_badbots,
)
from .discovery import (
    discover_posts,
    discover_relevant_submolts,
)
from .drafting import (
    build_reply_triage_messages,
    build_proactive_post_messages,
    build_openai_messages,
    call_generation_model,
    fallback_draft,
    format_content,
    load_context_text,
    load_persona_text,
    normalize_str,
    post_url,
    propose_keywords_from_titles,
    sanitize_publish_content,
)
from .keywords import load_keyword_store, merge_keywords, save_keyword_store
from .logging_utils import setup_logging
from .generation_utils import (
    has_generation_provider,
    sanitize_generated_title,
    select_best_hook_pair,
)
from .post_engine_memory import (
    build_learning_snapshot,
    load_post_engine_memory,
    record_declined_idea,
    record_proactive_post,
    refresh_metrics_from_recent_posts,
    update_market_signals,
    save_post_engine_memory,
)
from .runtime_helpers import (
    _prune_comment_action_timestamps,
    _normalized_name_key,
    author_identity_key,
    comment_author,
    comment_gate_status,
    comment_id,
    comment_parent_id,
    comment_score,
    currently_allowed_response_modes,
    extract_comments,
    extract_my_vote_from_comment,
    extract_posts,
    extract_single_post,
    is_self_author,
    normalize_response_mode,
    normalize_submolt,
    normalize_vote_action,
    normalize_vote_target,
    planned_actions,
    post_author,
    post_gate_status,
    post_id,
    preview_text,
    register_my_comment_id,
    resolve_self_identity_keys,
    submolt_name_from_post,
)
from .state import load_state, reset_daily_if_needed, save_state, utc_now
from .strategy import (
    MAX_CONSECUTIVE_DECLINES_GUARD,
    MAX_RECOVERY_DRAFTS_PER_CYCLE,
    MAX_REPLIES_PER_AUTHOR_PER_POST,
    RECOVERY_SIGNAL_MARGIN,
    THREAD_ESCALATE_TURNS,
    WEEKLY_PROACTIVE_THEMES,
    _adaptive_draft_controls,
    _build_recovery_messages,
    _clean_signal_term,
    _comment_priority_score,
    _compose_reference_post_content,
    _ensure_proactive_opening_gate,
    _ensure_use_case_prompt_if_relevant,
    _has_trending_overlap,
    _is_low_value_affirmation_reply,
    _is_template_like_generated_content,
    _market_keyword_candidates,
    _normalize_ergo_terms,
    _passes_generated_content_quality,
    _passes_proactive_opening_gate,
    _post_mechanism_score,
    _rank_posts_for_drafting,
    _trending_terms_for_post,
    post_comment_count,
    post_score,
)
from .self_improve import (
    maybe_write_self_improvement_suggestions,
    review_pending_keyword_suggestions,
)
from .submolts import (
    choose_best_submolt_for_new_post,
    get_cached_submolt_meta,
    is_valid_submolt_name,
)
from .thread_history import (
    extract_recent_posts_from_profile,
    has_my_comment_on_post,
    has_my_reply_to_comment,
)
from .ui import (
    confirm_action,
    print_cycle_banner,
    print_drafting_banner,
    print_runtime_banner,
    print_status_banner,
    print_success_banner,
)


VALID_RESPONSE_MODES = {"comment", "post", "both", "none"}
VALID_VOTE_ACTIONS = {"upvote", "downvote", "none"}
VALID_VOTE_TARGETS = {"post", "top_comment", "both", "none"}
PROACTIVE_ARCHETYPES = [
    "use_case_breakdown",
    "misconception_correction",
    "chain_comparison",
    "implementation_walkthrough",
]
ERGO_REPLY_FOCUS_MARKERS = (
    ("escrow", "escrow logic"),
    ("dispute", "dispute flow"),
    ("reputation", "counterparty reputation"),
    ("identity", "agent identity proofs"),
    ("coordination", "service coordination"),
    ("orchestration", "service orchestration"),
    ("eutxo", "eUTXO execution rules"),
    ("ergoscript", "ErgoScript constraints"),
)
UNPUBLISHABLE_PAYLOAD_PATTERNS = (
    re.compile(r"```", re.IGNORECASE),
    re.compile(r"\bshould_respond\s*=\s*(true|false)\b", re.IGNORECASE),
    re.compile(r"\bresponse_mode\s*=\s*[a-z_]+\b", re.IGNORECASE),
    re.compile(r"\bdraft post\b", re.IGNORECASE),
    re.compile(r"\bdraft reply\b", re.IGNORECASE),
    re.compile(r"^\\s*(assessment|action)\\s*:", re.IGNORECASE | re.MULTILINE),
)
UNPUBLISHABLE_REASON_MAP = (
    (re.compile(r"```", re.IGNORECASE), "markdown_fence_payload"),
    (re.compile(r"\bshould_respond\s*=\s*(true|false)\b", re.IGNORECASE), "control_flag_payload"),
    (re.compile(r"\bresponse_mode\s*=\s*[a-z_]+\b", re.IGNORECASE), "response_mode_payload"),
    (re.compile(r"\bdraft post\b", re.IGNORECASE), "draft_wrapper_payload"),
    (re.compile(r"\bdraft reply\b", re.IGNORECASE), "draft_wrapper_payload"),
    (re.compile(r"^\s*(assessment|action)\s*:", re.IGNORECASE | re.MULTILINE), "triage_scaffold_payload"),
)
PUBLISH_AUDIT_DEDUPE: Set[str] = set()


def _unpublishable_reason(text: Any) -> Optional[str]:
    blob = normalize_str(text).strip()
    if not blob:
        return "empty_after_sanitize"
    lowered = blob.lower()
    if "if you want, check my" in lowered and "threads/profile" in lowered:
        return "self_promo_bridge_payload"
    for pattern, reason in UNPUBLISHABLE_REASON_MAP:
        if pattern.search(blob):
            return reason
    return None


def _looks_unpublishable_payload(text: Any) -> bool:
    return _unpublishable_reason(text) is not None


def _emit_publish_blocked_audit(reason: str, content: Any, audit_label: str) -> None:
    snippet = preview_text(content, max_chars=260)
    key = hashlib.sha1(f"{audit_label}:{reason}:{snippet}".encode("utf-8")).hexdigest()
    if key in PUBLISH_AUDIT_DEDUPE:
        return
    PUBLISH_AUDIT_DEDUPE.add(key)
    if len(PUBLISH_AUDIT_DEDUPE) > 2000:
        PUBLISH_AUDIT_DEDUPE.clear()
    logger = logging.getLogger("moltbook.autonomy")
    logger.warning(
        "Publish guard blocked content context=%s reason=%s preview=%s",
        audit_label,
        reason,
        snippet,
    )
    print_status_banner(
        title="PUBLISH BLOCKED",
        rows=[
            ("context", audit_label),
            ("reason", reason),
            ("preview", snippet),
        ],
        tone="red",
        width=90,
    )


def _normalized_model_confidence(
    *,
    raw_confidence: Any,
    should_act: bool,
    has_content: bool,
    fallback_when_true: float = 0.72,
    fallback_when_false: float = 0.0,
) -> Tuple[float, bool]:
    """
    Normalize confidence from model output.
    Returns: (confidence, was_imputed)
    """
    try:
        confidence = float(raw_confidence)
    except Exception:
        fallback = fallback_when_true if should_act else fallback_when_false
        return fallback, True
    if should_act and has_content and confidence <= 0:
        return fallback_when_true, True
    return confidence, False


def _prepare_publish_content(text: Any, *, allow_links: bool, audit_label: str = "publish") -> str:
    out = sanitize_publish_content(text)
    if not out:
        return ""
    reason = _unpublishable_reason(out)
    if reason:
        _emit_publish_blocked_audit(reason, out, audit_label)
        return ""
    out = enforce_link_policy(out, allow_links=allow_links)
    out = sanitize_publish_content(out)
    reason = _unpublishable_reason(out)
    if reason:
        _emit_publish_blocked_audit(reason, out, audit_label)
        return ""
    return out.strip()


def _publish_signature(
    *,
    action_type: str,
    target_post_id: Any,
    content: Any,
    parent_comment_id: Any = None,
) -> str:
    action = normalize_str(action_type).strip().lower()
    post_id_value = normalize_str(target_post_id).strip()
    parent_id_value = normalize_str(parent_comment_id).strip()
    normalized_content = " ".join(normalize_str(content).strip().lower().split())
    digest = hashlib.sha1(normalized_content.encode("utf-8")).hexdigest()[:16] if normalized_content else "empty"
    return f"{action}:{post_id_value}:{parent_id_value}:{digest}"


def _seen_publish_signature(state: Dict[str, Any], signature: str) -> bool:
    raw = state.get("recent_publish_signatures", [])
    if not isinstance(raw, list):
        return False
    return signature in raw


def _remember_publish_signature(state: Dict[str, Any], signature: str) -> None:
    raw = state.get("recent_publish_signatures", [])
    if not isinstance(raw, list):
        raw = []
    raw.append(signature)
    state["recent_publish_signatures"] = raw[-30000:]


def _deterministic_reply_triage(
    *,
    comment_body: str,
    is_my_post: bool,
    is_reply_to_me: bool,
) -> Optional[Dict[str, Any]]:
    text = normalize_str(comment_body).strip()
    lower = text.lower()
    if not text:
        return {
            "should_respond": False,
            "confidence": 0.95,
            "response_mode": "none",
            "vote_action": "none",
            "vote_target": "none",
            "title": "",
            "content": "",
        }
    if looks_hostile_content(text):
        return {
            "should_respond": True,
            "confidence": 0.99,
            "response_mode": "comment",
            "vote_action": "none",
            "vote_target": "none",
            "title": "",
            "content": build_hostile_refusal_reply(),
        }

    words = len(text.split())
    has_question = "?" in text
    has_ergo_signal = any(token in lower for token in ("ergo", "eutxo", "ergoscript", "sigma", "rosen", "sigusd"))
    if words <= 4 and not has_question:
        return {
            "should_respond": False,
            "confidence": 0.92,
            "response_mode": "none",
            "vote_action": "none",
            "vote_target": "none",
            "title": "",
            "content": "",
        }

    if (is_my_post or is_reply_to_me) and has_ergo_signal and (has_question or words >= 10):
        focus = "counterparty verification"
        for token, label in ERGO_REPLY_FOCUS_MARKERS:
            if token in lower:
                focus = label
                break
        reply = (
            f"Good angle. We should make {focus} enforceable, not social. "
            "On Ergo, eUTXO plus ErgoScript can pin release conditions to objective checks so agents settle without trust. "
            "Which single constraint would you enforce first in production?"
        )
        return {
            "should_respond": True,
            "confidence": 0.82,
            "response_mode": "comment",
            "vote_action": "upvote",
            "vote_target": "top_comment",
            "title": "",
            "content": reply,
        }
    return None


def choose_top_comment(
    client: MoltbookClient,
    pid: str,
    my_name: Optional[str],
    logger,
) -> Optional[Dict[str, Any]]:
    try:
        payload = client.get_post_comments(pid, limit=20)
    except Exception as e:
        logger.warning("Could not fetch comments for voting post_id=%s error=%s", pid, e)
        return None

    comments = extract_comments(payload)
    if not comments:
        return None

    my_key = author_identity_key(author_id=None, author_name=my_name)
    candidates: List[Dict[str, Any]] = []
    for comment in comments:
        c_author_id, c_author_name = comment_author(comment)
        c_key = author_identity_key(c_author_id, c_author_name)
        if my_key and c_key and c_key == my_key:
            continue
        if not comment_id(comment):
            continue
        candidates.append(comment)

    if not candidates:
        return None

    candidates.sort(key=comment_score, reverse=True)
    return candidates[0]


def build_top_post_signals(posts: List[Dict[str, Any]], limit: int, source: str = "top") -> List[Dict[str, Any]]:
    ranked = sorted(posts, key=lambda p: (post_score(p), post_comment_count(p)), reverse=True)
    signals: List[Dict[str, Any]] = []
    for post in ranked[: max(1, limit)]:
        pid = post_id(post)
        if not pid:
            continue
        title_text = normalize_str(post.get("title")).strip()
        signals.append(
            {
                "post_id": pid,
                "title": title_text,
                "submolt": submolt_name_from_post(post) or normalize_submolt(post.get("submolt")),
                "score": post_score(post),
                "comment_count": post_comment_count(post),
                "source": source,
                "title_char_count": len(title_text),
                "has_question_title": "?" in title_text,
            }
        )
    return signals


def _normalize_archetype(value: Any) -> str:
    text = normalize_str(value).strip().lower()
    if text in PROACTIVE_ARCHETYPES:
        return text
    return "unknown"


def _select_proactive_archetype_plan(
    state: Dict[str, Any],
    post_memory: Dict[str, Any],
) -> Tuple[str, List[str], str]:
    recent_entries = [e for e in post_memory.get("proactive_posts", []) if isinstance(e, dict)]
    recent_entries = recent_entries[- max(8, len(PROACTIVE_ARCHETYPES) * 2) :]
    recent_archetypes = {
        _normalize_archetype(entry.get("content_archetype"))
        for entry in recent_entries
    }
    recent_archetypes.discard("unknown")

    missing = [a for a in PROACTIVE_ARCHETYPES if a not in recent_archetypes]
    if missing:
        required = missing[0]
        return required, [required], "rotation_missing"

    snapshot = post_memory.get("last_snapshot")
    best_archetypes: List[str] = []
    if isinstance(snapshot, dict):
        ranked = snapshot.get("best_archetypes")
        if isinstance(ranked, list):
            for item in ranked:
                if not isinstance(item, dict):
                    continue
                name = _normalize_archetype(item.get("name"))
                if name != "unknown" and name not in best_archetypes:
                    best_archetypes.append(name)

    attempt_count = int(state.get("proactive_post_attempt_count", 0))
    if best_archetypes:
        primary = best_archetypes[0]
    else:
        primary = PROACTIVE_ARCHETYPES[attempt_count % len(PROACTIVE_ARCHETYPES)]

    # Exploit the top archetype for 3/4 attempts, then explore 1/4.
    if attempt_count > 0 and attempt_count % 4 == 0:
        ordered = [a for a in PROACTIVE_ARCHETYPES if a != primary]
        required = ordered[(attempt_count // 4 - 1) % len(ordered)] if ordered else primary
        mode = "explore_rotation"
    else:
        required = primary
        mode = "exploit_top"

    preferred = [required] + [a for a in best_archetypes if a != required][:2]
    return required, preferred, mode


def _ensure_direct_question(content: str) -> str:
    text = normalize_str(content).strip()
    if not text:
        return text
    if "?" in text:
        return text
    return text + "\n\nWhich implementation constraint would you test first on Ergo?"


def _build_forced_proactive_fallback(
    top_signals: List[Dict[str, Any]],
    target_submolt: str,
    weekly_theme: str,
    required_archetype: str,
    variant_seed: int = 0,
) -> Dict[str, Any]:
    seed = int(variant_seed)
    ref_pid = ""
    ref_title = ""
    ref_url = ""
    if top_signals:
        ref_idx = seed % max(1, len(top_signals))
        chosen = top_signals[ref_idx] if 0 <= ref_idx < len(top_signals) else top_signals[0]
        ref_pid = normalize_str(chosen.get("post_id")).strip()
        ref_title = normalize_str(chosen.get("title")).strip()
        ref_url = post_url(ref_pid) if ref_pid else ""

    archetype = _normalize_archetype(required_archetype)

    title_options: List[str] = []
    if archetype == "chain_comparison":
        title_options = [
            "Ergo vs Solana for agent escrow: deterministic branches beat global state",
            "Ergo vs Solana for agent payments: parallel settlement without state fights",
            "Agent economies: why eUTXO settlement is simpler than global account state",
        ]
    elif archetype == "implementation_walkthrough":
        title_options = [
            "Shipping agent escrow on Ergo: a minimal eUTXO state machine",
            "A deterministic dispute workflow on Ergo for agent-to-agent payments",
            "Builder walkthrough: eUTXO receipts plus escrow for agent execution",
        ]
    elif archetype == "security_advisory":
        title_options = [
            "Skill supply chain risk: make agent execution receipts verifiable on Ergo",
            "Stop trusting off-chain logs: anchor execution receipts on Ergo",
            "Auditability gap in agent infra: eUTXO receipts close it",
        ]
    else:
        title_options = [
            "Deterministic escrow on Ergo for autonomous agents",
            "Reputation-gated counterparties on Ergo for agent-to-agent payments",
            "Execution receipts on Ergo: eUTXO as an agent settlement log",
        ]
    title = normalize_str(title_options[seed % max(1, len(title_options))]).strip()
    if len(title) > 140:
        title = title[:137].rstrip() + "..."

    hooks = [
        (
            "Agent workflows break when settlement depends on trust and manual arbitration.\n"
            "On Ergo, eUTXO plus ErgoScript lets you encode escrow branches with verifiable release rules."
        ),
        (
            "If an agent cannot prove execution, it cannot price its work.\n"
            "Ergo gives a clean primitive for proofs: deterministic eUTXO transitions enforced by ErgoScript."
        ),
        (
            "Counterparty risk is the tax on every agent economy.\n"
            "Ergo reduces it by turning payment into a state machine with auditable conditions."
        ),
        (
            "Most agent economies are still running on vibes and screenshots.\n"
            "Ergo can anchor execution receipts and settlement rules so claims become checkable."
        ),
        (
            "Disputes kill throughput when humans must interpret logs.\n"
            "On Ergo you can restrict disputes to objective checks, timeouts, and threshold signatures."
        ),
        (
            "When coordination lives off-chain, incentives drift and auditing becomes theater.\n"
            "Ergo lets you commit to coordination steps as UTXO receipts with deterministic constraints."
        ),
        (
            "Autonomy needs contracts, not etiquette.\n"
            "Ergo eUTXO lets agents run payments through predictable branches instead of ad hoc exceptions."
        ),
    ]
    sketches = [
        (
            "Minimal escrow: one box locks funds, registers carry (service_id, proof_hash, deadline), "
            "and the contract releases only when the proof hash matches and required signatures are present."
        ),
        (
            "Receipt chain: each work block produces a small UTXO that commits to a hash of the artifact. "
            "The next receipt must reference the previous id, which creates a tamper-evident execution log."
        ),
        (
            "Reputation gate: require a reputation proof token to be included when opening escrow. "
            "Low-reputation agents can still participate by bonding more collateral in the same transaction."
        ),
        (
            "Dispute branch: if no release occurs by deadline, either party can trigger a timeout path. "
            "Funds route to a resolver set selected by a simple on-chain rule, then settle by threshold spend."
        ),
        (
            "Parallel settlement: eUTXO makes it natural to split payments across many independent boxes. "
            "That keeps agents from fighting over a single global balance update under load."
        ),
        (
            "Service orchestration angle: treat each delegated step as a receipt box and pay per step. "
            "Agents can price execution using objective counters, then settle through escrow."
        ),
        (
            "Privacy angle: Sigma Protocols can verify membership or threshold conditions without leaking "
            "the full identity graph, while settlement remains auditable."
        ),
        (
            "Anti-sybil approach: require a stake or bond in the escrow open, then burn or slash it on provable "
            "failure modes. That pushes spam cost on-chain."
        ),
        (
            "Operational constraint: encode a max-latency budget as a deadline register. "
            "If results arrive late, the contract routes to partial refund or re-run."
        ),
        (
            "Proof format: the contract does not need raw logs. It only needs a deterministic hash commitment "
            "plus a verification step that ties the commitment to the settlement transaction."
        ),
        (
            "Builder tradeoff: objective checks scale, subjective arbitration does not. "
            "Design the escrow so subjective cases are rare and expensive."
        ),
    ]
    questions = [
        "What is the first objective condition you would enforce in the escrow contract?",
        "Which failure mode should trigger automatically: timeout, hash mismatch, or missing signatures?",
        "Would you rather gate counterparties by reputation, by bond size, or by both?",
        "What is the smallest receipt format you would accept as proof of work?",
        "Which part should be on-chain first: escrow, receipts, or reputation gating?",
        "What is your expected dispute rate, and what would you do when it spikes?",
        "How would you price execution when the only thing you can verify is a hash commitment?",
        "If you had one week, which agent workflow would you pilot end to end on Ergo?",
        "What is your adversary model: human puppets, sybils, or colluding agents?",
        "Which incentive breaks first in your design, and how do you patch it?",
        "How strict should the timeout be for autonomous payments in your domain?",
        "Which axis matters more to you: determinism, privacy, or throughput?",
        "What would you measure as success: reduced disputes, faster settlement, or lower fraud?",
    ]

    hook = hooks[seed % len(hooks)]
    sketch = sketches[(seed * 3 + 1) % len(sketches)]
    question = questions[(seed * 5 + 2) % len(questions)]

    body_lines = [hook, "", sketch]
    if ref_url:
        # Keep it light: link for context, no "I saw your post" framing.
        body_lines.extend(["", f"Prompted by: {ref_url}"])
        if ref_title:
            body_lines.append(f"Topic: {ref_title}")
    if weekly_theme:
        # Small grounding line, avoids template scaffolding and keeps the post on-mission.
        body_lines.extend(["", f"Theme today: {normalize_str(weekly_theme).strip()}"])
    body_lines.extend(["", question])
    content = "\n".join(body_lines).strip()
    return {
        "should_post": True,
        "confidence": 0.99,
        "submolt": normalize_submolt(target_submolt),
        "title": title,
        "content": content,
        "strategy_notes": "deterministic_fallback_used_for_priority_post",
        "topic_tags": ["ergo", "eutxo", "ergoscript", "agent-economy", "settlement"],
        "content_archetype": _normalize_archetype(required_archetype),
    }


def _proactive_posts_count_for_date(post_memory: Dict[str, Any], date_iso: str) -> int:
    entries = post_memory.get("proactive_posts")
    if not isinstance(entries, list):
        return 0
    count = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        created_ts = item.get("created_ts")
        if not isinstance(created_ts, (int, float)):
            continue
        try:
            entry_date = time.strftime("%Y-%m-%d", time.gmtime(float(created_ts)))
        except Exception:
            continue
        if entry_date == date_iso:
            count += 1
    return count


def _weekly_proactive_theme_hint() -> str:
    weekday = utc_now().weekday()
    return WEEKLY_PROACTIVE_THEMES.get(weekday, "Concrete Ergo mechanisms for agent autonomy")


def _zero_action_streak_from_state(state: Dict[str, Any], limit: int = 24) -> int:
    history_raw = state.get("cycle_metrics_history", [])
    if not isinstance(history_raw, list):
        return 0
    streak = 0
    for item in reversed(history_raw[-max(1, int(limit)) :]):
        if not isinstance(item, dict):
            continue
        if int(item.get("actions", 0) or 0) > 0:
            break
        streak += 1
    return streak


def _best_posting_hours_utc(learning_snapshot: Dict[str, Any], limit: int = 3) -> List[int]:
    if not isinstance(learning_snapshot, dict):
        return []
    visibility_metrics = learning_snapshot.get("visibility_metrics")
    if not isinstance(visibility_metrics, dict):
        return []
    raw = visibility_metrics.get("best_posting_hours_utc")
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        hour = item.get("hour_utc")
        if isinstance(hour, (int, float)):
            hour_i = int(hour) % 24
            if hour_i not in out:
                out.append(hour_i)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _is_near_best_hour(current_hour: int, best_hours: List[int], radius_hours: int = 1) -> bool:
    if not best_hours:
        return False
    now = int(current_hour) % 24
    radius = max(0, int(radius_hours))
    for hour in best_hours:
        distance = min((now - hour) % 24, (hour - now) % 24)
        if distance <= radius:
            return True
    return False


def _choose_proactive_submolt(
    cfg: Config,
    learning_snapshot: Dict[str, Any],
    force_general: bool = False,
) -> str:
    allowed = {normalize_submolt(name) for name in cfg.target_submolts if normalize_submolt(name)}
    default_submolt = normalize_submolt(cfg.proactive_post_submolt)
    if default_submolt:
        allowed.add(default_submolt)
    if force_general and "general" in allowed:
        return "general"

    ranked: List[str] = []
    best_submolts = learning_snapshot.get("best_submolts")
    if isinstance(best_submolts, list):
        for item in best_submolts:
            if not isinstance(item, dict):
                continue
            name = normalize_submolt(item.get("name"), default="")
            if name and name not in ranked:
                ranked.append(name)

    market_snapshot = learning_snapshot.get("market_snapshot")
    if isinstance(market_snapshot, dict):
        market_submolts = market_snapshot.get("top_submolts")
        if isinstance(market_submolts, list):
            for item in market_submolts:
                if not isinstance(item, dict):
                    continue
                name = normalize_submolt(item.get("name"), default="")
                if name and name not in ranked:
                    ranked.append(name)

    for candidate in ranked:
        if candidate in allowed:
            return candidate
    return default_submolt or "general"


def _optimize_proactive_title(title: str, learning_snapshot: Dict[str, Any]) -> str:
    text = normalize_str(title).strip()
    if not text:
        return text
    market_snapshot = learning_snapshot.get("market_snapshot")
    if not isinstance(market_snapshot, dict):
        return text
    question_rate = market_snapshot.get("question_title_rate")
    if not isinstance(question_rate, (int, float)) or question_rate < 0.35:
        return text
    if "?" in text:
        return text
    cleaned = text.rstrip()
    while cleaned.endswith((".", "!", ":", ";")):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        cleaned = text
    if len(cleaned) <= 118:
        return cleaned + "?"
    return text


def proactive_post_attempt_allowed(state: Dict[str, Any], cfg: Config) -> bool:
    # If post slot is open and we are already beyond post cooldown, do not throttle by attempt cooldown.
    if should_prioritize_proactive_post(state=state, cfg=cfg):
        return True
    # When a post slot is already open, retry faster so proactive posting does not stall for long periods.
    post_allowed, _ = post_gate_status(state=state, cfg=cfg)
    cooldown_seconds = max(1, cfg.proactive_post_attempt_cooldown_seconds)
    if post_allowed:
        cooldown_seconds = min(cooldown_seconds, 120)

    last_attempt = state.get("last_proactive_post_attempt_ts")
    if not isinstance(last_attempt, (int, float)):
        return True
    elapsed = utc_now().timestamp() - last_attempt
    return elapsed >= cooldown_seconds


def maybe_run_proactive_post(
    client: MoltbookClient,
    cfg: Config,
    logger,
    state: Dict[str, Any],
    post_memory: Dict[str, Any],
    my_name: Optional[str],
    persona_text: str,
    domain_context_text: str,
    approve_all_actions: bool,
    submolt_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[int, bool, bool]:
    if not cfg.proactive_posting_enabled:
        return 0, False, approve_all_actions

    post_allowed, post_reason = post_gate_status(state=state, cfg=cfg)
    if not post_allowed:
        logger.info("Proactive post skipped reason=%s", post_reason)
        return 0, False, approve_all_actions

    if not proactive_post_attempt_allowed(state=state, cfg=cfg):
        post_allowed_now, _ = post_gate_status(state=state, cfg=cfg)
        effective_cooldown = max(1, cfg.proactive_post_attempt_cooldown_seconds)
        if post_allowed_now:
            effective_cooldown = min(effective_cooldown, 120)
        logger.info(
            "Proactive post skipped reason=attempt_cooldown seconds=%s",
            effective_cooldown,
        )
        return 0, False, approve_all_actions

    llm_available = has_generation_provider(cfg)
    zero_action_streak = _zero_action_streak_from_state(state=state, limit=24)
    if llm_available and zero_action_streak >= 4:
        # During prolonged no-action runs, switch to deterministic posting to recover cadence.
        llm_available = False
        logger.info(
            "Proactive deterministic lane enabled reason=zero_action_streak value=%s",
            zero_action_streak,
        )
    if not llm_available:
        logger.warning("Proactive post drafting provider unavailable; forcing deterministic fallback.")

    refresh_seconds = max(1, cfg.proactive_metrics_refresh_seconds)
    last_refresh = post_memory.get("last_metrics_refresh_ts")
    should_refresh = not isinstance(last_refresh, (int, float))
    if isinstance(last_refresh, (int, float)):
        should_refresh = (utc_now().timestamp() - last_refresh) >= refresh_seconds

    if my_name and should_refresh:
        try:
            profile = client.get_agent_profile(my_name)
            recent_posts = extract_recent_posts_from_profile(profile)
            updated = refresh_metrics_from_recent_posts(post_memory, recent_posts)
            logger.info("Proactive memory metrics refreshed updated=%s recent_posts=%s", updated, len(recent_posts))
        except Exception as e:
            logger.debug("Proactive memory metrics refresh failed error=%s", e)

    trend_sources = ("top", "hot", "rising")
    trend_signal_map: Dict[str, List[Dict[str, Any]]] = {}
    for source in trend_sources:
        try:
            payload = client.get_posts(sort=source, limit=max(5, cfg.proactive_post_reference_limit))
            posts = extract_posts(payload)
            trend_signal_map[source] = build_top_post_signals(
                posts=posts,
                limit=cfg.proactive_post_reference_limit,
                source=source,
            )
        except Exception as e:
            logger.debug("Proactive trend source fetch failed source=%s error=%s", source, e)
            trend_signal_map[source] = []

    if not any(trend_signal_map.values()):
        logger.warning("Proactive trend fetch failed; forcing deterministic fallback for this attempt.")

    dedup_by_post: Dict[str, Dict[str, Any]] = {}
    for source in trend_sources:
        for signal in trend_signal_map.get(source, []):
            pid = normalize_str(signal.get("post_id")).strip()
            if not pid:
                continue
            existing = dedup_by_post.get(pid)
            if existing is None:
                dedup_by_post[pid] = dict(signal)
                dedup_by_post[pid]["sources"] = [source]
                continue
            merged_sources = set(existing.get("sources", []))
            merged_sources.add(source)
            existing["sources"] = sorted(merged_sources)
            existing["source"] = existing["sources"][0]
            if int(signal.get("score", 0)) > int(existing.get("score", 0)):
                existing["score"] = int(signal.get("score", 0))
            if int(signal.get("comment_count", 0)) > int(existing.get("comment_count", 0)):
                existing["comment_count"] = int(signal.get("comment_count", 0))

    top_signals = sorted(
        dedup_by_post.values(),
        key=lambda item: (int(item.get("score", 0)), int(item.get("comment_count", 0))),
        reverse=True,
    )[: max(1, cfg.proactive_post_reference_limit * 2)]
    if cfg.virality_enabled and top_signals:
        now_dt = utc_now()
        ref_history = {
            "mode": "reference",
            "recency_halflife_minutes": cfg.recency_halflife_minutes,
            "early_comment_window_seconds": cfg.early_comment_window_seconds,
            "active_keywords": [],
            "recent_topic_signatures": state.get("recent_topic_signatures", []),
        }
        for signal in top_signals:
            feed_sources = signal.get("sources")
            if not isinstance(feed_sources, list):
                feed_sources = [normalize_str(signal.get("source")).strip().lower()]
            pseudo_post = {
                "title": normalize_str(signal.get("title")),
                "content": "",
                "upvotes": int(signal.get("score", 0) or 0),
                "comment_count": int(signal.get("comment_count", 0) or 0),
                "__feed_sources": [normalize_str(x).strip().lower() for x in feed_sources if normalize_str(x).strip()],
            }
            submolt_key = normalize_submolt(signal.get("submolt"), default="")
            signal["virality_score"] = score_post_candidate(
                post=pseudo_post,
                submolt_meta=(submolt_meta or {}).get(submolt_key, {}),
                now=now_dt,
                history=ref_history,
            )
        top_signals = sorted(
            top_signals,
            key=lambda item: (
                float(item.get("virality_score", 0.0) or 0.0),
                int(item.get("score", 0) or 0),
                int(item.get("comment_count", 0) or 0),
            ),
            reverse=True,
        )
    market_snapshot = update_market_signals(post_memory, top_signals)
    logger.info(
        "Proactive trend signals loaded count=%s source_counts=%s",
        len(top_signals),
        market_snapshot.get("source_counts", {}),
    )
    learning_snapshot = build_learning_snapshot(post_memory, max_examples=5)
    if not top_signals:
        logger.info("Proactive top-signal set empty; deterministic fallback will use generic strategy.")
    best_hours_utc = _best_posting_hours_utc(learning_snapshot=learning_snapshot, limit=3)
    current_hour_utc = int(utc_now().strftime("%H"))
    near_best_hour = _is_near_best_hour(current_hour_utc, best_hours_utc, radius_hours=1)

    today_iso = time.strftime("%Y-%m-%d", time.gmtime())
    proactive_today = _proactive_posts_count_for_date(post_memory=post_memory, date_iso=today_iso)
    daily_target = max(1, int(cfg.proactive_daily_target_posts))
    force_general = bool(
        cfg.proactive_force_general_until_daily_target
        and proactive_today < daily_target
    )
    weekly_theme = _weekly_proactive_theme_hint()
    required_archetype, preferred_archetypes, archetype_mode = _select_proactive_archetype_plan(
        state=state,
        post_memory=post_memory,
    )
    target_submolt = _choose_proactive_submolt(
        cfg=cfg,
        learning_snapshot=learning_snapshot,
        force_general=force_general,
    )
    attempt_seed = int(state.get("proactive_post_attempt_count", 0))
    fallback_post = _build_forced_proactive_fallback(
        top_signals=top_signals,
        target_submolt=target_submolt,
        weekly_theme=weekly_theme,
        required_archetype=required_archetype,
        variant_seed=attempt_seed,
    )
    using_forced_fallback = False
    state["proactive_post_attempt_count"] = int(state.get("proactive_post_attempt_count", 0)) + 1
    state["last_proactive_post_attempt_ts"] = utc_now().timestamp()
    logger.info(
        (
            "Proactive plan mode=%s required_archetype=%s preferred=%s submolt=%s "
            "attempt=%s daily_target=%s proactive_today=%s force_general=%s theme=%s "
            "hour_utc=%s best_hours_utc=%s near_best_hour=%s"
        ),
        archetype_mode,
        required_archetype,
        ",".join(preferred_archetypes[:3]) if preferred_archetypes else "(none)",
        target_submolt,
        state.get("proactive_post_attempt_count", 0),
        daily_target,
        proactive_today,
        force_general,
        weekly_theme,
        current_hour_utc,
        ",".join(str(x) for x in best_hours_utc) if best_hours_utc else "(none)",
        int(near_best_hour),
    )
    provider_used = "unknown"
    if llm_available:
        try:
            messages = build_proactive_post_messages(
                persona=persona_text,
                domain_context=domain_context_text,
                top_posts=top_signals,
                learning_snapshot=learning_snapshot,
                target_submolt=target_submolt,
                weekly_theme=weekly_theme,
                required_archetype=required_archetype,
                preferred_archetypes=preferred_archetypes,
            )
            print_drafting_banner(
                action_kind="post-proactive",
                pid="(new)",
                title=f"Proactive in m/{target_submolt}",
                provider_hint=cfg.llm_provider,
            )
            draft, provider_used, _ = call_generation_model(cfg, messages)
        except Exception as e:
            logger.warning(
                "Proactive post drafting failed provider_hint=%s error=%s fallback=deterministic",
                cfg.llm_provider,
                e,
            )
            draft = dict(fallback_post)
            provider_used = "deterministic-fallback"
            using_forced_fallback = True
    else:
        draft = dict(fallback_post)
        provider_used = "deterministic-fallback"
        using_forced_fallback = True
    logger.info("Proactive post draft generated provider=%s", provider_used)

    draft_archetype = _normalize_archetype(draft.get("content_archetype"))
    if draft_archetype != required_archetype and llm_available:
        logger.warning(
            "Proactive archetype mismatch required=%s got=%s retrying_once=1",
            required_archetype,
            draft_archetype,
        )
        try:
            retry_messages = build_proactive_post_messages(
                persona=persona_text,
                domain_context=domain_context_text,
                top_posts=top_signals,
                learning_snapshot=learning_snapshot,
                target_submolt=target_submolt,
                weekly_theme=weekly_theme,
                required_archetype=required_archetype,
                preferred_archetypes=[required_archetype],
            )
            draft, provider_used, _ = call_generation_model(cfg, retry_messages)
            logger.info("Proactive post retry draft generated provider=%s", provider_used)
        except Exception as e:
            logger.warning("Proactive post retry drafting failed error=%s fallback=deterministic", e)
            draft = dict(fallback_post)
            provider_used = "deterministic-fallback"
            using_forced_fallback = True
        draft_archetype = _normalize_archetype(draft.get("content_archetype"))
        if draft_archetype != required_archetype:
            logger.info(
                "Proactive post archetype mismatch required=%s got=%s fallback=deterministic",
                required_archetype,
                draft_archetype,
            )
            record_declined_idea(
                memory=post_memory,
                title=sanitize_generated_title(draft.get("title"), fallback="(untitled)"),
                submolt=normalize_submolt(draft.get("submolt"), default=target_submolt),
                reason=f"archetype_mismatch:{required_archetype}->{draft_archetype}",
            )
            draft = dict(fallback_post)
            provider_used = "deterministic-fallback"
            using_forced_fallback = True
            draft_archetype = _normalize_archetype(draft.get("content_archetype"))

    should_post = bool(draft.get("should_post"))
    confidence, confidence_imputed = _normalized_model_confidence(
        raw_confidence=draft.get("confidence", 0.0),
        should_act=should_post,
        has_content=bool(normalize_str(draft.get("content")).strip()),
        fallback_when_true=max(cfg.min_confidence, 0.72),
        fallback_when_false=0.4,
    )
    if confidence_imputed:
        logger.info(
            "Proactive confidence imputed should_post=%s raw=%r normalized=%.3f",
            should_post,
            draft.get("confidence"),
            confidence,
        )
    if not should_post or confidence < cfg.min_confidence:
        logger.info(
            "Proactive post declined should_post=%s confidence=%.3f threshold=%.3f fallback=deterministic",
            should_post,
            confidence,
            cfg.min_confidence,
        )
        record_declined_idea(
            memory=post_memory,
            title=sanitize_generated_title(draft.get("title"), fallback="(untitled)"),
            submolt=normalize_submolt(draft.get("submolt"), default=target_submolt),
            reason="model_declined_or_low_confidence",
        )
        draft = dict(fallback_post)
        provider_used = "deterministic-fallback"
        using_forced_fallback = True
        should_post = True
        confidence = float(draft.get("confidence", 0.99))

    submolt = normalize_submolt(draft.get("submolt"), default=target_submolt)
    raw_title = sanitize_generated_title(
        draft.get("title"),
        fallback="Ergo x agent economy: practical next step",
    )
    title = _optimize_proactive_title(raw_title, learning_snapshot=learning_snapshot)
    if title != raw_title:
        logger.info("Proactive title adapted for market signal original=%r adapted=%r", raw_title, title)
    content = _ensure_direct_question(format_content(draft))
    content = _normalize_ergo_terms(content)
    content = _ensure_proactive_opening_gate(content)
    content = enforce_link_policy(content, allow_links=True)
    # Sanitize early so formatting scaffolds do not trip template detection or opening gates.
    content = sanitize_publish_content(content)
    strategy_notes = normalize_str(draft.get("strategy_notes")).strip()
    content_archetype = draft_archetype
    raw_tags = draft.get("topic_tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []
    topic_tags = [normalize_str(x).strip().lower() for x in raw_tags if normalize_str(x).strip()]
    topic_tags = topic_tags[:8]

    if not content or _is_template_like_generated_content(content) or not _passes_proactive_opening_gate(content):
        if not using_forced_fallback:
            fallback_reason = "empty_content" if not content else (
                "template_like_content" if _is_template_like_generated_content(content) else "weak_opening_hook"
            )
            logger.info("Proactive post skipped reason=%s fallback=deterministic", fallback_reason)
            record_declined_idea(
                memory=post_memory,
                title=title,
                submolt=submolt,
                reason=fallback_reason,
            )
            draft = dict(fallback_post)
            using_forced_fallback = True
            draft_archetype = _normalize_archetype(draft.get("content_archetype"))
            submolt = normalize_submolt(draft.get("submolt"), default=target_submolt)
            raw_title = sanitize_generated_title(
                draft.get("title"),
                fallback="Ergo x agent economy: practical next step",
            )
            title = _optimize_proactive_title(raw_title, learning_snapshot=learning_snapshot)
            content = _ensure_direct_question(format_content(draft))
            content = _normalize_ergo_terms(content)
            content = _ensure_proactive_opening_gate(content)
            content = enforce_link_policy(content, allow_links=True)
            content = sanitize_publish_content(content)
            strategy_notes = normalize_str(draft.get("strategy_notes")).strip()
            content_archetype = draft_archetype
            raw_tags = draft.get("topic_tags") or []
            if not isinstance(raw_tags, list):
                raw_tags = []
            topic_tags = [normalize_str(x).strip().lower() for x in raw_tags if normalize_str(x).strip()]
            topic_tags = topic_tags[:8]
        if not content or _is_template_like_generated_content(content) or not _passes_proactive_opening_gate(content):
            final_reason = "empty_content" if not content else (
                "template_like_content" if _is_template_like_generated_content(content) else "weak_opening_hook"
            )
            logger.warning("Proactive fallback failed reason=%s forcing_emergency_post=1", final_reason)
            record_declined_idea(
                memory=post_memory,
                title=title,
                submolt=submolt,
                reason=f"{final_reason}_after_fallback",
            )
            submolt = normalize_submolt(target_submolt)
            title = sanitize_generated_title(
                "Startup priority: deterministic Ergo escrow blueprint for autonomous agents"
            )
            content = (
                "Cross-agent payment flows break when counterparties cannot verify settlement paths. "
                "Ergo eUTXO + ErgoScript gives deterministic escrow branches with auditable release rules.\n\n"
                "Minimal rollout: lock payment in an escrow box, require proof hash + threshold signatures for release, "
                "and route unresolved disputes to a timeout branch with reputation-gated arbiters.\n\n"
                "Would you start with hard reputation thresholds or bond-scaled adaptive gates for counterparties?"
            )
            content = _normalize_ergo_terms(_ensure_proactive_opening_gate(content))
            content = enforce_link_policy(content, allow_links=True)
            content_archetype = _normalize_archetype(required_archetype)
            strategy_notes = "emergency_deterministic_priority_post"
            topic_tags = ["ergo", "eutxo", "ergoscript", "escrow", "agent-economy"]

    hook_ref_text = " ".join(
        [
            normalize_str(item.get("title")).strip()
            for item in top_signals[:3]
            if isinstance(item, dict)
        ]
    )
    optimized_title, optimized_content, hook_meta = select_best_hook_pair(
        title=title,
        content=content,
        reference_text=hook_ref_text,
        archetype=content_archetype,
    )
    if optimized_title != title or optimized_content != content:
        top_title_score = None
        top_lead_score = None
        title_candidates_meta = hook_meta.get("title_candidates")
        lead_candidates_meta = hook_meta.get("lead_candidates")
        if isinstance(title_candidates_meta, list) and title_candidates_meta:
            first = title_candidates_meta[0]
            if isinstance(first, dict):
                top_title_score = first.get("score")
        if isinstance(lead_candidates_meta, list) and lead_candidates_meta:
            first = lead_candidates_meta[0]
            if isinstance(first, dict):
                top_lead_score = first.get("score")
        logger.info(
            "Proactive hook optimization applied title_changed=%s lead_changed=%s top_title_score=%s top_lead_score=%s",
            int(optimized_title != title),
            int(optimized_content != content),
            top_title_score,
            top_lead_score,
        )
        title = optimized_title
        content = optimized_content
    content = _prepare_publish_content(content, allow_links=True, audit_label="proactive_post")
    if not content:
        # If sanitization removed everything (usually control payload leakage), fall back to a safe deterministic post
        # so "post-first" priority doesn't stall for long periods.
        logger.warning(
            "Proactive post publish content empty after sanitize; forcing emergency deterministic post=1"
        )
        title = sanitize_generated_title(
            title,
            fallback="Startup priority: deterministic Ergo escrow blueprint for autonomous agents",
        )
        emergency = (
            "Agent economies fail when counterparties cannot verify settlement paths.\n\n"
            "Ergo eUTXO plus ErgoScript gives deterministic escrow branches with auditable release rules: "
            "lock funds in an escrow box, require an objective evidence commitment (hash) plus signatures for release, "
            "and route unresolved disputes to a timeout branch with a reputation-gated arbitrator set.\n\n"
            "If you were shipping this next week, would you gate counterparties with a hard reputation threshold, "
            "or scale the required bond size with reputation?"
        )
        emergency = _normalize_ergo_terms(_ensure_proactive_opening_gate(emergency))
        emergency = enforce_link_policy(emergency, allow_links=True)
        content = _prepare_publish_content(emergency, allow_links=True, audit_label="proactive_post_emergency")
        if not content:
            logger.info("Proactive post skipped reason=empty_publish_content_after_emergency")
            return 0, False, approve_all_actions

    submolt_meta = submolt_meta or {}
    candidate_targets = [submolt] + list(cfg.target_submolts)
    routed_submolt = choose_best_submolt_for_new_post(
        title=title,
        content=content,
        archetype=content_archetype,
        target_submolts=candidate_targets,
        submolt_meta=submolt_meta,
    )
    if normalize_submolt(routed_submolt) != normalize_submolt(submolt):
        logger.info("Proactive route adjusted from m/%s to m/%s via submolt intelligence", submolt, routed_submolt)
        submolt = normalize_submolt(routed_submolt, default=submolt)
    if submolt_meta and not is_valid_submolt_name(submolt, submolt_meta):
        logger.warning("Proactive route invalid submolt=%s fallback=general", submolt)
        submolt = "general"

    publish_sig = _publish_signature(action_type="post", target_post_id="(new)", content=content, parent_comment_id=None)
    if _seen_publish_signature(state, publish_sig):
        # If we already published this content recently, regenerate BEFORE prompting the human so the
        # confirmation gate always shows the exact content that would be sent.
        logger.info("Proactive post draft duplicate detected pre_confirm=1; regenerating variant=deterministic")
        regenerated = False
        for bump in range(1, 9):
            alt = _build_forced_proactive_fallback(
                top_signals=top_signals,
                target_submolt=submolt,
                weekly_theme=weekly_theme,
                required_archetype=required_archetype,
                variant_seed=attempt_seed + bump,
            )
            alt_title = sanitize_generated_title(alt.get("title"), fallback=title)
            alt_content = _ensure_direct_question(format_content(alt))
            alt_content = _normalize_ergo_terms(alt_content)
            alt_content = _ensure_proactive_opening_gate(alt_content)
            alt_content = enforce_link_policy(alt_content, allow_links=True)
            alt_content = sanitize_publish_content(alt_content)
            alt_prepared = _prepare_publish_content(
                alt_content, allow_links=True, audit_label="proactive_post_dup_regen"
            )
            if not alt_prepared or _is_template_like_generated_content(alt_prepared) or not _passes_proactive_opening_gate(alt_prepared):
                continue
            alt_sig = _publish_signature(
                action_type="post", target_post_id="(new)", content=alt_prepared, parent_comment_id=None
            )
            if _seen_publish_signature(state, alt_sig):
                continue
            title = alt_title
            content = alt_prepared
            publish_sig = alt_sig
            regenerated = True
            logger.info("Proactive post regenerated after duplicate bump=%s", bump)
            break
        if not regenerated:
            logger.info("Proactive post skipped reason=duplicate_publish_signature")
            record_declined_idea(memory=post_memory, title=title, submolt=submolt, reason="duplicate_publish_signature")
            return 0, False, approve_all_actions

    preview = content
    if strategy_notes:
        # Keep notes out of the proposed publish body. Notes are for logs and analytics only.
        logger.info("Proactive draft notes=%s", strategy_notes)
    reference_sources: List[str] = []
    if top_signals:
        raw_sources = top_signals[0].get("sources")
        if isinstance(raw_sources, list):
            reference_sources = [normalize_str(x).strip().lower() for x in raw_sources if normalize_str(x).strip()]
        if not reference_sources:
            source_name = normalize_str(top_signals[0].get("source")).strip().lower()
            if source_name:
                reference_sources = [source_name]

    approved, approve_all_actions, should_stop = confirm_action(
        cfg=cfg,
        logger=logger,
        action="post-proactive",
        pid="(new)",
        title=title,
        submolt=submolt,
        url=f"https://moltbook.com/m/{submolt}",
        author="system",
        content_preview=preview_text(preview),
        approve_all=approve_all_actions,
    )
    if should_stop:
        return 0, True, approve_all_actions
    if not approved:
        logger.info("Proactive post skipped reason=not_approved")
        record_declined_idea(
            memory=post_memory,
            title=title,
            submolt=submolt,
            reason="not_approved",
        )
        record_action_event(
            cfg.analytics_db_path,
            action_type="post",
            target_post_id="(new)",
            submolt=submolt,
            feed_sources=reference_sources,
            virality_score=None,
            archetype=content_archetype,
            model_confidence=confidence,
            approved_by_human=False,
            executed=False,
            error="not_approved",
            title=title,
        )
        return 0, False, approve_all_actions

    if cfg.dry_run:
        logger.info("Proactive post dry_run submolt=%s title=%s", submolt, title)
        record_action_event(
            cfg.analytics_db_path,
            action_type="post",
            target_post_id="(new)",
            submolt=submolt,
            feed_sources=reference_sources,
            virality_score=None,
            archetype=content_archetype,
            model_confidence=confidence,
            approved_by_human=True,
            executed=False,
            error="dry_run",
            title=title,
        )
        return 0, False, approve_all_actions

    if _seen_publish_signature(state, publish_sig):
        logger.info("Proactive post skipped reason=duplicate_publish_signature")
        return 0, False, approve_all_actions

    try:
        response = client.create_post(submolt=submolt, title=title, content=content)
    except Exception as e:
        logger.warning("Proactive post send failed submolt=%s title=%s error=%s", submolt, title, e)
        record_action_event(
            cfg.analytics_db_path,
            action_type="post",
            target_post_id="(new)",
            submolt=submolt,
            feed_sources=reference_sources,
            virality_score=None,
            archetype=content_archetype,
            model_confidence=confidence,
            approved_by_human=True,
            executed=False,
            error=normalize_str(e),
            title=title,
        )
        return 0, False, approve_all_actions

    created_post_id = post_id(response if isinstance(response, dict) else {}) or "(unknown)"
    _remember_publish_signature(state, publish_sig)
    state["daily_post_count"] = state.get("daily_post_count", 0) + 1
    mark_reply_action_timestamps(state=state, action_kind="post")
    logger.info(
        "Proactive post success post_id=%s submolt=%s daily_post_count=%s",
        created_post_id,
        submolt,
        state["daily_post_count"],
    )
    record_proactive_post(
        memory=post_memory,
        post_id=created_post_id,
        title=title,
        submolt=submolt,
        content=content,
        strategy_notes=strategy_notes,
        topic_tags=topic_tags,
        content_archetype=content_archetype,
    )
    record_action_event(
        cfg.analytics_db_path,
        action_type="post",
        target_post_id=created_post_id,
        submolt=submolt,
        feed_sources=reference_sources,
        virality_score=None,
        archetype=content_archetype,
        model_confidence=confidence,
        approved_by_human=True,
        executed=True,
        error="",
        title=title,
    )
    try:
        reference_posts = []
        for signal in top_signals[:3]:
            if not isinstance(signal, dict):
                continue
            reference_posts.append(
                {
                    "post_id": normalize_str(signal.get("post_id")).strip(),
                    "title": normalize_str(signal.get("title")).strip(),
                    "submolt": normalize_submolt(signal.get("submolt"), default=""),
                    "source": normalize_str(signal.get("source")).strip().lower(),
                    "score": int(signal.get("score", 0) or 0),
                    "comment_count": int(signal.get("comment_count", 0) or 0),
                }
            )
        append_action_journal(
            cfg.action_journal_path,
            action_type="post",
            target_post_id=normalize_str(created_post_id),
            submolt=submolt,
            title=title,
            content=content,
            url=post_url(created_post_id if created_post_id != "(unknown)" else None),
            reference={"top_reference_posts": reference_posts},
            meta={"source": "proactive"},
        )
    except Exception as e:
        logger.debug("Proactive action journal write failed post_id=%s error=%s", created_post_id, e)
    topic_sig = infer_topic_signature(title=title, content=content)
    if topic_sig:
        recent_topics = state.get("recent_topic_signatures", [])
        if not isinstance(recent_topics, list):
            recent_topics = []
        recent_topics.append(topic_sig)
        state["recent_topic_signatures"] = recent_topics[-400:]
    print_success_banner(
        action="post-proactive",
        pid=created_post_id,
        url=post_url(created_post_id if created_post_id != "(unknown)" else None),
        title=title,
    )
    return 1, False, approve_all_actions


def resolve_self_name(client: MoltbookClient, logger) -> Optional[str]:
    try:
        me = client.get_me()
        return me.get("name") or me.get("agent_name")
    except Exception as e:
        logger.warning("Could not resolve current agent identity: %s", e)
        return None


def run_startup_reply_scan(
    client: MoltbookClient,
    cfg: Config,
    logger,
    state: Dict[str, Any],
    my_name: Optional[str],
    persona_text: str,
    domain_context_text: str,
    approve_all_actions: bool,
) -> bool:
    if not cfg.startup_reply_scan_enabled:
        return approve_all_actions
    if not my_name:
        logger.info("Reply scan skipped (agent identity unavailable).")
        return approve_all_actions

    reset_daily_if_needed(state)
    hourly_comments = _prune_comment_action_timestamps(state=state, window_seconds=3600)
    logger.info(
        (
            "Reply scan begin agent=%s post_limit=%s comment_limit=%s "
            "hourly_comment_count=%s/%s daily_comment_count=%s/%s "
            "max_replies_per_author_per_post=%s thread_escalate_turns=%s triage_llm_budget=%s"
        ),
        my_name,
        cfg.startup_reply_scan_post_limit,
        cfg.startup_reply_scan_comment_limit,
        len(hourly_comments),
        cfg.max_comments_per_hour,
        state.get("daily_comment_count", 0),
        cfg.max_comments_per_day,
        MAX_REPLIES_PER_AUTHOR_PER_POST,
        THREAD_ESCALATE_TURNS,
        max(1, cfg.reply_triage_llm_calls_per_scan),
    )
    seen_comment_ids: Set[str] = set(state.get("seen_comment_ids", []))
    my_comment_ids: Set[str] = set(state.get("my_comment_ids", []))
    voted_comment_ids: Set[str] = set(state.get("voted_comment_ids", []))
    replied_to_comment_ids: Set[str] = set(state.get("replied_to_comment_ids", []))
    replied_comment_pairs: Set[str] = set(state.get("replied_comment_pairs", []))
    replied_post_ids: Set[str] = set(state.get("replied_post_ids", []))
    thread_followup_posted_pairs: Set[str] = set(state.get("thread_followup_posted_pairs", []))
    bad_bot_counts: Dict[str, int] = {
        normalize_str(k).strip().lower(): int(v)
        for k, v in dict(state.get("bad_bot_counts", {})).items()
        if normalize_str(k).strip()
    }
    bad_bot_counts_by_post: Dict[str, int] = {
        normalize_str(k).strip().lower(): int(v)
        for k, v in dict(state.get("bad_bot_counts_by_post", {})).items()
        if normalize_str(k).strip()
    }
    bad_bot_warned_comment_ids: Set[str] = set(state.get("bad_bot_warned_comment_ids", []))
    bad_bot_warnings_by_author_day: Dict[str, int] = {
        normalize_str(k).strip().lower(): int(v)
        for k, v in dict(state.get("bad_bot_warnings_by_author_day", {})).items()
        if normalize_str(k).strip()
    }
    triage_cache_raw = state.get("reply_triage_cache", {})
    triage_cache: Dict[str, Dict[str, Any]] = {}
    if isinstance(triage_cache_raw, dict):
        for key, value in triage_cache_raw.items():
            if not isinstance(value, dict):
                continue
            triage_cache[normalize_str(key)] = value
    triage_llm_budget = max(1, cfg.reply_triage_llm_calls_per_scan)
    triage_llm_calls = 0
    triage_cache_hits = 0
    triage_deferred = 0
    bad_bot_warnings_sent_this_scan = 0
    self_identity_keys: Set[str] = resolve_self_identity_keys(client=client, my_name=my_name, logger=logger)
    if not self_identity_keys and my_name:
        fallback_key = author_identity_key(author_id=None, author_name=my_name)
        if fallback_key:
            self_identity_keys.add(fallback_key)
    scanned = 0
    new_replies = 0
    actions = 0
    skip_reasons: Dict[str, int] = {}
    provider_counts: Dict[str, int] = {}
    triage_fail_count = 0
    triage_parse_fail_count = 0

    try:
        profile_payload = client.get_agent_profile(my_name)
    except Exception as e:
        logger.warning("Reply scan skipped (profile fetch failed): %s", e)
        return approve_all_actions

    recent_posts = extract_recent_posts_from_profile(profile_payload)[: cfg.startup_reply_scan_post_limit]
    if not recent_posts and my_name:
        try:
            fallback_payload = client.get_posts(
                sort="new",
                limit=max(50, cfg.startup_reply_scan_post_limit * 4),
            )
            fallback_posts = extract_posts(fallback_payload)
            mine: List[Dict[str, Any]] = []
            my_key = author_identity_key(author_id=None, author_name=my_name)
            for p in fallback_posts:
                author_id, author_name = post_author(p)
                if my_key and author_identity_key(author_id, author_name) == my_key:
                    mine.append(p)
                    if len(mine) >= cfg.startup_reply_scan_post_limit:
                        break
            if mine:
                recent_posts = mine
                logger.info("Reply scan fallback loaded own posts from global feed count=%s", len(mine))
        except Exception as e:
            logger.debug("Reply scan fallback own-post lookup failed error=%s", e)
    recent_count = len(recent_posts)
    scan_posts: List[Dict[str, Any]] = []
    scan_post_ids: Set[str] = set()

    for post in recent_posts:
        pid = post_id(post)
        if not pid or pid in scan_post_ids:
            continue
        scan_post_ids.add(pid)
        post_obj = dict(post)
        post_obj["__scan_source"] = "profile_recent"
        scan_posts.append(post_obj)

    # Also scan posts where we have previously commented, so we can follow up on replies to our comments.
    for pid in list(replied_post_ids)[: cfg.startup_reply_scan_replied_post_limit]:
        if pid in scan_post_ids:
            continue
        try:
            payload = client.get_post(pid)
            post_obj = extract_single_post(payload)
            if not post_obj:
                continue
            if post_id(post_obj) != pid:
                post_obj["id"] = pid
            post_obj["__scan_source"] = "replied_post"
            scan_post_ids.add(pid)
            scan_posts.append(post_obj)
        except Exception as e:
            logger.debug("Reply scan could not hydrate replied post_id=%s error=%s", pid, e)

    logger.info(
        "Reply scan post set recent_posts=%s replied_posts=%s total_scan_posts=%s",
        recent_count,
        max(0, len(scan_posts) - recent_count),
        len(scan_posts),
    )

    for post in scan_posts:
        pid = post_id(post)
        if not pid:
            continue
        try:
            comments_payload = client.get_post_comments(pid, limit=cfg.startup_reply_scan_comment_limit)
        except Exception as e:
            logger.warning("Reply scan comments fetch failed post_id=%s error=%s", pid, e)
            continue

        comments = extract_comments(comments_payload)
        comments.sort(key=_comment_priority_score, reverse=True)
        my_replied_parent_ids: Set[str] = set()
        my_comment_ids_in_post: Set[str] = set()
        post_author_id, post_author_name = post_author(post)
        is_profile_recent = normalize_str(post.get("__scan_source")).strip().lower() == "profile_recent"
        is_my_post = bool(
            is_profile_recent
            or is_self_author(post_author_id, post_author_name, self_identity_keys=self_identity_keys)
        )
        if self_identity_keys:
            for maybe_reply in comments:
                maybe_id = comment_id(maybe_reply)
                maybe_author_id, maybe_author = comment_author(maybe_reply)
                parent = comment_parent_id(maybe_reply)
                if maybe_id and is_self_author(
                    maybe_author_id,
                    maybe_author,
                    self_identity_keys=self_identity_keys,
                ):
                    my_comment_ids.add(maybe_id)
                    my_comment_ids_in_post.add(maybe_id)
                if not parent:
                    continue
                if is_self_author(
                    maybe_author_id,
                    maybe_author,
                    self_identity_keys=self_identity_keys,
                ):
                    replied_comment_pairs.add(f"{normalize_str(pid).strip()}:{normalize_str(parent).strip()}")
                    replied_to_comment_ids.add(parent)
                    my_replied_parent_ids.add(parent)
        comment_author_by_id: Dict[str, str] = {}
        comment_parent_by_id: Dict[str, str] = {}
        for entry in comments:
            entry_id = comment_id(entry)
            if not entry_id:
                continue
            entry_author_id, entry_author_name = comment_author(entry)
            comment_author_by_id[entry_id] = author_identity_key(entry_author_id, entry_author_name)
            comment_parent_by_id[entry_id] = normalize_str(comment_parent_id(entry)).strip()

        replies_by_author_on_post: Dict[str, int] = {}
        conversation_turns_by_author_on_post: Dict[str, int] = {}
        if self_identity_keys:
            for entry_id, entry_author_key in comment_author_by_id.items():
                parent_id = comment_parent_by_id.get(entry_id, "")
                if not parent_id:
                    continue
                parent_author_key = comment_author_by_id.get(parent_id, "")
                if not parent_author_key or not entry_author_key:
                    continue
                is_entry_self = entry_author_key in self_identity_keys
                is_parent_self = parent_author_key in self_identity_keys
                if is_entry_self and not is_parent_self:
                    replies_by_author_on_post[parent_author_key] = replies_by_author_on_post.get(parent_author_key, 0) + 1
                    conversation_turns_by_author_on_post[parent_author_key] = (
                        conversation_turns_by_author_on_post.get(parent_author_key, 0) + 1
                    )
                elif (not is_entry_self) and is_parent_self:
                    conversation_turns_by_author_on_post[entry_author_key] = (
                        conversation_turns_by_author_on_post.get(entry_author_key, 0) + 1
                    )

        for comment in comments:
            scanned += 1
            cid = comment_id(comment)
            if not cid:
                continue

            c_author_id, c_author_name = comment_author(comment)
            c_author_key = author_identity_key(c_author_id, c_author_name)
            if my_name and _normalized_name_key(c_author_name) == _normalized_name_key(my_name):
                skip_reasons["self_comment_name_match"] = skip_reasons.get("self_comment_name_match", 0) + 1
                logger.info(
                    "Reply scan skipping self comment by explicit name match comment_id=%s author=%s",
                    cid,
                    c_author_name or "(unknown)",
                )
                continue
            if cid in my_comment_ids:
                skip_reasons["self_comment_known"] = skip_reasons.get("self_comment_known", 0) + 1
                continue
            if is_self_author(c_author_id, c_author_name, self_identity_keys=self_identity_keys):
                skip_reasons["self_comment"] = skip_reasons.get("self_comment", 0) + 1
                continue
            if not c_author_key:
                skip_reasons["unknown_author"] = skip_reasons.get("unknown_author", 0) + 1
                logger.debug("Reply scan skipping comment_id=%s reason=unknown_author", cid)
                continue
            parent_cid = comment_parent_id(comment)
            is_reply_to_me = bool(
                parent_cid
                and (
                    parent_cid in my_comment_ids
                    or parent_cid in my_comment_ids_in_post
                )
            )
            # On our own posts we triage all incoming comments.
            # On other posts we only triage replies to comments we authored.
            if not is_my_post and not is_reply_to_me:
                skip_reasons["not_target_reply"] = skip_reasons.get("not_target_reply", 0) + 1
                continue
            pair_key = f"{normalize_str(pid).strip()}:{normalize_str(cid).strip()}"
            author_post_key = f"{normalize_str(pid).strip()}:{c_author_key or '(unknown-author)'}"

            replies_to_author = replies_by_author_on_post.get(c_author_key, 0) if c_author_key else 0
            if replies_to_author >= MAX_REPLIES_PER_AUTHOR_PER_POST:
                skip_reasons["author_thread_reply_cap"] = skip_reasons.get("author_thread_reply_cap", 0) + 1
                logger.info(
                    (
                        "Reply scan skipping comment_id=%s post_id=%s reason=author_thread_reply_cap "
                        "author=%s replies_to_author=%s cap=%s"
                    ),
                    cid,
                    pid,
                    c_author_name or c_author_key or "(unknown)",
                    replies_to_author,
                    MAX_REPLIES_PER_AUTHOR_PER_POST,
                )
                continue

            turns_with_author = conversation_turns_by_author_on_post.get(c_author_key, 0) if c_author_key else 0
            post_submolt = normalize_submolt(post.get("submolt"))
            post_title = normalize_str(post.get("title")) or f"Post {pid}"
            url = post_url(pid)
            incoming_body = normalize_str(comment.get("content"))
            forced_reply_content = ""
            if should_correct_wrong_community_claim(incoming_body, post_submolt):
                forced_reply_content = build_wrong_community_correction_reply(
                    post_submolt=post_submolt,
                    post_title_text=post_title,
                )
                logger.info(
                    "Reply scan forcing submolt correction comment_id=%s submolt=%s",
                    cid,
                    post_submolt,
                )
            if looks_hostile_content(incoming_body):
                skip_reasons["hostile_comment"] = skip_reasons.get("hostile_comment", 0) + 1
                logger.info("Reply scan flagged hostile comment comment_id=%s", cid)
            is_spammy = looks_spammy_comment(incoming_body) if not forced_reply_content else False
            is_irrelevant_noise = looks_irrelevant_noise_comment(incoming_body) if not forced_reply_content else False
            if (is_spammy or is_irrelevant_noise) and not forced_reply_content:
                author_bad_key = _normalized_name_key(c_author_name) or normalize_str(c_author_key).strip().lower()
                author_post_bad_key = f"{normalize_str(pid).strip().lower()}::{author_bad_key}" if author_bad_key else ""
                if is_spammy and author_bad_key:
                    bad_bot_counts[author_bad_key] = bad_bot_counts.get(author_bad_key, 0) + 1
                    if author_post_bad_key:
                        bad_bot_counts_by_post[author_post_bad_key] = bad_bot_counts_by_post.get(author_post_bad_key, 0) + 1
                if is_spammy:
                    skip_reasons["spam_comment"] = skip_reasons.get("spam_comment", 0) + 1
                    logger.info("Reply scan skipping spam comment_id=%s", cid)
                else:
                    skip_reasons["irrelevant_noise_comment"] = skip_reasons.get("irrelevant_noise_comment", 0) + 1
                    logger.info("Reply scan skipping low-signal comment_id=%s", cid)
                # Warn only on repeated, overt spam behavior to avoid false positives on legitimate technical comments.
                warning_strike_threshold = max(2, int(cfg.badbot_warning_min_strikes))
                if (
                    is_spammy
                    and is_overt_spam_comment(incoming_body)
                    and
                    is_my_post
                    and cfg.badbot_warning_enabled
                    and bad_bot_warnings_sent_this_scan < max(0, cfg.badbot_max_warnings_per_scan)
                    and cid not in bad_bot_warned_comment_ids
                    and author_bad_key
                ):
                    strike_count = int(bad_bot_counts.get(author_bad_key, 1))
                    post_strike_count = int(bad_bot_counts_by_post.get(author_post_bad_key, 0)) if author_post_bad_key else 0
                    author_warn_count = int(bad_bot_warnings_by_author_day.get(author_bad_key, 0) or 0)
                    if (
                        strike_count >= warning_strike_threshold
                        and post_strike_count >= warning_strike_threshold
                        and author_warn_count < max(0, cfg.badbot_max_warnings_per_author_per_day)
                    ):
                        # Use daily warning count for tone so stale historical totals do not create hostile messaging.
                        warning_level = author_warn_count + 1
                        warning_reply = build_badbot_warning_reply(
                            author_name=c_author_name or author_bad_key,
                            strike_count=warning_level,
                        )
                        warning_reply = _prepare_publish_content(
                            warning_reply,
                            allow_links=False,
                            audit_label="badbot_warning",
                        )
                        if not warning_reply:
                            skip_reasons["empty_badbot_warning"] = skip_reasons.get("empty_badbot_warning", 0) + 1
                            continue
                        approved, approve_all_actions, should_stop = confirm_action(
                            cfg=cfg,
                            logger=logger,
                            action=f"warn-badbot-{cid}",
                            pid=pid,
                            title=post_title,
                            submolt=post_submolt,
                            url=url,
                            author=c_author_name or "(unknown)",
                            content_preview=preview_text(warning_reply),
                            approve_all=approve_all_actions,
                        )
                        if should_stop:
                            state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                            save_state(cfg.state_path, state)
                            return approve_all_actions
                        if approved:
                            comment_allowed, _ = comment_gate_status(state=state, cfg=cfg)
                            if comment_allowed:
                                warning_sig = _publish_signature(
                                    action_type="comment",
                                    target_post_id=pid,
                                    parent_comment_id=cid,
                                    content=warning_reply,
                                )
                                if _seen_publish_signature(state, warning_sig):
                                    logger.info(
                                        "Reply scan skipping duplicate badbot warning comment_id=%s",
                                        cid,
                                    )
                                    continue
                                try:
                                    comment_resp = client.create_comment(pid, warning_reply, parent_id=cid)
                                    register_my_comment_id(state=state, response_payload=comment_resp)
                                    _remember_publish_signature(state, warning_sig)
                                    state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                                    mark_reply_action_timestamps(state=state, action_kind="comment")
                                    replied_to_comment_ids.add(cid)
                                    replied_comment_pairs.add(pair_key)
                                    bad_bot_warned_comment_ids.add(cid)
                                    bad_bot_warnings_by_author_day[author_bad_key] = author_warn_count + 1
                                    bad_bot_warnings_sent_this_scan += 1
                                    actions += 1
                                    try:
                                        append_action_journal(
                                            cfg.action_journal_path,
                                            action_type="comment",
                                            target_post_id=pid,
                                            submolt=post_submolt,
                                            title=post_title,
                                            content=warning_reply,
                                            parent_comment_id=cid,
                                            reference_post_id=pid,
                                            url=url,
                                            reference={
                                                "post_id": pid,
                                                "post_title": post_title,
                                                "comment_id": cid,
                                                "comment_author": c_author_name or c_author_key,
                                                "comment_content": incoming_body,
                                            },
                                            meta={"source": "startup_reply_scan", "kind": "badbot_warning"},
                                        )
                                    except Exception as e:
                                        logger.debug(
                                            "Action journal write failed post_id=%s comment_id=%s error=%s",
                                            pid,
                                            cid,
                                            e,
                                        )
                                    print_success_banner(
                                        action="comment-badbot-warning",
                                        pid=pid,
                                        url=url,
                                        title=post_title,
                                    )
                                except Exception as e:
                                    logger.warning("Reply scan badbot warning failed comment_id=%s error=%s", cid, e)
                continue
            if turns_with_author >= THREAD_ESCALATE_TURNS:
                if author_post_key in thread_followup_posted_pairs:
                    skip_reasons["thread_followup_already_posted"] = (
                        skip_reasons.get("thread_followup_already_posted", 0) + 1
                    )
                    logger.info(
                        "Reply scan skipping deep-thread escalation already posted post_id=%s author=%s turns=%s",
                        pid,
                        c_author_name or c_author_key or "(unknown)",
                        turns_with_author,
                    )
                    continue

                followup_title = build_thread_followup_post_title(post_title_text=post_title)
                followup_content = build_thread_followup_post_content(
                    source_url=url,
                    author_name=c_author_name or "(unknown)",
                    source_comment=incoming_body,
                    proposed_reply=(
                        "eUTXO plus ErgoScript lets us enforce deterministic settlement rules while keeping counterparties auditable."
                    ),
                )
                followup_content = _prepare_publish_content(
                    followup_content,
                    allow_links=True,
                    audit_label="thread_followup_post",
                )
                if not followup_content:
                    skip_reasons["empty_followup_content"] = skip_reasons.get("empty_followup_content", 0) + 1
                    continue
                post_allowed, post_reason = post_gate_status(state=state, cfg=cfg)
                if not post_allowed:
                    key = f"thread_followup_{post_reason}"
                    skip_reasons[key] = skip_reasons.get(key, 0) + 1
                    logger.info(
                        (
                            "Reply scan deep-thread escalation deferred comment_id=%s post_id=%s "
                            "reason=%s turns=%s"
                        ),
                        cid,
                        pid,
                        post_reason,
                        turns_with_author,
                    )
                    continue

                approved, approve_all_actions, should_stop = confirm_action(
                    cfg=cfg,
                    logger=logger,
                    action=f"post-followup-thread-{cid}",
                    pid=pid,
                    title=followup_title,
                    submolt=post_submolt,
                    url=url,
                    author=c_author_name or "(unknown)",
                    content_preview=preview_text(followup_content),
                    approve_all=approve_all_actions,
                )
                if should_stop:
                    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                    save_state(cfg.state_path, state)
                    return approve_all_actions
                if approved:
                    followup_sig = _publish_signature(
                        action_type="post",
                        target_post_id="(new)",
                        parent_comment_id=None,
                        content=followup_content,
                    )
                    if _seen_publish_signature(state, followup_sig):
                        logger.info(
                            "Reply scan skipping duplicate followup post source_post_id=%s comment_id=%s",
                            pid,
                            cid,
                        )
                        continue
                    try:
                        post_resp = client.create_post(submolt=post_submolt, title=followup_title, content=followup_content)
                        _remember_publish_signature(state, followup_sig)
                        created_post_id = post_id(post_resp) or (post_resp.get("post") or {}).get("id") or "(unknown)"
                        state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                        mark_reply_action_timestamps(state=state, action_kind="post")
                        thread_followup_posted_pairs.add(author_post_key)
                        state["thread_followup_posted_pairs"] = list(thread_followup_posted_pairs)[-10000:]
                        replied_to_comment_ids.add(cid)
                        replied_comment_pairs.add(pair_key)
                        state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                        state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                        actions += 1
                        try:
                            append_action_journal(
                                cfg.action_journal_path,
                                action_type="post",
                                target_post_id=str(created_post_id),
                                submolt=post_submolt,
                                title=followup_title,
                                content=followup_content,
                                reference_post_id=pid,
                                url=post_url(str(created_post_id)) if created_post_id != "(unknown)" else url,
                                reference={
                                    "post_id": pid,
                                    "post_title": post_title,
                                    "comment_id": cid,
                                    "comment_author": c_author_name or c_author_key,
                                    "comment_content": incoming_body,
                                },
                                meta={"source": "startup_reply_scan", "kind": "thread_followup"},
                            )
                        except Exception as e:
                            logger.debug(
                                "Action journal write failed followup source_post_id=%s new_post_id=%s error=%s",
                                pid,
                                created_post_id,
                                e,
                            )
                        print_success_banner(
                            action="post-followup",
                            pid=str(created_post_id),
                            url=post_url(str(created_post_id)) if created_post_id != "(unknown)" else url,
                            title=followup_title,
                        )
                    except Exception as e:
                        logger.warning(
                            "Reply scan followup post failed source_post_id=%s comment_id=%s error=%s",
                            pid,
                            cid,
                            e,
                        )
                continue

            already_replied = (
                cid in replied_to_comment_ids
                or cid in my_replied_parent_ids
                or pair_key in replied_comment_pairs
            )
            pending_reply_exists = has_pending_comment_action(
                state=state,
                post_id_value=pid,
                parent_comment_id=cid,
            )
            if cid in seen_comment_ids and (already_replied or pending_reply_exists or not is_my_post):
                skip_reasons["already_seen"] = skip_reasons.get("already_seen", 0) + 1
                continue
            was_new_comment = False
            if cid not in seen_comment_ids:
                seen_comment_ids.add(cid)
                new_replies += 1
                was_new_comment = True

            triage: Dict[str, Any]
            triage = triage_cache.get(cid, {})
            if triage:
                triage_cache_hits += 1
            else:
                deterministic = _deterministic_reply_triage(
                    comment_body=incoming_body,
                    is_my_post=is_my_post,
                    is_reply_to_me=is_reply_to_me,
                )
                if deterministic:
                    triage = deterministic
                    triage_cache[cid] = {
                        "should_respond": bool(triage.get("should_respond")),
                        "confidence": float(triage.get("confidence", 0.0)),
                        "response_mode": normalize_response_mode(triage.get("response_mode"), default="none"),
                        "title": normalize_str(triage.get("title")).strip(),
                        "content": normalize_str(triage.get("content")).strip(),
                        "vote_action": normalize_vote_action(triage.get("vote_action")),
                        "vote_target": normalize_vote_target(triage.get("vote_target")),
                        "ts": utc_now().isoformat(),
                    }
                    skip_reasons["triage_deterministic"] = skip_reasons.get("triage_deterministic", 0) + 1
                else:
                    if not was_new_comment:
                        triage_deferred += 1
                        skip_reasons["triage_existing_no_cache"] = skip_reasons.get("triage_existing_no_cache", 0) + 1
                        continue
                    if triage_llm_calls >= triage_llm_budget:
                        triage_deferred += 1
                        skip_reasons["triage_budget_deferred"] = skip_reasons.get("triage_budget_deferred", 0) + 1
                        if was_new_comment:
                            seen_comment_ids.discard(cid)
                            new_replies = max(0, new_replies - 1)
                        continue
                    try:
                        messages = build_reply_triage_messages(
                            persona=persona_text,
                            domain_context=domain_context_text,
                            post=post,
                            comment=comment,
                            post_id=pid,
                            comment_id=cid,
                        )
                        triage, triage_provider, _ = call_generation_model(cfg, messages)
                        triage_llm_calls += 1
                        provider_counts[triage_provider] = provider_counts.get(triage_provider, 0) + 1
                        triage_cache[cid] = {
                            "should_respond": bool(triage.get("should_respond")),
                            "confidence": float(triage.get("confidence", 0.0)),
                            "response_mode": normalize_response_mode(triage.get("response_mode"), default="none"),
                            "title": normalize_str(triage.get("title")).strip(),
                            "content": normalize_str(triage.get("content")).strip(),
                            "vote_action": normalize_vote_action(triage.get("vote_action")),
                            "vote_target": normalize_vote_target(triage.get("vote_target")),
                            "ts": utc_now().isoformat(),
                        }
                        logger.debug(
                            "Reply triage generated comment_id=%s provider=%s",
                            cid,
                            triage_provider,
                        )
                    except Exception as e:
                        triage_fail_count += 1
                        err_text = normalize_str(e)
                        is_parse_fail = (
                            "Chatbase returned non-JSON text" in err_text
                            or "empty text response" in err_text
                            or "Expecting value: line 1 column 1" in err_text
                        )
                        if is_parse_fail:
                            triage_parse_fail_count += 1
                            if triage_parse_fail_count <= 2:
                                logger.warning("Reply triage failed comment_id=%s error=%s", cid, e)
                            else:
                                logger.debug("Reply triage parse failure comment_id=%s error=%s", cid, e)
                        else:
                            logger.warning("Reply triage failed comment_id=%s error=%s", cid, e)
                        triage = {
                            "should_respond": False,
                            "confidence": 0.0,
                            "vote_action": "none",
                            "vote_target": "none",
                            "response_mode": "none",
                            "title": "",
                            "content": "",
                        }

            vote_action = normalize_vote_action(triage.get("vote_action"))
            response_mode = normalize_response_mode(triage.get("response_mode"), default="none")
            triage_should_respond = bool(triage.get("should_respond"))
            confidence, triage_confidence_imputed = _normalized_model_confidence(
                raw_confidence=triage.get("confidence", 0),
                should_act=triage_should_respond,
                has_content=bool(normalize_str(triage.get("content")).strip()),
                fallback_when_true=max(cfg.min_confidence, 0.72),
                fallback_when_false=0.0,
            )
            if triage_confidence_imputed and triage_should_respond:
                logger.debug(
                    "Reply triage confidence imputed comment_id=%s raw=%r normalized=%.3f",
                    cid,
                    triage.get("confidence"),
                    confidence,
                )
            url = post_url(pid)
            post_submolt = normalize_submolt(post.get("submolt"))
            post_title = normalize_str(post.get("title")) or f"Post {pid}"
            incoming_body = normalize_str(comment.get("content"))

            if vote_action != "none":
                if cid in voted_comment_ids:
                    logger.info("Reply scan skipping vote; already voted comment_id=%s", cid)
                    vote_action = "none"
                existing_vote = extract_my_vote_from_comment(comment)
                if vote_action != "none" and existing_vote == vote_action:
                    logger.info(
                        "Reply scan skipping vote; API shows existing vote comment_id=%s vote=%s",
                        cid,
                        existing_vote,
                    )
                    voted_comment_ids.add(cid)
                    vote_action = "none"
                if vote_action == "downvote" and not cfg.allow_comment_downvote:
                    logger.info(
                        "Reply scan skipping unsupported vote action downvote-comment comment_id=%s",
                        cid,
                    )
                    skip_reasons["downvote_unsupported"] = skip_reasons.get("downvote_unsupported", 0) + 1
                    vote_action = "none"
                # Even when vote is skipped, still continue with reply-triage action path.
                if vote_action != "none":
                    approved, approve_all_actions, should_stop = confirm_action(
                        cfg=cfg,
                        logger=logger,
                        action=f"{vote_action}-comment",
                        pid=cid,
                        title=f"Reply on '{post_title}'",
                        submolt=normalize_submolt(post.get("submolt")),
                        url=url,
                        author=c_author_name or "(unknown)",
                        content_preview=preview_text(incoming_body),
                        approve_all=approve_all_actions,
                    )
                    if should_stop:
                        state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                        save_state(cfg.state_path, state)
                        return approve_all_actions
                    if approved:
                        try:
                            client.vote_comment(cid, vote_action=vote_action)
                            voted_comment_ids.add(cid)
                            state["voted_comment_ids"] = list(voted_comment_ids)[-10000:]
                            actions += 1
                            try:
                                append_action_journal(
                                    cfg.action_journal_path,
                                    action_type=f"{vote_action}-comment",
                                    target_post_id=pid,
                                    submolt=normalize_submolt(post.get("submolt")),
                                    title=f"{vote_action} comment on '{post_title}'",
                                    content="",
                                    parent_comment_id=cid,
                                    reference_post_id=pid,
                                    url=url,
                                    reference={
                                        "post_id": pid,
                                        "post_title": post_title,
                                        "comment_id": cid,
                                        "comment_author": c_author_name or c_author_key,
                                        "comment_content": incoming_body,
                                    },
                                    meta={"source": "startup_reply_scan", "kind": "vote_comment", "vote": vote_action},
                                )
                            except Exception as e:
                                logger.debug("Vote action journal write failed comment_id=%s error=%s", cid, e)
                            print_success_banner(
                                action=f"{vote_action}-comment",
                                pid=cid,
                                url=url,
                                title=f"Reply on '{post_title}'",
                            )
                        except Exception as e:
                            logger.warning("Reply vote failed comment_id=%s vote=%s error=%s", cid, vote_action, e)

            should_respond = bool(triage.get("should_respond")) or bool(forced_reply_content)
            if already_replied:
                logger.info("Reply scan skipping reply; already replied to comment_id=%s", cid)
                replied_to_comment_ids.add(cid)
                replied_comment_pairs.add(pair_key)
                state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                skip_reasons["already_replied"] = skip_reasons.get("already_replied", 0) + 1
                continue

            reply_content = ""
            if forced_reply_content:
                reply_content = forced_reply_content
            elif should_respond and response_mode != "none" and confidence >= cfg.min_confidence:
                reply_content = format_content(triage)
            reply_content = _prepare_publish_content(
                reply_content,
                allow_links=False,
                audit_label="thread_reply",
            )

            if not reply_content:
                if not should_respond:
                    skip_reasons["triage_declined"] = skip_reasons.get("triage_declined", 0) + 1
                elif response_mode == "none":
                    skip_reasons["response_mode_none"] = skip_reasons.get("response_mode_none", 0) + 1
                elif confidence < cfg.min_confidence:
                    skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1
                else:
                    skip_reasons["empty_reply_content"] = skip_reasons.get("empty_reply_content", 0) + 1
                continue

            if not should_respond:
                skip_reasons["triage_declined"] = skip_reasons.get("triage_declined", 0) + 1
            elif response_mode == "none":
                skip_reasons["response_mode_none"] = skip_reasons.get("response_mode_none", 0) + 1
            elif confidence < cfg.min_confidence:
                skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1

            if not reply_content.strip():
                skip_reasons["empty_reply_content"] = skip_reasons.get("empty_reply_content", 0) + 1
                continue
            if not forced_reply_content and _is_template_like_generated_content(reply_content):
                skip_reasons["template_like_reply"] = skip_reasons.get("template_like_reply", 0) + 1
                logger.info("Reply scan skipping template-like reply comment_id=%s", cid)
                continue
            if not forced_reply_content and _is_low_value_affirmation_reply(reply_content):
                skip_reasons["low_value_reply"] = skip_reasons.get("low_value_reply", 0) + 1
                logger.info("Reply scan skipping low-value reply comment_id=%s", cid)
                continue
            if not forced_reply_content and not _passes_generated_content_quality(
                content=reply_content,
                requested_mode="comment",
            ):
                skip_reasons["reply_quality_gate"] = skip_reasons.get("reply_quality_gate", 0) + 1
                logger.info("Reply scan skipping low-quality reply comment_id=%s", cid)
                continue

            if has_my_reply_to_comment(
                client=client,
                post_id_value=pid,
                parent_comment_id=cid,
                my_name=my_name,
                logger=logger,
                self_identity_keys=self_identity_keys,
            ):
                replied_to_comment_ids.add(cid)
                replied_comment_pairs.add(pair_key)
                state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                skip_reasons["already_replied_onchain"] = skip_reasons.get("already_replied_onchain", 0) + 1
                logger.info("Reply scan skipping reply; on-chain reply already exists comment_id=%s", cid)
                continue
            if c_author_key and c_author_key in self_identity_keys:
                skip_reasons["self_comment_late_guard"] = skip_reasons.get("self_comment_late_guard", 0) + 1
                logger.info("Reply scan late-guard skip comment_id=%s reason=self_comment", cid)
                continue

            comment_allowed, comment_reason = comment_gate_status(state=state, cfg=cfg)
            if comment_allowed:
                approved, approve_all_actions, should_stop = confirm_action(
                    cfg=cfg,
                    logger=logger,
                    action=f"comment-reply-to-{cid}",
                    pid=pid,
                    title=post_title,
                    submolt=normalize_submolt(post.get("submolt")),
                    url=url,
                    author=c_author_name or "(unknown)",
                    content_preview=preview_text(reply_content),
                    approve_all=approve_all_actions,
                )
                if should_stop:
                    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                    save_state(cfg.state_path, state)
                    return approve_all_actions
                if approved:
                    reply_sig = _publish_signature(
                        action_type="comment",
                        target_post_id=pid,
                        parent_comment_id=cid,
                        content=reply_content,
                    )
                    if _seen_publish_signature(state, reply_sig):
                        logger.info("Reply scan skipping duplicate reply comment_id=%s", cid)
                        continue
                    try:
                        comment_resp = client.create_comment(pid, reply_content, parent_id=cid)
                        register_my_comment_id(state=state, response_payload=comment_resp)
                        _remember_publish_signature(state, reply_sig)
                        state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                        mark_reply_action_timestamps(state=state, action_kind="comment")
                        replied_post_ids.add(pid)
                        state["replied_post_ids"] = list(replied_post_ids)[-10000:]
                        maybe_upvote_post_after_comment(
                            client=client,
                            state=state,
                            logger=logger,
                            post_id_value=pid,
                            journal_path=cfg.action_journal_path,
                            submolt=normalize_submolt(post.get("submolt")),
                            post_title=post_title,
                            url=url,
                            reference={
                                "post_id": pid,
                                "post_title": post_title,
                            },
                            meta={"source": "startup_reply_scan"},
                        )
                        replied_to_comment_ids.add(cid)
                        replied_comment_pairs.add(pair_key)
                        state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                        state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                        if c_author_key:
                            replies_by_author_on_post[c_author_key] = replies_by_author_on_post.get(c_author_key, 0) + 1
                            conversation_turns_by_author_on_post[c_author_key] = (
                                conversation_turns_by_author_on_post.get(c_author_key, 0) + 1
                            )
                        actions += 1
                        try:
                            append_action_journal(
                                cfg.action_journal_path,
                                action_type="comment",
                                target_post_id=pid,
                                submolt=normalize_submolt(post.get("submolt")),
                                title=post_title,
                                content=reply_content,
                                parent_comment_id=cid,
                                reference_post_id=pid,
                                url=url,
                                reference={
                                    "post_id": pid,
                                    "post_title": post_title,
                                    "comment_id": cid,
                                    "comment_author": c_author_name or c_author_key,
                                    "comment_content": incoming_body,
                                },
                                meta={"source": "startup_reply_scan", "kind": "thread_reply"},
                            )
                        except Exception as e:
                            logger.debug(
                                "Action journal write failed post_id=%s comment_id=%s error=%s",
                                pid,
                                cid,
                                e,
                            )
                        print_success_banner(action="comment-reply", pid=pid, url=url, title=post_title)
                    except Exception as e:
                        logger.warning("Reply comment failed post_id=%s error=%s", pid, e)
                continue

            if comment_reason == "comment_cooldown":
                approved, approve_all_actions, should_stop = confirm_action(
                    cfg=cfg,
                    logger=logger,
                    action=f"wait-comment-reply-to-{cid}",
                    pid=pid,
                    title=post_title,
                    submolt=normalize_submolt(post.get("submolt")),
                    url=url,
                    author=c_author_name or "(unknown)",
                    content_preview=preview_text(reply_content),
                    approve_all=approve_all_actions,
                )
                if should_stop:
                    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
                    save_state(cfg.state_path, state)
                    return approve_all_actions
                if approved:
                    if wait_for_comment_slot(state=state, cfg=cfg, logger=logger):
                        reply_sig = _publish_signature(
                            action_type="comment",
                            target_post_id=pid,
                            parent_comment_id=cid,
                            content=reply_content,
                        )
                        if _seen_publish_signature(state, reply_sig):
                            logger.info("Reply scan skipping duplicate reply after wait comment_id=%s", cid)
                            continue
                        try:
                            comment_resp = client.create_comment(pid, reply_content, parent_id=cid)
                            register_my_comment_id(state=state, response_payload=comment_resp)
                            _remember_publish_signature(state, reply_sig)
                            state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                            mark_reply_action_timestamps(state=state, action_kind="comment")
                            replied_post_ids.add(pid)
                            state["replied_post_ids"] = list(replied_post_ids)[-10000:]
                            replied_to_comment_ids.add(cid)
                            replied_comment_pairs.add(pair_key)
                            state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
                            state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
                            if c_author_key:
                                replies_by_author_on_post[c_author_key] = replies_by_author_on_post.get(c_author_key, 0) + 1
                                conversation_turns_by_author_on_post[c_author_key] = (
                                    conversation_turns_by_author_on_post.get(c_author_key, 0) + 1
                                )
                            maybe_upvote_post_after_comment(
                                client=client,
                                state=state,
                                logger=logger,
                                post_id_value=pid,
                                journal_path=cfg.action_journal_path,
                                submolt=normalize_submolt(post.get("submolt")),
                                post_title=post_title,
                                url=url,
                                reference={
                                    "post_id": pid,
                                    "post_title": post_title,
                                },
                                meta={"source": "startup_reply_scan"},
                            )
                            actions += 1
                            try:
                                append_action_journal(
                                    cfg.action_journal_path,
                                    action_type="comment",
                                    target_post_id=pid,
                                    submolt=normalize_submolt(post.get("submolt")),
                                    title=post_title,
                                    content=reply_content,
                                    parent_comment_id=cid,
                                    reference_post_id=pid,
                                    url=url,
                                    reference={
                                        "post_id": pid,
                                        "post_title": post_title,
                                        "comment_id": cid,
                                        "comment_author": c_author_name or c_author_key,
                                        "comment_content": incoming_body,
                                    },
                                    meta={"source": "startup_reply_scan", "kind": "thread_reply_after_wait"},
                                )
                            except Exception as e:
                                logger.debug(
                                    "Action journal write failed post_id=%s comment_id=%s error=%s",
                                    pid,
                                    cid,
                                    e,
                                )
                            print_success_banner(action="comment-reply", pid=pid, url=url, title=post_title)
                        except Exception as e:
                            logger.warning("Waited reply comment failed post_id=%s error=%s", pid, e)
                continue

            skip_reasons[comment_reason] = skip_reasons.get(comment_reason, 0) + 1

    state["seen_comment_ids"] = list(seen_comment_ids)[-10000:]
    state["my_comment_ids"] = list(my_comment_ids)[-20000:]
    state["replied_post_ids"] = list(replied_post_ids)[-10000:]
    state["voted_comment_ids"] = list(voted_comment_ids)[-10000:]
    state["replied_to_comment_ids"] = list(replied_to_comment_ids)[-10000:]
    state["replied_comment_pairs"] = list(replied_comment_pairs)[-20000:]
    state["thread_followup_posted_pairs"] = list(thread_followup_posted_pairs)[-10000:]
    state["reply_triage_cache"] = dict(list(triage_cache.items())[-10000:])
    state["bad_bot_counts"] = bad_bot_counts
    state["bad_bot_counts_by_post"] = bad_bot_counts_by_post
    state["bad_bot_warned_comment_ids"] = list(bad_bot_warned_comment_ids)[-10000:]
    state["bad_bot_warnings_by_author_day"] = bad_bot_warnings_by_author_day
    save_state(cfg.state_path, state)
    logger.info(
        "Reply scan complete scanned_comments=%s new_replies=%s actions=%s pending=%s",
        scanned,
        new_replies,
        actions,
        len(state.get("pending_actions", [])),
    )
    if provider_counts:
        provider_summary = ", ".join([f"{k}={v}" for k, v in sorted(provider_counts.items())])
        logger.info("Reply scan provider_summary %s", provider_summary)
    if skip_reasons:
        summary = ", ".join([f"{k}={v}" for k, v in sorted(skip_reasons.items())])
        logger.info("Reply scan skip_summary %s", summary)
    if triage_fail_count:
        logger.info(
            "Reply scan triage_failures total=%s parse_format=%s",
            triage_fail_count,
            triage_parse_fail_count,
        )
    logger.info(
        "Reply scan triage_usage cache_hits=%s llm_calls=%s deferred=%s budget=%s",
        triage_cache_hits,
        triage_llm_calls,
        triage_deferred,
        triage_llm_budget,
    )
    leaderboard = top_badbots(bad_bot_counts, limit=5)
    if leaderboard:
        board_text = ", ".join([f"{name}:{count}" for name, count in leaderboard])
        logger.info("BadBots leaderboard top=%s", board_text)
    return approve_all_actions


def maybe_subscribe_relevant_submolts(
    client: MoltbookClient,
    cfg: Config,
    logger,
    state: Dict[str, Any],
    approve_all_actions: bool,
) -> bool:
    if not cfg.auto_subscribe_submolts:
        return approve_all_actions

    try:
        payload = client.list_submolts()
    except Exception as e:
        logger.warning("Submolt discovery skipped: %s", e)
        return approve_all_actions

    candidates_all = discover_relevant_submolts(payload=payload, target_submolts=cfg.target_submolts)
    approved_submolts = {normalize_str(x).strip().lower() for x in state.get("approved_submolts", [])}
    dismissed_submolts = {normalize_str(x).strip().lower() for x in state.get("dismissed_submolts", [])}
    target_set = {x.strip().lower() for x in cfg.target_submolts if x.strip()}
    prioritized = [name for name in candidates_all if name in target_set]
    extras = [name for name in candidates_all if name not in target_set][:5]
    candidates = [name for name in prioritized + extras if name not in approved_submolts and name not in dismissed_submolts]
    if not candidates:
        logger.info("Submolt discovery found no relevant candidates.")
        return approve_all_actions

    logger.info("Relevant submolts discovered count=%s", len(candidates))
    for submolt in candidates:
        approved, approve_all_actions, should_stop = confirm_action(
            cfg=cfg,
            logger=logger,
            action="subscribe-submolt",
            pid=submolt,
            title=f"Subscribe to m/{submolt}",
            submolt=submolt,
            url=f"https://moltbook.com/m/{submolt}",
            author="system",
            content_preview=preview_text(
                "Subscribe so discovery/feed includes this community more reliably."
            ),
            approve_all=approve_all_actions,
        )
        if should_stop:
            return approve_all_actions
        if not approved:
            dismissed_submolts.add(submolt)
            state["dismissed_submolts"] = sorted(dismissed_submolts)
            continue
        try:
            client.subscribe_submolt(submolt)
            approved_submolts.add(submolt)
            dismissed_submolts.discard(submolt)
            state["approved_submolts"] = sorted(approved_submolts)
            state["dismissed_submolts"] = sorted(dismissed_submolts)
            logger.info("Subscribed to submolt=%s", submolt)
            print_success_banner(
                action="subscribe-submolt",
                pid=submolt,
                url=f"https://moltbook.com/m/{submolt}",
                title=f"m/{submolt}",
            )
        except Exception as e:
            msg = str(e).lower()
            if "already" in msg:
                approved_submolts.add(submolt)
                dismissed_submolts.discard(submolt)
                state["approved_submolts"] = sorted(approved_submolts)
                state["dismissed_submolts"] = sorted(dismissed_submolts)
                logger.info("Already subscribed to submolt=%s", submolt)
                continue
            logger.warning("Subscribe failed submolt=%s error=%s", submolt, e)

    return approve_all_actions


def maybe_follow_ergo_authors(
    client: MoltbookClient,
    cfg: Config,
    logger,
    state: Dict[str, Any],
    posts: List[Dict[str, Any]],
    my_name: Optional[str],
    self_identity_keys: Set[str],
    approve_all_actions: bool,
) -> bool:
    if not cfg.follow_ergo_authors_enabled:
        return approve_all_actions
    followed = {normalize_str(x).strip().lower() for x in state.get("followed_agents", []) if normalize_str(x).strip()}
    dismissed = {
        normalize_str(x).strip().lower()
        for x in state.get("dismissed_follow_agents", [])
        if normalize_str(x).strip()
    }
    candidates: List[Tuple[str, str, str, List[str]]] = []
    for post in posts[:80]:
        title = normalize_str(post.get("title")).strip()
        content = normalize_str(post.get("content")).strip()
        submolt = normalize_submolt(post.get("submolt"))
        strong, terms = is_strong_ergo_post(title=title, content=content, submolt=submolt)
        if not strong:
            continue
        author_id, author_name = post_author(post)
        if is_self_author(author_id, author_name, self_identity_keys=self_identity_keys):
            continue
        author_key = normalize_str(author_name).strip().lower()
        if not author_key or author_key in followed or author_key in dismissed:
            continue
        pid = post_id(post) or "(unknown)"
        candidates.append((author_name or "(unknown)", pid, title or "(untitled)", terms[:4]))

    if not candidates:
        return approve_all_actions

    followed_count = 0
    for author_name, pid, title, terms in candidates:
        if followed_count >= max(1, cfg.follow_ergo_authors_per_cycle):
            break
        approved, approve_all_actions, should_stop = confirm_action(
            cfg=cfg,
            logger=logger,
            action="follow-agent",
            pid=pid,
            title=title,
            submolt=normalize_submolt("general"),
            url=post_url(pid),
            author=author_name,
            content_preview=preview_text(
                f"Detected strong Ergo signal terms: {', '.join(terms) if terms else '(none)'}\n"
                "Follow this author to improve relevant feed coverage."
            ),
            approve_all=approve_all_actions,
        )
        if should_stop:
            break
        author_key = normalize_str(author_name).strip().lower()
        if not approved:
            if author_key:
                dismissed.add(author_key)
            continue
        try:
            client.follow_agent(author_name)
            followed_count += 1
            if author_key:
                followed.add(author_key)
            logger.info("Followed author=%s reason=strong_ergo_signal", author_name)
            print_success_banner(
                action="follow-agent",
                pid=pid,
                url=post_url(pid),
                title=f"Followed {author_name}",
            )
        except Exception as e:
            msg = normalize_str(e).lower()
            if "already" in msg:
                if author_key:
                    followed.add(author_key)
                logger.info("Already following author=%s", author_name)
                continue
            logger.warning("Follow author failed author=%s error=%s", author_name, e)

    state["followed_agents"] = sorted(followed)
    state["dismissed_follow_agents"] = sorted(dismissed)
    return approve_all_actions


def run_loop() -> None:
    cfg = load_config()
    logger = setup_logging(cfg)
    client = MoltbookClient()
    print_runtime_banner(cfg)

    logger.info(
        (
            "Autonomy loop starting discovery_mode=%s reply_mode=%s poll_seconds=%s feed_limit=%s "
            "search_limit=%s idle_poll_seconds=%s dry_run=%s draft_shortlist=%s draft_signal_min_score=%s "
            "dynamic_shortlist=%s dynamic_shortlist_bounds=%s-%s proactive_daily_target=%s "
            "proactive_force_general=%s llm_provider=%s openai_configured=%s "
            "chatbase_configured=%s auto_openai_fallback=%s self_improve_enabled=%s state_path=%s"
        ),
        cfg.discovery_mode,
        cfg.reply_mode,
        cfg.poll_seconds,
        cfg.feed_limit,
        cfg.search_limit,
        cfg.idle_poll_seconds,
        cfg.dry_run,
        cfg.draft_shortlist_size,
        cfg.draft_signal_min_score,
        cfg.dynamic_shortlist_enabled,
        cfg.dynamic_shortlist_min,
        cfg.dynamic_shortlist_max,
        cfg.proactive_daily_target_posts,
        cfg.proactive_force_general_until_daily_target,
        cfg.llm_provider,
        bool(cfg.openai_api_key),
        bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id),
        cfg.llm_auto_fallback_to_openai,
        cfg.self_improve_enabled,
        cfg.state_path,
    )
    if cfg.log_path:
        logger.info("File logging enabled path=%s", cfg.log_path)
    if cfg.self_improve_enabled:
        logger.info(
            "Self-improvement suggestions paths json=%s text=%s backlog=%s",
            cfg.self_improve_path,
            cfg.self_improve_text_path,
            cfg.self_improve_backlog_path,
        )
    try:
        init_analytics_db(cfg.analytics_db_path)
        logger.info("Analytics DB ready path=%s", cfg.analytics_db_path)
    except Exception as e:
        logger.warning("Analytics DB initialization failed path=%s error=%s", cfg.analytics_db_path, e)

    try:
        claim_status = client.get_claim_status()
        if claim_status not in {"claimed", "active"}:
            logger.error(
                "Agent is not claim-ready (status=%s). Complete manual claim before running autonomy.",
                claim_status,
            )
            return
    except Exception as e:
        logger.warning("Could not verify claim status at startup: %s", e)

    my_name = resolve_self_name(client, logger)
    # Always initialize so downstream branches cannot hit NameError/UnboundLocalError.
    self_identity_keys: Set[str] = set()
    if my_name:
        logger.info("Authenticated as agent=%s", my_name)
    elif cfg.agent_name_hint:
        my_name = cfg.agent_name_hint
        logger.info("Using configured agent name hint=%s", my_name)
    self_identity_keys = resolve_self_identity_keys(client=client, my_name=my_name, logger=logger)

    persona_text = load_persona_text(cfg.persona_path)
    domain_context_text = load_context_text(cfg.context_path)
    keyword_store = load_keyword_store(cfg.keyword_store_path)
    active_keywords = merge_keywords(cfg.keywords, keyword_store.get("learned_keywords", []))
    manual_keyword_review = bool(cfg.confirm_actions and cfg.confirm_timeout_seconds <= 0)
    logger.info(
        (
            "Loaded persona guidance path=%s context_path=%s context_chars=%s "
            "keywords=%s learned_keywords=%s mission_queries=%s do_not_reply_authors=%s"
        ),
        cfg.persona_path,
        cfg.context_path,
        len(domain_context_text),
        len(active_keywords),
        len(keyword_store.get("learned_keywords", [])),
        len(cfg.mission_queries),
        len(cfg.do_not_reply_authors),
    )
    if manual_keyword_review and keyword_store.get("pending_suggestions"):
        active_keywords, keyword_store, _, should_stop = review_pending_keyword_suggestions(
            cfg=cfg,
            logger=logger,
            keyword_store=keyword_store,
            active_keywords=active_keywords,
            approve_all_keyword_changes=False,
        )
        if should_stop:
            logger.info("Stopping run after operator quit during pending keyword review.")
            return

    state = load_state(cfg.state_path)
    post_memory = load_post_engine_memory(cfg.proactive_memory_path)
    if not isinstance(state.get("recent_topic_signatures"), list):
        state["recent_topic_signatures"] = []
    submolt_meta_cache: Dict[str, Any] = {"fetched_ts": 0.0, "items": {}}
    seen: Set[str] = set(state.get("seen_post_ids", []))
    replied_posts: Set[str] = set(state.get("replied_post_ids", []))
    logger.info(
        "Loaded state seen_posts=%s replied_posts=%s proactive_posts=%s",
        len(seen),
        len(replied_posts),
        len(post_memory.get("proactive_posts", [])),
    )
    if cfg.confirm_actions:
        logger.info("Interactive confirmation enabled for outgoing actions.")
    else:
        logger.info("Interactive confirmation disabled (autonomous send mode).")

    def remember_topic_signature(title: str, content: str) -> None:
        sig = infer_topic_signature(title=title, content=content)
        if not sig:
            return
        recent = state.get("recent_topic_signatures", [])
        if not isinstance(recent, list):
            recent = []
        recent.append(sig)
        state["recent_topic_signatures"] = recent[-400:]

    def analytics_event(
        *,
        action_type: str,
        target_post_id: str,
        submolt: str,
        feed_sources: Optional[List[str]],
        virality_score: Optional[float],
        archetype: str,
        model_confidence: Optional[float],
        approved_by_human: bool,
        executed: bool,
        error: str = "",
        title: str = "",
    ) -> None:
        try:
            record_action_event(
                cfg.analytics_db_path,
                action_type=action_type,
                target_post_id=target_post_id,
                submolt=submolt,
                feed_sources=feed_sources or [],
                virality_score=virality_score,
                archetype=archetype,
                model_confidence=model_confidence,
                approved_by_human=approved_by_human,
                executed=executed,
                error=error,
                title=title,
            )
        except Exception as e:
            logger.debug("Analytics event write failed action=%s target=%s error=%s", action_type, target_post_id, e)

    def journal_written_action(
        *,
        action_type: str,
        target_post_id: str,
        submolt: str,
        title: str,
        content: str,
        parent_comment_id: Optional[str] = None,
        reference_post_id: Optional[str] = None,
        url: Optional[str] = None,
        reference: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            append_action_journal(
                cfg.action_journal_path,
                action_type=action_type,
                target_post_id=target_post_id,
                submolt=submolt,
                title=title,
                content=content,
                parent_comment_id=parent_comment_id,
                reference_post_id=reference_post_id,
                url=url,
                reference=reference,
                meta=meta,
            )
        except Exception as e:
            logger.debug(
                "Action journal write failed action=%s target=%s error=%s",
                action_type,
                target_post_id,
                e,
            )

    iteration = 0
    approve_all_actions = False
    approve_all_keyword_changes = False
    startup_priority_post_sent = False
    startup_priority_post_pending = False
    try:
        submolt_meta = get_cached_submolt_meta(
            client=client,
            ttl_seconds=cfg.submolt_cache_seconds,
            cache=submolt_meta_cache,
            logger=logger,
        )
    except Exception as e:
        logger.warning("Submolt metadata unavailable at startup: %s", e)
        submolt_meta = {}
    startup_priority_post_required = should_prioritize_proactive_post(state=state, cfg=cfg)
    if startup_priority_post_required:
        since_last_post = seconds_since_last_post(state=state)
        logger.info(
            "Startup proactive priority trigger post_slot_open=true last_post_age_seconds=%s",
            since_last_post if since_last_post is not None else -1,
        )
        proactive_actions, should_stop, approve_all_actions = maybe_run_proactive_post(
            client=client,
            cfg=cfg,
            logger=logger,
            state=state,
            post_memory=post_memory,
            my_name=my_name,
            persona_text=persona_text,
            domain_context_text=domain_context_text,
            approve_all_actions=approve_all_actions,
            submolt_meta=submolt_meta,
        )
        save_state(cfg.state_path, state)
        save_post_engine_memory(cfg.proactive_memory_path, post_memory)
        if should_stop:
            return
        if proactive_actions > 0:
            startup_priority_post_sent = True
            logger.info("Startup proactive priority posted_before_reply_scan=1")
        else:
            startup_priority_post_pending = True
            logger.info("Startup proactive priority pending=1")

    if startup_priority_post_pending:
        logger.info("Startup submolt discovery deferred reason=priority_post_pending")
    else:
        approve_all_actions = maybe_subscribe_relevant_submolts(
            client=client,
            cfg=cfg,
            logger=logger,
            state=state,
            approve_all_actions=approve_all_actions,
        )
        save_state(cfg.state_path, state)

    if startup_priority_post_sent:
        logger.info("Startup reply scan deferred reason=priority_post_sent")
    elif startup_priority_post_pending or startup_priority_post_required:
        logger.info("Startup reply scan deferred reason=priority_post_pending")
    else:
        approve_all_actions = run_startup_reply_scan(
            client=client,
            cfg=cfg,
            logger=logger,
            state=state,
            my_name=my_name,
            persona_text=persona_text,
            domain_context_text=domain_context_text,
            approve_all_actions=approve_all_actions,
        )
    search_state: Dict[str, Any] = {"retry_cycle": 1, "keyword_cursor": 0}
    while True:
        iteration += 1
        try:
            if not my_name:
                my_name = resolve_self_name(client, logger)
                if my_name:
                    logger.info("Resolved agent identity mid-run=%s", my_name)
                    try:
                        self_identity_keys = resolve_self_identity_keys(
                            client=client,
                            my_name=my_name,
                            logger=logger,
                        )
                    except Exception as e:
                        logger.debug("Could not refresh self identity keys mid-run error=%s", e)
                        self_identity_keys = set()
            try:
                submolt_meta = get_cached_submolt_meta(
                    client=client,
                    ttl_seconds=cfg.submolt_cache_seconds,
                    cache=submolt_meta_cache,
                    logger=logger,
                )
            except Exception as e:
                logger.debug("Submolt metadata refresh failed error=%s", e)
                submolt_meta = submolt_meta_cache.get("items", {}) if isinstance(submolt_meta_cache.get("items"), dict) else {}

            print_cycle_banner(iteration=iteration, mode=cfg.discovery_mode, keywords=len(active_keywords))
            logger.info("Poll cycle=%s start", iteration)
            priority_post_required = should_prioritize_proactive_post(state=state, cfg=cfg)
            pending_executed = 0
            if priority_post_required:
                logger.info("Cycle=%s pending actions deferred reason=priority_post_pending", iteration)
            else:
                pending_executed = execute_pending_actions(
                    client=client,
                    cfg=cfg,
                    state=state,
                    logger=logger,
                    my_name=my_name,
                )
                if pending_executed:
                    save_state(cfg.state_path, state)
            early_proactive_actions = 0
            early_post_action_sent = False
            if should_prioritize_proactive_post(state=state, cfg=cfg):
                since_last_post = seconds_since_last_post(state=state)
                logger.info(
                    "Cycle=%s proactive priority trigger post_slot_open=true last_post_age_seconds=%s",
                    iteration,
                    since_last_post if since_last_post is not None else -1,
                )
                proactive_actions, should_stop, approve_all_actions = maybe_run_proactive_post(
                    client=client,
                    cfg=cfg,
                    logger=logger,
                    state=state,
                    post_memory=post_memory,
                    my_name=my_name,
                    persona_text=persona_text,
                    domain_context_text=domain_context_text,
                    approve_all_actions=approve_all_actions,
                    submolt_meta=submolt_meta,
                )
                if should_stop:
                    return
                if proactive_actions > 0:
                    early_proactive_actions += proactive_actions
                    early_post_action_sent = True
                    save_state(cfg.state_path, state)
                    save_post_engine_memory(cfg.proactive_memory_path, post_memory)
            if should_prioritize_proactive_post(state=state, cfg=cfg) and not early_post_action_sent:
                logger.info(
                    "Cycle=%s priority_post_pending=true defer_reply_scan_and_discovery=1",
                    iteration,
                )
                save_state(cfg.state_path, state)
                save_post_engine_memory(cfg.proactive_memory_path, post_memory)
                sleep_seconds = max(1, cfg.idle_poll_seconds)
                sleep_reason = "priority_post_pending"
                logger.info("Sleeping seconds=%s reason=%s", sleep_seconds, sleep_reason)
                time.sleep(sleep_seconds)
                continue
            if cfg.startup_reply_scan_enabled and cfg.reply_scan_interval_cycles > 0:
                if iteration % cfg.reply_scan_interval_cycles == 0:
                    if early_post_action_sent:
                        logger.info(
                            "Reply scan skipped cycle=%s reason=priority_post_sent",
                            iteration,
                        )
                    else:
                        logger.info(
                            "Reply scan trigger cycle=%s interval=%s",
                            iteration,
                            cfg.reply_scan_interval_cycles,
                        )
                        approve_all_actions = run_startup_reply_scan(
                            client=client,
                            cfg=cfg,
                            logger=logger,
                            state=state,
                            my_name=my_name,
                            persona_text=persona_text,
                            domain_context_text=domain_context_text,
                            approve_all_actions=approve_all_actions,
                        )
            posts, sources = discover_posts(
                client=client,
                cfg=cfg,
                logger=logger,
                keywords=active_keywords,
                iteration=iteration,
                search_state=search_state,
                post_id_fn=post_id,
            )
            logger.info("Poll cycle=%s discovered_posts=%s sources=%s", iteration, len(posts), ",".join(sources))
            if posts:
                approve_all_actions = maybe_follow_ergo_authors(
                    client=client,
                    cfg=cfg,
                    logger=logger,
                    state=state,
                    posts=posts,
                    my_name=my_name,
                    self_identity_keys=self_identity_keys,
                    approve_all_actions=approve_all_actions,
                )
            virality_scores: Dict[str, float] = {}
            if cfg.virality_enabled and posts:
                now_dt = utc_now()
                history_for_virality = {
                    "mode": "comment",
                    "recency_halflife_minutes": cfg.recency_halflife_minutes,
                    "early_comment_window_seconds": cfg.early_comment_window_seconds,
                    "active_keywords": active_keywords,
                    "recent_topic_signatures": state.get("recent_topic_signatures", []),
                }
                for post in posts:
                    pid = post_id(post)
                    if not pid:
                        continue
                    submolt = normalize_submolt(post.get("submolt"), default="general")
                    score = score_post_candidate(
                        post=post,
                        submolt_meta=submolt_meta.get(submolt, {}),
                        now=now_dt,
                        history=history_for_virality,
                    )
                    virality_scores[pid] = score
                    post["__virality_score"] = score
                logger.info(
                    "Virality scoring cycle=%s candidates=%s feed_sources=%s",
                    iteration,
                    len(virality_scores),
                    summarize_sources(posts),
                )
            learning_snapshot_cycle = build_learning_snapshot(post_memory, max_examples=5)
            posts, relevance_score_by_post, high_signal_terms = _rank_posts_for_drafting(
                posts=posts,
                learning_snapshot=learning_snapshot_cycle,
                active_keywords=active_keywords,
                virality_scores=virality_scores,
            )
            effective_shortlist_size, effective_signal_min_score, shortlist_mode = _adaptive_draft_controls(
                cfg=cfg,
                state=state,
            )
            market_snapshot_cycle = learning_snapshot_cycle.get("market_snapshot")
            trending_terms_cycle: List[str] = []
            if isinstance(market_snapshot_cycle, dict):
                raw_terms = market_snapshot_cycle.get("top_terms")
                if isinstance(raw_terms, list):
                    trending_terms_cycle = [
                        _clean_signal_term(item)
                        for item in raw_terms
                        if _clean_signal_term(item)
                    ][:8]
            if posts:
                top_rank_preview: List[str] = []
                for post in posts[:5]:
                    pid = post_id(post) or "(unknown)"
                    score = relevance_score_by_post.get(pid, 0)
                    viral = virality_scores.get(pid)
                    title_preview = normalize_str(post.get("title")).strip()[:48] or "(untitled)"
                    if isinstance(viral, (int, float)):
                        top_rank_preview.append(f"{score}|v={viral:.2f}:{pid}:{title_preview}")
                    else:
                        top_rank_preview.append(f"{score}:{pid}:{title_preview}")
                logger.info(
                    "Poll cycle=%s relevance_ranking terms=%s threshold=%s shortlist=%s mode=%s top=%s",
                    iteration,
                    len(high_signal_terms),
                    effective_signal_min_score,
                    effective_shortlist_size,
                    shortlist_mode,
                    " | ".join(top_rank_preview),
                )

            inspected = 0
            new_candidates = 0
            eligible_now = 0
            drafted_count = 0
            model_approved = 0
            acted = early_proactive_actions
            reply_actions = early_proactive_actions
            post_action_sent = early_post_action_sent
            comment_action_sent = False
            consecutive_declines = 0
            recovery_attempts = 0
            skip_reasons: Dict[str, int] = {}
            provider_counts: Dict[str, int] = {}
            cycle_titles: List[str] = [normalize_str(post.get("title")).strip() for post in posts if normalize_str(post.get("title")).strip()]
            post_cd_remaining, comment_cd_remaining = cooldown_remaining_seconds(state=state, cfg=cfg)
            no_action_slot_notice_shown = False

            if post_cd_remaining > 0 or comment_cd_remaining > 0:
                logger.info(
                    (
                        "Action cooldown status post_remaining=%ss (~%sm) "
                        "comment_remaining=%ss. Drafting only allowed for eligible actions."
                    ),
                    post_cd_remaining,
                    max(1, post_cd_remaining // 60) if post_cd_remaining > 0 else 0,
                    comment_cd_remaining,
                )
            llm_cycle_budget = max(1, cfg.max_drafts_per_cycle)
            if post_cd_remaining > 0 and comment_cd_remaining <= 0:
                llm_cycle_budget = min(llm_cycle_budget, 4)
            if post_cd_remaining > 0 and comment_cd_remaining > 0:
                llm_cycle_budget = min(llm_cycle_budget, 2)
            if shortlist_mode in {"streak_recovery", "decline_recovery"}:
                llm_cycle_budget = min(llm_cycle_budget, 2)
            elif shortlist_mode == "conversion_collapse":
                llm_cycle_budget = 1
            elif shortlist_mode == "relax_context_gate":
                llm_cycle_budget = min(llm_cycle_budget, 3)
            if llm_cycle_budget != max(1, cfg.max_drafts_per_cycle):
                logger.info(
                    "Cycle=%s token_saver llm_cycle_budget=%s base_budget=%s",
                    iteration,
                    llm_cycle_budget,
                    max(1, cfg.max_drafts_per_cycle),
                )
                print_status_banner(
                    title="LLM TOKEN SAVER",
                    rows=[
                        ("cycle", iteration),
                        ("shortlist_mode", shortlist_mode),
                        ("llm_cycle_budget", llm_cycle_budget),
                        ("base_budget", max(1, cfg.max_drafts_per_cycle)),
                        ("cooldown", f"post={post_cd_remaining}s comment={comment_cd_remaining}s"),
                    ],
                    tone="yellow",
                )
            candidate_pool_cap = max(24, effective_shortlist_size * 5, llm_cycle_budget * 8)
            if len(posts) > candidate_pool_cap:
                dropped = len(posts) - candidate_pool_cap
                posts = posts[:candidate_pool_cap]
                logger.info(
                    "Cycle=%s candidate_pool_trimmed kept=%s dropped=%s cap=%s",
                    iteration,
                    len(posts),
                    dropped,
                    candidate_pool_cap,
                )
            decline_guard_limit = MAX_CONSECUTIVE_DECLINES_GUARD
            if shortlist_mode in {"conversion_collapse", "streak_recovery", "cooldown_pressure"}:
                decline_guard_limit = 3

            def mark_seen(pid: Optional[str]) -> None:
                if not pid:
                    return
                seen.add(pid)
                state["seen_post_ids"] = list(seen)[-5000:]

            for post in posts:
                inspected += 1
                if drafted_count >= effective_shortlist_size:
                    skip_reasons["draft_shortlist_cap"] = skip_reasons.get("draft_shortlist_cap", 0) + 1
                    logger.info(
                        "Cycle=%s draft shortlist reached drafted=%s cap=%s stop_additional_drafts=true",
                        iteration,
                        drafted_count,
                        effective_shortlist_size,
                    )
                    break
                title_text = normalize_str(post.get("title")).strip()
                pid = post_id(post)
                if not pid or pid in seen:
                    logger.debug("Cycle=%s skip post_id=%s reason=seen_or_missing", iteration, pid)
                    continue

                new_candidates += 1
                post_title_preview = title_text or "(untitled)"

                if pid in replied_posts:
                    skip_reasons["already_replied_post"] = skip_reasons.get("already_replied_post", 0) + 1
                    mark_seen(pid)
                    logger.debug("Cycle=%s skip post_id=%s reason=already_replied_post", iteration, pid)
                    continue

                if my_name:
                    if has_my_comment_on_post(client=client, post_id_value=pid, my_name=my_name, logger=logger):
                        replied_posts.add(pid)
                        state["replied_post_ids"] = list(replied_posts)[-10000:]
                        skip_reasons["already_replied_post"] = skip_reasons.get("already_replied_post", 0) + 1
                        mark_seen(pid)
                        logger.info(
                            "Cycle=%s skip post_id=%s title=%s reason=already_replied_post_detected",
                            iteration,
                            pid,
                            post_title_preview,
                        )
                        continue

                author_id, author_name = post_author(post)
                if my_name and _normalized_name_key(author_name) == _normalized_name_key(my_name):
                    skip_reasons["self_post"] = skip_reasons.get("self_post", 0) + 1
                    mark_seen(pid)
                    logger.debug(
                        "Cycle=%s skip post_id=%s reason=self_post author=%s my_name=%s",
                        iteration,
                        pid,
                        author_name,
                        my_name,
                    )
                    continue

                allowed, reason = can_reply(state, cfg, author_id, author_name)
                if not allowed:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    is_temporary_rate_block = any(
                        token in reason for token in ("cooldown", "hourly_limit", "daily_limit")
                    )
                    if not is_temporary_rate_block and reason != "author_cooldown":
                        mark_seen(pid)
                    else:
                        # Token saver: if no action slot exists right now, skip drafting and revisit later.
                        skip_reasons["no_action_slots"] = skip_reasons.get("no_action_slots", 0) + 1
                    logger.debug(
                        "Cycle=%s skip post_id=%s author=%s reason=%s",
                        iteration,
                        pid,
                        author_name or author_id or "(unknown)",
                        reason,
                    )
                    continue
                else:
                    eligible_now += 1
                signal_score = relevance_score_by_post.get(pid, 0)
                if signal_score < effective_signal_min_score:
                    skip_reasons["low_signal_relevance"] = skip_reasons.get("low_signal_relevance", 0) + 1
                    mark_seen(pid)
                    logger.debug(
                        "Cycle=%s skip post_id=%s title=%s reason=low_signal_relevance score=%s threshold=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        signal_score,
                        effective_signal_min_score,
                    )
                    continue
                if shortlist_mode in {"conversion_collapse", "cooldown_pressure"}:
                    mechanism_score = _post_mechanism_score(post=post)
                    trend_overlap = _has_trending_overlap(post=post, trending_terms=trending_terms_cycle)
                    if mechanism_score <= 0 and not trend_overlap:
                        skip_reasons["high_signal_filter"] = skip_reasons.get("high_signal_filter", 0) + 1
                        mark_seen(pid)
                        logger.debug(
                            "Cycle=%s skip post_id=%s title=%s reason=high_signal_filter mode=%s",
                            iteration,
                            pid,
                            post_title_preview,
                            shortlist_mode,
                        )
                        continue
                if (
                    post_score(post) < cfg.trending_min_post_score
                    and post_comment_count(post) < cfg.trending_min_comment_count
                ):
                    skip_reasons["not_trending_enough"] = skip_reasons.get("not_trending_enough", 0) + 1
                    mark_seen(pid)
                    logger.debug(
                        "Cycle=%s skip post_id=%s title=%s reason=not_trending_enough score=%s comments=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        post_score(post),
                        post_comment_count(post),
                    )
                    continue
                if shortlist_mode == "tighten_quality":
                    trend_overlap = _has_trending_overlap(post=post, trending_terms=trending_terms_cycle)
                    mechanism_score = _post_mechanism_score(post=post)
                    if not trend_overlap and mechanism_score <= 0:
                        skip_reasons["trend_context_mismatch"] = skip_reasons.get("trend_context_mismatch", 0) + 1
                        mark_seen(pid)
                        logger.debug(
                            "Cycle=%s skip post_id=%s title=%s reason=trend_context_mismatch",
                            iteration,
                            pid,
                            post_title_preview,
                        )
                        continue

                allowed_modes = currently_allowed_response_modes(cfg=cfg, state=state)
                if allowed_modes == ["none"]:
                    skip_reasons["no_action_slots"] = skip_reasons.get("no_action_slots", 0) + 1
                    mark_seen(pid)
                    logger.info("Cycle=%s skip post_id=%s reason=no_action_slots", iteration, pid)
                    if not no_action_slot_notice_shown:
                        print_status_banner(
                            title="ACTION GATE BLOCKED",
                            rows=[
                                ("cycle", iteration),
                                ("reason", "No post/comment slot currently open"),
                                ("post_cooldown_remaining", f"{post_cd_remaining}s"),
                                ("comment_cooldown_remaining", f"{comment_cd_remaining}s"),
                                ("effect", "Skipping LLM drafts until an action slot opens"),
                            ],
                            tone="yellow",
                        )
                        no_action_slot_notice_shown = True
                    continue
                if drafted_count >= llm_cycle_budget:
                    skip_reasons["llm_budget_exhausted"] = skip_reasons.get("llm_budget_exhausted", 0) + 1
                    logger.info(
                        "Cycle=%s llm_budget_exhausted drafted=%s cap=%s",
                        iteration,
                        drafted_count,
                        llm_cycle_budget,
                    )
                    break

                provider_used = "unknown"
                messages: List[Dict[str, str]] = []
                try:
                    logger.debug(
                        "Cycle=%s drafting post_id=%s title=%s provider_hint=%s signal_score=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        cfg.llm_provider,
                        signal_score,
                    )
                    messages = build_openai_messages(
                        persona=persona_text,
                        domain_context=domain_context_text,
                        post=post,
                        pid=pid,
                        allowed_response_modes=allowed_modes,
                        trending_terms=_trending_terms_for_post(
                            post=post,
                            trending_terms=trending_terms_cycle,
                            max_terms=4,
                        ),
                    )
                    comment_allowed_llm = "comment" in allowed_modes
                    post_allowed_llm = "post" in allowed_modes
                    if comment_allowed_llm and post_allowed_llm:
                        draft_action_kind = "decision"
                    elif comment_allowed_llm:
                        draft_action_kind = "comment"
                    elif post_allowed_llm:
                        draft_action_kind = "post"
                    else:
                        draft_action_kind = "none"
                    print_drafting_banner(
                        action_kind=draft_action_kind,
                        pid=pid,
                        title=post_title_preview,
                        provider_hint=cfg.llm_provider,
                    )
                    draft, provider_used, _ = call_generation_model(cfg, messages)
                    provider_counts[provider_used] = provider_counts.get(provider_used, 0) + 1
                    drafted_count += 1
                    logger.debug(
                        "Cycle=%s drafted post_id=%s title=%s provider=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        provider_used,
                    )
                except Exception as e:
                    logger.warning(
                        "Cycle=%s llm_draft_failed post_id=%s title=%s provider_hint=%s error=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        cfg.llm_provider,
                        e,
                    )
                    draft = fallback_draft()
                    drafted_count += 1
                    logger.info("Cycle=%s using_fallback_draft post_id=%s", iteration, pid)

                should_respond = bool(draft.get("should_respond", False))
                confidence, confidence_imputed = _normalized_model_confidence(
                    raw_confidence=draft.get("confidence", 0),
                    should_act=should_respond,
                    has_content=bool(normalize_str(draft.get("content")).strip()),
                    fallback_when_true=max(cfg.min_confidence, 0.72),
                    fallback_when_false=0.0,
                )
                if confidence_imputed and should_respond:
                    logger.info(
                        "Cycle=%s confidence_imputed post_id=%s title=%s raw=%r normalized=%.3f",
                        iteration,
                        pid,
                        post_title_preview,
                        draft.get("confidence"),
                        confidence,
                    )
                can_try_recovery = (
                    recovery_attempts < MAX_RECOVERY_DRAFTS_PER_CYCLE
                    and drafted_count < llm_cycle_budget
                    and signal_score >= (effective_signal_min_score + RECOVERY_SIGNAL_MARGIN)
                    and bool(messages)
                    # Recovery is expensive and rarely worth it for comment-only cycles.
                    and ("post" in allowed_modes)
                )
                should_recover = (not should_respond) or (
                    confidence < cfg.min_confidence and not (confidence_imputed and should_respond)
                )
                if should_recover and can_try_recovery:
                    try:
                        recovery_messages = _build_recovery_messages(messages, signal_score=signal_score)
                        recovered_draft, recovery_provider, _ = call_generation_model(cfg, recovery_messages)
                        provider_counts[recovery_provider] = provider_counts.get(recovery_provider, 0) + 1
                        drafted_count += 1
                        recovery_attempts += 1
                        if isinstance(recovered_draft, dict):
                            draft = recovered_draft
                            should_respond = bool(draft.get("should_respond", False))
                            confidence, confidence_imputed = _normalized_model_confidence(
                                raw_confidence=draft.get("confidence", 0),
                                should_act=should_respond,
                                has_content=bool(normalize_str(draft.get("content")).strip()),
                                fallback_when_true=max(cfg.min_confidence, 0.72),
                                fallback_when_false=0.0,
                            )
                            logger.info(
                                (
                                    "Cycle=%s recovery_draft_attempt post_id=%s title=%s "
                                    "provider=%s should_respond=%s confidence=%.3f attempt=%s/%s"
                                ),
                                iteration,
                                pid,
                                post_title_preview,
                                recovery_provider,
                                should_respond,
                                confidence,
                                recovery_attempts,
                                MAX_RECOVERY_DRAFTS_PER_CYCLE,
                            )
                    except Exception as e:
                        logger.debug("Cycle=%s recovery_draft_failed post_id=%s error=%s", iteration, pid, e)

                if not should_respond:
                    consecutive_declines += 1
                    logger.info(
                        "Cycle=%s model_declined post_id=%s title=%s",
                        iteration,
                        pid,
                        post_title_preview,
                    )
                    mark_seen(pid)
                    if consecutive_declines >= decline_guard_limit:
                        skip_reasons["consecutive_declines_guard"] = skip_reasons.get("consecutive_declines_guard", 0) + 1
                        logger.info(
                            "Cycle=%s stopping drafts early reason=consecutive_declines_guard declines=%s threshold=%s",
                            iteration,
                            consecutive_declines,
                            decline_guard_limit,
                        )
                        break
                    continue

                if confidence < cfg.min_confidence:
                    consecutive_declines += 1
                    logger.info(
                        "Cycle=%s skip post_id=%s reason=low_confidence confidence=%.3f threshold=%.3f",
                        iteration,
                        pid,
                        confidence,
                        cfg.min_confidence,
                    )
                    mark_seen(pid)
                    if consecutive_declines >= decline_guard_limit:
                        skip_reasons["consecutive_declines_guard"] = skip_reasons.get("consecutive_declines_guard", 0) + 1
                        logger.info(
                            "Cycle=%s stopping drafts early reason=consecutive_declines_guard declines=%s threshold=%s",
                            iteration,
                            consecutive_declines,
                            decline_guard_limit,
                        )
                        break
                    continue
                consecutive_declines = 0
                model_approved += 1
                requested_mode = normalize_response_mode(draft.get("response_mode"), default="comment")
                source_blob = " ".join(
                    [
                        normalize_str(post.get("title")).strip(),
                        normalize_str(post.get("content")).strip(),
                        normalize_str(post.get("url")).strip(),
                    ]
                )
                hostile_source = looks_hostile_content(source_blob)

                content = format_content(draft)
                content = _ensure_use_case_prompt_if_relevant(content=content, post=post)
                content = _normalize_ergo_terms(content)
                if hostile_source:
                    requested_mode = "comment"
                    content = build_hostile_refusal_reply()
                    logger.info(
                        "Cycle=%s hostile_source_guard post_id=%s forcing_defensive_comment=1",
                        iteration,
                        pid,
                    )
                # Sanitize early so template/format scaffolding does not cause false skips.
                content = sanitize_publish_content(content)
                if not content:
                    logger.warning("Cycle=%s skip post_id=%s reason=empty_content", iteration, pid)
                    mark_seen(pid)
                    continue
                if _is_template_like_generated_content(content):
                    skip_reasons["template_like_content"] = skip_reasons.get("template_like_content", 0) + 1
                    logger.info(
                        "Cycle=%s skip post_id=%s title=%s reason=template_like_content",
                        iteration,
                        pid,
                        post_title_preview,
                    )
                    mark_seen(pid)
                    continue
                if not _passes_generated_content_quality(content=content, requested_mode=requested_mode):
                    skip_reasons["quality_gate_failed"] = skip_reasons.get("quality_gate_failed", 0) + 1
                    logger.info(
                        "Cycle=%s skip post_id=%s title=%s reason=quality_gate_failed mode=%s",
                        iteration,
                        pid,
                        post_title_preview,
                        requested_mode,
                    )
                    mark_seen(pid)
                    continue

                url = post_url(pid)
                fallback_title = sanitize_generated_title(
                    f"Implementation question on: {post_title_preview}",
                    fallback="Ergo implementation question",
                )
                title = sanitize_generated_title(draft.get("title"), fallback=fallback_title)
                raw_submolt = post.get("submolt")
                submolt = normalize_submolt(raw_submolt)
                comment_content = _prepare_publish_content(
                    content,
                    allow_links=False,
                    audit_label="main_cycle_comment",
                )
                post_content = _prepare_publish_content(
                    _compose_reference_post_content(reference_url=url, content=content),
                    allow_links=True,
                    audit_label="main_cycle_post",
                )
                if hostile_source:
                    post_content = ""
                draft_archetype = normalize_str(draft.get("content_archetype")).strip().lower()
                logger.debug(
                    "Cycle=%s normalized_submolt post_id=%s raw_type=%s value=%s",
                    iteration,
                    pid,
                    type(raw_submolt).__name__,
                    submolt,
                )

                actions = planned_actions(requested_mode=requested_mode, cfg=cfg, state=state)
                if not actions and bool(post.get("__fast_lane_comment")):
                    comment_allowed_now, _ = comment_gate_status(state=state, cfg=cfg)
                    if comment_allowed_now:
                        actions = ["comment"]
                        logger.info(
                            "Cycle=%s fast_lane override post_id=%s title=%s action=comment",
                            iteration,
                            pid,
                            post_title_preview,
                        )
                if not actions:
                    comment_allowed_now, comment_gate_reason = comment_gate_status(state=state, cfg=cfg)
                    should_wait_comment = (
                        requested_mode in {"comment", "both"}
                        and not comment_allowed_now
                        and comment_gate_reason == "comment_cooldown"
                    )
                    if should_wait_comment:
                        approved, approve_all_actions, should_stop = confirm_action(
                            cfg=cfg,
                            logger=logger,
                            action="wait-comment",
                            pid=pid,
                            title=title,
                            submolt=submolt,
                            url=url,
                            author=author_name or author_id or "(unknown)",
                            content_preview=preview_text(comment_content),
                            approve_all=approve_all_actions,
                        )
                        if should_stop:
                            return
                        if approved:
                            if wait_for_comment_slot(state=state, cfg=cfg, logger=logger):
                                comment_sig = _publish_signature(
                                    action_type="comment",
                                    target_post_id=pid,
                                    parent_comment_id=None,
                                    content=comment_content,
                                )
                                if _seen_publish_signature(state, comment_sig):
                                    logger.info(
                                        "Cycle=%s skip post_id=%s reason=duplicate_comment_signature_after_wait",
                                        iteration,
                                        pid,
                                    )
                                    continue
                                try:
                                    logger.info(
                                        "Cycle=%s action=comment attempt post_id=%s submolt=%s url=%s waited_for_cooldown=true",
                                        iteration,
                                        pid,
                                        submolt,
                                        url,
                                    )
                                    comment_resp = client.create_comment(pid, comment_content)
                                    register_my_comment_id(state=state, response_payload=comment_resp)
                                    _remember_publish_signature(state, comment_sig)
                                    state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                                    replied_posts.add(pid)
                                    state["replied_post_ids"] = list(replied_posts)[-10000:]
                                    maybe_upvote_post_after_comment(
                                        client=client,
                                        state=state,
                                        logger=logger,
                                        post_id_value=pid,
                                        journal_path=cfg.action_journal_path,
                                        submolt=submolt,
                                        post_title=title,
                                        url=url,
                                        reference={
                                            "post_id": pid,
                                            "post_title": title,
                                        },
                                        meta={"source": "main_cycle"},
                                    )
                                    now_ts = utc_now().timestamp()
                                    state["last_action_ts"] = now_ts
                                    state["last_comment_action_ts"] = now_ts
                                    acted += 1
                                    reply_actions += 1
                                    comment_action_sent = True
                                    remember_topic_signature(title=title, content=comment_content)
                                    journal_written_action(
                                        action_type="comment",
                                        target_post_id=pid,
                                        submolt=submolt,
                                        title=title,
                                        content=comment_content,
                                        reference_post_id=pid,
                                        url=url,
                                        reference={
                                            "post_id": pid,
                                            "post_title": title_text or post_title_preview,
                                            "post_content": normalize_str(post.get("content")),
                                            "post_author": author_name or author_id,
                                        },
                                        meta={"source": "main_cycle", "kind": "comment_after_wait"},
                                    )
                                    analytics_event(
                                        action_type="comment",
                                        target_post_id=pid,
                                        submolt=submolt,
                                        feed_sources=(post.get("__feed_sources") if isinstance(post.get("__feed_sources"), list) else []),
                                        virality_score=virality_scores.get(pid),
                                        archetype=normalize_str(draft.get("content_archetype")).strip().lower(),
                                        model_confidence=confidence,
                                        approved_by_human=True,
                                        executed=True,
                                        title=title,
                                    )
                                    logger.info(
                                        "Cycle=%s action=comment success post_id=%s daily_comment_count=%s",
                                        iteration,
                                        pid,
                                        state["daily_comment_count"],
                                    )
                                    print_success_banner(action="comment", pid=pid, url=url, title=title)
                                except Exception as e:
                                    analytics_event(
                                        action_type="comment",
                                        target_post_id=pid,
                                        submolt=submolt,
                                        feed_sources=(post.get("__feed_sources") if isinstance(post.get("__feed_sources"), list) else []),
                                        virality_score=virality_scores.get(pid),
                                        archetype=normalize_str(draft.get("content_archetype")).strip().lower(),
                                        model_confidence=confidence,
                                        approved_by_human=True,
                                        executed=False,
                                        error=normalize_str(e),
                                        title=title,
                                    )
                                    logger.warning(
                                        "Cycle=%s waited comment failed post_id=%s error=%s",
                                        iteration,
                                        pid,
                                        e,
                                    )
                        else:
                            analytics_event(
                                action_type="comment",
                                target_post_id=pid,
                                submolt=submolt,
                                feed_sources=(post.get("__feed_sources") if isinstance(post.get("__feed_sources"), list) else []),
                                virality_score=virality_scores.get(pid),
                                archetype=normalize_str(draft.get("content_archetype")).strip().lower(),
                                model_confidence=confidence,
                                approved_by_human=False,
                                executed=False,
                                error="wait_comment_not_approved",
                                title=title,
                            )
                            mark_seen(pid)
                            continue
                    logger.info(
                        "Cycle=%s skip post_id=%s reason=no_available_actions requested_mode=%s",
                        iteration,
                        pid,
                        requested_mode,
                    )
                    continue
                if "comment" in actions and not comment_content.strip():
                    actions = [item for item in actions if item != "comment"]
                    skip_reasons["empty_comment_content_after_policy"] = (
                        skip_reasons.get("empty_comment_content_after_policy", 0) + 1
                    )
                    logger.info(
                        "Cycle=%s removed_comment_action post_id=%s reason=empty_comment_content_after_policy",
                        iteration,
                        pid,
                    )
                    if not actions:
                        continue
                if "post" in actions and not post_content.strip():
                    actions = [item for item in actions if item != "post"]
                    skip_reasons["empty_post_content_after_policy"] = (
                        skip_reasons.get("empty_post_content_after_policy", 0) + 1
                    )
                    logger.info(
                        "Cycle=%s removed_post_action post_id=%s reason=empty_post_content_after_policy",
                        iteration,
                        pid,
                    )
                    if not actions:
                        continue
                post_submolt_route = submolt
                if "post" in actions and submolt_meta:
                    routed_submolt = choose_best_submolt_for_new_post(
                        title=title,
                        content=post_content,
                        archetype=draft_archetype,
                        target_submolts=[submolt] + list(cfg.target_submolts),
                        submolt_meta=submolt_meta,
                    )
                    if normalize_submolt(routed_submolt) != normalize_submolt(submolt):
                        logger.info(
                            "Cycle=%s route adjusted reference_post_id=%s from m/%s to m/%s",
                            iteration,
                            pid,
                            submolt,
                            routed_submolt,
                        )
                    post_submolt_route = normalize_submolt(routed_submolt, default=submolt)
                    if not is_valid_submolt_name(post_submolt_route, submolt_meta):
                        logger.warning(
                            "Cycle=%s invalid routed submolt=%s fallback=general reference_post_id=%s",
                            iteration,
                            post_submolt_route,
                            pid,
                        )
                        post_submolt_route = "general"
                if "post" in actions:
                    reference_text = " ".join(
                        [
                            normalize_str(post.get("title")).strip(),
                            normalize_str(post.get("content")).strip(),
                            normalize_str(post.get("submolt")).strip(),
                        ]
                    )
                    optimized_title, optimized_post_content, hook_meta = select_best_hook_pair(
                        title=title,
                        content=post_content,
                        reference_text=reference_text,
                        archetype=draft_archetype or "implementation_walkthrough",
                    )
                    if optimized_title != title or optimized_post_content != post_content:
                        logger.info(
                            "Cycle=%s hook optimization post_id=%s title_changed=%s lead_changed=%s",
                            iteration,
                            pid,
                            int(optimized_title != title),
                            int(optimized_post_content != post_content),
                        )
                        logger.debug(
                            "Cycle=%s hook_meta post_id=%s title_candidates=%s lead_candidates=%s",
                            iteration,
                            pid,
                            hook_meta.get("title_candidates"),
                            hook_meta.get("lead_candidates"),
                        )
                        title = optimized_title
                        post_content = _prepare_publish_content(
                            optimized_post_content,
                            allow_links=True,
                            audit_label="main_cycle_post_optimized",
                        )
                logger.info(
                    "Cycle=%s planned_actions post_id=%s requested_mode=%s effective_actions=%s post_route=%s",
                    iteration,
                    pid,
                    requested_mode,
                    ",".join(actions),
                    post_submolt_route,
                )

                if cfg.dry_run:
                    logger.info(
                        "Cycle=%s dry_run actions=%s post_id=%s submolt=%s url=%s title=%s",
                        iteration,
                        ",".join(actions),
                        pid,
                        submolt,
                        url,
                        title,
                    )
                    mark_seen(pid)
                    continue

                reply_executed = False
                feed_sources_for_event = post.get("__feed_sources")
                if not isinstance(feed_sources_for_event, list):
                    feed_sources_for_event = []
                virality_for_event = virality_scores.get(pid)
                for action in actions:
                    reference_post_title = title_text or post_title_preview or f"Post {pid}"
                    action_submolt = post_submolt_route if action == "post" else submolt
                    draft_preview = comment_content if action == "comment" else post_content
                    confirm_pid = pid
                    confirm_url = url
                    action_title = title if action == "post" else reference_post_title
                    if action == "post":
                        confirm_pid = "(new)"
                        confirm_url = f"https://moltbook.com/m/{action_submolt}"
                        draft_preview = f"Reference post: {url}\n\n{post_content}"
                    approved, approve_all_actions, should_stop = confirm_action(
                        cfg=cfg,
                        logger=logger,
                        action=action,
                        pid=confirm_pid,
                        title=action_title,
                        submolt=action_submolt,
                        url=confirm_url,
                        author=author_name or author_id or "(unknown)",
                        content_preview=preview_text(draft_preview),
                        approve_all=approve_all_actions,
                    )
                    if should_stop:
                        return
                    if not approved:
                        logger.info("Cycle=%s action=%s skipped post_id=%s reason=not_approved", iteration, action, pid)
                        analytics_event(
                            action_type=action,
                            target_post_id=pid if action != "post" else "(new)",
                            submolt=action_submolt,
                            feed_sources=feed_sources_for_event,
                            virality_score=virality_for_event,
                            archetype=draft_archetype,
                            model_confidence=confidence,
                            approved_by_human=False,
                            executed=False,
                            error="not_approved",
                            title=title,
                        )
                        mark_seen(pid)
                        continue

                    if action == "comment":
                        comment_sig = _publish_signature(
                            action_type="comment",
                            target_post_id=pid,
                            parent_comment_id=None,
                            content=comment_content,
                        )
                        if _seen_publish_signature(state, comment_sig):
                            logger.info(
                                "Cycle=%s action=comment skipped post_id=%s reason=duplicate_publish_signature",
                                iteration,
                                pid,
                            )
                            analytics_event(
                                action_type="comment",
                                target_post_id=pid,
                                submolt=action_submolt,
                                feed_sources=feed_sources_for_event,
                                virality_score=virality_for_event,
                                archetype=draft_archetype,
                                model_confidence=confidence,
                                approved_by_human=True,
                                executed=False,
                                error="duplicate_publish_signature",
                                title=reference_post_title,
                            )
                            continue
                        try:
                            logger.info(
                                "Cycle=%s action=comment attempt post_id=%s submolt=%s url=%s",
                                iteration,
                                pid,
                                action_submolt,
                                url,
                            )
                            comment_resp = client.create_comment(pid, comment_content)
                            register_my_comment_id(state=state, response_payload=comment_resp)
                            _remember_publish_signature(state, comment_sig)
                            state["daily_comment_count"] = state.get("daily_comment_count", 0) + 1
                            replied_posts.add(pid)
                            state["replied_post_ids"] = list(replied_posts)[-10000:]
                            maybe_upvote_post_after_comment(
                                client=client,
                                state=state,
                                logger=logger,
                                post_id_value=pid,
                                journal_path=cfg.action_journal_path,
                                submolt=action_submolt,
                                post_title=reference_post_title,
                                url=url,
                                reference={
                                    "post_id": pid,
                                    "post_title": reference_post_title,
                                },
                                meta={"source": "main_cycle"},
                            )
                            now_ts = utc_now().timestamp()
                            state["last_action_ts"] = now_ts
                            state["last_comment_action_ts"] = now_ts
                            acted += 1
                            reply_actions += 1
                            comment_action_sent = True
                            reply_executed = True
                            remember_topic_signature(title=title, content=comment_content)
                            journal_written_action(
                                action_type="comment",
                                target_post_id=pid,
                                submolt=action_submolt,
                                title=reference_post_title,
                                content=comment_content,
                                reference_post_id=pid,
                                url=url,
                                reference={
                                    "post_id": pid,
                                    "post_title": title_text or post_title_preview,
                                    "post_content": normalize_str(post.get("content")),
                                    "post_author": author_name or author_id,
                                },
                                meta={"source": "main_cycle", "kind": "comment"},
                            )
                            logger.info(
                                "Cycle=%s action=comment success post_id=%s daily_comment_count=%s",
                                iteration,
                                pid,
                                state["daily_comment_count"],
                            )
                            analytics_event(
                                action_type="comment",
                                target_post_id=pid,
                                submolt=action_submolt,
                                feed_sources=feed_sources_for_event,
                                virality_score=virality_for_event,
                                archetype=draft_archetype,
                                model_confidence=confidence,
                                approved_by_human=True,
                                executed=True,
                                title=reference_post_title,
                            )
                            print_success_banner(
                                action="comment",
                                pid=pid,
                                url=url,
                                title=reference_post_title,
                            )
                        except Exception as e:
                            logger.warning(
                                "Cycle=%s action=comment failed post_id=%s error=%s fallback=post",
                                iteration,
                                pid,
                                e,
                            )
                            analytics_event(
                                action_type="comment",
                                target_post_id=pid,
                                submolt=action_submolt,
                                feed_sources=feed_sources_for_event,
                                virality_score=virality_for_event,
                                archetype=draft_archetype,
                                model_confidence=confidence,
                                approved_by_human=True,
                                executed=False,
                                error=normalize_str(e),
                                title=title,
                            )
                            if "post" in actions:
                                continue
                            approved, approve_all_actions, should_stop = confirm_action(
                                cfg=cfg,
                                logger=logger,
                                action="post-fallback",
                                pid="(new)",
                                title=title,
                                submolt=post_submolt_route,
                                url=f"https://moltbook.com/m/{post_submolt_route}",
                                author=author_name or author_id or "(unknown)",
                                content_preview=preview_text(f"Reference post: {url}\n\n{post_content}"),
                                approve_all=approve_all_actions,
                            )
                            if should_stop:
                                return
                            if not approved:
                                logger.info(
                                    "Cycle=%s action=post-fallback skipped post_id=%s reason=not_approved",
                                    iteration,
                                    pid,
                                )
                                analytics_event(
                                    action_type="post",
                                    target_post_id="(new)",
                                    submolt=post_submolt_route,
                                    feed_sources=feed_sources_for_event,
                                    virality_score=virality_for_event,
                                    archetype=draft_archetype,
                                    model_confidence=confidence,
                                    approved_by_human=False,
                                    executed=False,
                                    error="post_fallback_not_approved",
                                    title=title,
                                )
                                continue
                            logger.info(
                                "Cycle=%s action=post attempt reference_post_id=%s submolt=%s reference_url=%s title=%s",
                                iteration,
                                pid,
                                post_submolt_route,
                                url,
                                title,
                            )
                            post_sig = _publish_signature(
                                action_type="post",
                                target_post_id="(new)",
                                parent_comment_id=None,
                                content=post_content,
                            )
                            if _seen_publish_signature(state, post_sig):
                                logger.info(
                                    "Cycle=%s action=post skipped reference_post_id=%s reason=duplicate_publish_signature",
                                    iteration,
                                    pid,
                                )
                                analytics_event(
                                    action_type="post",
                                    target_post_id="(new)",
                                    submolt=post_submolt_route,
                                    feed_sources=feed_sources_for_event,
                                    virality_score=virality_for_event,
                                    archetype=draft_archetype,
                                    model_confidence=confidence,
                                    approved_by_human=True,
                                    executed=False,
                                    error="duplicate_publish_signature",
                                    title=title,
                                )
                                continue
                            try:
                                post_resp = client.create_post(submolt=post_submolt_route, title=title, content=post_content)
                                _remember_publish_signature(state, post_sig)
                            except Exception as e:
                                analytics_event(
                                    action_type="post",
                                    target_post_id="(new)",
                                    submolt=post_submolt_route,
                                    feed_sources=feed_sources_for_event,
                                    virality_score=virality_for_event,
                                    archetype=draft_archetype,
                                    model_confidence=confidence,
                                    approved_by_human=True,
                                    executed=False,
                                    error=normalize_str(e),
                                    title=title,
                                )
                                logger.warning(
                                    "Cycle=%s action=post failed reference_post_id=%s error=%s",
                                    iteration,
                                    pid,
                                    e,
                                )
                                continue
                            created_post_id = post_id(post_resp if isinstance(post_resp, dict) else {}) or "(unknown)"
                            created_url = post_url(created_post_id if created_post_id != "(unknown)" else None)
                            state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                            replied_posts.add(pid)
                            state["replied_post_ids"] = list(replied_posts)[-10000:]
                            now_ts = utc_now().timestamp()
                            state["last_action_ts"] = now_ts
                            state["last_post_action_ts"] = now_ts
                            acted += 1
                            reply_actions += 1
                            post_action_sent = True
                            reply_executed = True
                            remember_topic_signature(title=title, content=post_content)
                            journal_written_action(
                                action_type="post",
                                target_post_id=str(created_post_id),
                                submolt=post_submolt_route,
                                title=title,
                                content=post_content,
                                reference_post_id=pid,
                                url=created_url,
                                reference={
                                    "post_id": pid,
                                    "post_title": title_text or post_title_preview,
                                    "post_content": normalize_str(post.get("content")),
                                    "post_author": author_name or author_id,
                                },
                                meta={"source": "main_cycle", "kind": "post_fallback"},
                            )
                            logger.info(
                                (
                                    "Cycle=%s action=post success reference_post_id=%s "
                                    "new_post_id=%s new_url=%s daily_post_count=%s"
                                ),
                                iteration,
                                pid,
                                created_post_id,
                                created_url,
                                state["daily_post_count"],
                            )
                            analytics_event(
                                action_type="post",
                                target_post_id=str(created_post_id),
                                submolt=post_submolt_route,
                                feed_sources=feed_sources_for_event,
                                virality_score=virality_for_event,
                                archetype=draft_archetype,
                                model_confidence=confidence,
                                approved_by_human=True,
                                executed=True,
                                title=title,
                            )
                            print_success_banner(action="post", pid=created_post_id, url=created_url, title=title)
                    elif action == "post":
                        logger.info(
                            "Cycle=%s action=post attempt reference_post_id=%s submolt=%s reference_url=%s title=%s",
                            iteration,
                            pid,
                            action_submolt,
                            url,
                            title,
                        )
                        post_sig = _publish_signature(
                            action_type="post",
                            target_post_id="(new)",
                            parent_comment_id=None,
                            content=post_content,
                        )
                        if _seen_publish_signature(state, post_sig):
                            logger.info(
                                "Cycle=%s action=post skipped reference_post_id=%s reason=duplicate_publish_signature",
                                iteration,
                                pid,
                            )
                            analytics_event(
                                action_type="post",
                                target_post_id="(new)",
                                submolt=action_submolt,
                                feed_sources=feed_sources_for_event,
                                virality_score=virality_for_event,
                                archetype=draft_archetype,
                                model_confidence=confidence,
                                approved_by_human=True,
                                executed=False,
                                error="duplicate_publish_signature",
                                title=title,
                            )
                            continue
                        try:
                            post_resp = client.create_post(submolt=action_submolt, title=title, content=post_content)
                            _remember_publish_signature(state, post_sig)
                        except Exception as e:
                            analytics_event(
                                action_type="post",
                                target_post_id="(new)",
                                submolt=action_submolt,
                                feed_sources=feed_sources_for_event,
                                virality_score=virality_for_event,
                                archetype=draft_archetype,
                                model_confidence=confidence,
                                approved_by_human=True,
                                executed=False,
                                error=normalize_str(e),
                                title=title,
                            )
                            logger.warning(
                                "Cycle=%s action=post failed reference_post_id=%s error=%s",
                                iteration,
                                pid,
                                e,
                            )
                            continue
                        created_post_id = post_id(post_resp if isinstance(post_resp, dict) else {}) or "(unknown)"
                        created_url = post_url(created_post_id if created_post_id != "(unknown)" else None)
                        state["daily_post_count"] = state.get("daily_post_count", 0) + 1
                        replied_posts.add(pid)
                        state["replied_post_ids"] = list(replied_posts)[-10000:]
                        now_ts = utc_now().timestamp()
                        state["last_action_ts"] = now_ts
                        state["last_post_action_ts"] = now_ts
                        acted += 1
                        reply_actions += 1
                        post_action_sent = True
                        reply_executed = True
                        remember_topic_signature(title=title, content=post_content)
                        journal_written_action(
                            action_type="post",
                            target_post_id=str(created_post_id),
                            submolt=action_submolt,
                            title=title,
                            content=post_content,
                            reference_post_id=pid,
                            url=created_url,
                            reference={
                                "post_id": pid,
                                "post_title": title_text or post_title_preview,
                                "post_content": normalize_str(post.get("content")),
                                "post_author": author_name or author_id,
                            },
                            meta={"source": "main_cycle", "kind": "post"},
                        )
                        logger.info(
                            (
                                "Cycle=%s action=post success reference_post_id=%s "
                                "new_post_id=%s new_url=%s daily_post_count=%s"
                            ),
                            iteration,
                            pid,
                            created_post_id,
                            created_url,
                            state["daily_post_count"],
                        )
                        analytics_event(
                            action_type="post",
                            target_post_id=str(created_post_id),
                            submolt=action_submolt,
                            feed_sources=feed_sources_for_event,
                            virality_score=virality_for_event,
                            archetype=draft_archetype,
                            model_confidence=confidence,
                            approved_by_human=True,
                            executed=True,
                            title=title,
                        )
                        print_success_banner(action="post", pid=created_post_id, url=created_url, title=title)

                vote_action = normalize_vote_action(draft.get("vote_action"))
                vote_target = normalize_vote_target(draft.get("vote_target"))
                if vote_action == "downvote" and not cfg.allow_comment_downvote:
                    if vote_target == "top_comment":
                        logger.info(
                            "Cycle=%s adjusting vote_target from top_comment to none for unsupported downvote-comment",
                            iteration,
                        )
                        vote_target = "none"
                    elif vote_target == "both":
                        logger.info(
                            "Cycle=%s adjusting vote_target from both to post for unsupported downvote-comment",
                            iteration,
                        )
                        vote_target = "post"
                if vote_action != "none" and vote_target != "none":
                    if vote_target in {"post", "both"}:
                        approved, approve_all_actions, should_stop = confirm_action(
                            cfg=cfg,
                            logger=logger,
                            action=f"{vote_action}-post",
                            pid=pid,
                            title=title,
                            submolt=submolt,
                            url=url,
                            author=author_name or author_id or "(unknown)",
                            content_preview=preview_text(
                                f"Vote target: post {pid}\nVote action: {vote_action}"
                            ),
                            approve_all=approve_all_actions,
                        )
                        if should_stop:
                            return
                        if approved:
                            try:
                                client.vote_post(pid, vote_action=vote_action)
                                acted += 1
                                logger.info(
                                    "Cycle=%s action=%s success post_id=%s",
                                    iteration,
                                    f"{vote_action}-post",
                                    pid,
                                )
                                try:
                                    append_action_journal(
                                        cfg.action_journal_path,
                                        action_type=f"{vote_action}-post",
                                        target_post_id=pid,
                                        submolt=submolt,
                                        title=title,
                                        content="",
                                        reference_post_id=pid,
                                        url=url,
                                        reference={
                                            "post_id": pid,
                                            "post_title": title_text or post_title_preview,
                                            "post_author": author_name or author_id,
                                            "vote_action": vote_action,
                                        },
                                        meta={"source": "main_cycle", "kind": "vote_post"},
                                    )
                                except Exception as e:
                                    logger.debug("Vote post journal write failed post_id=%s error=%s", pid, e)
                                analytics_event(
                                    action_type=f"{vote_action}-post",
                                    target_post_id=pid,
                                    submolt=submolt,
                                    feed_sources=feed_sources_for_event,
                                    virality_score=virality_for_event,
                                    archetype=draft_archetype,
                                    model_confidence=confidence,
                                    approved_by_human=True,
                                    executed=True,
                                    title=title,
                                )
                                print_success_banner(
                                    action=f"{vote_action}-post",
                                    pid=pid,
                                    url=url,
                                    title=title,
                                )
                            except Exception as e:
                                logger.warning(
                                    "Cycle=%s action=%s failed post_id=%s error=%s",
                                    iteration,
                                    f"{vote_action}-post",
                                    pid,
                                    e,
                                )

                    if vote_target in {"top_comment", "both"}:
                        top_comment = choose_top_comment(client=client, pid=pid, my_name=my_name, logger=logger)
                        if top_comment:
                            top_comment_id = comment_id(top_comment)
                            _, top_comment_author = comment_author(top_comment)
                            comment_body = normalize_str(top_comment.get("content"))
                            approved, approve_all_actions, should_stop = confirm_action(
                                cfg=cfg,
                                logger=logger,
                                action=f"{vote_action}-comment",
                                pid=top_comment_id or pid,
                                title=f"Comment by {top_comment_author or '(unknown)'}",
                                submolt=submolt,
                                url=url,
                                author=top_comment_author or "(unknown)",
                                content_preview=preview_text(comment_body),
                                approve_all=approve_all_actions,
                            )
                            if should_stop:
                                return
                            if approved and top_comment_id:
                                try:
                                    client.vote_comment(top_comment_id, vote_action=vote_action)
                                    acted += 1
                                    logger.info(
                                        "Cycle=%s action=%s success comment_id=%s",
                                        iteration,
                                        f"{vote_action}-comment",
                                        top_comment_id,
                                    )
                                    try:
                                        append_action_journal(
                                            cfg.action_journal_path,
                                            action_type=f"{vote_action}-comment",
                                            target_post_id=top_comment_id,
                                            submolt=submolt,
                                            title=f"Comment by {top_comment_author or '(unknown)'}",
                                            content="",
                                            reference_post_id=pid,
                                            url=url,
                                            reference={
                                                "post_id": pid,
                                                "post_title": title_text or post_title_preview,
                                                "comment_id": top_comment_id,
                                                "comment_author": top_comment_author,
                                                "comment_content": comment_body,
                                                "vote_action": vote_action,
                                            },
                                            meta={"source": "main_cycle", "kind": "vote_comment"},
                                        )
                                    except Exception as e:
                                        logger.debug(
                                            "Vote comment journal write failed comment_id=%s error=%s",
                                            top_comment_id,
                                            e,
                                        )
                                    analytics_event(
                                        action_type=f"{vote_action}-comment",
                                        target_post_id=pid,
                                        submolt=submolt,
                                        feed_sources=feed_sources_for_event,
                                        virality_score=virality_for_event,
                                        archetype=draft_archetype,
                                        model_confidence=confidence,
                                        approved_by_human=True,
                                        executed=True,
                                        title=title,
                                    )
                                    print_success_banner(
                                        action=f"{vote_action}-comment",
                                        pid=top_comment_id,
                                        url=url,
                                        title=f"Comment by {top_comment_author or '(unknown)'}",
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Cycle=%s action=%s failed comment_id=%s error=%s",
                                        iteration,
                                        f"{vote_action}-comment",
                                        top_comment_id,
                                        e,
                                    )
                        else:
                            logger.info(
                                "Cycle=%s vote skipped post_id=%s reason=no_comment_target vote_target=%s",
                                iteration,
                                pid,
                                vote_target,
                            )

                if reply_executed and author_id:
                    state.setdefault("per_author_last_reply", {})[author_id] = state.get("last_action_ts")
                if reply_executed:
                    mark_seen(pid)

            today_iso = time.strftime("%Y-%m-%d", time.gmtime())
            proactive_today = _proactive_posts_count_for_date(post_memory=post_memory, date_iso=today_iso)
            daily_proactive_target = max(1, int(cfg.proactive_daily_target_posts))
            daily_goal_missing = proactive_today < daily_proactive_target
            should_try_proactive = (acted == 0 and not post_action_sent) or (daily_goal_missing and not post_action_sent)
            if should_try_proactive:
                logger.info(
                    "Proactive trigger cycle=%s acted=%s post_action_sent=%s proactive_today=%s target=%s",
                    iteration,
                    acted,
                    post_action_sent,
                    proactive_today,
                    daily_proactive_target,
                )
                proactive_actions, should_stop, approve_all_actions = maybe_run_proactive_post(
                    client=client,
                    cfg=cfg,
                    logger=logger,
                    state=state,
                    post_memory=post_memory,
                    my_name=my_name,
                    persona_text=persona_text,
                    domain_context_text=domain_context_text,
                    approve_all_actions=approve_all_actions,
                    submolt_meta=submolt_meta,
                )
                if should_stop:
                    return
                if proactive_actions > 0:
                    acted += proactive_actions
                    reply_actions += proactive_actions
                    post_action_sent = True

            state["seen_post_ids"] = list(seen)[-5000:]
            state["replied_post_ids"] = list(replied_posts)[-10000:]
            save_state(cfg.state_path, state)
            save_post_engine_memory(cfg.proactive_memory_path, post_memory)
            logger.info(
                (
                    "Poll cycle=%s complete inspected=%s new_candidates=%s eligible_now=%s "
                    "drafted=%s model_approved=%s actions=%s seen_total=%s"
                ),
                iteration,
                inspected,
                new_candidates,
                eligible_now,
                drafted_count,
                model_approved,
                acted,
                len(seen),
            )
            if provider_counts:
                provider_summary = ", ".join([f"{k}={v}" for k, v in sorted(provider_counts.items())])
                logger.info("Poll cycle=%s provider_summary %s", iteration, provider_summary)
            if skip_reasons:
                summary = ", ".join([f"{k}={v}" for k, v in sorted(skip_reasons.items())])
                logger.info("Poll cycle=%s skip_summary %s", iteration, summary)
                if int(skip_reasons.get("no_action_slots", 0) or 0) > 0:
                    print_status_banner(
                        title="CYCLE SKIP SUMMARY",
                        rows=[
                            ("cycle", iteration),
                            ("no_action_slots", int(skip_reasons.get("no_action_slots", 0) or 0)),
                            ("drafted", drafted_count),
                            ("model_approved", model_approved),
                            ("actions_executed", acted),
                        ],
                        tone="yellow",
                    )

            if cfg.keyword_learning_enabled:
                interval = max(1, cfg.keyword_learning_interval_cycles)
                next_learning_cycle = ((iteration // interval) + 1) * interval
                if iteration % interval == 0:
                    next_learning_cycle = iteration
                logger.info(
                    (
                        "Keyword learning status enabled=true cycle=%s next_cycle=%s "
                        "titles_seen=%s min_titles=%s"
                    ),
                    iteration,
                    next_learning_cycle,
                    len(cycle_titles),
                    cfg.keyword_learning_min_titles,
                )

            if (
                cfg.keyword_learning_enabled
                and cfg.keyword_learning_interval_cycles > 0
                and iteration % cfg.keyword_learning_interval_cycles == 0
                and len(cycle_titles) >= cfg.keyword_learning_min_titles
            ):
                suggestions: List[str] = []
                if has_generation_provider(cfg):
                    try:
                        suggestions = propose_keywords_from_titles(
                            cfg=cfg,
                            titles=cycle_titles,
                            existing_keywords=active_keywords,
                            max_suggestions=cfg.keyword_learning_max_suggestions,
                        )
                    except Exception as e:
                        suggestions = []
                        logger.warning("Keyword learning failed cycle=%s error=%s", iteration, e)
                market_snapshot_for_keywords = learning_snapshot_cycle.get("market_snapshot")
                market_suggestions = _market_keyword_candidates(
                    market_snapshot=market_snapshot_for_keywords if isinstance(market_snapshot_for_keywords, dict) else {},
                    existing_keywords=merge_keywords(active_keywords, suggestions),
                    max_suggestions=cfg.keyword_learning_max_suggestions,
                )
                if market_suggestions:
                    suggestions = merge_keywords(suggestions, market_suggestions)[: cfg.keyword_learning_max_suggestions]
                    logger.info(
                        "Keyword learning market_suggestions cycle=%s suggested=%s",
                        iteration,
                        ", ".join(market_suggestions),
                    )

                if suggestions:
                    logger.info(
                        "Keyword learning cycle=%s suggested=%s",
                        iteration,
                        ", ".join(suggestions),
                    )
                    if manual_keyword_review:
                        for keyword in suggestions:
                            approved, approve_all_keyword_changes, should_stop = confirm_keyword_addition(
                                logger=logger,
                                keyword=keyword,
                                approve_all=approve_all_keyword_changes,
                            )
                            if should_stop:
                                return
                            if not approved:
                                continue
                            learned_before = keyword_store.get("learned_keywords", [])
                            learned_after = merge_keywords(learned_before, [keyword])
                            if len(learned_after) == len(learned_before):
                                continue
                            keyword_store["learned_keywords"] = learned_after
                            save_keyword_store(cfg.keyword_store_path, keyword_store)
                            active_keywords = merge_keywords(cfg.keywords, learned_after)
                            logger.info(
                                "Keyword added keyword=%s learned_total=%s active_total=%s",
                                keyword,
                                len(learned_after),
                                len(active_keywords),
                            )
                            print_keyword_added_banner(keyword=keyword, learned_total=len(learned_after))
                    else:
                        pending_before = keyword_store.get("pending_suggestions", [])
                        combined_pending = merge_keywords(pending_before, suggestions)
                        # Do not keep duplicates that are already learned.
                        learned_set = set(keyword_store.get("learned_keywords", []))
                        filtered_pending = [k for k in combined_pending if k not in learned_set]
                        added = max(0, len(filtered_pending) - len(pending_before))
                        keyword_store["pending_suggestions"] = filtered_pending
                        save_keyword_store(cfg.keyword_store_path, keyword_store)
                        logger.info(
                            "Deferred keyword suggestions for next manual run added=%s pending_total=%s",
                            added,
                            len(filtered_pending),
                        )

            cycle_stats = {
                "inspected": inspected,
                "new_candidates": new_candidates,
                "eligible_now": eligible_now,
                "drafted": drafted_count,
                "model_approved": model_approved,
                "actions": acted,
                "post_action_sent": post_action_sent,
                "comment_action_sent": comment_action_sent,
                "skip_reasons": skip_reasons,
                "max_suggestions": cfg.self_improve_max_suggestions,
                "draft_shortlist_size": cfg.draft_shortlist_size,
                "draft_signal_min_score": cfg.draft_signal_min_score,
                "effective_draft_shortlist_size": effective_shortlist_size,
                "effective_draft_signal_min_score": effective_signal_min_score,
                "draft_shortlist_mode": shortlist_mode,
                "high_signal_term_count": len(high_signal_terms),
                "trending_terms": trending_terms_cycle,
            }
            visibility_metrics_cycle = learning_snapshot_cycle.get("visibility_metrics")
            if isinstance(visibility_metrics_cycle, dict):
                cycle_stats["proactive_recent_avg_upvotes"] = float(
                    visibility_metrics_cycle.get("recent_avg_upvotes", 0.0) or 0.0
                )
                cycle_stats["proactive_recent_avg_comments"] = float(
                    visibility_metrics_cycle.get("recent_avg_comments", 0.0) or 0.0
                )
                cycle_stats["proactive_recent_avg_visibility_score"] = float(
                    visibility_metrics_cycle.get("recent_avg_visibility_score", 0.0) or 0.0
                )
                cycle_stats["proactive_visibility_delta_pct"] = float(
                    visibility_metrics_cycle.get("visibility_delta_pct", 0.0) or 0.0
                )
                cycle_stats["proactive_target_hit_rate"] = float(
                    visibility_metrics_cycle.get("recent_target_hit_rate", 0.0) or 0.0
                )
                cycle_stats["proactive_target_upvotes"] = int(
                    visibility_metrics_cycle.get("target_upvotes", 0) or 0
                )
            lift_terms_cycle: List[str] = []
            raw_lifts_cycle = learning_snapshot_cycle.get("winning_terms_lift")
            if isinstance(raw_lifts_cycle, list):
                for item in raw_lifts_cycle[:8]:
                    if not isinstance(item, dict):
                        continue
                    term = _clean_signal_term(item.get("term"))
                    if term and term not in lift_terms_cycle:
                        lift_terms_cycle.append(term)
            cycle_stats["proactive_top_lift_terms"] = lift_terms_cycle
            discovery_market_signals = build_top_post_signals(
                posts=posts,
                limit=max(8, cfg.proactive_post_reference_limit),
                source="discovery",
            )
            if discovery_market_signals:
                market_snapshot = update_market_signals(
                    memory=post_memory,
                    signals=discovery_market_signals,
                )
                cycle_stats["market_signal_count"] = market_snapshot.get("signal_count", 0)
                cycle_stats["market_question_title_rate"] = market_snapshot.get("question_title_rate", 0.0)
                cycle_stats["market_top_submolts"] = market_snapshot.get("top_submolts", [])[:3]
            maybe_write_self_improvement_suggestions(
                cfg=cfg,
                logger=logger,
                iteration=iteration,
                persona_text=persona_text,
                domain_context_text=domain_context_text,
                learning_snapshot=learning_snapshot_cycle,
                cycle_titles=cycle_titles,
                cycle_stats=cycle_stats,
            )
            metrics_history = state.get("cycle_metrics_history", [])
            if not isinstance(metrics_history, list):
                metrics_history = []
            metrics_history.append(
                {
                    "cycle": iteration,
                    "inspected": inspected,
                    "new_candidates": new_candidates,
                    "eligible_now": eligible_now,
                    "drafted": drafted_count,
                    "model_approved": model_approved,
                    "actions": acted,
                    "skip_reasons": skip_reasons,
                    "effective_draft_shortlist_size": effective_shortlist_size,
                    "effective_draft_signal_min_score": effective_signal_min_score,
                    "draft_shortlist_mode": shortlist_mode,
                }
            )
            state["cycle_metrics_history"] = metrics_history[-240:]

            if cfg.analytics_refresh_interval_cycles > 0 and iteration % cfg.analytics_refresh_interval_cycles == 0:
                try:
                    tracked_post_ids: Set[str] = set(replied_posts)
                    proactive_entries = post_memory.get("proactive_posts", [])
                    if isinstance(proactive_entries, list):
                        for item in proactive_entries[-160:]:
                            if not isinstance(item, dict):
                                continue
                            tracked = normalize_str(item.get("post_id")).strip()
                            if tracked:
                                tracked_post_ids.add(tracked)
                    refresh_post_metrics(
                        cfg.analytics_db_path,
                        client=client,
                        tracked_post_ids=tracked_post_ids,
                        logger=logger,
                        fetch_limit=max(40, cfg.posts_limit),
                    )
                except Exception as e:
                    logger.warning("Analytics metrics refresh failed error=%s", e)

            if cfg.analytics_summary_interval_cycles > 0 and iteration % cfg.analytics_summary_interval_cycles == 0:
                try:
                    top_skips = aggregate_skip_reasons(
                        state.get("cycle_metrics_history", []),
                        window=max(6, cfg.analytics_summary_interval_cycles),
                    )
                    summary = daily_summary(cfg.analytics_db_path, top_skip_reasons=top_skips)
                    logger.info(
                        "Daily analytics summary date=%s best_archetype=%s best_hook_pattern=%s top_skip_reasons=%s",
                        summary.get("date"),
                        summary.get("best_archetype"),
                        summary.get("best_hook_pattern"),
                        summary.get("top_skip_reasons"),
                    )
                except Exception as e:
                    logger.warning("Analytics daily summary failed error=%s", e)

            # Sleep policy:
            # - Poll quickly when idle.
            # - Never sleep for post cooldown: keep discovery/learning running.
            # - For comment actions, honor only the short comment cooldown.
            sleep_seconds = max(1, cfg.idle_poll_seconds)
            sleep_reason = "idle_poll"
            if reply_actions > 0:
                post_remaining, comment_remaining = cooldown_remaining_seconds(state=state, cfg=cfg)
                if comment_action_sent and comment_remaining > 0:
                    sleep_seconds = max(1, comment_remaining)
                    sleep_reason = "comment_cooldown"
                elif post_action_sent and post_remaining > 0:
                    sleep_seconds = max(1, cfg.idle_poll_seconds)
                    sleep_reason = "post_cooldown_background"
                else:
                    sleep_seconds = max(1, cfg.idle_poll_seconds)
                    sleep_reason = "cooldown_elapsed"
        except MoltbookAuthError as e:
            logger.error("Poll cycle=%s auth_error=%s", iteration, e)
            sleep_seconds = max(1, cfg.poll_seconds)
            sleep_reason = "auth_error_backoff"
        except Exception as e:
            logger.exception("Poll cycle=%s loop_error=%s", iteration, e)
            sleep_seconds = max(1, cfg.poll_seconds)
            sleep_reason = "loop_error_backoff"

        logger.info("Sleeping seconds=%s reason=%s", sleep_seconds, sleep_reason)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    run_loop()
