from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import Config
from .drafting import build_self_improvement_messages, call_generation_model, normalize_str
from .keywords import merge_keywords, save_keyword_store
from .post_engine_memory import (
    append_improvement_suggestions,
    append_improvement_suggestions_text,
    build_improvement_diagnostics,
    build_improvement_feedback_context,
    load_recent_improvement_entries,
    load_recent_improvement_raw_entries,
    sanitize_improvement_suggestions,
    update_improvement_backlog,
)
from .ui import confirm_keyword_addition, print_keyword_added_banner


def has_generation_provider(cfg: Config) -> bool:
    if cfg.llm_provider == "chatbase":
        return bool(cfg.chatbase_api_key and cfg.chatbase_chatbot_id)
    if cfg.llm_provider == "groq":
        return bool(cfg.groq_api_key)
    if cfg.llm_provider == "ollama":
        return bool(cfg.ollama_model and cfg.ollama_base_url)
    if cfg.llm_provider == "openai":
        return bool(cfg.openai_api_key)
    return bool(
        (cfg.chatbase_api_key and cfg.chatbase_chatbot_id)
        or cfg.groq_api_key
        or (cfg.ollama_model and cfg.ollama_base_url)
        or cfg.openai_api_key
    )


def review_pending_keyword_suggestions(
    cfg: Config,
    logger,
    keyword_store: Dict[str, Any],
    active_keywords: List[str],
    approve_all_keyword_changes: bool,
) -> Tuple[List[str], Dict[str, Any], bool, bool]:
    pending = list(keyword_store.get("pending_suggestions", []))
    if not pending:
        return active_keywords, keyword_store, approve_all_keyword_changes, False

    logger.info("Pending keyword suggestions awaiting review count=%s", len(pending))
    remaining: List[str] = []
    should_stop_run = False
    for keyword in pending:
        approved, approve_all_keyword_changes, should_stop = confirm_keyword_addition(
            logger=logger,
            keyword=keyword,
            approve_all=approve_all_keyword_changes,
        )
        if should_stop:
            remaining.append(keyword)
            should_stop_run = True
            continue
        if not approved:
            continue
        learned_before = keyword_store.get("learned_keywords", [])
        learned_after = merge_keywords(learned_before, [keyword])
        if len(learned_after) == len(learned_before):
            continue
        keyword_store["learned_keywords"] = learned_after
        active_keywords = merge_keywords(cfg.keywords, learned_after)
        logger.info(
            "Keyword approved from pending keyword=%s learned_total=%s active_total=%s",
            keyword,
            len(learned_after),
            len(active_keywords),
        )
        print_keyword_added_banner(keyword=keyword, learned_total=len(learned_after))

    keyword_store["pending_suggestions"] = merge_keywords([], remaining)
    save_keyword_store(cfg.keyword_store_path, keyword_store)
    return active_keywords, keyword_store, approve_all_keyword_changes, should_stop_run


def _deterministic_improvement_hints(
    diagnostics: Dict[str, Any],
    cycle_stats: Dict[str, Any],
    learning_snapshot: Dict[str, Any],
    feedback_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    hints: List[str] = []
    bottleneck = normalize_str(diagnostics.get("bottleneck_label")).strip()
    approval_rate = float(diagnostics.get("approval_rate", 0.0) or 0.0)
    execution_rate = float(diagnostics.get("execution_rate", 0.0) or 0.0)
    drafted = int(diagnostics.get("drafted", 0) or 0)
    eligible_now = int(diagnostics.get("eligible_now", 0) or 0)
    actions = int(diagnostics.get("actions", 0) or 0)

    if bottleneck == "model_rejection" or (drafted >= 12 and approval_rate < 0.1):
        hints.append("Model approval is weak. Prioritize relevance filters and stricter reject criteria before drafting.")
    if bottleneck == "execution_blocked" or (eligible_now >= 20 and actions == 0):
        hints.append("Execution conversion is weak. Focus on reducing non-actionable drafts and improving action readiness.")
    if bottleneck == "cooldown_limited":
        hints.append("Cooldown pressure is high. Prioritize comments or defer post-like actions while preserving discovery.")
    if bottleneck == "duplication_pressure":
        hints.append("Duplication pressure detected. Tighten dedupe/reply-once checks before generating new replies.")
    if execution_rate < 0.05 and eligible_now >= 15:
        hints.append("Eligible volume is high but execution is near zero. Add shortlist ranking before LLM calls.")

    market_snapshot = learning_snapshot.get("market_snapshot")
    if isinstance(market_snapshot, dict):
        q_rate = market_snapshot.get("question_title_rate")
        if isinstance(q_rate, (int, float)) and q_rate >= 0.35:
            hints.append("Question-style titles are trending. Prefer direct question hooks in proactive posts.")
        top_terms = market_snapshot.get("top_terms")
        if isinstance(top_terms, list) and top_terms:
            joined = ", ".join([normalize_str(t).strip() for t in top_terms[:6] if normalize_str(t).strip()])
            if joined:
                hints.append(f"Top market terms now: {joined}. Use them only when context actually fits.")
    visibility_metrics = learning_snapshot.get("visibility_metrics")
    if isinstance(visibility_metrics, dict):
        target_upvotes = int(visibility_metrics.get("target_upvotes", 0) or 0)
        hit_rate = float(visibility_metrics.get("recent_target_hit_rate", 0.0) or 0.0)
        delta_pct = float(visibility_metrics.get("visibility_delta_pct", 0.0) or 0.0)
        if target_upvotes > 0 and hit_rate < 0.35:
            hints.append(
                (
                    "Visibility under target. Raise opening-hook specificity and implementation detail density "
                    f"until recent target hit rate improves (target_upvotes={target_upvotes}, hit_rate={round(hit_rate, 3)})."
                )
            )
        if delta_pct <= -0.15:
            hints.append(
                f"Visibility momentum is negative ({round(delta_pct * 100, 1)}%). Prioritize high-lift themes only."
            )

    raw_skip = cycle_stats.get("skip_reasons")
    if isinstance(raw_skip, dict):
        items = sorted(raw_skip.items(), key=lambda x: int(x[1]), reverse=True)
        if items:
            label, count = items[0]
            hints.append(f"Dominant skip reason this cycle: {normalize_str(label)}={int(count)}.")
        if int(raw_skip.get("quality_gate_failed", 0) or 0) >= 3:
            hints.append(
                "Many drafts fail the quality gate. Tighten prompt specificity and keep mechanism-first phrasing."
            )
        if int(raw_skip.get("trend_context_mismatch", 0) or 0) >= 5:
            hints.append(
                "Trend/context mismatch is high. Favor candidates with both market-term overlap and clear Ergo mechanism."
            )
    if isinstance(feedback_context, dict):
        zero_action_streak = int(feedback_context.get("zero_action_streak", 0) or 0)
        avg_approval = float(feedback_context.get("avg_approval_rate", 0.0) or 0.0)
        avg_execution = float(feedback_context.get("avg_execution_rate", 0.0) or 0.0)
        if zero_action_streak >= 4:
            hints.append(
                (
                    "Zero-action streak is elevated. Enable streak recovery controls: "
                    "lower LLM draft budget, tighten shortlist, and avoid strict trend mismatch pruning."
                )
            )
        if zero_action_streak >= 6 and avg_approval < 0.2 and avg_execution < 0.2:
            hints.append(
                (
                    "Conversion collapse detected. Prioritize deterministic proactive post lane until "
                    "approval/execution rates recover."
                )
            )
    return hints[:8]


def _deterministic_improvement_suggestions(
    diagnostics: Dict[str, Any],
    cycle_stats: Dict[str, Any],
    learning_snapshot: Dict[str, Any],
    feedback_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hints = _deterministic_improvement_hints(
        diagnostics=diagnostics,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
        feedback_context=feedback_context,
    )
    prompt_changes: List[Dict[str, Any]] = []
    code_changes: List[Dict[str, Any]] = []
    strategy_experiments: List[Dict[str, Any]] = []

    bottleneck = normalize_str(diagnostics.get("bottleneck_label")).strip()
    approval_rate = float(diagnostics.get("approval_rate", 0.0) or 0.0)
    drafted = int(diagnostics.get("drafted", 0) or 0)
    eligible_now = int(diagnostics.get("eligible_now", 0) or 0)
    actions = int(diagnostics.get("actions", 0) or 0)
    execution_rate = float(diagnostics.get("execution_rate", 0.0) or 0.0)

    if bottleneck == "model_rejection" or (drafted >= 12 and approval_rate < 0.1):
        prompt_changes.append(
            {
                "target": "drafting relevance gate",
                "proposed_change": (
                    "Require one explicit Ergo mechanism plus one thread-specific implementation angle before should_respond=true."
                ),
                "reason": "Low model approval means current candidates are too broad or generic.",
                "expected_impact": "Higher approval rate and fewer wasted drafts.",
            }
        )
    if eligible_now >= 20 and actions == 0:
        code_changes.append(
            {
                "file_hint": "src/moltbook/autonomy/runner.py",
                "proposed_change": (
                    "Add pre-draft shortlist ranking so only top N eligible candidates reach the LLM each cycle."
                ),
                "reason": "High eligible volume with zero actions indicates poor conversion efficiency.",
                "risk": "May miss edge-case opportunities if shortlist is too small.",
            }
        )
    if bottleneck == "duplication_pressure":
        code_changes.append(
            {
                "file_hint": "src/moltbook/autonomy/runner.py",
                "proposed_change": (
                    "Strengthen reply dedupe by caching parent-comment fingerprints with longer retention in state."
                ),
                "reason": "Repeated reply targets hurt trust and consume action budget.",
                "risk": "Over-aggressive dedupe may skip valid follow-up contexts.",
            }
        )
    if execution_rate < 0.08 and eligible_now >= 15:
        strategy_experiments.append(
            {
                "idea": "Enable dynamic shortlist size based on last 3-cycle approval/execution rates.",
                "metric": "execution_rate and actions per cycle after shortlist enabled",
                "stop_condition": "Disable if execution_rate does not improve after 12 cycles",
            }
        )
    if isinstance(feedback_context, dict):
        zero_action_streak = int(feedback_context.get("zero_action_streak", 0) or 0)
        if zero_action_streak >= 4:
            code_changes.append(
                {
                    "file_hint": "src/moltbook/autonomy/strategy.py",
                    "proposed_change": (
                        "Add explicit streak recovery mode in adaptive controls to lower draft budgets "
                        "and avoid trend-context over-pruning after repeated zero-action cycles."
                    ),
                    "reason": "Recurring zero-action streak indicates the current adaptive loop is stuck.",
                    "risk": "May reduce topic diversity temporarily while recovery mode is active.",
                }
            )
            strategy_experiments.append(
                {
                    "idea": "Track recovery-mode effectiveness vs. baseline over 12 cycles.",
                    "metric": "approval_rate, execution_rate, and zero_action_streak",
                    "stop_condition": "Disable if zero_action_streak does not improve within 12 cycles",
                }
            )
    visibility_metrics = learning_snapshot.get("visibility_metrics")
    if isinstance(visibility_metrics, dict):
        hit_rate = float(visibility_metrics.get("recent_target_hit_rate", 0.0) or 0.0)
        target_upvotes = int(visibility_metrics.get("target_upvotes", 0) or 0)
        if target_upvotes > 0 and hit_rate < 0.35:
            prompt_changes.append(
                {
                    "target": "visibility targeting",
                    "proposed_change": (
                        "Require proactive drafts to open with one concrete pain point plus one Ergo mechanism in the first two lines."
                    ),
                    "reason": "Recent posts are underperforming the target upvote threshold.",
                    "expected_impact": (
                        f"Higher share of posts crossing {target_upvotes}+ upvotes by improving hook clarity."
                    ),
                }
            )
            code_changes.append(
                {
                    "file_hint": "src/moltbook/autonomy/runner.py",
                    "proposed_change": (
                        "Bias ranking toward terms with positive lift from proactive memory and penalize repeated low-lift terms."
                    ),
                    "reason": "Visibility target hit rate is low, so selection should follow measured term lift.",
                    "risk": "Overfitting to short-term language trends can reduce topic diversity.",
                }
            )
    raw_skip = cycle_stats.get("skip_reasons")
    if isinstance(raw_skip, dict):
        if int(raw_skip.get("quality_gate_failed", 0) or 0) >= 3:
            prompt_changes.append(
                {
                    "target": "draft content shape",
                    "proposed_change": (
                        "Require one concrete Ergo mechanism sentence before any question; reject abstract framing."
                    ),
                    "reason": "Quality gate failures indicate drafts are still too generic.",
                    "expected_impact": "Higher quality-pass rate and fewer dropped drafts.",
                }
            )

    summary = (
        "Deterministic diagnostics suggest conversion-focused tuning."
        if hints
        else "Deterministic diagnostics found no additional changes."
    )
    return {
        "summary": summary,
        "priority": "medium",
        "prompt_changes": prompt_changes,
        "code_changes": code_changes,
        "strategy_experiments": strategy_experiments,
    }


def _suggestion_signature(kind: str, item: Dict[str, Any]) -> str:
    if kind == "prompt_changes":
        raw = " ".join([normalize_str(item.get("target")), normalize_str(item.get("proposed_change"))])
    elif kind == "code_changes":
        raw = " ".join([normalize_str(item.get("file_hint")), normalize_str(item.get("proposed_change"))])
    else:
        raw = " ".join(
            [
                normalize_str(item.get("idea")),
                normalize_str(item.get("metric")),
                normalize_str(item.get("stop_condition")),
            ]
        )
    raw = re.sub(r"[^a-z0-9]+", " ", raw.lower())
    return " ".join(raw.split())[:260]


def _merge_improvement_payloads(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    max_items: int,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "summary": normalize_str(primary.get("summary")).strip() or normalize_str(fallback.get("summary")).strip(),
        "priority": normalize_str(primary.get("priority")).strip() or "medium",
        "prompt_changes": [],
        "code_changes": [],
        "strategy_experiments": [],
    }
    max_items = max(1, int(max_items))
    for kind in ("prompt_changes", "code_changes", "strategy_experiments"):
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for source in (primary, fallback):
            raw = source.get(kind)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                sig = _suggestion_signature(kind, item)
                if not sig or sig in seen:
                    continue
                seen.add(sig)
                out.append(item)
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break
        merged[kind] = out
    return merged


def maybe_write_self_improvement_suggestions(
    cfg: Config,
    logger,
    iteration: int,
    persona_text: str,
    domain_context_text: str,
    learning_snapshot: Dict[str, Any],
    cycle_titles: List[str],
    cycle_stats: Dict[str, Any],
) -> None:
    if not cfg.self_improve_enabled:
        return
    if cfg.self_improve_interval_cycles <= 0:
        return
    if iteration % cfg.self_improve_interval_cycles != 0:
        return
    if len(cycle_titles) < cfg.self_improve_min_titles:
        logger.info(
            "Self-improvement skipped cycle=%s reason=insufficient_titles titles=%s min_titles=%s",
            iteration,
            len(cycle_titles),
            cfg.self_improve_min_titles,
        )
        return
    diagnostics = build_improvement_diagnostics(cycle_stats)
    feedback_context = build_improvement_feedback_context(
        path=cfg.self_improve_path,
        current_cycle_stats=cycle_stats,
    )
    use_llm = has_generation_provider(cfg)
    if use_llm and int(feedback_context.get("zero_action_streak", 0) or 0) >= 6:
        use_llm = False
        logger.info(
            "Self-improvement cycle=%s running deterministic-only mode reason=zero_action_streak",
            iteration,
        )
    if not use_llm:
        logger.info(
            "Self-improvement cycle=%s running deterministic-only mode reason=no_generation_provider_or_suppressed",
            iteration,
        )
    deterministic_hints = _deterministic_improvement_hints(
        diagnostics=diagnostics,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
        feedback_context=feedback_context,
    )
    deterministic_payload = _deterministic_improvement_suggestions(
        diagnostics=diagnostics,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
        feedback_context=feedback_context,
    )

    provider_used = "unknown"
    llm_payload: Dict[str, Any] = {}
    if use_llm:
        try:
            prior_suggestions = load_recent_improvement_entries(path=cfg.self_improve_path, limit=8)
            messages = build_self_improvement_messages(
                persona=persona_text,
                domain_context=domain_context_text,
                learning_snapshot=learning_snapshot,
                recent_titles=cycle_titles,
                cycle_stats=cycle_stats,
                prior_suggestions=prior_suggestions,
                feedback_context=feedback_context,
                deterministic_hints=deterministic_hints,
            )
            generated, provider_used, _ = call_generation_model(cfg, messages)
            if isinstance(generated, dict):
                llm_payload = generated
            else:
                logger.warning("Self-improvement returned non-object payload cycle=%s", iteration)
                provider_used = "deterministic"
        except Exception as e:
            logger.warning("Self-improvement failed cycle=%s error=%s", iteration, e)
            provider_used = "deterministic"
            llm_payload = {}
    else:
        provider_used = "deterministic"

    max_suggestions = max(1, cfg.self_improve_max_suggestions)
    recent_raw = load_recent_improvement_raw_entries(path=cfg.self_improve_path, limit=72)
    llm_suggestions = sanitize_improvement_suggestions(
        suggestions=llm_payload,
        recent_raw_entries=recent_raw,
        max_items=max_suggestions,
    )
    combined_payload = _merge_improvement_payloads(
        primary=llm_suggestions,
        fallback=deterministic_payload,
        max_items=max_suggestions,
    )
    suggestions = sanitize_improvement_suggestions(
        suggestions=combined_payload,
        recent_raw_entries=recent_raw,
        max_items=max_suggestions,
    )
    if isinstance(suggestions.get("prompt_changes"), list):
        suggestions["prompt_changes"] = suggestions["prompt_changes"][:max_suggestions]
    if isinstance(suggestions.get("code_changes"), list):
        suggestions["code_changes"] = suggestions["code_changes"][:max_suggestions]
    if isinstance(suggestions.get("strategy_experiments"), list):
        suggestions["strategy_experiments"] = suggestions["strategy_experiments"][:max_suggestions]

    prompt_changes = suggestions.get("prompt_changes")
    code_changes = suggestions.get("code_changes")
    strategy_experiments = suggestions.get("strategy_experiments")
    if not any(
        (
            isinstance(prompt_changes, list) and prompt_changes,
            isinstance(code_changes, list) and code_changes,
            isinstance(strategy_experiments, list) and strategy_experiments,
        )
    ):
        logger.info(
            "Self-improvement produced no novel actionable suggestions cycle=%s bottleneck=%s",
            iteration,
            diagnostics.get("bottleneck_label"),
        )
        return

    append_improvement_suggestions(
        path=cfg.self_improve_path,
        cycle=iteration,
        provider=provider_used,
        suggestions=suggestions,
        cycle_stats=cycle_stats,
        diagnostics=diagnostics,
    )
    append_improvement_suggestions_text(
        path=cfg.self_improve_text_path,
        cycle=iteration,
        provider=provider_used,
        suggestions=suggestions,
        cycle_stats=cycle_stats,
        learning_snapshot=learning_snapshot,
        diagnostics=diagnostics,
        feedback_context=feedback_context,
    )
    update_improvement_backlog(
        path=cfg.self_improve_backlog_path,
        cycle=iteration,
        provider=provider_used,
        suggestions=suggestions,
        diagnostics=diagnostics,
    )
    logger.info(
        (
            "Self-improvement suggestions saved cycle=%s provider=%s json_path=%s "
            "text_path=%s backlog_path=%s bottleneck=%s"
        ),
        iteration,
        provider_used,
        cfg.self_improve_path,
        cfg.self_improve_text_path,
        cfg.self_improve_backlog_path,
        diagnostics.get("bottleneck_label"),
    )
