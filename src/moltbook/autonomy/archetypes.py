from __future__ import annotations

from typing import Dict, List, Tuple

from .drafting import normalize_str


ARCHETYPE_SECTION_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "security_advisory": ("threat model", "mitigation", "question"),
    "build_log": ("goal", "steps", "what broke", "result", "ergo mechanism framing", "question"),
    "mechanism_explainer": ("concept", "worked example", "constraint", "trade-off", "question"),
    "operator_reliability": ("reliability principle", "checklist", "failure mode", "ergo mechanism framing", "question"),
    "myth_correction": ("claim", "evidence", "correction", "ergo mechanism framing", "question"),
    "agent_economy_teardown": ("numbers", "incentives", "design suggestion", "trade-off", "question"),
}


def validate_archetype_template(name: str, template_text: str) -> Tuple[bool, List[str]]:
    archetype = normalize_str(name).strip().lower()
    body = normalize_str(template_text).strip().lower()
    required = ARCHETYPE_SECTION_REQUIREMENTS.get(archetype, ())
    if not required:
        return False, [f"unknown_archetype:{archetype or '(empty)'}"]

    missing: List[str] = []
    for section in required:
        if section not in body:
            missing.append(section)
    return (len(missing) == 0), missing
