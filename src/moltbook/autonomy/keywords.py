import json
from pathlib import Path
from typing import Dict, List


def _normalize(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = str(item).strip().lower()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_keyword_store(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {"learned_keywords": [], "pending_suggestions": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"learned_keywords": [], "pending_suggestions": []}

    learned = data.get("learned_keywords")
    if not isinstance(learned, list):
        learned = []
    pending = data.get("pending_suggestions")
    if not isinstance(pending, list):
        pending = []
    return {
        "learned_keywords": _normalize(learned),
        "pending_suggestions": _normalize(pending),
    }


def save_keyword_store(path: Path, store: Dict[str, List[str]]) -> None:
    learned = _normalize(store.get("learned_keywords", []))
    pending = [k for k in _normalize(store.get("pending_suggestions", [])) if k not in set(learned)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "learned_keywords": learned,
                "pending_suggestions": pending,
            },
            f,
            indent=2,
            sort_keys=True,
        )


def merge_keywords(base_keywords: List[str], learned_keywords: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for keyword in base_keywords + learned_keywords:
        k = str(keyword).strip().lower()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        merged.append(k)
    return merged
