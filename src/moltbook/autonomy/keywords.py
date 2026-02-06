import json
from pathlib import Path
from typing import Dict, List


def load_keyword_store(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {"learned_keywords": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"learned_keywords": []}

    learned = data.get("learned_keywords")
    if not isinstance(learned, list):
        learned = []
    return {"learned_keywords": [str(item).strip().lower() for item in learned if str(item).strip()]}


def save_keyword_store(path: Path, store: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, sort_keys=True)


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
