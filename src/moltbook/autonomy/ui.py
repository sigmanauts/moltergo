from __future__ import annotations

import os
import select
import sys
import textwrap
from typing import Any, Dict, List, Tuple

from .config import Config
from .drafting import normalize_str


def supports_color() -> bool:
    return sys.stdout.isatty() and not bool(os.getenv("NO_COLOR"))


def _ui_palette() -> Dict[str, str]:
    if not supports_color():
        return {
            "reset": "",
            "bold": "",
            "dim": "",
            "blue": "",
            "cyan": "",
            "green": "",
            "yellow": "",
            "magenta": "",
            "white": "",
        }
    return {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "blue": "\033[1;34m",
        "cyan": "\033[1;36m",
        "green": "\033[1;32m",
        "yellow": "\033[1;33m",
        "magenta": "\033[1;35m",
        "white": "\033[1;37m",
    }


def _ui_paint(text: str, tone: str = "", bold: bool = False, dim: bool = False) -> str:
    palette = _ui_palette()
    reset = palette["reset"]
    if not reset:
        return text
    chunks: List[str] = []
    if bold:
        chunks.append(palette["bold"])
    if dim:
        chunks.append(palette["dim"])
    if tone:
        chunks.append(palette.get(tone, ""))
    chunks.append(text)
    chunks.append(reset)
    return "".join(chunks)


def _ui_wrap_lines(value: Any, width: int) -> List[str]:
    text = normalize_str(value).strip()
    if width < 8:
        width = 8
    if not text:
        return [""]
    out: List[str] = []
    for raw_line in text.splitlines():
        line = normalize_str(raw_line).rstrip()
        if not line:
            out.append("")
            continue
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        out.extend(wrapped or [""])
    return out or [""]


def _ui_print_panel(
    title: str,
    rows: List[Tuple[str, Any]],
    tone: str = "cyan",
    width: int = 74,
) -> None:
    inner = max(30, width - 4)
    border = "+" + ("-" * (inner + 2)) + "+"
    print("")
    print(_ui_paint(border, tone=tone, bold=True))
    title_text = normalize_str(title).strip() or "INFO"
    print(_ui_paint(f"| {title_text:<{inner}} |", tone=tone, bold=True))
    print(_ui_paint(border, tone=tone))
    for key, value in rows:
        label = normalize_str(key).strip()
        value_lines = _ui_wrap_lines(value, width=(inner - (len(label) + 2) if label else inner))
        for idx, line in enumerate(value_lines):
            if label:
                prefix = f"{label}: " if idx == 0 else (" " * (len(label) + 2))
            else:
                prefix = ""
            content = f"{prefix}{line}"
            print(f"| {content:<{inner}} |")
    print(_ui_paint(border, tone=tone))
    print("")


def print_success_banner(action: str, pid: str, url: str, title: str) -> None:
    _ui_print_panel(
        title=f"[SUCCESS] {action.upper()}",
        rows=[
            ("post_id", pid),
            ("title", title),
            ("url", url),
        ],
        tone="green",
    )


def print_cycle_banner(iteration: int, mode: str, keywords: int) -> None:
    summary = f"discovery={mode} | keywords={keywords}"
    _ui_print_panel(
        title=f"CYCLE {iteration}",
        rows=[("", summary)],
        tone="blue",
        width=62,
    )


def print_runtime_banner(cfg: Config) -> None:
    provider = cfg.llm_provider
    if provider == "auto":
        provider = "auto(chatbase-first)"
    _ui_print_panel(
        title="MOLTBOOK AUTONOMY",
        rows=[
            ("mode", f"discovery={cfg.discovery_mode} reply={cfg.reply_mode}"),
            ("llm", f"{provider} | fallback_openai={int(cfg.llm_auto_fallback_to_openai)}"),
            (
                "limits",
                f"post/{cfg.min_seconds_between_posts}s "
                f"comment/{cfg.min_seconds_between_comments}s "
                f"triage_budget/{max(1, cfg.reply_triage_llm_calls_per_scan)}",
            ),
        ],
        tone="magenta",
    )


def print_keyword_added_banner(keyword: str, learned_total: int) -> None:
    _ui_print_panel(
        title="KEYWORD ADDED",
        rows=[
            ("keyword", keyword),
            ("learned_total", str(learned_total)),
        ],
        tone="green",
        width=56,
    )


def confirm_action(
    cfg: Config,
    logger,
    action: str,
    pid: str,
    title: str,
    submolt: str,
    url: str,
    author: str,
    content_preview: str | None,
    approve_all: bool,
) -> Tuple[bool, bool, bool]:
    if not cfg.confirm_actions:
        return True, approve_all, False

    if approve_all:
        return True, approve_all, False

    if not sys.stdin.isatty():
        logger.warning(
            "Skipping action=%s post_id=%s because confirmation is enabled but stdin is not interactive. "
            "Set MOLTBOOK_CONFIRM_ACTIONS=0 for unattended mode.",
            action,
            pid,
        )
        return False, approve_all, False

    _ui_print_panel(
        title="[CONFIRM] PROPOSED ACTION",
        rows=[
            ("action", action),
            ("post_id", pid),
            ("author", author),
            ("submolt", submolt),
            ("url", url),
            ("title", title),
        ],
        tone="yellow",
    )
    if content_preview:
        _ui_print_panel(
            title="DRAFT PREVIEW",
            rows=[("content", content_preview)],
            tone="cyan",
        )
    prompt = "Proceed? [y]es / [n]o / [a]ll remaining / [q]uit: "
    choice = ""
    if cfg.confirm_timeout_seconds > 0:
        default_choice = cfg.confirm_default_choice
        print(
            _ui_paint(
                (
                    f"[auto] default='{default_choice}' in {cfg.confirm_timeout_seconds}s "
                    "if no input is provided."
                ),
                tone="magenta",
                bold=True,
            )
        )
        print(prompt, end="", flush=True)
        ready, _, _ = select.select([sys.stdin], [], [], cfg.confirm_timeout_seconds)
        if ready:
            choice = sys.stdin.readline().strip().lower()
        else:
            print("")
            choice = default_choice
    else:
        choice = input(prompt).strip().lower()

    if choice in {"y", "yes"}:
        return True, approve_all, False
    if choice in {"a", "all"}:
        logger.info("Operator approved all remaining actions for this run.")
        return True, True, False
    if choice in {"q", "quit"}:
        logger.info("Operator requested stop from confirmation prompt.")
        return False, approve_all, True
    return False, approve_all, False


def confirm_keyword_addition(
    logger,
    keyword: str,
    approve_all: bool,
) -> Tuple[bool, bool, bool]:
    if approve_all:
        return True, approve_all, False

    if not sys.stdin.isatty():
        logger.warning(
            "Skipping learned keyword '%s' because stdin is non-interactive.",
            keyword,
        )
        return False, approve_all, False

    _ui_print_panel(
        title="[CONFIRM] LEARNED KEYWORD",
        rows=[("keyword", keyword)],
        tone="yellow",
        width=56,
    )
    choice = input("Add this keyword to learning store? [y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
    if choice in {"y", "yes"}:
        return True, approve_all, False
    if choice in {"a", "all"}:
        logger.info("Operator approved all remaining keyword suggestions for this run.")
        return True, True, False
    if choice in {"q", "quit"}:
        logger.info("Operator requested stop from keyword confirmation prompt.")
        return False, approve_all, True
    return False, approve_all, False
