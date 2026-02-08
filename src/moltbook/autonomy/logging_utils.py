import logging
import os
import sys
from .config import Config


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[31m",
}


def _stream_supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    force = os.getenv("FORCE_COLOR", "").strip().lower()
    if force in {"1", "true", "yes"}:
        return True
    return bool(sys.stderr.isatty())


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        level_name = record.levelname.upper()
        color = _COLORS.get(level_name, "")
        if not color:
            return message

        # Highlight key runtime phases so operators can scan the console quickly.
        if "LLM request" in message:
            painted = f"{_BOLD}{_CYAN}[LLM REQUEST] {message}{_RESET}"
        elif "LLM response" in message:
            painted = f"{_BOLD}{_MAGENTA}[LLM RESPONSE] {message}{_RESET}"
        elif "drafting post_id=" in message:
            painted = f"{_BOLD}{_CYAN}[DRAFT] {message}{_RESET}"
        elif "action=post attempt" in message:
            painted = f"{_BOLD}{_MAGENTA}[POST ATTEMPT] {message}{_RESET}"
        elif "action=comment attempt" in message:
            painted = f"{_BOLD}{_CYAN}[COMMENT ATTEMPT] {message}{_RESET}"
        elif "skip post_id=" in message and "reason=no_action_slots" in message:
            painted = f"{_BOLD}{_YELLOW}[ACTION SLOT BLOCKED] {message}{_RESET}"
        elif "token_saver llm_cycle_budget=" in message:
            painted = f"{_BOLD}{_YELLOW}[TOKEN SAVER] {message}{_RESET}"
        elif "ACTION SUCCESS" in message or "Proactive post success" in message:
            painted = f"{_BOLD}{_GREEN}[SUCCESS] {message}{_RESET}"
        elif "Sleeping seconds=" in message:
            painted = f"{_DIM}{color}{message}{_RESET}"
        elif "WARNING" in message:
            painted = f"{_BOLD}{_YELLOW}{message}{_RESET}"
        else:
            painted = f"{color}{message}{_RESET}"
        return painted


def setup_logging(cfg: Config) -> logging.Logger:
    level = getattr(logging, cfg.log_level, logging.INFO)
    logger = logging.getLogger("moltbook.autonomy")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    if _stream_supports_color():
        stream_handler.setFormatter(
            ColorFormatter(
                fmt="%(asctime)sZ %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    else:
        stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if cfg.log_path:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
