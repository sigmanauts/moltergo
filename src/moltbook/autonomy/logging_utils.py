import logging
from .config import Config


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
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if cfg.log_path:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
