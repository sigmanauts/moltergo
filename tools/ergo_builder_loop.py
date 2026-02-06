#!/usr/bin/env python3
"""Compatibility shim for the older loop entrypoint.

Prefer running `python -m moltbook.autonomy.runner`.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from moltbook.autonomy.runner import run_loop


if __name__ == "__main__":
    run_loop()
