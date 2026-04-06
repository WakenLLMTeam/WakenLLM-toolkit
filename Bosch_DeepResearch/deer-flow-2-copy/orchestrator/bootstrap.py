"""Ensure the DeerFlow harness package is importable when running from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_deerflow_importable() -> Path | None:
    """Insert ``backend/packages/harness`` into ``sys.path`` if present.

    Returns the harness directory when it was found, else ``None``.
    """
    root = Path(__file__).resolve().parent.parent
    harness = root / "backend" / "packages" / "harness"
    if harness.is_dir():
        s = str(harness)
        if s not in sys.path:
            sys.path.insert(0, s)
        return harness
    return None
