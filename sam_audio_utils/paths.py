from __future__ import annotations

import os
from pathlib import Path


def env_path(env_key: str, default: Path | str) -> Path:
    raw = os.environ.get(env_key)
    return Path(raw) if raw else Path(default)
