from __future__ import annotations

from pathlib import Path
from typing import Optional


def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_file(path: str | Path, suffix: str) -> Optional[Path]:
    p = Path(path)
    files = sorted(p.glob(f'*{suffix}'))
    return files[-1] if files else None
