from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from domains.bend.services.promotion_readiness import load_program_status


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATUS_PATH = PROJECT_ROOT / "docs/research/artifacts/f1_program_status_current.json"


def _status_path() -> Path:
    configured = os.getenv("SHERMAN_QC_F1_PROGRAM_STATUS_PATH")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_STATUS_PATH


def load_bend_inspection_readiness(path: Path | None = None) -> Dict[str, Any]:
    return load_program_status(path or _status_path())


async def get_bend_inspection_readiness():
    return load_bend_inspection_readiness()
