"""Location and thread count of the zs_rust binary.

Environment:
  ZS_RUST_BIN      path of the binary (default research/zeroshot/zs_rust/target/release/zs_rust)
  ZS_RUST_THREADS  rayon threads (default 1; the output is byte-identical for every thread count)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CRATE = REPO / "research/zeroshot/zs_rust"
BIN_DEFAULT = CRATE / "target/release/zs_rust"


def binary() -> Path:
    return Path(os.environ.get("ZS_RUST_BIN") or BIN_DEFAULT)


def threads() -> int:
    try:
        return max(1, int(os.environ.get("ZS_RUST_THREADS", "1")))
    except ValueError:
        return 1


def available() -> bool:
    b = binary()
    return b.is_file() and os.access(b, os.X_OK)
