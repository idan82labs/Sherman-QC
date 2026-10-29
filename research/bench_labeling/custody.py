"""Custody guard: refuse forbidden or unrelated inputs by path/name only.

The guard never opens a file. It is applied to every scan the tool reads and
every asset it writes a label for.
"""

from __future__ import annotations

import re
from pathlib import Path


class CustodyError(PermissionError):
    """Raised when a path is forbidden for this workflow."""


_FORBIDDEN_TOKENS = (
    "holdout", "hold_out", "hold-out", "heldout", "held_out", "held-out",
    "sealed", "eho4", "one_shot", "one-shot", "oneshot", "authorization",
)
_OPAQUE_N = re.compile(r"(?<![A-Za-z0-9])N0(0[1-9]|1[0-4])(?![0-9])")
_UNRELATED_PARTS = ("rhino-preview", "82labs", "node_modules", "library")
_UNRELATED_SUFFIX = (".app",)


def check_path(path: str | Path) -> Path:
    """Return the resolved path, or raise CustodyError if it is forbidden."""
    resolved = Path(path).expanduser().resolve()
    text = resolved.as_posix()
    lower = text.casefold()
    for token in _FORBIDDEN_TOKENS:
        if token in lower:
            raise CustodyError(f"forbidden custody token {token!r} in path: {resolved.name}")
    if _OPAQUE_N.search(text):
        raise CustodyError(f"opaque N001-N014 identifier in path: {resolved.name}")
    home = Path.home().resolve()
    parts = [part.casefold() for part in resolved.parts]
    for part in parts:
        if part in _UNRELATED_PARTS or part.endswith(_UNRELATED_SUFFIX):
            raise CustodyError(f"path is outside Sherman scope ({part}): {resolved}")
    try:
        rel = resolved.relative_to(home)
        if rel.parts and rel.parts[0].casefold() in {"rhino-preview", "82labs", "library"}:
            raise CustodyError(f"path is outside Sherman scope: {resolved}")
    except ValueError:
        pass
    return resolved
