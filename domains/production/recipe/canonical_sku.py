"""Canonical SKU form (CONTRACTS §1).

``canonical_sku``: NFKC normalise, strip, replace internal whitespace runs with
``-``, uppercase ASCII letters. The result must match ``^[A-Z0-9][A-Z0-9._-]{0,63}$``
(charset ``[A-Z0-9._-]``, length 1-64, first character ``[A-Z0-9]``); anything
else raises :class:`SkuError` with code ``SKU_INVALID``. Uniqueness in the SKU
registry is enforced on the canonical form, so ``"abc 1"`` and ``"ABC-1"``
collide.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from contracts.production.part_recipe import SKU_PATTERN

__all__ = ["SkuError", "SKU_ERROR_CODES", "canonical_sku", "is_canonical_sku", "find_canonical_collisions"]

SKU_ERROR_CODES = frozenset({"SKU_INVALID", "FILENAME_REGEX_INVALID"})
_SKU_RE = re.compile(SKU_PATTERN)
_WS_RUN = re.compile(r"\s+")


class SkuError(ValueError):
    """A SKU or filename-pattern problem. ``code`` is one of :data:`SKU_ERROR_CODES`."""

    def __init__(self, code: str, message: str = "") -> None:
        if code not in SKU_ERROR_CODES:
            raise ValueError(f"unknown SkuError code {code!r}")
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


def _upper_ascii(text: str) -> str:
    return "".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in text)


def canonical_sku(raw: str) -> str:
    """Return the canonical SKU or raise ``SkuError("SKU_INVALID")``."""
    if not isinstance(raw, str):
        raise SkuError("SKU_INVALID", "SKU must be a string")
    text = unicodedata.normalize("NFKC", raw).strip()
    text = _upper_ascii(_WS_RUN.sub("-", text))
    if not _SKU_RE.match(text):
        raise SkuError("SKU_INVALID", f"{raw!r} is not a valid SKU (allowed [A-Z0-9._-], 1-64, starts [A-Z0-9])")
    return text


def is_canonical_sku(value: str) -> bool:
    try:
        return canonical_sku(value) == value
    except SkuError:
        return False


def find_canonical_collisions(skus: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Group raw SKUs by canonical form; return only groups with two or more entries.

    Raises ``SkuError("SKU_INVALID")`` for any invalid entry (never skipped silently).
    """
    groups: dict[str, list[str]] = {}
    for raw in skus:
        groups.setdefault(canonical_sku(raw), []).append(raw)
    return {canon: tuple(sorted(raws)) for canon, raws in sorted(groups.items()) if len(raws) > 1}
