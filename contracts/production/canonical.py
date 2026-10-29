"""Canonical JSON bytes and content hashes for production contracts.

Conventions (plan CONTRACTS §0):

- UTF-8, ``sort_keys=True``, separators ``(",", ":")``, ``ensure_ascii=False``,
  ``allow_nan=False``.
- Every float is normalised with ``round(x, 6)``; ``-0.0`` becomes ``0.0``.
  Integral floats stay floats (``1.0`` is serialised as ``1.0``), so a value's
  JSON type never changes between producer and consumer.
- NaN and +/-inf are rejected with :class:`CanonicalJsonError`; they are never
  silently coerced to ``null``.
- Hashes are lowercase sha256 hex over the canonical bytes.

Pydantic models are dumped with ``model_dump(mode="json")`` before
normalisation, so tuples become lists and enums/literals become strings.

This module is pure: no file IO, no third-party imports beyond pydantic.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

__all__ = [
    "CanonicalJsonError",
    "FLOAT_DECIMALS",
    "canonical_json_bytes",
    "normalise",
    "sha256_hex",
    "sha256_bytes",
    "strip_keys",
]

FLOAT_DECIMALS = 6


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be canonicalised (NaN/inf, unsupported type)."""


def _norm_float(value: float) -> float:
    if not math.isfinite(value):
        raise CanonicalJsonError(f"non-finite float is not canonicalisable: {value!r}")
    rounded = round(value, FLOAT_DECIMALS)
    if rounded == 0.0:
        return 0.0  # also folds -0.0
    return rounded


def normalise(obj: Any) -> Any:
    """Return a JSON-ready copy of ``obj`` with canonical float normalisation.

    Mapping keys must be strings. Sets are rejected because their order is not
    defined; callers must sort them explicitly.
    """
    if isinstance(obj, BaseModel):
        return normalise(obj.model_dump(mode="json"))
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _norm_float(obj)
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"mapping keys must be str, got {type(key).__name__}")
            out[key] = normalise(value)
        return out
    if isinstance(obj, (set, frozenset)):
        raise CanonicalJsonError("sets have no canonical order; sort them first")
    if isinstance(obj, (list, tuple)):
        return [normalise(item) for item in obj]
    # numpy scalars and similar expose .item(); avoid importing numpy here.
    item = getattr(obj, "item", None)
    if callable(item):
        return normalise(item())
    raise CanonicalJsonError(f"unsupported type for canonical JSON: {type(obj).__name__}")


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialise ``obj`` to canonical JSON bytes (see module docstring)."""
    return json.dumps(
        normalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Lowercase sha256 hex of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_hex(obj: Any) -> str:
    """Lowercase sha256 hex over :func:`canonical_json_bytes` of ``obj``."""
    return sha256_bytes(canonical_json_bytes(obj))


def strip_keys(d: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Shallow copy of ``d`` without the top-level ``keys`` (missing keys are fine)."""
    drop = set(keys)
    return {k: v for k, v in d.items() if k not in drop}
