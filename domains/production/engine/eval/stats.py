"""Small, explicit statistics for the E4 scorecards (numpy only).

Conventions:

- Every function returns ``None`` (never 0 or NaN) when its population is
  empty, so an empty population can never look like a passing value.
- Percentiles use numpy's default linear interpolation (Hyndman-Fan type 7),
  recorded in the threshold document.
- :func:`cluster_bootstrap` resamples whole clusters (part families) with
  replacement, which respects the correlation between cases of one family; the
  seed is explicit so a scorecard is reproducible byte for byte.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

__all__ = [
    "finite",
    "mean",
    "median",
    "mae",
    "percentile",
    "p95_abs",
    "rate",
    "ols_slope",
    "cluster_bootstrap",
    "round_or_none",
]

T = TypeVar("T")


def finite(values: Sequence[float | None]) -> np.ndarray:
    """Finite float array from ``values`` (``None`` / NaN / inf dropped)."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return arr[np.isfinite(arr)] if arr.size else arr


def mean(values: Sequence[float | None]) -> float | None:
    arr = finite(values)
    return float(arr.mean()) if arr.size else None


def median(values: Sequence[float | None]) -> float | None:
    arr = finite(values)
    return float(np.median(arr)) if arr.size else None


def mae(values: Sequence[float | None]) -> float | None:
    arr = finite(values)
    return float(np.abs(arr).mean()) if arr.size else None


def percentile(values: Sequence[float | None], q: float) -> float | None:
    arr = finite(values)
    return float(np.percentile(arr, q)) if arr.size else None


def p95_abs(values: Sequence[float | None]) -> float | None:
    """95th percentile of ``|value|``."""
    arr = finite(values)
    return float(np.percentile(np.abs(arr), 95.0)) if arr.size else None


def rate(numerator: int, denominator: int) -> float | None:
    """``numerator / denominator``, or ``None`` for an empty denominator."""
    if denominator <= 0:
        return None
    return numerator / denominator


def ols_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float] | None:
    """Ordinary least-squares ``(slope, intercept)`` of ``y`` on ``x``; ``None``
    when fewer than 2 points or ``x`` has no spread."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2:
        return None
    xm, ym = xa.mean(), ya.mean()
    sxx = float(((xa - xm) ** 2).sum())
    if sxx <= 1e-18:
        return None
    slope = float(((xa - xm) * (ya - ym)).sum() / sxx)
    return slope, float(ym - slope * xm)


def cluster_bootstrap(
    groups: Mapping[Hashable, T],
    statistic: Callable[[list[T]], float | None],
    *,
    resamples: int,
    seed: int,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI of ``statistic`` under a cluster bootstrap.

    ``groups`` maps a cluster (part family) to its payload (for example a tuple
    of numpy arrays). Clusters are resampled with replacement, as many as there
    are; ``statistic`` receives the list of drawn payloads (repeats included).
    Resamples where the statistic is undefined are skipped and counted.
    Returns ``{"lo", "hi", "clusters", "resamples_used", "resamples_undefined"}``
    with ``lo``/``hi`` ``None`` when fewer than 2 clusters or no defined resample.
    """
    keys = sorted(groups, key=lambda k: str(k))
    out: dict[str, Any] = {"lo": None, "hi": None, "clusters": len(keys), "resamples_used": 0,
                           "resamples_undefined": 0, "ci": ci, "seed": seed}
    if len(keys) < 2 or resamples <= 0:
        return out
    rng = np.random.default_rng(seed)
    values: list[float] = []
    undefined = 0
    for _ in range(resamples):
        draw = rng.integers(0, len(keys), size=len(keys))
        v = statistic([groups[keys[int(i)]] for i in draw])
        if v is None or not math.isfinite(v):
            undefined += 1
            continue
        values.append(float(v))
    out["resamples_used"] = len(values)
    out["resamples_undefined"] = undefined
    if values:
        alpha = (1.0 - ci) / 2.0
        arr = np.asarray(values)
        out["lo"] = float(np.percentile(arr, 100.0 * alpha))
        out["hi"] = float(np.percentile(arr, 100.0 * (1.0 - alpha)))
    return out


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    out = round(float(value), digits)
    return 0.0 if out == 0.0 else out
