"""Shared building blocks for the production contracts.

Every production contract model derives from :class:`ContractModel`:
pydantic v2, ``extra="forbid"``, ``frozen=True``. Models that are persisted or
exchanged as documents carry a literal ``schema`` tag; because ``schema``
shadows a ``BaseModel`` attribute, the Python attribute is ``schema_`` with the
JSON alias ``schema`` (serialisation always uses the alias).

Units are mm and degrees everywhere. Coordinates are finite 3-vectors.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ContractModel",
    "FiniteFloat",
    "NonNegFloat",
    "PosFloat",
    "CFloat",
    "CNonNegFloat",
    "CPosFloat",
    "round6",
    "Vec3",
    "Mat4",
    "Sha256Hex",
    "UtcIsoTimestamp",
    "RelPath",
    "Tolerance",
    "ToleranceRef",
    "ToleranceSource",
    "TOLERANCE_SOURCES",
    "ToleranceOrigin",
    "TOLERANCE_ORIGINS",
    "Principal",
    "FRAME_CAD_MODEL",
    "is_utc_iso",
]

#: The single declared frame of every production geometry artifact: the STEP
#: model coordinates of the recipe CAD, in millimetres (plan §10 amendment 1).
FRAME_CAD_MODEL = "cad_model_mm"


class ContractModel(BaseModel):
    """Base model: strict shape, immutable, alias-serialised."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PosFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def round6(value: float) -> float:
    """Canonical rounding (6 decimals, -0.0 -> 0.0), identical to canonical JSON."""
    out = round(float(value), 6)
    return 0.0 if out == 0.0 else out


def _round6_pos(value: float) -> float:
    out = round6(value)
    if not out > 0:
        raise ValueError("value must stay > 0 after rounding to 6 decimals")
    return out


#: Floats stored already canonically rounded, so a canonical-JSON round trip is
#: lossless and validators that compare derived values stay consistent.
CFloat = Annotated[float, Field(allow_inf_nan=False), AfterValidator(round6)]
CNonNegFloat = Annotated[float, Field(ge=0, allow_inf_nan=False), AfterValidator(round6)]
CPosFloat = Annotated[float, Field(gt=0, allow_inf_nan=False), AfterValidator(_round6_pos)]
Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Mat4 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def is_utc_iso(value: str) -> bool:
    """True when ``value`` is an ISO-8601 timestamp with a zero UTC offset."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def _check_utc_iso(value: str) -> str:
    if not is_utc_iso(value):
        raise ValueError("timestamp must be ISO-8601 with a UTC offset (Z or +00:00)")
    return value


UtcIsoTimestamp = Annotated[str, AfterValidator(_check_utc_iso)]

_REL_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._@+\-]+$")


def _check_rel_path(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError("relative path must be non-empty, POSIX and not absolute")
    for segment in value.split("/"):
        if segment in ("", ".", "..") or not _REL_PATH_SEGMENT.match(segment):
            raise ValueError(f"invalid relative path segment: {segment!r}")
    return value


#: A POSIX path relative to a declared root; no ``..``, no absolute paths.
RelPath = Annotated[str, AfterValidator(_check_rel_path)]

ToleranceSource = Literal[
    "DRAWING",
    "ISO_2768_F",
    "ISO_2768_M",
    "ISO_2768_C",
    "SHOP_STANDARD",
    "CUSTOMER_SPEC",
]
TOLERANCE_SOURCES: frozenset[str] = frozenset(ToleranceSource.__args__)  # type: ignore[attr-defined]


class Tolerance(ContractModel):
    """Bilateral tolerance band ``[-minus, +plus]`` around a nominal.

    Both sides are non-negative and the band has positive width. Asymmetric
    (including unilateral) bands are allowed.
    """

    plus: CNonNegFloat
    minus: CNonNegFloat

    @model_validator(mode="after")
    def _positive_width(self) -> "Tolerance":
        if not (self.plus + self.minus > 0):
            raise ValueError("tolerance band must have positive width (plus + minus > 0)")
        return self


ToleranceOrigin = Literal["RECIPE_DEFAULT", "RECIPE_BEND_OVERRIDE", "RECIPE_MEASUREMENT_OVERRIDE"]
TOLERANCE_ORIGINS: frozenset[str] = frozenset(ToleranceOrigin.__args__)  # type: ignore[attr-defined]


class ToleranceRef(Tolerance):
    """A resolved tolerance plus where it came from inside the recipe.

    ``standard`` is the recipe-level tolerance standard
    (``default_tolerances.source``) the value was released under.
    """

    origin: ToleranceOrigin
    standard: ToleranceSource


class Principal(ContractModel):
    """Who performed an action. ``kind`` distinguishes users from devices/services."""

    kind: Literal["user", "device", "service"]
    id: str = Field(min_length=1, max_length=128)
    role: str | None = Field(default=None, max_length=64)
