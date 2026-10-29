"""Explicit compensation inputs; a mathematical draft is never a shop release.

No defaults, corpus identities, scan data, or UI approval strings enter this seam.
K is neutral-axis distance / thickness (ANSI). Angles are bend sweeps, not
the included angle printed by the upstream SheetMetal BendInfo object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

SCHEMA = "SHERMAN_COMPENSATION_PROFILE_V1"
UNFOLDER_COMMIT = "3141e205ea340d68e69ff0e257a44e27d9bcb591"
FREECAD_VERSION = "1.1.3"


class CompensationRefused(ValueError):
    pass


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompensationRefused(f"INVALID_{name.upper()}")
    try:
        value = float(value)
    except OverflowError as exc:
        raise CompensationRefused(f"INVALID_{name.upper()}") from exc
    if not math.isfinite(value):
        raise CompensationRefused(f"INVALID_{name.upper()}")
    return value


@dataclass(frozen=True)
class CompensationProfile:
    profile_id: str
    revision: str
    material: str
    process: str
    tooling: str
    thickness_mm: float
    k_factor_ansi: float
    inner_radius_min_mm: float
    inner_radius_max_mm: float
    sweep_min_deg: float
    sweep_max_deg: float
    evidence_sha256: str | None
    purpose: str

    def __post_init__(self) -> None:
        # Direct construction has the same gate as deserializing a profile.
        self._validated_values({"schema": SCHEMA, **asdict(self)})

    @classmethod
    def _validated_values(cls, source: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(source, dict) or source.get("schema") != SCHEMA:
            raise CompensationRefused("INVALID_PROFILE_SCHEMA")
        fields = set(cls.__dataclass_fields__)
        if set(source) != fields | {"schema"}:
            raise CompensationRefused("PROFILE_FIELDS_MISMATCH")
        values = {name: source[name] for name in fields}
        for name in ("profile_id", "revision", "material", "process", "tooling"):
            if not isinstance(values[name], str) or not values[name].strip() or len(values[name]) > 160:
                raise CompensationRefused(f"INVALID_{name.upper()}")
        for name in ("thickness_mm", "k_factor_ansi", "inner_radius_min_mm",
                     "inner_radius_max_mm", "sweep_min_deg", "sweep_max_deg"):
            values[name] = _number(values[name], name)
        if not 0 < values["thickness_mm"] <= 50:
            raise CompensationRefused("THICKNESS_OUT_OF_DOMAIN")
        if not 0 <= values["k_factor_ansi"] <= 1:
            raise CompensationRefused("K_FACTOR_OUT_OF_DOMAIN")
        if not 0 < values["inner_radius_min_mm"] <= values["inner_radius_max_mm"] <= 10000:
            raise CompensationRefused("RADIUS_DOMAIN_INVALID")
        if not 0 < values["sweep_min_deg"] <= values["sweep_max_deg"] < 180:
            raise CompensationRefused("SWEEP_DOMAIN_INVALID")
        if not isinstance(values["purpose"], str) or values["purpose"] not in {"synthetic_development", "shop_candidate"}:
            raise CompensationRefused("INVALID_PROFILE_PURPOSE")
        digest = values["evidence_sha256"]
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in "0123456789abcdef" for c in digest)):
            raise CompensationRefused("INVALID_PROFILE_EVIDENCE_HASH")
        # A syntactically correct evidence hash does not prove approval or calibration.
        return values

    @classmethod
    def from_dict(cls, source: dict[str, Any]) -> "CompensationProfile":
        return cls(**cls._validated_values(source))

    @property
    def sha256(self) -> str:
        raw = json.dumps({"schema": SCHEMA, **asdict(self)}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def allowance_mm(self, inner_radius_mm: float, sweep_deg: float,
                     thickness_mm: float, *, material: str, process: str,
                     tooling: str) -> float:
        radius = _number(inner_radius_mm, "inner_radius_mm")
        sweep = _number(sweep_deg, "sweep_deg")
        thickness = _number(thickness_mm, "thickness_mm")
        if (material, process, tooling) != (self.material, self.process, self.tooling):
            raise CompensationRefused("PROFILE_APPLICABILITY_MISMATCH")
        # These values are declarations, not measured manufacturing uncertainty.
        if thickness != self.thickness_mm:
            raise CompensationRefused("PROFILE_THICKNESS_MISMATCH")
        if not self.inner_radius_min_mm <= radius <= self.inner_radius_max_mm:
            raise CompensationRefused("PROFILE_RADIUS_OUT_OF_RANGE")
        if not self.sweep_min_deg <= sweep <= self.sweep_max_deg:
            raise CompensationRefused("PROFILE_SWEEP_OUT_OF_RANGE")
        return math.radians(sweep) * (radius + self.k_factor_ansi * thickness)


def from_upstream_bend(selected_radius_mm: float, signed_included_deg: float,
                       thickness_mm: float) -> dict[str, float | str]:
    """Decode the exact pinned upstream contract, not arbitrary bend conventions."""
    radius = _number(selected_radius_mm, "selected_radius_mm")
    angle = _number(signed_included_deg, "signed_included_deg")
    thickness = _number(thickness_mm, "thickness_mm")
    if thickness <= 0 or radius <= 0 or not 0 < abs(angle) < 180:
        raise CompensationRefused("UNSUPPORTED_UPSTREAM_BEND")
    direction = "UP" if angle > 0 else "DOWN"
    inner_radius = radius if direction == "UP" else radius - thickness
    if inner_radius <= 0:
        raise CompensationRefused("NONPOSITIVE_INNER_RADIUS")
    sweep = 180 - abs(angle)
    if not 0 < sweep < 180:
        raise CompensationRefused("UNREPRESENTABLE_UPSTREAM_SWEEP")
    return {"inner_radius_mm": inner_radius, "sweep_deg": sweep,
            "included_angle_deg": abs(angle), "direction_relative_to_root": direction}


def manufacturing_release_status(profile: CompensationProfile | None) -> dict[str, Any]:
    """Qualification is not implemented as an operator checkbox or a hash string."""
    reasons = ["SHOP_DIMENSIONAL_VALIDATION_MISSING", "COMPLETE_NATIVE_COVERAGE_UNQUALIFIED",
               "BEND_ID_BRIDGE_UNQUALIFIED", "CUT_EXPORT_ROUNDTRIP_UNQUALIFIED"]
    if profile is None:
        reasons.insert(0, "COMPENSATION_PROFILE_MISSING")
    else:
        reasons.insert(0, "PROFILE_EVIDENCE_NOT_INDEPENDENTLY_VERIFIED")
    return {"manufacturing_ready": False, "cut_release_allowed": False,
            "profile_sha256": profile.sha256 if profile else None, "blocking_reasons": reasons}
