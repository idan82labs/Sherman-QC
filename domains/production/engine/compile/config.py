"""Pinned onboarding-compile configuration (E1).

The effective configuration is hashed (canonical JSON) into
``CompiledPartV1.compiler.config_sha256`` so a compile is reproducible from
its inputs plus this record. Every key has a documented default; an unknown key
or an out-of-range value is refused (``CONFIG_INVALID``), never ignored.

The per-feature UNSUPPORTED rule thresholds are *not* configurable: they live
in :mod:`domains.production.engine.compile.unsupported` and were frozen before
any evaluation (plan §10 amendment 4d).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from contracts.production.canonical import sha256_hex

__all__ = ["CompileConfig", "load_config", "COMPILER_VERSION", "COMPILER_REVISION", "DEFAULT_SHOP_K"]

#: Version of the compile algorithm; recorded in ``compiler.version`` (inside ``compiled_part_sha256``).
#: Kept at 1.1.0 by the 2026-10-17 evening revision so the pre-A1 freeze's compile hashes stay valid:
#: that revision changes the output only of compiles with a bound flat (Method B), which no frozen
#: DEV recipe has (verified hash-identical). The 2026-10-19 revision (r3) likewise changes the output
#: only of CAD files that did not compile before (a multi-body STEP with a dominant solid, a stepped
#: bend skin, bend skins that are not coaxial): every frozen A1 / DEV-CHECK recipe and every
#: previously compiling corpus STEP recompiles hash-identical (fixes/FIXES.md). The next change that
#: alters other outputs bumps it.
COMPILER_VERSION = "e1-1.1.0"
#: Code revision within ``COMPILER_VERSION``; recorded in ``diagnostics.compiler_revision``, which is
#: outside ``compiled_part_sha256`` (ENGINE_COMPILE.md §11 lists what each revision changed).
COMPILER_REVISION = ("e1-1.1.0-r3 (2026-10-19: dominant solid of a multi-body STEP; stepped bend skins; offset "
                     "(non-coaxial) bend skins unrolled onto [Ri, Ri + t]; DEV policy 2 in dev_recipe)")
#: Shop K-factor used when no flat is bound (plan §3.1).
DEFAULT_SHOP_K = 0.33


@dataclass(frozen=True)
class CompileConfig:
    """Engineering policy of the compile. Units: mm, degrees, seconds."""

    #: Target spacing of the face-labelled stage sample clouds (contract 0.5 mm).
    sample_spacing_mm: float = 0.5
    #: Cap on points per stage cloud; the spacing grows (and is recorded) beyond it.
    max_points_per_stage: int = 1_600_000
    #: Edge junctions with outward normals closer than this are smooth (tangent/coplanar).
    smooth_dihedral_deg: float = 1.0
    #: Coaxial cylinder grouping: axis direction / axis line tolerances.
    coaxial_angle_deg: float = 0.25
    coaxial_offset_mm: float = 0.02
    #: Skin separation agreement (absolute + relative to t) before THICKNESS_INCONSISTENT.
    thickness_tol_mm: float = 0.05
    thickness_tol_rel: float = 0.02
    #: Standard uncertainty attached to the shop K-factor when it is defaulted.
    shop_k_uncertainty: float = 0.05
    #: Compiled vs recipe included angle beyond this -> ANGLE_CONVENTION_DISAGREES_WITH_RECIPE.
    angle_disagreement_deg: float = 0.5
    #: Section stations per flange/backgauge nominal along the bend's axial corridor.
    flange_stations: int = 21
    #: Reconciliation (ENGINE_IO §2.1): direction tolerance and radial margin beyond Ri + t.
    reconcile_parallel_deg: float = 2.0
    reconcile_radial_margin_mm: float = 1.0
    #: A mark whose runner-up graph bend is within this cost margin is MARK_AMBIGUOUS.
    reconcile_ambiguity_margin_mm: float = 1.0
    #: Display-mesh tessellation.
    mesh_linear_deflection_mm: float = 0.1
    mesh_angular_deflection_rad: float = 0.35
    #: Symmetry self-registration: source sample size and target decimation.
    symmetry_source_points: int = 12_000
    symmetry_target_spacing_mm: float = 1.0
    symmetry_icp_iterations: int = 12
    #: Flat QA (DevelopedLengthCheckV1 threshold) and binding acceptance.
    developed_length_threshold_mm: float = 0.5
    flat_line_offset_tol_mm: float = 1.5
    flat_line_angle_tol_deg: float = 3.0
    #: Soft wall-clock budget inside the 120 s runner budget.
    time_budget_s: float = 110.0

    def __post_init__(self) -> None:
        positive = (
            "sample_spacing_mm",
            "smooth_dihedral_deg",
            "coaxial_angle_deg",
            "coaxial_offset_mm",
            "thickness_tol_mm",
            "angle_disagreement_deg",
            "reconcile_parallel_deg",
            "reconcile_radial_margin_mm",
            "reconcile_ambiguity_margin_mm",
            "mesh_linear_deflection_mm",
            "mesh_angular_deflection_rad",
            "symmetry_target_spacing_mm",
            "developed_length_threshold_mm",
            "flat_line_offset_tol_mm",
            "flat_line_angle_tol_deg",
            "time_budget_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("thickness_tol_rel", "shop_k_uncertainty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")
        for name in ("max_points_per_stage", "flange_stations", "symmetry_source_points", "symmetry_icp_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.05 <= self.sample_spacing_mm <= 5.0:
            raise ValueError("sample_spacing_mm must lie in [0.05, 5]")
        if self.flange_stations < 3:
            raise ValueError("flange_stations must be >= 3")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        """Canonical sha256 of the effective configuration."""
        return sha256_hex({"compiler_version": COMPILER_VERSION, "config": self.to_dict()})


def load_config(path: str | Path | None) -> CompileConfig:
    """Defaults overridden by the keys of a JSON object file; raises ``ValueError``."""
    if path is None:
        return CompileConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("compile config must be a JSON object")
    known = {f.name for f in fields(CompileConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown compile config keys: {unknown}")
    return CompileConfig(**raw)
