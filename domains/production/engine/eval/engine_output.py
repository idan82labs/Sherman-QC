"""Load and check E3 output directories (ENGINE_IO §4.1) and their CompiledPart / recipe.

:func:`load_engine_output` never raises for a bad engine output: it returns an
:class:`EngineOutput` whose ``status`` says what is wrong, because a malformed
result is an engine failure the scorecard must count, not a harness crash:

- ``OK``: ``result.json`` validates as ``ShopfloorInspectionResultV1`` and
  ``check_result_against_compiled`` (when the CompiledPart and recipe are
  given) returns no problem;
- ``RESULT_MISSING`` / ``RESULT_INVALID`` (model validation failed) /
  ``RESULT_INCONSISTENT`` (cross-document problems, listed in ``problems``).

Geometry helpers convert between the engine frame (``cad_model_mm``) and the
raw scan frame: ``frame = raw_to_frame @ (unit_factor * raw)`` (ENGINE_IO §1).

Characteristics are selected by kind and side (:func:`row_characteristic`),
never by position in the row. :func:`measured_outer_apex` turns the engine's
measured OML virtual sharp into the visible outer-apex line at the *measured*
included angle (the G2 declared locus and the solid G4 line): E1's loci obey
``|apex - OML| = (Ri + t)(1/sin(theta/2) - 1)`` along the bisector, which sits
at ``theta/2`` from the parent flange direction, so both the offset length and
the bisector follow the measured angle instead of the nominal one.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contracts.production.compiled_part import CompiledPartV1, StageModelV1
from contracts.production.consistency import check_result_against_compiled
from contracts.production.inspection_result import CharacteristicResultV1, ShopfloorInspectionResultV1
from contracts.production.layout import ENGINE_OUTPUT_FILES
from contracts.production.part_recipe import PartRecipeV1

from .plyio import read_f32, read_ply, read_u16

__all__ = [
    "EngineOutput",
    "load_engine_output",
    "load_compiled",
    "load_recipe",
    "frame_to_raw",
    "raw_to_frame_points",
    "stage_transform",
    "apply4",
    "stage_locus",
    "decoded_stage",
    "row_characteristic",
    "measured_included_angle",
    "measured_outer_apex",
    "stage_reference_points",
    "APEX_SOURCES",
    "SEGMENT_PANEL_BASE",
    "SEGMENT_BEND_BASE",
    "SEGMENT_UNPLANNED",
]

#: Where a measured outer-apex line came from (recorded on line records and renders).
APEX_SOURCES = frozenset({
    "MEASURED_ANGLE",          # the INCLUDED_ANGLE characteristic is MEASURED
    "STATE_EVIDENCE_ANGLE",    # the bend row's state_evidence.measured_included_deg
    "OML_IS_APEX",             # E1 publishes the apex as the OML locus (hems): no offset
    "NOMINAL_OFFSET_UNSUPPORTED",  # an unsupported bend: E1's nominal offset
    "NOMINAL_ANGLE_FALLBACK",  # no measured angle: nominal offset (line metric only, flagged)
    "OML_FALLBACK_NO_ANGLE",   # no measured angle: the OML sharp itself (renders, stated on the image)
})

#: ``segment_labels.u16`` encoding (ENGINE_IO §4.1).
SEGMENT_PANEL_BASE = 1
SEGMENT_BEND_BASE = 1001
SEGMENT_UNPLANNED = 65535


@dataclass
class EngineOutput:
    out_dir: Path
    status: str
    result: ShopfloorInspectionResultV1 | None = None
    problems: list[str] = field(default_factory=list)
    error: str | None = None
    result_sha256: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def path(self, key: str) -> Path:
        return self.out_dir / ENGINE_OUTPUT_FILES[key]

    def registered_points(self) -> np.ndarray | None:
        """Registered scan points (cad_model_mm) or None when not published."""
        if self.result is None or self.result.viewer_assets.registered_scan is None:
            return None
        path = self.out_dir / self.result.viewer_assets.registered_scan
        if not path.is_file():
            return None
        return read_ply(path, with_faces=False).vertices

    def deviations(self, count: int) -> np.ndarray | None:
        if self.result is None or self.result.viewer_assets.deviations is None:
            return None
        path = self.out_dir / self.result.viewer_assets.deviations
        return read_f32(path, count) if path.is_file() else None

    def segment_labels(self, count: int) -> np.ndarray | None:
        if self.result is None or self.result.viewer_assets.segment_labels is None:
            return None
        path = self.out_dir / self.result.viewer_assets.segment_labels
        return read_u16(path, count) if path.is_file() else None


def load_compiled(path: str | os.PathLike) -> CompiledPartV1:
    """A CompiledPart (the hash is recomputed by the model validator)."""
    return CompiledPartV1.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_recipe(path: str | os.PathLike) -> PartRecipeV1:
    return PartRecipeV1.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_engine_output(
    out_dir: str | os.PathLike,
    *,
    compiled: CompiledPartV1 | None = None,
    recipe: PartRecipeV1 | None = None,
) -> EngineOutput:
    """Validate ``<out_dir>/result.json`` (module docstring)."""
    import hashlib

    from pydantic import ValidationError

    root = Path(out_dir)
    path = root / ENGINE_OUTPUT_FILES["result"]
    if not path.is_file():
        return EngineOutput(root, "RESULT_MISSING", error="result.json not written")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        result = ShopfloorInspectionResultV1.model_validate(json.loads(raw.decode("utf-8")))
    except (ValueError, ValidationError) as exc:
        msg = str(exc).splitlines()
        return EngineOutput(root, "RESULT_INVALID", error=" | ".join(msg[:6])[:2000], result_sha256=sha)
    problems: list[str] = []
    if compiled is not None and recipe is not None:
        problems = check_result_against_compiled(result, compiled, recipe)
    status = "RESULT_INCONSISTENT" if problems else "OK"
    return EngineOutput(root, status, result=result, problems=problems, result_sha256=sha)


def _mat(m: Any) -> np.ndarray:
    return np.asarray(m, dtype=np.float64).reshape(4, 4)


def apply4(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def raw_to_frame_points(result: ShopfloorInspectionResultV1, raw: np.ndarray) -> np.ndarray:
    reg = result.inspection_validity.registration
    if reg.raw_to_frame is None:
        raise ValueError("registration published no transform")
    return apply4(_mat(reg.raw_to_frame), np.asarray(raw, dtype=np.float64) * float(reg.unit_factor))


def frame_to_raw(result: ShopfloorInspectionResultV1, pts: np.ndarray) -> np.ndarray:
    """Engine-frame (cad_model_mm) points back to the raw scan frame."""
    reg = result.inspection_validity.registration
    if reg.raw_to_frame is None:
        raise ValueError("registration published no transform")
    inv = np.linalg.inv(_mat(reg.raw_to_frame))
    return apply4(inv, pts) / float(reg.unit_factor)


def decoded_stage(result: ShopfloorInspectionResultV1 | None, compiled: CompiledPartV1) -> StageModelV1:
    """The stage model the result was decoded against, else the final prefix stage S_N."""
    sid = None if result is None else result.stage_decision.stage_model_id
    if sid is not None:
        try:
            return compiled.stage(sid)
        except KeyError:
            pass
    return compiled.stage(f"S{len(compiled.bends)}")


def stage_transform(stage: StageModelV1, panel_id: str) -> np.ndarray:
    return _mat(stage.panel_transforms[panel_id])


def stage_locus(compiled: CompiledPartV1, stage: StageModelV1, bend_uid: str, locus: str) -> np.ndarray:
    """A final-pose bend locus (``loci.<locus>``) moved to ``stage``: every locus of a
    bend moves with its parent panel's transform (E1 stages.py)."""
    bend = next(b for b in compiled.bends if b.bend_uid == bend_uid)
    line = getattr(bend.loci, locus)
    pts = np.asarray([line.start_mm, line.end_mm], dtype=np.float64)
    return apply4(stage_transform(stage, bend.parent_panel_id), pts)


def row_characteristic(row: Any, kind: str, side: str | None = None) -> CharacteristicResultV1 | None:
    """The characteristic of ``kind`` (and ``side`` when given) of one bend row, selected
    by identity rather than by its position in the row."""
    for char in row.characteristics:
        if char.kind == kind and (side is None or char.side == side):
            return char
    return None


def measured_included_angle(row: Any) -> tuple[float | None, str]:
    """``(angle_deg, source)``: the MEASURED INCLUDED_ANGLE, else the row's
    ``state_evidence.measured_included_deg``, else ``(None, "NO_MEASURED_ANGLE")``."""
    angle = row_characteristic(row, "INCLUDED_ANGLE")
    if angle is not None and angle.state == "MEASURED" and angle.actual is not None:
        return float(angle.actual), "MEASURED_ANGLE"
    evidence = getattr(row.state_evidence, "measured_included_deg", None)
    if evidence is not None:
        return float(evidence), "STATE_EVIDENCE_ANGLE"
    return None, "NO_MEASURED_ANGLE"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _apex_factor(theta_deg: float) -> float:
    """``1/sin(theta/2) - 1``: OML-sharp-to-outer-apex distance per unit outer radius."""
    return 1.0 / math.sin(math.radians(theta_deg) / 2.0) - 1.0


def measured_outer_apex(compiled: CompiledPartV1, stage: StageModelV1, row: Any, *,
                        fallback: str = "OML") -> tuple[np.ndarray | None, str]:
    """``(line, source)``: the measured outer-apex line in ``cad_model_mm``.

    The engine publishes the measured OML virtual sharp (``display.measured_polyline_mm``).
    The apex lies on the bisector of the two outer flange planes, at
    ``(Ri + t)(1/sin(theta/2) - 1)`` from the sharp. E1's nominal offset is scaled
    to the measured angle and the bisector is rotated to ``theta_m/2`` from the
    parent flange direction (the engine's articulated fit walks from the datum, so
    a mis-bend moves the child flange). ``source`` is one of :data:`APEX_SOURCES`.

    Without a measured angle the result depends on ``fallback``: ``"OML"`` returns
    the OML sharp itself (renders say so on the image), ``"NOMINAL"`` E1's nominal
    offset (the line metric records it), ``"NONE"`` returns no line. Without a
    measured line the result is ``(None, "NO_MEASURED_LINE")``.
    """
    poly = row.display.measured_polyline_mm
    if poly is None:
        return None, "NO_MEASURED_LINE"
    oml = np.asarray([poly[0], poly[-1]], dtype=np.float64)
    bend = next(b for b in compiled.bends if b.bend_uid == row.bend_uid)
    oml_nom = stage_locus(compiled, stage, bend.bend_uid, "oml_virtual_sharp")
    apex_nom = stage_locus(compiled, stage, bend.bend_uid, "outer_apex")
    v_nom = apex_nom.mean(axis=0) - oml_nom.mean(axis=0)
    mag_nom = float(np.linalg.norm(v_nom))
    if not bend.supported:
        return oml + v_nom, "NOMINAL_OFFSET_UNSUPPORTED"
    if mag_nom < 1e-9:
        return oml.copy(), "OML_IS_APEX"
    theta_m, source = measured_included_angle(row)
    if theta_m is None:
        if fallback == "NOMINAL":
            return oml + v_nom, "NOMINAL_ANGLE_FALLBACK"
        if fallback == "OML":
            return oml.copy(), "OML_FALLBACK_NO_ANGLE"
        return None, "NO_MEASURED_ANGLE"
    axis = _unit(oml[1] - oml[0])
    b_nom = _unit(v_nom - axis * float(v_nom @ axis))
    parent_side = next(s for s in bend.sides if s.panel_id == bend.parent_panel_id)
    d_p = stage_transform(stage, bend.parent_panel_id)[:3, :3] @ np.asarray(parent_side.outward_dir, dtype=np.float64)
    d_p = _unit(d_p - axis * float(d_p @ axis))
    e2 = b_nom - d_p * float(b_nom @ d_p)
    if float(np.linalg.norm(e2)) < 1e-6:  # degenerate section basis: keep the nominal bisector
        b_meas = b_nom
    else:
        e2 = _unit(e2)
        half = math.radians(theta_m) / 2.0
        b_meas = math.cos(half) * d_p + math.sin(half) * e2
    f_nom, f_meas = _apex_factor(bend.nominal_included_deg), _apex_factor(theta_m)
    if f_nom > 1e-6:
        mag = mag_nom * f_meas / f_nom
    else:
        mag = (bend.inner_radius_mm + bend.thickness_mm) * f_meas
    return oml + b_meas * mag, source


def stage_reference_points(compiled: CompiledPartV1, stage: StageModelV1, *, max_points: int = 2000) -> np.ndarray:
    """Points spread over the whole stage model: every panel's outer boundary moved
    to ``stage`` (evenly subsampled to at most ``max_points``)."""
    pts = [apply4(stage_transform(stage, p.panel_id), np.asarray(p.outer_boundary_mm, dtype=np.float64))
           for p in compiled.panels if p.panel_id in stage.panel_transforms]
    allp = np.vstack(pts) if pts else np.zeros((0, 3))
    if len(allp) > max_points:
        allp = allp[np.unique(np.linspace(0, len(allp) - 1, max_points).astype(np.int64))]
    return allp
