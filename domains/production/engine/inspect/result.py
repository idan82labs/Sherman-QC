"""Result assembly (E3 step 8): ``ShopfloorInspectionResultV1`` rows, stage decision and summary.

Rows are built only through the contract factories
(``CharacteristicResultV1.measured / not_measured / not_applicable``), which
take the CompiledPart nominal and derive the expected value, deviation,
guard-band decision, tolerance and requiredness; this module never computes a
decision itself and never copies a nominal into ``actual``.

Per bend (sequence order), characteristics in the fixed order FLANGE_LENGTH A,
FLANGE_LENGTH B, INCLUDED_ANGLE, BEND_LINE_POSITION, then the backgauge of the
decoded mid-sequence stage on its just-formed bend:

- NOT_YET_FORMED: every characteristic NOT_APPLICABLE (BEND_NOT_YET_FORMED, or
  NOT_INSPECTED_BY_RECIPE for a side the recipe does not inspect);
- UNKNOWN: NOT_MEASURED with the bend's reason;
- formed-like: the measurement when there is one, else NOT_MEASURED with its
  reason; nominals the compile could not give stay UNKNOWN
  (UNSUPPORTED_FEATURE / NOMINAL_UNAVAILABLE); a degenerate in-plane pose
  makes the bend-line position NOT_MEASURED(REGISTRATION_UNTRUSTED).

``stage_decision`` and ``summary`` come from the pure contract derivations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from contracts.production.common import round6
from contracts.production.inspection_result import (
    BendDisplayV1,
    BendRowV1,
    BendStateEvidenceV1,
    CharacteristicEvidenceV1,
    CharacteristicResultV1,
    CoverageV1,
    EngineInfoV1,
    GeneralMeasurementRowV1,
    InspectionValidityV1,
    RecipeRefV1,
    RegistrationV1,
    ResultContextV1,
    ScanInfoV1,
    ShopfloorInspectionResultV1,
    StageDecisionV1,
    StationStatsV1,
    UnplannedCreaseV1,
    ViewerAssetsV1,
    derive_stage_decision,
    derive_summary,
)
from domains.production.engine.inspect.metrology import Measurement
from domains.production.engine.inspect.part import PartView

__all__ = ["BendOutcome", "ResultInputs", "build_result", "utc_now"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _v3(p) -> tuple[float, float, float]:
    a = np.asarray(p, dtype=np.float64)
    return (round6(float(a[0])), round6(float(a[1])), round6(float(a[2])))


@dataclass
class BendOutcome:
    """Everything the row builder needs for one bend."""

    state: str
    state_reasons: tuple[str, ...] = ()
    measured_included_deg: float | None = None
    formed_fraction: float | None = None
    prior: float | None = None
    support_parent: int | None = None
    support_child: int | None = None
    coverage: float | None = None
    #: characteristic id -> Measurement (MEASURED when value is not None)
    measurements: dict[str, Measurement] = field(default_factory=dict)
    #: characteristic id -> UNKNOWN reason forced for that characteristic
    forced_unknown: dict[str, str] = field(default_factory=dict)
    nominal_line: tuple[np.ndarray, np.ndarray] | None = None
    measured_line: tuple[np.ndarray, np.ndarray] | None = None
    side_anchor: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    backgauge_cid: str | None = None
    #: the decoded stage's backgauge does not describe this formed set (NOT_APPLICABLE_AT_STAGE)
    backgauge_na: bool = False
    #: flange characteristic ids bounded by an adjacent bend that is NOT_YET_FORMED: the length is undefined at
    #: this stage (NOT_APPLICABLE_AT_STAGE, contract amendment ENGINE_IO-A1, owner decision 2026-10-19)
    stage_na: set[str] = field(default_factory=set)


@dataclass
class ResultInputs:
    part: PartView
    recipe: Any
    scan_sha256: str
    scan_format: str
    vertex_count: int
    face_count: int
    context: ResultContextV1
    engine: EngineInfoV1
    validity_state: str
    validity_reasons: tuple[str, ...]
    registration: RegistrationV1
    coverage: CoverageV1
    declared_stage: int | None
    declared_source: str | None
    stage_model_id: str | None
    stage_detail: str
    bends: list[BendOutcome]
    viewer: ViewerAssetsV1
    creases: tuple[UnplannedCreaseV1, ...] = ()
    warnings: tuple[str, ...] = ()
    created_at: str | None = None
    general_measurements: dict[str, Measurement] = field(default_factory=dict)
    general_anchors: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    general_not_applicable: set[str] = field(default_factory=set)


def _evidence(m: Measurement) -> CharacteristicEvidenceV1:
    notes = tuple(n for n in m.notes if n)[:8]
    comp = tuple(f"{k}={v:.4g}" for k, v in sorted(m.components.items()) if isinstance(v, (int, float)))
    return CharacteristicEvidenceV1(
        method=(m.method or "UNSPECIFIED")[:96],
        support_points=int(m.support) if m.support else None,
        coverage_ratio=None if m.coverage is None else float(min(1.0, max(0.0, m.coverage))),
        fit_rms_mm=None if m.fit_rms is None else float(max(0.0, m.fit_rms)),
        notes=(notes + comp)[:12],
    )


def _row_char(nominal, outcome: BendOutcome, *, expected_source: str) -> CharacteristicResultV1:
    cid = nominal.characteristic_id
    if nominal.recipe_requirement == "NOT_INSPECTED":
        return CharacteristicResultV1.not_applicable(nominal, reason="NOT_INSPECTED_BY_RECIPE",
                                                     expected_source=expected_source)
    if outcome.state == "NOT_YET_FORMED":
        return CharacteristicResultV1.not_applicable(nominal, reason="BEND_NOT_YET_FORMED", expected_source=expected_source)
    if outcome.state == "UNKNOWN":
        reason = outcome.state_reasons[0] if outcome.state_reasons else "BEND_STATE_UNKNOWN"
        return CharacteristicResultV1.not_measured(nominal, reason=reason, expected_source=expected_source)
    if cid in outcome.stage_na:
        return CharacteristicResultV1.not_applicable(nominal, reason="NOT_APPLICABLE_AT_STAGE",
                                                     expected_source=expected_source)
    if cid in outcome.forced_unknown:
        return CharacteristicResultV1.not_measured(nominal, reason=outcome.forced_unknown[cid],
                                                   expected_source=expected_source)
    if nominal.nominal is None:
        reason = nominal.unavailable_reason or "NOMINAL_UNAVAILABLE"
        return CharacteristicResultV1.not_measured(nominal, reason=reason, expected_source=expected_source)
    m = outcome.measurements.get(cid)
    if m is None or m.value is None or m.u is None:
        reason = (m.reason if m is not None else None) or "OCCLUDED"
        ev = _evidence(m) if m is not None and m.method else None
        return CharacteristicResultV1.not_measured(nominal, reason=reason, expected_source=expected_source, evidence=ev)
    u = max(float(m.u), 1e-6)
    stations = None
    if m.count and m.vmin is not None and m.vmax is not None and m.median is not None:
        lo, hi = round6(m.vmin), round6(m.vmax)
        med = min(max(round6(m.median), lo), hi)
        stations = StationStatsV1(count=int(m.count), median=med, min=lo, max=hi)
    return CharacteristicResultV1.measured(nominal, actual=float(m.value), uncertainty=u,
                                           expected_source=expected_source, stations=stations, evidence=_evidence(m))


def _bend_row(part: PartView, j: int, outcome: BendOutcome, invalid_reason: str | None) -> BendRowV1:
    bend = part.bends[j]
    uid = bend.uid
    chars = []
    for kind, side in (("FLANGE_LENGTH", "A"), ("FLANGE_LENGTH", "B"), ("INCLUDED_ANGLE", None),
                       ("BEND_LINE_POSITION", None)):
        cid = f"{uid}/{kind}" + (f"/{side}" if side else "")
        nominal = part.nominals[cid]
        src = "NOMINAL_LOCUS_REFERENCE" if kind == "BEND_LINE_POSITION" else "COMPILED_NOMINAL"
        if invalid_reason is not None:
            if nominal.recipe_requirement == "NOT_INSPECTED":
                chars.append(CharacteristicResultV1.not_applicable(nominal, reason="NOT_INSPECTED_BY_RECIPE",
                                                                   expected_source=src))
            else:
                chars.append(CharacteristicResultV1.not_measured(nominal, reason=invalid_reason, expected_source=src))
            continue
        chars.append(_row_char(nominal, outcome, expected_source=src))
    if outcome.backgauge_cid is not None and invalid_reason is None:
        nominal = part.nominals[outcome.backgauge_cid]
        if outcome.state == "NOT_YET_FORMED":
            chars.append(CharacteristicResultV1.not_applicable(nominal, reason="BEND_NOT_YET_FORMED",
                                                               expected_source="STAGE_MODEL_NOMINAL"))
        elif outcome.backgauge_na:
            chars.append(CharacteristicResultV1.not_applicable(nominal, reason="NOT_APPLICABLE_AT_STAGE",
                                                               expected_source="STAGE_MODEL_NOMINAL"))
        else:
            chars.append(_row_char(nominal, outcome, expected_source="STAGE_MODEL_NOMINAL"))
    state = outcome.state if invalid_reason is None else "UNKNOWN"
    reasons = outcome.state_reasons if invalid_reason is None else (invalid_reason,)
    # a partial bend whose angle passes the guard band is, by definition, formed
    if state == "PARTIALLY_FORMED" and chars[2].decision == "PASS":
        state = "FORMED"
    if state in ("FORMED", "NOT_YET_FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"):
        reasons = ()
    nominal_line = outcome.nominal_line
    if nominal_line is None:
        line = part.prefix_stage(part.n).bend_lines[uid]
        nominal_line = (np.asarray(line.start_mm), np.asarray(line.end_mm))
    measured = None
    if chars[3].state == "MEASURED" and outcome.measured_line is not None:
        measured = (_v3(outcome.measured_line[0]), _v3(outcome.measured_line[1]))
    elif chars[3].state == "MEASURED":
        # contract: a measured position always carries its line
        raise ValueError(f"bend {uid}: BEND_LINE_POSITION measured without a measured line")
    anchor = 0.5 * (np.asarray(nominal_line[0]) + np.asarray(nominal_line[1]))
    if measured is not None:
        anchor = 0.5 * (np.asarray(measured[0]) + np.asarray(measured[1]))
    evidence = BendStateEvidenceV1(
        nominal_included_deg=float(min(180.0, max(1e-6, bend.model.nominal_included_deg))),
        measured_included_deg=None if outcome.measured_included_deg is None or invalid_reason is not None
        else float(min(180.0, max(1e-3, outcome.measured_included_deg))),
        formed_fraction=None if outcome.formed_fraction is None or invalid_reason is not None
        else float(outcome.formed_fraction),
        parent_support_points=outcome.support_parent if invalid_reason is None else None,
        child_support_points=outcome.support_child if invalid_reason is None else None,
        coverage_ratio=None if outcome.coverage is None or invalid_reason is not None
        else float(min(1.0, max(0.0, outcome.coverage))),
        prior_formed_probability=None if outcome.prior is None else float(min(1.0, max(0.0, outcome.prior))),
    )
    display = BendDisplayV1(
        nominal_polyline_mm=(_v3(nominal_line[0]), _v3(nominal_line[1])),
        measured_polyline_mm=measured,
        label_anchor_mm=_v3(anchor),
        side_a_anchor_mm=_v3(outcome.side_anchor["A"]) if "A" in outcome.side_anchor and invalid_reason is None else None,
        side_b_anchor_mm=_v3(outcome.side_anchor["B"]) if "B" in outcome.side_anchor and invalid_reason is None else None,
    )
    diag = {k: float(v) for k, v in outcome.diagnostics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
    if invalid_reason is not None:
        diag = {}
    return BendRowV1(bend_uid=uid, label=bend.label, sequence_index=bend.seq,
                     machine_move_index=bend.model.machine_move_index, state=state, state_reasons=tuple(reasons),
                     state_evidence=evidence, characteristics=tuple(chars), display=display, diagnostics=diag,
                     notes=tuple(outcome.notes)[:16] if invalid_reason is None else ())


def _general_rows(inp: ResultInputs, invalid_reason: str | None) -> tuple[GeneralMeasurementRowV1, ...]:
    rows = []
    for gm in inp.recipe.general_measurements:
        cid = f"gm/{gm.measurement_id}"
        nominal = inp.part.nominals[cid]
        src = "RECIPE_TARGET" if gm.nominal_source == "OPERATOR_TARGET" else "CAD_DISTANCE"
        ms = me = None
        if invalid_reason is not None:
            char = CharacteristicResultV1.not_measured(nominal, reason=invalid_reason, expected_source=src)
        elif cid in inp.general_not_applicable:
            char = CharacteristicResultV1.not_applicable(nominal, reason="NOT_APPLICABLE_AT_STAGE", expected_source=src)
        else:
            m = inp.general_measurements.get(cid)
            if m is None or m.value is None or m.u is None:
                char = CharacteristicResultV1.not_measured(nominal, reason=(m.reason if m else None) or "UNSUPPORTED_FEATURE",
                                                           expected_source=src)
            else:
                char = CharacteristicResultV1.measured(nominal, actual=float(m.value), uncertainty=max(float(m.u), 1e-6),
                                                       expected_source=src, evidence=_evidence(m))
                anchors = inp.general_anchors.get(cid)
                if anchors is not None:
                    ms, me = _v3(anchors[0]), _v3(anchors[1])
                else:  # pragma: no cover - a measured general measurement always records its anchors
                    raise ValueError("measured general measurement without anchors")
        rows.append(GeneralMeasurementRowV1(measurement_id=gm.measurement_id, label=gm.label, characteristic=char,
                                            nominal_start_mm=_v3(gm.point_start_mm), nominal_end_mm=_v3(gm.point_end_mm),
                                            measured_start_mm=ms, measured_end_mm=me))
    return tuple(rows)


def build_result(inp: ResultInputs) -> ShopfloorInspectionResultV1:
    """Assemble and validate the result (the contract validators run on construction)."""
    part, recipe = inp.part, inp.recipe
    invalid_reason = None
    if inp.validity_state != "VALID":
        invalid_reason = "TIMEOUT" if inp.validity_state == "TIMEOUT" else "INVALID_INSPECTION"
    rows = tuple(_bend_row(part, j, inp.bends[j], invalid_reason) for j in range(part.n))
    gms = _general_rows(inp, invalid_reason)
    creases = inp.creases if invalid_reason is None else ()
    stage = derive_stage_decision(inspection_stage=recipe.inspection_stage, declared_stage=inp.declared_stage,
                                  bends=rows, pose_ambiguous=inp.registration.pose_ambiguous,
                                  validity_state=inp.validity_state)
    stage_model_id = None if invalid_reason is not None else inp.stage_model_id
    decision = StageDecisionV1(inspection_stage=recipe.inspection_stage, declared_stage=inp.declared_stage,
                               declared_stage_source=inp.declared_source, stage_model_id=stage_model_id,
                               detail=inp.stage_detail[:1000], **stage)
    summary = derive_summary(validity_state=inp.validity_state, bends=rows, general_measurements=gms,
                             stage_decision=decision.decision, stage_status=decision.status,
                             unplanned_creases=len(creases))
    return ShopfloorInspectionResultV1(
        created_at=inp.created_at or utc_now(),
        context=inp.context,
        sku=recipe.sku,
        part_name=recipe.part_name,
        recipe=RecipeRefV1(recipe_id=recipe.recipe_id, version=recipe.recipe_version,
                           recipe_sha256=recipe.recipe_sha256, content_sha256=recipe.content_sha256),
        scan=ScanInfoV1(sha256=inp.scan_sha256, format=inp.scan_format, vertex_count=inp.vertex_count,
                        face_count=inp.face_count),
        engine=inp.engine,
        inspection_validity=InspectionValidityV1(state=inp.validity_state, reasons=inp.validity_reasons,
                                                 registration=inp.registration, coverage=inp.coverage),
        stage_decision=decision,
        summary=summary,
        bends=rows,
        general_measurements=gms,
        unplanned_creases=creases,
        viewer_assets=inp.viewer,
        warnings=tuple(inp.warnings)[:64],
    )
