"""Catalog-bound operator setup drafts; no scan matching or QC decision logic."""
from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from apps.api.legacy_context import legacy
from backend.vendor_flange_preview import FlangePreviewUnavailable, build_flange_previews
from backend.vendor_unfold_preview import unavailable_unfold_preview
from backend.vendor_unfold_worker import build_unfold_preview_bounded
from backend.vendor_unfold_worker_v2 import build_unfold_preview_bounded as build_unfold_preview_v2_bounded
from backend.vendor_unfold_worker_v3 import build_unfold_preview_bounded as build_unfold_preview_v3_bounded
from backend.vendor_unfold_contract_v3 import unavailable as unavailable_unfold_preview_v4
from backend.vendor_unfold_worker_v4 import build_unfold_preview_bounded as build_unfold_preview_v4_bounded
from backend.vendor_unfold_contract_v4 import unavailable as unavailable_unfold_preview_v5


class BendMark(BaseModel):
    feature_id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=100)
    source: Literal["cad_candidate", "manual"]
    status: Literal["active", "removed"] = "active"
    review_status: Literal["pending", "confirmed"] = "pending"
    issue: Optional[Literal["missing", "wrong_location", "not_a_bend"]] = None
    corrects_feature_id: Optional[str] = None
    note: str = Field(default="", max_length=500)
    line_start: Optional[list[float]] = None
    line_end: Optional[list[float]] = None
    target_angle: Optional[float] = None
    target_radius: Optional[float] = None


class MatchingSearchSettings(BaseModel):
    search_radius_mm: Optional[float] = Field(default=None, gt=0, le=1000)
    search_angle_deg: Optional[float] = Field(default=None, gt=0, le=180)


class GeneralMeasurement(BaseModel):
    """Operator-authored straight-line CAD reference, not a scan measurement."""

    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=100)
    kind: Literal["bend_pair", "points"]
    bend_ids: list[str] = Field(default_factory=list, max_length=2)
    point_start: list[float] = Field(min_length=3, max_length=3)
    point_end: list[float] = Field(min_length=3, max_length=3)
    cad_distance: float = Field(gt=0, allow_inf_nan=False)
    target_mm: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    note: str = Field(default="", max_length=500)


class VendorSetupDraft(BaseModel):
    schema_version: Literal[1] = 1
    cad_file_hash: str = Field(min_length=1, max_length=128)
    cad_uploaded_at: Optional[str] = None
    part_revision: Optional[str] = None
    drawing_file_hash: Optional[str] = None
    candidate_generation: str = Field(min_length=64, max_length=64)
    expected_draft_revision: int = Field(default=0, ge=0)
    marks: list[BendMark] = Field(default_factory=list, max_length=200)
    sequence: list[str] = Field(default_factory=list, max_length=200)
    machine_moves: list[list[str]] = Field(default_factory=list, max_length=200)
    general_measurements: list[GeneralMeasurement] = Field(default_factory=list, max_length=100)
    machine_moves_explicit: bool = False
    operator_sequence_confirmed: bool = False
    completed_steps: int = Field(default=0, ge=0)
    matching_search: MatchingSearchSettings = Field(default_factory=MatchingSearchSettings)


def _line(raw: Optional[str]) -> Optional[list[float]]:
    if not raw:
        return None
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if len(value) == 3 and all(math.isfinite(float(v)) for v in value):
            return [float(v) for v in value]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def _validate_draft(draft: VendorSetupDraft, candidates: dict[str, BendMark]) -> None:
    ids = [mark.feature_id for mark in draft.marks]
    if len(ids) != len(set(ids)):
        raise HTTPException(422, "Bend feature IDs must be unique")
    by_id = {mark.feature_id: mark for mark in draft.marks}
    if not set(candidates).issubset(by_id):
        raise HTTPException(422, "Every CAD candidate must remain represented")
    for mark in draft.marks:
        points = (mark.line_start, mark.line_end)
        if (points[0] is None) != (points[1] is None):
            raise HTTPException(422, "A line needs both endpoints")
        if points[0] is not None:
            if any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in points):
                raise HTTPException(422, "Line endpoints must be finite 3D coordinates")
            if points[0] == points[1]:
                raise HTTPException(422, "Line endpoints must differ")
        if mark.source == "cad_candidate":
            original = candidates.get(mark.feature_id)
            if original is None:
                raise HTTPException(422, "Unknown CAD candidate feature ID")
            if (mark.label != original.label or mark.line_start != original.line_start
                    or mark.line_end != original.line_end
                    or mark.target_angle != original.target_angle
                    or mark.target_radius != original.target_radius):
                raise HTTPException(422, "CAD candidate geometry is immutable; add a manual correction")
            if mark.corrects_feature_id is not None:
                raise HTTPException(422, "CAD candidates cannot correct other marks")
        elif not mark.feature_id.startswith("manual:") or points[0] is None:
            raise HTTPException(422, "Manual marks need a manual ID and two endpoints")
        if mark.source == "manual" and mark.issue == "wrong_location":
            corrected = by_id.get(mark.corrects_feature_id or "")
            if (corrected is None or corrected.source != "cad_candidate"
                    or corrected.status != "removed" or corrected.issue != "wrong_location"):
                raise HTTPException(422, "Location corrections must identify a removed CAD candidate")
        if mark.status == "removed" and not mark.issue:
            raise HTTPException(422, "Removed marks require an issue reason")
        if mark.status == "removed" and mark.review_status != "pending":
            raise HTTPException(422, "Removed marks cannot remain review-confirmed")
        if mark.review_status == "confirmed" and points[0] is None:
            raise HTTPException(422, "Only located bend marks can be review-confirmed")
        if mark.source == "manual" and not mark.issue:
            raise HTTPException(422, "Manual corrections require an issue reason")
    # This association is operator-authored, not proof that an anchor belongs
    # to a bend, flange, or mesh surface. Distances are straight-line CAD
    # coordinate distances (assumed mm), with no scan or conformance claim.
    measurements = getattr(draft, "general_measurements", [])
    if len(measurements) > 100:
        raise HTTPException(422, "At most 100 general measurements are allowed")
    measurement_ids = set()
    for measurement in measurements:
        if (not isinstance(measurement.id, str) or not measurement.id.strip()
                or len(measurement.id) > 100):
            raise HTTPException(422, "Measurement IDs must be nonempty and at most 100 characters")
        if measurement.id in measurement_ids:
            raise HTTPException(422, "Measurement IDs must be unique")
        measurement_ids.add(measurement.id)
        if (not isinstance(measurement.label, str) or not measurement.label.strip()
                or len(measurement.label) > 100):
            raise HTTPException(422, "Measurement labels must be nonempty and at most 100 characters")
        if not isinstance(measurement.note, str) or len(measurement.note) > 500:
            raise HTTPException(422, "Measurement notes must be at most 500 characters")
        anchors = (measurement.point_start, measurement.point_end)
        try:
            anchors_valid = all(len(point) == 3 and all(math.isfinite(v) for v in point)
                                for point in anchors)
        except (TypeError, ValueError):
            anchors_valid = False
        if not anchors_valid:
            raise HTTPException(422, "Measurement anchors must be finite 3D coordinates")
        if anchors[0] == anchors[1]:
            raise HTTPException(422, "Measurement anchors must differ")
        anchor_distance = math.dist(*anchors)
        if (not math.isfinite(anchor_distance) or anchor_distance <= 0
                or not math.isfinite(measurement.cad_distance) or measurement.cad_distance <= 0):
            raise HTTPException(422, "CAD reference distance must be finite and positive")
        # Numerical serialization/round-off allowance, not a manufacturing tolerance.
        if not math.isclose(measurement.cad_distance, anchor_distance, rel_tol=1e-9, abs_tol=1e-12):
            raise HTTPException(422, "CAD reference distance must match the straight-line anchors")
        if (measurement.target_mm is not None
                and (not math.isfinite(measurement.target_mm) or measurement.target_mm <= 0)):
            raise HTTPException(422, "Measurement target must be finite and positive when provided")
        if measurement.kind == "bend_pair":
            if len(measurement.bend_ids) != 2 or len(set(measurement.bend_ids)) != 2:
                raise HTTPException(422, "Bend-pair measurements need two distinct bend IDs")
            for bend_id in measurement.bend_ids:
                linked = by_id.get(bend_id)
                if (linked is None or linked.status != "active"
                        or linked.line_start is None or linked.line_end is None):
                    raise HTTPException(422, "Bend-pair measurements must reference active located bends")
        elif measurement.kind == "points":
            if measurement.bend_ids:
                raise HTTPException(422, "Point measurements cannot reference bend IDs")
        else:
            raise HTTPException(422, "Unknown general measurement kind")
    active = {key for key, mark in by_id.items() if mark.status == "active"}
    if len(draft.sequence) != len(set(draft.sequence)) or set(draft.sequence) != active:
        raise HTTPException(422, "Sequence must contain each active mark exactly once")
    # Moves are optional, explicitly tagged, ordered contiguous subsets.
    # Bends outside a move retain their order but have no machine-phase claim.
    move_cursor = 0
    for move in draft.machine_moves:
        if not move:
            raise HTTPException(422, "Machine moves must be ordered contiguous bend groups")
        try:
            start = draft.sequence.index(move[0], move_cursor)
        except ValueError:
            raise HTTPException(422, "Machine moves must be ordered contiguous bend groups") from None
        if draft.sequence[start:start + len(move)] != move:
            raise HTTPException(422, "Machine moves must be ordered contiguous bend groups")
        move_cursor = start + len(move)
    if active and not draft.operator_sequence_confirmed:
        raise HTTPException(422, "Operator must confirm bending order")
    if draft.completed_steps > len(draft.sequence):
        raise HTTPException(422, "Completed steps exceed the planned sequence")


def _natural_label_key(label: str) -> list:
    """B2 before B10: the catalog returns bend specs ORDER BY bend_id (text), which put B10-B12
    between B1 and B2 in the operator's initial list on parts with ten or more bends."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(label))]


async def get_vendor_setup(part_id: str):
    catalog = legacy.get_catalog()
    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")
    # Explicit development path; activation is not production qualification.
    runtime = os.environ.get("SHERMAN_VENDOR_UNFOLD_RUNTIME")
    controlled_unfold = bool(runtime)
    unavailable_factory = (unavailable_unfold_preview_v5 if runtime == "v4-development"
                           else unavailable_unfold_preview_v4 if runtime == "v3-development" else unavailable_unfold_preview)
    unfold_worker = (build_unfold_preview_v4_bounded if runtime == "v4-development"
                     else build_unfold_preview_v3_bounded if runtime == "v3-development"
                     else build_unfold_preview_v2_bounded if runtime == "v2-development"
                     else build_unfold_preview_bounded if not runtime else None)
    cad_path, cad_hash = part.cad_file_path, part.cad_file_hash
    part_snapshot = part.to_dict()
    generation = catalog.candidate_generation(part_id)
    # All response state belongs to this pre-await snapshot. A concurrent
    # upload/save must not pair a newer operator plan with the old CAD rows.
    draft_snapshot = copy.deepcopy(catalog.get_vendor_setup_draft(part_id))
    candidates = []
    for spec in catalog.get_bend_specs_for_part(part_id):
        candidates.append({
            "feature_id": spec.id, "label": spec.bend_id,
            "source": "cad_candidate", "status": "active", "issue": None,
            "review_status": "pending",
            "note": "", "line_start": _line(spec.bend_line_start),
            "line_end": _line(spec.bend_line_end),
            "target_angle": spec.target_angle, "target_radius": spec.target_radius,
        })
    # Natural label order for the initial (unconfirmed) list; stable, never an operator order.
    candidates.sort(key=lambda c: _natural_label_key(c["label"]))
    flange_surfaces = []
    flange_surface_status = "CAD_OR_PREVIEW_UNAVAILABLE"
    if controlled_unfold:
        # The legacy folded-flange aid executes native work in an unbounded
        # thread. Never fall back to that path from the controlled runtime.
        flange_surface_status = "CONTROLLED_PATH_FOLDED_FLANGE_NOT_QUALIFIED"
    elif cad_path and cad_hash and part.preview_mesh_path:
        try:
            flange_surfaces = await run_in_threadpool(
                build_flange_previews, cad_path, cad_hash, candidates,
            )
            flange_surface_status = "AVAILABLE" if flange_surfaces else "NO_UNAMBIGUOUS_ASSOCIATION"
        except FlangePreviewUnavailable as exc:
            flange_surface_status = str(exc)
        except Exception:
            # A visual aid may fail independently of the saved candidate list.
            flange_surface_status = "FLANGE_PREVIEW_UNAVAILABLE"
    unfold_preview = unavailable_factory("CAD_SOURCE_UNAVAILABLE" if unfold_worker else "UNFOLD_RUNTIME_UNRECOGNIZED")
    if cad_path and cad_hash and unfold_worker:
        try:
            unfold_preview = await run_in_threadpool(
                unfold_worker,
                cad_path, cad_hash, candidates,
            )
        except Exception:
            unfold_preview = unavailable_factory("UNFOLD_WORKER_UNAVAILABLE")
    current_part = catalog.get_part(part_id)
    if (catalog.candidate_generation(part_id) != generation or not current_part
            or current_part.cad_file_hash != cad_hash):
        unfold_preview = unavailable_factory("CAD_CANDIDATES_CHANGED_DURING_PREVIEW")
        flange_surfaces = []
        flange_surface_status = "CAD_CANDIDATES_CHANGED_DURING_PREVIEW"
    return {
        "part": part_snapshot, "candidates": candidates,
        "candidate_generation": generation,
        "draft": draft_snapshot,
        "flange_surfaces": flange_surfaces,
        "flange_surface_status": flange_surface_status,
        "unfold_preview": unfold_preview,
    }


async def save_vendor_setup(part_id: str, draft: VendorSetupDraft):
    catalog = legacy.get_catalog()
    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")
    candidates = {
        s.id: BendMark(feature_id=s.id, label=s.bend_id,
                       source="cad_candidate", line_start=_line(s.bend_line_start),
                       line_end=_line(s.bend_line_end),
                       target_angle=s.target_angle, target_radius=s.target_radius)
        for s in catalog.get_bend_specs_for_part(part_id)
    }
    _validate_draft(draft, candidates)
    try:
        saved = catalog.save_vendor_setup_draft(part_id, draft.model_dump())
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"draft": saved}
