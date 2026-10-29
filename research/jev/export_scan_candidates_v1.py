"""Exporters for scan-side bend candidates (family B, C10 v1).

Three sources, all from explicit inputs (custody: :func:`.export_onboarding_v1.
check_input_path`):

``SCAN_E3``
    The production engine's ``<job>/engine/result.json``
    (``ShopfloorInspectionResultV1``). Candidates are the planned bend rows (per-bend
    state evidence and characteristic evidence) and the unplanned creases (the
    residual lane). Geometry is in ``cad_model_mm``.
``SCAN_LEGACY_REPORT``
    The legacy worker's ``bend_inspection_report.json`` (``matches`` with a
    detection, and ``unmatched_detections``). Geometry is in ``reference_mesh_raw``.
``SCAN_F1_POOL``
    Legacy F1 candidate-pool rows: the selector-v2 descriptor names
    (``research/f1_multifamily_selector_v2/model.py`` ``BASE_FEATURE_NAMES``, copied in
    :data:`F1_BASE_FEATURE_NAMES` and checked by a parity test) plus line geometry.
    Each row must carry ``group_sku``, ``case_key`` and ``coordinate_frame``.

State blocks. ``candidate`` holds only what the detector or engine observed and
what exists for every candidate of that source, with the same key set for every
candidate of that source. Everything whose value **or presence** depends on the
CAD or on the binding to a recipe bend goes into ``cad_relation`` (nominal
angles, deviations, assignment kind and scores, pipeline observability around a
bound CAD bend, the nearest-planned-bend relation, and for E3 every per-bend
engine field that only planned rows carry: support counts, coverage, fit RMS,
characteristic states, bend state and reasons, the crease reason), so the CAD-free
ablation is exactly a drop of that block. The pipeline's own observability/bend
state is a feature there, never a label.

Non-candidates are skipped and counted, never exported with CAD geometry:

- E3 planned bends without a measured (scan-fitted) polyline
  (``e3_bends_without_measured_line``);
- legacy matches without a detected line (``legacy_matches_without_detection``) and
  legacy *placeholder* matches (``legacy_matches_cad_placeholder``): those with
  ``assignment_source == CAD_LOCAL_NEIGHBORHOOD`` or
  ``detected_line_geometry_source == CAD_REFERENCE_PLACEHOLDER``, whose
  "detected" line is the CAD line copied into the legacy container
  (``backend/feature_detection/bend_detector.py`` ``to_dict``), and any match whose
  detected line equals its CAD line (older reports without the flag).

Labels (see :mod:`.labels_v1`):

- human marker reviews, only when supplied explicitly through ``labels_root``
  (``<labels_root>/<job dir name>/bend_instance_labels.json`` in frame
  ``reference_mesh_raw`` for legacy reports, or ``jev_candidate_labels.json``,
  a :data:`.labels_v1.CANDIDATE_LABEL_DOC` in the source's frame);
- gold bend lines for F1 pool rows (``jev_gold_lines_v1``; matching thresholds
  of ``scripts/evaluate_f1_against_manual_gold.py``: 5 deg, 4 mm, 0.6/0.6 overlap);
- weak labels in their own column: bound + FORMED + measured -> 1, unbound and
  more than 10 mm from every recipe bend -> 0.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.jev.export_onboarding_v1 import (
    REPO_ROOT,
    ExportError,
    ExportResult,
    canonical_sha256,
    check_input_dir,
    check_input_path,
    load_authorization,
    load_json_records,
    validate_provenance_flags,
    write_export,
)
from research.jev.labels_v1 import (
    LabelError,
    LabelRow,
    make_label,
    parse_candidate_label_doc,
    parse_marker_reviews,
)
from research.jev.state_v1 import (
    CandidateRecord,
    Line,
    build_state,
    enum_token,
    line_from_endpoints,
    line_from_polyline,
    line_pair_metrics,
    neighbour_summary,
    record_id_for,
    relation_to_reference,
    segment_distance_mm,
)

__all__ = [
    "F1_BASE_FEATURE_NAMES",
    "F1_STATE_KEYS",
    "GOLD_SCHEMA",
    "E3_FRAME",
    "LEGACY_FRAME",
    "WEAK_FAR_MM",
    "ScanExport",
    "e3_records",
    "legacy_report_records",
    "f1_pool_records",
    "export_scan",
    "synthetic_e3_result",
    "synthetic_legacy_report",
    "synthetic_f1_pool",
]

E3_FRAME = "cad_model_mm"
LEGACY_FRAME = "reference_mesh_raw"
E3_RESULT_SCHEMA = "sherman_shopfloor_inspection_result_v1"
GOLD_SCHEMA = "jev_gold_lines_v1"
#: An unbound candidate farther than this from every recipe bend gets weak label 0.
WEAK_FAR_MM = 10.0
#: Legacy statuses that mean "measured" (a value was produced).
LEGACY_MEASURED_STATUSES = frozenset({"PASS", "FAIL", "WARNING"})
#: Gold matching thresholds (scripts/evaluate_f1_against_manual_gold.py Thresholds).
GOLD_ANGLE_DEG = 5.0
GOLD_OFFSET_MM = 4.0
GOLD_OVERLAP = 0.60

#: Copy of research/f1_multifamily_selector_v2/model.py BASE_FEATURE_NAMES (parity-tested).
F1_BASE_FEATURE_NAMES: tuple[str, ...] = (
    "span_mm", "radius_mm", "transverse_offset_mm", "vote_count", "support_count",
    "normal_spread_deg", "score", "consensus_count", "radius_consensus_count",
    "finite_support_fraction", "finite_support_longest_run_fraction", "endpoint_support_fraction",
    "median_line_distance_mm", "p90_line_distance_mm", "corridor_points_per_mm",
    "transverse_angular_span_deg", "transverse_angular_bin_fraction", "median_corridor_radius_mm",
    "normal_second_eigenvalue", "normal_spread_deg_scan", "two_sided_transition_score",
    "two_sided_normal_angle_deg", "two_sided_support_balance", "two_sided_min_support_fraction",
    "two_sided_normal_coherence", "tight_normal_second_eigenvalue_r2mm",
    "tight_normal_second_eigenvalue_r6mm", "tight_to_wide_normal_second_ratio",
    "cluster_member_count", "cluster_source_count", "case_percentile_support_count",
    "case_percentile_score", "case_percentile_consensus_count", "case_percentile_finite_support_fraction",
    "case_percentile_finite_support_longest_run_fraction", "case_percentile_endpoint_support_fraction",
    "case_percentile_normal_second_eigenvalue", "case_percentile_normal_spread_deg_scan",
    "case_percentile_tight_normal_second_eigenvalue_r2mm", "case_percentile_tight_to_wide_normal_second_ratio",
    "case_percentile_corridor_points_per_mm", "case_percentile_transverse_angular_span_deg",
    "case_percentile_cluster_member_count", "case_percentile_cluster_source_count",
    "case_percentile_two_sided_transition_score", "case_percentile_two_sided_normal_angle_deg",
    "case_percentile_two_sided_support_balance", "case_percentile_two_sided_min_support_fraction",
    "case_percentile_two_sided_normal_coherence",
)
_F1_RENAMES = {
    "score": "pool_score",
    "endpoint_support_fraction": "end_region_support_fraction",
    "normal_spread_deg_scan": "normal_spread_cloud_deg",
    "two_sided_support_balance": "two_sided_support_balance_ratio",
    "two_sided_normal_coherence": "two_sided_normal_coherence_score",
    "tight_normal_second_eigenvalue_r2mm": "tight_r2mm_normal_second_eigenvalue",
    "tight_normal_second_eigenvalue_r6mm": "tight_r6mm_normal_second_eigenvalue",
}


def _f1_key(name: str) -> str:
    if name.startswith("case_percentile_"):
        base = name[len("case_percentile_"):]
        return f"rank_{_F1_RENAMES.get(base, base)}_quantile"
    return _F1_RENAMES.get(name, name)


#: F1 descriptor name -> unit-suffixed, identity-free state key.
F1_STATE_KEYS: dict[str, str] = {name: _f1_key(name) for name in F1_BASE_FEATURE_NAMES}


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _min_distance_to_refs(line: Line | None, refs: Sequence[Line]) -> float | None:
    """True segment-to-segment distance from the candidate to the nearest reference segment.

    A long line that crosses a recipe bend is at distance 0 (never "far"), whatever
    its midpoint and endpoints do.
    """
    if line is None or not refs:
        return None
    return min(segment_distance_mm(line, ref) for ref in refs)


def _same_line(a: Line | None, b: Line | None, tol_mm: float = 1e-6) -> bool:
    """``a`` is ``b`` (either orientation) within ``tol_mm``."""
    if a is None or b is None:
        return False
    return ((math.dist(a.start, b.start) <= tol_mm and math.dist(a.end, b.end) <= tol_mm)
            or (math.dist(a.start, b.end) <= tol_mm and math.dist(a.end, b.start) <= tol_mm))


@dataclass
class ScanExport:
    records: list[tuple[CandidateRecord, LabelRow]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


def _human_sources(labels: Mapping[str, str] | None, key: str) -> str:
    if labels and key in labels:
        return labels[key]
    return "unreviewed"


# --------------------------------------------------------------------------
# E3 result.json
# --------------------------------------------------------------------------


def _reason_flags(reasons: Any) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for reason in reasons or ():
        token = enum_token(reason)
        if token:
            out[f"has_reason_{token.lower()}"] = True
    return out


def e3_records(result: Mapping[str, Any], *, group_sku: str | None,
               candidate_labels: Mapping[str, str] | None = None,
               confuser_kinds: Mapping[str, str] | None = None,
               provenance_flags: Sequence[str] = ()) -> ScanExport:
    """Candidates of one E3 ``result.json``: planned bend rows and unplanned creases."""
    if result.get("schema") != E3_RESULT_SCHEMA:
        raise ExportError(f"E3 result schema must be {E3_RESULT_SCHEMA}")
    out = ScanExport()
    source_sha = canonical_sha256(result)
    validity = result.get("inspection_validity") or {}
    registration = validity.get("registration") or {}
    bends = [b for b in result.get("bends") or () if isinstance(b, Mapping)]
    creases = [c for c in result.get("unplanned_creases") or () if isinstance(c, Mapping)]
    context = {
        "inspection_validity_state": enum_token(validity.get("state")) or "UNKNOWN",
        "registration_datum_fitness_score": _num(registration.get("datum_fitness")),
        "registration_rmse_mm": _num(registration.get("rmse_mm")),
        "unplanned_creases_count": len(creases),
    }
    nominal_lines: list[tuple[Line, dict[str, Any]]] = []
    for bend in bends:
        line = line_from_polyline((bend.get("display") or {}).get("nominal_polyline_mm"))
        if line is not None:
            evidence = bend.get("state_evidence") or {}
            nominal_lines.append((line, {"nominal_included_deg": _num(evidence.get("nominal_included_deg"))}))
    ref_lines = [line for line, _ in nominal_lines]

    rows: list[tuple[str, Line | None, dict[str, Any], dict[str, Any], str | None]] = []
    skipped_unmeasured = 0
    for bend in bends:
        uid = str(bend.get("bend_uid"))
        evidence = bend.get("state_evidence") or {}
        chars = [c for c in bend.get("characteristics") or () if isinstance(c, Mapping)]
        angle = next((c for c in chars if c.get("kind") == "INCLUDED_ANGLE"), {})
        position = next((c for c in chars if c.get("kind") == "BEND_LINE_POSITION"), {})
        flange_states = [c.get("state") for c in chars if c.get("kind") == "FLANGE_LENGTH"]
        display = bend.get("display") or {}
        measured_line = line_from_polyline(display.get("measured_polyline_mm"))
        if measured_line is None:
            # No scan-fitted line: the row is a CAD-located evaluation, not a scan candidate.
            skipped_unmeasured += 1
            continue
        nominal_line = line_from_polyline(display.get("nominal_polyline_mm"))
        parent = _num(evidence.get("parent_support_points"))
        child = _num(evidence.get("child_support_points"))
        balance = (min(parent, child) / max(parent, child)) if parent and child else None
        angle_ev = angle.get("evidence") or {}
        measured = _num(evidence.get("measured_included_deg"))
        # Exactly the keys an unplanned crease also has (scan evidence common to both).
        candidate = {
            "measured_included_deg": measured,
            "measured_line_length_mm": measured_line.length,
        }
        nominal_included = _num(evidence.get("nominal_included_deg"))
        cad_relation = {
            "is_planned_bend": True,
            "nominal_included_deg": nominal_included,
            "angle_deviation_deg": (measured - nominal_included) if (measured is not None and nominal_included is not None) else None,
            "formed_fraction": _num(evidence.get("formed_fraction")),
            "prior_formed_probability": _num(evidence.get("prior_formed_probability")),
            "nominal_line_length_mm": None if nominal_line is None else nominal_line.length,
            "planned_bends_count": len(bends),
            # Engine evidence that exists only for planned rows (presence encodes the binding).
            "parent_support_count": parent,
            "child_support_count": child,
            "side_support_balance_ratio": balance,
            "coverage_ratio": _num(evidence.get("coverage_ratio")),
            "angle_fit_rms_mm": _num(angle_ev.get("fit_rms_mm")),
            "angle_support_count": _num(angle_ev.get("support_points")),
            "engine_bend_state": enum_token(bend.get("state")) or "UNKNOWN",
            "angle_measurement_state": enum_token(angle.get("state")) or "UNKNOWN",
            "position_measurement_state": enum_token(position.get("state")) or "UNKNOWN",
            "flange_measured_count": sum(1 for st in flange_states if st == "MEASURED"),
            "engine_reasons": _reason_flags(bend.get("state_reasons")),
        }
        weak = ("recipe_binding_bound_formed"
                if bend.get("state") == "FORMED" and angle.get("state") == "MEASURED" else None)
        # Only the measured (scan-fitted) line feeds the neighbour summary: the nominal line is
        # CAD geometry and must stay inside cad_relation.
        rows.append((f"e3|bend|{uid}", measured_line, candidate, cad_relation, weak))
    for crease in creases:
        cid = str(crease.get("crease_id"))
        line = line_from_polyline(crease.get("polyline_mm"))
        candidate = {
            "measured_included_deg": _num(crease.get("included_deg")),
            "measured_line_length_mm": None if line is None else line.length,
        }
        relation = relation_to_reference(line, nominal_lines)
        relation["is_planned_bend"] = False
        relation["planned_bends_count"] = len(bends)
        # EXTRA_BEND_SUSPECTED vs UNPLANNED_CREASE is decided against the recipe.
        relation["engine_crease_reason"] = enum_token(crease.get("reason")) or "UNKNOWN"
        distance = _min_distance_to_refs(line, ref_lines)
        weak = "recipe_binding_unbound_far" if (distance is not None and distance > WEAK_FAR_MM) else None
        rows.append((f"e3|crease|{cid}", line, candidate, relation, weak))

    lines = [row[1] for row in rows]
    for index, (key, _line, candidate, relation, weak) in enumerate(rows):
        record = CandidateRecord(source_sha, key, "SCAN_E3", candidate,
                                 neighbours=neighbour_summary(index, lines), context=context,
                                 cad_relation=relation)
        rid = record_id_for(source_sha, key)
        out.records.append((record, make_label(
            rid, label_source=_human_sources(candidate_labels, key), group_sku=group_sku,
            weak_label_source=weak, confuser_kind=(confuser_kinds or {}).get(key),
            candidate_key=key, source_sha256=source_sha, provenance_flags=provenance_flags)))
    out.bump("e3_bend_rows", len(bends))
    out.bump("e3_bends_without_measured_line", skipped_unmeasured)
    out.bump("e3_unplanned_creases", len(creases))
    return out


# --------------------------------------------------------------------------
# Legacy bend_inspection_report.json
# --------------------------------------------------------------------------


def legacy_report_records(report: Mapping[str, Any], *, group_sku: str | None,
                          marker_labels: Mapping[str, str] | None = None,
                          candidate_labels: Mapping[str, str] | None = None,
                          provenance_flags: Sequence[str] = ()) -> ScanExport:
    """Candidates of one legacy report: matches that have a detection, and unmatched detections."""
    out = ScanExport()
    source_sha = canonical_sha256(report)
    matches = [m for m in report.get("matches") or () if isinstance(m, Mapping)]
    unmatched = [d for d in report.get("unmatched_detections") or () if isinstance(d, Mapping)]
    cad_lines: list[tuple[Line, dict[str, Any]]] = []
    for match in matches:
        line = line_from_endpoints(match.get("cad_line_start"), match.get("cad_line_end"))
        if line is not None:
            cad_lines.append((line, {"target_angle_deg": _num(match.get("target_angle"))}))
    ref_lines = [line for line, _ in cad_lines]

    id_counts: dict[str, int] = {}
    for item in [*matches, *unmatched]:
        bid = str(item.get("bend_id"))
        id_counts[bid] = id_counts.get(bid, 0) + 1

    def _folded(measured: float | None) -> dict[str, Any]:
        return {
            "measured_folded_included_deg": measured,
            "unfold_a_included_deg": measured,
            "unfold_b_included_deg": None if measured is None else 180.0 - measured,
        }

    rows: list[tuple[str, str, Line | None, dict[str, Any], dict[str, Any], str | None]] = []
    for match in matches:
        line = line_from_endpoints(match.get("detected_line_start"), match.get("detected_line_end"))
        if line is None:
            out.bump("legacy_matches_without_detection")
            continue
        cad_line = line_from_endpoints(match.get("cad_line_start"), match.get("cad_line_end"))
        if (match.get("assignment_source") == "CAD_LOCAL_NEIGHBORHOOD"
                or match.get("detected_line_geometry_source") == "CAD_REFERENCE_PLACEHOLDER"
                or _same_line(line, cad_line)):
            # The "detected" line is the nominal CAD line (and older reports may carry the
            # target radius as measured_radius): nominal must never be exported as actual.
            out.bump("legacy_matches_cad_placeholder")
            continue
        bid = str(match.get("bend_id"))
        candidate = {**_folded(_num(match.get("measured_angle"))),
                     "measured_radius_mm": _num(match.get("measured_radius")),
                     "detected_line_length_mm": line.length}
        relation = {
            "is_bound_to_planned_bend": True,
            "assignment_kind": enum_token(match.get("assignment_candidate_kind")) or "NONE",
            "assignment_source": enum_token(match.get("assignment_source")) or "NONE",
            "assignment_confidence_score": _num(match.get("assignment_confidence")),
            "assignment_candidate_score": _num(match.get("assignment_candidate_score")),
            "assignment_null_score": _num(match.get("assignment_null_score")),
            "assignment_candidates_count": _num(match.get("assignment_candidate_count")),
            "engine_confidence_score": _num(match.get("confidence")),
            "observability_state": enum_token(match.get("observability_state")) or "UNKNOWN",
            "observability_confidence_score": _num(match.get("observability_confidence")),
            "visibility_score": _num(match.get("visibility_score")),
            "surface_visibility_ratio": _num(match.get("surface_visibility_ratio")),
            "local_support_score": _num(match.get("local_support_score")),
            "side_balance_score": _num(match.get("side_balance_score")),
            "local_evidence_score": _num(match.get("local_evidence_score")),
            "local_points_count": _num(match.get("local_point_count")),
            "observed_surfaces_count": _num(match.get("observed_surface_count")),
            "measurement_mode": enum_token(match.get("measurement_mode")) or "UNKNOWN",
            "bend_form": enum_token(match.get("bend_form")) or "UNKNOWN",
            "target_angle_deg": _num(match.get("target_angle")),
            "angle_deviation_deg": _num(match.get("angle_deviation")),
            "line_center_deviation_mm": _num(match.get("line_center_deviation_mm")),
            "line_length_deviation_mm": _num(match.get("line_length_deviation_mm")),
            "planned_bends_count": len(matches),
        }
        weak = ("recipe_binding_bound_formed"
                if match.get("physical_completion_state") == "FORMED"
                and match.get("status") in LEGACY_MEASURED_STATUSES else None)
        rows.append((f"legacy|match|{bid}", bid, line, candidate, relation, weak))
    for det in unmatched:
        line = line_from_endpoints(det.get("bend_line_start"), det.get("bend_line_end"))
        bid = str(det.get("bend_id"))
        candidate = {**_folded(_num(det.get("measured_angle"))),
                     "measured_radius_mm": _num(det.get("measured_radius")),
                     "detected_line_length_mm": _num(det.get("bend_line_length")) if line is None else line.length}
        relation = relation_to_reference(line, cad_lines)
        relation.update({
            "is_bound_to_planned_bend": False,
            "assignment_kind": "UNMATCHED",
            "engine_confidence_score": _num(det.get("confidence")),
            "flange_a_area_mm2": _num(det.get("flange1_area")),
            "flange_b_area_mm2": _num(det.get("flange2_area")),
            "planned_bends_count": len(matches),
        })
        distance = _min_distance_to_refs(line, ref_lines)
        weak = "recipe_binding_unbound_far" if (distance is not None and distance > WEAK_FAR_MM) else None
        rows.append((f"legacy|unmatched|{bid}", bid, line, candidate, relation, weak))

    lines = [row[2] for row in rows]
    context = {"detected_candidates_count": len(rows)}
    for index, (key, bid, _line, candidate, relation, weak) in enumerate(rows):
        source = "unreviewed"
        if candidate_labels and key in candidate_labels:
            source = candidate_labels[key]
        elif marker_labels and bid in marker_labels:
            if id_counts.get(bid, 0) > 1:
                out.bump("legacy_marker_ambiguous_id")
            else:
                source = marker_labels[bid]
        record = CandidateRecord(source_sha, key, "SCAN_LEGACY_REPORT", candidate,
                                 neighbours=neighbour_summary(index, lines), context=context,
                                 cad_relation=relation)
        rid = record_id_for(source_sha, key)
        out.records.append((record, make_label(
            rid, label_source=source, group_sku=group_sku, weak_label_source=weak,
            candidate_key=key, source_sha256=source_sha, provenance_flags=provenance_flags)))
    out.bump("legacy_candidates", len(rows))
    return out


# --------------------------------------------------------------------------
# F1 candidate pool rows
# --------------------------------------------------------------------------


def _f1_line(row: Mapping[str, Any]) -> Line | None:
    ends = row.get("span_endpoints")
    if isinstance(ends, list) and len(ends) >= 2:
        return line_from_endpoints(ends[0], ends[1])
    anchor, axis, span = row.get("anchor"), row.get("axis_direction"), _num(row.get("span_mm"))
    if isinstance(anchor, list) and isinstance(axis, list) and len(anchor) == 3 and len(axis) == 3 and span:
        norm = math.sqrt(sum(float(a) ** 2 for a in axis))
        if norm <= 0:
            return None
        unit = [float(a) / norm for a in axis]
        return line_from_endpoints([a - 0.5 * span * u for a, u in zip(anchor, unit)],
                                   [a + 0.5 * span * u for a, u in zip(anchor, unit)])
    return None


def parse_gold(doc: Any) -> tuple[str, bool, dict[str, list[Line]]]:
    """``(frame, complete, {case_key: [gold lines]})`` from a ``jev_gold_lines_v1`` doc."""
    if not isinstance(doc, Mapping) or doc.get("schema") != GOLD_SCHEMA:
        raise LabelError(f"gold document schema must be {GOLD_SCHEMA}")
    frame = doc.get("coordinate_frame")
    if not isinstance(frame, str) or not frame:
        raise LabelError("gold document needs a coordinate_frame")
    status = doc.get("review_status")
    if status not in ("complete", "in_progress"):
        raise LabelError("gold review_status must be complete or in_progress")
    cases = doc.get("cases")
    if not isinstance(cases, Mapping):
        raise LabelError("gold cases must be an object")
    out: dict[str, list[Line]] = {}
    for case, lines in cases.items():
        parsed = []
        for item in lines or ():
            line = line_from_endpoints((item or {}).get("line_start"), (item or {}).get("line_end"))
            if line is None:
                raise LabelError(f"gold line in {case!r} is degenerate")
            parsed.append(line)
        out[str(case)] = parsed
    return frame, status == "complete", out


def _gold_match(gold: Sequence[Line], line: Line | None) -> bool:
    if line is None:
        return False
    for ref in gold:
        m = line_pair_metrics(ref, line)
        if (m["axis_angle_deg"] <= GOLD_ANGLE_DEG and m["transverse_offset_mm"] <= GOLD_OFFSET_MM
                and m["reference_overlap_fraction"] >= GOLD_OVERLAP and m["other_overlap_fraction"] >= GOLD_OVERLAP):
            return True
    return False


def f1_pool_records(rows: Sequence[Mapping[str, Any]], *, source_sha256: str,
                    gold: tuple[str, bool, dict[str, list[Line]]] | None = None,
                    provenance_flags: Sequence[str] = ()) -> ScanExport:
    """States for F1 pool rows, grouped by ``case_key`` for the neighbour summary."""
    out = ScanExport()
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        case = row.get("case_key")
        if not isinstance(case, str) or not case:
            raise ExportError("every F1 pool row needs a case_key")
        if not isinstance(row.get("coordinate_frame"), str):
            raise ExportError("every F1 pool row needs a coordinate_frame")
        by_case.setdefault(case, []).append(row)
    for case in sorted(by_case):
        members = sorted(by_case[case], key=lambda r: str(r.get("candidate_id")))
        lines = [_f1_line(r) for r in members]
        frames = {r["coordinate_frame"] for r in members}
        if len(frames) != 1:
            raise ExportError(f"case {case!r} mixes coordinate frames")
        gold_lines = None
        complete = False
        if gold is not None:
            gold_frame, complete, cases = gold
            if case in cases:
                if gold_frame != next(iter(frames)):
                    raise LabelError(f"gold frame {gold_frame!r} differs from pool frame for {case!r}")
                gold_lines = cases[case]
        for index, row in enumerate(members):
            key = f"f1|{case}|{row.get('candidate_id')}"
            candidate: dict[str, Any] = {F1_STATE_KEYS[n]: _num(row.get(n)) for n in F1_BASE_FEATURE_NAMES}
            candidate["generator_source"] = enum_token(row.get("representative_source")) or "UNKNOWN"
            relation = {"engine_confidence_score": _num(row.get("case_percentile_score"))}
            if gold_lines is None:
                label_source = "unreviewed"
            elif _gold_match(gold_lines, lines[index]):
                label_source = "gold_line_matched"
            else:
                label_source = "gold_line_unmatched" if complete else "gold_line_unmatched_incomplete"
            record = CandidateRecord(source_sha256, key, "SCAN_F1_POOL", candidate,
                                     neighbours=neighbour_summary(index, lines),
                                     context={"pool_candidates_count": len(members)},
                                     cad_relation=relation)
            rid = record_id_for(source_sha256, key)
            out.records.append((record, make_label(
                rid, label_source=label_source, group_sku=row.get("group_sku"), candidate_key=key,
                source_sha256=source_sha256, provenance_flags=provenance_flags)))
        out.bump("f1_pool_rows", len(members))
    return out


# --------------------------------------------------------------------------
# Export driver
# --------------------------------------------------------------------------


def _job_sku(job_dir: Path, fallback: Any, auth: Any, repo_root: Path) -> str | None:
    for name in ("recipe.json", "job.json"):
        candidate = job_dir / name
        if candidate.is_file():
            path = check_input_path(candidate, authorization=auth, repo_root=repo_root)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ExportError(f"{candidate}: invalid JSON") from exc
            if isinstance(doc, Mapping) and isinstance(doc.get("sku"), str):
                return doc["sku"]
    return fallback if isinstance(fallback, str) else None


def export_scan(
    out_dir: Path,
    *,
    jobs_root: Path | None = None,
    labels_root: Path | None = None,
    f1_pool: Path | None = None,
    gold: Path | None = None,
    authorization: Path | None = None,
    provenance_flags: Sequence[str] = (),
    repo_root: Path = REPO_ROOT,
) -> ExportResult:
    """Export scan candidates from job workspaces under ``jobs_root`` and/or an F1 pool file."""
    if jobs_root is None and f1_pool is None:
        raise ExportError("give --jobs-root and/or --f1-pool")
    provenance_flags = validate_provenance_flags(list(provenance_flags))
    auth = load_authorization(authorization, repo_root=repo_root)
    states: list[tuple[str, dict[str, Any]]] = []
    labels: list[LabelRow] = []
    counts: dict[str, int] = {}
    warnings: list[str] = []
    inputs: list[dict[str, str]] = []

    def _read(path: Path) -> Any:
        resolved = check_input_path(path, authorization=auth, repo_root=repo_root)
        data = resolved.read_bytes()
        inputs.append({"name": f"{path.parent.parent.name}/{path.parent.name}/{path.name}",
                       "sha256": hashlib.sha256(data).hexdigest()})
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExportError(f"{path}: invalid JSON") from exc

    def _collect(part: ScanExport) -> None:
        for record, label in part.records:
            rid, state = build_state(record)
            states.append((rid, state))
            labels.append(label)
        for key, value in part.counts.items():
            counts[key] = counts.get(key, 0) + value
        warnings.extend(part.warnings)

    if labels_root is not None:
        # Checked before anything inside it is even stat'ed.
        labels_root = check_input_dir(labels_root, authorization=auth, repo_root=repo_root)
    if jobs_root is not None:
        root = Path(jobs_root)
        # The root itself must pass custody before it is listed.
        root = check_input_dir(root, authorization=auth, repo_root=repo_root)
        for job_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            e3_path = job_dir / "engine" / "result.json"
            legacy_paths = [job_dir / "engine" / "bend_inspection_report.json", job_dir / "bend_inspection_report.json"]
            job_labels_dir = None if labels_root is None else Path(labels_root) / job_dir.name
            marker = None
            cand_doc_path = None
            if job_labels_dir is not None:
                if (job_labels_dir / "bend_instance_labels.json").is_file():
                    marker = job_labels_dir / "bend_instance_labels.json"
                if (job_labels_dir / "jev_candidate_labels.json").is_file():
                    cand_doc_path = job_labels_dir / "jev_candidate_labels.json"
            found = False
            if e3_path.is_file():
                found = True
                result = _read(e3_path)
                cand_labels, kinds = (None, None)
                if cand_doc_path is not None:
                    cand_labels, kinds = parse_candidate_label_doc(_read(cand_doc_path), required_frame=E3_FRAME)
                sku = _job_sku(job_dir, result.get("sku") if isinstance(result, Mapping) else None, auth, repo_root)
                _collect(e3_records(result, group_sku=sku, candidate_labels=cand_labels,
                                    confuser_kinds=kinds, provenance_flags=provenance_flags))
            for legacy_path in legacy_paths:
                if legacy_path.is_file():
                    found = True
                    report = _read(legacy_path)
                    markers = parse_marker_reviews(_read(marker)) if marker is not None else None
                    cand_labels = None
                    if cand_doc_path is not None and not e3_path.is_file():
                        cand_labels, _ = parse_candidate_label_doc(_read(cand_doc_path), required_frame=LEGACY_FRAME)
                    sku = _job_sku(job_dir, None, auth, repo_root)
                    _collect(legacy_report_records(report, group_sku=sku, marker_labels=markers,
                                                   candidate_labels=cand_labels, provenance_flags=provenance_flags))
                    break
            if not found:
                counts["jobs_without_candidates"] = counts.get("jobs_without_candidates", 0) + 1
            counts["jobs"] = counts.get("jobs", 0) + 1

    if f1_pool is not None:
        pool_path = check_input_path(f1_pool, authorization=auth, repo_root=repo_root)
        data = pool_path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        inputs.append({"name": pool_path.name, "sha256": sha})
        rows = [r for r in load_json_records(pool_path) if isinstance(r, Mapping)]
        gold_doc = parse_gold(_read(Path(gold))) if gold is not None else None
        _collect(f1_pool_records(rows, source_sha256=sha, gold=gold_doc, provenance_flags=provenance_flags))

    by_source: dict[str, int] = {}
    for row in labels:
        by_source[row.label_source] = by_source.get(row.label_source, 0) + 1
        if row.weak_label_source:
            by_source[f"weak:{row.weak_label_source}"] = by_source.get(f"weak:{row.weak_label_source}", 0) + 1
    counts.update({f"label_source:{k}": v for k, v in sorted(by_source.items())})
    manifest = {
        "exporter": "export_scan_candidates_v1",
        "family": "SCAN_ENGINE_CANDIDATE",
        "inputs": inputs,
        "authorization_sha256": None if auth is None else auth.sha256,
        "options": {"labels_root_given": labels_root is not None, "gold_given": gold is not None,
                    "provenance_flags": sorted(set(provenance_flags))},
        "counts": counts,
        "warnings": sorted(warnings),
        "leakage_guards": [
            "everything whose value or presence depends on CAD or recipe binding is in cad_relation",
            "E3 candidate block has the same keys for planned rows and creases",
            "legacy CAD placeholder matches and rows without a scan-fitted line are skipped, never exported",
            "observability/bend state is a feature, never a label",
            "weak labels are a separate column and never evaluation truth",
            "human labels only from explicitly supplied documents with a matching frame",
        ],
    }
    return write_export(out_dir, states=states, labels=labels, manifest=manifest)


# --------------------------------------------------------------------------
# Synthetic examples (tests and acceptance runs only)
# --------------------------------------------------------------------------


def synthetic_e3_result(seed: int = 0, *, sku: str = "SYN-SCAN-A", n_bends: int = 3,
                        n_creases: int = 2) -> dict[str, Any]:
    """A minimal fake E3 result with planned bends and unplanned creases (not contract-valid)."""
    rng = random.Random(seed)
    bends = []
    for i in range(n_bends):
        y = 50.0 * i
        formed = i < n_bends - 1
        measured = 90.0 + rng.uniform(-1, 1) if formed else 178.0
        bends.append({
            "bend_uid": f"bu_{seed:04d}{i:016d}",
            "label": f"B{i + 1}",
            "sequence_index": i + 1,
            "state": "FORMED" if formed else "NOT_YET_FORMED",
            "state_reasons": [] if formed else ["COVERAGE_LOW"],
            "state_evidence": {"nominal_included_deg": 90.0, "measured_included_deg": measured,
                               "formed_fraction": (180 - measured) / 90.0, "parent_support_points": 900 + i,
                               "child_support_points": 700 + 3 * i, "coverage_ratio": 0.8},
            "characteristics": [
                {"kind": "FLANGE_LENGTH", "side": "A", "state": "MEASURED" if formed else "NOT_APPLICABLE"},
                {"kind": "FLANGE_LENGTH", "side": "B", "state": "MEASURED" if formed else "NOT_APPLICABLE"},
                {"kind": "INCLUDED_ANGLE", "state": "MEASURED" if formed else "NOT_APPLICABLE",
                 "evidence": {"method": "plane_fit", "support_points": 1500, "fit_rms_mm": 0.05}},
                {"kind": "BEND_LINE_POSITION", "state": "MEASURED" if formed else "NOT_APPLICABLE"},
            ],
            "display": {"nominal_polyline_mm": [[0.0, y, 0.0], [120.0, y, 0.0]],
                        "measured_polyline_mm": [[0.2, y + 0.1, 0.0], [119.8, y + 0.1, 0.0]] if formed else None},
        })
    creases = []
    for j in range(n_creases):
        far = j % 2 == 0
        y = 400.0 if far else 50.0 + 2.0
        creases.append({"crease_id": f"crease-{j + 1}", "reason": "UNPLANNED_CREASE",
                        "polyline_mm": [[10.0, y, 5.0], [10.0 + rng.uniform(10, 40), y, 5.0]],
                        "included_deg": rng.uniform(120, 175)})
    return {
        "schema": E3_RESULT_SCHEMA,
        "created_at": "2026-10-17T00:00:00Z",
        "sku": sku,
        "scan": {"scan_file_name": f"{sku}_SCAN_1.ply", "sha256": "1" * 64},
        "inspection_validity": {"state": "VALID", "registration": {"datum_fitness": 0.91, "rmse_mm": 0.12}},
        "bends": bends,
        "unplanned_creases": creases,
    }


def synthetic_legacy_report(seed: int = 0, *, n_matches: int = 3, n_unmatched: int = 2) -> dict[str, Any]:
    """A minimal fake legacy report (``matches`` + ``unmatched_detections``)."""
    rng = random.Random(seed)
    matches = []
    for i in range(n_matches):
        y = 60.0 * i
        detected = i < n_matches - 1
        matches.append({
            "bend_id": f"CAD_B{i + 1}", "status": "PASS" if detected else "NOT_DETECTED",
            "target_angle": 90.0, "measured_angle": 90.5 + rng.uniform(-1, 1) if detected else None,
            "measured_radius": 2.1 if detected else None, "confidence": 0.8 + 0.05 * i,
            "cad_line_start": [0.0, y, 0.0], "cad_line_end": [100.0, y, 0.0],
            "detected_line_start": [0.5, y + 0.3, 0.0] if detected else None,
            "detected_line_end": [99.0, y + 0.3, 0.0] if detected else None,
            "physical_completion_state": "FORMED" if detected else "UNKNOWN",
            "observability_state": "FORMED" if detected else "UNKNOWN",
            "observability_confidence": 0.9, "visibility_score": 0.8, "surface_visibility_ratio": 0.7,
            "local_support_score": 0.6, "side_balance_score": 0.5, "assignment_source": "LOCAL",
            "assignment_confidence": 0.7, "assignment_candidate_kind": "DETECTED_BEND",
            "assignment_candidate_score": 0.66, "assignment_null_score": 0.1, "assignment_candidate_count": 3,
            "observed_surface_count": 2, "local_point_count": 1200, "local_evidence_score": 0.5,
            "measurement_mode": "SCAN", "bend_form": "STANDARD", "angle_deviation": 0.5,
            "line_center_deviation_mm": 0.3, "line_length_deviation_mm": -1.5,
        })
    unmatched = []
    for j in range(n_unmatched):
        y = 500.0 + 10 * j if j % 2 == 0 else 3.0
        unmatched.append({"bend_id": f"DET_{j + 1}", "measured_angle": rng.uniform(100, 175),
                          "measured_radius": rng.uniform(0.2, 15.0), "bend_line_length": 25.0,
                          "bend_line_start": [5.0, y, 1.0], "bend_line_end": [30.0, y, 1.0],
                          "confidence": rng.uniform(0.2, 0.9), "flange1_area": 1200.0, "flange2_area": 80.0})
    return {"part_id": "PART-REAL-NAME", "matches": matches, "unmatched_detections": unmatched}


def synthetic_f1_pool(seed: int = 0, *, group_sku: str = "SYN-F1-A", case_key: str = "case-a",
                      n_real: int = 3, n_fake: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fake F1 pool rows and a matching complete gold document (frame ``reference_mesh_raw``)."""
    rng = random.Random(seed)
    rows, gold = [], []
    for i in range(n_real + n_fake):
        real = i < n_real
        y = 40.0 * i
        start, end = [0.0, y, 0.0], [120.0 if real else 30.0, y, 0.0]
        row: dict[str, Any] = {n: rng.uniform(0.0, 1.0) for n in F1_BASE_FEATURE_NAMES}
        row.update({"candidate_id": f"{case_key}-c{i}", "case_key": case_key, "group_sku": group_sku,
                    "coordinate_frame": LEGACY_FRAME, "span_endpoints": [start, end],
                    "span_mm": math.dist(start, end), "representative_source": "plane_lattice",
                    "part_id": "PART-REAL-NAME", "filename": "secret.ply"})
        rows.append(row)
        if real:
            gold.append({"line_start": [0.0, y + 0.5, 0.0], "line_end": [118.0, y + 0.5, 0.0]})
    doc = {"schema": GOLD_SCHEMA, "coordinate_frame": LEGACY_FRAME, "review_status": "complete",
           "cases": {case_key: gold}}
    return rows, doc
