"""G2 line-localization metric adapter (plan §5 G2a, §10 amendment 6a).

The matcher is the one of ``scripts/evaluate_f1_against_manual_gold.py``
(re-implemented here so the evaluation package never imports ``scripts/``; a
parity test runs both on random lines): one-to-one minimum-cost Hungarian
matching where a pair is feasible when the undirected axis angle <= 5 deg, the
symmetric transverse offset <= 4 mm and both span overlaps >= 0.60.

Amendment 6a rules implemented here:

- **Raw-scan frame.** Engine lines are in ``cad_model_mm`` and are mapped back
  with the result's own registration (``raw = inv(raw_to_frame) @ frame /
  unit_factor``). Gold lines drawn in a reference frame need a recovered
  raw-to-reference transform with residual <= 0.01 mm; cases without one are
  listed and excluded **before** any scoring (:func:`prepare_gold`).
- **One declared locus**: the visible-surface apex (``OUTER_APEX``). The engine
  publishes the measured OML virtual sharp; the apex line is that line moved
  along the *measured* bisector by ``(Ri + t)(1/sin(theta/2) - 1)`` at the
  *measured* included angle (:func:`.engine_output.measured_outer_apex`), so a
  partially formed or mis-bent bend is scored at its real apex, not at the
  nominal offset (which is wrong by up to about 0.38 (Ri + t) there). Without
  a measured angle the nominal offset is used and the line record says so
  (``apex_source``). The OML sharp itself is a diagnostic locus only.
- **Only MEASURED positions emit lines**: a bend emits a line only when its
  BEND_LINE_POSITION is MEASURED (``display.measured_polyline_mm`` present);
  NOT_YET_FORMED and UNKNOWN bends emit none, and an invalid inspection emits
  nothing.
- **Aggregate as the script's evaluate()**: precision and recall are 0.0 when
  their denominator is empty and F1 is 0.0 when precision + recall = 0, so an
  engine that emits no line on usable Gold scores F1 = 0 and FAILs; only a run
  without any usable Gold case is NOT_EVALUABLE.
- The result is labelled "recipe-guided vs blind union, not like-for-like" and
  carries the exact-part rate, compared with the legacy arm on the same scans
  when given.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.production.compiled_part import CompiledPartV1
from contracts.production.inspection_result import ShopfloorInspectionResultV1

from .engine_output import decoded_stage, frame_to_raw, measured_outer_apex, row_characteristic
from .thresholds import G2_LINE_THRESHOLDS

__all__ = [
    "Thresholds",
    "DEFAULT_THRESHOLDS",
    "line_endpoints",
    "pair_metrics",
    "is_match",
    "match_cost",
    "match_lines",
    "engine_lines",
    "GoldCase",
    "prepare_gold",
    "score_line_metric",
]


@dataclass(frozen=True)
class Thresholds:
    angle_deg: float = 5.0
    transverse_offset_mm: float = 4.0
    gold_span_overlap: float = 0.60
    prediction_span_overlap: float = 0.60


_M = G2_LINE_THRESHOLDS["matcher"]
DEFAULT_THRESHOLDS = Thresholds(_M["axis_angle_deg"], _M["transverse_offset_mm"], _M["gold_span_overlap"],
                                _M["prediction_span_overlap"])


def line_endpoints(record: Mapping[str, Any]) -> np.ndarray:
    """Endpoints of a line record in any of the script's shapes: ``span_endpoints``,
    ``line_start``/``line_end`` or ``anchor`` + ``axis_direction`` + ``span_mm``."""
    endpoints = record.get("span_endpoints")
    if endpoints:
        return np.asarray(endpoints, dtype=np.float64)
    if record.get("line_start") is not None and record.get("line_end") is not None:
        return np.asarray([record["line_start"], record["line_end"]], dtype=np.float64)
    anchor = np.asarray(record["anchor"], dtype=np.float64)
    direction = np.asarray(record["axis_direction"], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    half = direction * float(record["span_mm"]) / 2.0
    return np.asarray([anchor - half, anchor + half])


def pair_metrics(gold: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Axis angle, symmetric transverse offset and span overlaps of two segments
    (identical to the script's ``pair_metrics``)."""
    gold = np.asarray(gold, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    gold_mid = gold.mean(axis=0)
    pred_mid = prediction.mean(axis=0)
    gd = gold[1] - gold[0]
    pd = prediction[1] - prediction[0]
    gl = float(np.linalg.norm(gd))
    pl = float(np.linalg.norm(pd))
    if gl <= 1e-9 or pl <= 1e-9:
        raise ValueError("bend lines must have nonzero length")
    ga = gd / gl
    pa = pd / pl
    dot = float(np.clip(np.dot(ga, pa), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    if dot < 0:
        pa = -pa
    delta = pred_mid - gold_mid
    off_g = float(np.linalg.norm(delta - ga * np.dot(delta, ga)))
    off_p = float(np.linalg.norm(delta - pa * np.dot(delta, pa)))
    gi = sorted(float(np.dot(p - gold_mid, ga)) for p in gold)
    pi = sorted(float(np.dot(p - gold_mid, ga)) for p in prediction)
    overlap = max(0.0, min(gi[1], pi[1]) - max(gi[0], pi[0]))
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": max(off_g, off_p),
        "gold_span_overlap": overlap / gl,
        "prediction_span_overlap": overlap / pl,
        "shorter_span_overlap": overlap / min(gl, pl),
        "gold_length_mm": gl,
        "prediction_length_mm": pl,
    }


def is_match(m: Mapping[str, float], t: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    return (
        m["axis_angle_deg"] <= t.angle_deg
        and m["transverse_offset_mm"] <= t.transverse_offset_mm
        and m["gold_span_overlap"] >= t.gold_span_overlap
        and m["prediction_span_overlap"] >= t.prediction_span_overlap
    )


def match_cost(m: Mapping[str, float], t: Thresholds = DEFAULT_THRESHOLDS) -> float:
    return (
        m["axis_angle_deg"] / t.angle_deg
        + m["transverse_offset_mm"] / t.transverse_offset_mm
        + (1.0 - min(m["gold_span_overlap"], 1.0))
        + (1.0 - min(m["prediction_span_overlap"], 1.0))
    )


def match_lines(gold_records: Sequence[Mapping[str, Any]], prediction_records: Sequence[Mapping[str, Any]],
                t: Thresholds = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """One-to-one minimum-cost matching (same output shape as the script)."""
    from scipy.optimize import linear_sum_assignment

    gold = [(str(r["bend_id"]), line_endpoints(r)) for r in gold_records]
    preds = [(str(r["bend_id"]), line_endpoints(r)) for r in prediction_records]
    matrix = np.full((len(gold), len(preds)), 1e9, dtype=np.float64)
    cache: dict[tuple[int, int], dict[str, float]] = {}
    for gi, (_, gl) in enumerate(gold):
        for pi, (_, pl) in enumerate(preds):
            m = pair_metrics(gl, pl)
            cache[(gi, pi)] = m
            if is_match(m, t):
                matrix[gi, pi] = match_cost(m, t)
    matches = []
    mg: set[int] = set()
    mp: set[int] = set()
    if gold and preds:
        rows, cols = linear_sum_assignment(matrix)
        for gi, pi in zip(rows, cols):
            if matrix[gi, pi] >= 1e9:
                continue
            mg.add(int(gi))
            mp.add(int(pi))
            matches.append({"gold_id": gold[gi][0], "prediction_id": preds[pi][0],
                            **{k: round(v, 6) for k, v in cache[(int(gi), int(pi))].items()}})
    return {
        "gold_count": len(gold),
        "prediction_count": len(preds),
        "true_positive": len(matches),
        "false_positive": len(preds) - len(matches),
        "false_negative": len(gold) - len(matches),
        "exact": len(matches) == len(gold) == len(preds),
        "matches": matches,
        "unmatched_gold_ids": [g[0] for i, g in enumerate(gold) if i not in mg],
        "unmatched_prediction_ids": [p[0] for i, p in enumerate(preds) if i not in mp],
    }


def engine_lines(result: ShopfloorInspectionResultV1, compiled: CompiledPartV1, *, locus: str = "OUTER_APEX",
                 frame: str = "raw") -> list[dict[str, Any]]:
    """Engine bend lines for the line metric (module docstring rules).

    ``locus`` is ``OUTER_APEX`` (declared) or ``OML_VIRTUAL_SHARP`` (diagnostic);
    ``frame`` is ``raw`` (scoring) or ``cad`` (rendering)."""
    if locus not in ("OUTER_APEX", "OML_VIRTUAL_SHARP"):
        raise ValueError(f"unknown locus {locus!r}")
    if result.inspection_validity.state != "VALID":
        return []
    stage = decoded_stage(result, compiled)
    out = []
    for row in result.bends:
        pos = row_characteristic(row, "BEND_LINE_POSITION")
        line = row.display.measured_polyline_mm
        if pos is None or pos.state != "MEASURED" or line is None or row.state == "NOT_YET_FORMED":
            continue
        pts = np.asarray([line[0], line[-1]], dtype=np.float64)
        source = "OML_VIRTUAL_SHARP"
        if locus == "OUTER_APEX":
            apex, source = measured_outer_apex(compiled, stage, row, fallback="NOMINAL")
            if apex is None:  # pragma: no cover - a measured line always yields an apex with fallback NOMINAL
                continue
            pts = apex
        if frame == "raw":
            pts = frame_to_raw(result, pts)
        out.append({"bend_id": row.bend_uid, "label": row.label, "sequence_index": row.sequence_index,
                    "locus": locus, "apex_source": source, "frame": frame,
                    "line_start": pts[0].tolist(), "line_end": pts[1].tolist()})
    return out


@dataclass(frozen=True)
class GoldCase:
    """Gold line references of one scan.

    ``frame`` is ``raw_scan`` (lines drawn on the raw scan) or ``reference``
    (lines drawn in another frame, then ``raw_to_reference`` and its recovery
    ``residual_mm`` are required)."""

    case_id: str
    gold_lines: tuple[Mapping[str, Any], ...]
    frame: str
    raw_to_reference: np.ndarray | None = None
    residual_mm: float | None = None
    transform_method: str | None = None


def prepare_gold(cases: Sequence[GoldCase], *, max_residual_mm: float | None = None
                 ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Gold lines in the raw-scan frame, and the cases excluded before scoring.

    A ``reference``-frame case is excluded unless it has a transform whose
    recovery residual is <= ``max_residual_mm`` (default 0.01 mm)."""
    limit = G2_LINE_THRESHOLDS["transform_residual_max_mm"] if max_residual_mm is None else max_residual_mm
    usable: dict[str, list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    for case in cases:
        if case.frame == "raw_scan":
            usable[case.case_id] = [dict(r) for r in case.gold_lines]
            continue
        if case.frame != "reference":
            excluded.append({"case_id": case.case_id, "reason": f"UNKNOWN_FRAME:{case.frame}"})
            continue
        if case.raw_to_reference is None or case.residual_mm is None:
            excluded.append({"case_id": case.case_id, "reason": "NO_RECOVERABLE_TRANSFORM"})
            continue
        if not (case.residual_mm <= limit):
            excluded.append({"case_id": case.case_id, "reason": "TRANSFORM_RESIDUAL_TOO_LARGE",
                             "residual_mm": case.residual_mm})
            continue
        inv = np.linalg.inv(np.asarray(case.raw_to_reference, dtype=np.float64))
        lines = []
        for r in case.gold_lines:
            pts = line_endpoints(r)
            raw = pts @ inv[:3, :3].T + inv[:3, 3]
            lines.append({"bend_id": r["bend_id"], "line_start": raw[0].tolist(), "line_end": raw[1].tolist()})
        usable[case.case_id] = lines
    return usable, excluded


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 exactly as the script's ``evaluate()``: an empty
    denominator gives 0.0, and F1 is 0.0 when precision + recall = 0."""
    pc, gc = tp + fp, tp + fn
    precision = tp / pc if pc else 0.0
    recall = tp / gc if gc else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_line_metric(
    gold_cases: Sequence[GoldCase],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    legacy_predictions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Score engine lines (raw frame, declared locus) against Gold.

    ``predictions[case_id]`` are line records in the raw-scan frame. A case with
    gold but no prediction entry is an engine miss (all its gold lines are false
    negatives), never a skipped case. Exclusions are decided by
    :func:`prepare_gold` before matching."""
    gold, excluded = prepare_gold(gold_cases)

    def run(preds: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        cases = []
        for cid in sorted(gold):
            res = match_lines(gold[cid], list(preds.get(cid) or []), thresholds)
            res["case_id"] = cid
            cases.append(res)
        tp = sum(c["true_positive"] for c in cases)
        fp = sum(c["false_positive"] for c in cases)
        fn = sum(c["false_negative"] for c in cases)
        exact = sum(1 for c in cases if c["exact"])
        if not cases:  # no usable Gold at all: nothing can be evaluated
            p = r = f1 = None
        else:
            p, r, f1 = _prf(tp, fp, fn)
        return {"cases": cases, "true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": p,
                "recall": r, "f1": f1, "exact_part_count": exact,
                "exact_part_rate": exact / len(cases) if cases else None, "case_count": len(cases),
                "gold_count": tp + fn, "prediction_count": tp + fp}

    engine = run(predictions)
    out: dict[str, Any] = {
        "schema": "sherman_e4_g2_line_metric_v1",
        "label": G2_LINE_THRESHOLDS["label"],
        "declared_locus": G2_LINE_THRESHOLDS["declared_locus"],
        "frame": "raw_scan",
        "thresholds": {"matcher": dict(G2_LINE_THRESHOLDS["matcher"]), "f1_min": G2_LINE_THRESHOLDS["f1_min"],
                       "precision_min": G2_LINE_THRESHOLDS["precision_min"]},
        "excluded_before_scoring": excluded,
        "engine": engine,
    }
    gates = {
        "f1": _gate(engine["f1"], G2_LINE_THRESHOLDS["f1_min"]),
        "precision": _gate(engine["precision"], G2_LINE_THRESHOLDS["precision_min"]),
    }
    if legacy_predictions is not None:
        legacy = run(legacy_predictions)
        out["legacy"] = legacy
        e, lg = engine["exact_part_rate"], legacy["exact_part_rate"]
        gates["exact_part_rate_beats_legacy"] = (
            "NOT_EVALUABLE" if e is None or lg is None else ("PASS" if e > lg else "FAIL"))
    else:
        gates["exact_part_rate_beats_legacy"] = "NOT_EVALUABLE"
    out["gates"] = gates
    out["verdict"] = "PASS" if all(v == "PASS" for v in gates.values()) else (
        "FAIL" if any(v == "FAIL" for v in gates.values()) else "NOT_EVALUABLE")
    return out


def _gate(value: float | None, minimum: float) -> str:
    if value is None:
        return "NOT_EVALUABLE"
    return "PASS" if value >= minimum else "FAIL"
