"""Pure scoring bridge for explicitly declared finite support-line references.

The geometric matcher below preserves the pure finite-line rule from
scripts/evaluate_f1_against_manual_gold.py; compatibility tests execute that
exact source separately. No corpus/path defaults, I/O, truth loader, or inference
imports are present. A declared reference convention is not its qualification.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


LINE_CONVENTION = "CYLINDER_CENTER_AXIS_SUPPORT"
MAX_LINES = 256


@dataclass(frozen=True)
class Thresholds:
    angle_deg: float = 5.0
    transverse_offset_mm: float = 4.0
    gold_span_overlap: float = 0.60
    prediction_span_overlap: float = 0.60

    def __post_init__(self):
        for name, limit in (("angle_deg", 90.), ("transverse_offset_mm", 1e6),
                            ("gold_span_overlap", 1.), ("prediction_span_overlap", 1.)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= limit):
                raise ValueError(f"invalid matching threshold {name}")


def pair_metrics(gold, prediction):
    """Same unoriented finite-line metrics as the retained corpus matcher."""
    gold_midpoint = np.mean(gold, axis=0)
    prediction_midpoint = np.mean(prediction, axis=0)
    gold_delta = gold[1] - gold[0]
    prediction_delta = prediction[1] - prediction[0]
    gold_length = float(np.linalg.norm(gold_delta))
    prediction_length = float(np.linalg.norm(prediction_delta))
    if gold_length <= 1e-9 or prediction_length <= 1e-9:
        raise ValueError("bend lines must have nonzero length")
    gold_axis = gold_delta / gold_length
    prediction_axis = prediction_delta / prediction_length
    dot = float(np.clip(np.dot(gold_axis, prediction_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    if dot < 0:
        prediction_axis *= -1
    midpoint_delta = prediction_midpoint - gold_midpoint
    gold_axis_offset = float(np.linalg.norm(midpoint_delta - gold_axis * np.dot(midpoint_delta, gold_axis)))
    prediction_axis_offset = float(np.linalg.norm(
        midpoint_delta - prediction_axis * np.dot(midpoint_delta, prediction_axis)))
    transverse_offset = max(gold_axis_offset, prediction_axis_offset)
    gold_interval = sorted(float(np.dot(point - gold_midpoint, gold_axis)) for point in gold)
    prediction_interval = sorted(float(np.dot(point - gold_midpoint, gold_axis)) for point in prediction)
    overlap_length = max(0.0, min(gold_interval[1], prediction_interval[1])
                         - max(gold_interval[0], prediction_interval[0]))
    return {"axis_angle_deg": angle, "transverse_offset_mm": transverse_offset,
            "gold_span_overlap": overlap_length / gold_length,
            "prediction_span_overlap": overlap_length / prediction_length,
            "shorter_span_overlap": overlap_length / min(gold_length, prediction_length),
            "gold_length_mm": gold_length, "prediction_length_mm": prediction_length}


def _match_lines(gold_records, prediction_records, thresholds):
    gold = [(record["bend_id"], np.asarray([record["line_start"], record["line_end"]], dtype=float))
            for record in gold_records]
    predictions = [(record["bend_id"], np.asarray([record["line_start"], record["line_end"]], dtype=float))
                   for record in prediction_records]
    matrix = np.full((len(gold), len(predictions)), 1e9, dtype=np.float64)
    pair_cache = {}
    for gi, (_, gold_line) in enumerate(gold):
        for pi, (_, prediction_line) in enumerate(predictions):
            metrics = pair_metrics(gold_line, prediction_line)
            pair_cache[(gi, pi)] = metrics
            if (metrics["axis_angle_deg"] <= thresholds.angle_deg
                    and metrics["transverse_offset_mm"] <= thresholds.transverse_offset_mm
                    and metrics["gold_span_overlap"] >= thresholds.gold_span_overlap
                    and metrics["prediction_span_overlap"] >= thresholds.prediction_span_overlap):
                matrix[gi, pi] = (
                    metrics["axis_angle_deg"] / thresholds.angle_deg
                    + metrics["transverse_offset_mm"] / thresholds.transverse_offset_mm
                    + (1.0 - min(metrics["gold_span_overlap"], 1.0))
                    + (1.0 - min(metrics["prediction_span_overlap"], 1.0)))
    matches, matched_gold, matched_predictions = [], set(), set()
    if len(gold) and len(predictions):
        gi_values, pi_values = linear_sum_assignment(matrix)
        for gi, pi in zip(gi_values, pi_values):
            if matrix[gi, pi] >= 1e9:
                continue
            matched_gold.add(int(gi))
            matched_predictions.add(int(pi))
            matches.append({"gold_id": gold[gi][0], "prediction_id": predictions[pi][0],
                            **{key: round(value, 6) for key, value in pair_cache[(gi, pi)].items()}})
    return {"gold_count": len(gold), "prediction_count": len(predictions),
            "true_positive": len(matches), "false_positive": len(predictions) - len(matches),
            "false_negative": len(gold) - len(matches),
            "exact": len(matches) == len(gold) == len(predictions), "matches": matches,
            "unmatched_gold_ids": [row[0] for i, row in enumerate(gold) if i not in matched_gold],
            "unmatched_prediction_ids": [row[0] for i, row in enumerate(predictions)
                                         if i not in matched_predictions]}


def _lines(records, *, reference_frame, predictions=False):
    if not isinstance(records, (list, tuple)) or len(records) > MAX_LINES:
        raise ValueError("line records require a bounded list")
    ids, lines = set(), []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("line records must be dictionaries")
        bend_id = record.get("bend_id")
        if not isinstance(bend_id, str) or not bend_id.strip() or bend_id in ids:
            raise ValueError("line identities must be nonempty and unique")
        ids.add(bend_id)
        if record.get("line_convention", LINE_CONVENTION) != LINE_CONVENTION:
            raise ValueError("per-line convention mismatch")
        if record.get("unit", "mm") != "mm":
            raise ValueError("per-line unit mismatch")
        if record.get("reference_frame", reference_frame) != reference_frame:
            raise ValueError("per-line frame mismatch")
        if predictions and (record.get("line_convention") != LINE_CONVENTION
                            or record.get("unit") != "mm"
                            or record.get("observation_status") != "SAMPLE_SUPPORTED_CONDITIONAL"
                            or record.get("endpoint_semantics") != "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM"
                            or record.get("physical_trim_endpoints") is not None):
            raise ValueError("prediction is not an accepted sampled-support instance")
        raw = np.asarray([record.get("line_start"), record.get("line_end")])
        if raw.shape != (2, 3) or raw.dtype.kind not in "fiu":
            raise ValueError("finite explicit line_start/line_end are required")
        value = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(value).all() or float(np.max(np.abs(value))) > 1e6:
            raise ValueError("invalid line numerics")
        if float(np.linalg.norm(value[1] - value[0])) <= 1e-9:
            raise ValueError("bend lines must have nonzero length")
        lines.append({"bend_id": bend_id, "line_start": value[0].tolist(),
                      "line_end": value[1].tolist()})
    return lines


def score_support_case(*, case_id, reference_records, prediction_payload,
                       reference_frame, reference_unit, reference_convention,
                       thresholds=None):
    """Score supplied references AFTER inference under an explicit same-frame rule.

    No nearest-CAD-feature join or reference-based filtering is performed.
    PARTIAL cases retain accepted predictions and their TP/FP/FN, but neither
    partial nor unavailable inference earns exact count / exact-set credit.
    This includes unavailable inference when reference count happens to be zero.
    """
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be nonempty")
    if not isinstance(prediction_payload, dict):
        raise ValueError("prediction_payload must be a dictionary")
    if not isinstance(reference_frame, str) or not reference_frame.strip():
        raise ValueError("reference_frame must be explicitly declared")
    if reference_convention != LINE_CONVENTION or reference_unit != "mm":
        raise ValueError("unsupported or undeclared reference convention/unit")
    if (prediction_payload.get("reference_frame") != reference_frame
            or prediction_payload.get("unit") != reference_unit
            or prediction_payload.get("line_convention") != reference_convention):
        raise ValueError("reference/prediction frame, unit, or line convention mismatch")
    status = prediction_payload.get("instance_extraction_status")
    if status not in {"COMPLETE", "PARTIAL", "UNAVAILABLE"}:
        raise ValueError("explicit instance extraction status is required")
    references = _lines(reference_records, reference_frame=reference_frame)
    predictions = _lines(prediction_payload.get("finite_support_instances"),
                         reference_frame=reference_frame, predictions=True)
    if status == "UNAVAILABLE" and predictions:
        raise ValueError("unavailable inference cannot publish predictions")
    if thresholds is None:
        thresholds = Thresholds()
    if not isinstance(thresholds, Thresholds):
        raise ValueError("thresholds must be Thresholds")
    result = _match_lines(references, predictions, thresholds)
    # Keep the retained matcher's raw flag visible; qualify scored success by
    # execution status rather than changing its geometric matching rule.
    return {**result, "case_id": case_id, "instance_extraction_status": status,
            "raw_exact_localized_set": result["exact"],
            "exact": status == "COMPLETE" and result["exact"],
            "exact_count": status == "COMPLETE" and len(references) == len(predictions),
            "exact_localized_target_set": status == "COMPLETE" and result["exact"],
            "reference_frame": reference_frame, "unit": reference_unit,
            "line_convention": reference_convention,
            "reference_contract_status": "DECLARED_NOT_PHYSICALLY_QUALIFIED",
            "physical_bend_count": None, "promotion_eligible": False}


def score_support_cases(rows, *, expected_case_ids):
    """Aggregate every requested case, including failed inference denominators.

    Count/F1 arithmetic is the retained score_localized_cases rule, with explicit
    status qualification added only to the exact-case success numerators.
    Undefined ratios remain None as in that scorer, rather than becoming a pass.
    """
    if (not isinstance(rows, (list, tuple)) or not rows or len(rows) > 4096
            or not isinstance(expected_case_ids, (list, tuple))
            or not 1 <= len(expected_case_ids) <= 4096
            or any(not isinstance(value, str) or not value.strip() for value in expected_case_ids)
            or len(set(expected_case_ids)) != len(expected_case_ids)):
        raise ValueError("at least one scored case is required")
    ids, tp, fp, fn, exact_count, exact_set, complete = set(), 0, 0, 0, 0, 0, 0
    contract = None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("scored rows must be dictionaries")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case_id must be present and unique")
        ids.add(case_id)
        row_contract = (row.get("reference_frame"), row.get("unit"), row.get("line_convention"))
        if (not isinstance(row_contract[0], str) or not row_contract[0].strip()
                or row_contract[1:] != ("mm", LINE_CONVENTION)):
            raise ValueError("scored rows require explicit frame/unit/convention")
        if contract is None:
            contract = row_contract
        elif row_contract != contract:
            raise ValueError("mixed scoring frame/unit/convention")
        for key in ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if (row["true_positive"] + row["false_negative"] != row["gold_count"]
                or row["true_positive"] + row["false_positive"] != row["prediction_count"]):
            raise ValueError("inconsistent matched counts")
        status = row.get("instance_extraction_status")
        if status not in {"COMPLETE", "PARTIAL", "UNAVAILABLE"}:
            raise ValueError("explicit instance extraction status is required")
        if status == "UNAVAILABLE" and row["prediction_count"]:
            raise ValueError("unavailable inference cannot publish predictions")
        tp += row["true_positive"]
        fp += row["false_positive"]
        fn += row["false_negative"]
        complete += int(status == "COMPLETE")
        exact_count += int(status == "COMPLETE" and row["prediction_count"] == row["gold_count"])
        exact_set += int(status == "COMPLETE" and row["true_positive"] == row["gold_count"]
                         == row["prediction_count"] and row["false_positive"] == row["false_negative"] == 0)
    if ids != set(expected_case_ids):
        raise ValueError("scored cases do not match the complete requested case set")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"case_count": len(rows), "complete_case_count": complete,
            "inference_complete_rate": complete / len(rows),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "exact_count_cases": exact_count, "exact_count_accuracy": exact_count / len(rows),
            "exact_localized_target_set_cases": exact_set,
            "exact_localized_target_set_accuracy": exact_set / len(rows),
            "physical_bend_count": None, "promotion_eligible": False}
