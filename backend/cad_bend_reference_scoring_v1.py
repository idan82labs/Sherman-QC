"""Pure native-face reference scoring; no geometry or physical input access.

Exact face-set identity is the deliberately frozen V1 matching rule. Partial
references support verified-positive recall only, never a full precision/F1.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def _ids(value, field, allowed):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    if any(type(item) is not int or item < 1 for item in value):
        raise ValueError(f"{field} must contain positive integer native face IDs")
    if len(set(value)) != len(value) or not set(value) <= allowed:
        raise ValueError(f"{field} contains duplicate or invalid native face IDs")
    return frozenset(value)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_parts(parts: Sequence[Mapping]) -> dict:
    """Score supplied independent references and predictions deterministically.

    Each part supplies part_id, face_ids, reference_status, reference_bends,
    unknown_face_ids, verified_nonbend_face_ids, predictions and optionally
    prediction_status/failure_reason. The three ownership domains partition the
    full face census; complete is an independent feature-completeness assertion,
    not inferred from face coverage. Prediction failure must include an empty
    predictions list and remains in every denominator.
    Each reference bend owns disjoint nonempty inner_face_ids/outer_face_ids.
    Each prediction supplies prediction_id and its complete face_ids group.
    Hash/provenance validation is the caller's responsibility.
    """
    results = []
    seen_parts = set()
    totals = {"tp": 0, "fp": 0, "fn": 0}
    partial_tp = partial_reference_count = fully_referenced = exact_count = 0
    for part in parts:
        part_id = part.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in seen_parts:
            raise ValueError("part_id must be unique nonempty text")
        seen_parts.add(part_id)
        raw_faces = part.get("face_ids")
        if not isinstance(raw_faces, (list, tuple)):
            raise ValueError("face_ids must be a list")
        faces = _ids(raw_faces, "face_ids", set(raw_faces))
        if not faces:
            raise ValueError("native face census cannot be empty")
        status = part.get("reference_status")
        if status not in ("complete", "partial"):
            raise ValueError("reference_status must be complete or partial")
        unknown = _ids(part.get("unknown_face_ids"), "unknown_face_ids", faces)
        if status == "complete" and unknown:
            raise ValueError("complete reference cannot contain unknown faces")
        if status == "partial" and not unknown:
            raise ValueError("partial reference must declare unknown faces")
        references = {}
        occupied = set()
        for bend in part.get("reference_bends", []):
            identity = bend.get("bend_id")
            if not isinstance(identity, str) or not identity or identity in references:
                raise ValueError("bend_id must be unique nonempty text")
            inner = _ids(bend.get("inner_face_ids"), "inner_face_ids", faces)
            outer = _ids(bend.get("outer_face_ids"), "outer_face_ids", faces)
            group = inner | outer
            if not inner or not outer or inner & outer or group & occupied or group & unknown:
                raise ValueError("reference bend face ownership is empty, overlapping or unknown")
            references[identity] = group
            occupied.update(group)
        nonbend = _ids(part.get("verified_nonbend_face_ids"), "verified_nonbend_face_ids", faces)
        if nonbend & (occupied | unknown) or occupied | set(nonbend) | set(unknown) != set(faces):
            raise ValueError("reference bend, verified nonbend and unknown faces must partition the census")
        predictions = {}
        for prediction in part.get("predictions", []):
            identity = prediction.get("prediction_id")
            if not isinstance(identity, str) or not identity or identity in predictions:
                raise ValueError("prediction_id must be unique nonempty text")
            group = _ids(prediction.get("face_ids"), "prediction face_ids", faces)
            if not group:
                raise ValueError("prediction face group cannot be empty")
            predictions[identity] = group
        prediction_status = part.get("prediction_status", "ok")
        if prediction_status not in ("ok", "failed"):
            raise ValueError("prediction_status must be ok or failed")
        if prediction_status == "failed" and predictions:
            raise ValueError("failed extraction cannot carry predictions")
        unmatched_references = set(references)
        matches = []
        unmatched_predictions = []
        for prediction_id in sorted(predictions):
            candidates = sorted(identity for identity in unmatched_references
                                if references[identity] == predictions[prediction_id])
            if candidates:
                reference_id = candidates[0]
                unmatched_references.remove(reference_id)
                matches.append({"prediction_id": prediction_id, "bend_id": reference_id})
            else:
                unmatched_predictions.append(prediction_id)
        tp = len(matches)
        complete = status == "complete"
        if complete:
            fully_referenced += 1
            totals["tp"] += tp
            totals["fp"] += len(unmatched_predictions)
            totals["fn"] += len(unmatched_references)
            exact_count += int(prediction_status == "ok" and len(predictions) == len(references))
        else:
            partial_tp += tp
            partial_reference_count += len(references)
        results.append({
            "part_id": part_id, "reference_status": status,
            "prediction_status": prediction_status, "failure_reason": part.get("failure_reason"),
            "reference_count": len(references), "prediction_count": len(predictions),
            "unknown_face_ids": sorted(unknown), "matches": matches, "tp": tp,
            "native_face_count": len(faces), "unknown_face_count": len(unknown),
            "fp": len(unmatched_predictions) if complete else None,
            "fn": len(unmatched_references) if complete else None,
            "precision": _ratio(tp, len(predictions)) if complete else None,
            "recall": _ratio(tp, len(references)) if complete else None,
            "f1": _ratio(2 * tp, len(predictions) + len(references)) if complete else None,
            "unmatched_reference_ids": sorted(unmatched_references),
            "unmatched_prediction_ids": unmatched_predictions,
            "unscored_prediction_ids": unmatched_predictions if not complete else [],
            "verified_positive_recall": _ratio(tp, len(references)),
            "exact_count_correct": (prediction_status == "ok" and len(predictions) == len(references)) if complete else None,
        })
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        "schema": "cad_bend_reference_score_v1", "matching_rule": "exact_native_face_union_one_to_one",
        "geometrical_location_angle_accuracy": "unscored",
        "exact_count_semantics": "count_agreement_only_not_detection_correctness",
        "partial_recall_semantics": "verified_reference_subset_only_not_full_recall",
        "coverage": {"total_parts": len(results), "fully_referenced_parts": fully_referenced,
                     "partial_parts": len(results) - fully_referenced,
                     "full_reference_part_fraction": _ratio(fully_referenced, len(results))},
        "full_reference_micro": {**totals, "reference_bend_denominator": tp + fn,
                                 "prediction_denominator": tp + fp,
                                 "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                                 "f1": _ratio(2 * tp, 2 * tp + fp + fn)},
        "full_reference_macro": {
            metric: {"value": _ratio(sum(r[metric] for r in results if r[metric] is not None),
                                      sum(r[metric] is not None for r in results)),
                     "defined_part_denominator": sum(r[metric] is not None for r in results)}
            for metric in ("precision", "recall", "f1")
        },
        "partial_verified_positive": {"matched": partial_tp, "reference_denominator": partial_reference_count,
                                      "recall": _ratio(partial_tp, partial_reference_count)},
        "all_verified_positive_recall": {
            "matched": tp + partial_tp,
            "reference_denominator": tp + fn + partial_reference_count,
            "recall": _ratio(tp + partial_tp, tp + fn + partial_reference_count),
            "semantics": "known_reference_positives_only_not_full_corpus_recall",
            "precision": _ratio(tp, tp + fp) if fully_referenced == len(results) else None,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn) if fully_referenced == len(results) else None,
        },
        "exact_count_accuracy": {"correct": exact_count, "part_denominator": fully_referenced,
                                 "accuracy": _ratio(exact_count, fully_referenced)},
        "parts": results,
        "failure_rates": {"refusal_count": sum(r["prediction_status"] == "failed" for r in results),
                          "refusal_rate": _ratio(sum(r["prediction_status"] == "failed" for r in results), len(results)),
                          "zero_output_count": sum(r["prediction_count"] == 0 for r in results),
                          "zero_output_rate": _ratio(sum(r["prediction_count"] == 0 for r in results), len(results))},
        "face_coverage": {"native_face_count": sum(r["native_face_count"] for r in results),
                          "unknown_face_count": sum(r["unknown_face_count"] for r in results)},
    }
