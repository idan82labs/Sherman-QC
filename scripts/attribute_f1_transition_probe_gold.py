#!/usr/bin/env python3
"""Post-lock gold attribution for the frozen F1 transition-mode probe.

The inference-safe census and prediction bundle must already exist and carry
matching scan/graph/A0-solution hashes.  This script never modifies or
regenerates those records.  Gold is opened only to perform the frozen line
matching and to label the first failed diagnostic stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


KIND = "postlock_gold_transition_failure_attribution"
A0_FALSE_PREDICTION_BASELINE = 33
MIN_RECOVERED_UNREPRESENTED = 14
MIN_INDEPENDENT_GROUPS = 8
MAX_FALSE_CERTIFICATE_FRACTION = 0.25
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_cases(payload: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("cases") or ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case records must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        result[token] = dict(raw)
    return result


def _require_hash(value: Any, *, field: str) -> str:
    result = str(value or "").lower()
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _record_id(record: Mapping[str, Any], *, fallback: str) -> str:
    return str(
        record.get("bend_id")
        or record.get("latent_path_id")
        or record.get("probe_record_id")
        or record.get("component_id")
        or record.get("birth_eligibility_id")
        or fallback
    )


def _line_record(record: Mapping[str, Any], *, fallback: str) -> dict[str, Any] | None:
    geometry = record.get("geometry")
    source = dict(geometry) if isinstance(geometry, Mapping) else dict(record)
    if source.get("span_endpoints") is None and record.get("intersection_span_endpoints") is not None:
        source["span_endpoints"] = deepcopy(record["intersection_span_endpoints"])
    has_line = (
        source.get("span_endpoints") is not None
        or (source.get("line_start") is not None and source.get("line_end") is not None)
        or (
            source.get("anchor") is not None
            and source.get("axis_direction") is not None
            and source.get("span_mm") is not None
        )
    )
    if not has_line:
        return None
    if source.get("span_endpoints") is not None:
        endpoints = source["span_endpoints"]
        if not isinstance(endpoints, Sequence) or len(endpoints) != 2:
            return None
        delta = [float(endpoints[1][index]) - float(endpoints[0][index]) for index in range(3)]
        if not all(math.isfinite(value) for value in delta) or sum(value * value for value in delta) <= 1e-18:
            return None
    elif source.get("line_start") is not None and source.get("line_end") is not None:
        delta = [
            float(source["line_end"][index]) - float(source["line_start"][index])
            for index in range(3)
        ]
        if not all(math.isfinite(value) for value in delta) or sum(value * value for value in delta) <= 1e-18:
            return None
    elif float(source.get("span_mm") or 0.0) <= 1e-9:
        return None
    source["bend_id"] = _record_id(record, fallback=fallback)
    return source


def _probe_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_records: list[Mapping[str, Any]] = []
    for field in (
        "probe_records",
        "birth_eligibility_records",
        "transition_component_records",
        "plane_pair_records",
    ):
        values = case.get(field) or ()
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            raw_records.extend(value for value in values if isinstance(value, Mapping))
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        record = deepcopy(dict(raw))
        record_id = _record_id(record, fallback=f"PROBE_{index:06d}")
        if record_id in seen:
            continue
        seen.add(record_id)
        line = _line_record(record, fallback=record_id)
        if line is None:
            continue
        record["probe_record_id"] = record_id
        record["_line"] = line
        result.append(record)
    return sorted(result, key=lambda item: str(item["probe_record_id"]))


def _path_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = case.get("existing_paths") or case.get("latent_paths") or ()
    result = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping) or not bool(raw.get("incidence_complete")):
            continue
        eligibility = raw.get("label_eligibility")
        if isinstance(eligibility, Mapping) and not bool(
            eligibility.get("BEND_STRAIGHT_COUNTED")
        ):
            continue
        line = _line_record(raw, fallback=f"PATH_{index:06d}")
        if line is not None:
            result.append(line)
    return sorted(result, key=lambda item: str(item["bend_id"]))


def _prediction_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = case.get("a0_predictions") or case.get("predictions") or ()
    result = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            continue
        line = _line_record(raw, fallback=f"PRED_{index:06d}")
        if line is not None:
            result.append(line)
    return sorted(result, key=lambda item: str(item["bend_id"]))


def _matching_ids(
    gold: Mapping[str, Any], records: Sequence[Mapping[str, Any]], thresholds: Thresholds
) -> list[str]:
    result = []
    for record in records:
        match = match_lines([dict(gold)], [dict(record)], thresholds)
        if match["true_positive"] == 1:
            result.append(str(record["bend_id"]))
    return sorted(set(result))


def _record_stage(record: Mapping[str, Any]) -> int:
    raw = record.get("first_failed_stage")
    if raw is None:
        raw = record.get("failure_stage")
    if raw is None and record.get("passed_stage_count") is not None:
        raw = int(record["passed_stage_count"]) + 1
    if raw is None and _certificate_passed(record):
        raw = 6
    stage = int(raw or 1)
    if stage < 1 or stage > 6:
        raise ValueError("probe first_failed_stage must be in [1,6]")
    return stage


def _certificate_passed(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("valid_two_mode_connected_run")
        or record.get("transition_certificate_passed")
        or record.get("valid_transition_certificate")
    )


def _matching_probe_records(
    gold: Mapping[str, Any], records: Sequence[Mapping[str, Any]], thresholds: Thresholds
) -> list[dict[str, Any]]:
    matched = []
    for record in records:
        if match_lines([dict(gold)], [dict(record["_line"])], thresholds)[
            "true_positive"
        ] == 1:
            matched.append(record)
    # Prefer the record that progressed furthest through the fixed cascade.
    return sorted(
        matched,
        key=lambda item: (-_record_stage(item), str(item["probe_record_id"])),
    )


def _bounded_ids(record: Mapping[str, Any], *fields: str) -> list[str]:
    return sorted(
        {
            str(value)
            for field in fields
            for value in (record.get(field) or ())
            if str(value)
        }
    )


def attribute_transition_probe_gold(
    *,
    census: Mapping[str, Any],
    predictions: Mapping[str, Any],
    sealed_gold: Mapping[str, Any],
    grouped_split: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Join frozen inference records to gold without mutating inference state."""

    if census.get("inference_safe") is not True:
        raise ValueError("diagnostic census must declare inference_safe=true")
    if census.get("truth_inputs_consumed") not in (None, False):
        raise ValueError("diagnostic census must not consume truth")
    if predictions.get("inference_safe") is not True:
        raise ValueError("prediction bundle must declare inference_safe=true")
    hashes = {
        key: _require_hash(source_hashes.get(key), field=key)
        for key in (
            "diagnostic_census_sha256",
            "prediction_sha256",
            "gold_sha256",
            "grouped_split_sha256",
        )
    }

    census_cases = _index_cases(census, label="census")
    prediction_cases = _index_cases(predictions, label="prediction")
    gold_cases = _index_cases(sealed_gold, label="gold")
    split_cases = _index_cases(grouped_split, label="grouped split")
    case_tokens = set(gold_cases)
    if case_tokens != set(census_cases) or case_tokens != set(prediction_cases) or case_tokens != set(split_cases):
        raise ValueError("census, prediction, gold, and split case-token sets must match")

    thresholds = Thresholds()
    rows: list[dict[str, Any]] = []
    recovered_unrepresented: list[tuple[str, str, str]] = []
    false_certificate_pass_count = 0
    observed_false_predictions = 0

    for token in sorted(case_tokens):
        census_case = census_cases[token]
        prediction_case = prediction_cases[token]
        gold_case = gold_cases[token]
        split_case = split_cases[token]
        for field in (
            "scan_sha256",
            "candidate_pool_sha256",
            "graph_sha256",
            "a0_solution_sha256",
        ):
            left = _require_hash(census_case.get(field), field=f"{token}.census.{field}")
            right = _require_hash(prediction_case.get(field), field=f"{token}.prediction.{field}")
            if left != right:
                raise ValueError(f"{token}: frozen {field} mismatch")
        if str(gold_case.get("source_scan_sha256") or "").lower() != str(
            census_case["scan_sha256"]
        ).lower():
            raise ValueError(f"{token}: sealed-gold scan hash mismatch")

        gold_records = sorted(
            (dict(item) for item in gold_case.get("manual_bends") or ()),
            key=lambda item: str(item.get("bend_id") or ""),
        )
        a0_predictions = _prediction_records(prediction_case)
        a0_match = match_lines(gold_records, a0_predictions, thresholds)
        missed_ids = set(a0_match["unmatched_gold_ids"])
        false_ids = set(a0_match["unmatched_prediction_ids"])
        existing_paths = _path_records(prediction_case)
        probe_records = _probe_records(census_case)
        certificate_lines = [
            dict(record["_line"])
            for record in probe_records
            if _certificate_passed(record)
        ]
        false_predictions = [
            record for record in a0_predictions if str(record["bend_id"]) in false_ids
        ]
        observed_false_predictions += len(false_predictions)
        false_certificate_pass_count += int(
            match_lines(false_predictions, certificate_lines, thresholds)["true_positive"]
        )

        group_id = str(
            split_case.get("split_group_id")
            or split_case.get("part_key")
            or split_case.get("part_family")
            or ""
        )
        if not group_id:
            raise ValueError(f"{token}: grouped split lacks an independent group ID")

        for gold in gold_records:
            gold_id = str(gold["bend_id"])
            if gold_id not in missed_ids:
                continue
            matching_paths = _matching_ids(gold, existing_paths, thresholds)
            if matching_paths:
                rows.append(
                    {
                        "case_token": token,
                        "split_group_id": group_id,
                        "gold_id": gold_id,
                        "first_failed_stage": 7,
                        "reason_code": "REPRESENTED_EXISTING_PATH_REJECTED_BY_MAP",
                        "matching_path_ids": matching_paths,
                        "matching_probe_record_ids": [],
                        "matching_plane_pair_ids": [],
                        "matching_parent_ids": [],
                        "matching_component_ids": [],
                        "valid_two_mode_connected_run": False,
                        "bounded_counters": {},
                    }
                )
                continue

            matching_probe = _matching_probe_records(gold, probe_records, thresholds)
            selected = matching_probe[0] if matching_probe else None
            if selected is None:
                stage = 1
                reason = "NO_MATCHING_INFERENCE_SAFE_PROBE_RECORD"
                certificate = False
                plane_ids: list[str] = []
                parent_ids: list[str] = []
                component_ids: list[str] = []
                counters: dict[str, Any] = {}
            else:
                stage = _record_stage(selected)
                reason = str(
                    selected.get("reason_code")
                    or selected.get("first_rejection_code")
                    or f"PROBE_STAGE_{stage}_FAILURE"
                )
                certificate = any(_certificate_passed(record) for record in matching_probe)
                plane_ids = _bounded_ids(
                    selected, "matching_plane_pair_ids", "plane_pair_ids", "plane_ids"
                )
                parent_ids = _bounded_ids(
                    selected, "matching_parent_ids", "support_parent_ids", "parent_ids"
                )
                component_ids = _bounded_ids(
                    selected, "matching_component_ids", "component_ids"
                )
                counters = deepcopy(dict(selected.get("bounded_counters") or {}))
            if certificate:
                recovered_unrepresented.append((token, gold_id, group_id))
            rows.append(
                {
                    "case_token": token,
                    "split_group_id": group_id,
                    "gold_id": gold_id,
                    "first_failed_stage": stage,
                    "reason_code": reason,
                    "matching_path_ids": [],
                    "matching_probe_record_ids": [
                        str(record["probe_record_id"]) for record in matching_probe
                    ],
                    "matching_plane_pair_ids": plane_ids,
                    "matching_parent_ids": parent_ids,
                    "matching_component_ids": component_ids,
                    "valid_two_mode_connected_run": certificate,
                    "bounded_counters": counters,
                }
            )

    rows.sort(key=lambda item: (item["case_token"], item["gold_id"]))
    recovered_count = len(recovered_unrepresented)
    recovered_groups = sorted({group for _token, _gold, group in recovered_unrepresented})
    false_fraction = false_certificate_pass_count / A0_FALSE_PREDICTION_BASELINE
    criteria = {
        "recovered_unrepresented_objects": recovered_count >= MIN_RECOVERED_UNREPRESENTED,
        "independent_family_groups": len(recovered_groups) >= MIN_INDEPENDENT_GROUPS,
        "a0_false_certificate_fraction": false_fraction <= MAX_FALSE_CERTIFICATE_FRACTION,
    }
    stage_counts = Counter(int(row["first_failed_stage"]) for row in rows)
    return {
        "schema_version": 1,
        "kind": KIND,
        "research_only": True,
        "postlock": True,
        "prediction_sha256": hashes["prediction_sha256"],
        "diagnostic_census_sha256": hashes["diagnostic_census_sha256"],
        "gold_sha256": hashes["gold_sha256"],
        "grouped_split_sha256": hashes["grouped_split_sha256"],
        "line_matching_contract": {
            "axis_angle_deg": thresholds.angle_deg,
            "transverse_offset_mm": thresholds.transverse_offset_mm,
            "gold_span_overlap": thresholds.gold_span_overlap,
            "prediction_span_overlap": thresholds.prediction_span_overlap,
        },
        "summary": {
            "a0_missed_gold_count": len(rows),
            "stage_counts": {
                str(stage): int(stage_counts.get(stage, 0)) for stage in range(1, 8)
            },
            "represented_existing_path_miss_count": int(stage_counts.get(7, 0)),
            "candidate_unrepresented_miss_count": len(rows) - int(stage_counts.get(7, 0)),
        },
        "professor_probe_gate": {
            "thresholds": {
                "minimum_recovered_unrepresented_objects": MIN_RECOVERED_UNREPRESENTED,
                "minimum_independent_family_groups": MIN_INDEPENDENT_GROUPS,
                "a0_false_prediction_baseline": A0_FALSE_PREDICTION_BASELINE,
                "maximum_false_certificate_fraction": MAX_FALSE_CERTIFICATE_FRACTION,
            },
            "metrics": {
                "recovered_unrepresented_objects": recovered_count,
                "independent_family_group_count": len(recovered_groups),
                "independent_family_group_ids": recovered_groups,
                "observed_a0_false_prediction_count": observed_false_predictions,
                "a0_false_predictions_passing_certificate": false_certificate_pass_count,
                "a0_false_certificate_fraction_of_frozen_33": round(false_fraction, 12),
            },
            "criteria": criteria,
            "passed": all(criteria.values()),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--sealed-gold", required=True, type=Path)
    parser.add_argument("--grouped-split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "diagnostic_census_sha256": args.census.resolve(),
        "prediction_sha256": args.predictions.resolve(),
        "gold_sha256": args.sealed_gold.resolve(),
        "grouped_split_sha256": args.grouped_split.resolve(),
    }
    result = attribute_transition_probe_gold(
        census=_read(paths["diagnostic_census_sha256"]),
        predictions=_read(paths["prediction_sha256"]),
        sealed_gold=_read(paths["gold_sha256"]),
        grouped_split=_read(paths["grouped_split_sha256"]),
        source_hashes={key: _sha256(path) for key, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "row_count": len(result["rows"]),
                "professor_probe_gate_passed": result["professor_probe_gate"]["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
