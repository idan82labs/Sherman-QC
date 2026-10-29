from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts.attribute_f1_transition_probe_gold import (
    _line_record,
    attribute_transition_probe_gold,
)


HASHES = {
    "diagnostic_census_sha256": "a" * 64,
    "prediction_sha256": "b" * 64,
    "gold_sha256": "c" * 64,
    "grouped_split_sha256": "d" * 64,
}


def _line(bend_id: str, y: float) -> dict:
    return {
        "bend_id": bend_id,
        "span_endpoints": [[0.0, y, 0.0], [20.0, y, 0.0]],
    }


def _case_hashes(index: int) -> dict:
    token = f"{index:064x}"
    return {
        "scan_sha256": token,
        "candidate_pool_sha256": f"{index + 50:064x}",
        "graph_sha256": f"{index + 100:064x}",
        "a0_solution_sha256": f"{index + 200:064x}",
    }


def test_zero_length_probe_geometry_is_not_matchable() -> None:
    assert (
        _line_record(
            {
                "component_id": "TC_ZERO",
                "intersection_span_endpoints": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            },
            fallback="TC_ZERO",
        )
        is None
    )


def test_postlock_partition_is_deterministic_and_does_not_mutate_probe() -> None:
    hashes = _case_hashes(1)
    census = {
        "inference_safe": True,
        "cases": [
            {
                "case_token": "C001",
                **hashes,
                "probe_records": [
                    {
                        **_line("PROBE_SHALLOW", 20.0),
                        "probe_record_id": "PROBE_SHALLOW",
                        "first_failed_stage": 2,
                        "reason_code": "NO_RIGHT_SIDE_MODE",
                    },
                    {
                        **_line("PROBE_DEEP", 20.0),
                        "probe_record_id": "PROBE_DEEP",
                        "first_failed_stage": 4,
                        "reason_code": "CONNECTIVITY_BROKEN",
                        "valid_two_mode_connected_run": True,
                        "support_parent_ids": ["P2", "P1"],
                    },
                ],
            }
        ],
    }
    predictions = {
        "inference_safe": True,
        "cases": [
            {
                "case_token": "C001",
                **hashes,
                "a0_predictions": [_line("A0_TP", 0.0), _line("A0_FP", 40.0)],
                "existing_paths": [
                    {
                        **_line("LP_REPRESENTED", 10.0),
                        "latent_path_id": "LP_REPRESENTED",
                        "incidence_complete": True,
                        "label_eligibility": {"BEND_STRAIGHT_COUNTED": True},
                    }
                ],
            }
        ],
    }
    gold = {
        "cases": [
            {
                "case_token": "C001",
                "source_scan_sha256": hashes["scan_sha256"],
                "manual_bends": [
                    _line("G_SELECTED", 0.0),
                    _line("G_REPRESENTED", 10.0),
                    _line("G_UNREPRESENTED", 20.0),
                ],
            }
        ]
    }
    split = {"cases": [{"case_token": "C001", "part_key": "FAMILY_A"}]}
    frozen_census = deepcopy(census)

    first = attribute_transition_probe_gold(
        census=census,
        predictions=predictions,
        sealed_gold=gold,
        grouped_split=split,
        source_hashes=HASHES,
    )
    assert census == frozen_census
    census["cases"][0]["probe_records"].reverse()
    second = attribute_transition_probe_gold(
        census=census,
        predictions=predictions,
        sealed_gold=gold,
        grouped_split=split,
        source_hashes=HASHES,
    )

    assert census != frozen_census  # only the test reordered its input
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [row["first_failed_stage"] for row in first["rows"]] == [7, 4]
    assert first["rows"][0]["reason_code"] == "REPRESENTED_EXISTING_PATH_REJECTED_BY_MAP"
    assert first["rows"][1]["matching_probe_record_ids"] == [
        "PROBE_DEEP",
        "PROBE_SHALLOW",
    ]
    assert first["rows"][1]["matching_parent_ids"] == ["P1", "P2"]
    assert first["summary"]["represented_existing_path_miss_count"] == 1
    assert first["summary"]["candidate_unrepresented_miss_count"] == 1
    assert first["professor_probe_gate"]["metrics"]["recovered_unrepresented_objects"] == 1
    assert first["professor_probe_gate"]["passed"] is False
    assert first["diagnostic_census_sha256"] == "a" * 64


def test_professor_probe_gate_uses_frozen_33_false_baseline() -> None:
    census_cases = []
    prediction_cases = []
    gold_cases = []
    split_cases = []
    gold_index = 0
    for case_index in range(8):
        token = f"C{case_index + 1:03d}"
        hashes = _case_hashes(case_index + 1)
        count = 2 if case_index < 6 else 1  # 14 recovered objects across 8 groups.
        gold_lines = []
        probe_lines = []
        for local_index in range(count):
            y = float(local_index * 10)
            gold_index += 1
            gold_lines.append(_line(f"G{gold_index:03d}", y))
            probe_lines.append(
                {
                    **_line(f"R{gold_index:03d}", y),
                    "probe_record_id": f"R{gold_index:03d}",
                    "first_failed_stage": 5,
                    "valid_two_mode_connected_run": True,
                }
            )
        # One false prediction per group passes a separate certificate: 8/33.
        false_y = 100.0
        probe_lines.append(
            {
                **_line(f"RF{case_index:03d}", false_y),
                "probe_record_id": f"RF{case_index:03d}",
                "first_failed_stage": 5,
                "valid_two_mode_connected_run": True,
            }
        )
        census_cases.append(
            {"case_token": token, **hashes, "probe_records": probe_lines}
        )
        prediction_cases.append(
            {
                "case_token": token,
                **hashes,
                "a0_predictions": [_line(f"FP{case_index:03d}", false_y)],
                "existing_paths": [],
            }
        )
        gold_cases.append(
            {
                "case_token": token,
                "source_scan_sha256": hashes["scan_sha256"],
                "manual_bends": gold_lines,
            }
        )
        split_cases.append(
            {"case_token": token, "split_group_id": f"GROUP_{case_index:02d}"}
        )

    result = attribute_transition_probe_gold(
        census={"inference_safe": True, "cases": census_cases},
        predictions={"inference_safe": True, "cases": prediction_cases},
        sealed_gold={"cases": gold_cases},
        grouped_split={"cases": split_cases},
        source_hashes=HASHES,
    )

    metrics = result["professor_probe_gate"]["metrics"]
    assert metrics["recovered_unrepresented_objects"] == 14
    assert metrics["independent_family_group_count"] == 8
    assert metrics["a0_false_predictions_passing_certificate"] == 8
    assert metrics["a0_false_certificate_fraction_of_frozen_33"] == pytest.approx(8 / 33)
    assert result["professor_probe_gate"]["passed"] is True


def test_hash_or_scan_mismatch_fails_closed() -> None:
    hashes = _case_hashes(1)
    census = {"inference_safe": True, "cases": [{"case_token": "C001", **hashes}]}
    prediction_case = {"case_token": "C001", **hashes}
    prediction_case["graph_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="graph_sha256 mismatch"):
        attribute_transition_probe_gold(
            census=census,
            predictions={"inference_safe": True, "cases": [prediction_case]},
            sealed_gold={
                "cases": [
                    {
                        "case_token": "C001",
                        "source_scan_sha256": hashes["scan_sha256"],
                        "manual_bends": [],
                    }
                ]
            },
            grouped_split={"cases": [{"case_token": "C001", "part_key": "F"}]},
            source_hashes=HASHES,
        )
