from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.attribute_f1_transition_probe_gold import attribute_transition_probe_gold
from scripts.build_f1_transition_probe_postlock_bundles import (
    build_postlock_bundles,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(identifier: str, y: float) -> dict:
    return {
        "bend_id": identifier,
        "span_endpoints": [[0.0, y, 0.0], [20.0, y, 0.0]],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    structured = tmp_path / "structured"
    manifest_rows = []
    split_rows = []
    for index, token in enumerate(("C002", "C001"), start=1):
        root = structured / "cases" / token
        graph_path = root / "typed_union_graph.json"
        solution_path = root / "cpsat_solution.json"
        prediction_path = root / "prediction_owned_regions.json"
        graph = {
            "latent_paths": [
                {
                    "latent_path_id": "LP_Z",
                    "incidence_complete": False,
                    "label_eligibility": {"BEND_STRAIGHT_COUNTED": True},
                    **_line("IGNORED", 30.0),
                },
                {
                    "latent_path_id": "LP_B",
                    "incidence_complete": True,
                    "label_eligibility": {"BEND_STRAIGHT_COUNTED": False},
                    **_line("IGNORED_TOO", 20.0),
                },
                {
                    "latent_path_id": "LP_A",
                    "incidence_complete": True,
                    "label_eligibility": {"BEND_STRAIGHT_COUNTED": True},
                    **_line("LP_A_LINE", 10.0),
                },
            ]
        }
        solution = {"solver": {"status": "OPTIMAL"}, "atom_owners": []}
        prediction = {
            "owned_bend_regions": [_line("P2", 2.0), _line("P1", 1.0)]
        }
        _write(graph_path, graph)
        _write(solution_path, solution)
        _write(prediction_path, prediction)
        case_hashes = {
            "scan_sha256": f"{index:064x}",
            "candidate_pool_sha256": f"{index + 10:064x}",
            "graph_sha256": _sha(graph_path),
            "a0_solution_sha256": _sha(solution_path),
        }
        census_path = tmp_path / "probe" / f"{token}.json"
        _write(
            census_path,
            {
                "schema_version": 1,
                "kind": "f1_transition_diagnostic_census_v1",
                "inference_safe": True,
                "truth_inputs_consumed": False,
                "prediction_mutated": False,
                **case_hashes,
                "plane_pair_records": [{"plane_pair_id": "PP1"}],
                "multimodal_parent_records": [],
                "transition_component_records": [
                    {
                        "component_id": "TC1",
                        "intersection_span_endpoints": [[0.0, 5.0, 0.0], [20.0, 5.0, 0.0]],
                        "first_failed_stage": 4,
                    }
                ],
                "birth_eligibility_records": [
                    {
                        "birth_diagnostic_id": "BD1",
                        "component_id": "TC1",
                        "intersection_span_endpoints": [[0.0, 5.0, 0.0], [20.0, 5.0, 0.0]],
                        "first_failed_stage": 5,
                        "valid_two_mode_connected_run": True,
                    }
                ],
            },
        )
        manifest_rows.append(
            {
                "case_token": token,
                "path": str(census_path),
                "output_sha256": _sha(census_path),
                **case_hashes,
            }
        )
        split_rows.append(
            {"case_token": token, "part_key": f"FAMILY_{index}"}
        )
    manifest_path = tmp_path / "probe" / "manifest.json"
    split_path = tmp_path / "grouped_split.json"
    _write(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "f1_transition_mode_probe_batch",
            "inference_safe": True,
            "birth_columns_enabled": False,
            "prediction_changed": False,
            "case_count": 2,
            "cases": manifest_rows,
        },
    )
    _write(split_path, {"cases": list(reversed(split_rows))})
    return manifest_path, structured, split_path


def test_bundles_are_verified_deterministic_and_directly_consumable(tmp_path: Path) -> None:
    manifest, structured, split = _fixture(tmp_path)
    first = build_postlock_bundles(
        probe_manifest_path=manifest,
        structured_root=structured,
        grouped_split_path=split,
    )
    second = build_postlock_bundles(
        probe_manifest_path=manifest,
        structured_root=structured,
        grouped_split_path=split,
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    census, predictions = first
    assert [case["case_token"] for case in census["cases"]] == ["C001", "C002"]
    assert census["cases"][0]["plane_pair_records"] == [{"plane_pair_id": "PP1"}]
    assert census["cases"][0]["birth_eligibility_records"][0][
        "intersection_span_endpoints"
    ] == [[0.0, 5.0, 0.0], [20.0, 5.0, 0.0]]
    assert [row["bend_id"] for row in predictions["cases"][0]["a0_predictions"]] == ["P1", "P2"]
    assert [row["latent_path_id"] for row in predictions["cases"][0]["existing_paths"]] == ["LP_A"]
    assert "part_key" not in json.dumps(predictions, sort_keys=True)

    gold_cases = [
        {
            "case_token": case["case_token"],
            "source_scan_sha256": case["scan_sha256"],
            "manual_bends": [],
        }
        for case in predictions["cases"]
    ]
    result = attribute_transition_probe_gold(
        census=census,
        predictions=predictions,
        sealed_gold={"cases": gold_cases},
        grouped_split=json.loads(split.read_text(encoding="utf-8")),
        source_hashes={
            "diagnostic_census_sha256": "a" * 64,
            "prediction_sha256": "b" * 64,
            "gold_sha256": "c" * 64,
            "grouped_split_sha256": _sha(split),
        },
    )
    assert result["kind"] == "postlock_gold_transition_failure_attribution"
    assert result["rows"] == []


def test_tampered_census_fails_manifest_hash_verification(tmp_path: Path) -> None:
    manifest, structured, split = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    census_path = Path(payload["cases"][0]["path"])
    census_path.write_text(census_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="census artifact SHA-256 mismatch"):
        build_postlock_bundles(
            probe_manifest_path=manifest,
            structured_root=structured,
            grouped_split_path=split,
        )


def test_grouped_split_must_cover_exact_probe_cases(tmp_path: Path) -> None:
    manifest, structured, split = _fixture(tmp_path)
    _write(split, {"cases": [{"case_token": "C001", "part_key": "FAMILY"}]})
    with pytest.raises(ValueError, match="case-token sets must match"):
        build_postlock_bundles(
            probe_manifest_path=manifest,
            structured_root=structured,
            grouped_split_path=split,
        )
