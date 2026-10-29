from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import scripts.run_f1_v51_prospective_review_scoring_v1 as lane


FREEZE = lane.ROOT / "research/f1_v51_20260831_prospective_three_arm_freeze_v1.json"


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lane._canonical_bytes(payload))


def _source_result(predictions: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": "synthetic_source_only_result",
        "candidate_count": len(predictions),
        "predictions": predictions,
        "source_only": True,
        "research_only": True,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
    }


def _line(start: list[float], end: list[float]) -> dict:
    return {"line_start": start, "line_end": end}


def _truth_free_fields() -> dict:
    return {
        "source_only": True,
        "truth_opened": False,
        "labels_used": False,
        "external_counts_used": False,
        "cad_used": False,
        "drawings_used": False,
        "part_identity_used": False,
        "evaluator_inputs_used": False,
    }


def _make_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_count: int = 2,
) -> tuple[Path, Path, Path]:
    scan_root = tmp_path / "opaque_scans"
    scan_root.mkdir()
    monkeypatch.setattr(lane, "OPAQUE_SCAN_ROOT", scan_root.resolve())
    scan_rows = []
    for ordinal in range(1, case_count + 1):
        scan = scan_root / f"T{ordinal:03d}.ply"
        scan.write_bytes(
            b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
            + str(ordinal).encode("ascii")
        )
        scan_rows.append({"scan_path": str(scan.resolve()), "scan_sha256": lane._sha(scan)})
    scan_manifest = tmp_path / "scan_manifest.json"
    _json(
        scan_manifest,
        {"schema_version": 1, "kind": lane.SCAN_INPUT_KIND, "cases": scan_rows},
    )

    execution_root = tmp_path / "execution"
    case_manifest_hashes: dict[str, str] = {}
    arm_rows: dict[str, list[dict]] = {arm_id: [] for arm_id in lane.ARM_IDS}
    exact = _line([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
    far = _line([0.0, 20.0, 0.0], [10.0, 20.0, 0.0])

    for ordinal, scan_row in enumerate(scan_rows, start=1):
        case_id = f"T{ordinal:03d}"
        arm_payloads = {
            lane.ARM_IDS[0]: _source_result([exact] if ordinal == 1 else []),
            lane.ARM_IDS[1]: _source_result([]),
            lane.ARM_IDS[2]: _source_result([exact, far] if ordinal == 1 else []),
        }
        case_arms = {}
        for arm_id, payload in arm_payloads.items():
            primary_relative = f"primary/{case_id}/arms/{lane.ARM_FILENAMES[arm_id]}"
            replay_relative = f"replay/{case_id}/arms/{lane.ARM_FILENAMES[arm_id]}"
            primary = execution_root / primary_relative
            replay = execution_root / replay_relative
            _json(primary, payload)
            _json(replay, payload)
            row = {
                "case_id": case_id,
                "scan_sha256": scan_row["scan_sha256"],
                "primary_artifact": primary_relative,
                "replay_artifact": replay_relative,
                "result_sha256": lane._sha(primary),
                "candidate_count": len(payload["predictions"]),
                "pipeline_status": "SUCCESS",
                "primary_replay_byte_identical": True,
            }
            if arm_id == lane.ARM_IDS[2]:
                intermediate = {"kind": "synthetic_recall_intermediate", "source_only": True}
                primary_intermediate = (
                    execution_root
                    / "primary"
                    / case_id
                    / "arms"
                    / "arm_c_recall_r3_intermediate.json"
                )
                replay_intermediate = (
                    execution_root
                    / "replay"
                    / case_id
                    / "arms"
                    / "arm_c_recall_r3_intermediate.json"
                )
                _json(primary_intermediate, intermediate)
                _json(replay_intermediate, intermediate)
                row["recall_intermediate_sha256"] = lane._sha(primary_intermediate)
            case_arms[arm_id] = row
            arm_rows[arm_id].append(deepcopy(row))

        case_manifest = {
            "schema_version": 1,
            "kind": lane.CASE_MANIFEST_KIND,
            "case_id": case_id,
            "scan_sha256": scan_row["scan_sha256"],
            **_truth_free_fields(),
            "base_primary_status": "SUCCESS",
            "base_replay_status": "SUCCESS",
            "arms": case_arms,
        }
        case_path = execution_root / "case_manifests" / f"{case_id}.json"
        _json(case_path, case_manifest)
        case_manifest_hashes[case_id] = lane._sha(case_path)

    arm_manifest_hashes = {}
    for arm_id in lane.ARM_IDS:
        arm_manifest = {
            "schema_version": 1,
            "kind": lane.ARM_MANIFEST_KIND,
            "arm_id": arm_id,
            **_truth_free_fields(),
            "case_count": case_count,
            "all_primary_replay_byte_identical": True,
            "cases": arm_rows[arm_id],
        }
        arm_path = execution_root / "arm_manifests" / f"{arm_id.lower()}.json"
        _json(arm_path, arm_manifest)
        arm_manifest_hashes[arm_id] = lane._sha(arm_path)

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    component_lock = freeze["component_lock"]
    orchestrator = lane.ROOT / lane.ORCHESTRATOR_RELATIVE_PATH
    execution = {
        "schema_version": 1,
        "kind": lane.EXECUTION_KIND,
        "status": "PROSPECTIVE_PREDICTION_ONLY_COMPLETE",
        "research_only": True,
        "scoring_opened": False,
        **_truth_free_fields(),
        "all_arms_primary_replay_byte_identical": True,
        "case_count": case_count,
        "case_ids": [f"T{ordinal:03d}" for ordinal in range(1, case_count + 1)],
        "input_manifest_sha256": lane._sha(scan_manifest),
        "freeze_packet_sha256": lane._sha(FREEZE),
        "component_lock": component_lock,
        "component_lock_sha256": lane._sha_bytes(lane._canonical_bytes(component_lock)),
        "orchestrator_binding": {
            "orchestrator_path": lane.ORCHESTRATOR_RELATIVE_PATH,
            "orchestrator_sha256": lane._sha(orchestrator),
            "bound_before_scan_manifest_open": True,
        },
        "case_manifest_sha256": case_manifest_hashes,
        "arm_manifest_sha256": arm_manifest_hashes,
    }
    _json(execution_root / "prospective_execution_manifest.json", execution)
    return scan_manifest, execution_root, FREEZE


def _catalog(
    path: Path,
    *,
    cases: list[dict],
    render_sha: str,
    seal_sha: str,
    freeze_sha: str,
) -> Path:
    rows = []
    for case in cases:
        case_id = case["case_id"]
        reviews = []
        for ordinal in range(1, 4):
            reviewer_id = f"R{ordinal:03d}"
            bends = (
                [
                    {
                        "bend_id": f"{reviewer_id}-B001",
                        "line_start": [0.0, 0.0, 0.0],
                        "line_end": [10.0, 0.0, 0.0],
                    }
                ]
                if case_id == "T001"
                else []
            )
            reviews.append(
                {
                    "reviewer_id": reviewer_id,
                    "reviewer_class": "high_performance_agent",
                    "independent": True,
                    "manual_bends": bends,
                }
            )
        final = (
            [
                {
                    "bend_id": f"{case_id}-B001",
                    "line_start": [0.0, 0.0, 0.0],
                    "line_end": [10.0, 0.0, 0.0],
                }
            ]
            if case_id == "T001"
            else []
        )
        rows.append(
            {
                "case_id": case_id,
                "scan_sha256": case["scan_sha256"],
                "reviews": reviews,
                "reconciliation": {
                    "status": "SEALED",
                    "blind_to_algorithm_outputs": True,
                    "source_reviewer_ids": ["R001", "R002", "R003"],
                    "adjudicator_ids": ["A001", "A002"],
                    "manual_bends": final,
                },
            }
        )
    payload = {
        "schema_version": 1,
        "kind": lane.LABEL_CATALOG_KIND,
        "status": "SEALED_BEFORE_SCORING",
        "cohort_date": "2026-08-31",
        "render_manifest_sha256": render_sha,
        "execution_seal_sha256": seal_sha,
        "freeze_packet_sha256": freeze_sha,
        "review_protocol": deepcopy(lane.REVIEW_PROTOCOL),
        "cases": rows,
    }
    _json(path, payload)
    lane._write_sidecar(path)
    return path


def _prepare_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path]:
    scan_manifest, execution_root, freeze = _make_execution(tmp_path, monkeypatch)
    review_root = tmp_path / "review"
    lane.prepare_prediction_hidden_review(scan_manifest, execution_root, freeze, review_root)
    cases = lane.validate_scan_manifest(scan_manifest)
    render_path = review_root / "raw_scan_only_render_manifest.json"
    seal_path = review_root / "execution_identity_seal.json"
    catalog = _catalog(
        tmp_path / "labels.json",
        cases=cases,
        render_sha=lane._sha(render_path),
        seal_sha=lane._sha(seal_path),
        freeze_sha=lane._sha(freeze),
    )
    return scan_manifest, execution_root, freeze, review_root, catalog


def _walk_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(key)
            keys.extend(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_walk_keys(nested))
    return keys


def test_prepare_review_validates_every_arm_and_emits_zero_prediction_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_manifest, execution_root, freeze = _make_execution(tmp_path, monkeypatch)
    result = lane.prepare_prediction_hidden_review(
        scan_manifest, execution_root, freeze, tmp_path / "review"
    )
    assert result["all_arms_primary_replay_byte_identical"] is True
    render_path = Path(result["raw_scan_only_render_manifest"])
    payload = json.loads(render_path.read_text(encoding="utf-8"))
    forbidden_fragments = ("prediction", "candidate", "artifact", "arm", "line")
    assert not [
        key
        for key in _walk_keys(payload)
        if any(fragment in key.lower() for fragment in forbidden_fragments)
    ]
    assert payload["custody"] == lane.RENDER_CUSTODY
    assert [row["case_id"] for row in payload["cases"]] == ["T001", "T002"]
    assert lane._validate_sidecar(render_path) == lane._sha(render_path)


def test_prepare_review_fails_on_any_arm_replay_drift_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_manifest, execution_root, freeze = _make_execution(tmp_path, monkeypatch)
    replay = execution_root / "replay/T002/arms/arm_b_v51_v43_guard.json"
    replay.write_bytes(replay.read_bytes() + b" ")
    with pytest.raises(lane.CustodyError, match="primary/replay byte identity failed"):
        lane.prepare_prediction_hidden_review(
            scan_manifest, execution_root, freeze, tmp_path / "review"
        )
    assert not (tmp_path / "review").exists()


def test_prepare_review_fails_on_c_recall_intermediate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_manifest, execution_root, freeze = _make_execution(tmp_path, monkeypatch)
    replay = execution_root / "replay/T001/arms/arm_c_recall_r3_intermediate.json"
    replay.write_bytes(replay.read_bytes() + b" ")
    with pytest.raises(lane.CustodyError, match="recall replay identity failed"):
        lane.prepare_prediction_hidden_review(
            scan_manifest, execution_root, freeze, tmp_path / "review"
        )


def test_label_catalog_requires_many_independent_reviews_and_two_adjudicators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_manifest, execution_root, freeze, review_root, catalog = _prepare_fixture(
        tmp_path, monkeypatch
    )
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["cases"][0]["reviews"] = payload["cases"][0]["reviews"][:2]
    _json(catalog, payload)
    lane._write_sidecar(catalog)
    with pytest.raises(lane.CustodyError, match="fewer than three"):
        lane.score_sealed_three_arm(
            scan_manifest,
            execution_root,
            freeze,
            review_root,
            catalog,
            tmp_path / "score",
        )

    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["cases"][0]["reviews"].append(deepcopy(payload["cases"][1]["reviews"][2]))
    payload["cases"][0]["reconciliation"]["source_reviewer_ids"] = ["R001", "R002", "R003"]
    payload["cases"][0]["reconciliation"]["adjudicator_ids"] = ["A001"]
    _json(catalog, payload)
    lane._write_sidecar(catalog)
    with pytest.raises(lane.CustodyError, match="at least two opaque adjudicators"):
        lane.score_sealed_three_arm(
            scan_manifest,
            execution_root,
            freeze,
            review_root,
            catalog,
            tmp_path / "score",
        )


def test_invalid_label_seal_fails_before_prediction_json_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_manifest, execution_root, freeze, review_root, catalog = _prepare_fixture(
        tmp_path, monkeypatch
    )
    catalog.write_bytes(catalog.read_bytes() + b" ")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prediction parser opened before label seal validated")

    monkeypatch.setattr(lane, "_load_prediction_records", forbidden)
    with pytest.raises(lane.CustodyError, match="sidecar mismatch"):
        lane.score_sealed_three_arm(
            scan_manifest,
            execution_root,
            freeze,
            review_root,
            catalog,
            tmp_path / "score",
        )


def test_score_all_three_arms_with_exact_canonical_thresholds_and_no_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_manifest, execution_root, freeze, review_root, catalog = _prepare_fixture(
        tmp_path, monkeypatch
    )
    result = lane.score_sealed_three_arm(
        scan_manifest,
        execution_root,
        freeze,
        review_root,
        catalog,
        tmp_path / "score",
    )
    report_path = Path(result["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["thresholds"] == lane.CANONICAL_THRESHOLDS
    assert report["case_exclusions"] == []
    assert report["all_cases_included"] is True
    assert report["retuning_after_freeze"] is False
    assert report["threshold_overrides"] is False
    assert set(report["arms"]) == set(lane.ARM_IDS)
    assert report["arms"][lane.ARM_IDS[0]]["metrics"] == {
        "case_count": 2,
        "gold_bend_count": 1,
        "prediction_count": 1,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "exact_case_count": 2,
    }
    assert report["arms"][lane.ARM_IDS[1]]["metrics"]["false_negative"] == 1
    assert report["arms"][lane.ARM_IDS[2]]["metrics"]["false_positive"] == 1
    assert lane._validate_sidecar(report_path) == result["report_sha256"]


def test_cli_exposes_no_threshold_or_case_selection_override() -> None:
    source = inspect.getsource(lane.main)
    for forbidden in (
        "--angle",
        "--offset",
        "--overlap",
        "--threshold",
        "--case-map",
        "--exclude",
        "--expected-count",
    ):
        assert forbidden not in source


def test_label_catalog_rejects_algorithm_output_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_manifest, execution_root, freeze, review_root, catalog = _prepare_fixture(
        tmp_path, monkeypatch
    )
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["cases"][0]["prediction_count"] = 1
    _json(catalog, payload)
    lane._write_sidecar(catalog)
    with pytest.raises(lane.CustodyError, match="forbidden or missing fields"):
        lane.score_sealed_three_arm(
            scan_manifest,
            execution_root,
            freeze,
            review_root,
            catalog,
            tmp_path / "score",
        )

