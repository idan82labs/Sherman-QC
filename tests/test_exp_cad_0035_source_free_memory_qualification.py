from __future__ import annotations

import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

from scripts import generate_exp_cad_0035_source_free_fixtures as FIXTURES
from scripts import run_exp_cad_0035_source_free_memory_qualification as Q


def _small_fixture_set(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "fixtures"
    manifest = FIXTURES.generate_fixture_set(
        root, row_counts={"N009": 32, "N010": 40}
    )
    return root / FIXTURES.MANIFEST_NAME, dict(manifest)


def _bind_small_fixture(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
    *,
    bind_row_counts: bool = False,
) -> None:
    monkeypatch.setattr(Q, "FIXTURE_ROOT", manifest_path.parent)
    monkeypatch.setattr(Q, "FIXTURE_MANIFEST", manifest_path)
    if bind_row_counts:
        monkeypatch.setattr(Q, "ROW_COUNTS", {"N009": 32, "N010": 40})


def _bind_nonisolated_unit_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Keep unit-scale controller mocks separate from the real -I entry gate."""

    identity = Q._runtime_identity_snapshot(strict_sys_path=False)
    monkeypatch.setattr(
        Q,
        "_runtime_identity_snapshot",
        lambda *, strict_sys_path: identity,
    )
    return identity


def _read_ply_rows(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"end_header\n"
    offset = payload.index(marker) + len(marker)
    return np.frombuffer(payload[offset:], dtype="<f8").reshape(-1, 6)


def test_generator_creates_only_manifest_and_four_declared_plys(tmp_path: Path) -> None:
    manifest_path, manifest = _small_fixture_set(tmp_path)
    files = sorted(
        path.relative_to(manifest_path.parent).as_posix()
        for path in manifest_path.parent.rglob("*")
        if path.is_file()
    )
    assert files == [
        "EXP_CAD_0035_SOURCE_FREE_FIXTURE_MANIFEST.json",
        "boundary-equivalence/N009.ply",
        "boundary-equivalence/N010.ply",
        "maximal-retention/N009.ply",
        "maximal-retention/N010.ply",
    ]
    assert manifest["schema"] == FIXTURES.SCHEMA
    assert len(manifest["fixtures"]) == 4
    assert manifest["physical_scan_byte_open_operation_count"] == 0
    assert manifest["evaluator_role_label_gold_quality_or_expected_count_access_count"] == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_manifest_rejects_redirected_fixture_path_before_any_fixture_stat_hash_or_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fixtures"][0]["path"] = (
        "data/bend_corpus/incoming/physical-redirect/N009.ply"
    )
    Q._strict_write(manifest_path, payload)

    def forbidden_hash(_path: Path) -> str:
        raise AssertionError("fixture metadata gate must precede all SHA/stat/open work")

    monkeypatch.setattr(Q, "_sha256_path", forbidden_hash)
    with pytest.raises(ValueError, match="identity/path/metadata drift"):
        Q._fixture_inputs(manifest_path, require_exact_counts=False)


def test_manifest_requires_exact_generator_identity_before_fixture_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["generator_path"] = "scripts/not-the-exp35-generator.py"
    Q._strict_write(manifest_path, payload)

    def forbidden_hash(_path: Path) -> str:
        raise AssertionError("generator identity gate must precede fixture access")

    monkeypatch.setattr(Q, "_sha256_path", forbidden_hash)
    with pytest.raises(ValueError, match="identity/access drift"):
        Q._fixture_inputs(manifest_path, require_exact_counts=False)


def test_boundary_fixture_has_exact_frozen_bit_patterns_and_formula(tmp_path: Path) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    rows = _read_ply_rows(manifest_path.parent / "boundary-equivalence/N009.ply")
    assert rows.shape == (32, 6)
    assert rows[0, 0:1].view("<u8")[0] == 0x7FF8000000000000
    assert rows[0, 3:4].view("<u8")[0] == 0x7FF8000000000000
    assert np.isposinf(rows[1, 0])
    assert np.isposinf(rows[3, 3])
    assert np.array_equal(rows[5, 3:], [0.0, 0.0, 0.0])
    assert rows[6, 3].hex() == "0x1.19799812dea11p-40"
    assert rows[7, 3].hex() == "0x1.19799812dea10p-40"
    assert rows[8, 3].hex() == "0x1.19799812dea12p-40"
    assert np.array_equal(rows[9], [32.0 + 9 * 2.0**-10, -16.0, 4.0, 0.0, 0.0, 1.0])


def test_boundary_reports_are_exact_and_nonfinite_point_wins_precedence(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    reports = []
    for token in ("N009", "N010"):
        rows = _read_ply_rows(
            manifest_path.parent / f"boundary-equivalence/{token}.ply"
        )
        filtered = Q.INTAKE.filter_source_rows(
            rows[:, :3],
            rows[:, 3:],
            member_token=token,
            original_row_ids=np.arange(len(rows), dtype=np.uint64),
        )
        report = filtered["report"]
        assert report["raw_predicate_counts"] == {
            "NONFINITE_POINT": 2,
            "NONFINITE_NORMAL": 4,
            "NORMAL_NORM_LE_1E_12": 3,
        }
        assert report["exclusive_rejection_counts"] == {
            "NONFINITE_POINT": 2,
            "NONFINITE_NORMAL": 3,
            "NORMAL_NORM_LE_1E_12": 3,
        }
        assert report["rejected_original_row_ids_sorted"] == list(range(8))
        assert report["retained_row_count"] == len(rows) - 8
        reports.append(
            {
                "raw_predicate_counts": report["raw_predicate_counts"],
                "exclusive_rejection_counts": report["exclusive_rejection_counts"],
                "rejected_original_row_ids_sorted": report[
                    "rejected_original_row_ids_sorted"
                ],
            }
        )
    assert reports[0] == reports[1]


def test_boundary_plys_cross_actual_open3d_legacy_and_sequential_modes_with_exact_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    _manifest_payload, inputs = Q._fixture_inputs(
        manifest_path,
        require_exact_counts=False,
        fixture="BOUNDARY_EQUIVALENCE",
    )
    frozen = Q.consumed_sha256_map(manifest_path)
    legacy_first, legacy_second, legacy_pair, _legacy_lifecycle = Q._legacy_intake(
        inputs,
        fixture_manifest=manifest_path,
        frozen_consumed_sha256=frozen,
    )
    sequential_first, sequential_second, sequential_pair, _sequential_lifecycle = (
        Q._sequential_intake(
            inputs,
            fixture_manifest=manifest_path,
            frozen_consumed_sha256=frozen,
        )
    )

    def projection(first: object, second: object, pair: object) -> dict[str, object]:
        first_row = first
        second_row = second
        return {
            "retained": {
                "N009": Q._projection(first_row),
                "N010": Q._projection(second_row),
            },
            "reports": {
                "N009": first_row["report"],
                "N010": second_row["report"],
            },
            "pair_actionability": pair,
        }

    legacy_projection = projection(legacy_first, legacy_second, legacy_pair)
    sequential_projection = projection(
        sequential_first, sequential_second, sequential_pair
    )
    assert Q._strict_bytes(legacy_projection) == Q._strict_bytes(
        sequential_projection
    )
    for report in sequential_projection["reports"].values():
        assert report["raw_predicate_counts"] == {
            "NONFINITE_POINT": 2,
            "NONFINITE_NORMAL": 4,
            "NORMAL_NORM_LE_1E_12": 3,
        }
        assert report["exclusive_rejection_counts"] == {
            "NONFINITE_POINT": 2,
            "NONFINITE_NORMAL": 3,
            "NORMAL_NORM_LE_1E_12": 3,
        }
        assert report["rejected_original_row_ids_sorted"] == list(range(8))


def test_small_legacy_oracle_calls_exact_hash_bound_exp34_wrapper_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    _manifest_payload, inputs = Q._fixture_inputs(
        manifest_path, require_exact_counts=False
    )
    frozen = Q.consumed_sha256_map(manifest_path)
    exact_legacy = Q._load_legacy_runner()
    assert Q._sha256_path(Path(exact_legacy.__file__)) == Q.LEGACY_SHA256
    exact_wrapper = exact_legacy.execute_source_exp34
    wrapper_calls: list[tuple[str, str]] = []

    def wrapper_spy(**kwargs: object) -> dict[str, object]:
        wrapper_calls.append(
            (str(kwargs["first_token"]), str(kwargs["second_token"]))
        )
        return exact_wrapper(**kwargs)

    exact_legacy.execute_source_exp34 = wrapper_spy
    monkeypatch.setattr(Q, "_load_legacy_runner", lambda: exact_legacy)
    outcome = Q._legacy_execution(
        inputs,
        fixture_manifest=manifest_path,
        frozen_consumed_sha256=frozen,
    )
    assert wrapper_calls == [("N009", "N010")]
    assert outcome["call_accounting"] == {
        "exact_exp34_execute_source_exp34_call_count": 1,
        "exact_exp35_execute_filtered_source_exp35_call_count": 0,
        "exp31_filter_source_rows_call_count_by_token": {"N009": 1, "N010": 1},
        "exp31_pair_actionability_call_count": 1,
        "qualification_parent_v5_stub_call_count": 0,
        "exp32_source_free_v5_wrapper_import_or_call_count": 0,
        "exp32_v5_filter_entry_point_call_count": 0,
    }


def test_subscale_actionable_workers_call_identical_parent_v5_stub_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "actionable-fixtures"
    FIXTURES.generate_fixture_set(
        root, row_counts={"N009": 100_000, "N010": 100_000}
    )
    manifest_path = root / FIXTURES.MANIFEST_NAME
    _bind_small_fixture(monkeypatch, manifest_path)
    legacy = Q.worker_compute(
        mode="LEGACY_BATCH",
        fixture_manifest=manifest_path,
        require_exact_counts=False,
    )
    sequential = Q.worker_compute(
        mode="SEQUENTIAL",
        fixture_manifest=manifest_path,
        require_exact_counts=False,
    )
    assert legacy["qualification_call_accounting"][
        "exact_exp34_execute_source_exp34_call_count"
    ] == 1
    assert sequential["qualification_call_accounting"][
        "exact_exp35_execute_filtered_source_exp35_call_count"
    ] == 1
    assert legacy["qualification_call_accounting"][
        "qualification_parent_v5_stub_call_count"
    ] == 1
    assert sequential["qualification_call_accounting"][
        "qualification_parent_v5_stub_call_count"
    ] == 1
    assert legacy["nested_payload_projection"] == sequential[
        "nested_payload_projection"
    ]
    assert legacy["nested_payload_projection"][
        "scientific_execute_source_reached"
    ] is True


def test_frozen_actionability_boundaries_are_inclusive_and_exact() -> None:
    points = np.zeros((200_000, 3), dtype=np.float64)
    normals = np.tile([0.0, 0.0, 1.0], (200_000, 1))
    exactly = points.copy()
    exactly[:2_000, 0] = np.nan
    exact_result = Q.INTAKE.filter_source_rows(
        exactly,
        normals,
        member_token="N009",
        original_row_ids=np.arange(200_000, dtype=np.uint64),
    )
    assert exact_result["report"]["invalid_fraction"] == 0.01
    assert exact_result["report"]["actionable"] is True
    above = exactly.copy()
    above[2_000, 0] = np.nan
    above_result = Q.INTAKE.filter_source_rows(
        above,
        normals,
        member_token="N009",
        original_row_ids=np.arange(200_000, dtype=np.uint64),
    )
    assert above_result["report"]["invalid_fraction"] > 0.01
    assert above_result["report"]["actionable"] is False
    assert "INVALID_FRACTION_EXCEEDS_0.01" in above_result["report"]["actionability_reasons"]

    minimum = Q.INTAKE.filter_source_rows(
        points[:100_000],
        normals[:100_000],
        member_token="N009",
        original_row_ids=np.arange(100_000, dtype=np.uint64),
    )
    below = Q.INTAKE.filter_source_rows(
        points[:99_999],
        normals[:99_999],
        member_token="N009",
        original_row_ids=np.arange(99_999, dtype=np.uint64),
    )
    assert minimum["report"]["retained_row_count"] == 100_000
    assert minimum["report"]["actionable"] is True
    assert below["report"]["retained_row_count"] == 99_999
    assert below["report"]["actionable"] is False
    assert "VALID_ROW_COUNT_BELOW_100000" in below["report"]["actionability_reasons"]


def test_fixture_generation_refuses_regeneration(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    FIXTURES.generate_fixture_set(root, row_counts={"N009": 16, "N010": 16})
    with pytest.raises(FileExistsError, match="regeneration"):
        FIXTURES.generate_fixture_set(root, row_counts={"N009": 16, "N010": 16})


def test_small_actual_open3d_legacy_and_sequential_workers_are_equivalent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    legacy = Q.worker_compute(
        mode="LEGACY_BATCH",
        fixture_manifest=manifest_path,
        require_exact_counts=False,
    )
    sequential = Q.worker_compute(
        mode="SEQUENTIAL",
        fixture_manifest=manifest_path,
        require_exact_counts=False,
    )
    assert Q.scientific_equivalence_projection(legacy) == Q.scientific_equivalence_projection(
        sequential
    )
    assert legacy["consumed_sha256_map"] == sequential["consumed_sha256_map"]
    assert set(legacy["consumed_sha256_map"]) == {
        "memory_worker",
        "exp29_cad_core",
        "exp29_frozen_source",
        "exp29_frozen_runner",
        "exp35_source_runner",
        "exp35_process_monitor",
        "exp35_execution_contract",
        "exp34_legacy_runner",
        "exp34_execution_contract",
        "exp34_environment_attestation",
        "exp31_filter_module",
        "exp32_v5_engine",
        "exp30_preregistration",
        "exp31_preregistration",
        "exp32_preregistration",
        "exp34_preregistration",
        "exp35_preregistration",
        "fixture_generator",
        "fixture_manifest",
        "fixture_boundary_N009",
        "fixture_boundary_N010",
        "fixture_maximal_N009",
        "fixture_maximal_N010",
    }
    assert legacy["lifecycle"] == {
        "explicit_gc_collect_call_count": 0,
        "maximum_live_source_cloud_count": 2,
        "maximum_live_unfiltered_member_bundle_count": 2,
        "N009_release_verified": False,
        "N010_release_verified": False,
        "filter_source_rows_call_count_by_token": {"N009": 1, "N010": 1},
        "pair_actionability_call_count": 1,
    }
    assert sequential["lifecycle"]["explicit_gc_collect_call_count"] == 2
    assert sequential["lifecycle"]["mode"] == "PRIMARY"
    assert sequential["lifecycle"]["status"] == "COMPLETE_RELEASE_VERIFIED"
    assert sequential["lifecycle"]["maximum_live_source_cloud_count"] == 1
    assert sequential["lifecycle"]["maximum_live_unfiltered_member_bundle_count"] == 1
    assert sequential["lifecycle"]["N009_release_verified"] is True
    assert sequential["lifecycle"]["N010_release_verified"] is True
    assert sequential["lifecycle"]["filter_source_rows_call_count_by_token"] == {
        "N009": 1,
        "N010": 1,
    }
    assert sequential["lifecycle"]["pair_actionability_call_count"] == 1
    assert legacy["nested_payload_projection"] == sequential[
        "nested_payload_projection"
    ]
    assert legacy["qualification_call_accounting"] == {
        "exact_exp34_execute_source_exp34_call_count": 1,
        "exact_exp35_execute_filtered_source_exp35_call_count": 0,
        "exp31_filter_source_rows_call_count_by_token": {"N009": 1, "N010": 1},
        "exp31_pair_actionability_call_count": 1,
        "qualification_parent_v5_stub_call_count": 0,
        "exp32_source_free_v5_wrapper_import_or_call_count": 0,
        "exp32_v5_filter_entry_point_call_count": 0,
    }
    assert sequential["qualification_call_accounting"] == {
        "exact_exp34_execute_source_exp34_call_count": 0,
        "exact_exp35_execute_filtered_source_exp35_call_count": 1,
        "exp31_filter_source_rows_call_count_by_token": {"N009": 1, "N010": 1},
        "exp31_pair_actionability_call_count": 1,
        "qualification_parent_v5_stub_call_count": 0,
        "exp32_source_free_v5_wrapper_import_or_call_count": 0,
        "exp32_v5_filter_entry_point_call_count": 0,
    }
    for result in (legacy, sequential):
        assert result["synthetic_fixture_hash_call_count"] == 2
        assert result["synthetic_fixture_point_cloud_load_call_count"] == 2
        assert result["synthetic_fixture_byte_open_operation_count"] == 4
        assert result["physical_scan_hash_call_count"] == 0
        assert result["physical_scan_point_cloud_load_call_count"] == 0
        assert result["physical_scan_byte_open_operation_count"] == 0
        assert result["evaluator_role_label_gold_quality_or_expected_count_access_count"] == 0
        assert result["source_free_v5_wrapper_import_or_call_count"] == 0
        assert result["v5_filter_entry_point_call_count"] == 0


def test_exact_scale_gate_rejects_small_unit_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    _bind_nonisolated_unit_runtime(monkeypatch)
    with pytest.raises(ValueError, match="exact-scale"):
        Q.worker_compute(
            mode="SEQUENTIAL",
            fixture_manifest=manifest_path,
            require_exact_counts=True,
        )


def test_controller_inventory_and_schedule_are_exact_without_execution() -> None:
    assert Q.SCHEDULE == (
        (1, "LEGACY_BATCH", "trial-01-legacy-batch"),
        (2, "SEQUENTIAL", "trial-02-sequential"),
        (3, "LEGACY_BATCH", "trial-03-legacy-batch"),
        (4, "SEQUENTIAL", "trial-04-sequential"),
        (5, "LEGACY_BATCH", "trial-05-legacy-batch"),
        (6, "SEQUENTIAL", "trial-06-sequential"),
    )
    help_text = Q.build_parser().format_help()
    assert "worker" in help_text and "qualify" in help_text
    assert Q.RESULT_NAME == "SOURCE_FREE_MEMORY_QUALIFICATION_RESULT.json"
    assert Q.MEASUREMENT_NAME == "EXTERNAL_PROCESS_MEASUREMENT.json"
    assert Q.AGGREGATE_NAME == "EXP_CAD_0035_SOURCE_FREE_MEMORY_QUALIFICATION_AGGREGATE.json"


def _valid_measurement(
    *,
    ordinal: int,
    mode: str,
    peak: int,
    runtime_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    if runtime_identity is None:
        runtime_identity = Q._runtime_identity_snapshot(strict_sys_path=False)
    environment = runtime_identity["runtime_environment_fingerprint"]
    return {
        "schema": Q.MEASUREMENT_SCHEMA,
        "experiment_id": Q.EXPERIMENT_ID,
        "trial_ordinal": ordinal,
        "mode": mode,
        "returncode": 0,
        "timed_out": False,
        "rss_limit_exceeded": False,
        "sampler_failed": False,
        "resource_kill_reason": None,
        "elapsed_wall_seconds": 1.0,
        "sampling_interval_seconds": Q.SAMPLE_INTERVAL_SECONDS,
        "sampled_peak_process_tree_rss_bytes": peak - 1,
        "time_l_maximum_resident_set_size_bytes": peak,
        "peak_process_tree_rss_bytes": peak,
        "applicable_rss_cap_bytes": (
            Q.SEQUENTIAL_RSS_CAP_BYTES
            if mode == "SEQUENTIAL"
            else Q.LEGACY_RSS_CAP_BYTES
        ),
        "runtime_identity": runtime_identity,
        "worker_runtime_identity_validated": True,
        "worker_python_invocation_path": environment[
            "python_executable_invocation_path"
        ],
        "worker_python_realpath": environment["python_executable_realpath"],
        "worker_python_sha256": environment["python_executable_sha256"],
        "worker_python_isolated_flag": True,
        "worker_python_no_bytecode_flag": True,
        "stdout": "",
        "stderr": "",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("time_l_maximum_resident_set_size_bytes", None),
        ("elapsed_wall_seconds", 180.0),
    ),
)
def test_external_measurement_requires_time_l_rss_and_strict_sub_180_wall(
    field: str, value: object
) -> None:
    measurement = _valid_measurement(
        ordinal=1, mode="SEQUENTIAL", peak=1_000_000
    )
    measurement[field] = value
    runtime_identity = measurement["runtime_identity"]
    with pytest.raises(ValueError, match="strict gate"):
        Q._validate_external_measurement(
            measurement,
            ordinal=1,
            mode="SEQUENTIAL",
            frozen_runtime_identity=runtime_identity,
        )


def test_noncanonical_json_result_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / Q.RESULT_NAME
    path.write_text('{\n  "schema": "NOT_CANONICAL"\n}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="strict canonical bytes"):
        Q._load_canonical_json(path)


def test_mocked_six_trial_controller_success_claims_root_first_and_rechecks_final_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    qualification_root = tmp_path / "qualification"
    _bind_small_fixture(monkeypatch, manifest_path)
    monkeypatch.setattr(Q, "QUALIFICATION_ROOT", qualification_root)
    runtime_identity = _bind_nonisolated_unit_runtime(monkeypatch)

    small_results = {
        mode: Q.worker_compute(
            mode=mode,
            fixture_manifest=manifest_path,
            require_exact_counts=False,
        )
        for mode in ("LEGACY_BATCH", "SEQUENTIAL")
    }
    results: dict[str, dict[str, object]] = {}
    for mode, small in small_results.items():
        result = json.loads(Q._strict_bytes(small).decode("utf-8"))
        projection = result["scientific_equivalence_projection"]
        for token in ("N009", "N010"):
            count = Q.ROW_COUNTS[token]
            report = projection["reports"][token]
            report["original_row_count"] = count
            report["retained_row_count"] = count
            report["rejected_row_count"] = 0
            report["invalid_fraction"] = 0.0
            report["actionable"] = True
            report["actionability_reasons"] = []
            projection["retained"][token]["points"]["shape"] = [count, 3]
            projection["retained"][token]["normals"]["shape"] = [count, 3]
            projection["retained"][token]["original_row_ids"]["shape"] = [count]
        projection["pair_actionability"] = {
            "actionable": True,
            "failed_member_tokens": [],
            "status": "PASS_SOURCE_INTERFACE_ACTIONABLE",
        }
        result["report_sha256"] = {
            token: Q.hashlib.sha256(
                Q._strict_bytes(projection["reports"][token])
            ).hexdigest()
            for token in ("N009", "N010")
        }
        result["pair_actionability_sha256"] = Q.hashlib.sha256(
            Q._strict_bytes(projection["pair_actionability"])
        ).hexdigest()
        input_projection = {
            token: {
                "points_dtype": projection["retained"][token]["points"]["dtype"],
                "points_shape": projection["retained"][token]["points"]["shape"],
                "points_tobytes_sha256": projection["retained"][token]["points"][
                    "tobytes_sha256"
                ],
                "normals_dtype": projection["retained"][token]["normals"]["dtype"],
                "normals_shape": projection["retained"][token]["normals"]["shape"],
                "normals_tobytes_sha256": projection["retained"][token]["normals"][
                    "tobytes_sha256"
                ],
            }
            for token in ("N009", "N010")
        }
        counters = {
            "kd_node_query_count": 306_926,
            "local_fit_call_count": 38,
            "robustness_row_count": 32,
            "unique_member_fit_count": 38,
        }
        result["nested_payload_projection"] = {
            "status": "PASS_SOURCE_FREE_QUALIFICATION_PARENT_V5_STUB",
            "reason": None,
            "scientific_execute_source_reached": True,
            "semantic_payload": {
                "schema": "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1",
                "experiment_id": "EXP_CAD_0030",
                "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
                "status": "PASS_SOURCE_FREE_QUALIFICATION_PARENT_V5_STUB",
                "reason": None,
                "qualification_only_parent_v5_stub": True,
                "input_projection": input_projection,
            },
            "visual_payload": {
                "schema": "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1",
                "qualification_only_parent_v5_stub": True,
                "input_projection": input_projection,
            },
            "engine_metrics": {
                "schema": "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1",
                "experiment_id": "EXP_CAD_0032",
                "architecture": Q.V5.ARCHITECTURE,
                "operation_counters": counters,
                **counters,
                "full_fit_inventory_gate_passed": True,
                "full_fit_batched_kd_gate_passed": True,
                "full_fit_kd_query_gate_passed": True,
                "fit_or_robustness_row_skipped": False,
                "scientific_threshold_or_gate_changed": False,
                "qualification_only_parent_v5_stub": True,
                "input_projection": input_projection,
            },
            "engine_accounting": {
                "schema": "EXP_CAD_0034_ENGINE_INVOCATION_ACCOUNTING_V1",
                "v5_engine_invocation_count": 1,
                "unique_member_fit_count": 38,
                "kd_node_query_count": 306_926,
                "local_fit_call_count": 38,
                "robustness_row_count": 32,
                "exp32_engine_metrics_payload_present": True,
                "deterministic_exp32_operation_counter_projection": counters,
            },
        }
        result["nested_payload_projection_sha256"] = Q.hashlib.sha256(
            Q._strict_bytes(result["nested_payload_projection"])
        ).hexdigest()
        result["qualification_call_accounting"][
            "qualification_parent_v5_stub_call_count"
        ] = 1
        results[mode] = result
    calls: list[tuple[int, str]] = []

    def fake_trial(**kwargs: object) -> dict[str, object]:
        ordinal = int(kwargs["ordinal"])
        mode = str(kwargs["mode"])
        trial_root = Path(kwargs["trial_root"])
        calls.append((ordinal, mode))
        peak = 100_000_000 if mode == "LEGACY_BATCH" else 10_000_000
        assert kwargs["frozen_runtime_identity"] == runtime_identity
        measurement = _valid_measurement(
            ordinal=ordinal,
            mode=mode,
            peak=peak,
            runtime_identity=runtime_identity,
        )
        Q._strict_write(trial_root / Q.RESULT_NAME, results[mode])
        Q._strict_write(trial_root / Q.MEASUREMENT_NAME, measurement)
        return {"result": results[mode], "measurement": measurement}

    original_preflight = Q._fixture_inputs

    def asserted_preflight(*args: object, **kwargs: object) -> object:
        assert qualification_root.is_dir()
        kwargs["require_exact_counts"] = False
        return original_preflight(*args, **kwargs)

    contexts: list[str] = []
    original_assert = Q._assert_consumed_unchanged

    def assert_spy(*args: object, **kwargs: object) -> None:
        contexts.append(str(kwargs["context"]))
        original_assert(*args, **kwargs)

    monkeypatch.setattr(Q, "_fixture_inputs", asserted_preflight)
    monkeypatch.setattr(Q, "_assert_consumed_unchanged", assert_spy)
    stage_events: list[str] = []

    def fake_boundary(**kwargs: object) -> dict[str, object]:
        assert calls == []
        stage_events.append("BOUNDARY_BEFORE_TRIALS")
        return {
            "status": "PASS_EXACT_SIZE_OPEN3D_BOUNDARY_EQUIVALENCE",
            "formal_stage": (
                "AFTER_QUALIFICATION_ROOT_CLAIM_BEFORE_SIX_MEASURED_MAXIMAL_TRIALS"
            ),
            "runtime_identity": kwargs["frozen_runtime_identity"],
        }

    monkeypatch.setattr(Q, "_run_boundary_equivalence_stage", fake_boundary)
    monkeypatch.setattr(Q, "_run_trial", fake_trial)
    aggregate = Q.qualify(
        fixture_manifest=manifest_path,
        qualification_root=qualification_root,
    )
    assert calls == [(row[0], row[1]) for row in Q.SCHEDULE]
    assert stage_events == ["BOUNDARY_BEFORE_TRIALS"]
    assert aggregate["status"] == "PASS_SOURCE_FREE_QUALIFICATION"
    assert aggregate["qualification_root_claimed_before_fixture_preflight"] is True
    assert aggregate["per_mode_result_bytes_identical"] == {
        "LEGACY_BATCH": True,
        "SEQUENTIAL": True,
    }
    assert aggregate["final_consumed_sha256_recheck_after_trial_6_passed"] is True
    assert aggregate["boundary_equivalence_completed_before_measured_trial_1"] is True
    assert contexts[-1] == "immediately after qualification trial 6"
    persisted = json.loads(
        (qualification_root / Q.AGGREGATE_NAME).read_text(encoding="utf-8")
    )
    assert persisted == aggregate


def test_fixture_preflight_failure_after_root_claim_always_publishes_terminal_aggregate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    qualification_root = tmp_path / "qualification"
    _bind_small_fixture(monkeypatch, manifest_path, bind_row_counts=True)
    monkeypatch.setattr(Q, "QUALIFICATION_ROOT", qualification_root)
    _bind_nonisolated_unit_runtime(monkeypatch)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fixtures"][3]["path"] = "data/physical/N010.ply"
    Q._strict_write(manifest_path, payload)
    with pytest.raises(ValueError, match="identity/path/metadata drift"):
        Q.qualify(
            fixture_manifest=manifest_path,
            qualification_root=qualification_root,
        )
    aggregate = json.loads(
        (qualification_root / Q.AGGREGATE_NAME).read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "TERMINAL_PRE_PHYSICAL_NONQUALIFICATION"
    assert aggregate["qualification_root_claimed_before_fixture_preflight"] is True
    assert aggregate["failed_trial_ordinal"] is None
    assert aggregate["completed_trials"] == []


def test_formal_boundary_failure_is_terminal_before_any_measured_trial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    qualification_root = tmp_path / "qualification"
    _bind_small_fixture(monkeypatch, manifest_path, bind_row_counts=True)
    monkeypatch.setattr(Q, "QUALIFICATION_ROOT", qualification_root)
    _bind_nonisolated_unit_runtime(monkeypatch)

    def fail_boundary(**_kwargs: object) -> object:
        assert qualification_root.is_dir()
        assert not any(path.is_dir() for path in qualification_root.iterdir())
        raise RuntimeError("synthetic boundary oracle mismatch")

    def forbidden_trial(**_kwargs: object) -> object:
        raise AssertionError("measured trial launched after boundary failure")

    monkeypatch.setattr(Q, "_run_boundary_equivalence_stage", fail_boundary)
    monkeypatch.setattr(Q, "_run_trial", forbidden_trial)
    with pytest.raises(RuntimeError, match="boundary oracle mismatch"):
        Q.qualify(
            fixture_manifest=manifest_path,
            qualification_root=qualification_root,
        )
    aggregate = json.loads(
        (qualification_root / Q.AGGREGATE_NAME).read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "TERMINAL_PRE_PHYSICAL_NONQUALIFICATION"
    assert aggregate["failed_qualification_stage"] == (
        "EXACT_SIZE_OPEN3D_BOUNDARY_EQUIVALENCE"
    )
    assert aggregate["failed_trial_ordinal"] is None
    assert aggregate["completed_trials"] == []


@pytest.mark.parametrize(
    ("reason", "timed_out", "rss_exceeded", "sampler_failed"),
    (
        ("WALL_LIMIT", True, False, False),
        ("RSS_LIMIT", False, True, False),
        ("RSS_SAMPLER_FAILURE", False, False, True),
    ),
)
def test_trial_fails_closed_on_watchdog_timeout_rss_or_sampler_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: str,
    timed_out: bool,
    rss_exceeded: bool,
    sampler_failed: bool,
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    runtime_identity = _bind_nonisolated_unit_runtime(monkeypatch)
    frozen = Q.consumed_sha256_map(manifest_path)
    trial = tmp_path / "trial"
    trial.mkdir()

    def fake_monitor(command: object, *, measurement_root: Path, **_kwargs: object) -> dict[str, object]:
        assert command[:4] == [
            str(Q.EXPECTED_PYTHON_INVOCATION),
            "-I",
            "-B",
            str(Q.ROOT / "scripts/run_exp_cad_0035_source_free_memory_qualification.py"),
        ]
        measurement_root.mkdir(parents=True)
        (measurement_root / "stdout.txt").write_text("", encoding="utf-8")
        (measurement_root / "stderr.txt").write_text(reason, encoding="utf-8")
        return {
            "status": "FAIL_TERMINAL",
            "returncode": -15,
            "timed_out": timed_out,
            "rss_limit_exceeded": rss_exceeded,
            "sampler_failed": sampler_failed,
            "resource_kill_reason": reason,
            "wall_seconds": 0.1,
            "sampled_process_tree_peak_rss_bytes": 1,
            "exact_worker_process_max_rss_bytes": None,
            "peak_process_tree_rss_bytes": 1,
        }

    monkeypatch.setattr(Q.MONITOR, "run_measured_command", fake_monitor)
    with pytest.raises(RuntimeError, match="trial 1 failed"):
        Q._run_trial(
            ordinal=1,
            mode="SEQUENTIAL",
            trial_root=trial,
            fixture_manifest=manifest_path,
            frozen_consumed_sha256=frozen,
            frozen_runtime_identity=runtime_identity,
        )
    measurement = json.loads((trial / Q.MEASUREMENT_NAME).read_text(encoding="utf-8"))
    assert measurement["resource_kill_reason"] == reason
    assert measurement["timed_out"] is timed_out
    assert measurement["rss_limit_exceeded"] is rss_exceeded
    assert measurement["sampler_failed"] is sampler_failed


def test_first_trial_failure_publishes_terminal_aggregate_and_stops_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    qualification_root = tmp_path / "qualification"
    calls: list[int] = []

    def fail_trial(**kwargs: object) -> object:
        calls.append(int(kwargs["ordinal"]))
        raise RuntimeError("synthetic watchdog failure")

    _bind_small_fixture(monkeypatch, manifest_path, bind_row_counts=True)
    monkeypatch.setattr(Q, "QUALIFICATION_ROOT", qualification_root)
    runtime_identity = _bind_nonisolated_unit_runtime(monkeypatch)
    monkeypatch.setattr(
        Q,
        "_run_boundary_equivalence_stage",
        lambda **_kwargs: {
            "status": "PASS_EXACT_SIZE_OPEN3D_BOUNDARY_EQUIVALENCE",
            "runtime_identity": runtime_identity,
        },
    )
    monkeypatch.setattr(Q, "_run_trial", fail_trial)
    with pytest.raises(RuntimeError, match="synthetic watchdog failure"):
        Q.qualify(
            fixture_manifest=manifest_path,
            qualification_root=qualification_root,
        )
    assert calls == [1]
    aggregate = json.loads(
        (qualification_root / Q.AGGREGATE_NAME).read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "TERMINAL_PRE_PHYSICAL_NONQUALIFICATION"
    assert aggregate["failed_trial_ordinal"] == 1
    assert aggregate["later_trials_launched"] is False
    assert aggregate["physical_scan_access_count"] == 0
    assert aggregate["evaluator_access_count"] == 0


def test_consumed_code_or_fixture_drift_rejects_before_worker_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    runtime_identity = _bind_nonisolated_unit_runtime(monkeypatch)
    frozen = Q.consumed_sha256_map(manifest_path)
    frozen["memory_worker"] = "0" * 64
    trial = tmp_path / "trial"
    trial.mkdir()

    def forbidden_monitor(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("TOCTOU gate must precede worker spawn")

    monkeypatch.setattr(Q.MONITOR, "run_measured_command", forbidden_monitor)
    with pytest.raises(RuntimeError, match="TOCTOU before worker spawn"):
        Q._run_trial(
            ordinal=1,
            mode="SEQUENTIAL",
            trial_root=trial,
            fixture_manifest=manifest_path,
            frozen_consumed_sha256=frozen,
            frozen_runtime_identity=runtime_identity,
        )
    assert not (trial / Q.RESULT_NAME).exists()
    assert not (trial / Q.MEASUREMENT_NAME).exists()


def test_runtime_identity_records_full_live_fingerprint_and_exact_module_map() -> None:
    identity = Q._runtime_identity_snapshot(strict_sys_path=False)
    environment = identity["runtime_environment_fingerprint"]
    modules = identity["loaded_module_origin_path_sha256_map"]
    assert set(environment) == set(Q.G35.ENVIRONMENT_FINGERPRINT_KEYS)
    assert identity["runtime_environment_fingerprint_sha256"] == Q.hashlib.sha256(
        Q._strict_bytes(environment)
    ).hexdigest()
    assert set(modules) == {
        "numpy",
        "scipy",
        "open3d",
        "exp29_cad_core",
        "exp29_frozen_source",
        "exp29_frozen_runner",
        "exp31_intake",
        "exp32_v5",
        "exp34_contract",
        "exp35_contract",
        "exp35_monitor",
        "exp35_sequential_runner",
    }
    assert identity["loaded_module_origin_path_sha256_map_sha256"] == (
        Q.hashlib.sha256(Q._strict_bytes(modules)).hexdigest()
    )
    for key, expected in Q.IMMUTABLE_INHERITED_MODULE_SHA256.items():
        assert modules[key]["sha256"] == expected


def test_exact_runtime_gate_rejects_nonisolated_python() -> None:
    with pytest.raises(RuntimeError, match="preloaded candidate modules|isolated Python -I"):
        Q._runtime_identity_snapshot(strict_sys_path=True)


def test_exact_isolated_python_runtime_probe_passes() -> None:
    code = (
        f"import sys;sys.path.insert(0,{str(Q.ROOT)!r});"
        "from scripts import run_exp_cad_0035_source_free_memory_qualification as q;"
        "q._runtime_identity_snapshot(strict_sys_path=True);print('PASS')"
    )
    completed = subprocess.run(
        [str(Q.EXPECTED_PYTHON_INVOCATION), "-I", "-B", "-c", code],
        cwd=Q.ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PASS"


def test_exact_runtime_rejects_preloaded_candidate_before_call() -> None:
    code = (
        f"import sys;sys.path.insert(0,{str(Q.ROOT)!r});"
        "from scripts import exp_cad_0035_execution_contract as g;"
        "g.runtime_environment_fingerprint=lambda:print('CALLED');"
        "from scripts import run_exp_cad_0035_source_free_memory_qualification as q;"
        "\ntry:q._runtime_identity_snapshot(strict_sys_path=True)\n"
        "except RuntimeError as e:\n"
        " print('REJECTED' if 'preloaded candidate modules' in str(e) else repr(e))"
    )
    completed = subprocess.run(
        [str(Q.EXPECTED_PYTHON_INVOCATION), "-I", "-B", "-c", code],
        cwd=Q.ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "REJECTED"


def test_exact_runtime_rejects_custom_project_loader_before_call() -> None:
    target = "scripts.exp_cad_0035_execution_contract"
    target_path = str(Q.PROJECT_MODULE_PATHS["exp35_contract"])
    code = f'''
import importlib.abc,importlib.machinery,importlib.util,sys
from pathlib import Path
root=Path({str(Q.ROOT)!r});sys.path.insert(0,str(root))
target={target!r};target_path={target_path!r}
class Loader(importlib.abc.Loader):
 def create_module(self,spec): return None
 def exec_module(self,module):
  importlib.machinery.SourceFileLoader(target,target_path).exec_module(module)
  module.__loader__=self;module.__spec__.loader=self
  module.runtime_environment_fingerprint=lambda:print("CALLED")
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname==globals()["target"]:
   return importlib.util.spec_from_file_location(fullname,target_path,loader=Loader())
  return None
sys.meta_path.insert(0,Finder())
from scripts import run_exp_cad_0035_source_free_memory_qualification as q
try:q._runtime_identity_snapshot(strict_sys_path=True)
except RuntimeError as e:
 print("REJECTED" if "pre-fingerprint contract module drift" in str(e) else repr(e))
'''
    completed = subprocess.run(
        [str(Q.EXPECTED_PYTHON_INVOCATION), "-I", "-B", "-c", code],
        cwd=Q.ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "REJECTED"


def test_runtime_gate_rejects_pythonpath_before_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/alternate-pythonpath")

    def forbidden_fingerprint() -> object:
        raise AssertionError("unsafe environment must fail before fingerprinting")

    monkeypatch.setattr(Q.G35, "runtime_environment_fingerprint", forbidden_fingerprint)
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_runtime_gate_rejects_untrusted_sys_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Q.sys, "path", [str(Q.ROOT), str(tmp_path)])
    with pytest.raises(RuntimeError, match="untrusted entry"):
        Q._validate_safe_sys_path()


def test_runtime_gate_rejects_symlinked_module_path(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("# source-free unit fixture\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink component"):
        Q._require_regular_unsymlinked_file(link, label="synthetic module")


def test_runtime_gate_rejects_alternate_sys_modules_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate = types.ModuleType("numpy")
    alternate.__package__ = Q.np.__package__
    alternate.__file__ = Q.np.__file__
    alternate.__spec__ = Q.np.__spec__
    alternate.__version__ = Q.np.__version__
    monkeypatch.setitem(Q.sys.modules, "numpy", alternate)
    with pytest.raises(RuntimeError, match="sys.modules object identity drift: numpy"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_runtime_gate_rejects_stale_v5_dependency_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Q.V5, "FROZEN_INTAKE", object())
    with pytest.raises(RuntimeError, match="dependency object identity drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


@pytest.mark.parametrize("metadata_field", ("__file__", "spec_origin"))
def test_runtime_gate_rejects_module_file_or_spec_origin_drift(
    monkeypatch: pytest.MonkeyPatch, metadata_field: str
) -> None:
    if metadata_field == "__file__":
        monkeypatch.setattr(Q.INTAKE, "__file__", "/tmp/alternate-intake.py")
    else:
        monkeypatch.setattr(Q.INTAKE.__spec__, "origin", "/tmp/alternate-intake.py")
    with pytest.raises(RuntimeError, match="metadata/origin drift: exp31_intake"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_runtime_gate_rejects_stale_function_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Q.INTAKE, "pair_actionability", lambda *_args: {})
    with pytest.raises(RuntimeError, match="critical function identity drift: exp31_intake"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_runtime_gate_rejects_in_place_canonical_function_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace: dict[str, object] = {}
    exec(
        compile(
            "def replacement(*_args, **_kwargs):\n    return {}\n",
            str(Q.PROJECT_MODULE_PATHS["exp31_intake"]),
            "exec",
        ),
        Q.INTAKE.__dict__,
        namespace,
    )
    replacement = namespace["replacement"]
    assert replacement.__globals__ is Q.INTAKE.__dict__
    assert replacement.__code__.co_filename == str(
        Q.PROJECT_MODULE_PATHS["exp31_intake"]
    )
    monkeypatch.setattr(Q.INTAKE, "pair_actionability", replacement)
    with pytest.raises(RuntimeError, match="critical function identity drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_runtime_gate_rejects_fingerprint_dependency_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def poisoned_compiler() -> str:
        calls.append("CALLED")
        return "POISONED_COMPILER"

    monkeypatch.setattr(Q.G35.platform, "python_compiler", poisoned_compiler)
    with pytest.raises(RuntimeError, match="runtime dependency callable drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)
    assert calls == []


def test_runtime_gate_rejects_scripts_namespace_contamination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = Q.sys.modules["scripts"]
    monkeypatch.setattr(
        namespace.__spec__,
        "submodule_search_locations",
        [str(Q.ROOT / "scripts"), str(tmp_path)],
    )
    with pytest.raises(RuntimeError, match="namespace package path/identity drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_runtime_gate_rejects_interpreter_invocation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Q.sys, "executable", "/tmp/alternate-python")
    with pytest.raises(RuntimeError, match="Python invocation path drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_runtime_gate_rejects_full_g35_fingerprint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = Q.G35.runtime_environment_fingerprint()
    live["python_compiler"] = "alternate compiler"
    monkeypatch.setattr(Q.G35, "runtime_environment_fingerprint", lambda: live)
    with pytest.raises(RuntimeError, match="pre-fingerprint contract function drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_swapped_g35_object_is_rejected_before_fingerprint_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sentinel = types.ModuleType("scripts.exp_cad_0035_execution_contract")

    def forbidden_fingerprint() -> object:
        calls.append("CALLED")
        raise AssertionError("swapped G35 fingerprint must never execute")

    sentinel.runtime_environment_fingerprint = forbidden_fingerprint
    monkeypatch.setattr(Q, "G35", sentinel)
    with pytest.raises(RuntimeError, match="pre-fingerprint contract module drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)
    assert calls == []


def test_runtime_gate_rejects_immutable_inherited_module_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = dict(Q.IMMUTABLE_INHERITED_MODULE_SHA256)
    expected["exp29_cad_core"] = "0" * 64
    monkeypatch.setattr(Q, "IMMUTABLE_INHERITED_MODULE_SHA256", expected)
    with pytest.raises(RuntimeError, match="immutable inherited module SHA drift"):
        Q._runtime_identity_snapshot(strict_sys_path=False)


def test_worker_runtime_failure_precedes_fixture_access_and_science(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_runtime(*, strict_sys_path: bool) -> object:
        assert strict_sys_path is True
        raise RuntimeError("synthetic interpreter drift")

    def forbidden_fixture(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fixture accessed after runtime preflight failure")

    monkeypatch.setattr(Q, "_runtime_identity_snapshot", fail_runtime)
    monkeypatch.setattr(Q, "_fixture_inputs", forbidden_fixture)
    with pytest.raises(RuntimeError, match="synthetic interpreter drift"):
        Q.worker_compute(
            mode="SEQUENTIAL",
            fixture_manifest=tmp_path / "must-not-be-read.json",
            require_exact_counts=True,
        )


def test_controller_runtime_failure_is_root_first_terminal_before_fixture_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    qualification_root = tmp_path / "qualification"
    _bind_small_fixture(monkeypatch, manifest_path, bind_row_counts=True)
    monkeypatch.setattr(Q, "QUALIFICATION_ROOT", qualification_root)

    def fail_runtime(*, strict_sys_path: bool) -> object:
        assert qualification_root.is_dir()
        assert strict_sys_path is True
        raise RuntimeError("synthetic poisoned module")

    def forbidden_fixture(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fixture accessed after runtime preflight failure")

    monkeypatch.setattr(Q, "_runtime_identity_snapshot", fail_runtime)
    monkeypatch.setattr(Q, "_fixture_inputs", forbidden_fixture)
    with pytest.raises(RuntimeError, match="synthetic poisoned module"):
        Q.qualify(
            fixture_manifest=manifest_path,
            qualification_root=qualification_root,
        )
    terminal = json.loads(
        (qualification_root / Q.AGGREGATE_NAME).read_text(encoding="utf-8")
    )
    assert terminal["failed_qualification_stage"] == (
        "RUNTIME_ENVIRONMENT_PREFLIGHT_AFTER_ROOT_CLAIM"
    )
    assert terminal["runtime_identity"] is None
    assert terminal["completed_trials"] == []


def test_trial_runtime_toctou_rejects_before_monitor_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, _manifest = _small_fixture_set(tmp_path)
    _bind_small_fixture(monkeypatch, manifest_path)
    frozen_runtime = Q._runtime_identity_snapshot(strict_sys_path=False)
    frozen_consumed = Q.consumed_sha256_map(manifest_path)
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    changed = json.loads(Q._strict_bytes(frozen_runtime).decode("utf-8"))
    changed["runtime_environment_fingerprint"]["python_compiler"] = "drift"
    changed["runtime_environment_fingerprint_sha256"] = Q.hashlib.sha256(
        Q._strict_bytes(changed["runtime_environment_fingerprint"])
    ).hexdigest()
    monkeypatch.setattr(
        Q,
        "_runtime_identity_snapshot",
        lambda *, strict_sys_path: changed,
    )

    def forbidden_monitor(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("monitor spawned after runtime TOCTOU")

    monkeypatch.setattr(Q.MONITOR, "run_measured_command", forbidden_monitor)
    with pytest.raises(RuntimeError, match="runtime/module identity TOCTOU"):
        Q._run_trial(
            ordinal=1,
            mode="SEQUENTIAL",
            trial_root=trial_root,
            fixture_manifest=manifest_path,
            frozen_consumed_sha256=frozen_consumed,
            frozen_runtime_identity=frozen_runtime,
        )
    assert not (trial_root / Q.MEASUREMENT_NAME).exists()
