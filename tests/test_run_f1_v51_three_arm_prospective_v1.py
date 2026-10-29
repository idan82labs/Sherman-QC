from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scripts.run_f1_v51_three_arm_prospective_v1 as prospective
from scripts.run_f1_v51_three_arm_prospective_v1 import (
    ARM_FILENAMES,
    COMPONENT_LOCK_KIND,
    COMPONENT_RELATIVE_PATHS,
    FREEZE_PACKET_KIND,
    FROZEN_V51_HELPER_RELATIVE_PATH,
    INPUT_KIND,
    ROOT,
    SEALED_FREEZE_PACKET_SHA256,
    run_prospective,
    validate_freeze_packet,
    validate_scan_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_freeze_packet(tmp_path: Path) -> Path:
    sealed = ROOT / "research/f1_v51_20260831_prospective_three_arm_freeze_v1.json"
    packet = tmp_path / "freeze.json"
    packet.write_bytes(sealed.read_bytes())
    return packet


def _write_scan_manifest(tmp_path: Path, count: int = 2) -> Path:
    rows = []
    scan_root = tmp_path / "opaque_scans"
    scan_root.mkdir(parents=True, exist_ok=True)
    for ordinal in range(1, count + 1):
        scan = scan_root / f"T{ordinal:03d}.ply"
        scan.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
            encoding="ascii",
        )
        rows.append({"scan_path": str(scan), "scan_sha256": _sha(scan)})
    manifest = tmp_path / "scans.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": INPUT_KIND,
                "cases": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _source_result(predictions: list[dict] | None = None) -> dict:
    rows = list(predictions or [])
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_single_axis_parallel_return_recovery_result",
        "profile": "synthetic_mock",
        "candidate_count": len(rows),
        "predictions": rows,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
    }


def _mock_base_runner(
    case: dict,
    root: Path,
    pass_name: str,
    _v9: Path,
    _v4: Path,
) -> dict:
    case_root = root / pass_name / case["case_id"]
    pool_root = case_root / "pool"
    pool_root.mkdir(parents=True, exist_ok=True)
    (case_root / "semantic_result.json").write_text(
        json.dumps(_source_result(), sort_keys=True), encoding="utf-8"
    )
    (case_root / "v16.json").write_text(
        json.dumps(_source_result(), sort_keys=True), encoding="utf-8"
    )
    v43 = _source_result()
    v43.update(
        {
            "retained_v16_prediction_count": 0,
            "v43_post_merge_addition_count": 0,
        }
    )
    (case_root / "v43.json").write_text(json.dumps(v43, sort_keys=True), encoding="utf-8")
    (pool_root / f"{case['case_id']}.json").write_text(
        json.dumps(
            {"working_point_budget": 200000, "line_objects": [], "candidates": []},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "case_id": case["case_id"],
        "pass": pass_name,
        "scan_sha256": case["scan_sha256"],
    }


def test_scan_manifest_is_ordered_opaque_and_side_information_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prospective, "OPAQUE_SCAN_ROOT", (tmp_path / "opaque_scans").resolve()
    )
    manifest = _write_scan_manifest(tmp_path, count=3)
    cases = validate_scan_manifest(manifest)
    assert [row["case_id"] for row in cases] == ["T001", "T002", "T003"]
    assert all(set(row) == {"case_id", "scan_path", "scan_sha256"} for row in cases)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_count"] = 7
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only scan_path and scan_sha256"):
        validate_scan_manifest(manifest)


def test_scan_manifest_rejects_identity_bearing_filename(tmp_path: Path) -> None:
    scan = tmp_path / "customer-part-123.ply"
    scan.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
    manifest = tmp_path / "scans.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": INPUT_KIND,
                "cases": [{"scan_path": str(scan), "scan_sha256": _sha(scan)}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="opaque case ID"):
        validate_scan_manifest(manifest)


def test_scan_manifest_rejects_identity_bearing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prospective, "OPAQUE_SCAN_ROOT", (tmp_path / "canonical/opaque_scans").resolve()
    )
    scan = tmp_path / "customer-123/T001.ply"
    scan.parent.mkdir(parents=True)
    scan.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
    manifest = tmp_path / "scans.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": INPUT_KIND,
                "cases": [{"scan_path": str(scan), "scan_sha256": _sha(scan)}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical opaque scan root"):
        validate_scan_manifest(manifest)


def test_stdlib_bootstrap_and_help_do_not_preload_scientific_modules(
    tmp_path: Path,
) -> None:
    runner = ROOT / "scripts/run_f1_v51_three_arm_prospective_v1.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    forbidden_static = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {
            "backend",
            "scripts",
        }:
            forbidden_static.append(node.module)
        if isinstance(node, ast.Import):
            forbidden_static.extend(
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] in {"backend", "scripts"}
            )
    assert forbidden_static == []

    probe = textwrap.dedent(
        f"""
        import runpy
        import sys
        names = {sorted(prospective.LOCKED_SCIENTIFIC_MODULE_NAMES)!r}
        import scripts.run_f1_v51_three_arm_prospective_v1
        assert not [name for name in names if name in sys.modules]
        sys.argv = [{str(runner)!r}, '--help']
        try:
            runpy.run_path({str(runner)!r}, run_name='__main__')
        except SystemExit as exc:
            assert exc.code == 0
        assert not [name for name in names if name in sys.modules]
        """
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_authoritative_entry_has_no_runner_override_and_rejects_preload(
    tmp_path: Path,
) -> None:
    assert "base_case_runner" not in inspect.signature(run_prospective).parameters
    probe = textwrap.dedent(
        """
        from pathlib import Path
        import backend.f1_post_v51_v43_explosion_guard_v1
        from scripts.run_f1_v51_three_arm_prospective_v1 import run_prospective
        try:
            run_prospective(Path('missing-scans'), Path('missing-freeze'), Path('missing-output'))
        except RuntimeError as exc:
            assert 'requires a fresh process' in str(exc)
        else:
            raise AssertionError('preloaded scientific module was not rejected')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_freeze_packet_is_hard_pinned_and_helper_path_is_explicit(tmp_path: Path) -> None:
    packet = _write_freeze_packet(tmp_path)
    lock, resolved = validate_freeze_packet(packet)
    assert all(key in lock for key in COMPONENT_RELATIVE_PATHS)
    assert resolved["axis_span_model"].is_file()
    assert resolved["frozen_v51_helper"] == (ROOT / FROZEN_V51_HELPER_RELATIVE_PATH).resolve()

    sidecar = packet.with_suffix(".sha256")
    sidecar.write_text(
        f"{SEALED_FREEZE_PACKET_SHA256}  {packet.name}\n", encoding="ascii"
    )
    validate_freeze_packet(packet)
    sidecar.write_text(f"{'0' * 64}  {packet.name}\n", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        validate_freeze_packet(packet)
    sidecar.unlink()

    payload = json.loads(packet.read_text(encoding="utf-8"))
    payload["component_lock"]["axis_span_model"]["path"] = (
        "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1.json"
    )
    packet.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hard-pinned seal"):
        validate_freeze_packet(packet)

    packet = _write_freeze_packet(tmp_path)
    payload = json.loads(packet.read_text(encoding="utf-8"))
    payload["component_lock"]["recall_model"]["sha256"] = "0" * 64
    packet.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hard-pinned seal"):
        validate_freeze_packet(packet)

    packet = _write_freeze_packet(tmp_path)
    _, resolved = validate_freeze_packet(packet)
    wrong_helper = dict(resolved)
    wrong_helper["frozen_v51_helper"] = resolved["v51_case_runner"]
    with pytest.raises(RuntimeError, match="frozen V51 helper.*locked runtime path"):
        prospective._load_runtime_bundle(wrong_helper)


def test_mock_three_arm_primary_replay_and_deterministic_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prospective, "OPAQUE_SCAN_ROOT", (tmp_path / "opaque_scans").resolve()
    )
    packet = _write_freeze_packet(tmp_path)
    scans = _write_scan_manifest(tmp_path, count=2)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    validation_calls: list[Path] = []
    original_validate = prospective.validate_freeze_packet

    def tracked_validate(path: Path):
        validation_calls.append(Path(path))
        return original_validate(path)

    monkeypatch.setattr(prospective, "validate_freeze_packet", tracked_validate)

    first = prospective._run_prospective_core(
        scans,
        packet,
        first_root,
        max_workers=2,
        _base_case_runner=_mock_base_runner,
        _test_allow_preloaded_scientific_modules=True,
    )
    second = prospective._run_prospective_core(
        scans,
        packet,
        second_root,
        max_workers=1,
        _base_case_runner=_mock_base_runner,
        _test_allow_preloaded_scientific_modules=True,
    )

    assert first == second
    assert len(validation_calls) == 8  # four full-inventory validations per execution
    assert first["case_ids"] == ["T001", "T002"]
    assert first["all_arms_primary_replay_byte_identical"] is True
    assert first["truth_opened"] is False
    assert first["labels_used"] is False
    assert first["external_counts_used"] is False
    assert first["cad_used"] is False
    assert first["part_identity_used"] is False
    assert set(first["arm_manifest_sha256"]) == set(ARM_FILENAMES)

    first_json = {
        path.relative_to(first_root): path.read_bytes()
        for path in first_root.rglob("*.json")
    }
    second_json = {
        path.relative_to(second_root): path.read_bytes()
        for path in second_root.rglob("*.json")
    }
    assert first_json == second_json
    for pass_name in ("primary", "replay"):
        for case_id in ("T001", "T002"):
            for filename in ARM_FILENAMES.values():
                assert (first_root / pass_name / case_id / "arms" / filename).is_file()


def test_base_failure_is_three_arm_fail_closed_and_replay_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prospective, "OPAQUE_SCAN_ROOT", (tmp_path / "opaque_scans").resolve()
    )
    packet = _write_freeze_packet(tmp_path)
    scans = _write_scan_manifest(tmp_path, count=1)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    output = tmp_path / "failed"
    result = prospective._run_prospective_core(
        scans,
        packet,
        output,
        _base_case_runner=fail,
        _test_allow_preloaded_scientific_modules=True,
    )
    assert result["all_arms_primary_replay_byte_identical"] is True
    case = json.loads((output / "case_manifests/T001.json").read_text(encoding="utf-8"))
    assert case["base_primary_status"] == "FAILED_ABSTAIN"
    assert case["base_replay_status"] == "FAILED_ABSTAIN"
    assert all(arm["candidate_count"] == 0 for arm in case["arms"].values())


def test_any_arm_replay_drift_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prospective, "OPAQUE_SCAN_ROOT", (tmp_path / "opaque_scans").resolve()
    )
    packet = _write_freeze_packet(tmp_path)
    scans = _write_scan_manifest(tmp_path, count=1)

    def drifting_runner(case: dict, root: Path, pass_name: str, _v9: Path, _v4: Path) -> dict:
        _mock_base_runner(case, root, pass_name, _v9, _v4)
        prediction = {
            "bend_id": "mock-bend",
            "line_start": [0.0, 0.0, 0.0],
            "line_end": [10.0, 0.0 if pass_name == "primary" else 1.0, 0.0],
            "source_kind": "mock",
            "selector_score": 1.0,
        }
        semantic = root / pass_name / case["case_id"] / "semantic_result.json"
        semantic.write_text(
            json.dumps(_source_result([prediction]), sort_keys=True), encoding="utf-8"
        )
        return {
            "case_id": case["case_id"],
            "pass": pass_name,
            "scan_sha256": case["scan_sha256"],
        }

    with pytest.raises(RuntimeError, match="primary/replay bytes differ"):
        prospective._run_prospective_core(
            scans,
            packet,
            tmp_path / "drift",
            _base_case_runner=drifting_runner,
            _test_allow_preloaded_scientific_modules=True,
        )


def test_helper_scan_identity_mismatch_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prospective, "OPAQUE_SCAN_ROOT", (tmp_path / "opaque_scans").resolve()
    )
    packet = _write_freeze_packet(tmp_path)
    scans = _write_scan_manifest(tmp_path, count=1)

    def wrong_identity_runner(
        case: dict, root: Path, pass_name: str, v9: Path, v4: Path
    ) -> dict:
        record = _mock_base_runner(case, root, pass_name, v9, v4)
        record["scan_sha256"] = "0" * 64
        return record

    with pytest.raises(prospective.CustodyIntegrityError, match="helper record scan"):
        prospective._run_prospective_core(
            scans,
            packet,
            tmp_path / "wrong-identity",
            _base_case_runner=wrong_identity_runner,
            _test_allow_preloaded_scientific_modules=True,
        )


def test_inflight_scan_mutation_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prospective, "OPAQUE_SCAN_ROOT", (tmp_path / "opaque_scans").resolve()
    )
    packet = _write_freeze_packet(tmp_path)
    scans = _write_scan_manifest(tmp_path, count=1)

    def mutating_runner(
        case: dict, root: Path, pass_name: str, v9: Path, v4: Path
    ) -> dict:
        record = _mock_base_runner(case, root, pass_name, v9, v4)
        with Path(case["scan_path"]).open("ab") as handle:
            handle.write(b"mutated")
        return record

    with pytest.raises(prospective.CustodyIntegrityError, match="scan bytes changed"):
        prospective._run_prospective_core(
            scans,
            packet,
            tmp_path / "mutated-scan",
            _base_case_runner=mutating_runner,
            _test_allow_preloaded_scientific_modules=True,
        )


def test_direct_entry_help_from_foreign_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/run_f1_v51_three_arm_prospective_v1.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--freeze-packet" in completed.stdout
