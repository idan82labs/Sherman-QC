from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import scripts.optimize_f1_train18_source_only_v1 as optimizer

from backend.f1_train18_frozen_source_only_candidate_v1 import (
    MODEL_KIND,
    REPRESENTATIONS,
    canonical_bytes,
    compose_frozen_candidate,
    feature_names,
    sha256_bytes,
    validate_model,
)
from scripts.optimize_f1_train18_source_only_v1 import (
    ATTESTATION_KIND,
    PROJECTION_KIND,
    TRAIN_CACHE_ATTESTATION_KIND,
    TRAIN_CACHE_MANIFEST_KIND,
    LEARNER_SPECS,
    CaseData,
    build_guarded_reader,
    case_folds,
    load_lock_context,
    load_train_cases,
    fit_portable_classifier,
    predict_with_classifier,
    run_optimizer,
    training_gate,
)


def _candidate(token: str, *, y: float = 0.0, score: float = 0.8) -> dict:
    return {
        "candidate_id": token,
        "anchor": [5.0, y, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 10.0,
        "span_endpoints": [[0.0, y, 0.0], [10.0, y, 0.0]],
        "candidate_source": "plane_lattice",
        "representative_source": "plane_lattice",
        "score": score,
        "finite_support_fraction": 0.9,
        "normal_second_eigenvalue": 0.4,
        "two_sided_transition_score": 0.8,
    }


def _pool(*candidates: dict) -> dict:
    rows = list(candidates)
    return {
        "working_point_budget": 200000,
        "candidates": rows,
        "line_objects": rows,
    }


def _v51(*predictions: dict, case_id: str | None = None, scan_sha256: str | None = None) -> dict:
    payload = {
        "kind": "synthetic-v51",
        "predictions": list(predictions),
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
    }
    if case_id is not None:
        payload["case_id"] = case_id
    if scan_sha256 is not None:
        payload["scan_sha256"] = scan_sha256
    return payload


def _model(*, cap: object = 2, classifier: dict | None = None) -> dict:
    names = feature_names("absolute_quality")
    return {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "freeze_status": "FROZEN_BEFORE_HOLDOUT_SCORE",
        "holdout_score_opened": False,
        "source_only": True,
        "case_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "count_feature_used": False,
        "representation": "absolute_quality",
        "universe": "raw_reselect",
        "feature_names": list(names),
        "classifier": classifier
        or {
            "kind": "constant",
            "imputer_medians": [0.0] * len(names),
            "positive_probability": 1.0,
        },
        "profile_name": "SYNTHETIC_TEST",
        "profile": {
            "probability_threshold": 0.5,
            "output_cap": cap,
            "duplicate_angle_deg": 5.0,
            "duplicate_offset_mm": 4.0,
            "duplicate_span_overlap": 0.60,
            "span_scales": [1.0],
            "fixed_spans_mm": [20.0],
        },
        "split_sha256": "1" * 64,
        "train_allowlist_sha256": "2" * 64,
        "preregistration_sha256": "3" * 64,
        "label_projection_attestation_sha256": "4" * 64,
        "train_cache_manifest_sha256": "8" * 64,
        "train_cache_manifest_attestation_sha256": "9" * 64,
        "train_source_cache_binding_sha256": "a" * 64,
        "optimizer_code_sha256": "5" * 64,
        "runtime_code_sha256": "6" * 64,
        "fit_audit_sha256": "7" * 64,
    }


def test_feature_contract_has_no_identity_filename_cad_or_count_features() -> None:
    for representation in REPRESENTATIONS:
        names = feature_names(representation)
        assert names
        lowered = [name.lower() for name in names]
        assert not any("count" in name for name in lowered)
        assert not any("case_id" in name for name in lowered)
        assert not any("filename" in name for name in lowered)
        assert not any("part_id" in name for name in lowered)
        assert not any("cad" in name for name in lowered)


@pytest.mark.parametrize("cap", [0, -1, 2.5, True, "2"])
def test_model_rejects_nonpositive_fractional_or_noninteger_output_cap(cap: object) -> None:
    with pytest.raises(ValueError, match="output cap"):
        validate_model(_model(cap=cap))


def test_model_rejects_negative_internal_tree_feature_index() -> None:
    names = feature_names("absolute_quality")
    classifier = {
        "kind": "tree",
        "imputer_medians": [0.0] * len(names),
        "tree": {
            "children_left": [1, -1, -1],
            "children_right": [2, -1, -1],
            "feature_index": [-1, -2, -2],
            "threshold": [0.5, -2.0, -2.0],
            "positive_probability": [0.5, 0.0, 1.0],
        },
    }
    with pytest.raises(ValueError, match="internal feature index"):
        validate_model(_model(classifier=classifier))


@pytest.mark.parametrize(
    "field",
    [
        "split_sha256",
        "train_allowlist_sha256",
        "preregistration_sha256",
        "label_projection_attestation_sha256",
        "train_cache_manifest_sha256",
        "train_cache_manifest_attestation_sha256",
        "train_source_cache_binding_sha256",
        "optimizer_code_sha256",
        "runtime_code_sha256",
        "fit_audit_sha256",
    ],
)
def test_model_requires_every_pre_holdout_cryptographic_binding(field: str) -> None:
    model = _model()
    model.pop(field)
    with pytest.raises(ValueError, match="binding"):
        validate_model(model)


def test_model_rejects_any_prior_holdout_score_event() -> None:
    model = _model()
    model["holdout_score_opened"] = True
    with pytest.raises(ValueError, match="before any holdout score"):
        validate_model(model)


def test_source_only_compose_is_deterministic_and_pool_order_invariant() -> None:
    first = _candidate("A", y=0.0, score=0.9)
    second = _candidate("B", y=20.0, score=0.8)
    model = _model()
    one = compose_frozen_candidate(_v51(), _pool(first, second), model)
    two = compose_frozen_candidate(_v51(), _pool(second, first), model)
    assert canonical_bytes(one) == canonical_bytes(two)
    assert len(one["predictions"]) == 2
    assert one["labels_used"] is False
    assert one["case_id_used"] is False
    assert one["filename_used"] is False
    assert one["count_feature_used"] is False


@pytest.mark.parametrize(
    "flag", ["labels_used", "expected_count_used", "cad_used", "drawing_used", "part_identity_used"]
)
def test_source_only_compose_rejects_forbidden_runtime_inputs(flag: str) -> None:
    source = _v51()
    source[flag] = True
    with pytest.raises(ValueError, match="forbidden inference input"):
        compose_frozen_candidate(source, _pool(_candidate("A")), _model())


def test_sealed_lock_derives_exact_18_before_label_path_resolution(tmp_path: Path) -> None:
    lock = load_lock_context()
    assert len(lock.train_case_ids) == 18
    assert lock.train_case_ids[0] == "A30-001"
    assert "A30-002" not in lock.train_case_ids
    reader = build_guarded_reader(
        lock,
        label_attestation=tmp_path / "train_projection_attestation.json",
        label_attestation_sha256="1" * 64,
        cache_attestation=tmp_path / "train_cache_attestation.json",
        cache_attestation_sha256="2" * 64,
    )
    full_catalog = (
        Path(__file__).resolve().parents[1]
        / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
    )
    with pytest.raises(PermissionError, match="exact allowlist"):
        reader.read_bytes(full_catalog, purpose="TRAIN_LABEL_PROJECTION")
    assert reader.denied_attempts[-1]["path"] == str(full_catalog.resolve())


def _write_json(path: Path, payload: dict) -> bytes:
    data = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _binding(root: Path, path: Path, data: bytes) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


def _build_synthetic_train_projection_and_sources(
    tmp_path: Path, monkeypatch, *, shared_custody_evidence: bool = False
):
    lock = load_lock_context()
    monkeypatch.setattr(optimizer, "ROOT", tmp_path)
    cache = tmp_path / "cache"
    cases = []
    cache_records = []
    runtime_ids = [f"T{index:03d}" for index in range(28, 10, -1)]
    for index, (case_id, runtime_id) in enumerate(zip(lock.train_case_ids, runtime_ids)):
        scan_sha = lock.scan_sha_by_case[case_id]
        bends = [
            {
                "bend_id": f"{case_id}-B001",
                "line_start": [0.0, float(index), 0.0],
                "line_end": [10.0, float(index), 0.0],
                "reviewer_support_count": index % 3 + 1,
            }
        ]
        cases.append({"case_id": case_id, "scan_sha256": scan_sha, "bends": bends})
        semantic = _v51(case_id=runtime_id, scan_sha256=scan_sha)
        pool = _pool(_candidate(f"{runtime_id}-C1", y=float(index)))
        pool["case_id"] = runtime_id
        pool["inference_contract"] = {
            "part_identifier_exposed": False,
            "expected_count_exposed": False,
            "manual_truth_exposed": False,
            "cad_exposed": False,
        }
        for pass_name in ("primary", "replay"):
            root = cache / pass_name / runtime_id
            semantic_path = root / "semantic_result.json"
            pool_path = root / "pool" / f"{runtime_id}.json"
            semantic_bytes = _write_json(semantic_path, semantic)
            pool_bytes = _write_json(pool_path, pool)
            cache_records.append(
                {
                    "case_id": case_id,
                    "runtime_case_id": runtime_id,
                    "pass": pass_name,
                    "scan_sha256": scan_sha,
                    "semantic_path": str(semantic_path.relative_to(tmp_path)),
                    "semantic_sha256": sha256_bytes(semantic_bytes),
                    "pool_path": str(pool_path.relative_to(tmp_path)),
                    "pool_sha256": sha256_bytes(pool_bytes),
                }
            )
    projection = {
        "schema_version": 1,
        "kind": PROJECTION_KIND,
        "case_count": 18,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "cases": cases,
    }
    projection_path = tmp_path / "custody/train_projection.json"
    projection_bytes = _write_json(projection_path, projection)
    label_builder_path = tmp_path / "custody/label_builder.py"
    label_validator_path = tmp_path / "custody/label_validator.py"
    label_staged_pass_path = tmp_path / "custody/label_staged_validator_pass.json"
    label_access_path = tmp_path / "custody/label_access.json"
    label_builder_bytes = _write_json(label_builder_path, {"source_only_test": True})
    label_validator_bytes = _write_json(label_validator_path, {"independent": True})
    label_staged_pass_bytes = _write_json(label_staged_pass_path, {"status": "PASS"})
    label_access_bytes = _write_json(label_access_path, {"optimizer_holdout_access_count": 0})
    attestation = {
        "schema_version": 1,
        "kind": ATTESTATION_KIND,
        "case_count": 18,
        "file_inventory_count": 18,
        "aggregate_catalog_read_count": 1,
        "holdout_label_row_emit_count": 0,
        "holdout_case_file_emit_count": 0,
        "optimizer_holdout_access_count": 0,
        "train_exact_match": True,
        "train_holdout_intersection_count": 0,
        "projection_contains_only_train": True,
        "independent_validation_passed": True,
        "train_set_commitment_sha256": lock.train_commitment_sha256,
        "blinded_complement_commitment_sha256": "b" * 64,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "projection_binding": {
            "path": str(projection_path.relative_to(tmp_path)),
            "sha256": sha256_bytes(projection_bytes),
            "size_bytes": len(projection_bytes),
            "kind": PROJECTION_KIND,
        },
        "builder_binding": _binding(tmp_path, label_builder_path, label_builder_bytes),
        "validator_binding": _binding(tmp_path, label_validator_path, label_validator_bytes),
        "staged_validator_pass_binding": _binding(
            tmp_path, label_staged_pass_path, label_staged_pass_bytes
        ),
        "access_evidence_binding": _binding(tmp_path, label_access_path, label_access_bytes),
    }
    attestation_path = tmp_path / "custody/train_projection_attestation.json"
    attestation_bytes = _write_json(attestation_path, attestation)

    cache_manifest = {
        "schema_version": 1,
        "kind": TRAIN_CACHE_MANIFEST_KIND,
        "source_only": True,
        "holdout_entry_count": 0,
        "case_count": 18,
        "record_count": 36,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "train_set_commitment_sha256": lock.train_commitment_sha256,
        "records": cache_records,
    }
    cache_manifest_path = tmp_path / "custody/train_cache_manifest.json"
    cache_manifest_bytes = _write_json(cache_manifest_path, cache_manifest)
    cache_builder_path = (
        label_builder_path
        if shared_custody_evidence
        else tmp_path / "custody/cache_builder.py"
    )
    cache_validator_path = (
        label_validator_path
        if shared_custody_evidence
        else tmp_path / "custody/cache_validator.py"
    )
    cache_staged_pass_path = (
        label_staged_pass_path
        if shared_custody_evidence
        else tmp_path / "custody/cache_staged_validator_pass.json"
    )
    cache_access_path = tmp_path / "custody/cache_access.json"
    if shared_custody_evidence:
        cache_builder_bytes = label_builder_bytes
        cache_validator_bytes = label_validator_bytes
        cache_staged_pass_bytes = label_staged_pass_bytes
    else:
        cache_builder_bytes = _write_json(cache_builder_path, {"source_only": True})
        cache_validator_bytes = _write_json(cache_validator_path, {"independent": True})
        cache_staged_pass_bytes = _write_json(cache_staged_pass_path, {"status": "PASS"})
    cache_access_bytes = _write_json(cache_access_path, {"holdout_entry_count": 0})
    cache_attestation = {
        "schema_version": 1,
        "kind": TRAIN_CACHE_ATTESTATION_KIND,
        "case_count": 18,
        "record_count": 36,
        "train_exact_match": True,
        "train_holdout_intersection_count": 0,
        "holdout_entry_count": 0,
        "holdout_status_count": 0,
        "holdout_hash_count": 0,
        "optimizer_holdout_access_count": 0,
        "independent_validation_passed": True,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "train_set_commitment_sha256": lock.train_commitment_sha256,
        "blinded_complement_commitment_sha256": "c" * 64,
        "manifest_binding": {
            "path": str(cache_manifest_path.relative_to(tmp_path)),
            "sha256": sha256_bytes(cache_manifest_bytes),
            "size_bytes": len(cache_manifest_bytes),
            "kind": TRAIN_CACHE_MANIFEST_KIND,
        },
        "builder_binding": _binding(tmp_path, cache_builder_path, cache_builder_bytes),
        "validator_binding": _binding(tmp_path, cache_validator_path, cache_validator_bytes),
        "staged_validator_pass_binding": _binding(
            tmp_path, cache_staged_pass_path, cache_staged_pass_bytes
        ),
        "access_evidence_binding": _binding(tmp_path, cache_access_path, cache_access_bytes),
    }
    cache_attestation_path = tmp_path / "custody/train_cache_attestation.json"
    cache_attestation_bytes = _write_json(cache_attestation_path, cache_attestation)
    return {
        "lock": lock,
        "cache": cache,
        "projection_path": projection_path,
        "attestation_path": attestation_path,
        "attestation_sha": sha256_bytes(attestation_bytes),
        "cache_manifest_path": cache_manifest_path,
        "cache_attestation_path": cache_attestation_path,
        "cache_attestation_sha": sha256_bytes(cache_attestation_bytes),
    }


def test_exact_train_projection_and_case_scoped_sources_load_with_zero_holdout_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    lock = fixture["lock"]
    reader = build_guarded_reader(
        lock,
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
    )
    loaded = load_train_cases(
        lock,
        reader,
        label_attestation=fixture["attestation_path"],
        cache_attestation=fixture["cache_attestation_path"],
    )
    cases = loaded[0]
    assert [case.case_id for case in cases] == list(lock.train_case_ids)
    assert cases[0].runtime_case_id == "T028"
    assert cases[0].runtime_case_id != "T001"
    assert len(cases) == 18
    assert len({case.scan_sha256 for case in cases}) == 18
    assert all(len(value) == 64 for value in loaded[1:])
    assert reader.denied_attempts == []
    case_reads = [row for row in reader.entries if row["case_id"] is not None]
    assert len(case_reads) == 18 * 4
    assert {row["case_id"] for row in case_reads} == set(lock.train_case_ids)


def test_train_projection_rejects_a_holdout_case(tmp_path: Path, monkeypatch) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    lock = fixture["lock"]
    projection_path = fixture["projection_path"]
    attestation_path = fixture["attestation_path"]
    projection = json.loads(projection_path.read_text())
    projection["cases"][0]["case_id"] = "A30-002"
    projection_bytes = _write_json(projection_path, projection)
    attestation = json.loads(attestation_path.read_text())
    attestation["projection_binding"]["sha256"] = sha256_bytes(projection_bytes)
    attestation["projection_binding"]["size_bytes"] = len(projection_bytes)
    attestation_bytes = _write_json(attestation_path, attestation)
    reader = build_guarded_reader(
        lock,
        label_attestation=attestation_path,
        label_attestation_sha256=sha256_bytes(attestation_bytes),
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
    )
    with pytest.raises(PermissionError, match="non-TRAIN case"):
        load_train_cases(
            lock,
            reader,
            label_attestation=attestation_path,
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(
        row["case_id"] == "A30-002" for row in reader.entries
    )


def _refresh_cache_attestation(fixture: dict) -> str:
    manifest_path = fixture["cache_manifest_path"]
    manifest_bytes = canonical_bytes(json.loads(manifest_path.read_text()))
    manifest_path.write_bytes(manifest_bytes)
    attestation_path = fixture["cache_attestation_path"]
    attestation = json.loads(attestation_path.read_text())
    attestation["manifest_binding"]["sha256"] = sha256_bytes(manifest_bytes)
    attestation["manifest_binding"]["size_bytes"] = len(manifest_bytes)
    attestation_bytes = _write_json(attestation_path, attestation)
    return sha256_bytes(attestation_bytes)


def test_invalid_label_attestation_rejects_before_projection_or_case_source_open(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    attestation = json.loads(fixture["attestation_path"].read_text())
    attestation["train_exact_match"] = False
    data = _write_json(fixture["attestation_path"], attestation)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=sha256_bytes(data),
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
    )
    with pytest.raises(ValueError, match="custody attestation failed"):
        load_train_cases(
            fixture["lock"],
            reader,
            label_attestation=fixture["attestation_path"],
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(row["purpose"] == "TRAIN_LABEL_PROJECTION" for row in reader.entries)
    assert not any(row["case_id"] is not None for row in reader.entries)


def test_independent_attestation_hash_mismatch_rejects_before_projection(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256="0" * 64,
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
    )
    with pytest.raises(ValueError, match="authorized read hash mismatch"):
        load_train_cases(
            fixture["lock"],
            reader,
            label_attestation=fixture["attestation_path"],
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(row["purpose"] == "TRAIN_LABEL_PROJECTION" for row in reader.entries)


def test_full_28_case_execution_manifest_is_never_authorized(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
    )
    full_manifest = tmp_path / "cache/development_execution_manifest.json"
    _write_json(full_manifest, {"case_count": 28})
    with pytest.raises(PermissionError, match="exact allowlist"):
        reader.read_bytes(full_manifest, purpose="TRAIN_CACHE_MANIFEST")


def test_cache_attestation_cannot_redirect_to_full_28_case_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    full_manifest = tmp_path / "custody/development_execution_manifest.json"
    full_bytes = _write_json(full_manifest, {"case_count": 28})
    attestation = json.loads(fixture["cache_attestation_path"].read_text())
    attestation["manifest_binding"].update(
        {
            "path": str(full_manifest.relative_to(tmp_path)),
            "sha256": sha256_bytes(full_bytes),
            "size_bytes": len(full_bytes),
        }
    )
    attestation_bytes = _write_json(fixture["cache_attestation_path"], attestation)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=sha256_bytes(attestation_bytes),
    )
    with pytest.raises(PermissionError, match="full 28-case execution manifest"):
        load_train_cases(
            fixture["lock"],
            reader,
            label_attestation=fixture["attestation_path"],
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(row["purpose"] == "TRAIN_LABEL_PROJECTION" for row in reader.entries)


def test_train_cache_manifest_rejects_any_19th_case_record(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    manifest = json.loads(fixture["cache_manifest_path"].read_text())
    extra = dict(manifest["records"][0])
    extra["case_id"] = "A30-002"
    manifest["records"].append(extra)
    _write_json(fixture["cache_manifest_path"], manifest)
    cache_attestation_sha = _refresh_cache_attestation(fixture)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=cache_attestation_sha,
    )
    with pytest.raises(ValueError, match="exactly 36"):
        load_train_cases(
            fixture["lock"],
            reader,
            label_attestation=fixture["attestation_path"],
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(row["purpose"] == "TRAIN_LABEL_PROJECTION" for row in reader.entries)


def test_train_cache_hash_binding_rejects_tampered_pool_before_labels(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    manifest = json.loads(fixture["cache_manifest_path"].read_text())
    pool_path = tmp_path / manifest["records"][0]["pool_path"]
    pool = json.loads(pool_path.read_text())
    pool["tampered"] = True
    _write_json(pool_path, pool)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
    )
    with pytest.raises(ValueError, match="authorized read hash mismatch"):
        load_train_cases(
            fixture["lock"],
            reader,
            label_attestation=fixture["attestation_path"],
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(row["purpose"] == "TRAIN_LABEL_PROJECTION" for row in reader.entries)


def test_label_and_cache_attestations_may_share_exact_custody_evidence_once(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(
        tmp_path, monkeypatch, shared_custody_evidence=True
    )
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
    )
    loaded = load_train_cases(
        fixture["lock"],
        reader,
        label_attestation=fixture["attestation_path"],
        cache_attestation=fixture["cache_attestation_path"],
    )
    assert len(loaded[0]) == 18
    shared_purposes = {
        "CUSTODY_BUILDER_BINDING",
        "CUSTODY_VALIDATOR_BINDING",
        "CUSTODY_STAGED_VALIDATOR_PASS_BINDING",
    }
    for purpose in shared_purposes:
        assert sum(row["purpose"] == purpose for row in reader.entries) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remove_pass", "missing staged_validator_pass"),
        ("optimizer_access", "custody attestation failed"),
    ],
)
def test_cache_attestation_rejects_missing_pass_or_optimizer_holdout_access(
    tmp_path: Path, monkeypatch, mutation: str, message: str
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    attestation = json.loads(fixture["cache_attestation_path"].read_text())
    if mutation == "remove_pass":
        attestation.pop("staged_validator_pass_binding")
    else:
        attestation["optimizer_holdout_access_count"] = 1
    data = _write_json(fixture["cache_attestation_path"], attestation)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=sha256_bytes(data),
    )
    with pytest.raises(ValueError, match=message):
        load_train_cases(
            fixture["lock"],
            reader,
            label_attestation=fixture["attestation_path"],
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(row["purpose"] == "TRAIN_LABEL_PROJECTION" for row in reader.entries)


def test_training_loader_enforces_runtime_source_only_validator_before_labels(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    manifest = json.loads(fixture["cache_manifest_path"].read_text())
    first_case = manifest["records"][0]["case_id"]
    for record in manifest["records"]:
        if record["case_id"] != first_case:
            continue
        semantic_path = tmp_path / record["semantic_path"]
        semantic = json.loads(semantic_path.read_text())
        semantic["labels_used"] = True
        semantic_bytes = _write_json(semantic_path, semantic)
        record["semantic_sha256"] = sha256_bytes(semantic_bytes)
    _write_json(fixture["cache_manifest_path"], manifest)
    cache_attestation_sha = _refresh_cache_attestation(fixture)
    reader = build_guarded_reader(
        fixture["lock"],
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=cache_attestation_sha,
    )
    with pytest.raises(ValueError, match="forbidden inference input"):
        load_train_cases(
            fixture["lock"],
            reader,
            label_attestation=fixture["attestation_path"],
            cache_attestation=fixture["cache_attestation_path"],
        )
    assert not any(row["purpose"] == "TRAIN_LABEL_PROJECTION" for row in reader.entries)


def test_synthetic_end_to_end_fit_freezes_one_pre_holdout_model(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _build_synthetic_train_projection_and_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(optimizer, "load_lock_context", lambda: fixture["lock"])
    monkeypatch.setattr(optimizer, "LEARNER_SPECS", (optimizer.LEARNER_SPECS[0],))
    monkeypatch.setattr(optimizer, "PROBABILITY_THRESHOLDS", (0.35,))
    monkeypatch.setattr(optimizer, "OUTPUT_CAPS", (2,))
    monkeypatch.setattr(optimizer, "OUTER_FOLDS", 3)
    monkeypatch.setattr(optimizer, "INNER_FOLDS", 2)
    monkeypatch.setattr(optimizer, "FINAL_SELECTION_FOLDS", 3)
    output = tmp_path / "frozen-output"
    result = run_optimizer(
        label_attestation=fixture["attestation_path"],
        label_attestation_sha256=fixture["attestation_sha"],
        cache_attestation=fixture["cache_attestation_path"],
        cache_attestation_sha256=fixture["cache_attestation_sha"],
        output_dir=output,
    )
    assert result["holdout_score_opened"] is False
    assert result["external_anchor_required_before_holdout_score"] is True
    model_path = Path(result["frozen_model"])
    model = json.loads(model_path.read_text())
    validate_model(model)
    assert sha256_bytes(model_path.read_bytes()) == result["frozen_model_sha256"]
    assert (output / "fit_audit_core.json").is_file()
    assert sha256_bytes((output / "fit_audit_core.json").read_bytes()) == model["fit_audit_sha256"]
    assert (output / "train_read_ledger.json").is_file()
    assert (output / "freeze_lock.json").is_file()
    assert (output / "pre_holdout_seal.json").is_file()
    assert (output / "pre_holdout_seal.sha256").is_file()


def _empty_case(index: int) -> CaseData:
    token = f"{index:064x}"[-64:]
    return CaseData(
        case_id=f"A30-{index:03d}",
        runtime_case_id=f"T{index:03d}",
        scan_sha256=token,
        bends=[],
        primary_v51={},
        primary_pool={},
        replay_v51={},
        replay_pool={},
    )


def test_case_folds_hold_out_complete_cases_deterministically() -> None:
    cases = [_empty_case(index) for index in range(1, 19)]
    first = case_folds(cases, 6, seed="synthetic")
    second = case_folds(list(reversed(cases)), 6, seed="synthetic")
    assert [[row.case_id for row in fold] for fold in first] == [
        [row.case_id for row in fold] for fold in second
    ]
    flattened = [row.case_id for fold in first for row in fold]
    assert len(flattened) == len(set(flattened)) == 18
    assert all(len(fold) == 3 for fold in first)


def _synthetic_learning_case(index: int) -> CaseData:
    true = _candidate(f"TRUE-{index}", y=float(index), score=0.95)
    false = _candidate(f"FALSE-{index}", y=float(index + 20), score=0.05)
    pool = _pool(true, false)
    pool["case_id"] = f"R{index:03d}"
    pool["inference_contract"] = {
        "part_identifier_exposed": False,
        "expected_count_exposed": False,
        "manual_truth_exposed": False,
        "cad_exposed": False,
    }
    prediction = {
        "bend_id": f"P-{index}",
        "line_start": true["span_endpoints"][0],
        "line_end": true["span_endpoints"][1],
        "selector_score": 0.95,
        "source_hypothesis_id": true["candidate_id"],
        "source_kind": "plane_lattice",
    }
    semantic = _v51(prediction)
    return CaseData(
        case_id=f"A30-{index:03d}",
        runtime_case_id=f"R{index:03d}",
        scan_sha256=f"{index:064x}"[-64:],
        bends=[
            {
                "bend_id": f"G-{index}",
                "line_start": true["span_endpoints"][0],
                "line_end": true["span_endpoints"][1],
                "reviewer_support_count": index % 3 + 1,
            }
        ],
        primary_v51=deepcopy(semantic),
        primary_pool=deepcopy(pool),
        replay_v51=deepcopy(semantic),
        replay_pool=deepcopy(pool),
    )


def test_every_preregistered_representation_policy_classifier_fits_portably() -> None:
    cases = [_synthetic_learning_case(index) for index in range(1, 7)]
    for spec in LEARNER_SPECS:
        classifier = fit_portable_classifier(spec, cases[:4])
        first = predict_with_classifier(
            cases[4], spec, classifier, threshold=0.35, cap=2
        )
        replay = predict_with_classifier(
            cases[4], spec, classifier, threshold=0.35, cap=2
        )
        assert canonical_bytes(first) == canonical_bytes(replay)


def _metrics(*, f1: float, tp: int, fp: int, fn: int, exact: int, tier_tp=(1, 1, 1)) -> dict:
    return {
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_case_count": exact,
        "label_support_tiers": {
            f"REVIEWER_SUPPORT_{index}": {"matched": tier_tp[index - 1]}
            for index in (1, 2, 3)
        },
    }


def test_training_gate_is_conjunctive_including_every_label_support_tier() -> None:
    baseline = _metrics(f1=0.20, tp=10, fp=20, fn=70, exact=1, tier_tp=(2, 3, 5))
    candidate = _metrics(f1=0.31, tp=15, fp=22, fn=45, exact=1, tier_tp=(2, 3, 10))
    assert training_gate(candidate, baseline)["all_conjunctive_checks_passed"] is True
    harmed = deepcopy(candidate)
    harmed["label_support_tiers"]["REVIEWER_SUPPORT_2"]["matched"] = 2
    result = training_gate(harmed, baseline)
    assert result["all_conjunctive_checks_passed"] is False
    assert result["label_support_tier_checks"]["REVIEWER_SUPPORT_2"] is False
