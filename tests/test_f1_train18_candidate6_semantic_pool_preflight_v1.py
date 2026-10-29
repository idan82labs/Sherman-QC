from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as runtime
from scripts import preflight_f1_train18_candidate6_semantic_pool_v1 as preflight


ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "preflight": preflight.sha256_bytes(preflight.PREFLIGHT_SOURCE.read_bytes()),
        "preflight_tests": preflight.sha256_bytes(
            preflight.PREFLIGHT_TESTS_SOURCE.read_bytes()
        ),
        "runtime": preflight.sha256_bytes(preflight.RUNTIME_SOURCE.read_bytes()),
        "runtime_tests": preflight.sha256_bytes(
            preflight.RUNTIME_TESTS_SOURCE.read_bytes()
        ),
    }


def _semantic() -> bytes:
    return runtime.canonical_bytes(
        {
            "cad_used": False,
            "candidate_count": 1,
            "expected_count_used": False,
            "labels_used": False,
            "part_identity_used": False,
            "predictions": [
                {
                    "bend_id": "V51-B0",
                    "line_end": [100.0, 0.0, 0.0],
                    "line_start": [0.0, 0.0, 0.0],
                    "selection_route": "v51-synthetic",
                    "selector_score": None,
                    "source_hypothesis_id": "V51-H0",
                    "source_kind": "plane_lattice",
                }
            ],
            "research_only": True,
            "source_only": True,
        }
    )


def _pool() -> bytes:
    return runtime.canonical_bytes(
        {
            "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
            "candidate_count": 50,
            "candidate_source_counts": {"ignored": 50},
            "candidates": [{"candidate_id": "RAW-IGNORED"}],
            "inference_contract": {
                "cad_exposed": False,
                "expected_count_exposed": False,
                "manual_truth_exposed": False,
                "part_identifier_exposed": False,
            },
            "line_object_count": 1,
            "line_objects": [
                {
                    "candidate_id": "IGNORED-ID",
                    "endpoint_support_fraction": 0.8,
                    "finite_support_fraction": 0.8,
                    "normal_second_eigenvalue": 0.2,
                    "representative_source": "plane_lattice",
                    "score": 0.8,
                    "span_endpoints": [
                        [0.0, 8.0, 0.0],
                        [100.0, 8.0, 0.0],
                    ],
                    "two_sided_transition_score": 0.8,
                }
            ],
            "schema_version": 3,
            "source_point_serialization": False,
            "working_point_budget": 200000,
        }
    )


def _manifest_fixture(
    *, train_ids: tuple[str, ...], scan_by_case: dict[str, str]
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for ordinal, case_id in enumerate(train_ids):
        runtime_id = f"T{900 + ordinal:03d}"
        semantic_sha = _sha(f"semantic-{case_id}")
        pool_sha = _sha(f"pool-{case_id}")
        for pass_name in ("primary", "replay"):
            records.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "pool_path": str(
                        (
                            preflight.ALLOWED_CACHE_ROOT
                            / pass_name
                            / runtime_id
                            / "pool"
                            / f"{runtime_id}.json"
                        ).relative_to(preflight.ROOT)
                    ),
                    "pool_sha256": pool_sha,
                    "runtime_case_id": runtime_id,
                    "scan_sha256": scan_by_case[case_id],
                    "semantic_path": str(
                        (
                            preflight.ALLOWED_CACHE_ROOT
                            / pass_name
                            / runtime_id
                            / "semantic_result.json"
                        ).relative_to(preflight.ROOT)
                    ),
                    "semantic_sha256": semantic_sha,
                }
            )
    commitment = preflight.sha256_bytes(
        preflight.canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    return {
        "case_count": 18,
        "holdout_entry_count": 0,
        "kind": preflight.CACHE_MANIFEST_KIND,
        "record_count": 36,
        "records": records,
        "schema_version": 1,
        "source_only": True,
        "split_sha256": preflight.FIXED_CONTROLS["SEALED_SPLIT"][1],
        "train_allowlist_sha256": preflight.FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "train_set_commitment_sha256": commitment,
        "v51_scan_only_manifest_sha256": preflight.V51_SCAN_ONLY_MANIFEST_SHA256,
    }


def _entry(path: Path, purpose: str, *, case_id: str | None = None) -> dict[str, object]:
    data = path.read_bytes()
    metadata = os.lstat(path)
    return {
        "case_id": case_id,
        "device_id": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "path": str(path.relative_to(preflight.ROOT)),
        "purpose": purpose,
        "sha256": preflight.sha256_bytes(data),
        "size_bytes": len(data),
    }


def test_candidate6_preflight_top_level_is_stdlib_only() -> None:
    tree = ast.parse(preflight.PREFLIGHT_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "numpy", "scipy", "scripts", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_help_does_not_import_runtime_evaluator_or_numeric_packages() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(preflight.PREFLIGHT_SOURCE), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "f1_train18_candidate6_conservative_recall_recovery_v1",
        "evaluate_f1_against_manual_gold",
        "numpy",
        "scipy",
        "sklearn",
    ):
        assert forbidden not in completed.stderr


def test_source_check_only_verifies_four_exact_sources_and_no_artifacts() -> None:
    hashes = _source_hashes()
    completed = subprocess.run(
        [
            sys.executable,
            str(preflight.PREFLIGHT_SOURCE),
            "--runtime-sha256",
            hashes["runtime"],
            "--preflight-sha256",
            hashes["preflight"],
            "--runtime-tests-sha256",
            hashes["runtime_tests"],
            "--preflight-tests-sha256",
            hashes["preflight_tests"],
            "--source-check-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "artifact_input_opened": False,
        "label_or_holdout_opened": False,
        "source_binding_count": 4,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def test_tampered_source_binding_fails_before_runtime_execution_or_artifacts(
    tmp_path: Path,
) -> None:
    hashes = _source_hashes()
    sentinel = tmp_path / "runtime-executed"
    code = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_c6_preflight_tamper', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_c6_preflight_tamper'] = module
spec.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_loader(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('EXECUTED')
    raise AssertionError('runtime executed after source mismatch')
module._load_verified_runtime = forbidden_loader
try:
    module.bootstrap_verified_sources(
        runtime_sha256={'0' * 63 + '1'!r},
        preflight_sha256={hashes['preflight']!r},
        runtime_tests_sha256={hashes['runtime_tests']!r},
        preflight_tests_sha256={hashes['preflight_tests']!r},
    )
except ValueError as error:
    assert 'binding mismatch' in str(error)
else:
    raise AssertionError('tampered source binding was accepted')
assert not pathlib.Path({str(sentinel)!r}).exists()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_preloaded_symlink_alias_rejects_before_any_controlled_source_read(
    tmp_path: Path,
) -> None:
    hashes = _source_hashes()
    alias = tmp_path / "innocent_name.py"
    alias.symlink_to(preflight.RUNTIME_SOURCE)
    sentinel = tmp_path / "read-sentinel"
    code = f"""
import importlib.util, pathlib, sys
alias = pathlib.Path({str(alias)!r})
spec = importlib.util.spec_from_file_location('arbitrary_alias', alias)
loaded = importlib.util.module_from_spec(spec)
sys.modules['arbitrary_alias'] = loaded
spec.loader.exec_module(loaded)
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec2 = importlib.util.spec_from_file_location('isolated_c6_preflight', path)
module = importlib.util.module_from_spec(spec2)
sys.modules['isolated_c6_preflight'] = module
spec2.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_read(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('READ')
    raise AssertionError('source read preceded alias rejection')
module._read_exact_source = forbidden_read
try:
    module.bootstrap_verified_sources(
        runtime_sha256={hashes['runtime']!r},
        preflight_sha256={hashes['preflight']!r},
        runtime_tests_sha256={hashes['runtime_tests']!r},
        preflight_tests_sha256={hashes['preflight_tests']!r},
    )
except RuntimeError as error:
    assert 'preloaded source alias' in str(error)
else:
    raise AssertionError('symlink-loaded alias was accepted')
assert not pathlib.Path({str(sentinel)!r}).exists()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_verified_loader_executes_the_exact_already_read_runtime_bytes() -> None:
    lineage = {
        f"{name}_terminal": {"sha256": sha}
        for name, sha in preflight.PREDECESSOR_TERMINAL_SHA256_NOT_OPENED.items()
    }
    source = (
        "ATTEMPT_ORDINAL = 6\n"
        f"PREDECESSOR_LINEAGE = {lineage!r}\n"
        "def fixed_rule_space():\n"
        "    return tuple({} for _ in range(32))\n"
    ).encode("utf-8")
    module = preflight._load_verified_runtime(Path("synthetic_runtime.py"), source)
    assert module.ATTEMPT_ORDINAL == 6
    assert len(module.fixed_rule_space()) == 32
    sys.modules.pop("_candidate6_verified_runtime", None)


def test_guarded_reader_denies_unbound_label_full28_holdout_and_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    reader = preflight.GuardedReader()
    ordinary = root / "ordinary.json"
    ordinary.write_bytes(b"{}\n")
    with pytest.raises(PermissionError, match="NOT_EXACTLY_AUTHORIZED"):
        reader.read_bytes(ordinary, purpose="ORDINARY")
    for name, purpose in (
        ("train_label_projection.json", "INNOCENT"),
        ("artifact.json", "TRAIN_LABEL_DATA"),
        ("labels-shadow.json", "CACHE"),
        ("full28-cache.json", "CACHE"),
        ("holdout-shadow.json", "CACHE"),
    ):
        path = root / name
        if not path.exists():
            path.write_bytes(b"{}\n")
        with pytest.raises(PermissionError):
            reader.authorize(
                path,
                purpose=purpose,
                sha256=preflight.sha256_bytes(path.read_bytes()),
                size_bytes=path.stat().st_size,
            )
    target = root / "target.json"
    target.write_bytes(b"{}\n")
    alias = root / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        reader.authorize(
            alias,
            purpose="CACHE",
            sha256=preflight.sha256_bytes(target.read_bytes()),
            size_bytes=target.stat().st_size,
        )


def test_guarded_reader_rejects_leaf_and_ancestor_swaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    parent = root / "bound"
    parent.mkdir(parents=True)
    monkeypatch.setattr(preflight, "ROOT", root)
    value = parent / "value.json"
    expected = b"expected\n"
    value.write_bytes(expected)
    reader = preflight.GuardedReader()
    reader.authorize(
        value,
        purpose="VALUE",
        sha256=preflight.sha256_bytes(expected),
        size_bytes=len(expected),
    )
    replacement = parent / "replacement.json"
    replacement.write_bytes(b"replacement\n")
    original_open = preflight.os.open
    swapped = False

    def swap_leaf(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if not swapped and Path(path).name == value.name and kwargs.get("dir_fd") is not None:
            swapped = True
            value.unlink()
            replacement.rename(value)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(preflight.os, "open", swap_leaf)
    with pytest.raises(ValueError, match="identity changed"):
        reader.read_bytes(value, purpose="VALUE")

    monkeypatch.setattr(preflight.os, "open", original_open)
    if value.exists():
        value.unlink()
    value.write_bytes(expected)
    reader = preflight.GuardedReader()
    reader.authorize(
        value,
        purpose="VALUE",
        sha256=preflight.sha256_bytes(expected),
        size_bytes=len(expected),
    )
    foreign = root / "foreign"
    foreign.mkdir()
    (foreign / value.name).write_bytes(b"foreign\n")
    moved = root / "moved"
    swapped = False

    def swap_parent(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if not swapped and Path(path) == parent:
            swapped = True
            parent.rename(moved)
            parent.symlink_to(foreign, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(preflight.os, "open", swap_parent)
    with pytest.raises(OSError):
        reader.read_bytes(value, purpose="VALUE")


def test_guarded_reader_rejects_hardlink_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    first = root / "first.json"
    second = root / "second.json"
    first.write_bytes(b"{}\n")
    os.link(first, second)
    reader = preflight.GuardedReader()
    with pytest.raises(ValueError, match="non-hard-linked"):
        reader.authorize(
            first,
            purpose="FIRST",
            sha256=preflight.sha256_bytes(first.read_bytes()),
            size_bytes=first.stat().st_size,
        )


def test_manifest_accepts_exact_nonordinal_bijective_18_by_2_inventory() -> None:
    train_ids = tuple(f"TRAIN-X{index:02d}" for index in range(18))
    scan_by_case = {case_id: _sha(f"scan-{case_id}") for case_id in train_ids}
    manifest = _manifest_fixture(train_ids=train_ids, scan_by_case=scan_by_case)
    mapping = preflight._validate_cache_manifest(
        manifest,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=str(manifest["train_set_commitment_sha256"]),
    )
    assert len(mapping) == 36
    assert mapping[(train_ids[0], "primary")]["runtime_case_id"] == "T900"


def test_cache_attestation_header_is_exact_and_zero_holdout() -> None:
    commitment = _sha("train-commitment")
    binding = {
        "kind": "synthetic-binding",
        "path": "synthetic.json",
        "sha256": _sha("binding"),
        "size_bytes": 1,
    }
    attestation = {
        "access_evidence_binding": deepcopy(binding),
        "blinded_complement_commitment_sha256": _sha("complement"),
        "builder_binding": deepcopy(binding),
        "canonical_json": True,
        "case_count": 18,
        "holdout_entry_count": 0,
        "holdout_hash_count": 0,
        "holdout_status_count": 0,
        "independent_validation_passed": True,
        "kind": preflight.CACHE_ATTESTATION_KIND,
        "manifest_binding": deepcopy(binding),
        "optimizer_holdout_access_count": 0,
        "record_count": 36,
        "schema_version": 1,
        "split_sha256": preflight.FIXED_CONTROLS["SEALED_SPLIT"][1],
        "staged_validator_pass_binding": deepcopy(binding),
        "train_allowlist_sha256": preflight.FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "train_exact_match": True,
        "train_holdout_intersection_count": 0,
        "train_set_commitment_sha256": commitment,
        "validator_binding": deepcopy(binding),
    }
    checks = preflight._validate_cache_attestation_header(
        attestation, train_commitment=commitment
    )
    assert checks and all(checks.values())
    for mutation in ("extra", "holdout", "commitment"):
        changed = deepcopy(attestation)
        if mutation == "extra":
            changed["unexpected"] = False
        elif mutation == "holdout":
            changed["holdout_hash_count"] = 1
        else:
            changed["train_set_commitment_sha256"] = _sha("wrong")
        with pytest.raises(ValueError):
            preflight._validate_cache_attestation_header(
                changed, train_commitment=commitment
            )


@pytest.mark.parametrize(
    "mutation",
    (
        "nineteenth",
        "wrong_order",
        "wrong_scan",
        "extra_key",
        "duplicate_path",
        "nonbijective_runtime_id",
        "wrong_root",
        "replay_hash_mismatch",
    ),
)
def test_manifest_rejects_any_inventory_or_binding_expansion(mutation: str) -> None:
    train_ids = tuple(f"TRAIN-X{index:02d}" for index in range(18))
    scan_by_case = {case_id: _sha(f"scan-{case_id}") for case_id in train_ids}
    manifest = _manifest_fixture(train_ids=train_ids, scan_by_case=scan_by_case)
    records = manifest["records"]
    assert isinstance(records, list)
    if mutation == "nineteenth":
        records.append(deepcopy(records[-1]))
        manifest["record_count"] = 37
    elif mutation == "wrong_order":
        records[0], records[1] = records[1], records[0]
    elif mutation == "wrong_scan":
        records[0]["scan_sha256"] = _sha("wrong")
    elif mutation == "extra_key":
        manifest["unexpected"] = False
    elif mutation == "duplicate_path":
        records[1]["semantic_path"] = records[0]["semantic_path"]
    elif mutation == "nonbijective_runtime_id":
        records[2]["runtime_case_id"] = records[0]["runtime_case_id"]
        records[3]["runtime_case_id"] = records[0]["runtime_case_id"]
    elif mutation == "wrong_root":
        records[0]["pool_path"] = "output/research/elsewhere/pool.json"
    else:
        records[1]["pool_sha256"] = _sha("changed-replay")
    with pytest.raises((PermissionError, ValueError)):
        preflight._validate_cache_manifest(
            manifest,
            train_ids=train_ids,
            scan_by_case=scan_by_case,
            train_commitment=str(
                _manifest_fixture(
                    train_ids=train_ids, scan_by_case=scan_by_case
                )["train_set_commitment_sha256"]
            ),
        )


def test_runtime_preflight_is_aggregate_source_only_and_honestly_named() -> None:
    semantic = _semantic()
    pool = _pool()
    value = runtime.preflight_candidate6_source_bytes(
        semantic,
        pool,
        v51_semantic_sha256=runtime.sha256_bytes(semantic),
        pool_sha256=runtime.sha256_bytes(pool),
    )
    preflight._assert_runtime_preflight_safe(value)
    assert value["runtime_threshold_match_implies_conservative_conflict_for_bound_pairs"]
    assert "evaluator_match_implies_ownership_conflict_for_bound_pairs" not in value
    assert value["fixed_allowed_source_kind_eligibility_exercised"] is True
    assert value["deterministic_geometry_source_tie_break_exercised"] is True
    assert not preflight.FORBIDDEN_AGGREGATE_KEYS.intersection(value)


def _synthetic_context_and_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    replay_semantic_mismatch: bool = False,
    invalid_first_pool: bool = False,
) -> tuple[preflight.BootstrapContext, tuple[str, ...]]:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "ALLOWED_CACHE_ROOT", root / "cache")
    source_specs = (
        ("preflight", "CANDIDATE6_SOURCE_PREFLIGHT"),
        ("preflight_tests", "CANDIDATE6_SOURCE_PREFLIGHT_TESTS"),
        ("runtime", "CANDIDATE6_SOURCE_RUNTIME"),
        ("runtime_tests", "CANDIDATE6_SOURCE_RUNTIME_TESTS"),
    )
    bootstrap_entries: list[dict[str, object]] = []
    source_bindings: dict[str, dict[str, object]] = {}
    for name, purpose in source_specs:
        path = root / f"{name}.py"
        path.write_bytes(f"# {name}\n".encode("ascii"))
        bootstrap_entries.append(_entry(path, purpose))
        source_bindings[name] = {
            "path": path.name,
            "sha256": preflight.sha256_bytes(path.read_bytes()),
            "size_bytes": path.stat().st_size,
        }
    context = preflight.BootstrapContext(
        runtime=runtime,
        source_bindings=source_bindings,
        source_bytes={name: b"" for name, _purpose in source_specs},
        bootstrap_entries=tuple(bootstrap_entries),
    )
    train_ids = tuple(f"TRAIN-X{index:02d}" for index in range(18))
    semantic = _semantic()
    pool = _pool()
    records: dict[tuple[str, str], dict[str, object]] = {}
    for ordinal, case_id in enumerate(train_ids):
        runtime_id = f"T{900 + ordinal:03d}"
        for pass_name in ("primary", "replay"):
            directory = root / "cache" / pass_name / runtime_id
            pool_directory = directory / "pool"
            pool_directory.mkdir(parents=True)
            semantic_path = directory / "semantic_result.json"
            pool_path = pool_directory / f"{runtime_id}.json"
            semantic_value = semantic
            if replay_semantic_mismatch and ordinal == 0 and pass_name == "replay":
                changed = json.loads(semantic)
                changed["candidate_count"] = 999
                semantic_value = runtime.canonical_bytes(changed)
            pool_value = pool
            if invalid_first_pool and ordinal == 0 and pass_name == "primary":
                changed_pool = json.loads(pool)
                changed_pool["line_objects"][0]["ground_truth"] = True
                pool_value = runtime.canonical_bytes(changed_pool)
            semantic_path.write_bytes(semantic_value)
            pool_path.write_bytes(pool_value)
            records[(case_id, pass_name)] = {
                "case_id": case_id,
                "pass": pass_name,
                "pool_path": pool_path,
                "pool_sha256": runtime.sha256_bytes(pool_value),
                "runtime_case_id": runtime_id,
                "semantic_path": semantic_path,
                "semantic_sha256": runtime.sha256_bytes(semantic_value),
            }

    def fake_validate_controls(reader: preflight.GuardedReader):  # type: ignore[no-untyped-def]
        purposes = (
            "SEALED_SPLIT",
            "ORIGINAL_PREREGISTRATION",
            "TRAIN_ALLOWLIST",
            "CANDIDATE5_TERMINAL_DISPOSITION",
            "TRAIN_CACHE_ATTESTATION",
            "CACHE_CUSTODY_ACCESS_EVIDENCE",
            "CACHE_CUSTODY_BUILDER",
            "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
            "CACHE_CUSTODY_VALIDATOR",
            "TRAIN_CACHE_MANIFEST",
        )
        for index, purpose in enumerate(purposes):
            path = root / f"control-{index:02d}.json"
            data = preflight.canonical_bytes({"kind": f"synthetic-{index}"})
            path.write_bytes(data)
            reader.authorize(
                path,
                purpose=purpose,
                sha256=preflight.sha256_bytes(data),
                size_bytes=len(data),
            )
            reader.read_bytes(path, purpose=purpose)
        scan_by_case = {case_id: _sha(f"scan-{case_id}") for case_id in train_ids}
        return train_ids, scan_by_case, _sha("commitment"), records, {"synthetic": True}

    monkeypatch.setattr(preflight, "_validate_controls", fake_validate_controls)
    return context, train_ids


def test_build_pass_payload_has_exact_86_read_inventory_and_aggregate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, train_ids = _synthetic_context_and_controls(tmp_path, monkeypatch)
    payloads = preflight.build_preflight_payloads(context)
    assert set(payloads) == preflight.PASS_FILENAMES
    report = json.loads(payloads["preflight_report.json"])
    lock = json.loads(payloads["preflight_lock.json"])
    ledger = json.loads(payloads["preflight_read_ledger.json"])
    assert report["status"] == "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
    assert report["case_count"] == 18 and report["record_count"] == 36
    for name, sha in preflight.PREDECESSOR_TERMINAL_SHA256_NOT_OPENED.items():
        assert report["bindings"][f"{name}_terminal_sha256_not_opened"] == sha
    assert ledger["assertions"]["exact_train_inventory_completed"] is True
    assert ledger["assertions"]["label_attestation_or_projection_opened"] is False
    assert len(ledger["reads"]) == 86
    assert lock["preflight_report_sha256"] == preflight.sha256_bytes(
        payloads["preflight_report.json"]
    )
    assert report["preflight_read_ledger_sha256"] == preflight.sha256_bytes(
        payloads["preflight_read_ledger.json"]
    )
    for value in (report, lock):
        preflight._assert_aggregate_safe(value)
    assert tuple(
        entry["case_id"]
        for entry in ledger["reads"]
        if str(entry["purpose"]).endswith("_V51_POOL")
    ) == tuple(case_id for case_id in train_ids for _pass in range(2))


@pytest.mark.parametrize("failure", ("replay_mismatch", "forbidden_pool_key"))
def test_any_post_control_compatibility_failure_publishes_terminal_and_ledger_only(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _train_ids = _synthetic_context_and_controls(
        tmp_path,
        monkeypatch,
        replay_semantic_mismatch=failure == "replay_mismatch",
        invalid_first_pool=failure == "forbidden_pool_key",
    )
    payloads = preflight.build_preflight_payloads(context)
    assert set(payloads) == preflight.FAILURE_FILENAMES
    terminal = json.loads(payloads["preflight_terminal.json"])
    ledger = json.loads(payloads["preflight_read_ledger.json"])
    assert terminal["status"] == "FAILED_ENGINEERING_SOURCE_COMPATIBILITY"
    assert terminal["continuation_authorized"] is False
    assert terminal["model_fit_performed"] is False
    assert terminal["label_attestation_or_projection_opened"] is False
    assert ledger["status"] == "TERMINAL_FAILURE"
    assert ledger["assertions"]["exact_train_inventory_completed"] is False
    assert "preflight_report.json" not in payloads
    assert "preflight_lock.json" not in payloads


def test_success_inventory_validator_rejects_missing_duplicate_or_denied_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, train_ids = _synthetic_context_and_controls(tmp_path, monkeypatch)
    reader = preflight.GuardedReader(entries=[dict(row) for row in context.bootstrap_entries])
    preflight._validate_controls(reader)
    records = preflight._validate_controls(reader)[3]
    preflight._read_train_sources(context, reader, train_ids=train_ids, records=records)
    with pytest.raises((PermissionError, ValueError), match="not exact|repeated input"):
        preflight._validate_complete_success_read_inventory(reader, train_ids=train_ids)


def _pass_payloads() -> dict[str, bytes]:
    return {
        "preflight_lock.json": preflight.canonical_bytes({"kind": "lock"}),
        "preflight_read_ledger.json": preflight.canonical_bytes({"kind": "ledger"}),
        "preflight_report.json": preflight.canonical_bytes({"kind": "report"}),
    }


def _publication_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    output = parent / "candidate6-preflight"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", output)
    return parent, output


def test_atomic_publication_success_is_exact_read_only_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    payloads = _pass_payloads()
    preflight.publish_atomically(output, payloads)
    assert {path.name for path in output.iterdir()} == set(payloads)
    for name, data in payloads.items():
        path = output / name
        assert path.read_bytes() == data
        assert path.stat().st_mode & 0o777 == 0o400
    assert output.stat().st_mode & 0o777 == 0o500
    assert not tuple(parent.glob(f".{output.name}.staging-*"))
    assert not tuple(parent.glob(f".{output.name}.quarantine-*"))


def test_publication_never_clobbers_prior_or_raced_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    output.mkdir()
    marker = output / "foreign.txt"
    marker.write_text("foreign", encoding="utf-8")
    with pytest.raises(FileExistsError):
        preflight.publish_atomically(output, _pass_payloads())
    assert marker.read_text(encoding="utf-8") == "foreign"

    output.rename(parent / "prior-output")

    def race(_staging: Path) -> None:
        output.mkdir()
        (output / "race.txt").write_text("race", encoding="utf-8")

    with pytest.raises(FileExistsError):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_stage_write=race,
        )
    assert (output / "race.txt").read_text(encoding="utf-8") == "race"
    assert not tuple(parent.glob(f".{output.name}.staging-*"))
    quarantines = tuple(parent.glob(f".{output.name}.quarantine-*"))
    assert len(quarantines) == 1
    assert set(path.name for path in quarantines[0].iterdir()) == set(_pass_payloads())


def test_postrename_replacement_is_quarantined_and_canonical_output_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    moved = parent / "moved-owned-output"

    def replace_after_rename(_output: Path) -> None:
        output.rename(moved)
        output.mkdir()
        (output / "foreign.txt").write_text("foreign", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed"):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_rename=replace_after_rename,
        )
    assert output.exists()
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert moved.exists()
    quarantines = tuple(parent.glob(f".{output.name}.quarantine-*"))
    assert not quarantines


def test_output_parent_symlink_swap_is_detected_without_foreign_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    root = preflight.ROOT
    foreign = root / "foreign-parent"
    foreign.mkdir()
    moved_parent = root / "moved-parent"

    def swap_parent(_staging: Path) -> None:
        parent.rename(moved_parent)
        parent.symlink_to(foreign, target_is_directory=True)

    with pytest.raises((RuntimeError, ValueError), match="parent|symlink"):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_stage_write=swap_parent,
        )
    assert list(foreign.iterdir()) == []


def test_staging_inode_swap_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    moved: list[Path] = []

    def swap_stage(staging: Path) -> None:
        detached = parent / "detached-owned"
        staging.rename(detached)
        staging.mkdir()
        (staging / "foreign.txt").write_text("foreign", encoding="utf-8")
        moved.append(detached)

    with pytest.raises(RuntimeError, match="identity changed"):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_stage_write=swap_stage,
        )
    assert not output.exists()
    assert moved and moved[0].exists()
    residue = tuple(parent.glob(f".{output.name}.staging-*"))
    assert len(residue) == 1
    assert (residue[0] / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not tuple(parent.glob(f".{output.name}.quarantine-*"))


@pytest.mark.parametrize(
    "payloads",
    (
        {
            "preflight_read_ledger.json": b"{}\n",
            "preflight_terminal.json": b"{}\n",
        },
        {
            "preflight_lock.json": b"{}\n",
            "preflight_read_ledger.json": b"{}\n",
            "preflight_report.json": b"{}\n",
        },
    ),
)
def test_both_pass_and_terminal_failure_inventories_publish_atomically(
    payloads: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, output = _publication_root(tmp_path, monkeypatch)
    preflight.publish_atomically(output, payloads)
    assert {path.name for path in output.iterdir()} == set(payloads)


def test_aggregate_artifacts_forbid_candidate_volume_and_feature_details() -> None:
    forbidden = (
        "candidate_count",
        "feature_row_count",
        "line_object_count",
        "missing_numeric_feature_counts",
        "prediction_count",
        "prediction_row_count",
        "predictions",
        "rule_path_check_count",
        "selected_candidate_geometry_digest",
        "selector_score",
        "source_kind",
    )
    for key in forbidden:
        with pytest.raises(ValueError, match="forbidden key"):
            preflight._assert_aggregate_safe({key: 0})


def test_run_returns_only_value_free_status_inventory_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    context = SimpleNamespace()
    monkeypatch.setattr(
        preflight,
        "build_preflight_payloads",
        lambda _context: _pass_payloads(),
    )
    value = preflight.run(context, output_root=output)
    assert set(value) == {
        "case_count",
        "label_attestation_or_projection_opened",
        "output_inventory",
        "published_artifact_sha256",
        "record_count",
        "status",
    }
    assert value["label_attestation_or_projection_opened"] is False
    assert value["status"] == "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
    assert output.exists()
    assert not tuple(parent.glob(f".{output.name}.staging-*"))
