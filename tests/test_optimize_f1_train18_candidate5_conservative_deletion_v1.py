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

from backend import f1_train18_candidate5_conservative_deletion_v1 as runtime
from scripts import evaluate_f1_against_manual_gold as real_evaluator
from scripts import optimize_f1_train18_candidate5_conservative_deletion_v1 as optimizer


ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Thresholds:
    def __init__(self, *values: float) -> None:
        self.values = values


def _match_lines(gold, predictions, _thresholds):  # type: ignore[no-untyped-def]
    gold_by_id = {str(row["bend_id"]): row for row in gold}
    prediction_by_id = {str(row["bend_id"]): row for row in predictions}
    matched = sorted(set(gold_by_id).intersection(prediction_by_id))
    return {
        "exact": len(matched) == len(gold) == len(predictions),
        "false_negative": len(gold) - len(matched),
        "false_positive": len(predictions) - len(matched),
        "matches": [
            {"gold_id": bend_id, "prediction_id": bend_id} for bend_id in matched
        ],
        "prediction_count": len(predictions),
        "true_positive": len(matched),
    }


EVALUATOR = SimpleNamespace(Thresholds=_Thresholds, match_lines=_match_lines)


def _prediction(
    bend_id: str,
    *,
    source_kind: str,
    selector_score: float | None | type(Ellipsis) = 0.5,
) -> dict[str, object]:
    row: dict[str, object] = {
        "bend_id": bend_id,
        "line_start": [0.0, 0.0, 0.0],
        "line_end": [10.0, 0.0, 0.0],
        "selection_route": "full_resolution",
        "source_hypothesis_id": f"H-{bend_id}",
        "source_kind": source_kind,
    }
    if selector_score is not ...:
        row["selector_score"] = selector_score
    return row


def _semantic(predictions: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "source_only": True,
            "research_only": True,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "expected_count_used": False,
            "candidate_count": len(predictions),
            "audit": {"source_partition_count": 1},
            "predictions": predictions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _case(index: int, *, include_fp: bool = True) -> optimizer.PreparedCase:
    case_id = f"C{index:02d}"
    tp = _prediction(case_id, source_kind="plane_lattice")
    predictions = [tp]
    if include_fp:
        predictions.append(
            _prediction(
                f"NOISE-{case_id}",
                source_kind="v41_fragment_tolerant_parent_bridge_sidecar",
            )
        )
    semantic = _semantic(predictions)
    digest = runtime.sha256_bytes(semantic)
    _source, sanitized = runtime._validate_semantic_bytes(
        semantic, expected_sha256=digest
    )
    gold = (
        {
            "bend_id": case_id,
            "line_start": [0.0, 0.0, 0.0],
            "line_end": [10.0, 0.0, 0.0],
            "reviewer_support_count": (index % 3) + 1,
        },
    )
    return optimizer.PreparedCase(
        case_id=case_id,
        runtime_case_id=f"T{index:03d}",
        scan_sha256=_sha(f"scan-{index}"),
        gold=gold,
        primary_semantic_bytes=semantic,
        primary_semantic_sha256=digest,
        primary_predictions=tuple(dict(row) for row in sanitized),
        replay_semantic_bytes=semantic,
        replay_semantic_sha256=digest,
        replay_predictions=tuple(dict(row) for row in sanitized),
    )


def _cases(*, include_fp: bool = True) -> list[optimizer.PreparedCase]:
    optimizer.configure_source_free_dependencies(runtime, EVALUATOR)
    return [_case(index, include_fp=include_fp) for index in range(18)]


def _binding(path: Path, label: str, size: int = 1) -> dict[str, object]:
    return {
        "path": str(path.relative_to(optimizer.ROOT)),
        "sha256": _sha(label),
        "size_bytes": size,
    }


def _prereg(source_bindings: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": optimizer.PREREG_KIND,
        "status": "SEALED_ENGINEERING_ONLY_FIT_UNAUTHORIZED",
        "candidate_id": 5,
        "attempt_ordinal": 5,
        "holdout_score_opened": False,
        "adaptive_search_disclosed": True,
        "prospective_or_unseen_claim_authorized": False,
        "lineage_bindings": optimizer.expected_lineage_bindings(),
        "preflight_bindings": optimizer.expected_preflight_bindings(),
        "custody_bindings": optimizer.expected_custody_bindings(),
        "source_bindings": source_bindings,
        "source_binding_commitment_sha256": optimizer.sha256_bytes(
            optimizer.canonical_bytes(source_bindings)
        ),
        "search_contract": optimizer.search_contract(),
        "nested_cv_contract": optimizer.nested_cv_contract(),
        "train_gate_contract": optimizer.train_gate_contract(),
        "holdout_contract": optimizer.holdout_contract(),
        "authorization_contract": {
            "fit_authorization_artifact_required": True,
            "fit_authorization_path": str(optimizer.FIT_AUTHORIZATION.relative_to(optimizer.ROOT)),
            "fit_execution_authorized_by_preregistration": False,
            "fresh_process_required": True,
            "single_authorized_fit_run_required": True,
        },
        "deployment_identity_contract": {
            "deployed_model_exact_bytes_match_evaluated_predictions_scope": "FINAL_RULE_SEPARATE_TRAIN_PRIMARY_REPLAY_EVALUATION_ONLY",
            "oof_predictions_are_not_final_model_predictions": True,
            "replay_must_complete_before_any_pass_artifact_publication": True,
        },
        "publication_contract": {
            "failed_gate_inventory": ["terminal_negative.json"],
            "no_clobber_atomic_directory_publication": True,
            "pass_inventory": sorted(optimizer.PASS_FILENAMES),
            "replay_failure_terminalizes": True,
        },
        "forbidden_access_contract": {
            "full_28_catalog_or_manifest": "FORBIDDEN",
            "holdout_paths_scores_status_hashes": "FORBIDDEN",
            "label_projection": "ONE_FD_GUARDED_READ_LAST_AFTER_SEPARATE_AUTHORIZATION",
            "pool_artifacts": "FORBIDDEN",
        },
        "sealed_preflight_test_temporal_note": {
            "after_one_shot_publication_expected_result": "110_PASS_1_EXPECTED_TEMPORAL_ABSENCE_ASSERTION_FAILURE",
            "bound_test_bytes_must_not_be_mutated": True,
            "prepublication_result": "111_PASS",
        },
    }


def test_optimizer_top_level_imports_are_stdlib_only() -> None:
    tree = ast.parse(optimizer.OPTIMIZER_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "numpy", "scipy", "scripts", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_fixed_search_space_is_finite_deterministic_and_source_only() -> None:
    first = optimizer.fixed_rule_space()
    second = optimizer.fixed_rule_space()
    assert first == second
    assert len(first) == 72
    assert optimizer.search_contract()["rule_count"] == 72
    assert optimizer.search_contract()["randomness_used"] is False
    assert {rule["feature"] for rule in first} == {
        "line_length_mm",
        "selector_score",
        "selection_route",
        "source_kind",
    }
    encoded = optimizer.canonical_bytes(list(first)).decode("utf-8")
    for forbidden in ("case_id", "filename", "expected_count", "candidate_count", "audit"):
        assert forbidden not in encoded


def test_prereg_validator_binds_lineage_preflight_custody_and_source_commitment() -> None:
    sources = {
        "evaluator": _binding(optimizer.EVALUATOR_SOURCE, "evaluator"),
        "optimizer": _binding(optimizer.OPTIMIZER_SOURCE, "optimizer"),
        "optimizer_tests": _binding(optimizer.OPTIMIZER_TESTS_SOURCE, "optimizer-tests"),
        "publication_helper": _binding(optimizer.PREFLIGHT_SOURCE, "helper"),
        "runtime": _binding(optimizer.RUNTIME_SOURCE, "runtime"),
        "runtime_tests": _binding(optimizer.RUNTIME_TESTS_SOURCE, "runtime-tests"),
        "semantic_cache_preflight_tests": _binding(optimizer.PREFLIGHT_TESTS_SOURCE, "preflight-tests"),
    }
    prereg = _prereg(sources)
    optimizer._validate_preregistration(prereg, prereg_sha256=_sha("prereg"))
    for field in ("lineage_bindings", "preflight_bindings", "custody_bindings"):
        tampered = deepcopy(prereg)
        tampered[field] = {}
        with pytest.raises(ValueError, match="binding"):
            optimizer._validate_preregistration(tampered, prereg_sha256=_sha("prereg"))
    tampered = deepcopy(prereg)
    tampered["source_bindings"]["runtime"]["sha256"] = _sha("tampered")
    with pytest.raises(ValueError, match="commitment"):
        optimizer._validate_preregistration(tampered, prereg_sha256=_sha("prereg"))


def test_preloaded_symlink_runtime_rejects_before_prereg_read(tmp_path) -> None:
    alias = tmp_path / "innocent.py"
    alias.symlink_to(optimizer.RUNTIME_SOURCE)
    sentinel = tmp_path / "prereg-read.txt"
    code = f"""
import importlib.util, pathlib, sys
runtime_spec = importlib.util.spec_from_file_location('arbitrary_alias', pathlib.Path({str(alias)!r}))
runtime_alias = importlib.util.module_from_spec(runtime_spec)
sys.modules['arbitrary_alias'] = runtime_alias
runtime_spec.loader.exec_module(runtime_alias)
path = pathlib.Path({str(optimizer.OPTIMIZER_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_candidate5_optimizer', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_candidate5_optimizer'] = module
spec.loader.exec_module(module)
def forbidden_read(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('READ')
    raise AssertionError('prereg read occurred')
module._read_exact_once = forbidden_read
try:
    module.bootstrap_verified_sources(preregistration_sha256='1' * 64)
except RuntimeError as error:
    assert 'preloaded source alias' in str(error)
else:
    raise AssertionError('preloaded runtime alias was accepted')
assert not pathlib.Path({str(sentinel)!r}).exists()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_guarded_projection_is_single_fd_read_last(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(optimizer, "ROOT", root)
    control = root / "control.json"
    control.write_bytes(b"{}\n")
    projection = root / "projection.json"
    projection.write_bytes(b'{"cases":[]}\n')
    reader = optimizer.GuardedReader()
    reader.authorize(
        control,
        purpose="CONTROL",
        sha256=optimizer.sha256_bytes(control.read_bytes()),
        size_bytes=control.stat().st_size,
    )
    assert reader.read_bytes(control, purpose="CONTROL") == b"{}\n"
    reader.authorize_projection_last(
        projection,
        sha256=optimizer.sha256_bytes(projection.read_bytes()),
        size_bytes=projection.stat().st_size,
        all_nonprojection_reads_complete=True,
    )
    assert reader.read_projection_once_last() == b'{"cases":[]}\n'
    with pytest.raises(PermissionError):
        reader.read_projection_once_last()
    with pytest.raises(PermissionError, match="ARTIFACT_READ_AFTER_PROJECTION"):
        reader.read_bytes(control, purpose="CONTROL")


@pytest.mark.parametrize("projection_read", [False, True])
def test_guarded_read_rejects_ancestor_symlink_swap_after_first_bytes(
    tmp_path, monkeypatch, projection_read: bool
) -> None:
    root = tmp_path / "root"
    input_parent = root / "inputs"
    input_parent.mkdir(parents=True)
    detached_parent = root / "inputs-detached"
    foreign_parent = root / "foreign"
    foreign_parent.mkdir()
    path = input_parent / "bound.json"
    original_bytes = b'{"bound":true}\n'
    path.write_bytes(original_bytes)
    monkeypatch.setattr(optimizer, "ROOT", root)
    reader = optimizer.GuardedReader()
    if projection_read:
        reader.authorize_projection_last(
            path,
            sha256=optimizer.sha256_bytes(original_bytes),
            size_bytes=len(original_bytes),
            all_nonprojection_reads_complete=True,
        )
    else:
        reader.authorize(
            path,
            purpose="CONTROL",
            sha256=optimizer.sha256_bytes(original_bytes),
            size_bytes=len(original_bytes),
        )
    original_read = optimizer.os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            input_parent.rename(detached_parent)
            input_parent.symlink_to(foreign_parent, target_is_directory=True)
            (foreign_parent / path.name).write_bytes(b'{"evil":true}\n')
        return chunk

    monkeypatch.setattr(optimizer.os, "read", swapping_read)
    with pytest.raises((RuntimeError, ValueError), match="symlink|identity changed"):
        if projection_read:
            reader.read_projection_once_last()
        else:
            reader.read_bytes(path, purpose="CONTROL")
    assert swapped is True
    assert (foreign_parent / path.name).read_bytes() == b'{"evil":true}\n'
    assert not reader.entries
    if projection_read:
        assert reader.projection_read_count == 1
        assert reader.projection_read_complete is True
        with pytest.raises(PermissionError):
            reader.read_projection_once_last()


@pytest.mark.parametrize("projection_bytes", [b"{not-json}\n", b"{}\n"])
def test_projection_decode_or_schema_failure_after_read_is_wrapped_for_terminalization(
    tmp_path, monkeypatch, projection_bytes: bytes
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(optimizer, "ROOT", root)
    projection = root / "projection.json"
    projection.write_bytes(projection_bytes)
    reader = optimizer.GuardedReader()
    reader.authorize_projection_last(
        projection,
        sha256=optimizer.sha256_bytes(projection_bytes),
        size_bytes=len(projection_bytes),
        all_nonprojection_reads_complete=True,
    )
    state = optimizer.PostProjectionState(
        reader=reader,
        input_bindings={
            "label_projection_sha256": optimizer.sha256_bytes(projection_bytes),
            "train_cache_manifest_sha256": _sha("manifest"),
            "train_source_cache_binding_sha256": _sha("cache-binding"),
        },
        candidate3_terminal_bytes=b"{}\n",
        candidate4_terminal_bytes=b"{}\n",
    )
    with pytest.raises(optimizer.PostProjectionFailure) as raised:
        optimizer._read_projection_and_prepare_cases(
            reader,
            state,
            train_ids=(),
            scan_by_case={},
            source_cases=(),
        )
    assert raised.value.state is state
    assert raised.value.reason_class in {"JSONDecodeError", "ValueError"}
    assert reader.projection_read_count == 1
    assert reader.projection_read_complete is True
    with pytest.raises(PermissionError):
        reader.read_projection_once_last()


def test_truncated_projection_fd_read_is_consumed_and_terminalized(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(optimizer, "ROOT", root)
    projection = root / "projection.json"
    attested_bytes = b"{}\n"
    projection.write_bytes(attested_bytes)
    reader = optimizer.GuardedReader()
    reader.authorize_projection_last(
        projection,
        sha256=optimizer.sha256_bytes(attested_bytes),
        size_bytes=len(attested_bytes),
        all_nonprojection_reads_complete=True,
    )
    projection.write_bytes(b"")
    state = optimizer.PostProjectionState(
        reader=reader,
        input_bindings={
            "label_projection_sha256": optimizer.sha256_bytes(attested_bytes),
            "train_cache_manifest_sha256": _sha("manifest"),
            "train_source_cache_binding_sha256": _sha("cache-binding"),
        },
        candidate3_terminal_bytes=b"{}\n",
        candidate4_terminal_bytes=b"{}\n",
    )
    with pytest.raises(optimizer.PostProjectionFailure) as raised:
        optimizer._read_projection_and_prepare_cases(
            reader,
            state,
            train_ids=(),
            scan_by_case={},
            source_cases=(),
        )
    assert raised.value.reason_class == "ValueError"
    assert reader.projection_read_count == 1
    assert reader.projection_read_complete is True
    with pytest.raises(PermissionError):
        reader.read_projection_once_last()


def test_projection_sanitization_drops_conflicting_evaluator_geometry() -> None:
    train_ids = tuple(f"C{index:02d}" for index in range(18))
    scans = {case_id: _sha(f"scan-{case_id}") for case_id in train_ids}
    projection = {
        "case_count": 18,
        "cases": [
            {
                "case_id": case_id,
                "scan_sha256": scans[case_id],
                "bends": [
                    {
                        "bend_id": case_id,
                        "line_start": [0.0, 0.0, 0.0],
                        "line_end": [10.0, 0.0, 0.0],
                        "reviewer_support_count": 1,
                        "span_endpoints": [[0.0, 100.0, 0.0], [10.0, 100.0, 0.0]],
                        "arbitrary_label_metadata": "IGNORED",
                    }
                ],
            }
            for case_id in train_ids
        ],
        "kind": optimizer.PROJECTION_KIND,
        "schema_version": 1,
        "split_sha256": optimizer.FIXED_BINDINGS["sealed_split"][1],
        "train_allowlist_sha256": optimizer.FIXED_BINDINGS["train_allowlist"][1],
    }
    gold = optimizer._validate_projection(
        projection, train_ids=train_ids, scan_by_case=scans
    )
    assert set(gold[train_ids[0]][0]) == {
        "bend_id",
        "line_end",
        "line_start",
        "reviewer_support_count",
    }
    case = _case(0, include_fp=False)
    sanitized_case = optimizer.PreparedCase(
        **{**case.__dict__, "gold": gold[train_ids[0]]}
    )
    optimizer.configure_source_free_dependencies(runtime, real_evaluator)
    scored = optimizer._score_case(sanitized_case, None)
    assert scored["true_positive"] == 1
    assert scored["false_negative"] == 0


@pytest.mark.parametrize(
    ("name", "purpose"),
    [
        ("development_execution_manifest.json", "CONTROL"),
        ("nineteenth_semantic.json", "TRAIN_PRIMARY_V51_SEMANTIC"),
        ("pool.json", "TRAIN_POOL"),
        ("holdout_result.json", "CONTROL"),
    ],
)
def test_guarded_reader_denies_full28_nineteenth_pool_and_holdout(
    tmp_path, monkeypatch, name: str, purpose: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(optimizer, "ROOT", root)
    path = root / name
    path.write_bytes(b"{}\n")
    reader = optimizer.GuardedReader()
    if name == "nineteenth_semantic.json":
        with pytest.raises(PermissionError, match="NOT_EXACTLY_AUTHORIZED"):
            reader.read_bytes(path, purpose=purpose)
    else:
        with pytest.raises(PermissionError):
            reader.authorize(
                path,
                purpose=purpose,
                sha256=optimizer.sha256_bytes(path.read_bytes()),
                size_bytes=path.stat().st_size,
            )


def test_null_or_absent_selector_score_abstains_and_keeps_v51() -> None:
    cases = _cases()
    case = cases[0]
    rows = [dict(row) for row in case.primary_predictions]
    rows[0].pop("selector_score", None)
    modified = optimizer.PreparedCase(
        **{
            **case.__dict__,
            "primary_predictions": tuple(rows),
            "replay_predictions": tuple(rows),
        }
    )
    rule = {"feature": "selector_score", "operator": "le", "threshold": 1.0}
    retained = optimizer.predictions_after_rule(modified, rule)
    assert any(row["bend_id"] == modified.case_id for row in retained)


def test_complete_nested_oof_scores_each_case_once_and_discloses_not_deployed() -> None:
    nested = optimizer.complete_nested_oof(_cases())
    assert nested["checks"]["all_conjunctive_checks_passed"] is True
    assert nested["checks"]["complete_case_outer_coverage"] is True
    assert nested["checks"]["outer_train_validation_fold_disjoint"] is True
    assert nested["oof_predictions_are_not_final_model_predictions"] is True
    assert len(nested["folds"]) == 18
    assert len({fold["outer_validation_case_id"] for fold in nested["folds"]}) == 18
    assert nested["final_rule"] == {
        "feature": "source_kind",
        "operator": "eq",
        "value": "v41_fragment_tolerant_parent_bridge_sidecar",
    }
    assert "stable_rule_all_outer_and_final" not in nested["checks"]


def test_nested_outer_selector_never_receives_its_validation_case(monkeypatch) -> None:
    cases = _cases()
    original = optimizer.select_rule_by_fixed_inner_folds
    observed_inputs: list[tuple[str, ...]] = []

    def recording_selector(rows):  # type: ignore[no-untyped-def]
        observed_inputs.append(tuple(case.case_id for case in rows))
        return original(rows)

    monkeypatch.setattr(optimizer, "select_rule_by_fixed_inner_folds", recording_selector)
    nested = optimizer.complete_nested_oof(cases)
    assert len(observed_inputs) == 19
    for index, fold_inputs in enumerate(observed_inputs[:18]):
        assert len(fold_inputs) == len(set(fold_inputs)) == 17
        assert cases[index].case_id not in fold_inputs
    assert observed_inputs[-1] == tuple(case.case_id for case in cases)
    assert [fold["outer_validation_case_id"] for fold in nested["folds"]] == [
        case.case_id for case in cases
    ]


def test_nested_oof_rejects_duplicate_or_incomplete_case_groups() -> None:
    cases = _cases()
    with pytest.raises(ValueError, match="18 unique"):
        optimizer.complete_nested_oof(cases[:17])
    with pytest.raises(ValueError, match="18 unique"):
        optimizer.complete_nested_oof(cases[:17] + [cases[0]])


def _context() -> optimizer.BootstrapContext:
    source_names = (
        "evaluator",
        "optimizer",
        "optimizer_tests",
        "publication_helper",
        "runtime",
        "runtime_tests",
        "semantic_cache_preflight_tests",
    )
    bindings = {
        name: {
            "path": f"synthetic/{name}.py",
            "sha256": (
                runtime.sha256_bytes(Path(runtime.__file__).read_bytes())
                if name == "runtime"
                else _sha(name)
            ),
            "size_bytes": 1,
        }
        for name in source_names
    }
    return optimizer.BootstrapContext(
        preregistration={},
        preregistration_sha256=_sha("prereg"),
        source_bytes={},
        source_bindings=bindings,
        bootstrap_entries=(),
    )


def _synthetic_terminals(monkeypatch) -> tuple[bytes, bytes]:  # type: ignore[no-untyped-def]
    c3 = {
        "schema_version": 1,
        "kind": runtime.TERMINAL_KIND,
        "candidate_id": 3,
        "attempt_ordinal": 3,
        "status": runtime.CANDIDATE3_TERMINAL_STATUS,
        "candidate3_continuation_authorized": False,
        "holdout_execution_authorized": False,
        "holdout_access_count": 0,
        "holdout_score_opened": False,
        "promotion_eligible": False,
        "post_terminal_candidate3_model_or_threshold_mutation": "FORBIDDEN",
    }
    c3_bytes = runtime.canonical_bytes(c3)
    c3_sha = runtime.sha256_bytes(c3_bytes)
    c4 = {
        "schema_version": 1,
        "kind": runtime.TERMINAL_KIND,
        "candidate_id": 4,
        "attempt_ordinal": 4,
        "status": runtime.CANDIDATE4_TERMINAL_STATUS,
        "candidate4_continuation_authorized": False,
        "holdout_execution_authorized": False,
        "holdout_access_count": 0,
        "holdout_score_opened": False,
        "promotion_eligible": False,
        "post_terminal_candidate4_model_or_threshold_mutation": "FORBIDDEN",
        "artifact_bindings": {
            "candidate3_terminal": {
                "path": "research/f1_aug30_train18_candidate3_terminal_negative_v1.json",
                "sha256": c3_sha,
                "size_bytes": len(c3_bytes),
            }
        },
    }
    c4_bytes = runtime.canonical_bytes(c4)
    c4_sha = runtime.sha256_bytes(c4_bytes)
    monkeypatch.setattr(runtime, "CANDIDATE3_TERMINAL_SHA256", c3_sha)
    monkeypatch.setattr(runtime, "CANDIDATE3_TERMINAL_SIZE_BYTES", len(c3_bytes))
    monkeypatch.setattr(runtime, "CANDIDATE4_TERMINAL_SHA256", c4_sha)
    monkeypatch.setattr(runtime, "CANDIDATE4_TERMINAL_SIZE_BYTES", len(c4_bytes))
    monkeypatch.setitem(runtime.KNOWN_BINDINGS, "candidate3_terminal_sha256", c3_sha)
    monkeypatch.setitem(runtime.KNOWN_BINDINGS, "candidate4_terminal_sha256", c4_sha)
    c3_path, _old_c3, _old_c3_size = optimizer.FIXED_BINDINGS["candidate3_terminal"]
    c4_path, _old_c4, _old_c4_size = optimizer.FIXED_BINDINGS["candidate4_terminal"]
    monkeypatch.setitem(optimizer.FIXED_BINDINGS, "candidate3_terminal", (c3_path, c3_sha, len(c3_bytes)))
    monkeypatch.setitem(optimizer.FIXED_BINDINGS, "candidate4_terminal", (c4_path, c4_sha, len(c4_bytes)))
    return c3_bytes, c4_bytes


def _reader_after_projection() -> optimizer.GuardedReader:
    return optimizer.GuardedReader(
        entries=[
            {
                "case_id": None,
                "device_id": 1,
                "inode": 1,
                "path": "synthetic/projection.json",
                "purpose": "TRAIN_LABEL_PROJECTION_LAST_READ",
                "sha256": _sha("projection"),
                "size_bytes": 1,
            }
        ],
        projection_read_count=1,
        projection_read_complete=True,
    )


def test_run_terminalizes_post_projection_parse_failure_only(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    target = parent / "candidate5"
    monkeypatch.setattr(optimizer, "ROOT", root)
    monkeypatch.setattr(optimizer, "OUTPUT_ROOT", target)
    context = _context()
    reader = _reader_after_projection()
    state = optimizer.PostProjectionState(
        reader=reader,
        input_bindings={
            "label_projection_sha256": _sha("projection"),
            "train_cache_manifest_sha256": _sha("manifest"),
            "train_source_cache_binding_sha256": _sha("cache-binding"),
        },
        candidate3_terminal_bytes=b"{}\n",
        candidate4_terminal_bytes=b"{}\n",
    )
    monkeypatch.setattr(
        optimizer, "bootstrap_verified_sources", lambda **_kwargs: context
    )

    def fail_after_projection(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise optimizer.PostProjectionFailure(state, ValueError("synthetic schema failure"))

    monkeypatch.setattr(optimizer, "load_train_inputs", fail_after_projection)
    monkeypatch.setattr(
        optimizer,
        "build_fit_publication_payloads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fit builder must not run after projection parse failure")
        ),
    )
    result = optimizer.run(
        preregistration_sha256=_sha("prereg"),
        fit_authorization_sha256=_sha("authorization"),
        output_root=target,
    )
    assert result["published_inventory"] == ["terminal_negative.json"]
    terminal = json.loads((target / "terminal_negative.json").read_bytes())
    assert terminal["status"] == "REJECTED_POST_PROJECTION_INPUT_FAILURE"
    assert terminal["candidate5_metric_computation_started"] is False
    assert terminal["candidate5_metric_computed"] is False
    assert terminal["candidate5_model_published"] is False
    assert terminal["candidate5_pass_artifact_published"] is False
    assert terminal["candidate5_rule_published"] is False
    assert terminal["output_inventory"] == ["terminal_negative.json"]


def test_pass_payloads_exist_only_after_final_runtime_replay(monkeypatch) -> None:
    cases = _cases()
    c3, c4 = _synthetic_terminals(monkeypatch)
    context = _context()
    input_bindings = {
        "label_projection_sha256": _sha("projection"),
        "train_cache_manifest_sha256": _sha("manifest"),
        "train_source_cache_binding_sha256": _sha("cache-binding"),
    }
    payloads, result = optimizer.build_fit_publication_payloads(
        cases,
        _reader_after_projection(),
        context,
        input_bindings=input_bindings,
        fit_authorization_sha256=_sha("authorization"),
        candidate3_terminal_bytes=c3,
        candidate4_terminal_bytes=c4,
    )
    assert set(payloads) == optimizer.PASS_FILENAMES
    assert result["terminal_only"] is False
    freeze = json.loads(payloads["freeze_lock.json"])
    assert freeze["oof_predictions_are_not_final_model_predictions"] is True
    assert freeze["deployed_final_rule_train_replay"]["all_18_primary_replay_exact"] is True
    assert freeze["external_anchor_present"] is False


def test_replay_mismatch_terminalizes_with_no_pass_or_model(monkeypatch) -> None:
    cases = _cases()
    c3, c4 = _synthetic_terminals(monkeypatch)
    context = _context()
    input_bindings = {
        "label_projection_sha256": _sha("projection"),
        "train_cache_manifest_sha256": _sha("manifest"),
        "train_source_cache_binding_sha256": _sha("cache-binding"),
    }

    def mutate(result: dict[str, object]) -> dict[str, object]:
        changed = deepcopy(result)
        changed["predictions"] = []
        return changed

    payloads, result = optimizer.build_fit_publication_payloads(
        cases,
        _reader_after_projection(),
        context,
        input_bindings=input_bindings,
        fit_authorization_sha256=_sha("authorization"),
        candidate3_terminal_bytes=c3,
        candidate4_terminal_bytes=c4,
        replay_result_mutator=mutate,
    )
    assert set(payloads) == {"terminal_negative.json"}
    assert result["terminal_only"] is True
    terminal = json.loads(payloads["terminal_negative.json"])
    assert terminal["status"] == "REJECTED_TRAIN_REPLAY_DEPLOYMENT_MISMATCH"
    assert terminal["candidate5_model_published"] is False


def test_failed_train_gate_is_terminal_only(monkeypatch) -> None:
    cases = _cases(include_fp=False)
    c3, c4 = _synthetic_terminals(monkeypatch)
    payloads, result = optimizer.build_fit_publication_payloads(
        cases,
        _reader_after_projection(),
        _context(),
        input_bindings={
            "label_projection_sha256": _sha("projection"),
            "train_cache_manifest_sha256": _sha("manifest"),
            "train_source_cache_binding_sha256": _sha("cache-binding"),
        },
        fit_authorization_sha256=_sha("authorization"),
        candidate3_terminal_bytes=c3,
        candidate4_terminal_bytes=c4,
    )
    assert set(payloads) == {"terminal_negative.json"}
    assert result["status"] == "REJECTED_TRAIN_GATE"


def test_atomic_publication_no_clobber_and_no_partial(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    target = parent / "candidate5"
    monkeypatch.setattr(optimizer, "ROOT", root)
    monkeypatch.setattr(optimizer, "OUTPUT_ROOT", target)
    payloads = {"terminal_negative.json": b"{}\n"}
    optimizer.publish_atomically(target, payloads)
    assert (target / "terminal_negative.json").read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        optimizer.publish_atomically(target, payloads)

    failed = parent / "failed"
    monkeypatch.setattr(optimizer, "OUTPUT_ROOT", failed)
    with pytest.raises(RuntimeError, match="synthetic"):
        optimizer.publish_atomically(
            failed,
            payloads,
            after_stage_write=lambda _path: (_ for _ in ()).throw(RuntimeError("synthetic")),
        )
    assert not failed.exists()
    assert not tuple(parent.glob(f".{failed.name}.staging-*"))


def test_atomic_publication_preserves_raced_destination(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    target = parent / "candidate5"
    monkeypatch.setattr(optimizer, "ROOT", root)
    monkeypatch.setattr(optimizer, "OUTPUT_ROOT", target)
    payloads = {"terminal_negative.json": b"{}\n"}

    def race(_path: Path) -> None:
        target.mkdir()
        (target / "foreign.txt").write_text("FOREIGN")

    with pytest.raises(FileExistsError):
        optimizer.publish_atomically(target, payloads, after_stage_write=race)
    assert (target / "foreign.txt").read_text() == "FOREIGN"
    assert not tuple(parent.glob(f".{target.name}.staging-*"))


def test_atomic_publication_never_removes_swapped_staging_inode(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    target = parent / "candidate5"
    monkeypatch.setattr(optimizer, "ROOT", root)
    monkeypatch.setattr(optimizer, "OUTPUT_ROOT", target)
    moved_owned = parent / "moved-owned-staging"

    def swap(staging: Path) -> None:
        staging.rename(moved_owned)
        staging.mkdir()
        (staging / "foreign.txt").write_text("FOREIGN")

    with pytest.raises(RuntimeError, match="identity changed"):
        optimizer.publish_atomically(
            target,
            {"terminal_negative.json": b"{}\n"},
            after_stage_write=swap,
        )
    assert not target.exists()
    foreign_staging = tuple(parent.glob(f".{target.name}.staging-*"))
    assert len(foreign_staging) == 1
    assert (foreign_staging[0] / "foreign.txt").read_text() == "FOREIGN"
    assert (moved_owned / "terminal_negative.json").read_bytes() == b"{}\n"


def test_atomic_publication_rejects_visible_parent_symlink_swap(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    target = parent / "candidate5"
    detached_parent = root / "output" / "research-detached"
    foreign_parent = root / "foreign-parent"
    foreign_parent.mkdir()
    monkeypatch.setattr(optimizer, "ROOT", root)
    monkeypatch.setattr(optimizer, "OUTPUT_ROOT", target)
    captured: dict[str, tuple[int, int]] = {}

    def detach(staging: Path) -> None:
        metadata = os.lstat(staging)
        captured["staging"] = (int(metadata.st_dev), int(metadata.st_ino))
        parent.rename(detached_parent)
        parent.symlink_to(foreign_parent, target_is_directory=True)

    with pytest.raises((RuntimeError, ValueError), match="symlink|identity changed"):
        optimizer.publish_atomically(
            target,
            {"terminal_negative.json": b"{}\n"},
            after_stage_write=detach,
        )
    assert not (foreign_parent / target.name).exists()
    quarantines = tuple(detached_parent.glob(f".{target.name}.quarantine-*"))
    assert len(quarantines) == 1
    quarantine_metadata = os.lstat(quarantines[0])
    assert (int(quarantine_metadata.st_dev), int(quarantine_metadata.st_ino)) == captured[
        "staging"
    ]


def test_postrename_failure_quarantines_owned_output(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    target = parent / "candidate5"
    monkeypatch.setattr(optimizer, "ROOT", root)
    monkeypatch.setattr(optimizer, "OUTPUT_ROOT", target)
    captured: dict[str, tuple[int, int]] = {}

    def fail_after_rename(path: Path) -> None:
        metadata = os.lstat(path)
        captured["published"] = (int(metadata.st_dev), int(metadata.st_ino))
        raise RuntimeError("postrename")

    with pytest.raises(RuntimeError, match="postrename"):
        optimizer.publish_atomically(
            target,
            {"terminal_negative.json": b"{}\n"},
            after_rename=fail_after_rename,
        )
    assert not target.exists()
    quarantines = tuple(parent.glob(f".{target.name}.quarantine-*"))
    assert len(quarantines) == 1
    metadata = os.lstat(quarantines[0])
    assert (int(metadata.st_dev), int(metadata.st_ino)) == captured["published"]
    assert (quarantines[0] / "terminal_negative.json").read_bytes() == b"{}\n"


def test_fit_authorization_requires_fresh_separate_single_run() -> None:
    value = {
        "schema_version": 1,
        "kind": optimizer.FIT_AUTHORIZATION_KIND,
        "status": "AUTHORIZED_ONCE",
        "candidate_id": 5,
        "attempt_ordinal": 5,
        "fit_authorized": True,
        "fresh_process_verified": True,
        "parent_authorized": True,
        "auditor_authorized": True,
        "authorized_run_limit": 1,
        "authorized_scope": "CANDIDATE5_TRAIN18_FIT_ONLY",
        "cache_semantic_read_authorized": True,
        "label_projection_read_authorized_once_last": True,
        "pool_access_authorized": False,
        "full_28_access_authorized": False,
        "holdout_access_authorized": False,
        "candidate5_preregistration_sha256": _sha("prereg"),
        "source_binding_commitment_sha256": _sha("sources"),
    }
    optimizer._validate_fit_authorization(
        value,
        authorization_sha256=_sha("authorization"),
        preregistration_sha256=_sha("prereg"),
    )
    for field in ("fresh_process_verified", "parent_authorized", "auditor_authorized"):
        broken = dict(value)
        broken[field] = False
        with pytest.raises(ValueError, match="status contract"):
            optimizer._validate_fit_authorization(
                broken,
                authorization_sha256=_sha("authorization"),
                preregistration_sha256=_sha("prereg"),
            )


def test_preregistration_is_inert_without_distinct_fit_authorization() -> None:
    sources = {
        "evaluator": _binding(optimizer.EVALUATOR_SOURCE, "evaluator"),
        "optimizer": _binding(optimizer.OPTIMIZER_SOURCE, "optimizer"),
        "optimizer_tests": _binding(optimizer.OPTIMIZER_TESTS_SOURCE, "optimizer-tests"),
        "publication_helper": _binding(optimizer.PREFLIGHT_SOURCE, "helper"),
        "runtime": _binding(optimizer.RUNTIME_SOURCE, "runtime"),
        "runtime_tests": _binding(optimizer.RUNTIME_TESTS_SOURCE, "runtime-tests"),
        "semantic_cache_preflight_tests": _binding(
            optimizer.PREFLIGHT_TESTS_SOURCE, "preflight-tests"
        ),
    }
    contract = _prereg(sources)["authorization_contract"]
    assert contract["fit_authorization_artifact_required"] is True
    assert contract["fit_execution_authorized_by_preregistration"] is False
    assert contract["fresh_process_required"] is True
    assert contract["single_authorized_fit_run_required"] is True
