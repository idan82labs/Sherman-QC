from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_CORE = ROOT / "research/f1_ciip_v1/semantic_core.py"
CONTAINER = ROOT / "research/f1_ciip_v1/container.py"
RUNNER = ROOT / "scripts/run_f1_ciip_v1_smoke.py"
STALE_LOCK = ROOT / "docs/research/CIIP_V1_GOLD_FREE_SMOKE_PREEXECUTION_LOCK_20260809.json"
TAMPER_CONTRACT = ROOT / "tests/test_f1_ciip_v1_tamper_contract.py"
EXECUTION_RESULT = ROOT / "research/f1_ciip_v1/execution_result.py"
NEXT_PACKET_REQUIRED_INVENTORY = {
    "docs/research/CIIP_V1_GOLD_FREE_SMOKE_PREEXECUTION_LOCK_20260809.json": {
        "bytes": 9494,
        "sha256": "b80a7dde043a6ca5871121a3d203244a2f9079f79dc795e6e7b9cb47cff1e0d9",
    },
}
PRODUCTION_DELETION = '        "serialized_bytes": 0,\n'
AUTHORIZED_CHANGED_LOCKED_FILES = frozenset({
    "research/f1_ciip_v1/semantic_core.py",
    "research/f1_ciip_v1/container.py",
    "research/f1_ciip_v1/execution_lock.py",
    "research/f1_ciip_v1/independent_validator.py",
    "scripts/run_f1_ciip_v1_smoke.py",
})
AUTHORIZED_ADDITION_PREFIXES = ("tests/", "docs/research/", "output/professor_package/")
PROTECTED_PAPER_FILES = {
    "output/professor_package/Sherman_QC_F1_Cubical_IIP_V1_Preflight_Lock_Review_20260809_v7/05_V7_LOCKED_CUBICAL_IIP_V1_CONFIGURATION.json":
        "813acdd96f5a8108caf2c87dfdd733b5c856248c4522d2789d733e84016dce04",
    "output/professor_package/Sherman_QC_F1_Cubical_IIP_V1_Preflight_Lock_Review_20260809_v7/06_V7_LOCKED_CUBICAL_IIP_V1_SCHEMA.json":
        "e996566e2b7da94dfc99b97e8bb59e5b6f1fdb25692a16890b63731b5c16ea4f",
    "output/professor_package/Sherman_QC_F1_Cubical_IIP_V1_Preflight_Lock_Review_20260809_v7/07_V7_STATIC_CARDINALITY_EVIDENCE_RESOURCE_PROOF.json":
        "9726a1624afa115134135c0451ebf89e3f82241515000a18557988ab721da160",
    "output/professor_package/Sherman_QC_F1_Cubical_IIP_V1_Preflight_Lock_Review_20260809_v7/09_V7_SAFETY_AUTHORITY_AND_TERMINAL_CONTRACT.json":
        "dd8581b61ea83f22153c5c0e13ee4267312ec1f2b9356c1de3a989acf75a651d",
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stale_lock() -> dict[str, object]:
    value = json.loads(STALE_LOCK.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def _current_code_aggregate(lock: dict[str, object]) -> str:
    digest = hashlib.sha256(b"CIIP_CODE_LOCK_V1\0")
    code = lock["code"]
    assert isinstance(code, dict)
    rows = code["files"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        relative = row["path"]
        assert isinstance(relative, str)
        encoded = relative.encode("utf-8")
        raw = (ROOT / relative).read_bytes()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def test_semantic_resource_builder_omits_serialized_bytes_before_encoding() -> None:
    source = SEMANTIC_CORE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SEMANTIC_CORE))
    resource_dicts: list[ast.Dict] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "execute_semantic_role":
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == "resource_values"
                    and isinstance(child.value, ast.Dict)
                ):
                    resource_dicts.append(child.value)
    assert len(resource_dicts) == 1
    keys = {ast.literal_eval(key) for key in resource_dicts[0].keys if key is not None}
    assert keys == {
        "cpu_nanoseconds",
        "wall_nanoseconds",
        "peak_process_tree_rss_bytes",
        "scratch_bytes",
        "watchdog_poll_milliseconds",
        "resource_cap_tripped",
    }
    assert "serialized_bytes" not in keys
    assert PRODUCTION_DELETION not in source


def test_accounting_correction_invalidates_stale_code_lock() -> None:
    lock = _stale_lock()
    code = lock["code"]
    assert isinstance(code, dict)
    rows = code["files"]
    assert isinstance(rows, list)
    semantic_row = next(
        row for row in rows
        if isinstance(row, dict) and row.get("path") == "research/f1_ciip_v1/semantic_core.py"
    )
    current = SEMANTIC_CORE.read_bytes()
    assert _sha256(current) != semantic_row["sha256"]
    current_aggregate = _current_code_aggregate(lock)
    assert current_aggregate != lock["code_sha256"]

    rebound = json.loads(json.dumps(lock))
    rebound_code = rebound["code"]
    for relative in AUTHORIZED_CHANGED_LOCKED_FILES:
        rebound_row = next(
            row for row in rebound_code["files"]
            if row["path"] == relative
        )
        rebound_raw = (ROOT / relative).read_bytes()
        rebound_row["bytes"] = len(rebound_raw)
        rebound_row["sha256"] = _sha256(rebound_raw)
    rebound["code_sha256"] = current_aggregate
    rebound.pop("execution_lock_sha256")
    rebound_execution_digest = _sha256(b"CIIP_EXECUTION_LOCK_V1\0" + _canonical_json(rebound))
    assert rebound_execution_digest != lock["execution_lock_sha256"]

    runner_source = RUNNER.read_text(encoding="utf-8")
    assert "observed_code, observed_code_sha = code_record(ROOT, paths)" in runner_source
    assert 'observed_code_sha != lock["code_sha256"]' in runner_source
    assert 'raise RuntimeError("code changed after immutable execution lock")' in runner_source


def test_accounting_patch_changes_only_authorized_source_and_test_contract() -> None:
    lock = _stale_lock()
    code = lock["code"]
    assert isinstance(code, dict)
    rows = code["files"]
    assert isinstance(rows, list)
    changed: set[str] = set()
    locked_by_path: dict[str, dict[str, object]] = {}
    for row in rows:
        assert isinstance(row, dict)
        relative = row["path"]
        assert isinstance(relative, str)
        locked_by_path[relative] = row
        raw = (ROOT / relative).read_bytes()
        if len(raw) != row["bytes"] or _sha256(raw) != row["sha256"]:
            changed.add(relative)
    assert changed == AUTHORIZED_CHANGED_LOCKED_FILES

    semantic_row = locked_by_path["research/f1_ciip_v1/semantic_core.py"]
    current_semantic = SEMANTIC_CORE.read_bytes()
    assert semantic_row["bytes"] - len(current_semantic) == len(PRODUCTION_DELETION.encode("utf-8"))

    container_row = locked_by_path["research/f1_ciip_v1/container.py"]
    current_container = CONTAINER.read_bytes()
    assert len(current_container) != container_row["bytes"]
    assert _sha256(current_container) != container_row["sha256"]
    container_source = current_container.decode("utf-8")
    assert 'if "serialized_bytes" in built["resource"]:' in container_source
    assert '_u64_json(built["resource"]["serialized_bytes"], "resource.serialized_bytes")' in container_source

    protected_locked_paths = {
        "research/f1_ciip_v1/config.py",
        "research/f1_ciip_v1/fields.py",
        "research/f1_ciip_v1/graph.py",
        "research/f1_ciip_v1/independent_semantic_rebuild.py",
        "research/f1_ciip_v1/native/semantic_worker.cpp",
        "research/f1_ciip_v1/partition.py",
    }
    assert protected_locked_paths.isdisjoint(changed)
    for relative, expected_sha256 in PROTECTED_PAPER_FILES.items():
        assert _sha256((ROOT / relative).read_bytes()) == expected_sha256

    own_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    assert own_relative.startswith(AUTHORIZED_ADDITION_PREFIXES)
    own_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    imported_roots: set[str] = set()
    for node in own_tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            assert node.module is not None
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {"ast", "hashlib", "importlib", "json", "pathlib"}


def test_frozen_eighteen_class_tamper_contract_remains_collected_and_intact() -> None:
    tree = ast.parse(TAMPER_CONTRACT.read_text(encoding="utf-8"), filename=str(TAMPER_CONTRACT))
    collected = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_v7_tamper_")
    ]
    expected = [
        f"test_v7_tamper_{index:02d}_{suffix}"
        for index, suffix in enumerate((
            "source_config_schema_code_dependency_hash_mutation",
            "missing_duplicate_reordered_appended_or_trailing_payload",
            "forged_counts_offsets_lengths_row_sizes_or_container_length",
            "patricia_relation_subtree_digest_or_collision",
            "sample_census_membership_root_or_presampling_cap_domain",
            "actual_o_padding_or_sparse_formula_violation",
            "omitted_scale_cell_edge_interval_or_rejection",
            "filtration_tie_owner_status_or_pairing_mutation",
            "isotropic_preserved_payload_counterpart_or_dependent_id",
            "denominator_rounded_third_nonzero_q_or_survivor",
            "injected_metadata_key_object_string_or_descriptor_order",
            "length_whitespace_padding_or_hidden_xyz_block",
            "known_answer_or_signed_binding_mutation",
            "forbidden_xyz_member_variable_width_or_archive_payload",
            "nonfinite_negative_zero_overflow_q_format_or_eigensolver_disagreement",
            "forged_work_rss_scratch_time_or_serialized_accounting",
            "independent_validator_semantic_import_replacement",
            "top_k_tail_omission_or_changed_terminal_token",
        ), start=1)
    ]
    assert collected == expected
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in collected:
            decorator_names = {
                ast.unparse(decorator) for decorator in node.decorator_list
            }
            assert not any("skip" in name or "xfail" in name for name in decorator_names)


def _load_execution_result_module():
    spec = importlib.util.spec_from_file_location(
        "ciip_execution_result_source_contract", EXECUTION_RESULT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_execution_result(module) -> dict[str, object]:
    roles = [
        {
            "role": role,
            "return_code": 0,
            "cpu_nanoseconds": 1,
            "wall_nanoseconds": 1,
            "peak_process_tree_rss_bytes": 1,
            "scratch_bytes": 1,
            "stdout_sha256": "11" * 32,
            "stderr_sha256": "22" * 32,
            "failure": None,
            "container_sha256": "33" * 32,
        }
        for role in module.ROLE_ORDER
    ]
    return {
        "outcome": module.PASS_TOKEN,
        "failure": None,
        "execution_lock_sha256": "44" * 32,
        "raw_ply_sha256": "55" * 32,
        "roles": roles,
        "rerun_authorized": False,
        "replacement_input_authorized": False,
        "gold_or_labels_accessed": False,
    }


def _execution_result_rejects(module, value: dict[str, object]) -> bool:
    try:
        module.validate_execution_result(value)
    except module.ExecutionResultError:
        return True
    return False


def test_execution_result_validator_keeps_two_tokens_and_three_authority_flags_closed() -> None:
    module = _load_execution_result_module()
    assert {
        module.PASS_TOKEN,
        module.STOP_TOKEN,
    } == {
        "CII_PERSISTENCE_PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET",
        "INVALID_CII_PERSISTENCE_PREFLIGHT_STOP",
    }
    valid = _valid_execution_result(module)
    module.validate_execution_result(valid)

    stopped = dict(valid)
    stopped.update({"outcome": module.STOP_TOKEN, "failure": "TERMINAL", "roles": []})
    module.validate_execution_result(stopped)

    for changed_token in (
        "CII_PERSISTENCE_PREFLIGHT_VALID",
        "INVALID_CII_PERSISTENCE_PREFLIGHT",
        "PASS",
        "STOP",
        "",
    ):
        tampered = dict(valid)
        tampered["outcome"] = changed_token
        assert _execution_result_rejects(module, tampered)

    for flag in (
        "rerun_authorized",
        "replacement_input_authorized",
        "gold_or_labels_accessed",
    ):
        tampered = dict(valid)
        tampered[flag] = True
        assert _execution_result_rejects(module, tampered)


def test_next_packet_inventory_must_include_the_exact_stale_execution_lock_object() -> None:
    relative = STALE_LOCK.relative_to(ROOT).as_posix()
    assert relative in NEXT_PACKET_REQUIRED_INVENTORY
    required = NEXT_PACKET_REQUIRED_INVENTORY[relative]
    raw = STALE_LOCK.read_bytes()
    assert len(raw) == required["bytes"]
    assert _sha256(raw) == required["sha256"]
    lock = json.loads(raw.decode("ascii"))
    assert lock["execution_lock_sha256"] == (
        "54086ad146476dacd1e66217669f359c730646f80b80f6182101f32b19b0158b"
    )
