from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import types

import pytest

from scripts import run_f1_train18_candidate10_full_replay_adapter_preflight_once_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_f1_train18_candidate10_full_replay_adapter_preflight_once_v1.py"
ZERO_SHA = "0" * 64


def _base_cli() -> list[str]:
    return [
        "--runner-sha256",
        ZERO_SHA,
        "--runner-size-bytes",
        "1",
        "--runner-tests-sha256",
        "1" * 64,
        "--runner-tests-size-bytes",
        "1",
    ]


def test_top_level_imports_are_stdlib_only() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "errno",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "stat",
        "sys",
        "types",
        "typing",
    }
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed


def test_exact_final_source_bindings_and_new_namespaces() -> None:
    assert runner.FIXED_SOURCE_BINDINGS["c10_core"][1:] == (
        "3ef5a1c26ba138ff3d085e89483da706288d683797977fa5ad8bc47b9a135e53",
        33019,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c10_core_tests"][1:] == (
        "a612a5a6bd243daa6536bb1139d454f7435431291553c344ef20f72d975d5796",
        26994,
    )
    assert runner.FIXED_CONTROLS["CANDIDATE9_TERMINAL_DISPOSITION"][1:] == (
        "66ccbadbe49d5563355d407f8c319e665a20dd937b3506c74402fc2c97165ea2",
        4992,
    )
    assert len(runner.FIXED_SOURCE_BINDINGS) == 14
    assert runner.FIXED_SOURCE_BINDINGS["c8_runtime_tests"][1:] == (
        "85c8b684dcaf4acaccb9800ba7bb7747c3546b24baae7d3c0685169431f52cfb",
        14652,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c8_adapter_tests"][1:] == (
        "20bb70082daf1e681dd2ce3a1367da59befce4eebabcfd7a75944c5246091b8b",
        14197,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c6_hypothesis_tests"][1:] == (
        "98df5af442627ce71bf539bb6a801cb30e3ad5b3a10f4e680f55a07c6e4c452f",
        26864,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c7_runtime_tests"][1:] == (
        "fbfb3bb91f68fadcf415f2da2a585125e8a95b6b7ffedf3a68879684792c1ee8",
        12364,
    )
    assert runner.RECORD_COUNT == 36
    assert runner.PAIR_COUNT == 18
    assert runner.SOURCE_FILE_COUNT == 72
    for path in (
        runner.C10_QUALIFICATION,
        runner.C10_AUTHORIZATION,
        runner.C10_ANCHOR,
        runner.OUTPUT_ROOT,
    ):
        assert "candidate10" in str(path).lower()
        assert "candidate9-first-primary" not in str(path)


def test_cli_rejects_no_mode_and_conflicting_modes() -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(_base_cli())
    with pytest.raises(SystemExit):
        runner._parse_args(
            [*_base_cli(), "--source-check-only", "--source-control-check-only"]
        )


def test_cli_modes_and_authorization_are_fail_closed() -> None:
    assert runner._parse_args([*_base_cli(), "--source-check-only"]).source_check_only
    assert runner._parse_args(
        [*_base_cli(), "--source-control-check-only"]
    ).source_control_check_only
    with pytest.raises(SystemExit):
        runner._parse_args([*_base_cli(), "--preflight-once"])
    with pytest.raises(SystemExit):
        runner._parse_args(
            [*_base_cli(), "--source-check-only", "--authorization-sha256", ZERO_SHA]
        )
    parsed = runner._parse_args(
        [
            *_base_cli(),
            "--preflight-once",
            "--authorization-sha256",
            ZERO_SHA,
            "--authorization-size-bytes",
            "123",
        ]
    )
    assert parsed.preflight_once is True
    assert parsed.authorization_size_bytes == 123


def test_source_control_mode_dispatches_without_manifest_anchor_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    context = SimpleNamespace()
    monkeypatch.setattr(runner, "bootstrap_verified_sources", lambda **_kwargs: context)
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "qualification_result",
        lambda _context: {
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
            "cache_manifest_opened": False,
            "artifact_source_inventory_opened": False,
        },
    )
    monkeypatch.setattr(
        runner,
        "run_one_shot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("one-shot must not run")
        ),
    )
    assert runner.main([*_base_cli(), "--source-control-check-only"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["cache_manifest_opened"] is False
    assert value["artifact_source_inventory_opened"] is False
    assert not tree["manifest"].exists()
    assert not runner.C10_ANCHOR.exists()
    assert not runner.OUTPUT_ROOT.exists()


def test_decode_rejects_noncanonical_duplicate_and_nonfinite_json() -> None:
    assert runner._decode(b'{"a":1}\n', name="synthetic") == {"a": 1}
    for data in (b'{"a": 1}\n', b'{"a":1,"a":2}\n', b'{"a":NaN}\n'):
        with pytest.raises(ValueError):
            runner._decode(data, name="synthetic")


def _c9_disposition() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate9_terminal_disposition_v1",
        "attempt_ordinal": 9,
        "scientific_hypothesis_id": runner.SCIENTIFIC_HYPOTHESIS_ID,
        "status": "PASS_ENGINEERING_DIAGNOSTIC_NONAUTHORIZING",
        "check_code": "C8_ALL_DIAGNOSTIC_PHASES_PASSED",
        "candidate10_engineering_successor_authorized": True,
    }


def test_candidate9_disposition_semantics_are_exact() -> None:
    runner._validate_candidate9_disposition(_c9_disposition())
    for key, value in (
        ("candidate10_engineering_successor_authorized", False),
        ("check_code", "wrong"),
        ("status", "wrong"),
    ):
        changed = _c9_disposition()
        changed[key] = value
        with pytest.raises(runner.RunnerFailure):
            runner._validate_candidate9_disposition(changed)


def _write_synthetic_authority(
    monkeypatch: pytest.MonkeyPatch,
    tree: dict[str, Path],
    *,
    qualification_mutation: tuple[str, object] | None = None,
    authorization_mutation: tuple[str, object] | None = None,
) -> tuple[SimpleNamespace, bytes, bytes]:
    _patch_static_paths(monkeypatch, tree)
    context = SimpleNamespace(
        source_bindings={},
        source_entries=(),
        source_commitment_sha256="1" * 64,
        _issuance_token=object(),
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    control_commitment = "2" * 64
    qualification = SimpleNamespace(
        manifest_path=runner.CACHE_MANIFEST,
        manifest_sha256=runner.CACHE_MANIFEST_SHA256,
        manifest_size_bytes=runner.CACHE_MANIFEST_SIZE_BYTES,
        control_binding_commitment_sha256=control_commitment,
        control_raw_fd_opened_commitment_sha256="3" * 64,
        control_read_inventory_commitment_sha256="4" * 64,
        control_read_started_commitment_sha256="5" * 64,
        control_verified_inode_commitment_sha256="6" * 64,
    )
    qualification_value = runner._decode(
        runner.build_qualification_record(context, qualification), name="qualification"
    )
    if qualification_mutation is not None:
        qualification_value[qualification_mutation[0]] = qualification_mutation[1]
    qualification_bytes = runner.canonical_bytes(qualification_value)
    runner.C10_QUALIFICATION.write_bytes(qualification_bytes)
    qualification_binding = runner._binding(
        runner.C10_QUALIFICATION,
        runner.sha256_bytes(qualification_bytes),
        len(qualification_bytes),
        kind=runner.QUALIFICATION_KIND,
    )
    authorization_value = runner.build_authorization_contract(
        context,
        qualification_record_binding=qualification_binding,
        control_binding_commitment_sha256=control_commitment,
    )
    if authorization_mutation is not None:
        authorization_value[authorization_mutation[0]] = authorization_mutation[1]
    authorization_bytes = runner.canonical_bytes(authorization_value)
    runner.C10_AUTHORIZATION.write_bytes(authorization_bytes)
    return context, authorization_bytes, qualification_bytes


@pytest.mark.parametrize(
    ("qualification_mutation", "authorization_mutation"),
    [
        (("unexpected", True), None),
        (("status", "wrong"), None),
        (("source_inventory_commitment_sha256", "9" * 64), None),
        (None, ("status", "wrong")),
        (None, ("record_count", True)),
    ],
)
def test_authorization_or_qualification_drift_rejects_before_anchor(
    qualification_mutation: tuple[str, object] | None,
    authorization_mutation: tuple[str, object] | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    context, authorization_bytes, _qualification_bytes = _write_synthetic_authority(
        monkeypatch,
        tree,
        qualification_mutation=qualification_mutation,
        authorization_mutation=authorization_mutation,
    )
    freshness = runner.prevalidate_fresh_output()
    with pytest.raises((PermissionError, ValueError)):
        runner.authorize_preflight_once(
            context,
            authorization_sha256=runner.sha256_bytes(authorization_bytes),
            authorization_size_bytes=len(authorization_bytes),
            output_freshness=freshness,
        )
    assert not runner.C10_ANCHOR.exists()
    assert not runner.OUTPUT_ROOT.exists()


def test_valid_external_authority_reads_only_auth_and_qualification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    context, authorization_bytes, _qualification_bytes = _write_synthetic_authority(
        monkeypatch, tree
    )
    seal = runner.authorize_preflight_once(
        context,
        authorization_sha256=runner.sha256_bytes(authorization_bytes),
        authorization_size_bytes=len(authorization_bytes),
        output_freshness=runner.prevalidate_fresh_output(),
    )
    assert [entry["purpose"] for entry in seal.reader.entries] == [
        "CANDIDATE10_EXTERNAL_AUTHORIZATION",
        "CANDIDATE10_QUALIFICATION_RECORD",
    ]
    assert not runner.CACHE_MANIFEST.exists()
    assert not runner.C10_ANCHOR.exists()
    assert not runner.OUTPUT_ROOT.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "authorization_bytes_sha_size",
        "qualification_bytes_sha_size",
        "coordinated_authority_substitution",
        "control_commitment",
        "authority_reader_entry",
        "output_freshness",
    ],
)
def test_issued_authorization_rejects_post_issuance_substitution(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    context, authorization_bytes, _qualification_bytes = _write_synthetic_authority(
        monkeypatch, tree
    )
    seal = runner.authorize_preflight_once(
        context,
        authorization_sha256=runner.sha256_bytes(authorization_bytes),
        authorization_size_bytes=len(authorization_bytes),
        output_freshness=runner.prevalidate_fresh_output(),
    )

    if mutation == "authorization_bytes_sha_size":
        value = runner._decode(seal.authorization_bytes, name="authorization")
        value["status"] = "SUBSTITUTED_AFTER_ISSUANCE"
        data = runner.canonical_bytes(value)
        object.__setattr__(seal, "authorization_bytes", data)
        object.__setattr__(seal, "authorization_sha256", runner.sha256_bytes(data))
        object.__setattr__(seal, "authorization_size_bytes", len(data))
    elif mutation == "qualification_bytes_sha_size":
        value = runner._decode(seal.qualification_bytes, name="qualification")
        value["status"] = "SUBSTITUTED_AFTER_ISSUANCE"
        data = runner.canonical_bytes(value)
        object.__setattr__(seal, "qualification_bytes", data)
        object.__setattr__(seal, "qualification_sha256", runner.sha256_bytes(data))
        object.__setattr__(seal, "qualification_size_bytes", len(data))
    elif mutation == "coordinated_authority_substitution":
        replacement_commitment = "9" * 64
        qualification_value = runner._decode(
            seal.qualification_bytes, name="qualification"
        )
        qualification_value[
            "control_binding_commitment_sha256"
        ] = replacement_commitment
        qualification_data = runner.canonical_bytes(qualification_value)
        qualification_sha = runner.sha256_bytes(qualification_data)

        authorization_value = runner._decode(
            seal.authorization_bytes, name="authorization"
        )
        authorization_value[
            "control_binding_commitment_sha256"
        ] = replacement_commitment
        authorization_value["qualification_record_binding"][
            "sha256"
        ] = qualification_sha
        authorization_value["qualification_record_binding"][
            "size_bytes"
        ] = len(qualification_data)
        authorization_data = runner.canonical_bytes(authorization_value)

        object.__setattr__(seal, "qualification_bytes", qualification_data)
        object.__setattr__(seal, "qualification_sha256", qualification_sha)
        object.__setattr__(
            seal, "qualification_size_bytes", len(qualification_data)
        )
        object.__setattr__(
            seal, "control_binding_commitment_sha256", replacement_commitment
        )
        object.__setattr__(seal, "authorization_bytes", authorization_data)
        object.__setattr__(
            seal, "authorization_sha256", runner.sha256_bytes(authorization_data)
        )
        object.__setattr__(
            seal, "authorization_size_bytes", len(authorization_data)
        )
    elif mutation == "control_commitment":
        object.__setattr__(seal, "control_binding_commitment_sha256", "9" * 64)
    elif mutation == "authority_reader_entry":
        seal.reader.entries[0]["sha256"] = "9" * 64
    elif mutation == "output_freshness":
        object.__setattr__(
            seal,
            "output_freshness",
            runner.FreshOutputSeal(
                path=seal.output_freshness.path,
                parent_identity=(0, 0),
                _sentinel=seal.output_freshness._sentinel,
            ),
        )
    else:  # pragma: no cover - exhaustive parametrization guard
        raise AssertionError(mutation)

    with pytest.raises(PermissionError):
        runner._validate_authorization(context, seal)
    assert not runner.C10_ANCHOR.exists()
    assert not runner.OUTPUT_ROOT.exists()


def _patch_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    (root / "research").mkdir(parents=True)
    (root / "output/research").mkdir(parents=True)
    cache = root / "output/research/cache"
    cache.mkdir(parents=True)
    allowed = root / "output/research/train-cache"
    allowed.mkdir(parents=True)
    manifest = cache / "manifest.json"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", allowed)
    monkeypatch.setattr(runner, "CACHE_MANIFEST", manifest)
    monkeypatch.setattr(
        runner,
        "OUTPUT_ROOT",
        root
        / "output/research/f1-aug30-train18-candidate10-full-replay-adapter-preflight-v1",
    )
    monkeypatch.setattr(
        runner,
        "C10_ANCHOR",
        root
        / "research/f1_aug30_train18_candidate10_full_replay_adapter_preflight_consumption_v1.json",
    )
    monkeypatch.setattr(runner, "C10_AUTHORIZATION", root / "research/auth.json")
    monkeypatch.setattr(runner, "C10_QUALIFICATION", root / "research/qualification.json")
    return {"root": root, "cache": cache, "allowed": allowed, "manifest": manifest}


def _preanchor_reader_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, runner.GuardedReader]:
    _patch_tree(monkeypatch, tmp_path)
    purposes = [
        "CANDIDATE10_EXTERNAL_AUTHORIZATION",
        "CANDIDATE10_QUALIFICATION_RECORD",
        *runner._expected_control_purposes(),
    ]
    entries = [
        _synthetic_read_entry(index, purpose)
        for index, purpose in enumerate(purposes)
    ]
    reader = runner.GuardedReader(
        entries=entries,
        raw_fd_opened_purposes=list(purposes),
        verified_inode_purposes=list(purposes),
        read_started_purposes=list(purposes),
    )
    consumed = {
        runner._lexical(
            runner.ROOT / str(entry["path"]), role=str(entry["purpose"])
        )
        for entry in entries
    }
    reader.allowed = {path: {} for path in consumed}
    reader.consumed_paths = set(consumed)
    context = SimpleNamespace(source_entries=())
    authorization = SimpleNamespace(reader=reader)
    qualification = SimpleNamespace(reader=reader)
    runner._validate_preanchor_reader_state(context, authorization, qualification)
    return context, authorization, qualification, reader


@pytest.mark.parametrize("contamination", ["manifest_pre_read", "artifact_inventory"])
def test_preanchor_reader_state_rejects_manifest_or_inventory_contamination(
    contamination: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context, authorization, qualification, reader = _preanchor_reader_fixture(
        monkeypatch, tmp_path
    )
    if contamination == "manifest_pre_read":
        entry = {
            "device_id": 7,
            "inode": 99999,
            "path": str(runner.CACHE_MANIFEST.relative_to(runner.ROOT)),
            "purpose": "TRAIN_CACHE_MANIFEST",
            "sha256": "7" * 64,
            "size_bytes": 7,
        }
        reader.entries.append(entry)
        reader.raw_fd_opened_purposes.append("TRAIN_CACHE_MANIFEST")
        reader.verified_inode_purposes.append("TRAIN_CACHE_MANIFEST")
        reader.read_started_purposes.append("TRAIN_CACHE_MANIFEST")
        path = runner._lexical(runner.CACHE_MANIFEST, role="TRAIN_CACHE_MANIFEST")
        reader.allowed[path] = {}
        reader.consumed_paths.add(path)
    else:
        reader.artifact_inventory = object()

    with pytest.raises(PermissionError):
        runner._validate_preanchor_reader_state(
            context, authorization, qualification
        )
    assert not runner.C10_ANCHOR.exists()


def _manifest_fixture(
    tree: dict[str, Path],
    *,
    write_files: bool,
) -> tuple[dict[str, object], SimpleNamespace]:
    train_ids = tuple(f"case-{index:02d}" for index in range(18))
    scan_by_case = {
        case_id: runner.sha256_bytes(f"scan-{case_id}".encode()) for case_id in train_ids
    }
    train_commitment = runner.sha256_bytes(
        runner.canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    rows: list[dict[str, object]] = []
    for pair, case_id in enumerate(train_ids):
        runtime_id = f"T{pair:03d}"
        semantic = runner.canonical_bytes({"semantic": pair})
        pool = runner.canonical_bytes({"pool": pair})
        for pass_name in ("primary", "replay"):
            semantic_path = (
                tree["allowed"] / pass_name / runtime_id / "semantic_result.json"
            )
            pool_path = (
                tree["allowed"] / pass_name / runtime_id / "pool" / f"{runtime_id}.json"
            )
            if write_files:
                semantic_path.parent.mkdir(parents=True, exist_ok=True)
                pool_path.parent.mkdir(parents=True, exist_ok=True)
                semantic_path.write_bytes(semantic)
                pool_path.write_bytes(pool)
            rows.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "pool_path": str(pool_path.relative_to(tree["root"])),
                    "pool_sha256": runner.sha256_bytes(pool),
                    "runtime_case_id": runtime_id,
                    "scan_sha256": scan_by_case[case_id],
                    "semantic_path": str(semantic_path.relative_to(tree["root"])),
                    "semantic_sha256": runner.sha256_bytes(semantic),
                }
            )
    qualification = SimpleNamespace(
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment_sha256=train_commitment,
    )
    manifest = {
        "case_count": 18,
        "holdout_entry_count": 0,
        "kind": runner.CACHE_MANIFEST_KIND,
        "record_count": 36,
        "records": rows,
        "schema_version": 1,
        "source_only": True,
        "split_sha256": runner.FIXED_CONTROLS["SEALED_SPLIT"][1],
        "train_allowlist_sha256": runner.FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "train_set_commitment_sha256": train_commitment,
        "v51_scan_only_manifest_sha256": runner.V51_SCAN_ONLY_MANIFEST_SHA256,
    }
    return manifest, qualification


def test_full_manifest_exact_36_row_72_path_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    manifest, qualification = _manifest_fixture(tree, write_files=False)
    records = runner._validate_full_manifest(
        manifest, qualification, manifest_binding_sha256="2" * 64
    )
    assert len(records) == 36
    assert len({record.manifest_record_binding_sha256 for record in records}) == 36
    assert len({record.semantic_path for record in records} | {record.pool_path for record in records}) == 72
    assert [record.pass_name for record in records[:4]] == [
        "primary",
        "replay",
        "primary",
        "replay",
    ]


@pytest.mark.parametrize("mutation", ["reorder", "duplicate_path", "replay_hash", "bool_count"])
def test_full_manifest_rejects_order_path_hash_and_strict_type_drift(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    manifest, qualification = _manifest_fixture(tree, write_files=False)
    rows = manifest["records"]
    assert isinstance(rows, list)
    if mutation == "reorder":
        rows[0], rows[2] = rows[2], rows[0]
    elif mutation == "duplicate_path":
        rows[2]["semantic_path"] = rows[0]["semantic_path"]
    elif mutation == "replay_hash":
        rows[1]["pool_sha256"] = "3" * 64
    else:
        manifest["record_count"] = True
    with pytest.raises(runner.RunnerFailure):
        runner._validate_full_manifest(
            manifest, qualification, manifest_binding_sha256="2" * 64
        )


def test_guarded_reader_attests_72_files_and_raw_inode_read_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    specifications: list[tuple[Path, str, str]] = []
    for index in range(72):
        path = tree["allowed"] / f"file-{index:02d}.json"
        data = runner.canonical_bytes({"index": index})
        path.write_bytes(data)
        specifications.append((path, f"TRAIN_RECORD_{index:02d}_FILE", runner.sha256_bytes(data)))
    reader = runner.GuardedReader()
    inventory = reader.install_artifact_inventory(specifications)
    assert len(inventory.bindings) == 72
    first = inventory.bindings[specifications[0][0]]
    reader.authorize(
        first.path,
        purpose=first.purpose,
        sha256=first.sha256,
        size_bytes=first.size_bytes,
        artifact_inventory=inventory,
    )
    assert reader.read_bytes(first.path, purpose=first.purpose)
    assert reader.raw_fd_opened_purposes == [first.purpose]
    assert reader.verified_inode_purposes == [first.purpose]
    assert reader.read_started_purposes == [first.purpose]
    with pytest.raises(PermissionError):
        reader.read_bytes(first.path, purpose=first.purpose)


def test_guarded_reader_requires_exact_cache_capability_and_forbids_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    path = tree["allowed"] / "x.json"
    data = b"x"
    path.write_bytes(data)
    reader = runner.GuardedReader()
    with pytest.raises(PermissionError):
        reader.authorize(
            path,
            purpose="TRAIN_RECORD_00_FILE",
            sha256=runner.sha256_bytes(data),
            size_bytes=1,
        )
    label_path = tree["root"] / "research/labels.json"
    label_path.write_bytes(b"x")
    with pytest.raises(PermissionError):
        reader.authorize(
            label_path,
            purpose="LABEL_INPUT",
            sha256=runner.sha256_bytes(b"x"),
            size_bytes=1,
        )


def test_exact_source_read_rejects_tamper_symlink_and_hardlink_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    source = tree["root"] / "source.py"
    source.write_bytes(b"source bytes\n")
    data, entry = runner._read_exact_source(
        source,
        role="SYNTHETIC_SOURCE",
        sha256=runner.sha256_bytes(source.read_bytes()),
        size_bytes=source.stat().st_size,
    )
    assert data == b"source bytes\n"
    assert entry["size_bytes"] == len(data)
    with pytest.raises(ValueError):
        runner._read_exact_source(
            source,
            role="SYNTHETIC_SOURCE",
            sha256="0" * 64,
            size_bytes=source.stat().st_size,
        )
    symlink = tree["root"] / "source-link.py"
    symlink.symlink_to(source)
    with pytest.raises(ValueError):
        runner._read_exact_source(
            symlink,
            role="SYNTHETIC_SOURCE_LINK",
            sha256=runner.sha256_bytes(data),
            size_bytes=len(data),
        )
    alias = tree["root"] / "source-hardlink.py"
    os.link(source, alias)
    with pytest.raises(ValueError):
        runner._read_exact_source(
            source,
            role="SYNTHETIC_SOURCE_HARDLINK",
            sha256=runner.sha256_bytes(data),
            size_bytes=len(data),
        )


def test_reject_preloaded_controlled_name_and_file_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    source = tree["root"] / "controlled.py"
    source.write_bytes(b"x = 1\n")
    forbidden = types.ModuleType("synthetic.forbidden")
    monkeypatch.setattr(runner, "FORBIDDEN_PRELOADED", frozenset({"synthetic.forbidden"}))
    monkeypatch.setitem(__import__("sys").modules, "synthetic.forbidden", forbidden)
    with pytest.raises(RuntimeError):
        runner._reject_preloaded((source,))
    monkeypatch.delitem(__import__("sys").modules, "synthetic.forbidden")
    alias = types.ModuleType("synthetic.alias")
    alias.__file__ = str(source)
    monkeypatch.setitem(__import__("sys").modules, "synthetic.alias", alias)
    with pytest.raises(RuntimeError):
        runner._reject_preloaded((source,))


@pytest.mark.parametrize("failure_point", ["raw", "inode", "read_start"])
def test_raw_open_inode_and_read_start_are_distinct_fail_closed_events(
    failure_point: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    path = tree["root"] / "events.bin"
    path.write_bytes(b"events")
    events: list[str] = []

    def raw() -> None:
        events.append("raw")
        if failure_point == "raw":
            raise RuntimeError("raw")

    def inode(_identity: tuple[int, int]) -> None:
        events.append("inode")
        if failure_point == "inode":
            raise RuntimeError("inode")

    def start(_identity: tuple[int, int]) -> None:
        events.append("read_start")
        if failure_point == "read_start":
            raise RuntimeError("read_start")

    with pytest.raises(RuntimeError):
        runner._read_regular_file_once(
            path,
            role="SYNTHETIC_EVENTS",
            after_raw_fd_open=raw,
            after_verified_inode=inode,
            before_first_read=start,
        )
    expected = {
        "raw": ["raw"],
        "inode": ["raw", "inode"],
        "read_start": ["raw", "inode", "read_start"],
    }
    assert events == expected[failure_point]


@pytest.mark.parametrize("failure_point", ["inode", "read", "hash"])
def test_guarded_reader_partial_failure_records_exact_event_category(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    _patch_static_paths(monkeypatch, tree)
    path = runner.C10_AUTHORIZATION
    path.write_bytes(b"one")
    reader = runner.GuardedReader()
    expected_sha = runner.sha256_bytes(b"two" if failure_point == "hash" else b"one")
    reader.authorize(
        path,
        purpose="CANDIDATE10_EXTERNAL_AUTHORIZATION",
        sha256=expected_sha,
        size_bytes=3,
    )
    if failure_point == "inode":
        replacement = path.with_name("replacement")
        replacement.write_bytes(b"one")
        replacement.replace(path)
    elif failure_point == "read":
        monkeypatch.setattr(
            runner.os,
            "read",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic read")),
        )
    with pytest.raises((ValueError, OSError)):
        reader.read_bytes(path, purpose="CANDIDATE10_EXTERNAL_AUTHORIZATION")
    assert reader.raw_fd_opened_purposes == ["CANDIDATE10_EXTERNAL_AUTHORIZATION"]
    if failure_point == "inode":
        assert reader.verified_inode_purposes == []
        assert reader.read_started_purposes == []
    else:
        assert reader.verified_inode_purposes == ["CANDIDATE10_EXTERNAL_AUTHORIZATION"]
        assert reader.read_started_purposes == ["CANDIDATE10_EXTERNAL_AUTHORIZATION"]
    assert reader.entries == []


def test_file_bindings_are_domain_separated_and_record_bound() -> None:
    entry = {
        "device_id": 1,
        "inode": 2,
        "path": "x",
        "purpose": "TRAIN_RECORD_00_PRIMARY_SEMANTIC",
        "sha256": "4" * 64,
        "size_bytes": 5,
    }
    first = runner._artifact_file_binding(entry, manifest_record_binding_sha256="5" * 64)
    second = runner._artifact_file_binding(entry, manifest_record_binding_sha256="6" * 64)
    manifest = runner._manifest_file_binding(entry)
    assert len({first, second, manifest}) == 3


class FakeCore:
    class RecordBytes(SimpleNamespace):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)

    def __init__(self) -> None:
        self.records: tuple[SimpleNamespace, ...] = ()

    @staticmethod
    def make_verified_context(**_kwargs: object) -> object:
        return object()

    def preflight_all_records(
        self,
        _context: object,
        records: tuple[SimpleNamespace, ...],
        *,
        manifest_binding_sha256: str,
    ) -> dict[str, object]:
        self.records = records
        assert len(records) == 36
        assert all(records[index].semantic_bytes == records[index + 1].semantic_bytes for index in range(0, 36, 2))
        assert all(records[index].pool_bytes == records[index + 1].pool_bytes for index in range(0, 36, 2))
        return {
            "aggregate_evidence_commitment_sha256": "7" * 64,
            "input_binding_commitment_sha256": "8" * 64,
            "projection_receipt_attestation_commitment_sha256": "9" * 64,
            "replay_diagnostic_commitment_sha256": "a" * 64,
            "source_lineage_commitment_sha256": "b" * 64,
            "manifest_binding_sha256": manifest_binding_sha256,
            "adapted_pool_primary_replay_byte_identity": True,
            "all_d01_d05_diagnostics_passed": True,
            "diagnostic_primary_replay_byte_identity": True,
            "original_pool_primary_replay_byte_identity": True,
            "projection_receipt_attestation_complete": True,
            "semantic_primary_replay_byte_identity": True,
        }

    @staticmethod
    def _validate_public_result(_value: object) -> None:
        return None

    canonical_bytes = staticmethod(runner.canonical_bytes)


def _fake_public_outcome() -> SimpleNamespace:
    core = {
        "aggregate_evidence_commitment_sha256": "7" * 64,
        "input_binding_commitment_sha256": "8" * 64,
        "projection_receipt_attestation_commitment_sha256": "9" * 64,
        "replay_diagnostic_commitment_sha256": "a" * 64,
        "source_lineage_commitment_sha256": "b" * 64,
        "adapted_pool_primary_replay_byte_identity": True,
        "all_d01_d05_diagnostics_passed": True,
        "diagnostic_primary_replay_byte_identity": True,
        "original_pool_primary_replay_byte_identity": True,
        "projection_receipt_attestation_complete": True,
        "semantic_primary_replay_byte_identity": True,
    }
    return SimpleNamespace(
        core_result=core,
        core_result_bytes=runner.canonical_bytes(core),
        manifest_binding_sha256="e" * 64,
        artifact_inventory_commitment_sha256="d" * 64,
    )


def _patch_static_paths(monkeypatch: pytest.MonkeyPatch, tree: dict[str, Path]) -> None:
    fixed: dict[str, tuple[Path, str, int]] = {}
    for index, (name, (_path, sha, size)) in enumerate(runner.FIXED_CONTROLS.items()):
        fixed[name] = (tree["root"] / f"research/control-{index}.json", sha, size)
    monkeypatch.setattr(runner, "FIXED_CONTROLS", fixed)
    monkeypatch.setattr(
        runner,
        "CUSTODY_EVIDENCE_PATHS",
        {
            name: tree["root"] / f"research/custody-{name}.json"
            for name in ("access_evidence", "builder", "staged_validator_pass", "validator")
        },
    )


def _synthetic_execution_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Path], SimpleNamespace, SimpleNamespace, runner.GuardedReader, FakeCore]:
    tree = _patch_tree(monkeypatch, tmp_path)
    _patch_static_paths(monkeypatch, tree)
    manifest, qualification_base = _manifest_fixture(tree, write_files=True)
    manifest_bytes = runner.canonical_bytes(manifest)
    tree["manifest"].write_bytes(manifest_bytes)
    monkeypatch.setattr(runner, "CACHE_MANIFEST_SHA256", runner.sha256_bytes(manifest_bytes))
    monkeypatch.setattr(runner, "CACHE_MANIFEST_SIZE_BYTES", len(manifest_bytes))
    reader = runner.GuardedReader()
    qualification = SimpleNamespace(
        **vars(qualification_base),
        reader=reader,
        manifest_path=tree["manifest"],
        manifest_sha256=runner.sha256_bytes(manifest_bytes),
        manifest_size_bytes=len(manifest_bytes),
        candidate7_disposition_bytes=b"synthetic-c7",
    )
    fake_core = FakeCore()
    source_bindings = {
        name: {"sha256": runner.sha256_bytes(name.encode())}
        for name in (
            "builder",
            "c6_hypothesis",
            "c7_runtime",
            "c8_adapter",
            "c8_runtime",
            "c9_adapter",
            "c9_adapter_tests",
            "cluster",
        )
    }
    context = SimpleNamespace(
        c10_core=fake_core,
        c9_adapter=object(),
        c9_adapter_context=object(),
        c8_adapter=object(),
        c8_adapter_context=object(),
        c8_runtime=object(),
        source_bindings=source_bindings,
        source_entries=(),
    )
    monkeypatch.setattr(runner, "_validate_context", lambda _context: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_complete_inventory", lambda *_args: None)
    return tree, context, qualification, reader, fake_core


def test_synthetic_full36_constructs_exact_core_records_and_runs_once_each(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tree, context, qualification, reader, fake_core = _synthetic_execution_fixture(
        monkeypatch, tmp_path
    )
    anchor_checks: list[int] = []
    monkeypatch.setattr(
        runner,
        "revalidate_anchor",
        lambda *_args: anchor_checks.append(len(anchor_checks) + 1),
    )
    outcome = runner.read_adapt_diagnose_full_inventory(
        context, qualification, SimpleNamespace()
    )
    assert len(fake_core.records) == 36
    assert len({record.manifest_record_binding_sha256 for record in fake_core.records}) == 36
    assert len(
        {record.semantic_file_binding_sha256 for record in fake_core.records}
        | {record.pool_file_binding_sha256 for record in fake_core.records}
    ) == 72
    assert reader.raw_fd_opened_purposes[0] == "TRAIN_CACHE_MANIFEST"
    assert len(reader.raw_fd_opened_purposes) == 73
    assert len(anchor_checks) == 187
    assert outcome.core_result["all_d01_d05_diagnostics_passed"] is True


def test_post_manifest_pre_semantic_anchor_parent_swap_stops_before_artifact_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree, context, qualification, reader, _fake_core = _synthetic_execution_fixture(
        monkeypatch, tmp_path
    )
    checks = 0

    def revalidate(*_args: object) -> None:
        nonlocal checks
        checks += 1
        if checks == 5:
            original = tree["root"] / "research"
            original.rename(tree["root"] / "research-swapped-out")
            original.mkdir()
            raise PermissionError("synthetic anchor parent swap")

    monkeypatch.setattr(runner, "revalidate_anchor", revalidate)
    with pytest.raises(runner.RunnerFailure) as captured:
        runner.read_adapt_diagnose_full_inventory(
            context, qualification, SimpleNamespace()
        )
    assert captured.value.stage == "ARTIFACT_INVENTORY_OR_READ"
    assert str(captured.value) == "C10_ARTIFACT_INVENTORY_OR_READ_FAILED"
    assert checks == 5
    assert reader.raw_fd_opened_purposes == ["TRAIN_CACHE_MANIFEST"]


def _synthetic_read_entry(index: int, purpose: str) -> dict[str, object]:
    return {
        "device_id": 1,
        "inode": index + 1,
        "path": f"synthetic/{index}.json",
        "purpose": purpose,
        "sha256": f"{index + 1:064x}",
        "size_bytes": index + 1,
    }


def test_aggregate_ledger_contains_only_counts_booleans_and_commitments() -> None:
    source_count = len(runner.FIXED_SOURCE_BINDINGS) + 2
    source_entries = [
        _synthetic_read_entry(index, f"SOURCE_{index}") for index in range(source_count)
    ]
    entries = list(source_entries)
    entries.extend(
        [
            _synthetic_read_entry(source_count, "CANDIDATE10_EXTERNAL_AUTHORIZATION"),
            _synthetic_read_entry(source_count + 1, "CANDIDATE10_QUALIFICATION_RECORD"),
        ]
    )
    control_start = source_count + 2
    for offset, purpose in enumerate(
        runner._expected_control_purposes(), start=control_start
    ):
        entries.append(_synthetic_read_entry(offset, purpose))
    manifest_index = control_start + len(runner._expected_control_purposes())
    entries.append(_synthetic_read_entry(manifest_index, "TRAIN_CACHE_MANIFEST"))
    for index in range(72):
        entries.append(
            _synthetic_read_entry(
                manifest_index + 1 + index, f"TRAIN_RECORD_{index:02d}_FILE"
            )
        )
    reader = runner.GuardedReader(
        entries=entries,
        raw_fd_opened_purposes=[str(entry["purpose"]) for entry in entries],
        verified_inode_purposes=[str(entry["purpose"]) for entry in entries],
        read_started_purposes=[str(entry["purpose"]) for entry in entries],
    )
    reader.artifact_inventory = SimpleNamespace(bindings={index: index for index in range(72)})
    context = SimpleNamespace(
        source_entries=source_entries,
        source_commitment_sha256="b" * 64,
    )
    qualification = SimpleNamespace(
        reader=reader,
        control_binding_commitment_sha256="c" * 64,
    )
    outcome = SimpleNamespace(
        artifact_inventory_commitment_sha256="d" * 64,
        manifest_binding_sha256="e" * 64,
        core_result_bytes=b"{}\n",
        core_result={"aggregate_evidence_commitment_sha256": "f" * 64},
    )
    ledger = runner._ledger_value(
        context,
        SimpleNamespace(),
        qualification,
        consumption=SimpleNamespace(sha256="1" * 64),
        outcome=outcome,
        failure_stage=None,
        check_code="C10_FULL_REPLAY_PREFLIGHT_PASSED",
        projection_receipt_attestation_sha256="2" * 64,
    )
    assert ledger["expected_total_read_count"] == 101
    assert ledger["expected_artifact_file_count"] == 72
    encoded = json.dumps(ledger, sort_keys=True)
    for forbidden in (
        "synthetic/",
        '"path"',
        '"inode"',
        '"case_id"',
        "semantic_bytes",
        '"artifact_file_count":',
        '"total_read_count":',
    ):
        assert forbidden not in encoded


def test_terminal_partial_record_failure_exposes_no_dynamic_observed_counts() -> None:
    source_count = len(runner.FIXED_SOURCE_BINDINGS) + 2
    source_entries = [
        _synthetic_read_entry(index, f"SOURCE_{index}") for index in range(source_count)
    ]
    entries = list(source_entries)
    entries.extend(
        [
            _synthetic_read_entry(source_count, "CANDIDATE10_EXTERNAL_AUTHORIZATION"),
            _synthetic_read_entry(source_count + 1, "CANDIDATE10_QUALIFICATION_RECORD"),
        ]
    )
    start = source_count + 2
    for offset, purpose in enumerate(runner._expected_control_purposes(), start=start):
        entries.append(_synthetic_read_entry(offset, purpose))
    manifest_index = start + len(runner._expected_control_purposes())
    entries.append(_synthetic_read_entry(manifest_index, "TRAIN_CACHE_MANIFEST"))
    entries.append(_synthetic_read_entry(manifest_index + 1, "TRAIN_RECORD_00_PRIMARY_SEMANTIC"))
    purposes = [str(entry["purpose"]) for entry in entries]
    reader = runner.GuardedReader(
        entries=entries,
        raw_fd_opened_purposes=purposes,
        verified_inode_purposes=purposes,
        read_started_purposes=purposes,
    )
    ledger = runner._ledger_value(
        SimpleNamespace(
            source_entries=source_entries,
            source_commitment_sha256="b" * 64,
        ),
        SimpleNamespace(),
        SimpleNamespace(
            reader=reader,
            control_binding_commitment_sha256="c" * 64,
        ),
        consumption=SimpleNamespace(sha256="1" * 64),
        outcome=None,
        failure_stage="CORE_PREFLIGHT",
        check_code=runner.ENGINEERING_FAILURE_CODES["CORE_PREFLIGHT"],
    )
    assert ledger["assertions"]["any_artifact_raw_opened"] is True
    assert ledger["assertions"]["all72_artifact_files_completed"] is False
    assert ledger["expected_record_count"] == 36
    assert ledger["expected_source_file_count"] == 72
    encoded = json.dumps(ledger, sort_keys=True)
    for forbidden in (
        '"artifact_file_count":',
        '"total_read_count":',
        '"manifest_read_count":',
        '"record_count":0',
    ):
        assert forbidden not in encoded


def test_core_failure_after_full_read_keeps_custody_completion_true() -> None:
    source_count = len(runner.FIXED_SOURCE_BINDINGS) + 2
    source_entries = [
        _synthetic_read_entry(index, f"SOURCE_{index}") for index in range(source_count)
    ]
    entries = list(source_entries)
    entries.extend(
        [
            _synthetic_read_entry(source_count, "CANDIDATE10_EXTERNAL_AUTHORIZATION"),
            _synthetic_read_entry(source_count + 1, "CANDIDATE10_QUALIFICATION_RECORD"),
        ]
    )
    start = source_count + 2
    for offset, purpose in enumerate(runner._expected_control_purposes(), start=start):
        entries.append(_synthetic_read_entry(offset, purpose))
    manifest_index = start + len(runner._expected_control_purposes())
    entries.append(_synthetic_read_entry(manifest_index, "TRAIN_CACHE_MANIFEST"))
    for index in range(72):
        entries.append(
            _synthetic_read_entry(
                manifest_index + 1 + index,
                f"TRAIN_RECORD_{index:02d}_FILE",
            )
        )
    purposes = [str(entry["purpose"]) for entry in entries]
    reader = runner.GuardedReader(
        entries=entries,
        raw_fd_opened_purposes=purposes,
        verified_inode_purposes=purposes,
        read_started_purposes=purposes,
    )
    reader.artifact_inventory = SimpleNamespace(
        bindings={index: index for index in range(72)}
    )
    ledger = runner._ledger_value(
        SimpleNamespace(
            source_entries=source_entries,
            source_commitment_sha256="b" * 64,
        ),
        SimpleNamespace(),
        SimpleNamespace(
            reader=reader, control_binding_commitment_sha256="c" * 64
        ),
        consumption=SimpleNamespace(sha256="1" * 64),
        outcome=None,
        failure_stage="CORE_PREFLIGHT",
        check_code=runner.ENGINEERING_FAILURE_CODES["CORE_PREFLIGHT"],
    )
    assert ledger["status"] == "TERMINAL_ENGINEERING_FAILURE"
    assert ledger["assertions"]["all72_artifact_files_completed"] is True
    assert ledger["assertions"]["all_source_files_read_independently"] is True
    assert ledger["assertions"]["exact_manifest_record_vector_completed"] is True
    assert ledger["assertions"]["exact_primary_replay_byte_identity_verified"] is False


def test_projection_receipt_attestation_is_distinct_aggregate_only() -> None:
    fake_core = FakeCore()
    core = fake_core.preflight_all_records(
        object(), tuple(SimpleNamespace(semantic_bytes=b"x", pool_bytes=b"y") for _ in range(36)),
        manifest_binding_sha256="e" * 64,
    )
    outcome = SimpleNamespace(
        core_result=core,
        core_result_bytes=runner.canonical_bytes(core),
        manifest_binding_sha256="e" * 64,
    )
    value = runner._projection_receipt_attestation(
        SimpleNamespace(source_commitment_sha256="f" * 64), outcome
    )
    assert value["kind"] == runner.PROJECTION_ATTESTATION_KIND
    assert value["record_count"] == 36
    assert value["pair_count"] == 18
    assert value["adapted_cache_published"] is False
    assert value["projection_receipt_attestation_complete"] is True
    encoded = json.dumps(value, sort_keys=True)
    assert "semantic_bytes" not in value
    assert "pool_bytes" not in value
    for forbidden in ("case_id", '"path"', "semantic_fixture", "pool_fixture"):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    ("mutation", "key", "value"),
    [
        ("extra", "unexpected", True),
        ("missing", "kind", None),
        ("replace", "kind", "wrong"),
        ("replace", "status", "wrong"),
        ("replace", "schema_version", True),
        ("replace", "record_count", True),
        ("replace", "core_result_sha256", None),
    ],
)
def test_projection_attestation_validator_is_exact_and_strict(
    mutation: str, key: str, value: object
) -> None:
    valid = runner._projection_receipt_attestation(
        SimpleNamespace(source_commitment_sha256="f" * 64),
        _fake_public_outcome(),
    )
    changed = dict(valid)
    if mutation == "missing":
        changed.pop(key)
    else:
        changed[key] = value
    with pytest.raises(ValueError):
        runner._validate_projection_receipt_attestation(changed)


@pytest.mark.parametrize(
    ("mutation", "key", "value"),
    [
        ("extra", "unexpected", True),
        ("missing", "kind", None),
        ("replace", "kind", "wrong"),
        ("replace", "status", "wrong"),
        ("replace", "schema_version", True),
        ("replace", "pair_count", True),
        ("replace", "preflight_read_ledger_sha256", None),
    ],
)
def test_success_public_validator_is_exact_and_strict(
    mutation: str, key: str, value: object
) -> None:
    valid = runner._success_public_payload(
        _fake_public_outcome(),
        ledger_sha256="1" * 64,
        consumption=SimpleNamespace(sha256="2" * 64),
        projection_receipt_attestation_sha256="3" * 64,
    )
    changed = dict(valid)
    if mutation == "missing":
        changed.pop(key)
    else:
        changed[key] = value
    with pytest.raises(ValueError):
        runner._validate_success_public_payload(changed)


@pytest.mark.parametrize(
    ("mutation", "key", "value"),
    [
        ("extra", "unexpected", True),
        ("missing", "kind", None),
        ("replace", "kind", "wrong"),
        ("replace", "status", "wrong"),
        ("replace", "schema_version", True),
        ("replace", "attempt_ordinal", True),
        ("replace", "preflight_read_ledger_sha256", None),
    ],
)
def test_terminal_public_validator_is_exact_and_strict(
    mutation: str, key: str, value: object
) -> None:
    valid = runner._terminal_public_payload(
        failure_stage="CORE_PREFLIGHT", ledger_sha256="1" * 64
    )
    changed = dict(valid)
    if mutation == "missing":
        changed.pop(key)
    else:
        changed[key] = value
    with pytest.raises(ValueError):
        runner._validate_terminal_public_payload(changed)


@pytest.mark.parametrize(
    ("binding_key", "field", "replacement"),
    [
        (
            "preflight_read_ledger_binding",
            "filename",
            "redirected-ledger.json",
        ),
        (
            "projection_receipt_attestation_binding",
            "filename",
            "redirected-projection.json",
        ),
        ("result_binding", "filename", "redirected-result.json"),
        ("authorization_binding", "namespace", "research/redirected-auth.json"),
        (
            "consumption_anchor_binding",
            "namespace",
            "research/redirected-anchor.json",
        ),
        (
            "qualification_record_binding",
            "namespace",
            "research/redirected-qualification.json",
        ),
    ],
)
def test_lock_validator_requires_exact_filenames_and_namespaces(
    binding_key: str, field: str, replacement: str
) -> None:
    value = runner._lock_payload(
        SimpleNamespace(source_commitment_sha256="1" * 64),
        SimpleNamespace(
            authorization_sha256="2" * 64,
            authorization_size_bytes=2,
            qualification_sha256="3" * 64,
            qualification_size_bytes=3,
        ),
        SimpleNamespace(control_binding_commitment_sha256="4" * 64),
        SimpleNamespace(path=runner.C10_ANCHOR, sha256="5" * 64, size_bytes=5),
        SimpleNamespace(
            core_result={"aggregate_evidence_commitment_sha256": "6" * 64},
            manifest_binding_sha256="7" * 64,
        ),
        ledger_bytes=runner.canonical_bytes({"ledger": True}),
        result_bytes=runner.canonical_bytes({"result": True}),
        projection_attestation_bytes=runner.canonical_bytes({"projection": True}),
    )
    runner._validate_lock_payload(value)
    changed = json.loads(json.dumps(value))
    changed[binding_key][field] = replacement
    with pytest.raises(ValueError):
        runner._validate_lock_payload(changed)


def test_aggregate_guard_rejects_bytes_identifiers_and_paths() -> None:
    for value in (
        {"semantic_bytes": "x"},
        {"path": "secret"},
        {"safe": b"secret"},
    ):
        with pytest.raises(ValueError):
            runner._assert_aggregate_safe(value)


def test_consumption_anchor_is_durable_o_excl_and_nonretryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_preanchor_reader_state", lambda *_args: None)
    monkeypatch.setattr(runner, "build_qualification_record", lambda *_args: b"qualification\n")
    token = object()
    context = SimpleNamespace(
        _issuance_token=token,
        source_commitment_sha256="1" * 64,
    )
    shared_reader = object()
    authorization = SimpleNamespace(
        reader=shared_reader,
        qualification_bytes=b"qualification\n",
        qualification_sha256=runner.sha256_bytes(b"qualification\n"),
        qualification_size_bytes=len(b"qualification\n"),
        authorization_sha256="2" * 64,
        authorization_size_bytes=10,
        control_binding_commitment_sha256="3" * 64,
        anchor_parent_identity=runner._identity(os.lstat(runner.C10_ANCHOR.parent)),
    )
    qualification = SimpleNamespace(
        reader=shared_reader,
        control_binding_commitment_sha256="3" * 64,
    )
    seal = runner.consume_authorization(context, authorization, qualification)
    assert runner.C10_ANCHOR.read_bytes() == seal.bytes_value
    assert seal.sha256 == runner.sha256_bytes(seal.bytes_value)
    assert os.stat(runner.C10_ANCHOR).st_mode & 0o777 == 0o400
    with pytest.raises(PermissionError):
        runner.consume_authorization(context, authorization, qualification)


def test_consumption_rejects_anchor_parent_swap_before_o_excl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_preanchor_reader_state", lambda *_args: None)
    monkeypatch.setattr(runner, "build_qualification_record", lambda *_args: b"qualification\n")
    original_parent = runner._identity(os.lstat(runner.C10_ANCHOR.parent))
    moved = tree["root"] / "research-original"
    runner.C10_ANCHOR.parent.rename(moved)
    runner.C10_ANCHOR.parent.mkdir()
    context = SimpleNamespace(
        _issuance_token=object(), source_commitment_sha256="1" * 64
    )
    shared_reader = object()
    authorization = SimpleNamespace(
        reader=shared_reader,
        qualification_bytes=b"qualification\n",
        qualification_sha256=runner.sha256_bytes(b"qualification\n"),
        qualification_size_bytes=len(b"qualification\n"),
        authorization_sha256="2" * 64,
        authorization_size_bytes=10,
        control_binding_commitment_sha256="3" * 64,
        anchor_parent_identity=original_parent,
    )
    qualification = SimpleNamespace(
        reader=shared_reader, control_binding_commitment_sha256="3" * 64
    )
    with pytest.raises(PermissionError):
        runner.consume_authorization(context, authorization, qualification)
    assert not runner.C10_ANCHOR.exists()


def test_final_anchor_revalidation_failure_is_consumed_and_retry_blocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_preanchor_reader_state", lambda *_args: None)
    monkeypatch.setattr(runner, "build_qualification_record", lambda *_args: b"qualification\n")
    monkeypatch.setattr(
        runner,
        "revalidate_anchor",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic final revalidation")),
    )
    token = object()
    context = SimpleNamespace(
        _issuance_token=token, source_commitment_sha256="1" * 64
    )
    shared_reader = object()
    authorization = SimpleNamespace(
        reader=shared_reader,
        qualification_bytes=b"qualification\n",
        qualification_sha256=runner.sha256_bytes(b"qualification\n"),
        qualification_size_bytes=len(b"qualification\n"),
        authorization_sha256="2" * 64,
        authorization_size_bytes=10,
        control_binding_commitment_sha256="3" * 64,
        anchor_parent_identity=runner._identity(os.lstat(runner.C10_ANCHOR.parent)),
    )
    qualification = SimpleNamespace(
        reader=shared_reader, control_binding_commitment_sha256="3" * 64
    )
    with pytest.raises(runner.AnchorCreationFailure):
        runner.consume_authorization(context, authorization, qualification)
    assert runner.C10_ANCHOR.exists()
    with pytest.raises(PermissionError):
        runner.consume_authorization(context, authorization, qualification)


def test_post_o_excl_failure_publishes_terminal_truth_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_count = len(runner.FIXED_SOURCE_BINDINGS) + 2
    source_entries = [
        _synthetic_read_entry(index, f"SOURCE_{index}") for index in range(source_count)
    ]
    entries = list(source_entries)
    entries.extend(
        [
            _synthetic_read_entry(source_count, "CANDIDATE10_EXTERNAL_AUTHORIZATION"),
            _synthetic_read_entry(source_count + 1, "CANDIDATE10_QUALIFICATION_RECORD"),
        ]
    )
    start = source_count + 2
    for offset, purpose in enumerate(runner._expected_control_purposes(), start=start):
        entries.append(_synthetic_read_entry(offset, purpose))
    purposes = [str(entry["purpose"]) for entry in entries]
    reader = runner.GuardedReader(
        entries=entries,
        raw_fd_opened_purposes=purposes,
        verified_inode_purposes=purposes,
        read_started_purposes=purposes,
    )
    context = SimpleNamespace(
        source_entries=source_entries,
        source_commitment_sha256="1" * 64,
    )
    authorization = SimpleNamespace()
    qualification = SimpleNamespace(
        reader=reader, control_binding_commitment_sha256="2" * 64
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "consume_authorization",
        lambda *_args: (_ for _ in ()).throw(
            runner.AnchorCreationFailure("synthetic post-O_EXCL failure")
        ),
    )
    payloads = runner.build_payloads(context, authorization, qualification)
    assert set(payloads) == {
        "full_replay_preflight_read_ledger.json",
        "full_replay_preflight_terminal.json",
    }
    ledger = runner._decode(
        payloads["full_replay_preflight_read_ledger.json"], name="ledger"
    )
    terminal = runner._decode(
        payloads["full_replay_preflight_terminal.json"], name="terminal"
    )
    assert ledger["assertions"]["anchor_durable_entry_created_or_partial"] is True
    assert ledger["assertions"]["anchor_consumed_before_manifest_open"] is True
    assert ledger["assertions"]["anchor_full_receipt_available"] is False
    assert ledger["consumption_anchor_sha256"] is None
    assert terminal["retry_authorized"] is False


def test_atomic_publication_no_clobber_and_exact_owned_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    payloads = {
        "full_replay_preflight_lock.json": runner.canonical_bytes({"lock": True}),
        "full_replay_preflight_read_ledger.json": runner.canonical_bytes({"ledger": True}),
        "full_replay_preflight_result.json": runner.canonical_bytes({"result": True}),
        "projection_receipt_attestation.json": runner.canonical_bytes(
            {"projection": True}
        ),
    }
    runner.publish_atomically(runner.OUTPUT_ROOT, payloads, freshness=freshness)
    assert sorted(path.name for path in runner.OUTPUT_ROOT.iterdir()) == sorted(payloads)
    for name, data in payloads.items():
        assert (runner.OUTPUT_ROOT / name).read_bytes() == data
        assert os.stat(runner.OUTPUT_ROOT / name).st_mode & 0o777 == 0o400
    with pytest.raises((FileExistsError, RuntimeError)):
        runner.publish_atomically(runner.OUTPUT_ROOT, payloads, freshness=freshness)


def test_atomic_publication_rejects_partial_or_unknown_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    with pytest.raises(ValueError):
        runner.publish_atomically(
            runner.OUTPUT_ROOT,
            {"unexpected.json": b"{}\n"},
            freshness=freshness,
        )


@pytest.mark.parametrize("residue_kind", ["staging", "quarantine"])
def test_preanchor_freshness_rejects_publication_residue(
    residue_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    residue = runner.OUTPUT_ROOT.parent / (
        f".{runner.OUTPUT_ROOT.name}.{residue_kind}-foreign"
    )
    residue.mkdir()
    marker = residue / "foreign.txt"
    marker.write_text("owned elsewhere", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.prevalidate_fresh_output()
    assert marker.read_text(encoding="utf-8") == "owned elsewhere"
    assert not runner.C10_ANCHOR.exists()
    assert not runner.OUTPUT_ROOT.exists()


def _publication_payloads() -> dict[str, bytes]:
    return {
        "full_replay_preflight_lock.json": runner.canonical_bytes({"lock": True}),
        "full_replay_preflight_read_ledger.json": runner.canonical_bytes({"ledger": True}),
        "full_replay_preflight_result.json": runner.canonical_bytes({"result": True}),
        "projection_receipt_attestation.json": runner.canonical_bytes(
            {"projection": True}
        ),
    }


def test_output_parent_swap_after_freshness_rejects_without_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    parent = runner.OUTPUT_ROOT.parent
    moved = parent.with_name(parent.name + "-old")
    parent.rename(moved)
    parent.mkdir()
    with pytest.raises(RuntimeError):
        runner.publish_atomically(
            runner.OUTPUT_ROOT, _publication_payloads(), freshness=freshness
        )
    assert not runner.OUTPUT_ROOT.exists()


def test_unowned_staging_collision_is_never_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    payloads = _publication_payloads()
    digest = runner.sha256_bytes(
        runner.canonical_bytes(
            {name: runner.sha256_bytes(data) for name, data in sorted(payloads.items())}
        )
    )
    staging = runner.OUTPUT_ROOT.parent / (
        f".{runner.OUTPUT_ROOT.name}.staging-{digest[:24]}"
    )
    staging.mkdir()
    foreign = staging / "foreign.txt"
    foreign.write_text("owned elsewhere", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.publish_atomically(runner.OUTPUT_ROOT, payloads, freshness=freshness)
    assert foreign.read_text(encoding="utf-8") == "owned elsewhere"
    assert not runner.OUTPUT_ROOT.exists()


def test_failed_stage_never_unlinks_a_same_uid_leaf_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    payloads = _publication_payloads()
    digest = runner.sha256_bytes(
        runner.canonical_bytes(
            {name: runner.sha256_bytes(data) for name, data in sorted(payloads.items())}
        )
    )
    staging_name = f".{runner.OUTPUT_ROOT.name}.staging-{digest[:24]}"
    replacement_name = sorted(payloads)[0]
    original_verify = runner._verify_owned_directory
    swapped = False

    def swap_leaf_then_fail(
        parent_fd: int,
        *,
        name: str,
        directory_identity: tuple[int, int],
        leaf_identities: dict[str, tuple[int, int]],
        payloads: dict[str, bytes],
    ) -> None:
        nonlocal swapped
        if name == staging_name and not swapped:
            swapped = True
            stage_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=parent_fd,
            )
            try:
                os.fchmod(stage_fd, 0o700)
                os.rename(
                    replacement_name,
                    replacement_name + ".owned-hidden",
                    src_dir_fd=stage_fd,
                    dst_dir_fd=stage_fd,
                )
                foreign_fd = os.open(
                    replacement_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o400,
                    dir_fd=stage_fd,
                )
                try:
                    os.write(foreign_fd, b"foreign replacement")
                    os.fsync(foreign_fd)
                finally:
                    os.close(foreign_fd)
            finally:
                os.close(stage_fd)
            raise RuntimeError("synthetic post-check leaf replacement")
        original_verify(
            parent_fd,
            name=name,
            directory_identity=directory_identity,
            leaf_identities=leaf_identities,
            payloads=payloads,
        )

    monkeypatch.setattr(runner, "_verify_owned_directory", swap_leaf_then_fail)
    with pytest.raises(RuntimeError):
        runner.publish_atomically(
            runner.OUTPUT_ROOT, payloads, freshness=freshness
        )

    stage = runner.OUTPUT_ROOT.parent / staging_name
    assert swapped is True
    assert (stage / replacement_name).read_bytes() == b"foreign replacement"
    assert (stage / (replacement_name + ".owned-hidden")).is_file()
    assert not runner.OUTPUT_ROOT.exists()
    with pytest.raises(FileExistsError):
        runner.prevalidate_fresh_output()


def test_destination_race_after_freshness_is_not_clobbered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    runner.OUTPUT_ROOT.mkdir()
    marker = runner.OUTPUT_ROOT / "foreign.txt"
    marker.write_text("foreign", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.publish_atomically(
            runner.OUTPUT_ROOT, _publication_payloads(), freshness=freshness
        )
    assert marker.read_text(encoding="utf-8") == "foreign"


def test_post_rename_foreign_replacement_is_never_moved_or_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    original_verify = runner._verify_owned_directory
    hidden_owned = runner.OUTPUT_ROOT.parent / (
        f".{runner.OUTPUT_ROOT.name}.owned-hidden"
    )
    swapped = False

    def replace_final_before_verification(
        parent_fd: int,
        *,
        name: str,
        directory_identity: tuple[int, int],
        leaf_identities: dict[str, tuple[int, int]],
        payloads: dict[str, bytes],
    ) -> None:
        nonlocal swapped
        if name == runner.OUTPUT_ROOT.name and not swapped:
            swapped = True
            os.rename(
                name,
                hidden_owned.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            foreign_parent_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=parent_fd,
            )
            try:
                foreign_fd = os.open(
                    "foreign.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o400,
                    dir_fd=foreign_parent_fd,
                )
                try:
                    os.write(foreign_fd, b"foreign")
                    os.fsync(foreign_fd)
                finally:
                    os.close(foreign_fd)
            finally:
                os.close(foreign_parent_fd)
        original_verify(
            parent_fd,
            name=name,
            directory_identity=directory_identity,
            leaf_identities=leaf_identities,
            payloads=payloads,
        )

    monkeypatch.setattr(
        runner, "_verify_owned_directory", replace_final_before_verification
    )
    with pytest.raises(RuntimeError):
        runner.publish_atomically(
            runner.OUTPUT_ROOT, _publication_payloads(), freshness=freshness
        )
    assert swapped is True
    assert (runner.OUTPUT_ROOT / "foreign.txt").read_bytes() == b"foreign"
    assert hidden_owned.is_dir()
    assert not any(
        path.name.startswith(f".{runner.OUTPUT_ROOT.name}.quarantine-")
        for path in runner.OUTPUT_ROOT.parent.iterdir()
    )


def test_runner_source_contains_no_real_mode_autorun_or_adapted_cache_write() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "mutually_exclusive_group(required=True)" in text
    assert "os.O_EXCL" in text
    assert "before_first_read" in text
    assert "projection_receipts_are_not_adapted_cache" in text
    assert "adapted_cache_published\": False" in text
    assert "train_label_projection_attestation.json" in text
    assert "bend_line_development_catalog_v2_20260830.json" in text
    assert "if __name__ == \"__main__\"" in text
