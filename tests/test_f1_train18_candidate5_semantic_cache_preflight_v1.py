from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend import f1_train18_candidate5_conservative_deletion_v1 as runtime
from scripts import preflight_f1_train18_candidate5_semantic_cache_v1 as preflight


ROOT = Path(__file__).resolve().parents[1]


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "runtime": preflight.sha256_bytes(preflight.RUNTIME_SOURCE.read_bytes()),
        "preflight": preflight.sha256_bytes(preflight.PREFLIGHT_SOURCE.read_bytes()),
        "runtime_tests": preflight.sha256_bytes(
            preflight.RUNTIME_TESTS_SOURCE.read_bytes()
        ),
        "preflight_tests": preflight.sha256_bytes(
            preflight.PREFLIGHT_TESTS_SOURCE.read_bytes()
        ),
    }


def _manifest_fixture(
    *, train_ids: tuple[str, ...], scan_by_case: dict[str, str]
) -> dict:
    records = []
    for ordinal, case_id in enumerate(train_ids):
        runtime_id = f"T{900 + ordinal:03d}"
        semantic_sha = _hash(f"semantic-{case_id}")
        pool_sha = _hash(f"pool-{case_id}")
        for pass_name in ("primary", "replay"):
            records.append(
                {
                    "case_id": case_id,
                    "runtime_case_id": runtime_id,
                    "pass": pass_name,
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
        "schema_version": 1,
        "kind": preflight.TRAIN_CACHE_MANIFEST_KIND,
        "source_only": True,
        "holdout_entry_count": 0,
        "split_sha256": preflight.SPLIT_SHA256,
        "train_allowlist_sha256": preflight.ALLOWLIST_SHA256,
        "train_set_commitment_sha256": commitment,
        "v51_scan_only_manifest_sha256": preflight.V51_SCAN_ONLY_MANIFEST_SHA256,
        "case_count": 18,
        "record_count": 36,
        "records": records,
    }


def test_candidate5_preflight_top_level_is_stdlib_only() -> None:
    tree = ast.parse(preflight.PREFLIGHT_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "scripts", "numpy", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_help_imports_no_runtime_numpy_optimizer_or_label_module() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(preflight.PREFLIGHT_SOURCE), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "f1_train18_candidate5_conservative_deletion_v1",
        "numpy",
        "optimize_f1_train18_source_only_v1",
        "evaluate_f1_against_manual_gold",
    ):
        assert forbidden not in completed.stderr


def test_source_check_only_verifies_exact_bytes_without_cache_or_label_access() -> None:
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
    payload = json.loads(completed.stdout)
    assert payload == {
        "cache_or_label_artifact_opened": False,
        "profile": runtime.PROFILE,
        "source_binding_count": 4,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def test_preloaded_controlled_module_rejects_before_any_source_read(tmp_path) -> None:
    sentinel = tmp_path / "sentinel.txt"
    code = f"""
import importlib.util, pathlib, sys, types
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_candidate5_preflight', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_candidate5_preflight'] = module
spec.loader.exec_module(module)
sys.modules['backend'] = types.ModuleType('backend')
class SentinelPath:
    def read_bytes(self):
        pathlib.Path({str(sentinel)!r}).write_text('READ')
        return b''
module.RUNTIME_SOURCE = SentinelPath()
try:
    module.bootstrap_verified_sources(
        runtime_sha256='1' * 64,
        preflight_sha256='2' * 64,
        runtime_tests_sha256='3' * 64,
        preflight_tests_sha256='4' * 64,
    )
except RuntimeError as error:
    assert 'preloaded module' in str(error)
else:
    raise AssertionError('preloaded module was accepted')
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


def test_verified_loader_executes_the_already_hashed_runtime_bytes(tmp_path) -> None:
    runtime_path = tmp_path / "runtime.py"
    preflight_path = tmp_path / "preflight.py"
    runtime_tests = tmp_path / "runtime_tests.py"
    preflight_tests = tmp_path / "preflight_tests.py"
    sentinel = tmp_path / "sentinel.txt"
    safe_runtime = b"PROFILE = 'SAFE'\n"
    runtime_path.write_bytes(safe_runtime)
    preflight_path.write_bytes(b"PREFLIGHT = True\n")
    runtime_tests.write_bytes(b"RUNTIME_TESTS = True\n")
    preflight_tests.write_bytes(b"PREFLIGHT_TESTS = True\n")
    code = f"""
import builtins, hashlib, importlib.util, pathlib, sys
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_candidate5_preflight', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_candidate5_preflight'] = module
spec.loader.exec_module(module)
module.ROOT = pathlib.Path({str(tmp_path)!r})
module.RUNTIME_SOURCE = pathlib.Path({str(runtime_path)!r})
module.PREFLIGHT_SOURCE = pathlib.Path({str(preflight_path)!r})
module.RUNTIME_TESTS_SOURCE = pathlib.Path({str(runtime_tests)!r})
module.PREFLIGHT_TESTS_SOURCE = pathlib.Path({str(preflight_tests)!r})
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
tampered = "from pathlib import Path\\nPath(" + repr({str(sentinel)!r}) + ").write_text('EXECUTED')\\n"
original_compile = builtins.compile
def swap(source, filename, mode, *args, **kwargs):
    module.RUNTIME_SOURCE.write_text(tampered)
    return original_compile(source, filename, mode, *args, **kwargs)
builtins.compile = swap
runtime, _bindings, _entries = module.bootstrap_verified_sources(
    runtime_sha256=hashlib.sha256({safe_runtime!r}).hexdigest(),
    preflight_sha256=hashlib.sha256(module.PREFLIGHT_SOURCE.read_bytes()).hexdigest(),
    runtime_tests_sha256=hashlib.sha256(module.RUNTIME_TESTS_SOURCE.read_bytes()).hexdigest(),
    preflight_tests_sha256=hashlib.sha256(module.PREFLIGHT_TESTS_SOURCE.read_bytes()).hexdigest(),
)
assert runtime.PROFILE == 'SAFE'
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


def test_tampered_runtime_hash_rejects_without_executing_source(tmp_path) -> None:
    sentinel = tmp_path / "sentinel.txt"
    malicious = tmp_path / "runtime.py"
    malicious.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('EXECUTED')\n"
    )
    harmless = tmp_path / "harmless.py"
    harmless.write_text("VALUE = True\n")
    code = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_candidate5_preflight', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_candidate5_preflight'] = module
spec.loader.exec_module(module)
module.ROOT = pathlib.Path({str(tmp_path)!r})
module.RUNTIME_SOURCE = pathlib.Path({str(malicious)!r})
module.PREFLIGHT_SOURCE = pathlib.Path({str(harmless)!r})
module.RUNTIME_TESTS_SOURCE = pathlib.Path({str(harmless)!r})
module.PREFLIGHT_TESTS_SOURCE = pathlib.Path({str(harmless)!r})
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
try:
    module.bootstrap_verified_sources(
        runtime_sha256='1' * 64,
        preflight_sha256=module.sha256_bytes(module.PREFLIGHT_SOURCE.read_bytes()),
        runtime_tests_sha256=module.sha256_bytes(module.RUNTIME_TESTS_SOURCE.read_bytes()),
        preflight_tests_sha256=module.sha256_bytes(module.PREFLIGHT_TESTS_SOURCE.read_bytes()),
    )
except ValueError as error:
    assert 'hash mismatch' in str(error)
else:
    raise AssertionError('tampered runtime was accepted')
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


def test_verified_bootstrap_rejects_hardlinked_source_aliases(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    runtime_source = root / "runtime.py"
    runtime_source.write_bytes(b"PROFILE = 'SAFE'\n")
    preflight_source = root / "preflight.py"
    os.link(runtime_source, preflight_source)
    runtime_tests = root / "runtime_tests.py"
    preflight_tests = root / "preflight_tests.py"
    runtime_tests.write_bytes(b"RUNTIME_TESTS = True\n")
    preflight_tests.write_bytes(b"PREFLIGHT_TESTS = True\n")
    module = __import__(
        "scripts.preflight_f1_train18_candidate5_semantic_cache_v1",
        fromlist=["bootstrap_verified_sources"],
    )
    original_values = {
        "ROOT": module.ROOT,
        "RUNTIME_SOURCE": module.RUNTIME_SOURCE,
        "PREFLIGHT_SOURCE": module.PREFLIGHT_SOURCE,
        "RUNTIME_TESTS_SOURCE": module.RUNTIME_TESTS_SOURCE,
        "PREFLIGHT_TESTS_SOURCE": module.PREFLIGHT_TESTS_SOURCE,
        "FORBIDDEN_PRELOADED_MODULES": module.FORBIDDEN_PRELOADED_MODULES,
    }
    try:
        module.ROOT = root
        module.RUNTIME_SOURCE = runtime_source
        module.PREFLIGHT_SOURCE = preflight_source
        module.RUNTIME_TESTS_SOURCE = runtime_tests
        module.PREFLIGHT_TESTS_SOURCE = preflight_tests
        module.FORBIDDEN_PRELOADED_MODULES = ("never.preloaded",)
        with pytest.raises(ValueError, match="hard-link aliases"):
            module.bootstrap_verified_sources(
                runtime_sha256=module.sha256_bytes(runtime_source.read_bytes()),
                preflight_sha256=module.sha256_bytes(preflight_source.read_bytes()),
                runtime_tests_sha256=module.sha256_bytes(runtime_tests.read_bytes()),
                preflight_tests_sha256=module.sha256_bytes(preflight_tests.read_bytes()),
            )
    finally:
        for name, value in original_values.items():
            setattr(module, name, value)


def test_verified_bootstrap_rejects_runtime_preloaded_under_an_alias() -> None:
    hashes = _source_hashes()
    code = f"""
import importlib.util, pathlib, sys
runtime_path = pathlib.Path({str(preflight.RUNTIME_SOURCE)!r})
runtime_spec = importlib.util.spec_from_file_location('arbitrary_runtime_alias', runtime_path)
runtime_alias = importlib.util.module_from_spec(runtime_spec)
sys.modules['arbitrary_runtime_alias'] = runtime_alias
runtime_spec.loader.exec_module(runtime_alias)
preflight_path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
preflight_spec = importlib.util.spec_from_file_location('isolated_candidate5_preflight', preflight_path)
module = importlib.util.module_from_spec(preflight_spec)
sys.modules['isolated_candidate5_preflight'] = module
preflight_spec.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
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
    raise AssertionError('runtime alias was accepted')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_verified_bootstrap_rejects_symlink_loaded_alias_before_source_read(
    tmp_path,
) -> None:
    hashes = _source_hashes()
    alias_path = tmp_path / "innocent_alias.py"
    alias_path.symlink_to(preflight.RUNTIME_SOURCE)
    sentinel = tmp_path / "controlled-source-read.txt"
    code = f"""
import importlib.util, pathlib, sys
alias_path = pathlib.Path({str(alias_path)!r})
runtime_spec = importlib.util.spec_from_file_location('arbitrary_symlink_alias', alias_path)
runtime_alias = importlib.util.module_from_spec(runtime_spec)
sys.modules['arbitrary_symlink_alias'] = runtime_alias
runtime_spec.loader.exec_module(runtime_alias)
preflight_path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
preflight_spec = importlib.util.spec_from_file_location('isolated_candidate5_preflight', preflight_path)
module = importlib.util.module_from_spec(preflight_spec)
sys.modules['isolated_candidate5_preflight'] = module
preflight_spec.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_read(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('CONTROLLED_SOURCE_READ')
    raise AssertionError('source read occurred before alias rejection')
module._read_verified_source_once = forbidden_read
try:
    module.bootstrap_verified_sources(
        runtime_sha256={hashes['runtime']!r},
        preflight_sha256={hashes['preflight']!r},
        runtime_tests_sha256={hashes['runtime_tests']!r},
        preflight_tests_sha256={hashes['preflight_tests']!r},
    )
except RuntimeError as error:
    assert 'resolved module alias' in str(error)
else:
    raise AssertionError('symlink-loaded runtime alias was accepted')
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


def test_guarded_reader_is_fail_closed_for_path_hash_size_and_conflict(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    value = root / "value.json"
    value.write_bytes(b"{}\n")
    digest = preflight.sha256_bytes(value.read_bytes())
    reader = preflight.GuardedReader()

    with pytest.raises(PermissionError, match="read denied"):
        reader.read_bytes(value, role="VALUE")
    reader.authorize(value, role="VALUE", sha256=digest, size_bytes=3)
    assert reader.read_bytes(value, role="VALUE") == b"{}\n"
    with pytest.raises(ValueError, match="conflicting"):
        reader.authorize(value, role="OTHER", sha256=digest, size_bytes=3)
    value.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.read_bytes(value, role="VALUE")


def test_guarded_reader_rejects_label_projection_and_symlink_capabilities(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    projection = root / "train_label_projection.json"
    with pytest.raises(PermissionError, match="forbids path"):
        preflight.GuardedReader().authorize(
            projection, role="NOT_A_LABEL", sha256=_hash("projection")
        )

    target = root / "target.json"
    target.write_text("{}\n")
    symlink = root / "link.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        preflight.GuardedReader().authorize(
            symlink,
            role="VALUE",
            sha256=preflight.sha256_bytes(target.read_bytes()),
        )

    dangling_parent = root / "dangling-parent"
    dangling_parent.symlink_to(root / "absent-directory", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        preflight.GuardedReader().authorize(
            dangling_parent / "renamed-input.json",
            role="VALUE",
            sha256=_hash("unreachable"),
        )


def test_guarded_reader_rejects_hard_links_across_bootstrap_and_authorized_inputs(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    source = root / "verified_source.py"
    source.write_bytes(b"SOURCE = True\n")
    alias = root / "renamed_label_like_input.json"
    os.link(source, alias)
    metadata = os.lstat(source)
    bootstrap_entry = {
        "case_id": None,
        "path": source.name,
        "role": "VERIFIED_SOURCE",
        "sha256": preflight.sha256_bytes(source.read_bytes()),
        "size_bytes": source.stat().st_size,
        "device_id": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
    }
    reader = preflight.GuardedReader(entries=[bootstrap_entry])
    with pytest.raises(ValueError, match="hard-link alias"):
        reader.authorize(
            alias,
            role="TRAIN_V51_SEMANTIC",
            sha256=preflight.sha256_bytes(alias.read_bytes()),
        )


def test_guarded_reader_rejects_symlink_and_regular_file_swaps_before_fd_read(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    value = root / "value.json"
    original = b"original\n"
    value.write_bytes(original)
    reader = preflight.GuardedReader()
    reader.authorize(
        value,
        role="VALUE",
        sha256=preflight.sha256_bytes(original),
        size_bytes=len(original),
    )

    secret = root / "secret.json"
    secret.write_bytes(b"must-not-open\n")
    value.unlink()
    value.symlink_to(secret)
    with pytest.raises(ValueError, match="symlink"):
        reader.read_bytes(value, role="VALUE")

    value.unlink()
    value.write_bytes(original)
    reader = preflight.GuardedReader()
    reader.authorize(
        value,
        role="VALUE",
        sha256=preflight.sha256_bytes(original),
        size_bytes=len(original),
    )
    replacement = root / "replacement.json"
    replacement.write_bytes(b"replacement\n")
    original_open = preflight.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and Path(path).name == value.name
            and kwargs.get("dir_fd") is not None
        ):
            swapped = True
            value.unlink()
            replacement.rename(value)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(preflight.os, "open", swap_before_open)
    with pytest.raises(ValueError, match="identity changed during open"):
        reader.read_bytes(value, role="VALUE")

    monkeypatch.setattr(preflight.os, "open", original_open)
    value.unlink()
    value.write_bytes(original)
    reader = preflight.GuardedReader()
    reader.authorize(
        value,
        role="VALUE",
        sha256=preflight.sha256_bytes(original),
        size_bytes=len(original),
    )
    swapped = False

    def symlink_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and Path(path).name == value.name
            and kwargs.get("dir_fd") is not None
        ):
            swapped = True
            value.unlink()
            value.symlink_to(secret)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(preflight.os, "open", symlink_before_open)
    with pytest.raises(OSError) as captured:
        reader.read_bytes(value, role="VALUE")
    assert captured.value.errno in (errno.ELOOP, errno.ENOTDIR)


def test_guarded_reader_rejects_ancestor_symlink_swap_before_child_open(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    source_parent = root / "source-parent"
    source_parent.mkdir(parents=True)
    monkeypatch.setattr(preflight, "ROOT", root)
    value = source_parent / "value.json"
    expected = b"expected\n"
    value.write_bytes(expected)
    reader = preflight.GuardedReader()
    reader.authorize(
        value,
        role="VALUE",
        sha256=preflight.sha256_bytes(expected),
        size_bytes=len(expected),
    )
    foreign_parent = root / "foreign-parent"
    foreign_parent.mkdir()
    (foreign_parent / value.name).write_bytes(b"must-not-read\n")
    moved_parent = root / "moved-source-parent"
    original_open = preflight.os.open
    swapped = False

    def swap_parent_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == source_parent:
            swapped = True
            source_parent.rename(moved_parent)
            source_parent.symlink_to(foreign_parent, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(preflight.os, "open", swap_parent_before_open)
    with pytest.raises(OSError) as captured:
        reader.read_bytes(value, role="VALUE")
    assert captured.value.errno in (errno.ELOOP, errno.ENOTDIR)


def test_manifest_validation_accepts_nonordinal_bijective_runtime_mapping() -> None:
    train_ids = tuple(f"A30-X{index:02d}" for index in range(18))
    scan_by_case = {case_id: _hash(f"scan-{case_id}") for case_id in train_ids}
    manifest = _manifest_fixture(train_ids=train_ids, scan_by_case=scan_by_case)
    commitment = manifest["train_set_commitment_sha256"]
    records = preflight._validate_cache_manifest(
        manifest,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=commitment,
    )
    assert len(records) == 36
    assert records[(train_ids[0], "primary")]["runtime_case_id"] == "T900"


def test_manifest_pool_fields_are_inert_and_never_resolved(monkeypatch) -> None:
    train_ids = tuple(f"A30-X{index:02d}" for index in range(18))
    scan_by_case = {case_id: _hash(f"scan-{case_id}") for case_id in train_ids}
    manifest = _manifest_fixture(train_ids=train_ids, scan_by_case=scan_by_case)
    original = preflight._reject_symlink_chain

    def reject_any_pool_resolution(path, *, role, require_exists=False):  # type: ignore[no-untyped-def]
        assert "POOL" not in role
        return original(path, role=role, require_exists=require_exists)

    monkeypatch.setattr(preflight, "_reject_symlink_chain", reject_any_pool_resolution)
    records = preflight._validate_cache_manifest(
        manifest,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=manifest["train_set_commitment_sha256"],
    )
    assert len(records) == 36


@pytest.mark.parametrize(
    "mutation",
    [
        "nineteenth",
        "wrong_order",
        "wrong_scan",
        "extra_envelope",
        "alias_replay",
        "alias_cross_case",
    ],
)
def test_manifest_validation_rejects_membership_order_and_scan_tamper(
    mutation: str,
) -> None:
    train_ids = tuple(f"A30-X{index:02d}" for index in range(18))
    scan_by_case = {case_id: _hash(f"scan-{case_id}") for case_id in train_ids}
    manifest = _manifest_fixture(train_ids=train_ids, scan_by_case=scan_by_case)
    if mutation == "nineteenth":
        manifest["records"].append(dict(manifest["records"][-1]))
        manifest["record_count"] = 37
    elif mutation == "wrong_order":
        manifest["records"][0], manifest["records"][1] = (
            manifest["records"][1],
            manifest["records"][0],
        )
    elif mutation == "wrong_scan":
        manifest["records"][0]["scan_sha256"] = _hash("wrong-scan")
    elif mutation == "extra_envelope":
        manifest["unexpected"] = False
    elif mutation == "alias_replay":
        manifest["records"][1]["semantic_path"] = manifest["records"][0][
            "semantic_path"
        ]
    else:
        manifest["records"][2]["semantic_path"] = manifest["records"][0][
            "semantic_path"
        ]
    with pytest.raises((ValueError, PermissionError)):
        preflight._validate_cache_manifest(
            manifest,
            train_ids=train_ids,
            scan_by_case=scan_by_case,
            train_commitment=_manifest_fixture(
                train_ids=train_ids, scan_by_case=scan_by_case
            )["train_set_commitment_sha256"],
        )


def test_cache_attestation_requires_exact_evidence_and_zero_holdout(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    evidence = {}
    evidence_paths = {}
    for prefix in ("builder", "validator", "staged_validator_pass", "access_evidence"):
        path = root / f"{prefix}.json"
        data = preflight.canonical_bytes({"kind": prefix})
        path.write_bytes(data)
        binding = {
            "path": path.name,
            "sha256": preflight.sha256_bytes(data),
            "size_bytes": len(data),
        }
        if prefix == "staged_validator_pass":
            binding["kind"] = preflight.CUSTODY_VALIDATION_KIND
        evidence[f"{prefix}_binding"] = binding
        evidence_paths[prefix] = path
    monkeypatch.setattr(preflight, "CACHE_EVIDENCE_PATHS", evidence_paths)
    monkeypatch.setattr(
        preflight,
        "CUSTODY_PASS_SHA256",
        evidence["staged_validator_pass_binding"]["sha256"],
    )
    manifest_path = root / "manifest.json"
    manifest_data = preflight.canonical_bytes({"kind": "manifest"})
    manifest_path.write_bytes(manifest_data)
    monkeypatch.setattr(preflight, "CACHE_MANIFEST", manifest_path)
    commitment = _hash("train-commitment")
    attestation = {
        "schema_version": 1,
        "kind": preflight.TRAIN_CACHE_ATTESTATION_KIND,
        "canonical_json": True,
        "case_count": 18,
        "record_count": 36,
        "split_sha256": preflight.SPLIT_SHA256,
        "train_allowlist_sha256": preflight.ALLOWLIST_SHA256,
        "train_set_commitment_sha256": commitment,
        "blinded_complement_commitment_sha256": _hash("blinded-complement"),
        "train_exact_match": True,
        "train_holdout_intersection_count": 0,
        "holdout_entry_count": 0,
        "holdout_hash_count": 0,
        "holdout_status_count": 0,
        "optimizer_holdout_access_count": 0,
        "independent_validation_passed": True,
        "manifest_binding": {
            "kind": preflight.TRAIN_CACHE_MANIFEST_KIND,
            "path": manifest_path.name,
            "sha256": preflight.sha256_bytes(manifest_data),
            "size_bytes": len(manifest_data),
        },
        **evidence,
    }
    reader = preflight.GuardedReader()
    path, sha, size, checks = preflight._validate_cache_attestation(
        attestation, train_commitment=commitment, reader=reader
    )
    assert path == manifest_path.resolve()
    assert sha == preflight.sha256_bytes(manifest_data)
    assert size == len(manifest_data)
    assert all(checks.values())
    assert len(reader.entries) == 4

    broken = dict(attestation)
    broken["optimizer_holdout_access_count"] = 1
    with pytest.raises(ValueError, match="boundary checks"):
        preflight._validate_cache_attestation(
            broken,
            train_commitment=commitment,
            reader=preflight.GuardedReader(),
        )

    extra_envelope = dict(attestation, unexpected=False)
    with pytest.raises(ValueError, match="exact-key schema"):
        preflight._validate_cache_attestation(
            extra_envelope,
            train_commitment=commitment,
            reader=preflight.GuardedReader(),
        )

    extra_binding = dict(attestation)
    extra_binding["validator_binding"] = {
        **attestation["validator_binding"],
        "unexpected": False,
    }
    with pytest.raises(ValueError, match="binding exact-key schema"):
        preflight._validate_cache_attestation(
            extra_binding,
            train_commitment=commitment,
            reader=preflight.GuardedReader(),
        )

    renamed = root / "renamed-safe-looking.json"
    renamed_data = preflight.canonical_bytes({"kind": "renamed"})
    renamed.write_bytes(renamed_data)
    alternate_path = dict(attestation)
    alternate_path["builder_binding"] = {
        "path": renamed.name,
        "sha256": preflight.sha256_bytes(renamed_data),
        "size_bytes": len(renamed_data),
    }
    alternate_reader = preflight.GuardedReader()
    with pytest.raises(PermissionError, match="exact custody allowlist"):
        preflight._validate_cache_attestation(
            alternate_path,
            train_commitment=commitment,
            reader=alternate_reader,
        )
    assert alternate_reader.entries == []


@pytest.mark.parametrize(
    "payload",
    [
        {"prediction_count": 1},
        {"nested": {"deleted_prediction_count": 2}},
        {"rows": [{"selector_score": None}]},
    ],
)
def test_aggregate_preflight_outputs_reject_source_volume_details(payload: dict) -> None:
    with pytest.raises(ValueError, match="forbidden keys"):
        preflight._assert_aggregate_safe(payload)


def test_atomic_publication_is_all_or_nothing(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "output"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    payloads = {
        "preflight_lock.json": b"lock\n",
        "preflight_read_ledger.json": b"ledger\n",
        "preflight_report.json": b"report\n",
    }
    preflight._publish_atomically(target, payloads)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == payloads

    failed_target = root / "failed-output"
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", failed_target)
    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        preflight._publish_atomically(
            failed_target,
            payloads,
            after_stage_write=lambda _path: (_ for _ in ()).throw(
                RuntimeError("synthetic publication failure")
            ),
        )
    assert not failed_target.exists()
    assert not tuple(root.glob(f".{failed_target.name}.staging-*"))
    failed_quarantines = tuple(root.glob(f".{failed_target.name}.quarantine-*"))
    assert len(failed_quarantines) == 1
    assert {path.name: path.read_bytes() for path in failed_quarantines[0].iterdir()} == payloads

    post_rename_target = root / "post-rename-failure"
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", post_rename_target)
    original_rename = preflight._renameat_directory_noreplace

    def rename_then_fail(parent_fd, source, destination) -> None:
        original_rename(parent_fd, source, destination)
        raise RuntimeError("synthetic post-rename failure")

    monkeypatch.setattr(preflight, "_renameat_directory_noreplace", rename_then_fail)
    with pytest.raises(RuntimeError, match="synthetic post-rename failure"):
        preflight._publish_atomically(post_rename_target, payloads)
    assert not post_rename_target.exists()
    assert not tuple(root.glob(f".{post_rename_target.name}.staging-*"))
    post_quarantines = tuple(root.glob(f".{post_rename_target.name}.quarantine-*"))
    assert len(post_quarantines) == 1
    assert {path.name: path.read_bytes() for path in post_quarantines[0].iterdir()} == payloads


def test_publication_never_clobbers_a_raced_destination(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "output"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    payloads = {
        "preflight_lock.json": b"lock\n",
        "preflight_read_ledger.json": b"ledger\n",
        "preflight_report.json": b"report\n",
    }

    def race_destination(_staging: Path) -> None:
        target.mkdir()
        (target / "foreign-sentinel.txt").write_text("FOREIGN")

    with pytest.raises(FileExistsError, match="raced into existence"):
        preflight._publish_atomically(
            target, payloads, after_stage_write=race_destination
        )
    assert (target / "foreign-sentinel.txt").read_text() == "FOREIGN"
    assert not tuple(root.glob(f".{target.name}.staging-*"))
    assert len(tuple(root.glob(f".{target.name}.quarantine-*"))) == 1


def test_publication_rejects_output_parent_symlink_swap_without_foreign_write(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    output_parent = root / "output-parent"
    output_parent.mkdir(parents=True)
    moved_parent = root / "moved-output-parent"
    foreign_parent = root / "foreign-output-parent"
    foreign_parent.mkdir()
    target = output_parent / "output"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    payloads = {
        "preflight_lock.json": b"lock\n",
        "preflight_read_ledger.json": b"ledger\n",
        "preflight_report.json": b"report\n",
    }

    def swap_parent(_staging: Path) -> None:
        output_parent.rename(moved_parent)
        output_parent.symlink_to(foreign_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        preflight._publish_atomically(
            target, payloads, after_stage_write=swap_parent
        )
    assert not (foreign_parent / target.name).exists()
    assert not tuple(moved_parent.glob(f".{target.name}.staging-*"))


def test_publication_quarantines_a_foreign_source_swapped_at_rename(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "output"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    payloads = {
        "preflight_lock.json": b"lock\n",
        "preflight_read_ledger.json": b"ledger\n",
        "preflight_report.json": b"report\n",
    }
    original_rename = preflight._renameat_directory_noreplace
    moved_owned_name = ".attacker-moved-owned-staging"
    acted = False

    def swap_source_then_rename(parent_fd, source, destination) -> None:
        nonlocal acted
        if not acted and destination == target.name:
            acted = True
            original_rename(parent_fd, source, moved_owned_name)
            os.mkdir(source, mode=0o700, dir_fd=parent_fd)
            foreign = root / source / "foreign-sentinel.txt"
            foreign.write_text("FOREIGN")
        original_rename(parent_fd, source, destination)

    monkeypatch.setattr(
        preflight, "_renameat_directory_noreplace", swap_source_then_rename
    )
    with pytest.raises(RuntimeError, match="published directory identity mismatch"):
        preflight._publish_atomically(target, payloads)
    assert not target.exists()
    assert (root / moved_owned_name).is_dir()
    assert not tuple(root.glob(f".{target.name}.staging-*"))
    quarantines = tuple(root.glob(f".{target.name}.quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "foreign-sentinel.txt").read_text() == "FOREIGN"


def test_publication_quarantines_owned_output_if_parent_detaches_inside_rename(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    output_parent = root / "output-parent"
    output_parent.mkdir(parents=True)
    moved_parent = root / "moved-output-parent"
    foreign_parent = root / "foreign-output-parent"
    target = output_parent / "output"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    payloads = {
        "preflight_lock.json": b"lock\n",
        "preflight_read_ledger.json": b"ledger\n",
        "preflight_report.json": b"report\n",
    }
    original_rename = preflight._renameat_directory_noreplace
    acted = False

    def detach_parent_then_rename(parent_fd, source, destination) -> None:
        nonlocal acted
        if not acted and destination == target.name:
            acted = True
            output_parent.rename(moved_parent)
            foreign_parent.mkdir()
            foreign_parent.rename(output_parent)
        original_rename(parent_fd, source, destination)

    monkeypatch.setattr(
        preflight, "_renameat_directory_noreplace", detach_parent_then_rename
    )
    with pytest.raises(FileNotFoundError, match="published output path must exist"):
        preflight._publish_atomically(target, payloads)
    assert not target.exists()
    assert not (moved_parent / target.name).exists()
    assert len(tuple(moved_parent.glob(f".{target.name}.quarantine-*"))) == 1
    assert not tuple(moved_parent.glob(f".{target.name}.staging-*"))


def test_publication_rejects_stale_staging_and_output_symlink_chains(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    payloads = {
        "preflight_lock.json": b"lock\n",
        "preflight_read_ledger.json": b"ledger\n",
        "preflight_report.json": b"report\n",
    }
    target = root / "output"
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    stale = root / f".{target.name}.staging-stale"
    stale.mkdir()
    with pytest.raises(FileExistsError, match="stale staging"):
        preflight._publish_atomically(target, payloads)
    assert stale.is_dir()

    stale.rmdir()
    dangling = root / "missing-target"
    target.symlink_to(dangling)
    with pytest.raises(ValueError, match="symlink"):
        preflight._validate_output_root(target)
    target.unlink()

    real_parent = root / "real-parent"
    real_parent.mkdir()
    linked_parent = root / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_target = linked_parent / "output"
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", linked_target)
    with pytest.raises(ValueError, match="symlink"):
        preflight._validate_output_root(linked_target)

    linked_parent.unlink()
    linked_parent.symlink_to(root / "absent-output-parent", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        preflight._validate_output_root(linked_target)


def test_staging_quarantine_requires_and_preserves_the_exact_owned_inode(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "output"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    parent_fd, _identity = preflight._open_stable_directory(
        root, role="synthetic output parent"
    )
    try:
        staging_name, staging_identity = preflight._create_owned_staging(
            parent_fd, target
        )
        staging_fd = preflight._open_owned_staging(
            parent_fd, staging_name, staging_identity
        )
        try:
            preflight._write_staging_file(staging_fd, "owned.json", b"owned\n")
        finally:
            os.close(staging_fd)
        wrong_identity = (staging_identity[0], staging_identity[1] + 1)
        with pytest.raises(RuntimeError, match="unexpected inode"):
            preflight._quarantine_directory(
                parent_fd,
                staging_name,
                target,
                expected_identity=wrong_identity,
            )
        assert (root / staging_name / "owned.json").read_bytes() == b"owned\n"
        quarantine_name, quarantined_identity = preflight._quarantine_directory(
            parent_fd,
            staging_name,
            target,
            expected_identity=staging_identity,
        )
        assert quarantined_identity == staging_identity
        assert (root / quarantine_name / "owned.json").read_bytes() == b"owned\n"
        preflight._assert_no_staging_residue(parent_fd, target)
        with pytest.raises(FileExistsError, match="prior publication quarantine"):
            preflight._assert_no_prior_quarantine(parent_fd, target)
    finally:
        os.close(parent_fd)


def test_output_root_rejects_parent_traversal_even_when_normalized_path_matches(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "output"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", target)
    traversal = root / "unused" / ".." / target.name
    with pytest.raises(ValueError, match="parent traversal"):
        preflight._validate_output_root(traversal)


def test_runtime_preflight_result_is_aggregate_and_exercises_all_rule_forms() -> None:
    semantic = runtime.canonical_bytes(
        {
            "schema_version": 1,
            "kind": "synthetic_v51",
            "source_only": True,
            "research_only": True,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "expected_count_used": False,
            "candidate_count": 3,
            "audit": {"source_partition_count": 2},
            "predictions": [
                {
                    "bend_id": "base",
                    "line_start": [0, 0, 0],
                    "line_end": [1, 0, 0],
                    "source_kind": "plane_lattice",
                    "selection_route": "v16",
                    "source_hypothesis_id": "h-base",
                },
                {
                    "bend_id": "s1",
                    "line_start": [0, 1, 0],
                    "line_end": [1, 1, 0],
                    "source_kind": "scan_mesh_adjacency",
                    "selection_route": "v41",
                    "source_hypothesis_id": "h-s1",
                    "selector_score": None,
                },
                {
                    "bend_id": "s2",
                    "line_start": [0, 2, 0],
                    "line_end": [1, 2, 0],
                    "source_kind": "cylindrical_standard",
                    "selection_route": "v41",
                    "source_hypothesis_id": "h-s2",
                    "selector_score": 0.25,
                },
            ],
        }
    )
    result = runtime.preflight_candidate5_semantic_bytes(
        semantic, v51_semantic_sha256=runtime.sha256_bytes(semantic)
    )
    assert result["all_frozen_rule_forms_exercised"] is True
    assert result["feature_extraction_complete"] is True
    assert result["generic_source_count_and_audit_metadata_ignored"] is True
    assert result["prediction_output_projection_sanitized"] is True
    preflight._assert_aggregate_safe(result)


def test_model_contract_requires_preflight_source_report_lock_ledger_and_tests() -> None:
    for field in (
        "candidate5_preflight_source_sha256",
        "candidate5_preflight_tests_sha256",
        "candidate5_preflight_report_sha256",
        "candidate5_preflight_lock_sha256",
        "candidate5_preflight_read_ledger_sha256",
    ):
        assert field in runtime.BINDING_KEYS


def test_no_real_candidate5_preflight_prereg_or_fit_artifact_exists() -> None:
    assert not preflight.OUTPUT_ROOT.exists()
    assert not (
        ROOT / "research/f1_aug30_train18_candidate5_preregistration_v1.json"
    ).exists()
