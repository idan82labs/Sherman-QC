from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.pwts_v1.config import canonical_json_bytes, get_config_copy
from research.pwts_v1.schema import SCHEMA, SCHEMA_SHA256
from scripts import materialize_pwts_preflight_locks_v1 as materializer


def test_materializes_exact_canonical_locks_and_compact_manifest(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "locks"
    manifest = materializer.materialize_locks(output_dir)

    configuration_bytes = (output_dir / "configuration.json").read_bytes()
    schema_bytes = (output_dir / "schema.json").read_bytes()
    manifest_bytes = (output_dir / "locks_manifest.json").read_bytes()

    assert configuration_bytes == canonical_json_bytes(get_config_copy())
    assert schema_bytes == canonical_json_bytes(SCHEMA)
    assert not configuration_bytes.endswith(b"\n")
    assert not schema_bytes.endswith(b"\n")
    assert not manifest_bytes.endswith(b"\n")
    assert hashlib.sha256(configuration_bytes).hexdigest() == (
        "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49"
    )
    assert hashlib.sha256(schema_bytes).hexdigest() == SCHEMA_SHA256
    assert json.loads(manifest_bytes) == manifest
    assert manifest_bytes == canonical_json_bytes(manifest)
    assert manifest["lock_file_sha256"] == {
        "configuration.json": hashlib.sha256(configuration_bytes).hexdigest(),
        "schema.json": hashlib.sha256(schema_bytes).hexdigest(),
    }
    assert set(manifest["code_sha256"]) == {
        "research/pwts_v1/config.py",
        "research/pwts_v1/schema.py",
        "scripts/materialize_pwts_preflight_locks_v1.py",
    }
    assert all(manifest["closed_lanes"].values())
    assert manifest["source_points_serialized"] is False


def test_repeated_materialization_is_byte_identical(tmp_path: Path) -> None:
    output_dir = tmp_path / "locks"
    materializer.materialize_locks(output_dir)
    first = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    materializer.materialize_locks(output_dir)
    second = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    assert first == second


def test_configuration_drift_fails_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materializer, "EXPECTED_CONFIG_SHA256", "0" * 64)
    output_dir = tmp_path / "locks"
    with pytest.raises(RuntimeError, match="stated configuration digest"):
        materializer.materialize_locks(output_dir)
    assert not output_dir.exists()


def test_cli_requires_only_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "locks"
    script = Path(materializer.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "configuration.json",
        "locks_manifest.json",
        "schema.json",
    ]
