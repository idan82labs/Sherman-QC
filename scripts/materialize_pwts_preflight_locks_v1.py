#!/usr/bin/env python3
"""Materialize the neutral, professor-authorized PWTS lock files.

This utility serializes only the already-locked configuration and compact
schema.  It deliberately has no scan input and performs no PWTS semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.pwts_v1.config import (
    EXPECTED_CONFIG_SHA256,
    canonical_json_bytes,
    get_config_copy,
)
from research.pwts_v1.schema import SCHEMA, SCHEMA_SHA256


AUTHORIZED_CONFIG_SHA256 = (
    "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49"
)
PREFLIGHT_ID = "PWTS_REPRESENTATION_PREFLIGHT_V1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _code_hashes() -> dict[str, str]:
    relative_paths = (
        "research/pwts_v1/config.py",
        "research/pwts_v1/schema.py",
        "scripts/materialize_pwts_preflight_locks_v1.py",
    )
    return {
        relative_path: _sha256_file(REPOSITORY_ROOT / relative_path)
        for relative_path in relative_paths
    }


def materialize_locks(output_dir: Path) -> dict[str, Any]:
    """Write canonical lock files and return their compact manifest."""
    configuration_bytes = canonical_json_bytes(get_config_copy())
    schema_bytes = canonical_json_bytes(SCHEMA)
    observed_configuration_sha256 = _sha256_bytes(configuration_bytes)
    observed_schema_sha256 = _sha256_bytes(schema_bytes)

    if EXPECTED_CONFIG_SHA256 != AUTHORIZED_CONFIG_SHA256:
        raise RuntimeError(
            "PWTS stated configuration digest differs from the professor lock: "
            f"stated={EXPECTED_CONFIG_SHA256}, authorized={AUTHORIZED_CONFIG_SHA256}"
        )
    if observed_configuration_sha256 != AUTHORIZED_CONFIG_SHA256:
        raise RuntimeError(
            "PWTS configuration payload differs from the professor lock: "
            f"observed={observed_configuration_sha256}, "
            f"authorized={AUTHORIZED_CONFIG_SHA256}"
        )
    if observed_schema_sha256 != SCHEMA_SHA256:
        raise RuntimeError(
            "PWTS schema payload digest mismatch: "
            f"observed={observed_schema_sha256}, stated={SCHEMA_SHA256}"
        )

    manifest: dict[str, Any] = {
        "preflight_id": PREFLIGHT_ID,
        "kind": "pwts_neutral_lock_materialization_v1",
        "lock_file_sha256": {
            "configuration.json": observed_configuration_sha256,
            "schema.json": observed_schema_sha256,
        },
        "code_sha256": _code_hashes(),
        "closed_lanes": {
            "cad": True,
            "gold": True,
            "legacy_f1": True,
            "raw_scans": True,
            "runtime": True,
            "semantic_compute": True,
        },
        "source_points_serialized": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)

    output_dir = output_dir.resolve()
    _atomic_write(output_dir / "configuration.json", configuration_bytes)
    _atomic_write(output_dir / "schema.json", schema_bytes)
    _atomic_write(output_dir / "locks_manifest.json", manifest_bytes)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    materialize_locks(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
