#!/usr/bin/env python3
"""Generate the four deterministic source-free EXP_CAD_0035 PLY fixtures.

This generator has no physical-scan or evaluator input.  Its CLI is fixed to
the preregistered canonical output root and exact physical-scale row counts.
Tests may call ``generate_fixture_set`` with smaller row counts in a temporary
directory; those calls are unit evidence only and cannot satisfy the CLI gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "EXP_CAD_0035"
PREREGISTRATION_SHA256 = (
    "462274a567cd0ed84b041b8e0576f93ceb1ce6b4df9abe75e63c15ec1f268cb2"
)
SCHEMA = "EXP_CAD_0035_SOURCE_FREE_FIXTURE_MANIFEST_V1"
GENERATOR_RELATIVE_PATH = "scripts/generate_exp_cad_0035_source_free_fixtures.py"
CANONICAL_ROOT = ROOT / (
    "output/research/autoresearch-loop/exp-cad-0035-source-free-fixtures"
)
MANIFEST_NAME = "EXP_CAD_0035_SOURCE_FREE_FIXTURE_MANIFEST.json"
ROW_COUNTS = {"N009": 1_156_302, "N010": 1_323_486}
FIXTURE_PATHS = {
    "boundary_N009": Path("boundary-equivalence/N009.ply"),
    "boundary_N010": Path("boundary-equivalence/N010.ply"),
    "maximal_N009": Path("maximal-retention/N009.ply"),
    "maximal_N010": Path("maximal-retention/N010.ply"),
}
CHUNK_ROWS = 65_536


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _header(row_count: int) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {int(row_count)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "end_header\n"
    ).encode("ascii")


def _float_from_bits(bits: int) -> np.float64:
    return np.asarray([bits], dtype="<u8").view("<f8")[0]


def _apply_boundary_overrides(rows: np.ndarray, *, start: int) -> None:
    """Apply the nine frozen overrides when their global rows are in chunk."""

    def set_row(index: int, values: Mapping[int, float]) -> None:
        local = index - start
        if 0 <= local < len(rows):
            for column, value in values.items():
                rows[local, column] = value

    quiet_nan = _float_from_bits(0x7FF8000000000000)
    set_row(0, {0: quiet_nan, 3: quiet_nan})
    set_row(1, {0: np.float64(np.inf)})
    set_row(2, {3: quiet_nan})
    set_row(3, {3: np.float64(np.inf)})
    huge = np.float64(float.fromhex("0x1.4e718d7d7625ap+664"))
    set_row(4, {3: huge, 4: huge, 5: huge})
    set_row(5, {3: 0.0, 4: 0.0, 5: 0.0})
    set_row(6, {3: float.fromhex("0x1.19799812dea11p-40"), 4: 0.0, 5: 0.0})
    set_row(7, {3: float.fromhex("0x1.19799812dea10p-40"), 4: 0.0, 5: 0.0})
    set_row(8, {3: float.fromhex("0x1.19799812dea12p-40"), 4: 0.0, 5: 0.0})


def write_fixture_ply(
    path: Path,
    *,
    row_count: int,
    boundary_overrides: bool,
    chunk_rows: int = CHUNK_ROWS,
) -> None:
    """Write one exact binary little-endian, six-float64-property PLY."""

    if row_count < 9 or chunk_rows <= 0:
        raise ValueError("fixture row/chunk count is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_header(row_count))
            for start in range(0, int(row_count), int(chunk_rows)):
                stop = min(start + int(chunk_rows), int(row_count))
                identities = np.arange(start, stop, dtype=np.uint64)
                rows = np.empty((stop - start, 6), dtype="<f8", order="C")
                rows[:, 0] = 32.0 + (identities % 4096).astype(np.float64) * (2.0 ** -10)
                rows[:, 1] = -16.0 + (
                    (identities // 4096) % 4096
                ).astype(np.float64) * (2.0 ** -10)
                rows[:, 2] = 4.0 + (identities // 16_777_216).astype(
                    np.float64
                ) * (2.0 ** -10)
                rows[:, 3:5] = 0.0
                rows[:, 5] = 1.0
                if boundary_overrides:
                    _apply_boundary_overrides(rows, start=start)
                handle.write(rows.tobytes(order="C"))
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def generate_fixture_set(
    output_root: Path,
    *,
    row_counts: Mapping[str, int] = ROW_COUNTS,
) -> Mapping[str, Any]:
    """Generate one deterministic fixture set into an absent directory."""

    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise ValueError("EXP35 fixture root must be an exact absolute path")
    if output_root.exists():
        raise FileExistsError("EXP35 fixture root already exists; regeneration is forbidden")
    if set(row_counts) != set(ROW_COUNTS):
        raise ValueError("fixture row-count token inventory mismatch")
    output_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for key, relative in FIXTURE_PATHS.items():
        token = relative.stem
        boundary = key.startswith("boundary_")
        path = output_root / relative
        write_fixture_ply(
            path,
            row_count=int(row_counts[token]),
            boundary_overrides=boundary,
        )
        try:
            canonical_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            canonical_path = path.resolve().as_posix()
        rows.append(
            {
                "key": key,
                "fixture": "BOUNDARY_EQUIVALENCE" if boundary else "MAXIMAL_RETENTION",
                "opaque_token": token,
                "path": canonical_path,
                "row_count": int(row_counts[token]),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_path(path),
            }
        )
    generator = ROOT / GENERATOR_RELATIVE_PATH
    if Path(__file__).resolve(strict=True) != generator:
        raise RuntimeError("EXP35 fixture generator is not the exact canonical file")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_SOURCE_FREE_FIXTURES_GENERATED_ONCE",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "generator_path": GENERATOR_RELATIVE_PATH,
        "generator_sha256": _sha256_path(generator),
        "common_formula": {
            "point_x": "32+(i%4096)*2^-10",
            "point_y": "-16+((i//4096)%4096)*2^-10",
            "point_z": "4+(i//16777216)*2^-10",
            "normal": [0.0, 0.0, 1.0],
            "binary_ply_scalar_dtype": "LITTLE_ENDIAN_FLOAT64",
        },
        "fixtures": rows,
        "physical_scan_hash_call_count": 0,
        "physical_scan_point_cloud_load_call_count": 0,
        "physical_scan_byte_open_operation_count": 0,
        "evaluator_role_label_gold_quality_or_expected_count_access_count": 0,
    }
    _atomic_write(output_root / MANIFEST_NAME, _strict_bytes(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.output_root.is_absolute() or str(args.output_root) != str(CANONICAL_ROOT):
        raise ValueError("fixture CLI output root is not the exact canonical path")
    manifest = generate_fixture_set(args.output_root, row_counts=ROW_COUNTS)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": manifest["status"],
                "fixture_count": len(manifest["fixtures"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
