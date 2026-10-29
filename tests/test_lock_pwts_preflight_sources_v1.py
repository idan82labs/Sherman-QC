from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from scripts import lock_pwts_preflight_sources_v1 as locker


def _ascii_ply(path: Path) -> None:
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\n"
        "element face 0\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        "0 -0 1 5\n0 0 1 6\nnan 1 2 7\n2 3 4 8\n",
        encoding="ascii",
    )


def _binary_ply(path: Path) -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 2\nproperty float x\nproperty float y\nproperty float z\n"
        b"element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    )
    path.write_bytes(header + struct.pack("<ffffff", 1, 2, 3, 4, 5, 6))


def test_inspect_ply_counts_only_finite_canonical_xyz(tmp_path: Path) -> None:
    ply = tmp_path / "source.ply"
    _ascii_ply(ply)
    row = locker.inspect_ply(ply, scratch_root=tmp_path / "scratch")
    assert row["parsed_vertex_count"] == 4
    assert row["finite_coordinate_count"] == 3
    assert row["exact_duplicate_count"] == 1
    assert row["ignored_property_name_count"] == 2
    assert row["raw_ply_sha256"] == hashlib.sha256(ply.read_bytes()).hexdigest()
    assert len(row["canonical_coordinate_multiset_sha256"]) == 64


def test_binary_little_endian_xyz_is_supported(tmp_path: Path) -> None:
    ply = tmp_path / "source.ply"
    _binary_ply(ply)
    row = locker.inspect_ply(ply, scratch_root=tmp_path / "scratch")
    assert row["ply_format"] == "binary_little_endian"
    assert row["parsed_vertex_count"] == 2
    assert row["finite_coordinate_count"] == 2
    assert row["exact_duplicate_count"] == 0


def test_lock_sources_sanitizes_case_mapping_and_self_binds(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(locker, "EXPECTED_CASE_TOKENS", ("C001", "C002"))
    paths = []
    for token in ("C001", "C002"):
        path = tmp_path / f"{token}.ply"
        _binary_ply(path)
        paths.append(path)
    case_map = tmp_path / "scan_only_map.json"
    case_map.write_text(json.dumps({
        "gold_bend_count": 999,
        "cases": [
            {
                "case_token": token, "source_scan_path": str(path),
                "source_scan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_scan_bytes": path.stat().st_size, "part_key": "must-not-copy",
            }
            for token, path in zip(("C001", "C002"), paths)
        ],
    }))
    output = tmp_path / "input_manifest.json"
    result = locker.lock_sources(case_map, output, scratch_root=tmp_path / "scratch")
    encoded = output.read_bytes()
    assert b"gold_bend_count" not in encoded
    assert b"part_key" not in encoded
    asserted = result.pop("input_manifest_sha256")
    assert asserted == hashlib.sha256(locker.canonical_json_bytes(result)).hexdigest()
    assert result["semantic_work_performed"] is False
    assert result["source_points_serialized"] is False
