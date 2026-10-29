from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from research.f1_ciip_v1.primary_parser import ParserLimits, PlyParseError, parse_ply_fd


def _parse(tmp_path: Path, raw: bytes, *, limits: ParserLimits | None = None):
    path = tmp_path / "fixture.ply"
    path.write_bytes(raw)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.lseek(fd, min(3, len(raw)), os.SEEK_SET)
        before = os.lseek(fd, 0, os.SEEK_CUR)
        result = parse_ply_fd(fd, limits=limits or ParserLimits())
        assert os.lseek(fd, 0, os.SEEK_CUR) == before
        return result
    finally:
        os.close(fd)


def _multiset_digest(rows: list[tuple[float, float, float]]) -> str:
    packed = [struct.pack("<ddd", *(0.0 if value == 0.0 else value for value in row)) for row in rows]
    digest = hashlib.sha256(b"PWTS_COORDINATE_MULTISET_V1\x00")
    for row in sorted(packed):
        digest.update(row)
    return digest.hexdigest()


def test_binary_little_endian_exact_diagnostics_and_fixed_array(tmp_path: Path) -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 3\nproperty double x\nproperty double y\nproperty double z\n"
        b"property uchar red\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n"
    )
    rows = [(1.0, 2.0, 3.0), (-0.0, 0.0, -0.0), (1.0, 2.0, 3.0)]
    payload = b"".join(struct.pack("<dddB", *row, index) for index, row in enumerate(rows))
    result = _parse(tmp_path, header + payload)
    assert result.xyz.shape == (3, 3)
    assert result.xyz.dtype == np.dtype("<f8")
    assert result.xyz.flags.c_contiguous and not result.xyz.flags.writeable
    assert result.xyz[1].view(np.uint64).tolist() == [0, 0, 0]
    expected_ignored = ["face:vertex_indices", "vertex:red"]
    expected_ignored_digest = hashlib.sha256(
        json.dumps(expected_ignored, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    diagnostics = result.diagnostics
    assert diagnostics.raw_byte_count == len(header + payload)
    assert diagnostics.raw_ply_sha256 == hashlib.sha256(header + payload).hexdigest()
    assert diagnostics.ply_header_byte_count == len(header)
    assert diagnostics.ply_header_sha256 == hashlib.sha256(header).hexdigest()
    assert diagnostics.ply_format == "binary_little_endian"
    assert diagnostics.parsed_vertex_count == diagnostics.finite_coordinate_count == 3
    assert diagnostics.exact_duplicate_count == 1
    assert diagnostics.ignored_property_name_count == 2
    assert diagnostics.ignored_property_name_digest == expected_ignored_digest
    assert diagnostics.canonical_coordinate_multiset_sha256 == _multiset_digest(rows)
    assert diagnostics.canonical_coordinate_multiset_digest == diagnostics.canonical_coordinate_multiset_sha256


def test_ascii_vertices_are_batched_and_declared_types_are_validated(tmp_path: Path) -> None:
    header = (
        b"ply\r\nformat ascii 1.0\r\n"
        b"element vertex 2\r\nproperty float x\r\nproperty double y\r\nproperty int z\r\n"
        b"property uchar quality\r\nend_header\r\n"
    )
    body = b"1.25 2.5 3 255\n-0 0 -4 0\n"
    result = _parse(tmp_path, header + body)
    np.testing.assert_array_equal(result.xyz, np.array([[1.25, 2.5, 3.0], [0.0, 0.0, -4.0]]))
    assert result.diagnostics.ply_format == "ascii"
    assert result.ignored_properties == ("vertex:quality",)


def test_ascii_nonvertex_list_is_consumed_exactly(tmp_path: Path) -> None:
    raw = (
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property double x\nproperty double y\nproperty double z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        b"0 0 0\n3 0 1 2\n"
    )
    result = _parse(tmp_path, raw)
    assert result.xyz.shape == (1, 3)
    assert result.ignored_properties == ("face:vertex_indices",)


def test_binary_nonvertex_list_is_consumed_exactly(tmp_path: Path) -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property double x\nproperty double y\nproperty double z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    raw = header + struct.pack("<ddd", 0.0, 0.0, 0.0) + struct.pack("<Biii", 3, 0, 1, 2)
    result = _parse(tmp_path, raw)
    assert result.xyz.shape == (1, 3)


@pytest.mark.parametrize(
    "header, message",
    [
        (b"ply\nformat ascii 1.1\nelement vertex 0\nproperty double x\nproperty double y\nproperty double z\nend_header\n", "version 1.0"),
        (b"ply\nformat binary_big_endian 1.0\nelement vertex 0\nproperty double x\nproperty double y\nproperty double z\nend_header\n", "only ASCII or little-endian"),
        (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty double x\nproperty double y\nproperty double z\nproperty list uchar int neighbors\nend_header\n", "fixed-width scalar"),
    ],
)
def test_header_contract_rejects_unfrozen_or_variable_vertex_grammar(
    tmp_path: Path, header: bytes, message: str
) -> None:
    with pytest.raises(PlyParseError, match=message):
        _parse(tmp_path, header)


def test_nonfinite_coordinate_stops_before_negative_zero_normalization(tmp_path: Path) -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property double x\nproperty double y\nproperty double z\nend_header\n"
    )
    with pytest.raises(PlyParseError, match="nonfinite"):
        _parse(tmp_path, header + struct.pack("<ddd", float("nan"), -0.0, 0.0))


def test_binary_truncation_and_trailing_bytes_fail_closed(tmp_path: Path) -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property double x\nproperty double y\nproperty double z\nend_header\n"
    )
    with pytest.raises(PlyParseError, match="truncated"):
        _parse(tmp_path, header + bytes(23))
    with pytest.raises(PlyParseError, match="trailing undeclared"):
        _parse(tmp_path, header + bytes(24) + b"X")


@pytest.mark.parametrize(
    "body, message",
    [
        (b"0 0\n", "malformed, missing, or extra"),
        (b"0 0 0 1\n", "malformed, missing, or extra"),
        (b"0 0 0\nundeclared\n", "trailing undeclared"),
    ],
)
def test_ascii_missing_extra_or_trailing_fields_fail_closed(
    tmp_path: Path, body: bytes, message: str
) -> None:
    header = (
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property double x\nproperty double y\nproperty double z\nend_header\n"
    )
    with pytest.raises(PlyParseError, match=message):
        _parse(tmp_path, header + body)


def test_ascii_balanced_total_token_count_cannot_hide_per_row_mismatch(tmp_path: Path) -> None:
    header = (
        b"ply\nformat ascii 1.0\nelement vertex 2\n"
        b"property double x\nproperty double y\nproperty double z\nend_header\n"
    )
    with pytest.raises(PlyParseError, match="malformed, missing, or extra"):
        _parse(tmp_path, header + b"0 0 0 1\n0 0\n")


def test_ascii_declared_integer_rejects_fractional_value(tmp_path: Path) -> None:
    header = (
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property double x\nproperty double y\nproperty double z\nproperty uchar quality\nend_header\n"
    )
    with pytest.raises(PlyParseError, match="invalid ASCII integer"):
        _parse(tmp_path, header + b"0 0 0 1.5\n")
    with pytest.raises(PlyParseError, match="invalid ASCII integer"):
        _parse(tmp_path, header + b"0 0 0 1e0\n")


def test_declared_vertex_and_raw_byte_caps_stop_before_allocation(tmp_path: Path) -> None:
    header = (
        b"ply\nformat ascii 1.0\nelement vertex 2\n"
        b"property double x\nproperty double y\nproperty double z\nend_header\n"
    )
    with pytest.raises(PlyParseError, match="exceeds its frozen bound"):
        _parse(tmp_path, header, limits=ParserLimits(vertex_records_maximum=1))
    with pytest.raises(PlyParseError, match="raw PLY byte count"):
        _parse(tmp_path, header, limits=ParserLimits(raw_bytes_maximum=len(header) - 1))


def test_parser_accepts_only_an_open_integer_fd(tmp_path: Path) -> None:
    path = tmp_path / "x.ply"
    path.write_bytes(b"ply\n")
    with pytest.raises(TypeError, match="integer file descriptor"):
        parse_ply_fd(path)  # type: ignore[arg-type]
    with pytest.raises(PlyParseError, match="open file descriptor"):
        parse_ply_fd(-1)
