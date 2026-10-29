#!/usr/bin/env python3
"""Create the nonsemantic PWTS C001-C037 raw-PLY source lock.

This program deliberately stops at byte/header/coordinate accounting.  It does
not sample points, construct neighborhoods, estimate normals, or import any F1
or PWTS semantic module.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence


PREFLIGHT_ID = "PWTS_REPRESENTATION_PREFLIGHT_V1"
SCHEMA_VERSION = 1
EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
SAFE_CASE_MAP_FIELDS = frozenset(
    {"case_token", "source_scan_path", "source_scan_sha256", "source_scan_bytes"}
)

_SCALAR_FORMATS = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(stream: BinaryIO) -> tuple[bytes, str, list[dict[str, Any]]]:
    raw_lines: list[bytes] = []
    for _ in range(100_000):
        line = stream.readline()
        if not line:
            raise ValueError("PLY header ended before end_header")
        raw_lines.append(line)
        if line.rstrip(b"\r\n") == b"end_header":
            break
    else:
        raise ValueError("PLY header line cap exceeded")
    header = b"".join(raw_lines)
    try:
        lines = [line.decode("ascii") for line in header.splitlines()]
    except UnicodeDecodeError as exc:
        raise ValueError("PLY header must be ASCII") from exc
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    format_rows = [line.split() for line in lines if line.startswith("format ")]
    if len(format_rows) != 1 or len(format_rows[0]) != 3:
        raise ValueError("PLY must declare exactly one format")
    ply_format = format_rows[0][1]
    if ply_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format: {ply_format}")
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] in {"comment", "obj_info", "format", "end_header"}:
            continue
        if fields[0] == "element":
            if len(fields) != 3:
                raise ValueError("malformed PLY element declaration")
            current = {"name": fields[1], "count": int(fields[2]), "properties": []}
            if current["count"] < 0:
                raise ValueError("negative PLY element count")
            elements.append(current)
        elif fields[0] == "property":
            if current is None:
                raise ValueError("PLY property declared before an element")
            if len(fields) == 3:
                kind, name = fields[1], fields[2]
                if kind not in _SCALAR_FORMATS:
                    raise ValueError(f"unsupported PLY scalar type: {kind}")
                current["properties"].append(
                    {"name": name, "kind": "scalar", "scalar_type": kind}
                )
            elif len(fields) == 5 and fields[1] == "list":
                count_kind, item_kind, name = fields[2], fields[3], fields[4]
                if count_kind not in _SCALAR_FORMATS or item_kind not in _SCALAR_FORMATS:
                    raise ValueError("unsupported PLY list type")
                current["properties"].append(
                    {
                        "name": name, "kind": "list", "count_type": count_kind,
                        "item_type": item_kind,
                    }
                )
            else:
                raise ValueError("malformed PLY property declaration")
        else:
            raise ValueError(f"unsupported PLY header directive: {fields[0]}")
    vertices = [element for element in elements if element["name"] == "vertex"]
    if len(vertices) != 1:
        raise ValueError("PLY must declare exactly one vertex element")
    names = [prop["name"] for prop in vertices[0]["properties"]]
    if any(names.count(axis) != 1 for axis in ("x", "y", "z")):
        raise ValueError("PLY vertex must have exactly one x, y, and z property")
    return header, ply_format, elements


def _unpack_scalar(stream: BinaryIO, endian: str, scalar_type: str) -> int | float:
    fmt = endian + _SCALAR_FORMATS[scalar_type]
    encoded = stream.read(struct.calcsize(fmt))
    if len(encoded) != struct.calcsize(fmt):
        raise ValueError("truncated binary PLY vertex data")
    return struct.unpack(fmt, encoded)[0]


def _binary_vertex_records(
    stream: BinaryIO, *, count: int, properties: Sequence[Mapping[str, Any]], endian: str,
) -> Iterator[tuple[float, float, float]]:
    for _ in range(count):
        values: dict[str, float] = {}
        for prop in properties:
            if prop["kind"] == "scalar":
                value = _unpack_scalar(stream, endian, str(prop["scalar_type"]))
                if prop["name"] in {"x", "y", "z"}:
                    values[str(prop["name"])] = float(value)
            else:
                size = int(_unpack_scalar(stream, endian, str(prop["count_type"])))
                if size < 0:
                    raise ValueError("negative PLY list length")
                item_fmt = endian + _SCALAR_FORMATS[str(prop["item_type"])]
                encoded = stream.read(struct.calcsize(item_fmt) * size)
                if len(encoded) != struct.calcsize(item_fmt) * size:
                    raise ValueError("truncated binary PLY list")
        yield values["x"], values["y"], values["z"]


def _ascii_vertex_records(
    stream: BinaryIO, *, count: int, properties: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[float, float, float]]:
    for _ in range(count):
        line = stream.readline()
        if not line:
            raise ValueError("truncated ASCII PLY vertex data")
        fields = line.split()
        offset = 0
        values: dict[str, float] = {}
        for prop in properties:
            if prop["kind"] == "scalar":
                if offset >= len(fields):
                    raise ValueError("short ASCII PLY vertex row")
                if prop["name"] in {"x", "y", "z"}:
                    values[str(prop["name"])] = float(fields[offset])
                offset += 1
            else:
                if offset >= len(fields):
                    raise ValueError("short ASCII PLY list")
                size = int(fields[offset])
                if size < 0 or offset + 1 + size > len(fields):
                    raise ValueError("invalid ASCII PLY list")
                offset += 1 + size
        if offset != len(fields):
            raise ValueError("extra ASCII PLY vertex fields")
        yield values["x"], values["y"], values["z"]


def _normalized_packed_xyz(x: float, y: float, z: float) -> bytes | None:
    if not all(math.isfinite(value) for value in (x, y, z)):
        return None
    values = tuple(0.0 if value == 0.0 else value for value in (x, y, z))
    return struct.pack("<ddd", *values)


def _flush_sorted_chunk(records: list[bytes], directory: Path) -> Path:
    records.sort()
    descriptor, name = tempfile.mkstemp(prefix="pwts-coordinates-", suffix=".bin", dir=directory)
    with os.fdopen(descriptor, "wb") as stream:
        stream.writelines(records)
    return Path(name)


def _chunk_records(path: Path) -> Iterator[bytes]:
    with path.open("rb") as stream:
        while True:
            record = stream.read(24)
            if not record:
                return
            if len(record) != 24:
                raise ValueError("corrupt PWTS coordinate sort chunk")
            yield record


def _coordinate_census(
    records: Iterable[tuple[float, float, float]], *, scratch_root: Path,
) -> tuple[int, int, str]:
    finite_count = 0
    chunk: list[bytes] = []
    chunks: list[Path] = []
    digest = hashlib.sha256(b"PWTS_COORDINATE_MULTISET_V1\0")
    try:
        for xyz in records:
            packed = _normalized_packed_xyz(*xyz)
            if packed is None:
                continue
            finite_count += 1
            chunk.append(packed)
            if len(chunk) >= 250_000:
                chunks.append(_flush_sorted_chunk(chunk, scratch_root))
                chunk = []
        if chunk:
            chunks.append(_flush_sorted_chunk(chunk, scratch_root))
        previous: bytes | None = None
        unique_count = 0
        for packed in heapq.merge(*(_chunk_records(path) for path in chunks)):
            digest.update(packed)
            if packed != previous:
                unique_count += 1
                previous = packed
        return finite_count, finite_count - unique_count, digest.hexdigest()
    finally:
        for path in chunks:
            path.unlink(missing_ok=True)


def inspect_ply(path: Path, *, scratch_root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        header, ply_format, elements = _read_header(stream)
        vertex = next(element for element in elements if element["name"] == "vertex")
        if elements.index(vertex) != 0:
            raise ValueError("PWTS source lock requires vertex to be the first PLY element")
        if ply_format == "ascii":
            records = _ascii_vertex_records(
                stream, count=int(vertex["count"]), properties=vertex["properties"]
            )
        else:
            records = _binary_vertex_records(
                stream, count=int(vertex["count"]), properties=vertex["properties"],
                endian="<" if ply_format == "binary_little_endian" else ">",
            )
        scratch = (scratch_root or path.parent).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        finite_count, duplicate_count, multiset_sha = _coordinate_census(
            records, scratch_root=scratch
        )
    ignored = sorted(
        f"{element['name']}:{prop['name']}"
        for element in elements
        for prop in element["properties"]
        if not (element["name"] == "vertex" and prop["name"] in {"x", "y", "z"})
    )
    return {
        "raw_path": str(path),
        "raw_byte_count": path.stat().st_size,
        "raw_ply_sha256": sha256_file(path),
        "ply_header_byte_count": len(header),
        "ply_header_sha256": hashlib.sha256(header).hexdigest(),
        "ply_format": ply_format,
        "parsed_vertex_count": int(vertex["count"]),
        "finite_coordinate_count": finite_count,
        "exact_duplicate_count": duplicate_count,
        "ignored_property_name_count": len(ignored),
        "ignored_property_name_digest": hashlib.sha256(canonical_json_bytes(ignored)).hexdigest(),
        "canonical_coordinate_multiset_sha256": multiset_sha,
        "canonical_coordinate_multiset_digest": multiset_sha,
    }


def _safe_case_mapping(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("case map must contain a cases array")
    result: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("case-map rows must be objects")
        missing = {"case_token", "source_scan_path"} - set(raw)
        if missing:
            raise ValueError(f"case-map row lacks {sorted(missing)}")
        # Deliberately extract only source-lock fields.  Top-level catalog,
        # family, count, and truth metadata are never copied or interpreted.
        result.append({key: raw[key] for key in SAFE_CASE_MAP_FIELDS if key in raw})
    result.sort(key=lambda row: str(row["case_token"]))
    if tuple(str(row["case_token"]) for row in result) != EXPECTED_CASE_TOKENS:
        raise ValueError("case map must resolve exactly C001-C037")
    return result


def lock_sources(case_map_path: Path, output_path: Path, *, scratch_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mapping in _safe_case_mapping(case_map_path.resolve()):
        row = {"case_token": str(mapping["case_token"]), **inspect_ply(
            Path(str(mapping["source_scan_path"])), scratch_root=scratch_root
        )}
        asserted_sha = mapping.get("source_scan_sha256")
        asserted_bytes = mapping.get("source_scan_bytes")
        if asserted_sha is not None and str(asserted_sha) != row["raw_ply_sha256"]:
            raise ValueError(f"{row['case_token']}: prior source SHA mismatch")
        if asserted_bytes is not None and int(asserted_bytes) != row["raw_byte_count"]:
            raise ValueError(f"{row['case_token']}: prior source byte-count mismatch")
        rows.append(row)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "preflight_id": PREFLIGHT_ID,
        "kind": "pwts_source_lock_v1",
        "case_count": len(rows),
        "case_tokens": list(EXPECTED_CASE_TOKENS),
        "case_map_sha256": sha256_file(case_map_path.resolve()),
        "case_map_consumed_fields": sorted(SAFE_CASE_MAP_FIELDS),
        "coordinate_properties_consumed": ["x", "y", "z"],
        "other_properties_ignored": True,
        "semantic_work_performed": False,
        "sampling_performed": False,
        "source_points_serialized": False,
        "coordinate_multiset_digest_algorithm": "SHA256(PWTS_COORDINATE_MULTISET_V1\\0 || sorted(canonical_float64_xyz_with_duplicates))",
        "ignored_property_digest_algorithm": "SHA256(canonical_JSON(sorted(element:name)))",
        "cases": rows,
    }
    manifest["input_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = lock_sources(args.case_map, args.output, scratch_root=args.scratch_root)
    print(manifest["input_manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
