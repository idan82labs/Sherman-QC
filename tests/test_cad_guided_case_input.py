"""Invented bytes and owned pytest temp files; no CAD/native/application execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_case_input.py"


def module():
    name = "invented_case_input"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.modules.pop(name, None)
    return loaded


def ply(points=None, encoding="ascii", kind="double"):
    if points is None:
        points = np.arange(180, dtype=float).reshape(60, 3) / 10
    header = (f"ply\nformat {encoding} 1.0\ncomment invented only\nelement vertex {len(points)}\n"
              f"property uchar red\nproperty {kind} z\nproperty {kind} x\nproperty {kind} y\nend_header\n").encode()
    if encoding == "ascii":
        body = "".join(f"7 {z:.17g} {x:.17g} {y:.17g}\n" for x, y, z in points).encode()
    else:
        fmt = ("<" if encoding == "binary_little_endian" else ">") + "B" + ("f" if kind == "float" else "d") * 3
        body = b"".join(struct.pack(fmt, 7, z, x, y) for x, y, z in points)
    return header + body


def sidecar(scan, cad=b"invented cad bytes"):
    return {"schema_version": 1, "scan_sha256": hashlib.sha256(scan).hexdigest(),
            "cad_sha256": hashlib.sha256(cad).hexdigest(), "reference_frame": "CAD_REFERENCE",
            "unit": "mm", "selection_method": "EXPLICIT_STABLE_REGION_SELECTION",
            "cad_datum_points": np.arange(108, dtype=float).reshape(36, 3).tolist(),
            "cad_datum_normals": np.tile([1., 0., 0.], (36, 1)).tolist(),
            "scan_datum_row_indices": list(range(36))}


@pytest.mark.parametrize("encoding", ["ascii", "binary_little_endian", "binary_big_endian"])
@pytest.mark.parametrize("kind", ["float", "double"])
def test_declared_scalar_types_property_names_and_row_order_agree_across_encodings(encoding, kind):
    m = module()
    points = np.array([[.1, 1.00000006, -3.33333333333333], [4.125, -7., 123.456789]])
    expected = points.astype(np.float32 if kind == "float" else np.float64).astype(np.float64)
    actual = m.parse_point_cloud_ply(ply(points, encoding, kind))
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.writeable is False


@pytest.mark.parametrize("change", ["missing_xyz", "duplicate_property", "list_property", "mesh",
                                    "bad_format", "too_many", "short_binary", "long_binary",
                                    "too_few_rows", "too_many_rows", "nan", "underscore",
                                    "fractional_integer", "integer_overflow", "too_many_tokens",
                                    "coordinate_bound", "header_bound", "nonascii_header"])
def test_malformed_or_unsupported_ply_never_yields_a_prefix(change):
    m = module()
    blob = ply(np.array([[1., 2., 3.]]))
    if change == "missing_xyz": blob = blob.replace(b"double x", b"double nx")
    elif change == "duplicate_property": blob = blob.replace(b"double z", b"double x")
    elif change == "list_property": blob = blob.replace(b"property uchar red", b"property list uchar float red")
    elif change == "mesh": blob = blob.replace(b"end_header", b"element face 0\nend_header")
    elif change == "bad_format": blob = blob.replace(b"ascii 1.0", b"ascii 2.0")
    elif change == "too_many": blob = blob.replace(b"vertex 1", b"vertex 200001")
    elif change == "short_binary": blob = ply(encoding="binary_little_endian")[:-1]
    elif change == "long_binary": blob = ply(encoding="binary_big_endian") + b"\x00"
    elif change == "too_few_rows": blob = blob[:blob.index(b"7 3 1 2")]
    elif change == "too_many_rows": blob += b"7 3 1 2\n" * 1000
    elif change == "nan": blob = blob.replace(b"7 3 1 2", b"7 nan 1 2")
    elif change == "underscore": blob = blob.replace(b"7 3 1 2", b"7 3 1_0.5 2")
    elif change == "fractional_integer": blob = blob.replace(b"7 3 1 2", b"7.5 3 1 2")
    elif change == "integer_overflow": blob = blob.replace(b"7 3 1 2", b"256 3 1 2")
    elif change == "too_many_tokens": blob = blob.replace(b"7 3 1 2", b"7 3 1 2" + b" 0" * 10000)
    elif change == "coordinate_bound": blob = blob.replace(b"7 3 1 2", b"7 1000001 1 2")
    elif change == "header_bound": blob = b"ply\ncomment " + b"x" * 65536
    elif change == "nonascii_header": blob = blob.replace(b"invented", b"\xff")
    with pytest.raises(m.InputUnavailable):
        m.parse_point_cloud_ply(blob)


def test_binding_is_exact_ordered_snapshot_not_a_filename_or_count_assertion():
    m, scan, cad = module(), ply(), b"invented cad bytes"
    datum = json.dumps(sidecar(scan, cad)).encode()
    result = m.bind_registered_scan(scan_bytes=scan, cad_bytes=cad, datum_bytes=datum)
    assert result.scan_sha256 == hashlib.sha256(scan).hexdigest()
    assert result.datum_sha256 == hashlib.sha256(datum).hexdigest()
    np.testing.assert_array_equal(result.scan_datum_row_indices, np.arange(36))
    assert set(result.inference_arguments()) == {"scan_points", "cad_datum_points", "cad_datum_normals",
                                                "scan_datum_row_indices", "unit_status"}
    assert all(not array.flags.writeable for array in [result.scan_points, result.cad_datum_points,
                                                       result.cad_datum_normals, result.scan_datum_row_indices])
    for bad_scan, bad_cad in ((scan + b"\n", cad), (scan, cad + b"changed")):
        with pytest.raises(m.InputUnavailable, match="DATUM_SOURCE_BINDING_MISMATCH"):
            m.bind_registered_scan(scan_bytes=bad_scan, cad_bytes=bad_cad, datum_bytes=datum)


@pytest.mark.parametrize("change", ["bool_schema", "unknown_truth_key", "wrong_frame", "wrong_unit",
                                    "wrong_selection", "stale_scan", "stale_cad", "bool_index",
                                    "repeated_index", "negative_index", "huge_index", "too_few_datums",
                                    "all_rows_datums", "wrong_normals", "bool_coordinate",
                                    "missing_key", "duplicate_json_key", "nan_json", "ragged"])
def test_datum_contract_is_bounded_explicit_and_not_silently_coerced(change):
    m, scan, cad = module(), ply(), b"invented cad bytes"
    obj = sidecar(scan, cad)
    if change == "bool_schema": obj["schema_version"] = True
    elif change == "unknown_truth_key": obj["gold_count"] = 1
    elif change == "wrong_frame": obj["reference_frame"] = "SCAN_REFERENCE"
    elif change == "wrong_unit": obj["unit"] = "cm"
    elif change == "wrong_selection": obj["selection_method"] = "GOLD_ASSISTED"
    elif change == "stale_scan": obj["scan_sha256"] = "a" * 64
    elif change == "stale_cad": obj["cad_sha256"] = "b" * 64
    elif change == "bool_index": obj["scan_datum_row_indices"][0] = True
    elif change == "repeated_index": obj["scan_datum_row_indices"][0] = 1
    elif change == "negative_index": obj["scan_datum_row_indices"][0] = -1
    elif change == "huge_index": obj["scan_datum_row_indices"][0] = 2**64 - 1
    elif change == "too_few_datums": obj["cad_datum_points"].pop()
    elif change == "all_rows_datums": obj["scan_datum_row_indices"] = list(range(60))
    elif change == "wrong_normals": obj["cad_datum_normals"][0] = [2, 0, 0]
    elif change == "bool_coordinate": obj["cad_datum_points"][0][0] = True
    elif change == "missing_key": del obj["scan_sha256"]
    elif change == "nan_json": obj["cad_datum_points"][0][0] = float("nan")
    elif change == "ragged": obj["cad_datum_points"][0] = [1, 2]
    encoded = json.dumps(obj).encode()
    if change == "duplicate_json_key": encoded = encoded[:-1] + b", \"unit\": \"mm\"}"
    with pytest.raises(m.InputUnavailable):
        m.bind_registered_scan(scan_bytes=scan, cad_bytes=cad, datum_bytes=encoded)


def test_non_opted_in_loader_performs_no_file_access(monkeypatch):
    m = module()
    monkeypatch.setattr(m, "_read_regular_snapshot", lambda *args: pytest.fail("unexpected read"))
    assert m.load_registered_case(cad_path="absent", scan_path="absent", datum_path="absent")["status"] == "UNAVAILABLE"


def test_snapshot_reader_rejects_nonregular_symlink_relative_and_oversize(tmp_path):
    m = module()
    regular = tmp_path / "invented"
    regular.write_bytes(b"1234")
    assert m._read_regular_snapshot(regular, 4) == b"1234"
    link = tmp_path / "link"
    link.symlink_to(regular)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for path, limit in [(regular, 3), (link, 4), (fifo, 4), ("relative", 4), (tmp_path, 4)]:
        with pytest.raises((m.InputUnavailable, OSError)):
            m._read_regular_snapshot(path, limit)


def test_file_loader_selects_binding_lazily_and_imports_exact_owned_snapshot(tmp_path, monkeypatch):
    m, scan, cad = module(), ply(), b"invented cad bytes"
    paths = [tmp_path / name for name in ("source.step", "scan.ply", "datum.json")]
    for path, blob in zip(paths, (cad, scan, json.dumps(sidecar(scan, cad)).encode())):
        path.write_bytes(blob)
    calls = []
    native = SimpleNamespace(success=True, native_shape=object(), native_binding="ocp")

    class Importer:
        def __init__(self, **kwargs):
            assert kwargs == {"linear_deflection": .25, "angular_deflection": .5, "allow_ocp": True}
            self.native_binding = None

        def _check_occ_available(self):
            self.native_binding = "ocp"
            calls.append("select")
            return True

        def import_file(self, path, *, retain_native_shape):
            assert calls == ["select"] and retain_native_shape is True
            assert Path(path) != paths[0] and Path(path).read_bytes() == cad
            calls.append("read_owned_snapshot")
            return native

    monkeypatch.setitem(sys.modules, "cad_import", SimpleNamespace(CADImporter=Importer))
    loaded = m.load_registered_case(cad_path=paths[0], scan_path=paths[1], datum_path=paths[2], allow_ocp=True)
    assert loaded["status"] == "AVAILABLE" and loaded["cad_result"] is native
    assert calls == ["select", "read_owned_snapshot"]
    assert loaded["source_bindings"]["cad_sha256"] == hashlib.sha256(cad).hexdigest()
    assert loaded["production_use_allowed"] is False
