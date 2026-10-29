"""Invented-only admission/lineage controls; no protected source reads."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
REVIEW = runpy.run_path(str(ROOT / "scripts/review_gold11_inputs_v1.py"))


def test_fixed_scope_is_gold11_not_gold13():
    cases = REVIEW["CASES"]
    assert len(cases) == len({row["case_token"] for row in cases}) == 11
    assert len({row["cad_sha256"] for row in cases}) == 6
    assert not {"CG60B2F9102E", "CGC60DACC70E"} & {row["case_token"] for row in cases}


def test_regular_snapshot_binds_used_bytes(tmp_path):
    path = tmp_path / "invented.bin"
    path.write_bytes(b"invented")
    digest = hashlib.sha256(b"invented").hexdigest()
    blob, facts = REVIEW["bounded_read"](path, 8, digest)
    assert blob == b"invented" and facts["sha256"] == digest
    assert facts["snapshot_consistent"] and facts["bytes"] == 8


@pytest.mark.parametrize("blob,limit,expected", [(b"", 8, None), (b"123", 2, None),
                                                (b"123", 3, "0" * 64)])
def test_snapshot_refuses_bad_admission(tmp_path, blob, limit, expected):
    path = tmp_path / "invented.bin"
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        REVIEW["bounded_read"](path, limit, expected)


def test_snapshot_refuses_final_symlink(tmp_path):
    target, link = tmp_path / "invented.bin", tmp_path / "link.bin"
    target.write_bytes(b"abc")
    link.symlink_to(target)
    with pytest.raises(OSError):
        REVIEW["bounded_read"](link, 8)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json_refuses_nonfinite_or_duplicate(text):
    with pytest.raises(ValueError):
        json.loads(text, object_pairs_hook=REVIEW["unique_object"],
                   parse_constant=REVIEW["reject_constant"])


def test_stats_uses_all_points_no_pose_or_scale_fit():
    import numpy as np
    points = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=float)
    original = points.copy()
    result = REVIEW["stats"](points)
    assert result["centroid"] == [2, 0, 0]
    assert result["aabb_max"] == [4, 0, 0]
    assert result["principal_rms_lengths"] == pytest.approx([(8 / 3)**.5, 0, 0])
    assert np.array_equal(points, original)


def test_native_helper_import_is_inert():
    helper = runpy.run_path(str(ROOT / "scripts/gold11_cad_admission.py"))
    assert callable(helper["inspect_step_snapshot"])


def test_case_and_per_line_declarations_cannot_disappear():
    case = REVIEW["declarations"]({"reference_frame": "CAD_REFERENCE", "unit": "mm"})
    line = REVIEW["declarations"]({"reference_frame": "SCAN_RAW", "unit": "inch",
                                   "carrier_convention": "SKIN", "not_metadata": [1, 2, 3]})
    assert line == {"reference_frame": "SCAN_RAW", "unit": "inch", "carrier_convention": "SKIN"}
    differences = REVIEW["declaration_differences"](case, line)
    assert {row["key"] for row in differences} == {"reference_frame", "unit"}
    assert all(row["status"].startswith("EXPLICIT_DIFFERENCE") for row in differences)


def test_declaration_bounds_do_not_silently_drop_or_invent_fields():
    assert REVIEW["declarations"]({}) == {}
    assert REVIEW["declarations"]({"unit": None}) == {"unit": None}
    result = REVIEW["declarations"]({"frame": "x" * 1025})
    assert result["frame"]["status"] == "DECLARATION_VALUE_BUDGET"
    assert REVIEW["declaration_differences"]({"unit": "mm"}, {"unit": "mm"}) == []


def test_reference_review_retains_all_lines_without_any_scan_read():
    helper = runpy.run_path(str(ROOT / "scripts/gold11_admission_geometry.py"))
    rows = REVIEW["review_reference_rows"]({"unit": "mm", "bends": [
        {"line_start": [0, 0, 0], "line_end": [1, 0, 0], "unit": "inch"},
        {"line_start": [0, 0, 0]},
    ]}, helper["parse_reference_endpoints"])
    assert len(rows) == 2
    assert rows[0]["case_line_declaration_differences"][0]["key"] == "unit"
    assert rows[1]["status"] == "REFERENCE_REPRESENTATION_UNAVAILABLE"
    assert all("nearest_raw_vertex_distance" not in row for row in rows)


def test_case_output_has_no_unbounded_declaration_aliases(monkeypatch):
    helper = runpy.run_path(str(ROOT / "scripts/gold11_admission_geometry.py"))
    function = REVIEW["review_case"]
    monkeypatch.setitem(function.__globals__, "bounded_read", lambda *args: (b"invented", {}))
    result = function(REVIEW["CASES"][0], {"coordinate_frame": "x" * 1025, "bends": [
        {"line_start": [0, 0, 0], "line_end": [1, 0, 0]},
    ]}, lambda blob: {"points": None, "header": {}}, helper["parse_reference_endpoints"])
    assert result["case_declarations"]["coordinate_frame"]["status"] == "DECLARATION_VALUE_BUDGET"
    assert not {"reference_frame", "reference_unit", "reference_carrier"} & result.keys()
    assert len(result["references"]) == 1


def test_source_guard_has_no_backend_or_scoring_imports():
    import ast
    for name in ("review_gold11_inputs_v1.py", "gold11_admission_geometry.py", "gold11_cad_admission.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(value.startswith(("backend", "research", "apps")) for value in modules)
