import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0017_native_roundbend_patch_axes.py"
SPEC = importlib.util.spec_from_file_location("exp17", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _surface(name, radius):
    return {"id": f"surf:{name}", "geometry": {"kind": "cylinder", "origin": {"x": 5, "y": 3, "z": 100}, "axis": {"x": 0, "y": 0, "z": 1}, "radius": radius}}


def _cadir(intervals=((-20, -5), (7, 18))):
    faces, surfaces, tessellations = [], [], []
    for level, radius in (("inner", 1.0), ("outer", 3.0)):
        for index, (start, end) in enumerate(intervals):
            token = f"{level}:{index}"
            faces.append({"id": f"face:{token}", "surface": f"surf:{token}"})
            surfaces.append(_surface(token, radius))
            tessellations.append({"body": "body:1", "faces": [f"face:{token}"], "vertices": [{"x": 5, "y": 3 + radius, "z": start}, {"x": 5, "y": 3 + radius, "z": end}]})
    return {
        "native": {"sldprt": {"custom_properties": [{"name": "Bends", "values": [{"value": str(len(intervals))}]}, {"name": "Sheet Metal Thickness", "values": [{"value": "2"}]}], "arenas": {"features": [{"kind": "RoundBend", "name": f"RoundBend{i+1}", "suppressed": False} for i in range(len(intervals))]}}},
        "model": {"faces": faces, "surfaces": surfaces, "tessellations": tessellations},
    }


def test_preserves_disconnected_collinear_bend_patches():
    result = MODULE.extract_native_roundbend_patch_axes(_cadir())
    assert result["qualifying_axis_cluster_count"] == 1
    assert result["qualifying_event_count"] == 2
    assert [event["span_mm"] for event in result["events"]] == pytest.approx([15.0, 11.0])


def test_fails_closed_on_mismatched_inner_outer_support():
    cadir = _cadir()
    cadir["model"]["tessellations"][-1]["vertices"][-1]["z"] = 19
    with pytest.raises(ValueError, match="matching finite support"):
        MODULE.extract_native_roundbend_patch_axes(cadir)
