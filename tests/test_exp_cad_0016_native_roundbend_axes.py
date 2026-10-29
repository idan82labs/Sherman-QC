import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0016_native_roundbend_axes.py"
SPEC = importlib.util.spec_from_file_location("exp16", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _cadir(*, bends="1", thickness="2", second_radius=1.0):
    return {
        "native": {
            "sldprt": {
                "custom_properties": [
                    {"name": "Bends", "values": [{"value": bends}]},
                    {"name": "Sheet Metal Thickness", "values": [{"value": thickness}]},
                ],
                "arenas": {
                    "features": [
                        {"kind": "RoundBend", "name": "RoundBend1", "suppressed": False},
                    ]
                },
            }
        },
        "model": {
            "faces": [
                {"id": "face:outer", "surface": "surf:outer"},
                {"id": "face:inner", "surface": "surf:inner"},
            ],
            "surfaces": [
                {"id": "surf:outer", "geometry": {"kind": "cylinder", "origin": {"x": 5, "y": 3, "z": 100}, "axis": {"x": 0, "y": 0, "z": 1}, "radius": 3}},
                {"id": "surf:inner", "geometry": {"kind": "cylinder", "origin": {"x": 5, "y": 3, "z": -100}, "axis": {"x": 0, "y": 0, "z": -1}, "radius": second_radius}},
            ],
            "tessellations": [
                {"body": "body:1", "faces": ["face:outer"], "vertices": [{"x": 5, "y": 6, "z": -7}, {"x": 5, "y": 6, "z": 11}]},
                {"body": "body:1", "faces": ["face:inner"], "vertices": [{"x": 5, "y": 4, "z": -7}, {"x": 5, "y": 4, "z": 11}]},
            ],
        },
    }


def test_extracts_one_finite_source_native_axis():
    result = MODULE.extract_native_roundbend_axes(_cadir())
    assert result["source_bend_count"] == 1
    assert result["qualifying_pair_count"] == 1
    event = result["events"][0]
    assert event["radii_mm"] == [1.0, 3.0]
    assert event["line_start_mm"] == pytest.approx([5.0, 3.0, -7.0])
    assert event["line_end_mm"] == pytest.approx([5.0, 3.0, 11.0])


def test_fails_closed_when_property_and_feature_count_disagree():
    with pytest.raises(ValueError, match="disagrees"):
        MODULE.extract_native_roundbend_axes(_cadir(bends="2"))


def test_fails_closed_without_thickness_pair():
    with pytest.raises(ValueError, match="geometry yielded 0 pairs"):
        MODULE.extract_native_roundbend_axes(_cadir(second_radius=1.5))
