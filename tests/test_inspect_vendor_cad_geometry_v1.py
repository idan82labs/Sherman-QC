"""Pure inspector summaries/fakes: no native imports or physical input reads."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from backend.vendor_unfold_preview import UnfoldUnavailable
from scripts.inspect_vendor_cad_geometry_v1 import describe_surface, trim_knot_diagnostics


class Spline:
    def NbUPoles(self): return 3
    def NbVPoles(self): return 4
    def NbUKnots(self): return 2
    def NbVKnots(self): return 2
    def UKnot(self, i): return (0., 1.)[i-1]
    def VKnot(self, i): return (3., 9.)[i-1]
    def UDegree(self): return 2
    def VDegree(self): return 3
    def IsUPeriodic(self): return False
    def IsVPeriodic(self): return False
    def Weight(self, i, j): return (1., .7, 1.)[i-1]


class Surface:
    def GetType(self): return "GeomAbs_BSplineSurface"
    def BSpline(self): return Spline()
    def FirstUParameter(self): return -1e-12
    def LastUParameter(self): return 1.
    def FirstVParameter(self): return 3.
    def LastVParameter(self): return 9.


def test_trim_overshoot_retained_with_no_clamping():
    result = trim_knot_diagnostics((-1e-12, 1.+1e-12), (0., 1.))
    assert result["lower_overshoot"] == 1e-12
    assert result["upper_overshoot"] == pytest.approx(1e-12, abs=1e-16)
    assert result["strictly_contained"] is False and result["clamping_performed"] is False


def test_nonfinite_bounds_are_json_safe_not_zero():
    result = trim_knot_diagnostics((0., float("inf")), (0., 1.))
    assert result["trim"] == [0., None] and result["finite"] is False
    assert result["upper_overshoot"] is None
    json.dumps(result, allow_nan=False)


def test_describe_refusal_preserves_bounds_and_weight_coupling_without_retry():
    calls = []
    def refuse(surface):
        calls.append(surface)
        raise UnfoldUnavailable("INVALID_NATIVE_EXTRUSION_INTERVAL")
    surface = Surface()
    result = describe_surface(surface, 4, proof=refuse)
    assert calls == [surface]
    assert result["surface_proof_reason"] == "INVALID_NATIVE_EXTRUSION_INTERVAL"
    assert result["spline"]["u_interval"]["lower_overshoot"] == 1e-12
    assert result["spline"]["weights_identical_across_v"] is True
    assert result["spline"]["weights_identical_across_u"] is False
    json.dumps(result, allow_nan=False)


def test_certificate_arrays_become_json_safe_and_native_curve_is_not_exported():
    result = describe_surface(Surface(), 0, proof=lambda _: {
        "kind": "extruded_boundary", "pole_cost": 12,
        "axis": np.array([1., 0., 0.]), "v0": np.float64(0.),
        "extrusion_curve": object()})
    assert result["surface_certificate"]["axis"] == [1., 0., 0.]
    assert "extrusion_curve" not in result["surface_certificate"]
    assert result["surface_proof_status"] == "PROVED_CARRIER_ONLY"
    json.dumps(result, allow_nan=False)


def test_inspector_is_api_only_no_meshing_or_import_time_native_execution():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts/inspect_vendor_cad_geometry_v1.py").read_text())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not getattr(n, "module", "").startswith("OCP") for n in top_imports)
    source = ast.unparse(tree)
    assert "BRepMesh" not in source and "ArgumentParser" not in source
    assert "if __name__" not in source and "indexed_native_identity" in source
    assert "INSPECTOR_EDGE_USE_BUDGET_EXCEEDED" in source
    assert "INSPECTOR_FACE_SOLID_OWNERSHIP_MISMATCH" in source
