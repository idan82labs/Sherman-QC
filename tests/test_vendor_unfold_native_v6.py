"""Pure native-index protocol fakes and AST gates; no native geometry imports."""
import ast
from pathlib import Path

import pytest

from backend.vendor_unfold_native_v6 import indexed_native_identity, read_native_v6
from backend.vendor_unfold_preview import MAX_EDGE_USES, UnfoldUnavailable
from backend.vendor_surface_normalization_v2 import surface_proof_v2


class Edge:
    calls = 0
    def __init__(self, identity, location=0, orientation=1):
        self.identity, self.location, self.orientation = identity, location, orientation
    def IsSame(self, other):
        Edge.calls += 1
        return (self.identity, self.location) == (other.identity, other.location)


class Indexed:
    def __init__(self): self.by_key, self.items = {}, []
    def Add(self, edge):
        key = (edge.identity, edge.location)
        if key not in self.by_key:
            self.items.append(edge); self.by_key[key] = len(self.items)
        return self.by_key[key]
    def Extent(self): return len(self.items)
    def FindKey(self, index): return self.items[index-1]


def test_first_encounter_enumeration_ignores_orientation_not_location():
    index = Indexed()
    assert indexed_native_identity(index, Edge(9), 0) == (0, True)
    assert indexed_native_identity(index, Edge(9, orientation=-1), 1) == (0, False)
    assert indexed_native_identity(index, Edge(9, location=3), 1) == (1, True)
    assert indexed_native_identity(index, Edge(8), 2) == (2, True)


def test_indexed_python_verification_cost_is_one_per_use_not_quadratic():
    Edge.calls = 0; index = Indexed(); count = 0
    for i in range(2000):
        _, fresh = indexed_native_identity(index, Edge(i), count)
        count += int(fresh)
    for i in range(2000):
        _, fresh = indexed_native_identity(index, Edge(i, orientation=-1), count)
        assert not fresh
    assert Edge.calls == 4000 and count == 2000


def test_map_collision_cannot_replace_native_same_proof():
    index = Indexed(); indexed_native_identity(index, Edge(1), 0)
    index.by_key[(2, 0)] = 1
    with pytest.raises(UnfoldUnavailable, match="NATIVE_INDEXED_EDGE_IDENTITY_MISMATCH"):
        indexed_native_identity(index, Edge(2), 1)


def test_map_extent_and_native_ids_must_match_first_encounter_ledger():
    index = Indexed(); indexed_native_identity(index, Edge(1), 0)
    with pytest.raises(UnfoldUnavailable, match="NATIVE_INDEXED_EDGE_IDENTITY_MISMATCH"):
        indexed_native_identity(index, Edge(1), 0)


@pytest.mark.parametrize("count", [-1, MAX_EDGE_USES, True])
def test_use_budget_preserved(count):
    with pytest.raises(UnfoldUnavailable, match="EDGE_INCIDENCE_BUDGET_EXCEEDED"):
        indexed_native_identity(Indexed(), Edge(1), count)


def test_reader_default_v2_chart_proof_and_axis_extractor_injection_seam():
    assert read_native_v6.__kwdefaults__["surface_proof"] is surface_proof_v2
    assert read_native_v6.__kwdefaults__["axis_extractor"] is None


def test_incumbent_native_guards_and_observer_remain_in_separate_reader():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "backend/vendor_unfold_native_v6.py").read_text())
    reasons = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {
        "SINGLE_VALID_SOLID_REQUIRED", "NATIVE_FACE_SOLID_OWNERSHIP_MISMATCH",
        "FACE_BUDGET_EXCEEDED", "EDGE_INCIDENCE_BUDGET_EXCEEDED",
        "NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED", "DUPLICATE_NATIVE_CIRCULAR_RAIL_USE",
        "TESSELLATION_PREFLIGHT_BUDGET_EXCEEDED", "TRIANGLE_BUDGET_EXCEEDED",
        "VERTEX_BUDGET_EXCEEDED", "NATIVE_EXTRUSION_UV_OWNERSHIP_MISMATCH",
    } <= reasons
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not getattr(n, "module", "").startswith("OCP") for n in imports)
    source = ast.unparse(tree)
    assert "axis_extractor(path)" in source and "on_face(native_face, face)" in source
    assert "for i, previous in enumerate(native_edges)" not in source
