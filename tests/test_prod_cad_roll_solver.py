"""CAD roll solver (research/cad_roll_solver): exact recovery, bend/roll rule, E1 bend preservation.

Fixtures are real OCP B-reps built in-test (research/cad_roll_solver/synth.py, which uses E1's
testparts turtle profiles plus a revolved cone and a boolean tube), written to STEP in tmp_path and read
back through E1's reader. No real CAD, no scans, no server imports. Expected values are analytic.
"""
from __future__ import annotations

import math

import pytest

from research.cad_roll_solver import synth
from research.cad_roll_solver.solver import (
    SCHEMA,
    _arc_overlap,
    _arcs_cover,
    _covering_arc,
    _union_arcs,
    solve_step,
)
from research.cad_roll_solver.validate_synth import compare

pytest.importorskip("OCP")


def _solve_case(name, tmp_path):
    from domains.production.engine.compile.testparts import write_step

    shape, truth = synth.build(name)
    truth["expected"] = synth.CASES[name][2]
    p = write_step(shape, tmp_path / f"{name}.step")
    doc, _, _ = solve_step(p)
    return doc, truth


@pytest.mark.parametrize("name", sorted(synth.CASES))
def test_synthetic_exact_recovery(name, tmp_path):
    doc, truth = _solve_case(name, tmp_path)
    res = compare(doc, truth)
    assert res["pass"], res
    for r in res["rolls"]:
        assert r["exact"] and r["max_err_mm"] <= 1e-4 and r["max_err_deg"] <= 1e-3, r


def test_rule_boundary_ri_equal_10t_is_bend(tmp_path):
    doc, _ = _solve_case("L_control_ri_10t", tmp_path)
    assert doc["summary"]["rolls"] == 0
    assert [b["id"] for b in doc["solver_bends"]] == [b["graph_bend_id"] for b in doc["e1_bends"]]


def test_e1_rolled_a_bend_becomes_roll_with_e1_panels(tmp_path):
    doc, _ = _solve_case("B_roll_flat_runouts", tmp_path)
    (roll,) = doc["rolls"]
    assert roll["subtype"] == "E1_ROLLED_A_GRAPH_BEND" and roll["e1"]["class"] == "ROLLED"
    assert sorted(roll["neighbours"]) == ["P0", "P1"]
    assert all(ro["end_type"] == "TANGENT_RUNOUT" for ro in roll["runouts"])
    assert doc["solver_bends"] == []
    names = {c["check"]: c["status"] for c in roll["checks"]}
    assert names["RUNOUT_TANGENT_MM"] == "PASS" and names["SWEEP_VS_RUNOUT_PLANES_DEG"] == "PASS"


def test_ordinary_bends_preserved_and_no_rolls(tmp_path):
    doc, _ = _solve_case("I_control_bends_only", tmp_path)
    assert doc["summary"]["rolls"] == 0 and doc["summary"]["junction_bends"] == 0
    assert [b["id"] for b in doc["solver_bends"]] == [b["graph_bend_id"] for b in doc["e1_bends"]]
    for b in doc["e1_bends"]:
        x = b["solver_cross_check"]
        assert x["paired"] and abs(x["ri_diff_mm"]) < 1e-6 and abs(x["sweep_diff_deg"]) < 1e-6


def test_junction_bend_and_coherent_graph(tmp_path):
    doc, _ = _solve_case("D_junction_reverse", tmp_path)
    (j,) = doc["junction_bends"]
    assert j["r_inner_mm"] == pytest.approx(0.5, abs=1e-6) and j["sweep_deg"] == pytest.approx(27.0, abs=1e-4)
    assert "R1" in j["neighbours"]
    g = doc["graph"]
    assert g["coherent"] and g["components"] == 1
    assert {e["type"] for e in g["edges"]} == {"JUNCTION_BEND"}


def test_mixed_bend_and_roll_graph(tmp_path):
    doc, _ = _solve_case("C_bend_and_roll", tmp_path)
    g = doc["graph"]
    assert g["coherent"]
    types = sorted(e["type"] for e in g["edges"])
    assert types == ["BEND", "RUNOUT", "RUNOUT"]  # E1's ROLLED (a) bend edge is replaced by the roll node
    (b,) = doc["solver_bends"]
    assert b["source"] == "E1" and b["class"] is None


def test_cone_flag_and_taper(tmp_path):
    doc, truth = _solve_case("F_cone_shell", tmp_path)
    (r,) = doc["rolls"]
    assert r["is_cone"] and r["surface"] == "cone"
    assert r["taper"]["mm_per_m"] == pytest.approx(1000 * math.tan(math.radians(10.0)), abs=1e-3)
    assert r["thickness_mm"] == pytest.approx(2.0, abs=1e-5)


def test_closed_tube(tmp_path):
    doc, _ = _solve_case("H_closed_tube", tmp_path)
    (r,) = doc["rolls"]
    assert r["closed"] and r["sweep_deg"] == pytest.approx(360.0) and r["runouts"] == []


def test_schema_fields_present(tmp_path):
    doc, _ = _solve_case("E_junction_same_turn", tmp_path)
    assert doc["schema"] == SCHEMA
    r = doc["rolls"][0]
    for k in ("id", "kind", "is_cone", "axis", "r_inner_mm", "r_outer_mm", "thickness_mm", "sweep_deg",
              "axial_length_mm", "chord_inner_mm", "opening_mm", "taper", "convexity", "runouts", "neighbours",
              "stations", "checks", "e1"):
        assert k in r, k
    for ro in r["runouts"]:
        for skin in ("inner", "outer"):
            assert {"start_mm", "end_mm", "length_mm", "class", "neighbour_ids"} <= set(ro[skin])
        assert ro["end_type"] in {"TANGENT_RUNOUT", "FREE_END", "MIXED"}


def test_arc_helpers():
    two_pi = 2 * math.pi
    s, w = _covering_arc([math.radians(a) for a in (350, 10, 20)])
    assert math.degrees(w) == pytest.approx(30.0) and math.degrees(s) == pytest.approx(350.0)
    u = _union_arcs([(0.0, math.radians(100)), (math.radians(100), math.radians(100))], 1e-6)
    assert len(u) == 1 and math.degrees(u[0][1]) == pytest.approx(200.0)
    assert _union_arcs([(0.0, math.pi), (math.pi, math.pi)], 1e-6) == [(0.0, two_pi)]
    assert math.degrees(_arcs_cover([(0.0, math.radians(30)), (math.radians(60), math.radians(30))])[1]) == pytest.approx(90.0)
    assert _arc_overlap([(math.radians(350), math.radians(20))], [(0.0, math.radians(5))]) == pytest.approx(math.radians(5))
