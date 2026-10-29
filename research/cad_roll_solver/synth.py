"""Synthetic STEP cases with known roll values (exact-recovery check of the roll solver).

Built as real OCP B-reps (E1's ``testparts.arc_profile_part`` turtle profiles, plus revolved cones and a
boolean tube), written to STEP and read back through E1's reader, so the solver sees what it would see
from a CAD export. Truth is analytic, from the construction parameters only.

usage (heavy: through the geometry slot):
    .venv/bin/python -B -m research.cad_roll_solver.synth --out DIR
writes DIR/<case>.step and DIR/truth.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _rot2(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _profile_truth(ops, t, width, start=(0.0, 0.0), heading_deg=0.0):
    """Analytic arcs of an arc_profile_part walk: centre, radii, signed sweep, end radials (3D, axis +Y)."""
    p = np.asarray(start, float)
    h = np.array([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))])
    arcs = []
    for op in ops:
        if op[0] == "line":
            p = p + float(op[1]) * h
            continue
        sw, ri = math.radians(float(op[1])), float(op[2])
        left = sw > 0
        r0 = ri if left else ri + t
        right = np.array([h[1], -h[0]])
        c = p + r0 * (-right if left else right)
        u0 = (p - c) / r0
        u1 = _rot2(u0, sw)
        arcs.append({"centre": [c[0], 0.0, c[1]], "ri": ri, "ro": ri + t, "sweep_deg": abs(float(op[1])),
                     "radials": [[u0[0], 0.0, u0[1]], [u1[0], 0.0, u1[1]]], "axis": [0.0, 1.0, 0.0],
                     "length": float(width), "taper_mm_per_m": 0.0})
        p = c + r0 * u1
        h = _rot2(h, sw)
    return arcs


def _cone_shell(r0: float, semi_deg: float, height: float, t: float, sweep_deg: float):
    """Partial conical shell about +Z: inner skin r(z) = r0 + z tan(semi), outer offset t along the normal."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt

    k = math.tan(math.radians(semi_deg))
    dr = t / math.cos(math.radians(semi_deg))
    pts = [(r0, 0.0), (r0 + k * height, height), (r0 + k * height + dr, height), (r0 + dr, 0.0)]
    poly = BRepBuilderAPI_MakePolygon()
    for x, z in pts:
        poly.Add(gp_Pnt(x, 0.0, z))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
    shape = BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(sweep_deg)).Shape()
    a0 = [1.0, 0.0, 0.0]
    a1 = [math.cos(math.radians(sweep_deg)), math.sin(math.radians(sweep_deg)), 0.0]
    zm = 0.5 * height
    truth = {"centre": [0.0, 0.0, zm], "ri": r0 + k * zm, "ro": r0 + k * zm + dr, "t": t, "sweep_deg": sweep_deg,
             "radials": [a0, a1], "axis": [0.0, 0.0, 1.0], "length": height, "taper_mm_per_m": 1000.0 * k,
             "cone": True}
    return shape, [truth]


def _tube(ri: float, t: float, height: float):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(ax, ri + t, height).Shape()
    inner = BRepPrimAPI_MakeCylinder(ax, ri, height).Shape()
    shape = BRepAlgoAPI_Cut(outer, inner).Shape()
    return shape, [{"centre": [0.0, 0.0, 0.5 * height], "ri": ri, "ro": ri + t, "sweep_deg": 360.0, "radials": None,
                    "axis": [0.0, 0.0, 1.0], "length": height, "taper_mm_per_m": 0.0, "closed": True}]


#: name -> (builder kind, params, expected solver counts)
CASES = {
    "A_cyl_shell_free_ends": ("profile", {"ops": [("arc", 120.0, 40.0)], "t": 2.0, "width": 100.0},
                              {"rolls": 1, "junction_bends": 0, "e1_bends": 0, "ends": ["FREE_END", "FREE_END"]}),
    "B_roll_flat_runouts": ("profile", {"ops": [("line", 30.0), ("arc", 90.0, 50.0), ("line", 40.0)], "t": 2.0,
                                        "width": 80.0},
                            {"rolls": 1, "junction_bends": 0, "e1_bends": 1, "solver_bends": 0,
                             "ends": ["TANGENT_RUNOUT", "TANGENT_RUNOUT"]}),
    "C_bend_and_roll": ("profile", {"ops": [("line", 40.0), ("arc", 90.0, 2.0), ("line", 50.0), ("arc", 70.0, 60.0),
                                            ("line", 30.0)], "t": 2.0, "width": 60.0},
                        {"rolls": 1, "junction_bends": 0, "e1_bends": 2, "solver_bends": 1}),
    "D_junction_reverse": ("profile", {"ops": [("line", 30.0), ("arc", -27.0, 0.5), ("arc", 154.0, 30.0)], "t": 1.5,
                                       "width": 70.0},
                           {"rolls": 1, "junction_bends": 1, "e1_bends": 0, "solver_bends": 1}),
    "E_junction_same_turn": ("profile", {"ops": [("line", 30.0), ("arc", 30.0, 1.0), ("arc", 100.0, 40.0),
                                                 ("line", 25.0)], "t": 1.0, "width": 50.0},
                             {"rolls": 1, "junction_bends": 1, "solver_bends": 1}),
    "F_cone_shell": ("cone", {"r0": 40.0, "semi_deg": 10.0, "height": 80.0, "t": 2.0, "sweep_deg": 140.0},
                     {"rolls": 1, "cones": 1, "ends": ["FREE_END", "FREE_END"]}),
    "G_slit_tube_350": ("profile", {"ops": [("arc", 350.0, 20.0)], "t": 1.5, "width": 60.0},
                        {"rolls": 1, "ends": ["FREE_END", "FREE_END"]}),
    "H_closed_tube": ("tube", {"ri": 25.0, "t": 2.0, "height": 60.0}, {"rolls": 1, "closed": True}),
    "I_control_bends_only": ("profile", {"ops": [("line", 40.0), ("arc", 90.0, 2.0), ("line", 30.0),
                                                 ("arc", -90.0, 3.0), ("line", 20.0)], "t": 2.0, "width": 50.0},
                             {"rolls": 0, "junction_bends": 0, "e1_bends": 2, "solver_bends": 2}),
    "J_control_hem": ("hem", {}, {"rolls": 0, "junction_bends": 0}),
    "K_bump_small_sweep": ("profile", {"ops": [("line", 40.0), ("arc", 8.0, 300.0), ("line", 40.0)], "t": 2.0,
                                       "width": 50.0},
                           {"rolls": 1, "e1_bends": 1, "solver_bends": 0}),
    "L_control_ri_10t": ("profile", {"ops": [("line", 40.0), ("arc", 60.0, 20.0), ("line", 40.0)], "t": 2.0,
                                     "width": 50.0},
                         {"rolls": 0, "e1_bends": 1, "solver_bends": 1}),
}


def build(name: str):
    kind, prm, _ = CASES[name]
    from domains.production.engine.compile import testparts as tp

    if kind == "profile":
        shape, _ = tp.arc_profile_part(prm["ops"], prm["t"], prm["width"])
        arcs = _profile_truth(prm["ops"], prm["t"], prm["width"])
        rolls = [a for a in arcs if a["ri"] > 10.0 * prm["t"]]
        return shape, {"rolls": rolls, "all_arcs": arcs, "t": prm["t"]}
    if kind == "cone":
        shape, rolls = _cone_shell(**prm)
        return shape, {"rolls": rolls, "all_arcs": rolls, "t": prm["t"]}
    if kind == "tube":
        shape, rolls = _tube(**prm)
        return shape, {"rolls": rolls, "all_arcs": rolls, "t": prm["t"]}
    if kind == "hem":
        shape, _ = tp.hem_180_part()
        return shape, {"rolls": [], "all_arcs": [], "t": 1.0}
    raise KeyError(kind)


def main() -> None:
    from domains.production.engine.compile.testparts import write_step

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    for name in CASES:
        shape, tr = build(name)
        write_step(shape, out / f"{name}.step")
        tr["expected"] = CASES[name][2]
        truth[name] = tr
        print(name, "written")
    (out / "truth.json").write_text(json.dumps(truth, indent=1))


if __name__ == "__main__":
    main()
