"""Engine fixes after the one-time held-out evaluation (2026-10-19): E1 compile robustness and E3 false refusals.

Every fixture is synthetic (OCP B-reps built in-test, analytic datum models, E2 smoke cases): no real CAD, no
scans, no labels. Each test pins one general rule:

- E1: a multi-body STEP is read as its dominant solid; a stepped bend skin keeps the dominant skin level; bend
  skins that are not coaxial still pair into one bend and unroll flat; DEV policy 2 marks graph bends no extractor
  found and drops rolled-only candidates.
- E3: the round-hole fill test uses the inscribed circle (an obround's own bore wall is not fill); a hole holding
  installed hardware (material proud of the sheet, concentric) is occupied, not filled; a mirror-symmetric CAD is
  never refused as a mirrored part, a chiral one still is.

Run: PYT tests/test_prod_engine_fixes_2026_10_19.py -q
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.brep import CadReadError, load_step, shape_to_model
from domains.production.engine.compile.config import CompileConfig
from domains.production.engine.compile.flange_graph import build_flange_graph
from domains.production.engine.inspect.config import InspectConfig

CFG = CompileConfig()


# --------------------------------------------------------------------------- helpers
def _offset_skin_l(a: float = 60.0, b: float = 40.0, t: float = 4.0, ri: float = 8.0, ro: float = 14.0,
                   width: float = 50.0):
    """90 deg L whose outer bend arc is NOT coaxial with the inner one (centre (ro, ro) vs (t+ri, t+ri)).

    Profile in (x, z), extruded along +Y: base z in [0, t], wall x in [0, t]."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.gp import gp_Pnt, gp_Vec

    def P(x, z):
        return gp_Pnt(float(x), 0.0, float(z))

    ci = t + ri
    s = math.sqrt(0.5)
    pts_out = [(ro, 0.0), (ro - ro * s, ro - ro * s), (0.0, ro)]
    pts_in = [(t, ci), (ci - ri * s, ci - ri * s), (ci, t)]
    edges = [GC_MakeSegment(P(a, 0.0), P(*pts_out[0])).Value(),
             GC_MakeArcOfCircle(P(*pts_out[0]), P(*pts_out[1]), P(*pts_out[2])).Value(),
             GC_MakeSegment(P(*pts_out[2]), P(0.0, b)).Value(),
             GC_MakeSegment(P(0.0, b), P(t, b)).Value(),
             GC_MakeSegment(P(t, b), P(*pts_in[0])).Value(),
             GC_MakeArcOfCircle(P(*pts_in[0]), P(*pts_in[1]), P(*pts_in[2])).Value(),
             GC_MakeSegment(P(*pts_in[2]), P(a, t)).Value(),
             GC_MakeSegment(P(a, t), P(a, 0.0)).Value()]
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(BRepBuilderAPI_MakeEdge(e).Edge())
    face = BRepBuilderAPI_MakeFace(wire.Wire(), True).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0.0, float(width), 0.0)).Shape()


def _box(lo, hi):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    return BRepPrimAPI_MakeBox(gp_Pnt(*lo), gp_Pnt(*hi)).Shape()


# --------------------------------------------------------------------------- E1: multi-body STEP
def test_a_multi_body_step_is_read_as_its_dominant_solid_and_refused_only_when_ambiguous(tmp_path):
    shape, _ = tp.l_part(a=60.0, b=40.0, t=3.0, width=40.0)  # ~ 11.6 cm3
    bracket = _box((0.0, 100.0, 0.0), (20.0, 120.0, 4.0))  # 1.6 cm3, ~14 % (a weld-on bracket / hardware)
    model = load_step(tp.write_step(tp.compound_of(shape, bracket), tmp_path / "dominant.step"))
    assert model.solid_count == 2
    assert any(n.startswith("MULTIPLE_SOLID_BODIES_DOMINANT") for n in model.notes)
    graph = build_flange_graph(model, CFG)
    assert len(graph.bends) == 1 and len(graph.panels) == 2  # the L alone
    # two comparable bodies: which one is the part is ambiguous -> refused
    other = tp._transform(tp.l_part(a=60.0, b=40.0, t=3.0, width=40.0)[0], translate=(0.0, 100.0, 0.0))
    with pytest.raises(CadReadError) as exc:
        load_step(tp.write_step(tp.compound_of(shape, other), tmp_path / "two.step"))
    assert exc.value.reason == "MULTIPLE_SOLID_BODIES"


# --------------------------------------------------------------------------- E1: stepped bend skin
def test_a_stepped_bend_skin_keeps_the_dominant_skin_level():
    verts = [(0.0, 4.0), (40.0, 4.0), (40.0, 34.0)]  # skin 0 (inside) identical for both strips
    thick, _ = tp.profile_part(verts, 4.0, 13.0, 12.0)
    thin, _ = tp.profile_part(verts, 2.5, 13.0, 8.0)
    shape = tp._fuse(thick, tp._transform(thin, translate=(0.0, 12.0, 0.0)))
    graph = build_flange_graph(shape_to_model(shape), CFG)  # used to raise FLANGE_GRAPH_NOT_A_TREE (non-coplanar)
    assert len(graph.bends) == 1 and len(graph.panels) == 2
    b = next(iter(graph.bends.values()))
    assert b.unsupported is None and b.ri == pytest.approx(13.0, abs=1e-3) and b.ro == pytest.approx(17.0, abs=1e-3)
    assert graph.thickness == pytest.approx(4.0, abs=1e-3)
    assert any(f.reason.startswith("stepped bend skin") for f in graph.features)


# --------------------------------------------------------------------------- E1: offset (non-coaxial) bend skins
def test_bend_skins_that_are_not_coaxial_pair_into_one_bend_and_compile_flat(tmp_path):
    from domains.production.engine.compile import compile_part
    from domains.production.engine.compile.dev_recipe import dev_candidates, dev_recipe

    step = tp.write_step(_offset_skin_l(), tmp_path / "offset.step")
    graph = build_flange_graph(load_step(step), CFG)
    assert len(graph.bends) == 1 and len(graph.panels) == 2
    b = next(iter(graph.bends.values()))
    # described by its inner cylinder and the outer skin planes' distance from that axis: Ri + t
    assert b.ri == pytest.approx(8.0, abs=1e-3) and b.ro == pytest.approx(12.0, abs=1e-3)
    assert b.outer_radius == pytest.approx(14.0, abs=1e-3) and b.unsupported is None
    assert b.sweep_deg == pytest.approx(90.0, abs=1e-3)
    cands, report = dev_candidates(step, sources=())  # no extractor: E1's own graph supplies the mark
    assert [c["label"] for c in cands] == ["E1_GRAPH_G1"] and report["added_from_e1_graph"] == ["E1_GRAPH_G1"]
    res = compile_part(dev_recipe(step, cands, sku="SYN-OFFSET"), step, tmp_path / "out")
    assert res.compiled.status == "COMPILED", [(i.code, i.detail) for i in res.compiled.issues]  # S0 unrolls flat


def test_a_coaxial_part_is_unchanged_by_the_offset_skin_pass():
    shape, info = tp.l_part(width=30.0)
    b = next(iter(build_flange_graph(shape_to_model(shape), CFG).bends.values()))
    assert b.outer_radius is None and b.ro - b.ri == pytest.approx(info.thickness, abs=1e-6)


# --------------------------------------------------------------------------- E1: DEV policy 2
def test_dev_policy_marks_a_graph_bend_the_extractors_missed_and_keeps_extractor_marks(tmp_path):
    from domains.production.engine.compile.dev_recipe import DEV_POLICY_VERSION, merge_graph_duplicates

    assert DEV_POLICY_VERSION == "E1-DEV-POLICY-2"
    shape, info = tp.u_part(width=30.0)
    step = tp.write_step(shape, tmp_path / "u.step")
    marks = tp.candidates_from_info(info)
    # both bends marked by an extractor: nothing added, the marks are kept as given
    out, rep = merge_graph_duplicates(step, marks)
    assert rep["added_from_e1_graph"] == [] and [c["line_start"] for c in out] == [m["line_start"] for m in marks]
    # one extractor mark missing: E1's graph adds exactly the missing bend
    out1, rep1 = merge_graph_duplicates(step, marks[:1])
    assert len(out1) == 2 and len(rep1["added_from_e1_graph"]) == 1


def test_a_part_whose_only_bend_is_a_roll_yields_no_recipe(tmp_path):
    from domains.production.engine.compile.dev_recipe import dev_candidates, merge_graph_duplicates

    shape, infos = tp.arc_profile_part([("arc", 170.0, 40.0)], 1.0, 50.0)  # a rolled half shell, no flange
    step = tp.write_step(shape, tmp_path / "roll.step")
    roll = tp.arc_candidates(infos, 50.0)
    out, rep = merge_graph_duplicates(step, roll)
    assert out == [] and [e["rule"] for e in rep["excluded_by_unsupported_rule"]] == ["ROLLED"]
    cands, _ = dev_candidates(step, sources=())
    assert cands == []


# --------------------------------------------------------------------------- E3: hole fill on an obround
def _datum_with_hole(hole):
    from domains.production.engine.inspect.registration import DatumModel

    poly = np.array([[-25.0, -20.0], [25.0, -20.0], [25.0, 20.0], [-25.0, 20.0]])
    return DatumModel(p0=np.zeros(3), n0=np.array([0.0, 0.0, 1.0]), f1=np.array([1.0, 0.0, 0.0]),
                      f2=np.array([0.0, 1.0, 0.0]), t=6.0, polygon=poly, free=np.zeros(4, dtype=bool), holes=[hole],
                      area=2000.0, bbox=(-25.0, -20.0, 25.0, 20.0), hole_blocked=[np.zeros((0, 2))])


def _obround(r: float = 5.85, half: float = 2.0, n: int = 24) -> np.ndarray:
    a = np.linspace(-math.pi / 2, math.pi / 2, n)
    return np.vstack([np.c_[half + r * np.cos(a), r * np.sin(a)], np.c_[-half - r * np.cos(a), -r * np.sin(a)]])


def _grid(lo, hi, step=0.2):
    g0, g1 = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
    xx, yy = np.meshgrid(g0, g1)
    return np.c_[xx.ravel(), yy.ravel()]


def _hole_count(model, uv, proud=None, spacing=0.2):
    from domains.production.engine.inspect.registration import _inplane_constraints

    con = _inplane_constraints(uv, model, [], InspectConfig(), 0.0, spacing, np.zeros(2), 10.0, wide=False,
                               proud=proud)
    return con["hole_mismatch"], con["holes_hardware"]


def test_an_obround_holes_own_bore_wall_is_not_fill_but_material_across_it_is():
    from matplotlib.path import Path as MplPath

    from domains.production.engine.inspect.registration import _round_hole

    hole = _obround()
    assert _round_hole(hole)  # near-round: judged by the radial (disc) test
    model = _datum_with_hole(hole)
    sheet = _grid((-15.0, -15.0), (15.0, 15.0))
    around = sheet[~MplPath(hole).contains_points(sheet)]
    # the bore's cut wall on the straight sides (|y| = r), seen at many heights: inside a disc of the mean
    # radius minus 1 mm, outside the inscribed one
    xs = np.arange(-2.0, 2.0, 0.02)
    wall = np.vstack([np.c_[xs, np.full_like(xs, s * 5.85)] for s in (-1, 1) for _ in range(6)])
    assert _hole_count(model, np.vstack([around, wall]))[0] == 0
    # a part without the hole: sheet material across the opening
    assert _hole_count(model, sheet)[0] == 1


# --------------------------------------------------------------------------- E3: installed hardware
def test_a_hole_with_a_pressed_in_stud_is_occupied_not_filled():
    ang = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
    hole = np.c_[3.0 * np.cos(ang), 3.0 * np.sin(ang)]
    model = _datum_with_hole(hole)
    sheet = _grid((-12.0, -12.0), (12.0, 12.0))
    r = np.hypot(sheet[:, 0], sheet[:, 1])
    shank = sheet[r < 3.0]  # the stud's shank / nut barrel fills the hole at the skin heights
    proud_stud = sheet[r < 2.5]  # its body stands out of the sheet, concentric
    # hardware: occupied, not a wrong part
    assert _hole_count(model, sheet, proud=proud_stud) == (0, 1)
    # the same material without anything proud of the sheet: a hole that is not cut -> filled
    assert _hole_count(model, sheet, proud=np.zeros((0, 2))) == (1, 0)
    assert _hole_count(model, sheet) == (1, 0)
    # material proud of the sheet but not concentric (a strip crossing one side): still filled
    strip = sheet[(sheet[:, 0] > 1.5) & (r < 3.0)]
    assert _hole_count(model, sheet, proud=strip) == (1, 0)
    # an open hole with a stud is not a mismatch either way
    assert _hole_count(model, sheet[r >= 3.0], proud=proud_stud)[0] == 0
    assert len(shank) > 0


# --------------------------------------------------------------------------- E3: mirror-symmetric CAD
def test_mirror_self_maps_are_the_certified_mirror_candidates_of_the_decoded_stage_only():
    from domains.production.engine.inspect.engine import _mirror_self_maps

    m = np.diag([-1.0, 1.0, 1.0, 1.0]).tolist()
    r = np.diag([-1.0, -1.0, 1.0, 1.0]).tolist()

    def stage(sid, cands):
        return SimpleNamespace(stage_model_id=sid, symmetry=SimpleNamespace(threshold_rms_mm=0.3, candidates=[
            SimpleNamespace(kind=k, transform=t, rms_mm=e) for k, t, e in cands]))

    # a flat blank that is its own mirror image, a formed part that is not (a chiral bracket)
    part = SimpleNamespace(compiled=SimpleNamespace(stages=[stage("S0", [("MIRROR", m, 0.02)]),
                                                            stage("S2", [("MIRROR", m, 4.9), ("ROTATION", r, 0.01)])]))
    assert _mirror_self_maps(part, "S2") == []  # the scan shows S2: a mirrored S2 is a mirrored part
    assert len(_mirror_self_maps(part, "S0")) == 1  # a mirrored blank is the blank
    assert _mirror_self_maps(part, None) == []  # no exact decoded stage: nothing certified to rely on
    sym = SimpleNamespace(compiled=SimpleNamespace(stages=[stage("S1", [("MIRROR", m, 0.05), ("MIRROR", m, 0.04)])]))
    out = _mirror_self_maps(sym, "S1")
    assert len(out) == 1 and np.linalg.det(out[0][:3, :3]) < 0


def test_the_mirror_verdict_needs_asymmetric_evidence_on_a_mirror_symmetric_shape():
    from domains.production.engine.inspect.engine import _mirror_refusal

    # chiral shape: the mirrored fit explaining clearly more is enough (and holes must not contradict it)
    assert _mirror_refusal(2.01, 1.87, 0, 0, symmetric=False, margin=0.08).startswith("MIRROR_FITS_BETTER")
    assert _mirror_refusal(1.90, 1.87, 0, 0, symmetric=False, margin=0.08) is None
    assert _mirror_refusal(2.01, 1.87, 2, 0, symmetric=False, margin=0.08) is None
    # mirror-symmetric shape: the converted proper pose explains as well; a better total is no evidence ...
    assert _mirror_refusal(2.05, 1.40, 0, 0, symmetric=True, margin=0.08) is None
    # ... the datum holes are: the mirrored fit finds them open where every proper pose finds some filled
    assert _mirror_refusal(2.05, 1.97, 0, 2, symmetric=True, margin=0.08) == "MIRROR_HOLES_FIT_BETTER:0<2"


def _usym_case(root):
    """A hole-free U channel (its own mirror image), compiled, and a perfect scan of its final stage."""
    from domains.production.engine.compile import compile_part
    from domains.production.engine.compile.dev_recipe import dev_recipe
    from domains.production.engine.synthetic.ply import write_ply

    root.mkdir(parents=True, exist_ok=True)
    shape, info = tp.u_part(base=60.0, h1=30.0, h2=30.0, width=80.0)
    step = tp.write_step(shape, root / "u.step")
    rec = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-USYM")
    (root / "recipe.json").write_text(rec.model_dump_json())
    assert compile_part(rec, step, root / "compiled").compiled.status == "COMPILED"
    cp = json.loads((root / "compiled" / "compiled_part.json").read_text())
    pts = np.load(root / "compiled" / cp["stages"][-1]["sample_cloud"]["rel_path"])["points"]
    pts = pts + np.random.default_rng(7).normal(0.0, 0.02, pts.shape)
    a, b = 0.6, 0.3
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    write_ply(root / "scan.ply", pts @ R.T + np.array([100.0, -50.0, 20.0]))
    return root


@pytest.fixture(scope="module")
def smoke_cases(tmp_path_factory):
    import dataclasses

    from contracts.production.part_recipe import PartRecipeV1
    from domains.production.engine.compile import compile_part
    from domains.production.engine.synthetic.cases import generate_case, plan_cases

    root = tmp_path_factory.mktemp("fix_cases")
    specs = plan_cases("smoke", "DEV")
    out = {"usym": _usym_case(root / "FIX-usym")}
    for name, family, feats in (("hat", "HAT", {}), ("ucs", "U_CHANNEL", {"symmetric": True})):
        spec = next(s for s in specs if s.family == family)
        spec = dataclasses.replace(spec, case_id=f"FIX-{name}", scenario="COMPLETE_NOMINAL", exact=True, plan="FULL",
                                   features=dataclasses.replace(spec.features, **feats))
        generate_case(spec, root)
        d = root / spec.case_id
        recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
        assert compile_part(recipe, d / "part.step", d / "compiled", flat_path=d / "flat.dxf").status == "COMPILED"
        out[name] = d
    return out


def _run_mirrored(case, tmp_path):
    import hashlib
    import os
    import subprocess
    import sys
    from pathlib import Path

    from domains.production.engine.synthetic.ply import read_ply, write_ply

    repo = Path(__file__).resolve().parents[1]
    pts, _, _ = read_ply(case / "scan.ply")
    scan = tmp_path / f"{case.name}_mirrored.ply"
    write_ply(scan, pts * np.array([-1.0, 1.0, 1.0]))
    out = tmp_path / f"{case.name}_run"
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(case / "compiled"),
           "--recipe", str(case / "recipe.json"), "--scan", str(scan), "--out", str(out),
           "--scan-sha256", hashlib.sha256(scan.read_bytes()).hexdigest()]
    env = {**os.environ, "PYTHONPATH": f"{repo}:{repo / 'backend'}", "OMP_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=300, check=True)
    return json.loads((out / "result.json").read_text())


def test_mirrored_scans_true_mirrors_stay_refused_a_mirror_symmetric_part_does_not(smoke_cases, tmp_path):
    # a hole-free U is its own mirror image: its mirrored scan is the part, never a mirrored part
    usym = _run_mirrored(smoke_cases["usym"], tmp_path)
    assert usym["inspection_validity"]["state"] == "VALID"
    assert np.linalg.det(np.asarray(usym["inspection_validity"]["registration"]["raw_to_frame"])[:3, :3]) > 0
    # E2's "symmetric" U carries datum holes that break the mirror symmetry: a true mirrored part, refused
    ucs = _run_mirrored(smoke_cases["ucs"], tmp_path)
    assert ucs["inspection_validity"]["state"] in ("MIRRORED_PART_SUSPECTED", "WRONG_PART_SUSPECTED")
    assert ucs["summary"]["state"] != "ALL_MEASURED_IN_TOLERANCE"
    # a chiral part (the asymmetric hat): refused
    hat = _run_mirrored(smoke_cases["hat"], tmp_path)
    assert hat["inspection_validity"]["state"] in ("MIRRORED_PART_SUSPECTED", "WRONG_PART_SUSPECTED",
                                                   "UNTRUSTED_REGISTRATION")
    assert hat["summary"]["state"] != "ALL_MEASURED_IN_TOLERANCE"
