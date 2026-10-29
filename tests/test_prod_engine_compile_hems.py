"""E1 compile: hems at and past 180 deg (open, closed, teardrop), and unexpected-geometry robustness.

Fixtures are exact OCP solids built from arc angles (testparts.arc_profile_part), so a 180 deg
hem has exactly antiparallel end radials and a teardrop a reflex sweep. Expected values are
analytic from the fixture parameters.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from contracts.production.canonical import canonical_json_bytes
from contracts.production.compiled_part import CompiledPartV1, check_compiled_against_recipe
from domains.production.engine.compile import analyze_cad, compile_part, stage_points_from_compiled
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.api import load_part_model
from domains.production.engine.compile.config import CompileConfig
from domains.production.engine.compile.dev_recipe import dev_recipe
from domains.production.engine.compile.geometry import rotation_matrix
from domains.production.engine.compile.reconcile import Mark, pair_metrics

WIDTH = 30.0
CASES = {
    "open_180": lambda: tp.hem_180_part(ri=0.5, width=WIDTH),
    "closed_180": lambda: tp.hem_180_part(ri=0.05, width=WIDTH),
    "teardrop_200": lambda: tp.teardrop_part(sweep=200.0, width=WIDTH),
}


@pytest.fixture(scope="module")
def steps(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hems")
    out = {}
    for name, make in CASES.items():
        shape, infos = make()
        out[name] = (tp.write_step(shape, tmp / f"{name}.step"), infos)
    return tmp, out


def _compile(tmp, step, cands, sku):
    recipe = dev_recipe(step, cands, sku=sku)
    return recipe, compile_part(recipe, step, tmp / f"out_{sku}").compiled


def _out(tmp, sku):
    return tmp / f"out_{sku}"


@pytest.mark.parametrize("name, sweep", [("open_180", 180.0), ("closed_180", 180.0), ("teardrop_200", 200.0)])
def test_hems_at_and_past_180_deg_are_hem_features_with_finite_loci(steps, name, sweep):
    _, cases = steps
    step, infos = cases[name]
    summary = analyze_cad(step)  # used to raise "cannot normalise a zero vector" at 180 deg
    bends = sorted(summary["graph_bends"], key=lambda b: b["sweep_deg"])
    assert [round(b["sweep_deg"], 6) for b in bends] == [90.0, pytest.approx(sweep, abs=1e-6)]
    assert [b["unsupported"] for b in bends] == [None, "HEM"]  # a teardrop is a HEM by its true (reflex) sweep
    hem = load_part_model(step)
    (mb,) = [b for b in hem.bends.values() if b.unsupported == "HEM"]
    ref = infos[1]
    loci = mb.loci()
    centre = loci["cylinder_axis"][0]
    # the apex lines stand in for the virtual sharps (at infinity for 180 deg); all lie on the bisector
    for name_, radius in (("oml_virtual_sharp", ref.ro), ("outer_apex", ref.ro), ("iml_virtual_sharp", ref.ri),
                          ("inner_apex", ref.ri)):
        a, e = loci[name_]
        assert np.all(np.isfinite(a)) and np.linalg.norm(a - centre) == pytest.approx(radius, abs=1e-9), name_
        assert (a - centre) / radius == pytest.approx(ref.mid_dir, abs=1e-6), name_
        assert np.linalg.norm(e - a) == pytest.approx(WIDTH, abs=1e-6)
    # orientation: +sweep about the axis takes the parent radial to the child radial through the material
    r = rotation_matrix(mb.axis_dir, mb.sweep_rad)
    assert r @ mb.r_parent == pytest.approx(mb.r_child, abs=1e-6)
    half = rotation_matrix(mb.axis_dir, mb.sweep_rad / 2) @ mb.r_parent
    assert half == pytest.approx(ref.mid_dir, abs=1e-6)
    assert abs(float(mb.p_out @ mb.r_parent)) < 1e-9 and abs(float(mb.c_out @ mb.r_child)) < 1e-9


@pytest.mark.parametrize("name", list(CASES))
def test_unmarked_hem_is_a_rigid_follower_and_the_part_compiles(steps, name):
    tmp, cases = steps
    step, infos = cases[name]
    sku = f"SYN-HU-{name.upper()}"[:30]
    recipe, c = _compile(tmp, step, tp.arc_candidates(infos, WIDTH, include=[1]), sku)
    assert c.status == "COMPILED", [(i.code, i.detail) for i in c.issues]
    assert check_compiled_against_recipe(c, recipe) == []
    rigid = [f for f in c.unsupported_features if f.note.startswith("RIGID_FOLLOWER_OF:")]
    assert len(rigid) == 1 and rigid[0].feature_kind == "HEM"
    # the supported bend's base side ends on the hem: unsupported far boundary, listed by the hem record
    (b,) = c.bends
    side = next(x for x in c.characteristics if x.bend_uid == b.bend_uid and x.kind == "FLANGE_LENGTH"
                and x.nominal is None)
    assert side.unavailable_reason == "UNSUPPORTED_FEATURE"
    assert side.characteristic_id in rigid[0].affected_characteristic_ids
    assert c.diagnostics["s0_flatness_max_mm"] < 1e-6
    carrier = rigid[0].note.split(":", 1)[1].split()[0]
    for s in c.stages:
        assert s.panel_transforms[rigid[0].panel_id] == s.panel_transforms[carrier]
    # E3 rebuilds every stage cloud from the compiled part, following the rigid follower
    final = np.load(_out(tmp, sku) / c.stages[-1].sample_cloud.rel_path, allow_pickle=False)
    for s in c.stages:
        ref = np.load(_out(tmp, sku) / s.sample_cloud.rel_path, allow_pickle=False)
        pts, _ = stage_points_from_compiled(c, final["points"], final["normals"], final["segment_kind"],
                                            final["segment_index"], s.formed_bend_uids)
        assert np.max(np.abs(pts - ref["points"].astype(np.float64))) < 2e-3


@pytest.mark.parametrize("name", ["open_180", "closed_180"])
def test_marked_180_deg_hem_is_staged_at_the_published_sweep(steps, name):
    tmp, cases = steps
    step, infos = cases[name]
    sku = f"SYN-HM-{name.upper()}"[:30]
    recipe, c = _compile(tmp, step, tp.arc_candidates(infos, WIDTH), sku)
    assert c.status == "COMPILED", [(i.code, i.detail) for i in c.issues]
    assert check_compiled_against_recipe(c, recipe) == []
    hem = next(b for b in c.bends if not b.supported)
    assert hem.unsupported_feature == "HEM" and hem.sweep_deg == pytest.approx(179.999, abs=1e-9)
    assert all(x.nominal is None and x.unavailable_reason == "UNSUPPORTED_FEATURE"
               for x in c.characteristics if x.bend_uid == hem.bend_uid)
    a, b = (np.array(s.outward_dir) for s in hem.sides)
    assert float(a @ b) > 0.999999  # both flanges point the same way; the published sweep still names them
    # the supported bend's side toward the hem has no nominal and names the hem as its boundary
    sup = next(b for b in c.bends if b.supported)
    toward = next(s for s in sup.sides if s.adjacent_bend_uid == hem.bend_uid)
    nominal = c.nominal(f"{sup.bend_uid}/FLANGE_LENGTH/{toward.side}")
    assert toward.boundary_kind == "ADJACENT_BEND"
    assert nominal.nominal is None and nominal.unavailable_reason == "UNSUPPORTED_FEATURE"
    hem_rec = next(f for f in c.unsupported_features if f.bend_uid == hem.bend_uid)
    assert nominal.characteristic_id in hem_rec.affected_characteristic_ids
    assert "apex" in hem_rec.note
    # S0 is flat to well below the 0.05 mm self-check, and every stage cloud is reproducible from the
    # compiled part alone (the staged sweep equals the published one)
    assert c.diagnostics["s0_flatness_max_mm"] < 1e-3
    out = _out(tmp, sku)
    final = np.load(out / c.stages[-1].sample_cloud.rel_path, allow_pickle=False)
    for s in c.stages:
        ref = np.load(out / s.sample_cloud.rel_path, allow_pickle=False)
        pts, _ = stage_points_from_compiled(c, final["points"], final["normals"], final["segment_kind"],
                                            final["segment_index"], s.formed_bend_uids)
        assert np.max(np.abs(pts - ref["points"].astype(np.float64))) < 2e-3


def test_marked_teardrop_is_blocked_with_a_reason_not_a_crash(steps):
    tmp, cases = steps
    step, infos = cases["teardrop_200"]
    _, c = _compile(tmp, step, tp.arc_candidates(infos, WIDTH), "SYN-HM-TEARDROP")
    assert c.status == "BLOCKED" and c.blocker_codes == ("STAGE_MODEL_FAILED",)
    detail = next(i.detail for i in c.issues if i.code == "STAGE_MODEL_FAILED")
    assert "reflex" in detail and "200.000" in detail and "remove the mark" in detail


def test_hem_marks_are_feasible_across_the_whole_sector(steps):
    _, cases = steps
    cfg = CompileConfig()
    for name in ("open_180", "teardrop_200"):
        step, infos = cases[name]
        model = load_part_model(step)
        hem = next(b for b in model.bends.values() if b.unsupported == "HEM")
        ref = infos[1]
        centre = ref.center
        for frac in (0.05, 0.5, 0.95):  # near the parent end, the apex, near the child end (> 180 for the teardrop)
            radial = rotation_matrix(hem.axis_dir, frac * hem.sweep_rad) @ hem.r_parent
            p = centre + (ref.ri + 0.5 * (ref.ro - ref.ri)) * radial
            mark = Mark("x", tuple(p), tuple(p + np.array([0.0, WIDTH, 0.0])))
            ok, cost, info = pair_metrics(hem, mark, cfg)
            assert ok, (name, frac, info)
            if frac == 0.5:
                assert cost < 1e-6  # the apex mark is the native mid-radius half-turn line


def test_unexpected_exception_is_a_blocker_with_its_phase(tmp_path, monkeypatch):
    shape, info = tp.l_part(width=30.0)
    step = tp.write_step(shape, tmp_path / "l.step")
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-BOOM")

    def boom(*args, **kwargs):
        raise ValueError("synthetic geometry failure")

    monkeypatch.setattr("domains.production.engine.compile.compiler.flange_length_nominal", boom)
    out = tmp_path / "out"
    c = compile_part(recipe, step, out).compiled
    assert c.status == "BLOCKED" and c.blocker_codes == ("STAGE_MODEL_FAILED",)
    detail = next(i.detail for i in c.issues if i.code == "STAGE_MODEL_FAILED")
    assert "'nominals'" in detail and "synthetic geometry failure" in detail
    reloaded = CompiledPartV1.model_validate_json((out / "compiled_part.json").read_bytes())
    assert reloaded.status == "BLOCKED" and "Traceback" in (out / "compile.log").read_text()

    def graph_boom(*args, **kwargs):
        raise ZeroDivisionError("graph")

    monkeypatch.setattr("domains.production.engine.compile.compiler.build_flange_graph", graph_boom)
    c2 = compile_part(recipe, step, tmp_path / "out2").compiled
    assert c2.blocker_codes == ("FLANGE_GRAPH_NOT_A_TREE",)
    assert "'flange_graph'" in next(i.detail for i in c2.issues)


def test_cli_on_an_exact_180_deg_hem_never_exits_3(steps):
    import os
    import subprocess
    import sys
    from pathlib import Path

    tmp, cases = steps
    step, infos = cases["closed_180"]
    repo = Path(__file__).resolve().parents[1]
    for marked in (True, False):
        recipe = dev_recipe(step, tp.arc_candidates(infos, WIDTH, include=None if marked else [1]),
                            sku=f"SYN-HCLI-{int(marked)}")
        rpath = tmp / f"hcli_{int(marked)}.json"
        rpath.write_bytes(canonical_json_bytes(recipe))
        env = dict(os.environ, PYTHONPATH=f"{repo}:{repo / 'backend'}", OMP_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-m", "domains.production.engine.compile", "--recipe", str(rpath),
                            "--cad", str(step), "--out", str(tmp / f"hcli_out_{int(marked)}")],
                           cwd=repo, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
        kind, payload = r.stdout.strip().splitlines()[-1].split("\t", 1)
        assert kind == "RESULT" and json.loads(payload)["status"] == "COMPILED"


@pytest.mark.parametrize("sweep", [178.9, 179.1, 179.9])
def test_the_179_deg_code_path_boundary_is_continuous(tmp_path, sweep):
    # below 179 deg the bisector is unit(r1 + r2); at and above it the material's mid radial:
    # both sides of the switch must give the exact apex, and a marked hem must stage flat
    ops = [("line", 20.0), ("arc", 90.0, 1.0), ("line", 40.0), ("arc", sweep, 0.8), ("line", 10.0)]
    shape, infos = tp.arc_profile_part(ops, 1.0, WIDTH, start=(0.0, 20.0), heading_deg=-90.0)
    step = tp.write_step(shape, tmp_path / "h.step")
    hem = next(b for b in load_part_model(step).bends.values() if b.unsupported == "HEM")
    loci = hem.loci()
    apex, axis0 = loci["outer_apex"][0], loci["cylinder_axis"][0]
    assert (apex - axis0) / infos[1].ro == pytest.approx(infos[1].mid_dir, abs=1e-9)
    assert np.linalg.norm(loci["oml_virtual_sharp"][0] - axis0) == pytest.approx(infos[1].ro, abs=1e-9)
    recipe = dev_recipe(step, tp.arc_candidates(infos, WIDTH), sku=f"SYN-H{int(sweep * 10)}")
    c = compile_part(recipe, step, tmp_path / "out").compiled
    assert c.status == "COMPILED" and c.diagnostics["s0_flatness_max_mm"] < 1e-6
