"""E1 compile: reconciliation of recipe marks with the flange graph (ENGINE_IO §2.1) and blockers."""
from __future__ import annotations

import json

import numpy as np
import pytest

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.compile import compile_part
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.api import load_part_model
from domains.production.engine.compile.config import CompileConfig
from domains.production.engine.compile.dev_recipe import dev_recipe
from domains.production.engine.compile.reconcile import Mark, pair_metrics, reconcile_marks


@pytest.fixture(scope="module")
def u_step(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("u_rec")
    shape, info = tp.u_part(width=40.0)
    return tmp, tp.write_step(shape, tmp / "u.step"), info


def _compile(tmp, step, cands, sku):
    recipe = dev_recipe(step, cands, sku=sku)
    out = tmp / f"out_{sku}"
    return recipe, compile_part(recipe, step, out).compiled, out


def test_exact_native_marks_match_at_zero_cost(u_step):
    tmp, step, info = u_step
    model = load_part_model(step)
    cands = tp.candidates_from_info(info)
    marks = [Mark(f"bu_{i:020d}", tuple(c["line_start"]), tuple(c["line_end"])) for i, c in enumerate(cands)]
    rec = reconcile_marks(model, marks, CompileConfig())
    assert len(rec.matches) == 2 and not rec.blocked
    assert max(rec.costs.values()) < 1e-6


def test_missing_mark_is_unmarked_bend_and_the_blocked_part_is_still_written(u_step):
    tmp, step, info = u_step
    recipe, c, out = _compile(tmp, step, tp.candidates_from_info(info)[:1], "SYN-UMISS")
    assert c.status == "BLOCKED" and c.blocker_codes == ("UNMARKED_BEND",)
    issue = next(i for i in c.issues if i.code == "UNMARKED_BEND")
    assert issue.graph_bend_id and [g.graph_bend_id for g in c.unmarked_graph_bends] == [issue.graph_bend_id]
    assert c.stages == ()
    reloaded = CompiledPartV1.model_validate_json((out / "compiled_part.json").read_bytes())
    assert reloaded.status == "BLOCKED"


def test_mark_outside_every_bend_is_mark_not_in_flange_graph(u_step):
    tmp, step, info = u_step
    cands = tp.candidates_from_info(info) + [{"line_start": [30.0, 0.0, 15.0], "line_end": [30.0, 40.0, 15.0],
                                              "target_angle": 90.0, "target_radius": 2.0}]
    recipe, c, _ = _compile(tmp, step, cands, "SYN-UEXTRA")
    assert c.status == "BLOCKED" and c.blocker_codes == ("MARK_NOT_IN_FLANGE_GRAPH",)
    issue = next(i for i in c.issues if i.code == "MARK_NOT_IN_FLANGE_GRAPH")
    assert issue.bend_uid == recipe.bends[-1].bend_uid


def test_two_marks_on_one_bend_are_ambiguous(tmp_path):
    shape, info = tp.window_u_part()
    step = tp.write_step(shape, tmp_path / "window.step")
    # the extractor style: each bend reported as two pieces either side of the window
    pieces = tp.candidates_from_info(info, pieces=[(0.0, 35.0), (45.0, 80.0)])
    _, c, _ = _compile(tmp_path, step, pieces, "SYN-WPIECES")
    assert c.status == "BLOCKED" and "MARK_AMBIGUOUS" in c.blocker_codes
    full = tp.candidates_from_info(info)
    _, c2, _ = _compile(tmp_path, step, full, "SYN-WFULL")
    assert c2.status == "COMPILED", c2.blocker_codes


@pytest.mark.parametrize("mutation", ["radial_4mm", "rotated_5deg", "outside_axially"])
def test_feasibility_rejects_displaced_marks(u_step, mutation):
    _, step, info = u_step
    model = load_part_model(step)
    cfg = CompileConfig()
    c = tp.candidates_from_info(info)[0]
    a, b = np.array(c["line_start"]), np.array(c["line_end"])
    bend = min(model.bends.values(), key=lambda g: pair_metrics(g, Mark("x", tuple(a), tuple(b)), cfg)[1])
    ok, cost, _ = pair_metrics(bend, Mark("x", tuple(a), tuple(b)), cfg)
    assert ok and cost < 1e-6
    if mutation == "radial_4mm":
        shift = 4.0 * (bend.r_parent + bend.r_child) / np.linalg.norm(bend.r_parent + bend.r_child)
        a2, b2 = a + shift, b + shift
    elif mutation == "rotated_5deg":
        mid = 0.5 * (a + b)
        d = (b - a) / np.linalg.norm(b - a)
        perp = bend.r_parent
        d2 = np.cos(np.radians(5)) * d + np.sin(np.radians(5)) * perp
        half = 0.5 * np.linalg.norm(b - a)
        a2, b2 = mid - half * d2, mid + half * d2
    else:
        d = (b - a) / np.linalg.norm(b - a)
        a2, b2 = a + d * 60.0, b + d * 60.0
    ok2, _, info2 = pair_metrics(bend, Mark("x", tuple(a2), tuple(b2)), cfg)
    assert not ok2, info2


def test_unmarked_unsupported_bend_is_a_rigid_feature_not_a_blocker(tmp_path):
    shape, info = tp.hem_part()
    step = tp.write_step(shape, tmp_path / "hem.step")
    supported = [c for c, b in zip(tp.candidates_from_info(info), info.bends) if b.sweep_deg < 170]
    _, c, _ = _compile(tmp_path, step, supported, "SYN-HEMR")
    assert c.status == "COMPILED", c.blocker_codes
    feats = [f for f in c.unsupported_features if f.note.startswith("RIGID_FOLLOWER_OF:")]
    assert len(feats) == 1 and feats[0].feature_kind == "HEM"
    assert not c.unmarked_graph_bends
    warn = [i for i in c.issues if i.code == "UNSUPPORTED_FEATURE"]
    assert warn and all(i.severity == "WARNING" for i in warn)
    # the hem's panel is carried rigidly: identical transform to its carrier in every stage
    carrier = feats[0].note.split(":", 1)[1].split()[0]
    for s in c.stages:
        assert s.panel_transforms[feats[0].panel_id] == s.panel_transforms[carrier]


def test_marked_unsupported_bend_compiles_with_unavailable_nominals(tmp_path):
    shape, info = tp.hem_part()
    step = tp.write_step(shape, tmp_path / "hem2.step")
    recipe, c, _ = _compile(tmp_path, step, tp.candidates_from_info(info), "SYN-HEMM")
    assert c.status == "COMPILED"
    hem = next(b for b in c.bends if not b.supported)
    assert hem.unsupported_feature == "HEM"
    noms = [x for x in c.characteristics if x.bend_uid == hem.bend_uid]
    assert noms and all(x.nominal is None and x.unavailable_reason == "UNSUPPORTED_FEATURE" for x in noms)
    assert all(x.tolerance is not None for x in noms if x.kind in ("FLANGE_LENGTH", "INCLUDED_ANGLE"))
    rec = next(f for f in c.unsupported_features if f.bend_uid == hem.bend_uid)
    assert set(rec.affected_characteristic_ids) >= {f"{hem.bend_uid}/INCLUDED_ANGLE"}
    assert json.loads(c.model_dump_json())["status"] == "COMPILED"


def test_wizard_helpers_report_the_graph_and_the_draft_reconciliation(u_step):
    from domains.production.engine.compile import analyze_cad, reconcile_draft

    _, step, info = u_step
    summary = analyze_cad(step)
    assert [b["graph_bend_id"] for b in summary["graph_bends"]] == ["G1", "G2"]
    assert all(b["included_deg"] == pytest.approx(90.0) and b["unsupported"] is None for b in summary["graph_bends"])
    assert summary["datum_panel_id"] in {p["panel_id"] for p in summary["panels"] if p["is_datum"]}
    cands = tp.candidates_from_info(info)
    marks = [Mark(f"bu_{i:020d}", tuple(c["line_start"]), tuple(c["line_end"])) for i, c in enumerate(cands[:1])]
    rec = reconcile_draft(step, marks)
    assert rec["blocked"] and len(rec["unmarked_bend"]) == 1 and len(rec["matches"]) == 1


def test_features_list_the_characteristics_they_make_unavailable(tmp_path):
    # an unmarked hem is the far boundary of the base side of the other bend
    shape, info = tp.hem_part()
    step = tp.write_step(shape, tmp_path / "hem3.step")
    supported = [c for c, b in zip(tp.candidates_from_info(info), info.bends) if b.sweep_deg < 170]
    _, c, _ = _compile(tmp_path, step, supported, "SYN-HEMA")
    rigid = next(f for f in c.unsupported_features if f.note.startswith("RIGID_FOLLOWER_OF:"))
    unavailable = {x.characteristic_id for x in c.characteristics if x.unavailable_reason == "UNSUPPORTED_FEATURE"}
    assert unavailable and set(rigid.affected_characteristic_ids) == unavailable


def test_a_duplicate_mark_beyond_the_margin_is_ambiguous_not_missing(u_step):
    # a second mark on bend 1, 1.5 mm off the first (beyond the 1 mm runner-up margin): it is
    # feasible only for a graph bend another mark already holds, i.e. a duplicate
    tmp, step, info = u_step
    cands = tp.candidates_from_info(info)
    first = cands[0]
    model = load_part_model(step)
    bend = min(model.bends.values(), key=lambda g: pair_metrics(
        g, Mark("x", tuple(first["line_start"]), tuple(first["line_end"])), CompileConfig())[1])
    shift = 1.5 * bend.bisector()
    dup = dict(first, line_start=(np.array(first["line_start"]) + shift).tolist(),
               line_end=(np.array(first["line_end"]) + shift).tolist())
    ok, cost, _ = pair_metrics(bend, Mark("d", tuple(dup["line_start"]), tuple(dup["line_end"])), CompileConfig())
    assert ok and cost == pytest.approx(1.5, abs=1e-6)
    recipe, c, _ = _compile(tmp, step, cands + [dup], "SYN-UDUP")
    assert c.status == "BLOCKED" and c.blocker_codes == ("MARK_AMBIGUOUS",)
    issue = next(i for i in c.issues if i.code == "MARK_AMBIGUOUS")
    assert issue.graph_bend_id == bend.graph_bend_id
    assert recipe.bends[0].bend_uid in issue.detail and recipe.bends[-1].bend_uid in issue.detail
    assert not [i for i in c.issues if i.code == "MARK_NOT_IN_FLANGE_GRAPH"]
