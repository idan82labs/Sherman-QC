"""Engine fixes round 3 (2026-10-19, owner decisions D4, D2, D3): synthetic fixtures only (E2 smoke cases and
in-test analytic data); no real CAD, no scans, no labels.

- D4 (contract amendment ENGINE_IO-A1, e3-1.11.0): a flange bounded by an adjacent bend that is NOT_YET_FORMED is
  NOT_APPLICABLE(NOT_APPLICABLE_AT_STAGE); a flange whose adjacent bend IS formed is still measured; the contract
  rejects the reason anywhere else; per-bend scoring V3 counts such a side as satisfied.

Run: PYT tests/test_prod_engine_fixes3_2026_10_19.py -q
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from contracts.production.inspection_result import ShopfloorInspectionResultV1

REPO = Path(__file__).resolve().parents[1]


def _run(case: Path, out: Path, scan: Path | None = None) -> dict:
    scan = scan or case / "scan.ply"
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(case / "compiled"),
           "--recipe", str(case / "recipe.json"), "--scan", str(scan), "--out", str(out),
           "--scan-sha256", hashlib.sha256(scan.read_bytes()).hexdigest()]
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=300, check=True)
    return json.loads((out / "result.json").read_text())


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    from contracts.production.part_recipe import PartRecipeV1
    from domains.production.engine.compile import compile_part
    from domains.production.engine.synthetic.cases import generate_case, plan_cases

    root = tmp_path_factory.mktemp("fix3_cases")
    specs = plan_cases("smoke", "DEV")
    out = {}
    for name, family, scenario in (("hat_stage", "HAT", "STAGE_K"), ("hat", "HAT", "COMPLETE_NOMINAL")):
        spec = next(s for s in specs if s.family == family)
        spec = dataclasses.replace(spec, case_id=f"FIX3-{name}", scenario=scenario,
                                   exact=scenario == "COMPLETE_NOMINAL", plan="FULL")
        generate_case(spec, root)
        d = root / spec.case_id
        recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
        assert compile_part(recipe, d / "part.step", d / "compiled", flat_path=d / "flat.dxf").status == "COMPILED"
        out[name] = (d, _run(d, root / f"{name}_run"))
    return out


def _flanges(res):
    state = {b["bend_uid"]: b["state"] for b in res["bends"]}
    for b in res["bends"]:
        for c in b["characteristics"]:
            if c["kind"] == "FLANGE_LENGTH":
                yield b, c, state.get(c.get("adjacent_bend_uid"))


# --------------------------------------------------------------------------- D4
def test_d4_a_flange_next_to_an_unformed_bend_is_not_applicable_at_stage_a_formed_neighbour_is_measured(staged):
    _, res = staged["hat_stage"]
    assert res["engine"]["engine_version"] == "e3-1.14.0"
    assert res["inspection_validity"]["state"] == "VALID"
    na = formed_nb = 0
    for b, c, adj in _flanges(res):
        if b["state"] not in ("FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"):
            continue
        if c["boundary_kind"] == "ADJACENT_BEND" and adj == "NOT_YET_FORMED":
            assert (c["state"], c["reason"], c["decision"]) == ("NOT_APPLICABLE", "NOT_APPLICABLE_AT_STAGE",
                                                                "NOT_APPLICABLE"), c["characteristic_id"]
            assert c["required"] is False
            na += 1
        elif c["boundary_kind"] == "ADJACENT_BEND" and adj == "FORMED":
            assert c["state"] == "MEASURED", c["characteristic_id"]
            formed_nb += 1
        else:
            assert c["reason"] != "NOT_APPLICABLE_AT_STAGE"
    assert na >= 1, "the staged hat must have a formed bend bounded by an unformed one"
    # a staged part never goes green, with or without the amendment
    assert res["summary"]["state"] != "ALL_MEASURED_IN_TOLERANCE"
    ShopfloorInspectionResultV1.model_validate(res)


def test_d4_a_complete_part_has_no_stage_inapplicable_flange_and_every_adjacent_flange_is_measured(staged):
    _, res = staged["hat"]
    seen = 0
    for b, c, adj in _flanges(res):
        assert c["reason"] != "NOT_APPLICABLE_AT_STAGE"
        if c["boundary_kind"] == "ADJACENT_BEND":
            assert adj == "FORMED" and c["state"] == "MEASURED", c["characteristic_id"]
            seen += 1
    assert seen >= 2


def _mutated(res, pick):
    doc = json.loads(json.dumps(res))
    for b in doc["bends"]:
        for c in b["characteristics"]:
            if pick(b, c):
                for k in ("actual", "deviation", "uncertainty", "stations", "evidence", "exceedance_ratio"):
                    c.pop(k, None)
                c.update(state="NOT_APPLICABLE", decision="NOT_APPLICABLE", reason="NOT_APPLICABLE_AT_STAGE",
                         required=False)
                return doc
    raise AssertionError("no characteristic picked")


def test_d4_the_contract_rejects_stage_inapplicable_flanges_unless_the_adjacent_bend_is_unformed(staged):
    _, res = staged["hat_stage"]
    state = {b["bend_uid"]: b["state"] for b in res["bends"]}
    # a flange whose adjacent bend is FORMED cannot be skipped
    if any(c["kind"] == "FLANGE_LENGTH" and c["state"] == "MEASURED" and state.get(c.get("adjacent_bend_uid")) == "FORMED"
           for b in res["bends"] for c in b["characteristics"]):
        doc = _mutated(res, lambda b, c: c["kind"] == "FLANGE_LENGTH" and c["state"] == "MEASURED"
                       and state.get(c.get("adjacent_bend_uid")) == "FORMED")
        with pytest.raises(Exception, match="adjacent bend is NOT_YET_FORMED"):
            ShopfloorInspectionResultV1.model_validate(doc)
    # a FREE_EDGE flange (no adjacent bend) cannot be skipped either
    doc = _mutated(res, lambda b, c: c["kind"] == "FLANGE_LENGTH" and c.get("adjacent_bend_uid") is None)
    with pytest.raises(Exception, match="adjacent bend is NOT_YET_FORMED"):
        ShopfloorInspectionResultV1.model_validate(doc)
    # an angle never takes the reason
    doc = _mutated(res, lambda b, c: c["kind"] == "INCLUDED_ANGLE" and c["state"] == "MEASURED")
    with pytest.raises(Exception, match="cannot be NOT_APPLICABLE"):
        ShopfloorInspectionResultV1.model_validate(doc)


def _v3():
    spec = importlib.util.spec_from_file_location("per_bend_scoring_v3", REPO / "scripts/production/per_bend_scoring_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bend(flanges, state="FORMED"):
    chs = [{"kind": "FLANGE_LENGTH", **f} for f in flanges]
    chs += [{"kind": "INCLUDED_ANGLE", "state": "MEASURED", "decision": "PASS"},
            {"kind": "BEND_LINE_POSITION", "state": "MEASURED", "decision": "PASS"}]
    return {"bend_uid": "b1", "label": "B1", "state": state, "characteristics": chs}


def test_d4_scoring_v3_counts_a_stage_inapplicable_flange_as_satisfied_and_nothing_else():
    v3 = _v3()
    ok = {"state": "MEASURED", "decision": "PASS"}
    na_stage = {"state": "NOT_APPLICABLE", "decision": "NOT_APPLICABLE", "reason": "NOT_APPLICABLE_AT_STAGE"}
    adj_unknown = {"state": "NOT_MEASURED", "decision": "UNKNOWN", "reason": "ADJACENT_BEND_NOT_MEASURED"}
    na_recipe = {"state": "NOT_APPLICABLE", "decision": "NOT_APPLICABLE", "reason": "NOT_INSPECTED_BY_RECIPE"}

    def row(fl, truth="FORMED", state="FORMED"):
        r = v3.per_bend_rows("c", "f", {"bends": [_bend(fl, state)]}, lambda u: truth, lambda u: v3.MATCH, "G")
        return r[0]

    r = row([ok, na_stage])
    assert r["correct_strict"] and r["correct_riser_aware"] and not r["false_pass_proxy"]
    assert not row([ok, adj_unknown])["correct_riser_aware"]           # an UNKNOWN neighbour stays a miss
    assert not row([ok, na_recipe])["correct_strict"]                  # only the stage reason is amended
    assert row([na_stage, na_stage], truth="NOT_FORMED", state="FORMED")["false_pass_proxy"]  # NOT_FORMED unchanged
    assert v3.DEFINITION["id"] == "PER_BEND_GIVEN_CAD_V3"


# --------------------------------------------------------------------------- D2: free datum (49207000 class)
def _free_datum_compiled(case: Path, dest: Path) -> Path:
    """The case's compiled part with every datum cut edge unusable for registration and the datum's holes removed:
    a datum that fixes none of its in-plane directions (the flanges' own cut ends stay usable)."""
    import shutil

    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.compiled_part import CompiledPartV1

    comp = dest / "compiled"
    shutil.copytree(case / "compiled", comp)
    doc = json.loads((comp / "compiled_part.json").read_text())
    for panel in doc["panels"]:
        if panel["is_datum"]:
            for fe in panel["free_edges"]:
                fe["registration_usable"] = False
            panel["holes_mm"] = []
    doc.pop("compiled_part_sha256")
    (comp / "compiled_part.json").write_bytes(canonical_json_bytes(CompiledPartV1.finalize(**doc)))
    return comp


def _run_with(compiled: Path, recipe: Path, scan: Path, out: Path) -> dict:
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(compiled),
           "--recipe", str(recipe), "--scan", str(scan), "--out", str(out),
           "--scan-sha256", hashlib.sha256(scan.read_bytes()).hexdigest()]
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600, check=True)
    return json.loads((out / "result.json").read_text())


@pytest.fixture(scope="module")
def free_datum(staged, tmp_path_factory):
    """A chiral TWO_DIRECTION_TAB (base with flanges on two adjacent sides) whose datum fixes no in-plane direction,
    its own scan, its mirrored scan, and a wrong part (the HAT case's scan)."""
    import numpy as np

    from contracts.production.part_recipe import PartRecipeV1
    from domains.production.engine.compile import compile_part
    from domains.production.engine.synthetic.cases import generate_case, plan_cases
    from domains.production.engine.synthetic.ply import read_ply, write_ply

    root = tmp_path_factory.mktemp("fix3_free")
    spec = next(s for s in plan_cases("smoke", "DEV") if s.family == "TWO_DIRECTION_TAB")
    spec = dataclasses.replace(spec, case_id="FIX3-td", scenario="COMPLETE_NOMINAL", exact=True, plan="FULL")
    generate_case(spec, root)
    case = root / spec.case_id
    recipe = PartRecipeV1.model_validate_json((case / "recipe.json").read_bytes())
    assert compile_part(recipe, case / "part.step", case / "compiled", flat_path=case / "flat.dxf").status == "COMPILED"
    comp = _free_datum_compiled(case, root)
    pts, _, _ = read_ply(case / "scan.ply")
    mirrored = root / "mirrored.ply"
    write_ply(mirrored, pts * np.array([-1.0, 1.0, 1.0]))
    return {"case": case, "compiled": comp, "mirrored": mirrored, "root": root, "wrong": staged["hat"][0] / "scan.ply"}


def _explained(out: Path) -> float:
    line = next(x for x in (out / "engine.log").read_text().splitlines() if "best datum score=" in x)
    return float(line.split("explained=")[1].split()[0])


def test_d2_a_datum_that_fixes_no_in_plane_direction_triggers_the_free_datum_search_and_registers(free_datum):
    out = free_datum["root"] / "true"
    res = _run_with(free_datum["compiled"], free_datum["case"] / "recipe.json", free_datum["case"] / "scan.ply", out)
    assert "free datum (NO_IN_PLANE_CONSTRAINT)" in (out / "engine.log").read_text()
    assert res["inspection_validity"]["state"] == "VALID"
    assert _explained(out) >= 0.95
    assert all(b["state"] == "FORMED" for b in res["bends"])
    assert all(c["state"] == "MEASURED" for b in res["bends"] for c in b["characteristics"] if c["kind"] == "INCLUDED_ANGLE")


def test_d2_a_mirrored_chiral_part_is_still_refused_with_a_free_datum(free_datum):
    out = free_datum["root"] / "mirrored"
    res = _run_with(free_datum["compiled"], free_datum["case"] / "recipe.json", free_datum["mirrored"], out)
    assert "mirrored free datum (NO_IN_PLANE_CONSTRAINT)" in (out / "engine.log").read_text()
    assert res["inspection_validity"]["state"] == "MIRRORED_PART_SUSPECTED"
    assert res["summary"]["state"] != "ALL_MEASURED_IN_TOLERANCE"
    assert not any(c["decision"] == "PASS" for b in res["bends"] for c in b["characteristics"])


def test_d2_a_wrong_part_is_still_refused_with_a_free_datum(free_datum):
    out = free_datum["root"] / "wrong"
    res = _run_with(free_datum["compiled"], free_datum["case"] / "recipe.json", free_datum["wrong"], out)
    assert "free datum (NO_IN_PLANE_CONSTRAINT)" in (out / "engine.log").read_text()
    assert res["inspection_validity"]["state"] == "WRONG_PART_SUSPECTED"
    assert not any(c["decision"] == "PASS" for b in res["bends"] for c in b["characteristics"])


@pytest.mark.skipif(os.environ.get("SHERMAN_D2_CONSUMED_49207") is None,
                    reason="consumed real scan (FRESH 49207000): run only on purpose, see fixes3/tools/check_49207.sh")
def test_d2_the_49207000_weldment_registers_and_explains_at_least_0_85():
    out = Path(os.environ["SHERMAN_D2_CONSUMED_49207"])
    res = json.loads((out / "result.json").read_text())
    assert res["inspection_validity"]["state"] == "VALID" and _explained(out) >= 0.85


def test_d2_trigger_rules_and_symmetric_images():
    import numpy as np

    from domains.production.engine.inspect.config import InspectConfig
    from domains.production.engine.inspect.engine import _free_datum_trigger

    class Fit:
        def __init__(self, k):
            self.undetermined = np.zeros((k, 3))

    class Model:
        def __init__(self, sym):
            self._s = sym

        def rotational_symmetries(self):
            return ([np.pi] if self._s else []), np.zeros(2)

    cfg = InspectConfig()
    assert _free_datum_trigger(Model(False), [Fit(3)], cfg) == "NO_IN_PLANE_CONSTRAINT"
    assert _free_datum_trigger(Model(True), [Fit(1)], cfg) == "SYMMETRIC_DATUM"
    assert _free_datum_trigger(Model(False), [Fit(1)], cfg) is None      # asymmetric datum, one free direction
    assert _free_datum_trigger(Model(True), [Fit(0)], cfg) is None       # symmetric but its edges fix the pose
    assert _free_datum_trigger(Model(False), [Fit(3)], dataclasses.replace(cfg, free_datum_rank_k=0)) is None


def test_d2_a_rectangle_datum_has_its_half_turn_image_and_an_asymmetric_one_has_none():
    import numpy as np

    from domains.production.engine.inspect.registration import DatumModel

    def model(poly):
        poly = np.asarray(poly, dtype=float)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        return DatumModel(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                          2.0, poly, np.zeros(len(poly), dtype=bool), [], 1.0,
                          (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))

    rect = model([(0, 0), (60, 0), (60, 30), (0, 30)])
    angles, centre = rect.rotational_symmetries()
    assert [round(a, 6) for a in angles] == [round(np.pi, 6)] and np.allclose(centre, (30, 15), atol=0.5)
    T = np.eye(4)
    T[:3, 3] = (5.0, -3.0, 0.0)
    (img,) = rect.symmetric_images(T)
    # the image maps the datum's material onto itself: the rectangle's corners go to corners
    S = img @ np.linalg.inv(T)  # the symmetry in the CAD frame: the rectangle's corners go to corners
    corners = np.array([[0, 0, 0, 1], [60, 0, 0, 1], [60, 30, 0, 1], [0, 30, 0, 1.0]])
    moved = (S @ corners.T).T[:, :3]
    for c in moved:
        assert np.min(np.linalg.norm(corners[:, :3] - c, axis=1)) < 1.0  # raster centroid; the fine search refines it
    assert np.linalg.norm(moved[0] - corners[0, :3]) > 1.0  # a half turn, not the identity
    ell = model([(0, 0), (60, 0), (60, 10), (10, 10), (10, 30), (0, 30)])
    assert ell.rotational_symmetries()[0] == []


def test_d2_flange_end_ranking_keeps_a_hypothesis_unless_the_refined_pose_scores_higher(monkeypatch):
    import numpy as np

    import domains.production.engine.inspect.engine as E
    from domains.production.engine.inspect.config import InspectConfig

    class Fit:
        undetermined = np.zeros((3, 3))
        score = 0.5

    def hyp(total):
        h = E._Hyp(fit=Fit())
        h.art, h.total = object(), total
        return h

    a, b = hyp(1.0), hyp(0.9)
    better, worse = hyp(1.2), hyp(0.1)
    refined = {id(a): worse, id(b): better}
    monkeypatch.setattr(E, "_refine_free_directions", lambda h, *x, **k: refined[id(h)])

    class Budget:
        def left(self):
            return 1.0

    out = E._free_datum_rank([a, b], None, None, None, None, InspectConfig(), Budget(), lambda m: None, 0.5)
    assert out[0] is better and out[1] is a        # b replaced by its better refinement; a kept (its was worse)


# --------------------------------------------------------------------------- D3: tied symmetric twins (38968000 class)
@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """The symmetric U channel (E2 'symmetric' feature), nominal and with injected errors, compiled and inspected."""
    from contracts.production.part_recipe import PartRecipeV1
    from domains.production.engine.compile import compile_part
    from domains.production.engine.synthetic.cases import generate_case, plan_cases

    root = tmp_path_factory.mktemp("fix3_twins")
    specs = plan_cases("smoke", "DEV")
    out = {}
    for name, scenario in (("nominal", "COMPLETE_NOMINAL"), ("errors", "COMPLETE_ERRORS")):
        spec = next(s for s in specs if s.family == "U_CHANNEL")
        spec = dataclasses.replace(spec, case_id=f"FIX3-ucs-{name}", scenario=scenario, exact=scenario == "COMPLETE_NOMINAL",
                                   plan="FULL", features=dataclasses.replace(spec.features, symmetric=True))
        generate_case(spec, root)
        d = root / spec.case_id
        recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
        assert compile_part(recipe, d / "part.step", d / "compiled", flat_path=d / "flat.dxf").status == "COMPILED"
        out[name] = (d, _run(d, root / f"{name}_run"))
    return out


def _check_no_false_pass(case: Path, res: dict) -> int:
    truth = json.loads((case / "truth.json").read_text())["characteristics"]
    passes = 0
    for b in res["bends"]:
        for c in b["characteristics"]:
            if c["decision"] == "PASS":
                passes += 1
                t = truth.get(c["characteristic_id"])
                assert t is not None and t["truth_in_tolerance"], c["characteristic_id"]
    return passes


def _consistent(case: Path, res: dict) -> None:
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.consistency import check_result_against_compiled
    from contracts.production.part_recipe import PartRecipeV1

    model = ShopfloorInspectionResultV1.model_validate(res)
    compiled = CompiledPartV1.model_validate_json((case / "compiled" / "compiled_part.json").read_bytes())
    recipe = PartRecipeV1.model_validate_json((case / "recipe.json").read_bytes())
    assert check_result_against_compiled(model, compiled, recipe) == []


def test_d3_tied_twins_publish_the_readings_identical_under_both_and_mark_the_identity_ambiguous(twins):
    case, res = twins["nominal"]
    reg = res["inspection_validity"]["registration"]
    assert res["inspection_validity"]["state"] == "VALID"
    assert reg["pose_ambiguous"] and reg["identity_ambiguous"]
    assert any(w.startswith("IDENTITY_AMBIGUOUS_TWINS") for w in res["warnings"])
    measured = [c for b in res["bends"] for c in b["characteristics"] if c["state"] == "MEASURED"]
    assert any(c["kind"] == "INCLUDED_ANGLE" for c in measured)          # an agreeing reading is published
    for b in res["bends"]:
        pos = b["characteristics"][3]
        assert pos["state"] == "NOT_MEASURED" and pos["reason"] == "AMBIGUOUS_POSE"   # identity-dependent: withheld
        assert b["display"]["measured_polyline_mm"] is None
    assert res["viewer_assets"]["deviations"] is None and res["stage_decision"]["stage_model_id"] is None
    assert res["summary"]["state"] != "ALL_MEASURED_IN_TOLERANCE"
    _check_no_false_pass(case, res)
    _consistent(case, res)


def test_d3_with_injected_errors_no_pass_is_false_under_the_true_twin(twins):
    case, res = twins["errors"]
    if not res["inspection_validity"]["registration"]["identity_ambiguous"]:
        pytest.skip("the errors broke the tie: not a twin case")
    _check_no_false_pass(case, res)
    _consistent(case, res)
    # every characteristic is either a reading both twins agree on or UNKNOWN(AMBIGUOUS_POSE)/another UNKNOWN reason
    for b in res["bends"]:
        for c in b["characteristics"]:
            assert c["state"] in ("MEASURED", "NOT_MEASURED", "NOT_APPLICABLE")


def test_d3_the_agreement_rule_publishes_agreeing_readings_and_withholds_differing_ones():
    from types import SimpleNamespace

    from domains.production.engine.inspect.engine import _agree, _matched_spec_ok

    tol = SimpleNamespace(plus=1.0, minus=1.0)
    part = SimpleNamespace(nominals={
        "b1/INCLUDED_ANGLE": SimpleNamespace(nominal=90.0, tolerance=tol, recipe_requirement="REQUIRED"),
        "b2/INCLUDED_ANGLE": SimpleNamespace(nominal=90.0, tolerance=tol, recipe_requirement="REQUIRED"),
        "b3/INCLUDED_ANGLE": SimpleNamespace(nominal=120.0, tolerance=tol, recipe_requirement="REQUIRED"),
    })
    cid = "b1/INCLUDED_ANGLE"
    assert _agree(part, cid, [("M", 90.1, 0.3), ("M", 90.2, 0.3)])            # agreeing reading: published
    assert not _agree(part, cid, [("M", 90.1, 0.3), ("M", 91.6, 0.3)])        # differing: UNKNOWN
    assert not _agree(part, cid, [("M", 90.6, 0.3), ("M", 90.8, 0.3)])        # same within U, decisions differ
    assert not _agree(part, cid, [("M", 90.1, 0.3), ("UNKNOWN",)])            # the twin cannot read it
    # the matched characteristic (the twin's label of the same physical bend) must carry the same spec
    assert _matched_spec_ok(part, "b1", "INCLUDED_ANGLE", [[{"b1": "b2", "b2": "b1"}]])
    assert not _matched_spec_ok(part, "b1", "INCLUDED_ANGLE", [[{"b1": "b3", "b3": "b1"}]])


def test_d3_the_contract_keeps_an_ambiguous_pose_without_the_mark_measuring_nothing(twins):
    case, res = twins["nominal"]
    doc = json.loads(json.dumps(res))
    doc["inspection_validity"]["registration"]["identity_ambiguous"] = False
    with pytest.raises(Exception, match="an ambiguous pose measures nothing"):
        ShopfloorInspectionResultV1.model_validate(doc)
    doc = json.loads(json.dumps(res))
    doc["inspection_validity"]["registration"]["pose_ambiguous"] = False
    with pytest.raises(Exception, match="identity_ambiguous needs pose_ambiguous"):
        ShopfloorInspectionResultV1.model_validate(doc)


# --------------------------------------------------------------------------- follow-up: pose-borne flange lengths
def test_flange_lengths_that_depend_on_an_undetermined_pose_direction_are_withheld():
    from types import SimpleNamespace

    import numpy as np

    from domains.production.engine.inspect.engine import _pose_borne_lengths
    from domains.production.engine.inspect.metrology import Measurement
    from domains.production.engine.inspect.result import BendOutcome

    class Fit:
        """Undetermined: in-plane translation along x (a free datum direction)."""
        undetermined = np.zeros((1, 3))

        def position_determined(self, model, point, direction, tol=0.05):
            return abs(float(np.asarray(direction) @ np.array([1.0, 0.0, 0.0]))) <= tol

    def nominal(adj=None):
        return SimpleNamespace(adjacent_bend_uid=adj)

    bends = [SimpleNamespace(uid="b1", label="B1"), SimpleNamespace(uid="b2", label="B2"), SimpleNamespace(uid="b3", label="B3")]
    part = SimpleNamespace(bends=bends, bend_index={"b1": 0, "b2": 1, "b3": 2},
                           nominals={"b1/FLANGE_LENGTH/A": nominal(), "b1/FLANGE_LENGTH/B": nominal("b2"),
                                     "b2/FLANGE_LENGTH/A": nominal("b1"), "b2/FLANGE_LENGTH/B": nominal(),
                                     "b3/FLANGE_LENGTH/A": nominal(), "b3/FLANGE_LENGTH/B": nominal()})
    plane = lambda n: SimpleNamespace(n=np.asarray(n, dtype=float))  # noqa: E731
    geoms = {
        # B1: OML from the inner skin of a panel whose normal is the free direction -> its sharp is pose-borne
        0: SimpleNamespace(ok=True, one_skin=True, inner_parent=0.0, inner_child=0.5, plane_parent=plane([0, 0, 1]),
                           plane_child=plane([1, 0, 0])),
        # B2: both OML skins seen directly
        1: SimpleNamespace(ok=True, one_skin=False, inner_parent=0.0, inner_child=0.0, plane_parent=plane([0, 0, 1]),
                           plane_child=plane([0, 1, 0])),
        # B3: OML from the inner skin, but of a panel parallel to the datum (its height is fixed): trusted
        2: SimpleNamespace(ok=True, one_skin=True, inner_parent=0.5, inner_child=0.0, plane_parent=plane([0, 0, 1]),
                           plane_child=plane([0, 1, 0])),
    }
    lines = {"b1": (np.zeros(3), np.array([0.0, 100.0, 0.0])), "b2": (np.zeros(3), np.array([0.0, 100.0, 0.0])),
             "b3": (np.zeros(3), np.array([100.0, 0.0, 0.0]))}  # B3 runs along the free direction

    def m(method):
        return Measurement(value=10.0, u=0.1, method=method)

    outs = [BendOutcome(state="FORMED") for _ in bends]
    outs[0].measurements = {"b1/FLANGE_LENGTH/A": m("OML_SHARP_TO_FREE_EDGE_MEDIAN"), "b1/FLANGE_LENGTH/B": m("OML_SHARP_TO_OML_SHARP_MEDIAN")}
    outs[1].measurements = {"b2/FLANGE_LENGTH/A": m("OML_SHARP_TO_OML_SHARP_MEDIAN"), "b2/FLANGE_LENGTH/B": m("OML_SHARP_TO_FREE_EDGE_MEDIAN")}
    outs[2].measurements = {"b3/FLANGE_LENGTH/A": m("OML_SHARP_TO_FREE_EDGE_MEDIAN_VS_STATION_NOMINAL"),
                            "b3/FLANGE_LENGTH/B": m("OML_SHARP_TO_FREE_EDGE_MEDIAN")}
    _pose_borne_lengths(outs, part, geoms, lines, Fit(), model=object())
    unk = {cid for o in outs for cid in o.forced_unknown}
    assert unk == {"b1/FLANGE_LENGTH/A", "b1/FLANGE_LENGTH/B",   # from the pose-borne sharp of B1
                   "b2/FLANGE_LENGTH/A",                         # ends at B1's sharp
                   "b3/FLANGE_LENGTH/A"}                         # station compared along the free direction
    assert all(o.forced_unknown[c] == "REGISTRATION_UNTRUSTED" for o in outs for c in o.forced_unknown)

    class Determined(Fit):
        undetermined = np.zeros((0, 3))

        def position_determined(self, *a, **k):
            return True

    fresh = [BendOutcome(state="FORMED", measurements=dict(o.measurements)) for o in outs]
    _pose_borne_lengths(fresh, part, geoms, lines, Determined(), model=object())
    assert not any(o.forced_unknown for o in fresh)


def test_a_station_comparison_whose_decision_cannot_change_along_the_bend_is_kept():
    from types import SimpleNamespace

    from domains.production.engine.inspect.engine import _station_free_decision
    from domains.production.engine.inspect.metrology import Measurement

    nom = SimpleNamespace(tolerance=SimpleNamespace(plus=0.5, minus=0.5))
    flat = Measurement(value=35.05, u=0.2, method="X_VS_STATION_NOMINAL", profile_range=(34.95, 35.05), raw_median=35.05)
    steep = Measurement(value=36.0, u=0.2, method="X_VS_STATION_NOMINAL", profile_range=(31.0, 36.0), raw_median=36.0)
    assert _station_free_decision(flat, nom)        # PASS whichever station nominal applies: kept
    assert not _station_free_decision(steep, nom)   # PASS at one end, FAIL at the other: withheld
    assert not _station_free_decision(Measurement(value=35.0, u=0.2), nom)  # no profile recorded: withheld
    # FAIL short at one end, FAIL long at the other: the interval passes through PASS -> withheld
    across = Measurement(value=36.0, u=0.2, method="X_VS_STATION_NOMINAL", profile_range=(30.0, 40.0), raw_median=35.0)
    assert not _station_free_decision(across, nom)
    # FAIL long whichever station nominal applies: kept
    long_ = Measurement(value=40.0, u=0.2, method="X_VS_STATION_NOMINAL", profile_range=(34.0, 35.0), raw_median=40.0)
    assert _station_free_decision(long_, nom)
