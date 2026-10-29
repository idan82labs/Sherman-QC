"""E4a G4 renderer, canaries, review packets, verdict schema and the G4 gate.

E3-shaped output directories are written by the E4 truth oracle (not an engine)
for real E2 cases compiled by the real E1. Pixel checks read the PNGs back.
The gate tests label the oracle outputs ``source="E3"`` to exercise the gated
arm; a separate test shows that ``source="ORACLE"`` images are never gated.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from domains.production.engine.compile import compile_part
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
from domains.production.engine.eval.g4 import (
    CLOSEUP_CLASSES,
    QUESTIONS,
    G4VerdictV1,
    RenderJob,
    append_verdict,
    build_review_packet,
    closeup_class,
    evaluate_g4,
)
from domains.production.engine.eval.oracle import OracleSpec, write_oracle_output
from domains.production.engine.eval.render import (
    Corruption,
    fit_flange_direction,
    plan_closeups,
    render_closeup,
    render_global_views,
    scene_from_engine_output,
)
from domains.production.engine.eval.truth import load_truth_case
from domains.production.engine.synthetic.cases import generate_tier

CASES = ("E2-DEV-UC-000", "E2-DEV-HT-005", "E2-DEV-LB-002", "E2-DEV-UCS-000")
# test-only evaluation flags: small packets, reproducible (test-seeded) canaries
GATE = {"require_synthetic_sample": False, "allow_test_seed": True}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("e4a_render")
    tier = root / "tier"
    generate_tier("regression", "DEV", tier, case_ids=CASES)
    out = {}
    variants = {cid: OracleSpec() for cid in CASES}
    variants.update({"E2-DEV-UCS-000:AMBIGUOUS": OracleSpec(pose_ambiguous=True),
                     "E2-DEV-HT-005:AMBIGUOUS": OracleSpec(pose_ambiguous=True)})
    compiled_by_case = {}
    for cid in CASES:
        recipe = load_recipe(tier / cid / "recipe.json")
        cdir = root / "compiled" / cid
        assert compile_part(recipe, tier / cid / "part.step", cdir, flat_path=tier / cid / "flat.dxf").status == "COMPILED"
        compiled_by_case[cid] = (recipe, load_compiled(cdir / "compiled_part.json"), cdir)
    tc_ht = load_truth_case(tier / "E2-DEV-HT-005")
    formed = next(b.bend_uid for b in tc_ht.bends if b.state == "FORMED")
    variants["E2-DEV-HT-005:UNKNOWN"] = OracleSpec(bend_states={formed: "UNKNOWN"})
    second = load_truth_case(tier / "E2-DEV-UCS-000").bends[1].bend_uid
    variants["E2-DEV-UCS-000:AMBIGUOUS_S1"] = OracleSpec(pose_ambiguous=True, bend_states={second: "NOT_YET_FORMED"})
    for name, spec in variants.items():
        cid = name.split(":")[0]
        recipe, compiled, cdir = compiled_by_case[cid]
        eng = root / "engine" / name.replace(":", "_")
        write_oracle_output(eng, recipe=recipe, compiled=compiled, compiled_dir=cdir, truth=load_truth_case(tier / cid),
                            scan_path=tier / cid / "scan.ply", spec=spec)
        out[name] = RenderJob(engine_dir=eng, compiled_json=cdir / "compiled_part.json",
                              recipe_json=tier / cid / "recipe.json", case_ref=name, source="E3",
                              incomplete=cid == "E2-DEV-HT-005")
    return root, out


def _png(path: Path) -> np.ndarray:
    from matplotlib import image

    return image.imread(str(path))


def _scene(job: RenderJob):
    return scene_from_engine_output(job.engine_dir, job.compiled_json, job.recipe_json, title="test")


def _longest_dark_run(rows: np.ndarray) -> int:
    best = 0
    for row in rows:
        dark = row[:, :3].max(axis=1) < 0.25
        run = cur = 0
        for d in dark:
            cur = cur + 1 if d else 0
            run = max(run, cur)
        best = max(best, run)
    return best


def test_closeup_scale_is_exact_with_a_10_mm_bar(outputs, tmp_path):
    _, jobs = outputs
    scene = _scene(jobs["E2-DEV-UC-000"])
    plan = next(p for p in plan_closeups(scene) if scene.bend(p.bend_uid).measured_apex is not None)
    rec = render_closeup(scene, plan, tmp_path / "c.png")
    img = _png(tmp_path / "c.png")
    assert img.shape[1] == rec["width_px"] and img.shape[0] == rec["height_px"]
    assert rec["px_per_mm"] == 10.0 and rec["panel_px"][0] == round(rec["window_mm"][0] * 10.0)
    assert rec["apex_source"] == "MEASURED_ANGLE"  # the solid line sits on the apex at the measured angle
    # the scale bar under the SCAN panel spans exactly 10 mm = 100 px
    y0 = 64 + 26 + rec["panel_px"][1]
    run = _longest_dark_run(img[y0:y0 + 34, 12:12 + rec["panel_px"][0]])
    assert abs(run - 100) <= 3, run
    with pytest.raises(ValueError):
        plan_closeups(scene, px_per_mm=6.0)  # below the 8 px/mm floor


def _line_rows(img: np.ndarray, rec: dict) -> list[int]:
    """Rows of the SCAN panel that are mostly near-black (the solid measured line)."""
    y0 = 64 + 26
    panel = img[y0:y0 + rec["panel_px"][1], 12:12 + rec["panel_px"][0], :3]
    dark = (panel.max(axis=2) < 0.2).mean(axis=1)
    return [y0 + i for i in np.nonzero(dark > 0.8)[0]]


def test_line_shift_canary_moves_the_solid_line_by_the_requested_millimetres(outputs, tmp_path):
    _, jobs = outputs
    scene = _scene(jobs["E2-DEV-UC-000"])
    plan = next(p for p in plan_closeups(scene) if scene.bend(p.bend_uid).measured_apex is not None)
    clean = render_closeup(scene, plan, tmp_path / "clean.png")
    moved = render_closeup(scene, plan, tmp_path / "moved.png", corruption=Corruption("LINE_SHIFT", shift_mm=14.0))
    r0 = _line_rows(_png(tmp_path / "clean.png"), clean)
    r1 = _line_rows(_png(tmp_path / "moved.png"), moved)
    assert r0 and r1
    shift_px = abs(float(np.mean(r1)) - float(np.mean(r0)))
    assert abs(shift_px - 140.0) <= 4.0, shift_px  # 14 mm at 10 px/mm
    assert clean["sha256"] != moved["sha256"] and moved["label_shown"] == clean["label_shown"]


def test_label_swap_canary_swaps_every_label_in_the_scan_panel(outputs, tmp_path):
    _, jobs = outputs
    scene = _scene(jobs["E2-DEV-UC-000"])
    a, b = scene.bends[0], scene.bends[1]
    swap = Corruption("LABEL_SWAP", label_map={a.bend_uid: b.text, b.bend_uid: a.text})
    plan = next(p for p in plan_closeups(scene) if p.bend_uid == a.bend_uid)
    clean = render_closeup(scene, plan, tmp_path / "c.png")
    rec = render_closeup(scene, plan, tmp_path / "s.png", corruption=swap)
    assert rec["label_shown"] == b.text and rec["bend_uid"] == a.bend_uid
    # never two identical labels in the SCAN panel; the partner (if in view) shows the subject's label
    assert len(rec["scan_labels_drawn"]) == len(set(rec["scan_labels_drawn"]))
    assert sorted(rec["scan_labels_drawn"]) == sorted({a.text: b.text, b.text: a.text}.get(t, t)
                                                      for t in clean["scan_labels_drawn"])
    assert rec["cad_labels_drawn"] == clean["cad_labels_drawn"] and a.text in rec["cad_labels_drawn"]
    with pytest.raises(ValueError):
        render_closeup(scene, plan, tmp_path / "x.png", corruption=Corruption("BLUR"))
    with pytest.raises(ValueError):  # a swap map that does not name the subject
        render_closeup(scene, plan, tmp_path / "y.png", corruption=Corruption("LABEL_SWAP", label_map={b.bend_uid: a.text}))


def test_global_views_no_line_closeups_and_metric_free_dimensions(outputs, tmp_path):
    _, jobs = outputs
    scene = _scene(jobs["E2-DEV-HT-005"])
    recs = render_global_views(scene, tmp_path)
    assert [r["kind"] for r in recs] == ["GLOBAL_ISO", "GLOBAL_TOP", "GLOBAL_FRONT", "GLOBAL_SIDE",
                                         "GLOBAL_DIMENSIONS", "GLOBAL_HEATMAP"]
    for r in recs:
        assert _png(tmp_path / r["file"]).shape[:2] == (1200, 1600)
    dims = next(r for r in recs if r["kind"] == "GLOBAL_DIMENSIONS")
    value = re.compile(r"\d+\.\d+ mm")
    assert any(value.search(t) for t in dims["labels_drawn"])  # engineer render: values shown
    assert dims["arrows"]["fitted_direction"] > 0  # arrows follow the scan, not the nominal direction
    packet_view = render_global_views(scene, tmp_path / "packet", show_values=False)
    dims = next(r for r in packet_view if r["kind"] == "GLOBAL_DIMENSIONS")
    assert dims["labels_drawn"] and not any(value.search(t) for t in dims["labels_drawn"])  # reviewers see no metric
    nyf = [b for b in scene.bends if b.state == "NOT_YET_FORMED"]
    assert nyf and all(b.measured_apex is None for b in nyf)
    plan = next(p for p in plan_closeups(scene) if p.bend_uid == nyf[0].bend_uid)
    rec = render_closeup(scene, plan, tmp_path / "nyf.png")
    assert rec["measured_line_shown"] is False and rec["state_shown"] == "NOT_YET_FORMED"


def test_fit_flange_direction_recovers_a_tilted_flange():
    rng = np.random.default_rng(4)
    axis = np.array([0.0, 0.0, 1.0])
    nominal = np.array([1.0, 0.0, 0.0])
    tilt = np.radians(10.0)
    true_dir = np.array([np.cos(tilt), np.sin(tilt), 0.0])
    s = rng.uniform(0, 40, 4000)
    z = rng.uniform(-10, 10, 4000)
    pts = s[:, None] * true_dir + z[:, None] * axis + rng.normal(0, 0.01, (4000, 3))
    d, fitted = fit_flange_direction(pts, np.zeros(3), nominal, axis, 40.0)
    assert fitted and np.degrees(np.arccos(np.clip(d @ true_dir, -1, 1))) < 0.5
    d, fitted = fit_flange_direction(pts[:5], np.zeros(3), nominal, axis, 40.0)
    assert not fitted and np.allclose(d, nominal)  # too few points: the nominal direction, flagged


@pytest.fixture(scope="module")
def packet(outputs):
    root, jobs = outputs
    info = build_review_packet([jobs["E2-DEV-UC-000"], jobs["E2-DEV-HT-005"]], root / "packet", root / "sealed",
                               canary_fraction=0.5, seed=7)
    return root / "packet", root / "sealed", info


def test_packet_hides_canaries_and_case_ids_and_seals_the_key(packet):
    pdir, sdir, info = packet
    manifest = json.loads((pdir / "review_manifest.json").read_text())
    text = (pdir / "review_manifest.json").read_text()
    for cid in CASES:
        assert cid not in text
    for e in manifest["images"]:
        assert set(e) <= {"image_id", "file", "sha256", "kind", "scan_ref", "subject", "tile", "px_per_mm", "questions"}
        assert e["image_id"].startswith("img_") and len(e["image_id"]) == 16
    key_path = sdir / "canary_key.json"
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o400
    assert hashlib.sha256(key_path.read_bytes()).hexdigest() == manifest["canary_key_sha256"]
    key = json.loads(key_path.read_text())
    assert key["salt_source"] == "TEST_SEED" and info["salt_source"] == "TEST_SEED"
    closeups = [k for k in key["images"] if k["kind"] == "CLOSEUP"]
    canaries = [k for k in closeups if k["canary"]]
    assert len(canaries) >= int(np.ceil(0.5 * len(closeups))) and info["canaries"] == len(canaries)
    # a line shift needs a measured line and goes beyond the bend's LANDS tolerance
    for k in canaries:
        if k["canary"]["kind"] == "LINE_SHIFT":
            assert k["measured_line_shown"] and k["canary"]["margin_mm"] >= 3.0 - 1e-6
    swaps = [k for k in canaries if k["canary"]["kind"] == "LABEL_SWAP"]
    assert len(swaps) % 2 == 0 and all(k["canary"]["swapped_with_uid"] != k["true_subject_uid"] for k in swaps)
    assert {k["class"] for k in closeups} <= {"MEASURED", "NOT_YET_FORMED"}
    assert not list(pdir.rglob("canary_key.json"))
    with pytest.raises(ValueError):
        build_review_packet([], pdir, pdir / "sealed_inside")


def _answers(k: dict, reviewer: str, *, miss_canary_by: str | None = None) -> tuple[str, str, str | None]:
    """Correct answers for one close-up (canaries flagged unless this reviewer misses them)."""
    cls = k["class"]
    canary = (k.get("canary") or {}).get("kind")
    if cls == "MEASURED":
        lands, tag, uj = "YES", "YES", None
    elif cls.startswith("AMBIGUOUS") or cls == "INVALID_INSPECTION":
        lands, tag, uj = "NOT_APPLICABLE", "UNDETERMINABLE", None
    elif cls in ("REGISTRATION_RULE", "UNSUPPORTED_RULE"):
        lands, tag, uj = "NOT_APPLICABLE", "YES", None
    else:
        lands, tag, uj = "NOT_APPLICABLE", "YES", "YES"
    if canary == "LINE_SHIFT" and reviewer != miss_canary_by:
        lands = "NO"
    if canary == "LABEL_SWAP" and reviewer != miss_canary_by:
        tag = "NO"
    return lands, tag, uj


def _verdicts(pdir: Path, sdir: Path, path: Path, *, miss_canary_by: str | None = None,
              bad_lands: str | None = None, disagree: str | None = None, tiebreak: bool = False,
              override: dict | None = None) -> None:
    manifest = json.loads((pdir / "review_manifest.json").read_text())
    key = {k["image_id"]: k for k in json.loads((sdir / "canary_key.json").read_text())["images"]}
    for e in manifest["images"]:
        if e["kind"] != "CLOSEUP":
            continue
        k = key[e["image_id"]]
        for reviewer in ("rev-a", "rev-b"):
            lands, tag, uj = _answers(k, reviewer, miss_canary_by=miss_canary_by)
            if e["image_id"] == bad_lands:
                lands = "NO"
            if e["image_id"] == disagree and reviewer == "rev-b":
                lands = "NO"
            if override and e["image_id"] in override:
                lands, tag, uj = override[e["image_id"]]
            append_verdict(path, G4VerdictV1.make(
                image_id=e["image_id"], image_sha256=e["sha256"], bend_uid=e["subject"]["bend_uid"],
                reviewer_id=reviewer, role="PRIMARY", lands=lands, tag_ok=tag, unknown_justified=uj,
                reason="test verdict"))
        if e["image_id"] == disagree and tiebreak:
            append_verdict(path, G4VerdictV1.make(
                image_id=e["image_id"], image_sha256=e["sha256"], bend_uid=e["subject"]["bend_uid"],
                reviewer_id="rev-c", role="TIEBREAK", lands="YES", tag_ok="YES", reason="tie-break"))


def _clean(sdir: Path, cls: str = "MEASURED") -> str:
    key = json.loads((sdir / "canary_key.json").read_text())
    return next(k["image_id"] for k in key["images"] if k["kind"] == "CLOSEUP" and not k["canary"]
                and k["class"] == cls)


def test_g4_passes_with_correct_verdicts_and_every_canary_flagged(packet, tmp_path):
    pdir, sdir, _ = packet
    _verdicts(pdir, sdir, tmp_path / "v.jsonl")
    report = evaluate_g4(pdir, sdir, tmp_path / "v.jsonl", **GATE)
    assert report["verdict"] == "PASS" and report["review_valid"], report
    assert report["engine"]["lands_rate"] == 1.0 and report["engine"]["mistags"] == 0
    assert report["engine"]["no_line_justified"]["NOT_YET_FORMED"]["rate"] == 1.0
    margin = report["canaries"]["min_line_shift_margin_mm"]
    assert margin is None or margin >= 3.0


def test_a_missed_canary_invalidates_the_review(packet, tmp_path):
    pdir, sdir, _ = packet
    _verdicts(pdir, sdir, tmp_path / "v.jsonl", miss_canary_by="rev-b")
    report = evaluate_g4(pdir, sdir, tmp_path / "v.jsonl", **GATE)
    assert not report["review_valid"] and report["verdict"] != "PASS"
    assert report["canaries"]["misses_by_reviewer"].get("rev-b", 0) >= 1


def test_a_bend_that_does_not_land_fails_the_gate(packet, tmp_path):
    pdir, sdir, _ = packet
    _verdicts(pdir, sdir, tmp_path / "v.jsonl", bad_lands=_clean(sdir))
    report = evaluate_g4(pdir, sdir, tmp_path / "v.jsonl", **GATE)
    assert report["verdict"] == "FAIL" and "NOT_ALL_MEASURED_BENDS_LAND" in report["reasons"]


def test_a_non_canary_mistag_fails_the_gate(packet, tmp_path):
    pdir, sdir, _ = packet
    target = _clean(sdir)
    _verdicts(pdir, sdir, tmp_path / "v.jsonl", override={target: ("YES", "NO", None)})
    report = evaluate_g4(pdir, sdir, tmp_path / "v.jsonl", **GATE)
    assert report["verdict"] == "FAIL" and report["reasons"] == ["MISTAGGED_BENDS"] and report["engine"]["mistags"] == 1


def test_an_unjustified_not_yet_formed_claim_fails_the_gate(packet, tmp_path):
    pdir, sdir, _ = packet
    target = _clean(sdir, "NOT_YET_FORMED")
    _verdicts(pdir, sdir, tmp_path / "v.jsonl", override={target: ("NOT_APPLICABLE", "YES", "NO")})
    report = evaluate_g4(pdir, sdir, tmp_path / "v.jsonl", **GATE)
    assert report["verdict"] == "FAIL" and report["reasons"] == ["NOT_YET_FORMED_NOT_JUSTIFIED"]


def test_an_unjustified_unknown_fails_the_gate(outputs, tmp_path):
    root, jobs = outputs
    pdir, sdir = tmp_path / "p", tmp_path / "s"
    build_review_packet([jobs["E2-DEV-HT-005:UNKNOWN"], jobs["E2-DEV-UC-000"]], pdir, sdir, seed=3,
                        include_globals=False)
    key = json.loads((sdir / "canary_key.json").read_text())
    unknown = [k for k in key["images"] if k["class"] == "UNKNOWN"]
    assert unknown and all(k["state_reasons"] == ["OCCLUDED"] for k in unknown)
    _verdicts(pdir, sdir, tmp_path / "ok.jsonl")
    assert evaluate_g4(pdir, sdir, tmp_path / "ok.jsonl", **GATE)["verdict"] == "PASS"
    target = next(k["image_id"] for k in unknown if not k["canary"])
    _verdicts(pdir, sdir, tmp_path / "no.jsonl", override={target: ("NOT_APPLICABLE", "YES", "NO")})
    report = evaluate_g4(pdir, sdir, tmp_path / "no.jsonl", **GATE)
    assert report["verdict"] == "FAIL" and report["reasons"] == ["UNKNOWN_NOT_JUSTIFIED"]


def test_ambiguous_pose_is_justified_only_by_the_e1_symmetry_test(outputs, tmp_path):
    root, jobs = outputs
    # a symmetric U channel reported AMBIGUOUS_POSE: E1's orbit justifies it, no reviewer question
    pdir, sdir = tmp_path / "p1", tmp_path / "s1"
    build_review_packet([jobs["E2-DEV-UCS-000:AMBIGUOUS"], jobs["E2-DEV-UC-000"], jobs["E2-DEV-HT-005"]], pdir, sdir,
                        seed=5, include_globals=False)
    key = json.loads((sdir / "canary_key.json").read_text())
    amb = [k for k in key["images"] if k["class"].startswith("AMBIGUOUS")]
    assert amb and {k["class"] for k in amb} == {"AMBIGUOUS_E1_JUSTIFIED"} and not any(k["canary"] for k in amb)
    manifest = json.loads((pdir / "review_manifest.json").read_text())
    assert all(len(e["questions"]) == 1 and "symmetry test" in e["questions"][0] for e in manifest["images"]
               if e["image_id"] in {k["image_id"] for k in amb})
    _verdicts(pdir, sdir, tmp_path / "v1.jsonl")
    report = evaluate_g4(pdir, sdir, tmp_path / "v1.jsonl", **GATE)
    assert report["verdict"] == "PASS", report
    assert report["engine"]["ambiguous_justified_by_e1"] == len(amb)
    # an asymmetric part reported AMBIGUOUS_POSE: no E1 orbit, so it fails without any reviewer verdict on it
    pdir, sdir = tmp_path / "p2", tmp_path / "s2"
    build_review_packet([jobs["E2-DEV-HT-005:AMBIGUOUS"], jobs["E2-DEV-UC-000"], jobs["E2-DEV-HT-005"]], pdir, sdir,
                        seed=5, include_globals=False)
    key = json.loads((sdir / "canary_key.json").read_text())
    assert any(k["class"] == "AMBIGUOUS_NOT_E1" for k in key["images"])
    _verdicts(pdir, sdir, tmp_path / "v2.jsonl")
    report = evaluate_g4(pdir, sdir, tmp_path / "v2.jsonl", **GATE)
    assert report["verdict"] == "FAIL" and report["reasons"] == ["AMBIGUOUS_NOT_JUSTIFIED_BY_E1"]


def test_line_shift_canaries_always_exceed_the_lands_tolerance(outputs, tmp_path):
    """Large-radius bends (UC-000: Ri + t = 10.8 mm; LB-002: 13.6 mm) must never get a shift
    a reviewer applying the written LANDS rule would accept."""
    root, jobs = outputs
    lb = _scene(jobs["E2-DEV-LB-002"])
    assert max(b.lands_tolerance_mm for b in lb.bends) > 12.0  # no shift range left: never a LINE_SHIFT
    shifts = 0
    for seed in range(1, 5):
        pdir, sdir = tmp_path / f"p{seed}", tmp_path / f"s{seed}"
        build_review_packet([jobs["E2-DEV-UC-000"], jobs["E2-DEV-LB-002"], jobs["E2-DEV-HT-005"]], pdir, sdir,
                            canary_fraction=0.4, seed=seed, include_globals=False)
        key = json.loads((sdir / "canary_key.json").read_text())
        for k in key["images"]:
            c = k.get("canary")
            if c and c["kind"] == "LINE_SHIFT":
                shifts += 1
                assert abs(c["shift_mm"]) >= k["lands_tolerance_mm"] + 3.0 - 1e-6, k
                assert 8.0 <= abs(c["shift_mm"]) <= 15.0
            if k["case_ref"] == "E2-DEV-LB-002" and k["kind"] == "CLOSEUP":
                assert c is None or c["kind"] == "LABEL_SWAP"
    assert shifts > 0


def test_only_the_e3_arm_is_gated_and_the_synthetic_sample_is_required(outputs, packet, tmp_path):
    root, jobs = outputs
    pdir, sdir, _ = packet
    _verdicts(pdir, sdir, tmp_path / "v.jsonl")
    report = evaluate_g4(pdir, sdir, tmp_path / "v.jsonl")  # production defaults
    assert "TEST_SEED_PACKET" in report["problems"] and not report["review_valid"]
    assert "SYNTHETIC_SAMPLE_INSUFFICIENT" in report["reasons"] and report["verdict"] == "NOT_EVALUABLE"
    assert report["sample"]["synthetic_scans"] == 2 and report["sample"]["incomplete_synthetic_scans"] == 1
    oracle_job = replace(jobs["E2-DEV-HT-005"], source="ORACLE")
    p2, s2 = tmp_path / "p", tmp_path / "s"
    build_review_packet([oracle_job], p2, s2, seed=2, include_globals=False)
    _verdicts(p2, s2, tmp_path / "o.jsonl")
    report = evaluate_g4(p2, s2, tmp_path / "o.jsonl", **GATE)
    assert report["verdict"] == "NOT_EVALUABLE" and "NO_GATED_SCANS" in report["reasons"]
    assert report["engine"]["scans"] == 0 and report["by_source"]["ORACLE"]["scans"] == 1
    with pytest.raises(ValueError):
        RenderJob(engine_dir=None, compiled_json=None, recipe_json=None, case_ref="x", source="ENGINE")


def test_packet_titles_are_opaque(outputs, tmp_path):
    _, jobs = outputs
    scene = _scene(jobs["E2-DEV-UC-000"])
    job = RenderJob(engine_dir=None, compiled_json=None, recipe_json=None, case_ref="E2-DEV-UC-000",
                    title="E2-DEV-UC-000 SYN-LEAK", scene=scene)
    build_review_packet([job], tmp_path / "p", tmp_path / "s", seed=1, include_globals=False)
    assert scene.title.startswith("review scan scan_") and "E2-DEV" not in scene.title and "LEAK" not in scene.title


def test_disagreement_needs_a_tiebreak(packet, tmp_path):
    pdir, sdir, _ = packet
    target = _clean(sdir)
    _verdicts(pdir, sdir, tmp_path / "a.jsonl", disagree=target)
    report = evaluate_g4(pdir, sdir, tmp_path / "a.jsonl", **GATE)
    assert report["verdict"] == "NOT_EVALUABLE" and target in report["unresolved_images"]
    _verdicts(pdir, sdir, tmp_path / "b.jsonl", disagree=target, tiebreak=True)
    assert evaluate_g4(pdir, sdir, tmp_path / "b.jsonl", **GATE)["verdict"] == "PASS"


def test_tampering_is_detected(packet, tmp_path):
    pdir, sdir, _ = packet
    _verdicts(pdir, sdir, tmp_path / "v.jsonl")
    with open(tmp_path / "v.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"schema": "sherman_g4_verdict_v1", "image_id": "img_x"}\n')
    report = evaluate_g4(pdir, sdir, tmp_path / "v.jsonl", **GATE)
    assert not report["review_valid"] and any(p.startswith("INVALID_VERDICT_LINE") for p in report["problems"])
    fake = tmp_path / "sealed"
    fake.mkdir()
    doc = json.loads((sdir / "canary_key.json").read_text())
    doc["canary_count"] = 0
    (fake / "canary_key.json").write_text(json.dumps(doc))
    _verdicts(pdir, sdir, tmp_path / "w.jsonl")
    report = evaluate_g4(pdir, fake, tmp_path / "w.jsonl", **GATE)
    assert "CANARY_KEY_COMMITMENT_MISMATCH" in report["problems"] and report["verdict"] != "PASS"


def test_verdict_schema_rules():
    base = dict(image_id="img_0123456789ab", image_sha256="0" * 64, bend_uid="bu_" + "a" * 20, reviewer_id="r",
                role="PRIMARY", reason="x", recorded_at="2026-10-17T00:00:00Z")
    v = G4VerdictV1.make(**base, lands="YES", tag_ok="NO")
    assert v.verdict == "MISTAG"
    assert G4VerdictV1.make(**base, lands="NOT_APPLICABLE", tag_ok="YES", unknown_justified="NO").verdict == \
        "UNKNOWN_NOT_JUSTIFIED"
    assert G4VerdictV1.make(**base, lands="NOT_APPLICABLE", tag_ok="NO").verdict == "CONTEXT_MISTAG"
    with pytest.raises(ValueError):
        G4VerdictV1(**base, lands="YES", tag_ok="YES", verdict="NOT_LANDS")
    with pytest.raises(ValueError):
        G4VerdictV1.make(**base, lands="YES", tag_ok="YES", unknown_justified="YES")



def test_every_closeup_class_has_its_questions(outputs):
    _, jobs = outputs
    scene = _scene(jobs["E2-DEV-UC-000"])
    b = scene.bends[0]
    flat = replace(b, measured_apex=None, measured_oml=None)
    cases = {
        "MEASURED": b,
        "NOT_YET_FORMED": replace(flat, state="NOT_YET_FORMED"),
        "UNKNOWN": replace(flat, state="UNKNOWN", state_reasons=("OCCLUDED",)),
        "FORMED_NO_LINE": replace(flat, state="FORMED", position_reason="EDGE_NOT_SEEN"),
        "REGISTRATION_RULE": replace(flat, state="FORMED", position_reason="REGISTRATION_UNTRUSTED"),
        "UNSUPPORTED_RULE": replace(flat, state="UNKNOWN", state_reasons=("UNSUPPORTED_FEATURE",)),
        "AMBIGUOUS_NOT_E1": replace(flat, state="UNKNOWN", state_reasons=("AMBIGUOUS_POSE",)),
    }
    for cls, overlay in cases.items():
        assert closeup_class(scene, overlay) == cls
    scene.ambiguous_groups = ((b.bend_uid, scene.bends[1].bend_uid),)
    assert closeup_class(scene, cases["AMBIGUOUS_NOT_E1"]) == "AMBIGUOUS_E1_JUSTIFIED"
    scene.validity_state = "MIRRORED_PART_SUSPECTED"
    assert closeup_class(scene, b) == "INVALID_INSPECTION"
    from domains.production.engine.eval.g4 import _QUESTION_SET

    assert set(_QUESTION_SET) == set(CLOSEUP_CLASSES) and set(_QUESTION_SET.values()) <= set(QUESTIONS)



def test_ambiguity_without_a_decoded_stage_uses_the_stage_the_engine_implies(outputs):
    """A U channel reported AMBIGUOUS_POSE with its second bend NOT_YET_FORMED implies stage S1,
    which E1 found unambiguous: the complete part's orbit must not justify it."""
    _, jobs = outputs
    full = _scene(jobs["E2-DEV-UCS-000:AMBIGUOUS"])
    assert full.meta["ambiguity_stage"] == "S2" and full.ambiguous_groups
    staged = _scene(jobs["E2-DEV-UCS-000:AMBIGUOUS_S1"])
    assert staged.meta["ambiguity_stage"] == "S1" and staged.ambiguous_groups == ()
    first = staged.bends[0]
    assert first.state == "UNKNOWN" and "AMBIGUOUS_POSE" in first.state_reasons
    assert closeup_class(staged, first) == "AMBIGUOUS_NOT_E1"
    assert closeup_class(full, full.bends[0]) == "AMBIGUOUS_E1_JUSTIFIED"
