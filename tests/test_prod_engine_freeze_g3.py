"""E4b G3 truth-table material (amendment 7): display registration, images, template.

Synthetic E2 scans in a random pose with known truth (``cad_to_raw``) compiled
by the real E1: the display registration must recover the pose for a complete
part and for a staged one (only the prefix stage fits), the material must hold
the expected images and nothing from an engine, and the template one row per
(case, bend).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from domains.production.engine.compile import compile_part
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
from domains.production.engine.eval.freeze import common, g3_material as g3
from domains.production.engine.eval.plyio import read_ply
from domains.production.engine.synthetic.cases import generate_tier

CASES = ("E2-DEV-TD-000", "E2-DEV-ZS-005")


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    root = tmp_path_factory.mktemp("e4b_g3")
    generate_tier("regression", "DEV", root / "tier", case_ids=list(CASES))
    out = {}
    for cid in CASES:
        src = root / "tier" / cid
        cdir = root / "compiled" / cid
        assert compile_part(load_recipe(src / "recipe.json"), src / "part.step", cdir,
                            flat_path=src / "flat.dxf").status == "COMPILED"
        truth = json.loads((src / "truth.json").read_text())
        out[cid] = (src, cdir, load_compiled(cdir / "compiled_part.json"), truth)
    return root, out


def _stages(cdir: Path, compiled) -> list:
    st = sorted((s for s in compiled.stages if s.kind == "SEQUENCE_PREFIX"), key=lambda s: s.stage_index)
    res = []
    for s in st:
        with np.load(cdir / s.sample_cloud.rel_path) as z:
            res.append((s.stage_model_id, np.asarray(z["points"], float), np.asarray(z["normals"], float)))
    return res


def _pose_error(t: np.ndarray, truth_raw_to_cad: np.ndarray, pts: np.ndarray) -> float:
    a = pts @ t[:3, :3].T + t[:3, 3]
    b = pts @ truth_raw_to_cad[:3, :3].T + truth_raw_to_cad[:3, 3]
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


@pytest.mark.parametrize("cid,stage", [("E2-DEV-TD-000", None), ("E2-DEV-ZS-005", "S1")])
def test_display_registration_recovers_the_pose(parts, cid, stage):
    _, out = parts
    src, cdir, compiled, truth = out[cid]
    raw = read_ply(src / "scan.ply", with_faces=False).vertices
    reg = g3.display_register(raw, _stages(cdir, compiled))
    true_r2c = np.linalg.inv(np.asarray(truth["scan"]["cad_to_raw"], float))
    assert reg.confidence == "OK" and reg.inlier_share >= 0.7
    assert _pose_error(reg.raw_to_cad, true_r2c, raw[::50]) < 2.0
    if stage is not None:
        assert reg.stage_model_id == stage == truth["stage"]["expected_stage_model_id"]


def test_material_has_the_images_and_no_engine_output(parts, tmp_path, monkeypatch):
    _, out = parts
    monkeypatch.setattr(common, "custody_check", lambda p, purpose: Path(p).resolve())
    src, cdir, compiled, truth = out["E2-DEV-ZS-005"]
    case = {"case_ref": "pc_g3test000001", "scan_sha256": common.sha256_file(src / "scan.ply"), "recipe_sku": "DEV-X"}
    entry = g3.render_case_material(case, src / "scan.ply", cdir, tmp_path / "m", tmp_path / "blind", sections=True)
    names = set(entry["images"])
    assert {"raw_views.png", "cad_numbered_views.png", "overlay_views.png"} <= names
    assert {f"bend_{b.sequence_index:02d}.png" for b in compiled.bends} <= names
    assert entry["engine_output_used"] is False and entry["registration"]["confidence"] == "OK"
    text = json.dumps(entry)
    assert "stage_model" not in text and "inlier" not in text  # the fitted stage stays reviewer-blind
    blind = json.loads((tmp_path / "blind" / "pc_g3test000001.registration.json").read_text())
    assert blind["stage_model_fitted"] == "S1"
    for name, sha in entry["images"].items():
        assert common.sha256_file(tmp_path / "m" / name) == sha
    no_sections = g3.render_case_material(case, src / "scan.ply", cdir, tmp_path / "n", tmp_path / "blind",
                                          sections=False)
    assert not any(n.startswith("bend_") for n in no_sections["images"])
    no_cad = g3.render_case_material(case, src / "scan.ply", None, tmp_path / "o", tmp_path / "blind", sections=True)
    assert set(no_cad["images"]) == {"raw_views.png"} and no_cad["cad"] == "NO_COMPILED_PART"
    tmpl = g3.truth_table_template([entry])
    assert len(tmpl["rows"]) == len(compiled.bends) and tmpl["frozen_at"] is None
    assert all(r["reviewer_1"] is None and r["final"] is None for r in tmpl["rows"])


def test_icp_converges_and_voxel_thinning_is_deterministic():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 100, 4000), rng.uniform(0, 50, 4000), np.zeros(4000)])
    pts = np.vstack([pts, np.column_stack([np.zeros(2000), rng.uniform(0, 50, 2000), rng.uniform(0, 40, 2000)]),
                     np.column_stack([rng.uniform(0, 100, 2000), np.zeros(2000), rng.uniform(0, 40, 2000)])])
    nrm = np.vstack([np.tile([0, 0, 1.0], (4000, 1)), np.tile([1.0, 0, 0], (2000, 1)), np.tile([0, 1.0, 0], (2000, 1))])
    t_true = np.eye(4)
    c, s = np.cos(0.05), np.sin(0.05)
    t_true[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t_true[:3, 3] = [1.0, -0.5, 0.3]
    src = (pts - t_true[:3, 3]) @ t_true[:3, :3]  # src mapped by t_true lands on pts
    t, q = g3.icp_point_to_plane(src, pts, nrm, np.eye(4), iterations=30)
    assert q["inlier_share"] > 0.95 and np.allclose(t, t_true, atol=0.05)
    a, b = g3.voxel_thin(pts, 2.0, 500, seed=1), g3.voxel_thin(pts, 2.0, 500, seed=1)
    assert np.array_equal(a, b) and len(a) == 500


def test_refinement_finds_a_formed_set_that_is_not_a_prefix(tmp_path):
    """ORDER_SWAP: the scan's formed set is not a prefix of the recipe order; single-bend
    toggles from the best prefix fit must find it (display placement only)."""
    cid = "E2-DEV-UC-006"
    generate_tier("regression", "DEV", tmp_path / "tier", case_ids=[cid])
    src, cdir = tmp_path / "tier" / cid, tmp_path / "compiled"
    assert compile_part(load_recipe(src / "recipe.json"), src / "part.step", cdir,
                        flat_path=src / "flat.dxf").status == "COMPILED"
    compiled = load_compiled(cdir / "compiled_part.json")
    truth = json.loads((src / "truth.json").read_text())
    assert truth["stage"]["scenario"] == "ORDER_SWAP" and not truth["stage"]["is_sequence_prefix"]
    stages = _stages(cdir, compiled)
    raw = read_ply(src / "scan.ply", with_faces=False).vertices
    reg = g3.display_register(raw, stages)
    final = max(compiled.stages, key=lambda s: s.stage_index)
    with np.load(cdir / final.sample_cloud.rel_path) as z:
        fz = {k: np.asarray(z[k]) for k in z.files}
    start = set(next(s for s in compiled.stages if s.stage_model_id == reg.stage_model_id).formed_bend_uids)
    placed, pose, info = g3.refine_formed_set(raw, compiled, fz, start, reg.raw_to_cad)
    assert placed == set(truth["stage"]["formed_bend_uids"]) != start and info["toggles"]
    true_r2c = np.linalg.inv(np.asarray(truth["scan"]["cad_to_raw"], float))
    assert _pose_error(pose, true_r2c, raw[::50]) < 1.0
