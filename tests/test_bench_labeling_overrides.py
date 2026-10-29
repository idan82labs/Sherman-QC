"""bench_labeling per-bend overrides: margin_mm, extent_t / extent_xyz (validated), plane_from_points."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

from research.bench_labeling.cli import main as cli_main  # noqa: E402
from research.bench_labeling.geometry import fold_support  # noqa: E402
from research.bench_labeling.lines import parse_pairs, run_lines  # noqa: E402
from research.bench_labeling.prep import run_prep  # noqa: E402
from research.bench_labeling.segment import SegmentParams  # noqa: E402
from research.bench_labeling.synthetic import make_part, random_pose, write_part  # noqa: E402

POS_TOL_MM = 0.5
ANG_TOL_DEG = 0.5
SEED = 11
SLOT_Y = (40.0, 46.0)   # slot across the fold of the L (local width coordinate)


def _patch_at(probe, ds):
    d = np.linalg.norm(ds["points"] - probe, axis=1)
    near = ds["labels"][np.argsort(d)[:7]]
    near = near[near > 0]
    return int(np.bincount(near).argmax()) if len(near) else 0


def _outer_face(part, segment):
    sign = part.bends[0]["outer_sign"]
    return next(f for f in part.faces if f["segment"] == segment and f["skin"] == sign)


def _line_errors(start, end, truth):
    g0, g1 = (np.asarray(p, float) for p in truth)
    gd = (g1 - g0) / np.linalg.norm(g1 - g0)

    def perp(p):
        r = np.asarray(p, float) - g0
        return float(np.linalg.norm(r - (r @ gd) * gd))

    d = np.asarray(end, float) - np.asarray(start, float)
    d /= np.linalg.norm(d)
    angle = float(np.degrees(np.arccos(min(1.0, abs(d @ gd)))))
    span = sorted([float((np.asarray(start) - g0) @ gd), float((np.asarray(end) - g0) @ gd)])
    return max(perp(start), perp(end)), angle, span


def _slotted_l(tmp_path):
    """L bracket with a 6 mm slot across the fold that cuts the short flange in two."""
    part = make_part("L", seed=SEED)
    rot, trans = random_pose(SEED)
    local = (part.vertices - trans) @ rot          # (x, y=width, z) profile frame
    centroid = local[part.triangles].mean(axis=1)
    cut = (centroid[:, 1] > SLOT_Y[0]) & (centroid[:, 1] < SLOT_Y[1]) & (centroid[:, 0] > 37.0)
    part = dataclasses.replace(part, triangles=part.triangles[~cut])
    scan = write_part(part, tmp_path / "L_slot.ply")
    run_prep(scan, tmp_path / "prep")
    ds = np.load(tmp_path / "prep" / "points_ds.npz")
    pa = _patch_at(_outer_face(part, 0)["probe"], ds)
    sign = part.bends[0]["outer_sign"]
    # the short flange's outer skin is now two patches; the probe (15% of width) hits the big one
    big_b = _patch_at(_outer_face(part, 1)["probe"], ds)
    probe_b = _outer_face(part, 1)["probe"]
    far = (probe_b - trans) @ rot
    far[1] = 0.5 * (SLOT_Y[1] + part.width)                 # beyond the slot
    small_b = _patch_at(far @ rot.T + trans, ds)
    assert pa and big_b and small_b and len({pa, big_b, small_b}) == 3, (pa, big_b, small_b)
    return part, sign, pa, big_b


def _write_pairs(path, bends):
    path.write_text(json.dumps({"labeler": "test", "bends": bends}))
    return path


def test_extent_override_spans_slot_and_is_validated(tmp_path):
    part, sign, pa, pb = _slotted_l(tmp_path)
    truth = part.bends[0]["skins"][sign]
    kw = dict(thickness_mm=part.thickness, ri_mm=part.ri)

    # 1. default: the fold is cut short at the slot (side B patch ends there)
    auto = run_lines(tmp_path / "prep", _write_pairs(tmp_path / "p0.json", [{"patches": [pa, pb]}]),
                     tmp_path / "lines0", **kw)["bends"][0]
    assert auto["status"] == "ok" and auto["extent_source"] == "auto"
    assert auto["override_support_frac"] is None and auto["t_range"] == auto["t_range_auto"]
    assert auto["length_mm"] < 0.75 * part.width
    origin, d = np.asarray(auto["line_origin"]), np.asarray(auto["direction"])
    np.testing.assert_allclose(auto["sharp"]["start"], origin + auto["t_range"][0] * d, atol=1e-6)
    t_truth = sorted(float((np.asarray(q) - origin) @ d) for q in truth["apex"])

    # 2. explicit extent across the slot to the real fold ends: accepted, support ~ 54/60
    good = run_lines(tmp_path / "prep", _write_pairs(tmp_path / "p1.json", [
        {"patches": [pa, pb], "extent_t": t_truth}]), tmp_path / "lines1", **kw)["bends"][0]
    assert good["extent_source"] == "override"
    assert good["line_origin"] == pytest.approx(auto["line_origin"], abs=1e-9)
    assert 0.8 <= good["override_support_frac"] < 0.97
    assert good["confidence"] == "certain", good["confidence_reasons"]
    gaps = good["fold_support_gaps_t"]
    assert gaps and all(t_truth[0] < g[0] < g[1] < t_truth[1] for g in gaps)
    for locus in ("apex", "sharp"):
        perp, angle, span = _line_errors(good[locus]["start"], good[locus]["end"], truth[locus])
        assert perp < POS_TOL_MM and angle < ANG_TOL_DEG
        assert span[0] == pytest.approx(0.0, abs=0.5) and span[1] == pytest.approx(part.width, abs=0.5)

    # 3. same extent given as two 3D points
    xyz = run_lines(tmp_path / "prep", _write_pairs(tmp_path / "p2.json", [
        {"patches": [pa, pb], "extent_xyz": [list(truth["apex"][0]), list(truth["apex"][1])]}]),
        tmp_path / "lines2", **kw)["bends"][0]
    assert xyz["t_range"] == pytest.approx(t_truth, abs=1e-6)

    # 4. an extent running 30 mm past the part end is never silently accepted
    bad = run_lines(tmp_path / "prep", _write_pairs(tmp_path / "p3.json", [
        {"patches": [pa, pb], "extent_t": [t_truth[0], t_truth[1] + 30.0], "confidence": "certain"}]),
        tmp_path / "lines3", **kw)["bends"][0]
    assert bad["override_support_frac"] < 0.8
    assert bad["confidence"] == "uncertain" and bad["auto_confidence"] == "uncertain"
    assert any("override extent fold support" in r for r in bad["confidence_reasons"])
    assert any("downgraded" in r for r in bad["confidence_reasons"])


def test_margin_override_and_cli_default(tmp_path):
    part, sign, pa, pb = _slotted_l(tmp_path)
    pairs = _write_pairs(tmp_path / "p.json", [{"patches": [pa, pb], "margin_mm": 6.0},
                                               {"patches": [pa, pb]}])
    rc = cli_main(["lines", "--prep", str(tmp_path / "prep"), "--pairs", str(pairs),
                   "--out", str(tmp_path / "lines"), "--margin", "4.5",
                   "--thickness", str(part.thickness), "--ri", str(part.ri)])
    assert rc == 0
    doc = json.loads((tmp_path / "lines" / "lines.json").read_text())
    b1, b2 = doc["bends"]
    assert doc["line_params"]["margin_mm"] == 4.5
    assert b1["margin_mm"] == 6.0 and b1["overrides"] == {"margin_mm": 6.0}
    assert b2["margin_mm"] == 4.5 and "overrides" not in b2
    assert b1["zone_radius_mm"] - b2["zone_radius_mm"] == pytest.approx(1.5, abs=0.05)


@pytest.mark.parametrize("fmt", ["mesh", "cloud"])
def test_plane_from_points_replaces_missing_flange(tmp_path, fmt):
    part = make_part("L", seed=SEED)
    scan = write_part(part, tmp_path / "L.ply", as_point_cloud=(fmt == "cloud"),
                      with_normals=(fmt != "cloud"))
    # min_area above the short flange's area: segmentation drops that flange entirely
    run_prep(scan, tmp_path / "prep", params=SegmentParams(min_area_mm2=2000.0))
    ds = np.load(tmp_path / "prep" / "points_ds.npz")
    face_a, face_b = _outer_face(part, 0), _outer_face(part, 1)
    pa = _patch_at(face_a["probe"], ds)
    assert pa > 0 and _patch_at(face_b["probe"], ds) == 0, "flange B should be unsegmented"
    seed = [float(v) for v in face_b["probe"]]
    pairs = _write_pairs(tmp_path / "p.json", [
        {"patches": [pa, {"plane_from_points": {"seed_xyz": seed, "radius_mm": 10.0}}]}])
    doc = run_lines(tmp_path / "prep", pairs, tmp_path / "lines", thickness_mm=part.thickness,
                    ri_mm=part.ri)
    bend = doc["bends"][0]
    assert bend["status"] == "ok", bend.get("error")
    truth = part.bends[0]["skins"][part.bends[0]["outer_sign"]]
    for locus in ("apex", "sharp"):
        perp, angle, span = _line_errors(bend[locus]["start"], bend[locus]["end"], truth[locus])
        assert perp < POS_TOL_MM and angle < ANG_TOL_DEG, (locus, perp, angle)
        assert span[0] == pytest.approx(0.0, abs=1.5) and span[1] == pytest.approx(part.width, abs=1.5)
    assert bend["skin"] == "outer"
    assert bend["confidence"] == "certain", bend["confidence_reasons"]
    side = bend["plane_sides"]["b"]
    assert side["plane_from_points"]["seed_xyz"] == seed
    assert side["fit"]["raw_points"] > 1000 and side["fit"]["rms_mm"] < 0.1
    assert bend["patches"][1].startswith("plane@")
    assert (tmp_path / "lines" / "render" / "closeup_B1.png").is_file()
    # seed off the surface is a hard failure, not a guess
    off = (np.asarray(seed) + 25.0 * np.asarray(face_b["normal"])).tolist()
    bad = run_lines(tmp_path / "prep", _write_pairs(tmp_path / "q.json", [
        {"patches": [pa, {"plane_from_points": {"seed_xyz": off, "radius_mm": 10.0}}]}]),
        tmp_path / "lines_bad", thickness_mm=part.thickness, ri_mm=part.ri)["bends"][0]
    assert bad["status"] == "failed" and "seed" in bad["error"]


def test_fold_support_counts_both_half_spaces():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 50, 20000)
    side = rng.uniform(0.6, 4.0, 20000)
    a_side = rng.random(20000) < 0.5
    ua, ub = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    pts = np.stack([s * 0, s * 0, s], axis=1) + np.where(a_side[:, None], side[:, None] * ua,
                                                        side[:, None] * ub)
    d = np.array([0, 0, 1.0])
    full = fold_support(pts, np.zeros(3), d, ua, ub, 0, 50, reach_mm=4.0)
    assert full["frac"] == 1.0 and full["gaps"] == []
    one_sided = pts[a_side | (s < 25)]
    half = fold_support(one_sided, np.zeros(3), d, ua, ub, 0, 50, reach_mm=4.0)
    assert half["frac"] == pytest.approx(0.5, abs=0.03)
    assert half["gaps"][0][0] == pytest.approx(25.0, abs=1.0)


def test_parse_overrides():
    parsed = parse_pairs({"bends": [{"patches": [3, {"plane_from_points": {
        "seed_xyz": [1, 2, 3], "radius_mm": 8}}], "margin_mm": 5, "extent_t": [-10, 40]}]})
    bend = parsed["bends"][0]
    assert bend["sides"][0] == [3]
    assert bend["sides"][1] == {"plane_from_points": {"seed_xyz": [1.0, 2.0, 3.0], "radius_mm": 8.0,
                                                      "grow": True, "tol_mm": None}}
    assert bend["margin_mm"] == 5.0 and bend["extent_t"] == [-10.0, 40.0] and bend["extent_xyz"] is None
    plain = parse_pairs([[1, 2]])["bends"][0]
    assert plain["margin_mm"] is None and plain["extent_t"] is None
    for bad in ({"patches": [1, 2], "extent_t": [5, 1]},
                {"patches": [1, 2], "extent_t": [0, 5], "extent_xyz": [[0, 0, 0], [1, 1, 1]]},
                {"patches": [1, 2], "margin": 4},
                {"patches": [1, 2], "margin_mm": -1},
                {"patches": [1, {"plane_from_points": {"seed_xyz": [0, 0], "radius_mm": 5}}]},
                {"patches": [1, {"plane_from_points": {"seed_xyz": [0, 0, 0], "radius_mm": 5,
                                                       "radius": 3}}]},
                {"patches": [1, {"plane": {"seed_xyz": [0, 0, 0]}}]}):
        with pytest.raises(ValueError):
            parse_pairs({"bends": [bad]})


def test_apex_mode_station_median_override(tmp_path):
    """On a straight fold the station-median apex equals the default envelope apex."""
    part, sign, pa, pb = _slotted_l(tmp_path)
    pairs = _write_pairs(tmp_path / "p.json", [{"patches": [pa, pb], "apex_mode": "station_median"},
                                               {"patches": [pa, pb]}])
    rc = cli_main(["lines", "--prep", str(tmp_path / "prep"), "--pairs", str(pairs),
                   "--out", str(tmp_path / "lines"),
                   "--thickness", str(part.thickness), "--ri", str(part.ri)])
    assert rc == 0
    b1, b2 = json.loads((tmp_path / "lines" / "lines.json").read_text())["bends"]
    assert b1["apex_mode"] == "station_median" and b1["overrides"] == {"apex_mode": "station_median"}
    assert "apex_mode" not in b2 and "overrides" not in b2
    assert b1["apex_offset_mm"] == pytest.approx(b2["apex_offset_mm"], abs=0.1)
    assert parse_pairs([[1, 2]])["bends"][0]["apex_mode"] is None
    with pytest.raises(ValueError):
        parse_pairs({"bends": [{"patches": [1, 2], "apex_mode": "median"}]})
