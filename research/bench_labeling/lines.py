"""``lines``: bend lines from labeler-chosen patch pairs, plus verification renders."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import TOOL_VERSION
from .geometry import APEX_MODES, BendLineError, LineParams, compute_bend, fold_support
from .render import LINE_PALETTE, REQUIRED_VIEWS, render_closeup, render_verification
from .scanio import load_scan, sha256_file
from .segment import fit_plane

LOCI = ("apex", "sharp")


_TYPO_KEYS = {"margin": "margin_mm", "extent": "extent_t", "extent_mm": "extent_t",
              "t_range": "extent_t", "plane": "plane_from_points", "extent_points": "extent_xyz"}


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_plane_side(spec, where):
    """Normalise a {"plane_from_points": {...}} side (segmentation missed the flange)."""
    if set(spec) != {"plane_from_points"} or not isinstance(spec["plane_from_points"], dict):
        raise ValueError(f"{where}: an object side must be {{\"plane_from_points\": {{...}}}}")
    raw = spec["plane_from_points"]
    unknown = set(raw) - {"seed_xyz", "radius_mm", "grow", "tol_mm"}
    if unknown:
        raise ValueError(f"{where}.plane_from_points: unknown keys {sorted(unknown)}")
    seed = raw.get("seed_xyz")
    if not (isinstance(seed, list) and len(seed) == 3 and all(_finite_number(v) for v in seed)):
        raise ValueError(f"{where}.plane_from_points.seed_xyz must be [x, y, z]")
    radius = raw.get("radius_mm")
    if not (_finite_number(radius) and 1.0 <= radius <= 500.0):
        raise ValueError(f"{where}.plane_from_points.radius_mm must be a number in [1, 500]")
    grow = raw.get("grow", True)
    if not isinstance(grow, bool):
        raise ValueError(f"{where}.plane_from_points.grow must be true or false")
    tol = raw.get("tol_mm")
    if tol is not None and not (_finite_number(tol) and 0.02 <= tol <= 5.0):
        raise ValueError(f"{where}.plane_from_points.tol_mm must be a number in [0.02, 5]")
    return {"plane_from_points": {"seed_xyz": [float(v) for v in seed], "radius_mm": float(radius),
                                  "grow": grow, "tol_mm": None if tol is None else float(tol)}}


def is_plane_side(side):
    return isinstance(side, dict)


def side_label(side):
    """Short text for a side: '3+12' for patch ids, 'plane@x,y,z/r' for a plane side."""
    if is_plane_side(side):
        spec = side["plane_from_points"]
        return "plane@" + ",".join(f"{v:.1f}" for v in spec["seed_xyz"]) + f"/r{spec['radius_mm']:g}"
    return "+".join(map(str, side))


def parse_pairs(doc):
    """Accept {"bends": [{"patches": [a, b], ...}]} or a bare [[a, b], ...] list.

    Optional per-bend overrides (README_OVERRIDES.md): ``margin_mm``, ``extent_t``,
    ``extent_xyz``; a side may be ``{"plane_from_points": {...}}`` instead of ids.
    """
    if isinstance(doc, list):
        doc = {"bends": [item if isinstance(item, dict) else {"patches": item} for item in doc]}
    if not isinstance(doc, dict) or not isinstance(doc.get("bends"), list):
        raise ValueError("pairs file must be a list of pairs or an object with a 'bends' list")
    bends = []
    for k, entry in enumerate(doc["bends"]):
        if isinstance(entry, list):
            entry = {"patches": entry}
        if isinstance(entry, dict) and "patches" not in entry and "a" in entry and "b" in entry:
            entry = {**entry, "patches": [entry["a"], entry["b"]]}
        typos = sorted(key for key in entry if key in _TYPO_KEYS)
        if typos:
            raise ValueError(f"bends[{k}] has {typos}; did you mean "
                             f"{[_TYPO_KEYS[key] for key in typos]}?")
        sides = entry.get("patches") or entry.get("pair")
        if not isinstance(sides, list) or len(sides) != 2:
            raise ValueError(f"bends[{k}].patches must be [side_a, side_b]")
        norm_sides = []
        for j, side in enumerate(sides):
            if isinstance(side, dict):
                norm_sides.append(_parse_plane_side(side, f"bends[{k}].patches[{j}]"))
                continue
            ids = side if isinstance(side, list) else [side]
            if not ids or not all(isinstance(i, int) and not isinstance(i, bool) and i > 0 for i in ids):
                raise ValueError(f"bends[{k}] patch ids must be positive integers")
            norm_sides.append(sorted(set(ids)))
        id_sides = [set(side) for side in norm_sides if not is_plane_side(side)]
        if len(id_sides) == 2 and id_sides[0] & id_sides[1]:
            raise ValueError(f"bends[{k}] uses the same patch on both sides")
        confidence = entry.get("confidence", "auto")
        if confidence not in {"auto", "certain", "uncertain"}:
            raise ValueError(f"bends[{k}].confidence must be auto, certain or uncertain")
        note = entry.get("note", entry.get("notes", ""))
        if not isinstance(note, str):
            raise ValueError(f"bends[{k}].note must be a string")
        margin = entry.get("margin_mm")
        if margin is not None and not (_finite_number(margin) and 0.0 <= margin <= 50.0):
            raise ValueError(f"bends[{k}].margin_mm must be a number in [0, 50]")
        extent_t = entry.get("extent_t")
        if extent_t is not None:
            if not (isinstance(extent_t, list) and len(extent_t) == 2
                    and all(_finite_number(v) for v in extent_t) and extent_t[1] > extent_t[0]):
                raise ValueError(f"bends[{k}].extent_t must be [t0, t1] (mm) with t1 > t0")
            extent_t = [float(v) for v in extent_t]
        extent_xyz = entry.get("extent_xyz")
        if extent_xyz is not None:
            if not (isinstance(extent_xyz, list) and len(extent_xyz) == 2
                    and all(isinstance(q, list) and len(q) == 3 and all(_finite_number(v) for v in q)
                            for q in extent_xyz)):
                raise ValueError(f"bends[{k}].extent_xyz must be [[x, y, z], [x, y, z]]")
            extent_xyz = [[float(v) for v in q] for q in extent_xyz]
        if extent_t is not None and extent_xyz is not None:
            raise ValueError(f"bends[{k}]: give extent_t or extent_xyz, not both")
        apex_mode = entry.get("apex_mode")
        if apex_mode is not None and apex_mode not in APEX_MODES:
            raise ValueError(f"bends[{k}].apex_mode must be one of {list(APEX_MODES)}")
        bends.append({"sides": norm_sides, "confidence": confidence, "note": note,
                      "margin_mm": None if margin is None else float(margin),
                      "extent_t": extent_t, "extent_xyz": extent_xyz, "apex_mode": apex_mode})
    no_bends = bool(doc.get("no_bends_present", False))
    if no_bends and bends:
        raise ValueError("no_bends_present cannot be true when bends are listed")
    return {"bends": bends, "no_bends_present": no_bends, "notes": str(doc.get("notes", "")),
            "labeler": str(doc.get("labeler", ""))}


def assign_full_resolution(scan, ds_points, ds_labels, patches, voxel_mm, thickness_mm):
    """Label every raw scan point by its nearest picked point, gated by plane distance/normal."""
    tree = cKDTree(ds_points)
    dist, idx = tree.query(scan.points, k=1, distance_upper_bound=1.5 * voxel_mm + 1e-6)
    labels = np.zeros(len(scan.points), dtype=np.int64)
    hit = np.isfinite(dist)
    labels[hit] = ds_labels[idx[hit]]
    gate = min(0.6, 0.45 * thickness_mm)
    for patch in patches:
        sel = np.flatnonzero(labels == patch["id"])
        if not len(sel):
            continue
        n = np.asarray(patch["normal"])
        c = np.asarray(patch["centroid"])
        bad = np.abs((scan.points[sel] - c) @ n) > gate
        if scan.oriented and scan.normals is not None and patch.get("normal_oriented"):
            bad |= (scan.normals[sel] @ n) < np.cos(np.radians(35.0))
        labels[sel[bad]] = 0
    return labels


def _twin_outward(side, by_id):
    """Outward direction from the other-skin twin (unoriented clouds): away from material."""
    vecs = []
    for pid in side:
        patch = by_id[pid]
        if patch.get("twin_offset_mm") is None:
            return None
        vecs.append(-np.sign(patch["twin_offset_mm"]) * np.asarray(patch["normal"]))
    total = np.sum(vecs, axis=0)
    return total if np.linalg.norm(total) > 1e-6 else None


def plane_side_points(spec, scan, ds_points, ds_normals, ds_oriented, *, voxel_mm, thickness_mm):
    """Raw points of a flange given as ``plane_from_points`` (segmentation missed it).

    1. ``ball`` = downsampled points within ``radius_mm`` of ``seed_xyz``; the seed must
       lie within max(2 mm, 3 voxels) of a downsampled point.
    2. Plane: trimmed fit to the ball points whose normal is within 20 deg of the
       normal at the point nearest the seed and that lie within 0.45 t of that
       point's tangent plane (keeps one skin), refit on all ball points within
       ``tol_mm`` (default min(0.6, 0.45 t)) of the plane and normal-consistent.
    3. ``grow`` (default true): flood-fill from those inliers over downsampled points
       that stay within ``tol_mm`` of the plane and normal-consistent (neighbour
       step 2.5 voxels), so the whole connected flange is used, not just the ball.
    4. Raw points whose nearest downsampled point is in the region (within 1.5
       voxels) and that lie within ``tol_mm`` of the plane form the side.
    """
    cfg = spec["plane_from_points"]
    seed = np.asarray(cfg["seed_xyz"], dtype=float)
    tol = cfg["tol_mm"] if cfg["tol_mm"] is not None else min(0.6, 0.45 * thickness_mm)
    tree = cKDTree(ds_points)
    seed_dist, seed_idx = tree.query(seed, k=1)
    if seed_dist > max(2.0, 3.0 * voxel_mm):
        raise BendLineError(f"plane_from_points seed is {seed_dist:.1f} mm from the scanned "
                            "surface; put it on the flange")
    ball = np.asarray(tree.query_ball_point(seed, cfg["radius_mm"]), dtype=np.int64)
    n0 = ds_normals[seed_idx] / max(np.linalg.norm(ds_normals[seed_idx]), 1e-12)
    x0 = ds_points[seed_idx]
    cos20 = np.cos(np.radians(20.0))

    def normal_ok(idx, n):
        dots = ds_normals[idx] @ n
        return dots >= cos20 if ds_oriented else np.abs(dots) >= cos20

    cand = ball[normal_ok(ball, n0) & (np.abs((ds_points[ball] - x0) @ n0) <= 0.45 * thickness_mm)]
    if len(cand) < 20:
        raise BendLineError(f"plane_from_points: only {len(cand)} planar points near the seed")
    n, c, _, _ = fit_plane(ds_points[cand])
    if n @ n0 < 0:
        n = -n
    inl = ball[normal_ok(ball, n) & (np.abs((ds_points[ball] - c) @ n) <= tol)]
    if len(inl) < 20:
        raise BendLineError(f"plane_from_points: only {len(inl)} inliers within {tol:.2f} mm")
    n, c, rms, _ = fit_plane(ds_points[inl])
    if n @ n0 < 0:
        n = -n
    region = set(int(i) for i in inl)
    if cfg["grow"]:
        frontier = list(region)
        step = 2.5 * voxel_mm
        while frontier:
            nbrs = tree.query_ball_point(ds_points[frontier], step)
            new = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for v in nbrs]))
            new = np.array([i for i in new if i not in region], dtype=np.int64)
            if not len(new):
                break
            new = new[normal_ok(new, n) & (np.abs((ds_points[new] - c) @ n) <= tol)]
            region.update(int(i) for i in new)
            frontier = [int(i) for i in new]
    region_idx = np.fromiter(region, dtype=np.int64)
    in_region = np.zeros(len(ds_points), dtype=bool)
    in_region[region_idx] = True
    dist, idx = tree.query(scan.points, k=1, distance_upper_bound=1.5 * voxel_mm + 1e-6)
    hit = np.isfinite(dist)
    mask = np.zeros(len(scan.points), dtype=bool)
    mask[hit] = in_region[idx[hit]]
    mask &= np.abs((scan.points - c) @ n) <= tol
    if scan.oriented and scan.normals is not None:
        mask &= np.abs(scan.normals @ n) >= np.cos(np.radians(35.0))
    outward, outward_source = None, "none"
    if scan.oriented and scan.normals is not None and mask.any():
        outward, outward_source = scan.normals[mask].mean(axis=0), "scan_normals"
    else:
        # unoriented: the other skin sits at ~t on the material side (cf. patch twins)
        off = (ds_points[ball] - c) @ n
        par = np.abs(ds_normals[ball] @ n) >= cos20
        lo_t, hi_t = 0.6 * thickness_mm, 1.4 * thickness_mm
        plus = int((par & (off >= lo_t) & (off <= hi_t)).sum())
        minus = int((par & (off <= -lo_t) & (off >= -hi_t)).sum())
        if max(plus, minus) >= 20 and max(plus, minus) >= 2 * max(1, min(plus, minus)):
            outward = -n if plus > minus else n
            outward_source = f"twin_skin ({plus}/{minus} points at +-t)"
    info = {
        "normal": n.tolist(), "point": c.tolist(), "rms_mm": float(rms), "tol_mm": float(tol),
        "seed_surface_dist_mm": float(seed_dist), "ds_ball_points": int(len(ball)),
        "ds_inliers": int(len(inl)), "ds_region_points": int(len(region_idx)),
        "raw_points": int(mask.sum()), "outward_source": outward_source,
    }
    return mask, outward, info


def _auto_confidence(res, locus, *, override=False):
    reasons = []
    if locus == "apex" and res["apex"] is None:
        reasons.append("apex not found (bend surface not scanned)")
    if not override and res["coverage"] < 0.6:
        reasons.append(f"coverage {res['coverage']:.2f} < 0.60")
    if min(res["support_points"].values()) < 100:
        reasons.append("support < 100 points on a side")
    for key in ("plane_a", "plane_b"):
        if (res[key]["rms_mm"] or 0) > 0.25:
            reasons.append(f"{key} rms {res[key]['rms_mm']:.2f} mm > 0.25")
    if res["skin"] == "mixed":
        reasons.append("sides are on different skins")
    if res["zone_radius_used_mm"] > 1.6 * res["zone_radius_mm"]:
        reasons.append("support zone had to be widened")
    return ("certain" if not reasons else "uncertain"), reasons


def run_lines(prep_dir, pairs_path, out_dir, *, locus="apex", ri_mm=None, thickness_mm=None,
              scan_path=None, params: LineParams | None = None, margin_mm=None):
    if locus not in LOCI:
        raise ValueError(f"locus must be one of {LOCI}")
    if margin_mm is not None:
        params = dataclasses.replace(params or LineParams(), margin_mm=float(margin_mm))
    prep_dir, out_dir = Path(prep_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prep_dir / "prep_manifest.json").read_text())
    patches_doc = json.loads((prep_dir / "patches.json").read_text())
    patches = patches_doc["patches"]
    by_id = {p["id"]: p for p in patches}
    pairs_doc = json.loads(Path(pairs_path).read_text())
    pairs = parse_pairs(pairs_doc)
    if isinstance(pairs_doc, dict):
        thickness_mm = thickness_mm or pairs_doc.get("thickness_mm")
        ri_mm = ri_mm if ri_mm is not None else pairs_doc.get("ri_mm")
    t = float(thickness_mm or manifest["thickness_mm"])
    ri = float(ri_mm if ri_mm is not None else (manifest["ri_mm"] if not thickness_mm else t))
    for bend in pairs["bends"]:
        for side in bend["sides"]:
            if is_plane_side(side):
                continue
            missing = [i for i in side if i not in by_id]
            if missing:
                raise ValueError(f"unknown patch ids {missing}; prep has 1..{len(patches)}")

    scan = load_scan(scan_path or manifest["scan_path"])
    if scan.sha256 != manifest["scan_sha256"]:
        raise ValueError("scan sha256 differs from the prep manifest; re-run prep")
    ds = np.load(prep_dir / "points_ds.npz")
    labels = assign_full_resolution(scan, ds["points"], ds["labels"], patches,
                                    manifest["voxel_mm"], t)
    base_params = params or LineParams()
    ds_points, ds_normals = ds["points"], ds["normals"]
    ds_oriented = bool(manifest.get("normals_oriented", scan.oriented))
    voxel = float(manifest["voxel_mm"])
    results = []
    plane_masks = {}
    for k, bend in enumerate(pairs["bends"], start=1):
        bend_id = f"B{k}"
        entry = {"bend_id": bend_id, "patches": [side_label(side) if is_plane_side(side) else side
                                                 for side in bend["sides"]],
                 "note": bend["note"], "locus": locus}
        overrides = {key: bend[key] for key in ("margin_mm", "extent_t", "extent_xyz", "apex_mode")
                     if bend[key] is not None}
        plane_sides = {}
        masks, outward = [], [None, None]
        try:
            for j, side in enumerate(bend["sides"]):
                if is_plane_side(side):
                    mask, out_j, info = plane_side_points(side, scan, ds_points, ds_normals,
                                                          ds_oriented, voxel_mm=voxel, thickness_mm=t)
                    plane_sides["ab"[j]] = {**side, "fit": info}
                    plane_masks[(bend_id, j)] = mask
                    masks.append(mask)
                    outward[j] = out_j
                else:
                    masks.append(np.isin(labels, side))
        except BendLineError as exc:
            entry.update({"status": "failed", "error": str(exc)})
            if overrides:
                entry["overrides"] = overrides
            if plane_sides:
                entry["plane_sides"] = plane_sides
            results.append(entry)
            continue
        sides_pts = [scan.points[m] for m in masks]
        for j, side in enumerate(bend["sides"]):
            if is_plane_side(side):
                continue
            if scan.oriented and scan.normals is not None:
                outward[j] = scan.normals[masks[j]].mean(axis=0) if masks[j].any() else None
            else:
                outward[j] = _twin_outward(side, by_id)
        bend_params = base_params
        if bend["margin_mm"] is not None:
            bend_params = dataclasses.replace(base_params, margin_mm=bend["margin_mm"])
        try:
            res = compute_bend(scan.points, sides_pts[0], sides_pts[1], ri_mm=ri, thickness_mm=t,
                               params=bend_params,
                               outward_a=outward[0], outward_b=outward[1],
                               extent_t=bend["extent_t"], extent_xyz=bend["extent_xyz"],
                               apex_mode=bend["apex_mode"] or "envelope")
        except BendLineError as exc:
            entry.update({"status": "failed", "error": str(exc)})
            if overrides:
                entry["overrides"] = overrides
            if plane_sides:
                entry["plane_sides"] = plane_sides
            results.append(entry)
            continue
        line = res["apex"] if locus == "apex" and res["apex"] is not None else res["sharp"]
        is_override = res["extent_source"] == "override"
        auto, reasons = _auto_confidence(res, locus, override=is_override)
        # fold support of the final extent, from the downsampled cloud (all points, not patches)
        bis = np.asarray(res["bisector"])
        apex_shift = res["apex_offset_mm"] if line is res["apex"] else 0.0
        origin = np.asarray(res["line_origin"]) + apex_shift * bis
        support = fold_support(ds_points, origin, res["direction"], res["side_dirs"]["plane_a"],
                               res["side_dirs"]["plane_b"], res["t_range"][0], res["t_range"][1],
                               reach_mm=ri + t + 1.0,
                               min_points=bend_params.fold_support_min_points,
                               half_window_mm=max(0.5, 0.75 * voxel),
                               arc_band_mm=max(0.5, 0.5 * voxel))
        confidence = auto if bend["confidence"] == "auto" else bend["confidence"]
        if is_override:
            if support["frac"] < bend_params.fold_support_min_frac:
                reasons.append(f"override extent fold support {support['frac']:.2f} < "
                               f"{bend_params.fold_support_min_frac:.2f} (unsupported t: "
                               f"{support['gaps'][:6]})")
                auto = "uncertain"
                if confidence != "uncertain":
                    confidence = "uncertain"
                    if bend["confidence"] == "certain":
                        reasons.append("labeler 'certain' downgraded: override not supported by points")
        for j, key in enumerate("ab"):
            if key in plane_sides and outward[j] is None:
                reason = f"side {key} plane_from_points skin unknown (no normals, no twin skin)"
                reasons.append(reason)
                auto = "uncertain"
                if bend["confidence"] == "auto":
                    confidence = "uncertain"
        entry.update(res)
        entry.update({
            "status": "ok",
            "label_line": line,
            "label_locus": "apex" if line is res["apex"] else "sharp",
            "confidence": confidence,
            "auto_confidence": auto,
            "confidence_reasons": reasons,
            "fold_support_frac": support["frac"],
            "fold_support_gaps_t": support["gaps"],
            "override_support_frac": support["frac"] if is_override else None,
            "margin_mm": bend_params.margin_mm,
        })
        if overrides:
            entry["overrides"] = overrides
        if plane_sides:
            entry["plane_sides"] = plane_sides
        results.append(entry)

    ok = [r for r in results if r["status"] == "ok"]
    render_dir = out_dir / "render"
    images = {}
    title = manifest["scan_name"]
    images.update(render_verification(ds["points"], ds["normals"], ok, manifest["camera_frames"],
                                      render_dir, spacing_mm=manifest["voxel_mm"], thickness_mm=t,
                                      title=title, locus=locus))
    plane_id = max(by_id, default=0) + 1
    for k, res in enumerate(ok):
        color = LINE_PALETTE[k % len(LINE_PALETTE)]
        cmap = {pid: by_id[pid]["color"] for side in res["patches"] if isinstance(side, list)
                for pid in side}
        vis_labels = labels
        for j in range(2):
            mask = plane_masks.get((res["bend_id"], j))
            if mask is not None:
                vis_labels = vis_labels.copy() if vis_labels is labels else vis_labels
                vis_labels[mask] = plane_id + j
                cmap[plane_id + j] = ("#d62728", "#1f77b4")[j]
        path = render_closeup(scan.points, scan.normals, vis_labels, res, color,
                              render_dir / f"closeup_{res['bend_id']}.png",
                              patch_colors_map=cmap, spacing_mm=manifest["voxel_mm"])
        images[f"closeup_{res['bend_id']}"] = path
    doc = {
        "tool": TOOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "source_ply_raw",
        "scan_path": str(scan.path),
        "scan_name": scan.path.name,
        "scan_sha256": scan.sha256,
        "prep_dir": str(prep_dir.resolve()),
        "prep_image_sha256": manifest.get("image_sha256", {}),
        "pairs_file": str(Path(pairs_path).resolve()),
        "pairs_sha256": sha256_file(Path(pairs_path)),
        "labeler": pairs["labeler"],
        "locus": locus,
        "thickness_mm": t,
        "ri_mm": ri,
        "line_params": vars(params or LineParams()),
        "no_bends_present": pairs["no_bends_present"],
        "notes": pairs["notes"],
        "bends": results,
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "required_views": list(REQUIRED_VIEWS),
        "images": {k: str(Path(v).relative_to(out_dir)) for k, v in images.items()},
        "image_sha256": {k: sha256_file(Path(v)) for k, v in images.items()},
    }
    (out_dir / "lines.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc
