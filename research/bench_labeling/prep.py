"""``prep``: segment a scan into numbered planar patches and render them."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import TOOL_VERSION
from .geometry import BendLineError, LineParams, compute_bend
from .render import REQUIRED_VIEWS, camera_frames, patch_colors, render_patch_views
from .scanio import load_scan, sha256_file
from .segment import SegmentParams, segment, voxel_pick

DEFAULT_THICKNESS_MM = 2.0


def patch_adjacency(points, labels, radius_mm, coarse_mm=3.0):
    """Patch pairs that come within ``radius_mm`` of each other (bend arcs included)."""
    keep = labels > 0
    idx = np.flatnonzero(keep)
    if not len(idx):
        return {}
    sub = idx[voxel_pick(points[idx], coarse_mm)]
    tree = cKDTree(points[sub])
    pairs = tree.query_pairs(radius_mm, output_type="ndarray")
    adjacency: dict[int, set[int]] = {}
    if len(pairs):
        la, lb = labels[sub][pairs[:, 0]], labels[sub][pairs[:, 1]]
        diff = la != lb
        for a, b in np.unique(np.stack([np.minimum(la[diff], lb[diff]),
                                        np.maximum(la[diff], lb[diff])], axis=1), axis=0):
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
    return adjacency


def run_prep(scan_path, out_dir, *, params: SegmentParams | None = None, title=None,
             thickness_mm=None, ri_mm=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan = load_scan(scan_path)
    seg = segment(scan.points, scan.normals, scan.oriented, params)
    t = float(thickness_mm or seg.thickness_mm or DEFAULT_THICKNESS_MM)
    ri = float(ri_mm if ri_mm is not None else t)
    oriented_ds = seg.oriented
    for patch in seg.patches:
        patch["thin_strip"] = bool(patch["width_mm"] < 2.0 * t + 1.0)
    adjacency = patch_adjacency(seg.points, seg.labels, radius_mm=2.0 * (ri + t) + 4.0)
    for patch in seg.patches:
        patch["neighbors"] = sorted(adjacency.get(patch["id"], ()))
    colors = patch_colors(len(seg.patches), adjacency)
    for patch in seg.patches:
        patch["color"] = colors[patch["id"]]

    by_id = {p["id"]: p for p in seg.patches}
    candidates = []
    line_params = LineParams(min_support=15)
    for a in sorted(adjacency):
        for b in sorted(adjacency[a]):
            if b <= a:
                continue
            na, nb = np.asarray(by_id[a]["normal"]), np.asarray(by_id[b]["normal"])
            if abs(float(na @ nb)) > np.cos(np.radians(8.0)):
                continue
            pa = seg.points[seg.labels == a]
            pb = seg.points[seg.labels == b]
            oa = ob = None
            if oriented_ds:
                oa = seg.normals[seg.labels == a].mean(axis=0)
                ob = seg.normals[seg.labels == b].mean(axis=0)
            try:
                res = compute_bend(seg.points, pa, pb, ri_mm=ri, thickness_mm=t,
                                   params=line_params, outward_a=oa, outward_b=ob)
            except BendLineError:
                continue
            if res["zone_radius_used_mm"] > res["zone_radius_mm"] * 1.01:
                continue
            candidates.append({
                "patches": [a, b],
                "included_angle_deg": round(res["included_angle_deg"], 2),
                "length_mm": round(res["length_mm"], 1),
                "coverage": round(res["coverage"], 2),
                "skin": res["skin"],
                "involves_thin_strip": bool(by_id[a]["thin_strip"] or by_id[b]["thin_strip"]),
            })
    candidates.sort(key=lambda c: (c["involves_thin_strip"], c["skin"] == "mixed", -c["length_mm"]))

    frames = camera_frames(seg.points)
    title = title or scan.path.name
    render_dir = out_dir / "render"
    images = render_patch_views(seg.points, seg.normals, seg.labels, seg.patches, colors, frames,
                                render_dir, spacing_mm=seg.voxel_mm, thickness_mm=t, title=title)
    np.savez_compressed(out_dir / "points_ds.npz", points=seg.points, normals=seg.normals,
                        labels=seg.labels)
    patches_doc = {"patches": seg.patches, "candidate_pairs": candidates,
                   "candidate_note": ("Geometric suggestions only (adjacent, non-parallel, overlapping "
                                      "support). The labeler decides from the images; thin strips are "
                                      "usually thickness edges, not flanges.")}
    (out_dir / "patches.json").write_text(json.dumps(patches_doc, indent=2) + "\n")
    template = {
        "labeler": "claude-vision-ai-v1",
        "no_bends_present": False,
        "notes": "",
        "bends": [{"patches": [0, 0], "note": "", "confidence": "auto"}],
        "_help": ("Each bend: patches=[side_a, side_b]; a side is a patch id or a list of ids of the "
                  "same flange skin. Pick both sides on the SAME skin (both outer or both inner). "
                  "confidence: auto|certain|uncertain. Remove the placeholder entry."),
    }
    (out_dir / "pairs_template.json").write_text(json.dumps(template, indent=2) + "\n")
    manifest = {
        "tool": TOOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "source_ply_raw",
        "scan_path": str(scan.path),
        "scan_name": scan.path.name,
        "scan_sha256": scan.sha256,
        "scan_kind": scan.kind,
        "scan_points": int(len(scan.points)),
        "scan_faces": scan.n_faces,
        "normals_oriented": bool(scan.oriented),
        "voxel_mm": seg.voxel_mm,
        "ds_points": int(len(seg.points)),
        "segment_params": vars(params or SegmentParams()),
        "thickness_mm_estimated": seg.thickness_mm,
        "thickness_mm": t,
        "ri_mm": ri,
        "n_patches": len(seg.patches),
        "unsegmented_fraction": float((seg.labels == 0).mean()),
        "camera_frames": frames,
        "required_views": list(REQUIRED_VIEWS),
        "images": {k: str(Path(v).relative_to(out_dir)) for k, v in images.items()},
        "image_sha256": {k: sha256_file(Path(v)) for k, v in images.items()},
        "files": {"points": "points_ds.npz", "patches": "patches.json",
                  "pairs_template": "pairs_template.json"},
    }
    (out_dir / "prep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
