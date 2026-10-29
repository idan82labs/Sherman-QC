"""Scan-only candidate table for the Z4 selector (no labels, no CAD, no file names).

Per DEV scan:
1. Lines from the A0 liberal pool (``f1_uniform_scan_only_candidate_pool_v3``: candidates + line_objects)
   and, optionally, from extra eval-lines sources (Z2 / Z3 ``sherman.broad_baseline.eval_lines.v1``).
2. Near-identical lines are merged (2 deg / 1.5 mm / 0.8 mutual overlap, research/jev/pool_v2.cluster_lines);
   the representative is the best-supported member, or a source line when the cluster has no pool row.
3. Per cluster: pool descriptors of the representative, cluster consensus and generator flags, source flags,
   cross-section geometry (research/jev/features_v2.line_features_v2: dihedral, flange fits, apex side,
   corner distance, near-line support), neighbour context (other-skin parallels, joggle partners, collinear
   partners), boundary distance, and per-scan rank-normalised copies of the strongest descriptors.

Run (heavy, one process per slot):
  sherman_slot.sh .venv/bin/python -B -m research.zeroshot.z4_selector.candidates --worker 0 --workers 3 \
      [--source z2=<dir with <scan>/eval_lines.json>] [--tag base]
Writes WORK/z4_feats/<tag>/<scan>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.zeroshot.z4_selector.custody import (OLD_DEV_ROOT, POOL_ROOT, WORK, check_dev_scan,
                                                   check_source_path, dev_ids, dev_inventory)
from research.jev.pool_v2 import PoolLine, cluster_lines, load_pool_lines, pair_metrics_matrix

FEAT_ROOT = WORK / "z4_feats"
GENERATORS = ("cylindrical_standard", "cylindrical_wide", "plane_lattice", "bend_hypothesis",
              "control_support_strip", "strict_blind_owned", "line_object")
RANKED = ("two_sided_transition_score", "corridor_point_count", "finite_support_fraction",
          "two_sided_normal_coherence", "length_mm", "member_count")
MAX_POINTS = 1_500_000
#: numeric per-line fields an extra source may carry (Z2 eval_lines: score, cover, stations, ro_mm, sweep_deg)
SRC_META = ("score", "confidence", "cover", "stations", "ro_mm", "sweep_deg", "support", "slices", "n_support", "incl")


def pool_path(sid: str) -> Path | None:
    for root in (POOL_ROOT, OLD_DEV_ROOT):
        d = root / sid / "pool"
        if d.is_dir():
            files = sorted(p for p in d.glob("BB*.json") if not p.name.endswith(".row.json"))
            if files:
                return files[0]
    return None


def load_points(path: Path) -> np.ndarray:
    """Vertices plus an area-uniform surface sample (same recipe as research/jev/broad_ab_v1.load_points)."""
    import trimesh

    m = trimesh.load(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    pts = np.asarray(m.vertices, dtype=np.float64)
    faces = getattr(m, "faces", None)
    rng = np.random.default_rng(0)
    if faces is not None and len(faces):
        count = int(min(MAX_POINTS, max(200_000, len(pts))))
        sample, _ = trimesh.sample.sample_surface(m, count, seed=0)
        pts = np.vstack([pts, np.asarray(sample, dtype=np.float64)])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > MAX_POINTS:
        pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
    return pts


def load_source_lines(path: Path, name: str) -> list[PoolLine]:
    check_source_path(path)
    d = json.loads(Path(path).read_text())
    if d.get("frame", "scan_raw") != "scan_raw":
        raise ValueError(f"{path}: frame {d.get('frame')}")
    out = []
    for r in d.get("owned_bend_regions") or []:
        a, b = r.get("line_start"), r.get("line_end")
        if a is None or b is None:
            continue
        a, b = np.asarray(a, float), np.asarray(b, float)
        if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.linalg.norm(b - a) < 5.0:
            continue
        conf = r.get("confidence", r.get("score"))
        pl = PoolLine(a, b, f"src:{name}", {}, angle_deg=r.get("angle_deg") if isinstance(r.get("angle_deg"), (int, float)) else None,
                      vote_count=float(conf) if isinstance(conf, (int, float)) else None)
        pl.meta = {k: float(r[k]) for k in SRC_META if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)}
        out.append(pl)
    return out


def thickness(scan: Path) -> tuple[float | None, float]:
    """Sheet thickness from the D1 two-skin probe (research/zeroshot/d1_two_skin, read-only import)."""
    try:
        from research.zeroshot.d1_two_skin.two_skin import estimate_thickness, load_mesh

        est = estimate_thickness(load_mesh(str(scan)))
        return (float(est.t_mm) if est.t_mm else None), float(est.confidence or 0.0)
    except Exception:
        return None, 0.0


def neighbour_context(ends: np.ndarray) -> list[dict[str, Any]]:
    """Coordinate-free context against the scan's other candidates (as research/jev/input_opt_v2)."""
    n = len(ends)
    if n == 0:
        return []
    m = pair_metrics_matrix(ends, ends)
    ang, off, ov = m["axis_angle_deg"], m["transverse_offset_mm"], m["shorter_span_overlap"]
    out = []
    for i in range(n):
        ns = np.ones(n, bool)
        ns[i] = False
        par = ns & (ang[i] <= 5.0) & (ov[i] >= 0.5)
        skin = par & (off[i] > 0.4) & (off[i] <= 6.0)
        fam = par & (off[i] > 6.0) & (off[i] <= 40.0)
        col = ns & (ang[i] <= 3.0) & (off[i] <= 3.0) & (m["gold_span_overlap"][i] <= 0.05)
        so = off[i][skin]
        out.append({"ctx_near_duplicate_count": int((par & (off[i] <= 4.0)).sum()),
                    "ctx_parallel_within_6mm_count": int(skin.sum()),
                    "ctx_nearest_parallel_offset_mm": float(so.min()) if len(so) else 99.0,
                    "ctx_parallel_6_to_40mm_count": int(fam.sum()),
                    "ctx_collinear_count": int(col.sum())})
    return out


def build_scan(sid: str, inv: dict, sources: dict[str, Path]) -> dict[str, Any]:
    from research.jev.features_v2 import ScanClouds, line_features_v2

    scan = check_dev_scan(sid, inv)
    t0 = time.monotonic()
    lines: list[PoolLine] = []
    meta: dict[str, Any] = {"scan_id": sid, "sources": {}}
    pp = pool_path(sid)
    if pp is not None:
        pl, pm = load_pool_lines(pp)
        lines += pl
        meta["sources"]["a0_pool"] = {"path": str(pp), "rows": len(pl)}
    for name, root in sources.items():
        root, fname = (root if isinstance(root, tuple) else (root, "eval_lines.json"))
        f = root / sid / fname
        if f.is_file():
            sl = load_source_lines(f, name)
            lines += sl
            meta["sources"][name] = {"path": str(f), "rows": len(sl)}
    clusters = cluster_lines(lines) if lines else []
    # representative: prefer a pool row (has descriptors); source-only clusters keep their own line
    reps = []
    for c in clusters:
        pool_members = [j for j in c["members"] if not lines[j].generator.startswith("src:")]
        reps.append(c["rep"] if not pool_members or not lines[c["rep"]].generator.startswith("src:") else pool_members[0])
    # geometry: an extra source's line (in --source order) when the cluster has one, else the representative
    prio = {f"src:{n}": i for i, n in enumerate(sources)}
    geoms = []
    for c, r in zip(clusters, reps):
        src = sorted((prio[lines[j].generator], j) for j in c["members"] if lines[j].generator in prio)
        geoms.append(src[0][1] if src else r)
    ends = np.array([[lines[g].start, lines[g].end] for g in geoms], float).reshape(-1, 2, 3)
    ctx = neighbour_context(ends)
    pts = load_points(scan)
    clouds = ScanClouds(pts)
    t_mm, t_conf = thickness(scan)
    meta["thickness_mm"], meta["thickness_conf"] = t_mm, t_conf
    lo, hi = pts.min(0), pts.max(0)
    rows = []
    for k, (c, r, gi) in enumerate(zip(clusters, reps, geoms)):
        ln = lines[r]
        geo = lines[gi]
        mem = [lines[j] for j in c["members"]]
        gens = {m.generator for m in mem}
        radii = [m.radius_mm for m in mem if m.radius_mm is not None]
        angs = [m.angle_deg for m in mem if m.angle_deg is not None]
        votes = [m.vote_count for m in mem if m.vote_count is not None]
        f: dict[str, Any] = {"length_mm": geo.length, "member_count": len(mem),
                             "generator_count": len({g for g in gens if not g.startswith("src:")}),
                             "radius_median_mm": float(np.median(radii)) if radii else None,
                             "pool_angle_median_deg": float(np.median(angs)) if angs else None,
                             "vote_max": float(max(votes)) if votes else None}
        for g in GENERATORS:
            f[f"gen_{g}"] = int(g in gens or (g == "line_object" and "cross_source_line_object" in gens))
        for name in sources:
            f[f"src_{name}"] = int(f"src:{name}" in gens)
            sm = [getattr(m, "meta", {}) for m in mem if m.generator == f"src:{name}"]
            for key in SRC_META:
                vals = [x[key] for x in sm if key in x]
                if vals:
                    f[f"src_{name}_{key}"] = float(max(vals))
        f["pool_member_count"] = sum(not m.generator.startswith("src:") for m in mem)
        for kk, v in ln.descriptors.items():
            f[f"d_{kk}"] = v
        mid = 0.5 * (geo.start + geo.end)
        bd = np.minimum(mid - lo, hi - mid)
        f["bbox_min_dist_mm"] = float(bd.min())
        f["bbox_min_dist_rel"] = float(bd.min() / max(1e-6, clouds.extent))
        f["length_rel"] = geo.length / max(1e-6, clouds.extent)
        f.update(ctx[k])
        f.update({f"x_{kk}": v for kk, v in line_features_v2(clouds, geo.start, geo.end).items()})
        f["t_mm"], f["t_conf"] = t_mm, t_conf
        inc = f.get("x_median_included_angle_deg")
        if t_mm and isinstance(inc, (int, float)) and inc > 1.0:
            d2s = t_mm / max(0.05, math.sin(math.radians(inc) / 2.0)) + 1.0
            f["two_skin_dup_mm"] = d2s
            f["nearest_parallel_over_dup"] = f["ctx_nearest_parallel_offset_mm"] / d2s
        if t_mm and isinstance(f.get("x_median_apex_side_mm"), (int, float)):
            f["apex_side_over_t"] = f["x_median_apex_side_mm"] / t_mm
        rows.append({"k": k, "line_start": geo.start.tolist(), "line_end": geo.end.tolist(), "f": f})
    # per-scan rank normalisation
    for key in RANKED:
        col = [r["f"].get(key if key in ("length_mm", "member_count") else f"d_{key}") for r in rows]
        vals = np.array([v if isinstance(v, (int, float)) and v is not None else -1e9 for v in col], float)
        if len(vals):
            order = vals.argsort().argsort()
            for r, o in zip(rows, order):
                r["f"][f"rank_{key}"] = float(o / max(1, len(vals) - 1))
    for r in rows:
        r["f"]["scan_extent_mm"] = clouds.extent
        r["f"]["scan_cluster_count"] = len(rows)
        r["f"]["scan_point_spacing_mm"] = clouds.fine_spacing
        for name in sources:  # lets the model tell "source absent for this scan" from "source rejected line"
            r["f"][f"scan_has_{name}"] = int(name in meta["sources"])
    meta.update({"clusters": len(rows), "raw_lines": len(lines), "wall_s": round(time.monotonic() - t0, 1)})
    return {"meta": meta, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--source", action="append", default=[],
                    help="name=DIR[:FILE], reads DIR/<scan>/FILE (default eval_lines.json); order = geometry priority")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    sources = {}
    for s in a.source:
        n, d = s.split("=", 1)
        d, _, fname = d.partition(":")
        sources[n] = (check_source_path(d), fname or "eval_lines.json")
    inv = dev_inventory()
    out = FEAT_ROOT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ids = a.only or dev_ids()
    for sid in ids[a.worker::a.workers]:
        f = out / f"{sid}.json"
        if f.is_file() and not a.force:
            continue
        try:
            res = build_scan(sid, inv, sources)
        except Exception as e:  # keep the batch going; record the failure
            res = {"meta": {"scan_id": sid, "error": repr(e)}, "rows": []}
        f.write_text(json.dumps(res))
        with open(out / f"driver_{a.worker}.log", "a") as fo:
            fo.write(f"{time.strftime('%FT%T')} {sid} clusters={res['meta'].get('clusters')} "
                     f"wall={res['meta'].get('wall_s')} err={res['meta'].get('error')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
