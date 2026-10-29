"""Exploratory Jev (TypeSafe System One) experiment: can a Noul tell real press-brake
bends from confusers, given only coordinate-free, identity-free patch-pair descriptors?

Research evidence only. Not imported or wired anywhere.

Inputs (read-only): docs/production/runlog/benchmark_v2/label_prep/<scan>/{patches.json,
points_ds.npz, prep_manifest.json}, label_lines/<scan>/lines.json, BENCHMARK_V2.json.
Raw scans are never opened. Outputs: docs/production/runlog/benchmark_v2/jev/**.

What is sent to the API: per candidate pair, derived scalars (angles, areas, lengths,
gaps, ratios) plus plain-language buckets. Never coordinates, normals, file names,
SKUs, scan ids or patch ids. The key is read from .env.local and never printed.

Usage:
  OMP_NUM_THREADS=1 .venv/bin/python -m research.bench_labeling.jev_experiment build
  ... jev_experiment query --model jev-1.13.0
  ... jev_experiment evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .custody import check_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH = PROJECT_ROOT / "docs" / "production" / "runlog" / "benchmark_v2"
PREP = BENCH / "label_prep"
LINES = BENCH / "label_lines"
OUT = BENCH / "jev"
CANDIDATES = OUT / "candidates.jsonl"
API_URL = "https://api.typesafe.ai/v1/systemone"
MAX_CANDIDATES = 1500
QUESTIONS_PER_REQUEST = 40

QUESTION = ("Is this candidate pair of planar regions joined by a real press-brake bend "
            "(not a sheet-thickness edge, hole wall, outline or artifact)?")

CRITERIA = {
    "true": (
        "Yes: the two regions are faces of two different flanges of a sheet-metal part that "
        "was folded on a press brake, and the fold runs between them. Typical signs: the "
        "regions meet at a clear, non-trivial fold angle (not parallel); they share a long, "
        "straight common boundary that is a large part of at least one region's length; the "
        "gap between them is small (a few sheet thicknesses at most) and is filled by a "
        "narrow curved transition band whose surface normals sweep from one region's normal "
        "to the other's; and both regions are real faces of the sheet, clearly wider than "
        "one sheet thickness. Both regions may be on the same skin of the sheet or on "
        "opposite skins of the two flanges."),
    "false": (
        "No: anything other than a press-brake fold between two flanges. Confusers: "
        "(1) sheet-thickness edge - one region is a narrow strip about one sheet thickness "
        "wide meeting a face at roughly 90 degrees with a sharp corner and no curved band; "
        "(2) hole, slot or cutout wall - a small strip about one thickness wide around an "
        "opening; (3) the two skins of one flange - parallel regions facing opposite ways "
        "about one sheet thickness apart; (4) one flat face split in two - nearly parallel "
        "regions with almost no offset; (5) outline or perimeter facets, a short or "
        "point-like contact at a corner, scan noise, or regions that are near each other "
        "only because they belong to unrelated parts of the shape."),
}

STATE = {
    "setting": ("A sheet-metal part was 3D-scanned and its surface was split into flat "
                "(planar) regions by an automatic segmenter. Each question describes one "
                "candidate pair of regions that lie near each other. All quantities were "
                "measured in code from the scan; lengths are millimetres and angles are "
                "degrees. Sheet thickness is estimated per part from parallel opposite "
                "skins. 'Transition band' means scan surface lying between the two regions "
                "that the segmenter did not assign to any flat region (for a real bend this "
                "is the curved bend surface)."),
    "judgment": "Decide each pair independently from its own measurements.",
}

FORBIDDEN_DESCRIPTOR_KEYS = {"centroid", "anchor", "normal", "points", "scan", "scan_id",
                             "patch_id", "id", "sku", "part_id", "path", "file", "name"}


# --------------------------------------------------------------------------- selection

def _accepted_and_secondary():
    doc = json.loads((BENCH / "BENCHMARK_V2.json").read_text())
    accepted, secondary = [], []
    for part in doc["parts"].values():
        for scan in part["scans"]:
            run = scan.get("ai_label_run")
            if not run:
                continue
            row = {"scan_id": scan["scan_id"], "final_status": run["final_status"],
                   "role": run.get("role"), "saw_images": bool(run["verifier"]["saw_images"])}
            if run["final_status"] == "AI_GOLD_ACCEPTED":
                accepted.append(row)
            elif row["saw_images"]:
                secondary.append(row)
    return accepted, secondary


def _neighbor_pairs(patches):
    return sorted({tuple(sorted((p["id"], q))) for p in patches for q in p["neighbors"]})


def select_scans(cap=MAX_CANDIDATES):
    """All accepted scans; then whole NEEDS_HUMAN_REVIEW scans whose verifier saw images,
    smallest first, while the candidate cap allows."""
    accepted, secondary = _accepted_and_secondary()
    chosen, total = [], 0
    for row in sorted(accepted, key=lambda r: r["scan_id"]):
        n = len(_neighbor_pairs(json.loads((PREP / row["scan_id"] / "patches.json").read_text())["patches"]))
        chosen.append(dict(row, tier="accepted", n_pairs=n))
        total += n
    if total > cap:
        raise SystemExit(f"accepted scans alone have {total} candidate pairs > cap {cap}")
    sized = []
    for row in secondary:
        if not (LINES / row["scan_id"] / "lines.json").is_file():
            continue
        n = len(_neighbor_pairs(json.loads((PREP / row["scan_id"] / "patches.json").read_text())["patches"]))
        sized.append((n, row["scan_id"], row))
    for n, _, row in sorted(sized):
        if total + n > cap:
            continue
        chosen.append(dict(row, tier="secondary_unverified", n_pairs=n))
        total += n
    return chosen, total


# --------------------------------------------------------------------------- geometry

def _ang(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v))))))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _bucket(x, edges, names):
    for edge, name in zip(edges, names):
        if x < edge:
            return name
    return names[-1]


def _ratio_words(r):
    return _bucket(r, [0.6, 0.85, 1.2, 1.6, 2.5, 5.0, 1e9],
                   ["much less than one sheet thickness", "a bit under one sheet thickness",
                    "about one sheet thickness", "a bit over one sheet thickness",
                    "about two sheet thicknesses", "several sheet thicknesses",
                    "many sheet thicknesses"])


def _normal_spread_deg(normals, n, oriented):
    if len(normals) == 0:
        return 0.0
    d = normals @ n
    if not oriented:
        d = np.abs(d)
    ang = np.degrees(np.arccos(np.clip(d, -1, 1)))
    return float(np.sqrt(np.mean(ang ** 2)))


def scan_descriptors(scan_id):
    prep_dir = PREP / scan_id
    manifest = json.loads((prep_dir / "prep_manifest.json").read_text())
    check_path(manifest["scan_path"])                 # name-only custody guard; never opened
    doc = json.loads((prep_dir / "patches.json").read_text())
    patches = doc["patches"]
    by_id = {p["id"]: p for p in patches}
    geo_candidates = {tuple(sorted(c["patches"])) for c in doc["candidate_pairs"]}
    z = np.load(prep_dir / "points_ds.npz")
    pts, nrm, lab = z["points"], z["normals"], z["labels"]
    t = float(manifest["thickness_mm"])
    ri = float(manifest.get("ri_mm") or t)
    voxel = float(manifest["voxel_mm"])
    oriented = bool(manifest["normals_oriented"])
    radius = 2.0 * (ri + t) + 4.0                    # same adjacency radius as prep
    max_area = max(p["area_mm2"] for p in patches)
    idx = {p["id"]: np.flatnonzero(lab == p["id"]) for p in patches}
    trees = {pid: cKDTree(pts[ix]) for pid, ix in idx.items() if len(ix)}
    unl = np.flatnonzero(lab == 0)
    spread = {p["id"]: _normal_spread_deg(nrm[idx[p["id"]]], np.asarray(p["normal"]),
                                          oriented and p.get("normal_oriented", False))
              for p in patches}

    out = []
    for a, b in _neighbor_pairs(patches):
        pa_, pb_ = by_id[a], by_id[b]
        if a not in trees or b not in trees:
            continue
        both_oriented = oriented and pa_.get("normal_oriented") and pb_.get("normal_oriented")
        na, nb = _unit(np.asarray(pa_["normal"])), _unit(np.asarray(pb_["normal"]))
        ca, cb = np.asarray(pa_["centroid"]), np.asarray(pb_["centroid"])
        PA, PB = pts[idx[a]], pts[idx[b]]
        dA, _ = trees[b].query(PA, k=1, distance_upper_bound=radius)
        dB, _ = trees[a].query(PB, k=1, distance_upper_bound=radius)
        nearA, nearB = PA[np.isfinite(dA)], PB[np.isfinite(dB)]
        gap_min = float(min(dA.min(), dB.min())) if np.isfinite(min(dA.min(), dB.min())) else radius
        near_d = np.concatenate([dA[np.isfinite(dA)], dB[np.isfinite(dB)]])
        gap_med = float(np.median(near_d)) if len(near_d) else radius

        theta = _ang(na, nb)                          # angle between region normals
        folded = min(theta, 180.0 - theta)
        cross = np.cross(na, nb)
        sin = float(np.linalg.norm(cross))
        near_all = np.vstack([nearA, nearB]) if len(nearA) + len(nearB) else np.zeros((0, 3))
        if sin > math.sin(math.radians(8.0)):
            direction = cross / sin
        elif len(near_all) >= 3:
            w, v = np.linalg.eigh(np.cov((near_all - near_all.mean(0)).T))
            direction = v[:, -1]
        else:
            direction = None
        shared = 0.0
        if direction is not None and len(nearA) >= 5 and len(nearB) >= 5:
            sa, sb = nearA @ direction, nearB @ direction
            lo = max(np.percentile(sa, 2), np.percentile(sb, 2))
            hi = min(np.percentile(sa, 98), np.percentile(sb, 98))
            shared = float(max(0.0, hi - lo))

        # Plane-intersection line distance from each region's near edge (non-parallel only).
        line_off = None
        if sin > math.sin(math.radians(8.0)) and len(near_all):
            A = np.stack([na, nb])
            bb = np.array([na @ ca, nb @ cb])
            near_c = near_all.mean(0)
            p0 = near_c + A.T @ np.linalg.solve(A @ A.T, bb - A @ near_c)
            rel = near_all - p0
            perp = rel - np.outer(rel @ direction, direction)
            line_off = float(np.median(np.linalg.norm(perp, axis=1)))

        # Transition band: unsegmented points within reach of both regions.
        band_n, sweep = 0, None
        if len(unl):
            U = pts[unl]
            du_a, _ = trees[a].query(U, k=1, distance_upper_bound=radius)
            du_b, _ = trees[b].query(U, k=1, distance_upper_bound=radius)
            sel = np.isfinite(du_a) & np.isfinite(du_b)
            band_n = int(sel.sum())
            if band_n >= 5 and folded >= 8.0:
                m = nrm[unl[sel]]
                if both_oriented:
                    ta = np.degrees(np.arccos(np.clip(m @ na, -1, 1)))
                    tb = np.degrees(np.arccos(np.clip(m @ nb, -1, 1)))
                    span = theta
                else:
                    ta = np.degrees(np.arccos(np.clip(np.abs(m @ na), 0, 1)))
                    tb = np.degrees(np.arccos(np.clip(np.abs(m @ nb), 0, 1)))
                    span = folded
                between = np.abs(ta + tb - span) < 12.0
                strictly = between & (ta > 6.0) & (tb > 6.0)
                sweep = float(strictly.mean())
        band_area = band_n * voxel * voxel
        band_width = band_area / shared if shared > 1e-6 else None

        # Side relations (oriented only): is each region behind the other's outward face?
        side = None
        if both_oriented and folded >= 8.0:
            sa_ = float(np.median((PB - ca) @ na))
            sb_ = float(np.median((PA - cb) @ nb))
            if sa_ < 0 and sb_ < 0:
                side = "convex corner (each region lies behind the other's outward face)"
            elif sa_ > 0 and sb_ > 0:
                side = "concave corner (each region lies in front of the other's outward face)"
            else:
                side = "mixed (one region in front of, one behind, the other's outward face)"

        offset = None
        if folded < 15.0:
            offset = float(abs(np.median((PB - ca) @ na)) + abs(np.median((PA - cb) @ nb))) / 2.0

        small, large = sorted([pa_, pb_], key=lambda p: p["area_mm2"])
        if both_oriented:
            if theta < 10:
                rel_txt = "parallel, facing the same way"
            elif theta > 170:
                rel_txt = "parallel, facing opposite ways"
            else:
                rel_txt = _bucket(theta, [30, 60, 75, 105, 120, 150, 999],
                                  ["shallow fold", "moderate fold", "steep fold",
                                   "near right-angle fold", "steep fold past right angle",
                                   "sharp fold", "very sharp fold (nearly folded back)"])
        else:
            rel_txt = ("parallel (normal sign unknown)" if folded < 10 else
                       "folded (normal signs unknown, fold could be the angle or its supplement)")

        def region(p, other):
            return {
                "area_mm2": round(p["area_mm2"], 1),
                "length_mm": round(p["length_mm"], 1),
                "width_mm": round(p["width_mm"], 2),
                "width_in_sheet_thicknesses": round(p["width_mm"] / t, 2),
                "width_description": _ratio_words(p["width_mm"] / t),
                "is_narrow_strip_about_thickness_wide": bool(p["width_mm"] < 2.0 * t + 1.0),
                "share_of_largest_region_area": round(p["area_mm2"] / max_area, 3),
                "flatness_rms_mm": round(p["plane_rms_mm"], 3),
                "normal_spread_deg": round(spread[p["id"]], 1),
                "shared_boundary_as_share_of_this_region_length":
                    round(min(1.0, shared / max(p["length_mm"], 1e-6)), 2),
            }

        desc = {
            "sheet_thickness_mm": round(t, 2),
            "normals_have_known_outward_direction": bool(both_oriented),
            "relationship": rel_txt,
            "angle_between_region_normals_deg": round(theta if both_oriented else folded, 1),
            "fold_angle_deg": None if folded < 10 else round(theta if both_oriented else folded, 1),
            "included_angle_between_faces_deg": (None if folded < 10 else
                                                 round(180.0 - theta, 1) if both_oriented else None),
            "corner_side": side,
            "region_a": region(pa_, pb_),
            "region_b": region(pb_, pa_),
            "smaller_to_larger_area_ratio": round(small["area_mm2"] / large["area_mm2"], 3),
            "shared_boundary_length_mm": round(shared, 1),
            "closest_gap_mm": round(gap_min, 2),
            "closest_gap_description": _ratio_words(gap_min / t),
            "typical_edge_gap_mm": round(gap_med, 2),
            "typical_edge_gap_description": _ratio_words(gap_med / t),
            "plane_intersection_line_distance_from_near_edges_mm":
                None if line_off is None else round(line_off, 2),
            "transition_band_area_mm2": round(band_area, 1),
            "transition_band_mean_width_mm": None if band_width is None else round(band_width, 2),
            "transition_band_normals_sweeping_between_regions_share":
                None if sweep is None else round(sweep, 2),
            "parallel_plane_offset_mm": None if offset is None else round(offset, 2),
            "parallel_plane_offset_description": None if offset is None else _ratio_words(offset / t),
            "segmenter_marked_as_opposite_skin_twins": bool(pa_.get("twin") == b or pb_.get("twin") == a),
        }
        _assert_coordinate_free(desc)
        out.append({"scan_id": scan_id, "pair": [a, b], "descriptor": desc,
                    "prep_geometric_candidate": (a, b) in geo_candidates,
                    "twins": [pa_.get("twin"), pb_.get("twin")]})
    return out


def _assert_coordinate_free(obj, path="descriptor"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_DESCRIPTOR_KEYS:
                raise AssertionError(f"forbidden key {path}.{k}")
            _assert_coordinate_free(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        raise AssertionError(f"list value at {path} (coordinates/ids not allowed)")
    elif isinstance(obj, str) and ("sc_" in obj or ".ply" in obj):
        raise AssertionError(f"identity-like text at {path}")


# --------------------------------------------------------------------------- labels

def _label_bends(scan_id):
    lines = json.loads((LINES / scan_id / "lines.json").read_text())
    if lines.get("no_bends_present"):
        return [], True
    bends = []
    for b in lines["bends"]:
        bends.append({"sides": [sorted(b["patches"][0]), sorted(b["patches"][1])],
                      "status": b.get("status")})
    return bends, False


def assign_labels(rec, bends):
    a, b = rec["pair"]
    twin_a, twin_b = rec["twins"]
    fl_a = {a} | ({twin_a} if twin_a else set())
    fl_b = {b} | ({twin_b} if twin_b else set())
    strict = expanded = False
    for bend in bends:
        s0, s1 = set(bend["sides"][0]), set(bend["sides"][1])
        if (a in s0 and b in s1) or (a in s1 and b in s0):
            strict = True
        # flange-level: each side of the candidate lies on (either skin of) one labelled flange
        if (fl_a & s0 and fl_b & s1) or (fl_a & s1 and fl_b & s0):
            if not (fl_a & s0 and fl_a & s1):
                expanded = True
    return {"strict": strict, "flange": expanded or strict, "ambiguous": expanded and not strict}


# --------------------------------------------------------------------------- build

def cmd_build(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    chosen, total = select_scans(args.cap)
    rows = []
    for scan in chosen:
        recs = scan_descriptors(scan["scan_id"])
        bends, no_bends = _label_bends(scan["scan_id"])
        for rec in recs:
            rec.update(tier=scan["tier"], scan_role=scan.get("role"),
                       labels=assign_labels(rec, bends))
            key = hashlib.sha256(f"{rec['scan_id']}:{rec['pair'][0]}:{rec['pair'][1]}".encode()).hexdigest()[:16]
            rec["cand_key"] = key
            rows.append(rec)
        print(f"{scan['scan_id']} {scan['tier']:<22} pairs={len(recs)} bends={len(bends)}", flush=True)
    if len(rows) > args.cap:
        raise SystemExit(f"{len(rows)} candidates > cap {args.cap}")
    with CANDIDATES.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    sel = {"generated_at": datetime.now(timezone.utc).isoformat(), "cap": args.cap,
           "n_candidates": len(rows), "scans": chosen}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    print(f"wrote {len(rows)} candidates", flush=True)


# --------------------------------------------------------------------------- query

def _load_key():
    for line in (PROJECT_ROOT / ".env.local").read_text().splitlines():
        line = line.strip()
        if line.startswith("TYPESAFE_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                return v
    raise SystemExit("TYPESAFE_API_KEY not found in .env.local")


def _post(body, key, timeout=120):
    data = json.dumps(body, allow_nan=False).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8")), dict(resp.headers)


def cmd_query(args):
    key = _load_key()
    rows = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    raw_path = OUT / f"raw_{args.model}.jsonl"
    done = {}
    if raw_path.is_file():
        for line in raw_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("ok"):
                for k, v in rec["answers"].items():
                    done[k] = v
    todo = [r for r in rows if r["cand_key"] not in done]
    random.Random(20261017).shuffle(todo)            # mix scans/tiers inside each request
    print(f"model={args.model} candidates={len(rows)} already={len(done)} todo={len(todo)}", flush=True)
    with raw_path.open("a") as fh:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            questions = {
                f"c_{r['cand_key']}": {
                    "type": "noul",
                    "instructions": {"candidate_pair": r["descriptor"],
                                     "question": QUESTION.replace(
                                         "this candidate pair", "the `candidate_pair`")},
                    "criteria": CRITERIA,
                } for r in chunk}
            body = {"state": STATE, "model": args.model, "questions": questions}
            attempt, delay = 0, 2.0
            while True:
                attempt += 1
                t0 = time.time()
                try:
                    status, resp, headers = _post(body, key)
                    answers = {k[2:]: v.get("noul") for k, v in resp["answers"].items()}
                    fh.write(json.dumps({"ok": True, "at": datetime.now(timezone.utc).isoformat(),
                                         "request_model": args.model, "model": resp.get("model"),
                                         "usage": resp.get("usage"), "latency_s": round(time.time() - t0, 2),
                                         "request_id": headers.get("x-typesafe-request-id"),
                                         "answers": answers}) + "\n")
                    fh.flush()
                    print(f"  batch {start // args.batch + 1}: {len(answers)} answers "
                          f"model={resp.get('model')} usage={resp.get('usage')}", flush=True)
                    break
                except urllib.error.HTTPError as exc:
                    err = exc.read().decode("utf-8", "replace")[:800]
                    fh.write(json.dumps({"ok": False, "at": datetime.now(timezone.utc).isoformat(),
                                         "status": exc.code, "error": err,
                                         "request_id": exc.headers.get("x-typesafe-request-id")}) + "\n")
                    fh.flush()
                    print(f"  HTTP {exc.code}: {err[:300]}", flush=True)
                    if exc.code in (429, 529, 500, 502, 503, 504) and attempt < 6:
                        ra = exc.headers.get("retry-after")
                        wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                        time.sleep(wait)
                        delay *= 2
                        continue
                    raise SystemExit(f"API error {exc.code}; recorded in {raw_path.name}")
                except (urllib.error.URLError, TimeoutError) as exc:
                    fh.write(json.dumps({"ok": False, "at": datetime.now(timezone.utc).isoformat(),
                                         "status": None, "error": f"{type(exc).__name__}: {exc}"[:400]}) + "\n")
                    fh.flush()
                    if attempt < 4:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    raise SystemExit(f"API unreachable; recorded in {raw_path.name}")
            time.sleep(args.pause)


# --------------------------------------------------------------------------- evaluate

def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):                              # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def pr_at(y, s, thr):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pred = s >= thr
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"threshold": round(float(thr), 4), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


def best_threshold(y, s):
    best = None
    for thr in sorted(set(np.round(s, 4))):
        m = pr_at(y, s, thr)
        if best is None or m["f1"] > best["f1"] or (m["f1"] == best["f1"] and thr > best["threshold"]):
            best = m
    return best


def calibration(y, s, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0000001)):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()),
                         "mean_p": round(float(s[m].mean()), 3),
                         "observed_yes_rate": round(float(y[m].mean()), 3), "n_yes": int(y[m].sum())})
    return rows


def scan_bootstrap_auc(y, s, groups, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y, bool), np.asarray(s, float), np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([members[g] for g in pick])
        a = auc(y[ix], s[ix])
        if a is not None:
            vals.append(a)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def _load_answers(model):
    path = OUT / f"raw_{model}.jsonl"
    ans, meta = {}, {"requests_ok": 0, "requests_failed": 0, "input_tokens": 0, "models": set()}
    if not path.is_file():
        return ans, meta
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("ok"):
            meta["requests_ok"] += 1
            meta["input_tokens"] += int((rec.get("usage") or {}).get("input_tokens", 0))
            meta["models"].add(rec.get("model"))
            ans.update(rec["answers"])
        else:
            meta["requests_failed"] += 1
    meta["models"] = sorted(m for m in meta["models"] if m)
    return ans, meta


def evaluate_model(rows, model):
    ans, meta = _load_answers(model)
    res = {"model": model, "meta": meta, "answered": sum(r["cand_key"] in ans for r in rows)}
    for tier in ("accepted", "secondary_unverified"):
        sub = [r for r in rows if r["tier"] == tier and r["cand_key"] in ans]
        if not sub:
            continue
        tier_res = {}
        for variant in ("flange", "strict", "strict_excl_ambiguous"):
            use = [r for r in sub if not (variant == "strict_excl_ambiguous" and r["labels"]["ambiguous"])]
            lab = "strict" if variant.startswith("strict") else "flange"
            y = np.array([r["labels"][lab] for r in use])
            s = np.array([float(ans[r["cand_key"]]) for r in use])
            g = np.array([r["scan_id"] for r in use])
            prep = np.array([r["prep_geometric_candidate"] for r in use], float)
            a = auc(y, s)
            v = {"n": len(use), "n_pos": int(y.sum()), "roc_auc": None if a is None else round(a, 3),
                 "roc_auc_scan_bootstrap_95ci": scan_bootstrap_auc(y, s, g) if a is not None else None,
                 "best_f1": best_threshold(y, s) if y.any() else None,
                 "at_0_5": pr_at(y, s, 0.5), "at_0_9": pr_at(y, s, 0.9),
                 "brier": round(float(np.mean((s - y) ** 2)), 4),
                 "calibration": calibration(y, s),
                 "baseline_prep_geometric_candidate": pr_at(y, prep, 0.5)}
            tier_res[variant] = v
        # zero-bend control false-positive view and per-scan AUC
        per_scan = {}
        for sid in sorted({r["scan_id"] for r in sub}):
            rs = [r for r in sub if r["scan_id"] == sid]
            y = np.array([r["labels"]["flange"] for r in rs])
            s = np.array([float(ans[r["cand_key"]]) for r in rs])
            a = auc(y, s)
            per_scan[sid] = {"role": rs[0].get("scan_role"), "n": len(rs), "n_pos_flange": int(y.sum()),
                             "auc_flange": None if a is None else round(a, 3),
                             "n_p_ge_0_5": int((s >= 0.5).sum()), "n_p_ge_0_9": int((s >= 0.9).sum()),
                             "max_p_negative": round(float(s[~y].max()), 3) if (~y).any() else None,
                             "min_p_positive": round(float(s[y].min()), 3) if y.any() else None}
        tier_res["per_scan"] = per_scan
        # confuser breakdown on negatives (flange variant)
        neg = [r for r in sub if not r["labels"]["flange"]]
        cats = {}
        for r in neg:
            d = r["descriptor"]
            if d["segmenter_marked_as_opposite_skin_twins"]:
                c = "opposite-skin twin"
            elif d["fold_angle_deg"] is None:
                c = "parallel / coplanar"
            elif d["region_a"]["is_narrow_strip_about_thickness_wide"] or d["region_b"]["is_narrow_strip_about_thickness_wide"]:
                c = "involves thin strip (thickness edge / hole wall)"
            else:
                c = "non-parallel, both wide"
            cats.setdefault(c, []).append(float(ans[r["cand_key"]]))
        pos = [float(ans[r["cand_key"]]) for r in sub if r["labels"]["flange"]]
        tier_res["score_by_category"] = {
            k: {"n": len(v), "mean_p": round(float(np.mean(v)), 3), "share_ge_0_5": round(float(np.mean(np.array(v) >= 0.5)), 3)}
            for k, v in sorted(cats.items())}
        if pos:
            tier_res["score_by_category"]["POSITIVE (flange-level bend)"] = {
                "n": len(pos), "mean_p": round(float(np.mean(pos)), 3),
                "share_ge_0_5": round(float(np.mean(np.array(pos) >= 0.5)), 3)}
        tier_res["rule_baselines"] = rule_baselines(sub, ans)
        res[tier] = tier_res
    return res


def _rule_wide_fold(d):
    """Hand rule with no training: non-parallel, neither region a thin strip, not twins."""
    return (d["fold_angle_deg"] is not None and not d["segmenter_marked_as_opposite_skin_twins"]
            and not d["region_a"]["is_narrow_strip_about_thickness_wide"]
            and not d["region_b"]["is_narrow_strip_about_thickness_wide"])


def _feat(d, name):
    if name == "max_shared_boundary_share":
        return max(d["region_a"]["shared_boundary_as_share_of_this_region_length"],
                   d["region_b"]["shared_boundary_as_share_of_this_region_length"])
    if name == "min_shared_boundary_share":
        return min(d["region_a"]["shared_boundary_as_share_of_this_region_length"],
                   d["region_b"]["shared_boundary_as_share_of_this_region_length"])
    if name == "neg_typical_edge_gap_mm":
        return -d["typical_edge_gap_mm"]
    if name == "min_region_width_over_t":
        return min(d["region_a"]["width_in_sheet_thicknesses"], d["region_b"]["width_in_sheet_thicknesses"])
    v = d.get(name)
    return -1.0 if v is None else float(v)


BASELINE_FEATURES = ("shared_boundary_length_mm", "max_shared_boundary_share", "min_shared_boundary_share",
                     "transition_band_normals_sweeping_between_regions_share", "neg_typical_edge_gap_mm",
                     "min_region_width_over_t", "smaller_to_larger_area_ratio")


def rule_baselines(sub, ans):
    """Flange-level labels: hand rule, single descriptor features, and Jev inside the rule set.
    The single features are oracle-direction, untrained, unthresholded rankers (AUC only)."""
    y = np.array([r["labels"]["flange"] for r in sub])
    s = np.array([float(ans[r["cand_key"]]) for r in sub])
    rule = np.array([_rule_wide_fold(r["descriptor"]) for r in sub], float)
    out = {"rule_wide_fold_at_0_5": pr_at(y, rule, 0.5),
           "single_feature_auc_all": {f: (lambda a: None if a is None else round(a, 3))(
               auc(y, [_feat(r["descriptor"], f) for r in sub])) for f in BASELINE_FEATURES}}
    m = rule > 0.5
    if m.any() and y[m].any() and (~y[m]).any():
        sub_m = [r for r, k in zip(sub, m) if k]
        out["inside_rule_set"] = {
            "n": int(m.sum()), "n_pos": int(y[m].sum()),
            "jev_auc": round(auc(y[m], s[m]), 3),
            "jev_best_f1": best_threshold(y[m], s[m]),
            "single_feature_auc": {f: (lambda a: None if a is None else round(a, 3))(
                auc(y[m], [_feat(r["descriptor"], f) for r in sub_m])) for f in BASELINE_FEATURES},
        }
    jr = (s >= 0.5).astype(float) * rule
    out["jev_ge_0_5_and_rule"] = pr_at(y, jr, 0.5)
    return out


def cmd_evaluate(args):
    rows = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    results = {"generated_at": datetime.now(timezone.utc).isoformat(), "n_candidates": len(rows),
               "question": QUESTION, "criteria": CRITERIA, "state": STATE,
               "models": [evaluate_model(rows, m) for m in args.models]}
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=list) + "\n")
    # merged per-candidate scores
    with (OUT / "scores.jsonl").open("w") as fh:
        answers = {m: _load_answers(m)[0] for m in args.models}
        for r in rows:
            fh.write(json.dumps({"cand_key": r["cand_key"], "scan_id": r["scan_id"], "pair": r["pair"],
                                 "tier": r["tier"], "labels": r["labels"],
                                 "prep_geometric_candidate": r["prep_geometric_candidate"],
                                 "p_yes": {m: answers[m].get(r["cand_key"]) for m in args.models}}) + "\n")
    print(json.dumps({m["model"]: {t: {v: (m[t][v]["roc_auc"], m[t][v]["best_f1"]) for v in ("flange", "strict", "strict_excl_ambiguous")}
                                   for t in ("accepted", "secondary_unverified") if t in m}
                      for m in results["models"]}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cap", type=int, default=MAX_CANDIDATES)
    q = sub.add_parser("query")
    q.add_argument("--model", default="jev-1.13.0")
    q.add_argument("--batch", type=int, default=QUESTIONS_PER_REQUEST)
    q.add_argument("--pause", type=float, default=1.0)
    e = sub.add_parser("evaluate")
    e.add_argument("--models", nargs="+", default=["jev-1.13.0"])
    args = ap.parse_args(argv)
    {"build": cmd_build, "query": cmd_query, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
