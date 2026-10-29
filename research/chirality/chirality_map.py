"""Compile-time chirality map ``chirality_map_v0`` of a compiled part (E1 side, research).

usage: python -m research.chirality.chirality_map COMPILED_DIR [--out DIR] [--stages S2,S12]

Input is only the CAD: E1's compiled stage sample clouds (``stages/S<k>.npz``, the B-rep sampled on a 0.5 mm grid
of each face's own parameters) and the compiled panels (planes, outer boundaries, hole loops from the B-rep). No
scan is read. Nothing in ``compiled_part.json`` or the stage files is changed: the map is a new artifact,
``chirality_map_v0.json`` + ``chirality_map_v0.npz`` (probe points), written beside the compiled part or to --out.

Per stage S_k:

1. **Mirror fits.** The stage cloud C is mirrored (any reflection; all are equivalent up to a proper motion) and
   fitted back onto C by point-to-plane ICP from 24 starts (the 24 improper symmetries of the principal-axis cube),
   reusing E1's own symmetry-test ICP and scorer (``compile.symmetry._icp`` / ``_score``) read-only. Each distinct
   optimum H (an improper rigid map) is a *mirror hypothesis*: where a mirrored part would sit after a proper
   registration onto the CAD.
2. **Deviation.** For each H, at full sample density: d_A(p) = distance from a CAD point p to H(C) (material the
   proper image has and the mirror image lacks) and d_B(q) = distance from q in H(C) to C (material only the mirror
   image has).
3. **Features.** Points with d > ``dev_mm`` are clustered (``link_mm`` linkage); a cluster of at least
   ``min_cluster_points`` samples is a *distinguishing feature*. Its *core probes* are skin samples with
   d >= ``core_mm`` (where the two images certainly differ), its *ring probes* the CAD skin samples around it that
   both images share (coverage reference). Type/size come from the compiled panel's B-rep hole loops and outer
   boundary (HOLE / CUTOUT / NOTCH / TAB / FORM).
4. **Class.** ACHIRAL: some mirror fit leaves no distinguishing feature. CHIRAL_GROSS: the best fit still leaves
   more than ``gross_fraction`` of the surface deviating. CHIRAL_BY_FEATURES otherwise.

Probe points are stored in the **final-pose frame** of their owning panel (inverse of the stage's panel
transform), so E3 can carry them onto the scan with the articulated fit's per-panel pose.

Everything is geometry; no part number or part-specific rule is used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SCHEMA = "chirality_map_v0"
METHOD = "MIRROR_FIT_FEATURE_DIFF_V0"

#: Frozen 2026-10-28 before any synthetic evaluation (sha256 of the canonical JSON is recorded in the map).
THRESHOLDS: dict[str, float] = {
    "fit_voxel_mm": 1.0,            # ICP target decimation (as E1's symmetry test)
    "fit_source_points": 4000,      # ICP / scoring source sample
    "fit_iterations": 40,
    "eval_voxel_mm": 0.5,           # full-density deviation (E1's sample grid)
    "dev_mm": 1.0,                  # a sample deviating more than this is not shared by the two images
    "core_mm": 2.0,                 # probe core: the two images differ by at least this here
    "shared_mm": 0.5,               # ring probes: both images have material within this
    "link_mm": 3.0,                 # feature cluster linkage
    "min_cluster_points": 20,       # distinguishing feature: >= this many deviating samples (~5 mm^2 at 0.5 mm)
    "gross_fraction": 0.05,         # CHIRAL_GROSS: best mirror fit leaves > 5 % of samples deviating
    "hyp_overlap_min": 0.5,         # a mirror hypothesis must overlap the CAD on >= half its samples
    "hyp_max": 6,
    "hyp_dedupe_deg": 2.0,
    "hyp_dedupe_mm": 1.0,
    "ring_inner_mm": 1.0,
    "ring_outer_mm": 5.0,
    "probe_cap": 300,               # per feature and role
    "min_core_probes": 8,           # a feature is verifiable with >= this many core probes (one role)
    "side_dot": 0.5,                # skin side of a probe: |normal . panel normal| above this
}
THRESHOLDS_SHA256 = hashlib.sha256(json.dumps(THRESHOLDS, sort_keys=True).encode()).hexdigest()

ROLE_A, ROLE_B, ROLE_RING = 0, 1, 2  # proper-only material, mirror-only material, shared ring
#: distance cap (mm) of deviation queries: deviations are reported capped here (well above every threshold)
_CAP = 10.0


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _components(points: np.ndarray, link: float) -> np.ndarray:
    """Connected components of ``points`` under ``link`` linkage (labels)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(link, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab


def _vox_first(points: np.ndarray, cell: float) -> np.ndarray:
    keys = np.floor(points / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


class _Panels:
    """Final-pose panel planes, outer boundaries and hole loops (from the compiled B-rep), in 2-D."""

    def __init__(self, compiled: dict):
        from matplotlib.path import Path as MPath

        self.ids = [p["panel_id"] for p in compiled["panels"]]
        self.normal, self.origin, self.basis, self.outer, self.holes = [], [], [], [], []
        for p in compiled["panels"]:
            n = np.asarray(p["skin0_plane"]["normal"], float)
            n /= np.linalg.norm(n)
            o = np.asarray(p["skin0_plane"]["point_mm"], float)
            u, v = _plane_basis(n)
            self.normal.append(n)
            self.origin.append(o)
            self.basis.append((u, v))
            to2 = lambda poly: np.c_[(np.asarray(poly, float) - o) @ u, (np.asarray(poly, float) - o) @ v]  # noqa: E731
            self.outer.append(MPath(to2(p["outer_boundary_mm"])))
            self.holes.append([(MPath(to2(h)), to2(h)) for h in p.get("holes_mm", [])])

    def to2(self, i: int, x: np.ndarray) -> np.ndarray:
        u, v = self.basis[i]
        return np.c_[(x - self.origin[i]) @ u, (x - self.origin[i]) @ v]

    def classify_opening(self, i: int, x_final: np.ndarray) -> tuple[str, list[float] | None]:
        """What the CAD (final pose) has at points ``x_final`` of panel ``i`` where it has no material."""
        if i < 0:
            return "UNPLACED", None
        p2 = self.to2(i, x_final)
        c = p2.mean(axis=0)
        for path, poly in self.holes[i]:
            if path.contains_point(c) or path.contains_points(p2).mean() > 0.5:
                ext = poly.max(axis=0) - poly.min(axis=0)
                kind = "HOLE" if max(ext) <= 12.0 and min(ext) > 0.6 * max(ext) else "CUTOUT"
                return kind, sorted([round(float(e), 1) for e in ext], reverse=True)
        if self.outer[i].contains_points(p2).mean() > 0.5:
            return "OPENING", None
        return "NOTCH", None


def _stage_frame(compiled: dict, stage: dict) -> list[np.ndarray]:
    return [np.asarray(stage["panel_transforms"][pid], float) for pid in (p["panel_id"] for p in compiled["panels"])]


def _fit_candidates(P: np.ndarray, Nn: np.ndarray, th: dict) -> tuple[list[tuple[float, np.ndarray]], dict]:
    """Distinct improper self-fits (4x4) of the cloud, with E1's ICP and score: [(rms, M4)] best first."""
    from domains.production.engine.compile.symmetry import _icp, _score, voxel_decimate

    keep = voxel_decimate(P, th["fit_voxel_mm"])
    tgt, tgt_n = P[keep], Nn[keep]
    tree = cKDTree(tgt)
    step = max(1, len(tgt) // int(th["fit_source_points"]))
    src, src_n = tgt[::step], tgt_n[::step]
    center = tgt.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((tgt - center).T))
    refl = np.eye(3) - 2.0 * np.outer(evecs[:, 0], evecs[:, 0])
    found: list[tuple[float, np.ndarray]] = []
    for R in Rotation.create_group("O").as_matrix():
        lin0 = evecs @ R @ evecs.T @ refl
        lin, trans = _icp(src, src_n, tree, tgt, tgt_n, lin0, center, int(th["fit_iterations"]),
                          max_dist=4.0 * th["fit_voxel_mm"])
        rms = _score(src, src_n, tree, tgt, tgt_n, lin, trans, center, th["fit_voxel_mm"])
        m4 = np.eye(4)
        m4[:3, :3] = lin
        m4[:3, 3] = center + trans - lin @ center
        dup = False
        for k, (r2, o) in enumerate(found):
            rr = o[:3, :3].T @ m4[:3, :3]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rr) - 1) / 2))))
            if ang < th["hyp_dedupe_deg"] and np.linalg.norm(o[:3, 3] - m4[:3, 3]) < 5 * th["hyp_dedupe_mm"] and \
                    np.linalg.norm((src @ m4[:3, :3].T + m4[:3, 3]) - (src @ o[:3, :3].T + o[:3, 3]), axis=1).max() \
                    < th["hyp_dedupe_mm"]:
                if rms < r2:
                    found[k] = (rms, m4)
                dup = True
                break
        if not dup:
            found.append((rms, m4))
    found.sort(key=lambda t: t[0])
    return found, {"pca_eigenvalues": [round(float(e), 3) for e in evals], "fit_points": int(len(tgt))}


def stage_map(compiled: dict, cdir: Path, stage: dict, th: dict = THRESHOLDS) -> tuple[dict, dict]:
    """Chirality of one stage: (record, probe arrays)."""
    z = np.load(cdir / stage["sample_cloud"]["rel_path"], allow_pickle=False)
    P0 = z["points"].astype(np.float64)
    N0 = z["normals"].astype(np.float64)
    kind0, idx0 = z["segment_kind"].astype(np.int64), z["segment_index"].astype(np.int64)
    sel = _vox_first(P0, th["eval_voxel_mm"])
    P, Nn, kind, idx = P0[sel], N0[sel], kind0[sel], idx0[sel]
    panels = _Panels(compiled)
    owner = np.where(kind == 1, -1, idx)  # bend zones: no rigid owner (unrolled at unformed stages)
    TS = _stage_frame(compiled, stage)
    TSinv = [np.linalg.inv(T) for T in TS]
    t_mm = float(compiled["thickness"]["t_mm"])
    stage_normals = [TS[i][:3, :3] @ panels.normal[i] for i in range(len(TS))]
    stage_origin = [TS[i][:3, :3] @ panels.origin[i] + TS[i][:3, 3] for i in range(len(TS))]

    cands, fitinfo = _fit_candidates(P, Nn, th)
    # coarse screen (1 mm cloud): overlap and deviating share of every fit; only the most promising few are
    # evaluated at full density (the best by deviating share decides ACHIRAL; the best by RMS are hypotheses)
    ck = _vox_first(P, th["fit_voxel_mm"])
    Pc = P[ck]
    treePc = cKDTree(Pc)
    coarse = []
    for rank, (rms, H) in enumerate(cands):
        HPc = Pc @ H[:3, :3].T + H[:3, 3]
        a, _ = cKDTree(HPc).query(Pc, distance_upper_bound=_CAP)
        b, _ = treePc.query(HPc, distance_upper_bound=_CAP)
        a = np.where(np.isfinite(a), a, _CAP)
        b = np.where(np.isfinite(b), b, _CAP)
        coarse.append((rank, float(np.mean(a <= th["dev_mm"] + th["fit_voxel_mm"])),
                       float(max(np.mean(a > th["dev_mm"] + th["fit_voxel_mm"]), np.mean(b > th["dev_mm"] + th["fit_voxel_mm"])))))
    by_dev = [c[0] for c in sorted(coarse, key=lambda c: (c[2], cands[c[0]][0]))[:3] if c[2] < 2 * th["gross_fraction"]]
    by_rms = [c[0] for c in coarse if c[1] >= th["hyp_overlap_min"]][: int(th["hyp_max"])]
    chosen = sorted(set(by_dev) | set(by_rms) | {0})
    fitinfo["coarse"] = [{"rank": r, "rms_mm": round(cands[r][0], 4), "overlap_coarse": round(o, 4),
                          "deviating_coarse": round(dv, 5)} for r, o, dv in coarse]
    treeP = cKDTree(P)
    hyps = []
    achiral_by = None
    for rank in chosen:
        rms, H = cands[rank]
        HP = P @ H[:3, :3].T + H[:3, 3]
        dA, _ = cKDTree(HP).query(P, distance_upper_bound=_CAP)
        dB, nnB = treeP.query(HP, distance_upper_bound=_CAP)
        dA = np.where(np.isfinite(dA), dA, _CAP)
        dB = np.where(np.isfinite(dB), dB, _CAP)
        nnB = np.where(nnB < len(P), nnB, 0)
        overlap = float(np.mean(dA <= th["dev_mm"]))
        devA, devB = dA > th["dev_mm"], dB > th["dev_mm"]
        dev_frac = float(max(devA.mean(), devB.mean()))
        # clusters of deviating samples (A and B jointly: an offset hole is one feature)
        D = np.r_[P[devA], HP[devB]]
        role = np.r_[np.zeros(int(devA.sum()), int), np.ones(int(devB.sum()), int)]
        src_idx = np.r_[np.where(devA)[0], np.where(devB)[0]]
        feats = []
        if len(D):
            rep = _vox_first(D, 1.0)
            lab_rep = _components(D[rep], th["link_mm"])
            _, near = cKDTree(D[rep]).query(D)
            lab = lab_rep[near]
            sizes = np.bincount(lab)
            for c in np.where(sizes >= th["min_cluster_points"])[0]:
                feats.append({"mask": lab == c})
        hyps.append(dict(rank=rank, rms=rms, H=H, HP=HP, dA=dA, dB=dB, nnB=nnB, overlap=overlap, dev_frac=dev_frac,
                         D=D, role=role, src=src_idx, feats=feats))
        if not feats and achiral_by is None:
            achiral_by = rank
    best = min(hyps, key=lambda h: (h["dev_frac"], h["rms"]))
    if achiral_by is not None:
        cls = "ACHIRAL"
    elif best["dev_frac"] > th["gross_fraction"]:
        cls = "CHIRAL_GROSS"
    else:
        cls = "CHIRAL_BY_FEATURES"
    e1sym = stage["symmetry"]
    rec = {
        "stage_model_id": stage["stage_model_id"],
        "stage_index": stage["stage_index"],
        "formed_bend_uids": list(stage["formed_bend_uids"]),
        "sample_cloud_sha256": stage["sample_cloud"]["sha256"],
        "class": cls,
        "e1_symmetry": {"is_symmetric": bool(e1sym["is_symmetric"]),
                        "best_mirror_rms_mm": min([c["rms_mm"] for c in e1sym["candidates"] if c["kind"] == "MIRROR"],
                                                  default=None)},
        "fit": fitinfo,
        "mirror_fits": [{"rank": h["rank"], "rms_mm": round(h["rms"], 4), "overlap": round(h["overlap"], 4),
                         "deviating_fraction": round(h["dev_frac"], 5), "n_features": len(h["feats"])} for h in hyps],
        "achiral_fit_rank": achiral_by,
        "hypotheses": [],
    }
    probes = {k: [] for k in ("pos", "nrm", "owner", "side", "hyp", "feature", "role")}
    if cls == "ACHIRAL":
        return rec, probes
    keep_h = [h for h in sorted(hyps, key=lambda h: h["rms"]) if h["overlap"] >= th["hyp_overlap_min"]]
    if not keep_h:
        keep_h = [best]
    keep_h = keep_h[: int(th["hyp_max"])]
    rng = np.random.default_rng(0)
    treeA_shared = None
    for hi, h in enumerate(keep_h):
        H, HP = h["H"], h["HP"]
        Hinv = np.linalg.inv(H)
        hrec = {"hypothesis": hi, "fit_rank": h["rank"], "transform": [[round(float(v), 6) for v in r] for r in H],
                "rms_mm": round(h["rms"], 4), "overlap": round(h["overlap"], 4),
                "deviating_fraction": round(h["dev_frac"], 5), "features": []}
        shared = h["dA"] < th["shared_mm"]
        for fi, f in enumerate(sorted(h["feats"], key=lambda f: -int(f["mask"].sum()))):
            m = f["mask"]
            fr, fs = h["role"][m], h["src"][m]
            pts = h["D"][m]
            # owners: CAD samples own themselves; mirror-image samples take the nearest CAD sample's owner
            own = np.where(fr == ROLE_A, owner[fs], owner[h["nnB"][fs]])
            known = own[own >= 0]
            panel = int(np.bincount(known).argmax()) if len(known) else -1
            # core probes per role (skins only: side defined by the owner's panel normal)
            out = {}
            for role in (ROLE_A, ROLE_B):
                rm = fr == role
                if role == ROLE_A:
                    d = h["dA"][fs[rm]]
                    p_st, n_st = P[fs[rm]], Nn[fs[rm]]
                else:
                    d = h["dB"][fs[rm]]
                    p_st, n_st = HP[fs[rm]], Nn[fs[rm]] @ H[:3, :3].T
                o = own[rm]
                core = (d >= th["core_mm"]) & (o >= 0)
                out[role] = (p_st[core], n_st[core], o[core])
            # ring: shared CAD skin samples between ring_inner and ring_outer of the feature, on the feature's panels
            box_lo, box_hi = pts.min(axis=0) - th["ring_outer_mm"], pts.max(axis=0) + th["ring_outer_mm"]
            near_box = np.all((P >= box_lo) & (P <= box_hi), axis=1)
            dd = np.full(len(P), np.inf)
            if near_box.any():
                dd[near_box], _ = cKDTree(pts).query(P[near_box], distance_upper_bound=th["ring_outer_mm"])
            ring = shared & np.isfinite(dd) & (dd >= th["ring_inner_mm"]) & (owner >= 0)
            panels_f = set(int(v) for v in np.unique(known)) if len(known) else set()
            ring &= np.isin(owner, list(panels_f)) if panels_f else ring
            out[ROLE_RING] = (P[ring], Nn[ring], owner[ring])
            n_core = {}
            for role, (pp, nn, oo) in out.items():
                if len(pp) == 0:
                    n_core[role] = 0
                    continue
                pf = np.empty_like(pp)
                nf = np.empty_like(nn)
                for i in np.unique(oo):
                    s = oo == i
                    Ti = TSinv[int(i)]
                    pf[s] = pp[s] @ Ti[:3, :3].T + Ti[:3, 3]
                    nf[s] = nn[s] @ Ti[:3, :3].T
                side = np.zeros(len(pp), int)
                for i in np.unique(oo):
                    s = oo == i
                    dot = nf[s] @ panels.normal[int(i)]
                    side[s] = np.where(dot > th["side_dot"], 1, np.where(dot < -th["side_dot"], -1, 0))
                ok = side != 0
                pf, nf, oo2, side = pf[ok], nf[ok], oo[ok], side[ok]
                if len(pf) > th["probe_cap"]:
                    pick = np.sort(rng.choice(len(pf), int(th["probe_cap"]), replace=False))
                    pf, nf, oo2, side = pf[pick], nf[pick], oo2[pick], side[pick]
                n_core[role] = int(len(pf))
                probes["pos"].append(pf)
                probes["nrm"].append(nf)
                probes["owner"].append(oo2)
                probes["side"].append(side)
                probes["hyp"].append(np.full(len(pf), hi))
                probes["feature"].append(np.full(len(pf), fi))
                probes["role"].append(np.full(len(pf), role))
            # type and size from the B-rep panel loops (final pose)
            ftype, fsize = "UNCLASSIFIED", None
            out_of_plane = False
            if panel >= 0:
                dist_plane = np.abs((pts - stage_origin[panel]) @ stage_normals[panel])
                out_of_plane = bool(np.mean(dist_plane > t_mm + 0.5) > 0.3)
            if out_of_plane:
                ftype = "FORM"
            elif panel >= 0:
                bpts = pts[fr == ROLE_B]
                apts = pts[fr == ROLE_A]
                Tinv = TSinv[panel]
                if len(bpts) >= len(apts) and len(bpts):
                    # the CAD lacks material here (the mirror image has it)
                    ftype, fsize = panels.classify_opening(panel, bpts @ Tinv[:3, :3].T + Tinv[:3, 3])
                    ftype = {"NOTCH": "NOTCH", "OPENING": "OPENING"}.get(ftype, ftype)
                    where = "proper"
                elif len(apts):
                    # the mirror image lacks material here: the CAD's own opening, seen through H^-1
                    back = apts @ Hinv[:3, :3].T + Hinv[:3, 3]
                    _, nb = treeP.query(back)
                    ob = owner[nb]
                    okb = ob >= 0
                    pb = int(np.bincount(ob[okb]).argmax()) if okb.any() else -1
                    if pb >= 0:
                        Tb = TSinv[pb]
                        ftype, fsize = panels.classify_opening(pb, back @ Tb[:3, :3].T + Tb[:3, 3])
                    where = "mirror"
                if ftype == "NOTCH":
                    ftype = "NOTCH_OR_TAB"
            else:
                where = None
            u, v = (None, None)
            if panel >= 0:
                u, v = _plane_basis(stage_normals[panel])
            ext2 = None
            if u is not None:
                q = np.c_[pts @ u, pts @ v]
                ext2 = sorted([round(float(e), 1) for e in (q.max(0) - q.min(0))], reverse=True)
            centre = pts.mean(axis=0)
            cf = None
            if panel >= 0:
                Ti = TSinv[panel]
                cf = (Ti[:3, :3] @ centre + Ti[:3, 3]).round(2).tolist()
            hrec["features"].append({
                "feature": fi,
                "type": ftype,
                "opening_in": None if ftype in ("FORM", "UNCLASSIFIED") else (where if panel >= 0 else None),
                "size_mm": fsize or ext2,
                "extent_mm": ext2,
                "panel_id": panels.ids[panel] if panel >= 0 else None,
                "centre_stage_mm": centre.round(2).tolist(),
                "centre_final_mm": cf,
                "n_deviating": int(m.sum()),
                "max_deviation_mm": round(float(max(h["dA"][fs[fr == ROLE_A]].max(initial=0), h["dB"][fs[fr == ROLE_B]].max(initial=0))), 2),
                "probes": {"proper_material": n_core[ROLE_A], "mirror_material": n_core[ROLE_B], "ring": n_core[ROLE_RING]},
                "expected_evidence": {
                    "proper_material_probes": {"proper_image": "MATERIAL", "mirror_image": "NO_MATERIAL"},
                    "mirror_material_probes": {"proper_image": "NO_MATERIAL", "mirror_image": "MATERIAL"},
                },
                "verifiable": bool(max(n_core[ROLE_A], n_core[ROLE_B]) >= th["min_core_probes"]
                                   and n_core[ROLE_RING] >= th["min_core_probes"]),
            })
        hrec["verifiable"] = any(f["verifiable"] for f in hrec["features"])
        rec["hypotheses"].append(hrec)
    rec["verifiable"] = bool(rec["hypotheses"]) and all(h["verifiable"] for h in rec["hypotheses"])
    return rec, probes


def compute(cdir: Path, stages: list[str] | None = None, th: dict = THRESHOLDS) -> tuple[dict, dict[str, np.ndarray]]:
    cdir = Path(cdir)
    compiled = json.loads((cdir / "compiled_part.json").read_text())
    t0 = time.time()
    recs = []
    arrays: dict[str, list] = {k: [] for k in ("pos", "nrm", "owner", "side", "hyp", "feature", "role", "stage")}
    for si, st in enumerate(compiled["stages"]):
        if stages and st["stage_model_id"] not in stages:
            continue
        ts = time.time()
        rec, pr = stage_map(compiled, cdir, st, th)
        rec["seconds"] = round(time.time() - ts, 2)
        rec["stage_ordinal"] = len(recs)
        n = 0
        for k in ("pos", "nrm", "owner", "side", "hyp", "feature", "role"):
            if pr[k]:
                arrays[k].append(np.concatenate(pr[k]))
        if pr["pos"]:
            n = sum(len(a) for a in pr["pos"])
            arrays["stage"].append(np.full(n, len(recs)))
        rec["probe_count"] = n
        recs.append(rec)
    out = {k: (np.concatenate(v) if v else np.zeros((0, 3) if k in ("pos", "nrm") else 0)) for k, v in arrays.items()}
    out = {"pos": out["pos"].astype(np.float32), "nrm": out["nrm"].astype(np.float32),
           "owner": out["owner"].astype(np.int16), "side": out["side"].astype(np.int8),
           "hyp": out["hyp"].astype(np.int16), "feature": out["feature"].astype(np.int16),
           "role": out["role"].astype(np.uint8), "stage": out["stage"].astype(np.int16)}
    classes = [r["class"] for r in recs]
    final = recs[-1] if recs else None
    doc = {
        "schema": SCHEMA,
        "method": METHOD,
        "thresholds": th,
        "thresholds_sha256": hashlib.sha256(json.dumps(th, sort_keys=True).encode()).hexdigest(),
        "compiled_part_sha256": compiled["compiled_part_sha256"],
        "cad_sha256": compiled.get("cad_sha256"),
        "panel_ids": [p["panel_id"] for p in compiled["panels"]],
        "part_class": final["class"] if final else None,
        "any_stage_chiral": any(c != "ACHIRAL" for c in classes),
        "stages": recs,
        "seconds": round(time.time() - t0, 2),
        "probes_file": None,
        "probes_sha256": None,
    }
    return doc, out


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for k in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, b.getvalue())
    return buf.getvalue()


def write(cdir: Path, out_dir: Path | None = None, stages: list[str] | None = None) -> Path:
    cdir = Path(cdir)
    out_dir = Path(out_dir) if out_dir else cdir
    out_dir.mkdir(parents=True, exist_ok=True)
    doc, arrays = compute(cdir, stages)
    raw = _npz_bytes(arrays)
    (out_dir / f"{SCHEMA}.npz").write_bytes(raw)
    doc["probes_file"] = f"{SCHEMA}.npz"
    doc["probes_sha256"] = hashlib.sha256(raw).hexdigest()
    body = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    doc["map_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (out_dir / f"{SCHEMA}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return out_dir / f"{SCHEMA}.json"


def load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Map + probes, with the probe file's hash checked (raises ValueError when it does not match)."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError("not a chirality_map_v0")
    pz = path.parent / doc["probes_file"]
    if _sha_file(pz) != doc["probes_sha256"]:
        raise ValueError("chirality probe file hash mismatch")
    z = np.load(pz, allow_pickle=False)
    return doc, {k: z[k] for k in z.files}


def summary(doc: dict) -> dict:
    """Display summary for the setup screen / station badge (final stage + any chiral stage)."""
    final = doc["stages"][-1] if doc["stages"] else None
    feats = []
    if final and final["hypotheses"]:
        h0 = final["hypotheses"][0]
        for f in h0["features"]:
            feats.append({k: f[k] for k in ("feature", "type", "opening_in", "size_mm", "panel_id", "centre_final_mm",
                                            "max_deviation_mm", "verifiable")})
    return {"schema": "chirality_summary_v0", "part_class": doc["part_class"],
            "left_right_sensitive": doc["part_class"] in ("CHIRAL_GROSS", "CHIRAL_BY_FEATURES"),
            "stage_classes": {s["stage_model_id"]: s["class"] for s in doc["stages"]},
            "features": feats, "verifiable": bool(final and final.get("verifiable", True)),
            "map_sha256": doc.get("map_sha256"), "thresholds_sha256": doc["thresholds_sha256"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("compiled_dir")
    ap.add_argument("--out")
    ap.add_argument("--stages")
    a = ap.parse_args(argv)
    p = write(Path(a.compiled_dir), Path(a.out) if a.out else None, a.stages.split(",") if a.stages else None)
    doc = json.loads(p.read_text())
    print(json.dumps({"map": str(p), "part_class": doc["part_class"], "seconds": doc["seconds"],
                      "stages": {s["stage_model_id"]: [s["class"], len(s["hypotheses"]),
                                                       sum(len(h["features"]) for h in s["hypotheses"])]
                                 for s in doc["stages"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
