"""Synthetic scans of a compiled stage for the chirality evaluation (research).

usage: python -m research.chirality.synth_scans --compiled DIR --stage S12 --map MAP.json --seed N --out DIR
           [--variants G,MX,MY,MZ,T,O,OX]

The surface is E1's own stage display mesh (``stages/S<k>.ply``: the OCC tessellation of the CAD at that stage),
posed at a random pose in the raw scanner frame and scanned by E2's virtual ZEISS cell
(``domains.production.engine.synthetic``: ray casting with self-occlusion, triangulation shadowing, grazing and
specular dropout, edge erosion/flare, range noise, per-view jitter, multi-setup merge with duplicate skins,
outliers), with the amendment-4b scanner model card and the native-density calibration E2 uses. The part is the
nominal CAD: a genuine scan of it is in tolerance, so any mirrored scan the engine accepts would carry PASSes.

Variants (each its own PLY, points only):

- ``G``  genuine, FULL plan (two setups);
- ``MX`` / ``MY`` / ``MZ`` the genuine scan mirrored in one raw axis (a mirrored part scanned the same way);
- ``T``  genuine, TOP_ONLY plan (one setup: natural partial coverage);
- ``O``  genuine with every small distinguishing feature of the stage occluded (extent <= ``SMALL_FEATURE_MM``):
  scan points within ``OCCLUSION_MM`` of any probe (feature core and its ring) of those features are removed (an
  occluder hides the feature and its surroundings: not scanned, not absent);
- ``OX`` ``O`` mirrored in x.

The map is used only to place the occluder; the scan itself comes from the CAD mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OCCLUSION_MM = 8.0
SMALL_FEATURE_MM = 60.0
MODEL_CARD = Path(__file__).resolve().parents[2] / "docs/production/runlog/scanner_model_card.json"


def _surface(mesh_path: Path):
    from domains.production.engine.synthetic.brep import TessellatedSurface
    from domains.production.engine.synthetic.ply import read_ply

    v, f, _ = read_ply(mesh_path)
    n = len(f)
    return TessellatedSurface(v.astype(np.float64), f.astype(np.int64), np.zeros(n, np.uint8), np.zeros(n, np.int16),
                              np.zeros(n, np.uint8))


def _scan(surface_raw, plan: str, seed: int, model):
    from domains.production.engine.synthetic.cases import _scan as e2_scan

    area = float(surface_raw.triangle_areas.sum())
    res, density = e2_scan(plan, "points", None, surface_raw, model, seed, area, None)
    return res, density


def _write(path: Path, pts: np.ndarray, note: str) -> str:
    from domains.production.engine.synthetic.ply import write_ply

    rng = np.random.default_rng(int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16))
    pts = pts[rng.permutation(len(pts))]
    return write_ply(path, pts, None, (note,))


def generate(cdir: Path, stage_id: str, map_path: Path | None, seed: int, out: Path, variants: list[str]) -> dict:
    from domains.production.engine.synthetic.scanner import ScannerModel, random_pose

    cdir, out = Path(cdir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    compiled = json.loads((cdir / "compiled_part.json").read_text())
    stage = next(s for s in compiled["stages"] if s["stage_model_id"] == stage_id)
    model = ScannerModel.from_json(MODEL_CARD)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    pose = random_pose(rng)
    surf = _surface(cdir / stage["display_mesh"]["rel_path"]).transformed(pose)
    meta = {"compiled_dir": str(cdir), "stage": stage_id, "seed": seed, "pose": pose.round(9).tolist(),
            "scanner_model_sha256": model.sha256(), "occlusion_mm": OCCLUSION_MM, "files": {}}
    want = set(variants)
    full = None
    if want & {"G", "MX", "MY", "MZ", "O", "OX"}:
        full, dens = _scan(surf, "FULL", seed, model)
        meta["density_full"] = {k: v for k, v in dens.items() if not isinstance(v, (list, dict))} if isinstance(dens, dict) else None
        meta["points_full"] = int(len(full.points))
    if "G" in want:
        meta["files"]["G"] = _write(out / "G.ply", full.points, "chirality synthetic: genuine FULL")
    for ax, name in ((0, "MX"), (1, "MY"), (2, "MZ")):
        if name in want:
            p = full.points.copy()
            p[:, ax] = -p[:, ax]
            meta["files"][name] = _write(out / f"{name}.ply", p, f"chirality synthetic: genuine FULL mirrored {name[1].lower()}")
    if "T" in want:
        top, _ = _scan(surf, "TOP_ONLY", seed + 1, model)
        meta["points_top"] = int(len(top.points))
        meta["files"]["T"] = _write(out / "T.ply", top.points, "chirality synthetic: genuine TOP_ONLY")
    if want & {"O", "OX"}:
        from scipy.spatial import cKDTree

        from research.chirality.chirality_map import load as load_map

        doc, pr = load_map(map_path)
        ordinal = next(i for i, s in enumerate(doc["stages"]) if s["stage_model_id"] == stage_id)
        # only the small distinguishing features are hidden (extent <= SMALL_FEATURE_MM): a gross difference is a
        # whole wall or flange, which an occluder does not hide
        small = {(h["hypothesis"], f["feature"]) for h in doc["stages"][ordinal]["hypotheses"] for f in h["features"]
                 if max(f["extent_mm"] or [0.0]) <= SMALL_FEATURE_MM}
        sel = (pr["stage"] == ordinal) & np.array([(int(a), int(b)) in small for a, b in zip(pr["hyp"], pr["feature"])],
                                                  dtype=bool)
        pos, own = pr["pos"][sel].astype(np.float64), pr["owner"][sel]
        # probes (final frame of their panel) -> stage frame -> raw frame
        pids = [p["panel_id"] for p in compiled["panels"]]
        st = np.empty_like(pos)
        for i in np.unique(own):
            T = np.asarray(stage["panel_transforms"][pids[int(i)]])
            m = own == i
            st[m] = pos[m] @ T[:3, :3].T + T[:3, 3]
        raw_probe = st @ pose[:3, :3].T + pose[:3, 3]
        ideal = full.ideal_raw
        okp = np.isfinite(ideal).all(axis=1)
        drop = np.zeros(len(full.points), bool)
        if len(raw_probe):
            d, _ = cKDTree(raw_probe).query(np.where(okp[:, None], ideal, full.points), distance_upper_bound=OCCLUSION_MM)
            drop = np.isfinite(d)
        meta["occluded_points"] = int(drop.sum())
        meta["occluded_probe_count"] = int(len(raw_probe))
        po = full.points[~drop]
        if "O" in want:
            meta["files"]["O"] = _write(out / "O.ply", po, "chirality synthetic: genuine FULL, features occluded")
        if "OX" in want:
            p = po.copy()
            p[:, 0] = -p[:, 0]
            meta["files"]["OX"] = _write(out / "OX.ply", p, "chirality synthetic: features occluded, mirrored x")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--map")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="G,MX,MY,MZ,T,O,OX")
    a = ap.parse_args(argv)
    meta = generate(Path(a.compiled), a.stage, Path(a.map) if a.map else None, a.seed, Path(a.out), a.variants.split(","))
    print(json.dumps({k: meta.get(k) for k in ("stage", "points_full", "points_top", "occluded_points")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
