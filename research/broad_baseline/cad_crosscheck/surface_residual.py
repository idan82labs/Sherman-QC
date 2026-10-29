"""For each matched scan: median / p90 point-to-CAD-surface distance after the
stored registration (3000 scan points). Near-zero residual on a 'scan' means
the mesh is CAD-derived, not a physical capture."""
import json, sys
from pathlib import Path
import numpy as np, trimesh
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.broad_baseline.cad_crosscheck.render_cad import tessellate  # noqa
O = ROOT / "docs/production/runlog/broad_baseline"
B = ROOT / "docs/production/runlog/benchmark_v2"
cache = {}
res = {}
summ = json.loads((O / "cad_crosscheck_match/_run_summary.json").read_text())
for r in summ:
    if r.get("rc") != 0: continue
    M = json.loads((O / "cad_crosscheck_match" / f"{r['scan_id']}.json").read_text())
    raw = json.loads((O / "cad_crosscheck_raw" / f"{r['part_id']}.json").read_text())
    if r["part_id"] not in cache:
        T = tessellate(raw["step_path"]); V = T.reshape(-1, 3); F = np.arange(len(V)).reshape(-1, 3)
        cache[r["part_id"]] = trimesh.Trimesh(V, F, process=True)
    mesh = cache[r["part_id"]]
    P = np.load(Path(r["prep"]) / "points_ds.npz")["points"]
    P = P[np.random.default_rng(0).choice(len(P), min(3000, len(P)), replace=False)]
    Rm, tm = np.array(M["R"]), np.array(M["t"])
    Q = (P - tm) @ Rm  # scan -> CAD frame
    _, d, _ = trimesh.proximity.closest_point(mesh, Q)
    res[r["scan_id"]] = {"median_mm": round(float(np.median(d)), 4), "p90_mm": round(float(np.percentile(d, 90)), 4),
                         "frac_lt_0p02mm": round(float((d < 0.02).mean()), 3)}
    print(r["scan_id"], r["part_id"], res[r["scan_id"]], flush=True)
(O / "cad_crosscheck_match" / "_surface_residual.json").write_text(json.dumps(res, indent=1))
