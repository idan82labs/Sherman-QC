"""Option 4 offline check: are zs->CAD seeds within E3's capture range of E3's own final pose (VALID scans)?
Also writes per-scan seeds (proper scan and mirrored scan) for the seeded E3 runs.
usage: seed_eval.py COHORT.json OUT_DIR"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import seeds as SD
sys.path.insert(0, str(Path(__file__).resolve().parent))
from options_offline import sel

FLIP = np.diag([-1.0, 1.0, 1.0])


def main(cohort, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(cohort).read_text())["rows"]
    summ = []
    for r in rows:
        z = [[l["start"], l["end"]] for l in sel(r, cap=True)]
        st = {k: [[l["start"], l["end"]] for l in v] for k, v in r["cad_stages"].items() if v}
        t0 = time.time()
        sp = SD.seeds(z, st)
        zm = [[(np.asarray(a) @ FLIP).tolist(), (np.asarray(b) @ FLIP).tolist()] for a, b in z]
        sm = SD.seeds(zm, st)
        dt = time.time() - t0
        rec = {"scan": r["scan"], "e3": r["e3"].get("state"), "n_zs": len(z), "n_bends": r["n_bends"], "seed_s": round(dt, 2),
               "n_seeds": len(sp), "n_seeds_mirrored": len(sm)}
        T = r["e3"].get("T")
        if T is not None and sp:
            T = np.asarray(T, float)
            probe = np.asarray([np.mean(l, axis=0) for l in z]) if z else None
            errs = [SD.same_pose(s["T"], T, ang=2.0, dist=2.0, probe=probe) for s in sp]
            rec["top1"] = {"ok": errs[0][0], "deg": round(errs[0][1], 2), "mm": round(errs[0][2], 2), "inl": sp[0]["inliers"]}
            ok = [k for k, e in enumerate(errs) if e[0]]
            rec["rank_within_2deg_2mm"] = ok[0] + 1 if ok else None
            rec["best_err"] = min(((round(e[1], 2), round(e[2], 2)) for e in errs), key=lambda x: x[0] + x[1])
        summ.append(rec)
        (out / f"{r['scan']}.seeds.json").write_text(json.dumps(
            {"proper": [s["T"].tolist() for s in sp], "mirrored": [s["T"].tolist() for s in sm],
             "meta": [{k: v for k, v in s.items() if k != "T"} for s in sp]}, default=str))
        print(json.dumps(rec), flush=True)
    v = [s for s in summ if s["e3"] == "VALID"]
    agg = {"valid_scans": len(v), "with_seeds": sum(1 for s in v if s["n_seeds"]),
           "top1_within_2deg_2mm": sum(1 for s in v if s.get("top1", {}).get("ok")),
           "any_within_2deg_2mm": sum(1 for s in v if s.get("rank_within_2deg_2mm")),
           "seed_time_median_s": float(np.median([s["seed_s"] for s in summ])),
           "seed_time_max_s": float(np.max([s["seed_s"] for s in summ]))}
    print(json.dumps(agg))
    (out / "SEED_EVAL.json").write_text(json.dumps({"aggregate": agg, "per_scan": summ}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
