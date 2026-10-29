"""Timing summary of bench.sh outputs: per thread count P50 / P95 / max of the process wall (adapter + binary, as
lane_cli would see it) and of the binary's own lane wall, peak RSS of the binary, mean stage walls.
usage: .venv/bin/python -B docs/.../v3_rust/tools/timing_summary.py BENCH_ROOT [--out JSON]"""
import argparse, json, re
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", default=""); a = ap.parse_args()
res = {}
for td in sorted(Path(a.root).glob("t*"), key=lambda p: int(p.name[1:])):
    w, lw, rss, prss, st, fb = [], [], [], [], {}, []
    for d in sorted(td.iterdir()):
        f = d / "run_record.json"
        if not f.is_file(): continue
        r = json.loads(f.read_text())
        if r.get("rust_fallback"):
            fb.append(d.name); continue
        w.append(r["wall_s"]); lw.append(r["lane_wall_s"]); rss.append(r["peak_rss_gib"])
        tt = (d / "time.txt").read_text() if (d / "time.txt").is_file() else ""
        m = re.search(r"(\d+)\s+maximum resident set size", tt)
        if m: prss.append(int(m.group(1)) / 2**30)
        for k, v in r["stages"].items():
            if isinstance(v, dict) and "wall" in v: st.setdefault(k, []).append(v["wall"])
    q = lambda x, p: round(float(np.percentile(x, p)), 3)
    res[td.name] = {"n": len(w), "wall_p50": q(w, 50), "wall_p95": q(w, 95), "wall_max": round(max(w), 3), "wall_sum": round(sum(w), 1),
                    "lane_p50": q(lw, 50), "lane_p95": q(lw, 95), "lane_max": round(max(lw), 3), "lane_sum": round(sum(lw), 1),
                    "rust_rss_max_gib": round(max(rss), 3), "rust_rss_p50_gib": q(rss, 50),
                    "adapter_rss_max_gib": round(max(prss), 3) if prss else None,
                    "stage_mean_s": {k: round(float(np.mean(v)), 3) for k, v in st.items()},
                    "stage_sum_s": {k: round(float(np.sum(v)), 2) for k, v in st.items()},
                    "python_fallback_scans": fb}
print(json.dumps(res, indent=1))
if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
