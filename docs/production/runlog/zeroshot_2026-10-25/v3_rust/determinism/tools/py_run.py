"""Run the Python zs_lane_v3 (live package, config_v3.json = revision 3) over DEV, one process per scan, serially,
nice 15, OMP_NUM_THREADS=1; either unpatched (--mode native) or under the deterministic-math research patch
(--mode det: research.zeroshot.zs_lane_v3_det.run). The lane itself runs the gold guard and custody check.
Writes <out>/<sid>/{eval_lines.json, run_record.json, stderr.log} and <out>/runs.json (rc, wall, peak RSS).
usage: .venv/bin/python -B .../determinism/tools/py_run.py --mode det|native --out DIR [--ids a,b]"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[7]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["det", "native"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ids", default="")
ap.add_argument("--config", default="research/zeroshot/zs_lane_v3/config_v3.json")
ap.add_argument("--list", default=str(REPO / "docs/production/runlog/zeroshot_2026-10-25/integration/dev_scans.tsv"))
a = ap.parse_args()
scans = [l.rstrip("\n").split("\t")[:2] for l in Path(a.list).read_text().splitlines() if l.strip()]
if a.ids:
    want = set(a.ids.split(",")); scans = [s for s in scans if s[0] in want]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rp = out / "runs.json"
runs = json.loads(rp.read_text()) if rp.is_file() else {}
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
mod = "research.zeroshot.zs_lane_v3_det.run" if a.mode == "det" else "research.zeroshot.zs_lane_v3.lane"
for sid, path in scans:
    if sid in runs and runs[sid]["rc"] == 0:
        continue
    d = out / sid; d.mkdir(exist_ok=True)
    cmd = ["nice", "-n", "15", "/usr/bin/time", "-l", str(REPO / ".venv/bin/python"), "-B", "-m", mod, "--scan", path,
           "--out", str(d / "eval_lines.json"), "--config", a.config, "--variant", "zs", "--record", str(d / "run_record.json")]
    t0 = time.time(); r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True); wall = time.time() - t0
    (d / "stderr.log").write_text(r.stderr[-20000:])
    rss = next((int(l.split()[0]) for l in r.stderr.splitlines() if "maximum resident set size" in l), None)
    runs[sid] = {"rc": r.returncode, "wall_s": round(wall, 4), "peak_rss_gib": round(rss / 2**30, 3) if rss else None}
    print(sid, runs[sid], flush=True)
    rp.write_text(json.dumps(runs, indent=1))
