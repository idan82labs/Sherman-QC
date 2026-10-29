"""Run a zs_rust(_det) binary's `lane` over the DEV scan list with probe environment variables (ZS_HOST_MATH,
ZS_ACCEL, ZS_PROBE_OUT per scan), one process per scan, serially, nice 15. Gold guard + custody check first.
Writes <out>/<sid>/{eval_lines.json, run_record.json[, dump.json, probe.json]} and <out>/runs.json.
usage: .venv/bin/python -B .../determinism/tools/det_run.py --bin PATH --out DIR [--env K=V ...] [--threads N]
       [--dump] [--probe] [--arch x86_64] [--config CFG] [--list TSV] [--ids a,b]"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO))
ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--env", action="append", default=[]); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--dump", action="store_true"); ap.add_argument("--probe", action="store_true")
ap.add_argument("--arch", default="")
ap.add_argument("--config", default=str(REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json"))
ap.add_argument("--list", default=str(REPO / "docs/production/runlog/zeroshot_2026-10-25/integration/dev_scans.tsv"))
ap.add_argument("--ids", default="")
a = ap.parse_args()
scans = [l.rstrip("\n").split("\t")[:2] for l in Path(a.list).read_text().splitlines() if l.strip()]
if a.ids:
    want = set(a.ids.split(",")); scans = [s for s in scans if s[0] in want]
from research.zeroshot.common import gold_guard
from research.zeroshot.z3_slicing.custody import check_path
for sid, p in scans:
    gold_guard.require_runnable(p, final_testb=False); check_path(p)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
for e in a.env:
    k, v = e.split("=", 1); env[k] = v
runs = {}
for sid, path in scans:
    d = out / sid; d.mkdir(exist_ok=True)
    job = {"scan": path, "config": str(Path(a.config).resolve()), "repo": str(REPO), "out": str(d / "eval_lines.json"),
           "record": str(d / "run_record.json"), "lane": "zs_lane_v3_rs:zs"}
    if a.dump:
        job["dump"] = str(d / "dump.json"); job["splits"] = True
    e2 = dict(env)
    if a.probe:
        e2["ZS_PROBE_OUT"] = str(d / "probe.json")
    with tempfile.TemporaryDirectory(prefix="zs_det_") as td:
        Path(td, "job.json").write_text(json.dumps(job))
        cmd = ["nice", "-n", "15"] + (["/usr/bin/arch", "-" + a.arch] if a.arch else []) + [a.bin, "lane", "--job", f"{td}/job.json", "--out", f"{td}/o.json", "--threads", str(a.threads)]
        t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, env=e2); wall = time.time() - t0
    rec = {"rc": r.returncode, "wall_s": round(wall, 4)}
    if r.returncode == 0:
        rr = json.loads((d / "run_record.json").read_text()); rec.update(peak_rss_gib=rr.get("peak_rss_gib"), n_lines=rr.get("n_lines"))
    elif r.returncode != 10:
        rec["stderr"] = r.stderr[-400:]
    runs[sid] = rec
(out / "runs.json").write_text(json.dumps(runs, indent=1))
bad = {k: v for k, v in runs.items() if v["rc"] not in (0, 10)}
print(json.dumps({"scans": len(runs), "rc0": sum(v["rc"] == 0 for v in runs.values()), "rc10": sum(v["rc"] == 10 for v in runs.values()), "bad": bad}))
