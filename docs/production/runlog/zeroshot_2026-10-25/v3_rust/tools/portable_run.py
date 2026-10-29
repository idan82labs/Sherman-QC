"""Run a zs_rust binary's `lane` command over a scan list (DEV or dense meshes), one process per scan and thread
count, thread counts interleaved per scan, and (optionally) several binaries interleaved too, so timings share the
same host conditions. Writes <out>/<label>/t<N>/<sid>/{eval_lines.json, run_record.json[, dump.json]} and
<out>/<label>/runs.json (rc, process wall, peak RSS per run). Engineering only: reads no gold, scores nothing.

Every scan passes the gold guard and the custody check before anything runs (as the adapter does); a scan the
guard refuses stops the run. Scans without faces exit 10 (point cloud) and are recorded as such.

usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/portable_run.py --out ROOT
         --bin label=path [--bin label2=path2 ...] [--wrap label=arch_x86_64] --config CFG.json
         --list scans.tsv (sid<TAB>path) --threads "1 16" [--dump] [--ids a,b] [--guard/--no-guard]
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--bin", action="append", required=True, help="label=path")
ap.add_argument("--wrap", action="append", default=[], help="label=arch_x86_64 (run that binary under Rosetta)")
ap.add_argument("--config", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--threads", default="16")
ap.add_argument("--dump", action="store_true")
ap.add_argument("--ids", default="")
ap.add_argument("--no-guard", action="store_true", help="synthetic / derived meshes outside the custody registry")
a = ap.parse_args()

bins = dict(b.split("=", 1) for b in a.bin)
wraps = dict(w.split("=", 1) for w in a.wrap)
ths = [int(t) for t in a.threads.split()]
scans = [l.rstrip("\n").split("\t")[:2] for l in Path(a.list).read_text().splitlines() if l.strip()]
if a.ids:
    want = set(a.ids.split(","))
    scans = [s for s in scans if s[0] in want]
if not a.no_guard:
    from research.zeroshot.common import gold_guard
    from research.zeroshot.z3_slicing.custody import check_path
    for sid, p in scans:
        gold_guard.require_runnable(p, final_testb=False)
        check_path(p)
out = Path(a.out)
cfg = str(Path(a.config).resolve())
runs = {lab: (json.loads((out / lab / "runs.json").read_text()) if (out / lab / "runs.json").is_file() else {}) for lab in bins}
for sid, path in scans:
    for th in ths:
        for lab, b in bins.items():
            key = f"{sid}/t{th}"
            if key in runs[lab] and runs[lab][key]["rc"] in (0, 10):
                continue
            d = out / lab / f"t{th}" / sid
            d.mkdir(parents=True, exist_ok=True)
            job = {"scan": path, "config": cfg, "repo": str(REPO), "out": str(d / "eval_lines.json"),
                   "record": str(d / "run_record.json"), "lane": "zs_lane_v3_rs:zs"}
            if a.dump:  # as the adapter's dump mode: stage dump with the split-run evidence (lines unchanged)
                job["dump"] = str(d / "dump.json")
                job["splits"] = True
            with tempfile.TemporaryDirectory(prefix="zs_port_") as td:
                Path(td, "job.json").write_text(json.dumps(job))
                cmd = [b, "lane", "--job", f"{td}/job.json", "--out", f"{td}/o.json", "--threads", str(th)]
                if wraps.get(lab) == "arch_x86_64":
                    cmd = ["/usr/bin/arch", "-x86_64"] + cmd
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True)
                wall = time.time() - t0
            rec = {"rc": r.returncode, "wall_s": round(wall, 4)}
            if r.returncode == 0 and (d / "run_record.json").is_file():
                rr = json.loads((d / "run_record.json").read_text())
                rec.update(peak_rss_gib=rr.get("peak_rss_gib"), lane_wall_s=rr.get("lane_wall_s"), n_lines=rr.get("n_lines"))
            elif r.returncode != 10:
                rec["stderr"] = r.stderr[-500:]
            runs[lab][key] = rec
            print(lab, key, rec.get("rc"), rec.get("wall_s"), rec.get("n_lines"), flush=True)
            (out / lab / "runs.json").write_text(json.dumps(runs[lab], indent=1))
