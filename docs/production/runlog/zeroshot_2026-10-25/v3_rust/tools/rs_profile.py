"""Per-stage timers of the zs_rust lane (run record timers_s) on given DEV / N1 scans at given thread counts.
Guard + custody per scan; outputs go to a temporary directory (timing only, nothing scored).
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/rs_profile.py OUT.json "1 16" sid [sid ...]"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.common.gold_guard import require_runnable
from research.zeroshot.z3_slicing.custody import check_path
from research.zeroshot.zs_lane_v3_rs import rs
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
tsv = dict(l.rstrip("\n").split("\t") for l in open(P / "integration/dev_scans.tsv"))
tsv.update(dict(l.rstrip("\n").split("\t") for l in open(P / "newscans_N1/n1_scans.tsv")))
out, ths, sids = Path(sys.argv[1]), [int(x) for x in sys.argv[2].split()], sys.argv[3:]
res = json.loads(out.read_text()) if out.is_file() else {}
cfg = REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json"
for sid in sids:
    require_runnable(sid, final_testb=False)
    path = str(check_path(tsv[sid]))
    for th in ths:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "job.json").write_text(json.dumps({"scan": path, "config": str(cfg), "repo": str(REPO),
                                                       "out": f"{d}/e.json", "record": f"{d}/r.json"}))
            t0 = time.time()
            r = subprocess.run([str(rs.binary()), "lane", "--job", f"{d}/job.json", "--out", f"{d}/o.json", "--threads", str(th)], capture_output=True, text=True)
            if r.returncode != 0:
                res[f"{sid}|t{th}"] = {"rc": r.returncode, "err": r.stderr[-300:]}; continue
            rec = json.loads(Path(d, "r.json").read_text())
            lines = json.loads(Path(d, "e.json").read_text())["owned_bend_regions"]
            res[f"{sid}|t{th}"] = {"wall_s": round(time.time() - t0, 3), "rss_gib": round(rec["peak_rss_gib"], 3),
                                   "n_lines": len(lines), "lines_json_sha": __import__("hashlib").sha256(json.dumps(lines).encode()).hexdigest()[:16],
                                   "timers_s": {k: v["s"] for k, v in rec["timers_s"].items()}}
        print(sid, th, json.dumps(res[f"{sid}|t{th}"])[:600], flush=True)
        out.write_text(json.dumps(res, indent=1))
