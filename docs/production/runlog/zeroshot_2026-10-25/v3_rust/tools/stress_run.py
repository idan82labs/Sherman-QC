"""Density / stress timing (engineering only, no scoring): the zs_rust lane on a derived stress mesh, with the frozen
revision-2 face budget and with the budget raised or off, at several thread counts. Custody: the DEV source scan id
is guarded. Prints wall, the binary's peak RSS, the budget record and the stage times.
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/stress_run.py SOURCE_SID MESH OUT.json"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.common import gold_guard
from research.zeroshot.zs_lane_v3_rs import rs
sid, mesh, out = sys.argv[1:4]
gold_guard.require_runnable(sid, final_testb=False)
cfg = REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json"
res = json.loads(Path(out).read_text()) if Path(out).is_file() else {}
for budget in ("rev2_3M", "off"):
    for th in (1, 4, 8, 16):
        key = f"{budget}|t{th}"
        if key in res:
            continue
        with tempfile.TemporaryDirectory(prefix="zs_stress_") as d:
            job = {"scan": mesh, "config": str(cfg), "repo": str(REPO), "out": f"{d}/eval_lines.json",
                   "record": f"{d}/rec.json"}
            if budget == "off":
                job["budget_override"] = {"max_faces": None}
            Path(d, "job.json").write_text(json.dumps(job))
            t0 = time.time()
            r = subprocess.run([str(rs.binary()), "lane", "--job", f"{d}/job.json", "--out", f"{d}/o.json", "--threads", str(th)],
                               capture_output=True, text=True)
            wall = time.time() - t0
            if r.returncode != 0:
                res[key] = {"error": r.stderr[-400:]}
            else:
                rec = json.loads(Path(d, "rec.json").read_text())
                lines = json.loads(Path(d, "eval_lines.json").read_text())["owned_bend_regions"]
                st = rec["stages"]
                res[key] = {"wall_s": round(wall, 2), "peak_rss_gib": round(rec["peak_rss_gib"], 3), "n_lines": len(lines),
                            "lines_sha": __import__("hashlib").sha256(json.dumps(lines).encode()).hexdigest()[:16],
                            "budget": st.get("budget"), "stage_wall": {k: v.get("wall") for k, v in st.items() if isinstance(v, dict) and "wall" in v}}
        print(key, json.dumps(res[key])[:300], flush=True)
        Path(out).write_text(json.dumps(res, indent=1))
