"""Dense-mesh timing (engineering only; meshes from /private/tmp/claude-501/dense_ready/meshes, made by the
dense-readiness agent; nothing scored): the zs_rust lane with the frozen revision-2 face budget and, as an
experimental flag only, with the budget off, at several thread counts. Records wall, the binary's peak RSS, stage
timers and a hash of the output lines (budget-on runs must hash equal across thread counts). The Python
revision-2 baseline is read from the dense agent's run records (runs/<mesh>_zs_t1/record.json) when present.
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/dense_run.py OUT.json "rev2:1,4,8,16 rev3:1,4,8,16 off:1,16" mesh.ply [...]"""
import hashlib, json, subprocess, sys, tempfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v3_rs import rs
out, spec, meshes = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
modes = [(m.split(":")[0], [int(t) for t in m.split(":")[1].split(",")]) for m in spec.split()]
cfg = REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json"
cfg3 = REPO / "research/zeroshot/zs_lane_v3_rs/config_v3r3_rs.json"   # mode "rev3": revision 3's budget
res = json.loads(out.read_text()) if out.is_file() else {}
DR = Path("/private/tmp/claude-501/dense_ready")
for mesh in meshes:
    name = Path(mesh).stem
    pyrec = DR / "runs" / f"{name}_zs_t1" / "record.json"
    if pyrec.is_file() and f"{name}|python_rev2" not in res:
        r = json.loads(pyrec.read_text())
        res[f"{name}|python_rev2"] = {"wall_s": r.get("wall_s"), "peak_rss_gib": round((r.get("peak_rss_bytes") or 0) / 2**30, 3),
                                      "n_lines": r.get("n_lines"), "status": r.get("status"), "budget": (r.get("stages") or {}).get("budget")}
    for mode, ths in modes:
        for th in ths:
            key = f"{name}|{mode}|t{th}"
            if key in res:
                continue
            with tempfile.TemporaryDirectory(prefix="zs_dense_") as d:
                cf = cfg3 if mode == "rev3" else cfg
                job = {"scan": mesh, "config": str(cf), "repo": str(REPO), "out": f"{d}/e.json", "record": f"{d}/r.json"}
                if mode == "off":
                    job["budget_override"] = {"max_faces": None}
                Path(d, "job.json").write_text(json.dumps(job))
                t0 = time.time()
                p = subprocess.run([str(rs.binary()), "lane", "--job", f"{d}/job.json", "--out", f"{d}/o.json", "--threads", str(th)],
                                   capture_output=True, text=True)
                wall = time.time() - t0
                if p.returncode != 0:
                    res[key] = {"rc": p.returncode, "err": p.stderr[-300:], "wall_s": round(wall, 2)}
                else:
                    rec = json.loads(Path(d, "r.json").read_text())
                    lines = json.loads(Path(d, "e.json").read_text())["owned_bend_regions"]
                    st = rec["stages"]
                    res[key] = {"wall_s": round(wall, 2), "peak_rss_gib": round(rec["peak_rss_gib"], 3), "n_lines": len(lines),
                                "lines_sha": hashlib.sha256(json.dumps(lines).encode()).hexdigest()[:16],
                                "budget": st.get("budget"),
                                "stage_wall": {k: v.get("wall") for k, v in st.items() if isinstance(v, dict) and "wall" in v},
                                "timers_s": {k: v["s"] for k, v in rec["timers_s"].items()}}
            print(key, json.dumps({k: v for k, v in res[key].items() if k in ("wall_s", "peak_rss_gib", "n_lines", "lines_sha", "rc", "err")}), flush=True)
            out.write_text(json.dumps(res, indent=1))
