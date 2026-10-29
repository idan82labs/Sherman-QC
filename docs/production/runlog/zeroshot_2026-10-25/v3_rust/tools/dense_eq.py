"""Dense meshes: the frozen Python lane (zs_lane_v3 core.process + select_lines, one process per mesh, wall and peak
RSS recorded) against the Rust binary (16 threads), eval lines compared field by field. Engineering only.
usage: [ZS_REV=3] sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/dense_eq.py OUT.json mesh.ply [...]
ZS_REV=2 (default): revision 2 (the hash-checked copies, rev2ref) against config_v3_rs.json.
ZS_REV=3: revision 3 (the live zs_lane_v3, every file checked against config_v3.json's fingerprint dc23a905...)
against config_v3r3_rs.json."""
import hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v3_rs import rs
REV = int(os.environ.get("ZS_REV", "2"))
RS = "research/zeroshot/zs_lane_v3_rs/"
cfg = REPO / (RS + ("config_v3_rs.json" if REV == 2 else "config_v3r3_rs.json"))
PYCFG = REPO / (RS + "rev2_snapshot/config_v3.json" if REV == 2 else "research/zeroshot/zs_lane_v3/config_v3.json")
if REV == 3:
    c3 = json.loads(PYCFG.read_text())
    assert c3["revision"] == 3 and c3["code_fingerprint"]["sha256"].startswith("dc23a905")
    for f_, h_ in c3["code_fingerprint"]["files"].items():
        assert hashlib.sha256((REPO / f_).read_bytes()).hexdigest() == h_, f_
PY = r'''
import sys, json, pickle, time, resource
REPO, REV, PYCFG = sys.argv[3], int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, REPO)
if REV == 2:
    from research.zeroshot.zs_lane_v3_rs import rev2ref; rev2ref.install()
from research.zeroshot.zs_lane_v3 import core
cfg = json.load(open(PYCFG)); core.configure(cfg, "zs"); v = cfg["variants"]["zs"]
zs = pickle.load(open(REPO + "/" + v["model"], "rb")); z3 = pickle.load(open(REPO + "/" + v["z3_model"], "rb"))
t0 = time.time(); rec = core.process(sys.argv[1]); lines = core.select_lines(rec, v, zs, z3)
pay = core.payload(sys.argv[1], None, lines, status="OK", conversion={})
json.dump({"lines": pay["owned_bend_regions"], "wall_s": time.time() - t0,
           "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
           "budget": rec["stages"].get("budget")}, open(sys.argv[2], "w"), default=str)
'''
out = Path(sys.argv[1])
res = json.loads(out.read_text()) if out.is_file() else {}
for mesh in sys.argv[2:]:
    name = Path(mesh).stem
    if name in res:
        continue
    with tempfile.TemporaryDirectory() as d:
        Path(d, "py.py").write_text(PY)
        t0 = time.time()
        subprocess.run([str(REPO / ".venv/bin/python"), "-B", f"{d}/py.py", mesh, f"{d}/py.json", str(REPO), str(REV), str(PYCFG)],
                       check=True, capture_output=True, env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
        pw = time.time() - t0
        py = json.loads(Path(d, "py.json").read_text())
        Path(d, "job.json").write_text(json.dumps({"scan": mesh, "config": str(cfg), "repo": str(REPO), "out": f"{d}/e.json",
                                                   "record": f"{d}/r.json"}))
        subprocess.run([str(rs.binary()), "lane", "--job", f"{d}/job.json", "--out", f"{d}/o.json", "--threads", "16"], check=True, capture_output=True)
        rl = json.loads(Path(d, "e.json").read_text())["owned_bend_regions"]
        rb = json.loads(Path(d, "r.json").read_text())["stages"].get("budget")
    same = json.dumps(py["lines"]) == json.dumps(rl)
    pb = {k: v for k, v in (py["budget"] or {}).items() if k != "wall"}
    rbb = {k: v for k, v in (rb or {}).items() if k != "wall"}
    res[name] = {"identical": same, "budget_info_identical": json.dumps(pb) == json.dumps(rbb), "n_lines_py": len(py["lines"]),
                 "n_lines_rs": len(rl), "python_wall_s": round(py["wall_s"], 2), "python_process_s": round(pw, 2),
                 "python_rss_gib": round(py["peak_rss_gib"], 3), "budget": py["budget"], "budget_rs": rb}
    print(name, json.dumps(res[name])[:300], flush=True)
    out.write_text(json.dumps(res, indent=1))
print("identical", sum(v["identical"] for v in res.values()), "of", len(res),
      "budget info identical", sum(v["budget_info_identical"] for v in res.values()))
