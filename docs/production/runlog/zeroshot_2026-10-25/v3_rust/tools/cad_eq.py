"""CAD-prior path (production lane_cli.run_lane_v3): Python zs_lane_v3 core.process(thickness={"t_mm": t,
"source": "CAD"}, max_families=K) + select_lines + the n_bends cap (sorted by -score, stable) against the binary with
job "cad" {t_mm, max_families} and the same cap. The facts are synthetic (a fixed cycle of t / K / n per scan): this
checks the mechanism, not a part. Revision 2 (rev2ref copies) or ZS_REV=3 (frozen live revision 3). DEV scans by sid
(guard + custody); eval lines only, nothing scored.
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/cad_eq.py OUT.json sid [...]"""
import json, os, pickle, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
REV = int(os.environ.get("ZS_REV", "2"))
RS = REPO / "research/zeroshot/zs_lane_v3_rs"
if REV == 2:
    from research.zeroshot.zs_lane_v3_rs import rev2ref
    rev2ref.install()
    pycfg_p, rscfg = rev2ref.REV2_CONFIG, RS / "config_v3_rs.json"
else:
    pycfg_p, rscfg = REPO / "research/zeroshot/zs_lane_v3/config_v3.json", RS / "config_v3r3_rs.json"
from research.zeroshot.zs_lane_v3 import core
from research.zeroshot.zs_lane_v3_rs import core as RC
from research.zeroshot.common.gold_guard import require_runnable
from research.zeroshot.z3_slicing.custody import check_path
cfg = json.loads(Path(pycfg_p).read_text())
core.configure(cfg, "zs")
v = cfg["variants"]["zs"]
zs = pickle.load(open(REPO / v["model"], "rb"))
z3 = pickle.load(open(REPO / v["z3_model"], "rb"))
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
tsv = dict(l.rstrip("\n").split("\t") for l in open(P / "integration/dev_scans.tsv"))
T = [1.0, 1.5, 2.0, 3.0, 5.0]
K = [2, 3, 4, 6]
N = [2, 3, 5, 8]
out = Path(sys.argv[1])
res = json.loads(out.read_text()) if out.is_file() else {}
for i, sid in enumerate(sys.argv[2:]):
    if sid in res:
        continue
    require_runnable(sid, final_testb=False)
    path = str(check_path(tsv[sid]))
    facts = {"t_mm": T[i % len(T)], "max_families": K[i % len(K)], "n_bends": N[i % len(N)]}
    t0 = time.time()
    rec = core.process(path, thickness={"t_mm": facts["t_mm"], "source": "CAD"}, max_families=facts["max_families"])
    if rec.get("status") != "OK":
        res[sid] = {"python_status": rec.get("status")}
        continue
    sel = sorted(core.select_lines(rec, v, zs, z3), key=lambda c: -float(c["score"]))[: facts["n_bends"]]
    py = [{"s": [float(x) for x in c["start_s"]], "e": [float(x) for x in c["end_s"]], "score": round(float(c["score"]), 6),
           "op": str(c.get("op") or "keep"), "fix": c.get("fix")} for c in sel]
    pyw = time.time() - t0
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        RC.run_binary(path, str(rscfg), out=f"{d}/e.json", cad=facts)
        rl = json.loads(Path(d, "e.json").read_text())["owned_bend_regions"]
    rl = sorted(rl, key=lambda c: -float(c["score"]))[: facts["n_bends"]]
    rr = [{"s": c["line_start"], "e": c["line_end"], "score": round(float(c["score"]), 6), "op": c.get("op"), "fix": c.get("fix")}
          for c in rl]
    res[sid] = {"facts": facts, "identical": json.dumps(py) == json.dumps(rr), "n": len(py), "python_s": round(pyw, 2)}
    print(sid, res[sid], flush=True)
    out.write_text(json.dumps(res, indent=1))
print("identical", sum(r.get("identical", False) for r in res.values()), "of", len(res))
