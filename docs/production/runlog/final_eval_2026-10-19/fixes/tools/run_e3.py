"""POST-HOC diagnostic E3 run (consumed cases). usage: run_e3.py SCAN COMPILED_DIR RECIPE OUT [--mirror]"""
import sys, json, hashlib, shutil
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
import numpy as np
from domains.production.engine.eval.freeze import physical
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
from domains.production.engine.eval.procmon import run_monitored
from domains.production.engine.eval.runner import RunnerConfig, engine_env
from domains.production.engine.eval.plyio import read_ply, write_mesh_ply
scan, cdir, recipe_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
if "--mirror" in sys.argv:
    d = read_ply(scan); v = d.vertices * np.array([-1.0, 1.0, 1.0])
    f = d.faces[:, ::-1] if d.faces is not None else None
    scan = out / "scan_mirrored.ply"; write_mesh_ply(scan, v, f)
RC = physical._RC
compiled = load_compiled(cdir / "compiled_part.json"); recipe = load_recipe(recipe_p)
sha = hashlib.sha256(scan.read_bytes()).hexdigest()
argv = [sys.executable, *RC["engine_argv_tail"], "--compiled", str(cdir / "compiled_part.json"), "--recipe", str(recipe_p),
        "--scan", str(scan), "--out", str(out / "engine"), "--scan-sha256", sha, "--recipe-sha256", recipe.recipe_sha256,
        "--time-budget-s", f"{RC['engine_time_budget_s']:g}", "--compiled-sha256", compiled.compiled_part_sha256]
for a in sys.argv:
    if a.startswith("--cfg="):
        cf = out / "cfg.json"; cf.write_text(a[6:]); argv += ["--config", str(cf)]
env = engine_env(RunnerConfig(tier_dir=out, out_dir=out, repo_root=REPO))
proc = run_monitored(argv, cwd=REPO, env=env, timeout_s=float(RC["engine_timeout_s"]), rss_limit_bytes=int(RC["engine_rss_limit_bytes"]),
                     stderr_path=out / "stderr.log", stdout_path=out / "stdout.log")
r = json.loads((out / "engine" / "result.json").read_text()) if (out / "engine" / "result.json").exists() else {}
iv = r.get("inspection_validity") or {}
print(json.dumps({"rc": proc.returncode, "wall": round(proc.wall_s, 1), "rss_gb": round((proc.peak_rss_bytes or 0) / 2**30, 2), "state": iv.get("state"), "reasons": iv.get("reasons"),
                  "bends": [(b["label"], b["state"], [(c["kind"][:4], c["decision"], c.get("actual")) for c in b["characteristics"] if c["kind"] in ("INCLUDED_ANGLE",)]) for b in r.get("bends", [])]}))
print(open(out / "engine" / "engine.log").read()[-1500:] if (out / "engine" / "engine.log").exists() else "")
