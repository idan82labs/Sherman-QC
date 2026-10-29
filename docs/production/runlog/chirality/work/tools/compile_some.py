"""Recompile E1 DEV recipes + compiled parts for an explicit list of STEP files (same code path as
final_eval_2026-10-19/fixes/tools/compile_all.py, without the repo-wide STEP glob). No labels read.
usage: compile_some.py STEPS.json OUT_DIR    (STEPS.json: {sku: step_path})"""
import hashlib, json, sys, time
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze import recipes
from domains.production.engine.eval.freeze.common import dev_sku_for
from domains.production.engine.eval.runner import RunnerConfig, engine_env
from domains.production.engine.compile import dev_recipe as dr
steps = json.loads(Path(sys.argv[1]).read_text()); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
env = engine_env(RunnerConfig(tier_dir=OUT, out_dir=OUT))
stats = []
for want, p in steps.items():
    p = Path(p); sha = hashlib.sha256(p.read_bytes()).hexdigest(); sku = dev_sku_for(sha)
    assert sku == want, (want, sku, p)
    wd = OUT / sku; wd.mkdir(exist_ok=True); rec = {"sha": sha[:12], "sku": sku, "path": str(p)}; t0 = time.time()
    try:
        cands, rep = dr.dev_candidates(p)
    except Exception as exc:
        cands, rep = [], {"error": f"{type(exc).__name__}: {exc}"[:300]}
    rec["candidates"] = len(cands)
    if not cands:
        rec.update(status="NO_RECIPE", reason="NO_CANDIDATE_BENDS")
    else:
        try:
            recp = dr.dev_recipe(p, cands, sku=sku, inspection_stage="FINAL")
        except Exception as exc:
            rec.update(status="NO_RECIPE", reason=f"RECIPE_BUILD_REFUSED: {type(exc).__name__}: {exc}"[:300])
        else:
            rp = wd / "dev.recipe.json"; recipes._write_recipe(rp, recp)
            rec["recipe_sha256"] = hashlib.sha256(rp.read_bytes()).hexdigest()
            c = recipes.compile_recipe(rp, p, wd / "compiled", env=env)
            rec.update(status=c.get("status"), reason=c.get("reason"), bends=c.get("bends"))
    rec["wall_s"] = round(time.time() - t0, 1); stats.append(rec); print(json.dumps(rec), flush=True)
(OUT / "compile_some.json").write_text(json.dumps(stats, indent=1, default=str))
print("COMPILED", sum(r["status"] == "COMPILED" for r in stats), "/", len(stats))
