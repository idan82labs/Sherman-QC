"""For SKUs that have a frozen A1 recipe (freeze/recipes), compile THAT recipe (as A1 did) instead of the rebuilt one.
usage: recompile_frozen.py STEPS.json COMPILE_DIR"""
import json, shutil, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze import recipes
from domains.production.engine.eval.runner import RunnerConfig, engine_env
steps = json.loads(Path(sys.argv[1]).read_text()); OUT = Path(sys.argv[2])
env = engine_env(RunnerConfig(tier_dir=OUT, out_dir=OUT)); rows = []
for sku, step in steps.items():
    fr = REPO / "docs/production/runlog/freeze/recipes" / f"{sku}.recipe.json"
    if not fr.is_file():
        continue
    wd = OUT / sku; wd.mkdir(exist_ok=True)
    if (wd / "compiled").exists():
        shutil.rmtree(wd / "compiled")
    shutil.copyfile(fr, wd / "dev.recipe.json")
    c = recipes.compile_recipe(wd / "dev.recipe.json", Path(step), wd / "compiled", env=env)
    rows.append({"sku": sku, "recipe": "FROZEN_A1", "status": c.get("status"), "reason": c.get("reason"), "bends": c.get("bends")})
    print(json.dumps(rows[-1]), flush=True)
(OUT / "recompile_frozen.json").write_text(json.dumps(rows, indent=1))
