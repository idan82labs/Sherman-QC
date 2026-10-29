"""D2 synthetic probe: free-datum TWO_DIRECTION_TAB vs mirrored / wrong-family scans, D2 off (k=0) and on."""
import dataclasses, hashlib, json, os, subprocess, sys, shutil
from pathlib import Path
import numpy as np
REPO = Path("/Users/idantal/ShermanQC/FullProject"); sys.path[:0] = [str(REPO), str(REPO / "backend")]
from contracts.production.part_recipe import PartRecipeV1
from contracts.production.canonical import canonical_json_bytes
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.compile import compile_part
from domains.production.engine.synthetic.cases import generate_case, plan_cases
from domains.production.engine.synthetic.ply import read_ply, write_ply
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
specs = plan_cases("smoke", "DEV")
def gen(name, fam):
    spec = next(s for s in specs if s.family == fam)
    spec = dataclasses.replace(spec, case_id=f"S-{name}", scenario="COMPLETE_NOMINAL", exact=True, plan="FULL")
    d = root / spec.case_id
    if not d.exists():
        generate_case(spec, root)
        rec = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
        assert compile_part(rec, d / "part.step", d / "compiled", flat_path=d / "flat.dxf").status == "COMPILED"
    return d
base = gen(sys.argv[2], sys.argv[3])
others = [gen(f, f) for f in sys.argv[4:]]
comp = root / "free_compiled"
if comp.exists(): shutil.rmtree(comp)
shutil.copytree(base / "compiled", comp)
doc = json.loads((comp / "compiled_part.json").read_text())
for p in doc["panels"]:
    if p["is_datum"]:
        for fe in p["free_edges"]: fe["registration_usable"] = False
        p["holes_mm"] = []
doc.pop("compiled_part_sha256"); (comp / "compiled_part.json").write_bytes(canonical_json_bytes(CompiledPartV1.finalize(**doc)))
pts, _, _ = read_ply(base / "scan.ply"); mir = root / "mirrored.ply"; write_ply(mir, pts * np.array([-1.0, 1.0, 1.0]))
(root / "k0.json").write_text('{"free_datum_rank_k": 0}')
scans = {"true": base / "scan.ply", "mirrored": mir, **{f"wrong_{o.name}": o / "scan.ply" for o in others}}
env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
(root / "d2.json").write_text(os.environ.get("D2CFG", "{}"))
for k in os.environ.get("KS", "k0,d2").split(","):
    for name, sc in scans.items():
        out = root / f"{k}_{name}"
        if out.exists(): shutil.rmtree(out)
        cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(comp), "--recipe", str(base / "recipe.json"),
               "--scan", str(sc), "--out", str(out), "--scan-sha256", hashlib.sha256(sc.read_bytes()).hexdigest()]
        cmd += ["--config", str(root / ("k0.json" if k == "k0" else "d2.json"))]
        subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, timeout=600)
        r = json.loads((out / "result.json").read_text())
        log = (out / "engine.log").read_text()
        trig = "free datum (" in log
        best = [l for l in log.splitlines() if "best datum" in l][:1]
        print(k, name, r["inspection_validity"]["state"], r["inspection_validity"]["reasons"], "trigger", trig,
              sorted({c["decision"] for b in r["bends"] for c in b["characteristics"]}), best[0][12:] if best else "", flush=True)
