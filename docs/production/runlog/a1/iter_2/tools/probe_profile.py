"""Synthetic probe (DEV seeds, E2 generator, E1 compile, E3 CLI): custom profile parts (e.g. joggles).

usage: probe_profile.py OUT_ROOT NAME SCENARIO PLAN 'json design spec'
design spec: {"t":1.5,"width":150,"segments":[["PBASE",80],["PA",21.4],["PJ",3.6],["PT",13]],
              "sweeps":[90,35,35],"dirs":["UP","UP","DOWN"],"radii":[0.78,0.1,0.1],"seq":[0,1,2],"min_straight":2.0}
"""
import dataclasses, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

out_root, name, scenario, plan, spec_json = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
d = json.loads(spec_json)
from domains.production.engine.synthetic import families as F, cases as C
from domains.production.engine.synthetic.families import ProfileSegment, build_profile, _chiral_holes

if "min_straight" in d:
    F._min_straight = lambda t: float(d["min_straight"])

def custom(family, rng, req=F.FeatureRequest()):
    segs = []
    for i, (key, ln) in enumerate(d["segments"]):
        holes = _chiral_holes(rng, 3, 4.0) if i == 0 else ()
        segs.append(ProfileSegment(key, float(ln), "flange", holes))
    return build_profile(family="Z_STEP", thickness=float(d["t"]), width=float(d["width"]), segments=segs,
                         sweeps=[float(s) for s in d["sweeps"]], directions=d["dirs"], radii=[float(r) for r in d["radii"]],
                         sequence=d["seq"], groups=[], params={"probe": name})

C.build_family = custom
specs = C.plan_cases("smoke", "DEV")
base = next(s for s in specs if s.family == "Z_STEP")
spec = dataclasses.replace(base, case_id=f"PROBE-{name}", scenario=scenario, exact=scenario == "COMPLETE_NOMINAL", plan=plan,
                           point_target=None)
cdir = out_root / spec.case_id
if not cdir.exists():
    C.generate_case(spec, out_root)
from contracts.production.part_recipe import PartRecipeV1
from domains.production.engine.compile import compile_part
if not (cdir / "compiled" / "compiled_part.json").exists():
    recipe = PartRecipeV1.model_validate_json((cdir / "recipe.json").read_bytes())
    res = compile_part(recipe, cdir / "part.step", cdir / "compiled", flat_path=cdir / "flat.dxf")
    print("compile", res.status)
scan = next(cdir.glob("scan.*"))
sha = hashlib.sha256(scan.read_bytes()).hexdigest()
out = cdir / "e3"
if out.exists():
    import shutil; shutil.rmtree(out)
cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(cdir / "compiled"), "--recipe", str(cdir / "recipe.json"),
       "--scan", str(scan), "--out", str(out), "--scan-sha256", sha]
r = subprocess.run(cmd, capture_output=True, text=True)
print("e3 rc", r.returncode, r.stdout.strip().splitlines()[-1][:200] if r.stdout.strip() else r.stderr[-500:])
truth = json.loads((cdir / "truth.json").read_text())
res = json.loads((out / "result.json").read_text())
exp_states = truth["expected"]["bend_states"]
tc = truth["characteristics"]
print("validity", res["inspection_validity"]["state"], "stage", truth["stage"]["expected_stage_model_id"], "summary", res["summary"]["state"])
for b in res["bends"]:
    line = [f"{b['label']} {b['state']} (truth {exp_states.get(b['bend_uid'])})"]
    for ch in b["characteristics"]:
        t = tc.get(ch["characteristic_id"], {})
        td = t.get("true_deviation")
        k = ch["kind"][:4] + (ch.get("side") or "")
        dv = ch.get("deviation")
        err = None if (dv is None or td is None) else dv - td
        line.append(f"{k}:{ch['state'][:4]}/{ch['decision'][:3]}/{ch.get('reason') or ''} dev {None if dv is None else round(dv,3)} true {None if td is None else round(td,3)} err {None if err is None else round(err,3)} U {ch.get('uncertainty') and round(ch['uncertainty'],3)}")
    print("  " + " | ".join(line))
log = (out / "engine.log").read_text().splitlines()
print("\n".join(l for l in log if "degenerate" in l or "free" in l or "best datum" in l))
