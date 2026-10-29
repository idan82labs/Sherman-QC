"""Run E3 (repo engine or --engine-root) on copied iteration-2 E2 joggle probe cases and print truth comparisons."""
import hashlib, json, os, subprocess, sys, shutil
from pathlib import Path
cdir = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
scan = next(cdir.glob("scan.*"))
sha = hashlib.sha256(scan.read_bytes()).hexdigest()
out = cdir / ("e3" if root is None else "e3_base")
if out.exists():
    shutil.rmtree(out)
cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(cdir / "compiled"), "--recipe", str(cdir / "recipe.json"),
       "--scan", str(scan), "--out", str(out), "--scan-sha256", sha]
env, cwd = None, None
if root is not None:
    repo = Path.cwd()
    env = dict(os.environ, PYTHONPATH=f"{root}:{repo}:{repo / 'backend'}"); cwd = str(root)
r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
truth = json.loads((cdir / "truth.json").read_text())
res = json.loads((out / "result.json").read_text())
exp = truth["expected"]["bend_states"]
tc = truth["characteristics"]
print("==", cdir.name, "engine", res["engine"]["engine_version"], "validity", res["inspection_validity"]["state"], "summary", res["summary"]["state"])
nfail = 0
for b in res["bends"]:
    line = [f"{b['label']} {b['state']} (truth {exp.get(b['bend_uid'])})"]
    for ch in b["characteristics"]:
        t = tc.get(ch["characteristic_id"], {})
        td = t.get("true_deviation")
        dv = ch.get("deviation")
        err = None if (dv is None or td is None) else dv - td
        nfail += ch["decision"] == "FAIL"
        line.append(f"{ch['kind'][:4]}{ch.get('side') or ''}:{ch['decision'][:3]} dev {None if dv is None else round(dv,3)} true {None if td is None else round(td,3)} U {ch.get('uncertainty') and round(ch['uncertainty'],3)}")
    print("  " + " | ".join(line))
