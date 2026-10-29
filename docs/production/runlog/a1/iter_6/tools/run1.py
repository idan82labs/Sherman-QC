"""Run E3 (the repository engine, CLI) on named A1 cases' frozen inputs. usage: run1.py OUT_ROOT CASE [CASE ...]
Inputs come from the iteration-5 physical work dir (input/scan.ply + input/recipe.json, opaque names) and its
compiled parts; nothing SEALED is read."""
import hashlib, json, subprocess, sys, time
from pathlib import Path
SP = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
SRC = SP / "a1_iter5/phys"
REPO = Path("/Users/idantal/ShermanQC/FullProject")
cases = {c["case_ref"]: c for c in json.loads((REPO / "docs/production/runlog/freeze/cases_A1.json").read_text())["cases"]}
out_root = Path(sys.argv[1])
for cid in sys.argv[2:]:
    c = cases[cid]
    inp = SRC / "physical/E3" / cid / "input"
    scan = inp / "scan.ply"
    sha = hashlib.sha256(scan.read_bytes()).hexdigest()
    out = out_root / cid
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(SRC / "compiled/dev" / c["recipe_sku"]),
           "--recipe", str(inp / "recipe.json"), "--scan", str(scan), "--out", str(out / "engine"), "--scan-sha256", sha]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    (out / "stderr.log").write_text(r.stdout + r.stderr)
    res = out / "engine/result.json"
    if r.returncode != 0 or not res.exists():
        print(cid, "ERROR rc", r.returncode, (r.stdout + r.stderr)[-500:])
        continue
    d = json.loads(res.read_text())
    print(cid, c["family"], d["inspection_validity"]["state"], f"{time.time() - t0:.1f}s")
    for b in d["bends"]:
        chs = {x["kind"][:5] + (x.get("side") or ""): (x["decision"], None if x.get("deviation") is None else round(x["deviation"], 2), x.get("reason")) for x in b["characteristics"]}
        print(f"   {b['label']:4s} {b['state']:18s} notes={list(b['notes'])} {chs}")
    for name in ("registered_scan.ply", "deviations.f32", "segment_labels.u16"):
        p = out / "engine" / name
        if p.exists():
            p.unlink()
