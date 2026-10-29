"""POST-HOC, CONSUMED diagnostics driver: E3 (new engine) on consumed FRESH / DEV-CHECK / generality scans,
one case per lock hold. Output: OUT/<case>/ and OUT/posthoc.jsonl. Not a new held-out result."""
import json, os, subprocess, sys, time
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
S = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
W = S / "final_eval_w"; FX = S / "fixes"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
only = set(sys.argv[2:])
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze.common import dev_sku_for
LOCK = "/private/tmp/claude-501/sherman_geom.lock"
man = json.loads((W / "fresh" / "fresh_cases.json").read_text())
cases = []
NEW = FX / "compile_after1"
for p in man["parts"]:
    sku = dev_sku_for(p["step_sha256"])
    for s in p["scans"]:
        cid = "fr_" + s["scan_id"][3:]
        old = W / "fresh" / "compiled" / "dev" / sku
        if (old / "compiled_part.json").exists() and json.loads((old / "compiled_part.json").read_text())["status"] == "COMPILED":
            cdir, rec = old, W / "fresh" / "recipes" / sku / "dev.recipe.json"
        elif (NEW / sku / "compiled" / "compiled_part.json").exists():
            cdir, rec = NEW / sku / "compiled", NEW / sku / "dev.recipe.json"
        else:
            cases.append({"case": cid, "part": p["part_id"], "skip": "NO_COMPILE"}); continue
        cases.append({"case": cid, "part": p["part_id"], "scan": s["path"], "cdir": str(cdir), "recipe": str(rec),
                      "label": str(REPO / s["studio_label"])})
D = W / "devcheck_SEALED"
dc = json.loads((REPO / "docs/production/runlog/freeze/devcheck_SEALED.json").read_text())
for c in dc["cases"]:
    e = D / "physical" / "E3" / c["case_ref"]
    if not (e / "input" / "scan.ply").exists():
        continue
    cases.append({"case": c["case_ref"], "part": c["family"], "scan": str(e / "input" / "scan.ply"),
                  "cdir": str(D / "compiled" / "dev" / c["recipe_sku"]), "recipe": str(e / "input" / "recipe.json"), "label": None})
j = NEW / dev_sku_for("da738a95faec" + "0" * 52)  # placeholder, replaced below
for sku_dir in NEW.iterdir():
    if sku_dir.name.startswith("DEV-DA738A95FA"):
        cases.append({"case": "gc_sc_822f083e3dfc", "part": "37126000",
                      "scan": str(REPO / "data/bend_corpus/incoming_email_2026-10-17/20260129_19c08b45/37126000_D.ply"),
                      "cdir": str(sku_dir / "compiled"), "recipe": str(sku_dir / "dev.recipe.json"),
                      "label": str(REPO / "data/bend_corpus/annotations/ai_bend_lines_gold_v1/c769e053715d973985c0c85c.json")})
log = open(OUT / "posthoc.jsonl", "a")
for c in cases:
    if only and c["case"] not in only:
        continue
    if "skip" in c:
        log.write(json.dumps(c) + "\n"); log.flush(); continue
    while True:
        try:
            os.mkdir(LOCK); break
        except FileExistsError:
            time.sleep(3)
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1"}
        r1 = subprocess.run([sys.executable, str(FX / "run_e3.py"), c["scan"], c["cdir"], c["recipe"], str(OUT / c["case"])],
                            cwd=REPO, env=env, capture_output=True, text=True)
        head = json.loads(r1.stdout.splitlines()[0]) if r1.stdout else {"err": r1.stderr[-500:]}
        row = {**{k: c[k] for k in ("case", "part")}, "run": {k: head.get(k) for k in ("rc", "wall", "rss_gb", "state", "reasons")}}
        if c.get("label"):
            r2 = subprocess.run([sys.executable, str(FX / "posthoc_case.py"), str(OUT / c["case"] / "engine"), c["cdir"], c["label"]],
                                cwd=REPO, env=env, capture_output=True, text=True)
            row["lines"] = json.loads(r2.stdout.splitlines()[-1]) if r2.stdout else {"err": r2.stderr[-300:]}
        else:
            row["bends"] = head.get("bends")
    finally:
        os.rmdir(LOCK)
    log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row)[:400], flush=True)
