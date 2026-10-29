"""Build the job manifest: G1 card tier (50), A1 (27), DEV E3 cohort (50 OK rows of e3_driver.jsonl). Dedup physical scans
by (scan sha256, compiled sha, recipe sha). usage: build_manifest.py OUT.json"""
import json, sys, hashlib, os
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject"); H = REPO / "docs/production/runlog/zeroshot_2026-10-25"
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
W = Path("/private/tmp/claude-501/sherman_zeroshot_work/compile")
G1C = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/zsprod/g1c")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
entries = []
man = json.loads((G1C / "run_manifest.json").read_text())
for cid in man["case_ids"]:
    c = json.loads((G1C / "cases" / f"{cid}.json").read_text())
    cdir = Path(c["compile"]["compiled_dir"]); edir = Path(c["engine_output"]["dir"]).parent
    entries.append(dict(corpus="G1", id=cid, scan=str(edir / "input/scan.ply"), compiled_dir=str(cdir),
                        recipe=str(edir / "input/recipe.json"), ref_engine=str(edir / "engine")))
a1 = json.loads((REPO / "docs/production/runlog/freeze/cases_A1.json").read_text())
for c in a1["cases"]:
    sku = c["recipe_sku"]
    entries.append(dict(corpus="A1", id=c["case_ref"], scan=c["scan_path"], compiled_dir=str(W / sku / "compiled"),
                        recipe=str(W / sku / "dev.recipe.json"), sku=sku, compile_status=c["compile_status"]))
inv = {r["scan_id"]: r for r in json.loads((H / "INVENTORY.json").read_text())["scans"]}
excluded = []
dev = set(json.loads((H / "SPLIT_FROZEN.json").read_text())["dev"]["scans"])
for line in open(H / "baseline/e3_driver.jsonl"):
    r = json.loads(line)
    if r.get("status") != "OK": continue
    sid = r["case"]; assert sid in dev, sid
    i = inv[sid]
    if i["custody"] != "DEV_CONSUMED" or i["benchmark_split"] != "CONSUMED":
        excluded.append({"id": sid, "custody": i["custody"], "benchmark_split": i["benchmark_split"]}); continue
    sku = r["sku"]
    entries.append(dict(corpus="COH", id=sid, scan=os.path.expanduser(i["path"]), compiled_dir=str(W / sku / "compiled"),
                        recipe=str(W / sku / "dev.recipe.json"), sku=sku, split=i["benchmark_split"],
                        ref_engine=f"/private/tmp/claude-501/sherman_zeroshot_work/e3/{sid}/engine"))
uniq = {}
for e in entries:
    assert Path(e["scan"]).is_file() and Path(e["compiled_dir"], "compiled_part.json").is_file() and Path(e["recipe"]).is_file(), e
    e["scan_sha256"] = sha(e["scan"])
    e["compiled_sha256"] = load_compiled(Path(e["compiled_dir"]) / "compiled_part.json").compiled_part_sha256
    e["recipe_sha256"] = load_recipe(Path(e["recipe"])).recipe_sha256
    k = (e["scan_sha256"], e["compiled_sha256"], e["recipe_sha256"])
    e["key"] = hashlib.sha256(repr(k).encode()).hexdigest()[:16]
    uniq.setdefault(e["key"], e)
json.dump({"excluded_coh": excluded, "entries": entries, "unique_jobs": list(uniq.values())}, open(sys.argv[1], "w"), indent=1)
from collections import Counter
print(Counter(e["corpus"] for e in entries), "unique", len(uniq), Counter(e["corpus"] for e in uniq.values()))
print("excluded", Counter(x["custody"] for x in excluded)); print(Counter(e.get("split") for e in entries if e["corpus"] == "COH"))
