"""corpus.py -> work/corpus.json: 22 compiled real parts (compile_real) + the 50 G1 regression-tier parts (compiled by
the live E4 run's cache, g1_live/compile_cache). G1 stage = the case's expected stage model."""
import json
from pathlib import Path
W = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work")
inv = json.load(open(W.parents[1] / "zeroshot_2026-10-25/INVENTORY.json"))
part_of = {}
for s in inv["scans"]:
    sku = (s.get("cad") or {}).get("sku")
    if sku: part_of.setdefault(sku, s.get("part_id"))
corpus = []
for sku in json.load(open(W / "real_steps.json")):
    cd = W / "compile_real" / sku / "compiled"
    c = json.load(open(cd / "compiled_part.json"))
    if c["status"] != "COMPILED":
        continue
    corpus.append(dict(id=sku, corpus="REAL_DEV", part=part_of.get(sku), cdir=str(cd), recipe=str(W / "compile_real" / sku / "dev.recipe.json"),
                       n_stages=len(c["stages"]), stage=None, compiled_part_sha256=c["compiled_part_sha256"]))
for f in sorted((W / "g1_live/cases").glob("*.json")):
    d = json.load(open(f))
    sc = d["score"]["case"]
    cd = Path(d["compile"]["compiled_dir"])
    c = json.load(open(cd / "compiled_part.json"))
    if c["status"] != "COMPILED":
        continue
    corpus.append(dict(id=d["case_id"], corpus="G1", part=d["case_id"], cdir=str(cd), recipe=str(W / "g1tier" / d["case_id"] / "recipe.json"),
                       n_stages=len(c["stages"]), stage=sc.get("expected_stage_model_id"), scenario=sc.get("scenario"),
                       compiled_part_sha256=c["compiled_part_sha256"]))
json.dump(corpus, open(W / "corpus.json", "w"), indent=1)
print(len(corpus), sum(c["corpus"] == "G1" for c in corpus))
