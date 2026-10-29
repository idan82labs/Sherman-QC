"""eval_plan.py: the frozen synthetic evaluation plan (unchanged rule from FREEZE.json; paths moved to work/).
Part-stages: REAL_DEV parts: final stage S_N and mid stage S_floor(N/2) (when 1 <= mid < N); G1 parts: the case's
expected stage (final when none). Seed: 20261028 + 1000 * ordinal. Variants: G, MX, MY, MZ, T (+ O, OX when the
scanned stage's map class is not ACHIRAL); mid stages: G, MX (+ O, OX when chiral).
Writes plan.json, gen.txt and runs_syn.tsv (candidate) + runs_syn_live.tsv (live, same scans)."""
import json
from pathlib import Path
W = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work")
corpus = json.load(open(W / "corpus.json"))
items = []
for c in corpus:
    m = json.load(open(W / "maps" / c["id"] / "chirality_map_v0.json"))
    cls = {s["stage_model_id"]: s["class"] for s in m["stages"]}
    n = c["n_stages"] - 1
    if c["corpus"] == "G1":
        stages = [(c["stage"] or f"S{n}", "case")]
    else:
        stages = [(f"S{n}", "final")]
        mid = n // 2
        if 1 <= mid < n:
            stages.append((f"S{mid}", "mid"))
    for st, role in stages:
        chiral = cls[st] != "ACHIRAL"
        v = ["G", "MX", "MY", "MZ", "T"] if role != "mid" else ["G", "MX"]
        if chiral:
            v += ["O", "OX"]
        items.append(dict(id=c["id"], corpus=c["corpus"], part=c["part"], cdir=c["cdir"], recipe=c["recipe"], stage=st,
                          role=role, stage_class=cls[st], part_class=m["part_class"], variants=v,
                          map=str(W / "maps" / c["id"] / "chirality_map_v0.json")))
for i, it in enumerate(items):
    it["key"] = f"{it['id']}_{it['stage']}"
    it["seed"] = 20261028 + 1000 * i
json.dump(items, open(W / "plan.json", "w"), indent=1)
with open(W / "gen.txt", "w") as g, open(W / "runs_syn.tsv", "w") as r, open(W / "runs_syn_live.tsv", "w") as rl:
    for it in items:
        g.write(f"{it['key']} {it['cdir']} {it['stage']} {it['map']} {it['seed']} {','.join(it['variants'])}\n")
        for v in it["variants"]:
            r.write("\t".join([f"{it['key']}/{v}", f"{W}/scans/{it['key']}/{v}.ply", it["cdir"], it["recipe"], it["map"],
                               f"{W}/runs/{it['key']}/{v}"]) + "\n")
            rl.write("\t".join([f"L:{it['key']}/{v}", f"{W}/scans/{it['key']}/{v}.ply", it["cdir"], it["recipe"], "-",
                                f"{W}/runs_live/syn/{it['key']}/{v}"]) + "\n")
print(len(items), sum(len(it["variants"]) for it in items))
