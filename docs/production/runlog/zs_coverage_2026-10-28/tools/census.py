"""Coverage census (2026-10-28): sha256 of mesh files in the standard census roots, metadata only (no geometry).

Prunes by path BEFORE hashing (never hashed, opened or copied):
  *_SEALED*, Eyal50 quarantine (*eyal*), rhino-preview, 41880000_D, HOLDOUT10 source dir
  (2026-08-30_shaked_daily_scans), the FE1 members' source dirs/names (from SET_MANIFEST copies and source names),
  final_eval_2026-10-28_heldout/, fe1_stage_2026-10-28/, f1-sep02-email-prospective-holdout-v1/.
Exception: 30820000_A01 (V)/(X) (FE1 listed_not_scored_consumed, not FE1-scored) are hashed from ~/Downloads.
Writes census.json: [{path, size, sha256}] + pruned path count.
"""
import hashlib, json, os, re, sys
from pathlib import Path

HOME = Path.home()
REPO = HOME / "ShermanQC/FullProject"
OUT = Path(sys.argv[1])
paths_file = Path(sys.argv[2])
man = json.loads((REPO / "docs/production/runlog/final_eval_2026-10-28_heldout/SET_MANIFEST.json").read_text())
fe1_paths = set()
fe1_stems = set()
for m in man["scored"]:
    for c in m.get("copies") or []:
        fe1_paths.add(os.path.realpath(c["path"]))
    sn = m["source_name"]
    fe1_stems.add(re.sub(r"\s*\((V|X)\)$", "", sn))
PRUNE_SUB = ["_sealed", "eyal", "rhino", "41880000_d", "2026-08-30_shaked_daily_scans", "final_eval_2026-10-28_heldout",
             "fe1_stage_2026-10-28", "f1-sep02-email-prospective-holdout", "shaked_new_good_bad_pair"]
rows, pruned = [], 0
for ln in paths_file.read_text().splitlines():
    p = (HOME / ln)
    low = ln.lower()
    stem_hit = any(s.lower() in Path(ln).name.lower() for s in fe1_stems)
    if any(s in low for s in PRUNE_SUB) or stem_hit or os.path.realpath(p) in fe1_paths:
        pruned += 1
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    rows.append({"path": str(p), "size": p.stat().st_size, "sha256": h.hexdigest()})
for nm in ("30820000_A01 (V).ply", "30820000_A01 (X).ply"):
    p = HOME / "Downloads" / nm
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append({"path": str(p), "size": p.stat().st_size, "sha256": h})
OUT.write_text(json.dumps({"pruned_before_hash": pruned, "fe1_stems_pruned": sorted(fe1_stems), "rows": rows}, indent=1))
print(len(rows), "hashed;", pruned, "pruned;", len({r['sha256'] for r in rows}), "unique")
