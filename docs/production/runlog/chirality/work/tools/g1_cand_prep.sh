#!/bin/bash
# g1_cand_prep.sh: the candidate's G1 run reuses the live run's compiles; each compiled out/ dir gets its part's map
# (the candidate's default map location, <compiled_dir>/chirality_map_v0.json).
W=/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work
mkdir -p $W/g1_cand && rsync -a $W/g1_live/compile_cache/ $W/g1_cand/compile_cache/
python3 - <<'PY'
import json, shutil
from pathlib import Path
W = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work")
for c in json.load(open(W / "corpus.json")):
    if c["corpus"] != "G1": continue
    out = W / "g1_cand" / Path(c["cdir"]).relative_to(W / "g1_live")
    for f in ("chirality_map_v0.json", "chirality_map_v0.npz"):
        shutil.copyfile(W / "maps" / c["id"] / f, out / f)
print("ok")
PY
