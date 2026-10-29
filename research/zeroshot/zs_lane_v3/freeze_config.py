"""Write config_v3.json for zs_lane_v3: the zs variant, its two models (+ sha256), the fixed threshold, the frozen
zs_lane_v2 post-processing (copied from config_v2.json), the v3 fix parameters, the Z3 / Z2 settings and the pinned
code and scorer fingerprints. Revision 2 adds ``revision`` and the face ``budget``; revision 3 adds ``budget.cell_mm``
(the density-invariant fixed-grid resampling).
usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3.freeze_config --model PATH --z3-model PATH [--note TEXT]"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from research.zeroshot.zs_lane_v3 import budget as BUD
from research.zeroshot.zs_lane_v3 import fingerprint as FP
from research.zeroshot.zs_lane_v3.core import REPO
from research.zeroshot.zs_lane_v3.fixes import FixParams
from research.zeroshot.zs_lane_v2.postproc import PostParams

V2_CONFIG = REPO / "research/zeroshot/zs_lane_v2/config_v2.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--z3-model", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    v2 = json.loads(V2_CONFIG.read_text())
    v2zs = v2["variants"]["zs"]
    post = dict(v2zs["post"])
    PostParams(**post)                       # validates the keys
    fix = asdict(FixParams())
    v = {"zs": {"use_z3": True, "use_z2": True, "model": a.model, "model_sha256": FP.sha256_file(REPO / a.model),
                "z3_model": a.z3_model, "z3_model_sha256": FP.sha256_file(REPO / a.z3_model),
                "threshold": 0.5, "feats": "base", "post": post, "fix": fix,
                "features": "pooled GBM: 21 Z3 + 5 pool flags; Z3-alone GBM (Z3R only): 21 Z3",
                "placement": v2zs["placement"] + "; EXT lines span the union with the absorbed Z2 extent along "
                             "their own axis; Z3R lines at the candidate's snapped placement"}}
    cfg = {"schema": "sherman.zeroshot.zs_lane_v3.config.v1", "lane": "zs_lane_v3", "base": "zs_lane_v2",
           "base_config_sha256": FP.sha256_file(V2_CONFIG), "variants": v,
           "z3_params": v2.get("z3_params"), "z2_fallback": bool(v2.get("z2_fallback")),
           "threshold_rule": "fixed 0.5 on the pooled GBM probability (as zs_lane_v1 / v2); Z3R floor 0.9 on the "
                             "Z3-alone GBM probability",
           "revision": 3,
           "budget": {"max_faces": BUD.MAX_FACES, "cell_mm": BUD.CELL_MM,
                      "rule": "a scan with more triangles than max_faces (file header) runs on a resampling of itself "
                              "on a fixed absolute grid (cell cell_mm x 1.12^k, the smallest k whose output has at "
                              "most max_faces faces): one QEF vertex per (cell, skin) from the scan's grid-line "
                              "crossings, one quad per crossing (budget.py, revision 3); within the budget the scan "
                              "is used unchanged"},
           "revision_note": "revision 3 (budget step only): the revision-2 vertex clustering (cell grown with the "
                            "input density) is replaced by the density-invariant fixed-grid resampling (budget.py). "
                            "Same models, threshold, post-processing, fixes, trigger (max_faces) and every other "
                            "stage as revision 2 (code fingerprint ab31d061..., config sha 1c3f8cb7...)",
           "note": a.note, "code_fingerprint": FP.lane_fingerprint(), "scorer_fingerprint": FP.scorer_fingerprint()}
    p = Path(__file__).parent / "config_v3.json"
    p.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    print(p, FP.sha256_file(p), cfg["code_fingerprint"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
