"""Write config_v2.json for zs_lane_v2: the zs variant, its model (+ sha256), the fixed threshold, the span
post-processing parameters, the Z3 / Z2 settings and the pinned code fingerprint.
usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v2.freeze_config --model PATH --post JSON [--z3 JSON]
       [--z2-fallback] [--feats base]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.zeroshot.zs_lane_v2 import fingerprint as FP
from research.zeroshot.zs_lane_v2.core import REPO
from research.zeroshot.zs_lane_v2.postproc import PostParams


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--post", default="{}")
    ap.add_argument("--z3", default="")
    ap.add_argument("--z2-fallback", action="store_true")
    ap.add_argument("--feats", default="base")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    post = json.loads(a.post)
    PostParams(**post)        # validates the keys
    v = {"zs": {"use_z3": True, "use_z2": True, "model": a.model, "threshold": 0.5, "feats": a.feats,
                "model_sha256": FP.sha256_file(REPO / a.model), "post": post,
                "features": "21 Z3 + 5 pool flags" + (" + 6 station-sweep summaries" if "sweep" in a.feats else ""),
                "placement": "D1 gated outer-apex snap of every pooled candidate on the Z3 mesh; joined lines span "
                             "the extreme snapped piece ends along the longest piece"}}
    cfg = {"schema": "sherman.zeroshot.zs_lane_v2.config.v1", "lane": "zs_lane_v2", "variants": v,
           "z3_params": json.loads(a.z3) if a.z3 else None, "z2_fallback": bool(a.z2_fallback),
           "threshold_rule": "fixed 0.5 on the GBM probability (as zs_lane_v1)", "note": a.note,
           "code_fingerprint": FP.lane_fingerprint(), "scorer_fingerprint": FP.scorer_fingerprint()}
    p = Path(__file__).parent / "config_v2.json"
    p.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    print(p, FP.sha256_file(p), cfg["code_fingerprint"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
