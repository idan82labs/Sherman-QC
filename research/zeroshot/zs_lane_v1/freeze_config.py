"""Write config_v1.json for zs_lane_v1: the three variants, their models (+ sha256), the fixed threshold rule and the
pinned code fingerprint. usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v1.freeze_config"""
from __future__ import annotations

import json
from pathlib import Path

from research.zeroshot.zs_lane_v1 import fingerprint as FP
from research.zeroshot.zs_lane_v1.core import REPO

MODELS = "docs/production/runlog/zeroshot_2026-10-25/integration/models"
THRESHOLD_RULE = ("fixed 0.5 on the GBM probability for every fold and for the final model (no threshold tuning). "
                  "Rationale: the Bayes decision point of a probability model, set a priori; DEV OOF with fixed 0.5 "
                  "equals the fold-tuned rule within noise (zs 0.944 fixed vs 0.939 tuned; z3 0.914 vs 0.913), and "
                  "the Z3 audit found fixed 0.5 = 0.929 vs fold-tuned 0.930. Replaces lane.py's undocumented 0.55.")


def main() -> int:
    v = {
        "zs": {"use_z3": True, "use_z2": True, "model": f"{MODELS}/zs_lane_v1_zs_gbm.pkl", "threshold": 0.5,
               "features": "21 Z3 + 5 pool flags (has_z3, in_z2, in_a0[always 0], n_src, ext_log_len)",
               "placement": "D1 gated outer-apex snap on the Z3 mesh"},
        "z3": {"use_z3": True, "use_z2": False, "model": f"{MODELS}/zs_lane_v1_z3_gbm.pkl", "threshold": 0.5,
               "features": "21 Z3", "placement": "D1 gated outer-apex snap on the Z3 mesh"},
        "z2": {"use_z3": False, "use_z2": True, "model": None, "threshold": None,
               "features": "none (Z2 it08 config: max_ro_t 12, min_span 8 mm)", "placement": "Z2-internal D1 snap"},
    }
    for x in v.values():
        x["model_sha256"] = FP.sha256_file(REPO / x["model"]) if x["model"] else None
    cfg = {"schema": "sherman.zeroshot.zs_lane_v1.config.v1", "lane": "zs_lane_v1", "variants": v,
           "threshold_rule": THRESHOLD_RULE, "code_fingerprint": FP.lane_fingerprint(),
           "scorer_fingerprint": FP.scorer_fingerprint()}
    p = Path(__file__).parent / "config_v1.json"
    p.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    print(p, FP.sha256_file(p), cfg["code_fingerprint"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
