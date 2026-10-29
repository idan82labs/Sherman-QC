"""Fit the final zs_lane_v4 procedure on the whole pool and write the frozen config (config_v4.json) + model.

The final procedure = the chosen variant's procedure P (train.fit_procedure) run on all 217 pool scans: the inner
5-fold grouped CV over all 152 families picks the SelParams, then one GBM is fit on every pool scan. The config
pins the code fingerprint (fingerprint.py), the model sha256 and every runtime parameter.
usage (repo root): .venv/bin/python -B -m research.zeroshot.zs_lane_v4.freeze_config <variant> [--models DIR]
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

from research.zeroshot.zs_lane_v4 import fingerprint as FP, lofo as LO, train as TR, variants as VA
from research.zeroshot.zs_lane_v4.core import REPO

CONFIG = REPO / "research/zeroshot/zs_lane_v4/config_v4.json"
MODELS = REPO / "docs/production/runlog/zs_v4/models"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--models", default=str(MODELS))
    a = ap.parse_args(argv)
    spec = VA.get(a.variant)
    items = LO.load_items()
    data = TR.V4Data(items, hem=VA.hem_lines(spec, items), hyp=spec.hyp, stab=VA.stab_of(spec, items))
    fp = TR.fit_procedure(data, [it.case_id for it in items], spec, log=print)
    md = Path(a.models)
    md.mkdir(parents=True, exist_ok=True)
    mpath = md / f"zs_lane_v4_{a.variant}.pkl"
    mpath.write_bytes(pickle.dumps(fp["model"], protocol=5))
    from research.zeroshot.zs_lane_v4.hem import HemParams
    runtime = {"hyp": spec.hyp, "stab": spec.stab, "hem": bool(spec.hem and fp["params"].hem),
               "hem_params": asdict(HemParams()), "cols": spec.cols, "select": asdict(fp["params"])}
    cfg = {
        "schema": "sherman.zeroshot.zs_lane_v4.config.v1",
        "lane": "zs_lane_v4",
        "variant": a.variant,
        "spec": VA.describe(spec),
        "status": "RESEARCH CANDIDATE, NOT LIVE: production stays zs_lane_v1 until the fresh-scan promotion test "
                  "(docs/production/runlog/zs_v4/PREREG_FRESH.md) passes and the owner decides",
        "detection": {"budget": {"max_faces": 3000000, "cell_mm": 0.30}, "z3_params": {"fallback_frac_only": True},
                      "z2_fallback": True, "max_families": 20,
                      "thickness": "Z2 self-recipe estimate on the (budgeted) scan unless an external t is given"},
        "runtime": runtime,
        "inner_selection": {"chosen": fp["params"].key(), "inner_scores": fp["inner_scores"],
                            "grid_size": len(spec.grid)},
        "model": {"path": str(mpath.relative_to(REPO)), "sha256": FP.sha256_file(mpath),
                  "class": "research.zeroshot.zs_lane_v4.model.GBM", "trained_on": f"v4 pool, {len(items)} scans "
                  "(freeze-ledger seq 48), labels = frozen-gate match at the output placement"},
        "code_fingerprint": FP.lane_fingerprint(),
        "scorer_fingerprint": FP.scorer_fingerprint(),
    }
    CONFIG.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    print(CONFIG, FP.sha256_file(CONFIG), cfg["model"])
    print("chosen", fp["params"].key())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
