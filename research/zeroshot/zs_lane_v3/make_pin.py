"""Write the production pin for zs_lane_v3 (``domains/production/zeroshot/assets/zs_lane_v3.pin.json``) and vendor its
two models byte-identical next to it. The pin copies config_v3.json's code fingerprint, the model sha256s and the
frozen lane parameters (the engine bundle carries only .py sources, the pins and the models, not research JSON).
It does not change which pin production runs (``pin.DEFAULT_LANE`` stays zs_lane_v1).

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3.make_pin [--ledger TEXT]
"""
from __future__ import annotations

import argparse
import json
import shutil

from research.zeroshot.zs_lane_v3 import fingerprint as FP
from research.zeroshot.zs_lane_v3.core import REPO

CONFIG = REPO / "research/zeroshot/zs_lane_v3/config_v3.json"
ASSETS = "domains/production/zeroshot/assets"
PIN = f"{ASSETS}/zs_lane_v3.pin.json"
V1_PIN = REPO / f"{ASSETS}/zs_lane_v1.pin.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="")
    a = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text())
    v = cfg["variants"]["zs"]
    if FP.lane_fingerprint()["sha256"] != cfg["code_fingerprint"]["sha256"]:
        raise SystemExit("code on disk differs from config_v3.json: refuse to pin")
    models = {}
    for role, key, name in (("zs", "model", "zs_lane_v3_zs_gbm.pkl"), ("z3", "z3_model", "zs_lane_v3_z3_gbm.pkl")):
        src = REPO / v[key]
        if FP.sha256_file(src) != v[f"{key}_sha256"]:
            raise SystemExit(f"{v[key]} differs from config_v3.json")
        dst = REPO / ASSETS / name
        if not (dst.is_file() and FP.sha256_file(dst) == v[f"{key}_sha256"]):     # vendored copies are read-only
            shutil.copyfile(src, dst)
        assert FP.sha256_file(dst) == v[f"{key}_sha256"]
        models[role] = {"rel_path": f"{ASSETS}/{name}", "sha256": v[f"{key}_sha256"], "frozen_source": v[key],
                        "loader": "pickle (loaded only after the sha256 check); sklearn HistGradientBoostingClassifier "
                                  "wrapped in research.zeroshot.z3_slicing.selector.GBM"}
    models["zs"]["features"] = "21 Z3 + 5 pool flags (has_z3, in_z2, in_a0[always 0], n_src, ext_log_len)"
    models["z3"]["features"] = "21 Z3 (Z3R only)"
    v1 = json.loads(V1_PIN.read_text())
    pin = {
        "schema": "sherman_zeroshot_lane_pin_v1",
        "lane": "zs_lane_v3",
        "variant": "zs",
        "code_fingerprint": cfg["code_fingerprint"],
        "frozen_config": {"path": "research/zeroshot/zs_lane_v3/config_v3.json", "schema": cfg["schema"],
                          "sha256": FP.sha256_file(CONFIG)},
        "model": models["zs"],
        "extra_models": {"z3": models["z3"]},
        "threshold": v["threshold"],
        "threshold_rule": cfg["threshold_rule"],
        "lane_params": {"variant": {k: v[k] for k in ("threshold", "feats", "post", "fix")},
                        "z3_params": cfg["z3_params"], "z2_fallback": cfg["z2_fallback"],
                        "budget": cfg["budget"]},
        "revision": cfg["revision"],
        "placement": v["placement"],
        "cad_priors": dict(v1["cad_priors"], count_cap="at most n_bends lines (full-stage bend count of the "
                           "CompiledPart), highest line score first (joined / extended lines: the max pooled "
                           "probability of their pieces; a Z3R line: its own pooled probability, below 0.5)"),
        "ledger": a.ledger,
        "status": "PREPARED, NOT LIVE: pin.DEFAULT_LANE is zs_lane_v1; flip only per "
                  "docs/production/runlog/zeroshot_2026-10-25/v3/REPIN_PLAN.md after the fresh-scan test passes",
    }
    (REPO / PIN).write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")
    print(PIN, FP.sha256_file(REPO / PIN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
