"""Write the zs_lane_v4 pin with the existing pin tooling (domains/production/zeroshot/pin.py: same schema
``sherman_zeroshot_lane_pin_v1``, same combined code fingerprint, verified by ``pin.verify``) as a NEW pin file in the
research tree: research/zeroshot/zs_lane_v4/pin/zs_lane_v4.pin.json, with the model vendored byte-identical next to
it. Nothing under domains/production/zeroshot is created or changed: ``pin.DEFAULT_LANE`` stays zs_lane_v1 and
``pin.PIN_RELS`` does not list v4, so production cannot select it.
usage (repo root): .venv/bin/python -B -m research.zeroshot.zs_lane_v4.make_pin [--ledger TEXT]
"""
from __future__ import annotations

import argparse
import json
import shutil

from domains.production.zeroshot import pin as PIN
from research.zeroshot.zs_lane_v4 import fingerprint as FP
from research.zeroshot.zs_lane_v4.core import REPO

CONFIG = REPO / "research/zeroshot/zs_lane_v4/config_v4.json"
PIN_DIR = "research/zeroshot/zs_lane_v4/pin"
PIN_REL = f"{PIN_DIR}/zs_lane_v4.pin.json"
MODEL_NAME = "zs_lane_v4_gbm.pkl"


def build(ledger: str = "") -> dict:
    cfg = json.loads(CONFIG.read_text())
    fp = FP.lane_fingerprint()
    if fp["sha256"] != cfg["code_fingerprint"]["sha256"]:
        raise SystemExit("code on disk differs from config_v4.json: refuse to pin")
    src = REPO / cfg["model"]["path"]
    if PIN.sha256_file(src) != cfg["model"]["sha256"]:
        raise SystemExit("model differs from config_v4.json")
    (REPO / PIN_DIR).mkdir(parents=True, exist_ok=True)
    dst = REPO / PIN_DIR / MODEL_NAME
    if not (dst.is_file() and PIN.sha256_file(dst) == cfg["model"]["sha256"]):
        shutil.copyfile(src, dst)
    assert PIN.sha256_file(dst) == cfg["model"]["sha256"]
    files = dict(fp["files"])
    files["research/zeroshot/zs_lane_v4/config_v4.json"] = PIN.sha256_file(CONFIG)
    return {
        "schema": PIN.PIN_SCHEMA,
        "lane": "zs_lane_v4",
        "variant": cfg["variant"],
        "code_fingerprint": {"files": files, "sha256": PIN.combined(files)},
        "lane_code_fingerprint": cfg["code_fingerprint"]["sha256"],
        "frozen_config": {"path": "research/zeroshot/zs_lane_v4/config_v4.json", "schema": cfg["schema"],
                          "sha256": PIN.sha256_file(CONFIG)},
        "model": {"rel_path": f"{PIN_DIR}/{MODEL_NAME}", "sha256": cfg["model"]["sha256"],
                  "frozen_source": cfg["model"]["path"], "class": cfg["model"]["class"],
                  "loader": "pickle (loaded only after the sha256 check)"},
        "lane_params": cfg["runtime"],
        "detection": cfg["detection"],
        "ledger": ledger,
        "status": "RESEARCH CANDIDATE, NOT LIVE: not listed in domains/production/zeroshot/pin.PIN_RELS; production "
                  "default stays zs_lane_v1 until the fresh-scan promotion test passes and the owner decides",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="")
    a = ap.parse_args(argv)
    pin = build(a.ledger)
    p = REPO / PIN_REL
    p.write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")
    bad = PIN.verify(REPO, pin)
    print(PIN_REL, PIN.sha256_file(p), "verify:", bad)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
