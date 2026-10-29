"""Write config_v3_rs.json: the frozen zs_lane_v3 revision-2 config (config_v3.json, sha256 1c3f8cb7...; same models,
threshold, post-processing, fixes, Z3 / Z2 settings and face budget) plus the Rust pins: the exported GBM trees
(export_models.py), the release binary and Cargo.lock, and this lane's code fingerprint. Re-run after every rebuild.

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3_rs.freeze_config [--revision 3]
  --revision 3 writes config_v3r3_rs.json from the frozen revision 3 config (research/zeroshot/zs_lane_v3/
  config_v3.json, sha256 8e002f3e...; budget.cell_mm selects the revision-3 budget in the binary); the live
  zs_lane_v3 files must equal that config's code fingerprint (dc23a905...).
"""
from __future__ import annotations

import hashlib
import json
import sys

from research.zeroshot.zs_lane_v3_rs import fingerprint as FP
from research.zeroshot.zs_lane_v3_rs.rs import REPO

from research.zeroshot.zs_lane_v3_rs import rev2ref

REVS = {2: (rev2ref.REV2_CONFIG, "1c3f8cb7787185f33a61922dda8a2fced900c2e3011ea0105612f4fd5cc0fc53",
            REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json"),
        3: (REPO / "research/zeroshot/zs_lane_v3/config_v3.json",
            "8e002f3e9dac696fc6072d2d6a3abca6d964bd9704a25b54dd8f3b084b08b8b4",
            REPO / "research/zeroshot/zs_lane_v3_rs/config_v3r3_rs.json")}
# revision 3 after the production re-pin of 2026-10-26 (code fingerprint dc23a905... -> 49a16399...: only the four
# re-pinned production files changed; variants, budget, z3_params, z2_fallback are byte-equal to 8e002f3e...)
REV3_REPINNED = {"9b173ea82033561b4521c37ad6fcb7f243eda56bed88e38b18d01def4376fa87"}
MODELS = {"zs": "research/zeroshot/zs_lane_v3_rs/models/zs_lane_v3_zs_gbm.gbm.json",
          "z3": "research/zeroshot/zs_lane_v3_rs/models/zs_lane_v3_z3_gbm.gbm.json"}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    rev = int(argv[argv.index("--revision") + 1]) if "--revision" in argv else 2
    V3_CONFIG, V3_CONFIG_SHA, OUT = REVS[rev]
    raw = V3_CONFIG.read_bytes()
    if rev == 3 and hashlib.sha256(raw).hexdigest() in REV3_REPINNED:
        V3_CONFIG_SHA = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(raw).hexdigest() != V3_CONFIG_SHA:
        print(f"refusing: {V3_CONFIG} is not the frozen revision {rev} ({V3_CONFIG_SHA[:8]}...)", file=sys.stderr)
        return 2
    cfg = json.loads(raw)
    if rev == 3:
        bad = [f for f, h in cfg["code_fingerprint"]["files"].items() if FP.sha256_file(REPO / f) != h]
        if bad:
            print(f"refusing: live files differ from the revision 3 fingerprint: {bad}", file=sys.stderr)
            return 2
    fp = FP.lane_fingerprint(rev)
    if fp["binary_sha256"] == "MISSING":
        print("refusing: build the zs_rust release binary first", file=sys.stderr)
        return 2
    models = {}
    v = cfg["variants"]["zs"]
    for k, rel in MODELS.items():
        m = json.loads((REPO / rel).read_text())
        src_key = "model" if k == "zs" else "z3_model"
        if m["source_sha256"] != v[f"{src_key}_sha256"]:
            print(f"refusing: {rel} was not exported from the pinned {v[src_key]}", file=sys.stderr)
            return 2
        models[k] = {"path": rel, "sha256": FP.sha256_file(REPO / rel), "source": m["source"],
                     "source_sha256": m["source_sha256"], "format": m["format"]}
    out = dict(cfg)
    out.update(lane="zs_lane_v3_rs", schema="sherman.zeroshot.zs_lane_v3_rs.config.v1",
               base_v3_config_sha256=V3_CONFIG_SHA, base_v3_code_fingerprint=cfg["code_fingerprint"]["sha256"],
               base_v3_revision=cfg.get("revision") or rev,
               code_fingerprint={"files": fp["files"], "sha256": fp["sha256"]},
               rust={"binary": "research/zeroshot/zs_rust/target/release/zs_rust",
                     "binary_sha256": fp["binary_sha256"], "cargo_lock_sha256": fp["cargo_lock_sha256"],
                     "build": "cargo build --release --locked (research/zeroshot/zs_rust; target-cpu generic)",
                     "portable": True,
                     "math": "pure Rust: libm crate for every transcendental, explicit fma only, LAPACK 3.9 ports for "
                             "eigh / solve / det, Jacobi-SVD lstsq; no Accelerate, no CommonCrypto",
                     "targets": FP.target_binaries(),
                     "golden": {"dir": "research/zeroshot/zs_rust/golden",
                                "golden_json_sha256": FP.sha256_file(REPO / "research/zeroshot/zs_rust/golden/golden.json"),
                                "check": "zs_rust selftest --golden DIR --config CFG (exit 0 pass, 3 refuse)"},
                     "models": models},
               note=f"zs_lane_v3 revision {rev} as one portable Rust binary (zs_rust, the reference since 2026-10-26); models, thresholds, post-processing, fixes "
                    "and budget unchanged. Point clouds fall back to the Python revision 2 lane (unbudgeted: identical in revision 3). " + cfg.get("note", ""))
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{OUT}: code fingerprint {fp['sha256']} binary {fp['binary_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
