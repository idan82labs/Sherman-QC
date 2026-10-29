"""Pin records (PORT.md section 7 format, schema sherman.zs_lane_v3_rs.pin.v1) for the portable zs_rust binary as
the reference: one per revision, pins/zs_lane_v3_rs.rev{2,3}.pin.json. They record the binary sha256 per built
target, the Cargo.lock, the toolchain and profile, the exported models, the frozen adapter config, the golden
self-test set and the equivalence evidence. Nothing is installed anywhere: the production runner does not read
these files (integration is a later step). Re-run after every rebuild and freeze_config run.

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3_rs.make_pins [--results docs/.../v3_rust/portable_results.json]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from research.zeroshot.zs_lane_v3_rs import fingerprint as FP
from research.zeroshot.zs_lane_v3_rs.rs import REPO

ADAPTER = "research/zeroshot/zs_lane_v3_rs"
RESULTS = "docs/production/runlog/zeroshot_2026-10-25/v3_rust/portable_results.json"
CONFIGS = {2: f"{ADAPTER}/config_v3_rs.json", 3: f"{ADAPTER}/config_v3r3_rs.json"}
STATION = "x86_64-pc-windows-msvc+x86-64-v2"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    res_p = REPO / (argv[argv.index("--results") + 1] if "--results" in argv else RESULTS)
    results = json.loads(res_p.read_text()) if res_p.is_file() else {}
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.split()[1]
    targets = FP.target_binaries()
    golden = REPO / "research/zeroshot/zs_rust/golden"
    out_dir = REPO / ADAPTER / "pins"
    out_dir.mkdir(exist_ok=True)
    for rev, cfg_rel in CONFIGS.items():
        cfg = json.loads((REPO / cfg_rel).read_text())
        rust = cfg["rust"]
        if rust.get("binary_sha256") != targets["aarch64-apple-darwin"]["sha256"]:
            print(f"refusing: {cfg_rel} pins {rust.get('binary_sha256')}, built binary is "
                  f"{targets['aarch64-apple-darwin']['sha256']} (run freeze_config first)", file=sys.stderr)
            return 2
        statuses = (results.get("cross_arch") or {}).get("target_status", {})
        binaries = {t: {**v, "status": statuses.get(t, "built" if v["sha256"] else "not built here")}
                    for t, v in targets.items()}
        eq = (results.get("drift") or {}).get(f"rev{rev}", {})
        pin = {
            "schema": "sherman.zs_lane_v3_rs.pin.v1",
            "lane": "zs_lane_v3_rs", "revision": rev,
            "reference": "the portable zs_rust binary (since 2026-10-26); the frozen Python lane is a comparison",
            "base": {"lane": "zs_lane_v3", "code_fingerprint": cfg["base_v3_code_fingerprint"],
                     "config_sha256": cfg["base_v3_config_sha256"]},
            "binary": {"path": "bin/zs_rust.exe", "station_target": STATION,
                       "sha256_signed": None, "sha256_unsigned": targets[STATION]["sha256"],
                       "note": "the station build's sha256 is recorded by station/build_install.ps1 (not buildable on "
                               "the Mac without the MSVC linker / Windows SDK); every other target below is built",
                       "rustc": rustc, "cargo_lock_sha256": rust["cargo_lock_sha256"],
                       "profile": {"opt": 3, "lto": "fat", "codegen_units": 1, "panic": "abort",
                                   "target_cpu": {"aarch64-apple-darwin": "generic", "x86_64": "x86-64-v2 (station default; x86-64-v3 not cleared, PORTABLE.md)"}},
                       "features": [], "math": rust.get("math"),
                       "reference_build": {"target": "aarch64-apple-darwin", "sha256": targets["aarch64-apple-darwin"]["sha256"],
                                           "bytes": targets["aarch64-apple-darwin"]["bytes"]}},
            "binaries": binaries,
            "models": {k: {"path": m["path"], "sha256": m["sha256"], "format": m["format"],
                           "source_sha256": m["source_sha256"]} for k, m in rust["models"].items()},
            "config": {"path": cfg_rel, "sha256": FP.sha256_file(REPO / cfg_rel),
                       "code_fingerprint": cfg["code_fingerprint"]["sha256"]},
            "golden": {"dir": "research/zeroshot/zs_rust/golden",
                       "golden_json_sha256": FP.sha256_file(golden / "golden.json"),
                       "meshes": {p.name: FP.sha256_file(p) for p in sorted(golden.iterdir()) if p.suffix in (".ply", ".stl")},
                       "check": "zs_rust selftest --golden DIR --config CFG --repo ROOT --threads N: exit 0 pass, 3 refuse; "
                                "at install and at every start (station/build_install.ps1)"},
            "runtime": {"threads": "physical_cores (output identical for every thread count)",
                        "exit_codes": {"0": "ok", "2": "error", "3": "selftest refused", "4": "pin mismatch (adapter)",
                                       "10": "point cloud"},
                        "point_cloud": "refuse | python_fallback"},
            "equivalence": {"vs_python_accelerate": eq, "cross_arch": (results.get("cross_arch") or {}).get("summary"),
                            "evidence": RESULTS},
        }
        p = out_dir / f"zs_lane_v3_rs.rev{rev}.pin.json"
        p.write_text(json.dumps(pin, indent=1) + "\n")
        print(f"{p.relative_to(REPO)} {FP.sha256_file(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
