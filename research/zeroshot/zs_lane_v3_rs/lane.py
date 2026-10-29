"""zs_lane_v3_rs CLI: scan -> eval_lines.json through the zs_rust binary (zs_lane_v3 revision 2 in Rust).

usage (repo root, through the harness guard + slot):
  docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh <scan> -- \\
    [ZS_RUST_THREADS=n] .venv/bin/python -B -m research.zeroshot.zs_lane_v3_rs.lane --scan <scan path> \\
      --out <eval_lines.json> --config research/zeroshot/zs_lane_v3_rs/config_v3_rs.json --variant zs \\
      [--record run_record.json] [--strict-pins]
  DEV dump for the stage comparison / out-of-fold evaluation: --dump <dump.json> instead of --out/--config.

The adapter refuses to run when the gold guard refuses the scan or the path fails custody. With --strict-pins it
also refuses (exit 4, nothing written) when a pinned file (binary, Cargo.lock, Rust sources, exported models,
adapter, the revision-2 Python files of the point-cloud fallback) differs from config_v3_rs.json. A scan without
faces (point cloud) runs the frozen Python revision 2 lane instead; the record says so (rust_fallback).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from research.zeroshot.zs_lane_v3_rs import core, rs
from research.zeroshot.zs_lane_v3_rs.core import REPO

from research.zeroshot.zs_lane_v3_rs import rev2ref

V3_CONFIG = rev2ref.REV2_CONFIG  # the frozen revision 2 config (the live zs_lane_v3 now holds revision 3)


def _guard(scan: str) -> dict:
    from research.zeroshot.common import gold_guard
    return gold_guard.require_runnable(scan, final_testb=False)


def _custody(scan: str) -> Path:
    from research.zeroshot.z3_slicing.custody import check_path
    return check_path(scan)


def verify_pins(cfg: dict) -> list[str]:
    from research.zeroshot.zs_lane_v3_rs import fingerprint as FP
    fp = FP.lane_fingerprint(int(cfg.get("base_v3_revision") or 2))
    pinned = cfg.get("code_fingerprint", {})
    if fp["sha256"] == pinned.get("sha256"):
        return []
    files = pinned.get("files", {})
    bad = [f"CODE:{f}" for f in sorted(set(files) | set(fp["files"])) if files.get(f) != fp["files"].get(f)]
    return bad or ["CODE_FINGERPRINT"]


def _nan_fields(d: dict) -> dict:
    """JSON has no NaN: the binary writes null where Python's evidence holds NaN (flank lengths of Z2-only lines,
    empty gap covers). Restore NaN so the dump equals the Python dump value for value."""
    nan = float("nan")
    ev = d.get("evidence") or {}
    for p in ev.get("pairs") or []:
        for k in ("flank_i", "flank_j"):
            if k in p:
                p[k] = [nan if x is None else x for x in p[k]]
        for k in ("cov_a", "cov_b"):
            if k in p and p[k] is None:
                p[k] = nan
    for runs in (ev.get("splits") or {}).values():
        for r in runs:
            for k in ("cov_a", "cov_b", "raw_a", "raw_b"):
                if k in r and r[k] is None:
                    r[k] = nan
    return d


def _fallback(path: str, out: str | None, record: str | None, dump: str | None, reason: str) -> int:
    """Point cloud: the frozen Python revision 2 lane (byte copies in rev2_snapshot/, see rev2ref)."""
    rev2ref.install()
    from research.zeroshot.zs_lane_v3 import lane as L3, core as C3
    from research.zeroshot.zs_lane_v3 import budget
    if dump:
        from research.zeroshot.zs_lane_v2 import core as V2
        v2cfg = json.loads((REPO / "research/zeroshot/zs_lane_v2/config_v2.json").read_text())
        V2.Z3_PARAMS = v2cfg.get("z3_params") or None
        V2.Z2_FALLBACK = bool(v2cfg.get("z2_fallback", False))
        C3.BUDGET_MAX_FACES = budget.MAX_FACES
        rc = L3.run(path, "", cfg_path=None, variant=None, dump=dump, record=None)
    else:
        rc = L3.run(path, out, cfg_path=str(V3_CONFIG), variant="zs", dump=None, record=record)
    if record and Path(record).is_file():
        r = json.loads(Path(record).read_text())
        r["rust_fallback"] = reason
        Path(record).write_text(json.dumps(r, indent=1))
    return rc


def run(scan: str, out: str, *, cfg_path: str | None, variant: str | None, dump: str | None, record: str | None,
        strict_pins: bool = False) -> int:
    t0 = time.time()
    decision = _guard(scan)
    path = str(_custody(scan))
    if dump:
        try:
            core.run_binary(path, str(cfg_path or REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json"),
                            dump=dump, splits=True)
            d = _nan_fields(json.loads(Path(dump).read_text()))
            d["scan_id"], d["scan_path"] = decision["scan_id"], path
            Path(dump).write_text(json.dumps(d))
        except core.PointCloudScan as e:
            return _fallback(path, None, None, dump, f"POINT_CLOUD: {e}")
        print(f"{decision['scan_id']} dump written", flush=True)
        return 0
    cfg = json.loads(Path(cfg_path).read_text())
    v = cfg["variants"][variant]
    bad = verify_pins(cfg)
    if strict_pins and bad:
        print(json.dumps({"refused": "PINNED_HASH_MISMATCH", "mismatch": bad}), flush=True)
        return 4
    conv = {"rule": f"zs_lane_v3_rs variant {variant} (zs_lane_v3 revision 2 in Rust)", "variant": variant,
            "threshold": v.get("threshold"), "model": v.get("model"), "model_sha256": v.get("model_sha256"),
            "z3_model": v.get("z3_model"), "z3_model_sha256": v.get("z3_model_sha256"), "post": v.get("post"),
            "fix": v.get("fix"), "z3_params": cfg.get("z3_params"), "z2_fallback": cfg.get("z2_fallback"),
            "budget": cfg.get("budget"), "code_fingerprint": cfg.get("code_fingerprint", {}).get("sha256"),
            "base_v3_code_fingerprint": cfg.get("base_v3_code_fingerprint"), "pins_ok": not bad, "pin_mismatch": bad,
            "rust": {**(cfg.get("rust") or {}), "threads": rs.threads()}, "guard_basis": decision.get("basis")}
    try:
        res = core.run_binary(path, cfg_path, out=out, record=record, conversion=conv, lane=f"zs_lane_v3_rs:{variant}")
    except core.PointCloudScan as e:
        return _fallback(path, out, record, None, f"POINT_CLOUD: {e}")
    wall = round(time.time() - t0, 3)
    if record:
        r = json.loads(Path(record).read_text())
        r.update(scan_id=decision["scan_id"], variant=variant, wall_s=wall, pins_ok=not bad,
                 code_fingerprint=conv["code_fingerprint"], rust_threads=rs.threads())
        Path(record).write_text(json.dumps(r, indent=1))
    print(f"{decision['scan_id']} {variant} {res['status']} lines={res['n_lines']} wall={wall:.1f}s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--variant", default="zs", choices=("zs",))
    ap.add_argument("--dump", default="")
    ap.add_argument("--record", default="")
    ap.add_argument("--strict-pins", action="store_true")
    a = ap.parse_args(argv)
    if not a.dump and not (a.out and a.config):
        ap.error("--out and --config are required unless --dump")
    return run(a.scan, a.out, cfg_path=a.config or None, variant=a.variant, dump=a.dump or None,
               record=a.record or None, strict_pins=a.strict_pins)


if __name__ == "__main__":
    sys.exit(main())
