"""zs_lane_v3 CLI: scan -> eval_lines.json (sherman.broad_baseline.eval_lines.v1, raw scan frame, outer-apex locus).

usage (repo root, through the harness guard + slot):
  docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh <scan> -- \\
    .venv/bin/python -B -m research.zeroshot.zs_lane_v3.lane --scan <scan path> --out <eval_lines.json> \\
      --config research/zeroshot/zs_lane_v3/config_v3.json --variant zs [--record run_record.json] [--strict-pins]
  DEV dump for out-of-fold training (no model): --dump <dump.json> instead of --config/--variant.

Variant zs (config_v3.json): zs_lane_v2's pipeline and frozen post-processing, then EXT and Z3R (fixes.py).
The lane refuses to run when the gold guard refuses the scan or the path fails custody. With --strict-pins it
also refuses (exit 4, nothing written) when any pinned code file, model or the config's code fingerprint differs,
or when a repo module outside the fingerprint was loaded. A stage failure that leaves no output writes an empty
payload with status FAILED (scored as all false negatives). There is no TEST-B mode: TEST-B is consumed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

from research.zeroshot.zs_lane_v3 import core
from research.zeroshot.zs_lane_v3.core import REPO


def _guard(scan: str) -> dict:
    from research.zeroshot.common import gold_guard
    return gold_guard.require_runnable(scan, final_testb=False)


def _custody(scan: str) -> Path:
    from research.zeroshot.z3_slicing.custody import check_path
    return check_path(scan)


def verify_pins(cfg: dict, variant: str) -> list[str]:
    """Mismatches between the files on disk and the hashes pinned in the config (empty = all equal)."""
    from research.zeroshot.zs_lane_v3 import fingerprint as FP
    bad = []
    fp = FP.lane_fingerprint()
    pinned_fp = cfg.get("code_fingerprint", {})
    if fp["sha256"] != pinned_fp.get("sha256"):
        pinned = pinned_fp.get("files", {})
        bad += [f"CODE:{f}" for f in sorted(set(pinned) | set(fp["files"])) if pinned.get(f) != fp["files"].get(f)]
        if not bad:
            bad.append("CODE_FINGERPRINT")
    v = cfg["variants"][variant]
    for key in ("model", "z3_model"):
        if v.get(key) and FP.sha256_file(REPO / v[key]) != v.get(f"{key}_sha256"):
            bad.append(f"MODEL:{v[key]}")
    return bad


def load_models(v: dict) -> tuple[dict | None, dict | None]:
    zs = pickle.loads((REPO / v["model"]).read_bytes()) if v.get("model") else None
    z3 = pickle.loads((REPO / v["z3_model"]).read_bytes()) if v.get("z3_model") else None
    return zs, z3


def _dump(path: str, decision: dict, sha: str, dump: str) -> int:
    rec = core.process(path)
    X = rec.pop("X")
    rec.pop("_snap_pts", None)
    rec["X"] = [[None if x != x else float(x) for x in row] for row in X.tolist()]
    rec["scan_id"], rec["scan_path"], rec["scan_sha256"] = decision["scan_id"], path, sha
    from research.zeroshot.zs_lane_v2 import core as V2
    rec["z3_params"], rec["z2_fallback"] = V2.Z3_PARAMS, V2.Z2_FALLBACK
    for c in rec["candidates"]:
        c.pop("_src", None)
    Path(dump).parent.mkdir(parents=True, exist_ok=True)
    Path(dump).write_text(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{decision['scan_id']} {rec['status']} cands={len(rec['candidates'])} z2={len(rec['z2_lines'])} "
          f"wall={rec['wall_s']:.1f}s rss={rec['peak_rss_gib_process']:.2f}G", flush=True)
    return 0


def run(scan: str, out: str, *, cfg_path: str | None, variant: str | None, dump: str | None, record: str | None,
        strict_pins: bool = False) -> int:
    t0 = time.time()
    decision = _guard(scan)
    path = str(_custody(scan))
    sha = core.sha256_file(path)
    if dump:
        return _dump(path, decision, sha, dump)
    cfg = json.loads(Path(cfg_path).read_text())
    v = cfg["variants"][variant]
    core.configure(cfg, variant)
    bad = verify_pins(cfg, variant)
    if strict_pins:
        from research.zeroshot.zs_lane_v3 import fingerprint as FP
        bad += [f"UNPINNED_LOADED:{f}" for f in FP.check_loaded()]
        if bad:
            print(json.dumps({"refused": "PINNED_HASH_MISMATCH", "mismatch": bad}), flush=True)
            return 4
    zs_model, z3_model = load_models(v)
    status, lines, rec, err = "OK", [], {"stages": {}, "errors": {}}, None
    try:
        rec = core.process(path)
        t1 = time.time()
        lines = core.select_lines(rec, v, zs_model, z3_model)
        ops = {k: sum(1 for x in lines if x["op"] == k) for k in ("keep", "split", "join")}
        fixes = {k: sum(1 for x in lines if x.get("fix") == k) for k in ("EXT", "Z3R")}
        rec["stages"]["select_post"] = {"wall": round(time.time() - t1, 3), "n": len(lines), "ops": ops,
                                        "fixes": fixes}
        if rec.get("status") != "OK":
            status = "FAILED"
            lines = []
    except Exception as e:  # noqa: BLE001
        status, lines, err = "FAILED", [], f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    conv = {"rule": f"zs_lane_v3 variant {variant}", "variant": variant, "threshold": v.get("threshold"),
            "model": v.get("model"), "model_sha256": v.get("model_sha256"), "z3_model": v.get("z3_model"),
            "z3_model_sha256": v.get("z3_model_sha256"), "post": v.get("post"), "fix": v.get("fix"),
            "z3_params": cfg.get("z3_params"), "z2_fallback": cfg.get("z2_fallback"),
            "code_fingerprint": cfg.get("code_fingerprint", {}).get("sha256"), "pins_ok": not bad,
            "pin_mismatch": bad,
            "errors": {**{k: x for k, x in rec.get("errors", {}).items() if not k.endswith("_trace")},
                       **({"lane": err} if err else {})},
            "stages": rec.get("stages"), "guard_basis": decision.get("basis")}
    pay = core.payload(path, sha, lines, status=status, conversion=conv, lane=f"zs_lane_v3:{variant}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pay, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    wall = round(time.time() - t0, 3)
    if record:
        Path(record).write_text(json.dumps({"scan_id": decision["scan_id"], "variant": variant, "status": status,
                                            "n_lines": len(lines), "wall_s": wall,
                                            "lane_wall_s": rec.get("wall_s"),
                                            "peak_rss_gib_process": core.V2._rss_gib(), "stages": rec.get("stages"),
                                            "errors": conv["errors"], "pins_ok": not bad,
                                            "code_fingerprint": conv["code_fingerprint"]}, indent=1))
    print(f"{decision['scan_id']} {variant} {status} lines={len(lines)} wall={wall:.1f}s", flush=True)
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
