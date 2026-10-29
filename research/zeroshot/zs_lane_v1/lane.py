"""zs_lane_v1 CLI: scan -> eval_lines.json (sherman.broad_baseline.eval_lines.v1, raw scan frame, outer-apex locus).

usage (repo root, through the harness guard + slot):
  docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh <scan> [--final-testb] -- \\
    .venv/bin/python -B -m research.zeroshot.zs_lane_v1.lane --scan <scan path> --out <eval_lines.json> \\
      --config research/zeroshot/zs_lane_v1/config_v1.json --variant zs [--final-testb] [--record run_record.json]
  DEV dump for out-of-fold training (no model): --dump <dump.json> instead of --config/--variant.

Variants (config_v1.json): zs = Z3 + Z2 pooled GBM (primary candidate), z3 = Z3 alone GBM, z2 = Z2 alone.
The lane refuses to run when the gold guard refuses the scan, when the path fails custody, or (with --final-testb)
when any code / model / config hash differs from the pinned values. Any stage failure that leaves no output writes
an empty payload with status FAILED (scored as all false negatives).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

from research.zeroshot.zs_lane_v1 import core
from research.zeroshot.zs_lane_v1.core import REPO

Z3_FEATS = 21    # z3_slicing.selector.FEATS_BASE; the pooled matrix appends 5 pool flags


def _guard(scan: str, final_testb: bool) -> dict:
    from research.zeroshot.common import gold_guard
    return gold_guard.require_runnable(scan, final_testb=final_testb)


def _custody(scan: str) -> Path:
    from research.zeroshot.z3_slicing.custody import check_path
    return check_path(scan)


def verify_pins(cfg: dict, variant: str) -> list[str]:
    """Mismatches between the files on disk and the hashes pinned in the config (empty = all equal)."""
    from research.zeroshot.zs_lane_v1 import fingerprint as FP
    bad = []
    fp = FP.lane_fingerprint()
    if fp["sha256"] != cfg["code_fingerprint"]["sha256"]:
        pinned = cfg["code_fingerprint"]["files"]
        bad += [f"CODE:{f}" for f in sorted(set(pinned) | set(fp["files"])) if pinned.get(f) != fp["files"].get(f)]
    v = cfg["variants"][variant]
    if v.get("model"):
        if FP.sha256_file(REPO / v["model"]) != v["model_sha256"]:
            bad.append(f"MODEL:{v['model']}")
    return bad


def select_lines(rec: dict, variant: str, v: dict, model: dict | None) -> list[dict]:
    if variant == "z2":
        return [{"start": l["start"], "end": l["end"], "incl": l["incl"], "src": ["z2"]} for l in rec["z2_lines"]]
    C, X = rec["candidates"], rec["X"]
    if variant == "z3":
        keep = [i for i, c in enumerate(C) if c["has_z3"]]
        C, X = [C[i] for i in keep], (X[keep][:, :Z3_FEATS] if len(keep) else X[:0, :Z3_FEATS])
    if not C:
        return []
    prob = model["model"].predict(X)
    thr = float(v["threshold"])
    sel = []
    for c, q in zip(C, prob):
        if q >= thr:
            c = dict(c)
            c["score"] = float(q)
            sel.append(c)
    return sel


def run(scan: str, out: str, *, cfg_path: str | None, variant: str | None, final_testb: bool,
        dump: str | None, record: str | None) -> int:
    t0 = time.time()
    decision = _guard(scan, final_testb)
    path = str(_custody(scan))
    sha = core.sha256_file(path)
    if dump:
        rec = core.process(path, use_z2=True, snap_all=True)
        X = rec.pop("X")
        rec.pop("_snap_pts", None)
        rec["X"] = [[None if x != x else float(x) for x in row] for row in X.tolist()]
        rec["scan_id"], rec["scan_path"], rec["scan_sha256"] = decision["scan_id"], path, sha
        for c in rec["candidates"]:
            c.pop("_src", None)
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_text(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"{decision['scan_id']} {rec['status']} cands={len(rec['candidates'])} z2={len(rec['z2_lines'])} "
              f"wall={rec['wall_s']:.1f}s rss={rec['peak_rss_gib_process']:.2f}G", flush=True)
        return 0
    cfg = json.loads(Path(cfg_path).read_text())
    v = cfg["variants"][variant]
    bad = verify_pins(cfg, variant)
    if final_testb:
        from research.zeroshot.zs_lane_v1 import fingerprint as FP
        bad += [f"UNPINNED_LOADED:{f}" for f in FP.check_loaded()]
    if bad and final_testb:
        print(json.dumps({"refused": "PINNED_HASH_MISMATCH", "mismatch": bad}), flush=True)
        return 4
    model = pickle.loads((REPO / v["model"]).read_bytes()) if v.get("model") else None
    status, lines, rec, err = "OK", [], {"stages": {}, "errors": {}}, None
    try:
        rec = core.process(path, use_z2=bool(v["use_z2"]), use_z3=bool(v["use_z3"]), snap_all=False)
        lines = select_lines(rec, variant, v, model)
        if variant != "z2":         # Z2 lines are already D1-snapped inside Z2 (it08 config)
            t1 = time.time()
            core.snap(lines, rec.pop("_snap_pts", None))
            rec["stages"]["snap"] = {"wall": round(time.time() - t1, 3), "n": len(lines),
                                     "moved": sum(1 for c in lines if c["snap"].get("moved"))}
        if rec.get("status") != "OK":
            status = "FAILED"
            lines = []
    except Exception as e:  # noqa: BLE001
        status, lines, err = "FAILED", [], f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    conv = {"rule": f"zs_lane_v1 variant {variant}", "variant": variant, "threshold": v.get("threshold"),
            "model": v.get("model"), "model_sha256": v.get("model_sha256"),
            "code_fingerprint": cfg["code_fingerprint"]["sha256"], "pins_ok": not bad, "pin_mismatch": bad,
            "errors": {**{k: x for k, x in rec.get("errors", {}).items() if not k.endswith("_trace")},
                       **({"lane": err} if err else {})},
            "stages": rec.get("stages"), "final_testb": final_testb, "guard_basis": decision.get("basis")}
    pay = core.payload(path, sha, lines, status=status, conversion=conv, lane=f"zs_lane_v1:{variant}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pay, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    wall = round(time.time() - t0, 3)
    if record:
        Path(record).write_text(json.dumps({"scan_id": decision["scan_id"], "variant": variant, "status": status,
                                            "n_lines": len(lines), "wall_s": wall,
                                            "lane_wall_s": rec.get("wall_s"),
                                            "peak_rss_gib_process": core._rss_gib(), "stages": rec.get("stages"),
                                            "errors": conv["errors"], "pins_ok": not bad,
                                            "code_fingerprint": cfg["code_fingerprint"]["sha256"]}, indent=1))
    print(f"{decision['scan_id']} {variant} {status} lines={len(lines)} wall={wall:.1f}s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--variant", default="zs", choices=("zs", "z3", "z2"))
    ap.add_argument("--final-testb", action="store_true")
    ap.add_argument("--dump", default="")
    ap.add_argument("--record", default="")
    a = ap.parse_args(argv)
    if not a.dump and not (a.out and a.config):
        ap.error("--out and --config are required unless --dump")
    return run(a.scan, a.out, cfg_path=a.config or None, variant=a.variant, final_testb=a.final_testb,
               dump=a.dump or None, record=a.record or None)


if __name__ == "__main__":
    sys.exit(main())
