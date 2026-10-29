"""zs_lane_v2 CLI: scan -> eval_lines.json (sherman.broad_baseline.eval_lines.v1, raw scan frame, outer-apex locus).

usage (repo root, through the harness guard + slot):
  docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh <scan> -- \\
    .venv/bin/python -B -m research.zeroshot.zs_lane_v2.lane --scan <scan path> --out <eval_lines.json> \\
      --config research/zeroshot/zs_lane_v2/config_v2.json --variant zs [--record run_record.json]
  DEV dump for out-of-fold training (no model): --dump <dump.json> instead of --config/--variant.
  Post-hoc re-score of the CONSUMED TEST-B (never a held-out claim): --posthoc-consumed-testb. The gold guard
  seals TEST-B unless its one-time flag is given; TEST-B was consumed by zs_lane_v1 (testb/RESULTS.md), so this
  mode passes that flag only for TEST-B scans, only after checking the consumption record exists, and stamps every
  output POSTHOC_CONSUMED.

Variant zs (config_v2.json): Z3 (vendored z3s) + Z2 pooled candidates, v2 evidence, GBM selector at the fixed
threshold, span post-processing (postproc.py), D1-snapped placement. Every pooled candidate is snapped before
selection (as in the DEV dump), so the lane and the DEV evaluation run identical arithmetic.
The lane refuses to run when the gold guard refuses the scan, when the path fails custody, or (with --strict-pins)
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

from research.zeroshot.zs_lane_v2 import core
from research.zeroshot.zs_lane_v2.core import REPO

Z3_FEATS = 21    # z3_slicing.selector.FEATS_BASE; the pooled matrix appends 5 pool flags
PROG = REPO / "docs/production/runlog/zeroshot_2026-10-25"


def _guard(scan: str, posthoc_testb: bool) -> dict:
    from research.zeroshot.common import gold_guard
    if not posthoc_testb:
        return gold_guard.require_runnable(scan, final_testb=False)
    sid = gold_guard.identify(scan)
    split = json.loads((PROG / "SPLIT_FROZEN.json").read_text())
    if sid not in set(split["test_b"]["scans"]):
        raise PermissionError(f"{sid}: --posthoc-consumed-testb is only for TEST-B scans")
    if not (PROG / "testb/RESULTS.json").is_file():
        raise PermissionError("TEST-B consumption record testb/RESULTS.json missing: TEST-B is not consumed")
    d = gold_guard.require_runnable(scan, final_testb=True)
    d["basis"] = "POSTHOC_CONSUMED_TESTB (zs_lane_v1 one-time run consumed TEST-B; not a held-out result)"
    return d


def _custody(scan: str) -> Path:
    from research.zeroshot.z3_slicing.custody import check_path
    return check_path(scan)


def verify_pins(cfg: dict, variant: str) -> list[str]:
    """Mismatches between the files on disk and the hashes pinned in the config (empty = all equal)."""
    from research.zeroshot.zs_lane_v2 import fingerprint as FP
    bad = []
    fp = FP.lane_fingerprint()
    pinned_fp = cfg.get("code_fingerprint", {})
    if fp["sha256"] != pinned_fp.get("sha256"):
        pinned = pinned_fp.get("files", {})
        bad += [f"CODE:{f}" for f in sorted(set(pinned) | set(fp["files"])) if pinned.get(f) != fp["files"].get(f)]
    v = cfg["variants"][variant]
    if v.get("model") and FP.sha256_file(REPO / v["model"]) != v.get("model_sha256"):
        bad.append(f"MODEL:{v['model']}")
    return bad


def select_lines(rec: dict, v: dict, model: dict | None) -> list[dict]:
    from research.zeroshot.zs_lane_v2.postproc import PostParams, postprocess, features, sweep_matrix
    C = rec["candidates"]
    if not C or model is None:
        return []
    SW = sweep_matrix((rec.get("evidence") or {}).get("sweep"), len(C))
    X = features(rec["X"], SW, model.get("feats", "base"))
    prob = model["model"].predict(X)
    prm = PostParams(**(v.get("post") or {}))
    out = postprocess(C, prob, float(v["threshold"]), rec.get("evidence"), prm)
    for o in out:
        o.pop("_i", None)
    return out


def run(scan: str, out: str, *, cfg_path: str | None, variant: str | None, dump: str | None, record: str | None,
        posthoc_testb: bool = False, strict_pins: bool = False) -> int:
    t0 = time.time()
    decision = _guard(scan, posthoc_testb)
    path = str(_custody(scan))
    sha = core.sha256_file(path)
    if dump:
        rec = core.process(path, use_z2=True, snap_all=True)
        X = rec.pop("X")
        rec.pop("_snap_pts", None)
        rec["X"] = [[None if x != x else float(x) for x in row] for row in X.tolist()]
        rec["scan_id"], rec["scan_path"], rec["scan_sha256"] = decision["scan_id"], path, sha
        rec["z3_params"], rec["z2_fallback"] = core.Z3_PARAMS, core.Z2_FALLBACK
        for c in rec["candidates"]:
            c.pop("_src", None)
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_text(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"{decision['scan_id']} {rec['status']} cands={len(rec['candidates'])} z2={len(rec['z2_lines'])} "
              f"wall={rec['wall_s']:.1f}s rss={rec['peak_rss_gib_process']:.2f}G", flush=True)
        return 0
    cfg = json.loads(Path(cfg_path).read_text())
    v = cfg["variants"][variant]
    core.Z3_PARAMS = cfg.get("z3_params") or None
    core.Z2_FALLBACK = bool(cfg.get("z2_fallback", False))
    core.EVIDENCE_SPLITS = bool((v.get("post") or {}).get("split", False))
    bad = verify_pins(cfg, variant)
    if strict_pins:
        from research.zeroshot.zs_lane_v2 import fingerprint as FP
        bad += [f"UNPINNED_LOADED:{f}" for f in FP.check_loaded()]
        if bad:
            print(json.dumps({"refused": "PINNED_HASH_MISMATCH", "mismatch": bad}), flush=True)
            return 4
    model = pickle.loads((REPO / v["model"]).read_bytes()) if v.get("model") else None
    status, lines, rec, err = "OK", [], {"stages": {}, "errors": {}}, None
    try:
        rec = core.process(path, use_z2=bool(v["use_z2"]), use_z3=bool(v["use_z3"]), snap_all=True)
        t1 = time.time()
        lines = select_lines(rec, v, model)
        rec["stages"]["select_post"] = {"wall": round(time.time() - t1, 3), "n": len(lines),
                                        "ops": {k: sum(1 for x in lines if x["op"] == k) for k in ("keep", "split", "join")}}
        if rec.get("status") != "OK":
            status = "FAILED"
            lines = []
    except Exception as e:  # noqa: BLE001
        status, lines, err = "FAILED", [], f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    conv = {"rule": f"zs_lane_v2 variant {variant}", "variant": variant, "threshold": v.get("threshold"),
            "model": v.get("model"), "model_sha256": v.get("model_sha256"), "post": v.get("post"),
            "z3_params": cfg.get("z3_params"), "z2_fallback": cfg.get("z2_fallback"),
            "code_fingerprint": cfg.get("code_fingerprint", {}).get("sha256"), "pins_ok": not bad,
            "pin_mismatch": bad,
            "errors": {**{k: x for k, x in rec.get("errors", {}).items() if not k.endswith("_trace")},
                       **({"lane": err} if err else {})},
            "stages": rec.get("stages"), "posthoc_consumed_testb": posthoc_testb,
            "guard_basis": decision.get("basis")}
    pay = core.payload(path, sha, lines, status=status, conversion=conv, lane=f"zs_lane_v2:{variant}")
    if posthoc_testb:
        pay["ledger_note"] = "POSTHOC_CONSUMED: TEST-B was consumed by zs_lane_v1; this is not a held-out result"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pay, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    wall = round(time.time() - t0, 3)
    if record:
        Path(record).write_text(json.dumps({"scan_id": decision["scan_id"], "variant": variant, "status": status,
                                            "n_lines": len(lines), "wall_s": wall,
                                            "lane_wall_s": rec.get("wall_s"),
                                            "peak_rss_gib_process": core._rss_gib(), "stages": rec.get("stages"),
                                            "errors": conv["errors"], "pins_ok": not bad,
                                            "posthoc_consumed_testb": posthoc_testb,
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
    ap.add_argument("--posthoc-consumed-testb", action="store_true")
    ap.add_argument("--strict-pins", action="store_true")
    a = ap.parse_args(argv)
    if not a.dump and not (a.out and a.config):
        ap.error("--out and --config are required unless --dump")
    return run(a.scan, a.out, cfg_path=a.config or None, variant=a.variant, dump=a.dump or None,
               record=a.record or None, posthoc_testb=a.posthoc_consumed_testb, strict_pins=a.strict_pins)


if __name__ == "__main__":
    sys.exit(main())
