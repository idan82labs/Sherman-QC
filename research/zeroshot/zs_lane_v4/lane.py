"""zs_lane_v4 CLI (research candidate): scan -> eval_lines.json (sherman.broad_baseline.eval_lines.v1, raw scan frame,
outer-apex locus). Production stays on zs_lane_v1; this lane is not wired into any runner.

usage (repo root, one job at a time through the geometry slot):
  docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh .venv/bin/python -B -m research.zeroshot.zs_lane_v4.lane \\
      --scan <scan> --out <eval_lines.json> --config research/zeroshot/zs_lane_v4/config_v4.json [--strict-pins]
      [--record run_record.json]

The lane refuses (exit 4, nothing written) with --strict-pins when any pinned code file or the model differs from
the frozen config, or when a repo module outside the fingerprint was loaded. Blindness: the scan must be staged as
``scan.ply`` / ``scan.stl`` (the production workspace convention), so no original file name, serial or part number
reaches the lane; it reads only the scan bytes (no CAD, no labels). Which scans may be run is the driver's custody
decision. A failure that leaves no output writes an empty FAILED payload.
"""
from __future__ import annotations

import argparse
import json
import re
import pickle
import sys
import time
import traceback
from pathlib import Path

from research.zeroshot.zs_lane_v4 import core, fingerprint as FP, pipeline
from research.zeroshot.zs_lane_v4.core import REPO

SCAN_NAME_RE = re.compile(r"^scan\.(ply|stl)$")     # blindness: the driver stages the scan as scan.<ext>


def verify_pins(cfg: dict) -> list[str]:
    bad = []
    fp = FP.lane_fingerprint()
    pinned = cfg.get("code_fingerprint", {})
    if fp["sha256"] != pinned.get("sha256"):
        pf = pinned.get("files", {})
        bad += [f"CODE:{f}" for f in sorted(set(pf) | set(fp["files"])) if pf.get(f) != fp["files"].get(f)]
        if not bad:
            bad.append("CODE_FINGERPRINT")
    m = cfg["model"]
    if FP.sha256_file(REPO / m["path"]) != m["sha256"]:
        bad.append(f"MODEL:{m['path']}")
    return bad


def payload(path: str, sha: str, lines: list[dict], status: str, conv: dict) -> dict:
    import bb_common as bb
    regs = []
    for i, c in enumerate(lines):
        meta = {k: c[k] for k in ("score", "incl", "src", "op", "parts", "fix") if k in c}
        regs.append(bb.as_line(f"ZS{i + 1:03d}", c["start_s"], c["end_s"], **meta))
    failure = None if status == "OK" else {"error": conv.get("errors")}
    return bb.eval_payload(lane="zs_lane_v4", lines=regs, scan_path=Path(path), scan_sha256=sha,
                           source_kind="zs_lane_v4", source_path=None, conversion=conv, status=status, failure=failure)


def run(scan: str, out: str, cfg_path: str, record: str | None = None, strict_pins: bool = False) -> int:
    t0 = time.time()
    path = str(Path(scan))
    if not SCAN_NAME_RE.match(Path(path).name):
        print(json.dumps({"refused": "REFUSED_INPUT", "reason": "the scan must be staged as scan.<ply|stl>"}), flush=True)
        return 3
    sha = core.sha256_file(path)
    cfg = json.loads(Path(cfg_path).read_text())
    bad = verify_pins(cfg)
    if strict_pins:
        bad += [f"UNPINNED_LOADED:{f}" for f in FP.check_loaded()]
        if bad:
            print(json.dumps({"refused": "PINNED_HASH_MISMATCH", "mismatch": bad}), flush=True)
            return 4
    model = pickle.loads((REPO / cfg["model"]["path"]).read_bytes())
    status, lines, res, err = "OK", [], {}, None
    try:
        res = pipeline.run(path, cfg["runtime"], model)
        status = res["status"]
        lines = res["lines"] if status == "OK" else []
    except Exception as e:  # noqa: BLE001
        status, lines, err = "FAILED", [], f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    errors = {k: str(v)[:500] for k, v in (res.get("errors") or {}).items() if not k.endswith("_trace")}
    if err:
        errors["lane"] = err
    conv = {"rule": "zs_lane_v4", "config_sha256": FP.sha256_file(Path(cfg_path)),
            "code_fingerprint": cfg.get("code_fingerprint", {}).get("sha256"), "model_sha256": cfg["model"]["sha256"],
            "pins_ok": not bad, "pin_mismatch": bad, "errors": errors, "stages": res.get("stages")}
    pay = payload(path, sha, lines, status, conv)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pay, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    wall = round(time.time() - t0, 3)
    if record:
        import resource
        Path(record).write_text(json.dumps({"status": status, "n_lines": len(lines), "wall_s": wall,
                                            "lane_wall_s": res.get("wall_s"), "stages": res.get("stages"),
                                            "peak_rss_gib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30,
                                            "errors": errors, "pins_ok": not bad}, indent=1, default=str))
    print(f"zs_lane_v4 {status} lines={len(lines)} wall={wall:.1f}s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(REPO / "research/zeroshot/zs_lane_v4/config_v4.json"))
    ap.add_argument("--record", default="")
    ap.add_argument("--strict-pins", action="store_true")
    a = ap.parse_args(argv)
    return run(a.scan, a.out, a.config, a.record or None, a.strict_pins)


if __name__ == "__main__":
    sys.exit(main())
