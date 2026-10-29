"""Re-pin rehearsal (DEV, consumed): the production zs step and overlay merge with the v1 pin (default) and the v3
pin (selected non-default via SHERMAN_ZS_LANE_PIN=zs_lane_v3) on the DEV E3-eligible cohort of the CAD-path study:
stored e3-1.12.0 results + E1 compiled parts (/private/tmp/claude-501/sherman_zeroshot_work), the staged scan copied
byte-identical to <ws>/input/scan.<ext>.

Per scan: the gold guard with --cad-path (refuses the OD-3 DEV-CHECK cases before any file is read), then
  python -m domains.production.zeroshot.lane_cli   (one geometry slot, OMP 1)          -> zeroshot/zs_lines.json
  python -m domains.production.zeroshot.overlay_cli (stored result.json + compiled)    -> zeroshot_overlay_v1.json
for each pin. Checks: lane status OK and pins verified; the overlay validates; it never creates or upgrades a
pass-like state (FALSE PASS = an effective pass-like state that E3 did not have); runtime and RSS against the
runner's zs limits (90 s, 4 GiB). DEV is consumed: this is an engineering rehearsal, not an accuracy result.

usage (repo root): .venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/v3/tools/repin_rehearsal.py OUTDIR [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO)]
W = Path("/private/tmp/claude-501/sherman_zeroshot_work")
PROG = REPO / "docs/production/runlog/zeroshot_2026-10-25"
GUARD = PROG / "tools/guarded_slot.sh"
PY = str(REPO / ".venv/bin/python")
PASS_LIKE = {"ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE"}
LANES = ("zs_lane_v1", "zs_lane_v3")


def cohort() -> list[dict]:
    inv = {r["scan_id"]: r for r in json.loads((PROG / "INVENTORY.json").read_text())["scans"]}
    from research.zeroshot.zs_lane_v2 import scoring
    out = []
    for s in scoring.dev_ids():
        r = inv.get(s)
        sku = ((r or {}).get("cad") or {}).get("sku")
        comp = W / "compile" / str(sku) / "compiled/compiled_part.json"
        res = W / "e3" / s / "engine/result.json"
        if sku and comp.is_file() and res.is_file():
            out.append({"sid": s, "scan": r["path"], "compiled": str(comp), "result": str(res)})
    return out


def one(c: dict, out: Path) -> dict:
    from research.zeroshot.common.gold_guard import require_runnable, GoldGuardRefusal
    rec = {"sid": c["sid"]}
    try:
        require_runnable(c["sid"], cad_path=True)
    except GoldGuardRefusal as e:
        rec["refused"] = str(e.reason)
        return rec
    src = Path(c["scan"]).expanduser()
    ext = src.suffix.lower().lstrip(".")
    if ext not in ("ply", "stl"):
        rec["skipped"] = f"scan format {ext}"
        return rec
    e3 = json.loads(Path(c["result"]).read_text())
    rec["e3_state"], rec["e3_validity"] = e3["summary"]["state"], e3["inspection_validity"]["state"]
    for lane in LANES:
        ws = out / "ws" / c["sid"] / lane
        if ws.exists():
            shutil.rmtree(ws)
        (ws / "input").mkdir(parents=True)
        scan = ws / "input" / f"scan.{ext}"
        shutil.copyfile(src, scan)
        zs = ws / "zeroshot/zs_lines.json"
        env = {**os.environ, "SHERMAN_ZS_LANE_PIN": lane, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
        p = subprocess.run([str(GUARD), c["sid"], "--cad-path", "--", "/usr/bin/time", "-l", PY, "-B", "-m",
                            "domains.production.zeroshot.lane_cli", "--scan", str(scan), "--compiled", c["compiled"],
                            "--out", str(zs)], cwd=REPO, env=env, capture_output=True, text=True)
        r = {"rc": p.returncode}
        for ln in p.stderr.splitlines():
            if "maximum resident set size" in ln:
                r["rss_gib"] = round(int(ln.split()[0]) / 2 ** 30, 3)
        if zs.is_file():
            d = json.loads(zs.read_text())
            r.update(status=d.get("status"), lane=d.get("lane"), n_lines=len(d.get("lines") or []),
                     wall_s=d.get("wall_s"), n_selected=d.get("n_selected"), pin_mismatch=d.get("pin_mismatch"),
                     unpinned=d.get("unpinned_loaded"), errors=d.get("errors"),
                     fixes={k: sum(1 for x in d.get("lines") or [] if x.get("fix") == k) for k in ("EXT", "Z3R")})
        ov = ws / "zeroshot/zeroshot_overlay_v1.json"
        q = subprocess.run([PY, "-B", "-m", "domains.production.zeroshot.overlay_cli", "--result", c["result"],
                            "--compiled", c["compiled"], "--zs", str(zs), "--out", str(ov)], cwd=REPO, env=env,
                           capture_output=True, text=True)
        r["overlay_rc"] = q.returncode
        if ov.is_file():
            from domains.production.zeroshot.overlay import ZeroshotOverlayV1
            o = ZeroshotOverlayV1.model_validate_json(ov.read_bytes())
            r.update(effective=o.effective_summary_state, lowered=o.verdict_lowered,
                     kinds={k: sum(1 for x in o.lines if x.kind == k) for k in
                            ("WITHHELD_BEND", "UNEXPECTED_BEND", "REFUSED_SCAN_DETECTION")},
                     false_pass=bool(o.effective_summary_state in PASS_LIKE and o.effective_summary_state != o.e3_summary_state),
                     overlay_lane=o.lane.lane)
        else:
            r["overlay_tail"] = (q.stdout + q.stderr)[-300:]
        scan.unlink()
        rec[lane] = r
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cs = cohort()
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda c: one(c, out), cs))
    ran = [r for r in recs if all(l in r for l in LANES)]
    summ = {"cohort": len(cs), "ran": len(ran), "refused": [[r["sid"], r["refused"]] for r in recs if "refused" in r],
            "skipped": [[r["sid"], r["skipped"]] for r in recs if "skipped" in r]}
    for lane in LANES:
        rr = [r[lane] for r in ran]
        walls = sorted(float(x.get("wall_s") or 0) for x in rr)
        summ[lane] = {
            "status": {s: sum(1 for x in rr if x.get("status") == s) for s in sorted({str(x.get("status")) for x in rr})},
            "overlays": sum(1 for x in rr if x.get("overlay_rc") == 0),
            "false_pass": sum(1 for x in rr if x.get("false_pass")),
            "lowered_to_review": sum(1 for x in rr if x.get("lowered")),
            "lines": sum(int(x.get("n_lines") or 0) for x in rr),
            "kinds": {k: sum((x.get("kinds") or {}).get(k, 0) for x in rr) for k in
                      ("WITHHELD_BEND", "UNEXPECTED_BEND", "REFUSED_SCAN_DETECTION")},
            "fixes": {k: sum((x.get("fixes") or {}).get(k, 0) for x in rr) for k in ("EXT", "Z3R")},
            "wall_median_s": walls[len(walls) // 2] if walls else None, "wall_max_s": walls[-1] if walls else None,
            "rss_max_gib": max((float(x.get("rss_gib") or 0) for x in rr), default=None),
            "over_90s": sum(1 for w in walls if w > 90), "over_4gib": sum(1 for x in rr if (x.get("rss_gib") or 0) > 4)}
    summ["e3_pass_like_scans"] = sum(1 for r in ran if r.get("e3_state") in PASS_LIKE)
    summ["changed_effective_state_v1_vs_v3"] = [[r["sid"], r["e3_state"], r["zs_lane_v1"].get("effective"),
                                                 r["zs_lane_v3"].get("effective")] for r in ran
                                                if r["zs_lane_v1"].get("effective") != r["zs_lane_v3"].get("effective")]
    (out / "REHEARSAL.json").write_text(json.dumps({"summary": summ, "scans": recs}, indent=1))
    shutil.rmtree(out / "ws", ignore_errors=True)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
