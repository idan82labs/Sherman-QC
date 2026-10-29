"""Export the Z3 DEV deliverables: per-scan eval_lines.json (out-of-fold selected lines) and
candidates_eval_lines.json (birth-filtered candidate stream with out-of-fold scores), a run record per scan,
and RESULTS.json. usage: python -m research.zeroshot.z3_slicing.export RUN_DIR OOF.json TAG"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, zscore, gate  # noqa: E402

CODE = ["slicer.py", "selector.py", "gate.py", "custody.py", "run_dev.py"]


def fingerprint() -> str:
    h = hashlib.sha256()
    for f in CODE:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def main():
    run, oof, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    inv = custody.dev_inventory()
    O = json.loads(oof.read_text())
    out = custody.OUT / "runs" / tag
    fp = fingerprint()
    for sid in custody.dev_ids():
        src = run / f"{sid}.json"
        if not src.is_file():
            continue
        d = json.loads(src.read_text())
        row = inv[sid]
        scored = O["scored"].get(sid, [])
        thr = O["thresholds"].get(sid, 0.5)
        sel = [c for c in scored if c.get("score", 0) >= thr]
        stream = gate.g_birth({"candidates": scored}, sid)
        conv = {"rule": "z3 axis-family slicing + out-of-fold GBM selector (DEV)", "code": fp,
                "threshold": thr, "t_mm": d.get("t_mm"), "run_dir": str(run)}
        dst = out / sid
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "eval_lines.json").write_text(json.dumps(zscore.to_payload(
            sel, lane="z3_axis_slicing", scan_path=d.get("scan_path"), scan_sha=None, extra=conv,
            status=d.get("status", "OK")), indent=1))
        (dst / "candidates_eval_lines.json").write_text(json.dumps(zscore.to_payload(
            stream, lane="z3_axis_slicing_candidates", scan_path=d.get("scan_path"), scan_sha=None,
            extra={**conv, "rule": "z3 birth stream (gate.birth), score = out-of-fold selector probability"},
            status=d.get("status", "OK")), indent=1))
        (dst / "run_record.json").write_text(json.dumps({
            "scan_id": sid, "family": row["family"], "code_fingerprint": fp, "status": d.get("status"),
            "timing_s": d.get("timing_s"), "peak_rss_gib_process": d.get("peak_rss_gib"),
            "mesh": d.get("mesh"), "t_mm": d.get("t_mm"), "n_families": len(d.get("families", [])),
            "n_candidates_raw": len(d.get("candidates", [])), "n_stream": len(stream), "n_selected": len(sel)}, indent=1))
    print(f"wrote {out} (code {fp})")


if __name__ == "__main__":
    main()
