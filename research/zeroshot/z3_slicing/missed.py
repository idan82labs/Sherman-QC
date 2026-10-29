"""List gold lines that no Z3 candidate matches (candidate-level misses), with the nearest candidate's metrics
and the gold note's first words (labeler description). usage: python -m research.zeroshot.z3_slicing.missed RUN_DIR"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, zscore  # noqa: E402
from research.zeroshot.z3_slicing.score_run import load_run  # noqa: E402


def main():
    run = Path(sys.argv[1])
    inv = custody.dev_inventory()
    data = load_run(run, custody.dev_ids())
    causes = Counter()
    for sid, d in sorted(data.items()):
        rec = zscore.score_lines(sid, inv[sid], d.get("candidates", []))
        if rec.get("disposition") not in ("SCORED_COMPLETE_LABEL",):
            continue
        g = {b["bend_id"]: b for b in json.loads(Path(inv[sid]["gold"]["path"]).read_text())["bends"]}
        for f in rec.get("fn_rows", []):
            causes[f["cause"]] += 1
            b = g.get(f["gold_id"], {})
            L = np.linalg.norm(np.subtract(b.get("line_end", [0, 0, 0]), b.get("line_start", [0, 0, 0])))
            note = (b.get("notes") or "").split("labeler:")[-1].strip()[:110]
            print(f"{sid} {inv[sid]['family']:>10s} {f['gold_id']:>4s} L={L:6.1f} {f['cause']:<28s} "
                  f"ang={f.get('axis_angle_deg', '-')} off={f.get('transverse_offset_mm', '-')} "
                  f"ovg={f.get('gold_span_overlap', '-')} ovp={f.get('prediction_span_overlap', '-')} | {note}")
    print(causes)


if __name__ == "__main__":
    main()
