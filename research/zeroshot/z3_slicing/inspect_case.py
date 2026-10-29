"""Debug view of one scan: gold lines vs Z3 candidates (best match metrics per gold line), family axes.
usage: python -m research.zeroshot.z3_slicing.inspect_case RUN_DIR SID [--gate expr]"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, zscore  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # noqa: E402


def main():
    run, sid = Path(sys.argv[1]), sys.argv[2]
    inv = custody.dev_inventory()
    row = inv[sid]
    custody.check_dev_scan(sid, inv)
    d = json.loads((run / f"{sid}.json").read_text())
    gold = zscore.S.load_gold(zscore.case_spec(sid, row)["gold"], frame_records=None)["lines"]
    C = d["candidates"]
    print(sid, row["family"], "t=", round(d["t_mm"], 2), "cands", len(C), "raw", d.get("raw_count"), "timing", {k: round(v, 2) for k, v in d["timing_s"].items()})
    for f in d["families"]:
        print("  fam", np.round(f["axis"], 3), "score", round(f["score"], 4), "arc", round(f["arc_area"], 1))
    rec = zscore.score_lines(sid, row, C)
    print("ALL-cand: TP", rec.get("true_positive"), "FN", rec.get("false_negative"), "FP", rec.get("false_positive"))
    for g in gold:
        ge = np.array([g["line_start"], g["line_end"]])
        gd = ge[1] - ge[0]
        gd /= np.linalg.norm(gd)
        fam_best = min(math.degrees(math.acos(min(1, abs(gd @ np.asarray(f["axis"]))))) for f in d["families"]) if d["families"] else 99
        best = None
        for j, c in enumerate(C):
            m = pair_metrics(ge, np.array([c["start"], c["end"]]))
            if m["axis_angle_deg"] > 15 or m["transverse_offset_mm"] > 15:
                continue
            sc = m["axis_angle_deg"] / 5 + m["transverse_offset_mm"] / 4 + (1 - min(m["gold_span_overlap"], m["prediction_span_overlap"]))
            if best is None or sc < best[0]:
                best = (sc, j, m)
        L = np.linalg.norm(ge[1] - ge[0])
        if best is None:
            print(f"  {g['bend_id']} L={L:.1f} famdev={fam_best:.1f} MISSED")
        else:
            _, j, m = best
            c = C[j]
            print(f"  {g['bend_id']} L={L:.1f} famdev={fam_best:.1f} -> c{j} ang={m['axis_angle_deg']:.1f} off={m['transverse_offset_mm']:.2f} "
                  f"ovg={m['gold_span_overlap']:.2f} ovp={m['prediction_span_overlap']:.2f} | Lc={c['length']:.1f} sup={c['n_support']} "
                  f"sf={c['support_frac']:.2f} incl={c['incl']:.1f} out={c['frac_outer']:.2f} strip={c.get('strip_frac', 0):.2f} fl={c.get('flank_frac', 0):.2f}")


if __name__ == "__main__":
    main()


def near(run, sid, bend_id):
    inv = custody.dev_inventory()
    row = inv[sid]
    d = json.loads((Path(run) / f"{sid}.json").read_text())
    gold = {g["bend_id"]: g for g in zscore.S.load_gold(zscore.case_spec(sid, row)["gold"], frame_records=None)["lines"]}
    g = gold[bend_id]
    ge = np.array([g["line_start"], g["line_end"]])
    gd = (ge[1] - ge[0]) / np.linalg.norm(ge[1] - ge[0])
    for j, c in enumerate(d["candidates"]):
        m = pair_metrics(ge, np.array([c["start"], c["end"]]))
        if m["axis_angle_deg"] < 6 and m["transverse_offset_mm"] < 5:
            s0 = (np.array(c["start"]) - ge[0]) @ gd
            s1 = (np.array(c["end"]) - ge[0]) @ gd
            print(f"  c{j} [{min(s0, s1):.1f},{max(s0, s1):.1f}] of {np.linalg.norm(ge[1]-ge[0]):.1f} ang={m['axis_angle_deg']:.1f} off={m['transverse_offset_mm']:.2f} sup={c['n_support']} incl={c['incl']:.1f} fam={c['family']}")
