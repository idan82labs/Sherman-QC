"""Cross-check relabel lines against the auditors' independent apex lines (read-only)."""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics

lines_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
AUD = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit")
L = json.loads(lines_path.read_text())
ours = {b["bend_id"]: np.array([b["label_line"]["start"], b["label_line"]["end"]]) for b in L["bends"] if b["status"] == "ok"}
a1 = json.loads((AUD / "auditor1/sc_71cf0b216f43/a1_apex_lines.json").read_text())["folds"]
a3 = json.loads((AUD / "auditor3/sc_71cf0b216f43/a3_apex_lines.json").read_text())["lines"]
refs = {}
for k, v in a1.items():
    refs[f"a1:{k}({v['gold_id']})"] = np.array(v["apex_line_raw"])
for k, v in a3.items():
    refs[f"a3:{k}"] = np.array([v["start"], v["end"]])
out = {}
for bid, P in ours.items():
    rows = []
    for rid, G in refs.items():
        m = pair_metrics(G, P)
        if m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 6:
            rows.append({"ref": rid, **{k: round(v, 3) for k, v in m.items()}})
    rows.sort(key=lambda r: r["transverse_offset_mm"])
    out[bid] = rows[:3]
    print(bid, " | ".join(f"{r['ref']} off {r['transverse_offset_mm']:.2f} ang {r['axis_angle_deg']:.2f} ovl(ref/ours) {r['gold_span_overlap']:.2f}/{r['prediction_span_overlap']:.2f} len {r['gold_length_mm']:.0f}/{r['prediction_length_mm']:.0f}" for r in rows[:2]))
out_path.write_text(json.dumps(out, indent=1))
