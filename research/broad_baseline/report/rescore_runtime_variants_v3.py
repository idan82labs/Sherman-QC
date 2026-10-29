"""Re-score the two diagnostic runtime variants on case_list_broad_v3 (read-only, no engine run).

scores_runtime__runtime_gated.json and scores_runtime__runtime_with_placeholders.json in the
run log were scored on the 67-case v2 list. This rescoring uses the same prediction files from
runs/runtime/<case_id>/ and the same scorer (score_broad.build) on the 60-case v3 list, and
writes to docs/production/runlog/broad_baseline/report_rescore_v3/ so the stale files stay as
they were.

  predictions_runtime_gated.json            runtime lines cropped by the research-side raw-scan
                                            support gate (NOT part of the shipped runtime)
  predictions_runtime_with_placeholders.json runtime lines plus CAD_REFERENCE_PLACEHOLDER lines
                                            (nominal CAD axes, not scan detections)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score_broad as sb  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "docs/production/runlog/broad_baseline"
DEST = OUT / "report_rescore_v3"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for variant in ("predictions_runtime.json", "predictions_runtime_gated.json", "predictions_runtime_with_placeholders.json"):
        report = sb.build(OUT / "case_list_broad_v3.json", runs_root=OUT / "runs", lane="runtime", pred_file=variant,
                          frame_records_path=OUT / "broad_frame_records.json", title=f"broad runtime:{variant} (v3 rescore)")
        name = "scores_runtime__" + variant.removeprefix("predictions_").removesuffix(".json") + ".json"
        (DEST / name).write_text(json.dumps(report, indent=1) + "\n")
        a = report["cohorts"]["ALL"]
        print(f"{variant:45s} TP={a['true_positive']} FP={a['false_positive']} FN={a['false_negative']} "
              f"P={a['micro_precision']:.3f} R={a['micro_recall']:.3f} F1={a['micro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
