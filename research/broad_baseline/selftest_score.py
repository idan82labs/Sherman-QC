#!/usr/bin/env python3
"""Label-only self-test of score.py gold loaders (no scan geometry, no engine).

Builds synthetic eval-format predictions from real gold label files and checks
the scorer's dispositions: complete labels (TP/FP/FN), label_complete=false
(recall only, FPs unscored), zero-bend controls (FP only), sha-mismatch
exclusion, bench_lines (STL-only gold) and Gold11 frame-record exclusion.

  selftest_score.py --out docs/production/runlog/broad_baseline/scout/selftest_score.json
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import gold11_cases as g11  # noqa: E402
import score  # noqa: E402

AI_STUDIO = bb.AI_GOLD_ROOT / "1c62d21a1fceedefd236259a.json"           # 15651004_01.ply, 3 lines
AI_ZERO = bb.AI_GOLD_ROOT / "a92f04f2ef011fc53a09675c.json"             # 28870000_A_SCAN_1.ply, zero-bend
BENCH_STL = bb.ROOT / "docs/production/runlog/benchmark_v2/label_lines/sc_927a2fa13067/lines.json"  # STL-only


def _shift(line: dict, mm: float) -> dict:
    a, b = np.asarray(line["line_start"]), np.asarray(line["line_end"])
    axis = (b - a) / np.linalg.norm(b - a)
    normal = np.cross(axis, [0.0, 0.0, 1.0])
    if np.linalg.norm(normal) < 1e-6:
        normal = np.cross(axis, [0.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)
    return bb.as_line(line["bend_id"] + "_p", (a + mm * normal).tolist(), (b + mm * normal).tolist())


def _pred_file(tmp: Path, name: str, lines: list[dict], sha: str | None) -> Path:
    payload = bb.eval_payload(lane="synthetic", lines=lines, scan_path=None, scan_sha256=sha,
                              source_kind="synthetic_selftest", source_path=None, conversion={})
    return bb.write_json(tmp / f"{name}.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    checks = []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        gold = score.load_gold({"kind": "ai_studio", "path": str(AI_STUDIO)}, frame_records=None)
        g = gold["lines"]
        sha = gold["scan_sha256"]
        extra = bb.as_line("FP1", [0.0, 0.0, 0.0], [50.0, 0.0, 0.0])
        lines = [dict(g[0], bend_id="P1"), _shift(g[1], 2.0), _shift(g[2], 6.0), extra]
        pred = _pred_file(tmp, "ai", lines, sha)
        base = {"case_id": "ai", "tier": "AI_GOLD_ACCEPTED", "gold": {"kind": "ai_studio", "path": str(AI_STUDIO)},
                "predictions": str(pred)}
        full = score.score_case({**base, "label_complete": True}, frame_records=None)
        checks.append(("ai_complete", (full["true_positive"], full["false_positive"], full["false_negative"]), (2, 2, 1)))
        part = score.score_case({**base, "label_complete": False}, frame_records=None)
        checks.append(("ai_partial_recall_only",
                       (part["true_positive"], part["false_negative"], part["unscored_unmatched_predictions"],
                        part["precision"], part["disposition"]),
                       (2, 1, 2, None, "SCORED_RECALL_ONLY_PARTIAL_LABEL")))
        bad = _pred_file(tmp, "ai_bad_sha", lines, "0" * 64)
        mism = score.score_case({**base, "predictions": str(bad), "label_complete": True}, frame_records=None)
        checks.append(("ai_sha_mismatch_excluded", (mism["disposition"], mism["reason"]),
                       ("EXCLUDED_BEFORE_SCORING", "SCAN_SHA_MISMATCH")))
        zero = score.load_gold({"kind": "ai_studio", "path": str(AI_ZERO)}, frame_records=None)
        zpred = _pred_file(tmp, "zero", [extra], zero["scan_sha256"])
        zrow = score.score_case({"case_id": "zero", "tier": "AI_GOLD_ACCEPTED",
                                 "gold": {"kind": "ai_studio", "path": str(AI_ZERO)}, "predictions": str(zpred)},
                                frame_records=None)
        checks.append(("zero_bend_control", (zrow["disposition"], zrow["false_positive_lines"]),
                       ("ZERO_BEND_CONTROL", 1)))
        bench = score.load_gold({"kind": "bench_lines", "path": str(BENCH_STL)}, frame_records=None)
        bpred = _pred_file(tmp, "bench", [dict(x, bend_id=x["bend_id"] + "_p") for x in bench["lines"]],
                           bench["scan_sha256"])
        brow = score.score_case({"case_id": "bench", "tier": "AI_GOLD_ACCEPTED",
                                 "gold": {"kind": "bench_lines", "path": str(BENCH_STL)}, "predictions": str(bpred)},
                                frame_records=None)
        n = len(bench["lines"])
        checks.append(("bench_lines_identity", (brow["true_positive"], brow["false_positive"], brow["false_negative"]),
                       (n, 0, 0)))
        missing = score.score_case({"case_id": "missing", "tier": "AI_GOLD_ACCEPTED",
                                    "gold": {"kind": "ai_studio", "path": str(AI_STUDIO)},
                                    "predictions": str(tmp / "absent.json")}, frame_records=None)
        checks.append(("missing_predictions_all_fn", (missing["true_positive"], missing["false_negative"],
                                                      missing["lane_status"]), (0, 3, "PREDICTIONS_MISSING")))
        g11_spec = {"kind": "gold11_catalog", "scan_sha256": g11.scan_sha("CG4FAF273093")}
        nofr = score.score_case({"case_id": "CG4FAF273093", "gold": g11_spec, "predictions": str(pred)},
                                frame_records=None)
        checks.append(("gold11_reference_frame_without_record_excluded", (nofr["disposition"], nofr["reason"]),
                       ("EXCLUDED_BEFORE_SCORING", "NO_RECOVERABLE_TRANSFORM")))
    results = [{"check": name, "observed": list(obs) if isinstance(obs, tuple) else obs,
                "expected": list(exp) if isinstance(exp, tuple) else exp, "pass": obs == exp}
               for name, obs, exp in checks]
    report = {"schema": "sherman.broad_baseline.selftest.v1", "created_at": bb.utc_now(),
              "passed": sum(r["pass"] for r in results), "total": len(results), "checks": results}
    bb.write_json(args.out, report)
    for r in results:
        print(("PASS " if r["pass"] else "FAIL ") + r["check"], r["observed"])
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
