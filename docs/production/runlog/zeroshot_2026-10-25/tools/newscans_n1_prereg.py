"""Write newscans_N1/PREREG.json (pre-registration of the one-time zero-shot run on the 4 N1 new-family scans) and,
with --ledger, append the freeze-ledger NOTE (pre-registration) and RUN (run binding) entries via
domains.production.engine.eval.freeze.append_entry, as testb/ did (zs_lane_v1/prereg.py). Reads only hashes of the
frozen N1 gold labels (from the gold ledger), configs, models, fingerprints. Write-once.
usage (repo root): .venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/tools/newscans_n1_prereg.py [--ledger]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v1 import fingerprint as FP1  # noqa: E402
from research.zeroshot.zs_lane_v2 import fingerprint as FP2  # noqa: E402

P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
NB = P / "newscans_N1"
OUT = NB / "PREREG.json"
TSV = NB / "n1_scans.tsv"
CFG1 = REPO / "research/zeroshot/zs_lane_v1/config_v1.json"
CFG2 = REPO / "research/zeroshot/zs_lane_v2/config_v2.json"
FREEZE_LEDGER = REPO / "docs/production/runlog/freeze/LEDGER.json"
RUNS = NB / "runs"


def jsha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO))


def build() -> dict:
    c1, c2 = json.loads(CFG1.read_text()), json.loads(CFG2.read_text())
    if FP1.lane_fingerprint()["sha256"] != c1["code_fingerprint"]["sha256"]:
        raise SystemExit("zs_lane_v1 code fingerprint differs from config_v1 pin")
    if FP2.lane_fingerprint()["sha256"] != c2["code_fingerprint"]["sha256"]:
        raise SystemExit("zs_lane_v2 code fingerprint differs from config_v2 pin")
    reg = json.loads((NB / "REGISTRY.json").read_text())
    led = [json.loads(x) for x in (P / "gold/GOLD_LEDGER.jsonl").read_text().splitlines() if x.strip()]
    last = {}
    for e in led:
        last[e["scan_id"]] = e
    scans, labels = [], {}
    for r in sorted(reg["scans"], key=lambda r: r["part_id"]):
        sid = f"sc_{r['sha256'][:12]}"
        e = last[sid]
        assert e["status"] == "FROZEN" and e["portion"] == "NEW_UNSEEN_FAMILY_TEST" and e["batch_id"] == "N1"
        got = FP1.sha256_file(e["label_path"])
        if got != e["label_sha256"]:
            raise SystemExit(f"{sid}: frozen label sha mismatch")
        labels[sid] = {"part_id": r["part_id"], "path": rel(e["label_path"]), "sha256": got, "ledger_seq": e["seq"],
                       "ledger_entry_sha256": e["entry_sha256"]}
        scans.append((sid, r["original_path"], r["part_id"]))
    TSV.write_text("".join(f"{s}\t{p}\n" for s, p, _ in scans))
    ids = [s for s, _, _ in scans]
    v1_cmd = (f"{rel(P / 'integration/run_cohort.sh')} {rel(TSV)} {rel(RUNS / 'v1')} zs,z3,z2 0 1")
    v2_cmd = (f"research/zeroshot/zs_lane_v2/run_cohort.sh {rel(TSV)} {rel(RUNS / 'v2')} 0 1")
    arms = {
        "primary": {"lane": "zs_lane_v1", "variant": "zs", "name": "zs_lane_v1 (Z3 + Z2 pooled GBM + D1 snap)",
                    "model_sha256": c1["variants"]["zs"]["model_sha256"], "command": v1_cmd,
                    "output": rel(RUNS / "v1/zs")},
        "secondary_v2": {"lane": "zs_lane_v2", "variant": "zs",
                         "name": "zs_lane_v2 (v1 pool + sweep evidence + split/join post-processing)",
                         "model_sha256": c2["variants"]["zs"]["model_sha256"], "command": v2_cmd + " (--strict-pins)",
                         "output": rel(RUNS / "v2/zs")},
        "secondary_z3": {"lane": "zs_lane_v1", "variant": "z3", "name": "Z3 alone (GBM + D1 snap)",
                         "model_sha256": c1["variants"]["z3"]["model_sha256"], "command": v1_cmd,
                         "output": rel(RUNS / "v1/z3")},
        "secondary_z2": {"lane": "zs_lane_v1", "variant": "z2", "name": "Z2 alone (it08 flange graph, internal D1 snap)",
                         "model_sha256": c1["variants"]["z2"].get("model_sha256"), "command": v1_cmd,
                         "output": rel(RUNS / "v1/z2")},
    }
    return {
        "schema": "sherman.zeroshot.newscans_prereg.v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "programme": "zeroshot_2026-10-25",
        "claim_label": "one-time run on 4 never-seen part families (N1, received 2026-10-19); N=4 scans / 4 families "
                       "(3 of them zero-bend rolled shells, 26 gold lines in total), too small for a headline claim",
        "cohort": {"batch": "N1", "registry": rel(NB / "REGISTRY.json"),
                   "registry_sha256": FP1.sha256_file(NB / "REGISTRY.json"), "scan_ids": ids, "scan_ids_sha256": jsha(ids),
                   "scan_list": rel(TSV), "families_group5": sorted(p[:5] for _, _, p in scans),
                   "rule": reg["rule"]},
        "gold_of_record": {"root": "data/bend_corpus/annotations/ai_bend_lines_newscans_n1_v1/",
                           "provenance": "blind 2-marker marking + adjudication (gold/N1/ADJUDICATION_N1.json), canary "
                                         "F1 1.0/1.0, frozen in gold/GOLD_LEDGER.jsonl before any engine output",
                           "adjudication_sha256": FP1.sha256_file(P / "gold/N1/ADJUDICATION_N1.json"),
                           "gold_ledger_sha256": FP1.sha256_file(P / "gold/GOLD_LEDGER.jsonl"),
                           "gold_ledger_entries": len(led), "gold_ledger_last_entry_sha256": led[-1]["entry_sha256"],
                           "labels": labels, "labels_sha256": jsha({s: labels[s]["sha256"] for s in ids})},
        "arms": arms,
        "lane_configs": {"zs_lane_v1": {"path": rel(CFG1), "sha256": FP1.sha256_file(CFG1)},
                         "zs_lane_v2": {"path": rel(CFG2), "sha256": FP2.sha256_file(CFG2)}},
        "code_fingerprints": {"zs_lane_v1": c1["code_fingerprint"], "zs_lane_v2": c2["code_fingerprint"]},
        "threshold_rules": {"zs_lane_v1": c1["threshold_rule"], "zs_lane_v2": c2["threshold_rule"]},
        "scorer": {"matcher": "research/broad_baseline/score.py score_case (frozen): 5 deg / 4 mm / 0.6 of both "
                              "spans (gold and prediction), one-to-one Hungarian, apex locus, raw scan frame",
                   "settings": {"axis_angle_deg": 5.0, "offset_mm": 4.0, "gold_span_overlap": 0.6,
                                "prediction_span_overlap": 0.6},
                   "fingerprint_v1": FP1.scorer_fingerprint(), "fingerprint_v2": c2.get("scorer_fingerprint"),
                   "driver": "research/zeroshot/zs_lane_v1/scoring.py score_scan + aggregate (unchanged), called by "
                             "tools/newscans_n1_score.py"},
        "run_driver": {"v1": {"path": rel(P / "integration/run_cohort.sh"),
                              "sha256": FP1.sha256_file(P / "integration/run_cohort.sh")},
                       "v2": {"path": "research/zeroshot/zs_lane_v2/run_cohort.sh",
                              "sha256": FP1.sha256_file(REPO / "research/zeroshot/zs_lane_v2/run_cohort.sh")},
                       "guard": {"path": rel(P / "tools/guarded_slot.sh"),
                                 "sha256": FP1.sha256_file(P / "tools/guarded_slot.sh")},
                       "gold_guard_py_sha256": FP1.sha256_file(REPO / "research/zeroshot/common/gold_guard.py"),
                       "gold_guard_note": "gold_guard.py unchanged (pinned in both lane fingerprints); N1 scans are "
                                          "admitted by its existing GOLD_LEDGER_FROZEN basis after "
                                          "tools/gold_ledger_newscans.py registered and froze them",
                       "ledger_tool": {"path": rel(P / "tools/gold_ledger_newscans.py"),
                                       "sha256": FP1.sha256_file(P / "tools/gold_ledger_newscans.py")}},
        "metrics": {
            "per_scan": "TP / FP / FN and F1 from score_case. Zero-gold scans follow the frozen convention exactly: "
                        "score.py gives them disposition ZERO_BEND_CONTROL (no F1), and scoring.aggregate reports them "
                        "separately as FP lines (BASELINE.md, testb PREREG 'controls'). For the owner page only, a "
                        "display F1 is derived: 1.0 if the control has 0 predicted lines, else 0.0 (labelled as such).",
            "micro": "scoring.aggregate micro: summed TP/FP/FN over complete-label line scans (here gb_4dd5ba "
                     "47085000 only); also reported: 'micro incl. control FP lines' = the same sums with every "
                     "line predicted on a zero-bend scan added as FP",
            "angle_agreement": "for each matched line, engine 'incl' (included angle, deg) minus gold "
                               "included_angle_deg from the frozen final lines (labelling tool patch fit); joggle "
                               "lines flagged (production never shows a zs angle on a joggle, ANGLE_CHECK.md)",
            "ci": "family bootstrap as testb (n 10000, seed 20261025) computed but N=4 families makes it "
                  "uninformative",
            "failed_scans": "a missing or FAILED output = empty prediction (all gold FN; on a control 0 FP); no scan "
                            "is dropped",
            "runtime": "wall time and peak RSS per process from run_record.json"},
        "expected": "no numeric expectation is pre-registered for a decision; descriptive only. Context: TEST-B "
                    "zs_lane_v1 0.855 micro; DEV OOF v1 0.944, v2 0.958; DEV zero-bend controls v1 5 FP lines / 5 "
                    "scans, v2 1 FP line / 5 scans.",
        "decision_rule": "none (descriptive, N=4). Each arm runs exactly once per scan; no re-run, no tuning, no lane, "
                         "threshold or rule change on these scans. A crash is recorded, not retried.",
        "cad_path_E3": "NOT RUN: the customer sent SolidWorks .SLDPRT, not STEP; no conversion attempted, so E3 "
                       "(CAD-assisted) has no compiled recipe for these parts.",
        "run_procedure": ["each arm x scan through tools/guarded_slot.sh <sid> (gold guard: GOLD_LEDGER_FROZEN, one "
                          "geometry slot, OMP 1); zs_lane_v2 with --strict-pins (refuses on any pin mismatch or "
                          "unpinned loaded module); zs_lane_v1 records pins_ok",
                          "v1 cohort (zs,z3,z2) and v2 cohort run as 2 concurrent workers (RSS < 16 GiB)",
                          "freeze-ledger RUN entry appended before the first lane process; NOTE binds RESULTS sha"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args(argv)
    if OUT.is_file():
        raise SystemExit(f"{OUT} exists (pre-registration is write-once)")
    p = build()
    OUT.write_text(json.dumps(p, indent=1, sort_keys=True) + "\n")
    sha = FP1.sha256_file(OUT)
    print("PREREG", rel(OUT), sha)
    if a.ledger:
        from domains.production.engine.eval import freeze
        e = freeze.append_entry(FREEZE_LEDGER, kind="NOTE", recorded_by="zeroshot_2026-10-25/N1-runner",
                                label="PRE-REGISTRATION: zero-shot one-time run on N1 new-family scans (zs_lane_v1 "
                                      "primary; zs_lane_v2, Z3, Z2 secondary), 2026-10-25",
                                items={"prereg_path": rel(OUT), "prereg_sha256": sha,
                                       "scan_ids_sha256": p["cohort"]["scan_ids_sha256"],
                                       "gold_labels_sha256": p["gold_of_record"]["labels_sha256"],
                                       "gold_ledger_sha256": p["gold_of_record"]["gold_ledger_sha256"],
                                       "code_fingerprint_v1_sha256": p["code_fingerprints"]["zs_lane_v1"]["sha256"],
                                       "code_fingerprint_v2_sha256": p["code_fingerprints"]["zs_lane_v2"]["sha256"],
                                       "lane_config_v1_sha256": p["lane_configs"]["zs_lane_v1"]["sha256"],
                                       "lane_config_v2_sha256": p["lane_configs"]["zs_lane_v2"]["sha256"],
                                       "scorer_fingerprint_sha256": p["scorer"]["fingerprint_v1"]["sha256"],
                                       "matcher": "5 deg / 4 mm / 0.6 / 0.6",
                                       "claim_label": p["claim_label"]})
        print("NOTE", e["seq"], e["entry_id"], e["entry_sha256"])
        r = freeze.append_entry(FREEZE_LEDGER, kind="RUN", recorded_by="zeroshot_2026-10-25/N1-runner",
                                label=f"RUN: zero-shot one-time run on N1 new-family scans under PREREG (ledger seq "
                                      f"{e['seq']})",
                                items={"tier": "NEW_UNSEEN_FAMILY_TEST N1 (one-time, N=4)",
                                       "seedset": "zeroshot_2026-10-25 newscans_N1 REGISTRY",
                                       "case_ids_sha256": p["cohort"]["scan_ids_sha256"], "n_cases": 4,
                                       "lanes": ["zs_lane_v1:zs", "zs_lane_v2:zs", "zs_lane_v1:z3", "zs_lane_v1:z2"],
                                       "prereg_sha256": sha, "prereg_ledger_seq": e["seq"],
                                       "code_fingerprint_v1_sha256": p["code_fingerprints"]["zs_lane_v1"]["sha256"],
                                       "code_fingerprint_v2_sha256": p["code_fingerprints"]["zs_lane_v2"]["sha256"],
                                       "output_root": rel(RUNS), "flag": "none (guard basis GOLD_LEDGER_FROZEN)",
                                       "authorized_by": "programme N1 runner task (owner-directed one-time run)"})
        print("RUN", r["seq"], r["entry_id"], r["entry_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
