"""Write testb/PREREG.json (the pre-registration of the one TEST-B run) and, with --ledger, append the freeze-ledger
NOTE binding it (domains.production.engine.eval.freeze.append_entry). Reads only hashes of the frozen gold files
(never their content), the split, the gold ledger, the lane config and the DEV out-of-fold summary.

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v1.prereg [--ledger]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from research.zeroshot.zs_lane_v1 import fingerprint as FP, scoring
from research.zeroshot.zs_lane_v1.core import REPO

PROG = scoring.PROG
OUT = PROG / "testb" / "PREREG.json"
CFG = REPO / "research/zeroshot/zs_lane_v1/config_v1.json"
FREEZE_LEDGER = REPO / "docs/production/runlog/freeze/LEDGER.json"


def _jsha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def build() -> dict:
    cfg = json.loads(CFG.read_text())
    sp = scoring.split()
    tb = sorted(sp["test_b"]["scans"])
    led = scoring._ledger_frozen()
    inv = scoring._inv()
    labels, old = {}, {}
    for s in tb:
        e = led[s]
        got = FP.sha256_file(e["label_path"])
        if got != e["label_sha256"]:
            raise SystemExit(f"{s}: frozen label sha mismatch ({got} vs ledger {e['label_sha256']})")
        labels[s] = {"path": e["label_path"].replace(str(REPO) + "/", ""), "sha256": got, "ledger_seq": e["seq"],
                     "ledger_entry_sha256": e["entry_sha256"]}
        op = inv[s]["gold"]["path"]
        old[s] = {"path": op, "sha256": FP.sha256_file(REPO / op)}
    oof = json.loads((PROG / "integration/oof_v1/DEV_OOF.json").read_text())["arms"]
    gl = PROG / "gold/GOLD_LEDGER.jsonl"
    gl_entries = [json.loads(x) for x in gl.read_text().splitlines() if x.strip()]
    lanes = {}
    for role, v in (("primary", "zs"), ("secondary_1", "z3"), ("secondary_2", "z2")):
        vv = cfg["variants"][v]
        arm = {"zs": "zs|fixed|snap", "z3": "z3|fixed|snap", "z2": "z2|none|z2snap"}[v]
        a = oof[arm]
        lanes[role] = {
            "variant": v, "name": {"zs": "zs_lane_v1 (Z3 + Z2 pooled GBM + D1 snap)", "z3": "Z3 alone (GBM + D1 snap)",
                                   "z2": "Z2 alone (it08 flange graph, internal D1 snap)"}[v],
            "config": vv, "model_sha256": vv["model_sha256"],
            "dev_oof": {k: a[k] for k in ("scans", "families", "gold", "tp", "fp", "fn", "P", "R", "micro_f1",
                                          "family_macro_f1", "micro_ci95", "macro_ci95", "controls")},
            "command": ("docs/production/runlog/zeroshot_2026-10-25/integration/run_cohort.sh "
                        "docs/production/runlog/zeroshot_2026-10-25/testb/testb_scans.tsv docs/production/runlog/zeroshot_2026-10-25/testb/runs_TESTB_CONSUMED " + v + " <i> 3 --final-testb"),
        }
    prereg = {
        "schema": "sherman.zeroshot.testb_prereg.v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "programme": "zeroshot_2026-10-25",
        "claim_label": "programme-sealed, consumed-by-rule TEST-B (OD-1); never an unseen-family accuracy claim",
        "split": {"path": "docs/production/runlog/zeroshot_2026-10-25/SPLIT_FROZEN.json",
                  "sha256": FP.sha256_file(PROG / "SPLIT_FROZEN.json")},
        "testb": {"n_scans": len(tb), "n_families_group5": sp["test_b"].get("n_family_groups"), "scan_ids": tb,
                  "scan_ids_sha256": _jsha(tb)},
        "gold_of_record": {
            "root": "data/bend_corpus/annotations/ai_bend_lines_zs_v1_SEALED/",
            "provenance": "OD-2 blind re-mark (2 markers) + adjudication, frozen in gold/GOLD_LEDGER.jsonl",
            "gold_ledger_sha256": FP.sha256_file(gl), "gold_ledger_entries": len(gl_entries),
            "gold_ledger_last_entry_sha256": gl_entries[-1]["entry_sha256"],
            "labels": labels, "labels_sha256": _jsha({s: labels[s]["sha256"] for s in tb})},
        "gold_secondary_column": {
            "root": "data/bend_corpus/annotations/ai_bend_lines_fresh_v1_SEALED/",
            "note": "old FRESH labels (2026-10-17, 15 deg floor, not re-audited); comparison column only; read under "
                    "freeze-ledger seq 8 (consumed-set diagnostics)",
            "labels": old, "labels_sha256": _jsha({s: old[s]["sha256"] for s in tb})},
        "lanes": lanes,
        "lane_config": {"path": "research/zeroshot/zs_lane_v1/config_v1.json", "sha256": FP.sha256_file(CFG)},
        "code_fingerprint": cfg["code_fingerprint"],
        "threshold_rule": cfg["threshold_rule"],
        "scorer": {"matcher": "research/broad_baseline/score.py score_case (frozen): 5 deg / 4 mm / 0.6 of both "
                              "spans, one-to-one Hungarian, apex locus, raw scan frame",
                   "fingerprint": FP.scorer_fingerprint(),
                   "driver": {"path": "research/zeroshot/zs_lane_v1/score_testb.py",
                              "sha256": FP.sha256_file(REPO / "research/zeroshot/zs_lane_v1/score_testb.py")}},
        "tooling_not_executed_by_the_lane": {f: FP.sha256_file(REPO / "research/zeroshot/zs_lane_v1" / f) for f in
                                             ("train.py", "dev_dump.py", "freeze_config.py", "prereg.py", "cad_prior.py")},
        "run_driver": {"path": "docs/production/runlog/zeroshot_2026-10-25/integration/run_cohort.sh",
                       "sha256": FP.sha256_file(PROG / "integration/run_cohort.sh"),
                       "guard": {"path": "docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh",
                                 "sha256": FP.sha256_file(PROG / "tools/guarded_slot.sh")}},
        "dev_cli_consistency": "the frozen CLI (config_v1, final models) reproduced the dump-pipeline selections "
                               "exactly on all 125 DEV scans x 3 variants (375/375 identical line sets, pins_ok)",
        "metrics": {
            "primary_metric": "micro line-F1 of the primary lane against the re-audit gold of record",
            "micro": "summed TP/FP/FN over the complete-label line scans",
            "family_macro": "mean over part families (INVENTORY 'family') of family-micro F1",
            "ci": {"method": "percentile bootstrap over families", "n": 10000, "seed": 20261025, "level": 0.95},
            "controls": "zero-bend controls reported separately as FP lines (TEST-B has none by construction of "
                        "the split, so this is expected to be empty)",
            "failed_scans": "a scan whose output is missing or FAILED is scored as an empty prediction: every gold "
                            "line is a false negative; no scan is dropped; a matcher exclusion aborts scoring",
            "also_reported": "precision, recall, error budget (score_broad.diagnose, nearest counterpart within "
                             "15 deg / 15 mm), runtime per scan (median / P95 / max), each lane against the old "
                             "FRESH labels as a secondary column"},
        "expected_range": {"primary_micro": [0.84, 0.92],
                           "basis": "DEV OOF 0.944 (125 scans) shrunk for selection/rule-design optimism (Z3 audit: "
                                    "touched 0.86 vs untouched 0.96) and for the FRESH-type distribution of TEST-B "
                                    "(DEV FRESH-DEV 17 line scans zs 0.950 / macro 0.893; the 4 newly frozen zs_v1 DEV scans zs 0.841; "
                                    "both outside Z2's rule design)",
                           "secondary_z3_micro": [0.76, 0.88], "secondary_z2_micro": [0.82, 0.92]},
        "decision_rule": "programme goal met on TEST-B iff primary micro >= 0.85 (point estimate; CI reported). "
                         "Secondaries are descriptive. TEST-B is run exactly once per lane and is CONSUMED afterwards: "
                         "no re-run, no tuning, no lane change justified by TEST-B results.",
        "run_procedure": ["each lane x scan through tools/guarded_slot.sh <sid> --final-testb (gold guard, one slot, "
                          "OMP 1); the lane re-checks the guard and refuses on any pinned-hash mismatch or unpinned "
                          "loaded module",
                          "scoring: research/zeroshot/zs_lane_v1/score_testb.py (scoring.py) once, after all runs",
                          "a freeze-ledger RUN entry binds the run before scoring; a NOTE binds the results sha"],
        "dev_oof_selection": {"chosen": "zs", "why": "zs - z3 = +0.030 micro, paired family bootstrap 95 % CI "
                                                     "[+0.013, +0.050]; +0.039 macro [+0.015, +0.070]; 25 families "
                                                     "better, 6 worse",
                              "dev_oof_path": "docs/production/runlog/zeroshot_2026-10-25/integration/oof_v1/DEV_OOF.json",
                              "dev_oof_sha256": FP.sha256_file(PROG / "integration/oof_v1/DEV_OOF.json")},
    }
    return prereg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args(argv)
    if OUT.is_file():
        raise SystemExit(f"{OUT} exists (pre-registration is write-once)")
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=1, sort_keys=True))
    sha = FP.sha256_file(OUT)
    print("PREREG", OUT, sha)
    if a.ledger:
        from domains.production.engine.eval import freeze
        e = freeze.append_entry(FREEZE_LEDGER, label="PRE-REGISTRATION: zero-shot TEST-B one-time run (zs_lane_v1 "
                                "primary; Z3, Z2 secondary), 2026-10-25", kind="NOTE",
                                recorded_by="zeroshot_2026-10-25/integrator", items={
                                    "prereg_path": str(OUT.relative_to(REPO)), "prereg_sha256": sha,
                                    "split_sha256": p["split"]["sha256"],
                                    "testb_scan_ids_sha256": p["testb"]["scan_ids_sha256"],
                                    "gold_labels_sha256": p["gold_of_record"]["labels_sha256"],
                                    "gold_ledger_sha256": p["gold_of_record"]["gold_ledger_sha256"],
                                    "lane_config_sha256": p["lane_config"]["sha256"],
                                    "code_fingerprint_sha256": p["code_fingerprint"]["sha256"],
                                    "scorer_fingerprint_sha256": p["scorer"]["fingerprint"]["sha256"],
                                    "models_sha256": {r: l["model_sha256"] for r, l in p["lanes"].items()},
                                    "expected_range_primary_micro": p["expected_range"]["primary_micro"],
                                    "metrics": "micro, family-macro, family bootstrap 95% CI n=10000 seed 20261025, "
                                               "controls separate, failed scans = all FN"})
        print("LEDGER", e["seq"], e["entry_id"], e["entry_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
