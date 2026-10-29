#!/usr/bin/env python3
"""Make gold11_relabel_v1 the 11/11 AI relabel (owner decision 2026-10-17) and rescore it.

Inputs (read-only): the 11 AI relabel labels in data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1
(sc_07062eee17aa now at revision 2, from relabel/sc_07062eee17aa_r2, which passed two independent blind
verifiers: relabel/sc_07062eee17aa_r2/VERIFICATION.json), case_list_broad_v3, the frozen frame records and the
EXISTING best_f1 / runtime predictions under broad_baseline/runs. No engine, prep, lines or label step runs here.

Writes:
  truth/superseded_10of11/   byte copies of the 10/11 truth files before this update
  truth/gold11_relabel_v1.json, truth/gold11_relabel_v1_gold_override.json,
  truth/case_list_gold11__relabel_v1.json, truth/case_list_broad_v3__gold11_relabel_v1.json   (now 11/11 AI)
  rescore/relabel_11of11/{gold11,broad57}/scores_<lane>.json + RESCORE_SUMMARY.json

Run: OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/broad_baseline/gold11_audit/scripts/rescore_relabel_11of11.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g11a_common as C  # noqa: E402
import build_truth_variants as BT  # noqa: E402
import rescore_gold11_variants as RV  # noqa: E402

TRUTH = C.AUDIT / "truth"
OUT = C.AUDIT / "rescore" / "relabel_11of11"
R2 = C.AUDIT / "relabel" / "sc_07062eee17aa_r2"
HUMAN_CATALOG = C.ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
HUMAN_CATALOG_SHA = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"
FILES = ("gold11_relabel_v1.json", "gold11_relabel_v1_gold_override.json",
         "case_list_gold11__relabel_v1.json", "case_list_broad_v3__gold11_relabel_v1.json")


def main() -> int:
    assert C.bb.sha256_file(HUMAN_CATALOG) == HUMAN_CATALOG_SHA, "human Gold11 catalog changed"
    human_mirror = {p.name: C.bb.sha256_file(p) for p in sorted(C.HUMAN_MIRROR_ROOT.glob("*.json"))}
    verification = C.read_json(R2 / "VERIFICATION.json")
    assert verification["both_passed"] is True, "r2 relabel not passed by both verifiers"

    sup = TRUTH / "superseded_10of11"
    sup.mkdir(exist_ok=True)
    for name in FILES:
        if not (sup / name).exists():
            shutil.copy2(TRUTH / name, sup / name)

    base = C.read_json(C.CASE_LIST)
    entries = {c["case_id"]: c for c in base["cases"]}
    g11_ids = [c["case_id"] for c in base["cases"] if c["tier"] == "GOLD11"]
    assert sorted(g11_ids) == sorted(C.SCANS)
    old = C.read_json(sup / "gold11_relabel_v1.json")
    audit = {s["scan_id"]: s for s in C.read_json(C.AUDIT / "audit_results.json")["scans"]}

    cases = [BT.ai_entry(entries[i], i, "gold11_relabel_v1") for i in g11_ids]
    by_id = {c["case_id"]: c for c in cases}
    lists = {}
    lists["case_list_gold11__relabel_v1"] = BT.write_case_list(
        "case_list_gold11__relabel_v1", cases, base,
        "gold11_relabel_v1 (11/11, owner decision 2026-10-17): AI vision relabel on all 11 Gold11 scans; "
        "sc_07062eee17aa at label revision 2 (B2 refit, re-verified by 2 blind verifiers)")
    broad = [by_id.get(c["case_id"], c) for c in base["cases"]]
    lists["case_list_broad_v3__gold11_relabel_v1"] = BT.write_case_list(
        "case_list_broad_v3__gold11_relabel_v1", broad, base,
        "case_list_broad_v3 with the 11 Gold11 gold specs replaced by gold11_relabel_v1 (11/11 AI relabel); "
        "every non-Gold11 entry is byte-for-byte the v3 entry")
    C.write_json(TRUTH / "gold11_relabel_v1_gold_override.json", {i: by_id[i]["gold"] for i in g11_ids})

    per_scan = {}
    for i in g11_ids:
        label_path = C.RELABEL_ROOT / f"{C.RELABEL_ASSET[i]}.json"
        label = C.read_json(label_path)
        d = dict(old["per_scan"][i])
        d.update({"source": "AI_RELABEL", "relabel_lines": len(label["bends"]),
                  "relabel_label": C.rel(label_path), "relabel_label_sha256": C.bb.sha256_file(label_path),
                  "relabel_label_revision": label.get("revision")})
        if i == "sc_07062eee17aa":
            d["relabel_pass"] = True
            d["relabel_round"] = "r2 (relabel/sc_07062eee17aa_r2)"
            d["r1_verifiers_failed"] = d.pop("verifiers")
            d["verifiers"] = verification["verifiers"]
        per_scan[i] = d
    manifest = {
        "schema": "sherman.gold11_audit.truth_variant.v1",
        "created_at": C.bb.utc_now(),
        "variant": "gold11_relabel_v1",
        "status": "OFFICIAL - Gold11 gate reference (owner decision 2026-10-17, gold11_audit/OWNER_DECISION_2026-10-17.md)",
        "rule": "AI vision relabel (claude-vision-ai-v1, data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1) "
                "on all 11 Gold11 scans, each passed by two independent blind verifiers; line locus = outer-skin "
                "bend apex; matcher unchanged (5 deg / 4 mm / 60 %)",
        "human_gold_role": "historical diagnostic only; kept untouched (catalog and human_bend_lines_v1)",
        "counts": {"ai_relabel": 11, "human_gold": 0,
                   "gold_lines": sum(d["relabel_lines"] for d in per_scan.values()),
                   "human_gold_lines_replaced": sum(d["human_lines"] for d in per_scan.values())},
        "per_scan": per_scan,
        "gold_override_map": C.rel(TRUTH / "gold11_relabel_v1_gold_override.json"),
        "case_lists": lists,
        "superseded": {"what": "the 10/11 PROPOSED version (sc_07062eee17aa on human gold)",
                       "copies": C.rel(sup)},
        "stale_sensitivity_lists": "case_list_gold11__all11_ai_relabel.json (and the snapped-apex diagnostics) pin "
                                   "sc_07062eee17aa's label revision 1 by sha256; they are historical",
        "human_gold_untouched": {"catalog": C.rel(HUMAN_CATALOG), "catalog_sha256": C.bb.sha256_file(HUMAN_CATALOG),
                                 "expected_sha256": HUMAN_CATALOG_SHA},
    }
    C.write_json(TRUTH / "gold11_relabel_v1.json", manifest)

    # ---- rescore (frozen matcher via score.py broad; existing predictions only)
    reports, runs = {"gold11": {}, "broad57": {}}, {}
    for key, cl in (("gold11", TRUTH / "case_list_gold11__relabel_v1.json"),
                    ("broad57", TRUTH / "case_list_broad_v3__gold11_relabel_v1.json")):
        od = OUT / key
        od.mkdir(parents=True, exist_ok=True)
        for lane in RV.LANES:
            runs[f"{key}:{lane}"] = RV.run_score(cl, lane, od)
            reports[key][lane] = C.read_json(od / f"scores_{lane}.json")
    ceiling_g11 = RV.ceiling(TRUTH / "case_list_gold11__relabel_v1.json")
    prev = C.read_json(C.AUDIT / "rescore" / "RESCORE_SUMMARY.json")
    published = {lane: C.read_json(C.BB / f"scores_{lane}.json") for lane in RV.LANES}
    g11 = reports["gold11"]
    b = reports["broad57"]
    g11_ids_sorted = sorted(C.SCANS)
    all_ids = [r["case_id"] for r in b["best_f1"]["per_scan"] if r.get("disposition") == "SCORED_COMPLETE_LABEL"]

    # perfect-apex ceiling on the broad 57: Gold11 rows replaced by the ceiling rows, other rows = published v3 rows
    import score_broad as sb  # noqa: E402
    ceil_rows = {k: v for k, v in ceiling_g11["per_scan"].items()}
    broad_ceiling = {}
    for lane in RV.LANES:
        merged = []
        for r in b[lane]["cases"]:
            if r.get("tier") == "GOLD11" and r["case_id"] in ceil_rows:
                tp, fp, fn = ceil_rows[r["case_id"]]
                prf = RV.sc.prf(tp, fp, fn)
                r = dict(r, true_positive=tp, false_positive=fp, false_negative=fn, matches=None,
                         precision=prf["precision"], recall=prf["recall"], f1=prf["f1"])
            merged.append(r)
        broad_ceiling[lane] = {k: RV.block(sb.aggregate([r for r in merged if fn_(r)]))
                               for k, fn_ in sb.COHORTS.items() if k in ("ALL", "GOLD11", "NON_GOLD11", "CAD_AVAILABLE")}

    summary = {
        "schema": "sherman.gold11_audit.rescore_summary.v1",
        "created_at": C.bb.utc_now(),
        "variant": "gold11_relabel_v1 (11/11 AI relabel, official)",
        "not_redone": "no engine, prep, lines or labeling step was re-run here; existing predictions only",
        "matcher": g11["best_f1"]["matcher"],
        "score_runs": runs,
        "gold11": {
            "case_list": g11["best_f1"]["case_list"],
            "lanes": {lane: {"gold11": RV.block(g11[lane]["cohorts"]["GOLD11"]), "per_scan": RV.per_scan(g11[lane])}
                      for lane in RV.LANES},
            "bootstrap_f1_ci95": RV.bootstrap(g11, g11_ids_sorted),
            "perfect_apex_engine_ceiling": ceiling_g11,
            "previous_10of11": {lane: prev["gold11"]["relabel_v1"]["lanes"][lane]["gold11"] for lane in RV.LANES},
            "previous_10of11_ceiling": prev["gold11"]["relabel_v1"]["perfect_apex_engine_ceiling"],
            "human_historical": {lane: prev["gold11"]["human"]["lanes"][lane]["gold11"] for lane in RV.LANES},
            "delta_vs_human_f1_ci95": {lane: RV.paired_delta_ci(
                C.read_json(C.AUDIT / "rescore" / "human" / f"scores_{lane}.json"), g11[lane], g11_ids_sorted)
                for lane in RV.LANES},
        },
        "broad57": {
            "case_list": b["best_f1"]["case_list"],
            "lanes": {lane: {k: RV.block(b[lane]["cohorts"][k]) for k in RV.COHORTS_BROAD} for lane in RV.LANES},
            "published_v3": {lane: {k: RV.block(published[lane]["cohorts"][k]) for k in RV.COHORTS_BROAD}
                             for lane in RV.LANES},
            "previous_10of11": {lane: {k: prev["broad57"]["lanes"][lane][k] for k in ("ALL", "GOLD11", "NON_GOLD11",
                                                                                      "CAD_AVAILABLE")}
                                for lane in RV.LANES},
            "bootstrap_f1_ci95_ALL": RV.bootstrap(b, all_ids),
            "delta_vs_published_f1_ci95_ALL": {lane: RV.paired_delta_ci(published[lane], b[lane], all_ids)
                                               for lane in RV.LANES},
            "perfect_apex_ceiling_gold11_rows": broad_ceiling,
            "note": ("'lanes' = score.py on the broad v3 case list with gold11_relabel_v1 (11/11) substituted, label "
                     "files as on disk now (includes sc_2233372b8df7 AI gold revision 2, +2 FN vs published v3). "
                     "'perfect_apex_ceiling_gold11_rows' replaces only the 11 Gold11 rows of 'lanes' with a perfect apex "
                     "engine (the relabel lines as predictions); the 46 other rows keep the engine's scores, because "
                     "their gold is not an apex relabel, so a perfect-apex ceiling is defined for the Gold11 part only."),
        },
        "checks": {},
    }
    checks = summary["checks"]
    for lane in RV.LANES:
        checks[f"broad57_gold11_equals_gold11_{lane}"] = all(
            b[lane]["cohorts"]["GOLD11"][k] == g11[lane]["cohorts"]["GOLD11"][k]
            for k in ("true_positive", "false_positive", "false_negative", "micro_f1"))
        checks[f"broad57_non_gold11_equals_previous_{lane}"] = all(
            b[lane]["cohorts"]["NON_GOLD11"][k] == prev["broad57"]["lanes"][lane]["NON_GOLD11"][k]
            for k in ("true_positive", "false_positive", "false_negative", "micro_f1"))
        checks[f"excluded_before_scoring_{lane}"] = {"gold11": g11[lane]["excluded_before_scoring"],
                                                     "broad57": b[lane]["excluded_before_scoring"]}
    checks["ceiling_gold11_is_perfect"] = ceiling_g11["f1"] == 1.0
    checks["human_catalog_untouched"] = C.bb.sha256_file(HUMAN_CATALOG) == HUMAN_CATALOG_SHA
    checks["human_bend_lines_v1_untouched_during_run"] = human_mirror == {
        p.name: C.bb.sha256_file(p) for p in sorted(C.HUMAN_MIRROR_ROOT.glob("*.json"))}
    C.write_json(OUT / "RESCORE_SUMMARY.json", summary)

    for lane in RV.LANES:
        x = summary["gold11"]["lanes"][lane]["gold11"]
        print(f"gold11 {lane:8s} {x['true_positive']}/{x['false_positive']}/{x['false_negative']} F1={x['micro_f1']:.4f} "
              f"macro={x['macro_f1']}")
        for k in ("ALL", "GOLD11", "NON_GOLD11", "CAD_AVAILABLE"):
            c = summary["broad57"]["lanes"][lane][k]
            cc = broad_ceiling[lane][k]
            print(f"broad57 {lane:8s} {k:14s} {c['true_positive']}/{c['false_positive']}/{c['false_negative']} "
                  f"F1={c['micro_f1']:.4f} | gold11-perfect {cc['true_positive']}/{cc['false_positive']}/"
                  f"{cc['false_negative']} F1={cc['micro_f1']:.4f}")
    print("ceiling gold11", {k: ceiling_g11[k] for k in ("true_positive", "false_positive", "false_negative", "f1")})
    print(json.dumps({k: v for k, v in checks.items() if not k.startswith("excluded")}, indent=1))
    ok = all(v for k, v in checks.items() if not k.startswith("excluded"))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
