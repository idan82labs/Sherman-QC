"""Final held-out evaluation 2026-10-28, Stage 1 step 2: NOTE (owner authorization + set manifest sha) and FREEZE
(engines, matcher, bootstrap) in docs/production/runlog/freeze/LEDGER.json via freeze.append_entry. NOTE/FREEZE only.
Verifies both production pins with domains.production.zeroshot.pin before writing. usage: .venv/bin/python -B <this> note|freeze
"""
import hashlib, json, sys
from pathlib import Path

R = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(R))
from domains.production.engine.eval import freeze  # noqa: E402
from domains.production.zeroshot import pin  # noqa: E402

LEDGER = R / "docs/production/runlog/freeze/LEDGER.json"
D = Path(__file__).resolve().parents[1]
BY = "final_eval_2026-10-28_heldout/stage1-prepare"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(R))


def note():
    m = json.loads((D / "SET_MANIFEST.json").read_text())
    return freeze.append_entry(LEDGER, kind="NOTE", recorded_by=BY,
        label="AUTHORIZATION: final one-time held-out evaluation (HOLDOUT10 minus 41880000_D, N001-N016 minus consumed "
              "30820000_A01, EHO4), no-CAD first; set manifest bound",
        items={"date": "2026-10-28", "owner_choice": "Use them now, no-CAD first",
               "effect": "consumes the remaining held-out scans once; after the run they are CONSUMED and may never "
                         "again serve as held-out, tuning or selection data",
               "set_manifest": {"path": rel(D / "SET_MANIFEST.json"), "sha256": sha(D / "SET_MANIFEST.json")},
               "scored_scan_ids": m["scored_scan_ids"],
               "scored_scan_ids_sha256": hashlib.sha256(json.dumps(m["scored_scan_ids"]).encode()).hexdigest(),
               "counts": m["counts"],
               "listed_not_scored": {"30820000_A01 (N015/N016)": "consumed, freeze-ledger seq 34"},
               "excluded_untouched": ["41880000_D", "Eyal50 quarantine", "rhino-preview face scans", "*_SEALED* paths"],
               "protocol_model": "newscans_N1 (freeze seqs 17-19; gold ledger seqs 131-138; labels root "
                                 "ai_bend_lines_newscans_n1_v1)"})


def freeze_engines():
    out = {}
    for lane, role in (("zs_lane_v3", "primary"), ("zs_lane_v1", "comparator")):
        p = pin.load_pin(lane=lane)
        bad = pin.verify(pin=p)
        if bad:
            sys.exit(f"{lane} pin does not verify: {bad}")
        pr = R / pin.PIN_RELS[lane]
        out[lane] = {"role": role, "revision": p.get("revision"),
                     "code_fingerprint_sha256": p["code_fingerprint"]["sha256"],
                     "pin": {"path": pin.PIN_RELS[lane], "sha256": sha(pr)},
                     "models": {k: {"path": v["rel_path"], "sha256": v["sha256"]} for k, v in pin.pinned_models(p).items()},
                     "frozen_config": p["frozen_config"], "threshold": p["threshold"],
                     "pin_verify": "OK (domains.production.zeroshot.pin.verify returned [])"}
    assert out["zs_lane_v3"]["code_fingerprint_sha256"].startswith("871f23fa") and out["zs_lane_v3"]["pin"]["sha256"].startswith("711bd079")
    assert out["zs_lane_v1"]["code_fingerprint_sha256"].startswith("c712c15c") and out["zs_lane_v1"]["pin"]["sha256"].startswith("0087fb23")
    assert out["zs_lane_v3"]["revision"] == 3
    scorer = ["research/broad_baseline/score.py", "research/broad_baseline/bb_common.py",
              "research/broad_baseline/score_broad.py", "research/zeroshot/zs_lane_v1/scoring.py",
              "scripts/evaluate_f1_against_manual_gold.py"]
    sys.path.insert(0, str(R / "research/broad_baseline"))
    import bb_common as bb  # noqa: E402
    matcher = {"angle_deg": bb.MATCH_ANGLE_DEG, "offset_mm": bb.MATCH_OFFSET_MM,
               "gold_overlap": bb.MATCH_GOLD_OVERLAP, "pred_overlap": bb.MATCH_PRED_OVERLAP}
    assert matcher == {"angle_deg": 5.0, "offset_mm": 4.0, "gold_overlap": 0.6, "pred_overlap": 0.6}, matcher
    return freeze.append_entry(LEDGER, kind="FREEZE", recorded_by=BY,
        label="FREEZE: final held-out evaluation engines (zs_lane_v3 rev3 primary, zs_lane_v1 comparator), matcher "
              "5 deg / 4 mm / 0.6, family bootstrap n 10000 seed 20261018",
        items={"final_eval_engines": out,
               "final_eval_cad": "none: no-CAD first; no STEP exists for any member (SLDPRT only); E3 not run",
               "final_eval_matcher": dict(matcher, function="research/broad_baseline/score.py score_case",
                                          locus="apex", files={f: sha(R / f) for f in scorer}),
               "final_eval_bootstrap": {"unit": "family (part_id[:5])", "n": 10000, "seed": 20261018,
                                        "ci": "95 % percentile", "function": "research/zeroshot/zs_lane_v1/scoring.py "
                                        "aggregate/bootstrap with n_boot=10000, seed=20261018 passed explicitly "
                                        "(module default seed 20261025 not used)"},
               "final_eval_run_guard": {"research/zeroshot/common/gold_guard.py": sha(R / "research/zeroshot/common/gold_guard.py"),
                                        "docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh":
                                            sha(R / "docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh")}})


if __name__ == "__main__":
    e = {"note": note, "freeze": freeze_engines}[sys.argv[1]]()
    print(e["seq"], e["kind"], e["entry_id"], e["entry_sha256"])
    print(freeze.verify_ledger(LEDGER))
