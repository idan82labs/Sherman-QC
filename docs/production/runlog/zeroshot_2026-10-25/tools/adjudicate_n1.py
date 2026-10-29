"""Gold batch N1 adjudication (2026-10-25): A/B match with the frozen matcher, canary QC, final lines, labels
(new root ai_bend_lines_newscans_n1_v1), provenance sidecars, per-item adjudication.json, gold-ledger freeze
(tools/gold_ledger_newscans.py) and gold/N1/ADJUDICATION_N1.json. No engine/lane/CAD output is read.
Verdicts below were taken by the adjudicator from the marker renders/close-ups/sections (see DOUBTS).
usage (repo root): .venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/tools/adjudicate_n1.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO), str(REPO / "research/broad_baseline")]
import bb_common as bb  # noqa: E402
import score as S  # noqa: E402
from research.bench_labeling.label import write_label  # noqa: E402

P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
G = P / "gold"
N1 = G / "N1"
ROOT = REPO / "data/bend_corpus/annotations/ai_bend_lines_newscans_n1_v1"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
BY = "N1/adjudicator"
PROTOCOL = ("PLAN.md 2.4-2.6; frozen matcher research/broad_baseline/score.py score_case (5 deg / 4 mm / 60 % both "
            "spans); accept threshold 1.5 mm / 2 deg")

DOUBTS = {
    "gb_4dd5ba": [
        {"lines_A": ["B25", "B26"], "lines_B": ["B25", "B26"], "doubt": "shallow edge joggle: joggle vs emboss",
         "verdict": "KEEP_2_LINES_JOGGLE",
         "reason": "A closeup_B25/B26 sections show both skins following an S: plate -> steep web (~75-81 deg each "
                   "way) -> strip offset ~3 mm (3t), constant along the 38 mm strip. The strip is free on three "
                   "sides (window cut), so it is a lanced strip offset by two formed folds, not an emboss (an emboss "
                   "is a closed stretched bump with no free edge and no flat second plate). GOLD_SET_V2: joggle = 2 "
                   "lines, folds >= 10 deg count. Both markers independently gave the same 2 lines (offsets 0.02 / "
                   "0.01 mm)."},
        {"lines_A": ["B1"], "lines_B": ["B19"], "doubt": "end-flange apex position (plate outer skin unscanned)",
         "verdict": "KEEP_A_POSITION",
         "reason": "Two independent constructions (A: plane_from_points on the inner plate skin; B: margin 32 mm "
                   "onto plate 2 + extent_xyz) agree to 0.37 mm / 0.04 deg; both close-ups put the dot on the outer "
                   "corner of the 90 deg flange. The residual outer-apex uncertainty (<= ~1 mm) is far inside the "
                   "4 mm matcher tolerance. Line from A (batch convention), confidence stays uncertain."},
        {"lines_A": ["B12", "B14", "B16", "B18", "B20", "B22"], "lines_B": ["B3", "B6", "B9", "B12", "B15", "B18"],
         "doubt": "hook lips (5 mm, larger radius)", "verdict": "KEEP_6_LINES",
         "reason": "A closeup_B12/B14 and sweep/sec_legs: at every leg the foot (~13 mm) and lip (~5-10 mm flank) "
                   "are two flat plates with a defined apex, ~116 deg bend; both skins visible. A formed fold >= 10 "
                   "deg between two plates. B's ~1 mm tip curl beyond the lip is edge rounding and stays rejected. "
                   "B's understated angles on the left-leg hairpins do not affect position (A/B offsets <= 0.71 mm)."},
    ],
    "gb_e94771": [
        {"doubt": "roll-to-flat run-outs at the wing ends (marker B)", "verdict": "NOT_A_BEND",
         "reason": "A sweep/prof_axis: normal angle linear in arc length over the whole roll, then a tangent run-out "
                   "into the ~45 mm flats with no step; rolled shell, no localized fold (GOLD_SET_V2 rolled kinks "
                   "are not bends)."}],
    "gb_196152": [
        {"doubt": "none (rolled half-cylinder)", "verdict": "ZERO_BEND",
         "reason": "A sweep/prof_axis at 3 stations: normal angle linear end to end (few-deg roll ripples only)."}],
    "gb_550b16": [
        {"doubt": "D-bulkhead to shell junction", "verdict": "NOT_A_BEND",
         "reason": "Separate ~2 mm D plate T-joined inside the shell (patches 1/2, right view): weld-plate edge, not a "
                   "formed fold; shell itself is a smooth roll."}],
    "gb_3f9a9c": [
        {"lines_A": ["B2"], "lines_B": ["B2"], "doubt": "canary B2 softness", "verdict": "KEEP",
         "reason": "Clear apex ~17 deg along the full length in both markers; matches canary gold B1 (0.33 mm)."}],
}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ok_lines(doc):
    return [b for b in doc["bends"] if b.get("status") == "ok" and b.get("label_line")]


def score(goldspec, pred_bends):
    regs = [bb.as_line(b["bend_id"], b["label_line"]["start"], b["label_line"]["end"]) for b in pred_bends]
    case = {"case_id": "x", "family_id": "x", "tier": "N1_ADJ", "gold": goldspec, "label_complete": True}
    return S.score_case(case, frame_records=None,
                        preloaded_predictions={"owned_bend_regions": regs, "status": "OK", "scan": {}, "lane": "m"})


def f1_of(r):
    if r["disposition"] == "ZERO_BEND_CONTROL":
        return 1.0 if r["prediction_count"] == 0 else 0.0
    return r["f1"]


def main():
    batches = json.loads((G / "batches.json").read_text())
    items = [b for b in batches["batches"] if b["id"] == "N1"][0]["items"]
    can = json.loads((G / "canaries.json").read_text())
    ckey = [c for c in can["canaries"] if c["batch"] == "N1"][0]
    key = json.loads((P / "newscans_N1/PRIVATE_KEY.json").read_text())["key"]
    summary = {"schema": "sherman.zeroshot.gold_batch_adjudication.v1", "batch": "N1", "adjudicated_at": NOW,
               "adjudicator": f"{BY} (claude-vision-ai-v1)", "protocol": PROTOCOL,
               "not_seen": "no engine, lane, A0, pool, E3, best_f1, CAD or baseline output (none exists for N1)",
               "per_item": {}, "frozen": {}}
    tot = {"tp": 0, "nA": 0, "nB": 0, "accepted": 0, "outside": 0}
    for it in items:
        gid = it["gid"]
        d = N1 / gid
        pa, pb = d / "marker_A/lines/lines.json", d / "marker_B/lines/lines.json"
        A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
        la, lb = ok_lines(A), ok_lines(B)
        r = score({"kind": "bench_lines", "path": str(pa)}, lb)
        zero = r["disposition"] == "ZERO_BEND_CONTROL"
        matches = [] if zero else r["matches"]
        accepted = [m for m in matches if m["transverse_offset_mm"] <= 1.5 and m["axis_angle_deg"] <= 2.0]
        outside = [m for m in matches if m not in accepted]
        ab = {"nA": len(la), "nB": len(lb), "tp": 0 if zero else r["true_positive"], "accepted": len(accepted),
              "matched_not_accepted": [[m["gold_id"], m["prediction_id"]] for m in outside],
              "A_only": [] if zero else r["unmatched_gold_ids"], "B_only": [] if zero else r["unmatched_prediction_ids"],
              "f1": 1.0 if zero and not lb else f1_of(r), "zero_bend_both": zero and not la and not lb,
              "matches": [[m["gold_id"], m["prediction_id"], round(m["transverse_offset_mm"], 3),
                           round(m["axis_angle_deg"], 3), round(m["gold_span_overlap"], 3),
                           round(m["prediction_span_overlap"], 3)] for m in matches]}
        assert ab["f1"] == 1.0 and not ab["A_only"] and not ab["B_only"], (gid, ab)
        for k, v in (("tp", ab["tp"]), ("nA", ab["nA"]), ("nB", ab["nB"]), ("accepted", ab["accepted"]),
                     ("outside", len(outside))):
            tot[k] += v
        # final lines = marker A (all lines AGREED within 1.5 mm / 2 deg; batch convention "line from A")
        fin = copy.deepcopy(A)
        n = len(la)
        fin["notes"] = (f"Zero-shot gold N1 adjudicated (PLAN 2.4): blind markers A and B, {n} lines"
                        f"{' (no_bends_present)' if n == 0 else ''}; A/B F1 1.0; adjudication record in "
                        f"gold/N1/{gid}/adjudication.json.")
        fin["adjudicated_at"] = NOW
        fin["marker_lines_sha256"] = {"A": sha(pa), "B": sha(pb)}
        fdir = d / "final/lines"
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "lines.json").write_text(json.dumps(fin, indent=1) + "\n")
        fsha = sha(fdir / "lines.json")
        bmap = {m["gold_id"]: m for m in matches}
        lines = []
        for b in la:
            m = bmap[b["bend_id"]]
            lines.append({"final_id": b["bend_id"], "marker_A_id": b["bend_id"], "marker_B_id": m["prediction_id"],
                          "from": "A", "verdict": "AGREED", "ab_offset_mm": round(m["transverse_offset_mm"], 3),
                          "ab_angle_deg": round(m["axis_angle_deg"], 3),
                          "ab_span_overlap": [round(m["gold_span_overlap"], 3), round(m["prediction_span_overlap"], 3)],
                          "bend_angle_deg_A": round(b["bend_angle_deg"], 2), "length_mm_A": round(b["length_mm"], 1),
                          "reason": "A/B matched within 1.5 mm / 2 deg; line from A."})
        is_canary = gid == ckey["gid"]
        adj = {"schema": "sherman.zeroshot.gold_adjudication.v1", "batch": "N1", "gid": gid,
               "scan_id": ckey["scan_id"] if is_canary else f"sc_{key[gid]['sha256'][:12]}",
               "portion": "DEV_CANARY" if is_canary else "NEW_UNSEEN_FAMILY_TEST",
               "part_id": None if is_canary else key[gid]["part_id"],
               "original_path": ckey["original_path"] if is_canary else key[gid]["original_path"],
               "adjudicator": f"{BY} (claude-vision-ai-v1)", "adjudicated_at": NOW, "protocol": PROTOCOL,
               "inputs_seen": ["marker_A and marker_B lines/pairs/notes/renders/sweeps",
                               "canary GOLD_SET_V2 gold (QC only)" if is_canary else "private key (gid -> original)"],
               "not_seen": "no engine, lane, A0, pool, E3, best_f1, CAD or baseline output",
               "marker_A_lines_sha256": sha(pa), "marker_B_lines_sha256": sha(pb), "final_lines_sha256": fsha,
               "final_line_count": n, "no_bends_present": n == 0, "ab_match": ab, "lines": lines,
               "doubts_adjudicated": DOUBTS.get(gid, [])}
        if is_canary:
            q = {m: score({"kind": "ai_studio", "path": str(REPO / ckey["gold_path"])}, L)
                 for m, L in (("A", la), ("B", lb), ("final", ok_lines(fin)))}
            adj["final_vs_reference"] = {"ref": "CANARY_GOLD", "f1": q["final"]["f1"], "tp": q["final"]["true_positive"],
                                         "n_ref": q["final"]["gold_count"], "n_final": n,
                                         "ref_only": q["final"]["unmatched_gold_ids"],
                                         "final_only": q["final"]["unmatched_prediction_ids"],
                                         "matches": [[m["gold_id"], m["prediction_id"], round(m["transverse_offset_mm"], 3),
                                                      round(m["axis_angle_deg"], 3)] for m in q["final"]["matches"]]}
            adj["canary_qc"] = {"gold_path": ckey["gold_path"], "gold_n_lines": ckey["gold_n_lines"],
                                "marker_A_f1": q["A"]["f1"], "marker_B_f1": q["B"]["f1"],
                                "adjudicated_f1": q["final"]["f1"], "rule": ">= 0.95",
                                "pass": min(q["A"]["f1"], q["B"]["f1"]) >= 0.95, "label_written": False}
            if not adj["canary_qc"]["pass"]:
                (d / "adjudication.json").write_text(json.dumps(adj, indent=1) + "\n")
                raise SystemExit("CANARY FAILED: batch invalid")
            summary["canary"] = {"gid": gid, "scan_id": ckey["scan_id"], **{k: adj["canary_qc"][k] for k in
                                 ("marker_A_f1", "marker_B_f1", "adjudicated_f1", "pass", "label_written")}}
        (d / "adjudication.json").write_text(json.dumps(adj, indent=1) + "\n")
        summary["per_item"][gid] = {"final_lines": n, "ab_f1": ab["f1"], "verdicts": [x["verdict"] for x in DOUBTS.get(gid, [])],
                                    "role": "CANARY" if is_canary else "NEW", "part_id": adj["part_id"]}
    summary["ab_agreement"] = {"micro_f1": 1.0, "tp": tot["tp"], "n_A": tot["nA"], "n_B": tot["nB"],
                               "pairs_accepted_within_1p5mm_2deg": tot["accepted"],
                               "matched_outside_tolerance": tot["outside"], "zero_bend_items_agreed": 3}
    summary["verdicts"] = {"ACCEPT_A": 0, "ACCEPT_B": 0, "MERGE": 0, "NOT_A_BEND": 0, "MISSING_ADD": 0,
                           "doubts": {v: sum(1 for L in DOUBTS.values() for x in L if x["verdict"] == v)
                                     for v in sorted({x["verdict"] for L in DOUBTS.values() for x in L})}}
    (N1 / "ADJUDICATION_N1.partial.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def labels_and_freeze():
    summary = json.loads((N1 / "ADJUDICATION_N1.partial.json").read_text())
    key = json.loads((P / "newscans_N1/PRIVATE_KEY.json").read_text())["key"]
    ROOT.mkdir(parents=True, exist_ok=True)
    for gid, v in key.items():
        d = N1 / gid
        fdir = d / "final/lines"
        adj = json.loads((d / "adjudication.json").read_text())
        res = write_label(fdir, v["original_path"], labels_root=ROOT, confirm_no_bends=adj["no_bends_present"])
        lp = Path(res["label_path"])
        lsha = sha(lp)
        pp = Path(res["provenance_path"])
        prov = json.loads(pp.read_text())
        assert prov["asset_sha256"] == v["sha256"] and prov["coordinate_match"] == "sha256"
        renders = {}
        for m in ("A", "B"):
            for png in sorted((d / f"marker_{m}").rglob("*.png")):
                renders[str(png.relative_to(P))] = sha(png)
        mk = {}
        for m in ("A", "B"):
            md = d / f"marker_{m}"
            mk[f"marker_{m}"] = {"lines_json": json.loads((md / "lines/lines.json").read_text()),
                                 "lines_sha256": sha(md / "lines/lines.json"),
                                 "pairs": json.loads((md / "pairs.json").read_text()),
                                 "pairs_sha256": sha(md / "pairs.json"),
                                 "notes_md": (md / "notes.md").read_text(),
                                 "dir": str(md.relative_to(REPO))}
        prov["zeroshot_gold"] = {"schema": "sherman.zeroshot.gold_provenance.v1", "batch": "N1", "gid": gid,
                                 "scan_id": adj["scan_id"], "portion": "NEW_UNSEEN_FAMILY_TEST",
                                 "part_id": v["part_id"], "original_path": v["original_path"],
                                 "gold_copy_path": str((N1 / "scans" / f"{gid}.ply").relative_to(REPO)), **mk,
                                 "adjudication": adj, "adjudication_sha256": sha(d / "adjudication.json"),
                                 "final_lines_sha256": sha(fdir / "lines.json"), "render_sha256": renders,
                                 "label_sha256": lsha}
        pp.write_text(json.dumps(prov, indent=2, allow_nan=False) + "\n")
        out = subprocess.run([str(REPO / ".venv/bin/python"), "-B", str(P / "tools/gold_ledger_newscans.py"), "freeze",
                              "--scan", adj["scan_id"], "--label-path", str(lp), "--label-sha256", lsha, "--by", BY,
                              "--note", f"N1 adjudicated gold ({adj['final_line_count']} lines); batch A/B F1 1.0; canary "
                                        f"F1 A/B/adj {summary['canary']['marker_A_f1']}/{summary['canary']['marker_B_f1']}/"
                                        f"{summary['canary']['adjudicated_f1']}; adjudication "
                                        f"{(d / 'adjudication.json').relative_to(REPO)}"],
                             cwd=REPO, capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(out.stderr + out.stdout)
        e = json.loads(out.stdout.splitlines()[0])
        summary["frozen"][gid] = {"scan_id": adj["scan_id"], "part_id": v["part_id"],
                                  "final_lines": adj["final_line_count"],
                                  "label_path": str(lp.relative_to(REPO)), "label_sha256": lsha,
                                  "provenance": str(pp.relative_to(REPO)), "ledger_seq": e["seq"],
                                  "entry_sha256": e["entry_sha256"]}
        print(gid, summary["frozen"][gid])
    summary["labels_root"] = str(ROOT.relative_to(REPO))
    summary["ledger_mechanism"] = ("tools/gold_ledger_newscans.py (REGISTER + FROZEN for NEW_UNSEEN_FAMILY_TEST; "
                                   "gold_guard.py untouched, its pinned hash in both lane fingerprints intact)")
    (N1 / "ADJUDICATION_N1.json").write_text(json.dumps(summary, indent=1) + "\n")
    (N1 / "ADJUDICATION_N1.partial.json").unlink()


if __name__ == "__main__":
    if sys.argv[1:] == ["labels-and-freeze"]:
        labels_and_freeze()
    else:
        main()
