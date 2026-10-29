"""Build cad_crosscheck.json from the V10 runs, the registration/match runs and
BENCHMARK_V2 metadata. Read-only on inputs."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "docs/production/runlog/benchmark_v2"
O = ROOT / "docs/production/runlog/broad_baseline"
FULL_HINTS = {"FULL", "GOOD_V", "DEFECTIVE_X", "LIDAR_DEMO_CAPTURE"}
STAGED_HINTS = {"PARTIAL_SCAN_1", "PARTIAL_SCAN_2", "STAGE0_BLANK", "PRELIM"}


def stage_class(h):
    h = set(h)
    if h & STAGED_HINTS: return "STAGED"
    if any(x.startswith("REPEAT_") for x in h): return "FULL_REPEAT_SCAN"
    if h & FULL_HINTS: return "FULL"
    if not h: return "FULL_PRESUMED_NO_STAGE_HINT"
    return "UNKNOWN:" + ",".join(sorted(h))


def lines_dir(a):
    ld = (a.get("reverify_2026_10_17") or {}).get("lines_dir") or a["disk_at_manifest_time"]["label_lines_dir"]
    return Path(ld) if ld.startswith("/") else ROOT / ld


def presence(cb):
    m = cb["scan_measure"]
    if cb["scan_support_3mm"] < 0.5: return "NOT_IN_SCAN"
    if m.get("status") != "OK": return "UNMEASURABLE:" + m.get("status", "?")
    a = m["scan_bend_angle_deg"]
    if a >= max(5.0, 0.5 * cb["angle_deg"]): return "FORMED"
    if a < 3.0: return "FLAT_NOT_FORMED"
    return "WEAK"


def main():
    d = json.loads((B / "BENCHMARK_V2.json").read_text())
    parts, scans = [], []
    for raw in sorted((O / "cad_crosscheck_raw").glob("[0-9]*.json")):
        pid = raw.stem; R = json.loads(raw.read_text())
        P = d["parts"][pid]
        parts.append({"part_id": pid, "step_path": R["step_path"], "v10_bend_count": R["v10_bend_count"],
                      "lenient_bend_count": R["lenient_bend_count"],
                      "v10_angles_deg": [round(a["angular_sweeps_deg"][0], 1) for a in R["v10_bend_axes"]],
                      "v10_spans_mm": [round(a["axial_span_mm"], 1) for a in R["v10_bend_axes"]],
                      "lenient_extra_candidates": [{"radii_mm": g["radii_mm"], "sweep_deg": max(g["sweep_by_radius_deg"].values()), "span_mm": g["max_axial_span_mm"]}
                                                   for g in R["lenient_coaxial_groups"] if g["kind"] == "BEND_CANDIDATE"
                                                   and not set(g["face_indices"]) & {int(x) for a in R["v10_bend_axes"] for x in a["bend_id"].split("_F", 1)[1].split("_")}],
                      "extractor_elapsed_s": R["elapsed_s"]})
        for s in P["scans"]:
            if s.get("role") != "benchmark_scan": continue
            a = s.get("ai_label_run")
            row = {"scan_id": s["scan_id"], "part_id": pid, "file": s["paths"][0].split("/")[-1], "stage_hints": s["stage_hints"],
                   "stage_class": stage_class(s["stage_hints"]), "primary_truth": s["primary_truth"],
                   "ai_status": (a or {}).get("final_status"), "ai_count": (a or {}).get("n_bends_labelled"),
                   "cad_v10_count": R["v10_bend_count"], "cad_lenient_count": R["lenient_bend_count"],
                   "gold_counts_other_tiers": [{"ref": g.get("ref") or g.get("gold11_case"), "n": g.get("n_lines", g.get("n_refs")), "tier": g.get("tier")}
                                               for g in (s.get("gold_detail") or []) if g.get("n_lines") or g.get("n_refs")]}
            mp = O / "cad_crosscheck_match" / f"{s['scan_id']}.json"
            if a and mp.exists():
                M = json.loads(mp.read_text())
                L = json.loads((lines_dir(a) / "lines.json").read_text())
                ai = [b for b in L.get("bends", []) if b.get("status") == "ok" and b.get("label_line")]
                row["ai_count_ok_lines"] = len(ai)
                row["lines_json"] = str(lines_dir(a) / "lines.json")
                row["registration"] = M["registration"]
                Rm, tm = np.array(M["R"]), np.array(M["t"])
                cadb = []
                allc = [*[(f"C{i+1}", x) for i, x in enumerate(R["v10_bend_axes"])]]
                geo = {}
                for i, x in enumerate(R["v10_bend_axes"]):
                    geo[f"C{i+1}"] = (np.array(x["line_start"]), np.array(x["line_end"]))
                k = 0
                v10f = {int(x) for ax in R["v10_bend_axes"] for x in ax["bend_id"].split("_F", 1)[1].split("_")}
                for g in R["lenient_coaxial_groups"]:
                    if g["kind"] == "BEND_CANDIDATE" and not set(g["face_indices"]) & v10f:
                        k += 1; geo[f"L{k}"] = (np.array(g["canonical_line"][0]), np.array(g["canonical_line"][1]))
                for cb in M["cad_bends"]:
                    cb = dict(cb); cb["presence_in_scan"] = presence(cb)
                    if cb["matched_ai_bend"] is None:
                        s0, e0 = geo[cb["cad_id"]]; cs, ce = Rm @ s0 + tm, Rm @ e0 + tm
                        cdir = (ce - cs) / np.linalg.norm(ce - cs); clen = np.linalg.norm(ce - cs)
                        for b in ai:
                            p, q = np.array(b["label_line"]["start"]), np.array(b["label_line"]["end"])
                            adir = (q - p) / np.linalg.norm(q - p)
                            ang = math.degrees(math.acos(min(1, abs(float(adir @ cdir)))))
                            w = (p + q) / 2 - cs; perp = float(np.linalg.norm(w - cdir * (w @ cdir)))
                            # perpendicular distance of the CAD midpoint to the AI line
                            cm = (cs + ce) / 2 - p; perp2 = float(np.linalg.norm(cm - adir * (cm @ adir)))
                            u0, u1 = sorted([float((p - cs) @ cdir), float((q - cs) @ cdir)])
                            cov = max(0.0, min(u1, clen) - max(u0, 0.0)) / clen
                            if ang < 12 and perp2 < 10 and cov >= 0.6:
                                cb["covered_by_merged_ai_line"] = b["bend_id"]; break
                    cadb.append(cb)
                row["cad_bends"] = cadb
                row["unmatched_ai_lines"] = M["unmatched_ai_lines"]
                missed = [c for c in cadb if c["matched_ai_bend"] is None and not c.get("covered_by_merged_ai_line") and c["presence_in_scan"] == "FORMED"]
                merged = [c for c in cadb if c.get("covered_by_merged_ai_line")]
                flat = [c for c in cadb if c["matched_ai_bend"] is None and not c.get("covered_by_merged_ai_line") and c["presence_in_scan"] == "FLAT_NOT_FORMED"]
                unscanned = [c for c in cadb if c["matched_ai_bend"] is None and not c.get("covered_by_merged_ai_line") and c["presence_in_scan"].startswith(("NOT_IN_SCAN", "UNMEASURABLE", "WEAK"))]
                row["auto"] = {"cad_bends_missed_by_ai_but_formed_in_scan": [c["cad_id"] for c in missed],
                               "cad_bends_covered_by_merged_collinear_ai_line": [(c["cad_id"], c["covered_by_merged_ai_line"]) for c in merged],
                               "cad_bends_unmatched_flat_in_scan": [c["cad_id"] for c in flat],
                               "cad_bends_unmatched_not_measurable": [(c["cad_id"], c["presence_in_scan"]) for c in unscanned],
                               "ai_lines_unmatched_to_cad": [u["bend_id"] for u in M["unmatched_ai_lines"]]}
            scans.append(row)
    out = {"parts": parts, "scans": scans}
    (O / "cad_crosscheck_auto.json").write_text(json.dumps(out, indent=1))
    for r in scans:
        au = r.get("auto", {})
        print(r["part_id"], r["scan_id"], r["stage_class"][:12], r["ai_status"], "ai", r["ai_count"], "v10", r["cad_v10_count"], "len", r["cad_lenient_count"],
              "reg", (r.get("registration") or {}).get("scan_inlier_frac_2mm"), (r.get("registration") or {}).get("cad_coverage_2mm"),
              "| miss", au.get("cad_bends_missed_by_ai_but_formed_in_scan"), "merge", au.get("cad_bends_covered_by_merged_collinear_ai_line"),
              "flat", au.get("cad_bends_unmatched_flat_in_scan"), "nm", au.get("cad_bends_unmatched_not_measurable"), "aiX", au.get("ai_lines_unmatched_to_cad"))


if __name__ == "__main__":
    main()
