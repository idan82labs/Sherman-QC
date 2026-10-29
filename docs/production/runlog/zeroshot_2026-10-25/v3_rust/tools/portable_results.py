"""Assemble portable_results.json from the portable-binary runs (engineering evidence; DEV only for scoring).
usage: .venv/bin/python -B docs/.../v3_rust/tools/portable_results.py WORK_DIR OUT.json
WORK_DIR holds the comparison outputs written by the steps in PORTABLE.md section 9 (cmp_*.json, match_*.json,
ident_*.json, oof_*/, final/speed_*/<bin>/runs.json + t<N>/<sid>/run_record.json, golden_*.json, abl / probe notes)."""
import json, statistics, sys
from pathlib import Path

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path(__file__).resolve().parents[6]
RL = REPO / "docs/production/runlog/zeroshot_2026-10-25/v3_rust"


def J(p):
    p = W / p
    return json.loads(p.read_text()) if p.is_file() else None


def cmp_summary(c):
    if not c:
        return None
    non_incl = sorted({p for v in c["diff"].values() for p in v["problems"] if ".incl " not in p})
    return {"scans": c["scans"], "byte_identical_lines": c["identical"], "max_endpoint_mm": c["max_endpoint_mm"],
            "max_score": c["max_score"], "n_lines_changed": sum(1 for v in c["diff"].values() for p in v["problems"] if p.startswith("n_lines")),
            "status_changed": sum(1 for v in c["diff"].values() for p in v["problems"] if p.startswith("status")),
            "scans_incl_last_bits": sum(1 for v in c["diff"].values() if any(".incl " in p for p in v["problems"])),
            "non_incl_field_changes": non_incl,
            "scans_score_gt_1e-9": sorted(((round(v["max_score"], 6), k) for k, v in c["diff"].items() if v["max_score"] > 1e-9), reverse=True),
            "point_cloud_not_run_by_binary": c.get("missing", [])}


def match_summary(m):
    if not m:
        return None
    return {"scans": m["scans"], "line_sets_equal": m["line_sets_equal"], "differ": list(m["differ"]),
            "ref_aggregate": m["ref_aggregate"], "run_aggregate": m["run_aggregate"],
            "point_cloud_from_reference": m["point_cloud_from_reference"]}


def oof_summary(d):
    if not d:
        return None
    s = json.loads((d / "REV2_OOF_SUMMARY.json").read_text())
    f, r = s["rev2_fixed_v3_fold_models"]["agg"], s["rev2"]["agg"]
    pick = lambda a: {k: a[k] for k in ("micro_f1", "family_macro_f1", "tp", "fp", "fn")} | {"control_fp": a["controls"]["fp_lines"]}  # noqa: E731
    return {"fixed_v3_fold_models": pick(f), "refit": pick(r),
            "files_identical_to_accelerate": {n: (d / n).read_bytes() == (RL / "oof" / n).read_bytes()
                                              for n in ("OOF_v3r2_lines.json", "OOF_v3r2_per_case.json")}}


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def speed(run_dir):
    d = W / run_dir
    if not d.is_dir():
        return None
    out = {}
    bins = sorted(p.name for p in d.iterdir() if (p / "runs.json").is_file())
    runs = {b: json.loads((d / b / "runs.json").read_text()) for b in bins}
    ths = sorted({int(k.split("/t")[1]) for b in bins for k in runs[b]})
    for th in ths:
        row = {}
        common = None
        for b in bins:
            ks = {k.split("/t")[0] for k, v in runs[b].items() if k.endswith(f"/t{th}") and v["rc"] == 0}
            common = ks if common is None else common & ks
        for b in bins:
            w = [runs[b][f"{s}/t{th}"]["wall_s"] for s in sorted(common)]
            rss = [runs[b][f"{s}/t{th}"].get("peak_rss_gib") or 0 for s in sorted(common)]
            # stage timers summed over the scans (run_record timers_s)
            st = {}
            for s in sorted(common):
                rr = d / b / f"t{th}" / s / "run_record.json"
                if rr.is_file():
                    for k, v in json.loads(rr.read_text()).get("timers_s", {}).items():
                        st[k] = st.get(k, 0.0) + v["s"]
            row[b] = {"n": len(w), "p50_s": round(pct(w, 0.5), 3), "p95_s": round(pct(w, 0.95), 3), "max_s": round(max(w), 3),
                      "sum_s": round(sum(w), 2), "peak_rss_gib_max": round(max(rss), 3),
                      "stage_sums_s": {k: round(v, 2) for k, v in sorted(st.items(), key=lambda x: -x[1])[:14]}}
        if "portable" in row and "accelerate" in row:
            ratios = [runs["portable"][f"{s}/t{th}"]["wall_s"] / runs["accelerate"][f"{s}/t{th}"]["wall_s"] for s in sorted(common)]
            row["portable_over_accelerate"] = {"median_per_scan": round(statistics.median(ratios), 3),
                                               "sum": round(row["portable"]["sum_s"] / row["accelerate"]["sum_s"], 3)}
        out[f"t{th}"] = row
    return out


res = {"schema": "sherman.zs_rust.portable_results.v1", "date": "2026-10-26"}
res["binaries"] = J("binaries.json")
res["drift"] = {
    "rev2": {"dev_eval_lines_vs_python": cmp_summary(J("cmp_final_r2_t16.json")),
             "dev_matcher_vs_python": match_summary(J("match_final_r2.json")),
             "dev_oof": oof_summary(W / "oof_final_r2" if (W / "oof_final_r2").is_dir() else None)},
    "rev3": {"dev_eval_lines_vs_python": cmp_summary(J("cmp_final_r3_t16.json")),
             "dev_matcher_vs_python": match_summary(J("match_final_r3.json")),
             "dense_vs_accelerate_build_t16": cmp_summary(J("cmp_final_dense_t16.json")),
             "dense_vs_accelerate_build_t1": cmp_summary(J("cmp_final_dense_t1.json"))},
    "ablation_rev2_dev_t16": {
        "libm_only (Accelerate kept)": {"eval_lines": cmp_summary(J("cmp_abl_libm_only.json")), "oof": oof_summary(W / "oof_abl_libm_only")},
        "linalg_only (system libm kept)": {"eval_lines": cmp_summary(J("cmp_abl_linalg_only.json")), "oof": oof_summary(W / "oof_abl_linalg_only")}},
    "linalg_bit_agreement_with_accelerate": J("laprobe.json"),
}
res["cross_arch"] = J("cross_arch.json")
res["speed"] = {"dev_rev3": speed("final/speed_dev_r3"), "dense_rev3": speed("final/speed_dense_r3")}
res["golden"] = J("golden_runs.json")
res["windows"] = J("windows.json")
OUT.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in res.items()}, indent=1)[:2000])
