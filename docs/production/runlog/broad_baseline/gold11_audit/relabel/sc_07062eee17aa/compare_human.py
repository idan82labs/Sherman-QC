"""Gold11 relabel check for sc_07062eee17aa (read-only on every input; writes one JSON next to it).

Usage: OMP_NUM_THREADS=1 .venv/bin/python -I compare_human.py LINES_DIR OUT_JSON

Inputs (never modified):
  - human gold   data/bend_corpus/annotations/human_bend_lines_v1/a962166c3019434c10628722.json
  - relabel      LINES_DIR/lines.json (bench_labeling 'lines' output, label_line = apex locus)
  - raw scan     source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply (sha256 checked)
  - auditor apex lines (independent fits): auditor1 a1_apex_lines_raw.json, auditor3 a3x_apexlines.json

Outputs per relabel line: frozen-matcher metrics against every human line and the
Hungarian assignment (scripts/evaluate_f1_against_manual_gold.match_lines, default
Thresholds = 5 deg / 4 mm / 60 % span), distance from the line to the scanned
surface (nearest raw vertex, every 1 mm), and agreement with the auditors' apex lines.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics  # noqa: E402

HUMAN = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1/a962166c3019434c10628722.json"
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply")
SCAN_SHA = "07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880"
AUD = ROOT / "docs/production/runlog/broad_baseline/gold11_audit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        head = fh.read(8192)
        k = head.index(b"end_header")
        k = head.index(b"\n", k) + 1
        lines = head[:k].decode("ascii", "replace").splitlines()
        props = [ln.split()[-1] for ln in lines if ln.startswith("property") and "list" not in ln]
        nv = int([ln for ln in lines if ln.startswith("element vertex")][0].split()[-1])
        assert "binary_little_endian" in head[:k].decode("ascii", "replace")
        assert all(" float" in ln or " float32" in ln for ln in lines
                   if ln.startswith("property") and "list" not in ln), "expects float vertex props"
        fh.seek(k)
        vert = np.frombuffer(fh.read(nv * 4 * len(props)), dtype="<f4").reshape(nv, len(props))
    return vert[:, :3].astype(np.float64)


def line_of(b: dict) -> np.ndarray:
    return np.asarray([b["label_line"]["start"], b["label_line"]["end"]], dtype=float)


def main(lines_dir: str, out_json: str) -> int:
    lines_doc = json.loads((Path(lines_dir) / "lines.json").read_text())
    human = json.loads(HUMAN.read_text())
    assert lines_doc["scan_sha256"] == SCAN_SHA
    assert sha256(SCAN) == SCAN_SHA
    ok = [b for b in lines_doc["bends"] if b.get("status") == "ok"]
    gold = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]}
            for b in human["bends"]]
    pred = [{"bend_id": b["bend_id"], "line_start": b["label_line"]["start"],
             "line_end": b["label_line"]["end"]} for b in ok]
    th = Thresholds()
    result = match_lines(gold, pred, th)

    from scipy.spatial import cKDTree

    verts = load_vertices(SCAN)
    tree = cKDTree(verts)

    a1 = json.loads((AUD / "auditor1/sc_07062eee17aa/a1_apex_lines_raw.json").read_text())
    a3 = json.loads((AUD / "auditor3/sc_07062eee17aa/a3x_apexlines.json").read_text())["folds"]
    auditor_lines = {f"a1:{k}": np.asarray([v["line_start"], v["line_end"]]) for k, v in a1.items()}
    for k, v in a3.items():
        key = "apex_line_raw" if "apex_line_raw" in v else "axis_line_raw_mid_surface"
        auditor_lines[f"a3:{k}"] = np.asarray(v[key])

    per_line = []
    for b in ok:
        seg = line_of(b)
        n = max(2, int(np.ceil(np.linalg.norm(seg[1] - seg[0]))) + 1)
        samples = seg[0] + np.linspace(0.0, 1.0, n)[:, None] * (seg[1] - seg[0])
        dist, _ = tree.query(samples, k=1)
        vs_human = {}
        for g in gold:
            m = pair_metrics(np.asarray([g["line_start"], g["line_end"]], float), seg)
            vs_human[g["bend_id"]] = {k: round(v, 3) for k, v in m.items()}
        vs_aud = {}
        for name, al in auditor_lines.items():
            m = pair_metrics(al, seg)
            if m["axis_angle_deg"] <= 10 and m["transverse_offset_mm"] <= 10:
                vs_aud[name] = {k: round(v, 3) for k, v in m.items()
                                if k in ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap",
                                         "prediction_span_overlap", "gold_length_mm", "prediction_length_mm")}
        per_line.append({
            "bend_id": b["bend_id"],
            "note": b.get("note", ""),
            "patches": b["patches"],
            "label_locus": b["label_locus"],
            "skin": b["skin"],
            "confidence": b["confidence"],
            "confidence_reasons": b["confidence_reasons"],
            "included_angle_deg": round(b["included_angle_deg"], 2),
            "length_mm": round(b["length_mm"], 2),
            "extent_source": b["extent_source"],
            "t_range": [round(v, 2) for v in b["t_range"]],
            "t_range_auto": None if b["t_range_auto"] is None else [round(v, 2) for v in b["t_range_auto"]],
            "coverage_patch_pair": round(b["coverage"], 3),
            "fold_support_frac": round(b["fold_support_frac"], 3),
            "fold_support_gaps_t": b["fold_support_gaps_t"],
            "apex_offset_mm": None if b["apex_offset_mm"] is None else round(b["apex_offset_mm"], 3),
            "expected_apex_offset_mm": {k: round(v, 3) for k, v in b["expected_apex_offset_mm"].items()},
            "surface_dist_mm": {"median": round(float(np.median(dist)), 3),
                                "p90": round(float(np.percentile(dist, 90)), 3),
                                "max": round(float(dist.max()), 3),
                                "frac_le_0p5": round(float((dist <= 0.5).mean()), 3),
                                "n_samples": int(n)},
            "vs_human": vs_human,
            "vs_auditor_apex_lines": vs_aud,
            "line": {"start": b["label_line"]["start"], "end": b["label_line"]["end"]},
        })
    doc = {
        "schema": "sherman.gold11_audit.relabel_vs_human.v1",
        "scan_id": "sc_07062eee17aa",
        "scan_sha256": SCAN_SHA,
        "human_gold": str(HUMAN.relative_to(ROOT)),
        "human_gold_sha256": sha256(HUMAN),
        "lines_json": str((Path(lines_dir) / "lines.json").resolve()),
        "lines_json_sha256": sha256(Path(lines_dir) / "lines.json"),
        "matcher": "scripts/evaluate_f1_against_manual_gold.match_lines (Thresholds default: 5 deg, 4 mm, 60%/60%)",
        "human_count": len(gold),
        "relabel_count": len(pred),
        "match_human_as_gold": result,
        "per_relabel_line": per_line,
    }
    Path(out_json).write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"tp": result["true_positive"], "fp": result["false_positive"],
                      "fn": result["false_negative"], "matches": [(m["gold_id"], m["prediction_id"])
                                                                  for m in result["matches"]],
                      "unmatched_human": result["unmatched_gold_ids"],
                      "unmatched_relabel": result["unmatched_prediction_ids"]}, indent=1))
    for row in per_line:
        print(row["bend_id"], row["skin"], row["confidence"], row["length_mm"], "cov", row["coverage_patch_pair"],
              "support", row["fold_support_frac"], "surf", row["surface_dist_mm"]["median"],
              row["surface_dist_mm"]["p90"], row["surface_dist_mm"]["max"], "apex_off", row["apex_offset_mm"],
              "aud", {k: (v["axis_angle_deg"], v["transverse_offset_mm"]) for k, v in row["vs_auditor_apex_lines"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
