#!/usr/bin/env python3
"""Build the authorized broad-benchmark case list (metadata + sha checks only).

Reads docs/production/runlog/benchmark_v2/BENCHMARK_V2.json and keeps ONLY scans
that are safe to run now:
  * part split CONSUMED and scan eval_tier CONSUMED (drops FRESH, HELD_EYAL50_OVERLAP,
    EXCLUDED_EYAL50_QUARANTINE and FRESH_SCAN_OF_CONSUMED_PART, which is evaluated once
    at the end like FRESH);
  * primary truth GOLD_LINES_HUMAN restricted to the 11 Gold11 cases (the other human
    gold scans need an explicit --include-human-non-gold11), or AI_GOLD_ACCEPTED;
  * never a path under data/bend_corpus/incoming_codex_fetch_2026-10-17.
Broad run (v2, 2026-10-17): also drops role != benchmark_scan (EXCLUDED_NOT_A_SCAN ...),
quality flags POSSIBLE_CAD_REFERENCE_MESH / SYNTHETIC_RESAMPLE_NOT_A_SCAN, honours
primary_truth AI_GOLD_PARTIAL (label_complete=false, recall-only), prefers the review11
final_lines_dir when no studio label exists, records owner flags, and writes a
case_list_sha256 over the identity of every kept case (scan sha, gold file sha, CAD sha).
Optional --review-verdicts adds scans from the human-review delegation once their
verdicts exist: {"sc_...": {"status": "ACCEPT"|"ACCEPT_PARTIAL", "lines": ".../lines.json"}}.
ACCEPT_PARTIAL becomes label_complete=false (recall/position only).

For every kept scan it resolves the exact file the gold coordinates are bound to
(AI studio label source + provenance asset_sha256, bench lines.json scan_sha256, or the
Gold11 catalog scan_sha256), verifies its sha256, picks a STEP/IGES CAD when the part
has one (runtime + CAD half of best_f1 need it), and writes the score.py gold spec.

  build_case_list.py --out docs/production/runlog/broad_baseline/case_list_consumed_v1.json
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import gold11_cases as g11  # noqa: E402

FORBIDDEN_FRAGMENTS = ("incoming_codex_fetch_2026-10-17", "/opaque/", "/evaluator_only/", "HOLDOUT10",
                       "holdout", "sealed")
ASSET_ROOTS = {
    "source-assets": bb.ROOT.parent / "source-assets",
    "data": bb.ROOT / "data",
    "uploads": bb.ROOT / "uploads",
    "live_scans": bb.ROOT / "live_scans",
    "output": bb.ROOT / "output",
}
CAD_PRIORITY = {"stp": 0, "step": 0, "igs": 1, "iges": 1}
GOLD11_BY_SHA = {g11.scan_sha(cid): cid for cid in g11.CASES}


EXCLUDED_QUALITY_FLAGS = {"POSSIBLE_CAD_REFERENCE_MESH", "SYNTHETIC_RESAMPLE_NOT_A_SCAN"}


def gold_path_of(gold: dict[str, Any]) -> Path:
    path = Path(gold["path"])
    return path if path.is_absolute() else bb.ROOT / path


def _owner_flags(bench: dict[str, Any]) -> dict[str, list[str]]:
    """Owner/review flags recorded in the manifest summary (review11) -> per-scan case flags."""
    out: dict[str, list[str]] = {}
    review = ((bench.get("summary") or {}).get("ai_label_run") or {}).get("review11_2026_10_17") or {}
    for item in review.get("owner_flags") or []:
        issue = str(item.get("issue") or "")
        if "synthetic provenance" in issue or "_write_lidar_demo_scan" in issue:
            tag = "OWNER_FLAG_LIDAR_DEMO_SYNTHETIC_PROVENANCE"
        elif "probably misses" in issue or "re-reviewed" in issue:
            tag = "OWNER_FLAG_LABEL_MAY_MISS_A_BEND"
        else:
            continue
        for sid in item.get("scan_ids") or []:
            out.setdefault(sid, []).append(tag)
    return out


def _forbidden(path: str | None) -> bool:
    text = str(path or "")
    return any(fragment.lower() in text.lower() for fragment in FORBIDDEN_FRAGMENTS)


def _pick_cad(part: dict[str, Any], scan_name: str) -> tuple[str | None, list[str]]:
    files = [f for f in part.get("cad", {}).get("files", []) if f.get("format", "").lower() in CAD_PRIORITY]
    files = [f for f in files if not _forbidden(f.get("path")) and bb.expand(f["path"]).is_file()]
    if not files:
        return None, []
    stem = scan_name.split(".")[0].lower()

    def rank(f: dict[str, Any]):
        name = Path(f["path"]).stem.lower()
        common = len([1 for a, b in zip(name, stem) if a == b])
        return (CAD_PRIORITY[f["format"].lower()], -common, len(f["path"]), f["path"])

    ordered = sorted(files, key=rank)
    return str(bb.expand(ordered[0]["path"])), [str(bb.expand(f["path"])) for f in ordered[1:]]


def _ai_gold(scan: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None, str | None, str]:
    run = scan.get("ai_label_run") or {}
    studio = run.get("studio_label") or {}
    if studio.get("written") and studio.get("label_path"):
        label = bb.ROOT / studio["label_path"]
        payload = bb.read_json(label)
        prov_path = label.parent / "provenance" / label.name
        provenance = bb.read_json(prov_path) if prov_path.is_file() else {}
        root = ASSET_ROOTS.get(payload.get("source"))
        scan_path = (root / payload["source_relative_path"]).resolve() if root else None
        return ({"kind": "ai_studio", "path": str(label.relative_to(bb.ROOT))}, scan_path,
                provenance.get("asset_sha256"), "ai_studio")
    lines_dir = ((run.get("review11") or {}).get("final_lines_dir")
                 or (run.get("reverify_2026_10_17") or {}).get("lines_dir")
                 or (run.get("disk_at_manifest_time") or {}).get("label_lines_dir"))
    if lines_dir:
        lines = bb.expand(lines_dir if Path(lines_dir).is_absolute() else bb.ROOT / lines_dir) / "lines.json"
        if lines.is_file():
            payload = bb.read_json(lines)
            return ({"kind": "bench_lines", "path": str(lines.relative_to(bb.ROOT))},
                    Path(payload["scan_path"]), payload.get("scan_sha256"), "bench_lines")
    return None, None, None, "NO_GOLD_FILE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=bb.BENCHMARK_V2)
    parser.add_argument("--catalog", type=Path, default=bb.GOLD_CATALOG)
    parser.add_argument("--review-verdicts", type=Path)
    parser.add_argument("--include-human-non-gold11", action="store_true")
    parser.add_argument("--no-hash", action="store_true", help="skip sha256 verification of scan files")
    parser.add_argument("--keep-possible-cad-meshes", action="store_true",
                        help="v1 behaviour: keep POSSIBLE_CAD_REFERENCE_MESH scans as zero-bend controls")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    bench = bb.read_json(args.benchmark)
    catalog = {row["scan_sha256"]: row for row in bb.read_json(args.catalog)["cases"]}
    verdicts = bb.read_json(args.review_verdicts) if args.review_verdicts else {}
    cases, skipped = [], Counter()
    excluded_named: list[dict[str, Any]] = []
    owner_flags = _owner_flags(bench)
    for part_id, part in sorted(bench["parts"].items()):
        if part.get("split") != "CONSUMED":
            skipped[f"split:{part.get('split')}"] += len(part.get("scans") or [])
            continue
        for scan in part.get("scans") or []:
            sid = scan["scan_id"]
            if scan.get("eval_tier") != "CONSUMED":
                skipped[f"eval_tier:{scan.get('eval_tier')}"] += 1
                continue
            if any(_forbidden(p) for p in scan.get("paths") or []):
                skipped["forbidden_path"] += 1
                continue
            if scan.get("role") not in (None, "benchmark_scan"):
                skipped[f"role:{scan.get('role')}"] += 1
                excluded_named.append({"scan_id": sid, "reason": f"role:{scan.get('role')}"})
                continue
            bad_flags = sorted(set(scan.get("quality_flags") or []) & EXCLUDED_QUALITY_FLAGS)
            if bad_flags and not args.keep_possible_cad_meshes:
                skipped["quality_flag:" + "+".join(bad_flags)] += 1
                excluded_named.append({"scan_id": sid, "reason": "quality_flag:" + "+".join(bad_flags)})
                continue
            truth = scan.get("primary_truth")
            entry: dict[str, Any] = {"case_id": sid, "family_id": part_id, "scan_sha256": scan["sha256"],
                                     "format": scan.get("format"), "stage_hints": scan.get("stage_hints"),
                                     "staged_or_incomplete": scan.get("staged_or_incomplete"),
                                     "quality_flags": scan.get("quality_flags") or [],
                                     "in_gold11": scan["sha256"] in GOLD11_BY_SHA}
            flags: list[str] = []
            gold11_id = GOLD11_BY_SHA.get(scan["sha256"])
            if gold11_id is not None or truth == "GOLD_LINES_HUMAN":
                # Gold11 cases are authorized with their catalog lines whatever their manifest tier
                # (C05/C06 are Codex multiview labels); other human-gold scans need an explicit flag.
                row = catalog.get(scan["sha256"])
                if row is None:
                    skipped["human_gold_not_in_catalog"] += 1
                    continue
                if gold11_id is None and not args.include_human_non_gold11:
                    skipped["human_gold_outside_gold11 (needs --include-human-non-gold11)"] += 1
                    continue
                entry.update({"tier": "GOLD11" if gold11_id else "GOLD_LINES_HUMAN_NON_GOLD11",
                              "gold11_id": gold11_id,
                              "gold": {"kind": "gold11_catalog", "path": str(args.catalog.relative_to(bb.ROOT)),
                                       "scan_sha256": scan["sha256"]},
                              "label_complete": True, "zero_bend_control": False,
                              "scan_path": row["scan_path"], "gold_scan_sha256": row["scan_sha256"],
                              "label_frame": row["coordinate_frame"], "gold_line_count": row["bend_count"],
                              "gold_tier_detail": row.get("truth_tier")})
                if row.get("truth_tier") != "authoritative_user_verified":
                    flags.append("GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN")
            elif truth in {"AI_GOLD_ACCEPTED", "AI_GOLD_PARTIAL"} or sid in verdicts:
                verdict = verdicts.get(sid, {})
                if truth == "AI_GOLD_PARTIAL" and not verdict:
                    verdict = {"status": "ACCEPT_PARTIAL"}
                if truth not in {"AI_GOLD_ACCEPTED", "AI_GOLD_PARTIAL"} and \
                        verdict.get("status") not in {"ACCEPT", "ACCEPT_PARTIAL"}:
                    skipped[f"review_verdict:{verdict.get('status')}"] += 1
                    continue
                if verdict.get("lines"):
                    lines = bb.expand(verdict["lines"])
                    payload = bb.read_json(lines)
                    gold, scan_path, gold_sha, gkind = ({"kind": "bench_lines", "path": str(lines)},
                                                        Path(payload["scan_path"]), payload.get("scan_sha256"),
                                                        "bench_lines")
                else:
                    gold, scan_path, gold_sha, gkind = _ai_gold(scan)
                if gold is None:
                    skipped["ai_gold_file_missing"] += 1
                    continue
                zero = scan.get("role") == "zero_bend_control" or bool(
                    bb.read_json(bb.ROOT / gold["path"] if not Path(gold["path"]).is_absolute() else gold["path"])
                    .get("no_bends_present"))
                run = scan.get("ai_label_run") or {}
                label_complete = verdict.get("status") != "ACCEPT_PARTIAL" and run.get("label_complete") is not False
                entry.update({"tier": ("AI_GOLD_ACCEPTED" if label_complete else "AI_GOLD_PARTIAL")
                              if truth in {"AI_GOLD_ACCEPTED", "AI_GOLD_PARTIAL"}
                              else f"AI_GOLD_REVIEW_{verdict.get('status')}", "gold": gold,
                              "label_complete": label_complete,
                              "review11": bool(run.get("review11")),
                              "zero_bend_control": zero, "scan_path": None if scan_path is None else str(scan_path),
                              "gold_scan_sha256": gold_sha, "label_frame": "source_ply_raw",
                              "gold_line_count": (scan.get("ai_label_run") or {}).get("n_bends_labelled"),
                              "gold_file_kind": gkind,
                              "verifier_summary": ((scan.get("ai_label_run") or {}).get("verifier") or {}).get("summary")})
                if scan.get("scan_accuracy_scoring"):
                    flags.append("EXCLUDE_FROM_SCAN_ACCURACY: " + str(scan["scan_accuracy_scoring"]))
                if "~72%" in str(entry.get("verifier_summary")) or sid == "sc_752e2bd54aad":
                    flags.append("PARTIAL_LENGTH_LINE_B2_72PCT")
            else:
                skipped[f"truth:{truth}"] += 1
                continue
            if any(f in {"LIDAR_DEMO_CAPTURE"} for f in entry["quality_flags"]):
                flags.append("LIDAR_DEMO_CAPTURE_REPORT_SEPARATELY")
            for owner_flag in owner_flags.get(sid, []):
                flags.append(owner_flag)
            scan_path = entry.get("scan_path")
            if not scan_path or _forbidden(scan_path) or not bb.expand(scan_path).is_file():
                skipped["scan_file_missing_or_forbidden"] += 1
                continue
            entry["scan_path"] = str(bb.expand(scan_path))
            if not args.no_hash:
                actual = bb.sha256_file(entry["scan_path"])
                entry["scan_sha256_verified"] = actual == entry["scan_sha256"] == entry["gold_scan_sha256"]
                if not entry["scan_sha256_verified"]:
                    flags.append("SCAN_SHA_MISMATCH_DO_NOT_SCORE")
            cad, alternatives = _pick_cad(part, Path(entry["scan_path"]).name)
            entry["cad_path"] = cad
            entry["cad_alternatives"] = alternatives
            entry["cad_formats_on_disk"] = sorted({f.get("format") for f in part.get("cad", {}).get("files", [])})
            entry["lanes_runnable"] = ["best_f1(A0 half)"] + (["best_f1(CAD half)", "runtime"] if cad else [])
            gold_file = bb.expand(gold_path_of(entry["gold"]))
            entry["gold_file_sha256"] = bb.sha256_file(gold_file) if gold_file.is_file() else None
            entry["cad_sha256"] = bb.sha256_file(cad) if cad else None
            entry["flags"] = flags
            cases.append(entry)
    summary = {
        "cases": len(cases),
        "by_tier": dict(Counter(c["tier"] for c in cases)),
        "zero_bend_controls": sum(c["zero_bend_control"] for c in cases),
        "with_cad": sum(bool(c["cad_path"]) for c in cases),
        "line_cases_with_cad": sum(bool(c["cad_path"]) and not c["zero_bend_control"] for c in cases),
        "gold_lines_total": sum(int(c.get("gold_line_count") or 0) for c in cases if not c["zero_bend_control"]),
        "families": len({c["family_id"] for c in cases}),
        "skipped": dict(skipped),
    }
    summary["excluded_named"] = excluded_named
    summary["by_tier"] = dict(Counter(c["tier"] for c in cases))
    summary["label_complete_cases"] = sum(bool(c["label_complete"]) for c in cases)
    identity = [{k: c.get(k) for k in ("case_id", "scan_sha256", "gold", "gold_file_sha256", "label_complete",
                                        "zero_bend_control", "cad_sha256")} for c in cases]
    case_list_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"))
                                   .encode("utf-8")).hexdigest()
    payload = {"schema": "sherman.broad_baseline.case_list.v1", "created_at": bb.utc_now(),
               "case_list_sha256": case_list_sha,
               "case_list_sha256_definition": "sha256 of compact sorted-key JSON of [{case_id, scan_sha256, gold, "
                                               "gold_file_sha256, label_complete, zero_bend_control, cad_sha256}]",
               "benchmark": {"path": str(args.benchmark), "sha256": bb.sha256_file(args.benchmark)},
               "custody": "CONSUMED development scans only; FRESH / FRESH_SCAN_OF_CONSUMED_PART / Eyal-50 / "
                          "incoming_codex_fetch_2026-10-17 excluded by construction",
               "summary": summary, "cases": cases}
    bb.write_json(args.out, payload)
    print({k: v for k, v in summary.items() if k != "excluded_named"}, case_list_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
