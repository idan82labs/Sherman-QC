#!/opt/homebrew/bin/python3.11
"""
Structured CAD gold-set comparator for bend corpus runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLICE = ROOT / "data" / "bend_corpus" / "evaluation_slices" / "cad_gold_set.json"
DEFAULT_GOLD = ROOT / "data" / "bend_corpus" / "gold_sets" / "cad_structural_gold.json"
EVALUATOR = ROOT / "scripts" / "evaluate_bend_corpus.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_row_selectors(slice_payload: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    selectors: List[Tuple[str, Optional[str]]] = []
    for entry in slice_payload.get("entries") or []:
        part_key = str(entry.get("part_key") or "").strip()
        if not part_key:
            continue
        scan_name = str(entry.get("scan_name") or "").strip() or None
        selectors.append((part_key, scan_name))
    return selectors


def _filter_projected_rows(
    projected_rows: List[Dict[str, Any]],
    selectors: Iterable[Tuple[str, Optional[str]]],
) -> List[Dict[str, Any]]:
    selector_set = list(selectors)
    if not selector_set:
        return projected_rows
    filtered: List[Dict[str, Any]] = []
    for row in projected_rows:
        row_part_key = str(row.get("part_key") or "")
        row_scan_name = str(row.get("scan_name") or "")
        for part_key, scan_name in selector_set:
            if row_part_key != part_key:
                continue
            if scan_name is not None and row_scan_name != scan_name:
                continue
            filtered.append(row)
            break
    return filtered


def _stable_output_name(part_key: str, scan_name: str) -> str:
    raw = f"{part_key}::{scan_name}"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("._")
    if not safe:
        safe = "scan"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe}--{digest}.json"


def _extract_matched_bend_ids(matches: Iterable[Dict[str, Any]]) -> List[str]:
    matched: List[str] = []
    for match in matches or []:
        bend_id = str(match.get("bend_id") or "").strip()
        if not bend_id:
            continue
        if not bool(match.get("countable_in_regression", True)):
            continue
        status = str(match.get("status") or "").upper()
        if status == "NOT_DETECTED":
            continue
        matched.append(bend_id)
    return sorted(dict.fromkeys(matched))


def _project_result(item: Dict[str, Any]) -> Dict[str, Any]:
    details = dict(item.get("details") or {})
    metrics = dict(item.get("metrics") or {})
    cad_target = dict(details.get("cad_structural_target") or {})
    return {
        "part_key": str(item.get("part_key") or ""),
        "scan_name": str(item.get("scan_name") or ""),
        "cad_strategy": details.get("cad_strategy") or metrics.get("cad_strategy"),
        "profile_mode": details.get("profile_mode"),
        "nominal_source_class": details.get("nominal_source_class"),
        "expected_bend_count": cad_target.get("expected_bend_count", details.get("expected_bend_count")),
        "dominant_angle_families_deg": list(cad_target.get("dominant_angle_families_deg") or []),
        "blocker_attribution_breakdown": dict(metrics.get("blocker_attribution_breakdown") or {}),
        "correspondence_ready_bends": metrics.get("correspondence_decision_ready_bends"),
        "position_eligible_bends": metrics.get("position_eligible_bends"),
        "matched_bend_ids": _extract_matched_bend_ids(item.get("matches") or []),
        "scan_only_diagnostic_mismatch": details.get("scan_only_diagnostic_mismatch"),
    }


def _project_summary(summary_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        _project_result(item)
        for item in (summary_payload.get("results") or [])
        if str(item.get("part_key") or "").strip() and str(item.get("scan_name") or "").strip()
    ]


def _row_key(row: Dict[str, Any]) -> str:
    return f"{row.get('part_key')}::{row.get('scan_name')}"


def _compare_rows(projected_rows: List[Dict[str, Any]], gold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    actual = {_row_key(row): row for row in projected_rows}
    gold = {_row_key(row): row for row in gold_rows}
    all_keys = sorted(set(actual) | set(gold))
    mismatches: List[Dict[str, Any]] = []
    exact_matches = 0
    for key in all_keys:
        actual_row = actual.get(key)
        gold_row = gold.get(key)
        if actual_row is None:
            mismatches.append({"row_key": key, "type": "missing_in_actual", "gold": gold_row})
            continue
        if gold_row is None:
            mismatches.append({"row_key": key, "type": "missing_in_gold", "actual": actual_row})
            continue
        field_diffs: Dict[str, Dict[str, Any]] = {}
        for field_name in (
            "cad_strategy",
            "profile_mode",
            "nominal_source_class",
            "expected_bend_count",
            "dominant_angle_families_deg",
            "blocker_attribution_breakdown",
            "correspondence_ready_bends",
            "position_eligible_bends",
            "matched_bend_ids",
            "scan_only_diagnostic_mismatch",
        ):
            if actual_row.get(field_name) != gold_row.get(field_name):
                field_diffs[field_name] = {"actual": actual_row.get(field_name), "gold": gold_row.get(field_name)}
        if field_diffs:
            mismatches.append({"row_key": key, "type": "field_mismatch", "fields": field_diffs})
        else:
            exact_matches += 1
    return {
        "rows_evaluated": len(all_keys),
        "exact_matches": exact_matches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _run_corpus_slice(
    slice_path: Path,
    output_dir: Path,
    *,
    deterministic_seed: Optional[int],
    retry_seeds_json: Optional[str],
) -> Path:
    slice_payload = _load_json(slice_path)
    requested_parts = sorted(dict.fromkeys(part_key for part_key, _ in _slice_row_selectors(slice_payload)))
    command = [sys.executable, str(EVALUATOR), "--output", str(output_dir)]
    for part_key in requested_parts:
        command.extend(["--part", part_key])
    if deterministic_seed is not None:
        command.extend(["--deterministic-seed", str(int(deterministic_seed))])
    if retry_seeds_json not in {None, "", "None"}:
        command.extend(["--retry-seeds-json", str(retry_seeds_json)])
    subprocess.run(command, check=True)
    return output_dir / "summary.json"


def _write_projected_rows(output_dir: Path, projected_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest_entries: List[Dict[str, Any]] = []
    for row in projected_rows:
        filename = _stable_output_name(str(row.get("part_key") or ""), str(row.get("scan_name") or ""))
        path = output_dir / filename
        path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_entries.append(
            {
                "part_key": row.get("part_key"),
                "scan_name": row.get("scan_name"),
                "path": str(path),
            }
        )
    return manifest_entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate or compare a structured CAD gold set.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--run-evaluator", action="store_true")
    parser.add_argument("--slice-json", default=str(DEFAULT_SLICE))
    parser.add_argument("--gold-json", default=str(DEFAULT_GOLD))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-gold", action="store_true")
    parser.add_argument("--deterministic-seed", type=int, default=None)
    parser.add_argument("--retry-seeds-json", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path: Optional[Path] = Path(args.summary_json) if args.summary_json else None
    if args.run_evaluator:
        summary_path = _run_corpus_slice(
            Path(args.slice_json),
            output_dir / "corpus_eval",
            deterministic_seed=args.deterministic_seed,
            retry_seeds_json=args.retry_seeds_json,
        )
    if summary_path is None:
        raise SystemExit("Either --summary-json or --run-evaluator is required.")

    summary_payload = _load_json(summary_path)
    slice_path = Path(args.slice_json)
    slice_payload = _load_json(slice_path) if slice_path.exists() else {"entries": []}
    projected_rows = _filter_projected_rows(
        _project_summary(summary_payload),
        _slice_row_selectors(slice_payload),
    )
    projected_manifest = _write_projected_rows(output_dir, projected_rows)

    result_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_json": str(summary_path),
        "slice_json": str(slice_path),
        "projected_rows": projected_rows,
        "projected_row_count": len(projected_rows),
    }

    gold_path = Path(args.gold_json)
    if args.write_gold:
        gold_payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary_json": str(summary_path),
            "rows": projected_rows,
        }
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(json.dumps(gold_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        result_payload["gold_written"] = str(gold_path)
    elif gold_path.exists():
        gold_payload = _load_json(gold_path)
        comparison = _compare_rows(projected_rows, list(gold_payload.get("rows") or []))
        result_payload["comparison"] = comparison
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (output_dir / "manifest.json").write_text(
        json.dumps({"entries": projected_manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
