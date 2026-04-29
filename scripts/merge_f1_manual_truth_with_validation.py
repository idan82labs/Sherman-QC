#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _scan_name(value: Any) -> str:
    return Path(str(value or "")).name


def _key(*, track: Any, part_key: Any, scan_name: Any) -> Tuple[str, str, str]:
    return (str(track or ""), str(part_key or ""), _scan_name(scan_name))


def _result_key(result: Mapping[str, Any]) -> Tuple[str, str, str]:
    return _key(track=result.get("track"), part_key=result.get("part_key"), scan_name=_scan_name(result.get("scan_path")))


def _record_key(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    return _key(track=record.get("track"), part_key=record.get("part_key"), scan_name=record.get("scan_name") or _scan_name(record.get("scan_path")))


def _singleton_count(count_range: Sequence[Any]) -> Optional[int]:
    if len(count_range) < 2:
        return None
    try:
        low = int(count_range[0])
        high = int(count_range[1])
    except (TypeError, ValueError):
        return None
    return low if low == high else None


def _final_exact(algorithm: Mapping[str, Any]) -> Optional[int]:
    exact = algorithm.get("accepted_exact_bend_count")
    singleton = _singleton_count(list(algorithm.get("accepted_bend_count_range") or ()))
    if exact is None or singleton is None:
        return None
    try:
        return int(exact) if int(exact) == singleton else None
    except (TypeError, ValueError):
        return None


def _index_results(summary_paths: Sequence[Path]) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    indexed: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        for result in list(summary.get("results") or ()):
            row = dict(result)
            row["_source_summary_json"] = str(summary_path)
            indexed[_result_key(row)] = row
    return indexed


def merge_manual_truth_with_validation(
    *,
    manual_truth_json: Path,
    validation_summary_jsons: Sequence[Path],
) -> Dict[str, Any]:
    manual = _load_json(manual_truth_json)
    validation_by_key = _index_results(validation_summary_jsons)
    records = []
    unmatched_manual = []
    matched_keys = set()
    for record in list(manual.get("records") or manual.get("labels") or ()):
        merged = dict(record)
        key = _record_key(record)
        result = validation_by_key.get(key)
        if result is None:
            unmatched_manual.append({"track": key[0], "part_key": key[1], "scan_name": key[2]})
        else:
            matched_keys.add(key)
            algorithm = dict(result.get("algorithm") or {})
            render = dict(result.get("render_artifacts") or {})
            merged["algorithm_range"] = list(algorithm.get("accepted_bend_count_range") or ())
            merged["algorithm_observed_prediction"] = algorithm.get("observed_bend_count_prediction")
            merged["algorithm_exact"] = _final_exact(algorithm)
            merged["algorithm_raw_exact_field"] = algorithm.get("raw_exact_bend_count_field", algorithm.get("accepted_exact_bend_count"))
            merged["algorithm_candidate_source"] = algorithm.get("candidate_source")
            merged["algorithm_guard_reason"] = algorithm.get("guard_reason")
            merged["feature_count_semantics"] = result.get("feature_count_semantics")
            merged["contact_sheet"] = render.get("contact_sheet_1080p_path") or merged.get("contact_sheet")
            merged["latest_validation_summary_json"] = result.get("_source_summary_json")
        records.append(merged)
    unmatched_validation = [
        {"track": key[0], "part_key": key[1], "scan_name": key[2]}
        for key in sorted(validation_by_key)
        if key not in matched_keys
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_manual_truth_json": str(manual_truth_json),
        "source_validation_summary_jsons": [str(path) for path in validation_summary_jsons],
        "schema_version": 1,
        "label_semantics": manual.get("label_semantics"),
        "records": records,
        "merge_diagnostics": {
            "manual_record_count": len(list(manual.get("records") or manual.get("labels") or ())),
            "matched_record_count": len(matched_keys),
            "unmatched_manual_count": len(unmatched_manual),
            "unmatched_validation_count": len(unmatched_validation),
            "unmatched_manual": unmatched_manual,
            "unmatched_validation": unmatched_validation,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge manual F1 truth labels with latest validation outputs.")
    parser.add_argument("--manual-truth-json", required=True, type=Path)
    parser.add_argument("--validation-summary-json", action="append", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    payload = merge_manual_truth_with_validation(
        manual_truth_json=args.manual_truth_json,
        validation_summary_jsons=args.validation_summary_json,
    )
    _write_json(args.output_json, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["merge_diagnostics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
