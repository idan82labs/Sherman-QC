#!/usr/bin/env python3
"""Apply and score the source-only V43 explosion guard on frozen caches.

This evaluator is explicitly development-only.  It opens truth only after
primary/replay successor outputs have been materialized and byte-compared.
Frozen V51 and catalog artifacts are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_post_v51_v43_explosion_guard_v1 import (  # noqa: E402
    PROFILE,
    canonical_bytes,
    compose_v43_explosion_guard,
)
from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    Thresholds,
    match_lines,
)


DEFAULT_CATALOG = (
    ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
)
DEFAULT_TODAY_ROOT = (
    ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
)
DEFAULT_LEGACY_ROOT = (
    ROOT / "output/research/f1-v51-legacy37-scan-only-development-20260830-r3"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "output/research/f1-v51-v43-explosion-guard-development-20260831"
)
TODAY_SCAN_ROLE = "2026-08-30_ai_multiview_consensus_development"
THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if tp or fp or fn else 0.0
    return {
        "case_count": len(rows),
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(row["exact"]) for row in rows),
        "exact_case_rate": round(
            sum(bool(row["exact"]) for row in rows) / len(rows), 6
        ),
    }


def _delta(successor: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(float(successor[key]) - float(baseline[key]), 6)
        for key in (
            "prediction_count",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "exact_case_count",
        )
    }


def _today_cases(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        dict(row)
        for row in catalog.get("cases") or []
        if row.get("scan_role") == TODAY_SCAN_ROLE
    ]
    expected = [f"T{index:03d}" for index in range(1, 29)]
    if [row.get("case_id") for row in rows] != expected:
        raise ValueError("catalog does not contain ordered today T001..T028")
    return {str(row["case_id"]): row for row in rows}


def _legacy_cases(
    catalog: dict[str, Any], execution_root: Path
) -> dict[str, dict[str, Any]]:
    by_sha = {
        str(row["scan_sha256"]): dict(row)
        for row in catalog.get("cases") or []
        if row.get("scan_role") != TODAY_SCAN_ROLE
    }
    manifest = _read(execution_root / "development_execution_manifest.json")
    primary = [
        row for row in manifest.get("records") or [] if row.get("pass") == "primary"
    ]
    if len(primary) != 37:
        raise ValueError("legacy execution manifest must contain 37 primary records")
    result: dict[str, dict[str, Any]] = {}
    for record in primary:
        runtime_id = str(record["case_id"])
        case = by_sha.get(str(record["scan_sha256"]))
        if case is None:
            raise ValueError(f"{runtime_id}: legacy scan SHA is absent from catalog")
        result[runtime_id] = case
    return result


def _materialize_cohort(
    *,
    cohort: str,
    execution_root: Path,
    cases: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    primary_outputs: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    byte_identity: dict[str, bool] = {}
    triggered: list[str] = []
    input_hashes: dict[str, dict[str, str | None]] = {}
    for runtime_id, case in sorted(cases.items()):
        pass_results: dict[str, tuple[Path, dict[str, Any]]] = {}
        input_hashes[runtime_id] = {}
        for pass_name in ("primary", "replay"):
            case_root = execution_root / pass_name / runtime_id
            v51_path = case_root / "semantic_result.json"
            if not v51_path.is_file():
                raise FileNotFoundError(f"{cohort}/{pass_name}/{runtime_id}: V51 missing")
            v16_path = case_root / "v16.json"
            v43_path = case_root / "v43.json"
            have_stages = v16_path.is_file() and v43_path.is_file()
            if v16_path.is_file() != v43_path.is_file():
                raise ValueError(f"{cohort}/{pass_name}/{runtime_id}: partial stage cache")
            v51 = _read(v51_path)
            v16 = _read(v16_path) if have_stages else None
            v43 = _read(v43_path) if have_stages else None
            result = compose_v43_explosion_guard(v16, v43, v51)
            destination = output_root / cohort / pass_name / runtime_id / "semantic_result.json"
            _write(destination, result)
            pass_results[pass_name] = (destination, result)
            input_hashes[runtime_id][f"{pass_name}_v16_sha256"] = (
                _sha(v16_path) if have_stages else None
            )
            input_hashes[runtime_id][f"{pass_name}_v43_sha256"] = (
                _sha(v43_path) if have_stages else None
            )
            input_hashes[runtime_id][f"{pass_name}_v51_sha256"] = _sha(v51_path)
        primary_path, primary = pass_results["primary"]
        replay_path, _replay = pass_results["replay"]
        identical = primary_path.read_bytes() == replay_path.read_bytes()
        if not identical:
            raise RuntimeError(f"{cohort}/{runtime_id}: primary/replay bytes differ")
        byte_identity[runtime_id] = identical
        primary_outputs[runtime_id] = primary
        paths[str(case["case_id"])] = str(primary_path)
        if primary["guard_audit"]["guard_triggered"]:
            triggered.append(runtime_id)

    # Truth is consulted only after both passes have been written and compared.
    baseline_rows: list[dict[str, Any]] = []
    successor_rows: list[dict[str, Any]] = []
    regressed: list[str] = []
    per_case: list[dict[str, Any]] = []
    for runtime_id, case in sorted(cases.items()):
        baseline_payload = _read(
            execution_root / "primary" / runtime_id / "semantic_result.json"
        )
        baseline = match_lines(case["bends"], baseline_payload["predictions"], THRESHOLDS)
        successor = match_lines(
            case["bends"], primary_outputs[runtime_id]["predictions"], THRESHOLDS
        )
        baseline_rows.append(baseline)
        successor_rows.append(successor)
        baseline_gold = {str(row["gold_id"]) for row in baseline["matches"]}
        successor_gold = {str(row["gold_id"]) for row in successor["matches"]}
        regressed.extend(sorted(baseline_gold - successor_gold))
        per_case.append(
            {
                "runtime_case_id": runtime_id,
                "catalog_case_id": str(case["case_id"]),
                "guard_triggered": primary_outputs[runtime_id]["guard_audit"][
                    "guard_triggered"
                ],
                "baseline_true_positive": baseline["true_positive"],
                "successor_true_positive": successor["true_positive"],
                "baseline_false_positive": baseline["false_positive"],
                "successor_false_positive": successor["false_positive"],
            }
        )
    baseline_metrics = _metrics(baseline_rows)
    successor_metrics = _metrics(successor_rows)
    return {
        "cohort": cohort,
        "case_count": len(cases),
        "baseline_metrics": baseline_metrics,
        "successor_metrics": successor_metrics,
        "delta": _delta(successor_metrics, baseline_metrics),
        "triggered_runtime_case_ids": triggered,
        "regressed_gold_ids": regressed,
        "zero_true_positive_regression": not regressed,
        "all_primary_replay_byte_identical": all(byte_identity.values()),
        "case_primary_replay_byte_identity": byte_identity,
        "prediction_paths": paths,
        "input_hashes": input_hashes,
        "cases": per_case,
        "_baseline_rows": baseline_rows,
        "_successor_rows": successor_rows,
    }


def evaluate_cached(
    *,
    catalog_path: Path,
    today_root: Path,
    legacy_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    today_root = today_root.resolve()
    legacy_root = legacy_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("guard output root is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    catalog = _read(catalog_path)
    today = _materialize_cohort(
        cohort="today28",
        execution_root=today_root,
        cases=_today_cases(catalog),
        output_root=output_root,
    )
    legacy = _materialize_cohort(
        cohort="legacy37",
        execution_root=legacy_root,
        cases=_legacy_cases(catalog, legacy_root),
        output_root=output_root,
    )
    pooled_baseline = _metrics(legacy.pop("_baseline_rows") + today.pop("_baseline_rows"))
    pooled_successor = _metrics(
        legacy.pop("_successor_rows") + today.pop("_successor_rows")
    )
    prediction_map = {**legacy["prediction_paths"], **today["prediction_paths"]}
    if len(prediction_map) != 65:
        raise ValueError("pooled prediction map must contain exactly 65 cases")
    prediction_map_path = output_root / "prediction_map.json"
    _write(prediction_map_path, dict(sorted(prediction_map.items())))
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v43_explosion_guard_cached_development_result_v1",
        "profile": PROFILE,
        "evidence_class": "POST_HOC_CONSUMED_DEVELOPMENT",
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha(catalog_path),
        "today_execution_root": str(today_root),
        "legacy_execution_root": str(legacy_root),
        "prediction_map_path": str(prediction_map_path),
        "prediction_map_sha256": _sha(prediction_map_path),
        "today28": today,
        "legacy37": legacy,
        "pooled65": {
            "baseline_metrics": pooled_baseline,
            "successor_metrics": pooled_successor,
            "delta": _delta(pooled_successor, pooled_baseline),
        },
        "all_65_primary_replay_byte_identical": bool(
            today["all_primary_replay_byte_identical"]
            and legacy["all_primary_replay_byte_identical"]
        ),
        "zero_true_positive_regression_on_consumed_65": bool(
            today["zero_true_positive_regression"]
            and legacy["zero_true_positive_regression"]
        ),
        "labels_used_at_inference": False,
        "expected_count_used_at_inference": False,
        "cad_used_at_inference": False,
        "part_identity_used_at_inference": False,
        "development_tuned_cutoff": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "limitation": (
            "The cutoff was selected after inspecting these consumed development errors; "
            "a future-day prospective cohort is required before retention."
        ),
    }
    report_path = output_root / "cached_guard_development_report.json"
    _write(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--today-root", type=Path, default=DEFAULT_TODAY_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = evaluate_cached(
        catalog_path=args.catalog,
        today_root=args.today_root,
        legacy_root=args.legacy_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "today28": report["today28"]["successor_metrics"],
                "legacy37": report["legacy37"]["successor_metrics"],
                "pooled65": report["pooled65"]["successor_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

