#!/opt/homebrew/bin/python3.11
"""
Label-blind evaluator for zero-shot bend structure inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
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
DEFAULT_DEV_SLICE = ROOT / "data" / "bend_corpus" / "evaluation_slices" / "zero_shot_development.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_output_name(scan_name: str) -> str:
    raw_name = Path(str(scan_name)).name
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw_name).strip("._")
    if not safe_name:
        safe_name = "scan"
    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}--{digest}.json"


def _load_zero_shot_backend() -> Tuple[Any, Any]:
    candidates = [
        ROOT / "backend" / "zero_shot_bend_profile.py",
        ROOT / "backend" / "bend_inspection_pipeline.py",
    ]
    backend_module = None
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if hasattr(module, "run_zero_shot_bend_profile"):
            backend_module = module
            break
    if backend_module is None:
        raise RuntimeError(
            "Zero-shot backend entrypoint not found. Expected run_zero_shot_bend_profile in "
            "backend/zero_shot_bend_profile.py or backend/bend_inspection_pipeline.py."
        )
    loader = getattr(backend_module, "load_bend_runtime_config", None)
    if loader is None:
        pipeline_path = ROOT / "backend" / "bend_inspection_pipeline.py"
        spec = importlib.util.spec_from_file_location("bend_inspection_pipeline", pipeline_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load bend_inspection_pipeline.py for runtime config.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loader = getattr(module, "load_bend_runtime_config")
    return backend_module.run_zero_shot_bend_profile, loader


def _apply_seed_overrides(runtime_config: Any, deterministic_seed: Optional[int], retry_seeds: Optional[List[int]]) -> Dict[str, Any]:
    local_refinement = dict(getattr(runtime_config, "local_refinement_kwargs", {}) or {})
    if deterministic_seed is not None:
        local_refinement["deterministic_seed"] = int(deterministic_seed)
    if retry_seeds is not None:
        local_refinement["retry_seeds"] = [int(seed) for seed in retry_seeds]
    setattr(runtime_config, "local_refinement_kwargs", local_refinement)
    return {
        "deterministic_seed": local_refinement.get("deterministic_seed"),
        "retry_seeds": list(local_refinement.get("retry_seeds") or []),
    }


def _normalize_angle_family(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [round(float(values), 3)]
    if isinstance(values, dict):
        if "dominant" in values:
            return _normalize_angle_family(values.get("dominant"))
        if "values" in values:
            return _normalize_angle_family(values.get("values"))
    if isinstance(values, (list, tuple)):
        out: List[float] = []
        for value in values:
            try:
                out.append(round(float(value), 3))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _normalize_count_range(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, dict):
        low = value.get("min")
        high = value.get("max")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = value
    else:
        return None
    try:
        low_int = int(low)
        high_int = int(high)
    except (TypeError, ValueError):
        return None
    return (min(low_int, high_int), max(low_int, high_int))


def _extract_duplicate_rate(result: Dict[str, Any]) -> Optional[float]:
    candidates: Iterable[Any] = (
        result.get("duplicate_line_family_rate"),
        (result.get("support_breakdown") or {}).get("duplicate_line_family_rate"),
        (result.get("candidate_generator_summary") or {}).get("duplicate_line_family_rate"),
        (result.get("profile_selector_result") or {}).get("duplicate_line_family_rate"),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            continue
    return None


def _extract_strip_merge_count(result: Dict[str, Any]) -> Optional[int]:
    candidates: Iterable[Any] = (
        (result.get("support_breakdown") or {}).get("strip_duplicate_merge_count"),
        (result.get("candidate_generator_summary") or {}).get("strip_duplicate_merge_count"),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_strip_count(result: Dict[str, Any]) -> Optional[int]:
    strips = result.get("bend_support_strips")
    if isinstance(strips, list):
        return len(strips)
    support = (result.get("support_breakdown") or {}).get("bend_support_strip_count")
    if support is None:
        return None
    try:
        return int(support)
    except (TypeError, ValueError):
        return None


def _score_result(result: Dict[str, Any], expectation: Dict[str, Any]) -> Dict[str, Any]:
    expected_count = expectation.get("expected_bend_count")
    expected_angles = _normalize_angle_family(expectation.get("dominant_angle_families_deg"))
    expected_family = expectation.get("expected_profile_family")
    expected_families = expectation.get("expected_profile_families")
    expected_abstained = expectation.get("expected_abstained")
    predicted_count = result.get("estimated_bend_count")
    try:
        predicted_exact_count = int(predicted_count) if predicted_count is not None else None
    except (TypeError, ValueError):
        predicted_exact_count = None
    count_range = _normalize_count_range(result.get("estimated_bend_count_range"))
    abstained = bool(result.get("abstained"))
    exact_count_match = None
    if expected_count is not None and predicted_exact_count is not None and not abstained:
        exact_count_match = int(predicted_exact_count == int(expected_count))
    count_range_contains_truth = None
    if expected_count is not None and count_range is not None:
        count_range_contains_truth = int(count_range[0] <= int(expected_count) <= count_range[1])
    predicted_angles = _normalize_angle_family(result.get("dominant_angle_families_deg"))
    dominant_angle_family_match = None
    if expected_angles and predicted_angles:
        dominant_angle_family_match = int(
            any(abs(predicted - expected) <= 3.0 for predicted in predicted_angles for expected in expected_angles)
        )
    predicted_family = result.get("profile_family")
    profile_family_match = None
    allowed_families: Optional[List[str]] = None
    if expected_families is not None:
        allowed_families = [str(value) for value in expected_families if value is not None]
    elif expected_family is not None:
        allowed_families = [str(expected_family)]
    if allowed_families:
        profile_family_match = int(str(predicted_family) in allowed_families)
    abstention_correct = None
    if expected_abstained is not None:
        abstention_correct = int(bool(abstained) == bool(expected_abstained))
    return {
        "predicted_exact_count": predicted_exact_count,
        "exact_count": exact_count_match,
        "count_range": list(count_range) if count_range is not None else None,
        "count_range_contains_truth": count_range_contains_truth,
        "dominant_angle_family_match": dominant_angle_family_match,
        "profile_family_match": profile_family_match,
        "abstention_correct": abstention_correct,
        "duplicate_line_family_rate": _extract_duplicate_rate(result),
        "bend_support_strip_count": _extract_strip_count(result),
        "strip_duplicate_merge_count": _extract_strip_merge_count(result),
        "abstained": abstained,
    }


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [item for item in results if item.get("status") == "success"]
    exact_values = [item["metrics"]["exact_count"] for item in successful if item["metrics"].get("exact_count") is not None]
    range_values = [
        item["metrics"]["count_range_contains_truth"]
        for item in successful
        if item["metrics"].get("count_range_contains_truth") is not None
    ]
    angle_values = [
        item["metrics"]["dominant_angle_family_match"]
        for item in successful
        if item["metrics"].get("dominant_angle_family_match") is not None
    ]
    family_values = [
        item["metrics"]["profile_family_match"]
        for item in successful
        if item["metrics"].get("profile_family_match") is not None
    ]
    abstention_values = [
        item["metrics"]["abstention_correct"]
        for item in successful
        if item["metrics"].get("abstention_correct") is not None
    ]
    duplicate_values = [
        float(item["metrics"]["duplicate_line_family_rate"])
        for item in successful
        if item["metrics"].get("duplicate_line_family_rate") is not None
    ]
    strip_counts = [
        int(item["metrics"]["bend_support_strip_count"])
        for item in successful
        if item["metrics"].get("bend_support_strip_count") is not None
    ]
    strip_merge_counts = [
        int(item["metrics"]["strip_duplicate_merge_count"])
        for item in successful
        if item["metrics"].get("strip_duplicate_merge_count") is not None
    ]
    abstained_values = [1 if item["metrics"].get("abstained") else 0 for item in successful]
    profile_breakdown: Dict[str, int] = {}
    for item in successful:
        family = str((item.get("result") or {}).get("profile_family") or "UNKNOWN")
        profile_breakdown[family] = profile_breakdown.get(family, 0) + 1
    return {
        "scans_processed": len(results),
        "successful_scans": len(successful),
        "exact_count_evaluated": len(exact_values),
        "exact_count_accuracy": round(sum(exact_values) / len(exact_values), 3) if exact_values else None,
        "count_range_evaluated": len(range_values),
        "count_range_contains_truth_rate": round(sum(range_values) / len(range_values), 3) if range_values else None,
        "dominant_angle_family_evaluated": len(angle_values),
        "dominant_angle_family_match_rate": round(sum(angle_values) / len(angle_values), 3) if angle_values else None,
        "profile_family_evaluated": len(family_values),
        "profile_family_match_rate": round(sum(family_values) / len(family_values), 3) if family_values else None,
        "abstention_evaluated": len(abstention_values),
        "abstention_correctness_rate": round(sum(abstention_values) / len(abstention_values), 3)
        if abstention_values
        else None,
        "mean_duplicate_line_family_rate": round(sum(duplicate_values) / len(duplicate_values), 3)
        if duplicate_values
        else None,
        "mean_bend_support_strip_count": round(sum(strip_counts) / len(strip_counts), 3)
        if strip_counts
        else None,
        "mean_strip_duplicate_merge_count": round(sum(strip_merge_counts) / len(strip_merge_counts), 3)
        if strip_merge_counts
        else None,
        "abstained_count": int(sum(abstained_values)),
        "abstained_rate": round(sum(abstained_values) / len(abstained_values), 3) if abstained_values else None,
        "profile_family_breakdown": dict(sorted(profile_breakdown.items())),
    }


def _run_entry(
    entry: Dict[str, Any],
    *,
    run_zero_shot_bend_profile: Any,
    load_runtime_config: Any,
    deterministic_seed: Optional[int],
    retry_seeds: Optional[List[int]],
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime_config = load_runtime_config(None)
    runtime_seed_profile = _apply_seed_overrides(runtime_config, deterministic_seed, retry_seeds)
    part_key = entry.get("part_key")
    scan_path = str(entry["scan_path"])
    expectation = copy.deepcopy(entry.get("expectation") or {})
    result = run_zero_shot_bend_profile(
        **{
            "scan_path": scan_path,
            "runtime_config": runtime_config,
            "part_id": part_key,
            **({"experiment_config": experiment_config} if experiment_config is not None else {}),
        },
    )
    if hasattr(result, "to_dict"):
        result_payload = result.to_dict()
    elif isinstance(result, dict):
        result_payload = copy.deepcopy(result)
    else:
        raise TypeError("run_zero_shot_bend_profile must return a dict-like payload or support to_dict().")
    return {
        "part_key": part_key,
        "scan_path": scan_path,
        "scan_name": Path(scan_path).name,
        "expectation": expectation,
        "runtime_seed_profile": runtime_seed_profile,
        "result": result_payload,
        "experiment_config": copy.deepcopy(experiment_config) if experiment_config else None,
        "metrics": _score_result(result_payload, expectation),
        "status": "success",
    }


def _write_markdown(output_dir: Path, aggregate: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    lines = [
        "# Zero-Shot Bend Evaluation",
        "",
        f"- Scans processed: `{aggregate['scans_processed']}`",
        f"- Successful scans: `{aggregate['successful_scans']}`",
        f"- Exact-count accuracy: `{aggregate['exact_count_accuracy']}`",
        f"- Count-range containment: `{aggregate['count_range_contains_truth_rate']}`",
        f"- Dominant-angle-family match: `{aggregate['dominant_angle_family_match_rate']}`",
        f"- Mean duplicate-line-family rate: `{aggregate['mean_duplicate_line_family_rate']}`",
        f"- Mean bend-support strip count: `{aggregate['mean_bend_support_strip_count']}`",
        f"- Mean strip duplicate merges: `{aggregate['mean_strip_duplicate_merge_count']}`",
        f"- Abstained rate: `{aggregate['abstained_rate']}`",
        "",
        "| Scan | Profile | Exact count | Range contains truth | Angle-family match | Duplicate rate | Abstained |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        metrics = item.get("metrics") or {}
        result = item.get("result") or {}
        lines.append(
            f"| {item.get('scan_name')} | {result.get('profile_family')} | "
            f"{metrics.get('exact_count')} | {metrics.get('count_range_contains_truth')} | "
            f"{metrics.get('dominant_angle_family_match')} | {metrics.get('duplicate_line_family_rate')} | "
            f"{metrics.get('abstained')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot bend profiling on a labeled slice.")
    parser.add_argument("--slice-json", default=str(DEFAULT_DEV_SLICE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deterministic-seed", type=int, default=None)
    parser.add_argument("--retry-seeds-json", default=None)
    parser.add_argument("--experiment-config-json", default=None)
    args = parser.parse_args()

    slice_path = Path(args.slice_json).resolve()
    slice_payload = _load_json(slice_path)
    entries = list(slice_payload.get("entries") or [])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_zero_shot_bend_profile, load_runtime_config = _load_zero_shot_backend()
    retry_seeds = None
    retry_seeds_json = getattr(args, "retry_seeds_json", None)
    if retry_seeds_json not in {None, "", "None"}:
        retry_seeds = [int(v) for v in json.loads(retry_seeds_json)]
    experiment_config = None
    experiment_config_json = getattr(args, "experiment_config_json", None)
    if experiment_config_json not in {None, "", "None"}:
        experiment_config = json.loads(Path(experiment_config_json).read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = []
    for entry in entries:
        item = _run_entry(
            entry,
            run_zero_shot_bend_profile=run_zero_shot_bend_profile,
            load_runtime_config=load_runtime_config,
            deterministic_seed=args.deterministic_seed,
            retry_seeds=retry_seeds,
            experiment_config=experiment_config,
        )
        results.append(item)
        out_name = _stable_output_name(item["scan_name"])
        (output_dir / out_name).write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")

    aggregate = _aggregate(results)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "slice_json": str(slice_path),
        "output_dir": str(output_dir),
        "runtime_seed_profile": {
            "deterministic_seed": args.deterministic_seed,
            "retry_seeds": list(retry_seeds or []),
        },
        "experiment_config": copy.deepcopy(experiment_config) if experiment_config else None,
        "entries": [
            {
                "part_key": item.get("part_key"),
                "scan_name": item.get("scan_name"),
                "scan_path": item.get("scan_path"),
                "output_json": str(output_dir / _stable_output_name(str(item.get("scan_name")))),
                "status": item.get("status"),
            }
            for item in results
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_payload = {
        "aggregate": aggregate,
        "results": results,
        "manifest_path": str(manifest_path),
        "slice_json": str(slice_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(output_dir, aggregate, results)
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
