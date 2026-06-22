#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_zero_shot_repeated_near_90 import analyze_zero_shot_repeated_near_90
from scripts.compare_bend_corpus_summaries import _load_summary, compare_summaries


METHOD_FAMILIES: Dict[str, Dict[str, Any]] = {
    "fold_contour_diagnostics": {
        "hypothesis_id": "B8-H1",
        "hypothesis_statement": "Better upstream fold/trace diagnostics will show whether hard-case instability is caused by candidate dropout before strip grouping.",
        "conceptual_change": "Add non-decision fold and trace diagnostics to candidate and perturbation summaries.",
        "flags": {
            "fold_contour_diagnostics": True,
        },
        "code_paths_touched": [
            "backend/zero_shot_bend_profile.py",
            "scripts/evaluate_zero_shot_bends.py",
            "scripts/run_zero_shot_theory_experiment.py",
        ],
        "focused_tests": [
            "backend/tests/test_zero_shot_bend_profile.py",
            "tests/test_evaluate_zero_shot_bends.py",
            "tests/test_run_zero_shot_theory_experiment.py",
        ],
    },
    "trace_informed_evidence_atoms": {
        "hypothesis_id": "B8-H2",
        "hypothesis_statement": "Trace-informed atoms should improve strip identity inputs without changing selector policy.",
        "conceptual_change": "Carry trace endpoints, continuity, branch degree, and fold strength into evidence atoms.",
        "flags": {
            "fold_contour_diagnostics": True,
            "trace_informed_atoms": True,
        },
        "code_paths_touched": [
            "backend/zero_shot_bend_profile.py",
            "scripts/evaluate_zero_shot_bends.py",
            "scripts/run_zero_shot_theory_experiment.py",
        ],
        "focused_tests": [
            "backend/tests/test_zero_shot_bend_profile.py",
            "tests/test_evaluate_zero_shot_bends.py",
            "tests/test_run_zero_shot_theory_experiment.py",
        ],
    },
    "trace_aware_strip_grouping": {
        "hypothesis_id": "B8-H3",
        "hypothesis_statement": "Trace-aware same-strip decisions should reduce false strip fragmentation on repeated near-90 scans and preserve tight neighboring bends.",
        "conceptual_change": "Require real trace overlap or small trace gap with strong fold evidence before merging atoms into one strip.",
        "flags": {
            "fold_contour_diagnostics": True,
            "trace_informed_atoms": True,
            "trace_aware_strip_grouping": True,
        },
        "code_paths_touched": [
            "backend/zero_shot_bend_profile.py",
            "scripts/evaluate_zero_shot_bends.py",
            "scripts/run_zero_shot_theory_experiment.py",
        ],
        "focused_tests": [
            "backend/tests/test_zero_shot_bend_profile.py",
            "tests/test_evaluate_zero_shot_bends.py",
            "tests/test_run_zero_shot_theory_experiment.py",
        ],
    },
    "deterministic_candidate_graph_pruning": {
        "hypothesis_id": "B8-H4",
        "hypothesis_statement": "Pruning isolated weak atoms before strip formation will improve hard full-scan determinism more safely than another selector rescue.",
        "conceptual_change": "Keep an overcomplete upstream set longer, then deterministically prune weak isolated atoms using fold/trace evidence.",
        "flags": {
            "fold_contour_diagnostics": True,
            "trace_informed_atoms": True,
            "trace_aware_strip_grouping": True,
            "candidate_graph_pruning": True,
        },
        "code_paths_touched": [
            "backend/zero_shot_bend_profile.py",
            "scripts/evaluate_zero_shot_bends.py",
            "scripts/run_zero_shot_theory_experiment.py",
        ],
        "focused_tests": [
            "backend/tests/test_zero_shot_bend_profile.py",
            "tests/test_evaluate_zero_shot_bends.py",
            "tests/test_run_zero_shot_theory_experiment.py",
        ],
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _slice_specs_from_validation_contract(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    supplemental = dict(contract.get("supplemental_zero_shot_slices") or {})
    return [
        {
            "name": "zero_shot_development",
            "slice_json": str(ROOT / contract["zero_shot_development"]["slice_json"]),
            "protected": True,
        },
        {
            "name": "zero_shot_holdout",
            "slice_json": str(ROOT / contract["zero_shot_holdout"]["slice_json"]),
            "protected": True,
        },
        {
            "name": "known_count_plys",
            "slice_json": str(ROOT / supplemental["known_count_plys"]["slice_json"]),
            "protected": False,
        },
        {
            "name": "repeated_near_90",
            "slice_json": str(ROOT / supplemental["repeated_near_90"]["slice_json"]),
            "protected": False,
        },
        {
            "name": "small_tight_bends",
            "slice_json": str(ROOT / supplemental["small_tight_bends"]["slice_json"]),
            "protected": False,
        },
        {
            "name": "partial_total_bends",
            "slice_json": str(ROOT / supplemental["partial_total_bends"]["slice_json"]),
            "protected": False,
        },
    ]


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def _run_focused_tests(test_paths: List[str], logs_dir: Path) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "-q", *test_paths]
    output = _run(cmd, cwd=ROOT)
    (logs_dir / "focused_tests.log").write_text(output, encoding="utf-8")
    return {"command": cmd, "test_paths": test_paths, "status": "passed"}


def _run_slice(
    *,
    slice_name: str,
    slice_json: Path,
    output_dir: Path,
    seed_profile: Dict[str, Any],
    experiment_config_path: Optional[Path],
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_zero_shot_bends.py"),
        "--slice-json",
        str(slice_json),
        "--output-dir",
        str(output_dir),
        "--deterministic-seed",
        str(seed_profile["deterministic_seed"]),
        "--retry-seeds-json",
        json.dumps(seed_profile.get("retry_seeds") or []),
    ]
    if experiment_config_path is not None:
        cmd.extend(["--experiment-config-json", str(experiment_config_path)])
    output = _run(cmd, cwd=ROOT)
    (output_dir / "run.log").write_text(output, encoding="utf-8")
    summary_path = output_dir / "summary.json"
    return {
        "name": slice_name,
        "slice_json": str(slice_json),
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
    }


def _extract_result_by_part_key(summary_payload: Dict[str, Any], part_key: str) -> Optional[Dict[str, Any]]:
    for row in list(summary_payload.get("results") or []):
        if str(row.get("part_key") or "") == str(part_key):
            return row
    return None


def _count_range(row: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if not row:
        return None
    result = dict(row.get("result") or {})
    raw = result.get("estimated_bend_count_range")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return (int(raw[0]), int(raw[1]))


def _exact_count(row: Optional[Dict[str, Any]]) -> Optional[int]:
    if not row:
        return None
    result = dict(row.get("result") or {})
    value = result.get("estimated_bend_count")
    if value is None:
        return None
    return int(value)


def _range_width(value: Optional[Tuple[int, int]]) -> Optional[int]:
    if value is None:
        return None
    return int(value[1] - value[0])


def _is_not_wider(control: Optional[Tuple[int, int]], candidate: Optional[Tuple[int, int]]) -> bool:
    if control is None or candidate is None:
        return False
    return _range_width(candidate) <= _range_width(control)


def _is_tighter(control: Optional[Tuple[int, int]], candidate: Optional[Tuple[int, int]]) -> bool:
    if control is None or candidate is None:
        return False
    return _range_width(candidate) < _range_width(control)


def _metric(report: Dict[str, Any], metric_name: str) -> Optional[float]:
    for row in list(report.get("changed_metrics") or []):
        if row.get("metric") == metric_name:
            value = row.get("candidate")
            try:
                return float(value)
            except Exception:
                return None
    return None


def _aggregate_regressions(
    comparisons: Dict[str, Dict[str, Any]],
    control_summaries: Dict[str, Dict[str, Any]],
    candidate_summaries: Dict[str, Dict[str, Any]],
) -> List[str]:
    problems: List[str] = []
    dev_control = dict(control_summaries["zero_shot_development"].get("aggregate") or {})
    dev_candidate = dict(candidate_summaries["zero_shot_development"].get("aggregate") or {})
    if dev_candidate.get("exact_count_accuracy") != dev_control.get("exact_count_accuracy"):
        problems.append("development exact_count_accuracy changed")
    if dev_candidate.get("count_range_contains_truth_rate") != dev_control.get("count_range_contains_truth_rate"):
        problems.append("development count_range_contains_truth_rate changed")
    if float(dev_candidate.get("abstained_rate") or 0.0) > float(dev_control.get("abstained_rate") or 0.0):
        problems.append("development abstained_rate regressed")

    holdout_control = dict(control_summaries["zero_shot_holdout"].get("aggregate") or {})
    holdout_candidate = dict(candidate_summaries["zero_shot_holdout"].get("aggregate") or {})
    for metric_name in (
        "count_range_contains_truth_rate",
        "profile_family_match_rate",
        "abstention_correctness_rate",
    ):
        if holdout_candidate.get(metric_name) != holdout_control.get(metric_name):
            problems.append(f"holdout {metric_name} changed")
    if float(holdout_candidate.get("abstained_rate") or 0.0) > float(holdout_control.get("abstained_rate") or 0.0):
        problems.append("holdout abstained_rate regressed")

    for slice_name in ("known_count_plys", "repeated_near_90", "small_tight_bends"):
        control_agg = dict(control_summaries[slice_name].get("aggregate") or {})
        cand_agg = dict(candidate_summaries[slice_name].get("aggregate") or {})
        if cand_agg.get("count_range_contains_truth_rate") != control_agg.get("count_range_contains_truth_rate"):
            problems.append(f"{slice_name} count_range_contains_truth_rate changed")
        if slice_name == "repeated_near_90":
            if cand_agg.get("dominant_angle_family_match_rate") != control_agg.get("dominant_angle_family_match_rate"):
                problems.append("repeated_near_90 dominant_angle_family_match_rate changed")
        if float(cand_agg.get("abstained_rate") or 0.0) > float(control_agg.get("abstained_rate") or 0.0):
            problems.append(f"{slice_name} abstained_rate regressed")

    partial_control = dict(control_summaries["partial_total_bends"].get("aggregate") or {})
    partial_candidate = dict(candidate_summaries["partial_total_bends"].get("aggregate") or {})
    if partial_candidate.get("abstained_rate") != partial_control.get("abstained_rate"):
        problems.append("partial_total_bends abstained_rate changed")
    return problems


def _watch_case_summary(
    *,
    control_summaries: Dict[str, Dict[str, Any]],
    candidate_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    known_control = control_summaries["known_count_plys"]
    known_candidate = candidate_summaries["known_count_plys"]
    repeated_control = control_summaries["repeated_near_90"]
    repeated_candidate = candidate_summaries["repeated_near_90"]

    cases = {}
    for slice_name, part_keys in (
        ("known_count_plys", ["10839000", "47959001", "48963008", "48991006"]),
        ("repeated_near_90", ["10839000", "ysherman", "49204020_04"]),
    ):
        control_summary = control_summaries[slice_name]
        candidate_summary = candidate_summaries[slice_name]
        rows: Dict[str, Any] = {}
        for part_key in part_keys:
            control_row = _extract_result_by_part_key(control_summary, part_key)
            candidate_row = _extract_result_by_part_key(candidate_summary, part_key)
            rows[part_key] = {
                "control_exact_count": _exact_count(control_row),
                "candidate_exact_count": _exact_count(candidate_row),
                "control_count_range": list(_count_range(control_row) or []) or None,
                "candidate_count_range": list(_count_range(candidate_row) or []) or None,
                "control_profile_family": (control_row.get("result") or {}).get("profile_family") if control_row else None,
                "candidate_profile_family": (candidate_row.get("result") or {}).get("profile_family") if candidate_row else None,
            }
        cases[slice_name] = rows
    return cases


def _decide_experiment(
    *,
    method_config: Dict[str, Any],
    control_summaries: Dict[str, Dict[str, Any]],
    candidate_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    regressions = _aggregate_regressions({}, control_summaries, candidate_summaries)
    repeated_control = control_summaries["repeated_near_90"]
    repeated_candidate = candidate_summaries["repeated_near_90"]
    known_control = control_summaries["known_count_plys"]
    known_candidate = candidate_summaries["known_count_plys"]

    ysherman_control = _count_range(_extract_result_by_part_key(repeated_control, "ysherman"))
    ysherman_candidate = _count_range(_extract_result_by_part_key(repeated_candidate, "ysherman"))
    bend12_control = _count_range(_extract_result_by_part_key(repeated_control, "49204020_04"))
    bend12_candidate = _count_range(_extract_result_by_part_key(repeated_candidate, "49204020_04"))

    hard_case_regressions: List[str] = []
    if not _is_not_wider(ysherman_control, ysherman_candidate):
        hard_case_regressions.append("ysherman broadened")
    if not _is_not_wider(bend12_control, bend12_candidate):
        hard_case_regressions.append("49204020_04 broadened")

    exact_anchor_regressions: List[str] = []
    if _exact_count(_extract_result_by_part_key(known_candidate, "10839000")) != 10:
        exact_anchor_regressions.append("10839000 lost exact 10")
    if _exact_count(_extract_result_by_part_key(known_candidate, "47959001")) != 6:
        exact_anchor_regressions.append("47959001 lost exact 6")

    improvements: List[str] = []
    if _is_tighter(ysherman_control, ysherman_candidate):
        improvements.append("ysherman tightened")
    if _is_tighter(bend12_control, bend12_candidate):
        improvements.append("49204020_04 tightened")

    exact_secondary: List[str] = []
    for part_key in ("48963008", "48991006"):
        control_row = _extract_result_by_part_key(known_control, part_key)
        candidate_row = _extract_result_by_part_key(known_candidate, part_key)
        control_exact = _exact_count(control_row)
        candidate_exact = _exact_count(candidate_row)
        if control_exact is None and candidate_exact is not None:
            exact_secondary.append(f"{part_key} gained exact {candidate_exact}")
    improvements.extend(exact_secondary)

    problems = regressions + hard_case_regressions + exact_anchor_regressions
    keep = bool(improvements) and not problems
    rejection_reason = None if keep else "; ".join(problems or ["no hard-case improvement"])
    return {
        "keep": keep,
        "decision": "keep" if keep else "reject",
        "rejection_reason": rejection_reason,
        "improvements": improvements,
        "regressions": problems,
        "watch_cases": _watch_case_summary(
            control_summaries=control_summaries,
            candidate_summaries=candidate_summaries,
        ),
        "hypothesis_statement": method_config["hypothesis_statement"],
        "conceptual_change": method_config["conceptual_change"],
    }


def _professor_bundle_markdown(
    *,
    method_config: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    lines = [
        f"# {method_config['hypothesis_id']} — {method_config['conceptual_change']}",
        "",
        f"- Method family: `{method_config['name']}`",
        f"- Hypothesis: {method_config['hypothesis_statement']}",
        f"- Decision: `{decision['decision']}`",
        f"- Improvements: `{decision['improvements']}`",
        f"- Regressions: `{decision['regressions']}`",
        "",
        "## Watch Cases",
        "",
    ]
    for slice_name, rows in decision["watch_cases"].items():
        lines.append(f"### {slice_name}")
        lines.append("")
        for part_key, payload in rows.items():
            lines.append(
                f"- `{part_key}`: control range=`{payload['control_count_range']}`, candidate range=`{payload['candidate_count_range']}`, "
                f"control exact=`{payload['control_exact_count']}`, candidate exact=`{payload['candidate_exact_count']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated theory-to-reality zero-shot experiments against the frozen current control.")
    parser.add_argument("--method-family", required=True, choices=sorted(METHOD_FAMILIES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    method_config = dict(METHOD_FAMILIES[args.method_family])
    method_config["name"] = args.method_family

    validation_contract = _load_json(ROOT / "data" / "bend_corpus" / "gold_sets" / "validation_baselines.json")
    seed_profile = dict(validation_contract.get("seed_profile") or {"deterministic_seed": 101, "retry_seeds": [211, 307]})
    slice_specs = _slice_specs_from_validation_contract(validation_contract)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "hypothesis_id": method_config["hypothesis_id"],
        "method_family": args.method_family,
        "method_variant": "v1",
        "flags": dict(method_config["flags"]),
    }
    experiment_config_path = output_dir / "experiment_config.json"
    _write_json(experiment_config_path, experiment_config)

    test_report = None
    if not args.skip_tests:
        test_report = _run_focused_tests(method_config["focused_tests"], logs_dir)

    control_runs: Dict[str, Dict[str, Any]] = {}
    candidate_runs: Dict[str, Dict[str, Any]] = {}
    control_summaries: Dict[str, Dict[str, Any]] = {}
    candidate_summaries: Dict[str, Dict[str, Any]] = {}
    comparisons: Dict[str, Dict[str, Any]] = {}

    for slice_spec in slice_specs:
        slice_name = str(slice_spec["name"])
        slice_json = Path(slice_spec["slice_json"])
        control_output_dir = output_dir / "control" / slice_name
        candidate_output_dir = output_dir / "candidate" / slice_name
        control_runs[slice_name] = _run_slice(
            slice_name=slice_name,
            slice_json=slice_json,
            output_dir=control_output_dir,
            seed_profile=seed_profile,
            experiment_config_path=None,
        )
        candidate_runs[slice_name] = _run_slice(
            slice_name=slice_name,
            slice_json=slice_json,
            output_dir=candidate_output_dir,
            seed_profile=seed_profile,
            experiment_config_path=experiment_config_path,
        )
        control_summaries[slice_name] = _load_json(Path(control_runs[slice_name]["summary_json"]))
        candidate_summaries[slice_name] = _load_json(Path(candidate_runs[slice_name]["summary_json"]))
        report = {
            "baseline": control_runs[slice_name]["summary_json"],
            "candidate": candidate_runs[slice_name]["summary_json"],
            **compare_summaries(
                _load_summary(Path(control_runs[slice_name]["summary_json"])),
                _load_summary(Path(candidate_runs[slice_name]["summary_json"])),
            ),
        }
        comparisons[slice_name] = report
        _write_json(output_dir / "comparisons" / f"{slice_name}.json", report)

    repeated_analysis = analyze_zero_shot_repeated_near_90(Path(candidate_runs["repeated_near_90"]["summary_json"]))
    _write_json(output_dir / "repeated_near_90_analysis.json", repeated_analysis)

    decision = _decide_experiment(
        method_config=method_config,
        control_summaries=control_summaries,
        candidate_summaries=candidate_summaries,
    )
    _write_json(output_dir / "decision.json", decision)
    (output_dir / "PROFESSOR_BRIEF.md").write_text(
        _professor_bundle_markdown(method_config=method_config, decision=decision),
        encoding="utf-8",
    )
    (output_dir / "DECISION.md").write_text(
        "\n".join(
            [
                f"# {method_config['name']} decision",
                "",
                f"- Decision: `{decision['decision']}`",
                f"- Improvements: `{decision['improvements']}`",
                f"- Regressions: `{decision['regressions']}`",
                f"- Rejection reason: `{decision['rejection_reason']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis_id": method_config["hypothesis_id"],
        "method_family": args.method_family,
        "exact_code_paths_touched": method_config["code_paths_touched"],
        "slices_run": [slice_spec["name"] for slice_spec in slice_specs],
        "artifact_folder": str(output_dir),
        "keep_reject_decision": decision["decision"],
        "rejection_reason": decision["rejection_reason"],
        "experiment_config_json": str(experiment_config_path),
        "control_runs": control_runs,
        "candidate_runs": candidate_runs,
        "comparison_reports": {key: str(output_dir / "comparisons" / f"{key}.json") for key in comparisons},
        "repeated_near_90_analysis_json": str(output_dir / "repeated_near_90_analysis.json"),
        "professor_brief_markdown": str(output_dir / "PROFESSOR_BRIEF.md"),
        "test_report": test_report,
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "decision": decision["decision"], "improvements": decision["improvements"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
