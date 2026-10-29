#!/usr/bin/env python3
"""Run a typed CAD+PLY slice with all scoring truth disabled during inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_bend_corpus import (  # noqa: E402
    _aggregate,
    _heatmap_consistency_metrics,
    _run_case_subprocess,
    _score_scan,
    _structured_score_scan,
)


def _case_output_name(entry: Dict[str, Any]) -> str:
    scan_path = Path(str(entry["scan_path"])).expanduser().resolve()
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in scan_path.name).strip("._") or "scan"
    digest = hashlib.sha256(str(scan_path).encode("utf-8")).hexdigest()[:12]
    return f"{entry.get('part_key', 'part')}--{slug}--{digest}.json"


def _blind_inference_part_id(cad_path: str, scan_path: str) -> str:
    identity = f"{Path(cad_path).expanduser().resolve()}\0{Path(scan_path).expanduser().resolve()}"
    return f"blind-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def _typed_release_metrics(results: Sequence[Dict[str, Any]], failures: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    exact_correct = 0
    contract_correct = 0
    latencies = []
    for result in results:
        expected = (result.get("expectation") or {}).get("expected_completed_bends")
        predicted = (result.get("summary") or {}).get("completed_bends")
        exact_correct += int(expected is not None and predicted is not None and int(expected) == int(predicted))
        contract = dict(result.get("blind_inference_contract") or {})
        contract_correct += int(
            result.get("inference_mode") == "blind_cad_ply"
            and contract.get("scoring_truth_crossed_subprocess_boundary") is False
            and contract.get("spec_bend_angles_crossed_subprocess_boundary") is False
            and contract.get("spec_schedule_crossed_subprocess_boundary") is False
            and contract.get("corpus_part_identifier_exposed_to_pipeline") is False
        )
        latency = (result.get("timing") or {}).get("total_sec")
        if latency is not None:
            latencies.append(float(latency))
    total = len(results) + len(failures)
    ordered = sorted(latencies)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    return {
        "scans_processed": total,
        "successful_scans": len(results),
        "failed_scans": len(failures),
        "exact_count_correct": exact_correct,
        "exact_count_accuracy": round(exact_correct / total, 6) if total else None,
        "blind_inference_contract_rate": round(contract_correct / total, 6) if total else None,
        "job_success_rate": round(len(results) / total, 6) if total else None,
        "p95_pipeline_latency_sec": p95,
    }


def run_slice(
    *,
    slice_payload: Dict[str, Any],
    output_dir: Path,
    timeout_sec: int,
    selected_parts: set[str],
    limit: Optional[int],
) -> Dict[str, Any]:
    entries = list(slice_payload.get("entries") or ())
    if selected_parts:
        entries = [entry for entry in entries if str(entry.get("part_key") or "") in selected_parts]
    if limit is not None:
        entries = entries[: max(0, int(limit))]
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    results = []
    failures = []
    for index, entry in enumerate(entries, start=1):
        expectation = dict(entry.get("expectation") or {})
        observed = expectation.get("expected_conventional_bend_count", expectation.get("expected_bend_count"))
        state = str(expectation.get("state") or "unknown")
        whole_total = expectation.get("known_whole_part_total_context")
        expected_total = observed if state == "full" or whole_total is None else whole_total
        score_expectation = {
            "state": state,
            "expected_total_bends": expected_total,
            "expected_completed_bends": observed,
        }
        inference_expectation = {
            "state": "unknown",
            "expected_total_bends": None,
            "expected_completed_bends": None,
        }
        output_json = cases_dir / _case_output_name(entry)
        print(f"[{index}/{len(entries)}] START {entry.get('part_key')}: {Path(str(entry['scan_path'])).name}", flush=True)
        try:
            result = _run_case_subprocess(
                part_key=str(entry.get("part_key") or ""),
                cad_path=str(entry["cad_path"]),
                scan_path=str(entry["scan_path"]),
                output_json=output_json,
                expectation=inference_expectation,
                spec_bend_angles=[],
                spec_schedule_type=None,
                timeout_sec=timeout_sec,
                skip_heatmap_consistency=True,
                heatmap_cache_dir=None,
                inference_mode="blind_cad_ply",
                inference_part_id=_blind_inference_part_id(str(entry["cad_path"]), str(entry["scan_path"])),
            )
            if result.get("inference_mode") != "blind_cad_ply":
                raise RuntimeError("case result violated blind inference contract")
            subprocess_expectation = dict(result.get("expectation") or {})
            if any(
                subprocess_expectation.get(field) not in {None, "", "unknown"}
                for field in ("state", "expected_total_bends", "expected_completed_bends")
            ):
                raise RuntimeError("blind inference subprocess received scoring truth")
            result["blind_inference_contract"] = {
                "scoring_truth_crossed_subprocess_boundary": False,
                "subprocess_expectation": subprocess_expectation,
                "spec_bend_angles_crossed_subprocess_boundary": False,
                "spec_schedule_crossed_subprocess_boundary": False,
                "corpus_part_identifier_exposed_to_pipeline": False,
                "pipeline_part_identifier": _blind_inference_part_id(
                    str(entry["cad_path"]), str(entry["scan_path"])
                ),
            }
            result["expectation"] = score_expectation
            result["metrics"] = _score_scan(result, score_expectation)
            result["structured_metrics"] = _structured_score_scan(result, score_expectation)
            result["heatmap_consistency_metrics"] = _heatmap_consistency_metrics(result, score_expectation)
            result["typed_truth_record_id"] = expectation.get("truth_record_id")
            results.append(result)
            print(f"[{index}/{len(entries)}] DONE {entry.get('part_key')}", flush=True)
        except Exception as exc:
            failures.append({"part_key": entry.get("part_key"), "scan_path": entry.get("scan_path"), "error": str(exc)})
            print(f"[{index}/{len(entries)}] FAILED {entry.get('part_key')}: {exc}", flush=True)
    aggregate = _aggregate(results + failures)
    summary = {
        "schema_version": 1,
        "inference_mode": "blind_cad_ply",
        "target": "visible_observed_conventional_bends",
        "aggregate": aggregate,
        "typed_release_metrics": _typed_release_metrics(results, failures),
        "results": results,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--part-key", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-sec", type=int, default=420)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.refresh_existing:
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["typed_release_metrics"] = _typed_release_metrics(
            summary.get("results") or (),
            summary.get("failures") or (),
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"typed_release_metrics": summary["typed_release_metrics"]}, indent=2))
        return 0
    summary = run_slice(
        slice_payload=json.loads(args.slice_json.read_text(encoding="utf-8")),
        output_dir=args.output_dir,
        timeout_sec=args.timeout_sec,
        selected_parts={str(value) for value in args.part_key if str(value)},
        limit=args.limit,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "aggregate": summary["aggregate"]}, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
