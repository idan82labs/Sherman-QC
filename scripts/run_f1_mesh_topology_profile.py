#!/usr/bin/env python3
"""Run and freeze the geometry-only mesh-topology research profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.bend_inspection_pipeline import load_bend_runtime_config
from backend.f1_mesh_surface_topology_filter import (
    filter_mesh_surface_topology,
    route_mesh_topology_profile,
)
from backend.zero_shot_bend_profile import run_zero_shot_bend_profile
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


DEFAULT_CANDIDATES = ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/candidate_pool"
DEFAULT_LABELS = (
    ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/manual_adjudication/labels"
)
DEFAULT_OUTPUT = ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/mesh_topology_profile_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("[0-9][0-9]_C[0-9][0-9].json"))
    if not paths:
        raise ValueError(f"no candidate pools found under {root}")
    return paths


def _token(payload: dict[str, Any], path: Path) -> str:
    token = str(payload.get("case_id") or path.stem).split("_", 1)[-1]
    if not token.startswith("C"):
        raise ValueError(f"invalid case token for {path}: {token}")
    return token


def _topology_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = []
    for index, candidate in enumerate(result["selected"], start=1):
        record = dict(candidate)
        record["source_candidate_id"] = str(
            record.get("candidate_id") or record.get("bend_id") or f"mesh_{index:02d}"
        )
        record["bend_id"] = f"F1-{index:02d}"
        predictions.append(record)
    return predictions


def _formed_predictions(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = []
    for index, bend in enumerate(result_payload.get("renderable_bend_objects") or (), start=1):
        line = bend.get("detected_line") or {}
        start = line.get("raw_start")
        end = line.get("raw_end")
        if start is None or end is None:
            continue
        predictions.append(
            {
                "bend_id": str(bend.get("bend_id") or f"FT{index:02d}"),
                "line_start": start,
                "line_end": end,
                "candidate_source": "interior_formed_transition_completion",
                "source_kind": bend.get("source_kind"),
                "profile_family": bend.get("profile_family"),
                "metadata": bend.get("metadata") or {},
            }
        )
    return predictions


def predict(args: argparse.Namespace) -> None:
    runtime_config = load_bend_runtime_config(None)
    prediction_paths: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    routes: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for path in _candidate_paths(args.candidate_root):
        candidate_payload = _read(path)
        token = _token(candidate_payload, path)
        topology = filter_mesh_surface_topology(candidate_payload.get("candidates") or ())
        route = route_mesh_topology_profile(candidate_payload, topology)
        zero_shot_payload = None
        if route["enable_interior_formed_transition_completion"]:
            zero_shot_result = run_zero_shot_bend_profile(
                scan_path=str(Path(candidate_payload["scan_path"]).resolve()),
                runtime_config=runtime_config,
                part_id=None,
                experiment_config={
                    "flags": {
                        "interior_formed_transition_completion": True,
                        "prismatic_plane_lattice_completion": True,
                    }
                },
            )
            zero_shot_payload = zero_shot_result.to_dict()
            predictions = _formed_predictions(zero_shot_payload)
        else:
            predictions = _topology_predictions(topology)

        prediction = {
            "schema_version": 1,
            "algorithm": "f1_mesh_topology_profile_v1",
            "case_token": token,
            "coordinate_frame": candidate_payload.get("coordinate_frame") or "scan_raw",
            "route": route,
            "mesh_topology_diagnostics": topology["diagnostics"],
            "owned_bend_regions": predictions,
            "inference_contract": {
                "uses_part_identifier": False,
                "uses_expected_count": False,
                "uses_manual_truth": False,
                "uses_cad": False,
                "source_points_serialized": False,
            },
        }
        if zero_shot_payload is not None:
            prediction["formed_completion_diagnostics"] = {
                "profile_family": zero_shot_payload.get("profile_family"),
                "estimated_bend_count": zero_shot_payload.get("estimated_bend_count"),
                "abstained": zero_shot_payload.get("abstained"),
                "candidate_generator_summary": zero_shot_payload.get("candidate_generator_summary"),
                "support_breakdown": zero_shot_payload.get("support_breakdown"),
            }
        output_path = args.output_root / "predictions" / token / "decomposition_result.json"
        _write(output_path, prediction)
        prediction_paths[token] = str(output_path)
        prediction_hashes[token] = _sha256(output_path)
        input_hashes[token] = _sha256(path)
        routes[token] = route
        counts[token] = len(predictions)

    manifest = {
        "schema_version": 1,
        "algorithm": "f1_mesh_topology_profile_v1",
        "phase": "predictions_frozen_before_score",
        "case_tokens": sorted(prediction_paths),
        "candidate_input_hashes": input_hashes,
        "prediction_paths": prediction_paths,
        "prediction_hashes": prediction_hashes,
        "prediction_counts": counts,
        "routes": routes,
        "inference_contract": {
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
            "uses_cad": False,
            "source_points_serialized": False,
        },
        "limitations": [
            "Thresholds were developed after visual review of this 12-scan development set.",
            "Predictions are not a blind benchmark and are not eligible for enterprise promotion.",
        ],
    }
    _write(args.output_root / "prediction_manifest.json", manifest)
    print(json.dumps({"prediction_counts": counts, "routes": routes}, indent=2))


def score(args: argparse.Namespace) -> None:
    manifest_path = args.output_root / "prediction_manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("phase") != "predictions_frozen_before_score":
        raise ValueError("prediction manifest is not frozen")
    thresholds = Thresholds()
    cases = []
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    exact_geometry = exact_count = 0
    label_hashes = {}
    for token in manifest["case_tokens"]:
        prediction_path = Path(manifest["prediction_paths"][token])
        if _sha256(prediction_path) != manifest["prediction_hashes"][token]:
            raise ValueError(f"{token}: frozen prediction changed")
        label_path = args.label_root / token / "bend_instance_labels.json"
        label_hashes[token] = _sha256(label_path)
        gold = _read(label_path).get("manual_bends") or []
        predictions = _read(prediction_path).get("owned_bend_regions") or []
        result = match_lines(gold, predictions, thresholds)
        for key in totals:
            totals[key] += int(result[key])
        exact_geometry += int(result["exact"])
        exact_count += int(len(gold) == len(predictions))
        cases.append({"case_token": token, **result})
    precision = totals["true_positive"] / max(
        totals["true_positive"] + totals["false_positive"], 1
    )
    recall = totals["true_positive"] / max(
        totals["true_positive"] + totals["false_negative"], 1
    )
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    metrics = {
        "case_count": len(cases),
        "gold_count": sum(case["gold_count"] for case in cases),
        "prediction_count": sum(case["prediction_count"] for case in cases),
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_geometry_case_count": exact_geometry,
        "exact_geometry_case_rate": exact_geometry / max(len(cases), 1),
        "exact_count_case_count": exact_count,
        "exact_count_case_rate": exact_count / max(len(cases), 1),
    }
    report = {
        "schema_version": 1,
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": _sha256(manifest_path),
        "external_label_hashes": label_hashes,
        "score_phase_only": True,
        "thresholds": thresholds.__dict__,
        "metrics": metrics,
        "cases": cases,
        "limitations": [
            "The 12 scans are a development set, not a blind benchmark.",
            "Labels are assistant-reviewed, drawing-backed, and user-review-pending.",
        ],
    }
    _write(args.output_root / "locked_score.json", report)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.candidate_root = args.candidate_root.resolve()
    args.label_root = args.label_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.phase == "predict":
        predict(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
