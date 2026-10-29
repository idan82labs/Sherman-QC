#!/usr/bin/env python3
"""Run the complete geometry-only F1 routed profile on the frozen gold27 inputs.

Inference consumes only opaque scan-derived patch graphs, zero-shot summaries,
and scan-only candidate pools. The sealed gold manifest is not opened here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_axial_support_split import promote_axial_support_lobes  # noqa: E402
from backend.f1_boundary_truncated_residual_pruning import prune_boundary_truncated_residuals  # noqa: E402
from backend.f1_corridor_plane_snap import snap_corridor_axes_by_plane_pareto  # noqa: E402
from backend.f1_dominant_topology_replacement import replace_complete_dominant_topology  # noqa: E402
from backend.f1_interface_birth_consensus import promote_repeated_interface_births  # noqa: E402
from backend.f1_interface_birth_stitch import promote_stitched_interface_births  # noqa: E402
from backend.f1_parallel_skin_plane_snap import snap_stitches_to_supported_skin_plane  # noqa: E402
from backend.f1_parallel_width_lattice_completion import complete_parallel_width_lattice  # noqa: E402
from backend.f1_partial_lattice_completion import complete_partial_lattice  # noqa: E402
from backend.f1_patch_graph_endpoint_frame_completion import complete_patch_graph_endpoint_frame  # noqa: E402
from backend.f1_prismatic_junction_completion import complete_prismatic_junction  # noqa: E402
from backend.f1_prismatic_transition_pruning import prune_weak_prismatic_transitions  # noqa: E402
from backend.f1_rolled_local_bend_completion import complete_rolled_local_bends  # noqa: E402
from backend.f1_rolled_long_strip_pruning import prune_rolled_long_strips  # noqa: E402
from backend.f1_rolled_parallel_doublet_pruning import prune_rolled_parallel_doublet  # noqa: E402
from backend.f1_transition_boundary_refit import promote_supported_transition_boundary  # noqa: E402
from backend.f1_zero_shot_source_router import route_f1_zero_shot_sources  # noqa: E402


BASE = ROOT / "output/research/f1-gold27-20260803"
DEFAULT_INFERENCE = BASE / "baseline_uniform_v1/inference_summary.json"
DEFAULT_FREEZE = BASE / "freeze/inference_manifest.json"
DEFAULT_POOL = BASE / "candidate_pool_v1"
DEFAULT_OUTPUT = BASE / "routed_profile_v18_replay_v1"
DEFAULT_PATCH_ROOTS = (
    ROOT / "output/research/f1-corpus-evidence-20260729-v1/patch_graph_fresh/inference/isolated_cases",
    ROOT / "output/enterprise_benchmark/shaked-count-backed-12-blind-baseline-20260717/isolated_cases",
    ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/sanitized_baseline/isolated_cases",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_payload(root_paths: tuple[Path, ...], source_stem: str) -> tuple[dict[str, Any], Path]:
    matches = []
    for root in root_paths:
        case_dir = root / source_stem
        if case_dir.is_dir():
            matches.extend(case_dir.glob("result/**/decomposition_result.json"))
    if len(matches) != 1:
        raise ValueError(f"{source_stem}: expected one patch payload, found {matches}")
    path = matches[0].resolve()
    payload = _read(path)
    provenance = payload.get("provenance") or {}
    if payload.get("part_id") is not None or provenance.get("coordinate_frame") != "scan_raw":
        raise ValueError(f"{source_stem}: patch payload violates strict blind scan-frame contract")
    return payload, path


def _zero_predictions(overlay_path: Path, result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = _read(overlay_path)
    strip_families = {
        str(strip.get("strip_id")): str(strip.get("strip_family") or "")
        for strip in result_payload.get("bend_support_strips") or []
    }
    predictions = []
    for bend in manifest.get("bends") or []:
        line = bend.get("detected_line") or bend.get("cad_line") or {}
        if line.get("display_start") is None or line.get("display_end") is None:
            continue
        bend_id = str(bend.get("bend_id") or f"ZS{len(predictions) + 1}")
        predictions.append(
            {
                "bend_id": bend_id,
                "span_endpoints": [line["display_start"], line["display_end"]],
                "source_kind": "zero_shot_support_strip",
                "strip_family": strip_families.get(bend_id, ""),
            }
        )
    return predictions


def _diagnostic(payload: dict[str, Any], excluded: set[str] | None = None) -> dict[str, Any]:
    excluded = excluded or {"selected"}
    return {key: value for key, value in payload.items() if key not in excluded}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-summary", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-root", type=Path, action="append")
    args = parser.parse_args()

    inference = _read(args.inference_summary.resolve())
    freeze = _read(args.freeze_manifest.resolve())
    pool_manifest = _read(args.candidate_root.resolve() / "manifest.json")
    if len({inference["corpus_freeze_id"], freeze["corpus_freeze_id"], pool_manifest["corpus_freeze_id"]}) != 1:
        raise ValueError("freeze ID mismatch")
    patch_roots = tuple(path.resolve() for path in (args.patch_root or DEFAULT_PATCH_ROOTS))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pool_cases = {str(case["case_token"]): case for case in pool_manifest["cases"]}

    freeze_cases = list(freeze["cases"])
    for index, inference_case in enumerate(freeze_cases, start=1):
        token = str(inference_case["case_token"])
        pool = _read(Path(pool_cases[token]["path"]))
        source_stem = str(pool["source_case_id"])
        patch, patch_path = _patch_payload(patch_roots, source_stem)
        zero_paths = inference["result_map"][token]
        zero = _read(Path(zero_paths["zero_shot_result"]))
        zero_selected = _zero_predictions(Path(zero_paths["overlay_manifest"]), zero)
        line_objects = pool.get("line_objects") or []
        strips = zero.get("bend_support_strips") or []

        snapped = snap_corridor_axes_by_plane_pareto(
            patch.get("owned_bend_regions") or [],
            patch.get("bend_hypotheses") or [],
            patch.get("flange_hypotheses") or [],
        )
        route = route_f1_zero_shot_sources(snapped["selected"], zero_selected, zero)
        birth = promote_repeated_interface_births(route["selected"], patch.get("bend_hypotheses") or [])
        stitch = promote_stitched_interface_births(birth["selected"], patch.get("bend_hypotheses") or [])
        skin = snap_stitches_to_supported_skin_plane(stitch["selected"], patch.get("flange_hypotheses") or [])
        transition = promote_supported_transition_boundary(
            skin["selected"],
            patch.get("bend_hypotheses") or [],
            patch.get("flange_hypotheses") or [],
            patch.get("atom_graph") or {},
        )
        axial = promote_axial_support_lobes(
            transition["selected"], patch.get("bend_hypotheses") or [], patch.get("atom_graph") or {}
        )
        topology = replace_complete_dominant_topology(axial["selected"])
        dense = complete_parallel_width_lattice(topology["selected"], strips, line_objects)
        boundary = prune_boundary_truncated_residuals(dense["selected"], route)
        rolled_doublet = prune_rolled_parallel_doublet(boundary["selected"], route, strips)
        rolled_long = prune_rolled_long_strips(rolled_doublet["selected"], route, strips)
        frame = complete_patch_graph_endpoint_frame(rolled_long["selected"], route, line_objects)
        rolled_local = complete_rolled_local_bends(frame["selected"], route, strips, line_objects)
        prismatic_pruning = prune_weak_prismatic_transitions(rolled_local["selected"], route, strips, line_objects)
        junction = complete_prismatic_junction(prismatic_pruning["selected"], route, strips, line_objects)
        partial = complete_partial_lattice(junction["selected"], route, strips, line_objects)
        selected = [dict(item) for item in partial["selected"]]

        case_dir = output_dir / "cases" / token
        case_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = case_dir / "prediction_owned_regions.json"
        prediction_path.write_text(json.dumps({"owned_bend_regions": selected}, indent=2) + "\n", encoding="utf-8")
        decomposition_path = case_dir / "decomposition_result.json"
        decomposition_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "algorithm": "f1_geometry_only_routed_profile_v18_replay",
                    "corpus_freeze_id": inference["corpus_freeze_id"],
                    "case_token": token,
                    "inference_contract": {
                        "uses_part_identifier": False,
                        "uses_expected_count": False,
                        "uses_manual_truth": False,
                        "uses_cad": False,
                        "source_point_serialization": False,
                    },
                    "source_route": _diagnostic(route, {"selected", "formed_feature_selected"}),
                    "corridor_plane_snap": _diagnostic(snapped),
                    "interface_birth_consensus": _diagnostic(birth, {"selected", "promoted"}),
                    "interface_birth_stitch": _diagnostic(stitch, {"selected", "promoted"}),
                    "parallel_skin_plane_snap": _diagnostic(skin),
                    "transition_boundary_refit": _diagnostic(transition, {"selected", "promoted"}),
                    "axial_support_split": _diagnostic(axial, {"selected", "promoted"}),
                    "dominant_topology_replacement": _diagnostic(topology),
                    "parallel_width_lattice_completion": _diagnostic(dense, {"selected", "promoted"}),
                    "boundary_truncated_residual_pruning": _diagnostic(boundary),
                    "rolled_parallel_doublet_pruning": _diagnostic(rolled_doublet),
                    "rolled_long_strip_pruning": _diagnostic(rolled_long),
                    "patch_graph_endpoint_frame_completion": _diagnostic(frame),
                    "rolled_local_bend_completion": _diagnostic(rolled_local),
                    "prismatic_transition_pruning": _diagnostic(prismatic_pruning),
                    "prismatic_junction_completion": _diagnostic(junction),
                    "partial_lattice_completion": _diagnostic(partial),
                    "owned_bend_regions": selected,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "case_token": token,
                "source_stem": source_stem,
                "patch_payload": str(patch_path),
                "route": route.get("route"),
                "prediction_count": len(selected),
                "prediction_path": str(prediction_path),
                "decomposition_path": str(decomposition_path),
            }
        )
        print(f"[{index}/{len(freeze_cases)}] {token} route={route.get('route')} bends={len(selected)}", flush=True)

    summary_path = output_dir / "inference_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sherman_qc_f1_gold27_routed_profile_inference",
                "corpus_freeze_id": inference["corpus_freeze_id"],
                "case_count": len(rows),
                "source_point_serialization": False,
                "cases": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(rows), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
