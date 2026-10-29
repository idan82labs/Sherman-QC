#!/usr/bin/env python3
"""Apply the post-lock Gold-27 F1 profile to one unseen strict-blind scan.

The external decomposition and candidate pool must already be generated in
scan-raw coordinates.  This runner fits the mesh selector on the frozen
Gold-27 development corpus, applies the fixed routed and mesh stacks, routes
between them using the v21 geometry gate, and applies the v22 deterministic
postprocessors.  It accepts no part identifier, expected count, manual truth,
or CAD input for the external scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_axial_support_split import promote_axial_support_lobes  # noqa: E402
from backend.f1_boundary_truncated_residual_pruning import prune_boundary_truncated_residuals  # noqa: E402
from backend.f1_corridor_plane_snap import snap_corridor_axes_by_plane_pareto  # noqa: E402
from backend.f1_complete_routed_consensus_preference import (  # noqa: E402
    prefer_complete_routed_consensus,
)
from backend.f1_dominant_topology_replacement import replace_complete_dominant_topology  # noqa: E402
from backend.f1_formed_end_junction_reconstruction import reconstruct_formed_end_junction  # noqa: E402
from backend.f1_interface_birth_consensus import promote_repeated_interface_births  # noqa: E402
from backend.f1_interface_birth_stitch import promote_stitched_interface_births  # noqa: E402
from backend.f1_isolated_component_rewire import rewire_isolated_short_component  # noqa: E402
from backend.f1_multiscale_consensus_completion import complete_multiscale_consensus  # noqa: E402
from backend.f1_near_coincident_mesh_duplicate_pruning import (  # noqa: E402
    prune_near_coincident_mesh_duplicates,
)
from backend.f1_opposite_fragment_completion import complete_opposite_fragment_pair  # noqa: E402
from backend.f1_orientation_isolated_leaf_pruning import prune_orientation_isolated_weak_leaf  # noqa: E402
from backend.f1_overextended_mesh_fragment_replacement import replace_overextended_mesh_fragments  # noqa: E402
from backend.f1_parallel_return_doublet_completion import complete_parallel_return_doublet  # noqa: E402
from backend.f1_parallel_skin_plane_snap import snap_stitches_to_supported_skin_plane  # noqa: E402
from backend.f1_parallel_width_lattice_completion import complete_parallel_width_lattice  # noqa: E402
from backend.f1_partial_lattice_completion import complete_partial_lattice  # noqa: E402
from backend.f1_patch_graph_endpoint_frame_completion import complete_patch_graph_endpoint_frame  # noqa: E402
from backend.f1_prismatic_junction_completion import complete_prismatic_junction  # noqa: E402
from backend.f1_prismatic_transition_pruning import prune_weak_prismatic_transitions  # noqa: E402
from backend.f1_rectangular_frame_reconstruction import reconstruct_rectangular_frame  # noqa: E402
from backend.f1_residual_long_consensus_completion import complete_residual_long_consensus  # noqa: E402
from backend.f1_rolled_local_bend_completion import complete_rolled_local_bends  # noqa: E402
from backend.f1_rolled_long_strip_pruning import prune_rolled_long_strips  # noqa: E402
from backend.f1_rolled_parallel_doublet_pruning import prune_rolled_parallel_doublet  # noqa: E402
from backend.f1_seed_completed_outline_pruning import prune_seed_completed_outline_residuals  # noqa: E402
from backend.f1_surface_graph_dominance_pruning import prune_surface_graph_dominance  # noqa: E402
from backend.f1_surface_star_completion import complete_strong_surface_star_leaf  # noqa: E402
from backend.f1_surface_triangle_split import split_supported_surface_triangle  # noqa: E402
from backend.f1_terminal_cycle_completion import complete_terminal_surface_cycles  # noqa: E402
from backend.f1_transition_boundary_refit import promote_supported_transition_boundary  # noqa: E402
from backend.f1_transition_dominant_outline_pruning import prune_transition_dominant_outline_fan  # noqa: E402
from backend.f1_wall_end_corner_reconstruction import (  # noqa: E402
    has_wall_end_corner_signature,
    reconstruct_wall_end_corners,
)
from backend.f1_zero_shot_source_router import route_f1_zero_shot_sources  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds  # noqa: E402
from scripts.run_f1_gold27_geometry_gate import _load_transient_scan_points  # noqa: E402
from scripts.run_f1_gold27_mesh_crossfit import (  # noqa: E402
    _apply_stack as apply_mesh_stack,
    _fit_model,
    _predict_model,
)
from scripts.run_f1_gold27_mesh_lopo import FIXED_CONFIG  # noqa: E402
from scripts.run_f1_grouped_holdout_selector import _match_quality  # noqa: E402
from scripts.run_shaked_selector_evaluation import _mesh_case_features, _select  # noqa: E402


DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_TRAINING_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"


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


def zero_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert the zero-shot render contract to routed-profile line records."""

    predictions = []
    for bend in result.get("renderable_bend_objects") or []:
        line = bend.get("detected_line") or bend.get("cad_line") or {}
        start = line.get("display_start")
        end = line.get("display_end")
        if start is None or end is None:
            continue
        predictions.append(
            {
                "bend_id": str(bend.get("bend_id") or f"ZS{len(predictions) + 1}"),
                "span_endpoints": [start, end],
                "source_kind": "zero_shot_support_strip",
                "strip_family": str((bend.get("metadata") or {}).get("strip_family") or ""),
            }
        )
    return predictions


def routed_prediction(
    decomposition: dict[str, Any],
    pool: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    zero = dict(decomposition.get("current_zero_shot_result") or {})
    zero_selected = zero_predictions(zero)
    line_objects = pool.get("line_objects") or []
    strips = zero.get("bend_support_strips") or []

    snapped = snap_corridor_axes_by_plane_pareto(
        decomposition.get("owned_bend_regions") or [],
        decomposition.get("bend_hypotheses") or [],
        decomposition.get("flange_hypotheses") or [],
    )
    route = route_f1_zero_shot_sources(snapped["selected"], zero_selected, zero)
    birth = promote_repeated_interface_births(
        route["selected"], decomposition.get("bend_hypotheses") or []
    )
    stitch = promote_stitched_interface_births(
        birth["selected"], decomposition.get("bend_hypotheses") or []
    )
    skin = snap_stitches_to_supported_skin_plane(
        stitch["selected"], decomposition.get("flange_hypotheses") or []
    )
    transition = promote_supported_transition_boundary(
        skin["selected"],
        decomposition.get("bend_hypotheses") or [],
        decomposition.get("flange_hypotheses") or [],
        decomposition.get("atom_graph") or {},
    )
    axial = promote_axial_support_lobes(
        transition["selected"],
        decomposition.get("bend_hypotheses") or [],
        decomposition.get("atom_graph") or {},
    )
    topology = replace_complete_dominant_topology(axial["selected"])
    dense = complete_parallel_width_lattice(topology["selected"], strips, line_objects)
    boundary = prune_boundary_truncated_residuals(dense["selected"], route)
    rolled_doublet = prune_rolled_parallel_doublet(boundary["selected"], route, strips)
    rolled_long = prune_rolled_long_strips(rolled_doublet["selected"], route, strips)
    frame = complete_patch_graph_endpoint_frame(rolled_long["selected"], route, line_objects)
    rolled_local = complete_rolled_local_bends(frame["selected"], route, strips, line_objects)
    prismatic_pruning = prune_weak_prismatic_transitions(
        rolled_local["selected"], route, strips, line_objects
    )
    junction = complete_prismatic_junction(
        prismatic_pruning["selected"], route, strips, line_objects
    )
    partial = complete_partial_lattice(junction["selected"], route, strips, line_objects)
    return [dict(item) for item in partial["selected"]], {
        "source_route": route.get("route"),
        "zero_shot_line_count": len(zero_selected),
        "routed_prediction_count": len(partial["selected"]),
    }


def fit_gold27_mesh_model(
    gold_manifest: Path,
    training_pool_root: Path,
) -> tuple[Any, dict[str, Any]]:
    sealed = _read(gold_manifest)
    rows: dict[str, list[tuple[dict[str, Any], Any, int, float]]] = {}
    for case in sealed["cases"]:
        token = str(case["case_token"])
        payload = _read(training_pool_root / f"{token}.json")
        candidates = [
            dict(item)
            for item in payload.get("candidates") or []
            if item.get("candidate_source") == "scan_mesh_adjacency"
        ]
        features = _mesh_case_features(candidates, payload) if candidates else []
        gold = list(case.get("manual_bends") or [])
        rows[token] = [
            (
                candidate,
                feature,
                int((quality := _match_quality(candidate, gold, Thresholds())) > 0),
                quality,
            )
            for candidate, feature in zip(candidates, features)
        ]
    training_tokens = [token for token in sorted(rows) if rows[token]]
    model = _fit_model(rows, training_tokens, "logistic")
    return model, {
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "training_scan_count": len(sealed["cases"]),
        "training_case_count_with_mesh_candidates": len(training_tokens),
        "config": dict(FIXED_CONFIG),
    }


def mesh_prediction(
    pool: dict[str, Any],
    model: Any,
) -> list[dict[str, Any]]:
    candidates = [
        dict(item)
        for item in pool.get("candidates") or []
        if item.get("candidate_source") == "scan_mesh_adjacency"
    ]
    if not candidates:
        return []
    features = _mesh_case_features(candidates, pool)
    case_rows = [
        (candidate, feature, 0, 0.0)
        for candidate, feature in zip(candidates, features)
    ]
    raw = _predict_model(model, case_rows, "logistic")
    selected = _select(raw, FIXED_CONFIG)
    selected = apply_mesh_stack(selected, pool, FIXED_CONFIG)
    for index, candidate in enumerate(selected, start=1):
        candidate["bend_id"] = f"F1U{index}"
    return selected


def apply_v21_geometry_stack(
    selected: list[dict[str, Any]],
    pool: dict[str, Any],
    *,
    mesh_lane: bool,
    scan_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = pool.get("candidates") or []
    line_objects = pool.get("line_objects") or []
    outline = (
        prune_transition_dominant_outline_fan(selected)
        if mesh_lane
        else {"selected": selected, "pruned": [], "activated": False}
    )
    graph_first = (
        prune_surface_graph_dominance(outline["selected"])
        if mesh_lane
        else {"selected": outline["selected"], "pruned": [], "activated": False}
    )
    graph_second = (
        prune_surface_graph_dominance(graph_first["selected"])
        if graph_first["activated"]
        else {"selected": graph_first["selected"], "pruned": [], "activated": False}
    )
    seed = (
        prune_seed_completed_outline_residuals(graph_second["selected"])
        if mesh_lane
        else {"selected": graph_second["selected"], "pruned": [], "activated": False}
    )

    def complete(function: Any, current: list[dict[str, Any]], *extra: Any) -> dict[str, Any]:
        if mesh_lane:
            return function(current, *extra)
        return {"selected": current, "added": [], "activated": False, "evidence": []}

    parallel = complete(complete_parallel_return_doublet, seed["selected"], candidates)
    star = complete(complete_strong_surface_star_leaf, parallel["selected"], candidates)
    cycle = complete(complete_terminal_surface_cycles, star["selected"], candidates)
    triangle = (
        split_supported_surface_triangle(cycle["selected"], candidates)
        if mesh_lane
        else {"selected": cycle["selected"], "removed": [], "added": [], "activated": False}
    )
    rewire = (
        rewire_isolated_short_component(triangle["selected"], candidates)
        if mesh_lane
        else {"selected": triangle["selected"], "removed": [], "added": [], "activated": False}
    )
    leaf = (
        prune_orientation_isolated_weak_leaf(rewire["selected"])
        if mesh_lane
        else {"selected": rewire["selected"], "pruned": [], "activated": False}
    )
    multiscale = complete(complete_multiscale_consensus, leaf["selected"], line_objects)
    replacement = (
        replace_overextended_mesh_fragments(multiscale["selected"], line_objects)
        if mesh_lane
        else {"selected": multiscale["selected"], "removed": [], "added": [], "activated": False}
    )
    opposite = complete(
        complete_opposite_fragment_pair,
        replacement["selected"],
        candidates,
        line_objects,
    )
    residual = complete(
        complete_residual_long_consensus,
        opposite["selected"],
        line_objects,
    )
    rectangular = reconstruct_rectangular_frame(residual["selected"], candidates)
    formed = reconstruct_formed_end_junction(
        rectangular["selected"], candidates, line_objects
    )
    transient_points = (
        _load_transient_scan_points(scan_path)
        if has_wall_end_corner_signature(candidates)
        else None
    )
    wall = reconstruct_wall_end_corners(
        formed["selected"], candidates, transient_points, point_budget=200_000
    )
    near_duplicate = (
        prune_near_coincident_mesh_duplicates(wall["selected"])
        if mesh_lane
        else {
            "selected": wall["selected"],
            "pruned_candidate_ids": [],
            "conflicts": [],
            "activated": False,
        }
    )
    diagnostics = {
        "transition_outline_pruning": bool(outline["activated"]),
        "surface_graph_pruning": bool(graph_first["activated"] or graph_second["activated"]),
        "seed_outline_pruning": bool(seed["activated"]),
        "parallel_return_completion": bool(parallel["activated"]),
        "surface_star_completion": bool(star["activated"]),
        "terminal_cycle_completion": bool(cycle["activated"]),
        "surface_triangle_split": bool(triangle["activated"]),
        "isolated_component_rewire": bool(rewire["activated"]),
        "orientation_leaf_pruning": bool(leaf["activated"]),
        "multiscale_completion": bool(multiscale["activated"]),
        "overextended_fragment_replacement": bool(replacement["activated"]),
        "opposite_fragment_completion": bool(opposite["activated"]),
        "residual_long_completion": bool(residual["activated"]),
        "rectangular_frame_reconstruction": bool(rectangular["activated"]),
        "formed_end_junction_reconstruction": bool(formed["activated"]),
        "wall_end_corner_reconstruction": bool(wall["activated"]),
        "wall_end_sampled_point_count": int(wall["transient_sampled_point_count"]),
        "near_coincident_mesh_duplicate_pruning": bool(
            near_duplicate["activated"]
        ),
        "near_coincident_mesh_duplicate_pruned_candidate_ids": list(
            near_duplicate["pruned_candidate_ids"]
        ),
        "near_coincident_mesh_duplicate_conflicts": list(
            near_duplicate["conflicts"]
        ),
    }
    return [dict(item) for item in near_duplicate["selected"]], diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--training-pool-root", type=Path, default=DEFAULT_TRAINING_POOL)
    args = parser.parse_args()

    decomposition_path = args.decomposition.resolve()
    candidate_pool_path = args.candidate_pool.resolve()
    decomposition = _read(decomposition_path)
    pool = _read(candidate_pool_path)
    provenance = dict(decomposition.get("provenance") or {})
    if decomposition.get("part_id") is not None:
        raise ValueError("unseen inference requires opaque part identity")
    if provenance.get("coordinate_frame") != "scan_raw":
        raise ValueError("unseen inference requires scan_raw coordinates")
    contract = dict(pool.get("inference_contract") or {})
    if any(
        contract.get(key)
        for key in (
            "part_identifier_exposed",
            "expected_count_exposed",
            "manual_truth_exposed",
            "cad_exposed",
        )
    ):
        raise ValueError("candidate pool violates strict-blind inference contract")
    if pool.get("source_point_serialization") is not False:
        raise ValueError("candidate pool must not serialize source points")
    scan_path = Path(str(decomposition.get("scan_path") or "")).resolve()
    if not scan_path.is_file():
        raise FileNotFoundError(scan_path)

    routed, routed_diagnostics = routed_prediction(decomposition, pool)
    model, training = fit_gold27_mesh_model(
        args.gold_manifest.resolve(), args.training_pool_root.resolve()
    )
    mesh = mesh_prediction(pool, model)
    mesh_area = float(
        pool.get("generator_diagnostics", {})
        .get("scan_mesh_adjacency", {})
        .get("mesh_area")
        or 0.0
    )
    transition_score = max(
        (
            float(
                item.get("two_sided_transition_score")
                or item.get("cluster_max_two_sided_transition_score")
                or 0.0
            )
            for item in mesh
        ),
        default=0.0,
    )
    compact_mesh_evidence = mesh_area <= 75_000.0 and transition_score >= 0.55
    large_route_disagreement = (
        mesh_area >= 120_000.0
        and transition_score >= 0.90
        and (len(routed) <= 3 or len(routed) > 2.0 * max(len(mesh), 1))
    )
    lane = (
        "mesh_gold27_train_all_fixed"
        if compact_mesh_evidence or large_route_disagreement
        else "routed_profile_v18"
    )
    selected, post = apply_v21_geometry_stack(
        mesh if lane.startswith("mesh") else routed,
        pool,
        mesh_lane=lane.startswith("mesh"),
        scan_path=scan_path,
    )
    routed_consensus = (
        prefer_complete_routed_consensus(selected, routed)
        if lane.startswith("mesh")
        else {
            "selected": selected,
            "activated": False,
            "reason": "non_mesh_lane",
            "pairings": [],
        }
    )
    selected = [dict(item) for item in routed_consensus["selected"]]
    post["complete_routed_consensus_preference"] = bool(
        routed_consensus["activated"]
    )
    post["complete_routed_consensus_reason"] = str(
        routed_consensus["reason"]
    )
    post["complete_routed_consensus_pairings"] = list(
        routed_consensus["pairings"]
    )
    for index, candidate in enumerate(selected, start=1):
        candidate["bend_id"] = str(candidate.get("bend_id") or f"F1U_FINAL_{index}")

    output_dir = args.output_dir.resolve()
    prediction_path = output_dir / "prediction_owned_regions.json"
    _write(
        prediction_path,
        {"scan_path": str(scan_path), "owned_bend_regions": selected},
    )
    _write(
        output_dir / "routed_prediction_owned_regions.json",
        {"scan_path": str(scan_path), "owned_bend_regions": routed},
    )
    _write(
        output_dir / "mesh_prediction_owned_regions.json",
        {"scan_path": str(scan_path), "owned_bend_regions": mesh},
    )
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_unseen_profile_inference",
        "algorithm": "f1_gold27_v23_unseen_fixed_v1",
        "training": training,
        "external_inference_contract": {
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
            "uses_cad": False,
            "source_point_serialization": False,
            "human_operator_may_know_count": True,
        },
        "claim": "prediction_generated_without_external_truth_input",
        "inputs": {
            "scan_sha256": _sha256(scan_path),
            "decomposition_sha256": _sha256(decomposition_path),
            "candidate_pool_sha256": _sha256(candidate_pool_path),
        },
        "routing": {
            "selected_lane": lane,
            "mesh_area_mm2": mesh_area,
            "maximum_mesh_transition_score": transition_score,
            "compact_mesh_evidence": compact_mesh_evidence,
            "large_route_disagreement": large_route_disagreement,
            "routed_prediction_count": len(routed),
            "mesh_prediction_count": len(mesh),
        },
        "routed_diagnostics": routed_diagnostics,
        "v21_postprocessor_activations": post,
        "prediction_count": len(selected),
        "prediction_path": str(prediction_path),
    }
    _write(output_dir / "inference_summary.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
