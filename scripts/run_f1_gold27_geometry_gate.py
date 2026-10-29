#!/usr/bin/env python3
"""Apply the fixed identity-free geometry gate between routed and mesh F1 lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_transition_dominant_outline_pruning import (
    prune_transition_dominant_outline_fan,
)
from backend.f1_complete_routed_consensus_preference import (
    prefer_complete_routed_consensus,
)
from backend.f1_surface_graph_dominance_pruning import prune_surface_graph_dominance
from backend.f1_seed_completed_outline_pruning import prune_seed_completed_outline_residuals
from backend.f1_parallel_return_doublet_completion import complete_parallel_return_doublet
from backend.f1_surface_star_completion import complete_strong_surface_star_leaf
from backend.f1_terminal_cycle_completion import complete_terminal_surface_cycles
from backend.f1_surface_triangle_split import split_supported_surface_triangle
from backend.f1_isolated_component_rewire import rewire_isolated_short_component
from backend.f1_orientation_isolated_leaf_pruning import prune_orientation_isolated_weak_leaf
from backend.f1_multiscale_consensus_completion import complete_multiscale_consensus
from backend.f1_near_coincident_mesh_duplicate_pruning import (
    prune_near_coincident_mesh_duplicates,
)
from backend.f1_overextended_mesh_fragment_replacement import replace_overextended_mesh_fragments
from backend.f1_opposite_fragment_completion import complete_opposite_fragment_pair
from backend.f1_residual_long_consensus_completion import complete_residual_long_consensus
from backend.f1_rectangular_frame_reconstruction import reconstruct_rectangular_frame
from backend.f1_formed_end_junction_reconstruction import reconstruct_formed_end_junction
from backend.f1_wall_end_corner_reconstruction import (
    has_wall_end_corner_signature,
    reconstruct_wall_end_corners,
)


BASE = ROOT / "output/research/f1-gold27-20260803"
DEFAULT_FREEZE = BASE / "freeze/inference_manifest.json"
DEFAULT_POOL = BASE / "candidate_pool_v1"
DEFAULT_ROUTED = BASE / "routed_profile_v18_replay_v1"
DEFAULT_MESH = BASE / "mesh_nested_lopo_v1"
DEFAULT_OUTPUT = BASE / "geometry_gate_v23"


def _load_transient_scan_points(path: Path) -> Any:
    import open3d as o3d
    import numpy as np

    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if len(points):
        return points
    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points, dtype=np.float64)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction(root: Path, token: str) -> list[dict[str, Any]]:
    return list(_read(root / "cases" / token / "prediction_owned_regions.json")["owned_bend_regions"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--routed-root", type=Path, default=DEFAULT_ROUTED)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-mesh-area-mm2", type=float, default=75_000.0)
    parser.add_argument("--minimum-transition-score", type=float, default=0.55)
    parser.add_argument("--minimum-large-mesh-area-mm2", type=float, default=120_000.0)
    parser.add_argument("--minimum-large-transition-score", type=float, default=0.90)
    parser.add_argument("--maximum-small-routed-count", type=int, default=3)
    parser.add_argument("--maximum-routed-to-mesh-count-ratio", type=float, default=2.0)
    args = parser.parse_args()

    freeze = _read(args.freeze_manifest.resolve())
    pool_manifest = _read(args.candidate_root.resolve() / "manifest.json")
    routed_summary = _read(args.routed_root.resolve() / "inference_summary.json")
    mesh_report = _read(args.mesh_root.resolve() / "mesh_lopo_metrics.json")
    if len(
        {
            freeze["corpus_freeze_id"],
            pool_manifest["corpus_freeze_id"],
            routed_summary["corpus_freeze_id"],
            mesh_report["corpus_freeze_id"],
        }
    ) != 1:
        raise ValueError("freeze ID mismatch")
    pool_cases = {str(case["case_token"]): case for case in pool_manifest["cases"]}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in freeze["cases"]:
        token = str(case["case_token"])
        pool = _read(Path(pool_cases[token]["path"]))
        routed = _prediction(args.routed_root.resolve(), token)
        mesh = _prediction(args.mesh_root.resolve(), token)
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
        compact_mesh_evidence = (
            mesh_area <= args.maximum_mesh_area_mm2
            and transition_score >= args.minimum_transition_score
        )
        large_route_disagreement = (
            mesh_area >= args.minimum_large_mesh_area_mm2
            and transition_score >= args.minimum_large_transition_score
            and (
                len(routed) <= args.maximum_small_routed_count
                or len(routed)
                > args.maximum_routed_to_mesh_count_ratio * max(len(mesh), 1)
            )
        )
        lane = (
            "mesh_nested_lopo"
            if compact_mesh_evidence or large_route_disagreement
            else "routed_profile_v18"
        )
        selected = mesh if lane == "mesh_nested_lopo" else routed
        outline_pruning = (
            prune_transition_dominant_outline_fan(selected)
            if lane == "mesh_nested_lopo"
            else {"selected": selected, "pruned": [], "activated": False, "reason": "non_mesh_lane"}
        )
        graph_pruning_first = (
            prune_surface_graph_dominance(outline_pruning["selected"])
            if lane == "mesh_nested_lopo"
            else {"selected": outline_pruning["selected"], "pruned": [], "activated": False}
        )
        graph_pruning_second = (
            prune_surface_graph_dominance(graph_pruning_first["selected"])
            if graph_pruning_first["activated"]
            else {"selected": graph_pruning_first["selected"], "pruned": [], "activated": False}
        )
        seed_outline_pruning = (
            prune_seed_completed_outline_residuals(graph_pruning_second["selected"])
            if lane == "mesh_nested_lopo"
            else {"selected": graph_pruning_second["selected"], "pruned": [], "activated": False, "reason": "non_mesh_lane"}
        )
        parallel_return_completion = (
            complete_parallel_return_doublet(
                seed_outline_pruning["selected"],
                pool.get("candidates") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": seed_outline_pruning["selected"],
                "added": [],
                "activated": False,
                "evidence": [],
            }
        )
        surface_star_completion = (
            complete_strong_surface_star_leaf(
                parallel_return_completion["selected"],
                pool.get("candidates") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": parallel_return_completion["selected"],
                "added": [],
                "activated": False,
                "evidence": [],
            }
        )
        terminal_cycle_completion = (
            complete_terminal_surface_cycles(
                surface_star_completion["selected"],
                pool.get("candidates") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": surface_star_completion["selected"],
                "added": [],
                "activated": False,
                "evidence": [],
            }
        )
        surface_triangle_split = (
            split_supported_surface_triangle(
                terminal_cycle_completion["selected"],
                pool.get("candidates") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": terminal_cycle_completion["selected"],
                "removed": [],
                "added": [],
                "activated": False,
                "evidence": [],
            }
        )
        isolated_component_rewire = (
            rewire_isolated_short_component(
                surface_triangle_split["selected"],
                pool.get("candidates") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": surface_triangle_split["selected"],
                "removed": [],
                "added": [],
                "activated": False,
                "evidence": [],
            }
        )
        orientation_leaf_pruning = (
            prune_orientation_isolated_weak_leaf(isolated_component_rewire["selected"])
            if lane == "mesh_nested_lopo"
            else {
                "selected": isolated_component_rewire["selected"],
                "pruned": [],
                "activated": False,
                "evidence": [],
            }
        )
        multiscale_consensus_completion = (
            complete_multiscale_consensus(
                orientation_leaf_pruning["selected"],
                pool.get("line_objects") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": orientation_leaf_pruning["selected"],
                "added": [],
                "activated": False,
                "reason": "non_mesh_lane",
                "evidence": [],
            }
        )
        overextended_fragment_replacement = (
            replace_overextended_mesh_fragments(
                multiscale_consensus_completion["selected"],
                pool.get("line_objects") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": multiscale_consensus_completion["selected"],
                "removed": [],
                "added": [],
                "activated": False,
                "evidence": [],
            }
        )
        opposite_fragment_completion = (
            complete_opposite_fragment_pair(
                overextended_fragment_replacement["selected"],
                pool.get("candidates") or [],
                pool.get("line_objects") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": overextended_fragment_replacement["selected"],
                "added": [],
                "activated": False,
                "evidence": [],
            }
        )
        residual_long_completion = (
            complete_residual_long_consensus(
                opposite_fragment_completion["selected"],
                pool.get("line_objects") or [],
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": opposite_fragment_completion["selected"],
                "added": [],
                "activated": False,
                "reason": "non_mesh_lane",
                "evidence": [],
            }
        )
        rectangular_frame_reconstruction = reconstruct_rectangular_frame(
            residual_long_completion["selected"],
            pool.get("candidates") or [],
        )
        formed_end_reconstruction = reconstruct_formed_end_junction(
            rectangular_frame_reconstruction["selected"],
            pool.get("candidates") or [],
            pool.get("line_objects") or [],
        )
        transient_points = None
        if has_wall_end_corner_signature(pool.get("candidates") or []):
            scan_path = Path(str(pool.get("scan_path") or "")).resolve()
            if not scan_path.is_file():
                raise FileNotFoundError(f"wall-end reconstruction scan missing: {scan_path}")
            transient_points = _load_transient_scan_points(scan_path)
        wall_end_reconstruction = reconstruct_wall_end_corners(
            formed_end_reconstruction["selected"],
            pool.get("candidates") or [],
            transient_points,
            point_budget=200_000,
        )
        near_coincident_mesh_duplicate_pruning = (
            prune_near_coincident_mesh_duplicates(
                wall_end_reconstruction["selected"]
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": wall_end_reconstruction["selected"],
                "pruned_candidate_ids": [],
                "conflicts": [],
                "activated": False,
            }
        )
        complete_routed_consensus_preference = (
            prefer_complete_routed_consensus(
                near_coincident_mesh_duplicate_pruning["selected"], routed
            )
            if lane == "mesh_nested_lopo"
            else {
                "selected": near_coincident_mesh_duplicate_pruning["selected"],
                "activated": False,
                "reason": "non_mesh_lane",
                "pairings": [],
            }
        )
        selected = list(complete_routed_consensus_preference["selected"])
        case_dir = output_dir / "cases" / token
        case_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = case_dir / "prediction_owned_regions.json"
        prediction_path.write_text(
            json.dumps({"owned_bend_regions": selected}, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "case_token": token,
                "selected_lane": lane,
                "mesh_area_mm2": mesh_area,
                "maximum_mesh_transition_score": transition_score,
                "compact_mesh_evidence": compact_mesh_evidence,
                "large_route_disagreement": large_route_disagreement,
                "prediction_count": len(selected),
                "transition_dominant_outline_pruning": {
                    key: value for key, value in outline_pruning.items() if key not in {"selected", "pruned"}
                },
                "transition_dominant_outline_pruned_count": len(outline_pruning["pruned"]),
                "surface_graph_dominance_pruning": {
                    "activated": bool(graph_pruning_first["activated"] or graph_pruning_second["activated"]),
                    "passes": [
                        {"activated": bool(graph_pruning_first["activated"]), "pruned": list(graph_pruning_first["pruned"])},
                        {"activated": bool(graph_pruning_second["activated"]), "pruned": list(graph_pruning_second["pruned"])},
                    ],
                },
                "seed_completed_outline_pruning": {
                    "activated": bool(seed_outline_pruning["activated"]),
                    "reason": seed_outline_pruning["reason"],
                    "pruned_count": len(seed_outline_pruning["pruned"]),
                },
                "parallel_return_doublet_completion": {
                    "activated": bool(parallel_return_completion["activated"]),
                    "added_count": len(parallel_return_completion["added"]),
                    "evidence": list(parallel_return_completion["evidence"]),
                },
                "surface_star_completion": {
                    "activated": bool(surface_star_completion["activated"]),
                    "added_count": len(surface_star_completion["added"]),
                    "evidence": list(surface_star_completion["evidence"]),
                },
                "terminal_cycle_completion": {
                    "activated": bool(terminal_cycle_completion["activated"]),
                    "added_count": len(terminal_cycle_completion["added"]),
                    "evidence": list(terminal_cycle_completion["evidence"]),
                },
                "surface_triangle_split": {
                    "activated": bool(surface_triangle_split["activated"]),
                    "removed_count": len(surface_triangle_split["removed"]),
                    "added_count": len(surface_triangle_split["added"]),
                    "evidence": list(surface_triangle_split["evidence"]),
                },
                "isolated_component_rewire": {
                    "activated": bool(isolated_component_rewire["activated"]),
                    "removed_count": len(isolated_component_rewire["removed"]),
                    "added_count": len(isolated_component_rewire["added"]),
                    "evidence": list(isolated_component_rewire["evidence"]),
                },
                "orientation_isolated_leaf_pruning": {
                    "activated": bool(orientation_leaf_pruning["activated"]),
                    "pruned_count": len(orientation_leaf_pruning["pruned"]),
                    "evidence": list(orientation_leaf_pruning["evidence"]),
                },
                "multiscale_consensus_completion": {
                    "activated": bool(multiscale_consensus_completion["activated"]),
                    "added_count": len(multiscale_consensus_completion["added"]),
                    "reason": multiscale_consensus_completion["reason"],
                    "evidence": list(multiscale_consensus_completion["evidence"]),
                },
                "overextended_mesh_fragment_replacement": {
                    "activated": bool(overextended_fragment_replacement["activated"]),
                    "removed_count": len(overextended_fragment_replacement["removed"]),
                    "added_count": len(overextended_fragment_replacement["added"]),
                    "evidence": list(overextended_fragment_replacement["evidence"]),
                },
                "opposite_fragment_completion": {
                    "activated": bool(opposite_fragment_completion["activated"]),
                    "added_count": len(opposite_fragment_completion["added"]),
                    "evidence": list(opposite_fragment_completion["evidence"]),
                },
                "residual_long_consensus_completion": {
                    "activated": bool(residual_long_completion["activated"]),
                    "added_count": len(residual_long_completion["added"]),
                    "reason": residual_long_completion["reason"],
                    "evidence": list(residual_long_completion["evidence"]),
                },
                "rectangular_frame_reconstruction": {
                    "activated": bool(rectangular_frame_reconstruction["activated"]),
                    "reconstructed_count": len(rectangular_frame_reconstruction["reconstructed"]),
                    "reason": rectangular_frame_reconstruction["reason"],
                    "replaced_prediction_count": int(
                        rectangular_frame_reconstruction.get("replaced_prediction_count") or 0
                    ),
                },
                "formed_end_junction_reconstruction": {
                    "activated": bool(formed_end_reconstruction["activated"]),
                    "reconstructed_count": len(formed_end_reconstruction["reconstructed"]),
                    "reason": formed_end_reconstruction["reason"],
                    "inferred_half_width_mm": formed_end_reconstruction.get(
                        "inferred_half_width_mm"
                    ),
                    "replaced_prediction_count": int(
                        formed_end_reconstruction.get("replaced_prediction_count") or 0
                    ),
                },
                "wall_end_corner_reconstruction": {
                    "activated": bool(wall_end_reconstruction["activated"]),
                    "reconstructed_count": len(wall_end_reconstruction["reconstructed"]),
                    "reason": wall_end_reconstruction["reason"],
                    "transient_input_point_count": int(
                        wall_end_reconstruction["transient_input_point_count"]
                    ),
                    "transient_sampled_point_count": int(
                        wall_end_reconstruction["transient_sampled_point_count"]
                    ),
                    "point_budget": int(wall_end_reconstruction["point_budget"]),
                    "selected_return_candidate_id": wall_end_reconstruction.get(
                        "selected_return_candidate_id"
                    ),
                },
                "near_coincident_mesh_duplicate_pruning": {
                    "activated": bool(
                        near_coincident_mesh_duplicate_pruning["activated"]
                    ),
                    "pruned_candidate_ids": list(
                        near_coincident_mesh_duplicate_pruning[
                            "pruned_candidate_ids"
                        ]
                    ),
                    "conflicts": list(
                        near_coincident_mesh_duplicate_pruning["conflicts"]
                    ),
                },
                "complete_routed_consensus_preference": {
                    "activated": bool(
                        complete_routed_consensus_preference["activated"]
                    ),
                    "reason": str(
                        complete_routed_consensus_preference["reason"]
                    ),
                    "pairings": list(
                        complete_routed_consensus_preference["pairings"]
                    ),
                },
                "prediction_path": str(prediction_path),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_geometry_gate_inference",
        "corpus_freeze_id": freeze["corpus_freeze_id"],
        "inference_contract": {
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
            "uses_cad": False,
            "source_point_serialization": False,
        },
        "gate": {
            "maximum_mesh_area_mm2": args.maximum_mesh_area_mm2,
            "minimum_transition_score": args.minimum_transition_score,
            "minimum_large_mesh_area_mm2": args.minimum_large_mesh_area_mm2,
            "minimum_large_transition_score": args.minimum_large_transition_score,
            "maximum_small_routed_count": args.maximum_small_routed_count,
            "maximum_routed_to_mesh_count_ratio": args.maximum_routed_to_mesh_count_ratio,
            "development_corpus_exposed_during_gate_selection": True,
            "locked_blind_claim": False,
        },
        "case_count": len(rows),
        "lane_counts": {
            lane: sum(row["selected_lane"] == lane for row in rows)
            for lane in ("mesh_nested_lopo", "routed_profile_v18")
        },
        "cases": rows,
    }
    report_path = output_dir / "inference_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "lane_counts": report["lane_counts"], "summary": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
