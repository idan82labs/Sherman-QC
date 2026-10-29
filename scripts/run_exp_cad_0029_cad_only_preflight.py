#!/usr/bin/env python3
"""Scan-free CAD actionability preflight for EXP_CAD_0029 CF-PDG.

This CLI intentionally has no scan, evaluator, role, label, gold, expected-
count, or quality-name argument.  A PASS freezes the deterministic CAD graph;
a failure is the preregistered terminal CAD-only STOP and leaves scans unopened.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import exp_cad_0029_cfpdg_core as CORE


EXPECTED_SCHEMA = "EXP_CAD_0029_CFPDG_PREREGISTRATION_V2_PRE_SCAN_AUDIT_AMENDMENT"
EXPECTED_EXPERIMENT_ID = "EXP_CAD_0029"
EXPECTED_ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V2_HARDENED"
OUTPUT_PREFLIGHT_SCHEMA = "EXP_CAD_0029_CAD_ONLY_PREFLIGHT_V2"
IDENTITY_WRAPPER_PATH: Optional[Path] = None


def _structured_nodes(
    points: np.ndarray,
    normals: np.ndarray,
    first_tangent: np.ndarray,
    second_tangent: np.ndarray,
    parent_faces: np.ndarray,
    ordinals: np.ndarray,
    smooth_regions: np.ndarray,
    area_weights: np.ndarray,
    fold_ids: np.ndarray,
    graph_components: np.ndarray,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
            ("e1x", "<f8"), ("e1y", "<f8"), ("e1z", "<f8"),
            ("e2x", "<f8"), ("e2y", "<f8"), ("e2z", "<f8"),
            ("parent_face", "<i8"), ("candidate_ordinal", "<i8"),
            ("smooth_region", "<i8"), ("area_mm2", "<f8"),
            ("fold_id", "u1"), ("graph_component", "<i8"),
        ]
    )
    output = np.empty(len(points), dtype=dtype)
    for names, values in (
        (("x", "y", "z"), points),
        (("nx", "ny", "nz"), normals),
        (("e1x", "e1y", "e1z"), first_tangent),
        (("e2x", "e2y", "e2z"), second_tangent),
    ):
        for column, name in enumerate(names):
            output[name] = values[:, column]
    output["parent_face"] = parent_faces
    output["candidate_ordinal"] = ordinals
    output["smooth_region"] = smooth_regions
    output["area_mm2"] = area_weights
    output["fold_id"] = fold_ids
    output["graph_component"] = graph_components
    return output


def execute_preflight(
    *,
    preregistration_path: Path,
    source_cad_path: Path,
    mesh_path: Path,
    mesh_report_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve()
    source_cad_path = source_cad_path.resolve()
    mesh_path = mesh_path.resolve()
    mesh_report_path = mesh_report_path.resolve()
    output_root = output_root.resolve()
    prereg = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if prereg.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected EXP29 preregistration schema")
    if prereg.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("unexpected experiment id")
    if prereg.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("unexpected architecture")
    if prereg.get("preregistered_before_physical_scan_load") is not True:
        raise ValueError("preregistration does not assert pre-scan freeze")
    forbidden = prereg.get("forbidden_inputs", {})
    required_forbidden = {
        "evaluator_manifest", "roles", "labels", "gold", "expected_counts",
        "quality_tokens_or_filenames", "bend_event_or_ntrc_guard_rescue",
        "local_or_component_specific_pose_optimization",
    }
    if not required_forbidden.issubset(forbidden) or not all(
        forbidden[name] is True for name in required_forbidden
    ):
        raise ValueError("forbidden-input contract is incomplete")

    immutable = prereg["immutable_inputs"]
    exact_hashes = {
        source_cad_path: immutable["source_cad_sha256"],
        mesh_path: immutable["accepted_mesh_sha256"],
        mesh_report_path: immutable["mesh_report_sha256"],
    }
    for path, expected in exact_hashes.items():
        actual = CORE.sha256_path(path)
        if actual != expected:
            raise ValueError(f"immutable CAD-only input mismatch: {path.name}")
    mesh_report = json.loads(mesh_report_path.read_text(encoding="utf-8"))
    if mesh_report.get("evaluator_data_opened") is not False:
        raise ValueError("mesh report opened evaluator data")
    if mesh_report.get("labels_or_expected_counts_used") is not False:
        raise ValueError("mesh report used labels or expected counts")
    if mesh_report.get("status") != "SOURCE_ONLY_ENGINE_MESH_READY":
        raise ValueError("accepted CAD mesh report is not source-only ready")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError("accepted CAD mesh is empty")
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                 vertices[triangles[:, 2]] - vertices[triangles[:, 0]]), axis=1
    )
    if (
        not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(normals))
        or np.any(triangle_areas <= 1e-12)
    ):
        raise ValueError("accepted mesh contains invalid geometry")
    total_mesh_area = math.fsum(float(value) for value in triangle_areas)

    graph_config = prereg["cad_graph"]
    topology = CORE.welded_mesh_topology(
        vertices, triangles, normals,
        weld_tolerance_mm=float(graph_config["coordinate_weld_tolerance_mm"]),
        smooth_angle_deg=float(graph_config["smooth_region_max_dihedral_deg_inclusive"]),
    )
    normals = np.asarray(topology["welded_triangle_normals"], dtype=np.float64)
    candidates = CORE.deterministic_equal_area_candidates(
        vertices, triangles, normals,
        float(graph_config["candidate_target_area_mm2"]),
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.int64)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    diagnostic_probe_mask = CORE.cad_interior_mask(
        scene,
        candidates["points"],
        candidates["normals"],
        radii_mm=graph_config["interior_probe_radii_mm"],
        azimuth_count=int(graph_config["interior_probe_azimuth_count"]),
        maximum_distance_mm=float(graph_config["interior_probe_max_surface_distance_mm"]),
        maximum_normal_angle_deg=float(graph_config["interior_probe_max_normal_angle_deg"]),
        parent_faces=candidates["parent_faces"],
        smooth_region_by_face=topology["smooth_region_labels"],
    )
    boundary_clearance_mm = CORE.smooth_region_boundary_clearance_mm(
        candidates["points"],
        candidates["parent_faces"],
        topology["smooth_region_labels"],
        topology["welded_vertices"],
        topology["boundary_edges_by_smooth_region"],
    )
    interior_mask = boundary_clearance_mm >= float(
        graph_config["minimum_boundary_or_hole_mesh_geodesic_clearance_mm"]
    )
    node_candidate_indices = CORE.deterministic_poisson_nodes(
        candidates,
        interior_mask,
        topology["smooth_region_labels"],
        minimum_spacing_mm=float(graph_config["poisson_min_spacing_mm"]),
    )
    node_points = candidates["points"][node_candidate_indices]
    node_normals = candidates["normals"][node_candidate_indices]
    node_parents = candidates["parent_faces"][node_candidate_indices]
    node_ordinals = candidates["ordinals"][node_candidate_indices]
    node_smooth_regions = topology["smooth_region_labels"][node_parents]
    node_area_weights, assigned_candidate_mask = CORE.assign_node_area_weights(
        candidates,
        interior_mask,
        node_candidate_indices,
        topology["smooth_region_labels"],
        maximum_distance_mm=float(graph_config["node_area_assignment_max_distance_mm"]),
    )
    if np.any(node_area_weights <= 0.0):
        raise ValueError("a deterministic CAD node received zero area")
    edges, edge_geodesic_mm = CORE.topology_qualified_graph_edges(
        node_points,
        node_parents,
        node_smooth_regions,
        topology,
        maximum_length_mm=float(graph_config["graph_edge_max_length_mm"]),
        maximum_geodesic_to_euclidean_ratio=float(
            graph_config["graph_edge_max_surface_geodesic_to_euclidean_ratio"]
        ),
        atlas_edge_sample_spacing_mm=float(
            graph_config["surface_path_atlas_edge_sample_spacing_mm"]
        ),
    )
    if len(edges):
        edge_order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[edge_order]
        edge_geodesic_mm = edge_geodesic_mm[edge_order]
    graph_components = CORE.graph_component_labels(len(node_points), edges)
    degrees = np.bincount(edges.reshape(-1), minlength=len(node_points)) if len(edges) else np.zeros(len(node_points), dtype=np.int64)
    weighted_graph = CORE.weighted_node_graph(
        len(node_points), edges, edge_geodesic_mm
    )
    fold_ids, fold_basis = CORE.spatial_pca_fold_ids(
        node_points,
        node_area_weights,
        fold_count=int(prereg["cad_only_actionability"]["fold_count"]),
    )
    first_tangent, second_tangent = CORE.canonical_tangent_basis(node_normals)
    _design, area_centroid, singular = CORE.rigid_design_matrix(
        node_points, node_normals, node_area_weights
    )
    design_rank, design_condition = CORE.design_rank_condition(singular)
    halo_training_masks = CORE.fold_training_masks_with_halo(
        node_points,
        node_smooth_regions,
        fold_ids,
        weighted_graph,
        fold_count=int(prereg["cad_only_actionability"]["fold_count"]),
        halo_mm=float(
            prereg["cad_only_actionability"][
                "fold_exclusion_halo_mm_strictly_greater_than"
            ]
        ),
    )
    fold_design = []
    for fold, training in enumerate(halo_training_masks):
        _rows, _centroid, values = CORE.rigid_design_matrix(
            node_points[training],
            node_normals[training],
            node_area_weights[training],
            fixed_area_centroid=area_centroid,
        )
        rank, condition = CORE.design_rank_condition(values)
        fold_design.append(
            {
                "held_out_fold": fold,
                "held_out_node_count": int(np.count_nonzero(fold_ids == fold)),
                "post_halo_training_node_count": int(np.count_nonzero(training)),
                "post_halo_training_assigned_area_mm2": math.fsum(
                    float(value) for value in node_area_weights[training]
                ),
                "training_rank": rank,
                "training_condition_number": condition,
            }
        )

    candidate_total_area = math.fsum(
        float(value) for value in candidates["area_weights_mm2"]
    )
    interior_area = math.fsum(
        float(value)
        for value in np.asarray(candidates["area_weights_mm2"])[interior_mask]
    )
    assigned_area = math.fsum(float(value) for value in node_area_weights)
    unassigned_area = math.fsum(
        float(value)
        for value in np.asarray(candidates["area_weights_mm2"])[
            interior_mask & ~assigned_candidate_mask
        ]
    )
    excluded_boundary_area = math.fsum(
        float(value)
        for value in np.asarray(candidates["area_weights_mm2"])[~interior_mask]
    )
    component_areas = np.bincount(
        graph_components, weights=node_area_weights,
        minlength=int(np.max(graph_components)) + 1 if len(graph_components) else 0,
    )
    largest_component_area = float(np.max(component_areas)) if len(component_areas) else 0.0
    fold_areas = np.bincount(fold_ids, weights=node_area_weights, minlength=4)
    fold_area_fractions = fold_areas / assigned_area
    component_spans, component_span_records = CORE.component_two_sweep_geodesic_spans(
        graph_components, weighted_graph
    )
    mesh_component_count = int(len(np.unique(topology["mesh_component_labels"])))

    gates_config = prereg["cad_only_actionability"]
    gates = {
        "FINITE_NONDEGENERATE_TRIANGLES": True,
        "NONMANIFOLD_EDGE_GATE": topology["nonmanifold_edge_count"] <= int(gates_config["maximum_nonmanifold_edge_count"]),
        "MESH_COMPONENT_COUNT_GATE": mesh_component_count == int(gates_config["required_mesh_component_count"]),
        "NODE_COUNT_GATE": len(node_points) >= int(gates_config["minimum_node_count"]),
        "AREA_ACCOUNTING_GATE": bool(
            abs(candidate_total_area - interior_area - excluded_boundary_area)
            <= 1e-12 * max(1.0, candidate_total_area)
            and abs(interior_area - assigned_area - unassigned_area)
            <= 1e-12 * max(1.0, interior_area)
            and abs(candidate_total_area - total_mesh_area)
            <= 1e-12 * max(1.0, total_mesh_area)
        ),
        "INTERIOR_AREA_FRACTION_GATE": interior_area / total_mesh_area >= float(gates_config["minimum_interior_area_fraction"]),
        "ASSIGNED_INTERIOR_AREA_GATE": assigned_area / interior_area >= float(gates_config["minimum_assigned_fraction_of_interior_area"]),
        "UNASSIGNED_INTERIOR_AREA_GATE": unassigned_area / interior_area <= float(gates_config["maximum_unassigned_fraction_of_interior_area"]),
        "LARGEST_GRAPH_COMPONENT_AREA_GATE": largest_component_area / assigned_area >= float(gates_config["minimum_largest_graph_component_area_fraction"]),
        "DEGREE_TWO_AREA_FRACTION_GATE": math.fsum(
            float(value) for value in node_area_weights[degrees >= 2]
        ) / assigned_area >= float(gates_config["minimum_degree_two_assigned_area_fraction"]),
        "FOLD_AREA_GATE": bool(
            np.all(fold_area_fractions >= float(gates_config["minimum_fold_area_fraction"]))
            and np.all(fold_area_fractions <= float(gates_config["maximum_fold_area_fraction"]))
        ),
        "PCA_EIGENGAP_GATE": bool(
            fold_basis["leading_to_second_eigenvalue_ratio"]
            >= float(gates_config["minimum_pca_leading_to_second_eigenvalue_ratio"])
        ),
        "GLOBAL_RIGID_DESIGN_GATE": bool(
            design_rank == int(gates_config["required_rigid_design_rank"])
            and design_condition <= float(gates_config["maximum_normalized_rigid_design_condition_number"])
        ),
        "FOLD_RIGID_DESIGN_GATE": all(
            row["post_halo_training_node_count"]
            >= int(gates_config["minimum_post_halo_training_node_count"])
            and row["post_halo_training_assigned_area_mm2"] / assigned_area
            >= float(gates_config["minimum_post_halo_training_assigned_area_fraction"])
            and row["training_rank"] == int(gates_config["required_rigid_design_rank"])
            and row["training_condition_number"] <= float(gates_config["maximum_normalized_rigid_design_condition_number"])
            for row in fold_design
        ),
    }
    passed = all(gates.values())

    output_root.mkdir(parents=True, exist_ok=True)
    node_path = output_root / "CAD_GRAPH_NODES.npy"
    edge_path = output_root / "CAD_GRAPH_EDGES.npy"
    edge_weight_path = output_root / "CAD_GRAPH_EDGE_GEODESIC_MM.npy"
    node_array = _structured_nodes(
        node_points, node_normals, first_tangent, second_tangent,
        node_parents, node_ordinals, node_smooth_regions,
        node_area_weights, fold_ids, graph_components,
    )
    np.save(node_path, node_array, allow_pickle=False)
    np.save(edge_path, np.asarray(edges, dtype="<i8"), allow_pickle=False)
    np.save(edge_weight_path, np.asarray(edge_geodesic_mm, dtype="<f8"), allow_pickle=False)
    payload = {
        "schema": OUTPUT_PREFLIGHT_SCHEMA,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "architecture": EXPECTED_ARCHITECTURE,
        "status": "PASS_CAD_ONLY_ACTIONABLE" if passed else gates_config["failure_ruling"],
        "preregistration_sha256": CORE.sha256_path(preregistration_path),
        "source_cad_sha256": CORE.sha256_path(source_cad_path),
        "accepted_mesh_sha256": CORE.sha256_path(mesh_path),
        "mesh_report_sha256": CORE.sha256_path(mesh_report_path),
        "implementation_sha256": CORE.sha256_path(Path(__file__)),
        "core_implementation_sha256": CORE.sha256_path(Path(CORE.__file__)),
        "identity_wrapper_sha256": (
            CORE.sha256_path(IDENTITY_WRAPPER_PATH)
            if IDENTITY_WRAPPER_PATH is not None
            else None
        ),
        "physical_scan_load_performed": False,
        "physical_scan_open_call_count": 0,
        "evaluator_manifest_accessed": False,
        "evaluator_roles_opened": False,
        "labels_gold_expected_counts_or_quality_used": False,
        "cad_metrics": {
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(triangles)),
            "total_mesh_area_mm2": total_mesh_area,
            "welded_vertex_count": topology["welded_vertex_count"],
            "edge_count": topology["edge_count"],
            "boundary_edge_count_recorded_not_closed_mesh_gate": topology["boundary_edge_count"],
            "nonmanifold_edge_count": topology["nonmanifold_edge_count"],
            "mesh_component_count": mesh_component_count,
            "smooth_region_count": int(len(np.unique(topology["smooth_region_labels"]))),
            "candidate_count": int(len(candidates["points"])),
            "interior_candidate_count": int(np.count_nonzero(interior_mask)),
            "diagnostic_probe_all_pass_candidate_count": int(
                np.count_nonzero(diagnostic_probe_mask)
            ),
            "diagnostic_probe_disagrees_with_authoritative_interior_count": int(
                np.count_nonzero(diagnostic_probe_mask != interior_mask)
            ),
            "node_count": int(len(node_points)),
            "candidate_total_area_mm2": candidate_total_area,
            "interior_area_mm2": interior_area,
            "interior_area_fraction": interior_area / total_mesh_area,
            "assigned_area_mm2": assigned_area,
            "assigned_fraction_of_interior_area": assigned_area / interior_area,
            "unassigned_area_mm2": unassigned_area,
            "unassigned_fraction_of_interior_area": unassigned_area / interior_area,
            "excluded_boundary_area_mm2": excluded_boundary_area,
            "candidate_area_accounting_residual_mm2": candidate_total_area - interior_area - excluded_boundary_area,
            "interior_area_accounting_residual_mm2": interior_area - assigned_area - unassigned_area,
            "minimum_candidate_boundary_clearance_mm": float(np.min(boundary_clearance_mm)),
            "minimum_retained_node_boundary_clearance_mm": float(
                np.min(boundary_clearance_mm[node_candidate_indices])
            ),
            "graph_edge_count": int(len(edges)),
            "graph_edge_geodesic_min_mm": float(np.min(edge_geodesic_mm)) if len(edge_geodesic_mm) else None,
            "graph_edge_geodesic_max_mm": float(np.max(edge_geodesic_mm)) if len(edge_geodesic_mm) else None,
            "graph_component_count": int(len(component_areas)),
            "largest_graph_component_area_mm2": largest_component_area,
            "largest_graph_component_area_fraction": largest_component_area / assigned_area,
            "degree_two_node_fraction_diagnostic": float(np.mean(degrees >= 2)),
            "degree_two_assigned_area_fraction": math.fsum(
                float(value) for value in node_area_weights[degrees >= 2]
            ) / assigned_area,
            "fold_area_fractions": fold_area_fractions.tolist(),
            "fold_basis": fold_basis,
            "component_two_sweep_span_records": component_span_records,
            "rigid_design_rank": design_rank,
            "rigid_design_condition_number": design_condition,
            "rigid_design_singular_values": singular.tolist(),
            "fold_design": fold_design,
            "cad_interior_area_weighted_centroid_mm": area_centroid.tolist(),
        },
        "gates": gates,
        "artifacts": {
            "nodes": {"path": node_path.name, "sha256": CORE.sha256_path(node_path)},
            "edges": {"path": edge_path.name, "sha256": CORE.sha256_path(edge_path)},
            "edge_geodesic_mm": {"path": edge_weight_path.name, "sha256": CORE.sha256_path(edge_weight_path)},
        },
        "cad_actionable": bool(passed),
        "source_execution_authorized": False,
        "source_execution_additional_requirement": "INDEPENDENT_AUDIT_OF_EXACT_PREREG_PREFLIGHT_IMPLEMENTATION_AND_ARTIFACT_HASHES",
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
    }
    CORE.atomic_json(output_root / "CAD_ONLY_PREFLIGHT.json", payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--source-cad", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--mesh-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = execute_preflight(
        preregistration_path=args.preregistration,
        source_cad_path=args.source_cad,
        mesh_path=args.mesh,
        mesh_report_path=args.mesh_report,
        output_root=args.output_root,
    )
    print(json.dumps({"status": payload["status"], "cad_metrics": payload["cad_metrics"]}, indent=2))
    return 0 if payload["status"] == "PASS_CAD_ONLY_ACTIONABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
