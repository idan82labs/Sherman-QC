#!/usr/bin/env python3
"""Source-free EXP_CAD_0032 V5 invocation surface.

There is intentionally no ``main`` function and no point-cloud/file loader.
Callers provide already-loaded numeric arrays and source-free CAD fixtures.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scripts import exp_cad_0031_source_interface as FROZEN_INTAKE
from scripts import run_exp_cad_0031_cfpdg_source as FROZEN_V4
from scripts.exp_cad_0032_cfpdg_v5_engine import (
    ARCHITECTURE,
    EXPERIMENT_ID,
    V5Engine,
    execute_parent_science_v5,
    filter_source_rows_v5,
)


EXPECTED_PREREG_SCHEMA = (
    "EXP_CAD_0032_CFPDG_V5_EXECUTION_EQUIVALENT_PREREGISTRATION_V1"
)
EXPECTED_CAD_ARTIFACT_HASHES = {
    "cad_only_preflight_sha256": "e96daf9288f59a8d7f156981f6d13e9fd3204b344523d96bba587bbbd84c8f82",
    "cad_graph_nodes_sha256": "575c19d404efd6733051d611dd46f123fbbee3117be13ef001514d658e2618b0",
    "cad_graph_edges_sha256": "0bf773fc65af9152bb8491830f98faf2e14c10fe999e8e4bf81a3db687513e3d",
    "cad_graph_edge_geodesic_mm_sha256": "92605624e786450ca260134555b6e71bbac56875e9e3e49b4aceac6d5e9de185",
    "accepted_mesh_sha256": "9e6571704780383e99b6fa4f2800fa69a50a3e187467698fe5dd7414d9d2a716",
}


def _validate_v5_preregistration(v5_prereg: Mapping[str, Any]) -> None:
    if (
        v5_prereg.get("schema") != EXPECTED_PREREG_SCHEMA
        or v5_prereg.get("experiment_id") != EXPERIMENT_ID
        or v5_prereg.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("unexpected EXP32 V5 preregistration identity")
    science = v5_prereg.get("immutable_science")
    if not isinstance(science, Mapping):
        raise ValueError("EXP32 immutable science contract missing")
    if science.get("cad_node_count") != 8077:
        raise ValueError("EXP32 CAD node count must remain 8077")
    if science.get("cad_edge_count") != 19267:
        raise ValueError("EXP32 CAD edge count must remain 19267")
    if science.get("unique_member_fit_count") != 38:
        raise ValueError("EXP32 member-fit inventory must remain 38")
    if science.get("robustness_row_ids") != "R00-R31 exactly":
        raise ValueError("EXP32 robustness inventory must remain R00-R31")
    if v5_prereg.get("immutable_cad_artifacts") != EXPECTED_CAD_ARTIFACT_HASHES:
        raise ValueError("EXP32 immutable CAD artifact hash contract mismatch")
    forbidden = v5_prereg.get("forbidden_inputs_and_actions")
    if not isinstance(forbidden, Mapping) or forbidden.get("physical_cli") is not True:
        raise ValueError("EXP32 source-free execution lock missing")


def _nonactionable_metrics() -> dict[str, Any]:
    proof_engine = V5Engine()
    counters = {
        "unique_member_fit_count": 0,
        "batched_kd_fit_count": 0,
        "kd_node_query_count": 0,
        "supplemental_diagnostic_batched_kd_fit_count": 0,
        "supplemental_diagnostic_kd_node_query_count": 0,
        "local_fit_call_count": 0,
        "compact_local_fit_call_count": 0,
        "diagnostic_local_fit_call_count": 0,
        "nuisance_design_build_count": 0,
        "nuisance_design_cache_hit_count": 0,
        "nuisance_design_hit_count": 0,
        "induced_components_call_count": 0,
        "induced_components_cache_hit_count": 0,
        "component_streamed_dijkstra_row_count": 0,
        "component_dijkstra_batch_call_count": 0,
        "induced_components_streamed_dijkstra_count": 0,
        "robustness_row_count": 0,
    }
    return {
        "schema": "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1",
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        **counters,
        "operation_counters": dict(counters),
        "scientific_execute_source_reached": False,
        "full_fit_inventory_gate_passed": False,
        "full_fit_batched_kd_gate_passed": False,
        "full_fit_kd_query_gate_passed": False,
        "equivalence_checks": {
            "raycast_tie_differential_passed": proof_engine.raycast_tie_differential_passed,
            "raycast_basis": "actual ordinary-and-tied source-free fixture through the frozen and V5-bound paths",
            "kd_tie_differential_passed": proof_engine.kd_tie_differential_passed,
            "semantic_payload_logical_equivalence": None,
            "visual_payload_logical_equivalence": None,
        },
        "scientific_threshold_or_gate_changed": False,
        "fit_or_robustness_row_skipped": False,
    }


def execute_source_v5(
    *,
    v5_prereg: Mapping[str, Any],
    parent_prereg: Mapping[str, Any],
    cad_preflight: Mapping[str, Any],
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    mesh: Any,
    raw_first: np.ndarray,
    normals_first: np.ndarray,
    raw_second: np.ndarray,
    normals_second: np.ndarray,
    transform_first: np.ndarray,
    transform_second: np.ndarray,
    first_token: str,
    second_token: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Filter source rows, then execute the exact frozen parent payload."""

    _validate_v5_preregistration(v5_prereg)
    first = filter_source_rows_v5(
        raw_first, normals_first, member_token=first_token
    )
    second = filter_source_rows_v5(
        raw_second, normals_second, member_token=second_token
    )
    pair = FROZEN_INTAKE.pair_actionability(first, second)
    filtered = {
        "first": first,
        "second": second,
        "pair": pair,
    }
    if not pair["actionable"]:
        semantic = FROZEN_V4._interface_semantic(
            first=first, second=second, pair=pair
        )
        return semantic, None, filtered, _nonactionable_metrics()

    semantic, visual, metrics = execute_parent_science_v5(
        prereg=parent_prereg,
        cad_preflight=cad_preflight,
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        mesh=mesh,
        raw_first=first["points"],
        normals_first=first["normals"],
        raw_second=second["points"],
        normals_second=second["normals"],
        transform_first=transform_first,
        transform_second=transform_second,
        first_token=first_token,
        second_token=second_token,
    )
    metrics["scientific_execute_source_reached"] = True
    return dict(semantic), dict(visual), filtered, metrics


def source_free_contract_summary() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "physical_cli_present": False,
        "physical_source_execution_authorized": False,
        "unique_member_fit_count": 38,
        "robustness_rows": [f"R{index:02d}" for index in range(32)],
        "scientific_payload_relabelled": False,
    }
