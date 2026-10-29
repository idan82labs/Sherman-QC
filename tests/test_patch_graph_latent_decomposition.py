import json
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai_analyzer import (
    Model3DSnapshotRenderer,
    SNAPSHOT_RENDER_TRIANGLE_BUDGET,
    _bounded_snapshot_reference_mesh,
)

from patch_graph_latent_decomposition import (
    BendHypothesis,
    DecompositionResult,
    FlangeHypothesis,
    OwnedBendRegion,
    SurfaceAtom,
    SurfaceAtomGraph,
    TypedLabelAssignment,
    _activate_interface_bends,
    _allow_no_contact_birth_rescue_for_range,
    _bend_component_admissibility,
    _build_local_replacement_diagnostics,
    _dedupe_bend_hypotheses,
    _bend_component_count,
    _bend_hypothesis_corridor_support,
    _build_structural_target_assignment,
    _build_scan_corridor_completion_hypotheses,
    _build_supported_strip_assignment,
    _candidate_labels_for_atom,
    _connected_components,
    _duplicate_conflict_penalty,
    _drop_selected_pair_duplicate_bends,
    _final_duplicate_cluster_penalty,
    _fragmentation_penalty,
    _incidence_penalty,
    _junction_birth_support_arrangement_diagnostics,
    _missing_arm_birth_support_proposal,
    _build_two_sided_contact_pair_diagnostics,
    _two_sided_contact_corridor_support_proposal,
    _junction_local_map_promotion_readiness,
    _normalize_bend_hypotheses,
    _owned_region_marker_quality,
    _prune_and_refit_labels,
    _rescue_interface_birth_supports,
    _render_region_suppression_report,
    _load_cached_research_inputs,
    _research_cache_identity,
    _write_cached_research_inputs,
    _suppress_same_pair_marker_aliases,
    _suppress_tiny_cross_pair_marker_clusters,
    _supported_strip_auto_target,
    _supported_strip_auto_excludes_boundary_truncated,
    _supported_strip_detector_auto_completion,
    _supported_strip_complete_incidence_auto_selection,
    _typed_adjacency_penalty,
    build_missed_conventional_bend_line_candidates,
    build_owned_region_marker_admissibility,
    build_surface_atom_graph,
    extract_owned_bend_regions,
    generate_interface_birth_hypotheses,
    generate_bend_hypotheses,
    generate_flange_hypotheses,
    render_decomposition_artifacts,
    solve_typed_latent_decomposition,
)


def _toy_points():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [2.0, 0.0, 0.0],
            [2.2, 0.1, 0.0],
            [2.0, 0.2, 0.0],
            [1.0, 0.0, 0.7],
            [1.2, 0.1, 0.8],
            [0.8, 0.2, 0.75],
        ],
        dtype=np.float64,
    )
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.7, 0.7],
            [0.0, 0.7, 0.7],
            [0.0, 0.7, 0.7],
        ],
        dtype=np.float64,
    )
    return points, normals


def _bend(
    bend_id: str,
    bend_class: str = "transition_only",
    incident_flange_ids=("F1", "F2"),
    atom_ids=("A1",),
    anchor=(0.0, 0.0, 0.0),
    axis_direction=(1.0, 0.0, 0.0),
    angle_deg=90.0,
    radius_mm=None,
    visibility_state="internal",
    confidence=0.8,
    cross_width_mm=1.0,
    span_mm=2.0,
    canonical_flange_pair=None,
    canonical_bend_key=None,
    candidate_classes=None,
    source_bend_ids=None,
    seed_incident_flange_ids=None,
    candidate_incident_flange_pairs=None,
):
    canonical_pair = tuple(sorted(canonical_flange_pair if canonical_flange_pair is not None else incident_flange_ids))
    return BendHypothesis(
        bend_id=bend_id,
        bend_class=bend_class,
        source_kind="seed",
        incident_flange_ids=tuple(incident_flange_ids),
        canonical_flange_pair=canonical_pair,
        canonical_bend_key=canonical_bend_key or f"{','.join(canonical_pair)}::{bend_id}",
        atom_ids=tuple(atom_ids),
        anchor=anchor,
        axis_direction=axis_direction,
        angle_deg=angle_deg,
        radius_mm=radius_mm,
        visibility_state=visibility_state,
        confidence=confidence,
        cross_width_mm=cross_width_mm,
        span_mm=span_mm,
        candidate_classes=tuple(candidate_classes or (bend_class,)),
        source_bend_ids=tuple(source_bend_ids or (bend_id,)),
        seed_incident_flange_ids=tuple(seed_incident_flange_ids or incident_flange_ids),
        candidate_incident_flange_pairs=tuple(
            tuple(sorted(pair)) for pair in (candidate_incident_flange_pairs or (canonical_pair,))
        ),
    )


def test_build_surface_atom_graph_constructs_connected_atoms():
    points, normals = _toy_points()
    graph = build_surface_atom_graph(points, normals, voxel_size_mm=1.0, local_spacing_mm=0.2)

    assert len(graph.atoms) >= 3
    assert all(graph.adjacency[atom.atom_id] for atom in graph.atoms[:2])
    assert graph.voxel_size_mm == 1.0


def test_generate_flange_hypotheses_splits_connected_planar_support():
    graph = SimpleNamespace(
        atoms=(
            SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A2", (0, 1, 0), (1,), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A3", (1, 0, 0), (2,), (1.0, 0.0, 0.1), (1.0, 0.0, 0.0), 0.2, 0.01, (0.0, 1.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A4", (1, 1, 0), (3,), (1.0, 1.0, 0.1), (1.0, 0.0, 0.0), 0.2, 0.01, (0.0, 1.0, 0.0), 1.0, 1.0),
        ),
        adjacency={"A1": ("A2",), "A2": ("A1",), "A3": ("A4",), "A4": ("A3",)},
        local_spacing_mm=0.2,
    )
    bend = SimpleNamespace(
        bend_id="B1",
        flange1=SimpleNamespace(normal=np.asarray([0.0, 0.0, 1.0]), d=0.0),
        flange2=SimpleNamespace(normal=np.asarray([1.0, 0.0, 0.0]), d=-1.0),
    )
    flanges = generate_flange_hypotheses(
        atom_graph=graph,
        candidate_bundles={"original": {"single_scale": [bend], "multiscale": []}},
    )

    assert len(flanges) >= 2
    assert any(abs(np.dot(np.asarray(flange.normal), np.asarray([0.0, 0.0, 1.0]))) > 0.95 for flange in flanges)
    assert any(abs(np.dot(np.asarray(flange.normal), np.asarray([1.0, 0.0, 0.0]))) > 0.95 for flange in flanges)


def test_generate_flange_hypotheses_uses_zero_shot_strip_normals_when_detector_planes_are_degenerate():
    graph = SimpleNamespace(
        atoms=(
            SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A3", (0, 1, 0), (2,), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), 0.5, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A4", (1, 1, 0), (3,), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), 0.5, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0),
        ),
        adjacency={"A1": ("A2", "A3"), "A2": ("A1", "A4"), "A3": ("A1", "A4"), "A4": ("A2", "A3")},
        local_spacing_mm=0.5,
    )
    detector_bend = SimpleNamespace(
        bend_id="B1",
        flange1=SimpleNamespace(normal=np.asarray([0.0, 0.0, 1.0]), d=0.0),
        flange2=SimpleNamespace(normal=np.asarray([0.0, 0.0, 1.0]), d=0.0),
    )

    flanges = generate_flange_hypotheses(
        atom_graph=graph,
        candidate_bundles={"original": {"single_scale": [detector_bend], "multiscale": []}},
        current_zero_shot_result={
            "bend_support_strips": [
                {
                    "axis_direction": [1.0, 0.0, 0.0],
                    "axis_line_hint": {"midpoint": [0.5, 0.5, 0.0]},
                    "cross_fold_width_mm": 1.0,
                    "along_axis_span_mm": 2.0,
                    "adjacent_flange_normals": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                }
            ]
        },
    )

    assert any(flange.source_kind == "zero_shot_strip_normal_seed" for flange in flanges)
    assert any(abs(np.dot(np.asarray(flange.normal), np.asarray([0.0, 1.0, 0.0]))) > 0.95 for flange in flanges)


def test_generate_bend_hypotheses_attaches_incident_flanges():
    points, normals = _toy_points()
    graph = build_surface_atom_graph(points, normals, voxel_size_mm=1.0, local_spacing_mm=0.2)
    flanges = [
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.9),
        FlangeHypothesis("F2", "seed", ("A2",), (0.0, 1.0, 0.0), 0.0, (5.0, 0.0, 0.0), 0.9),
    ]
    strip_payload = {
        "bend_support_strips": [
            {
                "strip_id": "S1",
                "axis_direction": [1.0, 0.0, 0.0],
                "axis_line_hint": {"midpoint": [2.5, 0.1, 0.75]},
                "adjacent_flange_normals": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                "dominant_angle_deg": 90.0,
                "radius_mm": 1.0,
                "cross_fold_width_mm": 1.0,
                "along_axis_span_mm": 4.0,
                "confidence": 0.8,
            }
        ]
    }
    bends = generate_bend_hypotheses(
        atom_graph=graph,
        flange_hypotheses=flanges,
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
        current_zero_shot_result=strip_payload,
        bbox_points=points,
    )

    assert bends
    assert tuple(bends[0].incident_flange_ids) == ("F1", "F2")


def test_typed_adjacency_penalty_rewards_valid_bend_flange_contacts():
    bend = _bend("B1", atom_ids=("A3",))
    bend_lookup = {"B1": bend}
    label_types = {"F1": "flange", "F3": "flange", "B1": "bend"}

    assert _typed_adjacency_penalty("B1", "F1", bend_lookup, label_types) == 0.0
    assert _typed_adjacency_penalty("B1", "F3", bend_lookup, label_types) > 0.0


def test_extract_owned_bend_regions_collapses_connected_bend_support():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0)
    atom2 = SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0)
    atom3 = SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 1.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1, atom2, atom3),
        adjacency={"A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2",)},
    )
    bend = _bend("B1", atom_ids=("A2",), anchor=(1.0, 0.0, 0.0))
    assignment = TypedLabelAssignment(
        atom_labels={"A1": "F1", "A2": "B1", "A3": "F2"},
        label_types={"F1": "flange", "F2": "flange", "B1": "bend", "residual": "residual", "outlier": "outlier"},
        energy=1.0,
    )

    regions = extract_owned_bend_regions(
        atom_graph=graph,
        assignment=assignment,
        flange_hypotheses=[],
        bend_hypotheses=[bend],
        solution_payload={
            "solution_energies": [1.0, 1.5, 1.7],
            "object_level_confidence_summary": {bend.canonical_bend_key: 0.82},
        },
    )

    assert len(regions) == 1
    assert regions[0].incident_flange_ids == ("F1", "F2")
    assert regions[0].owned_atom_ids == ("A2",)
    assert regions[0].admissible is True
    assert regions[0].post_bend_class == "transition_only"


def test_extract_owned_bend_regions_exports_hypothesis_corridor_not_grown_face():
    atoms = tuple(
        SurfaceAtom(
            f"A{index + 1}",
            (index, 0, 0),
            (index,),
            (float(index), float(6 if index >= 3 else 0), 0.0),
            (0.0, 0.7, 0.7),
            1.0,
            0.02,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for index in range(6)
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={
            "A1": ("A2",),
            "A2": ("A1", "A3"),
            "A3": ("A2", "A4"),
            "A4": ("A3", "A5"),
            "A5": ("A4", "A6"),
            "A6": ("A5",),
        },
        local_spacing_mm=1.0,
    )
    bend = _bend(
        "B1",
        atom_ids=("A1", "A2", "A3"),
        anchor=(1.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        cross_width_mm=1.0,
        span_mm=4.0,
    )

    support = _bend_hypothesis_corridor_support(
        component=tuple(atom.atom_id for atom in atoms),
        bend=bend,
        atom_graph=graph,
        atom_lookup={atom.atom_id: atom for atom in atoms},
    )

    assert support == ["A1", "A2", "A3"]


def test_unsupported_surface_atom_is_not_offered_nearest_bend_label():
    atom = SurfaceAtom(
        "A9",
        (9, 0, 0),
        (9,),
        (90.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
        0.0,
        (1.0, 0.0, 0.0),
        1.0,
        1.0,
    )
    bend = _bend("B1", atom_ids=("A1", "A2"), anchor=(0.0, 0.0, 0.0))

    labels = _candidate_labels_for_atom(
        atom=atom,
        current_label=None,
        flange_hypotheses=(),
        bend_hypotheses=(bend,),
    )

    assert "B1" not in labels
    assert labels == ("residual", "outlier")


def test_structural_target_assignment_activates_disjoint_line_corridors_only():
    atoms = tuple(
        SurfaceAtom(
            f"A{index + 1}",
            (index, 0, 0),
            (index,),
            (float(index % 3), float((index // 3) * 10), 0.0),
            (0.0, 0.7, 0.7),
            1.0,
            0.03,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for index in range(6)
    )
    graph = SurfaceAtomGraph(
        atoms=atoms,
        adjacency={
            "A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2",),
            "A4": ("A5",), "A5": ("A4", "A6"), "A6": ("A5",),
        },
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    bends = (
        _bend("B1", atom_ids=("A1", "A2", "A3"), anchor=(1.0, 0.0, 0.0), span_mm=3.0),
        _bend("B2", atom_ids=("A4", "A5", "A6"), anchor=(1.0, 10.0, 0.0), span_mm=3.0),
    )

    labels, diagnostics = _build_structural_target_assignment(
        atom_graph=graph,
        flange_hypotheses=(),
        bend_hypotheses=bends,
        target_count=2,
    )

    assert {label for label in labels.values() if label.startswith("B")} == {"B1", "B2"}
    assert diagnostics["structural_selected_count"] == 2
    assert all(label in {"B1", "B2"} for label in labels.values())


def test_supported_strip_assignment_partitions_overlapping_atoms_by_nearest_line():
    atoms = tuple(
        SurfaceAtom(
            f"A{index + 1}",
            (index, 0, 0),
            (index,),
            (float((index % 3) * 10), float((index // 3) * 10), 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            0.03,
            (1.0, 0.0, 0.0),
            10.0,
            10.0,
        )
        for index in range(6)
    )
    graph = SurfaceAtomGraph(
        atoms=atoms,
        adjacency={
            "A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2",),
            "A4": ("A5",), "A5": ("A4", "A6"), "A6": ("A5",),
        },
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    shared = tuple(atom.atom_id for atom in atoms)
    bends = (
        replace(
            _bend("B1", atom_ids=shared, anchor=(10.0, 0.0, 0.0), span_mm=20.0, cross_width_mm=5.0),
            source_kind="zero_shot_strip_seed",
        ),
        replace(
            _bend("B2", atom_ids=shared, anchor=(10.0, 10.0, 0.0), span_mm=20.0, cross_width_mm=5.0),
            source_kind="zero_shot_strip_seed",
        ),
    )

    labels, diagnostics = _build_supported_strip_assignment(
        atom_graph=graph,
        flange_hypotheses=(),
        bend_hypotheses=bends,
    )

    assert {atom_id for atom_id, label in labels.items() if label == "B1"} == {"A1", "A2", "A3"}
    assert {atom_id for atom_id, label in labels.items() if label == "B2"} == {"A4", "A5", "A6"}
    assert diagnostics["supported_strip_selected_count"] == 2


def test_supported_strip_auto_target_requires_partial_policy_lower_bound_agreement():
    graph = SurfaceAtomGraph(atoms=(), adjacency={}, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=1.0)
    bends = tuple(
        replace(
            _bend(f"B{index}", atom_ids=("A1", "A2", "A3"), span_mm=20.0),
            source_kind="zero_shot_strip_seed",
        )
        for index in range(1, 4)
    )
    options = {
        "scan_family_route": {
            "feature_snapshot": {
                "scan_state_hint": "partial",
                "zero_shot_count_range": [3, 9],
            }
        }
    }

    assert _supported_strip_auto_target(
        atom_graph=graph,
        bend_hypotheses=bends,
        solver_options=options,
    ) == 3
    options["scan_family_route"]["feature_snapshot"]["zero_shot_count_range"] = [4, 9]
    assert _supported_strip_auto_target(
        atom_graph=graph,
        bend_hypotheses=bends,
        solver_options=options,
    ) is None


def test_supported_strip_auto_target_accepts_scan_exact_singleton_without_truth_input():
    graph = SurfaceAtomGraph(atoms=(), adjacency={}, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=1.0)
    bend = replace(
        _bend("B1", atom_ids=("A1", "A2", "A3"), span_mm=20.0),
        source_kind="zero_shot_strip_seed",
    )
    options = {"scan_family_route": {"feature_snapshot": {"scan_state_hint": "partial", "zero_shot_count_range": [1, 1]}}}
    assert _supported_strip_auto_target(
        atom_graph=graph, bend_hypotheses=(bend,), solver_options=options,
    ) == 1


def test_supported_strip_auto_target_excludes_only_boundary_truncated_surplus():
    graph = SurfaceAtomGraph(atoms=(), adjacency={}, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=1.0)
    bends = tuple(
        replace(
            _bend(f"B{index}", atom_ids=("A1", "A2", "A3"), span_mm=20.0),
            source_kind="zero_shot_strip_seed",
            visibility_state="boundary_truncated" if index > 3 else "internal",
        )
        for index in range(1, 6)
    )
    options = {
        "scan_family_route": {
            "feature_snapshot": {
                "scan_state_hint": "partial",
                "zero_shot_count_range": [3, 9],
            }
        }
    }

    assert _supported_strip_auto_target(
        atom_graph=graph,
        bend_hypotheses=bends,
        solver_options=options,
    ) == 3
    assert _supported_strip_auto_excludes_boundary_truncated(
        atom_graph=graph,
        bend_hypotheses=bends,
        solver_options=options,
    ) is True
    labels, diagnostics = _build_supported_strip_assignment(
        atom_graph=graph,
        flange_hypotheses=(),
        bend_hypotheses=bends,
        exclude_boundary_truncated=True,
    )
    assert diagnostics["supported_strip_candidate_count"] == 3
    assert diagnostics["supported_strip_exclude_boundary_truncated"] is True


def test_strip_detector_completion_requires_partial_range_and_fragment_promotion():
    graph = SurfaceAtomGraph(atoms=(), adjacency={}, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=2.0)
    strips = [
        replace(_bend(f"S{index}", atom_ids=("A1", "A2", "A3"), span_mm=80.0), source_kind="zero_shot_strip_seed", anchor=(0.0, index * 40.0, 0.0))
        for index in range(1, 4)
    ]
    fragment = replace(_bend("SF", atom_ids=("A1", "A2", "A3"), span_mm=20.0), source_kind="zero_shot_strip_seed", anchor=(0.0, 160.0, 0.0))
    detector = replace(_bend("D1", atom_ids=("A1", "A2", "A3", "A4", "A5"), span_mm=80.0), source_kind="detector_bend_seed", anchor=(0.0, 160.0, 0.0), confidence=0.9)
    options = {"scan_family_route": {"feature_snapshot": {"scan_state_hint": "partial", "zero_shot_count_range": [4, 8]}}}

    completion = _supported_strip_detector_auto_completion(
        atom_graph=graph,
        bend_hypotheses=(*strips, fragment, detector),
        solver_options=options,
    )
    assert completion is not None
    assert completion.selected_bend_ids == ("S1", "S2", "S3", "D1")

    options["scan_family_route"]["feature_snapshot"]["scan_state_hint"] = "full"
    assert _supported_strip_detector_auto_completion(
        atom_graph=graph,
        bend_hypotheses=(*strips, fragment, detector),
        solver_options=options,
    ) is None


def test_small_tight_complete_incidence_selection_matches_scan_policy_lower_bound():
    graph = SurfaceAtomGraph(atoms=(), adjacency={}, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=1.0)
    complete = tuple(
        replace(
            _bend(f"B{index}", atom_ids=("A1", "A2", "A3"), span_mm=20.0),
            source_kind="zero_shot_strip_seed",
        )
        for index in (1, 2)
    )
    weak = replace(
        _bend(
            "B3", incident_flange_ids=("F1",), canonical_flange_pair=("F1",),
            atom_ids=("A1", "A2", "A3"), span_mm=20.0,
        ),
        source_kind="zero_shot_strip_seed",
    )
    options = {
        "scan_family_route": {
            "route_id": "small_tight_bend_recovery",
            "feature_snapshot": {
                "scan_state_hint": "full", "small_tight_bend_signal": 0.33,
                "zero_shot_count_range": [2, 9],
            },
        }
    }
    assert _supported_strip_complete_incidence_auto_selection(
        atom_graph=graph, bend_hypotheses=(*complete, weak), solver_options=options,
    ) == ("B1", "B2")


def test_scan_corridor_completion_materializes_only_strict_route_candidate():
    graph = SurfaceAtomGraph(atoms=(), adjacency={}, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=1.0)
    flanges = (
        FlangeHypothesis("F1", "test", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 1.0),
        FlangeHypothesis("F2", "test", (), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 1.0),
    )
    strips = tuple(
        replace(
            _bend(
                f"S{index}", atom_ids=(f"S{index}A1", f"S{index}A2", f"S{index}A3"),
                anchor=(0.0, index * 40.0, 0.0), span_mm=30.0, cross_width_mm=4.0,
            ),
            source_kind="zero_shot_strip_seed",
        )
        for index in range(3)
    )
    births = tuple(
        replace(
            _bend(
                f"I{index}", atom_ids=(f"I{index}A1", f"I{index}A2", f"I{index}A3"),
                anchor=(-30.0 + index * 20.0, 120.0, 0.0), span_mm=15.0, cross_width_mm=4.0,
            ),
            source_kind="interface_birth",
        )
        for index in range(4)
    )
    route = {
        "route_id": "fragmented_repeat90_interface_birth",
        "feature_snapshot": {
            "scan_state_hint": "",
            "repeated_near90_strength": 0.82,
            "candidate_fragmentation": 1.6,
        },
    }

    materialized, diagnostics = _build_scan_corridor_completion_hypotheses(
        atom_graph=graph,
        flange_hypotheses=flanges,
        bend_hypotheses=strips,
        interface_birth_hypotheses=births,
        route_payload=route,
    )
    assert diagnostics["applied"] is True
    assert len(materialized) == 4
    assert all(item.source_kind == "scan_supported_corridor_completion_seed" for item in materialized)

    route["feature_snapshot"]["scan_state_hint"] = "partial"
    blocked, blocked_diagnostics = _build_scan_corridor_completion_hypotheses(
        atom_graph=graph,
        flange_hypotheses=flanges,
        bend_hypotheses=strips,
        interface_birth_hypotheses=births,
        route_payload=route,
    )
    assert blocked == []
    assert blocked_diagnostics["applied"] is False


def test_small_tight_supported_marker_accepts_short_surface_line_not_curve_family():
    atoms = tuple(
        SurfaceAtom(
            atom_id=f"A{index}", voxel_key=(index, 0, 0), point_indices=(index,),
            centroid=(float(index), 0.0, 0.0), normal=(0.0, 0.0, 1.0),
            local_spacing_mm=1.0, curvature_proxy=0.1, axis_direction=(1.0, 0.0, 0.0),
            weight=1.0, area_proxy_mm2=1.0,
        )
        for index in range(8)
    )
    graph = SurfaceAtomGraph(
        atoms=atoms,
        adjacency={
            atom.atom_id: tuple(
                f"A{neighbor}" for neighbor in (index - 1, index + 1) if 0 <= neighbor < len(atoms)
            )
            for index, atom in enumerate(atoms)
        },
        edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=1.0,
    )
    region = OwnedBendRegion(
        bend_id="OB1", bend_class="circular_strip", incident_flange_ids=("F1", "F2"),
        canonical_bend_key="small-tight", owned_atom_ids=tuple(atom.atom_id for atom in atoms),
        anchor=(3.5, 0.0, 0.0), axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(3.5, 0.0, 0.0), span_endpoints=((0.0, 0.0, 0.0), (7.0, 0.0, 0.0)),
        angle_deg=90.0, radius_mm=2.0, visibility_state="internal", support_mass=8.0,
        admissible=True, admissibility_reasons=(), post_bend_class="scan_supported_small_tight",
        debug_confidence=1.0,
    )

    assert _owned_region_marker_quality(region, atom_graph=graph)["accepted"] is True
    ordinary = replace(region, post_bend_class="circular_strip")
    assert "curve_family_not_conventional" in _owned_region_marker_quality(ordinary, atom_graph=graph)["reason_codes"]


def test_suppress_tiny_cross_pair_marker_clusters_drops_smallest_multi_issue_fragment():
    def region(bend_id, pair, anchor, atom_count, support_mass):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=pair,
            canonical_bend_key=f"{','.join(pair)}::{bend_id}",
            owned_atom_ids=tuple(f"{bend_id}_A{i}" for i in range(atom_count)),
            anchor=anchor,
            axis_direction=(1.0, 0.0, 0.0),
            support_centroid=anchor,
            span_endpoints=(anchor, anchor),
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=support_mass,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class="transition_only",
            debug_confidence=1.0,
        )

    large_a = region("OB3", ("F27", "F4"), (0.0, 0.0, 0.0), 18, 4007.0)
    large_b = region("OB4", ("F1", "F14"), (31.0, 0.0, 0.0), 16, 6021.0)
    tiny_multi_issue = region("OB9", ("F27", "F32"), (16.0, 0.0, 0.0), 3, 885.0)
    far_tiny = region("OB10", ("F8", "F9"), (100.0, 0.0, 0.0), 3, 100.0)

    filtered = _suppress_tiny_cross_pair_marker_clusters(
        [large_a, large_b, tiny_multi_issue, far_tiny],
        local_spacing_mm=1.5,
    )

    assert [item.bend_id for item in filtered] == ["OB3", "OB4", "OB10"]


def test_suppress_same_pair_marker_aliases_drops_close_tiny_alias_only():
    def region(bend_id, pair, anchor, atom_count, support_mass, axis=(1.0, 0.0, 0.0)):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=pair,
            canonical_bend_key=f"{','.join(pair)}::{bend_id}",
            owned_atom_ids=tuple(f"{bend_id}_A{i}" for i in range(atom_count)),
            anchor=anchor,
            axis_direction=axis,
            support_centroid=anchor,
            span_endpoints=(anchor, anchor),
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=support_mass,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class="transition_only",
            debug_confidence=0.8,
        )

    real_region = region("OB4", ("F1", "F2"), (0.0, 0.0, 0.0), 24, 5000.0)
    tiny_alias = region("OB5", ("F2", "F1"), (18.0, 0.0, 0.0), 5, 400.0)
    far_same_pair = region("OB6", ("F1", "F2"), (90.0, 0.0, 0.0), 4, 300.0)
    real_neighbor = region("OB7", ("F1", "F2"), (25.0, 0.0, 0.0), 20, 4500.0)

    filtered = _suppress_same_pair_marker_aliases(
        [real_region, tiny_alias, far_same_pair, real_neighbor],
        local_spacing_mm=2.0,
    )

    assert [item.bend_id for item in filtered] == ["OB4", "OB6", "OB7"]


def test_suppress_same_pair_marker_aliases_drops_collinear_contained_alias() -> None:
    def region(bend_id, anchor, atom_count, support_mass, span, post_class="transition_only"):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=("F1", "F8"),
            canonical_bend_key=f"F1,F8::{bend_id}",
            owned_atom_ids=tuple(f"{bend_id}_A{i}" for i in range(atom_count)),
            anchor=anchor,
            axis_direction=(1.0, 0.0, 0.0),
            support_centroid=anchor,
            span_endpoints=span,
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=support_mass,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class=post_class,
            debug_confidence=1.0,
        )

    dominant = region(
        "OB4", (0.0, 0.0, 0.0), 19, 18000.0,
        ((-100.0, 0.0, 0.0), (100.0, 0.0, 0.0)),
        post_class="scan_supported_strip",
    )
    contained = region("OB8", (65.0, 1.0, 0.0), 8, 2600.0, ((40.0, 1.0, 0.0), (85.0, 1.0, 0.0)))
    separate = region("OB9", (180.0, 0.0, 0.0), 6, 2000.0, ((160.0, 0.0, 0.0), (200.0, 0.0, 0.0)))
    parallel_offset = region("OB10", (65.0, 12.0, 0.0), 7, 2100.0, ((40.0, 12.0, 0.0), (85.0, 12.0, 0.0)))

    filtered = _suppress_same_pair_marker_aliases(
        [dominant, contained, separate, parallel_offset], local_spacing_mm=2.0
    )

    assert [item.bend_id for item in filtered] == ["OB4", "OB9", "OB10"]
    report = _render_region_suppression_report(
        [dominant, contained, separate, parallel_offset], filtered, local_spacing_mm=2.0
    )
    assert report[0]["bend_id"] == "OB8"
    assert report[0]["reason_codes"] == ["same_pair_marker_alias"]
    assert report[0]["nearest_kept_regions"][0]["span_overlap_min_ratio"] == 1.0


def test_render_region_suppression_report_explains_hidden_marker_reasons():
    def region(bend_id, pair, anchor, atom_count, support_mass, axis=(1.0, 0.0, 0.0)):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=pair,
            canonical_bend_key=f"{','.join(pair)}::{bend_id}",
            owned_atom_ids=tuple(f"{bend_id}_A{i}" for i in range(atom_count)),
            anchor=anchor,
            axis_direction=axis,
            support_centroid=anchor,
            span_endpoints=(anchor, anchor),
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=support_mass,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class="transition_only",
            debug_confidence=0.8,
        )

    kept_same_pair = region("OB4", ("F1", "F2"), (0.0, 0.0, 0.0), 24, 5000.0)
    suppressed_alias = region("OB5", ("F2", "F1"), (18.0, 0.0, 0.0), 5, 400.0)
    kept_cross_pair = region("OB7", ("F3", "F4"), (120.0, 0.0, 0.0), 20, 4500.0)
    suppressed_cross_pair = region("OB9", ("F3", "F8"), (135.0, 0.0, 0.0), 3, 250.0)

    report = _render_region_suppression_report(
        [kept_same_pair, suppressed_alias, kept_cross_pair, suppressed_cross_pair],
        [kept_same_pair, kept_cross_pair],
        local_spacing_mm=2.0,
    )

    by_id = {item["bend_id"]: item for item in report}
    assert by_id["OB5"]["reason_codes"] == ["same_pair_marker_alias"]
    assert by_id["OB5"]["nearest_kept_regions"][0]["bend_id"] == "OB4"
    assert by_id["OB9"]["reason_codes"] == ["tiny_cross_pair_marker_cluster"]
    assert by_id["OB9"]["nearest_kept_regions"][0]["bend_id"] == "OB7"


def test_build_owned_region_marker_admissibility_counts_and_explains_suppression():
    def region(bend_id, pair, anchor, atom_count, support_mass, axis=(1.0, 0.0, 0.0)):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=pair,
            canonical_bend_key=f"{','.join(pair)}::{bend_id}",
            owned_atom_ids=tuple(f"{bend_id}_A{i}" for i in range(atom_count)),
            anchor=anchor,
            axis_direction=axis,
            support_centroid=anchor,
            span_endpoints=(anchor, anchor),
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=support_mass,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class="transition_only",
            debug_confidence=0.8,
        )

    kept = region("OB4", ("F1", "F2"), (0.0, 0.0, 0.0), 24, 5000.0)
    suppressed = region("OB5", ("F2", "F1"), (18.0, 0.0, 0.0), 5, 400.0)
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index), 0.0, 0.0),
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate((*kept.owned_atom_ids, *suppressed.owned_atom_ids))
        ),
        adjacency={},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=2.0,
    )

    payload = build_owned_region_marker_admissibility([kept, suppressed], atom_graph=graph)

    assert payload["raw_owned_region_count"] == 2
    assert payload["marker_admissible_owned_region_count"] == 1
    assert payload["marker_suppressed_region_ids"] == ["OB5"]
    assert payload["marker_suppression_reason_counts"] == {"same_pair_marker_alias": 1}
    assert payload["marker_suppression_details"][0]["bend_id"] == "OB5"
    assert payload["fragment_upgrade_action_counts"] == {"merge_alias": 1}
    assert payload["fragment_upgrade_diagnostics"][0]["recommended_action"] == "merge_alias"
    assert payload["fragment_upgrade_diagnostics"][0]["upgrade_blockers"] == [
        "same_selected_flange_pair_as_kept_region"
    ]


def test_build_owned_region_marker_admissibility_flags_tiny_cross_pair_for_growth():
    def region(bend_id, pair, anchor, atom_count, support_mass, axis=(1.0, 0.0, 0.0)):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=pair,
            canonical_bend_key=f"{','.join(pair)}::{bend_id}",
            owned_atom_ids=tuple(f"{bend_id}_A{i}" for i in range(atom_count)),
            anchor=anchor,
            axis_direction=axis,
            support_centroid=anchor,
            span_endpoints=(anchor, anchor),
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=support_mass,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class="transition_only",
            debug_confidence=0.8,
        )

    kept_a = region("OB3", ("F27", "F4"), (0.0, 0.0, 0.0), 18, 4007.0)
    kept_b = region("OB4", ("F1", "F14"), (31.0, 0.0, 0.0), 16, 6021.0)
    tiny_cross = region("OB9", ("F27", "F32"), (16.0, 0.0, 0.0), 3, 885.0)
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index), 0.0, 0.0),
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate((*kept_a.owned_atom_ids, *kept_b.owned_atom_ids, *tiny_cross.owned_atom_ids))
        ),
        adjacency={},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.5,
    )

    payload = build_owned_region_marker_admissibility([kept_a, kept_b, tiny_cross], atom_graph=graph)

    assert payload["marker_suppressed_region_ids"] == ["OB9"]
    assert payload["marker_suppression_reason_counts"] == {"tiny_cross_pair_marker_cluster": 1}
    assert payload["fragment_upgrade_action_counts"] == {"requires_support_growth_before_counting": 1}
    diagnostic = payload["fragment_upgrade_diagnostics"][0]
    assert diagnostic["bend_id"] == "OB9"
    assert diagnostic["recommended_action"] == "requires_support_growth_before_counting"
    assert diagnostic["upgrade_blockers"] == [
        "insufficient_adjacent_unowned_growth_support",
        "insufficient_atoms_for_standalone_line_instance",
        "insufficient_grown_support_atoms",
        "insufficient_grown_support_span",
        "too_close_to_existing_kept_marker",
    ]
    assert diagnostic["growth_probe"]["candidate_unowned_atom_count"] == 0
    assert diagnostic["growth_separability"]["status"] == "no_growth_support"


def test_build_owned_region_marker_admissibility_detects_growth_probe_candidates():
    def region(bend_id, pair, atom_ids, anchor, atom_count, support_mass, axis=(1.0, 0.0, 0.0)):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=pair,
            canonical_bend_key=f"{','.join(pair)}::{bend_id}",
            owned_atom_ids=tuple(atom_ids),
            anchor=anchor,
            axis_direction=axis,
            support_centroid=anchor,
            span_endpoints=(anchor, anchor),
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=support_mass,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class="transition_only",
            debug_confidence=0.8,
        )

    kept = region("OB3", ("F27", "F4"), tuple(f"K{i}" for i in range(18)), (0.0, 0.0, 0.0), 18, 4007.0)
    tiny_cross = region("OB9", ("F27", "F32"), ("S1", "S2", "S3"), (16.0, 0.0, 0.0), 3, 885.0)
    extra_atoms = ("R1", "R2", "R3")
    all_atom_ids = (*kept.owned_atom_ids, *tiny_cross.owned_atom_ids, *extra_atoms)
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index), 0.0, 0.0),
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(all_atom_ids)
        ),
        adjacency={
            "S1": ("R1",),
            "S2": ("R2",),
            "S3": ("R3",),
            "R1": ("S1",),
            "R2": ("S2",),
            "R3": ("S3",),
        },
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.5,
    )

    payload = build_owned_region_marker_admissibility([kept, tiny_cross], atom_graph=graph)

    diagnostic = payload["fragment_upgrade_diagnostics"][0]
    assert diagnostic["recommended_action"] == "local_growth_probe_candidate"
    assert diagnostic["growth_needed_atoms"] == 3
    assert diagnostic["growth_probe"]["candidate_unowned_atom_count"] == 3
    assert diagnostic["growth_separability"]["status"] == "growth_available_but_not_separable"
    assert diagnostic["upgrade_blockers"] == [
        "insufficient_grown_support_span",
        "too_close_to_existing_kept_marker",
    ]


def test_build_owned_region_marker_admissibility_detects_separable_growth_candidates():
    def region(bend_id, pair, atom_ids, anchor, axis):
        return OwnedBendRegion(
            bend_id=bend_id,
            bend_class="transition_only",
            incident_flange_ids=pair,
            canonical_bend_key=f"{','.join(pair)}::{bend_id}",
            owned_atom_ids=tuple(atom_ids),
            anchor=anchor,
            axis_direction=axis,
            support_centroid=anchor,
            span_endpoints=(anchor, anchor),
            angle_deg=90.0,
            radius_mm=None,
            visibility_state="internal",
            support_mass=1000.0,
            admissible=True,
            admissibility_reasons=(),
            post_bend_class="transition_only",
            debug_confidence=0.8,
        )

    kept = region("OB3", ("F27", "F4"), tuple(f"K{i}" for i in range(18)), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    tiny_cross = region("OB9", ("F27", "F32"), ("S1", "S2", "S3"), (30.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    extra_atoms = ("R1", "R2", "R3")
    coordinates = {atom_id: (float(index), 0.0, 0.0) for index, atom_id in enumerate(kept.owned_atom_ids)}
    coordinates.update(
        {
            "S1": (30.0, 0.0, 0.0),
            "S2": (30.0, 2.0, 0.0),
            "S3": (30.0, 4.0, 0.0),
            "R1": (30.0, 8.0, 0.0),
            "R2": (30.0, 12.0, 0.0),
            "R3": (30.0, 16.0, 0.0),
        }
    )
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                coordinates[atom_id],
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (0.0, 1.0, 0.0) if atom_id.startswith(("S", "R")) else (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate((*kept.owned_atom_ids, *tiny_cross.owned_atom_ids, *extra_atoms))
        ),
        adjacency={
            "S1": ("R1",),
            "S2": ("R2",),
            "S3": ("R3",),
            "R1": ("S1",),
            "R2": ("S2",),
            "R3": ("S3",),
        },
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.5,
    )

    payload = build_owned_region_marker_admissibility([kept, tiny_cross], atom_graph=graph)

    assert payload["marker_suppressed_owned_region_count"] == 1
    assert payload["marker_admissible_owned_region_count"] == 1
    assert payload["marker_suppressed_region_ids"] == ["OB9"]
    assert payload["marker_quality_reason_counts"] == {
        "edge_dominated_support": 1,
        "hole_or_free_boundary_support": 1,
        "insufficient_line_span": 1,
    }
    assert payload["marker_suppression_details"][0]["reason_codes"] == [
        "edge_dominated_support",
        "hole_or_free_boundary_support",
        "insufficient_line_span",
    ]


def test_build_owned_region_marker_admissibility_accepts_long_line_like_support():
    atom_ids = tuple(f"A{i}" for i in range(13))
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index), 0.0, 0.0),
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(atom_ids)
        ),
        adjacency={atom_id: tuple(neighbor for neighbor in (f"A{index - 1}", f"A{index + 1}") if neighbor in atom_ids) for index, atom_id in enumerate(atom_ids)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    region = OwnedBendRegion(
        bend_id="OB1",
        bend_class="transition_only",
        incident_flange_ids=("F1", "F2"),
        canonical_bend_key="F1,F2::OB1",
        owned_atom_ids=atom_ids,
        anchor=(6.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(6.0, 0.0, 0.0),
        span_endpoints=((0.0, 0.0, 0.0), (12.0, 0.0, 0.0)),
        angle_deg=90.0,
        radius_mm=None,
        visibility_state="internal",
        support_mass=13.0,
        admissible=True,
        admissibility_reasons=(),
        post_bend_class="transition_only",
        debug_confidence=0.9,
    )

    payload = build_owned_region_marker_admissibility([region], atom_graph=graph)

    assert payload["marker_admissible_owned_region_count"] == 1
    assert payload["marker_quality_failed_region_ids"] == []


def test_build_owned_region_marker_admissibility_rejects_tiny_blob_support():
    atom_ids = ("A1", "A2", "A3")
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (0.2 * float(index), 0.0, 0.0),
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(atom_ids)
        ),
        adjacency={"A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2",)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    region = OwnedBendRegion(
        bend_id="OB1",
        bend_class="transition_only",
        incident_flange_ids=("F1", "F2"),
        canonical_bend_key="F1,F2::OB1",
        owned_atom_ids=atom_ids,
        anchor=(0.2, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(0.2, 0.0, 0.0),
        span_endpoints=((0.0, 0.0, 0.0), (0.4, 0.0, 0.0)),
        angle_deg=90.0,
        radius_mm=None,
        visibility_state="internal",
        support_mass=3.0,
        admissible=True,
        admissibility_reasons=(),
        post_bend_class="transition_only",
        debug_confidence=0.9,
    )

    payload = build_owned_region_marker_admissibility([region], atom_graph=graph)

    assert payload["marker_admissible_owned_region_count"] == 0
    assert payload["marker_suppressed_region_ids"] == ["OB1"]
    assert "insufficient_line_span" in payload["marker_quality_reason_counts"]


def test_build_owned_region_marker_admissibility_rejects_boundary_blob_without_incidence():
    atom_ids = ("A1", "A2", "A3", "A4")
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index) * 2.0, 0.0, 0.0),
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(atom_ids)
        ),
        adjacency={atom_id: tuple() for atom_id in atom_ids},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    region = OwnedBendRegion(
        bend_id="OB1",
        bend_class="transition_only",
        incident_flange_ids=("F1",),
        canonical_bend_key="F1::OB1",
        owned_atom_ids=atom_ids,
        anchor=(3.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(3.0, 0.0, 0.0),
        span_endpoints=((0.0, 0.0, 0.0), (6.0, 0.0, 0.0)),
        angle_deg=90.0,
        radius_mm=None,
        visibility_state="internal",
        support_mass=4.0,
        admissible=True,
        admissibility_reasons=("missing_designated_flange_contact",),
        post_bend_class="transition_only",
        debug_confidence=0.9,
    )

    payload = build_owned_region_marker_admissibility([region], atom_graph=graph)

    assert payload["marker_admissible_owned_region_count"] == 0
    assert "edge_dominated_support" in payload["marker_quality_reason_counts"]
    assert "weak_two_flange_incidence" in payload["marker_quality_reason_counts"]


def test_build_owned_region_marker_admissibility_rejects_curve_family_as_conventional():
    atom_ids = tuple(f"A{i}" for i in range(13))
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index), 0.0, 0.0),
                (0.0, 0.7, 0.7),
                1.0,
                0.02,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(atom_ids)
        ),
        adjacency={},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    region = OwnedBendRegion(
        bend_id="OB1",
        bend_class="circular_strip",
        incident_flange_ids=("F1", "F2"),
        canonical_bend_key="curve::OB1",
        owned_atom_ids=atom_ids,
        anchor=(6.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(6.0, 0.0, 0.0),
        span_endpoints=((0.0, 0.0, 0.0), (12.0, 0.0, 0.0)),
        angle_deg=90.0,
        radius_mm=20.0,
        visibility_state="internal",
        support_mass=13.0,
        admissible=True,
        admissibility_reasons=(),
        post_bend_class="circular_strip",
        debug_confidence=0.9,
    )

    payload = build_owned_region_marker_admissibility([region], atom_graph=graph)

    assert payload["marker_admissible_owned_region_count"] == 0
    assert payload["marker_quality_reason_counts"] == {"curve_family_not_conventional": 1}


def test_missed_conventional_line_candidates_find_long_uncounted_support():
    atom_ids = tuple(f"A{i}" for i in range(13))
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index), 0.5, 0.5),
                (0.0, 0.707, 0.707),
                1.0,
                0.03,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(atom_ids)
        ),
        adjacency={atom_id: tuple(neighbor for neighbor in (f"A{index - 1}", f"A{index + 1}") if neighbor in atom_ids) for index, atom_id in enumerate(atom_ids)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    assignment = TypedLabelAssignment(
        atom_labels={atom_id: "residual" for atom_id in atom_ids},
        label_types={"residual": "residual", "outlier": "outlier"},
        energy=0.0,
    )
    flanges = (
        FlangeHypothesis("F1", "test", tuple(), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.9),
        FlangeHypothesis("F2", "test", tuple(), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.9),
    )

    payload = build_missed_conventional_bend_line_candidates(
        atom_graph=graph,
        assignment=assignment,
        flange_hypotheses=flanges,
        owned_bend_regions=(),
    )

    assert payload["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["incident_flange_ids"] == ["F1", "F2"]
    assert candidate["atom_count"] == 13
    assert candidate["span_mm"] >= 12.0
    assert candidate["promotion_state"] == "not_counted"


def test_missed_conventional_line_candidates_skip_existing_owned_region():
    atom_ids = tuple(f"A{i}" for i in range(13))
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                (float(index), 0.5, 0.5),
                (0.0, 0.707, 0.707),
                1.0,
                0.03,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(atom_ids)
        ),
        adjacency={atom_id: tuple(neighbor for neighbor in (f"A{index - 1}", f"A{index + 1}") if neighbor in atom_ids) for index, atom_id in enumerate(atom_ids)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    assignment = TypedLabelAssignment(
        atom_labels={atom_id: "residual" for atom_id in atom_ids},
        label_types={"residual": "residual", "outlier": "outlier"},
        energy=0.0,
    )
    flanges = (
        FlangeHypothesis("F1", "test", tuple(), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.9),
        FlangeHypothesis("F2", "test", tuple(), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.9),
    )
    existing = OwnedBendRegion(
        bend_id="OB1",
        bend_class="transition_only",
        incident_flange_ids=("F1", "F2"),
        canonical_bend_key="F1,F2::OB1",
        owned_atom_ids=atom_ids,
        anchor=(6.0, 0.5, 0.5),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(6.0, 0.5, 0.5),
        span_endpoints=((0.0, 0.5, 0.5), (12.0, 0.5, 0.5)),
        angle_deg=90.0,
        radius_mm=None,
        visibility_state="internal",
        support_mass=13.0,
        admissible=True,
        admissibility_reasons=(),
        post_bend_class="transition_only",
        debug_confidence=0.9,
    )

    payload = build_missed_conventional_bend_line_candidates(
        atom_graph=graph,
        assignment=assignment,
        flange_hypotheses=flanges,
        owned_bend_regions=(existing,),
    )

    assert payload["candidate_count"] == 0
    assert payload["rejection_counts"]["near_existing_owned_region"] == 1


def test_missed_recovery_flags_branched_junction_complex():
    coords = {
        "A0": (0.0, 0.0, 0.0),
        "A1": (3.0, 0.0, 0.0),
        "A2": (6.0, 0.0, 0.0),
        "A3": (9.0, 0.0, 0.0),
        "A4": (12.0, 0.0, 0.0),
        "B1": (6.0, 3.0, 0.0),
        "B2": (6.0, 6.0, 0.0),
        "B3": (6.0, 9.0, 0.0),
        "B4": (6.0, 12.0, 0.0),
        "C1": (6.0, 0.0, 3.0),
        "C2": (6.0, 0.0, 6.0),
        "C3": (6.0, 0.0, 9.0),
        "C4": (6.0, 0.0, 12.0),
    }
    adjacency = {
        "A0": ("A1",),
        "A1": ("A0", "A2"),
        "A2": ("A1", "A3", "B1", "C1"),
        "A3": ("A2", "A4"),
        "A4": ("A3",),
        "B1": ("A2", "B2"),
        "B2": ("B1", "B3"),
        "B3": ("B2", "B4"),
        "B4": ("B3",),
        "C1": ("A2", "C2"),
        "C2": ("C1", "C3"),
        "C3": ("C2", "C4"),
        "C4": ("C3",),
    }
    graph = SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id,
                (index, 0, 0),
                (index,),
                coords[atom_id],
                (0.0, 0.707, 0.707),
                1.0,
                0.03,
                (1.0, 0.0, 0.0),
                1.0,
                1.0,
            )
            for index, atom_id in enumerate(coords)
        ),
        adjacency=adjacency,
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=1.0,
    )
    assignment = TypedLabelAssignment(
        atom_labels={atom_id: "residual" for atom_id in coords},
        label_types={"residual": "residual", "outlier": "outlier"},
        energy=0.0,
    )
    flanges = (
        FlangeHypothesis("F1", "test", ("A0", "A1"), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.9),
        FlangeHypothesis("F2", "test", ("B3", "B4"), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.9),
        FlangeHypothesis("F3", "test", ("C3", "C4"), (1.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.9),
    )

    payload = build_missed_conventional_bend_line_candidates(
        atom_graph=graph,
        assignment=assignment,
        flange_hypotheses=flanges,
        owned_bend_regions=(),
    )

    assert payload["junction_complex_candidate_count"] == 1
    complex_candidate = payload["junction_complex_candidates"][0]
    assert "branched_atom_graph_support" in complex_candidate["junction_evidence"]
    assert "non_line_like_support" in complex_candidate["junction_evidence"]
    assert complex_candidate["promotion_state"] == "not_counted_requires_skeleton_solver"
    assert complex_candidate["junction_path_candidate_count"] >= 2
    assert all(
        path_candidate["promotion_state"] == "not_counted_requires_local_map"
        for path_candidate in complex_candidate["junction_path_candidates"]
    )
    local_map = complex_candidate["local_junction_map"]
    assert local_map["enabled"] is True
    assert local_map["promotion_state"] == "not_counted_local_junction_map_diagnostic"
    assert local_map["selected_path_count"] >= 0
    assert local_map["selected_path_count"] + local_map["excluded_path_count"] == complex_candidate["junction_path_candidate_count"]
    assert complex_candidate["promotion_readiness"]["promotion_state"].startswith("not_counted")
    assert payload["recovery_promotion_readiness"]["promotion_state"].startswith("not_counted")


def test_junction_promotion_readiness_requires_multi_arm_conventional_arrangement():
    base_candidate = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F1", "F2"],
        "atom_count": 24,
        "span_mm": 80.0,
        "min_required_span_mm": 12.0,
        "two_flange_contact_count": 2,
        "mean_bend_signal": 0.35,
        "path_quality": {
            "accepted_for_local_map": True,
            "reason_codes": [],
            "candidate_class": "conventional_flange_pair",
        },
    }
    local_map = {
        "selection_enabled": True,
        "selected_path_count": 1,
        "selected_path_candidate_ids": ["JPC1"],
        "selected_adjusted_scores": [9.0],
    }

    readiness = _junction_local_map_promotion_readiness(
        junction_path_candidates=[base_candidate],
        local_junction_map=local_map,
        route_id="small_tight_bend_recovery",
    )

    assert readiness["eligible_path_count"] == 1
    assert readiness["status"] == "blocked_single_arm_needs_companion_or_existing_bend"
    assert readiness["manual_review_required"] is False

    second_candidate = {
        **base_candidate,
        "candidate_id": "JPC2",
        "incident_flange_ids": ["F2", "F3"],
    }
    two_path_map = {
        "selection_enabled": True,
        "selected_path_count": 2,
        "selected_path_candidate_ids": ["JPC1", "JPC2"],
        "selected_adjusted_scores": [9.0, 8.0],
    }
    two_path_readiness = _junction_local_map_promotion_readiness(
        junction_path_candidates=[base_candidate, second_candidate],
        local_junction_map=two_path_map,
        route_id="small_tight_bend_recovery",
    )

    assert two_path_readiness["eligible_path_count"] == 2
    assert two_path_readiness["status"] == "ready_for_manual_promotion_validation"
    assert two_path_readiness["manual_review_required"] is True


def test_junction_promotion_readiness_blocks_internal_raise_paths():
    candidate = {
        "candidate_id": "JPC1",
        "candidate_class": "internal_ridge_or_raise",
        "incident_flange_ids": [],
        "atom_count": 24,
        "span_mm": 80.0,
        "min_required_span_mm": 12.0,
        "two_flange_contact_count": 0,
        "mean_bend_signal": 0.35,
        "path_quality": {
            "accepted_for_local_map": True,
            "reason_codes": [],
            "candidate_class": "internal_ridge_or_raise",
        },
    }
    readiness = _junction_local_map_promotion_readiness(
        junction_path_candidates=[candidate],
        local_junction_map={
            "selection_enabled": True,
            "selected_path_count": 1,
            "selected_path_candidate_ids": ["JPC1"],
            "selected_adjusted_scores": [9.0],
        },
        route_id="small_tight_bend_recovery",
    )

    assert readiness["eligible_path_count"] == 0
    assert readiness["status"] == "blocked_internal_ridge_or_raise_not_conventional"
    assert readiness["blocked_reason_counts"]["internal_ridge_or_raise_not_conventional"] == 1


def test_junction_promotion_readiness_reports_near_miss_companion_arm():
    selected_candidate = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F1", "F2"],
        "atom_count": 24,
        "span_mm": 80.0,
        "min_required_span_mm": 12.0,
        "two_flange_contact_count": 2,
        "mean_bend_signal": 0.35,
        "score": 30.0,
        "path_quality": {
            "accepted_for_local_map": True,
            "reason_codes": [],
            "candidate_class": "conventional_flange_pair",
        },
    }
    near_miss = {
        "candidate_id": "JPC2",
        "incident_flange_ids": ["F2", "F3"],
        "atom_count": 20,
        "span_mm": 42.0,
        "min_required_span_mm": 12.0,
        "two_flange_contact_count": 1,
        "mean_bend_signal": 0.3,
        "score": 24.0,
        "path_quality": {
            "accepted_for_local_map": False,
            "reason_codes": ["junction_path_pair_axis_mismatch", "junction_path_weak_two_flange_contact"],
            "candidate_class": "conventional_flange_pair",
        },
    }

    readiness = _junction_local_map_promotion_readiness(
        junction_path_candidates=[selected_candidate, near_miss],
        local_junction_map={
            "selection_enabled": True,
            "selected_path_count": 1,
            "selected_path_candidate_ids": ["JPC1"],
            "selected_adjusted_scores": [9.0],
        },
        route_id="small_tight_bend_recovery",
    )

    assert readiness["status"] == "blocked_near_miss_companion_rejected_by_arrangement"
    assert readiness["eligible_path_count"] == 1
    assert readiness["near_miss_companion_count"] == 1
    assert readiness["near_miss_companion_paths"][0]["candidate_id"] == "JPC2"
    assert readiness["manual_review_required"] is False


def test_junction_promotion_readiness_reports_missing_arm_birth_targets():
    selected_candidate = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F1", "F2"],
        "atom_ids": [f"A{i}" for i in range(24)],
        "support_centroid": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "atom_count": 24,
        "span_mm": 80.0,
        "min_required_span_mm": 12.0,
        "two_flange_contact_count": 2,
        "mean_bend_signal": 0.35,
        "score": 30.0,
        "path_quality": {
            "accepted_for_local_map": True,
            "reason_codes": [],
            "candidate_class": "conventional_flange_pair",
        },
    }

    readiness = _junction_local_map_promotion_readiness(
        junction_path_candidates=[selected_candidate],
        local_junction_map={
            "selection_enabled": True,
            "selected_path_count": 1,
            "selected_path_candidate_ids": ["JPC1"],
            "selected_adjusted_scores": [9.0],
        },
        route_id="small_tight_bend_recovery",
        touching_flange_ids=["F1", "F2", "F3"],
    )

    assert readiness["status"] == "blocked_missing_arm_birth_required"
    birth_targets = readiness["missing_arm_birth_targets"]
    assert birth_targets["target_count"] == 2
    target_pairs = {tuple(item["target_incident_flange_ids"]) for item in birth_targets["targets"]}
    assert ("F1", "F3") in target_pairs
    assert ("F2", "F3") in target_pairs


def test_missing_arm_birth_support_proposal_finds_connected_pair_support():
    atoms = tuple(
        SurfaceAtom(
            f"B{i}",
            (i, 0, 0),
            (i,),
            (float(i) * 4.0, 0.0, 0.0),
            (0.0, 0.7071, 0.7071),
            1.0,
            0.04,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for i in range(10)
    ) + (
        SurfaceAtom("E1", (-1, 0, 0), (100,), (-8.0, 6.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.04, (1.0, 0.0, 0.0), 1.0, 1.0),
        SurfaceAtom("E2", (-2, 0, 0), (101,), (-4.0, 6.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.04, (1.0, 0.0, 0.0), 1.0, 1.0),
    )
    adjacency = {
        f"B{i}": tuple(neighbor for neighbor in (f"B{i - 1}", f"B{i + 1}") if neighbor in {f"B{j}" for j in range(10)})
        for i in range(10)
    }
    adjacency.update({"E1": ("E2",), "E2": ("E1",)})
    graph = SurfaceAtomGraph(atoms, adjacency, {}, 1.0, 1.0)
    flanges = {
        "F2": FlangeHypothesis("F2", "test", ("B0",), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 1.0),
        "F3": FlangeHypothesis("F3", "test", ("B9",), (0.0, 1.0, 0.0), 0.0, (36.0, 0.0, 0.0), 1.0),
    }

    support = _missing_arm_birth_support_proposal(
        target={"eligible_candidate_id": "JPC1", "target_incident_flange_ids": ["F2", "F3"]},
        component={atom.atom_id for atom in atoms},
        candidate_by_id={"JPC1": {"atom_ids": ["E1", "E2"]}},
        atom_graph=graph,
        atom_lookup={atom.atom_id: atom for atom in atoms},
        flange_lookup=flanges,
        local_spacing_mm=1.0,
    )

    assert support["status"] == "birth_support_found_not_counted"
    assert support["promotion_state"] == "not_counted_birth_support_diagnostic"
    assert support["atom_count"] == 10
    assert support["span_mm"] >= 30.0
    assert support["two_flange_contact_count"] == 2


def test_missing_arm_birth_support_proposal_reports_missing_contact_extension():
    support_atoms = tuple(
        SurfaceAtom(
            f"B{i}",
            (i, 0, 0),
            (i,),
            (float(i) * 4.0, 0.0, 0.0),
            (0.0, 0.7071, 0.7071),
            1.0,
            0.04,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for i in range(8)
    )
    extension_atoms = tuple(
        SurfaceAtom(
            f"X{i}",
            (100 + i, 0, 0),
            (100 + i,),
            (34.0 + float(i) * 4.0, 0.0, 0.0),
            (0.0, 0.7071, 0.7071),
            1.0,
            0.04,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for i in range(3)
    )
    atoms = support_atoms + extension_atoms
    adjacency = {
        f"B{i}": tuple(neighbor for neighbor in (f"B{i - 1}", f"B{i + 1}") if neighbor in {f"B{j}" for j in range(8)})
        for i in range(8)
    }
    adjacency.update(
        {
            "X0": ("X1",),
            "X1": ("X0", "X2"),
            "X2": ("X1",),
        }
    )
    graph = SurfaceAtomGraph(atoms, adjacency, {}, 1.0, 1.0)
    flanges = {
        "F2": FlangeHypothesis("F2", "test", ("B0",), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 1.0),
        "F3": FlangeHypothesis("F3", "test", ("X1", "X2"), (0.0, 1.0, 0.0), 0.0, (42.0, 0.0, 0.0), 1.0),
    }

    support = _missing_arm_birth_support_proposal(
        target={"eligible_candidate_id": "JPC1", "target_incident_flange_ids": ["F2", "F3"]},
        component={atom.atom_id for atom in atoms},
        candidate_by_id={},
        atom_graph=graph,
        atom_lookup={atom.atom_id: atom for atom in atoms},
        flange_lookup=flanges,
        local_spacing_mm=1.0,
    )

    assert support["status"] == "birth_support_found_not_counted"
    assert support["two_flange_contact_count"] == 1
    assert support["missing_contact_flange_ids"] == ["F3"]
    assert support["weak_contact_extension"]["status"] == "extension_support_found"
    assert support["weak_contact_extension"]["missing_contact_flange_ids"] == ["F3"]


def test_two_sided_contact_corridor_support_proposal_requires_two_flange_contact():
    atoms = tuple(
        SurfaceAtom(
            f"B{i}",
            (i, 0, 0),
            (i,),
            (float(i) * 6.0, 0.0, 0.0),
            (0.0, 0.7071, 0.7071),
            1.0,
            0.04,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for i in range(8)
    )
    adjacency = {
        f"B{i}": tuple(neighbor for neighbor in (f"B{i - 1}", f"B{i + 1}") if neighbor in {f"B{j}" for j in range(8)})
        for i in range(8)
    }
    graph = SurfaceAtomGraph(atoms, adjacency, {}, 1.0, 1.0)
    flanges = {
        "F2": FlangeHypothesis("F2", "test", ("B0", "B1", "B2", "B3", "B4", "B5"), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 1.0),
        "F3": FlangeHypothesis("F3", "test", ("B2", "B3", "B4", "B5", "B6", "B7"), (0.0, 1.0, 0.0), 0.0, (36.0, 0.0, 0.0), 1.0),
    }

    corridor = _two_sided_contact_corridor_support_proposal(
        target={"eligible_candidate_id": "JPC1", "target_incident_flange_ids": ["F2", "F3"]},
        component={atom.atom_id for atom in atoms},
        candidate_by_id={},
        atom_graph=graph,
        atom_lookup={atom.atom_id: atom for atom in atoms},
        flange_lookup=flanges,
        local_spacing_mm=1.0,
    )

    assert corridor["status"] == "two_sided_corridor_found_not_counted"
    assert corridor["atom_count"] >= 4
    assert corridor["span_mm"] >= 18.0


def test_two_sided_contact_pair_diagnostics_finds_span_sufficient_pair():
    atoms = tuple(
        SurfaceAtom(
            f"B{i}",
            (i, 0, 0),
            (i,),
            (float(i) * 6.0, 0.0, 0.0),
            (0.0, 0.7071, 0.7071),
            1.0,
            0.04,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for i in range(8)
    )
    atom_ids = {atom.atom_id for atom in atoms}
    adjacency = {
        f"B{i}": tuple(neighbor for neighbor in (f"B{i - 1}", f"B{i + 1}") if neighbor in atom_ids)
        for i in range(8)
    }
    graph = SurfaceAtomGraph(atoms, adjacency, {}, 1.0, 1.0)
    flanges = (
        FlangeHypothesis("F2", "test", ("B0", "B1", "B2", "B3", "B4", "B5"), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 1.0),
        FlangeHypothesis("F3", "test", ("B2", "B3", "B4", "B5", "B6", "B7"), (0.0, 1.0, 0.0), 0.0, (36.0, 0.0, 0.0), 1.0),
    )

    diagnostics = _build_two_sided_contact_pair_diagnostics(
        component=atom_ids,
        atom_graph=graph,
        atom_lookup={atom.atom_id: atom for atom in atoms},
        flange_hypotheses=flanges,
        touching_flange_ids=("F2", "F3"),
        local_spacing_mm=1.0,
    )

    assert diagnostics
    assert diagnostics[0]["status"] == "two_sided_pair_candidate"
    assert diagnostics[0]["incident_flange_ids"] == ["F2", "F3"]
    assert diagnostics[0]["atom_count"] >= 4
    assert diagnostics[0]["span_mm"] >= 18.0


def test_missing_arm_birth_support_proposal_blocks_tiny_support():
    atoms = tuple(
        SurfaceAtom(
            f"B{i}",
            (i, 0, 0),
            (i,),
            (float(i) * 2.0, 0.0, 0.0),
            (0.0, 0.7071, 0.7071),
            1.0,
            0.04,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for i in range(3)
    )
    graph = SurfaceAtomGraph(
        atoms,
        {"B0": ("B1",), "B1": ("B0", "B2"), "B2": ("B1",)},
        {},
        1.0,
        1.0,
    )
    flanges = {
        "F2": FlangeHypothesis("F2", "test", ("B0",), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 1.0),
        "F3": FlangeHypothesis("F3", "test", ("B2",), (0.0, 1.0, 0.0), 0.0, (4.0, 0.0, 0.0), 1.0),
    }

    support = _missing_arm_birth_support_proposal(
        target={"eligible_candidate_id": "JPC1", "target_incident_flange_ids": ["F2", "F3"]},
        component={atom.atom_id for atom in atoms},
        candidate_by_id={},
        atom_graph=graph,
        atom_lookup={atom.atom_id: atom for atom in atoms},
        flange_lookup=flanges,
        local_spacing_mm=1.0,
    )

    assert support["status"] == "birth_support_too_small"
    assert "birth_support_too_small" in support["reason_codes"]
    assert support["promotion_state"] == "not_counted_birth_support_diagnostic"


def test_junction_birth_support_arrangement_diagnostic_accepts_sparse_three_arm_path():
    eligible = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F11", "F14"],
        "atom_ids": [f"E{i}" for i in range(20)],
        "support_centroid": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 80.0,
        "two_flange_contact_count": 2,
    }
    supports = [
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F11", "F17"],
            "atom_ids": [f"L{i}" for i in range(20)],
            "support_centroid": [0.0, 25.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 45.0,
            "score": 18.0,
            "two_flange_contact_count": 2,
        },
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F1", "F14"],
            "atom_ids": [f"R{i}" for i in range(20)],
            "support_centroid": [30.0, 0.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 42.0,
            "score": 16.0,
            "two_flange_contact_count": 2,
        },
    ]

    diagnostic = _junction_birth_support_arrangement_diagnostics(
        eligible_paths=[eligible],
        birth_supports=supports,
    )

    assert diagnostic["status"] == "ready_for_manual_birth_arrangement_review"
    assert diagnostic["manual_review_required"] is True
    assert diagnostic["arm_count"] == 3
    assert diagnostic["flange_graph_connected"] is True
    assert diagnostic["flange_graph_edge_count"] == 3


def test_junction_birth_support_arrangement_diagnostic_blocks_weak_birth_contact():
    eligible = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F11", "F14"],
        "atom_ids": [f"E{i}" for i in range(20)],
        "support_centroid": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 80.0,
        "two_flange_contact_count": 2,
    }
    supports = [
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F11", "F17"],
            "atom_ids": [f"L{i}" for i in range(20)],
            "support_centroid": [0.0, 25.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 45.0,
            "score": 18.0,
            "two_flange_contact_count": 1,
        },
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F1", "F14"],
            "atom_ids": [f"R{i}" for i in range(20)],
            "support_centroid": [30.0, 0.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 42.0,
            "score": 16.0,
            "two_flange_contact_count": 2,
        },
    ]

    diagnostic = _junction_birth_support_arrangement_diagnostics(
        eligible_paths=[eligible],
        birth_supports=supports,
    )

    assert diagnostic["status"] == "blocked_birth_support_arrangement"
    assert "birth_support_weak_two_flange_contact" in diagnostic["reason_codes"]
    assert diagnostic["manual_review_required"] is False


def test_junction_birth_support_arrangement_diagnostic_allows_strong_single_sided_birth_arms_for_review():
    eligible = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F12", "F22"],
        "atom_ids": [f"E{i}" for i in range(40)],
        "support_centroid": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 110.0,
        "two_flange_contact_count": 2,
    }
    supports = [
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F1", "F22"],
            "atom_ids": [f"L{i}" for i in range(40)],
            "support_centroid": [0.0, 35.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 32.0,
            "score": 28.0,
            "two_flange_contact_count": 1,
            "median_line_residual_mm": 9.5,
            "median_normal_arc_cost": 0.42,
            "mean_bend_signal": 0.72,
        },
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F12", "F32"],
            "atom_ids": [f"R{i}" for i in range(40)],
            "support_centroid": [32.0, 0.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 29.0,
            "score": 16.0,
            "two_flange_contact_count": 1,
            "median_line_residual_mm": 12.8,
            "median_normal_arc_cost": 0.25,
            "mean_bend_signal": 0.81,
        },
    ]

    diagnostic = _junction_birth_support_arrangement_diagnostics(
        eligible_paths=[eligible],
        birth_supports=supports,
    )

    assert diagnostic["status"] == "ready_for_manual_birth_arrangement_review"
    assert diagnostic["manual_review_required"] is True
    assert diagnostic["single_sided_geometry_review_arm_count"] == 2
    assert "birth_support_weak_two_flange_contact" not in diagnostic["reason_codes"]


def test_junction_birth_support_arrangement_diagnostic_blocks_disconnected_chords():
    eligible = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F11", "F14"],
        "atom_ids": [f"E{i}" for i in range(20)],
        "support_centroid": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 80.0,
        "two_flange_contact_count": 2,
    }
    supports = [
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F1", "F22"],
            "atom_ids": [f"L{i}" for i in range(20)],
            "support_centroid": [0.0, 25.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 45.0,
            "score": 18.0,
            "two_flange_contact_count": 1,
        },
        {
            "status": "birth_support_found_not_counted",
            "target_incident_flange_ids": ["F12", "F32"],
            "atom_ids": [f"R{i}" for i in range(20)],
            "support_centroid": [30.0, 0.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
            "span_mm": 42.0,
            "score": 16.0,
            "two_flange_contact_count": 1,
        },
    ]

    diagnostic = _junction_birth_support_arrangement_diagnostics(
        eligible_paths=[eligible],
        birth_supports=supports,
    )

    assert diagnostic["status"] == "blocked_birth_support_arrangement"
    assert "arrangement_flange_graph_disconnected" in diagnostic["reason_codes"]
    assert diagnostic["manual_review_required"] is False


def test_junction_promotion_readiness_reports_repairable_two_arm_arrangement():
    selected_candidate = {
        "candidate_id": "JPC1",
        "incident_flange_ids": ["F1", "F2"],
        "atom_ids": [f"A{i}" for i in range(24)],
        "support_centroid": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "atom_count": 24,
        "span_mm": 80.0,
        "min_required_span_mm": 12.0,
        "two_flange_contact_count": 2,
        "mean_bend_signal": 0.35,
        "score": 30.0,
        "path_quality": {
            "accepted_for_local_map": True,
            "reason_codes": [],
            "candidate_class": "conventional_flange_pair",
        },
    }
    near_miss = {
        "candidate_id": "JPC2",
        "incident_flange_ids": ["F2", "F3"],
        "atom_ids": [f"B{i}" for i in range(20)],
        "support_centroid": [20.0, 8.0, 0.0],
        "axis_direction": [0.0, 1.0, 0.0],
        "atom_count": 20,
        "span_mm": 42.0,
        "min_required_span_mm": 12.0,
        "two_flange_contact_count": 1,
        "mean_bend_signal": 0.3,
        "score": 24.0,
        "path_quality": {
            "accepted_for_local_map": False,
            "reason_codes": ["junction_path_pair_axis_mismatch", "junction_path_weak_two_flange_contact"],
            "candidate_class": "conventional_flange_pair",
        },
    }

    readiness = _junction_local_map_promotion_readiness(
        junction_path_candidates=[selected_candidate, near_miss],
        local_junction_map={
            "selection_enabled": True,
            "selected_path_count": 1,
            "selected_path_candidate_ids": ["JPC1"],
            "selected_adjusted_scores": [9.0],
        },
        route_id="small_tight_bend_recovery",
    )

    assert readiness["status"] == "blocked_single_arm_with_repairable_companion_arrangement"
    arrangement = readiness["companion_arrangement_diagnostic"]
    assert arrangement["repairable_arrangement_candidate_count"] == 1
    assert arrangement["arrangement_candidates"][0]["repair_action"] == "refit_pair_axis_and_contact_corridor"


def test_dedupe_bend_hypotheses_collapses_overlap_even_when_flange_ids_drift():
    bend1 = _bend("B1", atom_ids=("A1", "A2", "A3", "A4"), cross_width_mm=2.0, span_mm=10.0, confidence=0.9)
    bend2 = _bend(
        "B2",
        incident_flange_ids=("F9", "F10"),
        atom_ids=("A2", "A3", "A4", "A5"),
        anchor=(1.5, 0.0, 0.0),
        angle_deg=92.0,
        cross_width_mm=2.0,
        span_mm=10.0,
    )

    kept = _dedupe_bend_hypotheses([bend1, bend2], {})

    assert len(kept) == 1
    assert kept[0].bend_id == "B1"


def test_prune_and_refit_labels_drops_tiny_disconnected_bend_fragments():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
        SurfaceAtom("A4", (10, 0, 0), (3,), (10.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={"A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2",), "A4": ()},
    )
    bend = _bend("B1", atom_ids=("A1", "A2", "A3", "A4"), anchor=(1.0, 0.0, 0.0), span_mm=4.0)

    labels = _prune_and_refit_labels(
        labels={"A1": "B1", "A2": "B1", "A3": "B1", "A4": "B1"},
        atom_graph=graph,
        flange_hypotheses=[],
        bend_hypotheses=[bend],
    )

    assert labels["A4"] == "residual"
    assert labels["A1"] == "B1"


def test_render_decomposition_artifacts_marks_raw_f1_ownership_semantics(tmp_path, monkeypatch):
    atom_ids = tuple(f"A{i}" for i in range(13))
    atoms = tuple(
        SurfaceAtom(
            atom_id,
            (index, 0, 0),
            (index,),
            (float(index), 0.0, 0.0),
            (0.0, 0.7, 0.7),
            0.2,
            0.02,
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
        )
        for index, atom_id in enumerate(atom_ids)
    )
    graph = SurfaceAtomGraph(
        atoms=atoms,
        adjacency={atom_id: tuple(neighbor for neighbor in (f"A{index - 1}", f"A{index + 1}") if neighbor in atom_ids) for index, atom_id in enumerate(atom_ids)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    region = OwnedBendRegion(
        bend_id="OB1",
        bend_class="transition_only",
        incident_flange_ids=("F1", "F2"),
        canonical_bend_key="F1,F2::B1",
        owned_atom_ids=atom_ids,
        anchor=(6.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(6.0, 0.0, 0.0),
        span_endpoints=((0.0, 0.0, 0.0), (12.0, 0.0, 0.0)),
        angle_deg=90.0,
        radius_mm=None,
        visibility_state="internal",
        support_mass=13.0,
        admissible=True,
        admissibility_reasons=(),
        post_bend_class="transition_only",
        debug_confidence=0.8,
    )
    result = DecompositionResult(
        scan_path="toy.ply",
        part_id="toy",
        atom_graph=graph,
        flange_hypotheses=(),
        bend_hypotheses=(),
        assignment=TypedLabelAssignment(
            atom_labels={atom_id: "B1" for atom_id in atom_ids},
            label_types={"B1": "bend"},
            energy=1.0,
        ),
        owned_bend_regions=(region,),
        exact_bend_count=1,
        bend_count_range=(1, 1),
        candidate_solution_energies=(1.0,),
        candidate_solution_counts=(1,),
        object_level_confidence_summary={"F1,F2::B1": 0.8},
        repair_diagnostics={},
        current_zero_shot_result={},
        current_zero_shot_count_range=(6, 6),
        current_zero_shot_exact_count=6,
    )

    class DummyRenderer:
        def render_bend_status_overlay(self, **kwargs):
            assert kwargs["title_override"].startswith(("Debug Raw F1", "Scan Context"))
            return b"dummy-png-bytes"

    import patch_graph_latent_decomposition as module

    monkeypatch.setattr(module, "Model3DSnapshotRenderer", DummyRenderer)
    artifacts = render_decomposition_artifacts(result=result, output_dir=tmp_path)

    manifest = json.loads((tmp_path / "owned_region_manifest.json").read_text())
    assert manifest["owned_region_semantics"] == "raw_f1_owned_support_regions_not_always_accepted_candidate_count"
    assert manifest["render_debug_semantics"] == "debug_raw_owned_regions_not_final_accepted_bends"
    assert manifest["raw_f1_owned_region_count"] == 1
    assert manifest["render_owned_region_count"] == 1
    assert manifest["all_raw_render_owned_region_count"] == 1
    assert manifest["render_suppressed_owned_region_count"] == 0
    assert manifest["render_suppressed_region_ids"] == []
    assert manifest["render_suppression_details"] == []
    assert manifest["all_raw_overview_image"] == "patch_graph_owned_region_overview_all_raw.png"
    assert "estimated_bend_count" not in manifest
    assert manifest["bends"][0]["metadata"]["render_anchor_source"] == "owned_atom_medoid"
    assert manifest["bends"][0]["anchor"] in manifest["bends"][0]["metadata"]["owned_support_points"]
    assert Path(artifacts["atom_projection_path"]).exists()
    assert Path(artifacts["all_raw_overview_path"]).exists()
    assert (tmp_path / "patch_graph_owned_region_atom_projection.png").read_bytes().startswith(b"\x89PNG")


def test_no_contact_birth_rescue_only_enables_for_wide_control_ranges():
    assert _allow_no_contact_birth_rescue_for_range((10, 12)) is True
    assert _allow_no_contact_birth_rescue_for_range((8, 12)) is True
    assert _allow_no_contact_birth_rescue_for_range((11, 12)) is False
    assert _allow_no_contact_birth_rescue_for_range((6, 6)) is False


def test_incidence_penalty_penalizes_multiple_components_for_same_bend_label():
    graph = SimpleNamespace(
        adjacency={
            "A1": ("F1A",),
            "A2": ("F2A",),
            "A3": ("F1B",),
            "A4": ("F2B",),
            "F1A": ("A1",),
            "F2A": ("A2",),
            "F1B": ("A3",),
            "F2B": ("A4",),
        }
    )
    bend = _bend("B1", atom_ids=("A1", "A2", "A3", "A4"), span_mm=4.0)
    label_types = {
        "B1": "bend",
        "F1": "flange",
        "F2": "flange",
    }
    split_labels = {
        "A1": "B1",
        "A2": "B1",
        "A3": "B1",
        "A4": "B1",
        "F1A": "F1",
        "F2A": "F2",
        "F1B": "F1",
        "F2B": "F2",
    }
    single_component_graph = SimpleNamespace(
        adjacency={
            "A1": ("A2", "F1A"),
            "A2": ("A1", "F2A"),
            "F1A": ("A1",),
            "F2A": ("A2",),
        }
    )
    single_labels = {
        "A1": "B1",
        "A2": "B1",
        "F1A": "F1",
        "F2A": "F2",
    }

    split_penalty = _incidence_penalty(graph, split_labels, {"B1": bend}, label_types)
    single_penalty = _incidence_penalty(single_component_graph, single_labels, {"B1": bend}, label_types)

    assert split_penalty > single_penalty


def test_solver_label_cost_prevents_trivial_bend_oversplitting():
    points, normals = _toy_points()
    graph = build_surface_atom_graph(points, normals, voxel_size_mm=1.0, local_spacing_mm=0.2)
    flanges = [
        FlangeHypothesis("F1", "seed", tuple(atom.atom_id for atom in graph.atoms[:1]), (0.0, 0.0, 1.0), 0.0, graph.atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", tuple(atom.atom_id for atom in graph.atoms[-1:]), (0.0, 1.0, 0.0), 0.0, graph.atoms[-1].centroid, 0.8),
    ]
    bends = [
        _bend("B1", atom_ids=tuple(atom.atom_id for atom in graph.atoms), anchor=graph.atoms[1].centroid, confidence=0.7, span_mm=4.0),
        _bend(
            "B2",
            bend_class="developable_strip",
            incident_flange_ids=tuple(),
            atom_ids=tuple(atom.atom_id for atom in graph.atoms[:1]),
            anchor=graph.atoms[0].centroid,
            confidence=0.4,
            span_mm=1.0,
            candidate_classes=("transition_only", "developable_strip"),
        ),
    ]

    solved = solve_typed_latent_decomposition(
        atom_graph=graph,
        flange_hypotheses=flanges,
        bend_hypotheses=bends,
    )

    count = _bend_component_count(graph, solved["assignment"].atom_labels, solved["assignment"].label_types)
    assert count <= 1


def test_normalize_bend_hypotheses_collapses_reversed_flange_order_into_one_family():
    normalized = _normalize_bend_hypotheses(
        [
            _bend("B1", incident_flange_ids=("F4", "F6"), atom_ids=("A1", "A2"), anchor=(0.0, 0.0, 0.0), source_bend_ids=("S1",)),
            _bend("B2", incident_flange_ids=("F6", "F4"), atom_ids=("A2", "A3"), anchor=(0.2, 0.0, 0.0), source_bend_ids=("S2",)),
        ]
    )

    assert len(normalized) == 1
    assert normalized[0].canonical_flange_pair == ("F4", "F6")
    assert normalized[0].bend_class == "transition_only"
    assert normalized[0].candidate_classes == ("transition_only",)


def test_bend_component_admissibility_rejects_leaky_low_mass_components():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 0.2, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1,),
        adjacency={"A1": ("F1A", "F3A"), "F1A": ("A1",), "F3A": ("A1",)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    bend = _bend("B1", atom_ids=("A1",))
    admissible, reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels={"A1": "B1", "F1A": "F1", "F3A": "F3"},
        label_types={"B1": "bend", "F1": "flange", "F2": "flange", "F3": "flange"},
        atom_lookup={"A1": atom1},
    )

    assert admissible is False
    assert "support_mass_too_small" in reasons


def test_bend_component_admissibility_can_reattach_to_best_flange_pair():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1,),
        adjacency={"A1": ("F1A", "F3A"), "F1A": ("A1",), "F3A": ("A1",)},
        edge_weights={"A1::F1A": 1.0, "A1::F3A": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    bend = _bend(
        "B1",
        incident_flange_ids=("F1", "F2"),
        atom_ids=("A1",),
        candidate_incident_flange_pairs=(("F1", "F2"), ("F1", "F3")),
    )

    admissible, reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels={"A1": "B1", "F1A": "F1", "F3A": "F3"},
        label_types={"B1": "bend", "F1": "flange", "F2": "flange", "F3": "flange"},
        atom_lookup={"A1": atom1},
    )

    assert admissible is True
    assert "missing_designated_flange_contact" not in reasons


def test_bend_component_admissibility_allows_no_share_pair_when_corridor_matches():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1,),
        adjacency={"A1": ("F3A", "F4A"), "F3A": ("A1",), "F4A": ("A1",)},
        edge_weights={"A1::F3A": 1.0, "A1::F4A": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F1": FlangeHypothesis("F1", "seed", (), (1.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F2": FlangeHypothesis("F2", "seed", (), (0.0, 1.0, 0.0), -50.0, (0.0, 50.0, 0.0), 0.8),
        "F3": FlangeHypothesis("F3", "seed", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F4": FlangeHypothesis("F4", "seed", (), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
    }
    bend = _bend(
        "B1",
        incident_flange_ids=("F1", "F2"),
        atom_ids=("A1",),
        anchor=(0.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        candidate_incident_flange_pairs=(("F1", "F2"), ("F3", "F4")),
    )

    admissible, reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels={"A1": "B1", "F3A": "F3", "F4A": "F4"},
        label_types={"B1": "bend", "F1": "flange", "F2": "flange", "F3": "flange", "F4": "flange"},
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
    )

    assert admissible is True
    assert "unsupported_full_reattachment" not in reasons


def test_bend_component_admissibility_rejects_no_share_pair_far_from_corridor():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 100.0, 100.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1,),
        adjacency={"A1": ("F3A", "F4A"), "F3A": ("A1",), "F4A": ("A1",)},
        edge_weights={"A1::F3A": 1.0, "A1::F4A": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F1": FlangeHypothesis("F1", "seed", (), (1.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F2": FlangeHypothesis("F2", "seed", (), (0.0, 1.0, 0.0), -50.0, (0.0, 50.0, 0.0), 0.8),
        "F3": FlangeHypothesis("F3", "seed", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F4": FlangeHypothesis("F4", "seed", (), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
    }
    bend = _bend(
        "B1",
        incident_flange_ids=("F1", "F2"),
        atom_ids=("A1",),
        anchor=(0.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        candidate_incident_flange_pairs=(("F1", "F2"), ("F3", "F4")),
    )

    admissible, reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels={"A1": "B1", "F3A": "F3", "F4A": "F4"},
        label_types={"B1": "bend", "F1": "flange", "F2": "flange", "F3": "flange", "F4": "flange"},
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
    )

    assert admissible is False
    assert reasons


def test_bend_component_admissibility_single_contact_requires_route_relaxation():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1,),
        adjacency={"A1": ("F3A",), "F3A": ("A1",)},
        edge_weights={"A1::F3A": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F3": FlangeHypothesis("F3", "seed", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F4": FlangeHypothesis("F4", "seed", (), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
    }
    bend = _bend(
        "B1",
        incident_flange_ids=("F3", "F4"),
        atom_ids=("A1",),
        anchor=(0.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        angle_deg=70.0,
        candidate_incident_flange_pairs=(("F3", "F4"),),
    )
    labels = {"A1": "B1", "F3A": "F3"}
    label_types = {"B1": "bend", "F3": "flange", "F4": "flange"}

    strict_admissible, strict_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
    )
    relaxed_admissible, relaxed_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
        allow_geometry_backed_single_contact=True,
    )

    assert strict_admissible is False
    assert "missing_designated_flange_contact" in strict_reasons
    assert relaxed_admissible is True
    assert "missing_designated_flange_contact" not in relaxed_reasons


def test_bend_component_admissibility_allows_corridor_backed_contact_recovery():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    atom2 = SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1, atom2),
        adjacency={"A1": ("A2", "F3A"), "A2": ("A1", "F3A"), "F3A": ("A1", "A2")},
        edge_weights={"A1::A2": 1.0, "A1::F3A": 1.0, "A2::F3A": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F3": FlangeHypothesis("F3", "seed", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F4": FlangeHypothesis("F4", "seed", (), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
    }
    bend = _bend(
        "B1",
        incident_flange_ids=("F3", "F4"),
        atom_ids=("A1",),
        anchor=(0.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        angle_deg=50.0,
        candidate_incident_flange_pairs=(("F3", "F4"),),
    )
    labels = {"A1": "B1", "A2": "B1", "F3A": "F3"}
    label_types = {"B1": "bend", "F3": "flange", "F4": "flange"}

    strict_admissible, strict_reasons = _bend_component_admissibility(
        component=("A1", "A2"),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1, "A2": atom2},
        flange_lookup=flanges,
    )
    recovered_admissible, recovered_reasons = _bend_component_admissibility(
        component=("A1", "A2"),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1, "A2": atom2},
        flange_lookup=flanges,
        allow_corridor_backed_contact_recovery=True,
    )

    assert strict_admissible is False
    assert "missing_designated_flange_contact" in strict_reasons
    assert recovered_admissible is True
    assert "missing_designated_flange_contact" not in recovered_reasons


def test_bend_component_admissibility_can_use_flange_hypothesis_support_contacts():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    support_atom = SurfaceAtom("F4A", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1, support_atom),
        adjacency={"A1": ("F3A", "F4A"), "F3A": ("A1",), "F4A": ("A1",)},
        edge_weights={"A1::F3A": 1.0, "A1::F4A": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F3": FlangeHypothesis("F3", "seed", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F4": FlangeHypothesis("F4", "seed", ("F4A",), (0.0, 1.0, 0.0), 0.0, (1.0, 0.0, 0.0), 0.8),
    }
    bend = _bend(
        "B1",
        incident_flange_ids=("F3", "F4"),
        atom_ids=("A1",),
        anchor=(0.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        angle_deg=70.0,
        candidate_incident_flange_pairs=(("F3", "F4"),),
    )
    labels = {"A1": "B1", "F3A": "F3", "F4A": "residual"}
    label_types = {"B1": "bend", "F3": "flange", "F4": "flange", "residual": "residual"}

    strict_admissible, strict_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1, "F4A": support_atom},
        flange_lookup=flanges,
    )
    support_admissible, support_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1, "F4A": support_atom},
        flange_lookup=flanges,
        allow_hypothesis_support_contacts=True,
    )

    assert strict_admissible is False
    assert "missing_designated_flange_contact" in strict_reasons
    assert support_admissible is True
    assert "missing_designated_flange_contact" not in support_reasons


def test_bend_component_admissibility_imbalanced_contacts_require_route_relaxation():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1,),
        adjacency={"A1": ("F3A", "F4A"), "F3A": ("A1",), "F4A": ("A1",)},
        edge_weights={"A1::F3A": 10.0, "A1::F4A": 0.1},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F3": FlangeHypothesis("F3", "seed", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F4": FlangeHypothesis("F4", "seed", (), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
    }
    bend = _bend(
        "B1",
        incident_flange_ids=("F3", "F4"),
        atom_ids=("A1",),
        anchor=(0.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        angle_deg=70.0,
        candidate_incident_flange_pairs=(("F3", "F4"),),
    )
    labels = {"A1": "B1", "F3A": "F3", "F4A": "F4"}
    label_types = {"B1": "bend", "F3": "flange", "F4": "flange"}

    strict_admissible, strict_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
    )
    relaxed_admissible, relaxed_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
        allow_geometry_backed_single_contact=True,
    )

    assert strict_admissible is False
    assert "contact_imbalance" in strict_reasons
    assert relaxed_admissible is True
    assert "contact_imbalance" not in relaxed_reasons


def test_bend_component_admissibility_can_match_cross_axis_seed_convention():
    atom1 = SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (0.0, 1.0, 0.0), 3.0, 1.0)
    graph = SimpleNamespace(
        atoms=(atom1,),
        adjacency={"A1": ("F3A", "F4A"), "F3A": ("A1",), "F4A": ("A1",)},
        edge_weights={"A1::F3A": 1.0, "A1::F4A": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F3": FlangeHypothesis("F3", "seed", (), (0.0, 0.0, 1.0), 0.0, (0.0, 0.0, 0.0), 0.8),
        "F4": FlangeHypothesis("F4", "seed", (), (0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 0.0), 0.8),
    }
    bend = _bend(
        "B1",
        incident_flange_ids=("F3", "F4"),
        atom_ids=("A1",),
        anchor=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 1.0, 0.0),
        angle_deg=55.0,
        candidate_incident_flange_pairs=(("F3", "F4"),),
    )
    labels = {"A1": "B1", "F3A": "F3", "F4A": "F4"}
    label_types = {"B1": "bend", "F3": "flange", "F4": "flange"}

    strict_admissible, strict_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
    )
    relaxed_admissible, relaxed_reasons = _bend_component_admissibility(
        component=("A1",),
        bend=bend,
        atom_graph=graph,
        labels=labels,
        label_types=label_types,
        atom_lookup={"A1": atom1},
        flange_lookup=flanges,
        allow_cross_axis_attachment_match=True,
    )

    assert strict_admissible is False
    assert "attachment_penalty_too_high" in strict_reasons
    assert relaxed_admissible is True
    assert "attachment_penalty_too_high" not in relaxed_reasons


def test_drop_selected_pair_duplicate_bends_prunes_nearby_tiny_alias():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 10.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 10.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 10.0, 1.0),
        SurfaceAtom("A4", (3, 0, 0), (3,), (5.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
        SurfaceAtom("F1A", (4, 0, 0), (4,), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 10.0, 1.0),
        SurfaceAtom("F2A", (5, 0, 0), (5,), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 10.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={
            "A1": ("A2", "F1A", "F2A"),
            "A2": ("A1", "A3", "F1A", "F2A"),
            "A3": ("A2", "F1A", "F2A"),
            "A4": ("F1A", "F2A"),
            "F1A": ("A1", "A2", "A3", "A4"),
            "F2A": ("A1", "A2", "A3", "A4"),
        },
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = {
        "F1": FlangeHypothesis("F1", "seed", ("F1A",), (0.0, 0.0, 1.0), 0.0, atoms[4].centroid, 0.8),
        "F2": FlangeHypothesis("F2", "seed", ("F2A",), (0.0, 1.0, 0.0), 0.0, atoms[5].centroid, 0.8),
    }
    bend_lookup = {
        "B_BIG": _bend("B_BIG", incident_flange_ids=("F1", "F2"), atom_ids=("A1", "A2", "A3"), anchor=(1.0, 0.0, 0.0), cross_width_mm=8.0, span_mm=20.0),
        "B_TINY": _bend("B_TINY", incident_flange_ids=("F1", "F2"), atom_ids=("A4",), anchor=(5.0, 0.0, 0.0), cross_width_mm=8.0, span_mm=20.0),
    }
    labels = {"A1": "B_BIG", "A2": "B_BIG", "A3": "B_BIG", "A4": "B_TINY", "F1A": "F1", "F2A": "F2"}
    label_types = {"B_BIG": "bend", "B_TINY": "bend", "F1": "flange", "F2": "flange"}
    diagnostics = {}

    next_labels, pruned = _drop_selected_pair_duplicate_bends(
        atom_graph=graph,
        labels=labels,
        bend_lookup=bend_lookup,
        label_types=label_types,
        flange_lookup=flanges,
        diagnostics=diagnostics,
    )

    assert pruned == ["B_TINY"]
    assert next_labels["A4"] == "residual"
    assert diagnostics["selected_pair_duplicate_pruned"][0]["kept"] == "B_BIG"

    skipped_labels, skipped_pruned = _drop_selected_pair_duplicate_bends(
        atom_graph=graph,
        labels=labels,
        bend_lookup=bend_lookup,
        label_types=label_types,
        flange_lookup=flanges,
        min_active_bends=3,
    )

    assert skipped_pruned == []
    assert skipped_labels == labels


def test_drop_selected_pair_duplicate_bends_can_prune_medium_alias_when_route_allows_it():
    bend_atoms = tuple(
        SurfaceAtom(f"A{i}", (i, 0, 0), (i,), (float(i) * 0.25, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 10.0, 1.0)
        for i in range(20)
    )
    alias_atoms = tuple(
        SurfaceAtom(f"M{i}", (100 + i, 0, 0), (100 + i,), (2.0 + float(i) * 0.15, 0.2, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 5.0, 1.0)
        for i in range(8)
    )
    flange_atoms = (
        SurfaceAtom("F1A", (200, 0, 0), (200,), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 10.0, 1.0),
        SurfaceAtom("F2A", (201, 0, 0), (201,), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 10.0, 1.0),
    )
    atoms = bend_atoms + alias_atoms + flange_atoms
    adjacency = {}
    for group in (bend_atoms, alias_atoms):
        ids = [atom.atom_id for atom in group]
        for index, atom_id in enumerate(ids):
            neighbors = {"F1A", "F2A"}
            if index > 0:
                neighbors.add(ids[index - 1])
            if index + 1 < len(ids):
                neighbors.add(ids[index + 1])
            adjacency[atom_id] = tuple(sorted(neighbors))
    adjacency["F1A"] = tuple(atom.atom_id for atom in bend_atoms + alias_atoms)
    adjacency["F2A"] = tuple(atom.atom_id for atom in bend_atoms + alias_atoms)
    graph = SimpleNamespace(atoms=atoms, adjacency=adjacency, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=0.2)
    flanges = {
        "F1": FlangeHypothesis("F1", "seed", ("F1A",), (0.0, 0.0, 1.0), 0.0, flange_atoms[0].centroid, 0.8),
        "F2": FlangeHypothesis("F2", "seed", ("F2A",), (0.0, 1.0, 0.0), 0.0, flange_atoms[1].centroid, 0.8),
    }
    bend_lookup = {
        "B_BIG": _bend("B_BIG", incident_flange_ids=("F1", "F2"), atom_ids=tuple(atom.atom_id for atom in bend_atoms), anchor=(2.0, 0.0, 0.0), cross_width_mm=8.0, span_mm=30.0),
        "B_MED": _bend("B_MED", incident_flange_ids=("F1", "F2"), atom_ids=tuple(atom.atom_id for atom in alias_atoms), anchor=(2.5, 0.2, 0.0), cross_width_mm=8.0, span_mm=30.0),
    }
    labels = {atom.atom_id: "B_BIG" for atom in bend_atoms}
    labels.update({atom.atom_id: "B_MED" for atom in alias_atoms})
    labels.update({"F1A": "F1", "F2A": "F2"})
    label_types = {"B_BIG": "bend", "B_MED": "bend", "F1": "flange", "F2": "flange"}

    default_labels, default_pruned = _drop_selected_pair_duplicate_bends(
        atom_graph=graph,
        labels=labels,
        bend_lookup=bend_lookup,
        label_types=label_types,
        flange_lookup=flanges,
    )

    assert default_pruned == []
    assert default_labels == labels

    route_labels, route_pruned = _drop_selected_pair_duplicate_bends(
        atom_graph=graph,
        labels=labels,
        bend_lookup=bend_lookup,
        label_types=label_types,
        flange_lookup=flanges,
        alias_atom_count_max=15,
    )

    assert route_pruned == ["B_MED"]
    assert all(route_labels[atom.atom_id] == "residual" for atom in alias_atoms)

    default_penalty, default_diag = _final_duplicate_cluster_penalty(
        atom_graph=graph,
        labels=labels,
        bend_lookup=bend_lookup,
        label_types=label_types,
        flange_lookup=flanges,
    )
    route_penalty, route_diag = _final_duplicate_cluster_penalty(
        atom_graph=graph,
        labels=labels,
        bend_lookup=bend_lookup,
        label_types=label_types,
        flange_lookup=flanges,
        alias_atom_count_max=15,
        same_pair_penalty=140.0,
    )

    assert default_penalty == 0.0
    assert default_diag["clusters"] == []
    assert route_penalty == 140.0
    assert route_diag["clusters"][0]["reason"] == "selected_pair_duplicate_cluster"


def test_activate_interface_bends_can_promote_atoms_from_flange_labels():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={
            "A1": ("A2",),
            "A2": ("A1", "A3", "N1", "N2"),
            "A3": ("A2",),
            "N1": ("A2",),
            "N2": ("A2",),
        },
        edge_weights={"A2::N1": 1.0, "A2::N2": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = [
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A3",), (0.0, 1.0, 0.0), 0.0, atoms[2].centroid, 0.8),
    ]
    bend = _bend("B1", atom_ids=("A2",), anchor=(1.0, 0.0, 0.0))
    next_labels, activated = _activate_interface_bends(
        atom_graph=graph,
        labels={"A1": "F1", "A2": "F1", "A3": "F2", "N1": "F1", "N2": "F2"},
        flange_hypotheses=flanges,
        bend_hypotheses=[bend],
        label_types={"F1": "flange", "F2": "flange", "B1": "bend", "residual": "residual", "outlier": "outlier"},
    )

    assert "B1" in activated
    assert next_labels["A2"] == "B1"


def test_activate_interface_bends_uses_latent_candidate_attachment():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 3.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={
            "A1": ("A2",),
            "A2": ("A1", "A3", "N1", "N3"),
            "A3": ("A2",),
            "N1": ("A2",),
            "N3": ("A2",),
        },
        edge_weights={"A2::N1": 1.0, "A2::N3": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = [
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A3",), (0.0, 0.0, 1.0), 0.0, atoms[2].centroid, 0.8),
        FlangeHypothesis("F3", "seed", ("A3",), (0.0, 1.0, 0.0), 0.0, atoms[2].centroid, 0.8),
    ]
    bend = _bend(
        "B1",
        incident_flange_ids=("F1", "F2"),
        atom_ids=("A2",),
        anchor=(1.0, 0.0, 0.0),
        candidate_incident_flange_pairs=(("F1", "F2"), ("F1", "F3")),
    )
    next_labels, activated = _activate_interface_bends(
        atom_graph=graph,
        labels={"A1": "F1", "A2": "residual", "A3": "F3", "N1": "F1", "N3": "F3"},
        flange_hypotheses=flanges,
        bend_hypotheses=[bend],
        label_types={"F1": "flange", "F2": "flange", "F3": "flange", "B1": "bend", "residual": "residual", "outlier": "outlier"},
    )

    assert "B1" in activated
    assert next_labels["A2"] == "B1"


def test_activate_interface_bends_can_steal_atoms_from_other_bend_labels():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={
            "A1": ("A2",),
            "A2": ("A1", "A3", "N1", "N2"),
            "A3": ("A2",),
            "N1": ("A2",),
            "N2": ("A2",),
        },
        edge_weights={"A2::N1": 1.0, "A2::N2": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = [
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A3",), (0.0, 1.0, 0.0), 0.0, atoms[2].centroid, 0.8),
    ]
    incumbent = _bend(
        "B0",
        incident_flange_ids=("F1", "F9"),
        atom_ids=("A2",),
        canonical_bend_key="incumbent",
        anchor=(20.0, 0.0, 0.0),
        axis_direction=(0.0, 1.0, 0.0),
        angle_deg=150.0,
        cross_width_mm=0.5,
    )
    bend = _bend("B1", atom_ids=("A2",), anchor=(1.0, 0.0, 0.0))
    next_labels, activated = _activate_interface_bends(
        atom_graph=graph,
        labels={"A1": "F1", "A2": "B0", "A3": "F2", "N1": "F1", "N2": "F2"},
        flange_hypotheses=flanges,
        bend_hypotheses=[incumbent, bend],
        label_types={"F1": "flange", "F2": "flange", "B0": "bend", "B1": "bend", "residual": "residual", "outlier": "outlier"},
    )

    assert "B1" in activated
    assert next_labels["A2"] == "B1"


def test_outer_object_penalties_charge_duplicate_and_fragmented_bends():
    graph = SimpleNamespace(
        atoms=(
            SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
            SurfaceAtom("A3", (10, 0, 0), (2,), (10.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
        ),
        adjacency={"A1": ("A2",), "A2": ("A1",), "A3": ()},
        voxel_size_mm=1.0,
    )
    bend1 = _bend("B1", atom_ids=("A1", "A2", "A3"), canonical_bend_key="family-1")
    bend2 = _bend("B2", atom_ids=("A1",), canonical_bend_key="family-1")
    labels = {"A1": "B1", "A2": "B2", "A3": "B1"}
    label_types = {"B1": "bend", "B2": "bend"}

    assert _duplicate_conflict_penalty(labels, {"B1": bend1, "B2": bend2}, label_types) > 0.0
    assert _fragmentation_penalty(graph, labels, {"B1": bend1, "B2": bend2}, label_types) > 0.0


def test_activate_interface_bends_uses_component_level_interface_contact():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.3, 0.95), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 0.95, 0.3), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={
            "A1": ("A2", "N1"),
            "A2": ("A1", "A3"),
            "A3": ("A2", "N2"),
            "N1": ("A1",),
            "N2": ("A3",),
        },
        edge_weights={"A1::N1": 1.0, "A3::N2": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = [
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A3",), (0.0, 1.0, 0.0), 0.0, atoms[2].centroid, 0.8),
    ]
    bend = _bend("B1", atom_ids=("A1", "A2", "A3"), anchor=(1.0, 0.0, 0.0), span_mm=3.0)
    next_labels, activated = _activate_interface_bends(
        atom_graph=graph,
        labels={"A1": "F1", "A2": "residual", "A3": "F2", "N1": "F1", "N2": "F2"},
        flange_hypotheses=flanges,
        bend_hypotheses=[bend],
        label_types={"F1": "flange", "F2": "flange", "B1": "bend", "residual": "residual", "outlier": "outlier"},
    )

    assert "B1" in activated
    assert all(next_labels[atom_id] == "B1" for atom_id in ("A1", "A2", "A3"))


def test_generate_interface_birth_hypotheses_creates_transition_bend_at_clean_interface():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.03, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 0.8, 0.6), 0.2, 0.03, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A4", (3, 0, 0), (3,), (3.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={"A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2", "A4"), "A4": ("A3",)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = [
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A4",), (0.0, 1.0, 0.0), 0.0, atoms[3].centroid, 0.8),
    ]

    births = generate_interface_birth_hypotheses(
        atom_graph=graph,
        flange_hypotheses=flanges,
        existing_bend_hypotheses=[],
    )

    assert births
    assert births[0].source_kind == "interface_birth"
    assert births[0].bend_class == "transition_only"
    assert births[0].candidate_incident_flange_pairs == (("F1", "F2"),)


def test_generate_interface_birth_hypotheses_can_split_long_interface_into_subspans():
    atoms = tuple(
        SurfaceAtom(
            f"A{index}",
            (index, 0, 0),
            (index,),
            (float(index) * 10.0, 0.0, 0.0),
            (0.0, 0.0, 1.0) if index == 0 else (0.0, 1.0, 0.0) if index == 10 else (0.0, 0.7, 0.7),
            1.0,
            0.01 if index in {0, 10} else 0.03,
            (1.0, 0.0, 0.0),
            2.0,
            1.0,
        )
        for index in range(21)
    )
    adjacency = {
        f"A{index}": tuple(
            f"A{neighbor}"
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(atoms)
        )
        for index in range(len(atoms))
    }
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency=adjacency,
        edge_weights={},
        voxel_size_mm=3.0,
        local_spacing_mm=1.0,
    )
    flanges = [
        FlangeHypothesis("F1", "seed", ("A0",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A20",), (0.0, 1.0, 0.0), 0.0, atoms[-1].centroid, 0.8),
    ]

    compact_births = generate_interface_birth_hypotheses(
        atom_graph=graph,
        flange_hypotheses=flanges,
        existing_bend_hypotheses=[],
        enable_subspan_births=False,
    )
    subspan_births = generate_interface_birth_hypotheses(
        atom_graph=graph,
        flange_hypotheses=flanges,
        existing_bend_hypotheses=[],
        enable_subspan_births=True,
    )

    assert len(compact_births) == 1
    assert len(subspan_births) > len(compact_births)
    assert all(birth.source_kind == "interface_birth" for birth in subspan_births)


def test_rescue_interface_birth_supports_can_claim_admissible_inactive_birth():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.03, (1.0, 0.0, 0.0), 3.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 0.8, 0.6), 0.2, 0.03, (1.0, 0.0, 0.0), 3.0, 1.0),
        SurfaceAtom("A4", (3, 0, 0), (3,), (3.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    graph = SimpleNamespace(
        atoms=atoms,
        adjacency={"A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2", "A4"), "A4": ("A3",)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = [
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A4",), (0.0, 1.0, 0.0), 0.0, atoms[3].centroid, 0.8),
    ]
    birth = generate_interface_birth_hypotheses(
        atom_graph=graph,
        flange_hypotheses=flanges,
        existing_bend_hypotheses=[],
    )[0]

    labels, rescued = _rescue_interface_birth_supports(
        atom_graph=graph,
        labels={"A1": "F1", "A2": "residual", "A3": "residual", "A4": "F2"},
        flange_hypotheses=flanges,
        bend_hypotheses=[birth],
        label_types={"F1": "flange", "F2": "flange", birth.bend_id: "bend", "residual": "residual", "outlier": "outlier"},
    )

    assert birth.bend_id in rescued
    assert any(label == birth.bend_id for label in labels.values())


def test_local_replacement_diagnostics_scores_inactive_seed_against_active_birth():
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.03, (1.0, 0.0, 0.0), 3.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 0.8, 0.6), 0.2, 0.03, (1.0, 0.0, 0.0), 3.0, 1.0),
        SurfaceAtom("A4", (3, 0, 0), (3,), (3.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.2, 0.01, (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    graph = SurfaceAtomGraph(
        atoms=atoms,
        adjacency={"A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2", "A4"), "A4": ("A3",)},
        edge_weights={},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    flanges = (
        FlangeHypothesis("F1", "seed", ("A1",), (0.0, 0.0, 1.0), 0.0, atoms[0].centroid, 0.8),
        FlangeHypothesis("F2", "seed", ("A4",), (0.0, 1.0, 0.0), 0.0, atoms[3].centroid, 0.8),
    )
    active_birth = BendHypothesis(
        "BIRTH_F1_F2_1",
        "transition_only",
        "interface_birth",
        ("F1", "F2"),
        ("F1", "F2"),
        "birth::F1,F2::1",
        ("A2", "A3"),
        (1.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        90.0,
        None,
        "internal",
        0.55,
        4.0,
        4.0,
        ("transition_only",),
        tuple(),
        ("F1", "F2"),
        (("F1", "F2"),),
    )
    inactive_seed = BendHypothesis(
        "B_SEED",
        "transition_only",
        "zero_shot_strip_seed",
        ("F1", "F2"),
        ("F1", "F2"),
        "seed::F1,F2",
        ("A2", "A3"),
        (1.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        90.0,
        None,
        "internal",
        0.8,
        4.0,
        4.0,
        ("transition_only",),
        tuple(),
        ("F1", "F2"),
        (("F1", "F2"),),
    )
    label_types = {"F1": "flange", "F2": "flange", active_birth.bend_id: "bend", inactive_seed.bend_id: "bend", "residual": "residual", "outlier": "outlier"}
    labels = {"A1": "F1", "A2": active_birth.bend_id, "A3": active_birth.bend_id, "A4": "F2"}

    diagnostics = _build_local_replacement_diagnostics(
        atom_graph=graph,
        labels=labels,
        flange_hypotheses=flanges,
        bend_hypotheses=(active_birth, inactive_seed),
        label_types=label_types,
    )

    assert diagnostics["active_bend_component_count"] == 1
    assert diagnostics["replacement_candidates"]
    assert diagnostics["replacement_candidates"][0]["candidate_bend_id"] == "B_SEED"
    assert diagnostics["replacement_candidates"][0]["overlap_min_ratio"] == 1.0
    assert "multi_object_candidates" in diagnostics
    assert "local_relabel_candidates" in diagnostics


def test_research_cache_invalidates_when_scan_content_changes(tmp_path):
    scan_path = tmp_path / "scan.ply"
    scan_path.write_bytes(b"first scan payload")
    runtime_config = SimpleNamespace(scan_detector_kwargs={"min_points": 10}, scan_detect_call_kwargs={})
    first_identity = _research_cache_identity(scan_path=str(scan_path), runtime_config=runtime_config)
    _write_cached_research_inputs(
        cache_dir=tmp_path / "cache",
        payload={"_cache_identity": first_identity, "sentinel": "first"},
    )

    cached = _load_cached_research_inputs(tmp_path / "cache", expected_identity=first_identity)
    assert cached is not None
    assert cached["sentinel"] == "first"

    scan_path.write_bytes(b"different scan payload")
    changed_identity = _research_cache_identity(scan_path=str(scan_path), runtime_config=runtime_config)

    assert changed_identity["scan_sha256"] != first_identity["scan_sha256"]
    assert _load_cached_research_inputs(tmp_path / "cache", expected_identity=changed_identity) is None


def test_research_cache_invalidates_when_detector_configuration_changes(tmp_path):
    scan_path = tmp_path / "scan.ply"
    scan_path.write_bytes(b"stable scan payload")
    first = SimpleNamespace(scan_detector_kwargs={"min_points": 10}, scan_detect_call_kwargs={"voxel_size": 2.0})
    second = SimpleNamespace(scan_detector_kwargs={"min_points": 11}, scan_detect_call_kwargs={"voxel_size": 2.0})

    first_identity = _research_cache_identity(scan_path=str(scan_path), runtime_config=first)
    second_identity = _research_cache_identity(scan_path=str(scan_path), runtime_config=second)

    assert first_identity["scan_sha256"] == second_identity["scan_sha256"]
    assert first_identity["runtime_config_sha256"] != second_identity["runtime_config_sha256"]


def test_research_cache_invalidates_when_upstream_semantics_change(tmp_path):
    scan_path = tmp_path / "scan.ply"
    scan_path.write_bytes(b"stable scan payload")
    runtime_config = SimpleNamespace(scan_detector_kwargs={}, scan_detect_call_kwargs={})
    first_identity = _research_cache_identity(
        scan_path=str(scan_path), runtime_config=runtime_config,
        research_semantics_version="semantic-v1",
    )
    second_identity = _research_cache_identity(
        scan_path=str(scan_path), runtime_config=runtime_config,
        research_semantics_version="semantic-v2",
    )
    _write_cached_research_inputs(
        cache_dir=tmp_path / "cache",
        payload={"_cache_identity": first_identity, "sentinel": "old-semantics"},
    )

    assert first_identity["scan_sha256"] == second_identity["scan_sha256"]
    assert first_identity["runtime_config_sha256"] == second_identity["runtime_config_sha256"]
    assert first_identity["research_semantics_version"] != second_identity["research_semantics_version"]
    assert _load_cached_research_inputs(tmp_path / "cache", expected_identity=second_identity) is None


def test_research_cache_uses_compact_bounds_schema_without_source_points(tmp_path):
    scan_path = tmp_path / "scan.ply"
    scan_path.write_bytes(b"stable scan payload")
    runtime_config = SimpleNamespace(scan_detector_kwargs={}, scan_detect_call_kwargs={})
    identity = _research_cache_identity(scan_path=str(scan_path), runtime_config=runtime_config)
    payload = {
        "_cache_identity": identity,
        "bbox_min": [-1.0, -2.0, -3.0],
        "bbox_max": [10.0, 20.0, 30.0],
        "source_point_count": 10_839_000,
        "sentinel": "compact",
    }

    _write_cached_research_inputs(cache_dir=tmp_path / "cache", payload=payload)
    raw_payload = json.loads((tmp_path / "cache" / "research_inputs.json").read_text(encoding="utf-8"))

    assert identity["cache_schema_version"] == 6
    assert identity["research_semantics_version"] == "2026-08-05.2"
    assert "bbox_points" not in raw_payload
    assert raw_payload["bbox_min"] == [-1.0, -2.0, -3.0]
    assert raw_payload["bbox_max"] == [10.0, 20.0, 30.0]
    assert raw_payload["source_point_count"] == 10_839_000
    assert _load_cached_research_inputs(tmp_path / "cache", expected_identity=identity)["sentinel"] == "compact"


def test_research_cache_writer_rejects_legacy_full_point_serialization(tmp_path):
    with pytest.raises(ValueError, match="forbidden field bbox_points"):
        _write_cached_research_inputs(
            cache_dir=tmp_path / "cache",
            payload={"_cache_identity": {}, "bbox_points": [[0.0, 0.0, 0.0]]},
        )


def test_snapshot_renderer_accepts_f1_visual_review_options():
    parameters = inspect.signature(Model3DSnapshotRenderer.render_bend_status_overlay).parameters

    assert "compact_bend_labels" in parameters
    assert "title_override" in parameters


def test_snapshot_reference_mesh_is_bounded_without_changing_source_bbox():
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=300)
    source_bbox = mesh.get_axis_aligned_bounding_box()
    source_center = source_bbox.get_center().copy()
    source_extent = source_bbox.get_extent().copy()

    bounded, stats = _bounded_snapshot_reference_mesh(mesh, triangle_budget=5_000)

    assert stats["source_triangle_count"] > 5_000
    assert stats["render_triangle_count"] <= 5_000
    assert stats["simplified"] is True
    np.testing.assert_allclose(source_center, bounded.get_axis_aligned_bounding_box().get_center(), atol=0.05)
    np.testing.assert_allclose(source_extent, bounded.get_axis_aligned_bounding_box().get_extent(), atol=0.1)
    assert SNAPSHOT_RENDER_TRIANGLE_BUDGET == 120_000


def test_snapshot_renderer_reuses_bounded_reference_mesh_for_all_views(tmp_path, monkeypatch):
    import ai_analyzer as module
    import open3d as o3d

    source_path = tmp_path / "source.stl"
    source_path.write_bytes(b"stable-source-identity")
    source_mesh = o3d.geometry.TriangleMesh.create_box(width=10.0, height=20.0, depth=30.0)
    read_calls = []
    bound_calls = []

    def fake_read_triangle_mesh(path):
        read_calls.append(path)
        return source_mesh

    def fake_bound(mesh):
        bound_calls.append(mesh)
        return mesh, {"render_triangle_count": len(mesh.triangles), "simplified": False}

    renderer = Model3DSnapshotRenderer()
    monkeypatch.setattr(renderer.o3d.io, "read_triangle_mesh", fake_read_triangle_mesh)
    monkeypatch.setattr(module, "_bounded_snapshot_reference_mesh", fake_bound)

    first = renderer._load_bounded_reference_mesh(str(source_path))
    second = renderer._load_bounded_reference_mesh(str(source_path))

    assert first is second
    assert len(read_calls) == 1
    assert len(bound_calls) == 1
