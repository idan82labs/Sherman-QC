import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
    _connected_components,
    _duplicate_conflict_penalty,
    _drop_selected_pair_duplicate_bends,
    _final_duplicate_cluster_penalty,
    _fragmentation_penalty,
    _incidence_penalty,
    _normalize_bend_hypotheses,
    _prune_and_refit_labels,
    _rescue_interface_birth_supports,
    _render_region_suppression_report,
    _suppress_same_pair_marker_aliases,
    _suppress_tiny_cross_pair_marker_clusters,
    _typed_adjacency_penalty,
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
    atoms = (
        SurfaceAtom("A1", (0, 0, 0), (0,), (0.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
        SurfaceAtom("A2", (1, 0, 0), (1,), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
        SurfaceAtom("A3", (2, 0, 0), (2,), (2.0, 0.0, 0.0), (0.0, 0.7, 0.7), 0.2, 0.02, (1.0, 0.0, 0.0), 1.0, 1.0),
    )
    graph = SurfaceAtomGraph(
        atoms=atoms,
        adjacency={"A1": ("A2",), "A2": ("A1", "A3"), "A3": ("A2",)},
        edge_weights={"A1|A2": 1.0, "A2|A3": 1.0},
        voxel_size_mm=1.0,
        local_spacing_mm=0.2,
    )
    region = OwnedBendRegion(
        bend_id="OB1",
        bend_class="transition_only",
        incident_flange_ids=("F1", "F2"),
        canonical_bend_key="F1,F2::B1",
        owned_atom_ids=("A1", "A2", "A3"),
        anchor=(1.0, 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(1.0, 0.0, 0.0),
        span_endpoints=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        angle_deg=90.0,
        radius_mm=None,
        visibility_state="internal",
        support_mass=3.0,
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
            atom_labels={"A1": "B1", "A2": "B1", "A3": "B1"},
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
