"""Pinned per-scan engine configuration (E3).

Every tunable of the inspection pipeline lives in :class:`InspectConfig`. The
effective configuration is hashed (canonical JSON) into
``result.engine.config_sha256``, so a result always names the exact settings
that produced it (plan §10 amendment 4: one engine config hash must pass every
gate). ``--config CONFIG.json`` overrides individual fields; unknown keys are
refused, so a typo can never silently fall back to a default.

Units are mm and degrees. The groups below follow the pipeline order
(ingest -> registration -> articulated fit -> decode -> metrology -> U ->
validity -> heatmap). Values marked *calibrated* were set on the E2 DEV seed
set only (never QUAL; see docs/production/ENGINE_INSPECT.md §6 and §9).

Every value is checked on construction (:meth:`InspectConfig.__post_init__`):
finite, inside its documented range, integers where counts are expected, and
the cross-field orders (``voxel_min_mm <= voxel_max_mm``, ``stations_min <=
stations_max``, ``flat_fraction_max < formed_fraction_min`` ...). A bad
``--config`` is therefore a ``CONFIG_INVALID`` usage error, never a NaN in a
hash or a division by zero in the middle of a measurement.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from contracts.production.canonical import sha256_hex

__all__ = ["ENGINE_VERSION", "ALGORITHM_VERSIONS", "InspectConfig", "ConfigError", "load_config"]

#: Version of the per-scan engine (``result.engine.engine_version``).
ENGINE_VERSION = "e3-1.14.0"

#: Algorithm identifiers recorded in ``result.engine.algorithm_versions``.
ALGORITHM_VERSIONS: dict[str, str] = {
    "ingest": "ply_stl_explicit_unit_dense_cell_thin_density_guard_v3",
    "registration": "datum_plane_outline_fft_wall_edge_hole_polygon_flange_ends_hole_pairs_weak_dirs_walk_distinct_inscribed_hole_fill_hardware_mirror_symmetric_convert_free_datum_plane_seeds_symmetric_images_flange_end_rank_tilt_wells_v7",
    "articulated_fit": "hinge_hough_region_grow_skin_levels_arc_fold_displaced_fold_window_short_child_window_support_tabs_one_skin_pose_on_plane_joggle_step_one_skin_reread_astray_wide_reversed_mirrored_hinge_v8",
    "thickness": "datum_tile_two_skin_separation_v1",
    "state_decode": "signed_dihedral_fraction_order_prior_symmetric_near_twins_hole_evidence_twin_agreement_v5",
    "metrology": "station_oml_section_crease_aware_unfolded_island_station_profile_short_side_consistency_footprint_edge_riser_model_reseat_signed_angle_reversed_position_mirrored_flat_start_flange_na_at_stage_pose_borne_lengths_one_skin_swap_branch_withhold_pose_alternative_withhold_v13",
    "uncertainty": "station_mad_residual_pose_erosion_bias_thickness_station_spread_riser_length_v4",
    "heatmap": "stage_cloud_nearest_signed_v1",
}


class ConfigError(ValueError):
    """An invalid ``--config`` file (unknown key, wrong type or out-of-range value)."""


@dataclass(frozen=True)
class InspectConfig:
    # -- ingest / unit check ------------------------------------------------------
    #: Scan-extent / CAD-extent ratio outside [1/x, x] (for both S_0 and S_N) is a
    #: suspected unit mismatch (an inch or metre export read as mm, or the reverse).
    unit_ratio_limit: float = 6.0
    #: Robust extent: percentile trimmed from each end of every axis.
    extent_trim_percent: float = 0.5
    #: Minimum points after cleaning before anything is attempted.
    min_scan_points: int = 2000
    #: Gross outliers: points farther than this many robust extents (the trimmed bounding-box
    #: diagonal) from the trimmed box centre are dropped at ingest, like non-finite vertices
    #: (a single 3e38 sentinel vertex would otherwise overflow the integer voxel keys).
    gross_outlier_extent_factor: float = 10.0
    #: e3-1.13 density guard (``engine._density_guard``), fail closed. A cleaned scan above the validated native
    #: envelope -- more than ``density_guard_ppc`` points per occupied ``density_guard_cell_mm`` cell (densest native
    #: scan surveyed: 12.94 at 0.63 mm) or more than ``density_guard_max_points`` points (largest native: 2.40 M) --
    #: is not measured: the result is written at once, every characteristic UNKNOWN, validity UNSUPPORTED_INPUT
    #: with the reason ``DENSITY_ABOVE_VALIDATED_RANGE``; it can never PASS. 0 switches a criterion off.
    density_guard_cell_mm: float = 0.63
    density_guard_ppc: float = 15.0
    density_guard_max_points: int = 3_000_000
    #: e3-1.14 dense ingest (1 on, 0 off = the e3-1.13 guard). A cloud above ``density_guard_ppc`` is thinned to one
    #: actual point per ``density_guard_cell_mm`` cell (the one nearest the cell's mean; ties by coordinates, then
    #: file order) and measured; the max-points criterion of the guard then applies to the thinned cloud.
    dense_thin: int = 1

    # -- clouds ---------------------------------------------------------------------
    #: Registration / segmentation voxel (mm): max(min, sqrt(area / target)).
    voxel_min_mm: float = 0.8
    voxel_max_mm: float = 2.5
    voxel_target_points: int = 220_000
    normal_neighbours: int = 14
    #: Neighbour graph radius in voxels (connectivity for region growing).
    graph_radius_voxels: float = 1.9
    #: A voxel point is planar when its neighbourhood spread along the normal is
    #: below this (mm) when the skins are resolved, and below
    #: ``thin_spread_factor * t + thin_spread_floor_mm`` in *slab mode*: when the
    #: sheet is thinner than ``slab_radius_factor`` neighbourhood radii, both skins
    #: share a neighbourhood and the sheet reads as one slab.
    planar_spread_mm: float = 0.15
    slab_radius_factor: float = 1.2
    #: Normal-estimation voxel for thin sheets: ``thin_normal_voxel_factor * t`` (>= the voxel).
    thin_normal_voxel_factor: float = 1.8
    thin_spread_factor: float = 0.6
    thin_spread_floor_mm: float = 0.08

    # -- coarse registration (stage independent: datum plane + free edges + holes) ----
    plane_normal_bin_deg: float = 4.0
    plane_offset_tol_mm: float = 0.6
    plane_normal_tol_deg: float = 7.0
    max_planes: int = 24
    #: Candidate datum planes: area >= this fraction of the datum skin area.
    datum_min_area_fraction: float = 0.12
    datum_candidates: int = 6
    coarse_pixel_mm: float = 2.0
    coarse_angle_step_deg: float = 3.0
    fine_pixel_mm: float = 1.0
    #: Fine exhaustive in-plane search (holes weighted) around each coarse peak.
    fine_search_mm: float = 20.0
    fine_search_deg: float = 2.0
    fine_search_pixel_mm: float = 0.5
    #: A best angle on the edge of the fine search's window re-centres the window there, at most this many
    #: times, while the score improves (``registration.fine_search``), for the ``fine_walk_candidates``
    #: best-scoring searches that stopped on their window's edge.
    fine_search_walks: int = 2
    fine_walk_candidates: int = 3
    #: Registration kernels penalise a hole's open core where it lies farther than this (mm) from compiled
    #: stage material inside the hole (a stepped or countersunk hole's bore, the gap around a tab flat in its
    #: window): open in every stage, so scan material there is a wrong pose.
    hole_core_clearance_mm: float = 0.75
    #: ... a non-round hole (a window or slot, whose material is a tab cut on its contour: kerf, edge flare and
    #: kink put scan points beside its compiled footprint) keeps at least this clearance.
    window_core_clearance_mm: float = 1.5
    #: Forbidden band outside each free edge (mm) and its penalty weight.
    free_edge_band_mm: float = 8.0
    violation_weight: float = 2.0
    #: Scans denser than this spacing are thinned to one point per cell of this size for the
    #: registration (the measurement always uses every point).
    registration_decimate_mm: float = 0.55
    #: Coarse peaks that get the fine (hole-weighted) search, and the best of those that
    #: get the full edge/hole fit.
    coarse_candidates_fine: int = 14
    #: Datums with 2 to ``hole_pair_max_holes`` round holes also get poses from pairs of their holes found in
    #: the scan (``registration.hole_pair_hypotheses``), at most this many, next to the coarse peaks.
    hole_pair_hypotheses: int = 4
    hole_pair_max_holes: int = 8
    pose_hypotheses: int = 6
    #: Best refined hypotheses that also get an articulated fit (pose ambiguity, selection).
    articulated_hypotheses: int = 3
    #: A datum that fixes none of its in-plane directions, or a symmetric datum that leaves one free
    #: (``engine._free_datum_trigger``): the in-plane pose must come from the flanges, for every hypothesis.
    #: Then every candidate plane's best raster peak (at most ``free_datum_seeds``; no in-plane constraint
    #: only) and the datum's symmetric images of the best ``free_datum_rank_k`` fits are searched too, and
    #: the best ``free_datum_rank_k`` hypotheses get the articulated fit and the flange-ends variant
    #: (``engine._free_datum_rank``). ``free_datum_rank_k`` equals ``pose_hypotheses``: every refined pose gets
    #: its symmetric images. 0 disables it.
    free_datum_rank_k: int = 6
    free_datum_seeds: int = 4

    # -- fine in-plane refinement (edges + holes) -------------------------------------
    #: 2D tile for the local skin assignment of the datum (thin sheets, small tilts).
    skin_tile_mm: float = 12.0
    #: Single visible datum skin: cut-wall points within this band of the free edges and holes
    #: (and at least this many) decide on which side of it the material lies.
    wall_side_band_mm: float = 1.5
    wall_side_min_points: int = 15
    edge_bin_mm: float = 3.0
    edge_band_mm: float = 3.0
    edge_end_trim_mm: float = 2.0
    hole_annulus_mm: float = 3.0
    hole_angular_bins: int = 24
    refine_iterations: int = 6
    #: *calibrated*: expected inward gap of the extreme skin point at a cut edge,
    #: as a fraction of the scan's median nearest-neighbour spacing.
    edge_gap_factor: float = 0.0
    #: A free-edge bin whose cut wall is seen (at least this many points between the skins
    #: near its extreme; holes: per angular bin) is observed at the wall itself. A skin edge
    #: seen without its wall sits at a range-image depth discontinuity, where scanners erode
    #: it inward by an unknown amount: such rows get ``edge_unwalled_weight`` and their
    #: possible bias enters the position U (``u_pose_bias_factor``).
    reg_wall_min_points: int = 3
    reg_wall_min_points_hole: int = 2
    edge_unwalled_weight: float = 0.15
    #: Erosion of unwalled skin edges (mm): this scan's walled-minus-unwalled datum edge
    #: offset, clipped to [min, max] scan spacings; ``default`` spacings when not measurable.
    erosion_min_spacing: float = 0.2
    erosion_max_spacing: float = 1.5
    erosion_default_spacing: float = 0.5
    #: In-plane pose is degenerate when the weakest constrained direction has
    #: fewer than this many effective edge/hole observations.
    degenerate_min_information: float = 3.0
    #: The in-plane refinement moves every direction with at least this many effective observations to its
    #: least-squares optimum (a weakly determined one included: it still counts as undetermined above);
    #: directions with less keep the fine search's pose.
    refine_min_information: float = 1.0
    #: ... or when the in-plane position at the outline (1 sigma) is worse than this (mm).
    degenerate_max_sigma_mm: float = 0.25
    #: A direction the datum leaves undetermined is fixed by the cut ends of fitted flanges reached only
    #: through bends whose axes are parallel to it (``registration.child_edge_rows``; no bend moves
    #: them along it, formed or flat): an edge's outward normal within this angle of that axis,
    lateral_axis_tol_deg: float = 5.0
    #: its rows' mean sensitivity to the free direction(s) at least this (1: it observes them fully),
    lateral_min_sensitivity: float = 0.5
    #: its bins consistent within this MAD-sigma (mm),
    lateral_edge_max_spread_mm: float = 0.25
    #: no used edge's median residual beyond this after the robust solve (mm; the worst is dropped),
    lateral_outlier_mm: float = 0.5
    #: at least this many consistent edges (no single cut feature fixes a direction),
    lateral_min_edges: int = 2
    #: and a frame shift of at most this (mm); otherwise the direction stays undetermined.
    lateral_max_shift_mm: float = 3.0

    # -- articulated fit ------------------------------------------------------------------
    hinge_band_extra_mm: float = 5.0
    theta_min_deg: float = -10.0
    theta_overbend_deg: float = 25.0
    theta_coarse_step_deg: float = 1.0
    hough_bin_mm: float = 0.04
    hough_window_mm: float = 0.12
    #: Line support starts this far (mm) past the arc's tangent zone ``sqrt(2 Ro window)``.
    hough_arc_clearance_mm: float = 0.3
    #: The fine angle search may look this many band radii along the child (sharper peak).
    hough_fine_reach: float = 2.5
    hinge_rho_slack_mm: float = 3.0
    #: The child's line is searched around the fold kinematics' offset at each angle; a bend line placed
    #: ``d`` off moves it by ``d sin(theta)`` only, so the half-width grows from this (mm, at flat) with
    #: ``slack |sin theta|`` up to the slack: a flat continuation must be the parent's own plane, not a
    #: parallel sheet millimetres off it (the far flange of a joggle).
    hinge_rho_slack_flat_mm: float = 0.4
    #: A formed child whose bend line sits beyond ``hinge_rho_slack_mm`` (a backgauge error, a flange
    #: cut long) leaves the narrow search on clutter (the parent's own continuation, the bend arc) at a
    #: flat or partial angle. For bends of at least ``displaced_min_sweep_deg`` the formed-like angles
    #: (half the sweep and up) are then searched again with this IML slack (mm) over the fine reach;
    #: the displaced line wins when its two-skin support is at least ``displaced_support_ratio`` times
    #: the narrow winner's (a distance-to-nominal gate must not drop a defective flange, plan §3.2).
    hinge_rho_slack_wide_mm: float = 10.0
    displaced_support_ratio: float = 1.5
    displaced_min_sweep_deg: float = 45.0
    #: ... and so are they when the narrow line reads formed but more than this (deg) off the nominal sweep: the
    #: narrow window then holds clutter (lines crossing a displaced sheet's two skins obliquely), or the bend is
    #: truly mis-formed, whose own line the wide search finds again and leaves standing.
    displaced_check_dev_deg: float = 8.0
    #: A child carrying a bend parallel to its own within the fine reach (a joggle's riser) is followed by
    #: a parallel sheet about one thickness off the parent: within this angle of flat its line counts
    #: only when its material is continuous from the hinge over the first ``flat_continuity_mm`` (every
    #: ``flat_continuity_step_mm`` sub-band supported; ``articulated._hough``).
    flat_continuity_max_deg: float = 10.0
    flat_continuity_mm: float = 2.0
    flat_continuity_step_mm: float = 0.5
    #: ... and such a short child grows only up to the tangent lines of that parallel bend plus this (mm).
    short_child_margin_mm: float = 1.0
    #: One visible skin: a line whose weaker skin (in both readings, inner or outer) holds at most this
    #: share of the stronger one's support is read as one skin; when the two readings score within
    #: ``one_skin_tie_fraction`` of each other, the one nearer the nominal kinematics is taken.
    one_skin_fraction: float = 0.05
    one_skin_tie_fraction: float = 0.1
    #: A formed-like line (at least this angle, deg) read on one skin in its window is read the other way too, its
    #: visible skin taken for the other skin even when that puts the pair up to one thickness past the window: the
    #: pair stands when its partner skin holds at least this share of the visible skin's support (a flange whose
    #: hinge frame sits a few millimetres off, its far skin just outside the window; ``articulated._hough``).
    one_skin_reread_min_deg: float = 20.0
    one_skin_reread_min_share: float = 0.3
    #: A fold formed the wrong way is searched about the mirrored hinge (the CAD axis reflected across the parent's
    #: mid-plane, ``articulated._reversed_reading``) on material no fitted panel owns. Its reading replaces the
    #: CAD-direction one only when its support is at least ``reversed_support_ratio`` times the CAD reading's and
    #: it shows a sheet: a level parallel to its visible skin, 0.4 to 1.6 thicknesses away (a thin lip's skins
    #: often read closer than its thickness), holding at least ``reversed_two_skin_share`` of that skin's support.
    reversed_support_ratio: float = 1.5
    reversed_two_skin_share: float = 0.2
    min_child_points: int = 25
    min_child_axial_coverage: float = 0.12
    grow_plane_tol_mm: float = 0.35
    grow_normal_tol_deg: float = 12.0
    grow_iterations: int = 8
    #: A fitted child's pose is turned about its hinge onto its fitted plane's normal when the two differ by at
    #: most this (deg): the fold kinematics alone take the Hough angle (none for a flat child), and the error
    #: carried down the tree moves every grandchild's hinge frame (``articulated._grow_child``).
    child_orient_max_deg: float = 10.0
    #: ... unless it is a riser: a child whose next parallel bend's tangent lines lie within this (mm) of its hinge.
    child_orient_min_depth_mm: float = 8.0
    #: A joggle: a short child (riser) whose parallel bend turns the sheet back to the parent's orientation.
    #: The far flange beyond the riser is found directly in the parent's hinge frame
    #: (``articulated._joggle_step``), over ``joggle_window_mm`` (along the parent's continuation) starting
    #: ``joggle_window_gap_mm`` past the far bend's child-side tangent lines at the nominal pose: it must lie
    #: within ``joggle_parallel_deg`` of the parent's orientation with at least ``joggle_min_points`` points
    #: on its skin. The fold kinematics turn its measured step into the riser angle that produces it; that
    #: angle decides the riser when the riser's own line or grown plane strays more than
    #: ``joggle_theta_tol_deg`` from it (a riser without flat material, a curved S whose convex skin offers a
    #: tangent line at any angle, or whose voxel normals grow no plane).
    joggle_window_gap_mm: float = 0.5
    joggle_window_mm: float = 6.0
    joggle_parallel_deg: float = 6.0
    joggle_min_points: int = 60
    joggle_theta_tol_deg: float = 8.0
    #: ... and the riser angle taken from the step carries the standard uncertainty of the riser's effective
    #: length (this fraction of it): ``tan(theta) * fraction`` rad joins the angle's U (``metrology``).
    joggle_riser_length_u_fraction: float = 0.1

    # -- state decode (formed fraction = measured sweep / nominal sweep) ---------------------
    flat_fraction_max: float = 0.12
    flat_sweep_max_deg: float = 4.0
    formed_fraction_min: float = 0.82
    #: Band around the thresholds in which the order prior breaks the tie.
    prior_band: float = 0.04
    #: The sweep is signed by the side the child lies on (``geom.signed_sweep_deg``): negative when it turned the
    #: other way about the hinge axis (formed the wrong way). A reading within this margin (deg) of 180 keeps its
    #: positive sign whatever its sense of turn (a hem's 180 deg is no wrong-way fold).
    reversed_wrap_margin_deg: float = 30.0

    # -- metrology -------------------------------------------------------------------------
    stations_min: int = 5
    stations_max: int = 11
    station_target_mm: float = 12.0
    min_visible_length_mm: float = 4.0
    #: Plane fits near a bend use points within this distance of the bend line.
    local_fit_depth_mm: float = 40.0
    #: A bend whose median station sweep differs from its region-wide panel planes' sweep by more than this
    #: (deg) is not measured (AMBIGUOUS_SEGMENTATION): its stations settled on another sheet.
    station_region_max_deg: float = 10.0
    #: A side's region-wide plane is re-seated on its own material in the station strip when that material's
    #: OML level lies more than ``local_reseat_min_mm`` (and at most ``local_reseat_max_mm``) off it, the refit
    #: tilted at most ``local_reseat_max_tilt_deg`` (a long or bowed panel near one of its bends).
    local_reseat_min_mm: float = 0.3
    local_reseat_max_mm: float = 3.0
    local_reseat_max_tilt_deg: float = 6.0
    local_reseat_min_share: float = 0.25
    #: e3-1.13 root fix (1 on, 0 off). A side whose panel region holds one skin may have taken its inner skin for
    #: the OML (or the reverse): when its station strip shows two parallel skins one measured thickness apart and
    #: the plane sits on the inner one (or one thickness outside the outer one), the plane is re-seated on the
    #: outer skin (a shift of +-t); the share guard then counts that two-skin band, not the whole strip.
    one_skin_swap: int = 1
    #: e3-1.13 root fix (1 on, 0 off). A discrete skin / window choice taken within a margin of its alternative
    #: (two candidate levels whose counts are within ``branch_tie_ratio``) is measured both ways; a characteristic
    #: whose value differs between the two carries the difference in its U (|delta| added), so it is decided only
    #: when both readings decide the same, and is REVIEW otherwise.
    branch_withhold: int = 1
    branch_tie_ratio: float = 0.75
    #: e3-1.13 safety net. Extra deterministic 95 % resamples of the scan (fixed seeds) run after the primary
    #: inspection (0 = off). ``resample_check_policy`` 1: a decided characteristic that any resample decides
    #: differently becomes REVIEW (its U widened); 2: U' = sqrt(U^2 + (2 sigma)^2), sigma over the primary and the
    #: resamples. Neither can create a PASS.
    resample_check_k: int = 2
    resample_check_policy: int = 1
    resample_check_fraction: float = 0.95
    #: 1: also write ``resample_check.json`` (primary and per-resample readings) next to the result (evaluation).
    resample_check_record: int = 0
    #: 1: run the resamples in worker processes (one per resample, ``spawn`` start method on every platform, so the
    #: behaviour is the same on Windows) instead of serially. Off by default (repin_2026-10-26/REPIN.md).
    resample_check_parallel: int = 0
    #: e3-1.14 pose wells (1 on, 0 off). The datum plane fit keeps only points within a thin band of the current
    #: plane, so a long datum started a fraction of a degree off can settle in a self-consistent tilted plane (a
    #: local optimum). The chosen proper pose is therefore refined again from tilted starts: about each in-plane
    #: axis through the datum's centroid, tilts that move the datum's far edge by +-``pose_well_edge_mm`` and
    #: +-2x that (at most ``pose_well_max_deg``). Distinct optima ("wells": more than ``pose_well_min_gap_mm``
    #: apart anywhere on the datum outline) whose datum score is within ``pose_well_score_slack`` of the chosen
    #: one (or better) get the articulated fit (at most ``pose_well_articulate_max``). A well whose total beats
    #: the chosen pose by more than ``pose_well_switch_margin`` replaces it; every other well within
    #: ``pose_well_alt_margin`` of the (final) chosen total is an alternative pose: the scan is measured under it
    #: too, and a decided characteristic the alternative does not decide the same way is withheld (REVIEW).
    pose_wells: int = 1
    pose_well_edge_mm: float = 1.5
    pose_well_max_deg: float = 2.0
    pose_well_min_gap_mm: float = 0.5
    pose_well_score_slack: float = 0.1
    pose_well_articulate_max: int = 3
    pose_well_switch_margin: float = 0.03
    pose_well_alt_margin: float = 0.06
    local_fit_min_depth_mm: float = 4.0
    #: Each side's station window starts on that side's own flat material (``metrology._flat_start``): a real inner
    #: radius larger than the CAD's, or a springback transition, keeps bending beyond the CAD tangent line (+0.5 mm),
    #: and that tail, bending toward the bend's centre, tilts the station planes toward an underbent reading. From
    #: the CAD start the window moves out by ``flat_start_step_mm`` until a ``flat_start_test_mm`` strip lies within
    #: ``flat_start_dev_mm`` of the line of up to ``flat_start_ref_mm`` of straight skin starting
    #: ``flat_start_gap_mm`` beyond it (axial chunks of ``flat_start_chunk_mm`` pooled); with no such strip within
    #: ``flat_start_max_shift_mm`` (0 turns the search off) the CAD start stays.
    flat_start_step_mm: float = 0.25
    flat_start_test_mm: float = 0.5
    flat_start_gap_mm: float = 1.0
    flat_start_ref_mm: float = 3.0
    flat_start_dev_mm: float = 0.02
    flat_start_max_shift_mm: float = 4.0
    flat_start_chunk_mm: float = 25.0
    min_station_points: int = 12
    #: A station's skin level may sit this far (mm, at most 0.4 t) from the panel plane.
    station_skin_reach_mm: float = 0.6
    #: A bend's OML sharp line follows its station sharps when they sit farther than this (mm)
    #: from the panel-plane intersection.
    sharp_line_consistency_mm: float = 0.15
    #: Rank (1 = max) of the far-edge order statistic per station.
    far_edge_rank: int = 3
    far_edge_material_slack_mm: float = 0.25
    #: Contiguity gap (in scan spacings) when the far-edge run continues past the grown region.
    far_edge_gap_factor: float = 2.5
    #: A FREE_EDGE far edge reads the panel's grown region only within its fitted outline grown by
    #: this (mm): a region that ran through a touching coplanar panel into another one is cut there
    #: (the run past the region still follows contiguous material, so a long flange stays long).
    free_edge_footprint_margin_mm: float = 5.0
    #: A visible cut wall (points between the skins near the extreme) gives the edge directly.
    wall_min_thickness_mm: float = 0.7
    wall_skin_margin_fraction: float = 0.25
    wall_zone_mm: float = 0.6
    wall_min_points: int = 5
    min_station_fraction: float = 0.6
    #: Backgauge across unformed bends: each island panel is unfolded onto the side panel's plane
    #: about the hinge where their fitted planes meet (a not-yet-formed bend may carry a kink of
    #: up to ``flat_fraction_max`` of its sweep); below this dihedral (deg) the planes are parallel.
    unfold_min_angle_deg: float = 0.05
    #: A far-edge run is truncated (the station is dropped) when at least ``truncation_min_points``
    #: points of the panel's grown region lie more than ``truncation_gap_mm`` beyond its end.
    truncation_gap_mm: float = 2.0
    truncation_min_points: int = 8
    #: A FREE_EDGE far edge without a seen cut wall at this share of its stations or more is a
    #: lower bound of the length (a clamp or an out-of-view edge can end the visible material
    #: short of the edge): a short-side FAIL there is UNKNOWN(EDGE_NOT_SEEN).
    bare_edge_station_share: float = 0.5

    # -- uncertainty (expanded, k = 2) ---------------------------------------------------------
    coverage_factor: float = 2.0
    #: *calibrated* standard-uncertainty floors.
    u_floor_length_mm: float = 0.03
    u_floor_angle_deg: float = 0.03
    u_floor_position_mm: float = 0.04
    #: *calibrated* systematic far-edge localization (fraction of NN spacing).
    u_edge_factor: float = 0.15
    #: *calibrated* plane systematics (multi-setup fusion steps, duplicate skins): a
    #: fitted skin plane tilts by ``u_plane_tilt_factor * rms / extent`` and the OML
    #: sharp moves by ``u_sharp_factor * rms / sin(sweep)``.
    u_plane_tilt_factor: float = 1.0
    u_sharp_factor: float = 1.0
    u_plane_rms_floor_mm: float = 0.02
    #: *calibrated* standard uncertainty of a scanned skin's level relative to the datum frame:
    #: the multi-setup fusion misregistration (the model card's "duplicate-skin offset"; the two
    #: skins of one panel often come from different setups). It joins the plane residual in every
    #: OML sharp's displacement (``hypot(rms, u_skin_level) / sin(sweep)``). Sized on the E2 DEV
    #: regression tier (ASSUMED scanner model: the datum's measured two-skin separation scatters
    #: by 0.036 mm around the CAD thickness); re-derive it from the amendment-4b model card.
    u_skin_level_mm: float = 0.015
    u_pose_factor: float = 1.0
    #: *calibrated* bias bounds as standard uncertainties (a bound ``b`` of a uniform error
    #: is ``b / sqrt(3)``): the datum pose's worst-case shift from eroded unwalled rows, and
    #: the erosion of flange / backgauge far edges seen without their cut wall.
    u_pose_bias_factor: float = 0.577
    u_erosion_factor: float = 0.577
    #: *calibrated* single visible skin: standard uncertainty of "the other skin is t away"
    #: (skin level and offset systematics), as a fraction of t. It is combined with the
    #: uncertainty of the t used: the scan-measured t (``thickness_*``), else the compiled t
    #: with ``u_thickness_unmeasured_fraction`` (sheet stock tolerance). Every one-skin offset
    #: enters U through its geometry: ``u_t / sin(sweep)`` along the other flange.
    u_thickness_fraction: float = 0.02
    #: Standard uncertainty (fraction of t) of the compiled t when the scan shows no datum area
    #: with both skins: a uniform +-7 % stock tolerance (EN 10131 / EN 485-4 class) is 0.04.
    u_thickness_unmeasured_fraction: float = 0.04

    # -- measured sheet thickness ------------------------------------------------------
    #: The datum's two skin levels, per ``skin_tile_mm`` tile: the second level is searched
    #: within +-``thickness_search_fraction`` of the compiled t (a sheet off by 10 % is still
    #: found), and needs ``thickness_min_tiles`` tiles with both skins.
    thickness_search_fraction: float = 0.25
    thickness_min_tiles: int = 5
    #: Standard uncertainty floor of a measured t (mm): skin-level systematics of the scanner.
    thickness_floor_mm: float = 0.01
    #: A measured t further than this fraction from the compiled t is reported as a warning.
    thickness_warn_fraction: float = 0.08

    # -- validity -----------------------------------------------------------------------------
    min_datum_coverage: float = 0.15
    #: Share of scan voxels within ``explained_tol_mm`` of the articulated model below
    #: which the part is suspect (wrong part / gross damage).
    wrong_part_explained_min: float = 0.6
    explained_tol_mm: float = 1.0
    #: Tight explained share (``fraction * t`` within [min, max] mm) and its weight in the pose score.
    explained_tight_fraction: float = 0.35
    explained_tight_min_mm: float = 0.25
    explained_tight_max_mm: float = 0.5
    explained_tight_weight: float = 0.5
    wrong_part_hole_mismatch_min: int = 1
    #: Mirror hypothesis must beat the proper one by this score margin.
    mirror_margin: float = 0.08
    #: The mirrored scan is registered too when the best proper pose looks suspicious.
    mirror_check_explained: float = 0.85
    mirror_check_score: float = 0.9
    #: Pose hypotheses within this score margin are ambiguous (if E1's symmetry test agrees).
    ambiguity_margin: float = 0.03
    symmetry_match_deg: float = 6.0
    symmetry_match_mm: float = 6.0
    #: A rival pose whose motion matches a rotation E1 found within this RMS (mm; E1 certifies a
    #: symmetry below 0.3 mm) is a near twin: within ``ambiguity_margin`` of the best total it joins
    #: the order-prior choice (never the ambiguity decision).
    near_twin_rms_mm: float = 1.0
    #: Certified symmetric twins that tie (``pose_ambiguous``): 1 publishes the readings identical under every
    #: tied twin (contract amendment ENGINE_IO-A2, ``engine._twin_agreement``), 0 measures nothing (before it).
    twin_agreement: int = 1
    #: Twins and near twins within ``ambiguity_margin`` are first told apart by the holes of every fitted panel
    #: (``engine._hole_evidence``): a hole is confirmed when its core (the loop shrunk by
    #: ``max(hole_evidence_margin_mm, 2 spacings)``) holds at most ``hole_evidence_open_fraction`` of the
    #: material its 3 mm annulus' density predicts, contradicted at ``hole_evidence_filled_fraction`` or more
    #: (material owned by another fitted panel, a tab still flat in its window, does not count); holes whose
    #: annulus shows less than ``hole_evidence_min_density`` points per mm2 or whose core would hold fewer
    #: than ``hole_evidence_min_expected`` points are not judged. A rival whose (confirmed - contradicted)
    #: exceeds the best total's by ``hole_evidence_min_diff`` or more is chosen (a symmetric twin maps the
    #: datum onto itself, but a hole the other image puts on solid sheet is evidence the totals cannot weigh).
    hole_evidence_margin_mm: float = 0.75
    hole_evidence_open_fraction: float = 0.1
    hole_evidence_filled_fraction: float = 0.4
    hole_evidence_min_density: float = 1.0
    hole_evidence_min_expected: float = 12.0
    hole_evidence_min_diff: int = 2

    # -- unplanned creases -------------------------------------------------------------------------
    crease_min_area_mm2: float = 60.0
    crease_min_angle_deg: float = 6.0
    #: How far from a crease region to look for the fitted panel it hinges on (mm).
    crease_hinge_search_mm: float = 6.0
    #: A crease takes its segment out of the panel's slab: the 90th-percentile height outside
    #: the slab must reach this (mm, at least t / 2).
    crease_min_departure_mm: float = 1.0

    # -- heatmap -------------------------------------------------------------------------------------
    heatmap_max_distance_mm: float = 6.0
    heatmap_max_lateral_mm: float = 1.5

    # -- runtime -------------------------------------------------------------------------------------
    time_budget_s: float = 170.0

    def __post_init__(self) -> None:
        problems = _range_problems(self)
        if problems:
            raise ConfigError("invalid config: " + "; ".join(problems))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def neighbourhood_radius(self, voxel: float) -> float:
        """Radius (mm) of a ``normal_neighbours`` PCA neighbourhood on a voxel cloud."""
        return voxel * math.sqrt(self.normal_neighbours / math.pi)

    def slab_mode(self, t: float, voxel: float) -> bool:
        return t < self.slab_radius_factor * self.neighbourhood_radius(voxel)

    def planar_spread(self, t: float, slab: bool) -> float:
        """Planarity threshold (mm) for a sheet of thickness ``t``."""
        if not slab:
            return self.planar_spread_mm
        return self.thin_spread_factor * t + self.thin_spread_floor_mm

    def sha256(self) -> str:
        """Canonical sha256 of the effective configuration (``engine.config_sha256``)."""
        return sha256_hex({"engine_version": ENGINE_VERSION, "config": self.to_json()})


#: Allowed range per field: (low, high, low_open, high_open); ``None`` is unbounded. Fields not
#: listed must be finite and >= 0. Integer fields (int defaults) must also be whole numbers.
_POS = (0.0, None, True, False)
_FRAC = (0.0, 1.0, False, False)
_RANGES: dict[str, tuple[float | None, float | None, bool, bool]] = {
    "unit_ratio_limit": (1.0, None, True, False), "extent_trim_percent": (0.0, 25.0, False, True),
    "min_scan_points": (10, None, False, False), "gross_outlier_extent_factor": (1.0, None, True, False),
    "density_guard_cell_mm": _POS, "density_guard_ppc": (0.0, None, False, False),
    "density_guard_max_points": (0, None, False, False), "dense_thin": (0, 1, False, False),
    "pose_wells": (0, 1, False, False), "pose_well_edge_mm": _POS, "pose_well_max_deg": (0.0, 10.0, True, False),
    "pose_well_min_gap_mm": _POS, "pose_well_score_slack": (0.0, 1.0, False, False),
    "pose_well_articulate_max": (0, 16, False, False), "pose_well_switch_margin": (0.0, 1.0, False, False),
    "pose_well_alt_margin": (0.0, 1.0, False, False),
    "voxel_min_mm": _POS, "voxel_max_mm": _POS, "voxel_target_points": (1000, None, False, False),
    "normal_neighbours": (4, None, False, False), "graph_radius_voxels": (1.0, None, False, False),
    "planar_spread_mm": _POS, "slab_radius_factor": _POS, "thin_normal_voxel_factor": _POS, "thin_spread_factor": _POS,
    "plane_normal_bin_deg": (0.0, 45.0, True, False), "plane_offset_tol_mm": _POS,
    "plane_normal_tol_deg": (0.0, 90.0, True, True), "max_planes": (1, None, False, False),
    "datum_min_area_fraction": (0.0, 1.0, True, False), "datum_candidates": (1, None, False, False),
    "coarse_pixel_mm": _POS, "coarse_angle_step_deg": (0.0, 90.0, True, False), "fine_pixel_mm": _POS,
    "fine_search_pixel_mm": _POS, "free_edge_band_mm": _POS, "registration_decimate_mm": _POS,
    "fine_search_walks": (0, 10, False, False), "fine_walk_candidates": (0, 64, False, False),
    "hole_core_clearance_mm": _POS, "window_core_clearance_mm": _POS,
    "coarse_candidates_fine": (1, None, False, False), "pose_hypotheses": (1, None, False, False),
    "hole_pair_hypotheses": (0, 32, False, False), "hole_pair_max_holes": (2, 64, False, False),
    "articulated_hypotheses": (1, None, False, False), "skin_tile_mm": _POS, "wall_side_band_mm": _POS,
    "free_datum_rank_k": (0, 16, False, False), "free_datum_seeds": (0, 64, False, False),
    "wall_side_min_points": (1, None, False, False), "edge_bin_mm": _POS, "edge_band_mm": _POS,
    "hole_annulus_mm": _POS, "hole_angular_bins": (8, None, False, False), "refine_iterations": (1, None, False, False),
    "edge_gap_factor": (-2.0, 2.0, False, False), "reg_wall_min_points": (1, None, False, False),
    "reg_wall_min_points_hole": (1, None, False, False), "edge_unwalled_weight": (0.0, 1.0, True, False),
    "degenerate_min_information": _POS, "degenerate_max_sigma_mm": _POS, "refine_min_information": _POS,
    "lateral_axis_tol_deg": (0.0, 45.0, True, False), "lateral_min_sensitivity": _POS,
    "lateral_edge_max_spread_mm": _POS, "lateral_outlier_mm": _POS, "lateral_min_edges": (1, None, False, False),
    "lateral_max_shift_mm": _POS,
    "theta_min_deg": (-45.0, 0.0, False, False), "theta_coarse_step_deg": (0.0, 10.0, True, False),
    "hough_bin_mm": _POS, "hough_window_mm": _POS, "hough_fine_reach": (1.0, None, False, False),
    "hinge_rho_slack_mm": _POS, "hinge_rho_slack_flat_mm": _POS, "hinge_rho_slack_wide_mm": _POS, "displaced_support_ratio": (1.0, None, False, False),
    "displaced_min_sweep_deg": (0.0, 180.0, False, False), "displaced_check_dev_deg": (0.0, 90.0, True, False),
    "flat_continuity_max_deg": (0.0, 45.0, False, False), "flat_continuity_mm": _POS, "flat_continuity_step_mm": _POS,
    "short_child_margin_mm": _POS, "one_skin_fraction": _FRAC, "one_skin_tie_fraction": _FRAC,
    "one_skin_reread_min_deg": (0.0, 90.0, False, False), "one_skin_reread_min_share": (0.0, 1.0, True, False),
    "reversed_support_ratio": (1.0, None, False, False), "reversed_two_skin_share": (0.0, 1.0, True, False),
    "min_child_points": (1, None, False, False), "min_child_axial_coverage": _FRAC,
    "grow_plane_tol_mm": _POS, "grow_normal_tol_deg": (0.0, 90.0, True, True), "grow_iterations": (1, None, False, False),
    "child_orient_max_deg": (0.0, 25.0, False, False), "child_orient_min_depth_mm": _POS,
    "local_reseat_min_mm": _POS, "local_reseat_max_mm": _POS, "local_reseat_max_tilt_deg": (0.0, 45.0, True, False),
    "local_reseat_min_share": _FRAC,
    "one_skin_swap": (0, 1, False, False), "branch_withhold": (0, 1, False, False), "branch_tie_ratio": _FRAC,
    "resample_check_k": (0, 8, False, False), "resample_check_policy": (1, 2, False, False),
    "resample_check_fraction": (0.5, 1.0, False, True), "resample_check_record": (0, 1, False, False),
    "resample_check_parallel": (0, 1, False, False),
    "joggle_window_gap_mm": (0.0, None, False, False), "joggle_window_mm": _POS,
    "joggle_parallel_deg": (0.0, 45.0, False, False), "joggle_min_points": (1, None, False, False),
    "joggle_theta_tol_deg": (0.0, 90.0, False, False), "joggle_riser_length_u_fraction": _FRAC,
    "flat_fraction_max": (0.0, 1.0, False, True), "formed_fraction_min": (0.0, 1.0, True, False),
    "reversed_wrap_margin_deg": (0.0, 90.0, False, True),
    "stations_min": (1, None, False, False), "stations_max": (1, None, False, False), "station_target_mm": _POS,
    "min_visible_length_mm": _POS, "local_fit_depth_mm": _POS, "local_fit_min_depth_mm": _POS,
    "station_region_max_deg": (0.0, 180.0, True, False),
    "min_station_points": (3, None, False, False), "station_skin_reach_mm": _POS,
    "flat_start_step_mm": _POS, "flat_start_test_mm": _POS, "flat_start_gap_mm": (0.0, None, False, False),
    "flat_start_ref_mm": _POS, "flat_start_dev_mm": _POS,
    "flat_start_max_shift_mm": (0.0, None, False, False), "flat_start_chunk_mm": _POS,
    "far_edge_rank": (1, None, False, False), "wall_skin_margin_fraction": (0.0, 0.5, False, True),
    "free_edge_footprint_margin_mm": _POS,
    "wall_zone_mm": _POS, "wall_min_points": (1, None, False, False), "min_station_fraction": (0.0, 1.0, True, False),
    "unfold_min_angle_deg": _POS, "truncation_gap_mm": _POS, "truncation_min_points": (1, None, False, False),
    "bare_edge_station_share": (0.0, 1.0, True, False),
    "coverage_factor": (1.0, None, False, False), "u_floor_length_mm": _POS, "u_floor_angle_deg": _POS,
    "u_floor_position_mm": _POS, "thickness_search_fraction": (0.0, 0.5, True, True),
    "thickness_min_tiles": (1, None, False, False), "thickness_warn_fraction": _POS,
    "min_datum_coverage": _FRAC, "wrong_part_explained_min": _FRAC, "explained_tol_mm": _POS,
    "explained_tight_fraction": _POS, "explained_tight_min_mm": _POS, "explained_tight_max_mm": _POS,
    "wrong_part_hole_mismatch_min": (1, None, False, False), "mirror_check_explained": _FRAC,
    "mirror_check_score": (None, None, False, False), "symmetry_match_deg": _POS, "symmetry_match_mm": _POS,
    "near_twin_rms_mm": _POS, "twin_agreement": (0, 1, False, False), "hole_evidence_margin_mm": _POS, "hole_evidence_open_fraction": _FRAC,
    "hole_evidence_filled_fraction": _FRAC, "hole_evidence_min_density": _POS, "hole_evidence_min_expected": _POS,
    "hole_evidence_min_diff": (1, None, False, False),
    "crease_min_area_mm2": _POS, "crease_min_angle_deg": _POS, "crease_hinge_search_mm": _POS,
    "heatmap_max_distance_mm": _POS, "heatmap_max_lateral_mm": _POS, "time_budget_s": _POS,
}


def _range_problems(cfg: "InspectConfig") -> list[str]:
    """Every value that is not finite, out of its range or not whole where a count is expected."""
    out: list[str] = []
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out.append(f"{f.name} must be a number")
            continue
        if not math.isfinite(float(v)):
            out.append(f"{f.name} must be finite")
            continue
        if isinstance(f.default, int) and not isinstance(f.default, bool) and float(v) != int(v):
            out.append(f"{f.name} must be a whole number")
        lo, hi, lo_open, hi_open = _RANGES.get(f.name, (0.0, None, False, False))
        if lo is not None and (v < lo or (lo_open and v == lo)):
            out.append(f"{f.name}={v} must be {'>' if lo_open else '>='} {lo}")
        if hi is not None and (v > hi or (hi_open and v == hi)):
            out.append(f"{f.name}={v} must be {'<' if hi_open else '<='} {hi}")
    if not out:
        pairs = (("voxel_min_mm", "voxel_max_mm"), ("stations_min", "stations_max"),
                 ("explained_tight_min_mm", "explained_tight_max_mm"), ("erosion_min_spacing", "erosion_max_spacing"),
                 ("local_fit_min_depth_mm", "local_fit_depth_mm"), ("hinge_rho_slack_mm", "hinge_rho_slack_wide_mm"),
                 ("hinge_rho_slack_flat_mm", "hinge_rho_slack_mm"))
        for a, b in pairs:
            if getattr(cfg, a) > getattr(cfg, b):
                out.append(f"{a} must be <= {b}")
        if cfg.flat_fraction_max + 2 * cfg.prior_band >= cfg.formed_fraction_min:
            out.append("flat_fraction_max + 2 prior_band must be < formed_fraction_min")
    return out


def load_config(path: str | Path | None, **overrides: Any) -> InspectConfig:
    """Defaults, then the JSON file at ``path`` (if any), then ``overrides``.

    Raises :class:`ConfigError` for unknown keys, non-numeric values where a
    number is expected, and any value outside its range (``__post_init__``).
    """
    data: dict[str, Any] = {}
    if path is not None:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read config {Path(path).name}: {exc}") from None
        if not isinstance(raw, dict):
            raise ConfigError("config must be a JSON object")
        data.update(raw)
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name: f for f in fields(InspectConfig)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"unknown config keys: {', '.join(unknown)}")
    defaults = InspectConfig()
    clean: dict[str, Any] = {}
    for key, value in data.items():
        default = getattr(defaults, key)
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            raise ConfigError(f"config key {key} is not overridable")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config key {key} must be a number")
        if not math.isfinite(float(value)):
            raise ConfigError(f"config key {key} must be finite")
        if isinstance(default, int) and float(value) != int(value):
            raise ConfigError(f"config key {key} must be a whole number")
        clean[key] = int(value) if isinstance(default, int) else float(value)
    return InspectConfig(**{**defaults.to_json(), **clean})
