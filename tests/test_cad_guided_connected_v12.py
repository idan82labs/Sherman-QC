"""UNEXECUTED independent C0/C1/C2 array packet; native/byte adapters not tested.

Construction and truth are independent of inference. The CAD and scan grids,
their diagonals and poses differ. CAD nominal declarations are inputs; measured
quantities, scan role labels, scanner pose and truth endpoints are not. Root
must freeze the complete runner/API/policies before any connected execution.
The original two planar controls remain their exact reviewed source fixtures;
this file owns only the four new curved constructions and their output oracles.
"""

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


EPSILON_MM = .05
ACTUAL_NOISE_CEILING_MM = .002
NOISE_ROUNDING_RESERVE = .999
ARMS = ("C0", "C1", "C2")
CAD_RADIUS_MM = 5.
CAD_SWEEP_DEG = 90.
CAD_LENGTH_MM = 30.
CAD_THICKNESS_MM = 1.
CAD_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
CAD_TRANSLATION = np.array([11., -7., 3.])
ROOT = Path(__file__).resolve().parents[1]
FULL_MATRIX_OPT_IN = "SHERMAN_RUN_CONNECTED_V12"
ARTIFACT_DIRECTORY_ENV = "SHERMAN_CONNECTED_V12_ARTIFACT_DIRECTORY"
RESOURCE_LIMITS = dict(wall_seconds=300, worker_cpu_seconds=240,
                       process_tree_rss_bytes=4*1024**3, stdout_bytes=1024**2)
ARTIFACT_LIMITS = dict(case_arm_count=18, per_case_arm_json_bytes=16*1024**2,
                       total_case_arm_json_bytes=288*1024**2,
                       per_start_json_bytes=4096, total_start_json_bytes=18*4096)
PLANAR_CASES = ("P-clean", "P-noisy")
PLANAR_SOURCE_HASHES = {
    "tests/test_cad_guided_plane_initialization_v11.py": "16a646404b4cf35975956b717a9143b52d8bca1e7ec5467f40096aefbc0edf1e",
    "tests/test_cad_guided_plane_initialization_v10.py": "4882b701af3806fd03ddde9bb0b1af823bd0124e859e6655207f683db1563eb0",
}

# Literal policy values copied from scripts/run_gold11_mesh_development_v3.py
# CONFIG; that program is NEVER imported or executed. The array route does not
# reproduce its bytes/native sampler or its effective 8192 refinement rows.
PHYSICAL_POLICIES = dict(
    registration=dict(coarse_points=2048, refine_points=8192, coarse_iterations=30,
        refine_iterations=30, trim_fraction=.85, huber_delta_mm=.25,
        coarse_distance_fraction=.15, max_correspondence_distance_mm=2.,
        max_inlier_plane_p95_mm=.5, min_scan_coverage=.75, min_cad_coverage=.25,
        min_observability_ratio=.005, pose_class_distance_mm=.1,
        ambiguity_plane_rms_margin_mm=.05, ambiguity_coverage_margin=.03,
        max_rotation_step_radians=.15, max_translation_step_mm=5.),
    measurement=dict(flange_window_mm=[2.5, 12.5], search_margin_mm=3.,
        max_fit_residual_mm=.05, plane_exclusion_mm=.075, max_tangency_error_mm=.1,
        fit_model="free_circle"),
    support=dict(max_axial_gap_mm=4., max_angular_gap_deg=5., min_arc_sweep_deg=20., min_local_points=3),
    noisy_policy=dict(tile_triangles=[128, 1024], halo_rings=1, min_support_rows=3,
        max_normal_angle_radians=math.pi/18., min_inplane_covariance_ratio=1e-8),
    cad_patch_policy=dict(max_pair_normal_sine=1e-8, max_pair_anchored_distance_mm=1e-8,
        max_full_support_distance_mm=1e-8, min_support_vertices=12,
        min_second_extent_mm=.25, min_inplane_covariance_ratio=1e-8),
    search_policy=dict(min_scan_support_count=3, max_scan_support_rms_mm=.1,
        min_normal_condition_ratio=.02, max_normal_alignment_extra_radians=.02,
        max_intersection_backward_residual_mm=1e-8, pair_angle_numeric_guard_radians=1e-10),
)
PHYSICAL_BUDGETS = dict(
    noisy_budgets=dict(max_raw_rows=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_core_records=65_536, max_core_triangle_rows=4096,
        max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
        max_full_support_row_tests=64_000_000, max_cone_row_tests=16_000_000,
        max_closure_triangle_tests=32_000_000, max_closure_edge_tests=96_000_000,
        max_fit_evaluations=65_552, max_closure_schedules=16,
        max_retained_array_bytes=536_870_912, max_metadata_bytes=67_108_864, chunk_rows=65_536),
    cad_patch_budgets=dict(max_vertices=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_components=65_536, max_output_patches=4096,
        max_support_memberships=6_000_000, max_full_vertex_tests=12_000_000, chunk_triangles=65_536),
    search_budgets=dict(max_scan_planes=65_552, max_cad_planes=2048, max_bank_planes=128,
        max_scan_triplets=256, max_scan_partner_tests=1_048_576, max_cad_pair_records=2_096_128,
        max_pair_join_tests=32_000_000, max_cad_pair_samples_per_triplet=12,
        max_unsigned_per_scan_triplet=4, max_compatible_rows_per_triplet=49_104,
        max_unsigned_assignments=1024, max_signed_assignments=8192,
        max_report_bytes=67_108_864, chunk_pair_rows=65_536),
    initialization_budgets=dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
        max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
        max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000, chunk_rows=65_536,
        max_node_visits=512_000_000, max_triangle_pairs=256_000_000, max_normal_pairs=32_000_000,
        max_work_units=4_000_000_000, max_face_build_nodes=131_071,
        max_face_build_triangle_visits=10_000_000, max_refined_start_groups=48,
        max_report_bytes=100_663_296),
    surface_budgets=dict(max_triangles=500_000, max_query_rows=200_000, leaf_size=16,
        max_build_nodes=131_071, max_build_triangle_visits=10_000_000,
        max_node_visits=32_000_000, max_triangle_pairs=32_000_000, max_normal_pairs=32_000_000,
        max_work_units=512_000_000, max_pair_batch=65_536, max_ties_per_query=32),
    registration_budgets=dict(max_query_calls=4096, max_query_rows=64_000_000,
        max_node_visits=2_000_000_000, max_triangle_pairs=2_000_000_000,
        max_normal_pairs=500_000_000, max_work_units=20_000_000_000),
)
CERTIFICATE_CAPS = dict(initialization_budgets=1_024_000_000,
                        surface_budgets=128_000_000, registration_budgets=8_000_000_000)
SURFACE_ROLE_CAPS = dict(max_raw_rows=1_000_000, max_triangles=2_000_000,
    max_proposals=65_552, max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
    max_encoded_metadata_bytes=67_108_864, max_decoded_metadata_bytes=268_435_456,
    max_incidence_corner_visits=6_000_000, chunk_triangles=65_536)


@dataclass(frozen=True)
class CurvedCase:
    case_id: str
    radius_mm: float
    sweep_deg: float
    localized: bool
    noisy: bool
    pose: str


CURVED_CASES = (
    CurvedCase("U-R", 5.30, 90., False, False, "A"),
    CurvedCase("U-A", 5., 90.5, False, True, "B"),
    CurvedCase("L-R", 5.30, 90., True, True, "B"),
    CurvedCase("L-A", 5., 90.5, True, False, "A"),
)
CASE_IDS = PLANAR_CASES + tuple(case.case_id for case in CURVED_CASES)
CASE_ARM_IDS = tuple((case, arm) for case in CASE_IDS for arm in ARMS)
CLEAN_CRITERIA = dict(radius_error_mm=2e-5, absolute_sweep_error_deg=2e-5,
                      generator_endpoint_error_mm=.02, axis_endpoint_error_mm=.02)
NOISY_CRITERIA = dict(radius_error_mm=.02, absolute_sweep_error_deg=.1,
                      generator_endpoint_error_mm=.05, axis_endpoint_error_mm=.05)


def _scanner_pose(name):
    """Quaternion oracle; never supplied as a registration start or datum."""
    if name == "A":
        direction, angle, translation = (1., -2., 3.), .91, (37., -23., 11.)
    else:
        assert name == "B"
        direction, angle, translation = (-2., 3., 1.), -1.17, (-17., 31., 47.)
    axis = np.asarray(direction) / math.sqrt(sum(value*value for value in direction))
    w = math.cos(angle/2)
    x, y, z = axis*math.sin(angle/2)
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return rotation, np.asarray(translation)


def _profile(radius, sweep_deg, first_distances, angles_deg, second_distances):
    """Far first flange -> curved arc -> far second flange; seams welded."""
    phi = math.radians(sweep_deg)
    first = [[radius, -float(distance), 0.] for distance in reversed(first_distances)]
    arc = [[radius*math.cos(math.radians(angle)), radius*math.sin(math.radians(angle)), 0.]
           for angle in angles_deg[1:]]
    last = [[radius*math.cos(phi)-float(distance)*math.sin(phi),
             radius*math.sin(phi)+float(distance)*math.cos(phi), 0.]
            for distance in second_distances[1:]]
    return np.asarray(first+arc+last, dtype=np.float64)


def _quad(rows, a, b, c, d, *, reverse=False, alternate=False):
    """a,b,c,d are a grid quad, not an arbitrary polygon fan across a hole."""
    triangles = ((a, b, d), (a, d, c)) if alternate else ((a, b, c), (c, b, d))
    rows.extend((x, z, y) if reverse else (x, y, z) for x, y, z in triangles)


def _cad_arrays():
    """Independently triangulated complete nominal strip, with CAD face IDs.

    The volumetric structured grid is only construction scratch. Returned
    triangles are exclusively its true boundary, including the concave cap.
    No scan row, triangulation, deformation, pose or truth enters this function.
    """
    first = np.array([0., 2.5, 7.5, 13., 20.])
    second = np.array([0., 3., 8., 15., 22., 29.])
    angles = np.linspace(0., CAD_SWEEP_DEG, 19)
    radii, axial = (5., 5.5, 6.), (0., 7., 19., 30.)
    profiles = [_profile(radius, CAD_SWEEP_DEG, first, angles, second) for radius in radii]
    size = len(profiles[0])
    assert size == 28
    nodes = np.asarray([point+np.array([0., 0., z]) for profile in profiles
                        for point in profile for z in axial])

    def node(radial, profile, station):
        return (radial*size+profile)*len(axial)+station

    triangles, owners = [], []

    def add(a, b, c, d, face, reverse=False):
        _quad(triangles, a, b, c, d, reverse=reverse, alternate=True)
        owners.extend((face, face))

    for radial in (0, 2):
        for profile in range(size-1):
            # The three native-like face roles are assigned by CAD construction,
            # not inferred from the scan or supplied as scan feature labels.
            face = (11 if profile < 4 else 22 if profile < 22 else 33) + 100*radial
            for station in range(len(axial)-1):
                add(node(radial, profile, station), node(radial, profile+1, station),
                    node(radial, profile, station+1), node(radial, profile+1, station+1),
                    face, reverse=radial == 0)
    for station in (0, len(axial)-1):
        for radial in range(2):
            for profile in range(size-1):
                add(node(radial, profile, station), node(radial, profile+1, station),
                    node(radial+1, profile, station), node(radial+1, profile+1, station),
                    44 if station == 0 else 55, reverse=station != 0)
    for profile in (0, size-1):
        for radial in range(2):
            for station in range(len(axial)-1):
                add(node(radial, profile, station), node(radial+1, profile, station),
                    node(radial, profile, station+1), node(radial+1, profile, station+1),
                    66 if profile == 0 else 77, reverse=profile != 0)
    triangle_rows = np.asarray(triangles, dtype=np.int64)
    points = nodes @ CAD_ROTATION.T+CAD_TRANSLATION
    counts, local = {}, []
    for face in owners:
        local.append(counts.get(face, 0))
        counts[face] = counts.get(face, 0)+1
    assert len(triangle_rows) == 564
    return dict(cad_triangles=np.ascontiguousarray(points[triangle_rows]),
                cad_face_ids=np.asarray(owners, dtype=np.int64),
                cad_source_triangle_ids=np.asarray(local, dtype=np.int64))


def _scan_mesh(case, *, alternate=False):
    first = np.linspace(0., 20., 11)
    second = np.r_[np.linspace(0., 28., 15), 29.]
    angles = np.linspace(0., case.sweep_deg, 25)
    axial = np.linspace(0., 30., 11) if case.localized else np.linspace(6., 24., 9)
    profile = _profile(case.radius_mm, case.sweep_deg, first, angles, second)
    assert len(profile) == 50
    points = [point+np.array([0., 0., z]) for point in profile for z in axial]
    triangles = []
    for i in range(len(profile)-1):
        for j in range(len(axial)-1):
            a = i*len(axial)+j
            _quad(triangles, a, a+len(axial), a+1, a+len(axial)+1,
                  reverse=True, alternate=alternate)
    cap_rows = []
    if case.localized:
        # Inner cap boundary reuses the actual sheet rows at z=0. Radial
        # strips remain inside the material, never across the concave hole.
        cap_ids = [np.arange(len(profile), dtype=np.int64)*len(axial)]
        for radius in (case.radius_mm+.5, case.radius_mm+1.):
            cap = _profile(radius, case.sweep_deg, first, angles, second)
            offset = len(points)
            points.extend(cap)
            cap_ids.append(np.arange(offset, offset+len(cap), dtype=np.int64))
        for k in range(2):
            for i in range(len(profile)-1):
                _quad(triangles, cap_ids[k][i], cap_ids[k][i+1], cap_ids[k+1][i], cap_ids[k+1][i+1],
                      alternate=alternate)
        cap_rows = np.concatenate(cap_ids).tolist()
    return np.asarray(points), np.asarray(triangles, dtype=np.int64), cap_rows, axial


def _perturb(raw, ceiling=ACTUAL_NOISE_CEILING_MM):
    # Pattern is defined on original row number alone. It has no CAD, normal,
    # feature, pose, geometric region, measurement or reference lookup.
    rows = np.arange(len(raw), dtype=np.int64)
    direction = np.column_stack((((17*rows+3) % 19-9)/9.,
                                 ((7*rows+5) % 23-11)/11.,
                                 ((11*rows+2) % 29-14)/14.))
    intended = NOISE_ROUNDING_RESERVE*ceiling/math.sqrt(3.)*direction
    perturbed = raw+intended
    return perturbed, perturbed-raw


def curved_packet(case):
    """Separate explicit inference inputs from inaccessible simulator truth."""
    model_points, triangles, cap_rows, axial = _scan_mesh(case)
    rotation, translation = _scanner_pose(case.pose)
    clean = model_points @ rotation.T+translation
    raw, actual_error = _perturb(clean) if case.noisy else (clean.copy(), np.zeros_like(clean))
    inputs = dict(scan_points=raw, scan_triangles=triangles,
                  position_error_bounds_mm=EPSILON_MM, **_cad_arrays())
    # This identifier binds ARRAY BYTES ONLY. It is not a file/capture receipt.
    array_sha = hashlib.sha256(np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
    phi = math.radians(case.sweep_deg)/2
    carrier = np.array([[case.radius_mm*math.cos(phi), case.radius_mm*math.sin(phi), z]
                        for z in (axial[0], axial[-1])])
    axis = np.array([[0., 0., axial[0]], [0., 0., axial[-1]]])
    truth = dict(case=case, model_points=model_points, clean_scan_points=clean,
                 actual_coordinate_errors=actual_error, cap_rows=cap_rows,
                 scanner_rotation=rotation, scanner_translation=translation,
                 raw_generator_endpoints=carrier @ rotation.T+translation,
                 raw_axis_endpoints=axis @ rotation.T+translation,
                 axial_containment_interval_mm=None if case.localized else (-6., 6.),
                 array_only_scan_sha256=array_sha)
    return inputs, truth


def nominal_feature_kwargs():
    """Declared nominal CAD context, NOT native extraction or scan truth."""
    return dict(feature_id="constructed-cad-inner-cylinder", nominal_radius_mm=5.,
                nominal_sweep_deg=90., origin=tuple(CAD_TRANSLATION), axis=tuple(CAD_ROTATION[:, 2]),
                start_radial=tuple(CAD_ROTATION[:, 0]), span_mm=(0., 30.),
                flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.05,
                plane_exclusion_mm=.075, max_tangency_error_mm=.1,
                nominal_source="EXPLICIT_ANALYTIC_ARRAY_FIXTURE_NOT_NATIVE_ADAPTER",
                fit_model="free_circle")


def _endpoint_error(actual, expected):
    actual = np.asarray(actual)
    return min(float(np.linalg.norm(actual-expected, axis=1).max()),
               float(np.linalg.norm(actual[::-1]-expected, axis=1).max()))


def assert_scan_derived_measurement(measurement, truth):
    """Post-inference oracle; no truth is passed into runtime measurement."""
    case = truth["case"]
    limits = NOISY_CRITERIA if case.noisy else CLEAN_CRITERIA
    assert measurement["fit_model"] == "free_circle"
    assert measurement["tangency_check_role"] == "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK"
    radius = float(measurement["measured_radius_mm"])
    sweep = abs(float(measurement["measured_sweep_deg"]))
    assert abs(radius-case.radius_mm) <= limits["radius_error_mm"]
    assert abs(sweep-case.sweep_deg) <= limits["absolute_sweep_error_deg"]
    if case.radius_mm != CAD_RADIUS_MM:
        assert abs(radius-CAD_RADIUS_MM) >= .25, "NOMINAL_RADIUS_SUBSTITUTION"
    if case.sweep_deg != CAD_SWEEP_DEG:
        assert abs(sweep-CAD_SWEEP_DEG) >= .35, "NOMINAL_ANGLE_SUBSTITUTION"
    assert measurement.get("physical_trim_endpoints") is None
    return dict(radius_error_mm=abs(radius-case.radius_mm), absolute_sweep_error_deg=abs(sweep-case.sweep_deg))


def assert_localized_raw_carrier(instance, truth):
    """True sampled support endpoints, never extrapolated physical CAD trims."""
    assert truth["case"].localized
    limits = NOISY_CRITERIA if truth["case"].noisy else CLEAN_CRITERIA
    assert instance["line_convention"] == "MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR"
    assert instance["physical_trim_endpoints"] is None
    assert instance["manufacturing_event_identity"] == "UNKNOWN"
    generator_error = _endpoint_error([instance["line_start"], instance["line_end"]],
                                      truth["raw_generator_endpoints"])
    axis_error = _endpoint_error([instance["axis_support_line_start"], instance["axis_support_line_end"]],
                                 truth["raw_axis_endpoints"])
    assert generator_error <= limits["generator_endpoint_error_mm"]
    assert axis_error <= limits["axis_endpoint_error_mm"]
    return dict(generator_endpoint_error_mm=generator_error, axis_endpoint_error_mm=axis_error)


@pytest.mark.parametrize("case", CURVED_CASES, ids=lambda case: case.case_id)
def test_independent_fixture_sizes_noise_and_real_connected_cap(case):
    inputs, truth = curved_packet(case)
    expected = (650, 1176) if case.localized else (450, 784)
    assert (len(inputs["scan_points"]), len(inputs["scan_triangles"])) == expected
    assert len(inputs["cad_triangles"]) == 564
    assert inputs["position_error_bounds_mm"] == .05
    actual = np.linalg.norm(truth["actual_coordinate_errors"], axis=1)
    assert float(actual.max()) <= (.002 if case.noisy else 0.)
    assert len(np.unique(inputs["scan_triangles"].ravel())) == expected[0]
    assert np.array_equal(inputs["cad_triangles"], _cad_arrays()["cad_triangles"])
    if case.localized:
        cap = truth["model_points"][truth["cap_rows"]]
        assert len(cap) == 150 and np.all(cap[:, 2] == 0.)
        delta = cap[:, :2]-cap[:, :2].mean(axis=0)
        assert float(np.linalg.eigvalsh(delta.T@delta/len(delta))[0]) > 1.
        # The cap is welded to the inner sheet, not a floating datum patch.
        sheet_rows = 50*11
        assert sum(index < sheet_rows for index in truth["cap_rows"]) == 50
    else:
        assert truth["cap_rows"] == []
        assert truth["axial_containment_interval_mm"] == (-6., 6.)


def test_near_bound_noise_is_not_relabelled_small_noise_or_calibration():
    inputs, _ = curved_packet(CURVED_CASES[0])
    perturbed, displacement = _perturb(inputs["scan_points"], .049)
    assert perturbed.shape == inputs["scan_points"].shape
    norms = np.linalg.norm(displacement, axis=1)
    assert .04 < float(norms.max()) <= .049 < EPSILON_MM
    # This is only a noise-envelope construction check; it does not establish
    # accurate radius under arbitrary .05 mm perturbations or a runtime pass.


def test_nominal_context_is_cad_only_and_free_circle():
    values = nominal_feature_kwargs()
    assert values["nominal_radius_mm"] == 5. and values["nominal_sweep_deg"] == 90.
    assert values["fit_model"] == "free_circle" and values["search_margin_mm"] == 3.
    assert values["max_fit_residual_mm"] == .05
    assert "scanner_rotation" not in values and "raw_generator_endpoints" not in values
    assert values["nominal_source"] == "EXPLICIT_ANALYTIC_ARRAY_FIXTURE_NOT_NATIVE_ADAPTER"


def _decode_config_literal(node):
    """Read the named CONFIG expression without importing/executing its runner."""
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_decode_config_literal(value) for value in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "dict" and not node.args
            and all(keyword.arg is not None for keyword in node.keywords)):
        assert len({keyword.arg for keyword in node.keywords}) == len(node.keywords)
        return {keyword.arg: _decode_config_literal(keyword.value) for keyword in node.keywords}
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "math" and node.attr == "pi"):
        return math.pi
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = _decode_config_literal(node.left), _decode_config_literal(node.right)
        assert type(left) in (int, float) and type(right) in (int, float) and right != 0
        return left/right
    raise AssertionError("UNREVIEWED_CONFIG_EXPRESSION")


def test_curved_policy_literals_match_named_physical_source_without_importing_runner():
    source = ROOT/"scripts/run_gold11_mesh_development_v3.py"
    with source.open("rb") as stream:
        blob = stream.read(512*1024+1)
    assert len(blob) <= 512*1024
    tree = ast.parse(blob)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "CONFIG"]
    assert len(assignments) == 1
    config = _decode_config_literal(assignments[0].value)
    assert config["position_error_bound_mm"] == EPSILON_MM
    for name, values in (PHYSICAL_POLICIES | PHYSICAL_BUDGETS).items():
        assert config[name] == values, name
    # The preserved fifty-row controls remain their different .5/.95 policy;
    # numerical equality of position and residual bounds is not calibration.
    assert config["registration"]["max_correspondence_distance_mm"] == 2.
    assert config["registration"]["min_scan_coverage"] == .75
    assert config["measurement"]["search_margin_mm"] == 3.


def test_config_decoder_rejects_callable_or_attribute_escape():
    for expression in ("__import__('os')", "math.__dict__", "dict(**other)", "list([1])"):
        with pytest.raises(AssertionError, match="UNREVIEWED_CONFIG_EXPRESSION"):
            _decode_config_literal(ast.parse(expression, mode="eval").body)


def test_cad_mesh_is_closed_oriented_material_boundary_not_hole_filling_fan():
    cad = _cad_arrays()
    triangles = cad["cad_triangles"]
    edges = {}
    for triangle in triangles:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            first, second = tuple(triangle[a]), tuple(triangle[b])
            edges.setdefault(tuple(sorted((first, second))), []).append((first, second))
    assert all(len(uses) == 2 and uses[0] == uses[1][::-1] for uses in edges.values())
    # Independently integrate the represented polygonal sector, not the true
    # circular arc's area; discretization is never silently called exact CAD.
    volume = float(np.einsum("ij,ij->i", triangles[:, 0],
                             np.cross(triangles[:, 1], triangles[:, 2])).sum()/6.)
    polygon_sector_area = .5*(6.**2-5.**2)*18*math.sin(math.radians(5.))
    expected = 30.*(polygon_sector_area+20.+29.)
    assert abs(volume-expected) < 1e-8


@pytest.mark.parametrize("case", CURVED_CASES, ids=lambda case: case.case_id)
def test_post_inference_oracle_does_not_accept_copied_nominal_measurement(case):
    _, truth = curved_packet(case)
    copied = dict(fit_model="free_circle", tangency_check_role="FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK",
                  measured_radius_mm=5., measured_sweep_deg=90., physical_trim_endpoints=None)
    with pytest.raises(AssertionError):
        assert_scan_derived_measurement(copied, truth)


def test_twenty_degree_arc_has_explicit_flat_alternative_at_declared_bound():
    angles = np.linspace(-math.pi/18., math.pi/18., 21)
    points = np.column_stack((5.*np.cos(angles), 5.*np.sin(angles)))
    strip_midline = 2.5*(1.+math.cos(math.pi/18.))
    half_width = 2.5*(1.-math.cos(math.pi/18.))
    assert float(np.max(np.abs(points[:, 0]-strip_midline))) <= half_width+1e-14
    assert half_width < .038 < EPSILON_MM
    # This explicit geometric witness is only a post-construction oracle. It
    # does not use sample density to turn a coherent .05 bound into tiny noise.


@pytest.mark.parametrize("case", CURVED_CASES[2:], ids=lambda case: case.case_id)
def test_undeleted_cap_exposes_position_only_support_role_contamination(case):
    """Pre-noise exact-pose predicate witness, not a runtime success/failure."""
    _, truth = curved_packet(case)
    p = truth["model_points"]
    first_flange_mask = ((-p[:, 1] >= 2.5) & (-p[:, 1] <= 12.5)
                        & (np.abs(p[:, 0]-5.) <= 3.) & (p[:, 2] >= 0.) & (p[:, 2] <= 30.))
    assert int(first_flange_mask.sum()) == 65
    off_inner = first_flange_mask & (np.abs(p[:, 0]-case.radius_mm) > .25)
    assert int(off_inner.sum()) == 10 and np.all(p[off_inner, 2] == 0.)
    assert float(np.min(np.abs(p[off_inner, 0]-case.radius_mm))) >= .5-1e-14
    # Middle arc station: first-contact profile index10 plus12 arc increments.
    cap_only = p[np.array([550+22, 600+22])]
    theta = np.rad2deg(np.arctan2(cap_only[:, 1], cap_only[:, 0]))
    assert np.all((theta >= 8.) & (theta <= 82.))
    assert np.all(np.linalg.norm(cap_only[:, :2], axis=1) <= 8.)
    second_normal = np.array([math.cos(math.radians(case.sweep_deg)),
                              math.sin(math.radians(case.sweep_deg)), 0.])
    assert np.all(np.abs(cap_only[:, 0]-case.radius_mm) > .075)
    assert np.all(np.abs(cap_only@second_normal-case.radius_mm) > .075)
    # The coordinates remain in the fixture. No simulator cap mask is supplied
    # to inference, and these witnesses cannot be called positive bend support.


def _planar_packet(noisy):
    """Exact original fifty-row geometry via named, hash-bound AST capsules."""
    sources = {}
    for relative, expected in PLANAR_SOURCE_HASHES.items():
        with (ROOT/relative).open("rb") as stream:
            blob = stream.read(256*1024+1)
        assert len(blob) <= 256*1024 and hashlib.sha256(blob).hexdigest() == expected
        sources[relative] = ast.parse(blob)
    old_tree = sources["tests/test_cad_guided_plane_initialization_v10.py"]
    rectangles = [node for node in old_tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_rectangle"]
    assert len(rectangles) == 1
    namespace = dict(np=np)
    exec(compile(ast.Module(body=rectangles, type_ignores=[]), "<reviewed-planar-rectangle>", "exec"), namespace)
    legacy = SimpleNamespace(_rectangle=namespace["_rectangle"])
    tree = sources["tests/test_cad_guided_plane_initialization_v11.py"]
    names = {"SPECS", "ROTATION", "TRANSLATION", "UNIT"}
    nodes = [node for node in tree.body if isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_two_rectangles"]
    assert len(nodes) == 4 and len(functions) == 1
    namespace = dict(np=np)
    exec(compile(ast.Module(body=nodes+functions, type_ignores=[]), "<reviewed-fifty-row-planar>", "exec"), namespace)
    values, nominal, clean_observed = namespace["_two_rectangles"](legacy, noisy)
    raw, triangles, cad, faces, local = values
    inputs = dict(scan_points=raw, scan_triangles=triangles, cad_triangles=cad,
                  cad_face_ids=faces, cad_source_triangle_ids=local,
                  position_error_bounds_mm=.05)
    truth = dict(nominal=nominal, clean_scan_points=clean_observed,
                 oracle_scope="ORIGINAL_PLANAR_START_ATTRIBUTION_NOT_CURVED_MEASUREMENT")
    return inputs, truth


def _policy_objects(arm, *, planar):
    """Identical numerical contracts across arms; only consumer versions differ."""
    from backend import cad_guided_plane_initialization_v11 as v11
    from backend import cad_guided_plane_initialization_v12 as v12
    from backend.cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1
    from backend.cad_guided_global_registration_v3 import SurfaceRegistrationBudgetsV3
    from backend.cad_guided_triangle_surface_v3 import TriangleSurfaceBudgets
    from backend.cad_guided_noisy_mesh_planes_v5 import NoisyMeshPlanePolicyV5, NoisyMeshPlaneBudgetsV5
    from backend.cad_guided_mesh_plane_proposals_v4 import CoplanarPatchPolicyV4, CoplanarPatchBudgetsV4
    from backend.cad_guided_plane_search_v6 import PlaneSearchPolicyV6, PlaneSearchBudgetsV6
    from backend.cad_guided_instances import SupportPolicy

    assert arm in ARMS
    initialize = v11 if arm == "C0" else v12
    budget_type = v11.PlaneInitializationBudgetsV11 if arm == "C0" else v12.PlaneInitializationBudgetsV12
    registration = dict(PHYSICAL_POLICIES["registration"])
    if planar:
        # Exact original _policy(50), including unchanged constructor defaults.
        policy = WholePartRegistrationPolicyV1(coarse_points=50, refine_points=50,
            coarse_iterations=2, refine_iterations=2, huber_delta_mm=.01,
            max_inlier_plane_p95_mm=.05, max_correspondence_distance_mm=.5,
            min_scan_coverage=.95, min_cad_coverage=.30)
    else:
        policy = WholePartRegistrationPolicyV1(**registration)
    noisy = dict(PHYSICAL_POLICIES["noisy_policy"])
    noisy["tile_triangles"] = tuple(noisy["tile_triangles"])
    search_budget = dict(PHYSICAL_BUDGETS["search_budgets"])
    if planar:
        search_budget["max_bank_planes"] = 2  # Original attribution control only.
    kwargs = dict(registration_policy=policy, unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        noisy_policy=NoisyMeshPlanePolicyV5(**noisy),
        noisy_budgets=NoisyMeshPlaneBudgetsV5(**PHYSICAL_BUDGETS["noisy_budgets"]),
        cad_patch_policy=CoplanarPatchPolicyV4(**PHYSICAL_POLICIES["cad_patch_policy"]),
        cad_patch_budgets=CoplanarPatchBudgetsV4(**PHYSICAL_BUDGETS["cad_patch_budgets"]),
        search_policy=PlaneSearchPolicyV6(**PHYSICAL_POLICIES["search_policy"]),
        search_budgets=PlaneSearchBudgetsV6(**search_budget),
        budgets=budget_type(**PHYSICAL_BUDGETS["initialization_budgets"],
                            max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]),
        surface_budgets=TriangleSurfaceBudgets(**PHYSICAL_BUDGETS["surface_budgets"],
                            max_certificate_pairs=CERTIFICATE_CAPS["surface_budgets"]),
        registration_budgets=SurfaceRegistrationBudgetsV3(**PHYSICAL_BUDGETS["registration_budgets"],
                            max_certificate_pairs=CERTIFICATE_CAPS["registration_budgets"]),
        two_plane_enabled=True)
    return initialize, kwargs, SupportPolicy(**PHYSICAL_POLICIES["support"])


def _array_digests(inputs):
    return {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in inputs.items() if type(value) is np.ndarray}


def run_array_arm(arm, inputs, *, planar):
    """No simulator truth, known transform or support labels in this interface."""
    initialize, kwargs, instance_policy = _policy_objects(arm, planar=planar)
    from backend.cad_guided_measurement import CADGuidedFeature
    from backend.cad_guided_local_measurement_v1 import inspect_full_raw_windows_v1
    from backend.cad_guided_context_measurement_v1 import (
        inspect_generated_context_windows_v1, validate_context_measurement_publication_v1)

    scan = inputs["scan_points"]
    scan_digest = hashlib.sha256(np.asarray(scan, dtype="<f8").tobytes()).hexdigest()
    cad_points = np.unique(inputs["cad_triangles"].reshape(-1, 3), axis=0)
    arguments = (scan, inputs["scan_triangles"], inputs["cad_triangles"],
                 inputs["cad_face_ids"], inputs["cad_source_triangle_ids"])
    kwargs.update(position_error_bounds_mm=inputs["position_error_bounds_mm"],
                  cad_initialization_points=cad_points)
    snapshot = {name: asdict(value) if hasattr(value, "__dataclass_fields__") else value
                for name, value in kwargs.items() if name != "cad_initialization_points"}
    snapshot.update(cad_initialization_point_contract="UNIQUE_DECLARED_CAD_TRIANGLE_VERTICES_NOT_NATIVE_SAMPLER",
                    cad_initialization_point_count=len(cad_points),
                    cad_initialization_points_sha256=hashlib.sha256(cad_points.tobytes()).hexdigest())
    owned = None
    if arm == "C2":
        registration, owned = initialize.register_measurement_contexts_v12(
            *arguments, **kwargs, connected_measurement_contexts=True)
    elif arm == "C1":
        registration = initialize.register_whole_part_planes_v12(
            *arguments, **kwargs, connected_measurement_contexts=False)
    else:
        registration = initialize.register_whole_part_planes_v11(*arguments, **kwargs)
    answer = dict(registration=registration, measurements=None,
                  measurement_disposition="NOT_RUN_NO_ELIGIBLE_REGISTRATION",
                  policy_snapshot=snapshot, scan_sha256=scan_digest,
                  scan_digest_scope="ARRAY_BYTES_ONLY_NOT_SCAN_FILE_AUTHENTICATION")
    if planar:
        answer["measurement_disposition"] = "NOT_A_CURVED_MEASUREMENT_CASE"
        return answer
    try:
        features = (CADGuidedFeature(**nominal_feature_kwargs()),)
        if arm == "C2":
            if owned is not None:
                answer["owned_surface_role_binding"] = {name: getattr(owned, name) for name in (
                    "points_sha256", "triangle_raw_rows_sha256", "position_error_bounds_mm_sha256",
                    "metadata_sha256", "raw_row_count", "triangle_count", "triangle_membership_count",
                    "raw_membership_count", "unscheduled_qualified_core_count")}
                answer["owned_surface_role_budgets"] = asdict(owned._budgets)
                assert answer["owned_surface_role_budgets"] == SURFACE_ROLE_CAPS
            if registration["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS":
                measured = inspect_generated_context_windows_v1(scan, features, registration=registration,
                    scan_sha256=scan_digest, instance_policy=instance_policy, surface_roles=owned)
                answer.update(measurements=measured, measurement_disposition="CONTEXT_MEASUREMENT_RETURNED")
                try:
                    validate_context_measurement_publication_v1(measured, raw_count=len(scan),
                        scan_sha256=scan_digest, raw_points_sha256=scan_digest)
                    answer["publication_validator"] = "PASSED_ACTUAL_VALIDATOR"
                except Exception as error:
                    # Preserve the exact returned answer for audit; the safety
                    # assertion below fails rather than losing scientific rows.
                    answer["publication_validator"] = "FAILED_ACTUAL_VALIDATOR"
                    answer["publication_validator_error"] = _error_record(error)
        elif registration["status"] == "CONDITIONAL_BEST_FIT":
            transform = np.asarray(registration["transform_scan_to_cad"])
            aligned = scan@transform[:3, :3].T+transform[:3, 3]
            measured = inspect_full_raw_windows_v1(aligned, features, instance_policy=instance_policy,
                scan_sha256=scan_digest, registration_status="CONDITIONAL_BEST_FIT")
            answer.update(measurements=measured, measurement_disposition="LEGACY_SINGLE_CONTEXT_MEASUREMENT_RETURNED")
    except Exception as error:
        # Unexpected later-stage failure must not erase the completed preceding
        # registration. The outer recorder marks this arm's execution failed.
        answer["measurement_execution_error"] = _error_record(error)
        answer["measurement_disposition"] = "MEASUREMENT_STAGE_RAISED"
    return answer


def _assert_safety(answer, arm):
    registration = answer["registration"]
    assert registration["transform_scan_to_cad"] is None or registration["status"] == "CONDITIONAL_BEST_FIT"
    evidence = registration["evidence"]
    assert evidence["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    if arm == "C2":
        assert registration["transform_scan_to_cad"] is None
        if registration["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS":
            assert evidence["accepted_registration"] is False
            assert evidence["joint_configuration_status"] == "NOT_ESTABLISHED"
            assert evidence["continuous_pose_family_coverage"] == "NOT_ESTABLISHED"
    measured = answer["measurements"]
    if measured is not None:
        assert measured["observed_bend_count"] is None and measured["production_use_allowed"] is False
        assert measured["conformance"] == "INDETERMINATE"
        if arm == "C2":
            assert answer["publication_validator"] == "PASSED_ACTUAL_VALIDATOR"
            assert measured["joint_configuration_status"] == "NOT_ESTABLISHED"
            assert measured["continuous_pose_family_coverage"] == "NOT_ESTABLISHED"
            assert measured["context_quantifier"] == "EXISTS_COMPLETED_GENERATED_CONTEXT"


def _assert_planar_attribution(answer, truth):
    initialization = answer["registration"]["evidence"]["initialization"]
    assert initialization["selected_policy_evaluation_complete"] is True
    assert initialization["numerical_consumer_complete"] is True
    groups = initialization["start_groups"]
    assert groups, initialization["reasons"]
    errors = []
    for group in groups:
        rotation = np.asarray(group["initialized_rotation"])
        translation = np.asarray(group["initialized_translation_mm"])
        errors.append(float(np.linalg.norm(
            truth["clean_scan_points"]@rotation.T+translation-truth["nominal"], axis=1).max()))
    assert min(errors) <= 1e-7
    assert answer["measurements"] is None
    return dict(minimum_generated_start_action_error_mm=min(errors), curved_measurement_claim=False)


def _assert_curved_output(answer, truth, arm):
    measured = answer["measurements"]
    assert measured is not None, answer["registration"].get("reasons")
    records = measured["support_records"] if arm == "C2" else measured["features"]
    usable = [row for row in records if row["status"] == "MEASURED_CONDITIONAL"]
    assert usable, measured.get("reasons")
    # EVERY emitted measurement must pass. There is no truth-based best record,
    # assignment, threshold, support subset or pose selection in this oracle.
    errors = [assert_scan_derived_measurement(row["measurements"], truth) for row in usable]
    if arm == "C2":
        for row in usable:
            value = row["measurements"]
            assert value["geometry_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
            assert value["radius_uncertainty_interval_mm"] is None
            assert value["sweep_uncertainty_interval_deg"] is None
            assert value["datum_relative_deviation"] is None
            for association in row["associations"]:
                assert association["accepted_registration"] is False
                expected_radius = value["measured_radius_mm"]-association["nominal_radius_mm"]
                expected_sweep = value["measured_sweep_magnitude_deg"]-abs(association["nominal_sweep_deg"])
                assert abs(association["canonical_radius_deviation_mm"]-expected_radius) <= 1e-12
                assert abs(association["canonical_sweep_magnitude_deviation_deg"]-expected_sweep) <= 1e-12
        if truth["case"].localized:
            instances = measured["predictions"]
            assert instances, "MISSING_REQUIRED_LOCALIZED_RAW_CARRIER"
            carrier_errors = [assert_localized_raw_carrier(instance, truth) for instance in instances]
        else:
            carrier_errors = []
            assert measured["registration_status"] == "NOT_ACCEPTED_CONTEXT_PROPOSALS_ONLY"
            assert measured["continuous_pose_family_coverage"] == "NOT_ESTABLISHED"
    else:
        carrier_errors = []  # Baselines retain their original CAD-frame output.
    return dict(measured_records=len(usable), measurement_errors=errors, carrier_errors=carrier_errors)


def _error_record(error):
    return dict(type=type(error).__name__, message=str(error)[:1000])


def _encode_bounded(value, maximum):
    chunks, total = [], 0
    for fragment in json.JSONEncoder(allow_nan=False, separators=(",", ":"), sort_keys=True).iterencode(value):
        block = fragment.encode("utf-8")
        total += len(block)
        if total > maximum:
            raise ValueError("CASE_ARM_PUBLICATION_BYTE_LIMIT")
        chunks.append(block)
    return b"".join(chunks)


def _write_complete_record(directory, name, value, maximum):
    blob = _encode_bounded(value, maximum)
    temporary, target = directory/(name+".partial"), directory/name
    with temporary.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(target)
    return dict(filename=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


def test_bounded_json_exact_edge_and_nonfinite_whole_rejection():
    blob = _encode_bounded({"status": "UNAVAILABLE", "answer": None}, 1024)
    assert _encode_bounded({"status": "UNAVAILABLE", "answer": None}, len(blob)) == blob
    with pytest.raises(ValueError, match="CASE_ARM_PUBLICATION_BYTE_LIMIT"):
        _encode_bounded({"status": "UNAVAILABLE", "answer": None}, len(blob)-1)
    for invalid in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            _encode_bounded({"prefix": "must not survive", "invalid": invalid}, 1024)


def test_complete_artifact_writer_never_leaves_prefix_on_encoding_failure(tmp_path):
    name = "ARM_RESULT_00_C0.json"
    with pytest.raises(ValueError, match="CASE_ARM_PUBLICATION_BYTE_LIMIT"):
        _write_complete_record(tmp_path, name, {"status": "too large"}, 2)
    assert not (tmp_path/name).exists() and not (tmp_path/(name+".partial")).exists()
    receipt = _write_complete_record(tmp_path, name, {"status": "retained"}, 1024)
    blob = (tmp_path/name).read_bytes()
    assert receipt == dict(filename=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
    assert json.loads(blob) == {"status": "retained"}
    assert not (tmp_path/(name+".partial")).exists()


def test_full_packet_without_opt_in_stops_before_creating_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv(FULL_MATRIX_OPT_IN, raising=False)
    path = tmp_path/"ARTIFACTS"
    with pytest.raises(AssertionError, match="FULL_MATRIX_REQUIRES_EXPLICIT_SUPERVISOR_OPT_IN"):
        run_connected_packet(path)
    assert not path.exists()


def run_connected_packet(artifact_directory):
    """Serial18-arm run; failures are retained and do not rescue/drop cases."""
    assert os.environ.get(FULL_MATRIX_OPT_IN) == "1", "FULL_MATRIX_REQUIRES_EXPLICIT_SUPERVISOR_OPT_IN"
    directory = Path(artifact_directory)
    assert directory.is_absolute() and directory.name == "ARTIFACTS"
    directory.mkdir(exist_ok=False)  # New root-owned location only; no old records read.
    records, total_result_bytes = [], 0
    cases = [(case_id, True, case_id == "P-noisy") for case_id in PLANAR_CASES]
    cases += [(case.case_id, False, case) for case in CURVED_CASES]
    assert tuple(case[0] for case in cases) == CASE_IDS
    for case_index, (case_id, planar, specification) in enumerate(cases):
        # Geometry is prepared once independently, then every arm receives new
        # arrays with identical bytes. One arm cannot influence a later input.
        inputs, truth = _planar_packet(specification) if planar else curved_packet(specification)
        common_digests = _array_digests(inputs)
        for arm in ARMS:
            start = dict(case_index=case_index, case_id=case_id, arm=arm, disposition="STARTED",
                input_digests=common_digests, input_shapes={key: list(value.shape)
                    for key, value in inputs.items() if type(value) is np.ndarray},
                position_error_bound_mm=.05, fixture_scope="INDEPENDENT_CONSTRUCTED_ARRAYS_ONLY")
            _write_complete_record(directory, f"ARM_START_{case_index:02d}_{arm}.json", start,
                                   ARTIFACT_LIMITS["per_start_json_bytes"])
            record = dict(start, disposition="COMPLETED_ARM", answer=None, execution_error=None,
                          safety_error=None, oracle_error=None, positive_oracle_passed=False,
                          required_positive=(arm in ("C1", "C2") if planar else arm == "C2"))
            owned_inputs = {name: value.copy() if type(value) is np.ndarray else value
                            for name, value in inputs.items()}
            try:
                record["answer"] = run_array_arm(arm, owned_inputs, planar=planar)
                record["execution_error"] = record["answer"].get("measurement_execution_error")
                assert _array_digests(owned_inputs) == common_digests, "INPUT_MUTATED_BY_ARM"
            except Exception as error:
                record["execution_error"] = _error_record(error)
            if record["answer"] is not None:
                try:
                    _assert_safety(record["answer"], arm)
                except Exception as error:
                    record["safety_error"] = _error_record(error)
                try:
                    record["oracle"] = (_assert_planar_attribution(record["answer"], truth) if planar
                                         else _assert_curved_output(record["answer"], truth, arm))
                    record["positive_oracle_passed"] = True
                except Exception as error:
                    record["oracle_error"] = _error_record(error)
            filename = f"ARM_RESULT_{case_index:02d}_{arm}.json"
            try:
                artifact = _write_complete_record(directory, filename, record,
                    ARTIFACT_LIMITS["per_case_arm_json_bytes"])
            except (TypeError, ValueError, OverflowError, RecursionError) as error:
                # Retention failure is not an inference abstention or a passing
                # geometry result. Publish a whole-discard receipt, not a prefix.
                record = dict(case_index=case_index, case_id=case_id, arm=arm,
                    disposition="RESULT_RETENTION_UNAVAILABLE", input_digests=common_digests,
                    answer=None, discarded_answer_whole=True, positive_oracle_passed=False,
                    required_positive=record["required_positive"], publication_error=_error_record(error))
                artifact = _write_complete_record(directory, filename, record, 4096)
            total_result_bytes += artifact["bytes"]
            assert total_result_bytes <= ARTIFACT_LIMITS["total_case_arm_json_bytes"]
            records.append(dict(case_id=case_id, arm=arm, disposition=record["disposition"],
                required_positive=record["required_positive"],
                positive_oracle_passed=record["positive_oracle_passed"],
                execution_error=record.get("execution_error"), safety_error=record.get("safety_error"),
                oracle_error=record.get("oracle_error"), publication_error=record.get("publication_error"),
                artifact=artifact))
    assert tuple((row["case_id"], row["arm"]) for row in records) == CASE_ARM_IDS
    completed = all(row["disposition"] == "COMPLETED_ARM" for row in records)
    safe = all(row["execution_error"] is None and row["safety_error"] is None for row in records)
    positives = all(row["positive_oracle_passed"] for row in records if row["required_positive"])
    return dict(schema="CONNECTED_V12_SIX_CASE_THREE_ARM_ARRAY_PACKET",
        status="PASS" if completed and safe and positives else "FAIL",
        exact_case_arm_inventory_complete=completed, all_arm_execution_and_safety_passed=safe,
        required_connected_positives_passed=positives, records=records,
        total_result_bytes=total_result_bytes, resource_limits=RESOURCE_LIMITS,
        artifact_limits=ARTIFACT_LIMITS, physical_or_native_qualification=False,
        observed_bend_count=None, production_use_allowed=False)


@pytest.mark.skipif(os.environ.get(FULL_MATRIX_OPT_IN) != "1",
                    reason="Explicit bounded supervisor/PRE required; no connected execution in source gate")
def test_connected_six_case_three_arm_packet():
    directory = os.environ.get(ARTIFACT_DIRECTORY_ENV)
    assert directory, "Fresh ARTIFACTS path required"
    receipt = run_connected_packet(directory)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    # Every requested arm is retained before this aggregate positive assertion.
    assert receipt["status"] == "PASS", receipt
