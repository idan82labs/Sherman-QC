"""Fixed invented STEP + mesh-PLY checks of the actual successor workflow.

Root must run this matrix once under an external <=60 s wall / <=45 s CPU /
<=1 GiB process-tree RSS supervisor. This module performs no native work at
import/ordinary collection. No physical file, Gold/reference, app, or datum is
read. The only files are a newly constructed STEP in an owned temporary folder
and the workflow's own byte snapshot. Native geometry and scan samples are
independently constructed; no CAD triangulation supplies scan observations.

Frozen cases: v1 nominal MUST measure one finite instance; v1 missing arc MUST
publish none; mesh-compatible r0 nominal MUST measure one finite instance.
An abstention in either nominal case is a failure, not an acceptable outcome.
The r0 branch is the NEW common-wrapper control, not the historic/frozen engine.

Numerical criteria: intrinsic radius and absolute sweep error <=2e-5 mm/degree;
predicted sampled generator endpoints <=.02 mm from the analytic scan-frame
oracle; recovered rigid pose action error <=.20 mm on all original rows. The
latter allows approximate triangulated-CAD sampling, not exact surface ICP.
These are synthetic algorithm checks, not manufacturing/calibration tolerances.
The modest synthetic registration policy is NOT the physical draft policy.
No nominal value, pose, datum, feature annotation, or correspondence is passed
to inference beyond the invented CAD bytes and fixed label-independent policy.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest


CONNECTED_CASES = (
    ("v1_nominal_oblique", "v1", False, 1),
    ("v1_missing_arc_oblique", "v1", True, 0),
    ("r0_common_wrapper_nominal_oblique", "r0", False, 1),
)
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_CONNECTED_V1"
RESOURCE_LIMITS = {"wall_seconds": 60, "cpu_seconds": 45, "process_tree_rss_bytes": 1024**3}
CRITERIA = {"radius_error_mm": 2e-5, "absolute_sweep_error_deg": 2e-5,
            "generator_endpoint_error_mm": .02, "pose_action_error_mm": .20}


def _asymmetric_native_strip():
    """Exact R5/R6 cross section, unequal 20/29 mm flanges, 30 mm prism."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec

    def point(value):
        return gp_Pnt(*map(float, value))

    def polar(radius, angle):
        return (radius * math.cos(angle), radius * math.sin(angle), 0.)

    def edge(first, second):
        builder = BRepBuilderAPI_MakeEdge(point(first), point(second))
        assert builder.IsDone()
        return builder.Edge()

    def arc(radius, first, last):
        curve = GC_MakeArcOfCircle(point(polar(radius, first)),
                                  point(polar(radius, (first + last) / 2)),
                                  point(polar(radius, last)))
        assert curve.IsDone()
        builder = BRepBuilderAPI_MakeEdge(curve.Value())
        assert builder.IsDone()
        return builder.Edge()

    # Counterclockwise outer boundary and clockwise inner arc form one closed
    # sheet strip. No detector/adapter feature or mesh is consulted here.
    edges = (
        edge((6., -20., 0.), (6., 0., 0.)), arc(6., 0., math.pi / 2),
        edge((0., 6., 0.), (-29., 6., 0.)),
        edge((-29., 6., 0.), (-29., 5., 0.)),
        edge((-29., 5., 0.), (0., 5., 0.)), arc(5., math.pi / 2, 0.),
        edge((5., 0., 0.), (5., -20., 0.)),
        edge((5., -20., 0.), (6., -20., 0.)),
    )
    wire = BRepBuilderAPI_MakeWire()
    for item in edges:
        wire.Add(item)
        assert wire.IsDone()
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)), wire.Wire(), True)
    assert face.IsDone()
    prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., 30.), True, True)
    assert prism.IsDone()
    return prism.Shape()


def _owned_step_bytes(shape):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    with tempfile.TemporaryDirectory(prefix="invented-connected-mesh-") as directory:
        path = Path(directory) / "asymmetric_invented_strip.step"
        writer = STEPControl_Writer()
        assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
        assert writer.Write(str(path)) == IFSelect_RetDone
        blob = path.read_bytes()  # Only the STEP just created in this owned directory.
        assert 1 <= len(blob) <= 8 * 1024**2
        return blob


def _oracle_pose():
    """Independent quaternion matrix; runtime uses PCA/Kabsch/Rodrigues."""
    unit = np.array([1., -2., 3.]) / math.sqrt(14.)
    w = math.cos(.91 / 2)
    x, y, z = unit * math.sin(.91 / 2)
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return rotation, np.array([37., -23., 11.])


def _analytic_mesh_scan(missing_arc):
    """Partial inner scan plus remote cap patches; never native-mesh samples.

    Remote z-normal cap patches make axial registration observable without
    entering the root's 3..12 mm fitting strips. Unequal flange extents break
    cross-section reflection symmetry. Every mesh face joins a real sample grid.
    """
    patches = []
    axial = np.linspace(1., 29., 29)
    if not missing_arc:
        angles = np.linspace(0., math.pi / 2, 47)
        patches.append(np.array([[[5 * math.cos(a), 5 * math.sin(a), z]
                                  for z in axial] for a in angles]))
    patches.append(np.array([[[5., -d, z] for z in axial] for d in np.linspace(2., 19., 18)]))
    patches.append(np.array([[[-d, 5., z] for z in axial] for d in np.linspace(2., 28., 27)]))
    for z in (0., 30.):
        patches.append(np.array([[[r, -d, z] for r in (5.1, 5.5, 5.9)]
                                 for d in np.linspace(15., 19., 5)]))
        patches.append(np.array([[[-d, r, z] for r in (5.1, 5.5, 5.9)]
                                 for d in np.linspace(20., 28., 9)]))
    points, faces, offset = [], [], 0
    for patch in patches:
        first, second = patch.shape[:2]
        points.append(patch.reshape((-1, 3)))
        for i in range(first - 1):
            for j in range(second - 1):
                a = offset + i * second + j
                b, c, d = a + second, a + 1, a + second + 1
                faces.extend(((a, b, c), (c, b, d)))
        offset += first * second
    cad_frame = np.vstack(points)
    assert len(cad_frame) <= 3000
    rotation, translation = _oracle_pose()
    raw = cad_frame @ rotation.T + translation
    return raw, np.asarray(faces, dtype=np.int32), cad_frame


def _ply_bytes(points, faces):
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment invented partial mesh, units/frame unverified by importer\n"
              f"element vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii")
    packed = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    packed["count"], packed["indices"] = 3, faces
    return header + np.asarray(points, dtype="<f8").tobytes() + packed.tobytes()


def _policies(branch):
    from backend.cad_guided_mesh_offline_v1 import MeshWorkflowPolicyV1
    from backend.cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1
    from backend.cad_guided_native_features import NativeGuidedPolicy
    from backend.cad_guided_instances import SupportPolicy

    return dict(
        workflow_policy=MeshWorkflowPolicyV1(
            max_registration_scan_rows=3000, cad_sample_count=8192,
            cad_linear_deflection_mm=.02, cad_angular_deflection_radians=.10,
            native_adapter_version=branch),
        registration_policy=WholePartRegistrationPolicyV1(
            coarse_points=1024, refine_points=1024, coarse_iterations=20, refine_iterations=20,
            huber_delta_mm=.01, max_correspondence_distance_mm=.5,
            max_inlier_plane_p95_mm=.05, min_scan_coverage=.95, min_cad_coverage=.30,
            pose_class_distance_mm=.05, ambiguity_plane_rms_margin_mm=.005),
        measurement_policy=NativeGuidedPolicy(
            flange_window_mm=(3., 12.), search_margin_mm=.20, max_fit_residual_mm=.01,
            plane_exclusion_mm=.05, max_tangency_error_mm=.05,
            fit_model="measured_flange_tangent"),
        instance_policy=SupportPolicy(max_axial_gap_mm=1.1, max_angular_gap_deg=5.,
                                      min_arc_sweep_deg=20., min_local_points=3),
    )


def _common_assertions(result, cad_blob, scan_blob):
    json.dumps(result, allow_nan=False)
    assert result["observed_bend_count"] is None
    assert result["production_use_allowed"] is False
    assert result["conformance"] == "INDETERMINATE"
    assert result["uncertainty_status"] == "UNCALIBRATED"
    assert result["cad_sha256"] == hashlib.sha256(cad_blob).hexdigest()
    assert result["input_preparation"]["source_bindings"]["scan_sha256"] == hashlib.sha256(scan_blob).hexdigest()
    units = result["cad_units"]
    assert units["target_unit"] == "mm" and units["transferred_roots"] == 1
    assert units["target_unit_action"] == "SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER"
    assert units["source_unit_names"]["length"]
    assert all(name.upper() in ("MILLIMETRE", "MILLIMETER", "MM")
               for name in units["source_unit_names"]["length"])
    assert units["physical_calibration"] == "NOT_ESTABLISHED"
    assert result["cad_surface_sampling"]["status"] == "AVAILABLE"
    assert result["cad_surface_sampling"]["native_surface_point_status"] == "NOT_CERTIFIED"
    assert result["cad_surface_sampling"]["coverage"]["sample_count"] == 8192
    assert result["registration"]["evidence"]["candidate_count"] == 24
    assert result["reference_frame_compatibility"] == "NOT_ASSERTED_BY_INFERENCE"


def _nominal_assertions(result, raw, cad_frame):
    assert result["status"] == "PARTIAL", result.get("reasons")
    assert result["registration"]["status"] == "CONDITIONAL_BEST_FIT"
    assert len(result["predictions"]) == 1
    transform = np.asarray(result["registration"]["transform_scan_to_cad"])
    pose_error = float(np.linalg.norm(raw @ transform[:3, :3].T + transform[:3, 3] - cad_frame, axis=1).max())
    assert pose_error <= CRITERIA["pose_action_error_mm"], pose_error
    measured = result["measurements"]
    assert measured["raw_input_rows"] == len(raw)
    assert measured["observed_bend_count"] is None
    rows = [row for row in measured["features"] if row["status"] == "MEASURED_CONDITIONAL"]
    assert len(rows) == 1
    row, value = rows[0], rows[0]["measurements"]
    radius_error = abs(value["measured_radius_mm"] - 5.)
    angle_error = abs(abs(value["measured_sweep_deg"]) - 90.)
    assert radius_error <= CRITERIA["radius_error_mm"], radius_error
    assert angle_error <= CRITERIA["absolute_sweep_error_deg"], angle_error
    assert value["fit_model"] == "measured_flange_tangent"
    assert value["tangency_check_role"] == "ENFORCED_CONSTRAINT_NOT_INDEPENDENT_VALIDATION"
    support = row["raw_support"]
    assert isinstance(support["arc_rows"], list)
    assert support["arc_rows_sha256"] == hashlib.sha256(np.asarray(support["arc_rows"], dtype="<i8").tobytes()).hexdigest()
    for ids, digest in zip(support["flange_rows"], support["flange_rows_sha256"]):
        assert digest == hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    prediction = result["predictions"][0]
    rotation, translation = _oracle_pose()
    expected = np.array([[5 / math.sqrt(2), 5 / math.sqrt(2), z] for z in (1., 29.)]) @ rotation.T + translation
    actual = np.asarray([prediction["line_start"], prediction["line_end"]])
    endpoint_error = min(float(np.linalg.norm(actual - expected, axis=1).max()),
                         float(np.linalg.norm(actual[::-1] - expected, axis=1).max()))
    assert endpoint_error <= CRITERIA["generator_endpoint_error_mm"], endpoint_error
    assert prediction["line_convention"] == "MEASURED_INNER_ARC_MID_ANGLE_GENERATOR"
    assert prediction["physical_trim_endpoints"] is None
    assert prediction["manufacturing_event_identity"] == "UNKNOWN"
    # Both axis endpoints must undergo the same CAD-to-original-scan inverse.
    axis_actual = np.asarray([prediction["axis_support_line_start"], prediction["axis_support_line_end"]])
    axis_expected = np.array([[0., 0., z] for z in (1., 29.)]) @ rotation.T + translation
    axis_error = min(float(np.linalg.norm(axis_actual - axis_expected, axis=1).max()),
                     float(np.linalg.norm(axis_actual[::-1] - axis_expected, axis=1).max()))
    assert axis_error <= CRITERIA["generator_endpoint_error_mm"], axis_error
    return dict(pose_action_error_mm=pose_error, radius_error_mm=radius_error,
                absolute_sweep_error_deg=angle_error, generator_endpoint_error_mm=endpoint_error,
                axis_endpoint_error_mm=axis_error)


def run_invented_mesh_workflow_checks():
    """Explicit root-supervised worker entry; every frozen case retains outcome."""
    from backend.cad_guided_mesh_offline_v1 import inspect_cad_guided_mesh_v1

    try:
        cad_blob = _owned_step_bytes(_asymmetric_native_strip())
    except Exception as error:
        return dict(cases=[dict(case=name, passed=False, phase="fixture",
                               error_type=type(error).__name__, error=str(error)[:1000])
                           for name, _, _, _ in CONNECTED_CASES],
                    expected_case_count=len(CONNECTED_CASES), criteria=CRITERIA,
                    external_resource_limits=RESOURCE_LIMITS)
    records = []
    for name, branch, missing_arc, expected_predictions in CONNECTED_CASES:
        record = dict(case=name, passed=False, phase="fixture", expected_predictions=expected_predictions)
        try:
            raw, faces, cad_frame = _analytic_mesh_scan(missing_arc)
            scan_blob = _ply_bytes(raw, faces)
            record.update(scan_rows=len(raw), mesh_faces=len(faces), cad_sha256=hashlib.sha256(cad_blob).hexdigest())
            record["phase"] = "actual_byte_to_workflow"
            result = inspect_cad_guided_mesh_v1(scan_bytes=scan_blob, cad_bytes=cad_blob,
                                               allow_ocp=True, **_policies(branch))
            record["workflow"] = result
            record["phase"] = "frozen_assertions"
            _common_assertions(result, cad_blob, scan_blob)
            assert result["input_preparation"]["source_bindings"]["raw_xyz_sha256"] == hashlib.sha256(
                np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
            assert result["input_preparation"]["coverage"]["raw_vertex_count"] == len(raw)
            assert len(result["predictions"]) == expected_predictions
            if missing_arc:
                assert result["status"] == "UNAVAILABLE"
                assert result["reasons"]
                # Pose ambiguity or local insufficiency may explain abstention;
                # it must never be converted to a defect or an observed bend.
                assert result["phase"] in ("WHOLE_PART_REGISTRATION", "COMPLETE")
            else:
                record["errors"] = _nominal_assertions(result, raw, cad_frame)
            record.update(passed=True, phase="complete")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:1000])
        records.append(record)
    return dict(cases=records, expected_case_count=len(CONNECTED_CASES), criteria=CRITERIA,
                external_resource_limits=RESOURCE_LIMITS,
                policy_scope="FIXED_SYNTHETIC_INTERFACE_CHECK_NOT_PHYSICAL_RUN_POLICY",
                evidence_class="INVENTED_NATIVE_CONNECTED_NOT_PHYSICAL_GENERALIZATION")


def test_fixed_invented_mesh_workflow_matrix():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Root must explicitly opt in under the fixed external native budget")
    receipt = run_invented_mesh_workflow_checks()
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    print(json.dumps({"connected_compact": [
        {key: row.get(key) for key in ("case", "passed", "phase", "error_type", "error", "errors")}
        for row in receipt["cases"]]}, sort_keys=True), flush=True)
    assert len(receipt["cases"]) == len(CONNECTED_CASES)
    assert all(row["passed"] for row in receipt["cases"]), receipt
