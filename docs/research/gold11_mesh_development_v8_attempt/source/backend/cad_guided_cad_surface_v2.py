"""Finite oriented CAD triangulation plus the unchanged V1 sampling protocol.

This additive seam keeps the native mesher invocation single-shot and retains
the exact triangles that generated the registration initialization samples.
It deliberately depends on the frozen V1 native traversal/numeric helpers;
both sources must be in every successor source closure. No input file, scan,
label, Gold reference, pose or nominal measurement enters this module.

Triangles approximate the caller-owned CAD handle, not certified analytic CAD
surfaces. Native execution requires an external CPU/wall/process-tree RSS cap.
The mesher may reuse/mutate caller triangulation caches; no cleanup/healing or
retry occurs. Geometry/source IDs are import-local, not universal face identity.
"""

import hashlib
import json

import numpy as np

from . import cad_guided_cad_sampling_v1 as _v1


_ARRAY_KEYS = (
    "triangles", "source_face_indices", "source_triangle_indices", "points",
    "normals", "sample_triangle_rows", "sample_source_face_indices",
    "sample_source_triangle_indices", "sample_barycentric_weights",
)


def _base():
    return {
        "schema": "CAD_GUIDED_FINITE_TRIANGLE_SURFACE_V2", "status": "UNAVAILABLE",
        "reasons": [], **dict.fromkeys(_ARRAY_KEYS), "production_use_allowed": False,
        "evidence_class": "APPROXIMATE_ORIENTED_TRIANGULATED_CAD_REGISTRATION_ONLY",
        "native_surface_distance_status": "NOT_ANALYTIC_OR_CERTIFIED",
        "calibrated_error_status": "NOT_ESTABLISHED",
        "source_unit_status": "CALLER_IMPORT_CONTRACT_NOT_VERIFIED_HERE",
        "frame_status": "NATIVE_MODEL_COORDINATES_NOT_SCAN_REGISTRATION",
        "native_validity_status": "NOT_ANALYTICALLY_OR_TOPOLOGICALLY_QUALIFIED",
        "source_identity_status": "CALLER_GEOMETRY_NOT_BYTE_AUTHORIZED_HERE",
        "face_identity_status": "INPUT_LOCAL_TAGS_NOT_UNIVERSAL_OWNERSHIP",
    }


def prepare_cad_triangle_surface_v2(triangles, *, source_face_indices,
                                    source_triangle_indices, sample_count=32768):
    """Pure finite-triangle export with unchanged area-distributed V1 samples.

    Admission, sampling and digests are performed on one owned immutable
    snapshot. Every output array is bytes-backed read-only; changes to caller
    arrays cannot change either the samples, finite surface or their binding.
    An unsupported triangle refuses the entire surface, never silently drops it.
    No nearest-surface index or distance claim is constructed in this seam.
    """
    base = _base()
    try:
        _v1._sample_count(sample_count)
        values = _v1._immutable(_v1._triangle_array(triangles), "<f8")
        faces = _v1._immutable(_v1._id_array(source_face_indices, len(values),
            0, _v1.MAX_FACES - 1, "INVALID_SOURCE_FACE_IDS"), "<i8")
        local_ids = _v1._immutable(_v1._id_array(source_triangle_indices, len(values),
            1, _v1.MAX_TRIANGLES, "INVALID_SOURCE_TRIANGLE_IDS"), "<i8")
        sampled = _v1.sample_triangle_interiors_v1(values, source_face_indices=faces,
            source_triangle_indices=local_ids, sample_count=sample_count)
        _v1._require(sampled["status"] == "AVAILABLE",
                     sampled["reasons"][0] if sampled["reasons"] else "CAD_SAMPLING_FAILED")
        _v1._require(sampled["digests"]["oriented_triangle_xyz_sha256"] == _v1._digest(values)
                     and sampled["digests"]["source_face_ids_sha256"] == _v1._digest(faces)
                     and sampled["digests"]["source_triangle_ids_sha256"] == _v1._digest(local_ids),
                     "TRIANGLE_SAMPLE_BINDING_MISMATCH")
        return {**sampled, **base, "status": "AVAILABLE", "triangles": values,
                "source_face_indices": faces, "source_triangle_indices": local_ids,
                **{key: sampled[key] for key in _ARRAY_KEYS if key in sampled},
                "sampling_policy": sampled["sampling_policy"],
                "coverage": sampled["coverage"], "digests": sampled["digests"]}
    except _v1.CADSamplingUnavailable as error:
        return {**base, "reasons": [str(error)]}
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return {**base, "reasons": ["CAD_TRIANGLE_SURFACE_NUMERICS_UNAVAILABLE"]}


def sample_native_cad_surface_v2(shape, *, allow_ocp=False, native_binding="ocp",
                                  sample_count=32768, linear_deflection_mm=.10,
                                  angular_deflection_radians=.20):
    """Export one oriented native mesh and its V1 initialization samples.

    The policy/traversal/location/winding/resource limits match V1. Only retaining
    the finite triangles is new. A downstream closest-point kernel must separately
    qualify its own geometric/numerical/operation budgets before consuming them.
    """
    base = {**_base(),
        "cache_effect": "MESHER_MAY_REUSE_OR_MUTATE_CALLER_NATIVE_TRIANGULATION_CACHE",
        "native_mesher_resource_enforcement": "EXTERNAL_SUPERVISOR_REQUIRED_BEFORE_POST_MESH_COUNT_CHECKS",
        "replay_scope": "SAME_NATIVE_TOPOLOGY_ENUMERATION_CACHE_AND_KERNEL_NOT_UNIVERSAL_EXPORT_IDENTITY"}
    if type(allow_ocp) is not bool or not allow_ocp or type(native_binding) is not str or native_binding != "ocp":
        return {**base, "reasons": ["EXPLICIT_OCP_BINDING_REQUIRED"]}
    try:
        _v1._sample_count(sample_count)
        linear = _v1._scalar(linear_deflection_mm, _v1.MIN_LINEAR_DEFLECTION_MM,
            _v1.MAX_LINEAR_DEFLECTION_MM, "LINEAR_DEFLECTION_POLICY")
        angular = _v1._scalar(angular_deflection_radians, _v1.MIN_ANGULAR_DEFLECTION_RADIANS,
            _v1.MAX_ANGULAR_DEFLECTION_RADIANS, "ANGULAR_DEFLECTION_POLICY")
        triangles, faces, local_ids, facts = _v1._native_triangles(shape, linear, angular)
        surface = prepare_cad_triangle_surface_v2(triangles, source_face_indices=faces,
            source_triangle_indices=local_ids, sample_count=sample_count)
        _v1._require(surface["status"] == "AVAILABLE",
                     surface["reasons"][0] if surface["reasons"] else "CAD_TRIANGLE_EXPORT_FAILED")
        policy = {
            "method": "OCP_BRepMesh_IncrementalMesh_CONSTRUCTOR_ONCE",
            "linear_deflection_mm": linear, "angular_deflection_radians": angular,
            "relative_deflection": False, "parallel": False, "max_faces": _v1.MAX_FACES,
            "max_topology_nodes": _v1.MAX_TOPOLOGY_NODES,
            "max_topology_depth": _v1.MAX_TOPOLOGY_DEPTH, "max_mesh_nodes": _v1.MAX_MESH_NODES,
            "max_triangles": _v1.MAX_TRIANGLES, "location_tolerance": _v1.LOCATION_TOLERANCE,
            "reflected_locations": "REFUSE", "nonrigid_locations": "REFUSE",
            "face_order": "DEPTH_FIRST_NATIVE_CHILD_ENUMERATION",
            "reversed_face_winding": "SWAP_INDICES_2_3", "native_shape_cache_cleanup": False,
            "deflection_error_is_certified": False, "accepted_status_mask": _v1.NATIVE_MESH_REUSED_FLAG,
            "accepted_status_meaning": "NO_ERROR_OR_INFORMATIONAL_REUSED_ONLY"}
        policy_bytes = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return {**base, **surface, "native_facts": facts, "mesher_policy": policy,
            "digests": {**surface["digests"],
                        "mesher_policy_sha256": hashlib.sha256(policy_bytes).hexdigest()}}
    except _v1.CADSamplingUnavailable as error:
        return {**base, "reasons": [str(error)],
                "native_mesher_status_flags": getattr(error, "native_mesher_status_flags", None)}
    except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
        return {**base, "reasons": ["NATIVE_CAD_TRIANGLE_SURFACE_UNAVAILABLE"]}
