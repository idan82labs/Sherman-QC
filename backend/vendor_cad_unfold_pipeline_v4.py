"""Explicit stage composition: V7 extraction, V6 geometry, V2 native oracle.

This separately frozen development pipeline does not relabel a V5 output as V6.
It invokes the unchanged V6 geometry developer and contracts on explicit-frame
V7 native features. Its reader, normalization and oracle have no fallback path.
Caller supplies a hash-bound snapshot under the shared time/RSS supervisor.
"""
import hashlib
from pathlib import Path
import re
import tempfile
from backend import vendor_unfold_preview as native
from backend.vendor_unfold_preview_v6 import develop_faces_v6, _envelope
from backend.vendor_unfold_native_v6 import read_native_v6
from backend.vendor_unfold_oracle_bridge_v2 import NativeOracleCapture, validate_captured_development
from scripts.extract_step_topology_bend_axes_v7 import extract_step_bend_axes


def build_unfold_preview(cad_path, expected_hash, candidates):
    source_hash = None
    try:
        native._require(type(expected_hash) is str and re.fullmatch(r"[0-9a-f]{64}", expected_hash),
                        "INVALID_EXPECTED_CAD_HASH")
        native._require(type(candidates) is list and 0 < len(candidates) <= native.MAX_CANDIDATES,
                        "CANDIDATE_BUDGET_OR_SCHEMA_INVALID")
        path = Path(cad_path)
        native._require(path.suffix.lower() in {".stp", ".step"} and path.is_file(), "STEP_SOURCE_REQUIRED")
        with path.open("rb") as stream:
            raw = stream.read(native.MAX_CAD_BYTES + 1)
        native._require(0 < len(raw) <= native.MAX_CAD_BYTES, "CAD_BYTE_BUDGET_EXCEEDED")
        source_hash = hashlib.sha256(raw).hexdigest()
        native._require(source_hash == expected_hash, "CAD_HASH_MISMATCH")
        with tempfile.TemporaryDirectory(prefix="sherman-cad-pipeline-v4-") as folder:
            snapshot = Path(folder) / "source.step"
            snapshot.write_bytes(raw)
            capture = NativeOracleCapture()
            faces, axes = read_native_v6(snapshot, on_face=capture.on_face,
                                         axis_extractor=extract_step_bend_axes)
            result = develop_faces_v6(faces, axes, candidates, source_hash)
            result["diagnostics"]["limitations"].remove("V3_RECTANGULAR_NATIVE_CYLINDER_FEATURES")
            result["diagnostics"]["limitations"].append("V7_EXACT_TRIMMED_PCURVE_CORRIDOR_CYLINDER_FEATURES")
            result["diagnostics"].update(pipeline_version=4, extractor_schema_version=7,
                                         angular_frame_provenance_required=True)
            oracle = validate_captured_development(result, capture, faces, axes, candidates)
            result["diagnostics"]["geometry_oracle"] = oracle
            result["diagnostics"]["independent_geometry_oracle"] = "VERIFIED_DISCRETE_NATIVE_METRIC_CHART"
            return result
    except native.UnfoldUnavailable as exc:
        result = _envelope(source_hash, str(exc))
    except (OSError, ImportError, RuntimeError, TypeError, ValueError, KeyError, IndexError) as exc:
        result = _envelope(source_hash, "CAD_PIPELINE_V4_DEVELOPMENT_UNAVAILABLE")
        result["diagnostics"].update(error_type=type(exc).__name__, error_reason=str(exc)[:200])
    result["diagnostics"].update(pipeline_version=4, extractor_schema_version=7)
    limitations = result["diagnostics"]["limitations"]
    if "V3_RECTANGULAR_NATIVE_CYLINDER_FEATURES" in limitations:
        limitations.remove("V3_RECTANGULAR_NATIVE_CYLINDER_FEATURES")
    limitations.append("V7_EXACT_TRIMMED_PCURVE_CORRIDOR_CYLINDER_FEATURES")
    return result
