"""Explicit-frame V4 extraction adapter; nominal CAD data, not measurements."""
import copy
import numpy as np
from backend.vendor_cad_bend_candidates_v3 import axis_rows_v3
from backend.vendor_unfold_contract_v2 import PreviewContractError


def axis_rows_v4(axes, tolerance_angle=5.0):
    # Preserve the V3 finite-line/radius/sweep checks; V4 changes frame transport,
    # not the meaning of nominal bend geometry or the downstream sweep limit.
    candidates = axis_rows_v3(axes, tolerance_angle)
    by_id = {a["feature_id"]: a for a in axes}
    for candidate in candidates:
        axis = by_id[candidate["feature_id"]]
        receipt = axis.get("carrier_receipt", {})
        frame = receipt.get("angular_frame")
        if not isinstance(frame, dict):
            raise PreviewContractError("V4_ANGULAR_FRAME_REQUIRED")
        try:
            matrix = np.stack([frame["x_axis"], frame["y_axis"], frame["axis"]]).astype(float)
        except (KeyError, ValueError, TypeError) as exc:
            raise PreviewContractError("V4_INVALID_ANGULAR_FRAME") from exc
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10, rtol=0)
                or abs(np.linalg.det(matrix) - 1) > 1e-10):
            raise PreviewContractError("V4_INVALID_ANGULAR_FRAME")
        sources = receipt.get("source_angular_frames")
        if (not isinstance(sources, list)
                or sorted(row.get("face_index") for row in sources if isinstance(row, dict))
                != axis["face_indices"]):
            raise PreviewContractError("V4_SOURCE_ANGULAR_FRAME_COVERAGE")
        provenance = candidate.pop("v3_provenance")
        provenance.update(extractor_schema_version=4, angular_frame=copy.deepcopy(frame),
                          source_angular_frames=copy.deepcopy(sources))
        candidate["v4_provenance"] = provenance
    return candidates


def extraction_to_candidates(payload, tolerance_angle=5.0):
    if (type(payload) is not dict or payload.get("schema_version") != 4
            or payload.get("kind") != "sherman_qc_step_topology_bend_axes"):
        raise PreviewContractError("V4_EXTRACTION_REQUIRED")
    return {"candidates": axis_rows_v4(payload.get("bend_axes"), tolerance_angle),
            "mapping_receipt": {"adapter_version": 4,
                                "native_reader_must_use_same_v4_features": True,
                                "manufacturing_ready": False}}
