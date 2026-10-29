"""Nominal V5 CAD features with complete native boundary-certificate custody."""
import copy
import math
from backend.vendor_cad_bend_candidates_v4 import axis_rows_v4
from backend.vendor_unfold_contract_v2 import PreviewContractError


def axis_rows_v5(axes, tolerance_angle=5.0):
    candidates = axis_rows_v4(axes, tolerance_angle)
    by_id = {axis["feature_id"]: axis for axis in axes}
    for candidate in candidates:
        axis = by_id[candidate["feature_id"]]
        domain = axis.get("domain_receipt", {})
        certificates = domain.get("native_domain_certificates")
        if (type(domain.get("certificate_schema_version")) is not int
                or domain.get("certificate_schema_version") != 5 or type(certificates) is not list
                or any(type(row) is not dict for row in certificates)
                or any(type(row.get("face_index")) is not int for row in certificates)
                or sorted(row.get("face_index", -1) for row in certificates) != axis["face_indices"]):
            raise PreviewContractError("V5_NATIVE_DOMAIN_COVERAGE_REQUIRED")
        for proof in certificates:
            tolerance = proof.get("declared_tolerance_mm")
            residual = proof.get("maximum_residual_mm")
            bounds = proof.get("bounds_metric_mm")
            if (proof.get("status") != "PROVED" or proof.get("type_proven") is not True
                    or proof.get("kind") != "NATIVE_PCURVE_RECTANGLE_SEGMENT_UNION"
                    or type(proof.get("wire_count")) is not int or proof.get("wire_count") != 1
                    or type(tolerance) not in (int, float) or not math.isfinite(tolerance)
                    or not 0 < tolerance <= 1e-5
                    or proof.get("tolerance_mm") != tolerance
                    or type(residual) not in (int, float) or not math.isfinite(residual)
                    or not 0 <= residual <= tolerance
                    or type(bounds) is not list or len(bounds) != 4
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds)
                    or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]
                    or proof.get("simple_connected_outer_loop") is not True
                    or proof.get("complete_side_coverage") is not True):
                raise PreviewContractError("V5_INVALID_NATIVE_DOMAIN_CERTIFICATE")
        provenance = candidate.pop("v4_provenance")
        provenance.update(extractor_schema_version=5,
                          native_domain_certificates=copy.deepcopy(certificates))
        candidate["v5_provenance"] = provenance
    return candidates


def extraction_to_candidates(payload, tolerance_angle=5.0):
    if (type(payload) is not dict or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 5
            or payload.get("kind") != "sherman_qc_step_topology_bend_axes"):
        raise PreviewContractError("V5_EXTRACTION_REQUIRED")
    return {"candidates": axis_rows_v5(payload.get("bend_axes"), tolerance_angle),
            "mapping_receipt": {"adapter_version": 5,
                                "native_reader_must_use_same_v5_features": True,
                                "manufacturing_ready": False}}
