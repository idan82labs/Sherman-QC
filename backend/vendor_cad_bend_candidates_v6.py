"""Normalize native NumPy float64 residuals before the unchanged V5 contract.

V5 geometry and its certificate schema remain unchanged. Native calculations
can produce np.float64 while JSON round trips produce float. Only that exact
dimensional scalar type is normalized; integer identities are never coerced.
"""
import copy
import numpy as np
from backend.vendor_cad_bend_candidates_v5 import axis_rows_v5
from backend.vendor_unfold_contract_v2 import PreviewContractError


def axis_rows_v6(axes, tolerance_angle=5.0):
    normalized = copy.deepcopy(axes)
    if type(normalized) is list:
        for axis in normalized:
            if type(axis) is not dict:
                continue
            domain = axis.get("domain_receipt")
            if type(domain) is not dict:
                continue
            certificates = domain.get("native_domain_certificates")
            if type(certificates) is not list:
                continue
            for certificate in certificates:
                if type(certificate) is dict and type(certificate.get("maximum_residual_mm")) is np.float64:
                    certificate["maximum_residual_mm"] = float(certificate["maximum_residual_mm"])
    candidates = axis_rows_v5(normalized, tolerance_angle)
    for candidate in candidates:
        candidate["v5_provenance"]["adapter_version"] = 6
        candidate["v5_provenance"]["scalar_normalization"] = "NUMPY_FLOAT64_RESIDUAL_TO_PYTHON_FLOAT_ONLY"
    return candidates


def extraction_to_candidates(payload, tolerance_angle=5.0):
    if (type(payload) is not dict or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 5
            or payload.get("kind") != "sherman_qc_step_topology_bend_axes"):
        raise PreviewContractError("V5_EXTRACTION_REQUIRED")
    return {"candidates": axis_rows_v6(payload.get("bend_axes"), tolerance_angle),
            "mapping_receipt": {"adapter_version": 6, "extractor_schema_version": 5,
                                "native_reader_must_use_same_v5_features": True,
                                "manufacturing_ready": False}}
