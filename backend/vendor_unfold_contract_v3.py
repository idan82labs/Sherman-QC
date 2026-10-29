"""Full transport-V3 admission for the reviewed finite-width V4 skin candidate.

Compose the common body/source/marker contract with the finite-width diagnostic
add-on. Neither a relabelled mechanism-3 receipt nor the add-on alone admits
geometry. Inputs are never changed. No CAD, native modules, catalog, files or
runtime pins are accessed here; this validates receipts, not physical support,
runtime source identity, neutral-axis blanks or manufacturing readiness.

Requires the separately reviewed private V2 common validator interface. The
public V2 validator remains strict mechanism 3 and is not used as a shadow V4
admission path.
"""
from __future__ import annotations

import re

from backend.vendor_unfold_contract_v2 import (
    MODEL, PreviewContractError, canonical_candidates, strict_json,
    _validate_preview as _validate_common_preview,
)
from backend.vendor_unfold_finite_width_contract_v4 import validate_finite_width_diagnostics


TRANSPORT_VERSION = 3
MECHANISM_VERSION = 4


def unavailable(reason, source_hash=None):
    """Construct a fresh typed, empty V4 refusal; never relabel other output."""
    if type(reason) is not str or not reason.strip() or len(reason) > 200:
        raise PreviewContractError("INVALID_UNAVAILABLE_REASON")
    if source_hash is not None and (type(source_hash) is not str
                                   or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None):
        raise PreviewContractError("INVALID_SOURCE_HASH")
    return {"status": "UNAVAILABLE", "reason": reason, "model": MODEL,
            "units": "assumed_mm", "units_verified": False, "manufacturing_ready": False,
            "source_sha256": source_hash, "vertices": [], "triangles": [],
            "bend_lines": [], "flange_surfaces": [],
            "diagnostics": {"transport_version": TRANSPORT_VERSION,
                            "mechanism_version": MECHANISM_VERSION, "visual_only": True}}


def validate_preview(payload, expected_hash, candidates):
    """Return the identical valid payload after BOTH mandatory receipt gates.

    Unavailable envelopes still need common source/empty-geometry/claim checks
    and typed transport/mechanism declarations, but no successful finite-width
    certificate is required for an abstention. This function never supplies or
    rewrites mechanism versions, diagnostic proofs, geometry or release flags.
    """
    result = _validate_common_preview(payload, expected_hash, candidates,
                                      mechanism_version=MECHANISM_VERSION)
    diagnostics = result["diagnostics"]
    if (type(diagnostics.get("transport_version")) is not int
            or diagnostics["transport_version"] != TRANSPORT_VERSION
            or type(diagnostics.get("mechanism_version")) is not int
            or diagnostics["mechanism_version"] != MECHANISM_VERSION):
        raise PreviewContractError("INVALID_V3_TRANSPORT_OR_V4_MECHANISM")
    if result["status"] == "AVAILABLE":
        validate_finite_width_diagnostics(result, candidates)
    return result
