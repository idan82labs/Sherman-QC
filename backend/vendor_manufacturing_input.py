"""Same-byte STEP unit custody for the manufacturing candidate.

Initial support is explicit native millimetre representations only. This is
file-unit provenance, not evidence of shop calibration or physical dimensions.
Call inside a time/RSS-bounded offline worker, never on an HTTP request thread.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

MAX_STEP_BYTES = 5_000_000


class ManufacturingInputRefused(ValueError):
    pass


def audit_step_bytes(raw: bytes, expected_sha256: str) -> dict:
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ManufacturingInputRefused("INVALID_EXPECTED_CAD_HASH")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_STEP_BYTES:
        raise ManufacturingInputRefused("STEP_BYTE_BUDGET_EXCEEDED")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ManufacturingInputRefused("CAD_BYTES_HASH_MISMATCH")
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from backend.vendor_manufacturing_step_units import StepUnitsRefused, validate_native_step_units

    with tempfile.TemporaryDirectory(prefix="sherman-step-unit-audit-") as folder:
        snapshot = Path(folder) / "source.step"
        snapshot.write_bytes(raw)
        reader = STEPControl_Reader()
        if reader.ReadFile(str(snapshot)) != IFSelect_RetDone:
            raise ManufacturingInputRefused("INVALID_STEP_SOURCE")
        try:
            unit_proof = validate_native_step_units(reader)
        except StepUnitsRefused as exc:
            raise ManufacturingInputRefused(str(exc)) from exc
        reader.SetSystemLengthUnit(1.0)
        if reader.SystemLengthUnit() != 1.0:
            raise ManufacturingInputRefused("STEP_SYSTEM_LENGTH_UNIT_MISMATCH")
    return {**unit_proof, "source_sha256": digest, "source_bytes": len(raw),
            "occt_system_length_unit": 1.0, "physical_calibration_verified": False}
