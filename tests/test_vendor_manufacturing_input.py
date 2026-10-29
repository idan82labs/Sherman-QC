"""Independent STEP bytes generated in temp folders; no physical input reads."""
import hashlib

import pytest

from backend.vendor_manufacturing_input import audit_step_bytes, ManufacturingInputRefused


def synthetic_step(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    writer = STEPControl_Writer()
    assert writer.Transfer(BRepPrimAPI_MakeBox(10, 20, 1.5).Shape(), STEPControl_AsIs) == IFSelect_RetDone
    path = tmp_path / "synthetic.step"
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path.read_bytes()


def test_native_source_mm_units_and_hash_are_explicit(tmp_path):
    raw = synthetic_step(tmp_path)
    result = audit_step_bytes(raw, hashlib.sha256(raw).hexdigest())
    assert result["units"] == "mm"
    assert result["units_verified_from_native_step"]
    assert not result["physical_calibration_verified"]


def test_hash_mismatch_refused_before_native_import(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("OCP"):
            raise AssertionError("native parser must not run")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ManufacturingInputRefused, match="HASH_MISMATCH"):
        audit_step_bytes(b"not STEP", "0" * 64)


def test_non_mm_units_refused(tmp_path):
    raw = synthetic_step(tmp_path)
    assert b"SI_UNIT(.MILLI.,.METRE.)" in raw
    # Change only the declared unit in independently generated synthetic bytes.
    raw = raw.replace(b"SI_UNIT(.MILLI.,.METRE.)", b"SI_UNIT($,.METRE.)")
    with pytest.raises(ManufacturingInputRefused, match="NON_MM_LENGTH"):
        audit_step_bytes(raw, hashlib.sha256(raw).hexdigest())


def test_invalid_step_refused():
    raw = b"invalid STEP"
    with pytest.raises(ManufacturingInputRefused, match="INVALID_STEP_SOURCE"):
        audit_step_bytes(raw, hashlib.sha256(raw).hexdigest())
