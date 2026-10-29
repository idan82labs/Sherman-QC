"""Refusal metadata stays tied to the attempted pipeline, without native work."""
from backend.vendor_cad_unfold_pipeline_v4 import build_unfold_preview


def test_unavailable_envelope_names_actual_frame_implementation():
    result = build_unfold_preview("not-read.step", "a" * 64, [])
    assert result["status"] == "UNAVAILABLE"
    assert result["reason"] == "CANDIDATE_BUDGET_OR_SCHEMA_INVALID"
    diagnostics = result["diagnostics"]
    assert diagnostics["pipeline_version"] == 4
    assert diagnostics["extractor_schema_version"] == 7
    assert "V7_EXACT_TRIMMED_PCURVE_CORRIDOR_CYLINDER_FEATURES" in diagnostics["limitations"]
    assert "V3_RECTANGULAR_NATIVE_CYLINDER_FEATURES" not in diagnostics["limitations"]
