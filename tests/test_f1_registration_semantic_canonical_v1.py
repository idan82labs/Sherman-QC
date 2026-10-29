import math

import pytest

from research.f1_registration_semantic_canonical_v1 import canonical_json_bytes, canonicalize, semantic_payload


def _raw(selected: bool = True) -> dict:
    return {
        "case_token": "C001",
        "scan_sha256": "a",
        "cad_sha256": "b",
        "cad_template_sha256": "c",
        "deterministic_reference_points_sha256": "d",
        "deterministic_reference_normals_sha256": "e",
        "processed_scan_point_count": 10,
        "reference_point_count": 20,
        "voxel_size_mm": 1.0,
        "datum_candidate_captured": True,
        "datum_selected": selected,
        "global_transform": [[1.0, -0.0, 0.0, 1.12345678904]],
        "datum_transform": [[1.0, 0.0, 0.0, 287.8249999798134]],
        "datum_selection_stats": {"fitness": 0.1},
        "material_local_reversal_event_ids": ["E1"],
        "global_metrics": {"median_distance_mm": 1.0},
    }


def test_sub_quantum_differences_are_byte_identical():
    left = semantic_payload(_raw())
    right_raw = _raw()
    right_raw["global_transform"][0][3] += 1e-13
    right = semantic_payload(right_raw)
    assert canonical_json_bytes(left) == canonical_json_bytes(right)


def test_rejected_datum_candidate_is_fail_closed():
    payload = semantic_payload(_raw(selected=False))
    assert payload["selected_datum_transform"] is None
    assert payload["datum_selection_stats"] is None
    assert payload["global_metrics"] is None
    assert payload["material_local_reversal_event_ids"] == []
    assert payload["rejected_datum_candidate_fail_closed"] is True


def test_negative_zero_is_normalized():
    assert canonicalize(-0.0) == 0.0
    assert math.copysign(1.0, canonicalize(-0.0)) == 1.0


def test_nonfinite_is_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        canonicalize({"bad": float("nan")})
