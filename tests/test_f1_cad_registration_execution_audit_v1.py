from __future__ import annotations

import numpy as np

from research.f1_cad_registration_execution_audit_v1 import (
    AuditConfig,
    apply_transform,
    audit_transform_pair,
    classify_reversal,
    line_samples,
)


def _metrics(*, support: float, count: int, median: float, p90: float) -> dict:
    return {
        "support_fraction": support,
        "supported_sample_count": count,
        "median_distance_mm": median,
        "p90_distance_mm": p90,
    }


def test_transform_and_line_samples_are_deterministic() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    transform = np.eye(4)
    transform[:3, 3] = [4.0, -2.0, 1.0]
    assert np.allclose(apply_transform(points, transform), points + [4.0, -2.0, 1.0])
    assert np.allclose(line_samples([0, 0, 0], [2, 0, 0], 3), [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_reversal_requires_pre_step_support_and_material_change() -> None:
    config = AuditConfig()
    before = _metrics(support=0.8, count=161, median=0.5, p90=1.0)
    after = _metrics(support=0.65, count=131, median=0.55, p90=1.1)
    result = classify_reversal(before, after, config)
    assert result["material_local_reversal"] is True
    assert result["support_trigger"] is True
    weak = classify_reversal(
        _metrics(support=0.2, count=40, median=0.5, p90=1.0),
        _metrics(support=0.0, count=0, median=2.0, p90=3.0),
        config,
    )
    assert weak["eligible_pre_step_support"] is False
    assert weak["material_local_reversal"] is False


def test_pair_audit_uses_same_fixed_geometry() -> None:
    x = np.linspace(-5.0, 5.0, 201)
    scan = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    reference = scan.copy()
    global_transform = np.eye(4)
    datum_transform = np.eye(4)
    datum_transform[1, 3] = 1.0
    result = audit_transform_pair(
        processed_scan_points=scan,
        reference_points=reference,
        reference_center=[0.0, 0.0, 0.0],
        global_transform=global_transform,
        datum_transform=datum_transform,
        events=[{"event_id": "CAD_EVENT_0001", "cad_line_start_mm": [-5, 0, 0], "cad_line_end_mm": [5, 0, 0]}],
    )
    assert result["event_count"] == 1
    assert result["records"][0]["global_icp"]["median_distance_mm"] == 0.0
    assert result["records"][0]["datum_refinement"]["median_distance_mm"] == 1.0
    assert result["records"][0]["classification"]["material_local_reversal"] is True

