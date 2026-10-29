from __future__ import annotations

import numpy as np

from backend import cad_guided_v11_topology_recovery_v1 as recovery


def test_recovery_is_scan_supported_and_keeps_nominal_reference_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        recovery,
        "extract_step_bend_axes",
        lambda _path: {
            "bend_axes": [
                {
                    "bend_id": "BEND_A",
                    "line_start": [0.0, 0.0, 0.0],
                    "line_end": [10.0, 0.0, 0.0],
                    "radii_mm": [1.0],
                    "angular_sweeps_deg": [90.0],
                    "surface_count": 2,
                }
            ]
        },
    )
    monkeypatch.setattr(
        recovery,
        "line_scan_support",
        lambda _points, start, end, *, distance_mm, sample_count=201: {
            "sample_count": sample_count,
            "distance_threshold_mm": distance_mm,
            "source_span_mm": 10.0,
            "bbox_overlap_fraction": 1.0,
            "support_fraction": 1.0,
            "longest_support_run_fraction": 1.0,
            "supported_span_mm": 10.0,
            "median_distance_mm": 0.4,
            "p90_distance_mm": 0.8,
            "trimmed_line_start": start,
            "trimmed_line_end": end,
            "frame_integrity_pass": True,
        },
    )
    result = recovery.recover_scan_supported_axes_v1(
        np.zeros((12, 3), dtype=np.float64), tmp_path / "placeholder.stp", np.eye(4)
    )
    assert result["status"] == "PARTIAL"
    assert result["accepted_count"] == 1
    assert result["gold_or_expected_count_used"] is False
    prediction = result["predictions"][0]
    assert prediction["line_start"] == [0.0, 0.0, 0.0]
    assert prediction["bend_data"]["median_scan_distance_mm"] == 0.4
    assert prediction["cad_reference"]["semantics"].startswith("CAD_NOMINAL")


def test_recovery_rejects_non_rigid_context_pose(tmp_path):
    # The transform guard is reached before any CAD access in a real run.
    # Exercise the same helper directly so this contract test stays source-only.
    with np.testing.assert_raises(ValueError):
        recovery._raw_from_cad(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                               np.diag([2.0, 1.0, 1.0, 1.0]))
