import numpy as np

from scripts.build_additional_line_label_review import (
    apply_review_decisions,
    attach_scan_integrity,
    build_labels,
    line_scan_support,
)


def test_provisional_labels_are_not_authoritative_and_preserve_scan_frame(tmp_path):
    scan = tmp_path / "scan.ply"
    payload = build_labels(
        "14_demo",
        {
            "expectation": {"expected_completed_bends": 1},
            "matches": [
                {
                    "bend_id": "CAD_B1",
                    "countable_in_regression": True,
                    "detected_line_start": [1, 2, 3],
                    "detected_line_end": [4, 5, 6],
                }
            ],
        },
        scan,
    )

    assert payload["authoritative_ground_truth"] is False
    assert payload["coordinate_frame"] == "scan_raw"
    assert payload["manual_bends"][0]["review_decision"] == "unreviewed"
    assert payload["manual_bends"][0]["source_bend_id"] == "CAD_B1"


def test_scan_integrity_rejects_a_mixed_frame_line_deterministically():
    axis = np.linspace(0.0, 20.0, 101)
    points = np.column_stack((axis, np.zeros_like(axis), np.zeros_like(axis)))

    supported = line_scan_support(points, [0, 0, 0], [20, 0, 0])
    floating = line_scan_support(points, [300, 300, 300], [320, 300, 300])

    assert supported["frame_integrity_pass"] is True
    assert supported["support_fraction"] == 1.0
    assert supported["trimmed_line_start"] == [0.0, 0.0, 0.0]
    assert supported["trimmed_line_end"] == [20.0, 0.0, 0.0]
    assert floating["frame_integrity_pass"] is False
    assert floating["support_fraction"] == 0.0


def test_acceptance_trims_to_supported_span_and_rejection_preserves_source(tmp_path):
    scan = tmp_path / "scan.ply"
    payload = build_labels(
        "review_demo",
        {
            "expectation": {"expected_completed_bends": 2},
            "matches": [
                {"bend_id": "B1", "detected_line_start": [-5, 0, 0], "detected_line_end": [25, 0, 0]},
                {"bend_id": "B2", "detected_line_start": [0, 0, 0], "detected_line_end": [20, 0, 0]},
            ],
        },
        scan,
    )
    axis = np.linspace(0.0, 20.0, 201)
    points = np.column_stack((axis, np.zeros_like(axis), np.zeros_like(axis)))
    attach_scan_integrity(payload, points)
    apply_review_decisions(
        payload,
        {"P01": ("accepted", "supported transition"), "P02": ("rejected", "semantic face ridge")},
    )

    accepted, rejected = payload["manual_bends"]
    assert accepted["review_decision"] == "accepted"
    assert accepted["source_line_start"] == [-5.0, 0.0, 0.0]
    assert accepted["line_start"][0] < 0.0  # 3 mm support corridor is intentionally retained.
    assert accepted["line_end"][0] > 20.0
    assert rejected["review_decision"] == "rejected"
    assert "source_line_start" not in rejected
    assert payload["review_status"] == "visual_review_complete_training_only"
