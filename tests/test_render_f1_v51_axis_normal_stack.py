import json

from scripts.render_f1_v51_axis_normal_stack import render


def test_inactive_case_still_emits_audit_artifact(tmp_path):
    scan = tmp_path / "O001.ply"
    scan.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.json"
    prediction.write_text(
        json.dumps({"single_axis_parallel_return_recovery_audit": {"gate_active": False, "reason": "requires_exactly_one_prediction"}}),
        encoding="utf-8",
    )
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"line_objects": []}), encoding="utf-8")
    output = tmp_path / "axis.png"
    report = render(scan, prediction, pool, output, "O001")
    assert report == {"case_id": "O001", "gate_active": False, "slice_count": 0, "source_point_count": 4}
    assert output.is_file() and output.stat().st_size > 1000
