import importlib.util
import numpy as np
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "trace_multiscale_raw_point_crease_candidates.py"
    spec = importlib.util.spec_from_file_location("trace_multiscale_raw_point_crease_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tracer = _load_module()


def _candidate(candidate_id, x_offset=0.0, axis=(1.0, 0.0, 0.0), span=40.0, thickness=1.0, linearity=0.9):
    points = [[x_offset + float(i), 0.0, 0.0] for i in range(20)]
    return {
        "candidate_id": candidate_id,
        "admissible": True,
        "sampled_points": points,
        "axis_direction": list(axis),
        "span_mm": span,
        "thickness_mm": thickness,
        "linearity": linearity,
    }


def test_dedupe_candidates_merges_overlapping_multiscale_lines():
    lhs = _candidate("FINE_RPC1", thickness=2.0, linearity=0.8)
    rhs = _candidate("STANDARD_RPC1", thickness=1.0, linearity=0.95)

    deduped = tracer._dedupe_candidates([lhs, rhs])

    assert len(deduped) == 1
    assert deduped[0]["candidate_id"] == "STANDARD_RPC1"
    assert "FINE_RPC1" in deduped[0]["duplicate_source_candidate_ids"]


def test_dedupe_candidates_keeps_spatially_separate_lines():
    lhs = _candidate("A")
    rhs = _candidate("B", x_offset=100.0)

    deduped = tracer._dedupe_candidates([lhs, rhs])

    assert len(deduped) == 2


def test_projection_overlap_rejects_disjoint_segments_on_same_line():
    lhs = _candidate("A")
    rhs = {
        **_candidate("B"),
        "sampled_points": [[100.0 + float(i), 0.0, 0.0] for i in range(20)],
    }
    axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    center = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    assert tracer._interval_overlap(
        tracer._projection_interval(lhs, axis, center),
        tracer._projection_interval(rhs, axis, center),
    ) == 0.0
