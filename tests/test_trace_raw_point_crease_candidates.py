import importlib.util
import numpy as np
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "trace_raw_point_crease_candidates.py"
    spec = importlib.util.spec_from_file_location("trace_raw_point_crease_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tracer = _load_module()


def test_line_stats_identifies_thin_line():
    points = np.asarray([[float(i), 0.05 * (i % 2), 0.0] for i in range(40)], dtype=np.float64)

    stats = tracer._line_stats(points)

    assert stats["span_mm"] > 35.0
    assert stats["thickness_mm"] < 0.2
    assert stats["linearity"] > 0.99


def test_connected_components_keeps_separated_crease_groups_apart():
    points = np.asarray(
        [[float(i), 0.0, 0.0] for i in range(5)]
        + [[100.0 + float(i), 0.0, 0.0] for i in range(5)],
        dtype=np.float64,
    )
    indexes = np.arange(len(points), dtype=np.int64)

    components = tracer._connected_components(indexes, points, radius=3.0)

    assert len(components) == 2
    assert sorted(len(component) for component in components) == [5, 5]
