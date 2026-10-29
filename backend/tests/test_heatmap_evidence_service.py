import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "domains" / "bend" / "services" / "heatmap_evidence_service.py"
spec = importlib.util.spec_from_file_location("heatmap_evidence_service", MODULE_PATH)
heatmap_evidence_service = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(heatmap_evidence_service)


def test_cache_bundle_round_trip(tmp_path):
    cache_root = tmp_path / "cache" / "abc123"
    output_dir = tmp_path / "out"
    aligned_points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    deviations = np.array([0.1, -0.2], dtype=np.float32)
    stats = {"count": 2, "mean_abs_mm": 0.15, "p95_abs_mm": 0.2, "max_abs_mm": 0.2}

    heatmap_evidence_service._write_cache_bundle(
        cache_root,
        aligned_points=aligned_points,
        deviations=deviations,
        deviation_stats=stats,
    )

    bundle = heatmap_evidence_service._load_cached_bundle(cache_root, output_dir)

    assert bundle is not None
    assert bundle["cache_hit"] is True
    assert np.allclose(bundle["aligned_points"], aligned_points)
    assert np.allclose(bundle["deviations"], deviations)
    assert bundle["deviation_stats"] == stats
    assert bundle["aligned_scan_path"] == output_dir / "aligned_scan.ply"
    assert bundle["deviations_path"] == output_dir / "deviations.npy"
    assert (output_dir / "deviation_stats.json").exists()


def test_build_heatmap_evidence_returns_cached_bundle_before_qc_engine(monkeypatch, tmp_path):
    cad_path = tmp_path / "cad.stp"
    scan_path = tmp_path / "scan.stl"
    cad_path.write_text("cad", encoding="utf-8")
    scan_path.write_text("scan", encoding="utf-8")

    cached_bundle = {
        "aligned_scan_path": tmp_path / "aligned_scan.ply",
        "aligned_points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "deviations_path": tmp_path / "deviations.npy",
        "deviations": np.array([0.1], dtype=np.float32),
        "deviation_stats": {"count": 1},
        "cache_hit": True,
        "cache_key": "abc123",
    }

    monkeypatch.setattr(heatmap_evidence_service, "_load_cached_bundle", lambda *_args, **_kwargs: cached_bundle)

    result = heatmap_evidence_service.build_bend_heatmap_evidence(
        str(cad_path),
        str(scan_path),
        tolerance_angle=1.0,
        runtime_cfg=SimpleNamespace(local_refinement_kwargs={"deterministic_seed": 17}),
        cache_dir=tmp_path / "cache",
    )

    assert result is cached_bundle
