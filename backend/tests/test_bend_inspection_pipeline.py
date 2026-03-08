"""Tests for shared bend inspection pipeline utilities."""

import json
import sys
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import bend_inspection_pipeline as pipeline
from feature_detection.bend_detector import BendInspectionReport, BendSpecification, BendMatch


def test_build_not_detected_match_returns_bendmatch():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    matcher = pipeline.ProgressiveBendMatcher()

    match = pipeline._build_not_detected_match(
        cad_bend=cad_bend,
        matcher=matcher,
        action_item="Need more scan coverage",
    )

    assert isinstance(match, BendMatch)
    assert match.status == "NOT_DETECTED"
    assert match.action_item == "Need more scan coverage"


def test_build_not_detected_match_supports_observability_fields():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    matcher = pipeline.ProgressiveBendMatcher()
    match = pipeline._build_not_detected_match(
        cad_bend=cad_bend,
        matcher=matcher,
        action_item="Observed but still flat",
        observability_state="OBSERVED_NOT_FORMED",
        observability_confidence=0.72,
        observed_surface_count=2,
        local_point_count=44,
        local_evidence_score=0.63,
        measurement_mode="cad_local_neighborhood",
        measurement_method="surface_classification",
        measurement_context={"search_radius_mm": 35.0},
    )

    assert match.observability_state == "OBSERVED_NOT_FORMED"
    assert match.observed_surface_count == 2
    assert match.measurement_mode == "cad_local_neighborhood"


def test_local_measurement_observability_classifies_flat_state():
    measurement = {
        "success": True,
        "measured_angle": 176.0,
        "measurement_confidence": "high",
        "point_count": 80,
        "side1_points": 40,
        "side2_points": 40,
        "cad_alignment1": 0.99,
        "cad_alignment2": 0.98,
    }
    obs = pipeline._local_measurement_observability(measurement, cad_angle=90.0)
    assert obs["state"] == "OBSERVED_NOT_FORMED"
    assert obs["observed_surface_count"] == 2


def test_load_cad_geometry_step_uses_cad_import(monkeypatch, tmp_path):
    step_path = tmp_path / "part.stp"
    step_path.write_text("dummy-step", encoding="utf-8")

    class _DummyMesh:
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

    class _DummyResult:
        success = True
        mesh = _DummyMesh()
        error = None

    seen = {}

    def _fake_import(path: str, deflection: float):
        seen["path"] = path
        seen["deflection"] = deflection
        return _DummyResult()

    monkeypatch.setattr(pipeline, "import_cad_file", _fake_import)

    verts, faces, source = pipeline.load_cad_geometry(str(step_path), cad_import_deflection=0.07)

    assert seen["path"] == str(step_path)
    assert abs(seen["deflection"] - 0.07) < 1e-9
    assert source == "cad_import"
    assert verts.shape == (3, 3)
    assert faces.shape == (1, 3)


def test_load_bend_runtime_config_supports_wrapped_payload(tmp_path):
    config_path = tmp_path / "config.json"
    payload = {
        "name": "best",
        "runtime_config": {
            "scan_detector_kwargs": {"min_plane_points": 33},
            "matcher_kwargs": {"max_angle_diff": 13.0},
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = pipeline.load_bend_runtime_config(str(config_path))

    assert cfg.scan_detector_kwargs["min_plane_points"] == 33
    assert cfg.matcher_kwargs["max_angle_diff"] == 13.0


def test_runtime_config_deepcopy_isolation():
    raw = {
        "scan_detector_kwargs": {"min_plane_points": 12},
        "matcher_kwargs": {"max_angle_diff": 11.0},
    }
    cfg = pipeline.BendInspectionRuntimeConfig.from_dict(raw)
    raw["scan_detector_kwargs"]["min_plane_points"] = 99
    assert cfg.scan_detector_kwargs["min_plane_points"] == 12

    serialized = cfg.to_dict()
    serialized["scan_detector_kwargs"]["min_plane_points"] = 77
    assert cfg.scan_detector_kwargs["min_plane_points"] == 12


def test_run_progressive_bend_inspection_surfaces_warnings(monkeypatch):
    dummy_report = BendInspectionReport(part_id="PART", matches=[])

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), None, "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: ([], "detector_point_cloud"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((30, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_scan_bends",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: dummy_report,
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert report is dummy_report
    assert details.cad_source == "mock"
    assert details.cad_bend_count == 0
    assert details.scan_bend_count == 0
    assert details.overdetected_count == 0
    assert "No bends extracted from CAD" in details.warnings
    assert "No bends detected in scan" in details.warnings


def test_run_progressive_bend_inspection_applies_expected_override(monkeypatch, tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            [
                {
                    "part": "PART",
                    "drawing_expected_bends": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BEND_EXPECTED_OVERRIDES", str(override_path))

    cad_bends = [
        BendSpecification(
            bend_id=f"B{i+1}",
            target_angle=90.0,
            target_radius=2.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0.0, 10.0, 0.0]),
        )
        for i in range(6)
    ]
    matches = []
    for i, b in enumerate(cad_bends):
        status = "PASS" if i < 3 else "NOT_DETECTED"
        matches.append(BendMatch(cad_bend=b, detected_bend=None, status=status))
    report = BendInspectionReport(part_id="PART", matches=matches)
    report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), None, "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "detector_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((30, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_scan_bends",
        lambda **kwargs: [object()] * 3,
    )
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: report,
    )

    _, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert details.cad_bend_count == 6
    assert details.expected_bend_count == 3
    assert abs(details.expected_progress_pct - 100.0) < 1e-9
    assert details.overdetected_count == 0
    assert any("Expected bend override applied" in w for w in details.warnings)


def test_run_progressive_bend_inspection_caps_expected_progress(monkeypatch, tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps([{"part": "PART", "drawing_expected_bends": 3}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("BEND_EXPECTED_OVERRIDES", str(override_path))

    cad_bends = [
        BendSpecification(
            bend_id=f"B{i+1}",
            target_angle=90.0,
            target_radius=2.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0.0, 10.0, 0.0]),
        )
        for i in range(6)
    ]
    matches = [BendMatch(cad_bend=b, detected_bend=None, status="PASS") for b in cad_bends]
    report = BendInspectionReport(part_id="PART", matches=matches)
    report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), None, "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "detector_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((30, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_scan_bends",
        lambda **kwargs: [object()] * 6,
    )
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: report,
    )

    _, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert details.expected_bend_count == 3
    assert details.overdetected_count == 3
    assert abs(details.expected_progress_pct - 100.0) < 1e-9
    assert any("Over-detected bends vs expected baseline" in w for w in details.warnings)


def test_estimate_scan_quality_flags_good_scan():
    x = np.linspace(0.0, 120.0, 80)
    y = np.linspace(0.0, 80.0, 60)
    xx, yy = np.meshgrid(x, y)
    cad = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    rng = np.random.default_rng(42)
    scan = cad + rng.normal(0.0, 0.15, cad.shape)

    quality = pipeline.estimate_scan_quality(
        scan_points=scan,
        cad_points=cad,
        expected_bends=8,
        completed_bends=6,
        passed_bends=6,
    )

    assert quality["status"] in {"GOOD", "FAIR"}
    assert quality["coverage_pct"] > 55.0
    assert quality["density_pts_per_cm2"] > 10.0


def test_estimate_scan_quality_flags_poor_scan():
    x = np.linspace(0.0, 120.0, 80)
    y = np.linspace(0.0, 80.0, 60)
    xx, yy = np.meshgrid(x, y)
    cad = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    # Sparse partial corner capture with low point count and limited coverage.
    sparse_idx = np.where((cad[:, 0] < 30.0) & (cad[:, 1] < 20.0))[0]
    scan = cad[sparse_idx][::8]

    quality = pipeline.estimate_scan_quality(
        scan_points=scan,
        cad_points=cad,
        expected_bends=8,
        completed_bends=1,
        passed_bends=1,
    )

    assert quality["status"] == "POOR"
    assert (
        quality["coverage_pct"] < 45.0
        or quality["density_pts_per_cm2"] < 8.0
    )


def _make_spec(
    bend_id: str,
    *,
    angle: float = 90.0,
    radius: float = 3.0,
    bend_form: str = "FOLDED",
) -> BendSpecification:
    return BendSpecification(
        bend_id=bend_id,
        target_angle=angle,
        target_radius=radius,
        bend_line_start=np.zeros(3),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
        flange1_area=100.0,
        flange2_area=100.0,
        bend_form=bend_form,
    )


def _make_match(
    cad_bend: BendSpecification,
    status: str,
    *,
    angle_deviation: float = 0.0,
    confidence: float = 0.8,
) -> BendMatch:
    return BendMatch(
        cad_bend=cad_bend,
        detected_bend=None,
        status=status,
        angle_deviation=angle_deviation,
        match_confidence=confidence,
    )


def test_merge_bend_reports_preserves_primary_detection_when_local_misses():
    b1 = _make_spec("B1")
    b2 = _make_spec("B2")
    b3 = _make_spec("B3")
    primary = BendInspectionReport(
        part_id="PART",
        matches=[
            _make_match(b1, "PASS", angle_deviation=0.4),
            _make_match(b2, "NOT_DETECTED"),
            _make_match(b3, "PASS", angle_deviation=0.6),
        ],
    )
    primary.compute_summary()
    local = BendInspectionReport(
        part_id="PART",
        matches=[
            _make_match(b1, "FAIL", angle_deviation=8.0),
            _make_match(b2, "FAIL", angle_deviation=6.0),
            _make_match(b3, "NOT_DETECTED"),
        ],
    )
    local.compute_summary()

    merged = pipeline.merge_bend_reports(primary, local, part_id="PART")

    assert merged.detected_count == 3
    assert [m.status for m in merged.matches] == ["PASS", "FAIL", "PASS"]


def test_merge_bend_reports_prefers_authoritative_local_absence_for_weak_global_match():
    b1 = _make_spec("B1")
    primary = BendInspectionReport(
        part_id="PART",
        matches=[_make_match(b1, "PASS", angle_deviation=0.4, confidence=0.78)],
    )
    primary.compute_summary()
    local = BendInspectionReport(
        part_id="PART",
        matches=[_make_match(b1, "NOT_DETECTED")],
    )
    local.compute_summary()

    merged = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        local_alignment_fitness=0.87,
    )
    assert merged.detected_count == 0
    assert merged.matches[0].status == "NOT_DETECTED"

    merged_low_fit = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        local_alignment_fitness=0.55,
    )
    assert merged_low_fit.detected_count == 1
    assert merged_low_fit.matches[0].status == "PASS"


def test_align_engine_with_retries_keeps_first_good_alignment():
    class _Aligned:
        def __init__(self, pts):
            self.points = pts

    class _Engine:
        def __init__(self):
            self.calls = []
            self.aligned_scan = _Aligned(np.zeros((2, 3), dtype=np.float64))

        def align(self, auto_scale=True, tolerance=1.0, random_seed=None):
            self.calls.append(random_seed)
            self.aligned_scan = _Aligned(np.full((2, 3), float(random_seed), dtype=np.float64))
            return 0.91, 0.25

    phase_events = []
    best_points, fitness, rmse, seed, warnings_local = pipeline._align_engine_with_retries(
        engine=_Engine(),
        part_id="PART",
        tolerance_angle=1.0,
        refinement_cfg={
            "deterministic_seed": 17,
            "retry_seeds": [29, 43],
            "retry_below_fitness": 0.65,
            "min_alignment_fitness": 0.55,
        },
        phase_cb=phase_events.append,
    )

    assert best_points is not None
    assert fitness == 0.91
    assert rmse == 0.25
    assert seed == 17
    assert warnings_local == []
    assert any("align attempt begin seed=17" in msg for msg in phase_events)


def test_align_engine_with_retries_selects_best_retry_and_discards_untrusted_fit():
    class _Aligned:
        def __init__(self, pts):
            self.points = pts

    class _Engine:
        def __init__(self):
            self.calls = []
            self.aligned_scan = _Aligned(np.zeros((2, 3), dtype=np.float64))
            self.results = {
                17: (0.30, 1.40),
                29: (0.88, 0.95),
                43: (0.51, 1.10),
            }

        def align(self, auto_scale=True, tolerance=1.0, random_seed=None):
            self.calls.append(random_seed)
            self.aligned_scan = _Aligned(np.full((2, 3), float(random_seed), dtype=np.float64))
            return self.results[int(random_seed)]

    engine = _Engine()
    best_points, fitness, rmse, seed, warnings_local = pipeline._align_engine_with_retries(
        engine=engine,
        part_id="PART",
        tolerance_angle=1.0,
        refinement_cfg={
            "deterministic_seed": 17,
            "retry_seeds": [29, 43],
            "retry_below_fitness": 0.65,
            "min_alignment_fitness": 0.55,
        },
        phase_cb=lambda _: None,
    )

    assert engine.calls == [17, 29, 43]
    assert best_points is not None
    assert np.all(best_points == 29.0)
    assert fitness == 0.88
    assert rmse == 0.95
    assert seed == 29
    assert any("retried alternate deterministic alignment seeds" in w for w in warnings_local)

    low_engine = _Engine()
    low_engine.results = {
        17: (0.30, 1.40),
        29: (0.45, 1.25),
        43: (0.51, 1.10),
    }
    best_points, fitness, rmse, seed, warnings_local = pipeline._align_engine_with_retries(
        engine=low_engine,
        part_id="PART",
        tolerance_angle=1.0,
        refinement_cfg={
            "deterministic_seed": 17,
            "retry_seeds": [29, 43],
            "retry_below_fitness": 0.65,
            "min_alignment_fitness": 0.55,
        },
        phase_cb=lambda _: None,
    )

    assert best_points is None
    assert fitness == 0.51
    assert seed == 43
    assert any("discarded: alignment fitness" in w for w in warnings_local)


def test_choose_mesh_cad_baseline_prefers_expected_hint():
    detector_specs = [_make_spec(f"D{i}", angle=90.0, radius=12.0, bend_form="ROLLED") for i in range(6)]
    legacy_specs = [_make_spec(f"L{i}") for i in range(10)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=10,
    )

    assert selected == legacy_specs
    assert strategy == "legacy_mesh_expected_hint"


def test_choose_mesh_cad_baseline_prefers_legacy_on_tray_like_rolled_undercount():
    detector_specs = [_make_spec(f"D{i}", angle=90.0, radius=12.0, bend_form="ROLLED") for i in range(6)]
    legacy_specs = [_make_spec(f"L{i}", angle=89.0) for i in range(11)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
    )

    assert selected == legacy_specs
    assert strategy == "legacy_mesh_tray_override"


def test_choose_mesh_cad_baseline_keeps_detector_without_clear_legacy_advantage():
    detector_specs = [_make_spec(f"D{i}", angle=60.0, radius=3.0, bend_form="FOLDED") for i in range(6)]
    legacy_specs = [_make_spec(f"L{i}", angle=45.0) for i in range(7)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
    )

    assert selected == detector_specs
    assert strategy == "detector_mesh"


def test_run_progressive_bend_inspection_passes_expected_hint_to_strategy(monkeypatch, tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps([{"part": "PART", "drawing_expected_bends": 10}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("BEND_EXPECTED_OVERRIDES", str(override_path))

    seen = {}
    cad_bends = [_make_spec("B1")]
    dummy_report = BendInspectionReport(
        part_id="PART",
        matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED")],
    )
    dummy_report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )

    def _fake_extract(**kwargs):
        seen["expected_hint"] = kwargs.get("expected_bend_count_hint")
        return cad_bends, "legacy_mesh_expected_hint"

    monkeypatch.setattr(pipeline, "extract_cad_bend_specs_with_strategy", _fake_extract)
    monkeypatch.setattr(pipeline, "load_scan_points", lambda scan_path: np.zeros((30, 3)))
    monkeypatch.setattr(
        pipeline,
        "detect_and_match_with_fallback",
        lambda **kwargs: (dummy_report, 0, False),
    )

    _, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert seen["expected_hint"] == 10
    assert details.cad_strategy == "legacy_mesh_expected_hint"


def test_apply_scan_runtime_profile_adapts_large_scans():
    ctor, call, profile = pipeline._apply_scan_runtime_profile(
        {"min_plane_points": 50, "min_plane_area": 30.0},
        {"preprocess": True},
        715_498,
        fallback=False,
    )

    assert profile == "full_scan_dense_primary"
    assert ctor["min_plane_points"] >= 55
    assert call["voxel_size"] >= 2.0
    assert call["multiscale"] is False


def test_detect_and_match_with_fallback_uses_adaptive_profile(monkeypatch):
    seen_calls = []
    cad_bends = [_make_spec("B1")]

    def _fake_detect_scan_bends(**kwargs):
        seen_calls.append(kwargs["scan_detect_call_kwargs"])
        return []

    monkeypatch.setattr(pipeline, "detect_scan_bends", _fake_detect_scan_bends)
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: BendInspectionReport(
            part_id="PART",
            matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED")],
        ),
    )

    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    report, selected_count, fallback_used = pipeline.detect_and_match_with_fallback(
        scan_points=np.zeros((500_000, 3)),
        cad_bends=cad_bends,
        part_id="PART",
        runtime_config=runtime_cfg,
        expected_bend_count=10,
    )

    assert report.part_id == "PART"
    assert selected_count == 0
    assert len(seen_calls) == 2
    assert seen_calls[0]["voxel_size"] >= 2.0
    assert seen_calls[0]["multiscale"] is False
    assert seen_calls[1]["voxel_size"] >= 1.75
    assert seen_calls[1]["multiscale"] is False


def test_detect_and_match_with_fallback_skips_dense_global_fallback(monkeypatch):
    seen_calls = []
    cad_bends = [_make_spec("B1")]

    def _fake_detect_scan_bends(**kwargs):
        seen_calls.append(kwargs["scan_detect_call_kwargs"])
        return []

    monkeypatch.setattr(pipeline, "detect_scan_bends", _fake_detect_scan_bends)
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: BendInspectionReport(
            part_id="PART",
            matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED")],
        ),
    )

    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    _, _, fallback_used = pipeline.detect_and_match_with_fallback(
        scan_points=np.zeros((715_498, 3)),
        cad_bends=cad_bends,
        part_id="PART",
        runtime_config=runtime_cfg,
        expected_bend_count=10,
    )

    assert fallback_used is False
    assert len(seen_calls) == 1
    assert seen_calls[0]["voxel_size"] >= 2.0


def test_run_progressive_bend_inspection_uses_dense_local_refinement(monkeypatch):
    cad_bends = [_make_spec("B1"), _make_spec("B2")]
    weak_report = BendInspectionReport(
        part_id="PART",
        matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="PASS"), BendMatch(cad_bend=cad_bends[1], detected_bend=None, status="NOT_DETECTED")],
    )
    weak_report.compute_summary()
    refined_report = BendInspectionReport(
        part_id="PART",
        matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="PASS"), BendMatch(cad_bend=cad_bends[1], detected_bend=None, status="PASS")],
    )
    refined_report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "legacy_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((600_000, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_and_match_with_fallback",
        lambda **kwargs: (weak_report, 1, False),
    )
    monkeypatch.setattr(
        pipeline,
        "refine_dense_scan_bends_via_alignment",
        lambda **kwargs: (refined_report, ["Dense-scan local refinement used CAD-guided ICP alignment (voxel=2.00mm, fitness=99.00%, rmse=0.100mm)"]),
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert report.detected_count == 2
    assert details.scan_bend_count == 2
    assert any("Dense-scan local refinement used CAD-guided ICP alignment" in w for w in details.warnings)
    assert any("Dense-scan local refinement promoted over primary global detection" in w for w in details.warnings)


def test_run_progressive_bend_inspection_merges_local_refinement_without_erasing_primary(monkeypatch):
    cad_bends = [_make_spec("B1"), _make_spec("B2"), _make_spec("B3")]
    primary_report = BendInspectionReport(
        part_id="PART",
        matches=[
            _make_match(cad_bends[0], "PASS", angle_deviation=0.5),
            _make_match(cad_bends[1], "NOT_DETECTED"),
            _make_match(cad_bends[2], "PASS", angle_deviation=0.7),
        ],
    )
    primary_report.compute_summary()
    local_report = BendInspectionReport(
        part_id="PART",
        matches=[
            _make_match(cad_bends[0], "NOT_DETECTED"),
            _make_match(cad_bends[1], "FAIL", angle_deviation=7.0),
            _make_match(cad_bends[2], "FAIL", angle_deviation=12.0),
        ],
    )
    local_report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "legacy_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((600_000, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_and_match_with_fallback",
        lambda **kwargs: (primary_report, 2, False),
    )
    monkeypatch.setattr(
        pipeline,
        "refine_dense_scan_bends_via_alignment",
        lambda **kwargs: (local_report, ["Dense-scan local refinement used CAD-guided ICP alignment (voxel=1.50mm, fitness=90.00%, rmse=1.000mm)"]),
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert report.detected_count == 3
    assert [m.status for m in report.matches] == ["PASS", "FAIL", "PASS"]
    assert details.scan_bend_count == 3
    assert any("Dense-scan local refinement merged with primary global detection" in w for w in details.warnings)
