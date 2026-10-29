from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_exp_cad_0025_analytic_brep_carrier_orientation.py"
SPEC = importlib.util.spec_from_file_location("exp25_carriers", PATH)
EXP25 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP25)


def carrier(*, full_circle: bool) -> dict:
    return {
        "face_id": "synthetic",
        "midpoint_mm": [0.0, 0.0, 0.0],
        "axis": [0.0, 0.0, 1.0],
        "radius_mm": 10.0,
        "full_circle": full_circle,
        "angular_start_rad": 0.0,
        "angular_span_rad": np.pi / 2.0,
        "angular_reference": [1.0, 0.0, 0.0],
        "angular_tangent": [0.0, 1.0, 0.0],
        "asymmetric_profile_carrier": not full_circle,
    }


def test_canonical_axis_is_unoriented_and_unit() -> None:
    first = EXP25.canonical_unoriented_axis(np.array([0.0, -2.0, 0.0]))
    second = EXP25.canonical_unoriented_axis(np.array([0.0, 5.0, 0.0]))
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_partial_carrier_enforces_analytic_angular_trim() -> None:
    angles = np.array([0.0, np.pi / 4.0, np.pi, 0.0])
    points = np.column_stack((10.0 * np.cos(angles), 10.0 * np.sin(angles), [0.0, 0.0, 0.0, 2.0]))
    partial = EXP25.carrier_masks(
        points, [carrier(full_circle=False)], sheet_thickness_mm=2.0, tau_mm=0.35
    )[0]
    full = EXP25.carrier_masks(
        points, [carrier(full_circle=True)], sheet_thickness_mm=2.0, tau_mm=0.35
    )[0]
    assert partial.tolist() == [True, True, False, False]
    assert full.tolist() == [True, True, True, False]


def test_ranking_prioritizes_asymmetric_support_before_total_density() -> None:
    clusters = [
        {"members": ["A"], "representative": {"candidate_id": "A"}},
        {"members": ["B"], "representative": {"candidate_id": "B"}},
    ]
    metrics = {
        ("A",): {
            "asymmetric_supported_carrier_count": 6,
            "supported_carrier_count": 15,
            "sum_log1p_count": 20.0,
            "total_count": 100,
        },
        ("B",): {
            "asymmetric_supported_carrier_count": 5,
            "supported_carrier_count": 15,
            "sum_log1p_count": 200.0,
            "total_count": 10000,
        },
    }
    ranked = EXP25.rank_clusters(clusters, metrics)
    assert ranked[0]["representative"]["candidate_id"] == "A"


def test_frozen_source_inventory_has_six_asymmetric_of_fifteen() -> None:
    assert len(EXP25.EXPECTED_CARRIER_FACE_IDS) == 15
    assert len(EXP25.ASYMMETRIC_PROFILE_FACE_IDS) == 6
    assert EXP25.ASYMMETRIC_PROFILE_FACE_IDS.issubset(set(EXP25.EXPECTED_CARRIER_FACE_IDS))


def test_decoded_source_yields_exact_analytic_inventory() -> None:
    source = ROOT / "data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/50476000_A.SLDPRT"
    cadmpeg = Path(
        "/Users/idantal/.local/share/codex-tools/cadmpeg/v0.5.5/"
        "cadmpeg-aarch64-apple-darwin/cadmpeg"
    )
    summary_path = ROOT / (
        "output/research/autoresearch-loop/exp-cad-0017-native-roundbend-patch-axes-v1/"
        "family-50476000_A/SOURCE_ONLY_SUMMARY.json"
    )
    if not source.exists() or not cadmpeg.exists() or not summary_path.exists():
        pytest.skip("frozen EXP25 integration dependencies unavailable")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cadir, report = EXP25.decode_cadir(source, cadmpeg)
    assert report["decode_report"]["geometry_transferred"] is True
    event_faces = [
        face_id
        for event in summary["extraction"]["events"]
        for face_id in event["face_ids"]
    ]
    body_id, shell_id, carriers = EXP25.analytic_through_thickness_carriers(
        cadir,
        native_event_face_ids=event_faces,
        sheet_thickness_mm=2.0,
    )
    assert body_id == "sldprt:brep:body#3"
    assert shell_id == "sldprt:brep:shell#7"
    assert [row["face_id"] for row in carriers] == EXP25.EXPECTED_CARRIER_FACE_IDS
    assert sum(row["asymmetric_profile_carrier"] for row in carriers) == 6
    partial_degrees = [
        np.degrees(row["angular_span_rad"]) for row in carriers if not row["full_circle"]
    ]
    assert np.allclose(partial_degrees, [100.0, 80.325872, 80.180114, 100.145904], atol=1e-5)
