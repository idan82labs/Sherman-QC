import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_raw_crease_scan_support_quality.py"
    spec = importlib.util.spec_from_file_location("evaluate_raw_crease_scan_support_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


quality = _load_module()


def _raw(count):
    return {
        "part_id": "demo",
        "scan_path": "/tmp/demo.ply",
        "raw_admissible_candidate_count_before_dedupe": count + 2,
        "admissible_candidate_count": count,
        "candidates": [{"candidate_id": f"RPC{i}", "admissible": True} for i in range(count)],
    }


def _bridge(safe_count, total_count, pair="FG1-FG2"):
    candidates = []
    for index in range(total_count):
        is_safe = index < safe_count
        candidates.append(
            {
                "raw_candidate_id": f"RPC{index}",
                "bridge_safe": is_safe,
                "best_flange_pair": pair.split("-"),
                "reason_codes": [] if is_safe else ["corridor_far_from_candidate"],
            }
        )
    return {
        "part_id": "demo",
        "scan_path": "/tmp/demo.ply",
        "family_bridge_candidate_count": total_count,
        "family_bridge_safe_candidate_count": safe_count,
        "candidates": candidates,
    }


def test_partial_case_below_target_routes_to_limited_observed_support():
    report = quality.evaluate_raw_crease_scan_support_quality(
        raw_creases=_raw(5),
        bridge_validation=_bridge(2, 5),
        human_target=5,
        track="partial",
    )

    assert report["decision"] == "limited_observed_support"
    assert report["recommended_gate"] == "scan_quality_gate"
    assert "bridge_safe_below_human_target" in report["reason_codes"]
    assert "partial_or_observed_scan_support_incomplete" in report["reason_codes"]


def test_clean_case_with_too_many_safe_candidates_routes_to_arrangement_needed():
    report = quality.evaluate_raw_crease_scan_support_quality(
        raw_creases=_raw(6),
        bridge_validation=_bridge(6, 6),
        human_target=4,
        track="known",
    )

    assert report["decision"] == "arrangement_needed"
    assert report["recommended_gate"] == "sparse_arrangement_gate"
    assert "bridge_safe_exceeds_human_target" in report["reason_codes"]


def test_matching_high_fraction_support_is_promotion_candidate_support():
    report = quality.evaluate_raw_crease_scan_support_quality(
        raw_creases=_raw(4),
        bridge_validation=_bridge(4, 4),
        human_target=4,
        track="known",
    )

    assert report["decision"] == "promotion_candidate_support"
    assert report["recommended_gate"] == "promotion_review"


def test_no_safe_support_abstains():
    report = quality.evaluate_raw_crease_scan_support_quality(
        raw_creases=_raw(3),
        bridge_validation=_bridge(0, 3),
        human_target=3,
        track="known",
    )

    assert report["decision"] == "insufficient_support_abstain"
    assert "no_bridge_safe_support" in report["reason_codes"]
