from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_f1_v2_complete_edge_causal_funnel_report import (
    EXPECTED_CASE_TOKENS,
    ROW_FIELDS,
    _frozen_targets,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "output/research/f1-gold27-20260803"


def _control_component(identifier: str, *, qualifying: bool) -> dict:
    return {
        "local_transition_component_id": identifier,
        "support_parent_count": 1,
        "support_mode_count": 2,
        "left_side_bin_count": 1,
        "right_side_bin_count": 1,
        "transition_rung_bin_count": 1,
        "intersection_span_endpoints": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0] if qualifying else [1.0, 0.0, 0.0]],
        "witness_ids_truncated": False,
        "witness_mode_ids": ["M2", "M1"],
    }


def test_frozen_target_extraction_is_exact_and_hash_committed() -> None:
    control = {
        "local_plane_pair_records": [{
            "components": [
                _control_component("LTC_YES", qualifying=True),
                _control_component("LTC_NO", qualifying=False),
            ]
        }]
    }
    targets = _frozen_targets(control)
    assert len(targets) == 1
    assert targets[0]["frozen_control_component_id"] == "LTC_YES"
    assert targets[0]["mode_ids"] == ("M1", "M2")
    expected = hashlib.sha256(
        json.dumps(
            ["M1", "M2"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    assert targets[0]["member_mode_ids_sha256"] == expected


def test_real_gold_free_report_recomputes_all_320_without_truth() -> None:
    bundle_path = RESEARCH / "v2_complete_same_surface_edge_probe_v1_bundle.json"
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    report = build_report(
        replacement_bundle_path=bundle_path,
        expected_bundle_sha256=bundle_sha,
        probe_root=RESEARCH / "v2_complete_same_surface_edge_probe_v1",
        candidate_pool_manifest_path=RESEARCH / "candidate_pool_transition_v2_probe/manifest.json",
        frozen_control_manifest_path=RESEARCH / "local_plane_fragment_probe_v2/manifest.json",
    )
    assert report["truth_inputs_consumed"] is False
    assert report["source_points_consumed"] is False
    assert report["case_count"] == len(EXPECTED_CASE_TOKENS)
    assert report["aggregate"]["frozen_two_sided_one_parent_target_count"] == 4_788
    assert report["aggregate"]["extended_component_count"] == 320
    assert report["aggregate"]["accepted_added_union_edge_count"] == 10_652
    assert report["aggregate"]["edge_dependent_fragment_count"] == 253
    assert report["aggregate"]["edge_dependent_component_count"] == 14_896
    assert report["aggregate"]["edge_dependent_valid_certificate_count"] == 15
    assert report["aggregate"]["control_equivalent_edge_dependent_valid_certificate_count"] == 15
    assert report["aggregate"]["novel_edge_dependent_valid_certificate_count"] == 0
    assert len(report["rows"]) == 320
    assert all(set(row) == ROW_FIELDS for row in report["rows"])
    assert report["aggregate"]["extended_novel_valid_certificate_count"] == 0
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    assert "mode_ids\"" not in encoded
    assert "manual_bends" not in encoded
