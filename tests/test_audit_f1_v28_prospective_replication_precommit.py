from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.audit_f1_v28_prospective_replication_precommit import (
    EXPECTED_V27_SHA256,
    EXPECTED_V28_SHA256,
    TERMINAL_TOKEN,
    build_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_f1_v28_prospective_replication_precommit.py"


def test_frozen_sources_match_professor_hashes():
    manifest = build_manifest(ROOT)
    assert manifest["frozen_sources"]["v27"]["actual_sha256"] == EXPECTED_V27_SHA256
    assert manifest["frozen_sources"]["v28"]["actual_sha256"] == EXPECTED_V28_SHA256


def test_rank_is_extracted_but_fitted_axis_bytes_are_not_order_canonical():
    manifest = build_manifest(ROOT)
    rank = manifest["rank_extraction"]
    assert rank["directions"] == [
        "ascending",
        "descending_via_negation",
        "ascending",
    ]
    assert rank["floating_point_canonicalization"] == "none"
    assert rank["total_order_for_distinct_candidate_ids"] is True
    group = manifest["candidate_group_construction"]
    assert group["eligible_appended_in_input_order"] is True
    assert group["fit_point_order_uses_component_input_indices"] is True
    assert group["fitted_axis_sign_anchors_component_first_member"] is True
    assert group["canonical_fitted_axis_bytes_proven_input_order_independent"] is False


def test_missing_post_v28_nms_is_terminal_precommit_failure():
    manifest = build_manifest(ROOT)
    pipeline = manifest["pipeline_contract"]
    assert pipeline["ordinary_output_nms_after_v28_present"] is False
    assert pipeline["existing_lane_survives_v28_duplicate_collision_enforced"] is False
    assert manifest["precommit_pass"] is False
    assert manifest["terminal_token"] == TERMINAL_TOKEN
    assert "POST_V28_ORDINARY_NMS_ABSENT" in manifest["terminal_failures"]
    assert manifest["fixtures_executed"] is False
    assert manifest["real_scans_executed"] is False
    assert manifest["detector_invoked"] is False


def test_cli_is_byte_deterministic(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for path in (first, second):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(ROOT), "--output", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        result = json.loads(completed.stdout)
        assert result["terminal_token"] == TERMINAL_TOKEN
    assert first.read_bytes() == second.read_bytes()
