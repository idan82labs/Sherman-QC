from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_professor_f1_v2_cross_parent_rung_package_20260805 import (
    build_package,
)


def _file(tmp_path: Path, name: str, content: bytes = b"evidence") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _common(tmp_path: Path) -> dict:
    return {
        "ruling_path": _file(tmp_path, "ruling.md"),
        "core_path": _file(tmp_path, "core.py"),
        "core_test_path": _file(tmp_path, "core_test.py"),
        "memory_test_path": _file(tmp_path, "memory_test.py"),
        "bundle_builder_path": _file(tmp_path, "bundle.py"),
        "corrected_baseline_bundle_path": _file(tmp_path, "baseline.json"),
        "prior_causal_report_path": _file(tmp_path, "causal.json"),
        "historical_anchor_pdf": _file(tmp_path, "anchor.pdf", b"%PDF-anchor"),
    }


def test_preflight_blocker_package_is_compact_and_explicit(tmp_path: Path) -> None:
    blocker = _file(
        tmp_path,
        "blocker.json",
        json.dumps({
            "decision": "INVALID_EXPERIMENT_STOP",
            "case_token": "C001",
            "frozen_target_component_id": "LTC_32d3609a79a9c314",
            "baseline_component_mapping_count": 0,
            "parent_line_distance_mm": 7.421545,
            "line_tube_tolerance_mm": 3.0,
            "gold_open_authorized": False,
        }).encode("utf-8"),
    )
    result = build_package(
        output_dir=tmp_path / "package",
        preflight_blocker_path=blocker,
        compact_bundle_path=None,
        batch_manifest_path=None,
        replay_attestation_path=None,
        synthetic_attestation_path=None,
        **_common(tmp_path),
    )
    assert result["mode"] == "PREFLIGHT_BLOCKER"
    assert result["individual_file_count"] <= 10
    assert result["gold_open_authorized"] is False
    prompt = (tmp_path / "package/00_GPT_PRO_PROMPT.md").read_text()
    assert "cannot enter its ordinary batch" in prompt
    assert "No authoritative batch, replay, synthetic result" in prompt
    assert "Do not ask us to silently remove the target" in prompt
    assert "gold_open_authorized` is always false" in prompt
    names = {row["name"] for row in result["files"]}
    assert "04_CORE_FAIL_CLOSED_TEST.py" in names
    assert "05_MEMORY_CAP_TEST.py" in names


def test_authoritative_package_accepts_only_exact_pre_gold_decision(tmp_path: Path) -> None:
    bundle = _file(
        tmp_path,
        "result.json",
        json.dumps({
            "pre_gold_decision": "STOP_NO_ELIGIBLE_CROSS_PARENT_RUNGS",
            "gold_open_authorized": False,
        }).encode("utf-8"),
    )
    result = build_package(
        output_dir=tmp_path / "package",
        preflight_blocker_path=None,
        compact_bundle_path=bundle,
        batch_manifest_path=_file(tmp_path, "batch.json"),
        replay_attestation_path=_file(tmp_path, "replay.json"),
        synthetic_attestation_path=_file(tmp_path, "synthetic.json"),
        **_common(tmp_path),
    )
    assert result["mode"] == "AUTHORITATIVE_RESULT"
    assert result["individual_file_count"] == 10
    assert result["gold_open_authorized"] is False


def test_package_rejects_gold_authorizing_bundle(tmp_path: Path) -> None:
    bundle = _file(
        tmp_path,
        "result.json",
        json.dumps({
            "pre_gold_decision": "GO_NEXT_PROFESSOR_PACKET",
            "gold_open_authorized": True,
        }).encode("utf-8"),
    )
    with pytest.raises(ValueError, match="gold-authorizing"):
        build_package(
            output_dir=tmp_path / "package",
            preflight_blocker_path=None,
            compact_bundle_path=bundle,
            batch_manifest_path=_file(tmp_path, "batch.json"),
            replay_attestation_path=_file(tmp_path, "replay.json"),
            synthetic_attestation_path=_file(tmp_path, "synthetic.json"),
            **_common(tmp_path),
        )
