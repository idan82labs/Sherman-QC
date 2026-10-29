from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_professor_f1_v2_edge_stop_package_20260805 import (
    build_package,
)


def _source(tmp_path: Path, name: str, content: bytes = b"evidence") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _inputs(tmp_path: Path) -> dict:
    report = _source(
        tmp_path,
        "report.json",
        (json.dumps({
            "kind": "f1_v2_complete_edge_gold_free_causal_funnel_v1",
            "aggregate": {
                "added_same_surface_edge_count": 94_820,
                "frozen_two_sided_one_parent_target_count": 4_788,
                "extended_component_count": 320,
                "extended_valid_certificate_count": 15,
                "extended_control_equivalent_valid_certificate_count": 15,
                "extended_novel_valid_certificate_count": 0,
            },
        }) + "\n").encode("utf-8"),
    )
    return {
        "causal_funnel_report_path": report,
        "historical_anchor_pdf": _source(tmp_path, "anchor.pdf", b"%PDF-anchor"),
        "prior_ruling_path": _source(tmp_path, "ruling.md"),
        "batch_manifest_path": _source(tmp_path, "batch.json"),
        "replay_attestation_path": _source(tmp_path, "replay.json"),
        "synthetic_resource_attestation_path": _source(tmp_path, "synthetic.json"),
        "edge_core_path": _source(tmp_path, "edge.py"),
        "frozen_lpf_core_path": _source(tmp_path, "lpf.py"),
    }


def test_fresh_professor_package_is_ten_individual_files_with_explicit_placeholder(
    tmp_path: Path,
) -> None:
    result = build_package(
        output_dir=tmp_path / "package",
        corrected_bundle_path=None,
        expected_corrected_bundle_sha256=None,
        **_inputs(tmp_path),
    )
    assert result["individual_file_count"] == 10
    assert result["fresh_conversation_required"] is True
    names = {row["name"] for row in result["files"]}
    assert "02_CORRECTED_BUNDLE_HASH_PENDING.txt" in names
    assert "09_FeatureExtractionFromPointClouds.pdf" in names
    prompt = (tmp_path / "package/00_GPT_PRO_PROMPT.md").read_text()
    assert "primary sources only" in prompt
    assert "Qualify novelty narrowly" in prompt
    assert "Choose exactly one" in prompt
    assert "Do not authorize opening gold" in prompt


def test_professor_package_commits_known_corrected_bundle(tmp_path: Path) -> None:
    bundle = _source(tmp_path, "corrected.json", b'{"ready_for_gold":false}')
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    result = build_package(
        output_dir=tmp_path / "package",
        corrected_bundle_path=bundle,
        expected_corrected_bundle_sha256=digest,
        **_inputs(tmp_path),
    )
    assert result["corrected_bundle_known"] is True
    assert (tmp_path / "package/02_CORRECTED_COMPACT_BUNDLE.json").read_bytes() == bundle.read_bytes()


def test_professor_package_refuses_wrong_corrected_bundle_hash(tmp_path: Path) -> None:
    bundle = _source(tmp_path, "corrected.json")
    with pytest.raises(ValueError, match="differs from expected"):
        build_package(
            output_dir=tmp_path / "package",
            corrected_bundle_path=bundle,
            expected_corrected_bundle_sha256="0" * 64,
            **_inputs(tmp_path),
        )
