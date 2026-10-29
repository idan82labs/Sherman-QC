from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import scripts.build_professor_f1_post_v51_protocol_invalidation_package_20260814 as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_truth_free_incident_packet(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    archive = tmp_path / "packet.zip"
    audit = tmp_path / "audit.json"
    result = builder.build_package(output_dir=output, archive_path=archive, audit_path=audit)
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert manifest["gold_truth_included"] is False
    assert manifest["gold_case_mapping_included"] is False
    assert manifest["per_case_gold_score_included"] is False
    assert manifest["raw_scans_included"] is False
    assert manifest["v51_gold37_rerun_requested"] is False
    assert result["archive_sha256"] == _sha(archive)
    with zipfile.ZipFile(archive) as package:
        assert sorted(package.namelist()) == sorted([*builder.FILES, "MANIFEST.json"])


def test_manifest_hashes_every_file(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    builder.build_package(output_dir=output, archive_path=tmp_path / "packet.zip", audit_path=None)
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = output / row["path"]
        assert path.stat().st_size == row["size_bytes"]
        assert _sha(path) == row["sha256"]


def test_prompt_requests_one_action_and_forbids_v51_gold37_rerun() -> None:
    prompt = (builder.ROOT / builder.FILES["00_GPT_PRO_PROMPT.md"]).read_text()
    for token in (
        "AUTHORIZE_FROZEN_V51_ON_NEW_UNTOUCHED_CORPUS_ONLY",
        "AUTHORIZE_ONE_SOURCE_ONLY_SUCCESSOR_V52",
        "OPEN_SMALLEST_SCIENTIFICALLY_NECESSARY_CAPABILITY",
    ):
        assert token in prompt
    assert "Do not authorize another V51 Gold37 run" in prompt
    assert "No result is being hidden" in prompt
