from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import scripts.build_professor_f1_v51_post_prospective_package_20260814 as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_hash_bound_gold_closed_packet(tmp_path: Path) -> None:
    output_dir = tmp_path / "packet"
    archive = tmp_path / "packet.zip"
    audit = tmp_path / "audit.json"
    result = builder.build_package(
        output_dir=output_dir, archive_path=archive, audit_path=audit
    )

    manifest = json.loads((output_dir / "MANIFEST.json").read_text())
    assert manifest["gold_truth_included"] is False
    assert manifest["per_case_locked_scores_included"] is False
    assert manifest["raw_scans_included"] is False
    assert manifest["source_only_action_requested"] is True
    assert manifest["file_count_excluding_manifest"] == len(builder.FILES)
    assert result["archive_sha256"] == _sha(archive)
    assert audit.is_file()

    with zipfile.ZipFile(archive) as package:
        assert sorted(package.namelist()) == sorted(
            [*builder.FILES, "MANIFEST.json"]
        )


def test_manifest_hashes_every_bound_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        archive_path=tmp_path / "packet.zip",
        audit_path=None,
    )
    manifest = json.loads((output_dir / "MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = output_dir / row["path"]
        assert path.stat().st_size == row["size_bytes"]
        assert _sha(path) == row["sha256"]


def test_prompt_requests_one_action_and_keeps_locked_truth_closed() -> None:
    prompt = (builder.ROOT / builder.FILES["00_GPT_PRO_PROMPT.md"]).read_text()
    for token in (
        "AUTHORIZE_FROZEN_V51_BLIND_GOLD37_EVALUATION_ONCE",
        "AUTHORIZE_ONE_SOURCE_ONLY_TRANSFER_CORRECTION_V52",
        "STOP_V51_TRANSFER_INSUFFICIENT_EVIDENCE",
    ):
        assert token in prompt
    assert "No per-case Gold37 truth" in prompt
    assert "no rescue tuning" in prompt.lower()
