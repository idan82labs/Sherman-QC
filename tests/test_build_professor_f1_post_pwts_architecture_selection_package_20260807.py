from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

import scripts.build_professor_f1_post_pwts_architecture_selection_package_20260807 as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_exact_ten_file_architecture_selection_only_packet(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "packet"
    audit_path = tmp_path / "audit.json"
    result = builder.build_package(
        output_dir=output_dir,
        audit_path=audit_path,
        verify_raw_sources=False,
    )

    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    assert len(files) == 10
    assert result["individual_file_count"] == 10
    assert result["no_archive"] is True
    assert result["architecture_selection_only"] is True
    assert result["geometry_execution_authorized"] is False
    assert result["gold_or_truth_consumed"] is False
    assert result["runtime_or_production_authorized"] is False
    assert result["execution_authority"] == "NONE"
    assert result["embedded_classification_response_byte_exact"] is True
    assert result["classification_response_sha256"] == builder.CLASSIFICATION_RESPONSE_SHA256
    assert result["decision_tokens"] == list(builder.DECISION_TOKENS)
    assert result["next_step_tokens"] == list(builder.NEXT_STEP_TOKENS)
    assert audit_path.is_file()


def test_carries_nine_file_failure_capsule_byte_identical(tmp_path: Path) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        audit_path=None,
        verify_raw_sources=False,
    )
    for name, binding in builder.stop_packet.SOURCE_BINDINGS.items():
        assert (output_dir / name).read_bytes() == Path(binding["path"]).read_bytes()
        assert _sha(output_dir / name) == binding["sha256"]


def test_prompt_has_closed_candidates_bounds_research_and_safety(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        audit_path=None,
        verify_raw_sources=False,
    )
    prompt = (
        output_dir / "00_GPT_PRO_POST_PWTS_ARCHITECTURE_SELECTION.md"
    ).read_text(encoding="utf-8")

    for name, binding in builder.stop_packet.SOURCE_BINDINGS.items():
        assert f"`{name}` — `{binding['sha256']}`" in prompt
    for token in builder.DECISION_TOKENS:
        assert token in prompt
    for token in builder.NEXT_STEP_TOKENS:
        assert token in prompt
    assert len(builder.POSITIVE_TOKENS) == 3
    assert "24,117,184" in prompt
    assert "124,780,480" in prompt
    assert "19,398,592" in prompt
    assert "PROPOSED_ENGINEERING_CONSTANT_NOT_LITERATURE_ESTABLISHED" in prompt
    assert "2L-1 = 65,535" in prompt
    assert "S*L*K_cell = 3,145,728" in prompt
    assert "10.1145/777792.777840" in prompt
    assert "10.1016/j.cagd.2008.01.002" in prompt
    assert "10.1007/s00454-004-1146-y" in prompt
    assert "10.1016/j.gmod.2012.04.008" in prompt
    assert "10.1016/j.cagd.2004.09.004" in prompt
    assert "curvature plateau with no unique strict" in prompt
    assert "isotropic\nsame-trace tensor" in prompt
    assert "affine-only component" in prompt
    assert "source normals/colors" in prompt
    assert "No top-k, reservoir, tail omission" in prompt
    assert "selection authorizes another Pro packet" in prompt
    assert "EXECUTION_AUTHORITY=NONE" in prompt


def test_prompt_excludes_stopped_families_and_execution_authority(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        audit_path=None,
        verify_raw_sources=False,
    )
    prompt = (
        output_dir / "00_GPT_PRO_POST_PWTS_ARCHITECTURE_SELECTION.md"
    ).read_text(encoding="utf-8")
    assert "SELECT_PAIRED_WEDGE_TRANSITION_STRIP" not in prompt
    assert "SELECT_LOCAL_PLANE_FRAGMENT" not in prompt
    assert "does not permit geometry execution" in prompt
    assert "Do not prescribe or authorize an executable configuration" in prompt
    assert "gold/truth/expected counts" in prompt
    assert "PWTS and the stopped local-plane-fragment family are excluded" in prompt


def test_embeds_complete_classification_response_byte_exact(tmp_path: Path) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        audit_path=None,
        verify_raw_sources=False,
    )
    prompt = (
        output_dir / "00_GPT_PRO_POST_PWTS_ARCHITECTURE_SELECTION.md"
    ).read_text(encoding="utf-8")
    begin = "-----BEGIN VERBATIM PWTS STOP CLASSIFICATION RESPONSE-----\n"
    end = "-----END VERBATIM PWTS STOP CLASSIFICATION RESPONSE-----"
    embedded = prompt.split(begin, 1)[1].split(end, 1)[0]
    assert embedded == builder.CLASSIFICATION_RESPONSE_PATH.read_text(encoding="utf-8")


def test_fails_closed_when_classification_response_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(builder, "CLASSIFICATION_RESPONSE_PATH", changed)
    with pytest.raises(ValueError, match="differs from expected SHA-256"):
        builder.build_package(
            output_dir=tmp_path / "packet",
            audit_path=None,
            verify_raw_sources=False,
        )


def test_refuses_existing_output_directory_and_raw_ply_verification(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        builder.build_package(
            output_dir=existing,
            audit_path=None,
            verify_raw_sources=False,
        )
    with pytest.raises(ValueError, match="may not read raw PLY"):
        builder.build_package(
            output_dir=tmp_path / "raw",
            audit_path=None,
            verify_raw_sources=True,
        )


def test_validator_rejects_cardinality_mutation_and_extra_file(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        audit_path=None,
        verify_raw_sources=False,
    )
    prompt_path = output_dir / "00_GPT_PRO_POST_PWTS_ARCHITECTURE_SELECTION.md"
    original = prompt_path.read_text(encoding="utf-8")
    prompt_path.write_text(original.replace("24,117,184", "24,117,185"), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture-selection evidence missing"):
        builder._validate_packet(output_dir)

    prompt_path.write_text(original, encoding="utf-8")
    (output_dir / "extra.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 10"):
        builder._validate_packet(output_dir)


def test_fails_closed_when_failure_capsule_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = deepcopy(builder.stop_packet.SOURCE_BINDINGS)
    bindings["08_C001_BASE_A_STDERR.txt"]["sha256"] = "0" * 64
    monkeypatch.setattr(builder.stop_packet, "SOURCE_BINDINGS", bindings)
    with pytest.raises(ValueError, match="differs from expected SHA-256"):
        builder.build_package(
            output_dir=tmp_path / "packet",
            audit_path=None,
            verify_raw_sources=False,
        )
