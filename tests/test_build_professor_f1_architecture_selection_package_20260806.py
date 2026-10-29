from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

import scripts.build_professor_f1_architecture_selection_package_20260806 as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_exact_ten_file_decision_only_packet(tmp_path: Path) -> None:
    output_dir = tmp_path / "packet"
    audit_path = tmp_path / "audit.json"
    result = builder.build_package(output_dir=output_dir, audit_path=audit_path)

    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    assert len(files) == 10
    assert result["individual_file_count"] == 10
    assert result["no_archive"] is True
    assert result["decision_only"] is True
    assert result["architecture_executed"] is False
    assert result["gold_open_authorized"] is False
    assert result["runtime_integration_authorized"] is False
    assert audit_path.is_file()

    assert (output_dir / "01_FORMAL_LPF_FAMILY_STOP.md").read_bytes() == Path(
        builder.SOURCE_BINDINGS["formal_lpf_stop"]["path"]
    ).read_bytes()
    assert (output_dir / "02_ORDINARY_COMPACT_EVIDENCE.json").read_bytes() == Path(
        builder.SOURCE_BINDINGS["ordinary_compact_evidence"]["path"]
    ).read_bytes()


def test_prompt_binds_all_nine_evidence_files_and_exact_decision_set(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "packet"
    result = builder.build_package(output_dir=output_dir, audit_path=None)
    prompt = (output_dir / "00_GPT_PRO_PROMPT.md").read_text(encoding="utf-8")
    matrix = (output_dir / "07_EXACT_DECISION_MATRIX.md").read_text(
        encoding="utf-8"
    )

    for path in sorted(output_dir.iterdir()):
        if path.name == "00_GPT_PRO_PROMPT.md":
            continue
        assert f"`{path.name}` — `{_sha(path)}`" in prompt

    assert result["positive_tokens"] == list(builder.POSITIVE_TOKENS)
    assert result["fail_closed_token"] == builder.STOP_TOKEN
    for token in (*builder.POSITIVE_TOKENS, builder.STOP_TOKEN):
        assert token in prompt
        assert token in matrix

    for forbidden_choice in (
        "SELECT_LOCAL_PLANE_FRAGMENT",
        "SELECT_CAD_REFERENCE",
        "SELECT_MESH",
        "SELECT_SCREENED_POISSON",
        "SELECT_EXISTING_RAW_TRACER",
    ):
        assert forbidden_choice not in prompt
        assert forbidden_choice not in matrix


def test_packet_contains_research_enterprise_and_contamination_contracts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(output_dir=output_dir, audit_path=None)
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output_dir.iterdir())
    )

    assert "10.1145/777792.777840" in text
    assert "10.1111/1467-8659.00675" in text
    assert "10.1016/j.cad.2006.12.005" in text
    assert "10.1016/j.gmod.2012.04.008" in text
    assert "10.1016/j.cagd.2004.09.004" in text
    assert "10.1109/3DIMPVT.2012.12" in text
    assert "10.1145/1015706.1015768" in text
    assert "label-conditioned" in text
    assert "one-side deletion" in text
    assert "source_points_serialized=false" in text
    assert "at most 200,000 finite sampled points" in text
    assert "hard fail before 16 GiB peak RSS" in text
    assert "25856776f6048501c5e364cdad3fd33d70f5c89c1fb8c4e05dfbb5c46a4125b6" in text
    assert "2586bc09b7328900d5140857ece7553fd0188fddd29f2c462e01195aac13bfb8" in text


def test_builder_fails_closed_when_bound_source_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = deepcopy(builder.SOURCE_BINDINGS)
    bindings["formal_lpf_stop"]["sha256"] = "0" * 64
    monkeypatch.setattr(builder, "SOURCE_BINDINGS", bindings)

    with pytest.raises(ValueError, match="differs from expected SHA-256"):
        builder.build_package(output_dir=tmp_path / "packet", audit_path=None)
