from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

import scripts.build_professor_f1_pwts_stop_decision_package_20260807 as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_exact_ten_file_nonexecution_packet(tmp_path: Path) -> None:
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
    assert result["decision_only"] is True
    assert result["geometry_executed_by_builder"] is False
    assert result["gold_or_truth_consumed"] is False
    assert result["pwts_rerun_authorized"] is False
    assert result["runtime_or_production_authorized"] is False
    assert result["validation"]["known_seed_count_lower_bound"] == 1025
    assert result["validation"]["exact_seed_count_known"] is False
    assert audit_path.is_file()
    assert not any(path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"} for path in files)


def test_copies_all_nine_authoritative_files_byte_exact(tmp_path: Path) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        audit_path=None,
        verify_raw_sources=False,
    )
    for name, binding in builder.SOURCE_BINDINGS.items():
        assert (output_dir / name).read_bytes() == Path(binding["path"]).read_bytes()
        assert _sha(output_dir / name) == binding["sha256"]


def test_prompt_binds_evidence_decisions_research_and_prohibitions(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "packet"
    builder.build_package(
        output_dir=output_dir,
        audit_path=None,
        verify_raw_sources=False,
    )
    prompt = (
        output_dir / "00_GPT_PRO_FAILURE_CLASSIFICATION_PROMPT.md"
    ).read_text(encoding="utf-8")

    for name, binding in builder.SOURCE_BINDINGS.items():
        assert f"`{name}` — `{binding['sha256']}`" in prompt
    for token in (*builder.CLASSIFICATION_TOKENS, *builder.NEXT_STEP_TOKENS):
        assert token in prompt

    assert "seed_count >= 1025" in prompt
    assert "exact count and spatial distribution are unknown" in prompt
    assert "10.1145/777792.777840" in prompt
    assert "10.1111/1467-8659.00675" in prompt
    assert "10.1109/3DIMPVT.2012.12" in prompt
    assert "10.1016/j.cad.2006.12.005" in prompt
    assert "cap increase" in prompt
    assert "PWTS-V2 rescue" in prompt
    assert "count-only diagnostic" in prompt
    assert "No semantic payload was serialized" in prompt
    assert "not an in-kernel memory cgroup" in prompt


def test_builder_fails_closed_when_bound_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = deepcopy(builder.SOURCE_BINDINGS)
    bindings["08_C001_BASE_A_STDERR.txt"]["sha256"] = "0" * 64
    monkeypatch.setattr(builder, "SOURCE_BINDINGS", bindings)
    with pytest.raises(ValueError, match="differs from expected SHA-256"):
        builder.build_package(
            output_dir=tmp_path / "packet",
            audit_path=None,
            verify_raw_sources=False,
        )


def test_builder_rejects_nonterminal_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path(
        builder.SOURCE_BINDINGS["06_RUN_V2_ORCHESTRATION_STOP.json"]["path"]
    )
    mutated = tmp_path / "orchestration.json"
    value = __import__("json").loads(original.read_text(encoding="utf-8"))
    value["terminal_status"] = "PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET"
    mutated.write_text(__import__("json").dumps(value), encoding="utf-8")
    bindings = deepcopy(builder.SOURCE_BINDINGS)
    bindings["06_RUN_V2_ORCHESTRATION_STOP.json"] = {
        "path": mutated,
        "sha256": _sha(mutated),
    }
    monkeypatch.setattr(builder, "SOURCE_BINDINGS", bindings)
    with pytest.raises(ValueError, match="terminal status"):
        builder.build_package(
            output_dir=tmp_path / "packet",
            audit_path=None,
            verify_raw_sources=False,
        )
