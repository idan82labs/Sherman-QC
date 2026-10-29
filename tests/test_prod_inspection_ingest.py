"""C06 ingest of the E3 output directory: model validation, the cross-document
consistency check, input integrity and the verdict-preservation rule.

Synthetic data only (SYN-HAT-1 contract example via the fake engine output).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from domains.production.inspection.engine_output import (
    VERDICT_FREE_FIELDS,
    load_engine_output,
    load_result,
    same_verdict,
)
from domains.production.inspection.errors import POSTPROCESS_ERROR_CODES, PostprocessError
from domains.production.inspection.fake_engine_output import SCENARIOS, build_fake_engine_output


def _load(fake):
    return load_engine_output(fake.result_path, fake.recipe_path, fake.compiled_path)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_fake_scenario_ingests(tmp_path: Path, scenario: str) -> None:
    fake = build_fake_engine_output(tmp_path / scenario, scenario=scenario)
    eo = _load(fake)
    assert eo.engine_dir == fake.engine_dir
    assert len(eo.result_sha256) == 64
    assert eo.result.recipe.recipe_id == eo.recipe.recipe_id


def _expect(code: str, fn) -> PostprocessError:
    with pytest.raises(PostprocessError) as info:
        fn()
    assert info.value.code == code, info.value
    assert info.value.exit_code == 2
    return info.value


def test_unparseable_result_is_engine_output_invalid(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws")
    fake.result_path.write_text("{not json")
    _expect("ENGINE_OUTPUT_INVALID", lambda: _load(fake))


def test_dishonest_summary_is_rejected(tmp_path: Path) -> None:
    def lie(data):
        data["summary"]["state"] = "ALL_MEASURED_IN_TOLERANCE"

    fake = build_fake_engine_output(tmp_path / "ws", mutate_result=lie)
    err = _expect("ENGINE_OUTPUT_INVALID", lambda: _load(fake))
    assert "derive_summary" in err.message


def test_unknown_promoted_to_pass_is_rejected(tmp_path: Path) -> None:
    def promote(data):
        char = data["bends"][3]["characteristics"][0]  # B4 flange A: NOT_MEASURED EDGE_NOT_SEEN
        assert char["state"] == "NOT_MEASURED"
        char["decision"] = "PASS"

    fake = build_fake_engine_output(tmp_path / "ws", mutate_result=promote)
    _expect("ENGINE_OUTPUT_INVALID", lambda: _load(fake))


def test_wrong_compiled_reference_is_inconsistent(tmp_path: Path) -> None:
    def other(data):
        data["engine"]["compiled_part_sha256"] = "0" * 64

    fake = build_fake_engine_output(tmp_path / "ws", mutate_result=other)
    err = _expect("ENGINE_OUTPUT_INVALID", lambda: _load(fake))
    assert "COMPILED_PART_REF_MISMATCH" in err.message


def test_altered_expected_value_is_inconsistent(tmp_path: Path) -> None:
    def shift(data):
        char = data["bends"][3]["characteristics"][0]  # NOT_MEASURED: expected only, no derived values
        char["expected"] = char["expected"] + 1.0

    fake = build_fake_engine_output(tmp_path / "ws", mutate_result=shift)
    err = _expect("ENGINE_OUTPUT_INVALID", lambda: _load(fake))
    assert "EXPECTED_MISMATCH" in err.message


def test_hidden_required_side_is_inconsistent(tmp_path: Path) -> None:
    # A REQUIRED flange side reported NOT_INSPECTED_BY_RECIPE would hide a value.
    def hide(data):
        char = data["bends"][3]["characteristics"][0]
        char.update(state="NOT_APPLICABLE", decision="NOT_APPLICABLE", reason="NOT_INSPECTED_BY_RECIPE", required=False)
        s = data["summary"]
        s["characteristics_required"] -= 1
        s["required_unknown"] -= 1
        s["reasons"] = [r for r in s["reasons"] if r != "REQUIRED_UNKNOWN"]

    fake = build_fake_engine_output(tmp_path / "ws", mutate_result=hide)
    err = _expect("ENGINE_OUTPUT_INVALID", lambda: _load(fake))
    assert "REQUIREMENT_MISMATCH" in err.message


def test_tampered_recipe_is_recipe_invalid(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws")
    data = json.loads(fake.recipe_path.read_text())
    data["part_name"] = "tampered"
    fake.recipe_path.write_text(json.dumps(data))
    _expect("RECIPE_INVALID", lambda: _load(fake))


def test_missing_inputs_are_input_missing(tmp_path: Path) -> None:
    fake = build_fake_engine_output(tmp_path / "ws")
    fake.compiled_path.unlink()
    _expect("INPUT_MISSING", lambda: _load(fake))
    _expect("INPUT_MISSING", lambda: load_result(tmp_path / "nope.json"))


def test_error_codes_are_closed() -> None:
    with pytest.raises(ValueError):
        PostprocessError("MADE_UP", "x")
    assert {"ENGINE_OUTPUT_INVALID", "RESULT_ALTERED", "INTERNAL_ERROR"} <= POSTPROCESS_ERROR_CODES


def test_same_verdict_allows_only_assets_artifacts_and_appended_warnings(tmp_path: Path) -> None:
    from contracts.production.inspection_result import ShopfloorInspectionResultV1

    fake = build_fake_engine_output(tmp_path / "ws")
    engine = load_result(fake.result_path)
    base = engine.model_dump(mode="json")
    assert VERDICT_FREE_FIELDS == {"viewer_assets", "artifacts", "warnings"}

    def variant(fn) -> ShopfloorInspectionResultV1:
        data = json.loads(json.dumps(base))
        fn(data)
        return ShopfloorInspectionResultV1.model_validate(data)

    ok = variant(lambda d: (d["warnings"].append("C06: extra"), d["artifacts"].update(pdf="report.pdf"),
                            d["viewer_assets"].update(registered_scan="viewer_scan.ply")))
    assert same_verdict(engine, ok)
    assert not same_verdict(engine, variant(lambda d: d.update(warnings=[])))  # dropped a warning
    assert not same_verdict(engine, variant(lambda d: d.update(warnings=["C06: x", *d["warnings"]])))  # not appended
    assert not same_verdict(engine, variant(lambda d: d.update(part_name="other")))
    assert not same_verdict(engine, variant(lambda d: d["context"].update(serial="SN-X")))
