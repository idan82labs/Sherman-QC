"""Contract parity (C11): every duplicated literal set equals the C01 contract it mirrors.

Wave-1 chunks could not import ``contracts/production`` (C02 must import without
pydantic, C04 is independent, the frontend is TypeScript), so they copied the
contract vocabularies. This test is the single place that pins every copy to
its source. A failing case names the copy and the difference.

Covered: execution statuses, job kinds, stages, failure codes, parse / resolve
statuses, ingest states, quarantine reasons, scan formats, event kinds, summary /
validity / stage states, priorities, attempts, the engine hard limits, the
job-result-summary fields, the E3 exit-2 codes (C02 copy vs E3 vs C01), the
data-root directory list and forbidden repo directories (C04 settings, C02
hotfolder, C04 bundle builder vs C01 layout), the app manifest constants, the
artifact allowlist (TypeScript station copy), capability names (C04 principals
vs CONTRACTS §9), the station's TypeScript state lists and the onboarding
states. A mutate-and-check case proves the comparison helpers fail on a
changed copy. Pure source / import checks: no data is read.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

import apps.production_line.store.vocabulary as c02
from contracts.production import events, layout, line_job, states
from contracts.production import characteristics as chars
from contracts.production import common

REPO = Path(__file__).resolve().parents[1]
STATION_TYPES = REPO / "frontend/react/src/features/station/model/types.ts"
ONBOARDING_TYPES = REPO / "frontend/react/src/features/part-onboarding/model/types.ts"

#: CONTRACTS §9 (survey architect plan) capability names, plus ``admin`` (C04 route policy rules).
CONTRACT_CAPABILITIES = frozenset(
    {"line.read", "line.operate", "parts.write", "parts.delete", "parts.import", "recipe.release",
     "devices.manage", "metrics.read", "admin"}
)


def _same(copy: Any, source: Any, what: str) -> None:
    a, b = set(copy), set(source)
    assert a == b, f"{what}: copy-only {sorted(a - b)}, contract-only {sorted(b - a)}"


def ts_const_array(path: Path, name: str) -> list[str]:
    """String members of ``export const NAME = [ ... ] as const`` in a TypeScript file."""
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"export const {re.escape(name)}\s*=\s*\[(.*?)\]\s*as const", text, re.S)
    assert m, f"{name} not found in {path.name}"
    return re.findall(r"'([^']*)'", m.group(1))


def ts_union(path: Path, name: str) -> list[str]:
    """String members of ``export type NAME = | 'A' | 'B'`` in a TypeScript file."""
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"export type {re.escape(name)}\s*=\s*((?:\s*\|?\s*'[^']*')+)", text)
    assert m, f"type {name} not found in {path.name}"
    return re.findall(r"'([^']*)'", m.group(1))


# ---------------------------------------------------------------------------- C02 store vocabulary

C02_SETS = [
    ("EXECUTION_STATUSES", line_job.EXECUTION_STATUSES),
    ("JOB_KINDS", line_job.JOB_KINDS),
    ("STAGES", line_job.STAGES),
    ("FAILURE_CODES", line_job.FAILURE_CODES),
    ("PARSE_STATUSES", line_job.PARSE_STATUSES),
    ("RESOLVE_STATUSES", line_job.RESOLVE_STATUSES),
    ("INGEST_STATES", line_job.INGEST_STATES),
    ("QUARANTINE_REASONS", line_job.QUARANTINE_REASONS),
    ("SCAN_FORMATS", line_job.SCAN_FORMATS),
    ("EVENT_KINDS", events.EVENT_KINDS),
    ("SUMMARY_STATES", states.SUMMARY_STATES),
    ("VALIDITY_STATES", states.VALIDITY_STATES),
    ("STAGE_STATUSES", states.STAGE_STATUSES),
]


@pytest.mark.parametrize("name,source", C02_SETS, ids=[n for n, _ in C02_SETS])
def test_c02_vocabulary_sets_equal_contracts(name: str, source: frozenset[str]) -> None:
    _same(getattr(c02, name), source, f"apps/production_line/store/vocabulary.py {name}")


def test_c02_scalars_and_maps_equal_contracts() -> None:
    assert c02.SCAN_FILE_SCHEMA == line_job.SCAN_FILE_SCHEMA
    assert c02.LINE_JOB_SCHEMA == line_job.LINE_JOB_SCHEMA
    assert c02.LINE_EVENT_SCHEMA == events.LINE_EVENT_SCHEMA
    assert c02.PRIORITY_BY_KIND == line_job.PRIORITY_BY_KIND
    assert c02.MAX_ATTEMPTS_BY_KIND == line_job.MAX_ATTEMPTS_BY_KIND
    assert c02.ENGINE_HARD_TIMEOUT_S == line_job.ENGINE_HARD_TIMEOUT_S == 180
    assert c02.ENGINE_RSS_LIMIT_BYTES == line_job.ENGINE_RSS_LIMIT_BYTES == 32 * 1024**3  # ENGINE_IO-A5
    assert c02.TERMINAL_STATUSES <= c02.EXECUTION_STATUSES
    assert c02.CONTENT_QUARANTINE_REASONS <= c02.QUARANTINE_REASONS
    assert set(c02.RESOLVE_STATUS_TO_QUARANTINE_REASON) == set(line_job.RESOLVE_STATUSES) - {"MATCHED"}
    assert set(c02.RESOLVE_STATUS_TO_QUARANTINE_REASON.values()) <= c02.QUARANTINE_REASONS


def test_job_result_summary_keys_equal_contract_model() -> None:
    fields = set(line_job.JobResultSummaryV1.model_fields)
    assert set(c02.JOB_RESULT_SUMMARY_KEYS) == fields


def test_engine_input_error_codes_three_copies_agree() -> None:
    from domains.production.engine.inspect.part import INPUT_ERROR_CODES

    _same(c02.ENGINE_INPUT_ERROR_CODES, INPUT_ERROR_CODES, "C02 ENGINE_INPUT_ERROR_CODES vs E3 INPUT_ERROR_CODES")
    missing = set(c02.ENGINE_INPUT_ERROR_CODES) - set(line_job.FAILURE_CODES)
    assert not missing, f"E3 exit-2 codes that are not job FAILURE_CODES: {sorted(missing)}"


# ---------------------------------------------------------------------------- layout duplicates


def test_data_root_layout_copies_equal_contract() -> None:
    from apps.production_line import engine_bundle, settings
    from apps.production_line.ingest import hotfolder

    assert tuple(settings.DATA_ROOT_DIRS) == tuple(layout.DATA_ROOT_DIRS)
    forbidden = tuple(layout.FORBIDDEN_DATA_ROOT_PARENTS)
    assert tuple(settings.FORBIDDEN_REPO_SUBDIRS) == forbidden
    assert tuple(hotfolder._FORBIDDEN_REPO_DIRS) == forbidden
    assert tuple(engine_bundle.RESEARCH_DIRS) == forbidden
    # The bundle builder never even lists a research directory.
    assert set(forbidden) <= set(engine_bundle.HARD_EXCLUDED_DIRS)


def test_app_manifest_constants_equal_bundle_builder() -> None:
    from apps.production_line import engine_bundle, settings

    assert settings.APP_MANIFEST_NAME == engine_bundle.MANIFEST_NAME
    assert settings.APP_MANIFEST_SCHEMA == engine_bundle.MANIFEST_SCHEMA


def test_station_id_rules_agree() -> None:
    from apps.production_line import settings
    from apps.production_line.ingest import hotfolder

    for good in ("cell-1", "a", "z" * 32):
        assert settings.STATION_ID_RE.match(good) and hotfolder._STATION_RE.match(good)
    # The settings rule is the stricter one; the hotfolder accepts everything settings accepts.
    for bad in ("Cell-1", "", "a" * 33, "cell_1"):
        assert not settings.STATION_ID_RE.match(bad)


def test_station_artifact_names_equal_allowlist() -> None:
    _same(ts_const_array(STATION_TYPES, "ARTIFACT_NAMES"), layout.ARTIFACT_ALLOWLIST, "station ARTIFACT_NAMES")
    for name, rel in layout.ARTIFACT_ALLOWLIST.items():
        assert rel == f"result/{name}" and name in layout.RESULT_FILES.values()


# ---------------------------------------------------------------------------- capabilities


def test_capabilities_equal_contract_section_9() -> None:
    from apps.production_line.security import principals, route_policy

    _same(principals.CAPABILITIES, CONTRACT_CAPABILITIES, "principals.CAPABILITIES")
    assert principals.DEVICE_CAPABILITIES == frozenset({"line.read"})
    assert principals.CAPABILITIES["line.read"] == frozenset({"viewer", "operator", "admin"})
    assert principals.CAPABILITIES["recipe.release"] == frozenset({"admin"})  # SHERMAN_RELEASE_ROLES default
    for rule in route_policy.DEFAULT_RULES:
        assert rule.action in ("deny", "authenticated") or rule.action in CONTRACT_CAPABILITIES


# ---------------------------------------------------------------------------- frontend copies

TS_ARRAYS = [
    (STATION_TYPES, "SUMMARY_STATES", states.SUMMARY_STATES),
    (STATION_TYPES, "STAGE_STATUSES", states.STAGE_STATUSES),
    (STATION_TYPES, "BEND_STATES", states.BEND_STATES),
    (STATION_TYPES, "UNKNOWN_REASONS", chars.UNKNOWN_REASONS),
    (ONBOARDING_TYPES, "TOLERANCE_SOURCES", common.TOLERANCE_SOURCES),
]


@pytest.mark.parametrize("path,name,source", TS_ARRAYS, ids=[f"{p.parent.parent.name}.{n}" for p, n, _ in TS_ARRAYS])
def test_typescript_state_lists_equal_contracts(path: Path, name: str, source: frozenset[str]) -> None:
    _same(ts_const_array(path, name), source, f"{path.relative_to(REPO)} {name}")


def test_typescript_validity_union_and_na_reasons_equal_contracts() -> None:
    from contracts.production.inspection_result import _NA_REASONS_BY_KIND as na_by_kind

    _same(ts_union(STATION_TYPES, "ValidityState"), states.VALIDITY_STATES, "station ValidityState")
    na = set().union(*na_by_kind.values())
    _same(ts_const_array(STATION_TYPES, "NOT_APPLICABLE_REASONS"), na, "station NOT_APPLICABLE_REASONS")


def test_onboarding_states_equal_c05() -> None:
    from apps.production_line.recipes.onboarding import ONBOARDING_STATES

    assert ts_const_array(ONBOARDING_TYPES, "ONBOARDING_STATES") == list(ONBOARDING_STATES)


# ---------------------------------------------------------------------------- the checks can fail


def test_comparison_helpers_detect_a_changed_copy(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="copy-only \\['QUEUED'\\]"):
        _same(set(c02.EXECUTION_STATUSES) | {"QUEUED"}, line_job.EXECUTION_STATUSES, "mutated")
    with pytest.raises(AssertionError, match="contract-only"):
        _same(set(states.SUMMARY_STATES) - {"STAGE_IN_TOLERANCE"}, states.SUMMARY_STATES, "mutated")
    ts = tmp_path / "t.ts"
    ts.write_text("export const SUMMARY_STATES = [\n  'ALL_MEASURED_IN_TOLERANCE',\n  'INCOMPLETE',\n] as const\n"
                  "export type ValidityState =\n  | 'VALID'\n  | 'TIMEOUT'\n", encoding="utf-8")
    assert ts_const_array(ts, "SUMMARY_STATES") == ["ALL_MEASURED_IN_TOLERANCE", "INCOMPLETE"]
    assert ts_union(ts, "ValidityState") == ["VALID", "TIMEOUT"]
    with pytest.raises(AssertionError):
        _same(ts_const_array(ts, "SUMMARY_STATES"), states.SUMMARY_STATES, "mutated ts")
    with pytest.raises(AssertionError, match="not found"):
        ts_const_array(ts, "MISSING_NAME")
