"""Report envelope V7: V6 semantics plus completion-lane discard accounting. Inert import; invented reports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if "backend" not in sys.modules:
        package = types.ModuleType("backend"); package.__path__ = [str(ROOT/"backend")]; sys.modules["backend"] = package
    spec = importlib.util.spec_from_file_location("backend."+name, ROOT/"backend"/f"{name}.py")
    module = importlib.util.module_from_spec(spec); sys.modules["backend."+name] = module; spec.loader.exec_module(module)
    return module


v6 = _load("cad_guided_report_envelope_v6")
v7 = _load("cad_guided_report_envelope_v7")


def _report(main=3, children=2, nested=False, candidates=1):
    lane = dict(status="COMPLETE", hypotheses=[dict(completion_index=i) for i in range(children)])
    inner = dict(schema_version="noisy_plane_initialization_prefilter_v12", reasons=["X"],
                 hypotheses=[dict(hypothesis_index=i) for i in range(main)], start_groups=[], completion_lane=lane)
    if not nested:
        return inner
    return dict(schema_version="noisy_plane_initialized_finite_registration_v12", reasons=["Y"],
                evidence=dict(candidates=[dict(index=i) for i in range(candidates)], initialization=inner))


@pytest.mark.parametrize("nested", [False, True])
def test_refusal_counts_completion_children_at_both_depths(nested):
    report = _report(nested=nested)
    out = v7.bound_initialization_report_v7(report, 10)
    assert out["schema_version"] == "initialization_report_refusal_envelope_v7"
    assert out["reasons"] == ["INITIALIZATION_REPORT_BUDGET_EXCEEDED"]
    assert (out["discarded_hypothesis_records"], out["discarded_hypothesis_count_status"]) == (3, "RETAINED_RECORD_COUNT")
    assert (out["discarded_completion_records"], out["discarded_completion_count_status"]) == (2, "RETAINED_RECORD_COUNT")
    assert out["discarded_candidate_records"] == (1 if nested else 0)
    assert v7.bound_initialization_report_v7(out, 10) == out  # idempotent, detached
    assert v7.bound_initialization_report_v7(copy.deepcopy(out), 0) == out


def test_missing_or_malformed_completion_container_is_reported_not_invented():
    out = v7.bound_initialization_report_v7(dict(schema_version="x", reasons=[], hypotheses=[1], start_groups=[]), 10)
    assert (out["discarded_completion_records"], out["discarded_completion_count_status"]) == (0, "NO_RETAINED_LIST_PRESENT")
    bad = v7.bound_initialization_report_v7(dict(schema_version="x", reasons=[], hypotheses=[1], start_groups=[],
                                                 completion_lane=dict(hypotheses="not-a-list")), 10)
    assert (bad["discarded_completion_records"], bad["discarded_completion_count_status"]) == (None, "UNKNOWN_MALFORMED_CONTAINER")
    worse = v7.bound_initialization_report_v7(dict(schema_version="x", reasons=[], hypotheses=[1], start_groups=[],
                                                   completion_lane=5), 10)
    assert worse["discarded_completion_count_status"] == "UNKNOWN_MALFORMED_CONTAINER"


def test_malformed_top_level_evidence_container_makes_completion_count_unknown_not_zero():
    out = v7.bound_initialization_report_v7(dict(schema_version="x", reasons=[], hypotheses=[1], start_groups=[],
                                                 evidence=5), 10)
    # The direct list is a known lower bound; the nested containers behind a malformed `evidence` are
    # unknown, for hypotheses (lower bound), candidates and completion children alike, never a plain zero.
    assert (out["discarded_hypothesis_records"], out["discarded_hypothesis_count_status"]) == (1, "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER")
    assert (out["discarded_completion_records"], out["discarded_completion_count_status"]) == (None, "UNKNOWN_MALFORMED_CONTAINER")
    assert (out["discarded_candidate_records"], out["discarded_candidate_count_status"]) == (None, "UNKNOWN_MALFORMED_CONTAINER")
    nested = v7.bound_initialization_report_v7(dict(schema_version="x", reasons=[], hypotheses=[1], start_groups=[],
                                                    evidence=dict(initialization=5)), 10)
    assert (nested["discarded_completion_records"], nested["discarded_completion_count_status"]) == (None, "UNKNOWN_MALFORMED_CONTAINER")
    direct = v7.bound_initialization_report_v7(dict(schema_version="x", reasons=[], hypotheses=[1], start_groups=[],
                                                    completion_lane=dict(hypotheses=[1, 2]), evidence=5), 10)
    # A direct completion list is a known lower bound even when the nested container is malformed.
    assert (direct["discarded_completion_records"], direct["discarded_completion_count_status"]) == (2, "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER")


def test_within_budget_report_is_returned_unchanged_and_v6_ignores_completion_children():
    report = _report()
    size = len(json.dumps(report, sort_keys=True, separators=(",", ":")))
    assert v7.bound_initialization_report_v7(report, size) is report
    old = v6.bound_initialization_report_v6(_report(), 10)
    assert "discarded_completion_records" not in old and old["schema_version"].endswith("_v6")


def test_v6_envelope_is_not_accepted_as_a_v7_canonical_envelope():
    old = v6.bound_initialization_report_v6(_report(), 10)
    out = v7.bound_initialization_report_v7(old, 10)
    assert out["schema_version"].endswith("_v7") and out["reasons"] == ["INVALID_INITIALIZATION_REFUSAL_ENVELOPE"]


@pytest.mark.parametrize("field,value", [("discarded_completion_records", -1), ("discarded_completion_records", 2.0),
                                         ("discarded_completion_count_status", "FORGED")])
def test_forged_completion_counts_break_canonical_identity(field, value):
    out = v7.bound_initialization_report_v7(_report(), 10)
    forged = dict(out, **{field: value})
    again = v7.bound_initialization_report_v7(forged, 10)
    assert again["reasons"] == ["INVALID_INITIALIZATION_REFUSAL_ENVELOPE"]
