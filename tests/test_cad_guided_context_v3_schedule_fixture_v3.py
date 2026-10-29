"""Five prospective mocked byte-workflow controls after the terminal V2 failure.

Actual V7 codec, actual V2 empty scheduler and actual scan/schedule/context
binders; explicitly invented metadata, not actual extraction or geometry/count
evidence. Frozen V2 fixture helpers are imported explicitly. No old test function
is imported or selected; the 93 passing cases and dense case are not rerun.
Source-only until a new protocol, independent exact PRE and root release.
"""

import copy
from dataclasses import asdict
import json

import pytest

from tests.test_cad_guided_context_v3_engine_integration_v2 import (
    modules,
    plumbing,
    _canonical,
    _expected_configuration,
    _sha,
)


SCOPE = "INVENTED_V7_METADATA_CODEC_ONLY_NOT_EXTRACTION_OR_GEOMETRY"


def _diagnostic(state, answer):
    """Small explicit reasons survive pytest's elision of the full answer."""
    registration = answer.get("registration")
    value = dict(status=answer.get("status"), phase=answer.get("phase"),
        reasons=answer.get("reasons"),
        registration_reasons=registration.get("reasons") if type(registration) is dict else None,
        bindings=state.bindings, trace=state.fixture_trace)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert len(encoded.encode()) <= 4096, "FIXED_DIAGNOSTIC_EXCEEDED_4096_BYTES"
    return encoded


@pytest.fixture
def repaired(plumbing, monkeypatch):
    state, m = plumbing, plumbing.m
    # Same actual codec used by V8 preparation; we deliberately do not call its
    # extractor. This record is an honest boundary stub, not a COMPLETE producer
    # report, observed support/count, or a registration-quality certificate.
    codec = m.frozen._previous._compact
    metadata = dict(schema_version="noisy_mesh_planes_v7", fixture_scope=SCOPE,
        status="INVENTED_BOUNDARY_METADATA_NOT_EXTRACTOR_RESULT")
    envelope = codec.encode_metadata_v7(metadata)
    assert codec.decode_metadata_v7(envelope) == metadata
    assert set(envelope) == {"format", "schemas", "root"}
    assert len(_canonical(envelope)) <= 4096

    cap = m.schedule2.TwoPlaneScheduleBudgetsV2()
    configuration = dict(mode="v2", budgets=asdict(cap))
    schedule = m.schedule2.schedule_two_plane_pairs_v2((), (),
        retained_triplet_hypothesis_count=3, retained_triplet_unsigned_count=3,
        max_joint_hypothesis_records=8192, max_joint_unsigned_assignments=1024,
        budgets=cap)
    assert schedule["status"] == "COMPLETE", schedule["reasons"]
    assert schedule["scheduled_matches"] == []
    evidence = state.registration["evidence"]
    evidence["two_plane_schedule_configuration"] = copy.deepcopy(configuration)
    evidence["initialization"].update(
        two_plane_schedule_configuration=copy.deepcopy(configuration),
        two_plane_schedule=schedule, budgets=asdict(m.init.PlaneInitializationBudgetsV12()),
        hypothesis_evidence=dict(scan_bank=[], scan_planes=[]),
        patch_evidence=dict(scan=envelope, cad=dict(patches=[])))
    state.fixture_codec = codec
    state.fixture_metadata = metadata
    state.fixture_schedule_cap = cap
    state.fixture_schedule_configuration = configuration
    state.fixture_trace = dict(context=[], schedule=[], scan=[])

    def tracked(name, actual):
        def delegate(*args, **kwargs):
            try:
                result = actual(*args, **kwargs)
            except Exception as error:
                # Observe the actual production exception and rethrow it
                # unchanged. No fake result, fallback, or broadened acceptance.
                state.fixture_trace[name].append(("FAIL", type(error).__name__, error.args))
                raise
            state.fixture_trace[name].append(("PASS",))
            return result
        return delegate

    monkeypatch.setattr(m.init, "_bind_context_publication",
        tracked("context", m.init._bind_context_publication))
    monkeypatch.setattr(m.workflow, "_two_plane_schedule_binding",
        tracked("schedule", m.workflow._two_plane_schedule_binding))
    monkeypatch.setattr(m.workflow, "_scan_plane_binding",
        tracked("scan", m.workflow._scan_plane_binding))
    return state


def _run(state):
    return state.run(two_plane_schedule_mode="v2",
        two_plane_schedule_budgets=state.fixture_schedule_cap)


def test_repaired_v2_schedule_workflow_binds_three_contexts_with_actual_v7_codec(repaired):
    state, m = repaired, repaired.m
    answer = _run(state)
    detail = _diagnostic(state, answer)
    assert answer["status"] == "PARTIAL", detail
    assert answer["phase"] == "COMPLETE" and answer["reasons"] == [], detail
    assert answer["schema"] == "CAD_GUIDED_MESH_OFFLINE_V13"
    assert answer["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V14_CONTEXT_V3"
    assert answer["production_use_allowed"] is False and answer["observed_bend_count"] is None
    assert state.bindings == [0, 1, 2], detail
    assert state.fixture_trace == dict(context=[("PASS",)]*10,
        schedule=[("PASS",)]*3, scan=[("PASS",)]*3), detail
    assert len(state.registration_calls) == len(state.measurement_calls) == len(state.validator_calls) == 1
    assert state.events[-5:] == ["bind:0", "bind:1", "bind:2", "measure_mock", "validate_mock"]
    assert answer["registration"] is state.registration
    assert answer["measurements"] is state.measured
    assert answer["predictions"] is state.measured["predictions"]
    assert answer["predictions"] != state.measured["source_context_result"]["predictions"]

    args, kwargs = state.registration_calls[0]
    assert args[0] is state.scan["raw_points"] and len(args[0]) == 48
    assert kwargs["registration_raw_rows"] is state.scan["engine_to_raw_row_indices"]
    assert kwargs["two_plane_schedule_mode"] == "v2" and kwargs["two_plane_context_mode"] == "v3"
    assert kwargs["two_plane_schedule_budgets"] is not state.fixture_schedule_cap
    assert asdict(kwargs["two_plane_schedule_budgets"]) == asdict(state.fixture_schedule_cap)
    assert type(kwargs["two_plane_context_budgets"]) is m.context.TwoPlaneContextBudgetsV3
    expected = _expected_configuration(m)
    expected_sha = _sha(_canonical(expected))
    expected_schedule_sha = _sha(_canonical(state.fixture_schedule_configuration))
    assert len(answer["registration_array_binding"]) == 3
    for binding in answer["registration_array_binding"]:
        assert binding["two_plane_schedule_mode"] == "v2"
        assert binding["two_plane_schedule_configuration_sha256"] == expected_schedule_sha
        assert binding["two_plane_context_mode"] == "v3"
        assert binding["two_plane_context_configuration_sha256"] == expected_sha
    for publication in (answer, answer["policies"], state.registration["evidence"],
                        state.registration["evidence"]["initialization"]):
        assert publication["two_plane_context_configuration"] == expected
        assert publication["two_plane_context_configuration_sha256"] == expected_sha
    envelope = state.registration["evidence"]["initialization"]["patch_evidence"]["scan"]
    assert state.fixture_codec.decode_metadata_v7(envelope) == state.fixture_metadata


@pytest.mark.parametrize("fault,exception_name,reason", (
    ("missing_scan", "KeyError", "scan"),
    ("missing_root", "ValueError", "INVALID_LEGACY_SCAN_PLANE_ENVELOPE"),
    ("invalid_root", "ValueError", "INVALID_LEGACY_SCAN_PLANE_ROOT"),
    ("wrong_schema", "ValueError", "UNREQUESTED_OR_INVALID_SCAN_PLANE_SCHEMA"),
), ids=("missing_scan", "missing_root", "invalid_root", "wrong_schema"))
def test_malformed_scan_envelope_reaches_exact_binder_and_refuses_whole(
        repaired, fault, exception_name, reason):
    state = repaired
    patch = state.registration["evidence"]["initialization"]["patch_evidence"]
    if fault == "missing_scan":
        del patch["scan"]
    elif fault == "missing_root":
        del patch["scan"]["root"]
    elif fault == "invalid_root":
        patch["scan"]["root"] = ["D", 0]
    else:
        wrong = dict(state.fixture_metadata, schema_version="composite_scan_planes_v1")
        patch["scan"] = state.fixture_codec.encode_metadata_v7(wrong)
        assert state.fixture_codec.decode_metadata_v7(patch["scan"]) == wrong
    answer = _run(state)
    detail = _diagnostic(state, answer)
    assert answer["status"] == "UNAVAILABLE" and answer["phase"] == "WHOLE_PART_REGISTRATION", detail
    assert answer["reasons"] == ["WORKFLOW_EXCEPTION:"+exception_name], detail
    assert answer["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V14_CONTEXT_V3"
    assert answer["predictions"] == [] and answer["measurements"] is None
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
    assert state.fixture_trace == dict(context=[("PASS",)]*6, schedule=[("PASS",)],
        scan=[("FAIL", exception_name, (reason,))]), detail
    assert state.bindings == [0], detail
    assert len(state.registration_calls) == 1
    assert state.events[-2:] == ["register_mock", "bind:0"]
    assert state.measurement_calls == state.validator_calls == []
    assert answer["registration_array_binding"] == []
    registration = answer["registration"]
    assert registration["status"] == "UNAVAILABLE" and registration["transform_scan_to_cad"] is None
    assert registration["reasons"] == ["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"]
    assert registration["evidence"]["measurement_context_indices"] == []
    assert registration["evidence"]["attempted_measurement_context_indices"] == [0, 1, 2]
