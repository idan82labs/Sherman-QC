from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

import backend.f1_integral_transverse_covariance_growth_veto_v73 as v73
from tests.fixtures.f1_v73_itcgv_v1_fixture_oracle import (
    CASE_IDS,
    build_case,
    literal_axial_oracle,
    literal_comparator_oracle,
    literal_radial_scale0_oracle,
    small_roll_reference,
)


def _run(case):
    assert case.source is not None
    return v73.evaluate_scan(case.source, list(case.candidates))


def _signature(evaluation: v73.CandidateEvaluation) -> tuple:
    return (
        evaluation.decision,
        evaluation.eligibility_mask,
        evaluation.flat_mask,
        evaluation.smooth_growth_mask,
        evaluation.smoothlike_mask,
        evaluation.longest_smoothlike_run,
        evaluation.metric_sha256,
    )


def _forbidden_key_scan(value, path="") -> list[str]:
    forbidden_fragments = (
        "coordinate",
        "normal",
        "source_index",
        "cell_key",
        "centroid",
        "membership",
        "multiplicity",
        "point_list",
        "filename",
        "path",
        "identity",
        "family",
        "expected",
        "label",
        "cad",
        "baseline_score",
        "provenance",
    )
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{path}.{key}" if path else str(key)
            if any(fragment in str(key).lower() for fragment in forbidden_fragments):
                found.append(location)
            found.extend(_forbidden_key_scan(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_key_scan(child, f"{path}[{index}]"))
    return found


def _publish(identifier: str, artifact) -> None:
    root = os.environ.get("V73_LOCK_ARTIFACT_ROOT")
    if root:
        v73.atomic_write_canonical(Path(root) / f"{identifier}.json", artifact)


@pytest.mark.parametrize("identifier", CASE_IDS)
def test_v73_source_free_lock_case(identifier: str) -> None:
    case = build_case(identifier)

    if identifier == "G25":
        observed_values = []
        for radius, expected in literal_radial_scale0_oracle():
            observed = v73._radial_shell(radius * radius) == 0
            assert observed is expected
            observed_values.append({"radius": radius.hex(), "scale0": observed})
        _publish(identifier, {"case": identifier, "observed": observed_values})
        return

    if identifier == "G26":
        observed_values = []
        for zeta, expected in literal_axial_oracle():
            observed = v73._within_axial_slab(zeta)
            assert observed is expected
            observed_values.append({"included": observed, "zeta": zeta.hex()})
        _publish(identifier, {"case": identifier, "observed": observed_values})
        return

    if identifier == "G27":
        observed_values = {}
        for name, (operands, expected) in literal_comparator_oracle().items():
            if name.startswith("flat_"):
                observed = v73._flat_ratio(*operands)
            elif name.startswith("inner_"):
                observed = v73._inner_ratio(*operands)
            elif name.startswith("outer_"):
                observed = v73._outer_ratio(*operands)
            elif name.startswith("growth_"):
                observed = v73._growth_ratio(*operands)
            elif name.startswith("spread_"):
                observed = v73._spread_ratio(*operands)
            elif name.startswith("psd_"):
                _determinant, observed = v73._psd_clamp(*operands)
            else:
                raise AssertionError(name)
            assert observed is expected, name
            observed_values[name] = observed
        _publish(identifier, {"case": identifier, "observed": observed_values})
        return

    audit, evaluations = _run(case)

    if identifier == "G29":
        assert audit["status"] == v73.ABORT
        assert audit["abort_code"] == "SOURCE_CAP"
        assert audit["candidates"] == []
        assert evaluations == tuple()
        _publish(identifier, audit)
        return

    if identifier == "G31":
        assert audit["status"] == v73.ABORT
        assert audit["abort_code"] == "CANDIDATE_CAP"
        assert audit["candidates"] == []
        assert evaluations == tuple()
        _publish(identifier, audit)
        return

    assert audit["status"] == v73.COMPLETE
    assert len(evaluations) == len(case.candidates)
    assert len(audit["candidates"]) == len(case.candidates)

    if identifier in {"G01", "G02", "G17"}:
        assert [value.decision for value in evaluations] == [v73.DECISION_VETO]
        assert all(value.flat_mask == 0x1FF for value in evaluations)
        assert all(value.smoothlike_mask == 0x1FF for value in evaluations)
    elif identifier == "G03":
        assert evaluations[0].decision == v73.DECISION_VETO
        assert evaluations[0].smooth_growth_mask == 0x1FF
        assert evaluations[0].smoothlike_mask == 0x1FF
    elif identifier == "G04":
        reference = small_roll_reference()
        ref_audit, ref_values = _run(reference)
        assert ref_audit["status"] == v73.COMPLETE
        assert _signature(evaluations[0]) == _signature(ref_values[0])
    elif identifier in {"G05", "G06", "G07", "G08", "G09", "G10", "G11", "G12", "G18"}:
        assert evaluations[0].decision != v73.DECISION_VETO
        if identifier == "G12":
            assert evaluations[0].decision != v73.DECISION_KEEP_SHORT
    elif identifier == "G13":
        assert evaluations[0].decision == v73.DECISION_KEEP_SHORT
        assert audit["work_counters"]["candidate_source_tests"] == 0
    elif identifier == "G14":
        assert len(evaluations) == 2
        assert all(value.decision != v73.DECISION_VETO for value in evaluations)
    elif identifier == "G15":
        together = evaluations
        for ordinal, row in enumerate(case.candidates):
            alone_audit, alone = v73.evaluate_scan(case.source, [row])
            assert alone_audit["status"] == v73.COMPLETE
            assert _signature(alone[0]) == _signature(together[ordinal])
    elif identifier == "G16":
        assert len(evaluations) == 2
        assert _signature(evaluations[0]) == _signature(evaluations[1])
        assert audit["candidates"][0]["canonical_candidate_ordinal"] == 0
        assert audit["candidates"][1]["canonical_candidate_ordinal"] == 1
    elif identifier in {"G19", "G20", "G21", "G22", "G23", "G24"}:
        reference = build_case(case.reference)
        ref_audit, ref_values = _run(reference)
        assert ref_audit["status"] == v73.COMPLETE
        if identifier in {"G19", "G20", "G21"}:
            assert _signature(evaluations[0]) == _signature(ref_values[0])
        elif identifier in {"G22", "G23"}:
            assert _signature(evaluations[0]) == _signature(ref_values[0])
        else:
            assert evaluations[0].geometry_sha256 == ref_values[0].geometry_sha256
            assert _signature(evaluations[0]) == _signature(ref_values[0])
        if identifier == "G20":
            assert v73.canonical_bytes(audit) == v73.canonical_bytes(ref_audit)
    elif identifier == "G28":
        assert audit["source_count"] == 200_000
        assert evaluations[0].decision == v73.DECISION_KEEP_SHORT
        assert audit["work_counters"] == {
            "candidate_source_tests": 0,
            "cell_finalizations": 0,
            "cell_station_updates": 0,
            "metric_tuples": 0,
        }
    elif identifier == "G30":
        assert len(evaluations) == 64
        assert all(value.decision == v73.DECISION_KEEP_SHORT for value in evaluations)
        assert audit["work_counters"] == {
            "candidate_source_tests": 0,
            "cell_finalizations": 0,
            "cell_station_updates": 0,
            "metric_tuples": 0,
        }
    elif identifier == "G32":
        payload = v73.canonical_bytes(audit)
        assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
        assert len(payload) < 1_048_576
        assert _forbidden_key_scan(audit) == []
        decoded = json.loads(payload)
        assert decoded == audit
        replay_audit, replay_values = _run(case)
        assert v73.canonical_bytes(replay_audit) == payload
        assert _signature(replay_values[0]) == _signature(evaluations[0])
        with pytest.raises(v73.V73ResourceAbort, match="RESULT_BYTE_CAP"):
            v73.canonical_bytes({"oversized": "x" * 1_048_576})
    else:
        raise AssertionError(identifier)

    assert audit["work_counters"]["candidate_source_tests"] <= 12_800_000
    assert audit["work_counters"]["cell_finalizations"] <= 12_800_000
    assert audit["work_counters"]["cell_station_updates"] <= 38_400_000
    assert audit["work_counters"]["metric_tuples"] <= 4_608
    _publish(identifier, audit)
