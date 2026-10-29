from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.run_f1_local_plane_fragment_probe_batch import MAX_PEAK_RSS_BYTES
from scripts.run_f1_local_plane_fragment_probe_replays import (
    KIND,
    SELECTION_REASONS,
    _resource_record,
    select_designated_cases,
    validate_replay_attestation,
)


def _pool_manifest() -> dict:
    cases = []
    for index in range(1, 38):
        token = f"C{index:03d}"
        cases.append(
            {
                "case_token": token,
                "working_point_count": 200_000,
                "transition_mode_count": 300 if token == "C030" else index,
                "transition_mode_edge_count": 400 if token == "C003" else index,
            }
        )
    return {"cases": cases}


def _attestation() -> tuple[dict, dict[str, dict]]:
    batch_rows = {}
    cases = []
    pool_rows = {row["case_token"]: row for row in _pool_manifest()["cases"]}
    for token in sorted(SELECTION_REASONS):
        digest = {"C001": "1", "C003": "3", "C030": "a"}[token] * 64
        batch_rows[token] = {"probe_sha256": digest}
        runs = [
            _resource_record(
                {
                    "probe_sha256": digest,
                    "peak_rss_bytes": 100_000_000 + index,
                    "wall_seconds": 1.0 + index,
                    "input_json_bytes": 1000,
                    "serialized_output_bytes": 2000,
                },
                index,
            )
            for index in (1, 2, 3)
        ]
        pool = pool_rows[token]
        cases.append(
            {
                "case_token": token,
                "selection_reason": SELECTION_REASONS[token],
                "working_point_count": pool["working_point_count"],
                "transition_mode_count": pool["transition_mode_count"],
                "transition_mode_edge_count": pool["transition_mode_edge_count"],
                "batch_probe_sha256": digest,
                "replay_count": 3,
                "replay_probe_sha256s": [digest] * 3,
                "maximum_peak_rss_bytes": 100_000_003,
                "runs": runs,
            }
        )
    return (
        {
            "schema_version": 2,
            "kind": KIND,
            "profile": "LOCAL_PLANE_FRAGMENT_PROBE_V2",
            "research_only": True,
            "inference_safe": True,
            "case_count": 3,
            "total_isolated_run_count": 9,
            "probe_manifest_sha256": "b" * 64,
            "inference_manifest_sha256": "c" * 64,
            "candidate_pool_manifest_sha256": "d" * 64,
            "cases": cases,
        },
        batch_rows,
    )


def test_designated_selection_is_recomputed_with_lexical_ties() -> None:
    manifest = _pool_manifest()
    assert select_designated_cases(manifest) == SELECTION_REASONS
    c029 = next(row for row in manifest["cases"] if row["case_token"] == "C029")
    c029["transition_mode_count"] = 300
    with pytest.raises(ValueError, match="differs from frozen ruling"):
        select_designated_cases(manifest)


def test_exact_nine_run_attestation_validates() -> None:
    attestation, rows = _attestation()
    validate_replay_attestation(
        attestation, batch_rows=rows, pool_manifest=_pool_manifest()
    )


def test_replay_rejects_nonidentical_or_nonbatch_hash() -> None:
    attestation, rows = _attestation()
    bad = deepcopy(attestation)
    bad["cases"][0]["replay_probe_sha256s"][1] = "f" * 64
    with pytest.raises(ValueError, match="byte-identical"):
        validate_replay_attestation(bad, batch_rows=rows)
    bad = deepcopy(attestation)
    bad["cases"][0]["batch_probe_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="differs from batch probe"):
        validate_replay_attestation(bad, batch_rows=rows)


def test_replay_rejects_rss_at_limit_and_size_drift() -> None:
    attestation, rows = _attestation()
    bad = deepcopy(attestation)
    run = bad["cases"][0]["runs"][0]
    run["peak_rss_bytes"] = MAX_PEAK_RSS_BYTES
    # Resource hash must be valid before the semantic RSS rejection.
    replacement = _resource_record(run, int(run["run_index"]))
    bad["cases"][0]["runs"][0] = replacement
    with pytest.raises(ValueError, match="16 GiB"):
        validate_replay_attestation(bad, batch_rows=rows)
    bad = deepcopy(attestation)
    run = bad["cases"][0]["runs"][0]
    run["serialized_output_bytes"] += 1
    bad["cases"][0]["runs"][0] = _resource_record(run, int(run["run_index"]))
    with pytest.raises(ValueError, match="serialized_output_bytes"):
        validate_replay_attestation(bad, batch_rows=rows)


def test_v1_or_extra_case_replay_attestation_is_rejected() -> None:
    attestation, rows = _attestation()
    bad = deepcopy(attestation)
    bad["profile"] = "LOCAL_PLANE_FRAGMENT_PROBE_V1"
    with pytest.raises(ValueError, match="V2 identity"):
        validate_replay_attestation(bad, batch_rows=rows)
    bad = deepcopy(attestation)
    extra = deepcopy(bad["cases"][0])
    extra["case_token"] = "C004"
    bad["cases"].append(extra)
    bad["case_count"] = 4
    with pytest.raises(ValueError, match="exact C001/C003/C030"):
        validate_replay_attestation(bad, batch_rows=rows)
