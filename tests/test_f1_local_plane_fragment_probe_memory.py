from __future__ import annotations

from copy import deepcopy
import hashlib
import resource
import sys
from time import perf_counter

import pytest

from backend.f1_local_plane_fragment_probe import (
    PEAK_RSS_LIMIT_BYTES,
    _table_sha256,
    build_local_plane_fragment_probe,
    canonical_json_bytes,
    project_lpf_input,
    validate_lpf_memory_profile_row,
)


def _bounded_chain_pool(parent_count: int = 128) -> dict:
    parents = []
    modes = []
    edges = []
    for index in range(parent_count):
        parent_id = f"P{index:04d}"
        left_id, right_id = f"LX{index:04d}", f"RY{index:04d}"
        z_value = 3.0 * index
        parents.append(
            {
                "parent_atom_id": parent_id,
                "centroid": [0.0, 0.0, z_value],
                "point_count": 8,
                "normalized_support_mass": 1,
                "mode_child_ids": [left_id, right_id],
                "mode_overflow": False,
                "transition_certificate_allowed": True,
            }
        )
        modes.extend(
            (
                {
                    "mode_atom_id": left_id,
                    "parent_atom_id": parent_id,
                    "centroid": [0.0, 0.0, z_value],
                    "normal_projective": [1.0, 0.0, 0.0],
                    "point_count": 4,
                    "surface_fit_p90_mm": 0.0,
                },
                {
                    "mode_atom_id": right_id,
                    "parent_atom_id": parent_id,
                    "centroid": [0.0, 0.0, z_value],
                    "normal_projective": [0.0, 1.0, 0.0],
                    "point_count": 4,
                    "surface_fit_p90_mm": 0.0,
                },
            )
        )
        edges.append(
            {
                "mode_edge_id": f"C{index:04d}",
                "mode_ids": [left_id, right_id],
                "edge_kind": "CROSS_MODE_LOCAL",
                "normal_residual": 0.0,
                "plane_residual": 0.0,
                "spatial_residual": 0.0,
            }
        )
        for prefix in ("LX", "RY"):
            if index:
                edges.append(
                    {
                        "mode_edge_id": f"S{prefix}{index:04d}",
                        "mode_ids": [f"{prefix}{index - 1:04d}", f"{prefix}{index:04d}"],
                        "edge_kind": "SAME_SURFACE",
                        "normal_residual": 0.0,
                        "plane_residual": 0.0,
                        "spatial_residual": 0.0,
                    }
                )
    pool = {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "support_parent_table": parents,
        "support_mode_table": modes,
        "support_mode_edges": edges,
        "plane_table": [],
        "transition_local_spacing_mm": 1.0,
        "transition_atom_caps": {
            "input_point_cap": 200_000,
            "parent_cap": 4_000,
            "mode_cap": 16_000,
            "mode_children_per_parent_cap": 4,
            "parent_plus_mode_cap": 20_000,
            "mode_edge_cap": 200_000,
        },
        "working_point_budget": 200_000,
        "transient_sampled_point_count": 200_000,
    }
    pool["transition_parent_table_sha256"] = _table_sha256(parents, "parent_atom_id")
    pool["transition_mode_table_sha256"] = _table_sha256(modes, "mode_atom_id")
    pool["transition_mode_edge_table_sha256"] = _table_sha256(edges, "mode_edge_id")
    return pool


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def test_bounded_chain_has_constant_structural_output_and_three_identical_replays() -> None:
    pool = _bounded_chain_pool()
    input_bytes = canonical_json_bytes(pool)
    outputs = []
    started = perf_counter()
    for _index in range(3):
        outputs.append(
            build_local_plane_fragment_probe(
                project_lpf_input(deepcopy(pool)),
                scan_sha256="0" * 64,
                candidate_pool_sha256=hashlib.sha256(input_bytes).hexdigest(),
                a0_prediction_sha256="1" * 64,
            )
        )
    wall_seconds = perf_counter() - started
    output_bytes = [canonical_json_bytes(output) for output in outputs]
    output_hashes = [hashlib.sha256(value).hexdigest() for value in output_bytes]
    counters = outputs[0]["case_counters"]

    assert len(outputs[0]["local_plane_fragment_table"]) == 2
    assert len(outputs[0]["local_plane_pair_records"]) == 1
    assert counters["local_transition_component_count"] == 1
    assert counters["geometry_certificate_count"] == 1
    assert len(output_bytes[0]) < 100_000
    assert len(set(output_hashes)) == 1
    assert outputs[0]["local_plane_fragment_table"][0]["witness_ids_truncated"] is True
    assert outputs[0]["local_plane_pair_records"][0]["components"][0]["witness_ids_truncated"] is True

    row = {
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "input_json_bytes": len(input_bytes),
        "serialized_output_bytes": len(output_bytes[0]),
        "case_counters": {
            key: counters[key]
            for key in (
                "local_plane_fragment_count",
                "local_plane_pair_count",
                "local_transition_component_count",
                "geometry_certificate_count",
            )
        },
        "replay_probe_sha256s": output_hashes,
        "replay_count": 3,
    }
    validate_lpf_memory_profile_row(row)
    assert row["peak_rss_bytes"] < PEAK_RSS_LIMIT_BYTES


def test_memory_profile_fails_closed_on_limit_schema_and_replay_mismatch() -> None:
    valid = {
        "wall_seconds": 1.0,
        "peak_rss_bytes": 100_000_000,
        "input_json_bytes": 1000,
        "serialized_output_bytes": 2000,
        "case_counters": {
            "local_plane_fragment_count": 2,
            "local_plane_pair_count": 1,
            "local_transition_component_count": 1,
            "geometry_certificate_count": 1,
        },
        "replay_probe_sha256s": ["a" * 64] * 3,
        "replay_count": 3,
    }
    validate_lpf_memory_profile_row(valid)
    over = {**valid, "peak_rss_bytes": PEAK_RSS_LIMIT_BYTES}
    with pytest.raises(ValueError, match="16 GiB"):
        validate_lpf_memory_profile_row(over)
    mismatch = {**valid, "replay_probe_sha256s": ["a" * 64, "b" * 64, "a" * 64]}
    with pytest.raises(ValueError, match="not byte-identical"):
        validate_lpf_memory_profile_row(mismatch)
    extra = {**valid, "unexpected": 1}
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_lpf_memory_profile_row(extra)
