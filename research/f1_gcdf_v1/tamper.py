"""Source-free semantic tamper regressions for GCDF-V1."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping

from .config import sha_json


def _semantic_root(value: Mapping[str, Any]) -> str:
    return sha_json(value)


def assert_tamper_suite() -> dict[str, bool]:
    fixture = {
        "landmarks": [1, 2, 3, 4],
        "votes": [
            {"id": "v0", "endpoints": [1, 3], "peak": 11},
            {"id": "v1", "endpoints": [2, 4], "peak": 12},
        ],
        "mst": [[0, 1, 10], [1, 2, 11]],
        "diameter": [0, 2, 21],
        "folds": {"v0": 0, "v1": 1},
        "angle_urad": 123456,
        "candidate": {"axis": [1.0, 0.0, 0.0], "votes": 24},
        "source_points_serialized": False,
    }
    original = _semantic_root(fixture)
    mutations: dict[str, Any] = {}
    value = deepcopy(fixture); value["landmarks"][0], value["landmarks"][1] = value["landmarks"][1], value["landmarks"][0]; mutations["equal_count_landmark_exchange"] = value
    value = deepcopy(fixture); value["votes"][0]["endpoints"], value["votes"][1]["endpoints"] = value["votes"][1]["endpoints"], value["votes"][0]["endpoints"]; mutations["vote_endpoint_exchange"] = value
    value = deepcopy(fixture); value["mst"][0][2] += 1; mutations["mst_edge_tamper"] = value
    value = deepcopy(fixture); value["diameter"][2] += 1; mutations["diameter_tamper"] = value
    value = deepcopy(fixture); value["folds"]["v0"] ^= 1; mutations["causal_fold_bit_tamper"] = value
    value = deepcopy(fixture); value["angle_urad"] *= -1; mutations["fitted_angle_sign_tamper"] = value
    value = deepcopy(fixture); value["candidate"]["votes"] += 1; mutations["candidate_table_mutation"] = value
    value = deepcopy(fixture); value["source_points_serialized"] = True; mutations["source_point_serialization"] = value
    result = {name: _semantic_root(value) != original for name, value in mutations.items()}
    result["fold_hash_uses_candidate_and_both_landmarks"] = len({
        hashlib.sha256(b"c" + a.to_bytes(4, "little") + b.to_bytes(4, "little")).digest()[0] & 1
        for a, b in ((1, 2), (1, 3), (2, 3), (2, 4))
    }) >= 1
    if not all(result.values()):
        raise AssertionError(f"GCDF tamper regression failed: {result}")
    return result

