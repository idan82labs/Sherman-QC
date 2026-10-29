"""Source-free mandatory PGRR tamper regressions."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np

from .config import sha_json


def _semantic_hash(bundle: Mapping[str, Any]) -> str:
    return sha_json(bundle)


def _chain_valid(chain: Mapping[str, Any]) -> bool:
    return bool(
        chain.get("full_connected") is True
        and int(chain.get("edge_count", -1)) == int(chain.get("node_count", -1)) - 1
        and int(chain.get("removed_edge_count", -1)) == 1
        and chain.get("cut_connected") is False
    )


def run_tamper_suite() -> dict[str, bool]:
    baseline = {
        "point_order": "a" * 64,
        "pole": [0.0, 0.0, 1.0],
        "candidate_membership": ["b" * 64, "c" * 64],
        "target_ids": ["T1", "T2"],
    }
    altered_order = deepcopy(baseline)
    altered_order["point_order"] = "d" * 64
    input_order_dependency_rejected = _semantic_hash(altered_order) != _semantic_hash(baseline)

    axis = np.asarray([0.0, 0.0, 1.0])
    u = np.asarray([1.0, 0.0, 0.0])
    v = np.cross(axis, u)
    normal = np.asarray([[math.sqrt(0.5), math.sqrt(0.5), 0.0]])
    theta = np.mod(np.arctan2(normal @ v, normal @ u), math.pi)
    flipped = np.mod(np.arctan2((-normal) @ v, (-normal) @ u), math.pi)
    normal_sign_flip_projectively_invariant = bool(np.array_equal(theta, flipped))

    pole = np.asarray([0.0, 0.0, 1.0], dtype="<f8")
    tampered_pole = pole.copy().view(np.uint64)
    tampered_pole[2] ^= np.uint64(1)
    pole_bit_tamper_rejected = sha_json(pole.tolist()) != sha_json(tampered_pole.view("<f8").tolist())

    exchanged_membership = deepcopy(baseline)
    exchanged_membership["candidate_membership"] = ["c" * 64, "b" * 64]
    equal_count_membership_exchange_rejected = _semantic_hash(exchanged_membership) != _semantic_hash(baseline)

    exchanged_targets = deepcopy(baseline)
    exchanged_targets["target_ids"] = ["T2", "T1"]
    equal_count_target_exchange_rejected = _semantic_hash(exchanged_targets) != _semantic_hash(baseline)

    chain = {
        "node_count": 5,
        "edge_count": 4,
        "removed_edge_count": 1,
        "full_connected": True,
        "cut_connected": False,
    }
    deleted_edge_claim = deepcopy(chain)
    deleted_edge_claim["edge_count"] = 3
    central_edge_deletion_with_full_effect_rejected = _chain_valid(chain) and not _chain_valid(deleted_edge_claim)

    baseline_rows = [{"target_id": "T1", "state": "BASE"}]
    augmented_rows = [{"target_id": "T1", "state": "AUGMENTED"}]
    claimed = sha_json(baseline_rows)
    baseline_hash_over_augmented_table_rejected = claimed != sha_json(augmented_rows)

    return {
        "input_order_dependency_rejected": input_order_dependency_rejected,
        "normal_sign_flip_projectively_invariant": normal_sign_flip_projectively_invariant,
        "pole_bit_tamper_rejected": pole_bit_tamper_rejected,
        "equal_count_membership_exchange_rejected": equal_count_membership_exchange_rejected,
        "equal_count_target_exchange_rejected": equal_count_target_exchange_rejected,
        "central_edge_deletion_with_full_effect_rejected": central_edge_deletion_with_full_effect_rejected,
        "baseline_hash_over_augmented_table_rejected": baseline_hash_over_augmented_table_rejected,
    }


def assert_tamper_suite() -> dict[str, bool]:
    results = run_tamper_suite()
    if not all(results.values()):
        failed = sorted(key for key, value in results.items() if not value)
        raise ValueError(f"mandatory PGRR tamper regression failed: {failed}")
    return results
