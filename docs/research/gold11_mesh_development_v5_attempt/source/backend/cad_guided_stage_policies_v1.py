"""Explicit stage semantics for the unqualified connected CAD challenger.

This module does not repair or relabel a frozen V3 query. Complete distance
evidence, a materialized optimization target, and usable tangent normals are
different predicates. A true predicate is never a pose-uniqueness certificate.
"""

from dataclasses import asdict, dataclass

import numpy as np

from . import cad_guided_global_registration_v3 as _v3


PROXIMITY = "COMPLETE_FINITE_DISTANCE_UPPER_WITHOUT_CORRESPONDENCE_CLAIM_V1"
PROPOSAL_TARGET = "UNCHANGED_V3_UNAMBIGUOUS_MATERIALIZED_TARGET_PROPOSAL_ONLY"
TANGENT = "UNCHANGED_V3_UNAMBIGUOUS_NONUNCERTAIN_FACE_INTERIOR"
MEASUREMENT = "SCAN_DERIVED_FREE_CIRCLE_QUANTITY_SPECIFIC_CONDITIONAL_V1"


@dataclass(frozen=True)
class StagePoliciesV1:
    proposal_proximity: str = PROXIMITY
    refinement_targets: str = PROPOSAL_TARGET
    final_proximity: str = PROXIMITY
    tangent_constraints: str = TANGENT
    measurement: str = MEASUREMENT


def snapshot_stage_policies_v1(value):
    """Only this exact tuple is implemented; arbitrary strings are not policies."""
    if type(value) is not StagePoliciesV1 or asdict(value) != asdict(StagePoliciesV1()):
        raise ValueError("UNSUPPORTED_CONNECTED_STAGE_POLICY_TUPLE")
    return asdict(StagePoliciesV1())


def ablation_stage_policies_v1(*, connected):
    if type(connected) is not bool:
        raise ValueError("INVALID_CONNECTED_CONTEXT_POLICY")
    result = snapshot_stage_policies_v1(StagePoliciesV1())
    if not connected:
        result.update(final_proximity=_v3.SUPPORT_POLICY,
                      measurement="UNCHANGED_SINGLE_CONDITIONAL_BEST_FIT_V11")
    return result


def classify_stage_rows_v1(query, gate):
    """Retain every complete row and all ambiguity diagnostics.

    Proximity concerns the represented point and represented finite triangle
    union. Scanner error, analytic CAD error and transform rounding are not
    included. A near-tied witness need not invalidate distance to that union.

    Optimizer targets and tangent rank retain the legacy conservative ambiguity
    exclusions. Failure of these predicates no longer erases independently
    complete proximity evidence. They are engineering, not interval-certified.
    """
    masks = _v3._support_masks(query, gate)  # Validates the entire query first.
    distance = masks["definite_support"]
    legacy = distance & _v3._unambiguous(query)
    tangent = (legacy & (query["closest_regions"] == "FACE")
               & ~query["region_uncertain"])
    return dict(
        definite_proximity=distance.copy(),
        definite_outside=masks["definite_outside"].copy(),
        unresolved_proximity=masks["unresolved"].copy(),
        proposal_target=legacy.copy(),
        tangent_constraint=tangent,
        possible_proximity=~masks["definite_outside"],
        legacy_unambiguous_support=legacy,
        correspondence_ambiguous=~_v3._unambiguous(query),
        face_identity_ambiguous=query["face_identity_ambiguous"].copy(),
    )


def stage_row_counts_v1(query, gate):
    masks = classify_stage_rows_v1(query, gate)
    return {key: int(np.count_nonzero(value)) for key, value in masks.items()}
