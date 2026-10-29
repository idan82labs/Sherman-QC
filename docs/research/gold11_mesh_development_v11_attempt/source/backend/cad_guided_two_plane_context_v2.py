"""Explicit residual-depth context admission with unchanged V1 geometry.

SOURCE-ONLY successor to the terminal connected depth V1 quota failure.
V1 remains byte-identical and keeps four ordered targets per source pair.
V2 permits up to 256 ordered targets per pair, but NOT more than 256 total:
four signs and eight slots still cost 32 slots per ordered target. All input,
visit, cache, payload, rank, unit and projection controls are inherited unchanged.
No schedule is generated here; no CAD measurement or accepted pose is supplied.

Only the orchestration is versioned. The shared V1 numeric/lineage helpers are
called directly, without module mutation, batching, reordering or quota refunds.
Caller qualification remains a premise, and all failures discard usable output.
No tests, native run, physical score or production qualification is implied.
"""
from dataclasses import dataclass
import itertools

import numpy as np

from . import cad_guided_two_plane_context_v1 as _v1
from .cad_guided_two_plane_context_v1 import (
    ContextPlaneV1, OrderedPlanePairV1, _Unavailable, _require, _vector,
    _Work, _inputs, _planes, _schedule, _family, _slots, _SLOTS, _GATES,
    _final_report, _json,
)


@dataclass(frozen=True)
class TwoPlaneContextBudgetsV2(_v1.TwoPlaneContextBudgetsV1):
    max_ordered_cad_matches_per_scan_pair: int = 256


# Derived from immutable V1 maxima, never its caller-visible default instance.
_MAXIMA = tuple((name, 256 if name == "max_ordered_cad_matches_per_scan_pair" else maximum)
    for name, maximum in _v1._MAXIMA)
_MAX_ORDERED_MATCHES = 256


def _configuration(budgets):
    _require(type(budgets) is TwoPlaneContextBudgetsV2, "INVALID_BUDGET_TYPE")
    snapshot = {key: getattr(budgets, key) for key, _ in _MAXIMA}
    _require(all(type(snapshot[key]) is int and 0 <= snapshot[key] <= maximum
        for key, maximum in _MAXIMA) and snapshot["chunk_rows"] >= 3,
        "INVALID_BUDGET_VALUES")
    return snapshot


def _base():
    result = _v1._base()
    result.update(schema_version="two_plane_context_v2",
        scheduling_capacity=dict(
            policy="RESIDUAL_DEPTH_WITH_UNCHANGED_GLOBAL_SIGN_AND_SLOT_LIMITS",
            maximum_ordered_matches=_MAX_ORDERED_MATCHES,
            maximum_distinct_scan_pairs=64,
            maximum_ordered_matches_per_scan_pair=256,
            signs_per_ordered_match=4, slots_per_sign=8))
    return result


def _global_schedule_preflight(matches, cap):
    # Admit the full declared denominator before raw arrays/supports are read.
    # Per-pair identity, reversal and duplicate checks remain in V1 _schedule.
    maximum = min(_MAX_ORDERED_MATCHES,
        cap["max_scan_pairs"]*cap["max_ordered_cad_matches_per_scan_pair"],
        cap["max_signed_families"]//4, cap["max_candidate_slots"]//32)
    _require(type(matches) is tuple and len(matches) <= maximum,
        "ORDERED_MATCH_TABLE_BUDGET")


def propose_two_plane_context_v2(scan_points, scan_triangles, cad_triangles, cad_face_ids,
        cad_source_triangle_ids, scan_planes, cad_planes, ordered_matches, *,
        scan_origin_mm, cad_origin_mm, unit_assumption, source_frame, target_frame,
        budgets=TwoPlaneContextBudgetsV2()):
    """Return complete caller-schedule evidence, or NO usable proposal prefix.

    Arrays must be ordinary native float64/int64 NumPy arrays with listed shapes;
    plane/match tables must be exact immutable typed tuples. Scan support rows
    are rederived from input triangle rows; CAD supports must equal full face-ID
    membership. Plane qualification is caller-declared, not recomputed here.
    The caller owns fair scheduling and omitted-bank identities; this API cannot
    attest them. Both CAD orders count separately; depth may be concentrated
    within a pair, but the global 256-match / 8192-slot ceiling is unchanged.
    """
    result, work = _base(), None
    try:
        cap = _configuration(budgets)
        work = _Work(cap)
        work.memory()
        _global_schedule_preflight(ordered_matches, cap)
        _require(all(type(x) is str and 0 < len(x) <= 128 and x.isascii() and x.strip()
            for x in (unit_assumption, source_frame, target_frame)), "EXPLICIT_MM_AND_FRAME_DECLARATIONS_REQUIRED")
        origins = tuple(np.asarray(_vector(x)) for x in (scan_origin_mm, cad_origin_mm))
        result.update(unit="mm", unit_assumption=unit_assumption, source_frame=source_frame,
            target_frame=target_frame, source_origin_mm=origins[0].tolist(), target_origin_mm=origins[1].tolist(),
            policy=dict(_GATES), budgets=cap,
            counter_semantics=dict(
                scan_and_cad_visits="CHARGED_BEFORE_FULL_BUFFER_VALIDATION_OR_CACHE_MISS_PROJECTION",
                index_visits="CHARGED_BINDING_AND_PROJECTION_INDEX_ELEMENTS",
                sort_work="N_TIMES_CEIL_LOG2_N_DECLARED_ABSTRACT_COMPARISON_WORK_NOT_PROFILED_OPS",
                hash_bytes="OWNED_ARRAY_SHA256_INPUT_BYTES_CHARGED_BEFORE_HASH",
                working_bytes="OWNED_ARRAY_AND_TEMP_NUMERIC_PLUS_ENCODED_CACHE_PROPOSAL_PAYLOAD_NOT_PYTHON_RSS",
                attempted_overflow="ATTEMPTED_COUNTER_CAN_EXCEED_CAP_WITH_WHOLE_REFUSAL_BEFORE_OPERATION",
                slots="PLANNED32_PER_ORDERED_MATCH_WITH_EXPLICIT_REJECTED_OR_ABSENT_OUTCOMES"))
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            context = _inputs(scan_points, scan_triangles, cad_triangles, cad_face_ids,
                cad_source_triangle_ids, work, result)
            scans, cads = _planes(scan_planes, "scan", context, work, result), _planes(cad_planes, "cad", context, work, result)
            schedule = _schedule(ordered_matches, scans, cads, cap)
            result["scheduled_pairs"] = schedule
            work.record(schedule)
            work.stats.update(signed_families_planned=4*len(schedule), candidate_slots_planned=32*len(schedule))
            poses = {}
            for assignment_id, assignment in enumerate(schedule):
                source = [scans[x] for x in assignment["scan_plane_ids"]]
                target = [cads[x] for x in assignment["cad_plane_ids"]]
                for signs in itertools.product((-1, 1), repeat=2):
                    work.stats["signed_families_attempted"] += 1
                    family = dict(signed_family_id=len(result["families"]), ordered_assignment_id=assignment_id,
                        normal_signs=list(signs), correspondences=[dict(scan_plane_id=a["plane_id"],
                            cad_plane_id=b["plane_id"], normal_sign=s) for a, b, s in zip(source, target, signs)], slots=[])
                    result["families"].append(family)
                    family.update(_family(source, target, signs, *origins))
                    available = family["status"] == "AVAILABLE"
                    if available:
                        votes, rotation, t0, u = _slots(family, source, target, origins,
                            (source_frame, target_frame), context, work, result)
                    else:
                        votes = [None]*8
                    for slot, vote in zip(_SLOTS, votes):
                        row = dict(slot=slot, status="PAIR_REJECTED", lambda_mm=None, candidate_index=None)
                        if available:
                            work.stats["slots_generation_attempted"] += 1
                            if vote is None:
                                row["status"] = "ABSENT_EMPTY_HEURISTIC_INTERVAL"
                                work.stats["absent_midpoint_slots"] += 1
                            else:
                                local = t0+vote*u
                                translation = origins[1]+local-rotation@origins[0]
                                _require(np.isfinite(local).all() and np.isfinite(translation).all(), "NONFINITE_AXIAL_POSE")
                                values = np.concatenate((rotation.ravel(), translation))
                                # Numerical equality treats signed zeros equally;
                                # no tolerance-based merge of distinct axial votes.
                                key = np.where(values == 0., 0., values).tobytes()
                                witness = dict(signed_family_id=family["signed_family_id"], slot=slot)
                                if key not in poses:
                                    index = len(result["candidates"])
                                    poses[key] = index
                                    candidate = dict(candidate_index=index, rotation=rotation.tolist(),
                                        translation_mm=translation.tolist(), anchored_local_translation_mm=local.tolist(),
                                        contributing_slots=[witness], accepted_registration=False)
                                    work.record(candidate)
                                    work.memory(len(key))
                                    work.stats["cache_payload_bytes"] += len(key)
                                    result["candidates"].append(candidate)
                                else:
                                    index = poses[key]
                                    work.record(witness)
                                    result["candidates"][index]["contributing_slots"].append(witness)
                                    work.stats["exact_pose_duplicates"] += 1
                                row.update(status="PROPOSED_SEED_ONLY", lambda_mm=float(vote), candidate_index=index)
                                work.stats["proposed_slot_count"] += 1
                        else:
                            work.stats["rejected_slot_count"] += 1
                        family["slots"].append(row)
                        work.stats["slot_records_completed"] += 1
                    work.record(family)
                    work.stats["signed_families_completed"] += 1
            _require(work.stats["signed_families_completed"] == work.stats["signed_families_planned"]
                and work.stats["slot_records_completed"] == work.stats["candidate_slots_planned"], "INCOMPLETE_SCHEDULED_SLOT_LEDGER")
            result.update(status="COMPLETE", selected_policy_evaluation_complete=True,
                usable_proposals=bool(result["candidates"]), reasons=[])
        return _final_report(result, work)
    except Exception as error:
        # No central rotation/translation family, lambda or candidate survives a
        # late malformed/budget/numerical failure. Retain bounded attempted work;
        # no stringification of arbitrary caller exceptions enters the report.
        refusal = _base()
        refusal.update(reasons=[str(error) if type(error) is _Unavailable else "CONTEXT_HELPER_EXCEPTION"],
            evidence_scope="ATTEMPTED_COUNTERS_ONLY_NO_PARTIAL_POSE_OR_GEOMETRY_REPORT",
            discarded_signed_families=len(result["families"]), discarded_candidates=len(result["candidates"]),
            statistics={} if work is None else dict(work.stats))
        _require(len(_json(refusal)) <= 4096, "INTERNAL_REFUSAL_ENVELOPE_BUDGET")
        return refusal
