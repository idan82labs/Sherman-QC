"""Opt-in breadth-preserving residual-depth schedule; no poses or acceptance.

The first pass uses the unchanged V1 selection, index, windows, layout clocks,
matches and completion accounting. Its complete report is retained untouched in
``breadth_ledger``. Only its remaining joint reservation capacity is then spent
on omitted compatible targets for already selected source pairs, in descending
existing normal-independence margin with original selected-order ties.

Each residual source pair is visited once. Fair bins partition its omitted ranks;
their first positions are mapped through at most two original breadth positions.
Only selected int32 cache entries are read. There is no full cached-layout decode,
filter, second CAD index, second angle window, new geometric test or outcome refill.
Every target pair still reserves BOTH orders, 64 hypotheses and two unsigned
assignments. Lower-priority pairs can be starved. No recall/order-invariance,
physical independence, 48-start refinement or final-output monotonicity is claimed.

New sorting, visits, selected-rank/index work, references and fixed-schema reports
are precharged separately. Accounting is declared payload, not Python/process RSS.
V1 and this policy retain binary64 engineering arithmetic and caller-stipulated
uncalibrated cones. Any late refusal clears ALL usable matches, including breadth.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import struct

import numpy as np

from . import cad_guided_two_plane_schedule_v1 as _v1


@dataclass(frozen=True)
class TwoPlaneScheduleBudgetsV2(_v1.TwoPlaneScheduleBudgetsV1):
    max_residual_priority_work: int = 384
    max_residual_pair_visits: int = 64
    max_residual_breadth_position_tests: int = 128
    max_residual_rank_selections: int = 128
    max_residual_skip_position_tests: int = 512
    max_residual_index_lookups: int = 128
    max_residual_cache_reference_records: int = 64
    max_residual_cache_reference_bytes: int = 32768
    max_residual_retained_bytes: int = 1048576
    max_residual_report_records: int = 512
    max_residual_report_reserved_bytes: int = 1048576
    max_final_ledger_tests: int = 320


_EXTRA_MAXIMA = dict(max_residual_priority_work=384, max_residual_pair_visits=64,
    max_residual_breadth_position_tests=128, max_residual_rank_selections=128,
    max_residual_skip_position_tests=512, max_residual_index_lookups=128,
    max_residual_cache_reference_records=64, max_residual_cache_reference_bytes=32768,
    max_residual_retained_bytes=1048576, max_residual_report_records=512,
    max_residual_report_reserved_bytes=1048576, max_final_ledger_tests=320)
_MAXIMA = dict(_v1._MAXIMA, **_EXTRA_MAXIMA)


def _configuration(budgets):
    _v1._require(type(budgets) is TwoPlaneScheduleBudgetsV2, "INVALID_SCHEDULE_V2_BUDGET_TYPE")
    cap = {name: getattr(budgets, name) for name in _MAXIMA}
    _v1._require(all(type(value) is int and 0 <= value <= _MAXIMA[name]
        for name, value in cap.items()) and cap["chunk_pair_rows"] > 0,
        "INVALID_SCHEDULE_V2_BUDGET_VALUES")
    return cap


def _base():
    result = _v1._base()
    # These V1 tables have breadth-only counters. Keep them exclusively in the
    # explicitly scoped nested ledger, never as misleading empty/final tables.
    for name in ("scan_planes", "cad_planes", "scan_pairs", "cad_windows"):
        result.pop(name)
    result.update(schema_version="two_plane_schedule_v2", breadth_ledger=None,
        residual_windows=[], final_scan_pair_ledger=[], breadth_statistics={},
        residual_statistics={},
        scheduling_policy="EXACT_V1_BREADTH_THEN_DESCENDING_NORMAL_INDEPENDENCE_MARGIN_RESIDUAL_DEPTH",
        cad_schedule="V1_BREADTH_THEN_FAIR_OMITTED_RANKS_BOTH_ORDERS_NO_REFILL",
        breadth_ledger_scope="UNCHANGED_COMPLETE_V1_REPORT_NOT_FINAL_V2_COUNTS",
        residual_rank_policy="ONE_FIRST_FAIR_BIN_VISIT_OVER_OMITTED_RANKS_PER_SELECTED_SOURCE_PAIR",
        residual_ties="ORIGINAL_SELECTED_ORDER_THEN_SCAN_PAIR_ID",
        cached_layout_access="SELECTED_LITTLE_ENDIAN_INT32_DIRECT_LOOKUPS_NO_FULL_CACHE_SCAN",
        residual_skip_test_scope="MAPPING_LOOP_PLUS_AT_MOST_TWO_EXCLUDED_POSITION_VALIDATION_COMPARISONS",
        final_ledger_test_scope="ONE_FIXED_SCHEMA_ROW_VALIDATION_PER_MATCH_OR_SELECTED_PAIR_NOT_SCALAR_COMPARISONS",
        proposal_prefix_preservation="CONDITIONAL_ON_COMPLETE_V2_EVALUATION_NOT_FINAL_OUTPUT_MONOTONICITY")
    return result


class _ResidualWork:
    def __init__(self, cap, breadth_work):
        self.cap = cap
        # Retained V1 allocations remain live; its completed counters never change.
        self.base_bytes = sum(breadth_work.stats[name] for name in (
            "numeric_owned_bytes", "temporary_bytes", "compatible_layout_bytes",
            "report_reserved_bytes", "fixed_scratch_bytes"))
        self.stats = dict(residual_priority_work=0, residual_pair_visits=0,
            residual_breadth_position_tests=0, residual_rank_selections=0,
            residual_skip_position_tests=0, residual_index_lookups=0,
            residual_cache_reference_records=0, residual_cache_reference_bytes=0,
            residual_retained_bytes=0, residual_report_records=0,
            residual_report_reserved_bytes=0, residual_temporary_bytes=0,
            residual_fixed_scratch_bytes=65536, final_ledger_tests=0,
            residual_scheduled_unordered_occurrences=0, residual_scheduled_ordered_matches=0,
            residual_reserved_hypothesis_records=0, residual_completed_windows=0,
            residual_cad_index_builds=0, residual_geometry_window_tests=0,
            working_peak_bytes=breadth_work.stats["working_peak_bytes"])

    def charge(self, field, amount):
        _v1._require(type(amount) is int and amount >= 0, "INVALID_RESIDUAL_WORK_CHARGE")
        self.stats[field] += amount
        _v1._require(self.stats[field] <= self.cap["max_"+field],
            "SCHEDULE_"+field.upper()+"_BUDGET")

    def memory(self, extra=0):
        total = self.base_bytes + sum(self.stats[name] for name in (
            "residual_cache_reference_bytes", "residual_retained_bytes",
            "residual_report_reserved_bytes", "residual_temporary_bytes",
            "residual_fixed_scratch_bytes")) + extra
        self.stats["working_peak_bytes"] = max(self.stats["working_peak_bytes"], total)
        _v1._require(total <= self.cap["max_working_bytes"], "SCHEDULE_V2_WORKING_PAYLOAD_BUDGET")

    def retain(self, size):
        self.charge("residual_retained_bytes", size)
        self.memory()

    @contextmanager
    def temporary(self, size):
        self.stats["residual_temporary_bytes"] += size
        try:
            self.memory()
            yield
        finally:
            self.stats["residual_temporary_bytes"] -= size

    def reserve_record(self, reservation):
        # Call BEFORE allocating each new fixed-schema record.
        self.charge("residual_report_records", 1)
        self.charge("residual_report_reserved_bytes", reservation)
        self.memory()


def _breadth_pass(scan_equations, cad_equations, counts, cap, state):
    """The exact V1 first pass; retain its sole CAD table and exact layout keys."""
    retained_h, retained_u, max_h, max_u = counts
    breadth_cap = {name: cap[name] for name in _v1._MAXIMA}
    result = _v1._base(); work = _v1._Work(breadth_cap)
    state.update(result=result, work=work)
    work.memory()
    _v1._require(all(type(value) is int and value >= 0 for value in counts)
        and retained_h <= max_h <= 8192 and retained_u <= max_u <= 1024,
        "INVALID_RETAINED_TRIPLET_OR_JOINT_BUDGET")
    remaining = min((max_h-retained_h)//32, max_u-retained_u)
    result.update(policy=dict(_v1._GATES), budgets=breadth_cap, joint_budget=dict(
        retained_triplet_hypothesis_count=retained_h, retained_triplet_unsigned_count=retained_u,
        max_joint_hypothesis_records=max_h, max_joint_unsigned_assignments=max_u,
        ordered_match_capacity_before_both_order_rounding=remaining,
        both_order_bundle_capacity=remaining//2, unused_odd_ordered_capacity=remaining % 2,
        count_provenance="CALLER_DECLARED_RETAINED_TRIPLET_COUNTS_ADAPTER_MUST_VERIFY"))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scan = _v1._equations(scan_equations, cap["max_scan_planes"], "scan", work, result)
        cad = _v1._equations(cad_equations, cap["max_cad_planes"], "cad", work, result)
        selected = _v1._scan_pairs(scan, work, result)
        table, layouts = None, {}
        for pair_id in selected:
            row = result["scan_pairs"][pair_id]
            if remaining < 2:
                row["status"] = "UNSCHEDULED_TRIPLET_FIRST_JOINT_QUOTA"
                continue
            if table is None:
                table = _v1._cad_index(cad, work, result)
            pair_ids = _v1._window(row, cad, table, remaining, layouts, work, result)
            for pair in pair_ids:
                a, b = int(table["left"][pair]), int(table["right"][pair])
                for first, second in ((a, b), (b, a)):
                    match = dict(ordered_match_index=len(result["scheduled_matches"]),
                        scan_pair_id=pair_id, scan_plane_ids=list(row["scan_plane_ids"]),
                        cad_plane_ids=[cad[first]["plane_id"], cad[second]["plane_id"]],
                        canonical_cad_pair_index=pair, reserved_hypothesis_records=32,
                        reserved_unsigned_assignments=1)
                    work.record(match, 512); result["scheduled_matches"].append(match)
                    work.stats["scheduled_ordered_matches"] += 1
                    work.stats["reserved_two_plane_slots"] += 32
                    row["scheduled_ordered_matches"] += 1; remaining -= 1
                    for index in row["bank_indices"]:
                        result["scan_planes"][index]["scheduled_pair_occurrences"] += 1
                        result["scan_planes"][index]["disposition"] = "USED_IN_SCHEDULED_MATCH"
                    for index in (first, second):
                        result["cad_planes"][index]["scheduled_pair_occurrences"] += 1
                        result["cad_planes"][index]["disposition"] = "USED_IN_SCHEDULED_MATCH"
            row["status"] = "EVALUATED_WITH_SCHEDULED_MATCHES" if pair_ids else "EVALUATED_NO_COMPATIBLE_CAD_PAIR"
    _v1._require(work.stats["scheduled_ordered_matches"] == len(result["scheduled_matches"])
        and work.stats["reserved_two_plane_slots"] == 32*len(result["scheduled_matches"])
        and work.stats["completed_windows"] == len(result["cad_windows"])
        and len(result["scheduled_matches"]) <= 256
        and retained_h+work.stats["reserved_two_plane_slots"] <= max_h
        and retained_u+len(result["scheduled_matches"]) <= max_u
        and all(result["scan_pairs"][index]["scheduled_ordered_matches"] in (0, 2, 4) for index in selected),
        "INTERNAL_COMPLETE_SLOT_OR_WINDOW_LEDGER_MISMATCH")
    result.update(status="COMPLETE", selected_policy_evaluation_complete=True,
        usable_schedule=bool(result["scheduled_matches"]), reasons=[])
    result["joint_budget"].update(reserved_two_plane_hypothesis_records=work.stats["reserved_two_plane_slots"],
        reserved_two_plane_unsigned_assignments=work.stats["scheduled_ordered_matches"],
        remaining_ordered_match_capacity=remaining,
        joint_hypothesis_records_reserved=retained_h+work.stats["reserved_two_plane_slots"],
        joint_unsigned_assignments_reserved=retained_u+work.stats["scheduled_ordered_matches"],
        pair_lane_starved_by_joint_quota=bool(selected and not result["cad_windows"] and remaining < 2))
    result["statistics"] = dict(work.stats)
    encoded = _v1._size(result, cap["max_report_bytes"])
    work.memory(encoded)
    result["statistics"] = dict(work.stats)
    _v1._size(result, cap["max_report_bytes"])
    return result, work, cad, table, layouts, remaining


def _cache_references(layouts, work):
    """At most 64 dictionary entries, never the millions of IDs inside keys."""
    work.charge("residual_cache_reference_records", len(layouts))
    work.charge("residual_cache_reference_bytes", 512*len(layouts))
    work.memory()
    references = [None]*len(layouts)
    for key, entry in layouts.items():
        index = entry["layout_index"]
        _v1._require(type(key) is bytes and type(index) is int
            and 0 <= index < len(references) and references[index] is None,
            "INVALID_RETAINED_COMPATIBLE_LAYOUT_INDEX")
        references[index] = (key, entry["sha256"])
    return references


def _residual_positions(window, reference, remaining, work):
    """Map selected omitted ranks by skipping at most two breadth positions."""
    used = window["scheduled_compatible_positions"]
    count = window["compatible_unordered_count"]
    work.charge("residual_breadth_position_tests", len(used))
    _v1._require(len(used) <= 2 and len(set(used)) == len(used)
        and all(type(index) is int and 0 <= index < count for index in used),
        "INVALID_BREADTH_COMPATIBLE_POSITION_LEDGER")
    key, digest = reference
    _v1._require(len(key) == 4*count and digest == window["compatible_pair_ids_sha256"],
        "RETAINED_EXACT_COMPATIBLE_LAYOUT_MISMATCH")
    omitted = count-len(used)
    take = min(omitted, remaining//2)
    work.charge("residual_rank_selections", take)
    work.charge("residual_skip_position_tests", 2*take*len(used))
    work.charge("residual_index_lookups", take)
    # Account retained rank/position/pair-index lists BEFORE their construction.
    work.retain(256+128*take)
    excluded = sorted(used)
    ranks = _v1._fair_positions(omitted, take, 0)
    positions, pair_ids = [], []
    for rank in ranks:
        position = rank
        for prior in excluded:
            if prior <= position:
                position += 1
        _v1._require(0 <= position < count and position not in excluded,
            "INTERNAL_OMITTED_RANK_MAPPING_MISMATCH")
        positions.append(position)
        pair_ids.append(struct.unpack_from("<i", key, 4*position)[0])
    return ranks, positions, pair_ids


def _residual_pass(breadth, cad, table, layouts, remaining, work, result):
    references = _cache_references(layouts, work)
    selected = breadth["selected_scan_pair_ids"]
    work.charge("residual_priority_work", len(selected)*max(0, (len(selected)-1).bit_length()))
    work.retain(512*len(selected)+256*len(breadth["cad_windows"])
        +16*len(breadth["scheduled_matches"]))
    priority = sorted(selected, key=lambda index: (
        -breadth["scan_pairs"][index]["normal_independence_margin"],
        breadth["scan_pairs"][index]["selected_order"], index))
    windows = {window["scan_pair_id"]: window for window in breadth["cad_windows"]}
    result["scheduled_matches"] = list(breadth["scheduled_matches"])
    residual_counts = {index: 0 for index in selected}
    for order, pair_id in enumerate(priority):
        work.charge("residual_pair_visits", 1)
        window = windows.get(pair_id)
        if window is None:
            continue  # V1 never opened this window; no new geometric access.
        row = breadth["scan_pairs"][pair_id]
        ranks, positions, pair_ids = _residual_positions(
            window, references[window["layout_index"]], remaining, work)
        reservation = 8192+64*len(pair_ids)
        work.reserve_record(reservation)
        record = dict(scan_pair_id=pair_id, residual_priority_order=order,
            normal_independence_margin=row["normal_independence_margin"],
            original_selected_order=row["selected_order"],
            layout_index=window["layout_index"], compatible_pair_ids_sha256=window["compatible_pair_ids_sha256"],
            compatible_pair_ids_bytes=window["compatible_pair_ids_bytes"],
            compatible_unordered_count=window["compatible_unordered_count"],
            breadth_scheduled_unordered_count=len(window["scheduled_canonical_pair_indices"]),
            breadth_omitted_compatible_unordered_count=window["omitted_compatible_unordered_count"],
            scheduled_omitted_ranks=ranks, scheduled_compatible_positions=positions,
            scheduled_canonical_pair_indices=pair_ids,
            residual_scheduled_unordered_count=len(pair_ids),
            final_omitted_compatible_unordered_count=window["omitted_compatible_unordered_count"]-len(pair_ids),
            remaining_ordered_capacity_before=remaining, actual_bin_count=len(pair_ids), occurrence_index=0,
            coverage_scope="ONE_FAIR_PARTITION_OF_THIS_OMITTED_SET_NOT_GLOBAL_COVERAGE",
            both_target_orders_required=True, reserved_hypothesis_records=64*len(pair_ids),
            reserved_unsigned_assignments=2*len(pair_ids),
            direct_cached_index_lookups=len(pair_ids), full_cached_index_visits=0,
            compatible_layout_reused_without_geometry=True, complete=True)
        _v1._size(record, reservation)
        result["residual_windows"].append(record)
        for pair in pair_ids:
            _v1._require(table is not None and 0 <= pair < len(table["left"]),
                "INVALID_RETAINED_CANONICAL_CAD_PAIR_INDEX")
            a, b = int(table["left"][pair]), int(table["right"][pair])
            for first, second in ((a, b), (b, a)):
                work.reserve_record(512)
                match = dict(ordered_match_index=len(result["scheduled_matches"]),
                    scan_pair_id=pair_id, scan_plane_ids=list(row["scan_plane_ids"]),
                    cad_plane_ids=[cad[first]["plane_id"], cad[second]["plane_id"]],
                    canonical_cad_pair_index=pair, reserved_hypothesis_records=32,
                    reserved_unsigned_assignments=1)
                _v1._size(match, 512); result["scheduled_matches"].append(match)
                work.stats["residual_scheduled_ordered_matches"] += 1
                work.stats["residual_reserved_hypothesis_records"] += 32
                residual_counts[pair_id] += 1; remaining -= 1
        work.stats["residual_scheduled_unordered_occurrences"] += len(pair_ids)
        work.stats["residual_completed_windows"] += 1
    return remaining, windows, residual_counts


def _finish(result, breadth, remaining, windows, residual_counts, work):
    matches = result["scheduled_matches"]
    work.charge("final_ledger_tests", len(matches)+len(breadth["selected_scan_pair_ids"]))
    work.retain(256*len(residual_counts))
    actual = {index: 0 for index in residual_counts}
    for index, match in enumerate(matches):
        _v1._require(match["ordered_match_index"] == index
            and match["scan_pair_id"] in actual and match["reserved_hypothesis_records"] == 32
            and match["reserved_unsigned_assignments"] == 1,
            "INTERNAL_FINAL_MATCH_LEDGER_MISMATCH")
        if index < len(breadth["scheduled_matches"]):
            _v1._require(match == breadth["scheduled_matches"][index], "V1_BREADTH_PREFIX_CHANGED")
        actual[match["scan_pair_id"]] += 1
    omitted = 0
    for pair_id in breadth["selected_scan_pair_ids"]:
        row, window = breadth["scan_pairs"][pair_id], windows.get(pair_id)
        expected = row["scheduled_ordered_matches"]+residual_counts[pair_id]
        _v1._require(actual[pair_id] == expected and expected % 2 == 0,
            "INTERNAL_FINAL_PAIR_LEDGER_MISMATCH")
        final_omitted = None if window is None else window["compatible_unordered_count"]-expected//2
        _v1._require(final_omitted is None or final_omitted >= 0, "NEGATIVE_FINAL_COMPATIBLE_OMISSION")
        omitted += 0 if final_omitted is None else final_omitted
        work.reserve_record(1024)
        record = dict(scan_pair_id=pair_id, breadth_status=row["status"],
            breadth_scheduled_ordered_matches=row["scheduled_ordered_matches"],
            residual_scheduled_ordered_matches=residual_counts[pair_id], scheduled_ordered_matches=expected,
            compatible_unordered_count=None if window is None else window["compatible_unordered_count"],
            final_omitted_compatible_unordered_count=final_omitted,
            compatible_window_was_evaluated=window is not None,
            omission_scope="POLICY_OMISSION_NOT_GEOMETRIC_DISPROOF_UNKNOWN_IF_BREADTH_WINDOW_UNVISITED")
        _v1._size(record, 1024); result["final_scan_pair_ledger"].append(record)
    joint = breadth["joint_budget"]
    depth = work.stats
    _v1._require(len(matches) == len(breadth["scheduled_matches"])+depth["residual_scheduled_ordered_matches"]
        and len(matches) <= 256 and len(matches) % 2 == 0
        and depth["residual_scheduled_ordered_matches"] == 2*depth["residual_scheduled_unordered_occurrences"]
        and depth["residual_reserved_hypothesis_records"] == 32*depth["residual_scheduled_ordered_matches"]
        and omitted == breadth["statistics"]["omitted_compatible_occurrences"]-depth["residual_scheduled_unordered_occurrences"]
        and joint["retained_triplet_hypothesis_count"]+32*len(matches) <= joint["max_joint_hypothesis_records"]
        and joint["retained_triplet_unsigned_count"]+len(matches) <= joint["max_joint_unsigned_assignments"]
        and remaining == joint["ordered_match_capacity_before_both_order_rounding"]-len(matches)
        and depth["residual_completed_windows"] == len(result["residual_windows"]),
        "INTERNAL_FINAL_RESERVATION_OR_OMISSION_LEDGER_MISMATCH")
    work.reserve_record(8192)
    result.update(status="COMPLETE", reasons=[], selected_policy_evaluation_complete=True,
        usable_schedule=bool(matches), breadth_ledger=breadth, breadth_statistics=breadth["statistics"],
        input_digests=breadth["input_digests"], policy=dict(_v1._GATES), budgets=dict(work.cap),
        selected_scan_pair_ids=breadth["selected_scan_pair_ids"],
        joint_budget=dict(joint, reserved_two_plane_hypothesis_records=32*len(matches),
            reserved_two_plane_unsigned_assignments=len(matches), remaining_ordered_match_capacity=remaining,
            joint_hypothesis_records_reserved=joint["retained_triplet_hypothesis_count"]+32*len(matches),
            joint_unsigned_assignments_reserved=joint["retained_triplet_unsigned_count"]+len(matches)),
        statistics=dict(breadth_scheduled_ordered_matches=len(breadth["scheduled_matches"]),
            residual_scheduled_ordered_matches=depth["residual_scheduled_ordered_matches"],
            scheduled_ordered_matches=len(matches), reserved_two_plane_slots=32*len(matches),
            compatible_occurrences=breadth["statistics"]["compatible_occurrences"],
            scheduled_unordered_occurrences=len(matches)//2, omitted_compatible_occurrences=omitted,
            completed_breadth_windows=len(breadth["cad_windows"]),
            completed_residual_windows=len(result["residual_windows"])))
    result["residual_statistics"] = dict(work.stats)
    encoded = _v1._size(result, work.cap["max_report_bytes"])
    work.memory(encoded)
    result["residual_statistics"] = dict(work.stats)
    _v1._size(result, work.cap["max_report_bytes"])


def schedule_two_plane_pairs_v2(scan_equations, cad_equations, *, retained_triplet_hypothesis_count,
        retained_triplet_unsigned_count, max_joint_hypothesis_records=8192,
        max_joint_unsigned_assignments=1024, budgets=TwoPlaneScheduleBudgetsV2()):
    """Explicit V2 opt-in; V1 defaults and all downstream gates are unchanged."""
    result, state, work = _base(), {}, None
    try:
        cap = _configuration(budgets)
        counts = (retained_triplet_hypothesis_count, retained_triplet_unsigned_count,
                  max_joint_hypothesis_records, max_joint_unsigned_assignments)
        breadth, breadth_work, cad, table, layouts, remaining = _breadth_pass(
            scan_equations, cad_equations, counts, cap, state)
        work = _ResidualWork(cap, breadth_work)
        work.memory()  # Keep attempted residual counters even if initial memory admission fails.
        work.reserve_record(8192)
        remaining, windows, residual_counts = _residual_pass(
            breadth, cad, table, layouts, remaining, work, result)
        _finish(result, breadth, remaining, windows, residual_counts, work)
        return result
    except Exception as error:
        refusal = _base()
        breadth = state.get("result", {})
        attempted_breadth = len(breadth.get("scheduled_matches", []))
        refusal.update(reasons=[str(error) if type(error) is _v1._Unavailable else "SCHEDULE_V2_HELPER_EXCEPTION"],
            discarded_scheduled_matches=max(attempted_breadth, len(result["scheduled_matches"])),
            discarded_breadth_scheduled_matches=attempted_breadth,
            discarded_residual_completed_windows=len(result["residual_windows"]),
            evidence_scope="ATTEMPTED_COUNTERS_ONLY_NO_USABLE_BREADTH_OR_RESIDUAL_PREFIX",
            breadth_statistics={} if "work" not in state else dict(state["work"].stats),
            residual_statistics={} if work is None else dict(work.stats))
        _v1._size(refusal, 16384)
        return refusal
