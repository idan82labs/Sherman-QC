"""Bounded, caller-bank two-plane schedule; no poses or geometric acceptance.

The selected <=128 scan bank and <=2048 CAD equations are UNVERIFIED caller
declarations. The connecting adapter must bind qualified equations to complete
owned raw supports/full faces. This module never reads geometry, files or truth.
IDs and input bank order are preserved; numeric ties use bank/index order, not
case labels. Neither ID-renumbering/order invariance nor exhaustive search is
claimed. Triplets reserve the joint quota first and can starve this lane.

Every selected unordered CAD pair reserves BOTH target orders (64 slots total).
No failed sign, duplicate pose or absent midpoint refunds a reserved slot.
Exact compatible pair-index bytes, not a digest or angular approximation, identify
repeated layouts. Fair-bin coverage concerns repeated identical layout exposure,
not each source pair, a correct physical correspondence or all proposal lanes.
For L exact ordered entries and m nonempty integer bins, bin j is
[floor(j*L/m), floor((j+1)*L/m)); its visit q chooses start+q mod length.
Each rank is exposed after max bin length=ceil(L/m) occurrences at that fixed
actual bin count. Separate clocks are used when residual quota changes m.

Worst declared index/window work is C(2048,2)=2,096,128 CAD pairs and 64 full
windows (134,152,192 pair tests); sort work is the explicit N*ceil(log2 N)
proxy. Up to 128 MiB of exact layout keys and 256 MiB of accounted payload are
admitted. These are capacity limits, not a runtime or physical recovery claim.

Working accounting covers numeric arrays, immutable layout keys and conservative
fixed-schema report reservations, NOT Python allocator/process RSS. Window scans
are complete and chunked; unexpected input/work/numeric/byte failure clears every
usable match. Conditional binary64 rank/angle arithmetic is engineering evidence,
not an interval certificate or calibrated cone model. No downstream gate changes.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from . import cad_guided_plane_search_v6 as _search


@dataclass(frozen=True)
class TwoPlaneScheduleBudgetsV1:
    max_scan_planes: int = 128
    max_cad_planes: int = 2048
    max_scan_pair_records: int = 8128
    max_selected_scan_pairs: int = 64
    max_scan_selection_tests: int = 520192
    max_cad_pair_records: int = 2096128
    max_cad_sort_work: int = 44018688
    max_window_pair_tests: int = 134152192
    max_compatible_layout_bytes: int = 134217728
    max_compatible_layouts: int = 64
    max_working_bytes: int = 268435456
    max_report_bytes: int = 33554432
    chunk_pair_rows: int = 65536


_MAXIMA = dict(max_scan_planes=128, max_cad_planes=2048, max_scan_pair_records=8128,
    max_selected_scan_pairs=64, max_scan_selection_tests=520192,
    max_cad_pair_records=2096128, max_cad_sort_work=44018688, max_window_pair_tests=134152192,
    max_compatible_layout_bytes=134217728, max_compatible_layouts=64,
    max_working_bytes=268435456, max_report_bytes=33554432, chunk_pair_rows=65536)
_GATES = dict(min_normal_condition_ratio=.02, max_normal_angle_radians=math.pi/18.,
              pair_angle_numeric_guard_radians=1e-10)
_FIELDS = ("plane_id", "normal", "anchor_mm", "offset_from_anchor_mm", "normal_angle_bound_radians")


class _Unavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unavailable(reason)


def _size(value, maximum):
    count = 0
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        count += len(part.encode())
        _require(count <= maximum, "SCHEDULE_REPORT_BYTE_BUDGET")
    return count


class _Work:
    def __init__(self, cap):
        self.cap = cap
        self.stats = dict(scan_pair_tests=0, scan_selection_tests=0, cad_pair_tests=0, cad_sort_work=0,
            window_pair_tests=0, completed_windows=0, numeric_owned_bytes=0,
            temporary_bytes=0, compatible_layout_bytes=0, compatible_layouts=0,
            report_reserved_bytes=0, fixed_scratch_bytes=65536, working_peak_bytes=0,
            compatible_occurrences=0, scheduled_unordered_occurrences=0,
            omitted_compatible_occurrences=0, scheduled_ordered_matches=0,
            reserved_two_plane_slots=0, repeated_layout_hits=0)

    def memory(self, extra=0):
        total = sum(self.stats[key] for key in ("numeric_owned_bytes", "temporary_bytes",
            "compatible_layout_bytes", "report_reserved_bytes", "fixed_scratch_bytes"))+extra
        self.stats["working_peak_bytes"] = max(total, self.stats["working_peak_bytes"])
        _require(total <= self.cap["max_working_bytes"], "SCHEDULE_WORKING_PAYLOAD_BUDGET")

    def charge(self, field, amount, maximum):
        _require(type(amount) is int and amount >= 0, "INVALID_SCHEDULE_WORK_CHARGE")
        self.stats[field] += amount
        _require(self.stats[field] <= self.cap[maximum], "SCHEDULE_"+field.upper()+"_BUDGET")

    def array(self, length, dtype):
        size = length*np.dtype(dtype).itemsize
        self.memory(size)
        answer = np.empty(length, dtype=dtype)
        self.stats["numeric_owned_bytes"] += size
        return answer

    @contextmanager
    def temporary(self, size):
        self.stats["temporary_bytes"] += size
        try:
            self.memory()
            yield
        finally:
            self.stats["temporary_bytes"] -= size

    def record(self, value, reservation):
        _size(value, reservation)
        self.memory(reservation)
        self.stats["report_reserved_bytes"] += reservation


def _base():
    return dict(schema_version="two_plane_schedule_v1", status="UNAVAILABLE", reasons=[],
        scheduled_matches=[], scan_planes=[], cad_planes=[], scan_pairs=[], cad_windows=[],
        selected_scan_pair_ids=[], statistics={}, input_digests={},
        selected_policy_evaluation_complete=False, usable_schedule=False,
        search_exhaustive=False, global_pose_uniqueness="NOT_ESTABLISHED",
        accepted_registration=False, production_use_allowed=False, observed_bend_count=None,
        normal_cone_status="CALLER_STIPULATED_UNCALIBRATED",
        finite_support_match_verified=False, translation_uncertainty="NOT_COMPUTED",
        support_provenance="UNVERIFIED_CALLER_BANK_EQUATIONS_ADAPTER_MUST_BIND_COMPLETE_SUPPORT",
        scheduling_policy="TRIPLET_FIRST_RESIDUAL_QUOTA_UNCOVERED_SCAN_PARTICIPATION_THEN_CONDITIONING",
        cad_schedule="TWO_FAIR_EXACT_COMPATIBLE_UNORDERED_PAIRS_BOTH_ORDERS_NO_REFILL",
        rank_arithmetic="BINARY64_TWO_ROW_SINGULAR_VALUES_FROM_CROSS_AND_GRAM_NOT_CERTIFIED",
        search_incompleteness="BANK_PAIR_AND_CAD_SCHEDULE_OMISSIONS_NOT_GEOMETRIC_DISPROOF",
        memory_scope="ACCOUNTED_NUMERIC_KEY_AND_REPORT_RESERVATIONS_NOT_TOTAL_PYTHON_RSS",
        counter_scope="CHARGED_ATTEMPTS_BEFORE_DISPATCH_INCLUDING_CAP_REFUSALS_COMPLETIONS_SEPARATE",
        sort_work_scope="N_TIMES_CEIL_LOG2_N_DECLARED_PROXY_NOT_MEASURED_COMPARISONS",
        identity_ties="PRESERVED_BANK_AND_CAD_TABLE_ORDER_NOT_RENUMBERING_OR_ORDER_INVARIANT")


def _configuration(budgets):
    _require(type(budgets) is TwoPlaneScheduleBudgetsV1, "INVALID_SCHEDULE_BUDGET_TYPE")
    values = {key: getattr(budgets, key) for key in _MAXIMA}
    _require(all(type(value) is int and 0 <= value <= _MAXIMA[key] for key, value in values.items())
        and values["chunk_pair_rows"] > 0, "INVALID_SCHEDULE_BUDGET_VALUES")
    return values


def _equations(values, maximum, role, work, result):
    _require(type(values) is tuple and len(values) <= maximum, "INVALID_"+role.upper()+"_BANK")
    owned, seen = [], set()
    for index, value in enumerate(values):
        _require(type(value) is _search.PlaneEquationV6, "EXACT_PLANE_EQUATION_V6_REQUIRED")
        snapshot = _search.PlaneEquationV6(**{name: getattr(value, name) for name in _FIELDS})
        try:
            equation = _search._equation(snapshot)
        except Exception:
            raise _Unavailable("INVALID_"+role.upper()+"_EQUATION") from None
        _require(equation["plane_id"] not in seen, "DUPLICATE_"+role.upper()+"_PLANE_ID")
        _require(equation["normal_angle_bound_radians"] <= _GATES["max_normal_angle_radians"],
                 "EQUATION_CONE_OUTSIDE_TWO_PLANE_HELPER_SCOPE")
        seen.add(equation["plane_id"])
        row = dict(equation, normal=list(equation["normal"]), anchor_mm=list(equation["anchor_mm"]),
            bank_index=index, scheduled_pair_occurrences=0,
            disposition="NOT_USED_IN_SCHEDULED_MATCH")
        work.record(row, 1024)
        result[role+"_planes"].append(row)
        owned.append(equation)
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(owned):
        digest.update(part.encode())
    result["input_digests"][role+"_bank_equations_sha256"] = digest.hexdigest()
    return owned


def _pair_geometry(first, second, first_alpha, second_alpha):
    """Actual represented 2x3 rows: s_max from Gram, s_min=||a x b||/s_max.

    This avoids subtracting almost equal Gram eigenvalues for s_min; it does
    not turn floating-point singular values/cone margins into certified bounds.
    """
    cross = np.cross(first, second)
    sine = np.sqrt(np.sum(cross*cross, axis=-1))
    dot = np.sum(first*second, axis=-1)
    aa, bb = np.sum(first*first, axis=-1), np.sum(second*second, axis=-1)
    largest = np.sqrt((aa+bb+np.hypot(aa-bb, 2.*dot))/2.)
    smallest = sine/largest
    eta = np.hypot(2.*np.sin(first_alpha/2.), 2.*np.sin(second_alpha/2.))
    ratio, margin = smallest/largest, smallest-eta
    angle = np.arctan2(sine, np.abs(dot))
    _require(np.isfinite(ratio).all() and np.isfinite(margin).all() and np.isfinite(angle).all(),
             "NONFINITE_PAIR_NORMAL_ARITHMETIC")
    return angle, ratio, margin


def _scan_pairs(scan, work, result):
    count = len(scan); planned = count*(count-1)//2
    _require(planned <= work.cap["max_scan_pair_records"], "SCAN_PAIR_PREFLIGHT_BUDGET")
    normals = np.asarray([row["normal"] for row in scan])
    eligible = []
    for first in range(count):
        for second in range(first+1, count):
            work.charge("scan_pair_tests", 1, "max_scan_pair_records")
            a, b = scan[first], scan[second]
            angle, ratio, margin = _pair_geometry(normals[first], normals[second],
                a["normal_angle_bound_radians"], b["normal_angle_bound_radians"])
            admitted = bool(ratio >= _GATES["min_normal_condition_ratio"] and margin > 0.)
            row = dict(scan_pair_id=len(result["scan_pairs"]), bank_indices=[first, second],
                scan_plane_ids=[a["plane_id"], b["plane_id"]], angle_radians=float(angle),
                normal_condition_ratio=float(ratio), normal_independence_margin=float(margin),
                cone_sum_radians=a["normal_angle_bound_radians"]+b["normal_angle_bound_radians"],
                status="UNSCHEDULED_SCAN_PAIR_QUOTA" if admitted else "TWO_ROW_NORMAL_PREREQUISITE_REFUSAL",
                selected_order=None, scheduled_ordered_matches=0)
            work.record(row, 1024); result["scan_pairs"].append(row)
            if admitted: eligible.append(row["scan_pair_id"])
    selected, covered = [], set()
    while eligible and len(selected) < work.cap["max_selected_scan_pairs"]:
        work.charge("scan_selection_tests", len(eligible), "max_scan_selection_tests")
        def key(index):
            row = result["scan_pairs"][index]
            uncovered = sum(vertex not in covered for vertex in row["bank_indices"])
            return (-uncovered, -row["normal_condition_ratio"], -row["normal_independence_margin"],
                    row["cone_sum_radians"], index)
        chosen = min(eligible, key=key); eligible.remove(chosen)
        row = result["scan_pairs"][chosen]
        row.update(status="SELECTED_NOT_EVALUATED", selected_order=len(selected))
        covered.update(row["bank_indices"]); selected.append(chosen)
    result["selected_scan_pair_ids"] = selected
    return selected


def _cad_index(cad, work, result):
    count = len(cad); planned = count*(count-1)//2
    _require(planned <= work.cap["max_cad_pair_records"], "CAD_PAIR_PREFLIGHT_BUDGET")
    left, right = work.array(planned, "<i4"), work.array(planned, "<i4")
    angle, qualified = work.array(planned, "<f8"), work.array(planned, "?")
    # Small input normals/cones and argsort workspace are conservatively charged.
    with work.temporary(64*count):
        normals = np.asarray([row["normal"] for row in cad])
        alpha = np.asarray([row["normal_angle_bound_radians"] for row in cad])
        cursor = 0
        for first in range(count-1):
            length = count-first-1
            left[cursor:cursor+length] = first
            right[cursor:cursor+length] = np.arange(first+1, count)
            cursor += length
        for start in range(0, planned, work.cap["chunk_pair_rows"]):
            stop = min(planned, start+work.cap["chunk_pair_rows"])
            work.charge("cad_pair_tests", stop-start, "max_cad_pair_records")
            with work.temporary(384*(stop-start)):
                a, b = left[start:stop], right[start:stop]
                angles, ratios, margins = _pair_geometry(normals[a], normals[b], alpha[a], alpha[b])
                angle[start:stop] = angles
                qualified[start:stop] = (ratios >= _GATES["min_normal_condition_ratio"]) & (margins > 0.)
                del angles, ratios, margins, a, b
        del normals, alpha
    # Stable sort preserves canonical input-pair order for equal represented angles.
    work.charge("cad_sort_work", planned*max(0, (planned-1).bit_length()), "max_cad_sort_work")
    with work.temporary(32*planned):
        order = np.argsort(angle, kind="stable")
        sorted_angle = angle[order]
        work.memory(int(order.nbytes+sorted_angle.nbytes))
        work.stats["numeric_owned_bytes"] += int(order.nbytes+sorted_angle.nbytes)
    with work.temporary(16*planned):
        order_hash = hashlib.sha256(order.astype("<i8", copy=False).tobytes()).hexdigest()
    qualified_count = int(np.count_nonzero(qualified))
    index_record = dict(pair_count=planned, rank_qualified_pair_count=qualified_count,
        rank_refused_pair_count=planned-qualified_count,
        canonical_order="INPUT_INDICES_I_LESS_THAN_J_THEN_STABLE_ANGLE_SORT",
        angle_order_sha256=order_hash, complete=True)
    work.record(index_record, 1024); result["cad_index"] = index_record
    return dict(left=left, right=right, angle=angle, qualified=qualified, order=order,
        sorted_angle=sorted_angle, max_pair_cone=2.*max((row["normal_angle_bound_radians"] for row in cad), default=0.))


def _fair_positions(length, maximum, occurrence):
    count = min(length, maximum)
    if not count: return []
    boundaries = [index*length//count for index in range(count+1)]
    return [boundaries[index]+occurrence % (boundaries[index+1]-boundaries[index]) for index in range(count)]


def _window(scan_pair, cad, table, remaining, layouts, work, result):
    guard = _GATES["pair_angle_numeric_guard_radians"]
    radius = scan_pair["cone_sum_radians"]+2.*guard
    low = max(0., scan_pair["angle_radians"]-radius-table["max_pair_cone"])
    high = min(math.pi/2., scan_pair["angle_radians"]+radius+table["max_pair_cone"])
    start = int(np.searchsorted(table["sorted_angle"], low, side="left"))
    stop = int(np.searchsorted(table["sorted_angle"], high, side="right"))
    work.charge("window_pair_tests", stop-start, "max_window_pair_tests")
    with work.temporary(8*(stop-start)+64*len(cad)):
        compatible = np.empty(stop-start, dtype="<i4"); used = 0
        alpha = np.asarray([row["normal_angle_bound_radians"] for row in cad])
        for at in range(start, stop, work.cap["chunk_pair_rows"]):
            ids = table["order"][at:min(stop, at+work.cap["chunk_pair_rows"])]
            with work.temporary(80*len(ids)):
                widths = alpha[table["left"][ids]]+alpha[table["right"][ids]]+radius
                keep = table["qualified"][ids] & (np.abs(table["angle"][ids]-scan_pair["angle_radians"]) <= widths)
                chosen = ids[keep]
                compatible[used:used+len(chosen)] = chosen
                used += len(chosen)
                del widths, keep, chosen, ids
        # Complete ordered IDs identify the layout. Hash-only equality is forbidden.
        key_size = used*4
        _require(key_size <= work.cap["max_compatible_layout_bytes"], "SINGLE_COMPATIBLE_LAYOUT_BYTE_BUDGET")
        with work.temporary(key_size):
            key = compatible[:used].tobytes()
            if key not in layouts:
                _require(len(layouts) < work.cap["max_compatible_layouts"], "COMPATIBLE_LAYOUT_COUNT_BUDGET")
                _require(work.stats["compatible_layout_bytes"]+key_size <= work.cap["max_compatible_layout_bytes"],
                         "COMPATIBLE_LAYOUT_TOTAL_BYTE_BUDGET")
                work.memory(key_size)
                layouts[key] = dict(layout_index=len(layouts), occurrences=0, bin_occurrences={},
                    sha256=hashlib.sha256(key).hexdigest())
                work.stats["compatible_layout_bytes"] += key_size
                work.stats["compatible_layouts"] = len(layouts)
            else: work.stats["repeated_layout_hits"] += 1
            entry = layouts[key]; layout_occurrence = entry["occurrences"]; entry["occurrences"] += 1
            # A CAD pair is an indivisible both-order bundle. Odd residual room
            # never schedules one target order without the other.
            take = min(2, remaining//2)
            bin_count = min(used, take)
            # Keep independent counters when the last residual bundle changes
            # two bins to one: each fixed-bin coverage claim has its own clock.
            occurrence = entry["bin_occurrences"].get(bin_count, 0)
            entry["bin_occurrences"][bin_count] = occurrence+1
            positions = _fair_positions(used, take, occurrence)
            pair_ids = [int(compatible[position]) for position in positions]
            del key
        window = dict(scan_pair_id=scan_pair["scan_pair_id"], index_window_start=start,
            index_window_stop=stop, indexed_pair_count=stop-start, examined_pair_count=stop-start,
            outside_index_window_pair_count=len(table["order"])-(stop-start),
            outside_index_window_scope="INDEXED_ANGLE_INTERVAL_PRUNING_NOT_FINITE_GEOMETRIC_DISPROOF",
            compatible_unordered_count=used, incompatible_or_rank_refused_count=stop-start-used,
            omitted_compatible_unordered_count=used-len(pair_ids),
            layout_index=entry["layout_index"], compatible_pair_ids_sha256=entry["sha256"],
            compatible_pair_ids_encoding="EXACT_ORDERED_LITTLE_ENDIAN_INT32_CANONICAL_CAD_PAIR_INDICES",
            compatible_pair_ids_bytes=key_size, occurrence_index=occurrence,
            layout_occurrence_index=layout_occurrence, occurrence_scope="EXACT_LAYOUT_AND_ACTUAL_BIN_COUNT",
            scheduled_compatible_positions=positions, scheduled_canonical_pair_indices=pair_ids,
            both_target_orders_required=True, unordered_slot_quota=take, actual_bin_count=bin_count,
            coverage_occurrences=None if not positions else (used+len(positions)-1)//len(positions),
            coverage_scope="REPEATED_EXACT_LAYOUT_AT_THIS_FIXED_SLOT_COUNT_NOT_ALL_SCAN_PAIRS",
            complete=True)
        work.record(window, 2048); result["cad_windows"].append(window)
        work.stats["completed_windows"] += 1
        work.stats["compatible_occurrences"] += used
        work.stats["scheduled_unordered_occurrences"] += len(pair_ids)
        work.stats["omitted_compatible_occurrences"] += used-len(pair_ids)
        return pair_ids


def schedule_two_plane_pairs_v1(scan_equations, cad_equations, *, retained_triplet_hypothesis_count,
        retained_triplet_unsigned_count, max_joint_hypothesis_records=8192,
        max_joint_unsigned_assignments=1024, budgets=TwoPlaneScheduleBudgetsV1()):
    """Plan complete ordered pairs only; adapter authenticates all declarations.

    Retained triplet counts reserve capacity, including its rejected records.
    They are caller declarations, not authenticated or inferred from any score.
    Ordinary declared quota omissions are COMPLETE but nonexhaustive. A fatal
    failure discards ALL scheduled matches while retaining attempted counters.
    """
    result, work = _base(), None
    try:
        cap = _configuration(budgets); work = _Work(cap); work.memory()
        values = (retained_triplet_hypothesis_count, retained_triplet_unsigned_count,
                  max_joint_hypothesis_records, max_joint_unsigned_assignments)
        _require(all(type(value) is int and value >= 0 for value in values)
            and retained_triplet_hypothesis_count <= max_joint_hypothesis_records <= 8192
            and retained_triplet_unsigned_count <= max_joint_unsigned_assignments <= 1024,
            "INVALID_RETAINED_TRIPLET_OR_JOINT_BUDGET")
        remaining = min((max_joint_hypothesis_records-retained_triplet_hypothesis_count)//32,
            max_joint_unsigned_assignments-retained_triplet_unsigned_count)
        result.update(policy=dict(_GATES), budgets=cap, joint_budget=dict(
            retained_triplet_hypothesis_count=retained_triplet_hypothesis_count,
            retained_triplet_unsigned_count=retained_triplet_unsigned_count,
            max_joint_hypothesis_records=max_joint_hypothesis_records,
            max_joint_unsigned_assignments=max_joint_unsigned_assignments,
            ordered_match_capacity_before_both_order_rounding=remaining,
            both_order_bundle_capacity=remaining//2, unused_odd_ordered_capacity=remaining % 2,
            count_provenance="CALLER_DECLARED_RETAINED_TRIPLET_COUNTS_ADAPTER_MUST_VERIFY"))
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scan = _equations(scan_equations, cap["max_scan_planes"], "scan", work, result)
            cad = _equations(cad_equations, cap["max_cad_planes"], "cad", work, result)
            selected = _scan_pairs(scan, work, result)
            table, layouts = None, {}
            for pair_id in selected:
                row = result["scan_pairs"][pair_id]
                if remaining < 2:
                    row["status"] = "UNSCHEDULED_TRIPLET_FIRST_JOINT_QUOTA"
                    continue
                if table is None: table = _cad_index(cad, work, result)
                pair_ids = _window(row, cad, table, remaining, layouts, work, result)
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
        _require(work.stats["scheduled_ordered_matches"] == len(result["scheduled_matches"])
            and work.stats["reserved_two_plane_slots"] == 32*len(result["scheduled_matches"])
            and work.stats["completed_windows"] == len(result["cad_windows"])
            and len(result["scheduled_matches"]) <= 256
            and retained_triplet_hypothesis_count+work.stats["reserved_two_plane_slots"] <= max_joint_hypothesis_records
            and retained_triplet_unsigned_count+len(result["scheduled_matches"]) <= max_joint_unsigned_assignments
            and all(result["scan_pairs"][index]["scheduled_ordered_matches"] in (0, 2, 4) for index in selected),
            "INTERNAL_COMPLETE_SLOT_OR_WINDOW_LEDGER_MISMATCH")
        result.update(status="COMPLETE", selected_policy_evaluation_complete=True,
            usable_schedule=bool(result["scheduled_matches"]), reasons=[])
        result["joint_budget"].update(reserved_two_plane_hypothesis_records=work.stats["reserved_two_plane_slots"],
            reserved_two_plane_unsigned_assignments=work.stats["scheduled_ordered_matches"],
            remaining_ordered_match_capacity=remaining,
            joint_hypothesis_records_reserved=retained_triplet_hypothesis_count+work.stats["reserved_two_plane_slots"],
            joint_unsigned_assignments_reserved=retained_triplet_unsigned_count+work.stats["scheduled_ordered_matches"],
            pair_lane_starved_by_joint_quota=bool(selected and not result["cad_windows"] and remaining < 2))
        result["statistics"] = dict(work.stats)
        encoded = _size(result, cap["max_report_bytes"])
        work.memory(encoded)
        result["statistics"] = dict(work.stats)
        _size(result, cap["max_report_bytes"])
        return result
    except Exception as error:
        refusal = _base()
        refusal.update(reasons=[str(error) if type(error) is _Unavailable else "SCHEDULE_HELPER_EXCEPTION"],
            discarded_scheduled_matches=len(result["scheduled_matches"]),
            discarded_scan_pair_records=len(result["scan_pairs"]),
            discarded_completed_windows=len(result["cad_windows"]),
            evidence_scope="ATTEMPTED_COUNTERS_ONLY_NO_USABLE_MATCH_PREFIX",
            statistics={} if work is None else dict(work.stats))
        _size(refusal, 8192)
        return refusal
