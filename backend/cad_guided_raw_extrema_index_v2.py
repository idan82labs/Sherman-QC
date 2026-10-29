"""Unqualified standalone sheared-pruning index for the unchanged V1 program.

V1 owns the raw snapshot, support identities, hashes, world-axis tree, seeds
and leaf evaluator. A global raw-seed shear only tightens pruning bounds;
old world bounds still order traversal. All added work consumes the SAME
coordinate budget, separately itemized, with no cap increase. Zero shears
disable optional metadata without refunding attempted chooser work.

This is not context/workflow integration, a pose fit or a speed/accuracy claim.
Whole-part supports may be folded/multiplane: one global shear can be useless.
Flat normal queries can also require every leaf despite valid tight geometry.
The caller's aggregate work/memory and whole-failure obligations still apply.
"""

from itertools import combinations
import json
import math

import numpy as np

from . import cad_guided_raw_extrema_index_v1 as _v1
from . import cad_guided_sheared_bounds_v1 as _shear


RawExtremaBudgetsV1 = _v1.RawExtremaBudgetsV1
ARITHMETIC = _v1.ARITHMETIC
LEAF_ROWS = _v1.LEAF_ROWS
BOUND_METHOD = _shear.BOUND_METHOD
METADATA_BYTES = 4096
_PHASES = ("chooser_steps", "build_steps", "merge_steps",
           "query_bound_steps", "packing_steps")
_require = _v1._require


class _IndexV2(_v1._Index):
    """Trusted internal owner, not a foreign-box or mutated-index validator."""

    def __init__(self, points, support_rows, budgets):
        try:
            # No override of _build, _memory, _charge or _evaluate: V1 snapshot,
            # validation, permutations, hashes and numerical leaves are intact.
            super().__init__(points, support_rows, budgets)
            self.valid = False
            self.stats.update({name: 0 for name in _PHASES})
            self.stats.update(sheared_bound_steps=0, chooser_triples_attempted=0,
                              sheared_leaves_completed=0, sheared_merges_completed=0,
                              sheared_single_child_copies=0)
            self._memory(METADATA_BYTES)
            self.stats["owned_bytes"] += METADATA_BYTES
            self._choose_shear()
            if self.shear_enabled:
                self._build_shear()
            else:
                self.shear_bounds = None
                self.shear_bounds_sha256 = None
            self.valid = True
        except BaseException:
            self.valid = False
            raise

    def _extra(self, phase, amount):
        _require(phase in _PHASES and type(amount) is int and amount >= 0,
                 "INVALID_SHEARED_WORK_CHARGE")
        # Attempted phase and aggregate charges persist even when the inherited
        # cumulative coordinate cap rejects this operation before it executes.
        self.stats[phase] += amount
        self.stats["sheared_bound_steps"] += amount
        self._charge("coordinate_visits", amount)

    def _choose_shear(self):
        ids = self.seed_rows  # Sorted original raw IDs, at most seven.
        self._extra("chooser_steps", 16)
        _require(0 < len(ids) <= 7, "INVALID_V1_SEED_DENOMINATOR")
        self._memory(32 * len(ids))
        self._charge("index_visits", len(ids))
        self._charge("row_visits", len(ids))
        self._extra("chooser_steps", 4 * len(ids))
        seeds = self.points[np.asarray(ids, dtype=np.int64)]
        best, best_ids, strength, unsupported = None, None, 0., False
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for ia, ib, ic in combinations(range(len(ids)), 3):
                self.stats["chooser_triples_attempted"] += 1
                # 6 differences, 6 products, 3 cross differences, finite/all,
                # abs/max and strength comparison; fixed conservative reserve.
                self._extra("chooser_steps", 32)
                try:
                    u = np.subtract(seeds[ib], seeds[ia])
                    v = np.subtract(seeds[ic], seeds[ia])
                    cross = np.asarray((
                        np.subtract(np.multiply(u[1], v[2]), np.multiply(u[2], v[1])),
                        np.subtract(np.multiply(u[2], v[0]), np.multiply(u[0], v[2])),
                        np.subtract(np.multiply(u[0], v[1]), np.multiply(u[1], v[0]))),
                        dtype=np.float64)
                    if not np.isfinite(cross).all():
                        unsupported = True
                        continue
                    candidate = float(np.max(np.abs(cross)))
                    if candidate > strength:  # First lexicographic triple on ties.
                        best, best_ids, strength = cross, (ids[ia], ids[ib], ids[ic]), candidate
                except FloatingPointError:
                    unsupported = True
            axes, slopes = (0, 1, 2), (0., 0.)
            status = ("ZERO_SHEAR_UNSUPPORTED_SEEDS" if unsupported
                      else "ZERO_SHEAR_DEGENERATE_SEEDS")
            if best is not None:
                self._extra("chooser_steps", 32)
                try:
                    k = int(np.argmax(np.abs(best)))  # Lowest axis on ties.
                    i, j = tuple(axis for axis in range(3) if axis != k)
                    a = float(np.divide(np.negative(best[i]), best[k]))
                    b = float(np.divide(np.negative(best[j]), best[k]))
                    if math.isfinite(a) and math.isfinite(b) and abs(a) <= 1. and abs(b) <= 1.:
                        axes, slopes, status = (i, j, k), (a, b), "FINITE_NONZERO_SEED_CROSS"
                    else:
                        status = "ZERO_SHEAR_UNSUPPORTED_RATIO"
                except FloatingPointError:
                    status = "ZERO_SHEAR_UNSUPPORTED_RATIO"
        self.shear_axes, self.shear_slopes = axes, slopes
        self._extra("chooser_steps", 2)
        self.shear_enabled = not (slopes[0] == 0. and slopes[1] == 0.)
        self.shear_chooser = dict(status=status, axes=axes, slopes=slopes,
            seed_rows=ids, selected_seed_triple=best_ids if status == "FINITE_NONZERO_SEED_CROSS" else None,
            triple_order="LEXICOGRAPHIC_ORIGINAL_RAW_IDS_FIRST_MAX_STRENGTH_TIE",
            scope="RAW_SEED_HEURISTIC_NOT_PLANE_OR_POSE_CERTIFICATION")
        if not self.shear_enabled:
            self.shear_chooser.update(status="DISABLED_ZERO_SHEAR", chooser_status=status)

    def _box(self, node):
        # 10 represented numeric values, start/end and subtraction, with reserve.
        self._extra("packing_steps", 16)
        packed = self.shear_bounds[node]
        return _shear.ShearedBox(self.shear_axes, self.shear_slopes,
            tuple(float(x) for x in self.low[node]),
            tuple(float(x) for x in self.high[node]),
            (float(packed[0]), float(packed[1])),
            (float(packed[2]), float(packed[3])),
            int(self.end[node]) - int(self.start[node]))

    def _pack(self, node, box):
        self._extra("packing_steps", 4)
        self.shear_bounds[node] = (*box.z_interval, *box.residual_interval)

    def _build_shear(self):
        size = 32 * self.node_count
        self._memory(size)
        self._extra("packing_steps", 4 * self.node_count)
        self.shear_bounds = np.zeros((self.node_count, 4), dtype=np.float64)
        self.stats["owned_bytes"] += size
        leaf_count = (self.count + LEAF_ROWS - 1) // LEAF_ROWS
        for leaf in range(leaf_count):
            node = self.base + leaf
            self._extra("packing_steps", 4)
            start, end = int(self.start[node]), int(self.end[node])
            count = end - start
            self._memory(_shear.WORKING_BYTES + 24 * count)
            self._charge("index_visits", count)
            self._charge("row_visits", count)
            self._extra("packing_steps", 3 * count)
            points = self.points[self.rows[start:end]]
            box, _ = _shear.build_leaf_box(points, self.shear_axes, self.shear_slopes,
                charge=lambda amount: self._extra("build_steps", amount))
            self._pack(node, box)
            self.stats["sheared_leaves_completed"] += 1
            del points, box
        for node in range(self.base - 1, 0, -1):
            self._charge("node_visits", 1)
            self._extra("packing_steps", 8)
            left, right = 2 * node, 2 * node + 1
            has_left = self.end[left] > self.start[left]
            has_right = self.end[right] > self.start[right]
            self._memory(_shear.WORKING_BYTES)
            if has_left and has_right:
                box, _ = _shear.merge_boxes(self._box(left), self._box(right),
                    charge=lambda amount: self._extra("merge_steps", amount))
                self._pack(node, box)
                self.stats["sheared_merges_completed"] += 1
                del box
            elif has_left or has_right:
                self._extra("packing_steps", 4)
                self.shear_bounds[node] = self.shear_bounds[left if has_left else right]
                self.stats["sheared_single_child_copies"] += 1
        # Keep the old mutable packed payload plus the new bytes copy live in
        # the preflight estimate; the retained final array is bytes-backed.
        self._memory(size)
        self._extra("packing_steps", 4 * self.node_count)
        self.shear_bounds = np.frombuffer(self.shear_bounds.tobytes(order="C"),
            dtype=np.float64).reshape(self.node_count, 4)
        self.shear_bounds_sha256 = self._digest(self.shear_bounds)
        self._memory()

    def _node_bounds(self, node, direction, origin, constant):
        if not self.shear_enabled:
            # Zero shear cannot tighten the original world-corner enclosure.
            # Retain chooser charges, but allocate/build/query no local boxes.
            lower, upper = super()._bounds(node, direction, origin, constant)
            return lower, upper, lower, upper
        self._charge("node_visits", 1)
        self._memory(_shear.WORKING_BYTES)
        # Query validation has already produced finite represented vectors.
        self._extra("packing_steps", 6)
        d, o = tuple(float(x) for x in direction), tuple(float(x) for x in origin)
        answer = _shear.bound_sheared_box(self._box(node), d, o, constant,
            charge=lambda amount: self._extra("query_bound_steps", amount))
        return (*answer["interval"], *answer["world_interval"])

    def query(self, direction, origin, constant):
        try:
            _require(self.valid, "INDEX_INVALID_OR_PREVIOUSLY_FAILED")
            direction, origin = _v1._vector(direction), _v1._vector(origin)
            _require(type(constant) in (int, float) and math.isfinite(constant) and abs(constant) <= 1e6,
                "INVALID_PROJECTION_CONSTANT")
            constant = float(constant)
            self._charge("queries", 1)
            before = dict(self.stats)
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                seeds = np.asarray(self.seed_rows, dtype=np.int64)
                low, high, low_id, high_id = self._evaluate(seeds, direction, origin, constant)
                lower, upper, _, _ = self._node_bounds(1, direction, origin, constant)
                stack = [(1, lower, upper)]
                while stack:
                    node, lower, upper = stack.pop()
                    first = int(self.minimum_raw[node])
                    no_minimum = lower > low or lower == low and first >= low_id
                    no_maximum = upper < high or upper == high and first >= high_id
                    if no_minimum and no_maximum:
                        self.stats["nodes_pruned"] += 1
                        continue
                    if node >= self.base:
                        ids = self.rows[int(self.start[node]):int(self.end[node])]
                        a, b, ai, bi = self._evaluate(ids, direction, origin, constant)
                        self.stats["leaves_evaluated"] += 1
                        if a < low or a == low and ai < low_id:
                            low, low_id = a, ai
                        if b > high or b == high and bi < high_id:
                            high, high_id = b, bi
                    else:
                        children = []
                        for child in (2*node, 2*node+1):
                            if self.end[child] > self.start[child]:
                                a, b, world_a, world_b = self._node_bounds(child, direction, origin, constant)
                                children.append((max(low-world_a, world_b-high), child, a, b))
                        children.sort(key=lambda item: (item[0], -item[1]))
                        stack.extend((child, a, b) for _, child, a, b in children)
                        _require(len(stack) <= 32, "INTERNAL_TRAVERSAL_STACK_BUDGET")
                return dict(interval=[low, high], minimum_raw_row=low_id, maximum_raw_row=high_id,
                    support_row_count=self.count,
                    work={name: self.stats[name]-before[name] for name in
                        ("row_visits", "index_visits", "coordinate_visits", "node_visits",
                         "query_rows_evaluated", "nodes_pruned", "leaves_evaluated",
                         "sheared_bound_steps", *_PHASES)})
        except BaseException:
            self.valid = False
            raise


def project_raw_extrema_v2(points, projections, *, support_rows=None,
        budgets=RawExtremaBudgetsV1()):
    """Standalone all-or-nothing batch; same V1 budget type and fixed maxima."""
    index, completed = None, []
    declarations = dict(schema_version="raw_extrema_index_v2", arithmetic=ARITHMETIC,
        bound_method=BOUND_METHOD,
        bound_scope="ENCLOSURE_OF_ORIGINAL_STAGED_ROUNDED_PROGRAM",
        geometric_scope="SUPPLIED_RAW_ROW_LINEAR_EXTREMA_ONLY",
        accepted_registration=False, observed_bend_count=None, production_use_allowed=False)
    try:
        _require(type(budgets) is RawExtremaBudgetsV1, "INVALID_BUDGET_TYPE")
        _require(type(projections) is tuple and 0 < len(projections) <= budgets.max_queries,
            "INVALID_OR_OVER_BUDGET_PROJECTIONS")
        _require(all(type(item) is tuple and len(item) == 3 for item in projections),
            "INVALID_PROJECTION_RECORD")
        index = _IndexV2.__new__(_IndexV2)
        index.__init__(points, support_rows, budgets)
        for direction, origin, constant in projections:
            record = index.query(direction, origin, constant)
            size = sum(len(part.encode()) for part in json.JSONEncoder(
                sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(record))
            index._charge("report_bytes", size)
            _require(index.stats["report_bytes"]+8192 <= index.cap["max_report_bytes"],
                "BUDGET_REPORT_ENVELOPE_BYTES")
            index._memory()
            completed.append(record)
        result = dict(declarations, status="COMPLETE", reasons=[], projections=completed,
            statistics=dict(index.stats), point_buffer_sha256=index.point_sha256,
            support_rows_sha256=index.support_sha256, support_scope=index.scope,
            support_row_count=index.count, sort_axis=index.sort_axis,
            leaf_rows=LEAF_ROWS, node_slots=index.node_count, shear_chooser=index.shear_chooser,
            packed_shear_bytes=0 if index.shear_bounds is None else int(index.shear_bounds.nbytes),
            packed_shear_sha256=index.shear_bounds_sha256,
            budget_type="RawExtremaBudgetsV1_UNCHANGED_MAXIMA",
            traversal_order="UNCHANGED_V1_WORLD_BOUND_PRIORITY_AND_RAW_ID_TIES",
            coordinate_work_model="V1_COORDINATE_CHARGES_PLUS_ALL_SHEARED_BOUND_STEPS",
            added_work_partition=list(_PHASES),
            counter_scope="ATTEMPTED_ABSTRACT_WORK_NOT_HARDWARE_INSTRUCTIONS",
            memory_scope="OWNED_PAYLOAD_PLUS_FIXED_SCRATCH_NOT_TOTAL_PROCESS_RSS")
        encoded = 0
        for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(result):
            encoded += len(part.encode())
            _require(encoded <= index.cap["max_report_bytes"], "BUDGET_FINAL_REPORT_BYTES")
        return result
    except Exception as error:
        if index is not None:
            index.valid = False
        return dict(declarations, status="UNAVAILABLE",
            reasons=[str(error) if type(error) in (_v1._Unavailable, _shear.BoundUnavailable)
                     else "EXTREMA_V2_EXCEPTION"],
            discarded_projection_count=len(completed), projections=[],
            statistics={} if index is None else dict(getattr(index, "stats", {})))
