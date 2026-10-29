"""Unqualified internal sheared-box bounds for the V1 rounded leaf program.

This is a source candidate, not an integrated index or a speed/accuracy result.
Only a <=256-row leaf is admitted. A caller must own/authenticate raw lineage,
charge aggregate work and memory, and clear its whole output after any failure.
The shear is a search enclosure, never pose, calibration or measured geometry.

For a permutation (i,j,k), z=RN(RN(x[k]-RN(a*x[i]))-RN(b*x[j])).
An outward interval encloses the exact-real residual
r=x[k]-a*x[i]-b*x[j]-z. No inverse/orthogonality assumption is needed.
The affine query in these coordinates is bounded in exact-real arithmetic,
then expanded to enclose rounding in the ORIGINAL sub/mul/add01/add2/+c
program. Leaf evaluation itself is not changed or provided by this module.

Assumptions: IEEE binary64 round-to-nearest, gradual underflow, separate
NumPy ufuncs, no fused/reassociated operations. Overflow/nonfinite bounds refuse.
Every numerical ufunc element, outward nextafter, reduction input, and admission
comparison is conservatively charged before execution. Initial charger-callability
validation, shape/type metadata and Python scalar bookkeeping,
tuple encoding and callback cost are not hardware-operation accounting. The
caller must reserve WORKING_BYTES in addition to its live owned arrays BEFORE
calling; this fixed numeric/small-Python envelope is not total-process RSS.
"""

from dataclasses import dataclass
import math

import numpy as np


MAX_LEAF_ROWS = 256
MAX_NODE_ROWS = 1_000_000
WORKING_BYTES = 131_072
MAX_LOCAL_STEPS = 65_536
BOUND_METHOD = "SHEAR_EXACT_RESIDUAL_PLUS_ORIGINAL_STAGE_ULP_ENCLOSURE_V1"


class BoundUnavailable(ValueError):
    pass


def _need(condition, reason):
    if not condition:
        raise BoundUnavailable(reason)


class _Ops:
    def __init__(self, charge):
        _need(callable(charge), "MISSING_PARENT_WORK_CHARGER")
        self.charge = charge
        self.steps = 0

    def take(self, count):
        _need(type(count) is int and count >= 0, "INVALID_BOUND_WORK_CHARGE")
        # Parent refusal occurs before any operation and remains its evidence.
        self.charge(count)
        self.steps += count
        _need(self.steps <= MAX_LOCAL_STEPS, "LOCAL_BOUND_WORK_LIMIT")

    def binary(self, operation, a, b):
        # All internal arrays are scalar or one-dimensional with <=256 entries.
        na, nb = int(np.size(a)), int(np.size(b))
        _need(na == nb or na == 1 or nb == 1, "BOUND_BROADCAST_MISMATCH")
        self.take(max(na, nb))
        return operation(a, b)

    def down(self, value):
        return self.binary(np.nextafter, value, -np.inf)

    def up(self, value):
        return self.binary(np.nextafter, value, np.inf)

    def add(self, a, b):
        return self.binary(np.add, a, b)

    def sub(self, a, b):
        return self.binary(np.subtract, a, b)

    def mul(self, a, b):
        return self.binary(np.multiply, a, b)

    def iadd(self, a, b):
        return self.down(self.add(a[0], b[0])), self.up(self.add(a[1], b[1]))

    def isub(self, a, b):
        return self.down(self.sub(a[0], b[1])), self.up(self.sub(a[1], b[0]))

    def scale(self, interval, coefficient):
        self.take(1)
        low, high = interval if coefficient >= 0. else interval[::-1]
        return self.down(self.mul(low, coefficient)), self.up(self.mul(high, coefficient))

    def imul(self, a, b):
        products = [self.mul(x, y) for x in a for y in b]
        low, high = products[0], products[0]
        for product in products[1:]:
            low = self.binary(np.minimum, low, product)
            high = self.binary(np.maximum, high, product)
        return self.down(low), self.up(high)

    def reduce(self, operation, values):
        self.take(int(np.size(values)))
        return float(operation(values))

    def checked(self, low, high):
        self.take(6 * max(int(np.size(low)), int(np.size(high))))
        _need(np.isfinite(low).all() and np.isfinite(high).all()
            and np.all(low <= high), "NONFINITE_OR_REVERSED_BOUND")


@dataclass(frozen=True)
class ShearedBox:
    # Internal trusted value: not an authentication boundary for foreign records.
    axes: tuple
    slopes: tuple
    world_low: tuple
    world_high: tuple
    z_interval: tuple
    residual_interval: tuple
    row_count: int


def _vector(value):
    _need(type(value) is tuple and len(value) == 3 and all(
        type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1e6
        for x in value), "INVALID_QUERY_VECTOR")
    return tuple(float(x) for x in value)


def build_leaf_box(points, axes, slopes, *, charge):
    """Construct a certified leaf summary; caller reserves WORKING_BYTES first.

The <=6144-byte local immutable snapshot prevents later mutation of this leaf
from changing its box. It does not authenticate the caller's larger raw buffer.
Returned numeric_steps is additional bound work, not V1's old work model.
"""
    ops = _Ops(charge)
    ops.take(64)
    _need(type(points) is np.ndarray and points.dtype == np.dtype("float64")
        and points.ndim == 2 and points.shape[1:] == (3,)
        and 0 < len(points) <= MAX_LEAF_ROWS, "INVALID_LEAF_POINTS")
    _need(type(axes) is tuple and len(axes) == 3
        and all(type(x) is int for x in axes) and sorted(axes) == [0, 1, 2],
        "INVALID_SHEAR_AXES")
    _need(type(slopes) is tuple and len(slopes) == 2 and all(
        type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1.
        for x in slopes), "INVALID_SHEAR_SLOPES")
    count = len(points)
    ops.take(3 * count)
    owned = np.frombuffer(points.tobytes(order="C"), dtype=np.float64).reshape(count, 3)
    ops.take(18 * count)
    _need(np.isfinite(owned).all() and np.all(np.abs(owned) <= 1e6),
        "UNSUPPORTED_LEAF_COORDINATE")
    a, b = tuple(float(x) for x in slopes)
    i, j, k = axes
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        u, v, w = owned[:, i], owned[:, j], owned[:, k]
        z = ops.sub(ops.sub(w, ops.mul(a, u)), ops.mul(b, v))
        residual = ops.isub(ops.isub(ops.isub((w, w), ops.scale((u, u), a)),
            ops.scale((v, v), b)), (z, z))
        ops.checked(*residual)
        low = tuple(ops.reduce(np.min, owned[:, axis]) for axis in range(3))
        high = tuple(ops.reduce(np.max, owned[:, axis]) for axis in range(3))
        zi = ops.reduce(np.min, z), ops.reduce(np.max, z)
        ri = ops.reduce(np.min, residual[0]), ops.reduce(np.max, residual[1])
        ops.checked(*zi)
        ops.checked(*ri)
    return ShearedBox(axes, (a, b), low, high, zi, ri, count), ops.steps


def merge_boxes(left, right, *, charge):
    """Exact min/max union; both children must share the represented shear."""
    ops = _Ops(charge)
    ops.take(64)
    _need(type(left) is ShearedBox and type(right) is ShearedBox
        and left.axes == right.axes and left.slopes == right.slopes,
        "INCOMPATIBLE_SHEARED_BOXES")
    _need(type(left.row_count) is int and type(right.row_count) is int
        and 0 < left.row_count <= MAX_NODE_ROWS and 0 < right.row_count <= MAX_NODE_ROWS
        and left.row_count <= MAX_NODE_ROWS - right.row_count,
        "MERGED_NODE_ROW_LIMIT")
    ops.take(11)
    box = ShearedBox(left.axes, left.slopes,
        tuple(min(a, b) for a, b in zip(left.world_low, right.world_low)),
        tuple(max(a, b) for a, b in zip(left.world_high, right.world_high)),
        (min(left.z_interval[0], right.z_interval[0]),
         max(left.z_interval[1], right.z_interval[1])),
        (min(left.residual_interval[0], right.residual_interval[0]),
         max(left.residual_interval[1], right.residual_interval[1])),
        left.row_count + right.row_count)
    return box, ops.steps


def _rounded_world_and_error(box, direction, origin, constant, ops):
    # Stage ranges use the ORIGINAL rounded operations, not outward exact-real
    # ranges. Monotonicity encloses all rounded stage outputs over the world box.
    delta, terms = [], []
    for axis in range(3):
        lo = ops.sub(box.world_low[axis], origin[axis])
        hi = ops.sub(box.world_high[axis], origin[axis])
        delta.append((lo, hi))
        ops.take(1)
        x, y = (lo, hi) if direction[axis] >= 0. else (hi, lo)
        terms.append((ops.mul(x, direction[axis]), ops.mul(y, direction[axis])))
    first = tuple(ops.add(terms[0][end], terms[1][end]) for end in (0, 1))
    second = tuple(ops.add(first[end], terms[2][end]) for end in (0, 1))
    final = tuple(ops.add(second[end], constant) for end in (0, 1))

    def gap(interval):
        ops.checked(*interval)
        ops.take(3)
        magnitude = max(abs(float(interval[0])), abs(float(interval[1])))
        neighbor = ops.up(magnitude)
        radius = ops.sub(neighbor, magnitude)
        ops.checked(0., radius)
        return radius

    errors = []
    for axis in range(3):
        ops.take(1)
        propagated = ops.up(ops.mul(abs(direction[axis]), gap(delta[axis])))
        errors.append(ops.up(ops.add(propagated, gap(terms[axis]))))
    error = errors[0]
    for term in (errors[1], errors[2], gap(first), gap(second), gap(final)):
        error = ops.up(ops.add(error, term))
    ops.checked(0., error)
    ops.take(2)
    return tuple(0. if x == 0. else float(x) for x in final), float(error)


def bound_sheared_box(box, direction, origin, constant, *, charge):
    """Enclose original staged row outputs; preserve world bounds for ordering.

An index should use intersection only to prune, retaining V1 tree/order/seeds
and original world-bound traversal priorities. This primitive is not an index,
does not select rows, and makes no speedup or physical uncertainty claim.
"""
    ops = _Ops(charge)
    ops.take(64)
    _need(type(box) is ShearedBox, "INVALID_SHEARED_BOX")
    direction, origin = _vector(direction), _vector(origin)
    _need(type(constant) in (int, float) and math.isfinite(constant)
        and abs(constant) <= 1e6, "INVALID_QUERY_CONSTANT")
    constant = float(constant)
    i, j, k = box.axes
    a, b = box.slopes
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        dk = direction[k]
        ci = ops.iadd((direction[i], direction[i]), ops.scale((dk, dk), a))
        cj = ops.iadd((direction[j], direction[j]), ops.scale((dk, dk), b))
        exact = ops.iadd(
            ops.imul(ci, (box.world_low[i], box.world_high[i])),
            ops.imul(cj, (box.world_low[j], box.world_high[j])))
        exact = ops.iadd(exact, ops.scale(box.z_interval, dk))
        exact = ops.iadd(exact, ops.scale(box.residual_interval, dk))
        for axis in range(3):
            exact = ops.isub(exact, ops.scale((origin[axis], origin[axis]), direction[axis]))
        exact = ops.iadd(exact, (constant, constant))
        ops.checked(*exact)
        world, error = _rounded_world_and_error(box, direction, origin, constant, ops)
        rounded = float(ops.down(ops.sub(exact[0], error))), float(ops.up(ops.add(exact[1], error)))
        ops.checked(*rounded)
        ops.take(2)
        intersection = max(world[0], rounded[0]), min(world[1], rounded[1])
        ops.checked(*intersection)
    return dict(interval=intersection, world_interval=world,
        exact_affine_interval=tuple(float(x) for x in exact),
        original_rounding_error_bound=error, numeric_steps=ops.steps,
        bound_method=BOUND_METHOD)
