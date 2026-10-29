"""Source-authored synthetic primitive controls; not executed by the author.

Forty prospective pytest items. No index, CAD, raw files, physical calibration,
or speedup claim. Exact Fraction oracles below do not call candidate arithmetic.
Largest leaf256 rows; no900k fixture or inherited suite is collected here.
"""
from fractions import Fraction
import importlib.util
from itertools import permutations
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_sheared_bounds_v1.py"
AXES = tuple(permutations((0, 1, 2)))


@pytest.fixture(scope="module")
def bounds():
    # One named pure module only; never import backend/__init__ or server code.
    name = "_independent_sheared_bounds_v1_fixture"
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


class Charges:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        assert type(value) is int and value >= 0
        self.calls.append(value)

    @property
    def total(self):
        return sum(self.calls)


def rational(value):
    return Fraction.from_float(float(value))


def scalar_staged(row, direction, origin, constant):
    # Each Python float operation is intentionally separate: not sum/dot/FMA.
    x = float(row[0]) - float(origin[0])
    y = float(row[1]) - float(origin[1])
    z = float(row[2]) - float(origin[2])
    tx = x * float(direction[0])
    ty = y * float(direction[1])
    tz = z * float(direction[2])
    first = tx + ty
    second = first + tz
    result = second + float(constant)
    return 0.0 if result == 0.0 else result


def exact_affine(row, direction, origin, constant):
    result = rational(constant)
    for axis in range(3):
        result += (rational(row[axis]) - rational(origin[axis])) * rational(direction[axis])
    return result


def sheared_oracle(row, axes, slopes):
    i, j, k = axes
    a, b = map(float, slopes)
    first_product = a * float(row[i])
    first_difference = float(row[k]) - first_product
    second_product = b * float(row[j])
    stored_z = first_difference - second_product
    residual = (rational(row[k]) - rational(a) * rational(row[i])
                - rational(b) * rational(row[j]) - rational(stored_z))
    return stored_z, residual


def assert_box_members(box, points):
    assert box.row_count == len(points)
    for axis in range(3):
        assert box.world_low[axis] == min(float(row[axis]) for row in points)
        assert box.world_high[axis] == max(float(row[axis]) for row in points)
    for row in points:
        z, residual = sheared_oracle(row, box.axes, box.slopes)
        assert rational(box.z_interval[0]) <= rational(z) <= rational(box.z_interval[1])
        assert rational(box.residual_interval[0]) <= residual <= rational(box.residual_interval[1])


def make_box(module, points, axes=(0, 1, 2), slopes=(0.625, -0.375)):
    charge = Charges()
    box, steps = module.build_leaf_box(points, axes, slopes, charge=charge)
    assert type(steps) is int and steps == charge.total
    assert 0 < steps <= module.MAX_LOCAL_STEPS
    assert_box_members(box, points)
    return box


def assert_query(module, box, points, direction, origin=(0., 0., 0.), constant=0.):
    charge = Charges()
    answer = module.bound_sheared_box(box, direction, origin, constant, charge=charge)
    assert answer["numeric_steps"] == charge.total
    assert 0 < charge.total <= module.MAX_LOCAL_STEPS
    assert answer["bound_method"] == "SHEAR_EXACT_RESIDUAL_PLUS_ORIGINAL_STAGE_ULP_ENCLOSURE_V1"
    assert set(answer) == {"interval", "world_interval", "exact_affine_interval",
                          "original_rounding_error_bound", "numeric_steps", "bound_method"}
    low, high = answer["interval"]
    alo, ahi = map(rational, answer["exact_affine_interval"])
    error = rational(answer["original_rounding_error_bound"])
    assert error >= 0 and math.isfinite(low) and math.isfinite(high) and low <= high
    assert answer["world_interval"][0] <= low <= high <= answer["world_interval"][1]
    corner_low = tuple(box.world_low[a] if direction[a] >= 0. else box.world_high[a]
                       for a in range(3))
    corner_high = tuple(box.world_high[a] if direction[a] >= 0. else box.world_low[a]
                        for a in range(3))
    assert answer["world_interval"] == (
        scalar_staged(corner_low, direction, origin, constant),
        scalar_staged(corner_high, direction, origin, constant))
    for endpoint in answer["world_interval"]:
        if endpoint == 0.:
            assert math.copysign(1., endpoint) == 1.
    for row in points:
        exact = exact_affine(row, direction, origin, constant)
        observed = scalar_staged(row, direction, origin, constant)
        assert alo <= exact <= ahi
        assert abs(rational(observed) - exact) <= error
        assert low <= observed <= high
    return answer


def plate(count=73, axes=(0, 1, 2), slopes=(0.625, -0.375)):
    result = np.empty((count, 3), dtype=np.float64)
    i, j, k = axes
    for row in range(count):
        u = float(row - count // 2) / 8.
        v = float((row * 17) % 31 - 15) / 16.
        first = float(slopes[0]) * u
        second = float(slopes[1]) * v
        w = (first + second) + float(row % 5 - 2) / 1024.
        result[row, i], result[row, j], result[row, k] = u, v, w
    return result


@pytest.mark.parametrize("axes", AXES)
def test_all_six_axis_permutations_enclose_every_original_member(bounds, axes):
    points = plate(73, axes)
    box = make_box(bounds, points, axes)
    normal = [0., 0., 0.]
    normal[axes[0]], normal[axes[1]], normal[axes[2]] = -.625, .375, 1.
    for direction, origin, constant in (
        (tuple(normal), (0., 0., 0.), 0.),
        (tuple(-v for v in normal), (1., -2., 3.), -.125),
        ((-.3, .7, -.9), (.1, -.2, .3), .11),
        ((0., -0., 0.), (-0., 0., -0.), -0.),
    ):
        assert_query(bounds, box, points, direction, origin, constant)


@pytest.mark.parametrize("slopes", ((.1, -.3), (1. / 3., 1. / 7.), (-.9, .11)))
def test_non_dyadic_represented_slopes_have_exact_rational_residuals(bounds, slopes):
    # Fractions describe the represented coefficients, not ideal1/3 or1/7.
    points = plate(97, slopes=slopes)
    points += np.asarray((999000., -998000., 512.))
    box = make_box(bounds, points, slopes=slopes)
    residuals = [sheared_oracle(row, box.axes, slopes)[1] for row in points]
    assert any(value != 0 for value in residuals)
    assert_query(bounds, box, points, (-slopes[0], -slopes[1], 1.),
                 (999000., -998000., 512.), .1)


TINY = float.fromhex("0x0.0000000000001p-1022")
NORMAL = float.fromhex("0x1p-1022")
ROUNDING = (
    (((TINY, 0., 0.), (-TINY, TINY, 0.), (2*TINY, -TINY, TINY)),
     (1., -.5, .25), (0., 0., 0.), 0.),
    (((NORMAL, 0., 0.), (-NORMAL, NORMAL, 0.), (0., -NORMAL, NORMAL)),
     (.5, -.5, .5), (0., 0., 0.), 0.),
    (((-0., 0., -0.), (0., -0., 0.)),
     (-0., -1., 0.), (-0., 0., -0.), -0.),
    (((1., 1., 1.), (1., 1., -1.), (1., -1., 1.)),
     (1., float.fromhex("0x1p-53"), -1.), (0., 0., 0.), 0.),
    (((float.fromhex("0x1.fffffffffffffp-1"), 1., 2.),
      (1., float.fromhex("0x1.0000000000001p+0"), 2.),
      (float.fromhex("0x1.0000000000001p+0"), -1., 0.)),
     (1., 1., -1.), (0., 0., 0.), float.fromhex("0x1p-53")),
    (((1e6, -1e6, 1.), (999999.9999999999, -999999.9999999999, -1.),
      (-1e6, 1e6, 0.)),
     (1e6, 1e6, -1e6), (-1e6, 1e6, 1e6), -1e6),
)


@pytest.mark.parametrize("coordinates,direction,origin,constant", ROUNDING)
def test_subnormal_zero_cancellation_binade_and_translated_rounding(
        bounds, coordinates, direction, origin, constant):
    points = np.asarray(coordinates, dtype=np.float64)
    box = make_box(bounds, points, slopes=(.1, -.3))
    assert_query(bounds, box, points, direction, origin, constant)


@pytest.mark.parametrize("axes", ((0, 1, 2), (2, 0, 1)))
def test_merged_summary_encloses_both_children_and_is_associative(bounds, axes):
    points = plate(256, axes, (.1, -.3))
    children = [make_box(bounds, p, axes, (.1, -.3))
                for p in (points[:63], points[63:151], points[151:])]
    def merge(a, b):
        charge = Charges()
        box, steps = bounds.merge_boxes(a, b, charge=charge)
        assert steps == charge.total == 75
        return box
    box = merge(merge(children[0], children[1]), children[2])
    other = merge(children[0], merge(children[1], children[2]))
    assert box == other
    assert_box_members(box, points)
    assert_query(bounds, box, points, (.7, -.9, .11), (3., -7., 11.), .1)


def test_exact_flat_normal_retains_nonzero_width_not_universal_acceleration(bounds):
    points = np.asarray([(float(i), float(i % 7), .5*i - .25*(i % 7))
                         for i in range(-32, 33)], dtype=np.float64)
    box = make_box(bounds, points, slopes=(.5, -.25))
    direction = (-.5, .25, 1.)
    assert all(scalar_staged(row, direction, (0., 0., 0.), 0.) == 0. for row in points)
    answer = assert_query(bounds, box, points, direction)
    # A conservative residual/ULP envelope can straddle every tied witness.
    assert answer["interval"][0] < 0. < answer["interval"][1]
    assert answer["original_rounding_error_bound"] > 0.


def test_leaf_snapshot_survives_caller_mutation_without_lineage_claim(bounds):
    points = plate()
    original = points.copy()
    box = make_box(bounds, points)
    before = assert_query(bounds, box, original, (.1, -.3, .7))
    points[:] = 999999.
    after = assert_query(bounds, box, original, (.1, -.3, .7))
    assert before == after
    assert not hasattr(box, "point_buffer_sha256")  # No invented raw-owner authentication.


@pytest.mark.parametrize("kind", ("list", "float32", "empty", "oversize", "nan", "range"))
def test_malformed_leaf_refuses_without_returning_summary(bounds, kind):
    points = plate(4)
    if kind == "list": points = points.tolist()
    elif kind == "float32": points = points.astype(np.float32)
    elif kind == "empty": points = np.empty((0, 3), dtype=np.float64)
    elif kind == "oversize": points = np.zeros((257, 3), dtype=np.float64)
    elif kind == "nan": points[0, 0] = float("nan")
    else: points[0, 0] = 1_000_000.0000000001
    charge = Charges()
    reason = "UNSUPPORTED_LEAF_COORDINATE" if kind in ("nan", "range") else "INVALID_LEAF_POINTS"
    with pytest.raises(bounds.BoundUnavailable, match="^" + reason + "$"):
        bounds.build_leaf_box(points, (0, 1, 2), (.5, -.25), charge=charge)
    assert charge.calls[0] == 64
    assert charge.total == (64 + 21*4 if kind in ("nan", "range") else 64)


@pytest.mark.parametrize("axes", ((0, 0, 2), (False, 1, 2)))
def test_axes_require_exact_permutation(bounds, axes):
    with pytest.raises(bounds.BoundUnavailable, match="^INVALID_SHEAR_AXES$"):
        bounds.build_leaf_box(plate(4), axes, (.5, -.25), charge=Charges())


@pytest.mark.parametrize("slopes", ((True, .5), (1.0000000000000002, 0.), (float("inf"), 0.)))
def test_slopes_admit_only_finite_represented_domain(bounds, slopes):
    with pytest.raises(bounds.BoundUnavailable, match="^INVALID_SHEAR_SLOPES$"):
        bounds.build_leaf_box(plate(4), (0, 1, 2), slopes, charge=Charges())


@pytest.mark.parametrize("which", ("direction", "origin", "constant"))
def test_malformed_query_has_positive_box_premise(bounds, which):
    points = plate(4)
    box = make_box(bounds, points)
    assert_query(bounds, box, points, (1., 0., 0.))
    direction, origin, constant = (1., 0., 0.), (0., 0., 0.), 0.
    if which == "direction": direction = (True, 0., 0.)
    elif which == "origin": origin = (float("inf"), 0., 0.)
    else: constant = float("nan")
    charge = Charges()
    reason = "INVALID_QUERY_CONSTANT" if which == "constant" else "INVALID_QUERY_VECTOR"
    with pytest.raises(bounds.BoundUnavailable, match="^" + reason + "$"):
        bounds.bound_sheared_box(box, direction, origin, constant, charge=charge)
    assert charge.calls == [64]


def test_mandatory_parent_charger_and_internal_box_type(bounds):
    points = plate(4)
    box = make_box(bounds, points)
    calls = (
        lambda: bounds.build_leaf_box(points, (0, 1, 2), (.5, -.25), charge=None),
        lambda: bounds.merge_boxes(box, box, charge=None),
        lambda: bounds.bound_sheared_box(box, (1., 0., 0.), (0., 0., 0.), 0., charge=None),
    )
    for call in calls:
        with pytest.raises(bounds.BoundUnavailable, match="^MISSING_PARENT_WORK_CHARGER$"):
            call()
    with pytest.raises(bounds.BoundUnavailable, match="^INVALID_SHEARED_BOX$"):
        bounds.bound_sheared_box({}, (1., 0., 0.), (0., 0., 0.), 0., charge=Charges())


@pytest.mark.parametrize("phase", ("build", "merge", "query"))
def test_parent_refusal_propagates_before_first_charged_operation(bounds, phase):
    points = plate(4)
    box = make_box(bounds, points)
    class ParentRefusal(RuntimeError):
        pass
    error = ParentRefusal("invented parent aggregate budget")
    attempts = []
    def refuse(count):
        attempts.append(count)
        raise error
    with pytest.raises(ParentRefusal) as captured:
        if phase == "build":
            bounds.build_leaf_box(points, (0, 1, 2), (.5, -.25), charge=refuse)
        elif phase == "merge":
            bounds.merge_boxes(box, box, charge=refuse)
        else:
            bounds.bound_sheared_box(box, (1., 0., 0.), (0., 0., 0.), 0., charge=refuse)
    assert captured.value is error
    assert attempts == [64]


def test_source_derived_work_counts_and_fixed_configuration(bounds):
    assert bounds.MAX_LEAF_ROWS == 256
    assert bounds.MAX_NODE_ROWS == 1_000_000
    assert bounds.WORKING_BYTES == 131072
    assert bounds.MAX_LOCAL_STEPS == 65536
    for count in (1, 17, 256):
        points = plate(count)
        charge = Charges()
        box, steps = bounds.build_leaf_box(points, (0, 1, 2), (.625, -.375), charge=charge)
        # Source-derived:64 admission +21N snapshot/validation +4N z
        # +20N+2 residual +6N residual checks +10N reductions +12 scalar checks.
        assert steps == charge.total == 78 + 61*count
        answer = assert_query(bounds, box, points, (.1, -.3, .7), (1., -2., 3.), .1)
        #165 affine/admission +207 original-stage/error/zero +18 intersection.
        assert answer["numeric_steps"] == 390
    # Legitimate immutable summaries of a repeated-point population, not forged
    # foreign row_count fields and not an index/raw-lineage authentication test.
    one = make_box(bounds, np.asarray([[1., 2., 3.]], dtype=np.float64))
    powers = [one]
    for _ in range(19):
        charge = Charges()
        doubled, steps = bounds.merge_boxes(powers[-1], powers[-1], charge=charge)
        assert steps == charge.total == 75
        powers.append(doubled)
    exact = None
    for bit, box in enumerate(powers):
        if 1_000_000 & (1 << bit):
            if exact is None:
                exact = box
            else:
                charge = Charges()
                exact, steps = bounds.merge_boxes(exact, box, charge=charge)
                assert steps == charge.total == 75
    assert exact.row_count == 1_000_000
    charge = Charges()
    with pytest.raises(bounds.BoundUnavailable, match="^MERGED_NODE_ROW_LIMIT$"):
        bounds.merge_boxes(exact, one, charge=charge)
    assert charge.calls == [64]


@pytest.mark.parametrize("difference", ("axes", "slopes"))
def test_merge_requires_same_represented_shear(bounds, difference):
    points = plate(4)
    left = make_box(bounds, points)
    right = make_box(bounds, points, axes=(1, 0, 2) if difference == "axes" else (0, 1, 2),
                     slopes=(.625, -.375) if difference == "axes" else (.625, -.25))
    charge = Charges()
    with pytest.raises(bounds.BoundUnavailable, match="^INCOMPATIBLE_SHEARED_BOXES$"):
        bounds.merge_boxes(left, right, charge=charge)
    assert charge.calls == [64]
