"""Prospective constructed stress: 8,192 cases, one fixed seed, no tuning.

Continuous sizes/angles, local axis tilt/translation, four sampling densities,
duplicates and bounded noise. No physical data or fitted/calibrated error law.
Advance rule frozen before first execution: >=75% conditional measurement yield,
all emitted radius errors <=1% nominal radius and sweep errors <=0.25 degree,
and >=90% reduction in paired normalized radius RMSE versus nominal copying.
The low-density 3-axial-station stratum may lose full end stations at the finite
CAD window under tilt; this is an expected coverage challenge, not a missing part.
Failures/abstentions remain in the all-case yield denominator and output receipt.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_measurement.py"
SPEC = importlib.util.spec_from_file_location("cad_guided_stress_kernel", SOURCE)
kernel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kernel
SPEC.loader.exec_module(kernel)
CASES = 8192
SEED = 20260908
DENSITIES = ((17, 3, 7), (33, 5, 11), (65, 9, 17), (129, 17, 23))


def rotation(vector, angle):
    axis = vector / np.linalg.norm(vector)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def construct(rng, index):
    # Independent parameterization, not the runtime windows/plane/circle helper.
    radius = float(np.exp(rng.uniform(math.log(.7), math.log(40.))))
    nominal_sweep = float(rng.uniform(60., 150.) * (-1 if index % 2 else 1))
    actual_radius = radius * float(rng.uniform(.9, 1.1))
    actual_sweep = nominal_sweep + float(rng.uniform(-5., 5.))
    sign = math.copysign(1., actual_sweep)
    phi = math.radians(actual_sweep)
    na, nz, nf = DENSITIES[index % len(DENSITIES)]
    angles = np.linspace(0., phi, na)
    z = np.linspace(0., 6 * radius, nz)
    arc = np.column_stack((np.repeat(actual_radius * np.cos(angles), nz),
                           np.repeat(actual_radius * np.sin(angles), nz),
                           np.tile(z, na)))
    distances = np.repeat(np.linspace(.6, 2.4, nf) * radius, nz)
    axial = np.tile(z, nf)
    first = np.column_stack((np.full(len(axial), actual_radius), -sign * distances, axial))
    second = np.column_stack((actual_radius * math.cos(phi) - sign * distances * math.sin(phi),
                              actual_radius * math.sin(phi) + sign * distances * math.cos(phi), axial))
    points = np.vstack((arc, first, second))
    local_rotation = rotation(np.array([0.3, 0.7, 0.]), math.radians(float(rng.uniform(-2., 2.))))
    local_shift = radius * rng.uniform(-.03, .03, size=3)
    points = points @ local_rotation.T + local_shift
    noise_fraction = (0., .00025, .0005, .001)[(index // 4) % 4]
    points += rng.uniform(-noise_fraction, noise_fraction, size=points.shape) * radius
    if index % 7 == 0:
        points = np.vstack((points, points[::11]))
    q = rotation(np.array([.2, -.7, .4]), float(rng.uniform(-math.pi, math.pi)))
    translation = radius * rng.uniform(-100., 100., size=3)
    points = points @ q.T + translation
    # Shuffle rows to ensure neither source order nor an advantageous prefix is
    # being treated as a surface label. This RNG is construction-only.
    points = points[rng.permutation(len(points))]
    feature = kernel.CADGuidedFeature(
        feature_id="declared-local-feature", nominal_radius_mm=radius,
        nominal_sweep_deg=nominal_sweep, origin=tuple(translation), axis=tuple(q[:, 2]),
        start_radial=tuple(q[:, 0]), span_mm=(0., 6 * radius),
        flange_window_mm=(.5 * radius, 2.5 * radius), search_margin_mm=.6 * radius,
        max_fit_residual_mm=.01 * radius, plane_exclusion_mm=.015 * radius,
        max_tangency_error_mm=.02 * radius,
        nominal_source="independently constructed CAD declaration",
    )
    return points, feature, actual_radius, actual_sweep, noise_fraction


def run_frozen_stress(fit_model="free_circle"):
    rng = np.random.default_rng(SEED)
    reasons = Counter()
    strata = defaultdict(lambda: {"cases": 0, "measured": 0})
    failures = []
    measured = input_rows = 0
    radius_square = baseline_square = 0.
    max_radius_relative_error = max_angle_error = 0.
    for index in range(CASES):
        points, feature, true_radius, true_sweep, noise = construct(rng, index)
        feature = replace(feature, fit_model=fit_model)
        input_rows += len(points)
        label = f"density{index % 4}:noise{noise}"
        strata[label]["cases"] += 1
        result = kernel.inspect_cad_guided_scan(points, [feature], registration_status="DECLARED_ALIGNED")
        row = result["features"][0]
        if (result["observed_bend_count"] is not None or result["production_use_allowed"]
                or result["conformance"] != "INDETERMINATE"):
            failures.append({"case": index, "failure": "invalid authority/count claim"})
        if row["status"] != "MEASURED_CONDITIONAL":
            reasons.update(row["reasons"])
            continue
        measured += 1
        strata[label]["measured"] += 1
        values = row["measurements"]
        error = (values["measured_radius_mm"] - true_radius) / feature.nominal_radius_mm
        angle_error = abs(values["measured_sweep_deg"] - true_sweep)
        radius_square += error**2
        baseline_square += ((feature.nominal_radius_mm - true_radius) / feature.nominal_radius_mm)**2
        max_radius_relative_error = max(max_radius_relative_error, abs(error))
        max_angle_error = max(max_angle_error, angle_error)
        if abs(error) > .01 or angle_error > .25:
            failures.append({"case": index, "radius_relative_error": error,
                             "sweep_error_deg": angle_error, "stratum": label})
    report = {
        "fit_model": fit_model,
        "seed": SEED, "cases": CASES, "raw_rows_processed": input_rows,
        "measured": measured, "abstained": CASES - measured,
        "all_case_conditional_measurement_yield": measured / CASES,
        "abstention_reasons": dict(reasons), "strata": dict(strata),
        "challenger_normalized_radius_rmse": math.sqrt(radius_square / measured) if measured else None,
        "paired_nominal_copy_normalized_radius_rmse": math.sqrt(baseline_square / measured) if measured else None,
        "max_radius_relative_error": max_radius_relative_error,
        "max_sweep_error_deg": max_angle_error, "failures": failures,
        "scope": "constructed conditional measurement, NOT physical accuracy/count/F1",
        "physical_evaluation": False, "production_use_allowed": False,
    }
    print("CAD_GUIDED_STRESS " + json.dumps(report, sort_keys=True))
    return report


def check_frozen_advance_rule(report):
    assert not report["failures"], report
    assert report["measured"] / CASES >= .75, report
    assert report["challenger_normalized_radius_rmse"] <= .1 * report["paired_nominal_copy_normalized_radius_rmse"], report


def test_frozen_8192_case_position_only_stress():
    # Retained known failing free-circle hypothesis; do not xfail, tune or erase.
    check_frozen_advance_rule(run_frozen_stress("free_circle"))


def test_same_8192_case_measured_flange_tangent_candidate():
    # Development comparison on the SAME consumed constructed cases/seed/rules.
    # This is not a fresh confirmation or an independent physical cohort.
    check_frozen_advance_rule(run_frozen_stress("measured_flange_tangent"))
