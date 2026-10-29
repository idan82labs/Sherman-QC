"""Exercise real ownership + support-instance glue with stipulated scan fits.

Only _measure is replaced. Feature validation, inspection admission, shared-role
ownership, real finite support builder, status and serialization are actual
source. This is not an additional circle/plane fit or native CAD experiment.
"""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _modules():
    names = ("cad_guided_instances", "cad_guided_measurement_instance_glue_test")
    previous = {name: sys.modules.get(name) for name in names}
    modules = []
    try:
        for name, filename in zip(names, ("cad_guided_instances.py", "cad_guided_measurement.py")):
            path = ROOT / "backend" / filename
            module = ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = ""
            sys.modules[name] = module
            exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
            modules.append(module)
        yield modules[1], modules[0]
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _feature(kernel, name, y=0.):
    return kernel.CADGuidedFeature(
        feature_id=name, nominal_radius_mm=5., nominal_sweep_deg=90.,
        origin=(0., y, 0.), axis=(0., 0., 1.), start_radial=(1., 0., 0.),
        span_mm=(0., 8.), flange_window_mm=(3., 10.), search_margin_mm=2.,
        max_fit_residual_mm=.01, plane_exclusion_mm=.1, max_tangency_error_mm=.05,
        nominal_source="INVENTED_FEATURE_SCOPE")


def _fixture(stations=(0., 2., 4., 6., 8.), shared_plane=False):
    angles = np.deg2rad(np.linspace(10., 80., 15))
    pieces, fits, offset = [], {}, 0

    def append(values):
        nonlocal offset
        block = np.asarray(values, dtype=float)
        pieces.append(block)
        indices = list(range(offset, offset + len(block)))
        offset += len(block)
        return indices

    shared = None
    for name, y in (("one", 0.), ("two", 30.)):
        arc = append([[6 * np.cos(theta), y + 6 * np.sin(theta), z]
                      for z in stations for theta in angles])
        if name == "two" and shared_plane:
            first = shared
        else:
            first = append([[6., y - d, z] for z in stations for d in (3., 4., 5., 6., 7.)])
            if name == "one":
                shared = first
        second = append([[-d, y + 6., z] for z in stations for d in (3., 4., 5., 6., 7.)])
        fits[name] = {
            "measured_axis_point": [0., y, 0.], "measured_axis_direction": [0., 0., 1.],
            "measured_radius_mm": 6., "circle_max_residual_mm": 1e-10,
            "radius_deviation_mm": 1.,
            "support_row_indices": sorted(set(arc + first + second)),
            "_instance_support": {"arc_row_indices": arc, "flange_row_indices": [first, second]},
        }
    return np.vstack(pieces), fits


def _install_fit(kernel, fits):
    calls = []

    def stipulated_fit(points, feature):
        calls.append(feature.feature_id)
        return copy.deepcopy(fits[feature.feature_id])

    kernel._measure = stipulated_fit
    return calls


def _inspect(kernel, instances, points, features):
    return kernel.inspect_cad_guided_scan(
        points, features, registration_status="DECLARED_ALIGNED",
        instance_policy=instances.SupportPolicy(2.1, 6., 30., 3))


def test_successful_fits_do_not_guarantee_an_observed_finite_support_instance():
    with _modules() as (kernel, instances):
        points, fits = _fixture(stations=(0., 2., 6., 8.))
        calls = _install_fit(kernel, fits)
        result = _inspect(kernel, instances, points, [_feature(kernel, "one")])
        assert calls == ["one"] and result["measured_conditional_feature_count"] == 1
        assert result["features"][0]["measurements"]["radius_deviation_mm"] == 1.
        assert result["observed_support_instance_count"] == 0
        assert result["finite_support_instances"] == []
        assert result["instance_extraction_status"] == "UNAVAILABLE"
        assert result["features"][0]["finite_support"]["reasons"] == ["DISCONNECTED_AXIAL_SUPPORT"]
        assert result["observed_bend_count"] is None


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_curved_support_invalidates_all_competing_features(reverse):
    with _modules() as (kernel, instances):
        points, fits = _fixture()
        fits["two"] = copy.deepcopy(fits["one"])
        _install_fit(kernel, fits)
        features = [_feature(kernel, "one"), _feature(kernel, "two", 30.)]
        result = _inspect(kernel, instances, points, features[::-1] if reverse else features)
        assert result["measured_conditional_feature_count"] == 0
        assert result["observed_support_instance_count"] == 0
        assert all(row["reasons"] == ["AMBIGUOUS_SHARED_FEATURE_SUPPORT"] for row in result["features"])
        assert all(row["measurements"] is None for row in result["features"])


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_planar_context_can_support_two_distinct_curved_roots(reverse):
    with _modules() as (kernel, instances):
        points, fits = _fixture(shared_plane=True)
        _install_fit(kernel, fits)
        features = [_feature(kernel, "one"), _feature(kernel, "two", 30.)]
        result = _inspect(kernel, instances, points, features[::-1] if reverse else features)
        assert result["measured_conditional_feature_count"] == 2
        assert result["observed_support_instance_count"] == 2
        assert result["instance_extraction_status"] == "COMPLETE"
        by_id = {row["feature_id"]: row for row in result["finite_support_instances"]}
        assert set(by_id) == {"one", "two"}
        np.testing.assert_allclose(by_id["one"]["line_start"], [0., 0., 0.])
        np.testing.assert_allclose(by_id["two"]["line_start"], [0., 30., 0.])
        assert all(row["manufacturing_event_identity"] == "UNKNOWN" for row in by_id.values())
        assert result["observed_bend_count"] is None


@pytest.mark.parametrize("reverse", [False, True])
def test_curved_support_cannot_be_reused_as_another_features_planar_context(reverse):
    with _modules() as (kernel, instances):
        points, fits = _fixture()
        borrowed = fits["one"]["_instance_support"]["arc_row_indices"][:3]
        fits["two"]["_instance_support"]["flange_row_indices"][0].extend(borrowed)
        fits["two"]["support_row_indices"].extend(borrowed)
        _install_fit(kernel, fits)
        features = [_feature(kernel, "one"), _feature(kernel, "two", 30.)]
        result = _inspect(kernel, instances, points, features[::-1] if reverse else features)
        assert result["measured_conditional_feature_count"] == result["observed_support_instance_count"] == 0
        assert all(row["reasons"] == ["AMBIGUOUS_SHARED_FEATURE_SUPPORT"] for row in result["features"])


def test_private_support_indices_do_not_escape_into_serialized_report():
    with _modules() as (kernel, instances):
        points, fits = _fixture()
        _install_fit(kernel, fits)
        result = _inspect(kernel, instances, points, [_feature(kernel, "one")])
        serialized = json.dumps(result, allow_nan=False)
        assert result["observed_support_instance_count"] == 1
        for key in ("support_row_indices", "_instance_support", "arc_row_indices", "flange_row_indices"):
            assert json.dumps(key) not in serialized


def test_default_mode_retains_unknown_count_and_its_conservative_shared_support_rule():
    with _modules() as (kernel, instances):
        points, fits = _fixture(shared_plane=True)
        _install_fit(kernel, fits)
        result = kernel.inspect_cad_guided_scan(
            points, [_feature(kernel, "one"), _feature(kernel, "two", 30.)],
            registration_status="DECLARED_ALIGNED")
        assert result["observed_bend_count"] is None
        assert result["measured_conditional_feature_count"] == 0
        assert "finite_support_instances" not in result
        assert "observed_support_instance_count" not in result
        assert all(row["reasons"] == ["AMBIGUOUS_SHARED_FEATURE_SUPPORT"] for row in result["features"])


def test_default_single_fit_remains_a_measurement_not_an_observed_count():
    with _modules() as (kernel, instances):
        points, fits = _fixture()
        _install_fit(kernel, fits)
        result = kernel.inspect_cad_guided_scan(
            points, [_feature(kernel, "one")], registration_status="DECLARED_ALIGNED")
        assert result["measured_conditional_feature_count"] == 1
        assert result["observed_bend_count"] is None
        assert "finite_support_instances" not in result
        assert "_instance_support" not in result["features"][0]["measurements"]


def test_registration_admission_precedes_stipulated_fits_and_instance_creation():
    with _modules() as (kernel, instances):
        points, fits = _fixture()
        calls = _install_fit(kernel, fits)
        result = kernel.inspect_cad_guided_scan(
            points, [_feature(kernel, "one")], instance_policy=instances.SupportPolicy(2.1, 6., 30., 3))
        assert calls == []
        assert result["status"] == result["instance_extraction_status"] == "UNAVAILABLE"
        assert result["observed_support_instance_count"] == 0 and result["observed_bend_count"] is None


def test_actual_feature_and_policy_types_are_required_at_the_real_entry():
    with _modules() as (kernel, instances):
        points, fits = _fixture()
        calls = _install_fit(kernel, fits)
        with pytest.raises(ValueError, match="CADGuidedFeature"):
            _inspect(kernel, instances, points, [object()])
        with pytest.raises(ValueError, match="SupportPolicy"):
            kernel.inspect_cad_guided_scan(points, [_feature(kernel, "one")], instance_policy=object())
        assert calls == []
