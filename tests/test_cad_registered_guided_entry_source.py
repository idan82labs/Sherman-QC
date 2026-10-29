"""Registered native entry boundary, exact source, invented arrays/doubles only."""
import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_offline.py"


def entry(registration=None, inspection=None):
    def importer(name, *args, **kwargs):
        if name == "__future__" or name == "numpy" or name.startswith("numpy."):
            return builtins.__import__(name, *args, **kwargs)
        assert name == "cad_guided_registration"
        assert registration is not None, "admission should have stopped before import"
        return SimpleNamespace(register_datum_clouds=registration)

    namespace = {
        "__name__": "invented_registered_core", "__package__": "",
        "__builtins__": {**vars(builtins), "__import__": importer},
    }
    # Execute the whole import-safe module, not copied or AST-pruned runtime.
    exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), namespace)
    namespace["_inspect_native"] = inspection
    return namespace["inspect_registered_native"]


def inputs():
    points = np.arange(180, dtype=float).reshape(60, 3) / 20
    return dict(cad_result=SimpleNamespace(success=True, native_binding="ocp"),
                scan_points=points,
                cad_datum_points=np.zeros((36, 3)), cad_datum_normals=np.ones((36, 3)),
                scan_datum_row_indices=np.arange(36), registration_policy=object(),
                measurement_policy=object(), instance_policy=object(),
                unit_status="DECLARED_MM", allow_ocp=True)


@pytest.mark.parametrize("change,reason", [
    ({"allow_ocp": False}, "EXPLICIT_OCP_BINDING_REQUIRED"),
    ({"unit_status": "UNVERIFIED"}, "EXPLICIT_NATIVE_MILLIMETRE_DECLARATION_REQUIRED"),
    ({"instance_policy": None}, "EXPLICIT_FINITE_SUPPORT_POLICY_REQUIRED"),
    ({"scan_points": np.zeros((200_001, 3))}, "INVALID_OR_OVER_BUDGET_SCAN"),
    ({"scan_points": np.full((60, 3), np.nan)}, "UNSUPPORTED_POINT_NUMERICS"),
    ({"scan_points": np.full((60, 3), np.iinfo(np.int64).min)}, "UNSUPPORTED_POINT_NUMERICS"),
    ({"scan_datum_row_indices": np.arange(36, dtype=float)}, "INVALID_DATUM_ROW_SELECTION"),
    ({"scan_datum_row_indices": np.zeros(36, dtype=int)}, "INVALID_DATUM_ROW_SELECTION"),
    ({"scan_datum_row_indices": np.arange(35)}, "INVALID_DATUM_ROW_SELECTION"),
    ({"scan_datum_row_indices": np.full(36, np.iinfo(np.uint64).max, dtype=np.uint64)}, "INVALID_DATUM_ROW_SELECTION"),
    ({"scan_datum_row_indices": np.arange(60)}, "NO_DISJOINT_INSPECTION_ROWS"),
])
def test_admission_precedes_registration_import(change, reason):
    arguments = inputs()
    arguments.update(change)
    payload = entry()(**arguments)
    assert payload["reasons"] == [reason]
    assert payload["registration_status"] == payload["instance_extraction_status"] == "UNAVAILABLE"
    assert payload["finite_support_instances"] == []
    assert payload["observed_bend_count"] is None
    assert payload["production_use_allowed"] is False


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_opt_in_strict(flag):
    arguments = inputs()
    arguments["allow_ocp"] = flag
    with pytest.raises(TypeError):
        entry()(**arguments)


def test_unavailable_registration_never_calls_measurement_or_emits_instances():
    result = {"status": "UNAVAILABLE", "transform_scan_to_cad": None,
              "reasons": ["AMBIGUOUS_DATUM_POSE"]}
    payload = entry(lambda *args, **kwargs: result)(**inputs())
    assert payload["registration"] is result
    assert payload["reasons"] == ["DATUM_REGISTRATION_UNAVAILABLE"]
    assert payload["expected_feature_count"] is None
    assert payload["finite_support_instances"] == []
    assert payload["observed_support_instance_count"] == 0
    assert payload["observed_bend_count"] is None


@pytest.mark.parametrize("matrix", [np.diag([-1., 1., 1., 1.]), np.diag([2., 2., 2., 1.]),
                                     np.zeros((3, 3)), np.full((4, 4), np.nan)])
def test_nonproper_or_scaled_transform_never_reaches_measurements(matrix):
    result = {"status": "ACCEPTED_BOUNDED", "transform_scan_to_cad": matrix}
    payload = entry(lambda *args, **kwargs: result)(**inputs())
    assert payload["reasons"] == ["REGISTRATION_NOT_PROPER_RIGID"]
    assert payload["finite_support_instances"] == []


def test_actual_entry_solves_only_datum_rows_and_measures_only_other_transformed_rows():
    arguments = inputs()
    points = arguments["scan_points"].copy()
    arguments["scan_datum_row_indices"] = np.arange(0, 60)[::1][:36][::-1]
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, [7, -3, 2]
    result = {"status": "ACCEPTED_BOUNDED", "transform_scan_to_cad": matrix.tolist(),
              "evidence": {"search_status": "SEARCH_BOUNDED_NOT_GLOBAL"}}

    def register(scan, cad, normals, *, policy):
        assert np.array_equal(scan, points[arguments["scan_datum_row_indices"]])
        assert cad is arguments["cad_datum_points"] and normals is arguments["cad_datum_normals"]
        assert policy is arguments["registration_policy"]
        return result

    def inspect(cad, aligned, **kwargs):
        assert cad is arguments["cad_result"]
        assert np.array_equal(aligned, points[36:] @ rotation.T + matrix[:3, 3])
        assert kwargs == {"measurement_policy": arguments["measurement_policy"], "unit_status": "DECLARED_MM",
                          "allow_ocp": True,
                          "instance_policy": arguments["instance_policy"]}
        return {
            "status": "COMPLETE", "observed_bend_count": None, "conformance": "INDETERMINATE",
            "finite_support_instances": [{"bend_id": "invented"}],
            "observed_support_instance_count": 1, "instance_extraction_status": "COMPLETE",
            "production_use_allowed": False}

    payload = entry(register, inspect)(**arguments)
    assert payload["registration_status"] == "ACCEPTED_BOUNDED"
    assert payload["registration"] is result
    assert payload["datum_input_row_count"] == 36 and payload["inspection_input_row_count"] == 24
    assert payload["datum_selection_status"] == "EXPLICIT_STABLE_REGION_SELECTION_NOT_AUTOMATIC"
    assert payload["reference_frame"] == "CAD_REFERENCE"
    assert payload["line_convention"] == "CYLINDER_CENTER_AXIS_SUPPORT"
    assert payload["observed_support_instance_count"] == 1 and payload["observed_bend_count"] is None
    assert np.array_equal(arguments["scan_points"], points)
