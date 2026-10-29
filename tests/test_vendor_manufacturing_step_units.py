"""Small authored STEP/unit entities only; no corpus or physical-input access."""
from types import SimpleNamespace

import pytest

from backend import vendor_manufacturing_step_units as audit


def _mm_unit():
    from OCP.StepBasic import StepBasic_SiUnitAndLengthUnit, StepBasic_spMilli, StepBasic_sunMetre
    unit = StepBasic_SiUnitAndLengthUnit()
    unit.Init(True, StepBasic_spMilli, StepBasic_sunMetre)
    return unit


def _model(*, context_units=None, extra_units=(), dimension=3, plain_context=False):
    from OCP.StepBasic import StepBasic_HArray1OfNamedUnit
    from OCP.StepData import StepData_StepModel
    from OCP.StepGeom import (StepGeom_GeometricRepresentationContext,
                             StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext)
    from OCP.StepShape import StepShape_ShapeRepresentation
    from OCP.TCollection import TCollection_HAsciiString
    unit = _mm_unit()
    model = StepData_StepModel()
    model.AddEntity(unit)
    for other in extra_units:
        model.AddEntity(other)
    units = [unit] if context_units is None else context_units
    array = StepBasic_HArray1OfNamedUnit(1, len(units)) if units else None
    for index, other in enumerate(units, 1):
        array.SetValue(index, other)
    text = TCollection_HAsciiString("")
    if plain_context:
        context = StepGeom_GeometricRepresentationContext()
        context.Init(text, text, dimension)
    else:
        context = StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext()
        context.Init(text, text, dimension, array)
    representation = StepShape_ShapeRepresentation()
    representation.Init(text, None, context)
    model.AddEntity(context)
    model.AddEntity(representation)
    reader = SimpleNamespace(WS=lambda: SimpleNamespace(Model=lambda: model))
    return reader, model, context, unit, representation


def test_exact_si_mm_entity_and_its_context_are_verified():
    reader, model, context, unit, representation = _model()
    result = audit.validate_native_step_units(reader)
    assert result["units_verified_from_native_step"]
    assert result["source_mm_per_coordinate"] == 1.
    assert not result["physical_calibration_verified"]
    assert result["shape_representation_unit_assignments"] == [{
        "representation_entity_index": 3,
        "context_entity_index": 2,
        "length_unit_entity_index": 1}]


@pytest.mark.parametrize("failure", ["metre", "centimetre", "missing_length_component"])
def test_names_or_prefixes_cannot_replace_the_exact_length_descriptor(failure):
    from OCP.StepBasic import StepBasic_SiUnit, StepBasic_spCenti, StepBasic_spMilli, StepBasic_sunMetre
    reader, model, context, unit, _ = _model()
    if failure == "metre":
        unit.UnSetPrefix()
    elif failure == "centimetre":
        unit.SetPrefix(StepBasic_spCenti)
    else:
        unknown = StepBasic_SiUnit()
        unknown.Init(True, StepBasic_spMilli, StepBasic_sunMetre)
        model.AddEntity(unknown)
    with pytest.raises(audit.StepUnitsRefused, match="NON_MM|UNSUPPORTED_NATIVE_LENGTH"):
        audit.validate_native_step_units(reader)


def test_conversion_based_mm_label_is_refused_even_if_unused():
    from OCP.StepBasic import StepBasic_ConversionBasedUnitAndLengthUnit, StepBasic_DimensionalExponents
    from OCP.TCollection import TCollection_HAsciiString
    dimensions = StepBasic_DimensionalExponents()
    dimensions.Init(1., 0., 0., 0., 0., 0., 0.)
    impostor = StepBasic_ConversionBasedUnitAndLengthUnit()
    # A name alone was the previous false authority. Even missing/untrusted
    # conversion factors must be rejected before anyone could rely on the label.
    impostor.Init(dimensions, TCollection_HAsciiString("millimetre"), None)
    reader, *_ = _model(extra_units=[impostor])
    with pytest.raises(audit.StepUnitsRefused, match="CONVERSION_BASED"):
        audit.validate_native_step_units(reader)


def test_unused_mm_entity_cannot_mask_unassigned_active_shape_context():
    reader, *_ = _model(plain_context=True)
    with pytest.raises(audit.StepUnitsRefused, match="GLOBAL_UNITS_MISSING"):
        audit.validate_native_step_units(reader)


@pytest.mark.parametrize("failure", ["outside_model", "duplicate_pointer", "two_length_units", "dimension_two"])
def test_each_actual_context_assignment_is_unique_and_native(failure):
    reader, model, context, unit, _ = _model(dimension=2 if failure == "dimension_two" else 3)
    if failure != "dimension_two":
        from OCP.StepBasic import StepBasic_HArray1OfNamedUnit
        units = [_mm_unit()] if failure == "outside_model" else [unit, unit]
        if failure == "two_length_units":
            units[1] = _mm_unit()
            model.AddEntity(units[1])
        array = StepBasic_HArray1OfNamedUnit(1, len(units))
        for index, value in enumerate(units, 1):
            array.SetValue(index, value)
        context.SetUnits(array)
    with pytest.raises(audit.StepUnitsRefused, match="EXTERNAL|DUPLICATE|ONE_EXPLICIT|THREE_DIMENSIONAL"):
        audit.validate_native_step_units(reader)


def test_one_good_representation_cannot_mask_another_missing_unit_context():
    from OCP.StepGeom import StepGeom_GeometricRepresentationContext
    from OCP.StepShape import StepShape_ShapeRepresentation
    from OCP.TCollection import TCollection_HAsciiString
    reader, model, *_ = _model()
    text = TCollection_HAsciiString("")
    context = StepGeom_GeometricRepresentationContext()
    context.Init(text, text, 3)
    representation = StepShape_ShapeRepresentation()
    representation.Init(text, None, context)
    model.AddEntity(context)
    model.AddEntity(representation)
    with pytest.raises(audit.StepUnitsRefused, match="GLOBAL_UNITS_MISSING"):
        audit.validate_native_step_units(reader)


@pytest.mark.parametrize("budget", ["MAX_ENTITIES", "MAX_UNIT_ENTITIES", "MAX_SHAPE_REPRESENTATIONS", "MAX_SHAPE_CONTEXTS"])
def test_native_entity_work_is_bounded(monkeypatch, budget):
    reader, *_ = _model()
    monkeypatch.setattr(audit, budget, 0)
    with pytest.raises(audit.StepUnitsRefused, match="BUDGET_EXCEEDED"):
        audit.validate_native_step_units(reader)


def test_missing_reader_is_a_typed_refusal():
    with pytest.raises(audit.StepUnitsRefused, match="AUDIT_UNAVAILABLE"):
        audit.validate_native_step_units(None)


def test_real_newly_generated_step_uses_its_actual_complex_context(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
    path = tmp_path / "authored-mm-only-box.step"
    writer = STEPControl_Writer()
    assert writer.Transfer(BRepPrimAPI_MakeBox(10., 20., 1.5).Shape(), STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_RetDone
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    result = audit.validate_native_step_units(reader)
    assert result["native_shape_representation_count"] == result["native_shape_context_count"] == 1
    assert result["native_si_mm_length_unit_entity_indices"]
    assert result["all_shape_representation_length_units_verified"]
