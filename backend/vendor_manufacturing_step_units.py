"""Strict native STEP length-unit entities, not unit-name string inference.

Invoke only on the reader for the caller's byte-bound snapshot, inside its
time/RSS supervisor. No transfer, geometry job, file access, or release decision
occurs here. Initial support deliberately excludes conversion-based units.
"""
from __future__ import annotations

import math

MAX_ENTITIES = 100_000
MAX_UNIT_ENTITIES = 32
MAX_SHAPE_REPRESENTATIONS = 32
MAX_SHAPE_CONTEXTS = 32
MAX_UNITS_PER_CONTEXT = 32


class StepUnitsRefused(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise StepUnitsRefused(reason)


def validate_native_step_units(reader) -> dict:
    """Certify every native shape representation's actual assigned length unit.

    A display label such as ``mm`` is never inspected. An unused millimetre
    entity cannot supply a missing unit to another representation's context.
    All length-unit entities, including unused ones, must be explicit SI
    milli/metre; this conservative initial policy cannot silently mix scales.
    """
    from OCP.StepBasic import (
        StepBasic_ConversionBasedUnit, StepBasic_ConversionBasedUnitAndLengthUnit, StepBasic_LengthUnit,
        StepBasic_NamedUnit, StepBasic_SiUnit, StepBasic_SiUnitAndLengthUnit,
        StepBasic_spMilli, StepBasic_sunMetre,
    )
    from OCP.StepGeom import (
        StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext,
        StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx,
    )
    from OCP.StepRepr import StepRepr_RepresentationMap
    from OCP.StepShape import StepShape_ShapeDefinitionRepresentation, StepShape_ShapeRepresentation

    context_types = (
        StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext,
        StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx,
    )
    try:
        model = reader.WS().Model()
        count = model.NbEntities()
        _require(0 < count <= MAX_ENTITIES, "STEP_ENTITY_BUDGET_EXCEEDED")
        # Strongly retain every native handle. OCCT's Number() binding can
        # report zero even for Value(i); pybind's live-handle registry supplies
        # pointer identity instead. A different wrapper is conservatively
        # refused, never matched by a label or equal descriptive fields.
        entities = [model.Value(index) for index in range(1, count + 1)]
        native_indices = {id(entity): index for index, entity in enumerate(entities, 1)}
        _require(len(native_indices) == count, "DUPLICATE_NATIVE_MODEL_ENTITY")

        def native_index(entity) -> int:
            index = native_indices.get(id(entity), 0)
            return index if index and entities[index - 1] is entity else 0

        length_units, representations, named_units = set(), [], 0

        def is_length(unit) -> bool:
            if isinstance(unit, (StepBasic_SiUnitAndLengthUnit,
                                 StepBasic_ConversionBasedUnitAndLengthUnit,
                                 StepBasic_LengthUnit)):
                return True
            # A malformed/non-complex SI metre descriptor must not masquerade
            # as a non-length unit just because its LENGTH_UNIT component is absent.
            if isinstance(unit, StepBasic_SiUnit) and unit.Name() == StepBasic_sunMetre:
                return True
            dimensions = unit.Dimensions()
            if dimensions is None:
                _require(isinstance(unit, StepBasic_SiUnit), "UNVERIFIABLE_NAMED_UNIT_DIMENSIONS")
                return False
            exponents = tuple(float(getattr(dimensions, name)()) for name in (
                "LengthExponent", "MassExponent", "TimeExponent", "ElectricCurrentExponent",
                "ThermodynamicTemperatureExponent", "AmountOfSubstanceExponent", "LuminousIntensityExponent"))
            _require(all(math.isfinite(value) for value in exponents), "INVALID_UNIT_DIMENSIONS")
            return exponents == (1., 0., 0., 0., 0., 0., 0.)

        for index, entity in enumerate(entities, 1):
            if isinstance(entity, StepBasic_NamedUnit):
                named_units += 1
                _require(named_units <= MAX_UNIT_ENTITIES, "STEP_UNIT_ENTITY_BUDGET_EXCEEDED")
                if is_length(entity):
                    _require(not isinstance(entity, StepBasic_ConversionBasedUnit),
                             "CONVERSION_BASED_LENGTH_UNIT_UNSUPPORTED")
                    _require(isinstance(entity, StepBasic_SiUnitAndLengthUnit),
                             "UNSUPPORTED_NATIVE_LENGTH_UNIT_DESCRIPTOR")
                    _require(entity.HasPrefix() and entity.Prefix() == StepBasic_spMilli
                             and entity.Name() == StepBasic_sunMetre, "STEP_NON_MM_LENGTH_UNIT_UNSUPPORTED")
                    length_units.add(index)
            if isinstance(entity, StepShape_ShapeRepresentation):
                representations.append((index, entity))
                _require(len(representations) <= MAX_SHAPE_REPRESENTATIONS,
                         "STEP_SHAPE_REPRESENTATION_BUDGET_EXCEEDED")
            if isinstance(entity, StepShape_ShapeDefinitionRepresentation):
                _require(isinstance(entity.UsedRepresentation(), StepShape_ShapeRepresentation),
                         "UNSUPPORTED_SHAPE_DEFINITION_REPRESENTATION")
            if isinstance(entity, StepRepr_RepresentationMap):
                _require(isinstance(entity.MappedRepresentation(), StepShape_ShapeRepresentation),
                         "UNSUPPORTED_MAPPED_SHAPE_REPRESENTATION")

        _require(bool(length_units), "EXPLICIT_SI_MM_LENGTH_UNIT_MISSING")
        _require(bool(representations), "NATIVE_SHAPE_REPRESENTATION_MISSING")
        contexts, rows = set(), []
        for representation_index, representation in representations:
            context = representation.ContextOfItems()
            _require(isinstance(context, context_types), "SHAPE_CONTEXT_GLOBAL_UNITS_MISSING")
            context_index = native_index(context)
            _require(0 < context_index <= count, "SHAPE_CONTEXT_NOT_IN_NATIVE_MODEL")
            _require(context.CoordinateSpaceDimension() == 3, "SHAPE_CONTEXT_NOT_THREE_DIMENSIONAL")
            contexts.add(context_index)
            _require(len(contexts) <= MAX_SHAPE_CONTEXTS, "STEP_SHAPE_CONTEXT_BUDGET_EXCEEDED")
            unit_count = context.NbUnits()
            _require(0 < unit_count <= MAX_UNITS_PER_CONTEXT, "SHAPE_CONTEXT_UNIT_BUDGET_EXCEEDED")
            assigned_length, assigned_units = [], set()
            for offset in range(1, unit_count + 1):
                unit = context.UnitsValue(offset)
                _require(isinstance(unit, StepBasic_NamedUnit), "INVALID_SHAPE_CONTEXT_UNIT")
                unit_index = native_index(unit)
                _require(0 < unit_index <= count and unit_index not in assigned_units,
                         "DUPLICATE_OR_EXTERNAL_SHAPE_CONTEXT_UNIT")
                assigned_units.add(unit_index)
                if is_length(unit):
                    _require(unit_index in length_units, "UNVERIFIED_SHAPE_CONTEXT_LENGTH_UNIT")
                    assigned_length.append(unit_index)
            _require(len(assigned_length) == 1, "SHAPE_CONTEXT_REQUIRES_ONE_EXPLICIT_MM_LENGTH_UNIT")
            rows.append({"representation_entity_index": representation_index,
                         "context_entity_index": context_index,
                         "length_unit_entity_index": assigned_length[0]})
        return {"unit_policy": "EXPLICIT_SI_MILLI_METRE_LENGTH_ONLY",
                "native_entity_count": count, "native_named_unit_count": named_units,
                "native_shape_context_count": len(contexts),
                "native_shape_representation_count": len(representations),
                "native_si_mm_length_unit_entity_indices": sorted(length_units),
                "shape_representation_unit_assignments": rows,
                "all_shape_representation_length_units_verified": True,
                "source_mm_per_coordinate": 1.0,
                "units": "mm", "units_verified_from_native_step": True,
                "physical_calibration_verified": False}
    except StepUnitsRefused:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise StepUnitsRefused("NATIVE_STEP_UNIT_ENTITY_AUDIT_UNAVAILABLE") from exc
