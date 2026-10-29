"""Cross-document consistency: result vs CompiledPart vs recipe (pure).

``ShopfloorInspectionResultV1`` validates only its own internal honesty rules.
A result that is internally consistent can still hide characteristics: drop a
recipe bend, drop a general measurement, mark a REQUIRED flange side
NOT_APPLICABLE(NOT_INSPECTED_BY_RECIPE), or carry an expected value or a
tolerance that differs from the compiled nominal. Any of those could produce a
green summary while skipping what the recipe asked for.

:func:`check_result_against_compiled` closes that gap. It returns a sorted list
of problem codes (empty = consistent):

- ``RECIPE_REF_MISMATCH`` / ``SKU_MISMATCH``: the result names another recipe.
- ``COMPILED_PART_REF_MISMATCH``: ``engine.compiled_part_sha256`` differs.
- ``COMPILE_BLOCKED``: the CompiledPart is not COMPILED.
- every code of :func:`contracts.production.compiled_part.check_compiled_against_recipe`.
- ``INSPECTION_STAGE_MISMATCH``: ``stage_decision.inspection_stage`` differs
  from the recipe's.
- ``BEND_INVENTORY_MISMATCH``: bend rows (uid, sequence, label, move) differ
  from the compiled bends.
- ``GENERAL_MEASUREMENT_MISMATCH``: general-measurement ids/order differ.
- ``CHARACTERISTIC_UNKNOWN``: a row id has no compiled nominal.
- ``CHARACTERISTIC_IDENTITY_MISMATCH``: kind/side/stage/unit/boundary differ.
- ``EXPECTED_MISMATCH``: ``expected`` != ``round6(nominal)``.
- ``TOLERANCE_MISMATCH``: the row's tolerance != the nominal's.
- ``REQUIREMENT_MISMATCH``: NOT_INSPECTED_BY_RECIPE on a REQUIRED nominal, a
  NOT_INSPECTED nominal reported as applicable, or ``required`` differing
  from :meth:`CharacteristicResultV1.required_for`.
- ``BACKGAUGE_STAGE_MISMATCH``: a backgauge row on a bend other than the
  just-formed bend of the inferred stage.
- ``BACKGAUGE_MISSING``: an order-consistent mid-sequence stage S_k whose
  compiled backgauge nominal is absent from bend k's row.
- ``STAGE_MODEL_MISMATCH``: ``stage_decision.stage_model_id`` is not a
  compiled stage, or it leaves a formed-like bend unformed or forms a
  NOT_YET_FORMED bend (UNKNOWN bends are unconstrained).
- ``STAGE_APPLICABILITY_MISMATCH``: a NOT_APPLICABLE_AT_STAGE row the stage
  does not justify. A ``bend_pair`` general measurement needs one of its two
  bends NOT_YET_FORMED; a ``points`` one needs some NOT_YET_FORMED bend. A
  backgauge row is rejected when the formed set is exactly the first k bends
  and the decode is the prefix model ``S<k>`` (its nominal then applies).

Producers and consumers (ENGINE_IO §4.3): E3 runs it on its own result before
writing ``result.json`` (a non-empty list is an internal error, exit 3); the
runner/C06 run it on ingest and fail the job with ENGINE_OUTPUT_INVALID; C07
may re-run it before serving.
"""
from __future__ import annotations

from contracts.production.common import round6
from contracts.production.compiled_part import CompiledPartV1, check_compiled_against_recipe
from contracts.production.inspection_result import (
    CharacteristicResultV1,
    ShopfloorInspectionResultV1,
    iter_characteristics,
)
from contracts.production.part_recipe import PartRecipeV1

__all__ = ["RESULT_CONSISTENCY_CODES", "check_result_against_compiled"]

RESULT_CONSISTENCY_CODES: frozenset[str] = frozenset(
    {
        "RECIPE_REF_MISMATCH",
        "SKU_MISMATCH",
        "COMPILED_PART_REF_MISMATCH",
        "COMPILE_BLOCKED",
        "INSPECTION_STAGE_MISMATCH",
        "BEND_INVENTORY_MISMATCH",
        "GENERAL_MEASUREMENT_MISMATCH",
        "CHARACTERISTIC_UNKNOWN",
        "CHARACTERISTIC_IDENTITY_MISMATCH",
        "EXPECTED_MISMATCH",
        "TOLERANCE_MISMATCH",
        "REQUIREMENT_MISMATCH",
        "BACKGAUGE_STAGE_MISMATCH",
        "BACKGAUGE_MISSING",
        "STAGE_MODEL_MISMATCH",
        "STAGE_APPLICABILITY_MISMATCH",
        # from check_compiled_against_recipe
        "CAD_SHA_MISMATCH",
        "FLAT_SHA_MISMATCH",
        "NOMINAL_UNKNOWN_REFERENCE",
    }
)

_IDENTITY_FIELDS = ("kind", "bend_uid", "side", "stage_index", "measurement_id", "unit", "boundary_kind", "adjacent_bend_uid")


def _check_characteristic(char: CharacteristicResultV1, compiled: CompiledPartV1, problems: list[str]) -> None:
    try:
        nom = compiled.nominal(char.characteristic_id)
    except KeyError:
        problems.append("CHARACTERISTIC_UNKNOWN")
        return
    if any(getattr(char, f) != getattr(nom, f) for f in _IDENTITY_FIELDS):
        problems.append("CHARACTERISTIC_IDENTITY_MISMATCH")
    expected = None if nom.nominal is None else round6(nom.nominal)
    if char.expected != expected:
        problems.append("EXPECTED_MISMATCH")
    if char.tolerance != nom.tolerance:
        problems.append("TOLERANCE_MISMATCH")
    if char.reason == "NOT_INSPECTED_BY_RECIPE" and nom.recipe_requirement == "REQUIRED":
        problems.append("REQUIREMENT_MISMATCH")
    if nom.recipe_requirement == "NOT_INSPECTED" and char.state != "NOT_APPLICABLE":
        problems.append("REQUIREMENT_MISMATCH")
    if char.state != "NOT_APPLICABLE" and char.required != CharacteristicResultV1.required_for(nom):
        problems.append("REQUIREMENT_MISMATCH")


def _check_stage_applicability(
    result: ShopfloorInspectionResultV1, recipe: PartRecipeV1, problems: list[str]
) -> None:
    """NOT_APPLICABLE_AT_STAGE only where the decoded stage really leaves the value undefined."""
    state_of = {row.bend_uid: row.state for row in result.bends}
    any_unformed = "NOT_YET_FORMED" in state_of.values()
    planned = {gm.measurement_id: gm for gm in recipe.general_measurements}
    for row in result.general_measurements:
        if row.characteristic.reason != "NOT_APPLICABLE_AT_STAGE":
            continue
        gm = planned.get(row.measurement_id)
        if gm is None:
            continue  # already GENERAL_MEASUREMENT_MISMATCH
        if gm.kind == "bend_pair":
            justified = any(state_of.get(uid) == "NOT_YET_FORMED" for uid in gm.bend_uids)
        else:
            justified = any_unformed
        if not justified:
            problems.append("STAGE_APPLICABILITY_MISMATCH")

    stage = result.stage_decision
    k = stage.inferred_stage
    prefix = {row.bend_uid for row in result.bends if row.sequence_index <= (k or 0)}
    for row in result.bends:
        for char in row.characteristics[4:]:
            if char.reason != "NOT_APPLICABLE_AT_STAGE" or k is None:
                continue
            if set(stage.formed_bend_uids) == prefix and stage.stage_model_id == f"S{k}":
                problems.append("STAGE_APPLICABILITY_MISMATCH")


def check_result_against_compiled(
    result: ShopfloorInspectionResultV1, compiled: CompiledPartV1, recipe: PartRecipeV1
) -> list[str]:
    """Problems between a result, its CompiledPart and its recipe (module docstring)."""
    problems: list[str] = []
    ref = result.recipe
    if (ref.recipe_id, ref.version, ref.recipe_sha256, ref.content_sha256) != (
        recipe.recipe_id,
        recipe.recipe_version,
        recipe.recipe_sha256,
        recipe.content_sha256,
    ):
        problems.append("RECIPE_REF_MISMATCH")
    if result.sku != recipe.sku:
        problems.append("SKU_MISMATCH")
    if result.engine.compiled_part_sha256 != compiled.compiled_part_sha256:
        problems.append("COMPILED_PART_REF_MISMATCH")
    if compiled.status != "COMPILED":
        problems.append("COMPILE_BLOCKED")
    problems.extend(check_compiled_against_recipe(compiled, recipe))

    stage = result.stage_decision
    if stage.inspection_stage != recipe.inspection_stage:
        problems.append("INSPECTION_STAGE_MISMATCH")

    rows = [(r.bend_uid, r.sequence_index, r.label, r.machine_move_index) for r in result.bends]
    planned = [(b.bend_uid, b.sequence_index, b.label, b.machine_move_index) for b in compiled.bends]
    if rows != planned:
        problems.append("BEND_INVENTORY_MISMATCH")
    if [gm.measurement_id for gm in result.general_measurements] != [
        gm.measurement_id for gm in recipe.general_measurements
    ]:
        problems.append("GENERAL_MEASUREMENT_MISMATCH")

    for char in iter_characteristics(result.bends, result.general_measurements):
        _check_characteristic(char, compiled, problems)
    _check_stage_applicability(result, recipe, problems)

    # Backgauge: only on the just-formed bend k of stage S_k, and present there
    # whenever the decode is an order-consistent mid-sequence prefix.
    k = stage.inferred_stage
    for row in result.bends:
        for char in row.characteristics[4:]:
            if char.stage_index != row.sequence_index or (k is not None and char.stage_index != k):
                problems.append("BACKGAUGE_STAGE_MISMATCH")
    n = len(result.bends)
    decodable = (
        result.inspection_validity.state == "VALID"
        and k is not None
        and 1 <= k <= n - 1
        and not {"OUT_OF_ORDER_SUSPECTED", "AMBIGUOUS_POSE", "NOT_EVALUATED"} & set(stage.flags)
    )
    if decodable and k is not None:
        row = result.bends[k - 1]
        wanted = f"{row.bend_uid}/BACKGAUGE_LENGTH/S{k}"
        has_nominal = any(c.characteristic_id == wanted for c in compiled.characteristics)
        if has_nominal and wanted not in {c.characteristic_id for c in row.characteristics}:
            problems.append("BACKGAUGE_MISSING")

    if stage.stage_model_id is not None:
        try:
            model = compiled.stage(stage.stage_model_id)
        except KeyError:
            problems.append("STAGE_MODEL_MISMATCH")
        else:
            # Bends of UNKNOWN state are free; known states must agree with the model.
            formed = set(model.formed_bend_uids)
            if not set(stage.formed_bend_uids) <= formed or formed & set(stage.not_yet_formed_bend_uids):
                problems.append("STAGE_MODEL_MISMATCH")
    return sorted(set(problems))
