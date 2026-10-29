"""Recipe draft and finalized recipe for an E2 case.

``recipe_draft.json`` (schema ``sherman_synthetic_recipe_draft_v1``) carries
exactly the inputs of the C01 recipe builder
(:func:`domains.production.recipe.builder.build_part_recipe`): the catalog
part, the CAD candidate spec rows, a Vendor-Setup-shaped draft (marks,
confirmed bending order, explicit מהלך groups), the release request
(tolerances, per-bend overrides, inspection stage) and the CAD / drawing /
flat references; plus a human-readable summary (bends in order, groups,
tolerances). ``recipe.json`` is the finalized ``PartRecipeV1`` that builder
returns, as canonical JSON, ready for ``python -m domains.production.engine.compile``.

Mark lines follow the native STEP candidate convention (the mid-thickness,
mid-sweep line of the bend, ``Ri + t/2`` from the axis), which E1's
reconciliation accepts (within ``Ri + t + 1 mm`` of the axis, parallel within
2°). ``target_angle`` is the sweep (``NATIVE_SWEEP``), so the recipe's included
angle equals the compiled flange-graph value. Recipes are marked
``DEV_AUTO_CONFIRMED`` (built from truth, not by an operator). The flat is
released ``UNBOUND``: its bend-line ids are E1's to assign; the generator's
own flat bend lines (with their bend_uids) are listed in the draft summary
for any consumer that wants to bind them.

Exclusions follow one pre-written per-feature rule
(:func:`flange_inspection_overrides`, amendment 4d): a flange side is
NOT_INSPECTED when its far boundary has no value, mixes boundary kinds (the
majority kind holds < 90 % of the stations) or is stepped without a dominant
level (no 0.05 mm level holds >= 80 % of the stations). Those sides have no
station-placement-independent nominal, so a careful operator would not
tolerance them. Neither file names the case or its scenario (amendment 4).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from contracts.production.canonical import canonical_json_bytes
from contracts.production.part_recipe import (
    BendOverrideV1,
    CadRef,
    DrawingRef,
    FlangeInspection,
    FlatBindingV1,
    FlatRef,
    PartRecipeV1,
    ReleasedBy,
    ReleaseRequestV1,
)
from domains.production.recipe.builder import build_part_recipe

from .geometry import PartDesign
from .truth import NominalModel

__all__ = [
    "RECIPE_DRAFT_SCHEMA",
    "DEFAULT_TOLERANCES",
    "RELEASED_AT",
    "RELEASED_BY",
    "CaseRecipe",
    "build_case_recipe",
    "flange_inspection_overrides",
    "MIN_KIND_FRACTION",
]

RECIPE_DRAFT_SCHEMA = "sherman_synthetic_recipe_draft_v1"
#: A flange side whose majority far-boundary kind holds fewer stations is NOT_INSPECTED.
MIN_KIND_FRACTION = 0.9
RELEASED_AT = "2026-10-17T00:00:00Z"
RELEASED_BY = {"user_id": "e2-synthetic", "username": "e2-synthetic", "role": "admin"}
#: Shop-standard defaults used by every synthetic recipe (per-bend overrides on top).
DEFAULT_TOLERANCES: dict[str, Any] = {
    "angle_deg": {"plus": 1.0, "minus": 1.0},
    "flange_mm": {"plus": 0.5, "minus": 0.5},
    "bend_line_position_mm": {"plus": 0.5, "minus": 0.5},
    "backgauge_mm": {"plus": 0.3, "minus": 0.3},
    "general_mm": {"plus": 0.5, "minus": 0.5},
    "surface_profile_mm": 0.5,
    "source": "SHOP_STANDARD",
    "note": "E2 synthetic suite default tolerances",
}


class CaseRecipe:
    """Draft document, finalized recipe and the design-key -> bend_uid map."""

    def __init__(self, draft_doc: dict[str, Any], recipe: PartRecipeV1, uid_by_key: dict[str, str]) -> None:
        self.draft_doc = draft_doc
        self.recipe = recipe
        self.uid_by_key = uid_by_key

    @property
    def recipe_bytes(self) -> bytes:
        return canonical_json_bytes(self.recipe)


def _r6(v: float) -> float:
    out = round(float(v), 6)
    return 0.0 if out == 0.0 else out


def build_case_recipe(
    *,
    design: PartDesign,
    nominal: NominalModel,
    sku: str,
    revision: str,
    part_name: str,
    cad_sha256: str,
    drawing_sha256: str,
    flat_sha256: str,
    inspection_stage: str,
    bend_overrides: dict[str, dict[str, Any]] | None = None,
) -> CaseRecipe:
    """Build the draft and finalize the recipe through the C01 builder.

    ``bend_overrides``: design bend key -> ``BendOverrideV1`` fields."""
    candidate_generation = hashlib.sha256(f"e2-synthetic-candidates|{cad_sha256}".encode()).hexdigest()
    part = {
        "id": f"part-{sku.lower()}",
        "part_number": sku,
        "part_name": part_name,
        "revision": revision,
        "material": f"S235 {design.thickness:g} mm (synthetic)",
    }
    marks: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    fid_by_key: dict[str, str] = {}
    for gi, b in enumerate(design.bends, start=1):
        line = nominal.final.loci(b.key)["mid_radius_apex"]
        start = [_r6(x) for x in line[0]]
        end = [_r6(x) for x in line[1]]
        fid = f"syn-{gi:03d}"
        fid_by_key[b.key] = fid
        label = f"SM-BEND-{gi}"
        specs.append({
            "id": fid, "part_id": part["id"], "bend_id": label, "target_angle": float(b.sweep_deg),
            "target_radius": float(b.inner_radius), "tolerance_angle": None, "tolerance_radius": None,
            "bend_line_start": json.dumps(start), "bend_line_end": json.dumps(end), "notes": None,
        })
        marks.append({
            "feature_id": fid, "label": label, "source": "cad_candidate", "status": "active",
            "review_status": "confirmed", "issue": None, "corrects_feature_id": None, "note": "",
            "line_start": start, "line_end": end, "target_angle": float(b.sweep_deg),
            "target_radius": float(b.inner_radius),
        })
    sequence = [fid_by_key[k] for k in design.sequence]
    moves = [[fid_by_key[k] for k in g] for g in design.groups]
    draft = {
        "schema_version": 1,
        "cad_file_hash": cad_sha256,
        "cad_uploaded_at": "2026-10-17T00:00:00",
        "part_revision": revision,
        "drawing_file_hash": drawing_sha256,
        "candidate_generation": candidate_generation,
        "expected_draft_revision": 1,
        "marks": marks,
        "sequence": sequence,
        "machine_moves": moves,
        "machine_moves_explicit": bool(moves),
        "general_measurements": [],
        "operator_sequence_confirmed": True,
        "completed_steps": 0,
        "matching_search": {"search_radius_mm": None, "search_angle_deg": None},
        "draft_revision": 1,
        "is_stale": False,
        "updated_at": "2026-10-17T00:00:00",
    }
    overrides = {fid_by_key[k]: BendOverrideV1(**v) for k, v in (bend_overrides or {}).items()}
    release = ReleaseRequestV1(
        expected_draft_revision=1,
        default_tolerances=DEFAULT_TOLERANCES,
        bend_overrides=overrides,
        thickness_mm=design.thickness,
        inspection_stage=inspection_stage,
        confirmation="DEV_AUTO_CONFIRMED",
        release_note="E2 synthetic part (DEV_AUTO_CONFIRMED)",
    )
    cad = CadRef(
        sha256=cad_sha256, original_filename="part.step", stored_rel_path=f"recipes/cad/{cad_sha256}.step",
        candidate_generation=candidate_generation, candidate_angle_convention="NATIVE_SWEEP",
    )
    drawing = DrawingRef(
        sha256=drawing_sha256, original_filename="drawing.pdf", stored_rel_path=f"recipes/drawings/{drawing_sha256}.pdf", format="PDF"
    )
    flat = FlatRef(
        sha256=flat_sha256, original_filename="flat.dxf", stored_rel_path=f"recipes/flats/{flat_sha256}.dxf",
        format="DXF", binding=FlatBindingV1(state="UNBOUND"),
    )
    recipe = build_part_recipe(
        part=part, spec_rows=specs, draft=draft, release=release, cad=cad, drawing=drawing, version=1,
        supersedes_version=None, released_by=ReleasedBy(**RELEASED_BY), released_at=RELEASED_AT, flat=flat,
        cad_units_verified=True,
    )
    uid_by_fid = {b.source_feature_id: b.bend_uid for b in recipe.bends}
    uid_by_key = {k: uid_by_fid[f] for k, f in fid_by_key.items()}
    label_by_key = {k: recipe.bend(uid_by_key[k]).label for k in fid_by_key}
    from .flat import flat_pattern

    pattern = flat_pattern(design)
    summary_bends = []
    for k in design.sequence:
        b = design.bend(k)
        rb = recipe.bend(uid_by_key[k])
        sides = {}
        for letter in ("A", "B"):
            fb = nominal.flange[(k, letter)]
            sides[letter] = {
                "panel": nominal.side_panel(k, letter),
                "oml_length_mm": _r6(fb.value_mm) if fb.value_mm is not None else None,
                "boundary_kind": fb.boundary_kind,
                "well_defined": fb.well_defined,
                "inspection": rb.flange_inspection.side_a if letter == "A" else rb.flange_inspection.side_b,
            }
        summary_bends.append({
            "label": label_by_key[k], "bend_uid": uid_by_key[k], "feature_id": fid_by_key[k], "design_key": k,
            "sequence_index": rb.sequence_index, "machine_move_index": rb.machine_move_index,
            "sweep_deg": float(b.sweep_deg), "included_deg": float(180.0 - b.sweep_deg),
            "inner_radius_mm": float(b.inner_radius), "direction_contract": nominal.final.contract_direction(k),
            "sides": sides,
        })
    flat_lines = [
        {"design_key": key, "bend_uid": uid_by_key[key], "start_xy_mm": [_r6(p0[0]), _r6(p0[1])],
         "end_xy_mm": [_r6(p1[0]), _r6(p1[1])], "direction_flat": d, "sweep_deg": float(sw)}
        for key, p0, p1, d, sw, _ri in pattern.bend_lines
    ]
    draft_doc = {
        "schema": RECIPE_DRAFT_SCHEMA,
        "sku": recipe.sku,
        "revision": revision,
        "part_name": part_name,
        "inspection_stage": inspection_stage,
        "summary": {
            "bends_in_order": summary_bends,
            "machine_moves": [[label_by_key[k] for k in g] for g in design.groups],
            "default_tolerances": DEFAULT_TOLERANCES,
            "bend_overrides": {label_by_key[k]: v for k, v in (bend_overrides or {}).items()},
            "thickness_mm": design.thickness,
            "k_factor": design.k_factor,
            "flat_bend_lines": flat_lines,
        },
        "builder_inputs": {
            "part": part,
            "spec_rows": specs,
            "draft": draft,
            "release": release.model_dump(mode="json"),
            "cad": cad.model_dump(mode="json"),
            "drawing": drawing.model_dump(mode="json"),
            "flat": flat.model_dump(mode="json"),
            "version": 1,
            "released_by": RELEASED_BY,
            "released_at": RELEASED_AT,
            "cad_units_verified": True,
        },
        "recipe_sha256": recipe.recipe_sha256,
    }
    return CaseRecipe(draft_doc, recipe, uid_by_key)


def flange_inspection_overrides(nominal: NominalModel, min_kind_fraction: float = MIN_KIND_FRACTION) -> dict[str, dict[str, Any]]:
    """NOT_INSPECTED for flange sides whose far boundary is ill-defined, as a careful
    operator would set it (module docstring): no value, mixed kinds along the bend
    (majority kind < ``min_kind_fraction``), or a stepped edge without a dominant
    level (:attr:`~.analytic.FarBoundary.well_defined`)."""
    out: dict[str, dict[str, Any]] = {}
    for b in nominal.design.bends:
        req = {}
        for letter in ("A", "B"):
            fb = nominal.flange[(b.key, letter)]
            if fb.value_mm is None or fb.kind_fraction < min_kind_fraction or not fb.well_defined:
                req["side_a" if letter == "A" else "side_b"] = "NOT_INSPECTED"
        if req:
            out[b.key] = {"flange_inspection": FlangeInspection(**req).model_dump(mode="json")}
    return out

