"""Build an immutable PartRecipeV1 from a catalog part and its confirmed Vendor Setup draft.

Pure function, no IO: the C05 release gate loads the part, bend spec rows and
the draft (``catalog.get_vendor_setup_draft``: includes ``is_stale`` and
``draft_revision``), copies the CAD/drawing/flat by sha256, and then calls
:func:`build_part_recipe`. Failure raises :class:`RecipeBuildError` with
**every** applicable reason code (sorted), never just the first.

Rules (mirroring apps/api/services/vendor_setup_service.py:89-204 without
importing it):

- Draft: present (DRAFT_MISSING), not stale and bound to the same candidate
  generation and CAD bytes (DRAFT_STALE / CAD_HASH_MISMATCH /
  DRAWING_HASH_MISMATCH), at the revision the releaser saw
  (DRAFT_REVISION_CHANGED), structurally valid (DRAFT_INVALID).
- Active marks are ``status == "active"``; removed marks are ignored for
  confirmation and recorded in ``removed_candidates``. At least one active
  mark (NO_ACTIVE_BENDS); bending order confirmed (SEQUENCE_NOT_CONFIRMED);
  every active mark review-confirmed (UNCONFIRMED_ACTIVE_MARKS); every active
  mark located (MARK_WITHOUT_LINE); the sequence equals the active set
  (DRAFT_INVALID).
- Nominal angle: CAD candidates under NATIVE_SWEEP give ``included = 180 -
  target_angle`` (sweep = target_angle, CAD_NATIVE_SWEEP); under
  MESH_INCLUDED_FOLDED ``included = target_angle`` (CAD_MESH_FOLDED). Both are
  ``convention_verified = False``. Manual marks need
  ``bend_overrides[feature_id].manual_included_angle_deg``
  (MANUAL_MARK_MISSING_NOMINAL_ANGLE), source MANUAL_ENTRY.
- Drawing required (DRAWING_MISSING); default tolerances required
  (TOLERANCES_MISSING / TOLERANCE_INVALID); override keys must be active marks
  (OVERRIDE_UNKNOWN_BEND) or general measurements
  (OVERRIDE_UNKNOWN_MEASUREMENT).
- מהלך groups (``machine_moves``, only when ``machine_moves_explicit``) are
  mapped from feature ids to ``bend_uid`` and must be ordered contiguous
  slices of the sequence (DRAFT_INVALID).
- ``label = "B{sequence_index}"``, ``cad_label`` = the draft mark label,
  ``sku = canonical_sku(part["part_number"])`` (SKU_INVALID).
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from contracts.production.canonical import sha256_hex, strip_keys
from contracts.production.part_recipe import (
    RECIPE_BUILD_REASON_CODES,
    AngleNominal,
    BendTolerances,
    CadRef,
    DrawingRef,
    FlangeInspection,
    FlatRef,
    PartRecipeV1,
    RecipeBendV1,
    RecipeGeneralMeasurementV1,
    ReleasedBy,
    ReleaseRequestV1,
    RemovedCandidate,
    SourceDraftRef,
    Units,
    make_recipe_id,
)
from domains.production.recipe.bend_identity import bend_uid
from domains.production.recipe.canonical_sku import SkuError, canonical_sku

__all__ = ["RecipeBuildError", "build_part_recipe", "draft_sha256", "DRAFT_VOLATILE_KEYS"]

#: Draft keys that are bookkeeping, not operator content (excluded from draft_sha256).
DRAFT_VOLATILE_KEYS: tuple[str, ...] = ("updated_at", "draft_revision", "is_stale", "expected_draft_revision")

_TOLERANCE_LOC_NAMES = frozenset(
    {
        "plus",
        "minus",
        "angle_deg",
        "flange_mm",
        "flange_a_mm",
        "flange_b_mm",
        "bend_line_position_mm",
        "backgauge_mm",
        "general_mm",
        "surface_profile_mm",
        "default_tolerances",
        "tolerance",
    }
)


class RecipeBuildError(ValueError):
    """Release refused. ``codes`` are sorted unique RECIPE_BUILD_REASON_CODES;
    ``details`` maps a code to human-readable specifics (feature ids, fields)."""

    def __init__(self, codes: Iterable[str], details: Mapping[str, Sequence[str]] | None = None) -> None:
        unique = tuple(sorted(set(codes)))
        unknown = set(unique) - RECIPE_BUILD_REASON_CODES
        if unknown:
            raise ValueError(f"unknown recipe build codes: {sorted(unknown)}")
        if not unique:
            raise ValueError("RecipeBuildError needs at least one code")
        self.codes = unique
        self.details: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in sorted((details or {}).items())}
        super().__init__(", ".join(unique))


class _Codes:
    def __init__(self) -> None:
        self.details: dict[str, list[str]] = {}

    def add(self, code: str, detail: str = "") -> None:
        bucket = self.details.setdefault(code, [])
        if detail and detail not in bucket:
            bucket.append(detail)

    def __bool__(self) -> bool:
        return bool(self.details)

    def raise_if_any(self) -> None:
        if self.details:
            raise RecipeBuildError(self.details.keys(), self.details)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_point(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        point = [float(v) for v in value]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(point) != 3 or not all(math.isfinite(v) for v in point):
        return None
    return point


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def draft_sha256(draft: Mapping[str, Any]) -> str:
    """Canonical sha256 of the draft's operator content (volatile keys removed)."""
    return sha256_hex(strip_keys(dict(draft), DRAFT_VOLATILE_KEYS))


def _validation_codes(exc: ValidationError) -> set[str]:
    codes: set[str] = set()
    for err in exc.errors():
        loc = {str(part) for part in err.get("loc", ())}
        codes.add("TOLERANCE_INVALID" if loc & _TOLERANCE_LOC_NAMES else "DRAFT_INVALID")
    return codes


def build_part_recipe(
    *,
    part: Mapping[str, Any] | Any,
    spec_rows: Sequence[Mapping[str, Any] | Any],
    draft: Mapping[str, Any] | None,
    release: ReleaseRequestV1 | Mapping[str, Any],
    cad: CadRef,
    drawing: DrawingRef | None,
    version: int,
    supersedes_version: int | None,
    released_by: ReleasedBy | Mapping[str, Any],
    released_at: str,
    flat: FlatRef | None = None,
    cad_units_verified: bool = False,
) -> PartRecipeV1:
    """Return the finalized recipe or raise :class:`RecipeBuildError` (module docstring)."""
    errors = _Codes()

    # -- identity --------------------------------------------------------
    raw_sku = _get(part, "part_number")
    sku: str | None = None
    try:
        sku = canonical_sku(raw_sku)
    except SkuError:
        errors.add("SKU_INVALID", repr(raw_sku))

    # -- release request -------------------------------------------------
    request: ReleaseRequestV1 | None = None
    if isinstance(release, ReleaseRequestV1):
        request = release
    else:
        try:
            request = ReleaseRequestV1.model_validate(release)
        except ValidationError as exc:
            for code in _validation_codes(exc):
                errors.add(code, "release request")
    if request is not None and request.default_tolerances is None:
        errors.add("TOLERANCES_MISSING")
    if drawing is None:
        errors.add("DRAWING_MISSING")

    if draft is None:
        errors.add("DRAFT_MISSING")
        errors.raise_if_any()
        raise AssertionError("unreachable")  # pragma: no cover - raise_if_any always raises here

    # -- draft binding -----------------------------------------------------
    draft_revision = draft.get("draft_revision")
    if draft.get("is_stale") is True:
        errors.add("DRAFT_STALE", "catalog reports the draft stale")
    if draft.get("candidate_generation") != cad.candidate_generation:
        errors.add("DRAFT_STALE", "candidate generation changed")
    if draft.get("cad_file_hash") not in (None, cad.sha256):
        errors.add("CAD_HASH_MISMATCH", "draft is bound to other CAD bytes")
    if drawing is not None and draft.get("drawing_file_hash") not in (None, drawing.sha256):
        errors.add("DRAWING_HASH_MISMATCH", "draft is bound to another drawing")
    if not isinstance(draft_revision, int) or isinstance(draft_revision, bool) or draft_revision < 1:
        errors.add("DRAFT_INVALID", "draft_revision missing")
    elif request is not None and request.expected_draft_revision != draft_revision:
        errors.add("DRAFT_REVISION_CHANGED", f"expected {request.expected_draft_revision}, found {draft_revision}")

    marks = draft.get("marks") or []
    sequence = list(draft.get("sequence") or [])
    if not isinstance(marks, list) or not all(isinstance(m, Mapping) for m in marks):
        errors.add("DRAFT_INVALID", "marks must be objects")
        errors.raise_if_any()
    ids = [m.get("feature_id") for m in marks]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        errors.add("DRAFT_INVALID", "mark feature ids must be unique non-empty strings")
        errors.raise_if_any()
    by_id: dict[str, Mapping[str, Any]] = {m["feature_id"]: m for m in marks}

    # Candidates must still equal the published CAD spec rows (geometry immutable).
    specs = {str(_get(row, "id")): row for row in spec_rows}
    for spec_id, row in specs.items():
        mark = by_id.get(spec_id)
        if mark is None or mark.get("source") != "cad_candidate":
            errors.add("DRAFT_STALE", f"CAD candidate {spec_id} not represented")
            continue
        same = (
            mark.get("label") == _get(row, "bend_id")
            and mark.get("line_start") == _parse_point(_get(row, "bend_line_start"))
            and mark.get("line_end") == _parse_point(_get(row, "bend_line_end"))
            and mark.get("target_angle") == _get(row, "target_angle")
            and mark.get("target_radius") == _get(row, "target_radius")
        )
        if not same:
            errors.add("DRAFT_STALE", f"CAD candidate {spec_id} geometry differs from the published CAD")
    for mark in marks:
        if mark.get("source") == "cad_candidate" and mark["feature_id"] not in specs:
            errors.add("DRAFT_STALE", f"unknown CAD candidate {mark['feature_id']}")
        if mark.get("source") not in ("cad_candidate", "manual") or mark.get("status") not in ("active", "removed"):
            errors.add("DRAFT_INVALID", f"mark {mark['feature_id']} has an invalid source/status")

    active = [m for m in marks if m.get("status") == "active"]
    removed = [m for m in marks if m.get("status") == "removed"]
    active_ids = {m["feature_id"] for m in active}
    if not active:
        errors.add("NO_ACTIVE_BENDS")
    if active and draft.get("operator_sequence_confirmed") is not True:
        errors.add("SEQUENCE_NOT_CONFIRMED")
    for mark in active:
        if mark.get("review_status") != "confirmed":
            errors.add("UNCONFIRMED_ACTIVE_MARKS", mark["feature_id"])
        start, end = _parse_point(mark.get("line_start")), _parse_point(mark.get("line_end"))
        if start is None or end is None or start == end:
            errors.add("MARK_WITHOUT_LINE", mark["feature_id"])
    if len(sequence) != len(set(sequence)) or set(sequence) != active_ids:
        errors.add("DRAFT_INVALID", "sequence must contain each active mark exactly once")
    for mark in removed:
        if mark.get("issue") not in ("missing", "wrong_location", "not_a_bend"):
            errors.add("DRAFT_INVALID", f"removed mark {mark['feature_id']} needs an issue")

    overrides = request.bend_overrides if request is not None else {}
    for key in overrides:
        if key not in active_ids:
            errors.add("OVERRIDE_UNKNOWN_BEND", key)
    raw_measurements = draft.get("general_measurements") or []
    measurement_ids = {str(gm.get("id")) for gm in raw_measurements if isinstance(gm, Mapping)}
    m_overrides = request.measurement_overrides if request is not None else {}
    for key in m_overrides:
        if key not in measurement_ids:
            errors.add("OVERRIDE_UNKNOWN_MEASUREMENT", key)

    # -- nominal angles ----------------------------------------------------
    nominals: dict[str, AngleNominal] = {}
    for mark in active:
        fid = mark["feature_id"]
        override = overrides.get(fid)
        manual_angle = override.manual_included_angle_deg if override is not None else None
        try:
            if mark.get("source") == "manual":
                if manual_angle is None:
                    errors.add("MANUAL_MARK_MISSING_NOMINAL_ANGLE", fid)
                    continue
                nominals[fid] = AngleNominal(
                    included_deg=manual_angle,
                    sweep_deg=180.0 - manual_angle,
                    source="MANUAL_ENTRY",
                    convention_verified=False,
                )
            else:
                if manual_angle is not None:
                    errors.add("DRAFT_INVALID", f"manual_included_angle_deg applies to manual marks only ({fid})")
                target = mark.get("target_angle")
                if not _finite_number(target):
                    errors.add("DRAFT_INVALID", f"CAD candidate {fid} has no nominal angle")
                    continue
                if cad.candidate_angle_convention == "NATIVE_SWEEP":
                    if not 0 < target < 180:
                        errors.add("DRAFT_INVALID", f"CAD candidate {fid} sweep out of range")
                        continue
                    nominals[fid] = AngleNominal(
                        included_deg=180.0 - float(target),
                        sweep_deg=float(target),
                        source="CAD_NATIVE_SWEEP",
                        convention_verified=False,
                    )
                else:
                    nominals[fid] = AngleNominal(
                        included_deg=float(target), source="CAD_MESH_FOLDED", convention_verified=False
                    )
        except ValidationError:
            errors.add("DRAFT_INVALID", f"nominal angle of {fid} out of range")

    # -- identities ----------------------------------------------------------
    uid_by_fid: dict[str, str] = {}
    for mark in active:
        start, end = _parse_point(mark.get("line_start")), _parse_point(mark.get("line_end"))
        if start is None or end is None or start == end:
            continue
        radius = mark.get("target_radius")
        try:
            uid_by_fid[mark["feature_id"]] = bend_uid(
                cad.sha256, mark.get("source", ""), start, end, float(radius) if _finite_number(radius) else None
            )
        except ValueError as exc:
            errors.add("DRAFT_INVALID", f"{mark['feature_id']}: {exc}")
    seen: dict[str, str] = {}
    for fid, uid in uid_by_fid.items():
        if uid in seen:
            errors.add("DUPLICATE_BEND_IDENTITY", f"{seen[uid]} and {fid}")
        seen.setdefault(uid, fid)

    # -- machine moves (מהלך) ----------------------------------------------
    moves_fids: list[list[str]] = []
    if draft.get("machine_moves_explicit") is True:
        moves_fids = [list(move) for move in (draft.get("machine_moves") or [])]
    cursor = 0
    move_index_by_fid: dict[str, int] = {}
    for number, move in enumerate(moves_fids, start=1):
        if not move:
            errors.add("DRAFT_INVALID", "machine moves must be non-empty")
            continue
        try:
            start_at = sequence.index(move[0], cursor)
        except ValueError:
            errors.add("DRAFT_INVALID", "machine moves must be ordered contiguous bend groups")
            continue
        if sequence[start_at:start_at + len(move)] != move:
            errors.add("DRAFT_INVALID", "machine moves must be ordered contiguous bend groups")
            continue
        for fid in move:
            move_index_by_fid[fid] = number
        cursor = start_at + len(move)

    # -- general measurements ------------------------------------------------
    measurements: list[RecipeGeneralMeasurementV1] = []
    for gm in raw_measurements:
        if not isinstance(gm, Mapping):
            errors.add("MEASUREMENT_INVALID", "measurement must be an object")
            continue
        gid = str(gm.get("id"))
        bend_ids = list(gm.get("bend_ids") or [])
        if gm.get("kind") == "bend_pair" and not all(b in uid_by_fid for b in bend_ids):
            errors.add("MEASUREMENT_INVALID", f"{gid}: bend-pair must reference active located bends")
            continue
        override = m_overrides.get(gid)
        target = gm.get("target_mm")
        try:
            measurements.append(
                RecipeGeneralMeasurementV1(
                    measurement_id=gid,
                    label=gm.get("label"),
                    kind=gm.get("kind"),
                    bend_uids=tuple(uid_by_fid[b] for b in bend_ids) if gm.get("kind") == "bend_pair" else (),
                    point_start_mm=gm.get("point_start"),
                    point_end_mm=gm.get("point_end"),
                    cad_distance_mm=gm.get("cad_distance"),
                    nominal_mm=target if target is not None else gm.get("cad_distance"),
                    nominal_source="OPERATOR_TARGET" if target is not None else "CAD_DISTANCE",
                    tolerance=override.tolerance if override is not None else None,
                    required=override.required if override is not None and override.required is not None else True,
                    note=gm.get("note") or "",
                )
            )
        except (ValidationError, TypeError) as exc:
            errors.add("MEASUREMENT_INVALID", f"{gid}: {exc.__class__.__name__}")

    # -- flat binding ----------------------------------------------------------
    if flat is not None:
        unknown = set(flat.binding.bend_line_map.values()) - set(uid_by_fid.values())
        if unknown:
            errors.add("FLAT_BINDING_UNKNOWN_BEND", ", ".join(sorted(unknown)))

    errors.raise_if_any()
    if request is None or request.default_tolerances is None or drawing is None or sku is None:
        raise AssertionError("unreachable: missing inputs were reported above")  # pragma: no cover

    # -- assemble -----------------------------------------------------------
    bends: list[RecipeBendV1] = []
    for index, fid in enumerate(sequence, start=1):
        mark = by_id[fid]
        override = overrides.get(fid)
        tolerances = BendTolerances()
        inspection = FlangeInspection()
        if override is not None:
            tolerances = BendTolerances(
                angle_deg=override.angle_deg,
                flange_a_mm=override.flange_a_mm,
                flange_b_mm=override.flange_b_mm,
                bend_line_position_mm=override.bend_line_position_mm,
                backgauge_mm=override.backgauge_mm,
            )
            inspection = override.flange_inspection or FlangeInspection()
        radius = mark.get("target_radius")
        bends.append(
            RecipeBendV1(
                bend_uid=uid_by_fid[fid],
                sequence_index=index,
                label=f"B{index}",
                cad_label=str(mark.get("label") or fid),
                source=mark["source"],
                source_feature_id=fid,
                corrects_feature_id=mark.get("corrects_feature_id"),
                machine_move_index=move_index_by_fid.get(fid),
                line_start_mm=_parse_point(mark["line_start"]),
                line_end_mm=_parse_point(mark["line_end"]),
                nominal_angle=nominals[fid],
                nominal_inner_radius_mm=float(radius) if _finite_number(radius) and radius > 0 else None,
                tolerances=tolerances,
                flange_inspection=inspection,
            )
        )
    removed_candidates = [
        RemovedCandidate(
            feature_id=m["feature_id"],
            label=str(m.get("label") or m["feature_id"]),
            issue=m["issue"],
            note=m.get("note") or "",
        )
        for m in removed
    ]
    vendor_schema_version = draft.get("schema_version")
    try:
        return PartRecipeV1.finalize(
            recipe_id=make_recipe_id(sku, version),
            sku=sku,
            sku_display=str(raw_sku).strip(),
            part_id=str(_get(part, "id")),
            part_name=_get(part, "part_name"),
            part_revision=_get(part, "revision"),
            material=_get(part, "material"),
            thickness_mm=request.thickness_mm,
            recipe_version=version,
            supersedes_version=supersedes_version,
            released_at=released_at,
            released_by=released_by,
            confirmation=request.confirmation,
            units=Units(cad_units_verified=cad_units_verified),
            cad=cad,
            drawing=drawing,
            flat=flat,
            source_draft=SourceDraftRef(
                draft_revision=draft_revision,
                draft_sha256=draft_sha256(draft),
                vendor_schema_version=vendor_schema_version if isinstance(vendor_schema_version, int) else 1,
            ),
            inspection_stage=request.inspection_stage,
            default_tolerances=request.default_tolerances,
            bends=bends,
            machine_moves=[[uid_by_fid[f] for f in move] for move in moves_fids],
            general_measurements=measurements,
            removed_candidates=removed_candidates,
            release_note=request.release_note,
        )
    except ValidationError as exc:
        codes = _validation_codes(exc)
        summary = str(exc).splitlines()[0]
        raise RecipeBuildError(codes, {code: [summary] for code in codes}) from exc
