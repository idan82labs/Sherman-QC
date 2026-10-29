"""Per-part onboarding checklist (C05, CONTRACTS §9): what is missing before release.

:func:`checklist` dry-runs the release gate without compiling or writing
anything: the SKU, CAD (STEP, hash), drawing, Vendor Setup draft (fresh,
order confirmed, marks confirmed), release-draft tolerances, the flat
decision, and the last compile attempt. It returns::

    {part, sku, checklist: [{key, ok, codes, detail}], blockers: [codes],
     ready_to_release, active_recipe?, onboarding_state, last_compile?,
     flat?, links: {vendor_setup, wizard}}

The last compile attempt blocks (step ``compile``) only when it was BLOCKED
and attempted exactly the recipe content a release would make now
(``content_sha256`` of the dry run's recipe): a BLOCKED compile is
deterministic, so releasing the same content again would block again
(MARK_NOT_IN_FLANGE_GRAPH, UNMARKED_BEND, FLAT_BINDING_INCONSISTENT,
THICKNESS_INCONSISTENT, ...). Anything the operator changes that reaches the
recipe (marks, order, tolerances, thickness, the flat binding) clears it.
COMPILE_TIMEOUT / COMPILE_FAILED / COMPILE_INCONSISTENT attempts are
transient or engine-side: they are shown under ``last_compile`` and as the
step's ``advisory`` codes, never as blockers (a retry may succeed). Attempts
logged before schema v2 carry no content hash and fall back to the CAD +
Vendor Setup draft basis (BLOCKED only).

``onboarding_state``:

- RELEASED / RELEASED_STALE when an active recipe exists. Stale means a
  release now would make a new version: the dry run passes and its content
  differs from the active recipe's, or (when the dry run is blocked) the
  catalog CAD hash, the drawing hash or the Vendor Setup draft revision
  differs from the active recipe's source.
- otherwise the first unmet step: NEEDS_CAD, NEEDS_DRAWING,
  NEEDS_BEND_ORDER (draft, marks, order, flat binding, compile blockers),
  NEEDS_TOLERANCES; else READY_TO_RELEASE.

``ready_to_release`` is true only when the dry run finds no blocker. The
compile itself (E1) still runs at release, so a READY_TO_RELEASE part can
still be refused with compile codes; those then show up here as the last
compile attempt.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from apps.production_line.recipes.store import RecipeStore, iso_utc, utc_now

__all__ = ["ONBOARDING_STATES", "STEP_CODES", "checklist", "onboarding_state_for"]

ONBOARDING_STATES: tuple[str, ...] = (
    "NEEDS_CAD",
    "NEEDS_DRAWING",
    "NEEDS_BEND_ORDER",
    "NEEDS_TOLERANCES",
    "READY_TO_RELEASE",
    "RELEASED",
    "RELEASED_STALE",
)

#: Checklist step -> the release codes that block it.
STEP_CODES: dict[str, frozenset[str]] = {
    "sku": frozenset({"SKU_INVALID", "SKU_COLLISION"}),
    "cad": frozenset({"CAD_MISSING", "CAD_NOT_STEP", "CAD_HASH_MISMATCH", "CAD_READ_FAILED"}),
    "drawing": frozenset({"DRAWING_MISSING", "DRAWING_HASH_MISMATCH", "DRAWING_FORMAT_UNSUPPORTED"}),
    "bend_order": frozenset(
        {
            "DRAFT_MISSING",
            "DRAFT_STALE",
            "DRAFT_INVALID",
            "DRAFT_REVISION_CHANGED",
            "SEQUENCE_NOT_CONFIRMED",
            "NO_ACTIVE_BENDS",
            "UNCONFIRMED_ACTIVE_MARKS",
            "MARK_WITHOUT_LINE",
            "MANUAL_MARK_MISSING_NOMINAL_ANGLE",
            "DUPLICATE_BEND_IDENTITY",
            "MEASUREMENT_INVALID",
            "OVERRIDE_UNKNOWN_BEND",
            "OVERRIDE_UNKNOWN_MEASUREMENT",
        }
    ),
    "flat": frozenset(
        {
            "FLAT_MISSING",
            "FLAT_HASH_MISMATCH",
            "FLAT_PARSE_FAILED",
            "FLAT_BINDING_UNCONFIRMED",
            "FLAT_BINDING_STALE",
            "FLAT_BINDING_UNKNOWN_BEND",
        }
    ),
    "tolerances": frozenset({"TOLERANCES_MISSING", "TOLERANCE_INVALID", "RELEASE_DRAFT_REVISION_CHANGED"}),
}
_STATE_FOR_STEP = {
    "sku": "NEEDS_CAD",
    "cad": "NEEDS_CAD",
    "drawing": "NEEDS_DRAWING",
    "bend_order": "NEEDS_BEND_ORDER",
    "flat": "NEEDS_BEND_ORDER",
    "compile": "NEEDS_BEND_ORDER",
    "tolerances": "NEEDS_TOLERANCES",
}
_STEP_ORDER = ("sku", "cad", "drawing", "bend_order", "flat", "compile", "tolerances")


def onboarding_state_for(blocking_steps: list[str], active: bool, stale: bool) -> str:
    if active:
        return "RELEASED_STALE" if stale else "RELEASED"
    for step in _STEP_ORDER:
        if step in blocking_steps:
            return _STATE_FOR_STEP[step]
    return "READY_TO_RELEASE"


def _dry_run_codes(
    part_id: str,
    *,
    catalog: Any,
    store: RecipeStore,
    data_root: Path,
    request: Mapping[str, Any],
    clock: Callable[[], datetime],
) -> tuple[dict[str, tuple[str, ...]], bool, str | None]:
    """Codes of the release gate's read-only checks (``prepare_release``),
    whether a release now would be ``unchanged`` (same content as the active
    recipe), and the content sha256 of the recipe it would release (None when
    the dry run is refused)."""
    from apps.production_line.recipes.release_service import (
        ReleaseGateError,
        prepare_release,
    )

    try:
        prepared = prepare_release(
            part_id,
            request,
            {"user_id": "onboarding-dry-run", "username": "onboarding-dry-run", "role": "admin"},
            catalog=catalog,
            store=store,
            data_root=data_root,
            clock=clock,
        )
    except ReleaseGateError as exc:
        return {c: exc.details.get(c, ()) for c in exc.codes}, False, None
    active = prepared.active
    content = str(prepared.recipe.content_sha256)
    return {}, active is not None and active.content_sha256 == content, content


def checklist(
    part_id: str,
    *,
    catalog: Any,
    recipe_store: RecipeStore,
    data_root: Path,
    clock: Callable[[], datetime] = utc_now,
    vendor_setup_link: str | None = None,
) -> dict[str, Any] | None:
    """The onboarding checklist of ``part_id`` (module docstring); None when the part is unknown."""
    from domains.production.recipe.canonical_sku import SkuError, canonical_sku

    part = catalog.get_part(part_id)
    if part is None:
        return None
    try:
        sku: str | None = canonical_sku(str(part.get("part_number") or ""))
    except SkuError:
        sku = None
    stored_draft = recipe_store.get_release_draft(part_id)
    request: dict[str, Any] = dict(stored_draft.request) if stored_draft is not None else {}
    draft = catalog.get_vendor_setup_draft(part_id)
    # The dry run checks the release draft *as saved*, against the current Vendor Setup draft.
    request.pop("expected_release_draft_revision", None)
    if draft is not None and isinstance(draft.get("draft_revision"), int):
        request["expected_draft_revision"] = draft["draft_revision"]
    else:
        request.setdefault("expected_draft_revision", 0)
    codes, unchanged, content_sha = _dry_run_codes(part_id, catalog=catalog, store=recipe_store,
                                                   data_root=Path(data_root), request=request, clock=clock)
    if stored_draft is None:
        codes.setdefault("TOLERANCES_MISSING", ("no release settings saved yet",))

    last = recipe_store.last_compile_attempt(part_id)
    last_doc: dict[str, Any] | None = None
    compile_codes: tuple[str, ...] = ()
    advisory_codes: tuple[str, ...] = ()
    if last is not None:
        from domains.production.recipe.builder import draft_sha256

        if last.content_sha256 is not None:
            # Same content as the release the operator would make now.
            applies = content_sha is not None and last.content_sha256 == content_sha
        else:  # a schema-v1 attempt: the CAD + Vendor Setup draft basis
            current = (part.get("cad_file_hash"), draft_sha256(draft) if draft is not None else None)
            applies = (last.cad_sha256, last.draft_sha256) == current
        blocking = applies and last.status == "BLOCKED"
        last_doc = {
            "status": last.status,
            "codes": list(last.codes),
            "at": last.created_at,
            "recipe_sha256": last.recipe_sha256,
            "content_sha256": last.content_sha256,
            "applies_to_current_draft": applies,
            "blocking": blocking,
            "report": last.report,
        }
        if blocking:
            compile_codes = last.codes
        elif applies and last.status != "COMPILED":
            advisory_codes = last.codes

    items = []
    blocking_steps: list[str] = []
    all_codes: set[str] = set(codes)
    for step in _STEP_ORDER:
        advisory: list[str] = []
        if step == "compile":
            step_codes = sorted(compile_codes)
            advisory = sorted(advisory_codes)
            if step_codes:
                detail = ["the last compile of this exact recipe content was BLOCKED; fix the listed issues"]
            elif advisory:
                detail = [f"the last compile attempt ended {last.status if last else ''} (advisory: retry the release)"]
            else:
                detail = []
        else:
            step_codes = sorted(c for c in codes if c in STEP_CODES[step])
            detail = [d for c in step_codes for d in codes.get(c, ())]
        ok = not step_codes
        if not ok:
            blocking_steps.append(step)
        item: dict[str, Any] = {"key": step, "ok": ok, "codes": step_codes, "detail": detail[:20]}
        if step == "compile":
            item["advisory"] = advisory
        items.append(item)
    all_codes.update(compile_codes)
    # Codes that belong to no step (e.g. a conflict) still block.
    known = set().union(*STEP_CODES.values()) | set(compile_codes)
    leftovers = sorted(c for c in codes if c not in known)
    if leftovers:
        items.append({"key": "other", "ok": False, "codes": leftovers,
                      "detail": [d for c in leftovers for d in codes.get(c, ())][:20]})
        blocking_steps.append("other")

    active = recipe_store.get_active(sku) if sku is not None else None
    if active is None:
        active = recipe_store.get_active_for_part(part_id)
    active_doc = None
    stale = False
    if active is not None:
        recipe = active.recipe()
        compiled = recipe_store.get_compiled(active.recipe_sha256)
        if content_sha is not None:  # the dry run passed: stale iff a release would make a new version
            stale = content_sha != active.content_sha256
        else:
            stale = (
                part.get("cad_file_hash") != recipe.cad.sha256
                or part.get("drawing_file_hash") != recipe.drawing.sha256
                or draft is None
                or draft.get("draft_revision") != recipe.source_draft.draft_revision
            )
        active_doc = {
            **active.summary(),
            "compile": None if compiled is None else compiled.to_dict(),
            "source": {
                "cad_sha256": recipe.cad.sha256,
                "drawing_sha256": recipe.drawing.sha256,
                "draft_revision": recipe.source_draft.draft_revision,
            },
        }
    flat = recipe_store.get_part_flat(part_id)
    state = onboarding_state_for(blocking_steps, active is not None, stale)
    links = {
        "vendor_setup": vendor_setup_link or f"/vendor-setup/{part_id}",
        "wizard": f"/production/parts/{part_id}",
    }
    return {
        "part": part,
        "sku": sku,
        "checklist": items,
        "blockers": sorted(all_codes),
        "ready_to_release": not all_codes and not unchanged,
        "release_would_be_unchanged": unchanged,
        "active_recipe": active_doc,
        "onboarding_state": state,
        "last_compile": last_doc,
        "flat": None if flat is None else {"state": flat.state, "format": flat.format, "flat_sha256": flat.flat_sha256,
                                           "revision": flat.revision},
        "release_draft_revision": stored_draft.revision if stored_draft is not None else 0,
        "evaluated_at": iso_utc(clock()),
        "links": links,
    }
