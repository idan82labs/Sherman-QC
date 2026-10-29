"""Production recipe API service (C05): parts, onboarding, release drafts, release, recipes, flats.

Thin, synchronous orchestration over :mod:`apps.production_line.recipes`; the
router (``apps/api/routes/production_recipes.py``) maps every call to HTTP.
Nothing here imports the legacy server at module import: the catalog arrives
as a :class:`~apps.production_line.recipes.catalog_port.CatalogPort`.

Errors are :class:`RecipeApiError` with stable codes; the router turns them
into ``HTTPException(status, detail={codes, messages, details?, ...})``:

- 404 PART_NOT_FOUND / RECIPE_NOT_FOUND / FLAT_NOT_FOUND;
- 409 SKU_COLLISION, DRAFT_REVISION_CHANGED, RELEASE_DRAFT_REVISION_CHANGED,
  CONCURRENT_RELEASE, FLAT_REVISION_CHANGED;
- 413 FLAT_TOO_LARGE; 502/504 flat tool failures;
- 422 everything else (release gate codes, SKU_INVALID, validation,
  CONFIRMATION_NOT_ALLOWED, FLAT_REVISION_REQUIRED).

Confirmation policy: over HTTP a release (and a saved release draft) is
always OPERATOR_CONFIRMED. Any other ``confirmation`` (DEV_AUTO_CONFIRMED) is
refused with 422 CONFIRMATION_NOT_ALLOWED before anything is stored or
compiled, because a DEV recipe would release an unreviewed flat proposal as
AUTO_PROPOSED and bypass FLAT_BINDING_UNCONFIRMED. E4 builds its DEV recipes
through the library; :attr:`RecipeContext.allow_dev_auto_confirm` exists only
for an explicit dev wiring (the shipped router never sets it).

Part creation reserves the canonical SKU first
(:meth:`~apps.production_line.recipes.store.RecipeStore.reserve_sku`), checks
the catalog, creates the catalog part and then binds the reservation, so two
concurrent creators never both get a catalog row; a failed catalog insert
releases the reservation.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from apps.production_line.recipes.flat_intake import FlatIntakeError, FlatIntakeService
from apps.production_line.recipes.store import RecipeStore, RecipeStoreError, utc_now

__all__ = ["RecipeApiError", "RecipeContext", "RecipeService"]

_LOG = logging.getLogger(__name__)

_STATUS = {
    "PART_NOT_FOUND": 404,
    "RECIPE_NOT_FOUND": 404,
    "FLAT_NOT_FOUND": 404,
    "SKU_COLLISION": 409,
    "DRAFT_REVISION_CHANGED": 409,
    "RELEASE_DRAFT_REVISION_CHANGED": 409,
    "CONCURRENT_RELEASE": 409,
    "FLAT_REVISION_CHANGED": 409,
    "PART_EXISTS": 409,
}


class RecipeApiError(Exception):
    """An API refusal: ``status``, sorted ``codes`` and a JSON ``detail`` body."""

    def __init__(self, status: int, codes: Any, messages: Mapping[str, str] | None = None,
                 extra: Mapping[str, Any] | None = None) -> None:
        self.status = int(status)
        self.codes = tuple(sorted(set(codes)))
        self.messages = {c: (messages or {}).get(c, c) for c in self.codes}
        self.extra = dict(extra or {})
        super().__init__(", ".join(self.codes))

    @classmethod
    def of(cls, code: str, message: str = "", **extra: Any) -> "RecipeApiError":
        return cls(_STATUS.get(code, 422), [code], {code: message or code}, extra)

    def detail(self) -> dict[str, Any]:
        return {"codes": list(self.codes), "messages": self.messages, **self.extra}


@dataclass
class RecipeContext:
    """Everything a request needs (built by the router's dependencies)."""

    catalog: Any
    store: RecipeStore
    data_root: Path
    line_store: Any = None
    compiler: Any = None
    flat_tool: Any = None
    clock: Callable[[], datetime] = utc_now
    #: Dev-only escape hatch: allow ``confirmation=DEV_AUTO_CONFIRMED`` over HTTP. Never set in production.
    allow_dev_auto_confirm: bool = False


def _part_summary(part: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("id", "part_number", "part_name", "customer", "revision", "material", "cad_file_hash", "cad_uploaded_at",
            "drawing_file_hash", "drawing_uploaded_at", "bend_count", "created_at", "updated_at")
    out = {k: part.get(k) for k in keys}
    out["has_cad"] = bool(part.get("cad_file_path"))
    out["cad_format"] = Path(str(part.get("cad_file_path") or "")).suffix.lower().lstrip(".") or None
    out["has_drawing"] = bool(part.get("drawing_file_path"))
    return out


def _flat_doc(record: Any) -> dict[str, Any] | None:
    if record is None:
        return None
    doc = record.to_dict()
    proposal = doc.get("proposal") or {}
    inner = proposal.get("proposal") or {}
    doc["review"] = {
        "needs_operator_review": inner.get("needs_operator_review"),
        "ambiguous": (inner.get("fit_choice") or {}).get("ambiguous"),
        "mirror_determined": (inner.get("fit_choice") or {}).get("mirror_determined"),
        "alternative_maps": (inner.get("fit_choice") or {}).get("alternative_maps"),
        "k_clamped": inner.get("k_clamped"),
        "confidence": inner.get("confidence"),
    } if inner else None
    return doc


class RecipeService:
    """Request-scoped operations (module docstring)."""

    def __init__(self, ctx: RecipeContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------ parts
    def _require_part(self, part_id: str) -> dict[str, Any]:
        part = self.ctx.catalog.get_part(part_id)
        if part is None:
            raise RecipeApiError.of("PART_NOT_FOUND", f"part {part_id} not found")
        return part

    def list_parts(self, search: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Parts with a cheap onboarding summary (no file hashing, no compile).

        ``onboarding_state`` here is a summary from presence checks only;
        ``GET /parts/{id}/onboarding`` is authoritative.
        """
        from domains.production.recipe.canonical_sku import SkuError, canonical_sku

        limit = max(1, min(int(limit), 500))
        rows = []
        for part in self.ctx.catalog.list_parts(search=search, limit=limit, offset=max(0, int(offset))):
            try:
                sku: str | None = canonical_sku(str(part.get("part_number") or ""))
            except SkuError:
                sku = None
            active = self.ctx.store.get_active_for_part(str(part["id"]))
            draft = self.ctx.catalog.get_vendor_setup_draft(str(part["id"]))
            release_draft = self.ctx.store.get_release_draft(str(part["id"]))
            if active is not None:
                recipe = active.recipe()
                stale = (part.get("cad_file_hash") != recipe.cad.sha256
                         or part.get("drawing_file_hash") != recipe.drawing.sha256
                         or draft is None or draft.get("draft_revision") != recipe.source_draft.draft_revision)
                state = "RELEASED_STALE" if stale else "RELEASED"
            elif not part.get("cad_file_path") or sku is None:
                state = "NEEDS_CAD"
            elif not part.get("drawing_file_path"):
                state = "NEEDS_DRAWING"
            elif draft is None or draft.get("is_stale") or not draft.get("operator_sequence_confirmed"):
                state = "NEEDS_BEND_ORDER"
            elif release_draft is None or not (release_draft.request or {}).get("default_tolerances"):
                state = "NEEDS_TOLERANCES"
            else:
                state = "READY_TO_RELEASE"
            rows.append({
                "part": _part_summary(part),
                "sku": sku,
                "onboarding_state": state,
                "state_source": "SUMMARY",
                "active_recipe": None if active is None else {"recipe_id": active.recipe_id, "version": active.version,
                                                              "recipe_sha256": active.recipe_sha256},
            })  # fmt: skip
        return {"parts": rows, "limit": limit, "offset": max(0, int(offset))}

    def create_part(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Create a catalog part and register its canonical SKU (SKU_INVALID / SKU_COLLISION)."""
        from domains.production.recipe.canonical_sku import SkuError, canonical_sku

        raw = str(fields.get("part_number") or "")
        try:
            sku = canonical_sku(raw)
        except SkuError as exc:
            raise RecipeApiError.of("SKU_INVALID", str(exc)) from None
        try:
            token = self.ctx.store.reserve_sku(sku, raw.strip())
        except RecipeStoreError as exc:
            raise RecipeApiError.of(exc.code, f"SKU {sku} is already registered or being created") from None
        try:
            for other in self.ctx.catalog.list_parts(limit=100000):
                try:
                    same = canonical_sku(str(other.get("part_number") or "")) == sku
                except SkuError:
                    continue
                if same:
                    raise RecipeApiError.of(
                        "SKU_COLLISION", f"catalog part {other.get('part_number')!r} has the same canonical SKU {sku}"
                    )
            payload = {k: fields.get(k) for k in ("part_name", "customer", "revision", "material", "notes")
                       if fields.get(k) is not None}
            part = self.ctx.catalog.create_part(part_number=raw.strip(), **payload)
        except BaseException:
            self.ctx.store.release_sku_reservation(sku, token)
            raise
        try:
            self.ctx.store.bind_sku_reservation(sku, token, str(part["id"]))
        except RecipeStoreError as exc:
            # The reservation expired and was taken (a creator stalled > TTL): remove our catalog row.
            _LOG.error("SKU %s reservation lost while creating part %s: %s", sku, part.get("id"), exc.message)
            self._discard_catalog_part(str(part["id"]))
            raise RecipeApiError.of("SKU_COLLISION", exc.message) from None
        return {"part": _part_summary(part), "sku": sku}

    def _discard_catalog_part(self, part_id: str) -> None:
        delete = getattr(self.ctx.catalog, "delete_part", None)
        if delete is None:
            _LOG.error("catalog port cannot delete the orphan part %s; remove it by hand", part_id)
            return
        try:
            delete(part_id)
        except Exception:
            _LOG.exception("could not delete the orphan catalog part %s", part_id)

    def onboarding(self, part_id: str) -> dict[str, Any]:
        from apps.production_line.recipes.onboarding import checklist

        doc = checklist(part_id, catalog=self.ctx.catalog, recipe_store=self.ctx.store, data_root=self.ctx.data_root,
                        clock=self.ctx.clock)
        if doc is None:
            raise RecipeApiError.of("PART_NOT_FOUND", f"part {part_id} not found")
        doc["part"] = _part_summary(doc["part"])
        doc["flat"] = _flat_doc(self.ctx.store.get_part_flat(part_id))
        return doc

    # ------------------------------------------------------------------ release draft
    def get_release_draft(self, part_id: str) -> dict[str, Any]:
        self._require_part(part_id)
        record = self.ctx.store.get_release_draft(part_id)
        if record is None:
            return {"part_id": part_id, "revision": 0, "request": None, "updated_at": None, "updated_by": None}
        return {"part_id": part_id, "revision": record.revision, "request": record.request,
                "updated_at": record.updated_at, "updated_by": record.updated_by}

    def put_release_draft(self, part_id: str, body: Mapping[str, Any], principal: Any) -> dict[str, Any]:
        from pydantic import ValidationError

        from contracts.production.part_recipe import ReleaseRequestV1

        self._require_part(part_id)
        self._require_operator_confirmation(body)
        try:
            request = ReleaseRequestV1.model_validate(dict(body))
        except ValidationError as exc:
            from apps.production_line.recipes.release_service import _validation_codes

            codes = _validation_codes(exc)
            raise RecipeApiError(422, codes.keys(), {c: "; ".join(v) for c, v in codes.items()}) from None
        expected = request.expected_release_draft_revision or 0
        stored = request.model_dump(mode="json", exclude={"expected_release_draft_revision"})
        try:
            self.ctx.store.put_release_draft(part_id, stored, expected, updated_by=principal)
        except RecipeStoreError as exc:
            raise RecipeApiError.of(exc.code, exc.message) from None
        return self.get_release_draft(part_id)

    # ------------------------------------------------------------------ release
    def _require_operator_confirmation(self, body: Mapping[str, Any]) -> None:
        """422 CONFIRMATION_NOT_ALLOWED for any confirmation but OPERATOR_CONFIRMED (module docstring)."""
        confirmation = body.get("confirmation", "OPERATOR_CONFIRMED") if isinstance(body, Mapping) else None
        if confirmation == "OPERATOR_CONFIRMED":
            return
        if confirmation == "DEV_AUTO_CONFIRMED" and self.ctx.allow_dev_auto_confirm:
            _LOG.warning("DEV_AUTO_CONFIRMED accepted over HTTP (allow_dev_auto_confirm is set)")
            return
        from apps.production_line.recipes.release_service import CODE_MESSAGES

        raise RecipeApiError(422, ["CONFIRMATION_NOT_ALLOWED"],
                             {"CONFIRMATION_NOT_ALLOWED": CODE_MESSAGES["CONFIRMATION_NOT_ALLOWED"]},
                             {"details": {"CONFIRMATION_NOT_ALLOWED": [f"confirmation {confirmation!r} is not accepted here"]}})

    def release(self, part_id: str, body: Mapping[str, Any], principal: Any) -> tuple[int, dict[str, Any]]:
        from apps.production_line.recipes.release_service import (
            ReleaseGateError,
            release,
        )

        self._require_operator_confirmation(body)
        if self.ctx.compiler is None:
            raise RecipeApiError(503, ["COMPILE_UNAVAILABLE"], {"COMPILE_UNAVAILABLE": "no compile launcher configured"})
        try:
            outcome = release(part_id, dict(body), principal, catalog=self.ctx.catalog, recipe_store=self.ctx.store,
                              data_root=self.ctx.data_root, compiler=self.ctx.compiler, line_store=self.ctx.line_store,
                              clock=self.ctx.clock)
        except ReleaseGateError as exc:
            detail = exc.to_detail()
            raise RecipeApiError(exc.http_status, exc.codes, detail["messages"],
                                 {k: v for k, v in detail.items() if k not in ("codes", "messages")}) from None
        recipe = outcome.recipe
        doc = {
            "outcome": outcome.outcome,
            "recipe_id": recipe.recipe_id,
            "version": recipe.recipe_version,
            "supersedes_version": recipe.supersedes_version,
            "recipe_sha256": recipe.recipe_sha256,
            "content_sha256": recipe.content_sha256,
            "recipe": recipe.model_dump(mode="json", by_alias=True),
            "compiled": None if outcome.compiled is None else outcome.compiled.to_dict(),
            "compile": outcome.compile_report,
            "warnings": outcome.warnings,
        }
        return (201 if outcome.outcome == "released" else 200), doc

    # ------------------------------------------------------------------ recipes
    def _stored(self, recipe_id: str) -> Any:
        stored = self.ctx.store.get(recipe_id)
        if stored is None:
            raise RecipeApiError.of("RECIPE_NOT_FOUND", f"recipe {recipe_id} not found")
        return stored

    def _recipe_doc(self, stored: Any) -> dict[str, Any]:
        compiled = self.ctx.store.get_compiled(stored.recipe_sha256)
        return {
            **stored.summary(),
            "recipe": stored.recipe().model_dump(mode="json", by_alias=True),
            "compiled": None if compiled is None else compiled.to_dict(),
            "benchmark_reference": self._benchmark_summary(stored.recipe_id),
        }

    def _benchmark_summary(self, recipe_id: str) -> dict[str, Any] | None:
        ref = self.ctx.store.get_benchmark_reference(recipe_id)
        if ref is None:
            return None
        return {k: ref[k] for k in ("status", "registered_at", "compiled_part_sha256", "algo_version")}

    def recipe_by_sku(self, raw_sku: str) -> dict[str, Any]:
        from domains.production.recipe.canonical_sku import SkuError, canonical_sku

        try:
            sku = canonical_sku(raw_sku)
        except SkuError as exc:
            raise RecipeApiError.of("SKU_INVALID", str(exc)) from None
        active = self.ctx.store.get_active(sku)
        if active is None:
            raise RecipeApiError.of("RECIPE_NOT_FOUND", f"no active recipe for {sku}")
        doc = self._recipe_doc(active)
        doc["versions"] = [v.summary() for v in self.ctx.store.list_versions(sku)]
        return doc

    def recipe(self, recipe_id: str) -> dict[str, Any]:
        return self._recipe_doc(self._stored(recipe_id))

    def expectations(self, recipe_id: str) -> dict[str, Any]:
        stored = self._stored(recipe_id)
        compiled = self.ctx.store.get_compiled(stored.recipe_sha256)
        row = self.ctx.store.get_expectations(stored.recipe_sha256,
                                              compiled.compiler_version if compiled is not None else None)
        doc: dict[str, Any] = {"recipe_id": recipe_id, "status": "PENDING", "algo_version": None,
                               "expectations": None, "job_id": None}
        if row is not None:
            doc.update(row)
        doc["recompute_job"] = self._open_compile_job(stored)
        return doc

    def _open_compile_job(self, stored: Any) -> dict[str, Any] | None:
        if self.ctx.line_store is None:
            return None
        try:
            jobs = self.ctx.line_store.list_jobs(statuses=("queued", "running"), sku=stored.sku, kind="recipe_compile")
        except Exception:
            _LOG.exception("could not list compile jobs")
            return None
        for job in jobs:
            if job.get("recipe_sha256") == stored.recipe_sha256:
                return {"job_id": job["job_id"], "execution_status": job["execution_status"], "stage": job["stage"]}
        return None

    def recompute(self, recipe_id: str, principal: Any) -> dict[str, Any]:
        """Enqueue a ``recipe_compile`` job (the runner recompiles; the resolver's sink installs it)."""
        stored = self._stored(recipe_id)
        if stored.status != "released":
            raise RecipeApiError.of("RECIPE_RETIRED", f"{recipe_id} is retired")
        if self.ctx.line_store is None:
            raise RecipeApiError(503, ["LINE_STORE_UNAVAILABLE"], {"LINE_STORE_UNAVAILABLE": "no line store"})
        existing = self._open_compile_job(stored)
        if existing is not None:
            return {"job_id": existing["job_id"], "created": False}
        job_id, created = self.ctx.line_store.enqueue_job(
            "recipe_compile",
            sku=stored.sku,
            recipe_id=stored.recipe_id,
            recipe_version=stored.version,
            recipe_sha256=stored.recipe_sha256,
            principal=_audit(principal),
        )
        return {"job_id": job_id, "created": created}

    def retire(self, recipe_id: str, principal: Any) -> dict[str, Any]:
        from apps.production_line.recipes.release_service import (
            notify_release_listeners,
        )

        stored = self._stored(recipe_id)
        was = stored.status
        retired = self.ctx.store.retire(recipe_id, principal)
        if was == "released" and self.ctx.line_store is not None:
            try:
                self.ctx.line_store.append_event("RECIPE_RETIRED", principal=_audit(principal),
                                                 payload={"recipe_id": recipe_id, "recipe_sha256": stored.recipe_sha256})
            except Exception:
                _LOG.exception("could not append RECIPE_RETIRED for %s", recipe_id)
        notify_release_listeners(retired)
        doc = retired.summary()
        active = self.ctx.store.get_active(retired.sku)
        doc["active_recipe_id"] = None if active is None else active.recipe_id
        return doc

    # ------------------------------------------------------------------ flats
    def _flats(self) -> FlatIntakeService:
        if self.ctx.flat_tool is None:
            raise RecipeApiError(503, ["FLAT_TOOL_UNAVAILABLE"], {"FLAT_TOOL_UNAVAILABLE": "no flat tool configured"})
        return FlatIntakeService(catalog=self.ctx.catalog, store=self.ctx.store, data_root=self.ctx.data_root,
                                 tool=self.ctx.flat_tool, clock=self.ctx.clock)

    @staticmethod
    def _flat_error(exc: FlatIntakeError) -> RecipeApiError:
        return RecipeApiError(exc.http_status, exc.codes, {c: exc.message or c for c in exc.codes}, exc.extra)

    def get_flat(self, part_id: str) -> dict[str, Any]:
        self._require_part(part_id)
        record = self.ctx.store.get_part_flat(part_id)
        if record is None:
            raise RecipeApiError.of("FLAT_NOT_FOUND", "the part has no flat")
        return _flat_doc(record) or {}

    def upload_flat(self, part_id: str, stream: IO[bytes], filename: str, principal: Any) -> dict[str, Any]:
        try:
            record = self._flats().ingest_stream(part_id, stream, filename, principal)
        except FlatIntakeError as exc:
            raise self._flat_error(exc) from None
        return _flat_doc(record) or {}

    def propose_flat(self, part_id: str, expected_revision: int | None) -> dict[str, Any]:
        try:
            record = self._flats().propose(part_id, expected_revision)
        except FlatIntakeError as exc:
            raise self._flat_error(exc) from None
        return _flat_doc(record) or {}

    def set_flat_binding(self, part_id: str, body: Mapping[str, Any], principal: Any) -> dict[str, Any]:
        expected = body.get("expected_revision")
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise RecipeApiError.of("FLAT_BINDING_INVALID", "expected_revision (integer) is required")
        bend_line_map = body.get("bend_line_map")
        if bend_line_map is not None and (not isinstance(bend_line_map, Mapping)
                                          or not all(isinstance(k, str) and isinstance(v, str)
                                                     for k, v in bend_line_map.items())):
            raise RecipeApiError.of("FLAT_BINDING_INVALID", "bend_line_map maps flat line ids to bend_uids")
        mirrored = body.get("mirrored")
        if mirrored is not None and not isinstance(mirrored, bool):
            raise RecipeApiError.of("FLAT_BINDING_INVALID", "mirrored must be a boolean")
        try:
            record = self._flats().set_binding(part_id, expected_revision=expected, state=str(body.get("state")),
                                               principal=principal, bend_line_map=bend_line_map, mirrored=mirrored)
        except FlatIntakeError as exc:
            raise self._flat_error(exc) from None
        return _flat_doc(record) or {}

    def delete_flat(self, part_id: str, expected_revision: int | None) -> dict[str, Any]:
        try:
            deleted = self._flats().delete(part_id, expected_revision)
        except FlatIntakeError as exc:
            raise self._flat_error(exc) from None
        if not deleted:
            raise RecipeApiError.of("FLAT_NOT_FOUND", "the part has no flat")
        return {"part_id": part_id, "deleted": True}


def _audit(principal: Any) -> Any:
    if principal is None:
        return None
    if hasattr(principal, "audit") and callable(principal.audit):
        return principal.audit()
    if isinstance(principal, Mapping):
        return {"kind": principal.get("kind", "user"), "id": str(principal.get("user_id", principal.get("id"))),
                "role": principal.get("role")}
    return None
