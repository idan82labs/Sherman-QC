"""The narrow port between the recipe back end (C05) and the legacy part catalog.

The release gate, the onboarding checklist and the flat intake read the
catalog only through :class:`CatalogPort`. The production implementation,
:class:`LegacyCatalogPort`, wraps ``backend/part_catalog.py``
(``PartCatalogManager``) and the Vendor Setup draft validator in
``apps/api/services/vendor_setup_service.py``. Both are imported lazily inside
methods, never at module import, so importing this module (or the router that
uses it) never loads the legacy server (``_server_legacy_impl``).

Shapes (plain dicts, JSON-ready):

- part: ``Part.to_dict()`` (``id``, ``part_number``, ``part_name``,
  ``revision``, ``material``, ``cad_file_path``, ``cad_file_hash``,
  ``cad_uploaded_at``, ``drawing_file_path``, ``drawing_file_hash``, ...).
- bend spec row: ``{id, part_id, bend_id, target_angle, target_radius,
  tolerance_angle, tolerance_radius, bend_line_start, bend_line_end, notes}``
  with the line endpoints as the catalog's JSON strings (the shape
  ``domains.production.recipe.builder`` expects).
- draft: exactly ``PartCatalogManager.get_vendor_setup_draft`` (including
  ``draft_revision`` and ``is_stale``), or None.

:meth:`CatalogPort.validate_vendor_draft` runs the same checks as
``save_vendor_setup`` (``vendor_setup_service.py:277-290``): it builds the
``VendorSetupDraft`` and CAD-candidate ``BendMark`` objects and calls
``_validate_draft``. An ``HTTPException(422, msg)`` becomes a
:class:`DraftValidationError` carrying the message; :func:`codes_for_draft_message`
maps known messages to release codes (SEQUENCE_NOT_CONFIRMED, DRAFT_STALE),
everything else is DRAFT_INVALID.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DraftValidationError",
    "CatalogPort",
    "LegacyCatalogPort",
    "CatalogManagerPort",
    "codes_for_draft_message",
]

#: ``_validate_draft`` messages that have a precise release code.
_MESSAGE_CODES: dict[str, str] = {
    "Operator must confirm bending order": "SEQUENCE_NOT_CONFIRMED",
    "Every CAD candidate must remain represented": "DRAFT_STALE",
    "Unknown CAD candidate feature ID": "DRAFT_STALE",
    "CAD candidate geometry is immutable; add a manual correction": "DRAFT_STALE",
}


def codes_for_draft_message(message: str) -> str:
    """Release code for a Vendor Setup validation message (default DRAFT_INVALID)."""
    return _MESSAGE_CODES.get(str(message).strip(), "DRAFT_INVALID")


class DraftValidationError(ValueError):
    """The Vendor Setup validator refused the draft; ``messages`` are its texts."""

    def __init__(self, messages: Sequence[str]) -> None:
        self.messages = tuple(str(m) for m in messages) or ("draft invalid",)
        super().__init__("; ".join(self.messages))

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(sorted({codes_for_draft_message(m) for m in self.messages}))


@runtime_checkable
class CatalogPort(Protocol):
    """What C05 needs from the part catalog (module docstring)."""

    def get_part(self, part_id: str) -> dict[str, Any] | None: ...

    def list_parts(self, search: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]: ...

    def get_bend_specs(self, part_id: str) -> list[dict[str, Any]]: ...

    def get_vendor_setup_draft(self, part_id: str) -> dict[str, Any] | None: ...

    def candidate_generation(self, part_id: str) -> str: ...

    def create_part(self, **fields: Any) -> dict[str, Any]: ...

    def validate_vendor_draft(self, part_id: str, draft: Mapping[str, Any]) -> None:
        """Raise :class:`DraftValidationError` when the draft fails Vendor Setup validation."""
        ...

    def resolve_file(self, stored_path: str) -> Path:
        """Absolute path of a catalog file path (CAD / drawing) as stored in the part row."""
        ...


def _spec_dict(spec: Any) -> dict[str, Any]:
    return {
        "id": spec.id,
        "part_id": spec.part_id,
        "bend_id": spec.bend_id,
        "target_angle": spec.target_angle,
        "target_radius": spec.target_radius,
        "tolerance_angle": spec.tolerance_angle,
        "tolerance_radius": spec.tolerance_radius,
        "bend_line_start": spec.bend_line_start,
        "bend_line_end": spec.bend_line_end,
        "notes": getattr(spec, "notes", None),
    }


#: Fields ``create_part`` accepts (``PartCatalogManager.create_part`` keywords).
_CREATE_FIELDS = frozenset({"part_number", "part_name", "customer", "revision", "material", "notes"})


class CatalogManagerPort:
    """:class:`CatalogPort` over a given ``PartCatalogManager`` instance.

    ``catalog_root`` resolves relative stored file paths (the legacy server
    stores absolute paths; relative ones are taken relative to it).
    ``vendor_setup_loader`` returns the Vendor Setup service module; the
    default imports ``apps.api.services.vendor_setup_service`` lazily.
    """

    def __init__(
        self,
        manager: Any,
        *,
        catalog_root: Path | None = None,
        vendor_setup_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._manager = manager
        self._root = Path(catalog_root) if catalog_root is not None else Path.cwd()
        self._vendor_setup_loader = vendor_setup_loader

    def _mgr(self) -> Any:
        """The ``PartCatalogManager`` (overridden for lazy resolution)."""
        return self._manager

    @property
    def manager(self) -> Any:
        return self._mgr()

    def get_part(self, part_id: str) -> dict[str, Any] | None:
        part = self._mgr().get_part(part_id)
        return None if part is None else dict(part.to_dict())

    def list_parts(self, search: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return [dict(p.to_dict()) for p in self._mgr().list_parts(search=search, limit=limit, offset=offset)]

    def get_bend_specs(self, part_id: str) -> list[dict[str, Any]]:
        return [_spec_dict(s) for s in self._mgr().get_bend_specs_for_part(part_id)]

    def get_vendor_setup_draft(self, part_id: str) -> dict[str, Any] | None:
        draft = self._mgr().get_vendor_setup_draft(part_id)
        return None if draft is None else dict(draft)

    def candidate_generation(self, part_id: str) -> str:
        return str(self._mgr().candidate_generation(part_id))

    def create_part(self, **fields: Any) -> dict[str, Any]:
        unknown = set(fields) - _CREATE_FIELDS
        if unknown:
            raise ValueError(f"unknown part fields: {sorted(unknown)}")
        part = self._mgr().create_part(**fields)
        return dict(part.to_dict())

    def delete_part(self, part_id: str) -> bool:
        """Remove a part this API just created and could not register (lost SKU reservation)."""
        return bool(self._mgr().delete_part(part_id))

    def _vendor_setup(self) -> Any:
        if self._vendor_setup_loader is not None:
            return self._vendor_setup_loader()
        import importlib

        return importlib.import_module("apps.api.services.vendor_setup_service")

    def validate_vendor_draft(self, part_id: str, draft: Mapping[str, Any]) -> None:
        from pydantic import ValidationError

        service = self._vendor_setup()
        payload = {k: v for k, v in dict(draft).items() if k not in ("updated_at", "draft_revision", "is_stale")}
        try:
            model = service.VendorSetupDraft.model_validate(payload)
        except ValidationError as exc:
            raise DraftValidationError([f"draft does not validate: {exc.errors()[0].get('msg', '')}"]) from None
        candidates = {
            s.id: service.BendMark(
                feature_id=s.id,
                label=s.bend_id,
                source="cad_candidate",
                line_start=service._line(s.bend_line_start),
                line_end=service._line(s.bend_line_end),
                target_angle=s.target_angle,
                target_radius=s.target_radius,
            )
            for s in self._mgr().get_bend_specs_for_part(part_id)
        }
        try:
            service._validate_draft(model, candidates)
        except Exception as exc:  # fastapi.HTTPException(422, msg); imported lazily by the service
            status = getattr(exc, "status_code", None)
            detail = getattr(exc, "detail", None)
            if status is None:
                raise
            raise DraftValidationError([str(detail)]) from None

    def resolve_file(self, stored_path: str) -> Path:
        path = Path(os.path.expanduser(str(stored_path)))
        return path if path.is_absolute() else (self._root / path)


class LegacyCatalogPort(CatalogManagerPort):
    """:class:`CatalogPort` over the legacy server's catalog singleton.

    ``legacy.get_catalog()`` is resolved on first use (``apps.api.legacy_context``
    imports the legacy server, so it is never imported at module import).
    Relative stored paths resolve against ``catalog_root`` (default: the
    current working directory, the source root the legacy server runs from).
    """

    def __init__(self, catalog_root: Path | None = None) -> None:
        super().__init__(None, catalog_root=catalog_root)

    def _mgr(self) -> Any:
        if self._manager is None:
            from apps.api.legacy_context import (
                legacy,  # lazy: imports the legacy server
            )

            self._manager = legacy.get_catalog()
        return self._manager
