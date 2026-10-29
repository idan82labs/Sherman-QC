"""Put an E2 synthetic case into a part catalog, as an operator would (tests, E2E, demos).

E2 (``domains.production.engine.synthetic``) writes, per case, a real STEP
BRep (``part.step``), its flat (``flat.dxf``), a drawing (``drawing.pdf``)
and ``recipe_draft.json`` with the CAD candidate rows and the bending order.
:func:`onboard_case` replays the onboarding steps on a
``PartCatalogManager``:

1. ``create_part`` with the requested part number (SKU), revision and name;
2. "upload" the STEP and the drawing: copied into ``upload_dir`` under
   opaque names, with their sha256 recorded on the part, as the legacy upload
   routes do (``_server_legacy_impl.py:3643-3700`` / ``:3815-3851``);
3. publish the CAD bend candidates (``add_bend_spec``), as the STEP upload's
   native extractor would;
4. save a Vendor Setup draft that confirms every candidate, the bending order
   and the מהלך groups (``save_vendor_setup_draft``).

It returns the part id, the saved draft and a ready ``ReleaseRequestV1``
payload (E2's shop-standard tolerances and per-bend NOT_INSPECTED
overrides, re-keyed to the catalog feature ids). Nothing here touches the
repository's ``data/`` or ``uploads/``: every path is the caller's.

:func:`generate_e2_case` builds one case with E2's library (no scan needed
by the recipe side, but E2 always writes one).
:func:`vendor_setup_loader` loads the Vendor Setup service with a stub
legacy context bound to the given catalog, so the release gate's draft
validator runs without importing the legacy server.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

__all__ = ["OnboardedPart", "generate_e2_case", "onboard_case", "vendor_setup_loader", "release_request"]


@dataclass(frozen=True)
class OnboardedPart:
    part_id: str
    part_number: str
    draft: dict[str, Any]
    #: E2 draft feature id -> catalog bend spec id.
    feature_map: dict[str, str]
    #: Catalog feature ids in bending order.
    sequence: list[str]
    case_dir: Path
    recipe_draft: dict[str, Any]


def generate_e2_case(out_dir: Path, case_id: str = "E2-DEV-SMHT-000", tier: str = "smoke") -> Path:
    """Generate one E2 DEV case into ``out_dir`` (library call, deterministic)."""
    from domains.production.engine.synthetic.cases import generate_case, plan_cases

    specs = [s for s in plan_cases(tier, "DEV") if s.case_id == case_id]
    if not specs:
        raise KeyError(f"no E2 case {case_id} in tier {tier}")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_case(specs[0], out_dir, ray_threads=1)
    case_dir = out_dir / case_id
    if not (case_dir / "part.step").is_file():
        raise FileNotFoundError(f"E2 did not write {case_dir / 'part.step'}")
    return case_dir


def _copy_upload(src: Path, upload_dir: Path) -> tuple[str, str]:
    import hashlib

    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}{src.suffix.lower()}"
    shutil.copyfile(src, dest)
    return str(dest), hashlib.sha256(dest.read_bytes()).hexdigest()


def onboard_case(
    catalog: Any,
    case_dir: Path,
    *,
    part_number: str,
    upload_dir: Path,
    revision: str = "A",
    part_name: str | None = None,
    with_drawing: bool = True,
    confirm_marks: bool = True,
    confirm_sequence: bool = True,
) -> OnboardedPart:
    """Onboard an E2 case into ``catalog`` (a ``PartCatalogManager``; module docstring)."""
    case_dir = Path(case_dir)
    doc = json.loads((case_dir / "recipe_draft.json").read_text(encoding="utf-8"))
    inputs = doc["builder_inputs"]
    part = catalog.create_part(part_number, part_name=part_name or doc.get("part_name"), revision=revision,
                               material=inputs["part"].get("material"))
    cad_path, cad_sha = _copy_upload(case_dir / "part.step", upload_dir)
    fields: dict[str, Any] = {"cad_file_path": cad_path, "cad_file_hash": cad_sha,
                              "cad_uploaded_at": datetime.now().isoformat()}
    if with_drawing:
        drawing_path, drawing_sha = _copy_upload(case_dir / "drawing.pdf", upload_dir)
        fields.update(drawing_file_path=drawing_path, drawing_file_hash=drawing_sha,
                      drawing_uploaded_at=datetime.now().isoformat())
    catalog.update_part(part.id, **fields)
    feature_map: dict[str, str] = {}
    for row in inputs["spec_rows"]:
        spec = catalog.add_bend_spec(
            part.id,
            row["bend_id"],
            row["target_angle"],
            target_radius=row["target_radius"],
            bend_line_start=json.loads(row["bend_line_start"]),
            bend_line_end=json.loads(row["bend_line_end"]),
        )
        feature_map[row["id"]] = spec.id
    e2_draft = inputs["draft"]
    current = catalog.get_part(part.id)
    marks = []
    for spec in catalog.get_bend_specs_for_part(part.id):
        marks.append({
            "feature_id": spec.id, "label": spec.bend_id, "source": "cad_candidate", "status": "active",
            "review_status": "confirmed" if confirm_marks else "pending", "issue": None,
            "corrects_feature_id": None, "note": "",
            "line_start": json.loads(spec.bend_line_start), "line_end": json.loads(spec.bend_line_end),
            "target_angle": spec.target_angle, "target_radius": spec.target_radius,
        })  # fmt: skip
    sequence = [feature_map[f] for f in e2_draft["sequence"]]
    moves = [[feature_map[f] for f in move] for move in e2_draft.get("machine_moves") or []]
    existing = catalog.get_vendor_setup_draft(part.id)
    draft = {
        "schema_version": 1,
        "cad_file_hash": current.cad_file_hash,
        "cad_uploaded_at": current.cad_uploaded_at,
        "part_revision": current.revision,
        "drawing_file_hash": current.drawing_file_hash,
        "candidate_generation": catalog.candidate_generation(part.id),
        "expected_draft_revision": existing["draft_revision"] if existing else 0,
        "marks": marks,
        "sequence": sequence,
        "machine_moves": moves,
        "machine_moves_explicit": bool(moves),
        "general_measurements": [],
        "operator_sequence_confirmed": bool(confirm_sequence),
        "completed_steps": 0,
        "matching_search": {"search_radius_mm": None, "search_angle_deg": None},
    }
    saved = catalog.save_vendor_setup_draft(part.id, draft)
    return OnboardedPart(
        part_id=part.id,
        part_number=part_number,
        draft=saved,
        feature_map=feature_map,
        sequence=sequence,
        case_dir=case_dir,
        recipe_draft=doc,
    )


def release_request(onboarded: OnboardedPart, *, inspection_stage: str = "FINAL", **overrides: Any) -> dict[str, Any]:
    """A ``ReleaseRequestV1`` payload for an onboarded case (E2 tolerances and per-bend overrides)."""
    release = onboarded.recipe_draft["builder_inputs"]["release"]
    bend_overrides = {
        onboarded.feature_map[fid]: value for fid, value in (release.get("bend_overrides") or {}).items()
    }
    payload: dict[str, Any] = {
        "expected_draft_revision": onboarded.draft["draft_revision"],
        "default_tolerances": release["default_tolerances"],
        "bend_overrides": bend_overrides,
        "thickness_mm": release.get("thickness_mm"),
        "inspection_stage": inspection_stage,
        "confirmation": "OPERATOR_CONFIRMED",
        "release_note": "synthetic onboarding",
    }
    payload.update(overrides)
    return payload


def vendor_setup_loader(catalog: Any) -> Callable[[], ModuleType]:
    """Loader for ``apps/api/services/vendor_setup_service.py`` with a stub legacy context.

    The module is executed under a private name with ``apps.api.legacy_context``
    replaced by a stub whose ``legacy.get_catalog()`` returns ``catalog``
    (the pattern of ``backend/tests/test_vendor_setup_service.py``); the real
    ``sys.modules`` entry is restored afterwards. The module is loaded once.
    """
    cache: dict[str, ModuleType] = {}

    def _load() -> ModuleType:
        if "m" in cache:
            return cache["m"]
        path = Path(__file__).resolve().parents[3] / "apps" / "api" / "services" / "vendor_setup_service.py"
        stub = ModuleType("apps.api.legacy_context")
        stub.legacy = SimpleNamespace(get_catalog=lambda: catalog)  # type: ignore[attr-defined]
        previous = sys.modules.get("apps.api.legacy_context")
        name = f"_c05_vendor_setup_service_{uuid.uuid4().hex[:8]}"
        sys.modules["apps.api.legacy_context"] = stub
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        finally:
            if previous is None:
                sys.modules.pop("apps.api.legacy_context", None)
            else:
                sys.modules["apps.api.legacy_context"] = previous
        cache["m"] = module
        return module

    return _load
