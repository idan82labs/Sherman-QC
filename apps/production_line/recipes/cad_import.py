"""SolidWorks STEP batch intake: validate converted STEP files and import them as draft catalog parts.

The STEP files come back from Sherman's SolidWorks conversion
(``research/zeroshot/zs_rust/station/sldprt_to_step/``): the VBA macro's
``manifest.csv``, the PowerShell batch's ``manifest.json`` / ``manifest.csv``,
or a bare folder (SolidWorks Task Scheduler "Export Files")::

    python -m apps.production_line.recipes.cad_import validate \\
        (--manifest M.json|M.csv | --step-dir DIR) [--timeout-s 300] [--no-isolate] --out VALIDATION.json
    python -m apps.production_line.recipes.cad_import import \\
        (--manifest ... | --step-dir DIR) --data-root LINE_DATA_ROOT --catalog-db CATALOG.sqlite \\
        --upload-dir UPLOADS [--timeout-s 300] [--no-isolate] --out IMPORT.json

**validate** checks each STEP: it is present and non-empty, its sha256 matches the manifest's
``step_sha256`` (only the PowerShell manifest has one), and it is read with the compile's own
OCP reader (:func:`domains.production.engine.compile.brep.load_step`), which must find a single
solid. The STEP header length unit must be mm and the bounding box must be plausible. The
check then runs the CAD stage of the onboarding compile (flange graph + rooted part model, the
code ``compile_part`` runs before reconciling marks; :mod:`domains.production.engine.compile.api`).
This records ``FLANGE_TREE_OK`` or the compile's blocker code (``FLANGE_GRAPH_NOT_A_TREE``,
``DATUM_PANEL_UNDETERMINED``, ``THICKNESS_INCONSISTENT``, ...). It also records the sheet
thickness and checks that every panel and bend agrees with it within the compile's own thickness
tolerance. A full ``compile_part`` needs a released recipe (marks, tolerances), so it is not
run here. Each file is checked in a child process with a wall-clock timeout (``--no-isolate``
runs in process).

**import** registers each VALID STEP whose filename gives a SKU (:func:`part_sku_from_step_name`)
as a draft catalog part. It goes through the same seam as the setup screen:
:meth:`apps.api.services.production_recipe_service.RecipeService.create_part` reserves and
registers the canonical SKU in ``sku_registry``. The STEP is then attached with the catalog's
own CAD upload protocol (``begin_cad_upload`` / ``publish_cad_upload``, unanalyzed, the
``analyze_bends=False`` upload path). No recipe is released and no bend candidates are
published. Bend analysis, drawing, bend order, tolerances and release stay human steps on the
setup screen. Import is idempotent. If a part with this SKU already carries this STEP sha256,
nothing is created. If a part with this SKU exists with another or no CAD, it is reported
(SKU_EXISTS) and left alone, unless that part was created by an earlier interrupted import (its
notes carry this module's marker), in which case the CAD is attached. A STEP already attached
to another part is reported (STEP_ON_OTHER_PART). Unparseable names are reported, never
guessed.

Nothing here reads scans. Every number is informational for the operator; none of it is an
inspection result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "IMPORT_MARKER",
    "VALIDATION_CODES",
    "load_manifest",
    "rows_from_step_dir",
    "part_sku_from_step_name",
    "step_length_unit",
    "validate_step",
    "validate_rows",
    "import_validated",
    "main",
]

#: Written into a created part's notes; lets a re-run finish an interrupted import.
IMPORT_MARKER = "cad_import:v1"
#: Manifest statuses that should have produced a STEP file.
_EXPORTED = frozenset({"OK", "SKIPPED_UP_TO_DATE"})
VALIDATION_CODES = frozenset({
    "CONVERSION_FAILED", "STEP_FILE_MISSING", "STEP_EMPTY", "STEP_SHA_MISMATCH", "STEP_UNITS_NOT_MM",
    "STEP_UNITS_UNKNOWN", "BBOX_IMPLAUSIBLE", "THICKNESS_NOT_CONSTANT", "VALIDATION_TIMEOUT",
    "VALIDATION_CRASHED",
})
#: Plausible part envelope (mm): largest extent, smallest extent.
_MAX_EXTENT_MM = 6000.0
_MIN_EXTENT_MM = 0.2
_MIN_LARGEST_EXTENT_MM = 2.0

_BASE = r"(?P<base>[0-9]{7,8}|[A-Z]{2,4}[0-9]{4,8})"
_REV = r"(?P<rev>[A-Z]{1,2}[0-9]{0,2}|[0-9]{2}[A-Z]?)"
_NAME_RE = re.compile(rf"^{_BASE}[_-]{_REV}(?:[_-](?P<tag>RFQ))?$")
_BASE_ONLY_RE = re.compile(rf"^{_BASE}$")
_SINGLE_DIGIT_REV_RE = re.compile(rf"^{_BASE}[_-][0-9]$")


# ---------------------------------------------------------------------------- small helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    item = getattr(obj, "item", None)
    return _json_safe(item()) if callable(item) else str(obj)


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(dict(doc)), indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------- manifests
def _norm_row(raw: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """One manifest row in the common shape (PowerShell JSON/CSV or VBA CSV)."""
    out_rel = _clean(raw.get("output_rel")) or _clean(raw.get("output_name"))
    out_path = _clean(raw.get("output_path"))
    step_path = None
    if out_rel:
        step_path = root / out_rel.replace("\\", "/")
    elif out_path:
        step_path = root / Path(out_path.replace("\\", "/")).name
    sha = (_clean(raw.get("step_sha256")) or "").lower() or None
    return {
        "source_rel": _clean(raw.get("source_rel")) or _clean(raw.get("source_name")),
        "source_name": _clean(raw.get("source_name")),
        "source_size": _clean(raw.get("source_size")),
        "source_sha256_before": _clean(raw.get("source_sha256_before")),
        "source_sha256_after": _clean(raw.get("source_sha256_after")),
        "configuration": _clean(raw.get("configuration")),
        "manifest_status": (_clean(raw.get("status")) or "").upper() or None,
        "manifest_error": _clean(raw.get("error")),
        "sw_revision": _clean(raw.get("sw_revision")),
        "step_rel": None if step_path is None else str(step_path.relative_to(root)),
        "step_path": None if step_path is None else str(step_path),
        "manifest_step_sha256": sha,
    }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Rows of a ``manifest.json`` (``{"files": [...]}``) or ``manifest.csv``; STEP paths resolve
    against the manifest's folder. Raises ValueError for an unreadable manifest."""
    path = Path(path)
    root = path.parent
    if path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_rows = doc.get("files") if isinstance(doc, dict) else doc
        if not isinstance(raw_rows, list):
            raise ValueError(f"{path}: no 'files' list")
        if isinstance(doc, dict) and doc.get("dry_run"):
            raise ValueError(f"{path}: a dry-run manifest has no STEP files")
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as fh:
            raw_rows = list(csv.DictReader(fh))
        if raw_rows and "status" not in raw_rows[0]:
            raise ValueError(f"{path}: no 'status' column")
    else:
        raise ValueError(f"{path}: manifest must be .json or .csv")
    return [_norm_row(r, root) for r in raw_rows if isinstance(r, Mapping)]


def rows_from_step_dir(step_dir: str | Path) -> list[dict[str, Any]]:
    """Rows for a folder of STEP files without a manifest (no sha256 to compare)."""
    root = Path(step_dir)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".step", ".stp") and ".~tmp." not in p.name:
            rows.append(_norm_row({"output_rel": str(p.relative_to(root)), "status": "OK",
                                   "source_name": p.name}, root))
    return rows


# ---------------------------------------------------------------------------- SKU from a filename
def part_sku_from_step_name(filename: str) -> dict[str, Any]:
    """SKU of a converted part from its STEP (or SLDPRT) filename, never guessed.

    ``<base>_<rev>[_RFQ][__<configuration>].step`` with base 7-8 digits (or a 2-4 letter
    prefix + digits, e.g. MEC003744) and a revision (``C``, ``A01``, ``02``, ``01A``). The SKU
    is ``<base>_<rev>`` in canonical form (:func:`domains.production.recipe.canonical_sku.canonical_sku`),
    the same SKU-with-revision form the scan-name parser matches; the result is checked against
    :func:`domains.production.recipe.sku.parse_scan_filename` so a scan named like the CAD
    resolves to this SKU. Anything else is ``UNPARSEABLE`` with a reason.
    """
    from domains.production.recipe.canonical_sku import SkuError, canonical_sku
    from domains.production.recipe.sku import parse_scan_filename

    name = Path(str(filename).replace("\\", "/")).name
    stem = re.sub(r"\.(step|stp|sldprt)$", "", name, flags=re.IGNORECASE)
    configuration = None
    if "__" in stem:
        stem, configuration = stem.split("__", 1)
    text = stem.strip().upper()

    def bad(reason: str) -> dict[str, Any]:
        return {"status": "UNPARSEABLE", "sku": None, "base": None, "revision": None, "rfq": False,
                "configuration": configuration, "stem": stem, "reason": reason}

    if not text:
        return bad("EMPTY_NAME")
    m = _NAME_RE.match(text)
    if m is None:
        if _BASE_ONLY_RE.match(text):
            return bad("NO_REVISION")
        if _SINGLE_DIGIT_REV_RE.match(text):
            return bad("REVISION_SINGLE_DIGIT")  # a revision or a copy counter: not decidable from the name
        return bad("NAME_NOT_BASE_REV")
    try:
        sku = canonical_sku(f"{m['base']}_{m['rev']}")
    except SkuError as exc:
        return bad(f"SKU_INVALID: {exc}")
    check = parse_scan_filename(f"{stem}.ply", [sku])
    if check.status != "OK" or check.sku != sku:
        return bad(f"SCAN_RULE_DISAGREES: {check.status} {check.reason}")
    return {"status": "OK", "sku": sku, "base": m["base"], "revision": m["rev"], "rfq": bool(m["tag"]),
            "configuration": configuration, "stem": stem, "reason": None}


# ---------------------------------------------------------------------------- STEP checks
_LENGTH_UNITS = (
    (re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'INCH'", re.IGNORECASE), "inch"),
    (re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'(FOOT|FEET)'", re.IGNORECASE), "foot"),
    (re.compile(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)"), "mm"),
    (re.compile(r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)"), "cm"),
    (re.compile(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)"), "m"),
)
_ENTITY_RE = re.compile(r"^\s*(?P<id>#\d+)\s*=\s*(?P<body>.*)$", re.DOTALL)
_GLOBAL_UNITS_RE = re.compile(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*\(\s*\(([^()]*)\)\s*\)")


def step_length_unit(path: str | Path) -> str | None:
    """The length unit declared in a STEP file: ``mm``, ``cm``, ``m``, ``inch``, ``foot``,
    ``mixed`` (several different) or None (none found). Reads the STEP text, not geometry."""
    text = Path(path).read_bytes().decode("latin-1")
    entities: dict[str, str] = {}
    for stmt in text.split(";"):
        m = _ENTITY_RE.match(stmt)
        if m:
            entities[m["id"]] = m["body"]

    def classify(body: str) -> str | None:
        for rx, unit in _LENGTH_UNITS:  # conversion-based units first: they also name their SI base
            if rx.search(body):
                return unit
        return None

    found = set()
    # the units the geometry context actually uses (a conversion unit refers to an SI unit entity too)
    for body in entities.values():
        ctx = _GLOBAL_UNITS_RE.search(body)
        if not ctx:
            continue
        for ref in re.findall(r"#\d+", ctx.group(1)):
            unit_body = entities.get(ref, "")
            if "LENGTH_UNIT" in unit_body:
                unit = classify(unit_body)
                if unit:
                    found.add(unit)
    if not found:  # no context found: every declared length unit
        for body in entities.values():
            if "LENGTH_UNIT" in body:
                unit = classify(body)
                if unit:
                    found.add(unit)
    if not found:
        return None
    return found.pop() if len(found) == 1 else "mixed"


def validate_step(step_path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Check one STEP file (module docstring). Returns a JSON-ready dict; never raises for bad input."""
    p = Path(step_path)
    res: dict[str, Any] = {"step_path": str(p), "codes": [], "warnings": [], "step_sha256": None, "step_size": None,
                           "sha_check": None, "units": None, "brep": None, "sheet_metal": None, "compile": None}
    codes: list[str] = res["codes"]
    if not p.is_file():
        codes.append("STEP_FILE_MISSING")
        return _verdict(res)
    res["step_size"] = p.stat().st_size
    if res["step_size"] == 0:
        codes.append("STEP_EMPTY")
        return _verdict(res)
    res["step_sha256"] = _sha256(p)
    if expected_sha256:
        res["sha_check"] = "MATCH" if expected_sha256.lower() == res["step_sha256"] else "MISMATCH"
        if res["sha_check"] == "MISMATCH":
            codes.append("STEP_SHA_MISMATCH")
            return _verdict(res)
    else:
        res["sha_check"] = "NO_MANIFEST_SHA"
    res["units"] = step_length_unit(p)

    from domains.production.engine.compile.brep import CadReadError, load_step
    from domains.production.engine.compile.config import CompileConfig
    from domains.production.engine.compile.flange_graph import GraphError, build_flange_graph
    from domains.production.engine.compile.part_model import build_part_model

    cfg = CompileConfig()
    try:
        brep = load_step(p, linear_deflection=cfg.mesh_linear_deflection_mm,
                         angular_deflection=cfg.mesh_angular_deflection_rad)
    except CadReadError as exc:
        codes.append(exc.reason)
        res["brep"] = {"error": str(exc)}
        return _verdict(res)
    if res["units"] is None:
        codes.append("STEP_UNITS_UNKNOWN")
    elif res["units"] != "mm":
        codes.append("STEP_UNITS_NOT_MM")
    extents = [float(v) for v in (brep.bbox_max - brep.bbox_min)]
    res["brep"] = {"solid_count": int(brep.solid_count), "volume_mm3": round(float(brep.volume), 3),
                   "bbox_extent_mm": [round(v, 4) for v in extents], "face_count": len(brep.faces),
                   "notes": list(brep.notes)}
    if brep.solid_count != 1:
        res["warnings"].append(f"{brep.solid_count} solids in the file; the compile reads the dominant one")
    if (max(extents) > _MAX_EXTENT_MM or min(extents) < _MIN_EXTENT_MM
            or max(extents) < _MIN_LARGEST_EXTENT_MM):
        codes.append("BBOX_IMPLAUSIBLE")
    try:
        model = build_part_model(build_flange_graph(brep, cfg))
    except GraphError as exc:
        codes.append(exc.code)
        res["compile"] = {"outcome": exc.code, "detail": exc.detail}
        return _verdict(res)
    except Exception as exc:  # geometry the compile does not handle: the compile maps it the same way
        codes.append("FLANGE_GRAPH_NOT_A_TREE")
        res["compile"] = {"outcome": "FLANGE_GRAPH_NOT_A_TREE", "detail": f"{type(exc).__name__}: {exc}"}
        return _verdict(res)
    t = float(model.thickness)
    tol = max(cfg.thickness_tol_mm, cfg.thickness_tol_rel * t)
    samples = [float(pn.graph.thickness) for pn in model.panels.values() if pn.graph.thickness is not None]
    samples += [float(b.thickness) for b in model.bends.values()]
    spread = max((abs(s - t) for s in samples), default=0.0)
    res["sheet_metal"] = {"thickness_mm": round(t, 4), "samples": len(samples), "max_deviation_mm": round(spread, 4),
                          "tolerance_mm": round(tol, 4), "constant": spread <= tol}
    if spread > tol:
        codes.append("THICKNESS_NOT_CONSTANT")
    res["compile"] = {"outcome": "FLANGE_TREE_OK", "panels": len(model.panels), "graph_bends": len(model.bends),
                      "unsupported_bends": sum(1 for b in model.bends.values() if b.unsupported),
                      "datum_panel_id": model.panels[model.datum].panel_id}
    return _verdict(res)


def _verdict(res: dict[str, Any]) -> dict[str, Any]:
    res["codes"] = sorted(set(res["codes"]))
    res["verdict"] = "VALID" if not res["codes"] else "INVALID"
    return res


def _validate_isolated(step_path: str, expected: str | None, timeout_s: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cad_import_") as td:
        out = Path(td) / "one.json"
        cmd = [sys.executable, "-m", "apps.production_line.recipes.cad_import", "_one", "--step", step_path,
               "--out", str(out)] + (["--expected-sha256", expected] if expected else [])
        env = dict(os.environ, OMP_NUM_THREADS="1")
        repo = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            proc = subprocess.run(cmd, env=env, cwd=repo, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return _verdict({"step_path": step_path, "codes": ["VALIDATION_TIMEOUT"], "warnings": [],
                             "detail": f"exceeded {timeout_s:.0f} s"})
        if proc.returncode != 0 or not out.is_file():
            return _verdict({"step_path": step_path, "codes": ["VALIDATION_CRASHED"], "warnings": [],
                             "detail": (proc.stderr or "")[-2000:]})
        return json.loads(out.read_text(encoding="utf-8"))


def validate_rows(rows: Sequence[Mapping[str, Any]], *, isolate: bool = True, timeout_s: float = 300.0) -> dict[str, Any]:
    """Validate manifest rows; returns ``{"rows": [...], "counts": {...}}``."""
    out = []
    for row in rows:
        rec = dict(row)
        status = rec.get("manifest_status")
        name = Path(rec.get("step_rel") or rec.get("source_name") or "").name
        rec["sku"] = part_sku_from_step_name(name) if name else part_sku_from_step_name("")
        if status not in _EXPORTED:
            rec["validation"] = _verdict({"codes": ["CONVERSION_FAILED"], "warnings": [],
                                          "detail": f"{status}: {rec.get('manifest_error') or ''}".strip()})
        elif not rec.get("step_path"):
            rec["validation"] = _verdict({"codes": ["STEP_FILE_MISSING"], "warnings": []})
        elif isolate:
            rec["validation"] = _validate_isolated(rec["step_path"], rec.get("manifest_step_sha256"), timeout_s)
        else:
            rec["validation"] = validate_step(rec["step_path"], rec.get("manifest_step_sha256"))
        out.append(rec)
    counts: dict[str, Any] = {"rows": len(out),
                              "valid": sum(1 for r in out if r["validation"]["verdict"] == "VALID"),
                              "sku_parsed": sum(1 for r in out if r["sku"]["status"] == "OK")}
    by_code: dict[str, int] = {}
    for r in out:
        for c in r["validation"]["codes"]:
            by_code[c] = by_code.get(c, 0) + 1
    counts["invalid_by_code"] = dict(sorted(by_code.items()))
    return {"rows": out, "counts": counts}


# ---------------------------------------------------------------------------- import
def _attach_cad(manager: Any, part_id: str, step_path: Path, sha: str, upload_dir: Path) -> str | None:
    """Copy the STEP under an opaque name and publish it unanalyzed (catalog CAD-upload protocol).
    Returns None on success, else an error text; never leaves a half-published CAD."""
    cad_dir = Path(upload_dir) / "parts_cad"
    cad_dir.mkdir(parents=True, exist_ok=True)
    dest = cad_dir / f"{uuid.uuid4().hex}{step_path.suffix.lower()}"
    shutil.copyfile(step_path, dest)
    published = False
    reservation = None
    try:
        if _sha256(dest) != sha:
            return "STEP changed while copying"
        reservation = manager.begin_cad_upload(part_id)
        published = bool(manager.publish_cad_upload(part_id, reservation, str(dest), sha, None, [], analyzed=False))
        return None if published else "CAD upload superseded (part changed meanwhile)"
    finally:
        if not published:
            if reservation is not None:
                try:
                    manager.cancel_cad_upload(part_id, reservation)
                except Exception:
                    pass
            dest.unlink(missing_ok=True)


def import_validated(report: Mapping[str, Any], *, ctx: Any, manager: Any, upload_dir: Path,
                     manifest_label: str = "") -> dict[str, Any]:
    """Import the VALID rows of a :func:`validate_rows` report (module docstring).

    ``ctx`` is a :class:`apps.api.services.production_recipe_service.RecipeContext` (catalog port +
    recipe store); ``manager`` the ``PartCatalogManager`` behind the port (CAD upload protocol).
    """
    from apps.api.services.production_recipe_service import RecipeApiError, RecipeService
    from domains.production.recipe.canonical_sku import SkuError, canonical_sku

    service = RecipeService(ctx)
    parts = ctx.catalog.list_parts(limit=1_000_000)
    by_sku: dict[str, dict[str, Any]] = {}
    by_cad: dict[str, dict[str, Any]] = {}
    for part in parts:
        try:
            by_sku[canonical_sku(str(part.get("part_number") or ""))] = part
        except SkuError:
            pass
        if part.get("cad_file_hash"):
            by_cad.setdefault(str(part["cad_file_hash"]), part)

    rows = list(report.get("rows") or [])
    shas_by_sku: dict[str, set[str]] = {}
    for r in rows:
        if r["validation"]["verdict"] == "VALID" and r["sku"]["status"] == "OK":
            shas_by_sku.setdefault(r["sku"]["sku"], set()).add(r["validation"]["step_sha256"])

    results = []
    for r in rows:
        v, s = r["validation"], r["sku"]
        res = {"step_rel": r.get("step_rel"), "source_name": r.get("source_name"), "sku": s.get("sku"),
               "step_sha256": v.get("step_sha256"), "part_id": None, "detail": None}
        results.append(res)
        if v["verdict"] != "VALID":
            res.update(outcome="NOT_VALID", detail=",".join(v["codes"]))
            continue
        if s["status"] != "OK":
            res.update(outcome="SKU_UNPARSEABLE", detail=s["reason"])
            continue
        sku, sha = s["sku"], v["step_sha256"]
        if len(shas_by_sku[sku]) > 1:
            res.update(outcome="DUPLICATE_SKU_IN_BATCH",
                       detail=f"{len(shas_by_sku[sku])} different STEP files name {sku} (configurations?); pick one")
            continue
        existing = by_sku.get(sku)
        if existing is not None:
            res["part_id"] = str(existing["id"])
            if existing.get("cad_file_hash") == sha:
                res.update(outcome="ALREADY_IMPORTED")
                continue
            if not existing.get("cad_file_path") and IMPORT_MARKER in str(existing.get("notes") or ""):
                err = _attach_cad(manager, str(existing["id"]), Path(v["step_path"]), sha, upload_dir)
                res.update(outcome="CAD_ATTACHED_RETRY" if err is None else "CAD_PUBLISH_FAILED", detail=err)
                if err is None:
                    existing["cad_file_hash"] = sha
                    by_cad[sha] = existing
                continue
            res.update(outcome="SKU_EXISTS", detail="a catalog part with this SKU already has "
                       + ("another CAD" if existing.get("cad_file_path") else "no CAD and was not created by this import")
                       + "; left unchanged for the operator")
            continue
        other = by_cad.get(sha)
        if other is not None:
            res.update(outcome="STEP_ON_OTHER_PART", part_id=str(other["id"]),
                       detail=f"this STEP is already the CAD of part {other.get('part_number')!r}")
            continue
        notes = (f"{IMPORT_MARKER} step_sha256={sha} source={r.get('source_name') or ''} "
                 f"configuration={r.get('configuration') or ''} manifest={manifest_label}").strip()
        try:
            created = service.create_part({"part_number": sku, "part_name": s["stem"], "revision": s["revision"],
                                           "notes": notes})
        except RecipeApiError as exc:
            res.update(outcome="CREATE_FAILED", detail=",".join(exc.codes))
            continue
        part = dict(created["part"])
        part.update(notes=notes, part_number=sku)
        res["part_id"] = str(part["id"])
        by_sku[sku] = part
        err = _attach_cad(manager, res["part_id"], Path(v["step_path"]), sha, upload_dir)
        if err is None:
            part["cad_file_hash"] = sha
            part["cad_file_path"] = "attached"
            by_cad[sha] = part
            res.update(outcome="IMPORTED")
        else:
            res.update(outcome="CAD_PUBLISH_FAILED", detail=err + " (part created; a re-run attaches the CAD)")
    counts: dict[str, int] = {}
    for res in results:
        counts[res["outcome"]] = counts.get(res["outcome"], 0) + 1
    return {"results": results, "counts": dict(sorted(counts.items()))}


# ---------------------------------------------------------------------------- CLI
def _rows_from_args(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if args.manifest:
        return load_manifest(args.manifest), str(Path(args.manifest).name)
    return rows_from_step_dir(args.step_dir), f"dir:{Path(args.step_dir).name}"


def _add_input(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", type=Path)
    g.add_argument("--step-dir", type=Path)
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--no-isolate", action="store_true", help="validate in this process (no per-file timeout)")
    p.add_argument("--out", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m apps.production_line.recipes.cad_import",
                                 description="Validate / import SolidWorks-converted STEP files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_input(sub.add_parser("validate"))
    imp = sub.add_parser("import")
    _add_input(imp)
    imp.add_argument("--data-root", type=Path, required=True, help="line data root (db/recipes.sqlite3)")
    imp.add_argument("--catalog-db", type=Path, required=True, help="part catalog SQLite")
    imp.add_argument("--upload-dir", type=Path, required=True, help="catalog uploads dir (parts_cad/ under it)")
    one = sub.add_parser("_one")
    one.add_argument("--step", required=True)
    one.add_argument("--expected-sha256")
    one.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "_one":
            _write_json(args.out, validate_step(args.step, args.expected_sha256))
            return 0
        try:
            rows, label = _rows_from_args(args)
        except (OSError, ValueError) as exc:
            print(f"cad_import: cannot read input: {exc}", file=sys.stderr)
            return 2
        report = validate_rows(rows, isolate=not args.no_isolate, timeout_s=args.timeout_s)
        report["input"] = label
        if args.cmd == "validate":
            _write_json(args.out, report)
            print(json.dumps(report["counts"], sort_keys=True))
            return 0 if report["counts"]["valid"] == report["counts"]["rows"] else 1
        backend = str(Path(__file__).resolve().parents[3] / "backend")
        if backend not in sys.path:
            sys.path.insert(0, backend)
        from part_catalog import PartCatalogManager  # type: ignore[import-not-found]

        from apps.api.services.production_recipe_service import RecipeContext
        from apps.production_line.recipes.catalog_port import CatalogManagerPort
        from apps.production_line.recipes.store import RecipeStore

        manager = PartCatalogManager(str(args.catalog_db))
        ctx = RecipeContext(catalog=CatalogManagerPort(manager), store=RecipeStore(args.data_root / "db" / "recipes.sqlite3"),
                            data_root=args.data_root)
        result = import_validated(report, ctx=ctx, manager=manager, upload_dir=args.upload_dir, manifest_label=label)
        _write_json(args.out, {"input": label, "validation": report, "import": result})
        print(json.dumps({"validation": report["counts"], "import": result["counts"]}, sort_keys=True))
        bad = {"CREATE_FAILED", "CAD_PUBLISH_FAILED"}
        return 1 if any(k in bad for k in result["counts"]) else 0
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
