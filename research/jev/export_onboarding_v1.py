"""Exporter for CAD onboarding bend candidates (family A, C10 v1).

Every CAD bend candidate of a released recipe becomes one state. The question is
whether the candidate is a real press-brake bend; the operator's confirmation
or removal at onboarding is the label.

Inputs (explicit paths only; see :func:`check_input_path` for custody):

- a ``PartRecipeV1`` JSON file (``schema: sherman_part_recipe_v1``), or
- an onboarding *bundle*, one JSON object or one per line of a ``.jsonl``::

      {"recipe": <PartRecipeV1>,
       "draft_marks": [<vendor BendMark>, ...],     # optional: the CAD candidate list
       "compiled_part": <CompiledPartV1>,           # optional: E1 output
       "provenance_flags": ["retrospective", ...]}  # optional

  ``draft_marks`` are the vendor-setup marks of the released draft
  (``apps/api/services/vendor_setup_service.py`` ``BendMark``). Their CAD
  candidate geometry is immutable, so removed and active candidates are
  described from the same source. This is the shape C05's benchmark export is
  expected to provide (C05 is not imported).

Leakage guards (amendments to the survey feature list, on purpose):

- With ``draft_marks``, **every** candidate is described from its CAD candidate
  geometry only (line length, target angle, target radius) plus part-level
  facts. Recipe-only fields (sequence, machine move, tolerances,
  ``flange_inspection``, ``extent_semantics``, side boundary kinds) exist only for
  confirmed bends, so their presence would encode the label; they are excluded.
- Without ``draft_marks`` the removed candidates have no geometry. Only the
  confirmed bends are exported, and the manifest warns ``NO_NEGATIVES``.
- The E1 relation (``cad_relation``) uses the whole flange graph
  (``bends`` plus ``unmarked_graph_bends``), which does not depend on the marks.
  It is CAD-assisted and dropped by the ``cad_free`` ablation.
- ``DEV_AUTO_CONFIRMED`` recipes are self-referential: their bends get a weak
  label only, never truth.

Truth fails closed: operator labels come only from a released recipe
(``released_at`` set) whose ``confirmation`` is ``OPERATOR_CONFIRMED`` (the
``PartRecipeV1`` default). Any other confirmation value, or an unreleased recipe,
makes every row ``unreviewed`` (no recall rows) with a manifest warning.
Bundle ``provenance_flags`` are limited to ``consumed``/``retrospective``; a
``holdout``/``sealed``/``one_shot`` flag refuses the export.

Labels (see :mod:`.labels_v1`): active ``cad_candidate`` bend ->
``operator_confirmed``; removed ``not_a_bend`` -> ``operator_removed_not_a_bend``;
removed ``wrong_location`` -> null; removed ``missing`` -> null; a manual bend
that corrects nothing -> ``operator_manual_missing`` recall row (no state); an
unmarked E1 graph bend -> ``e1_unmarked_graph_bend`` (null).

This module also hosts the shared export plumbing (custody checks, atomic
output directories) used by :mod:`.export_scan_candidates_v1`, and
:func:`synthetic_onboarding_bundle` for tests and the acceptance run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.jev import HARNESS_VERSION
from research.jev.labels_v1 import LabelRow, make_label, write_jsonl
from research.jev.state_v1 import (
    STATE_SCHEMA,
    CandidateRecord,
    Line,
    build_state,
    enum_token,
    line_from_endpoints,
    neighbour_summary,
    record_id_for,
    relation_to_reference,
)

__all__ = [
    "EXPORT_MANIFEST_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "PROTECTED_REPO_DIRS",
    "CustodyError",
    "ExportError",
    "ExportResult",
    "check_input_path",
    "check_input_dir",
    "validate_provenance_flags",
    "load_authorization",
    "write_export",
    "canonical_sha256",
    "export_onboarding",
    "onboarding_records",
    "synthetic_onboarding_bundle",
]

EXPORT_MANIFEST_SCHEMA = "jev_export_manifest_v1"
AUTHORIZATION_SCHEMA = "jev_input_authorization_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Repo subdirectories that hold physical data or custody artifacts.
PROTECTED_REPO_DIRS = ("data", "output", "outputs", "artifacts", "uploads", "live_scans", "research")
#: research/jev itself is source, not data.
_ALLOWED_RESEARCH_SUBDIR = ("research", "jev")
#: Never readable, with or without authorization (plan §9).
_ALWAYS_FORBIDDEN_PARTS = re.compile(
    r"(holdout|sealed|eho4|one[_-]?shot|^n0(0[1-9]|1[0-4])(\D|$))", re.IGNORECASE
)


class CustodyError(PermissionError):
    """An input path is not allowed (custody, plan §9)."""


class ExportError(ValueError):
    """An export input or output is unusable."""


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def canonical_sha256(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Authorization:
    sha256: str
    roots: tuple[Path, ...]


def load_authorization(path: Path | None, *, repo_root: Path = REPO_ROOT) -> Authorization | None:
    """Load ``{"schema": "jev_input_authorization_v1", "authorized_roots": [...], "note": ...}``.

    Only its sha256 and roots are used; the file is the owner's written permission
    to export from otherwise protected locations. The file itself must pass the
    custody check *without* an authorization before it is opened: it may not sit
    under a protected repo directory (``research/`` siblings hold one-shot and
    authorization artifacts of other protocols) or carry a never-readable path
    component.
    """
    if path is None:
        return None
    resolved = _check_custody(Path(path), None, repo_root, kind="authorization file")
    if not resolved.is_file():
        raise CustodyError(f"authorization file {path} is not a file")
    data = resolved.read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CustodyError(f"authorization file is not JSON: {path}") from exc
    if not isinstance(doc, Mapping) or doc.get("schema") != AUTHORIZATION_SCHEMA:
        raise CustodyError(f"authorization file schema must be {AUTHORIZATION_SCHEMA}")
    roots = doc.get("authorized_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(r, str) and r for r in roots):
        raise CustodyError("authorization needs a non-empty authorized_roots list")
    for root in roots:
        bad = _forbidden_component(Path(os.path.abspath(root)))
        if bad:
            raise CustodyError(f"authorized root {root!r}: {bad}")
    return Authorization(hashlib.sha256(data).hexdigest(), tuple(Path(os.path.abspath(r)) for r in roots))


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _fold_parts(path: Path) -> tuple[str, ...]:
    return tuple(part.casefold() for part in path.parts)


def _under_folded(path: Path, root: Path) -> tuple[str, ...] | None:
    """Case-insensitive ``relative_to``: the folded relative parts, or ``None`` when outside.

    macOS volumes are case-insensitive and ``Path.resolve`` keeps the caller's case, so
    ``DATA/x`` and ``/users/.../fullproject/data/x`` name the protected file too.
    """
    p, r = _fold_parts(path), _fold_parts(root)
    if len(p) < len(r) or p[: len(r)] != r:
        return None
    return p[len(r):]


def _forbidden_component(path: Path) -> str | None:
    for part in path.parts:
        if _ALWAYS_FORBIDDEN_PARTS.search(part):
            return f"path component {part!r} is never readable (holdout/sealed/one-shot/EHO/opaque)"
    return None


def _lexical_problem(path: Path, repo_root: Path, auth: Authorization | None) -> str | None:
    """Pure string check (no filesystem access).

    Protected directories are matched case-insensitively (fail closed); the
    ``research/jev`` exemption and ``authorized_roots`` only match exactly, so a case
    variant can only ever be refused, never let through.
    """
    bad = _forbidden_component(path)
    if bad:
        return bad
    rel = _under_folded(path, repo_root)
    if rel is None:
        return None
    if _under(path, repo_root) and path.relative_to(repo_root).parts[:2] == _ALLOWED_RESEARCH_SUBDIR:
        return None
    if rel and rel[0] in PROTECTED_REPO_DIRS:
        if auth is None:
            return f"inputs under the repo's {rel[0]}/ need --authorization"
        if not any(_under(path, root) for root in auth.roots):
            return "path is not under any authorized_roots entry"
    return None


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _identity_problem(resolved: Path, repo_root: Path, auth: Authorization | None) -> str | None:
    """Filesystem-identity check of an already resolved path.

    Walks the path's ancestors (deepest first) and compares each with the protected
    repo directories by ``(st_dev, st_ino)``, so any alias of ``data/`` (case variant,
    bind mount, hard-linked tree) is caught even when the strings differ.
    """
    exempt = _stat_key(repo_root.joinpath(*_ALLOWED_RESEARCH_SUBDIR))
    protected = {k: d for d in PROTECTED_REPO_DIRS if (k := _stat_key(repo_root / d)) is not None}
    keys = [k for a in (resolved, *resolved.parents) if (k := _stat_key(a)) is not None]
    hit: str | None = None
    for key in keys:  # deepest first: research/jev wins over research/
        if exempt is not None and key == exempt:
            return None
        if key in protected:
            hit = protected[key]
            break
    if hit is None:
        return None
    if auth is None:
        return f"inputs under the repo's {hit}/ need --authorization"
    auth_keys = {k for r in auth.roots if (k := _stat_key(r)) is not None}
    if any(key in auth_keys for key in keys):
        return None
    return "path is not under any authorized_roots entry"


def _check_custody(path: Path, auth: Authorization | None, repo_root: Path, *, kind: str = "input") -> Path:
    lexical = Path(os.path.normpath(os.path.abspath(str(path))))
    root = Path(os.path.normpath(os.path.abspath(str(repo_root))))
    problem = _lexical_problem(lexical, root, auth)
    if problem:
        raise CustodyError(f"{kind} {path}: {problem}")
    resolved = lexical.resolve()
    root_resolved = root.resolve()
    problem = _lexical_problem(resolved, root_resolved, auth) or _identity_problem(resolved, root_resolved, auth)
    if problem:
        raise CustodyError(f"{kind} {path} (resolved {resolved}): {problem}")
    return resolved


def check_input_path(path: Path | str, *, authorization: Authorization | None = None,
                     repo_root: Path = REPO_ROOT) -> Path:
    """Return the resolved path, or raise :class:`CustodyError`.

    The check is lexical first (no filesystem access for a refused path), then
    repeated on the resolved path so a symlink cannot escape it, and finally by
    filesystem identity of every ancestor so a case variant or other alias of a
    protected directory cannot either.
    """
    resolved = _check_custody(Path(path), authorization, repo_root)
    if not resolved.is_file():
        raise ExportError(f"{path}: not a file")
    return resolved


def check_input_dir(path: Path | str, *, authorization: Authorization | None = None,
                    repo_root: Path = REPO_ROOT) -> Path:
    """Custody check for a directory input (same rules as :func:`check_input_path`)."""
    resolved = _check_custody(Path(path), authorization, repo_root)
    if not resolved.is_dir():
        raise ExportError(f"{path}: not a directory")
    return resolved


#: Provenance flags a record may carry; holdout-type flags refuse the export.
ALLOWED_PROVENANCE_FLAGS = frozenset({"consumed", "retrospective"})
REFUSED_PROVENANCE_FLAGS = frozenset({"holdout", "sealed", "one_shot"})


def validate_provenance_flags(flags: Any) -> list[str]:
    """Return the flags as a sorted list, or raise.

    ``holdout``/``sealed``/``one_shot`` records are never exportable
    (:class:`CustodyError`); any other unknown flag is an :class:`ExportError`.
    """
    if not isinstance(flags, (list, tuple)) or not all(isinstance(f, str) for f in flags):
        raise ExportError("provenance_flags must be a list of strings")
    refused = sorted({f.casefold() for f in flags} & REFUSED_PROVENANCE_FLAGS)
    if refused:
        raise CustodyError(f"records flagged {refused} are never exported")
    unknown = sorted(set(flags) - ALLOWED_PROVENANCE_FLAGS)
    if unknown:
        raise ExportError(f"unknown provenance_flags {unknown}; allowed: {sorted(ALLOWED_PROVENANCE_FLAGS)}")
    return sorted(set(flags))


@dataclass
class ExportResult:
    out_dir: Path
    states: int
    labels: int
    manifest: dict[str, Any]


def write_export(
    out_dir: Path,
    *,
    states: Sequence[tuple[str, dict[str, Any]]],
    labels: Sequence[LabelRow],
    manifest: dict[str, Any],
) -> ExportResult:
    """Write ``states.jsonl``, ``labels.jsonl`` and ``manifest.json`` atomically into a new dir.

    ``out_dir`` must not exist. Everything is written into a sibling temp dir that is
    renamed into place, so a crash never leaves a half export at ``out_dir``.
    """
    out_dir = Path(out_dir)
    if os.path.lexists(out_dir):
        raise ExportError(f"{out_dir} already exists; exports always go to a new directory")
    parent = out_dir.parent
    if not parent.is_dir():
        raise ExportError(f"parent directory {parent} does not exist")
    ids = [rid for rid, _ in states]
    if len(set(ids)) != len(ids):
        raise ExportError("duplicate record ids in states (was the same input record given twice?)")
    label_ids = [row.record_id for row in labels]
    if len(set(label_ids)) != len(label_ids):
        raise ExportError("duplicate record ids in labels")
    with_state = {row.record_id for row in labels if row.has_state}
    if with_state != set(ids):
        raise ExportError("every state needs exactly one label row and vice versa")
    tmp = parent / f".{out_dir.name}.tmp-{os.getpid()}"
    if os.path.lexists(tmp):
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        state_rows = [
            {"record_id": rid, "state_sha256": canonical_sha256(state), "state": state}
            for rid, state in sorted(states, key=lambda item: item[0])
        ]
        write_jsonl(tmp / "states.jsonl", state_rows)
        write_jsonl(tmp / "labels.jsonl", [row.to_json() for row in sorted(labels, key=lambda r: r.record_id)])
        full = dict(manifest)
        full["schema"] = EXPORT_MANIFEST_SCHEMA
        full["harness_version"] = HARNESS_VERSION
        full["state_schema"] = STATE_SCHEMA
        full["states_sha256"] = hashlib.sha256((tmp / "states.jsonl").read_bytes()).hexdigest()
        full["labels_sha256"] = hashlib.sha256((tmp / "labels.jsonl").read_bytes()).hexdigest()
        full["counts"] = dict(full.get("counts", {}), states=len(state_rows), labels=len(labels))
        (tmp / "manifest.json").write_text(
            json.dumps(full, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return ExportResult(out_dir, len(state_rows), len(labels), full)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """A ``.jsonl`` file gives one record per line; any other file is one JSON value."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".jsonl":
        out = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ExportError(f"{path}:{n}: invalid JSON") from exc
        return out
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExportError(f"{path}: invalid JSON") from exc
    return value if isinstance(value, list) else [value]


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------
# Onboarding records
# --------------------------------------------------------------------------

RECIPE_SCHEMA = "sherman_part_recipe_v1"


@dataclass
class _Candidate:
    feature_id: str
    line: Line | None
    target_angle_deg: float | None
    target_radius_mm: float | None
    origin: str  # CAD_ONBOARDING | E1_FLANGE_GRAPH
    label_source: str
    weak_label_source: str | None = None


@dataclass
class OnboardingExport:
    records: list[tuple[CandidateRecord, LabelRow]] = field(default_factory=list)
    recall_rows: list[LabelRow] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _angles(target_angle: float | None, convention: str | None) -> tuple[float | None, float | None]:
    """``(sweep_deg, included_deg)`` from a CAD candidate target angle (CONTRACTS §0)."""
    if target_angle is None:
        return None, None
    if convention == "MESH_INCLUDED_FOLDED":
        return 180.0 - target_angle, target_angle
    # NATIVE_SWEEP (and the default): the target is the mean sweep.
    return target_angle, 180.0 - target_angle


def _bundle_parts(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any] | None, dict[str, Any] | None, list[str]]:
    if raw.get("schema") == RECIPE_SCHEMA:
        return dict(raw), None, None, []
    recipe = raw.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("schema") != RECIPE_SCHEMA:
        raise ExportError(f"onboarding record needs a {RECIPE_SCHEMA} recipe")
    marks = raw.get("draft_marks")
    if marks is not None and not isinstance(marks, list):
        raise ExportError("draft_marks must be an array")
    compiled = raw.get("compiled_part")
    if compiled is not None and not isinstance(compiled, Mapping):
        raise ExportError("compiled_part must be an object")
    flags = validate_provenance_flags(raw.get("provenance_flags") or [])
    return dict(recipe), marks, (dict(compiled) if compiled else None), flags


def _graph_refs(compiled: Mapping[str, Any] | None) -> list[tuple[Line, dict[str, Any], str, bool]]:
    """``(line, attrs, graph_bend_id, marked)`` for every flange-graph bend of a CompiledPart."""
    if not compiled:
        return []
    out: list[tuple[Line, dict[str, Any], str, bool]] = []
    for bend in compiled.get("bends") or ():
        locus = ((bend.get("loci") or {}).get("oml_virtual_sharp")) or {}
        line = line_from_endpoints(locus.get("start_mm"), locus.get("end_mm"))
        if line is not None:
            out.append((line, {"graph_sweep_deg": _num(bend.get("sweep_deg")),
                               "graph_inner_radius_mm": _num(bend.get("inner_radius_mm"))},
                        str(bend.get("graph_bend_id") or ""), True))
    for bend in compiled.get("unmarked_graph_bends") or ():
        locus = bend.get("oml_virtual_sharp") or {}
        line = line_from_endpoints(locus.get("start_mm"), locus.get("end_mm"))
        if line is not None:
            out.append((line, {"graph_sweep_deg": _num(bend.get("sweep_deg")),
                               "graph_inner_radius_mm": _num(bend.get("inner_radius_mm"))},
                        str(bend.get("graph_bend_id") or ""), False))
    return out


def onboarding_records(
    raw: Mapping[str, Any],
    *,
    include_unmarked_graph: bool = True,
) -> OnboardingExport:
    """Candidate records and label rows for one recipe (or bundle)."""
    recipe, marks, compiled, flags = _bundle_parts(raw)
    result = OnboardingExport(counts={"removed_without_geometry": 0, "manual_recall_rows": 0,
                                      "manual_corrections": 0, "unmarked_graph_bends": 0})
    source_sha = canonical_sha256(raw)
    sku = recipe.get("sku")
    convention = (recipe.get("cad") or {}).get("candidate_angle_convention")
    # Truth fails closed: only a released recipe whose confirmation is (or defaults to,
    # per PartRecipeV1) OPERATOR_CONFIRMED yields operator labels. DEV_AUTO_CONFIRMED is
    # weak-only; anything else (typo, future value, unreleased draft) is unreviewed.
    confirmation = recipe.get("confirmation", "OPERATOR_CONFIRMED")
    released = isinstance(recipe.get("released_at"), str) and bool(recipe["released_at"].strip())
    dev_auto = confirmation == "DEV_AUTO_CONFIRMED"
    trusted = confirmation == "OPERATOR_CONFIRMED" and released
    if not dev_auto and not trusted:
        result.warnings.append("RECIPE_NOT_RELEASED" if confirmation == "OPERATOR_CONFIRMED"
                               else "UNTRUSTED_CONFIRMATION")
    thickness = _num(recipe.get("thickness_mm"))
    if thickness is None and compiled:
        thickness = _num((compiled.get("thickness") or {}).get("t_mm"))

    bends = [b for b in recipe.get("bends") or () if isinstance(b, Mapping)]
    active_cad = {str(b.get("source_feature_id")): b for b in bends if b.get("source") == "cad_candidate"}
    removed = {str(r.get("feature_id")): r for r in recipe.get("removed_candidates") or () if isinstance(r, Mapping)}

    def _label_for(feature_id: str) -> tuple[str, str | None]:
        if not dev_auto and not trusted:
            return "unreviewed", None
        if feature_id in active_cad:
            return ("dev_auto_confirmed", "dev_auto_confirmed_active") if dev_auto else ("operator_confirmed", None)
        if feature_id in removed:
            issue = removed[feature_id].get("issue")
            if dev_auto:
                return "dev_auto_confirmed", None
            return {"not_a_bend": ("operator_removed_not_a_bend", None),
                    "wrong_location": ("operator_wrong_location", None),
                    "missing": ("operator_removed_issue_missing", None)}.get(str(issue), ("unreviewed", None))
        return "unreviewed", None

    candidates: list[_Candidate] = []
    if marks is not None:
        for mark in marks:
            if not isinstance(mark, Mapping) or mark.get("source") != "cad_candidate":
                continue
            fid = str(mark.get("feature_id"))
            label_source, weak = _label_for(fid)
            candidates.append(_Candidate(
                fid, line_from_endpoints(mark.get("line_start"), mark.get("line_end")),
                _num(mark.get("target_angle")), _num(mark.get("target_radius")),
                "CAD_ONBOARDING", label_source, weak))
        seen = {c.feature_id for c in candidates}
        missing_active = sorted(set(active_cad) - seen)
        if missing_active:
            result.warnings.append(f"ACTIVE_BEND_NOT_IN_DRAFT_MARKS:{len(missing_active)}")
    else:
        for fid, bend in sorted(active_cad.items()):
            angle = bend.get("nominal_angle") or {}
            target = _num(angle.get("sweep_deg")) if convention != "MESH_INCLUDED_FOLDED" else _num(angle.get("included_deg"))
            label_source, weak = _label_for(fid)
            candidates.append(_Candidate(
                fid, line_from_endpoints(bend.get("line_start_mm"), bend.get("line_end_mm")),
                target, _num(bend.get("nominal_inner_radius_mm")), "CAD_ONBOARDING", label_source, weak))
        result.counts["removed_without_geometry"] = len(removed)
        result.warnings.append("NO_NEGATIVES: no draft_marks, removed candidates have no geometry")

    graph = _graph_refs(compiled)
    if include_unmarked_graph:
        for line, attrs, gid, marked in graph:
            if marked:
                continue
            result.counts["unmarked_graph_bends"] += 1
            sweep = attrs.get("graph_sweep_deg")
            target = sweep if convention != "MESH_INCLUDED_FOLDED" or sweep is None else 180.0 - sweep
            candidates.append(_Candidate(f"graph:{gid}", line, target, attrs.get("graph_inner_radius_mm"),
                                         "E1_FLANGE_GRAPH", "e1_unmarked_graph_bend"))

    # Neighbour summaries are computed within one origin, so the (mark-dependent)
    # unmarked E1 graph rows never change a CAD candidate's descriptors.
    origin_lines: dict[str, list[Line | None]] = {}
    origin_index: list[int] = []
    for cand in candidates:
        bucket = origin_lines.setdefault(cand.origin, [])
        origin_index.append(len(bucket))
        bucket.append(cand.line)
    references = [(line, attrs) for line, attrs, _, _ in graph]
    for index, cand in enumerate(candidates):
        sweep, included = _angles(cand.target_angle_deg, convention)
        ratio = (cand.target_radius_mm / thickness) if (cand.target_radius_mm is not None and thickness) else None
        descriptors: dict[str, Any] = {
            "bend_line_length_mm": None if cand.line is None else cand.line.length,
            "nominal_sweep_deg": sweep,
            "nominal_included_deg": included,
            "nominal_inner_radius_mm": cand.target_radius_mm,
            "radius_to_thickness_ratio": ratio,
            "angle_convention": enum_token(convention) or "UNKNOWN",
        }
        # The CAD candidate count does not depend on the marks (graph rows excluded).
        context = {"sheet_thickness_mm": thickness,
                   "cad_candidates_count": len(origin_lines.get("CAD_ONBOARDING", ()))}
        cad_relation = None
        if compiled:
            # A graph row is never related to itself.
            refs = [(ref, attrs) for ref, attrs in references if ref is not cand.line]
            cad_relation = relation_to_reference(cand.line, refs)
        record = CandidateRecord(
            source_sha256=source_sha,
            candidate_key=f"onboarding|{cand.feature_id}",
            candidate_source=cand.origin,
            candidate=descriptors,
            neighbours=neighbour_summary(origin_index[index], origin_lines[cand.origin]),
            context=context,
            cad_relation=cad_relation,
        )
        rid = record_id_for(source_sha, record.candidate_key)
        label = make_label(rid, label_source=cand.label_source, group_sku=sku,
                           weak_label_source=cand.weak_label_source, candidate_key=record.candidate_key,
                           source_sha256=source_sha, provenance_flags=flags)
        result.records.append((record, label))

    for bend in bends:
        if bend.get("source") != "manual":
            continue
        if bend.get("corrects_feature_id"):
            result.counts["manual_corrections"] += 1
            continue
        if dev_auto or not trusted:
            continue
        key = f"onboarding_manual|{bend.get('source_feature_id')}"
        rid = record_id_for(source_sha, key)
        result.recall_rows.append(make_label(rid, label_source="operator_manual_missing", group_sku=sku,
                                             has_state=False, candidate_key=key, source_sha256=source_sha,
                                             provenance_flags=flags))
        result.counts["manual_recall_rows"] += 1
    return result


def export_onboarding(
    inputs: Sequence[Path | str],
    out_dir: Path,
    *,
    authorization: Path | None = None,
    include_unmarked_graph: bool = True,
    repo_root: Path = REPO_ROOT,
) -> ExportResult:
    """Export every candidate of every input recipe/bundle into a new ``out_dir``."""
    auth = load_authorization(authorization, repo_root=repo_root)
    if not inputs:
        raise ExportError("no inputs")
    states: list[tuple[str, dict[str, Any]]] = []
    labels: list[LabelRow] = []
    counts: dict[str, int] = {}
    warnings: list[str] = []
    manifest_inputs = []
    for raw_path in inputs:
        path = check_input_path(raw_path, authorization=auth, repo_root=repo_root)
        data = path.read_bytes()
        manifest_inputs.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest()})
        for n, raw in enumerate(load_json_records(path)):
            if not isinstance(raw, Mapping):
                raise ExportError(f"{path}: record {n} is not an object")
            part = onboarding_records(raw, include_unmarked_graph=include_unmarked_graph)
            for record, label in part.records:
                rid, state = build_state(record)
                states.append((rid, state))
                labels.append(label)
            labels.extend(part.recall_rows)
            for key, value in part.counts.items():
                counts[key] = counts.get(key, 0) + value
            warnings.extend(f"{path.name}#{n}:{w}" for w in part.warnings)
            counts["recipes"] = counts.get("recipes", 0) + 1
    by_source: dict[str, int] = {}
    for row in labels:
        by_source[row.label_source] = by_source.get(row.label_source, 0) + 1
    counts.update({f"label_source:{k}": v for k, v in sorted(by_source.items())})
    manifest = {
        "exporter": "export_onboarding_v1",
        "family": "CAD_ONBOARDING_CANDIDATE",
        "inputs": manifest_inputs,
        "authorization_sha256": None if auth is None else auth.sha256,
        "options": {"include_unmarked_graph": include_unmarked_graph},
        "counts": counts,
        "warnings": sorted(warnings),
        "leakage_guards": [
            "recipe-only fields (sequence, moves, tolerances, flange_inspection, extent_semantics, side kinds) excluded",
            "descriptors from CAD candidate geometry for every candidate",
            "cad_relation uses the full flange graph (marked and unmarked) and is ablatable",
            "DEV_AUTO_CONFIRMED labels are weak only",
        ],
    }
    return write_export(out_dir, states=states, labels=labels, manifest=manifest)


# --------------------------------------------------------------------------
# Synthetic example (tests and acceptance runs only; no real data)
# --------------------------------------------------------------------------


def synthetic_onboarding_bundle(seed: int = 0, *, sku: str = "SYN-ONB-A", n_real: int = 4,
                                n_fake: int = 3, confirmation: str = "OPERATOR_CONFIRMED",
                                with_compiled: bool = True) -> dict[str, Any]:
    """A deterministic fake onboarding bundle: a box-like part with real and fake candidates.

    Real candidates are long, ~90 deg sweep lines with a tooling radius; fake ones are
    short outline/hole-like lines with odd angles. One fake is ``not_a_bend``, one is
    ``wrong_location`` (with a manual correction), one is left unreviewed, and one manual
    bend the candidates missed is added as a recall row.
    """
    rng = random.Random(seed)
    t = 2.0
    marks: list[dict[str, Any]] = []
    recipe_bends: list[dict[str, Any]] = []
    graph_bends: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for i in range(n_real):
        y = 40.0 * i + rng.uniform(-1, 1)
        length = 150.0 + rng.uniform(-20, 20)
        start, end = [0.0, y, 0.0], [length, y, 0.0]
        sweep = 90.0 + rng.uniform(-1.0, 1.0)
        fid = f"cad:{i + 1}"
        marks.append({"feature_id": fid, "label": f"B{i + 1}", "source": "cad_candidate", "status": "active",
                      "review_status": "confirmed", "line_start": start, "line_end": end,
                      "target_angle": sweep, "target_radius": 2.0})
        recipe_bends.append({"bend_uid": f"bu_{i:020d}", "sequence_index": i + 1, "label": f"B{i + 1}",
                             "cad_label": f"B{i + 1}", "source": "cad_candidate", "source_feature_id": fid,
                             "line_start_mm": start, "line_end_mm": end,
                             "nominal_angle": {"included_deg": 180.0 - sweep, "sweep_deg": sweep,
                                               "source": "CAD_NATIVE_SWEEP", "convention_verified": True},
                             "nominal_inner_radius_mm": 2.0})
        graph_bends.append({"graph_bend_id": f"G{i + 1}", "sweep_deg": sweep, "inner_radius_mm": 2.0,
                            "loci": {"oml_virtual_sharp": {"start_mm": [0.0, y + 3.0, 0.0],
                                                           "end_mm": [length, y + 3.0, 0.0]}}})
    issues = ["not_a_bend", "wrong_location", None]
    for j in range(n_fake):
        x = 20.0 + 15.0 * j
        start, end = [x, -10.0, 5.0], [x, -10.0 + rng.uniform(8, 20), 5.0]
        fid = f"cad:{n_real + j + 1}"
        issue = issues[j % len(issues)]
        mark = {"feature_id": fid, "label": f"C{j + 1}", "source": "cad_candidate",
                "status": "removed" if issue else "active", "review_status": "pending",
                "line_start": start, "line_end": end, "target_angle": rng.uniform(20, 160),
                "target_radius": rng.uniform(0.1, 30.0), "issue": issue}
        marks.append(mark)
        if issue:
            removed.append({"feature_id": fid, "label": f"C{j + 1}", "issue": issue, "note": ""})
    # wrong_location correction and one missed bend
    recipe_bends.append({"bend_uid": "bu_manualcorrection0000", "sequence_index": len(recipe_bends) + 1,
                         "label": "M1", "cad_label": "M1", "source": "manual", "source_feature_id": "manual:1",
                         "corrects_feature_id": f"cad:{n_real + 2}", "line_start_mm": [1, 2, 3],
                         "line_end_mm": [4, 5, 6], "nominal_angle": {"included_deg": 90.0, "source": "MANUAL_ENTRY",
                                                                    "convention_verified": False}})
    recipe_bends.append({"bend_uid": "bu_manualmissing000000", "sequence_index": len(recipe_bends) + 1,
                         "label": "M2", "cad_label": "M2", "source": "manual", "source_feature_id": "manual:2",
                         "line_start_mm": [7, 8, 9], "line_end_mm": [10, 11, 12],
                         "nominal_angle": {"included_deg": 90.0, "source": "MANUAL_ENTRY",
                                           "convention_verified": False}})
    recipe = {
        "schema": RECIPE_SCHEMA,
        "recipe_id": f"{sku}@v1",
        "sku": sku,
        "part_id": f"part-{sku.lower()}",
        "thickness_mm": t,
        "recipe_version": 1,
        "released_at": "2026-10-17T00:00:00Z",
        "confirmation": confirmation,
        "cad": {"sha256": "0" * 64, "candidate_angle_convention": "NATIVE_SWEEP",
                "original_filename": f"{sku}.step"},
        "bends": recipe_bends,
        "removed_candidates": removed,
    }
    bundle: dict[str, Any] = {"recipe": recipe, "draft_marks": marks}
    if with_compiled:
        bundle["compiled_part"] = {
            "schema": "sherman_compiled_part_v1",
            "thickness": {"t_mm": t, "source": "MEASURED_BOTH_SKINS"},
            "bends": graph_bends,
            # An unsupported (hem-like) graph bend nobody marked.
            "unmarked_graph_bends": [{"graph_bend_id": "GX", "sweep_deg": 170.0, "inner_radius_mm": 1.0,
                                      "oml_virtual_sharp": {"start_mm": [0.0, 400.0, 0.0],
                                                            "end_mm": [60.0, 400.0, 0.0]}}],
        }
    return bundle


def write_synthetic_inputs(directory: Path, *, families: Iterable[tuple[str, int]] = (("SYN-ONB-A", 0),
                                                                                    ("SYN-ONB-B", 1),
                                                                                    ("SYN-ONB-C", 2))) -> Path:
    """Write a synthetic onboarding ``.jsonl`` (one bundle per SKU) and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "synthetic_onboarding.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for sku, seed in families:
            fh.write(json.dumps(synthetic_onboarding_bundle(seed, sku=sku), sort_keys=True) + "\n")
    return path
