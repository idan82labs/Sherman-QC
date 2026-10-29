"""Flat intake (C05): attach the unfolded part (DXF or flat STEP) during onboarding.

Flow (plan §3.1 "Flat binding", §6 step 1-2):

1. **Upload** (:meth:`FlatIntakeService.ingest_stream`). The bytes are streamed
   to ``<data_root>/recipes/flats/.incoming-*`` with a size cap
   (:data:`FLAT_MAX_BYTES`), hashed, sniffed (DXF: group-code text with a
   ``SECTION`` or the binary sentinel; STEP: ``ISO-10303-21``) and stored by
   content address ``recipes/flats/<sha256>.dxf|.step|.stp`` (tmp + fsync +
   rename, idempotent). E1's parser then runs in a supervised subprocess
   (:class:`FlatTool`, :mod:`.flat_intake_cli`); the parse summary and the
   display geometry (outline loops and bend lines, mm, flat frame) are stored
   with the part. State UPLOADED, or PARSE_FAILED with E1's reason.
2. **Propose** (:meth:`FlatIntakeService.propose`). E1's
   ``propose_flat_binding`` runs (subprocess) on the catalog STEP, the flat and
   the operator's current draft marks (active, located, in bending order, with
   their recipe ``bend_uid``). The proposal (``bend_line_map`` flat line id ->
   bend_uid, mirror flag, confidence, ``needs_operator_review``, K, the
   developed-length check and ``fit_choice``: ``ambiguous``,
   ``mirror_determined``, ``alternative_maps``) is stored for the operator.
   State PROPOSED. The proposal is bound to a *basis*: the CAD sha256 and the
   set of mark identities (:func:`binding_basis_sha256`).
3. **Decide** (:meth:`FlatIntakeService.set_binding`). The operator confirms a
   map (the proposal's, or an edited one: method MANUAL) -> CONFIRMED, or keeps
   the flat explicitly unbound -> DECLINED.

At release, :func:`release_flat_ref` turns the stored decision into the
recipe's ``FlatRef`` or refuses with an explicit code:

- FLAT_PARSE_FAILED: the flat could not be parsed;
- FLAT_MISSING / FLAT_HASH_MISMATCH: the stored bytes are gone or changed;
- FLAT_BINDING_UNCONFIRMED: uploaded or proposed but not decided. Only a
  ``DEV_AUTO_CONFIRMED`` release may release an un-reviewed proposal, as
  ``AUTO_PROPOSED`` (E1 then still gives no K from it unless the recipe is
  DEV_AUTO_CONFIRMED, and the wizard never offers this);
- FLAT_BINDING_STALE: the decision was made against other CAD bytes or other
  marks (bend identities); propose and confirm again.

A DECLINED flat is released ``UNBOUND`` (stored and shown, no K from it).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from apps.production_line.recipes.hashing import cached_sha256
from apps.production_line.recipes.store import (
    PartFlatRecord,
    RecipeStore,
    RecipeStoreError,
    iso_utc,
    utc_now,
)

__all__ = [
    "FLAT_MAX_BYTES",
    "FLAT_INTAKE_CODES",
    "FlatIntakeError",
    "FlatTool",
    "FlatIntakeService",
    "detect_flat_format",
    "binding_marks",
    "binding_basis_sha256",
    "release_flat_ref",
    "released_by_dict",
]

_LOG = logging.getLogger(__name__)

#: Largest flat accepted (a shop DXF is typically < 5 MB).
FLAT_MAX_BYTES = 64 * 1024**2
_HEAD_BYTES = 4096
_CHUNK = 1024 * 1024

FLAT_INTAKE_CODES: frozenset[str] = frozenset(
    {
        "PART_NOT_FOUND",
        "FLAT_NOT_FOUND",
        "FLAT_EMPTY",
        "FLAT_TOO_LARGE",
        "FLAT_FORMAT_UNSUPPORTED",
        "FLAT_PARSE_FAILED",
        "FLAT_TOOL_FAILED",
        "FLAT_TOOL_TIMEOUT",
        "FLAT_REVISION_CHANGED",
        "FLAT_REVISION_REQUIRED",
        "FLAT_BINDING_INVALID",
        "FLAT_BINDING_STALE",
        "FLAT_NOT_PARSED",
        "CAD_MISSING",
        "CAD_NOT_STEP",
        "CAD_HASH_MISMATCH",
        "CAD_READ_FAILED",
        "DRAFT_MISSING",
        "NO_ACTIVE_BENDS",
        "RECONCILIATION_BLOCKED",
    }
)
_STATUS_BY_CODE: dict[str, int] = {
    "PART_NOT_FOUND": 404,
    "FLAT_NOT_FOUND": 404,
    "FLAT_TOO_LARGE": 413,
    "FLAT_REVISION_CHANGED": 409,
    "FLAT_TOOL_FAILED": 502,
    "FLAT_TOOL_TIMEOUT": 504,
}

_STEP_EXTS = (".step", ".stp")
_DXF_GROUP_CODE = re.compile(rb"^\s*\d{1,4}\s*\r?\n")


class FlatIntakeError(Exception):
    """A refused flat-intake operation; ``codes`` from :data:`FLAT_INTAKE_CODES`."""

    def __init__(self, codes: Sequence[str], message: str = "", extra: Mapping[str, Any] | None = None) -> None:
        unique = tuple(sorted(set(codes)))
        unknown = set(unique) - FLAT_INTAKE_CODES
        if unknown or not unique:  # pragma: no cover - programming error
            raise ValueError(f"bad flat intake codes {sorted(unknown) or 'none'}")
        super().__init__(f"{', '.join(unique)}: {message}" if message else ", ".join(unique))
        self.codes = unique
        self.message = message
        self.extra = dict(extra or {})

    @property
    def http_status(self) -> int:
        return max((_STATUS_BY_CODE.get(c, 422) for c in self.codes), key=lambda s: (s != 422, s))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_flat_format(filename: str, head: bytes) -> tuple[str, str]:
    """``(format, ext)`` of an uploaded flat, from its extension *and* its first bytes.

    DXF: ``.dxf`` with a binary-DXF sentinel, or ASCII group-code text that
    contains ``SECTION`` in the first 4 KiB. STEP: ``.step`` / ``.stp`` whose
    head contains ``ISO-10303-21``. Anything else is FLAT_FORMAT_UNSUPPORTED
    (never guessed).
    """
    ext = Path(filename or "").suffix.lower()
    if ext == ".dxf":
        if head.startswith(b"AutoCAD Binary DXF"):
            return "DXF", ".dxf"
        text = head.lstrip(b"\xef\xbb\xbf")
        if _DXF_GROUP_CODE.match(text) and b"SECTION" in text:
            return "DXF", ".dxf"
        raise FlatIntakeError(["FLAT_FORMAT_UNSUPPORTED"], "the .dxf file is not DXF text or binary DXF")
    if ext in _STEP_EXTS:
        if b"ISO-10303-21" in head:
            return "STEP", ext
        raise FlatIntakeError(["FLAT_FORMAT_UNSUPPORTED"], "the STEP file has no ISO-10303-21 header")
    raise FlatIntakeError(["FLAT_FORMAT_UNSUPPORTED"], f"unsupported flat type {ext or '(none)'}; use DXF or STEP")


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


def binding_marks(draft: Mapping[str, Any] | None, cad_sha256: str) -> list[dict[str, Any]]:
    """Active, located draft marks in bending order with their recipe ``bend_uid``.

    Same identity rule as the C01 builder: ``bend_uid(cad_sha256, source,
    line_start, line_end, radius-if-finite)``. Marks without a line or not in
    the sequence are skipped (the release gate reports them).
    """
    from domains.production.recipe.bend_identity import bend_uid

    if not draft:
        return []
    by_id = {m.get("feature_id"): m for m in (draft.get("marks") or []) if isinstance(m, Mapping)}
    out: list[dict[str, Any]] = []
    for fid in draft.get("sequence") or []:
        mark = by_id.get(fid)
        if mark is None or mark.get("status") != "active":
            continue
        start, end = _parse_point(mark.get("line_start")), _parse_point(mark.get("line_end"))
        if start is None or end is None or start == end:
            continue
        radius = mark.get("target_radius")
        finite_radius = (
            float(radius)
            if isinstance(radius, (int, float)) and not isinstance(radius, bool) and math.isfinite(radius)
            else None
        )
        try:
            uid = bend_uid(cad_sha256, str(mark.get("source", "")), start, end, finite_radius)
        except ValueError:
            continue
        out.append(
            {
                "bend_uid": uid,
                "feature_id": fid,
                "label": str(mark.get("label") or fid),
                "line_start": start,
                "line_end": end,
            }
        )
    return out


def binding_basis_sha256(cad_sha256: str, marks: Sequence[Mapping[str, Any]]) -> str:
    """Identity of what a binding was made against: the CAD bytes and the set of bend identities."""
    doc = {"cad_sha256": cad_sha256, "bend_uids": sorted({str(m["bend_uid"]) for m in marks})}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def released_by_dict(principal: Any) -> dict[str, str]:
    """``ReleasedBy``-shaped ``{user_id, username, role}`` of a principal or mapping."""
    if isinstance(principal, Mapping):
        user_id = principal.get("user_id", principal.get("id"))
        username = principal.get("username", user_id)
        role = principal.get("role")
    else:
        user_id = getattr(principal, "user_id", None)
        username = getattr(principal, "username", None) or user_id
        role = getattr(principal, "role", None)
    if not user_id or not role:
        raise ValueError("principal needs a user id and a role")
    return {"user_id": str(user_id)[:128], "username": str(username)[:128], "role": str(role)[:64]}


# ---------------------------------------------------------------------------
# Subprocess tool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatTool:
    """Runs :mod:`.flat_intake_cli` under the C02 supervisor.

    ``source_root`` is the running source root (dev repo or app copy; it holds
    ``apps/``). ``engine_root`` is where E1 is imported from: the pinned engine
    bundle when one exists (the same E1 the release compile runs,
    :func:`.release_service.engine_root_for`), else None = ``source_root``.
    The subprocess cwd is the engine root and ``PYTHONPATH`` is
    ``engine_root:source_root:source_root/backend``, so ``domains`` (E1) comes
    from the bundle and ``apps`` (this CLI, absent from an engine bundle)
    from the source root.
    """

    source_root: Path
    python_exe: Path
    engine_root: Path | None = None
    parse_timeout_s: float = 60.0
    propose_timeout_s: float = 120.0
    rss_limit_bytes: int = 4 * 1024**3
    shop_k: float | None = None
    poll_s: float = 0.25
    kill_grace_s: float = 3.0
    extra_env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Any) -> "FlatTool":
        """Same E1 root as :meth:`.release_service.CompileLauncher.from_settings`."""
        from apps.production_line.recipes.release_service import engine_root_for

        source = Path(settings.repo_root)
        engine = engine_root_for(settings)
        return cls(
            source_root=source,
            python_exe=Path(settings.python_exe),
            engine_root=None if engine.resolve() == source.resolve() else engine,
            rss_limit_bytes=min(int(settings.rss_limit_bytes), 4 * 1024**3),
        )

    @property
    def run_root(self) -> Path:
        """The subprocess cwd (the E1 root)."""
        return Path(self.engine_root if self.engine_root is not None else self.source_root)

    def env(self) -> dict[str, str]:
        from apps.production_line.runner.runner import ENV_PASSTHROUGH, THREAD_ENV_VARS

        env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
        root = str(Path(self.source_root))
        entries = [root, str(Path(root) / "backend")]
        if self.engine_root is not None:
            entries.insert(0, str(Path(self.engine_root)))
        env["PYTHONPATH"] = os.pathsep.join(entries)
        for key in THREAD_ENV_VARS:
            env[key] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.update({str(k): str(v) for k, v in self.extra_env.items()})
        return env

    def _run(self, args: list[str], timeout_s: float) -> dict[str, Any]:
        from apps.production_line.runner.supervisor import run_supervised

        work = Path(tempfile.mkdtemp(prefix="sherman-flat-"))
        out = work / "out.json"
        tail: list[str] = []

        def _line(text: str) -> None:
            tail.append(text)
            del tail[:-20]

        try:
            cmd = [str(self.python_exe), "-m", "apps.production_line.recipes.flat_intake_cli", *args, "--out", str(out)]
            res = run_supervised(
                cmd,
                env=self.env(),
                cwd=self.run_root,
                timeout_s=timeout_s,
                rss_limit_bytes=self.rss_limit_bytes,
                poll_s=self.poll_s,
                on_line=_line,
                kill_grace_s=self.kill_grace_s,
                identity_tag=f"c05-flat-{uuid.uuid4().hex[:12]}",
            )
            if res.timed_out:
                raise FlatIntakeError(["FLAT_TOOL_TIMEOUT"], f"flat tool exceeded {timeout_s:g} s")
            if res.rss_exceeded:
                raise FlatIntakeError(["FLAT_TOOL_FAILED"], "flat tool exceeded its memory limit")
            if res.exit_code != 0 or not out.is_file():
                detail = " | ".join(tail[-5:])[:500]
                raise FlatIntakeError(["FLAT_TOOL_FAILED"], f"flat tool exit {res.exit_code}: {detail}")
            doc = json.loads(out.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or "status" not in doc:
                raise FlatIntakeError(["FLAT_TOOL_FAILED"], "flat tool wrote no status")
            return doc
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def parse(self, flat_path: Path, fmt: str) -> dict[str, Any]:
        return self._run(["parse", "--flat", str(flat_path), "--format", fmt], self.parse_timeout_s)

    def propose(self, cad_path: Path, flat_path: Path, fmt: str, marks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(
                [{"bend_uid": m["bend_uid"], "line_start": m["line_start"], "line_end": m["line_end"]} for m in marks],
                fh,
            )
            marks_path = fh.name
        try:
            args = ["propose", "--cad", str(cad_path), "--flat", str(flat_path), "--format", fmt, "--marks", marks_path]
            if self.shop_k is not None:
                args += ["--shop-k", repr(float(self.shop_k))]
            return self._run(args, self.propose_timeout_s)
        finally:
            try:
                os.unlink(marks_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class FlatIntakeService:
    """Upload, parse, propose and decide a part's flat (module docstring)."""

    def __init__(
        self,
        *,
        catalog: Any,
        store: RecipeStore,
        data_root: Path,
        tool: Any,
        clock: Callable[[], datetime] = utc_now,
        max_bytes: int = FLAT_MAX_BYTES,
    ) -> None:
        self.catalog = catalog
        self.store = store
        self.data_root = Path(data_root)
        self.tool = tool
        self.clock = clock
        self.max_bytes = int(max_bytes)

    @property
    def flats_dir(self) -> Path:
        return self.data_root / "recipes" / "flats"

    def _require_part(self, part_id: str) -> dict[str, Any]:
        part = self.catalog.get_part(part_id)
        if part is None:
            raise FlatIntakeError(["PART_NOT_FOUND"], f"part {part_id} not found")
        return part

    def get(self, part_id: str) -> PartFlatRecord | None:
        self._require_part(part_id)
        return self.store.get_part_flat(part_id)

    # -- upload ---------------------------------------------------------------
    def ingest_stream(self, part_id: str, stream: IO[bytes], filename: str, principal: Any = None) -> PartFlatRecord:
        """Store an uploaded flat by sha256, parse it with E1 and attach it to the part."""
        self._require_part(part_id)
        self.flats_dir.mkdir(parents=True, exist_ok=True)
        incoming = self.flats_dir / f".incoming-{uuid.uuid4().hex}.part"
        h = hashlib.sha256()
        size = 0
        head = b""
        try:
            with open(incoming, "xb") as out:
                while True:
                    chunk = stream.read(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise FlatIntakeError(["FLAT_TOO_LARGE"], f"flat exceeds {self.max_bytes} bytes")
                    if len(head) < _HEAD_BYTES:
                        head += chunk[: _HEAD_BYTES - len(head)]
                    h.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            if size == 0:
                raise FlatIntakeError(["FLAT_EMPTY"], "the uploaded flat is empty")
            fmt, ext = detect_flat_format(filename, head)
            sha = h.hexdigest()
            final = self.flats_dir / f"{sha}{ext}"
            if final.is_file() and _sha256_file(final) == sha:
                incoming.unlink()
            else:
                os.replace(incoming, final)
                _fsync_dir(self.flats_dir)
        except BaseException:
            try:
                incoming.unlink()
            except OSError:
                pass
            raise
        rel = f"recipes/flats/{sha}{ext}"
        parsed = self.tool.parse(final, fmt)
        state = "UPLOADED" if parsed.get("status") == "PARSED" else "PARSE_FAILED"
        return self.store.put_part_flat(
            part_id,
            flat_sha256=sha,
            format=fmt,
            original_filename=Path(filename or "").name or None,
            stored_rel_path=rel,
            size_bytes=size,
            state=state,
            parse=parsed,
            uploaded_by=principal,
        )

    # -- propose --------------------------------------------------------------
    def _current_basis(self, part: Mapping[str, Any]) -> tuple[Path, str, list[dict[str, Any]], str]:
        cad_rel = part.get("cad_file_path")
        cad_sha = part.get("cad_file_hash")
        if not cad_rel or not cad_sha:
            raise FlatIntakeError(["CAD_MISSING"], "upload the folded STEP first")
        if Path(str(cad_rel)).suffix.lower() not in _STEP_EXTS:
            raise FlatIntakeError(["CAD_NOT_STEP"], "production recipes require a STEP CAD")
        cad_path = self.catalog.resolve_file(str(cad_rel))
        if not cad_path.is_file():
            raise FlatIntakeError(["CAD_MISSING"], "the catalog CAD file is missing")
        if _sha256_file(cad_path) != cad_sha:
            raise FlatIntakeError(["CAD_HASH_MISMATCH"], "the catalog CAD bytes changed")
        draft = self.catalog.get_vendor_setup_draft(str(part["id"]))
        if draft is None:
            raise FlatIntakeError(["DRAFT_MISSING"], "save the bend tagging (Vendor Setup draft) first")
        marks = binding_marks(draft, str(cad_sha))
        if not marks:
            raise FlatIntakeError(["NO_ACTIVE_BENDS"], "the draft has no active located bends")
        return cad_path, str(cad_sha), marks, binding_basis_sha256(str(cad_sha), marks)

    def _require_flat(self, part_id: str, expected_revision: int | None) -> PartFlatRecord:
        record = self.store.get_part_flat(part_id)
        if record is None:
            raise FlatIntakeError(["FLAT_NOT_FOUND"], "the part has no flat")
        if expected_revision is not None and record.revision != expected_revision:
            raise FlatIntakeError(
                ["FLAT_REVISION_CHANGED"], f"flat is at revision {record.revision}, not {expected_revision}"
            )
        return record

    def propose(self, part_id: str, expected_revision: int | None = None) -> PartFlatRecord:
        """Run E1's binding proposal for the current draft marks and store it (PROPOSED)."""
        part = self._require_part(part_id)
        record = self._require_flat(part_id, expected_revision)
        if record.state == "PARSE_FAILED":
            raise FlatIntakeError(["FLAT_PARSE_FAILED"], "the flat could not be parsed")
        cad_path, cad_sha, marks, basis = self._current_basis(part)
        flat_path = self.data_root / record.stored_rel_path
        if not flat_path.is_file() or _sha256_file(flat_path) != record.flat_sha256:
            raise FlatIntakeError(["FLAT_NOT_FOUND"], "the stored flat bytes are missing or changed")
        result = self.tool.propose(cad_path, flat_path, record.format, marks)
        status = result.get("status")
        labels = {m["bend_uid"]: m["label"] for m in marks}
        proposal = {
            **result,
            "basis_sha256": basis,
            "cad_sha256": cad_sha,
            "bend_labels": labels,
            "proposed_at": iso_utc(self.clock()),
        }
        if status == "PROPOSED":
            return self.store.update_part_flat(part_id, record.revision, state="PROPOSED", proposal=proposal,
                                               binding=None)
        # Record the failed attempt (the wizard shows it). An earlier operator
        # decision (CONFIRMED / DECLINED) is kept; a superseded proposal is not.
        keep_state = record.state if record.state in ("CONFIRMED", "DECLINED") else "UPLOADED"
        self.store.update_part_flat(part_id, record.revision, state=keep_state, proposal=proposal)
        if status == "RECONCILIATION_BLOCKED":
            raise FlatIntakeError(["RECONCILIATION_BLOCKED"], "draft marks do not reconcile with the CAD flange graph",
                                  {"reconciliation": result.get("reconciliation")})
        if status == "PARSE_FAILED":
            raise FlatIntakeError(["FLAT_PARSE_FAILED"], f"{result.get('reason')}: {result.get('detail', '')}")
        if status == "CAD_READ_FAILED":
            raise FlatIntakeError(["CAD_READ_FAILED"], f"{result.get('reason')}: {result.get('detail', '')}")
        raise FlatIntakeError(["FLAT_TOOL_FAILED"], f"unexpected proposal status {status!r}")

    # -- decide ---------------------------------------------------------------
    def set_binding(
        self,
        part_id: str,
        *,
        expected_revision: int,
        state: str,
        principal: Any,
        bend_line_map: Mapping[str, str] | None = None,
        mirrored: bool | None = None,
    ) -> PartFlatRecord:
        """Operator decision: ``OPERATOR_CONFIRMED`` (a bend-line map) or ``UNBOUND`` (keep unbound)."""
        part = self._require_part(part_id)
        record = self._require_flat(part_id, expected_revision)
        if record.state == "PARSE_FAILED":
            raise FlatIntakeError(["FLAT_PARSE_FAILED"], "the flat could not be parsed")
        try:
            who = released_by_dict(principal)
        except ValueError as exc:
            raise FlatIntakeError(["FLAT_BINDING_INVALID"], str(exc)) from None
        now = iso_utc(self.clock())
        if state == "UNBOUND":
            declined: dict[str, Any] = {"state": "UNBOUND", "decided_by": who, "decided_at": now}
            return self.store.update_part_flat(part_id, expected_revision, state="DECLINED", binding=declined)
        if state != "OPERATOR_CONFIRMED":
            raise FlatIntakeError(["FLAT_BINDING_INVALID"], "state must be OPERATOR_CONFIRMED or UNBOUND")
        _cad_path, cad_sha, marks, basis = self._current_basis(part)
        proposal = record.proposal if (record.proposal or {}).get("status") == "PROPOSED" else None
        if proposal is not None and proposal.get("basis_sha256") != basis:
            proposal = None  # made against other marks/CAD: not a basis for this confirmation
        prop = (proposal or {}).get("proposal") or {}
        chosen = dict(bend_line_map) if bend_line_map is not None else dict(prop.get("bend_line_map") or {})
        if not chosen:
            raise FlatIntakeError(["FLAT_BINDING_INVALID"], "confirm a non-empty bend-line map (or propose first)")
        valid_lines = {str(b.get("line_id")) for b in ((record.parse or {}).get("summary") or {}).get("bend_lines", [])}
        valid_lines |= {str(b.get("line_id")) for b in prop.get("lines", []) if isinstance(b, Mapping)}
        uids = {m["bend_uid"] for m in marks}
        bad_lines = sorted(k for k in chosen if k not in valid_lines)
        bad_uids = sorted(v for v in chosen.values() if v not in uids)
        if bad_lines or bad_uids:
            raise FlatIntakeError(
                ["FLAT_BINDING_INVALID"],
                f"unknown flat lines {bad_lines} or bend identities {bad_uids}",
            )
        if len(set(chosen.values())) != len(chosen):
            raise FlatIntakeError(["FLAT_BINDING_INVALID"], "each bend may be bound to at most one flat line")
        # Mirror flag: the operator's, else the proposal's; unknown (None) for a manual map with no proposal.
        mirror: bool | None = mirrored if mirrored is not None else (bool(prop.get("mirrored")) if prop else None)
        same_as_proposal = bool(prop) and chosen == dict(prop.get("bend_line_map") or {}) and mirror == bool(
            prop.get("mirrored")
        )
        binding: dict[str, Any] = {
            "state": "OPERATOR_CONFIRMED",
            "bend_line_map": dict(sorted(chosen.items())),
            "mirrored": mirror,
            "method": "GRAPH_MATCH_RIGID_FIT" if same_as_proposal else "MANUAL",
            "basis_sha256": basis,
            "cad_sha256": cad_sha,
            "confirmed_by": who,
            "confirmed_at": now,
        }
        return self.store.update_part_flat(part_id, expected_revision, state="CONFIRMED", binding=binding)

    def delete(self, part_id: str, expected_revision: int | None) -> bool:
        """Detach the flat at ``expected_revision`` (required: a stale tab must not
        delete a flat it has not seen). The revision counter continues afterwards."""
        self._require_part(part_id)
        if not isinstance(expected_revision, int) or isinstance(expected_revision, bool):
            raise FlatIntakeError(["FLAT_REVISION_REQUIRED"], "deleting a flat needs its expected_revision")
        try:
            return self.store.delete_part_flat(part_id, expected_revision)
        except RecipeStoreError as exc:
            raise FlatIntakeError(["FLAT_REVISION_CHANGED"], exc.message) from None


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


def release_flat_ref(
    record: PartFlatRecord | None,
    *,
    draft: Mapping[str, Any] | None,
    cad_sha256: str | None,
    confirmation: str,
    data_root: Path,
) -> tuple[Any, Path | None, dict[str, list[str]]]:
    """``(FlatRef | None, absolute flat path | None, {code: [details]})`` for a release.

    No flat attached -> ``(None, None, {})``. Refusal codes are described in
    the module docstring.
    """
    from contracts.production.part_recipe import FlatBindingV1, FlatRef

    if record is None:
        return None, None, {}
    problems: dict[str, list[str]] = {}
    path = Path(data_root) / record.stored_rel_path
    if record.state == "PARSE_FAILED":
        reason = (record.parse or {}).get("reason")
        problems.setdefault("FLAT_PARSE_FAILED", []).append(f"the attached flat could not be parsed ({reason})")
    if not path.is_file():
        problems.setdefault("FLAT_MISSING", []).append("the stored flat file is missing")
    elif cached_sha256(path) != record.flat_sha256:
        problems.setdefault("FLAT_HASH_MISMATCH", []).append("the stored flat bytes changed")
    basis = binding_basis_sha256(cad_sha256, binding_marks(draft, cad_sha256)) if cad_sha256 else None
    binding: Any = None
    if record.state == "DECLINED":
        binding = FlatBindingV1(state="UNBOUND")
    elif record.state == "CONFIRMED":
        stored = record.binding or {}
        if basis is None or stored.get("basis_sha256") != basis:
            problems.setdefault("FLAT_BINDING_STALE", []).append(
                "the flat binding was confirmed against other CAD or bend marks; propose and confirm again"
            )
        else:
            binding = FlatBindingV1(
                state="OPERATOR_CONFIRMED",
                bend_line_map=stored.get("bend_line_map") or {},
                method=stored.get("method") or "MANUAL",
                mirrored=stored.get("mirrored"),
                confirmed_by=stored.get("confirmed_by"),
                confirmed_at=stored.get("confirmed_at"),
            )
    elif record.state == "PROPOSED" and confirmation == "DEV_AUTO_CONFIRMED":
        proposal = record.proposal or {}
        prop = proposal.get("proposal") or {}
        if basis is None or proposal.get("basis_sha256") != basis:
            problems.setdefault("FLAT_BINDING_STALE", []).append("the flat proposal was made against other marks")
        elif not prop.get("bend_line_map"):
            problems.setdefault("FLAT_BINDING_UNCONFIRMED", []).append("the proposal bound no bend line")
        else:
            binding = FlatBindingV1(
                state="AUTO_PROPOSED",
                bend_line_map=prop["bend_line_map"],
                method="GRAPH_MATCH_RIGID_FIT",
                mirrored=bool(prop.get("mirrored")),
            )
    elif record.state in ("UPLOADED", "PROPOSED"):
        problems.setdefault("FLAT_BINDING_UNCONFIRMED", []).append(
            "the flat binding is not confirmed: confirm the proposed bend lines or keep the flat unbound"
        )
    if problems or binding is None:
        return None, None, problems
    ref = FlatRef(
        sha256=record.flat_sha256,
        original_filename=record.original_filename,
        stored_rel_path=record.stored_rel_path,
        format=record.format,  # type: ignore[arg-type]
        binding=binding,
    )
    return ref, path, {}
