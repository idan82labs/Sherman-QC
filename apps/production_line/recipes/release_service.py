"""The recipe release gate (C05): catalog part + confirmed draft -> compiled, immutable recipe.

:func:`release` turns the catalog part (STEP CAD, drawing), its confirmed
Vendor Setup draft, the attached flat (optional) and a ``ReleaseRequestV1``
into an immutable ``PartRecipeV1`` version whose E1 ``CompiledPartV1`` is
hash-bound to it. Production recipes require STEP (plan §10 amendment 1).

Steps (every refusal is a :class:`ReleaseGateError` with **all** applicable
codes, sorted, and human-readable details):

1. Part exists (PART_NOT_FOUND, 404). The release request validates
   (TOLERANCE_INVALID / DRAFT_INVALID) and names the stored release-draft
   revision when it names one (RELEASE_DRAFT_REVISION_CHANGED, 409).
2. Canonical SKU (SKU_INVALID); registry and catalog uniqueness on the
   canonical form (SKU_COLLISION, 409).
3. The draft passes the Vendor Setup validator
   (:meth:`CatalogPort.validate_vendor_draft`: SEQUENCE_NOT_CONFIRMED,
   DRAFT_STALE, DRAFT_INVALID).
4. CAD: present (CAD_MISSING), STEP (CAD_NOT_STEP), bytes hash to the
   catalog's ``cad_file_hash`` (CAD_HASH_MISMATCH).
5. Drawing: present (DRAWING_MISSING), a supported format
   (DRAWING_FORMAT_UNSUPPORTED), bytes hash to ``drawing_file_hash``
   (DRAWING_HASH_MISMATCH).
6. Flat (when attached, :func:`.flat_intake.release_flat_ref`):
   FLAT_PARSE_FAILED, FLAT_MISSING, FLAT_HASH_MISMATCH,
   FLAT_BINDING_UNCONFIRMED, FLAT_BINDING_STALE.
7. ``domains.production.recipe.builder.build_part_recipe`` with version
   ``max + 1`` (every C01 code: DRAFT_STALE, UNCONFIRMED_ACTIVE_MARKS,
   TOLERANCES_MISSING, ...). ``supersedes_version`` is the active version.
8. Content unchanged against the active version (``content_sha256``) ->
   outcome ``unchanged`` (200); nothing is compiled or stored.
9. The CAD and drawing are frozen by content address under
   ``recipes/cad/<sha><ext>`` and ``recipes/drawings/<sha><ext>`` (tmp +
   fsync + rename, idempotent, re-hashed while copying).
10. E1 compile, supervised (:class:`CompileLauncher`): ``python -m
    domains.production.engine.compile --recipe --cad [--flat] --out``,
    120 s and 4 GiB, cwd = the engine bundle (or the source root in a dev
    checkout). COMPILE_TIMEOUT, COMPILE_FAILED (crash, invalid output, E1
    input error), COMPILE_BLOCKED plus every E1 blocker code
    (MARK_NOT_IN_FLANGE_GRAPH, UNMARKED_BEND, MARK_AMBIGUOUS, ...), and
    COMPILE_INCONSISTENT when ``check_compiled_against_recipe`` finds a
    problem. A refused compile's report (issues, unmarked graph bends) is
    returned to the wizard and logged in ``compile_attempts``.
11. The compile output is installed at ``recipes/compiled/<recipe_sha256>/``
    and the recipe is inserted with its compile record, READY expectations
    (the compiled nominals) and its ``benchmark_references`` row with
    per-bend nominals, in one store transaction (CONCURRENT_RELEASE when
    another release won; the installed compile is then removed).
12. Line events RECIPE_RELEASED and RECIPE_COMPILED (best effort: a failure
    is logged and never rolls the release back), and the release listeners
    (scan-resolver cache invalidation; bound methods are held weakly, so a
    discarded resolver is dropped instead of being notified forever).

Every compile attempt is logged with the attempted recipe's
``content_sha256``: the onboarding checklist blocks only on a BLOCKED compile
of the very content a release would make now (a BLOCKED result is
deterministic); FAILED / TIMEOUT / INCONSISTENT attempts are advisory.

The HTTP layer (``production_recipe_service``) additionally refuses any
``confirmation`` other than OPERATOR_CONFIRMED (CONFIRMATION_NOT_ALLOWED):
DEV_AUTO_CONFIRMED recipes are built through this library by E4 only.

Releases of one part are serialised in-process; across processes the store's
``max + 1`` rule refuses the loser.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import uuid
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from apps.production_line.recipes.flat_intake import release_flat_ref, released_by_dict
from apps.production_line.recipes.hashing import cached_sha256
from apps.production_line.recipes.store import (
    CompiledRecord,
    RecipeStore,
    RecipeStoreError,
    iso_utc,
    utc_now,
)

__all__ = [
    "C05_RELEASE_CODES",
    "RELEASE_GATE_CODES",
    "CONFLICT_CODES",
    "CODE_MESSAGES",
    "ReleaseGateError",
    "CompileOutcome",
    "CompileLauncher",
    "ReleaseOutcome",
    "PreparedRelease",
    "prepare_release",
    "release",
    "install_compiled",
    "archive_compiled",
    "compiled_history_rel_dir",
    "cached_sha256",
    "engine_root_for",
    "expectations_document",
    "benchmark_nominals",
    "add_release_listener",
    "release_listener_count",
    "notify_release_listeners",
    "remove_release_listener",
    "candidate_angle_convention",
    "drawing_format",
]

_LOG = logging.getLogger(__name__)

#: Codes the C05 gate adds to the C01 builder codes and the E1 blocker codes.
C05_RELEASE_CODES: frozenset[str] = frozenset(
    {
        "PART_NOT_FOUND",
        "CAD_NOT_STEP",
        "DRAWING_FORMAT_UNSUPPORTED",
        "FLAT_MISSING",
        "FLAT_BINDING_UNCONFIRMED",
        "FLAT_BINDING_STALE",
        "COMPILE_BLOCKED",
        "COMPILE_FAILED",
        "COMPILE_TIMEOUT",
        "COMPILE_INCONSISTENT",
        "CONFIRMATION_NOT_ALLOWED",
    }
)


def _gate_codes() -> frozenset[str]:
    from contracts.production.compiled_part import COMPILE_BLOCKER_CODES
    from contracts.production.part_recipe import RECIPE_BUILD_REASON_CODES

    return frozenset(RECIPE_BUILD_REASON_CODES | COMPILE_BLOCKER_CODES | C05_RELEASE_CODES)


#: Every code a release refusal may carry: C01 builder codes, E1 blocker codes, C05 codes.
RELEASE_GATE_CODES: frozenset[str] = _gate_codes()
#: Refusals that mean "someone else changed it; reload" (HTTP 409).
CONFLICT_CODES: frozenset[str] = frozenset(
    {"DRAFT_REVISION_CHANGED", "RELEASE_DRAFT_REVISION_CHANGED", "CONCURRENT_RELEASE", "SKU_COLLISION"}
)

#: Operator-facing text per code (English; the wizard localises by code).
CODE_MESSAGES: dict[str, str] = {
    "PART_NOT_FOUND": "The catalog part does not exist.",
    "DRAFT_MISSING": "Tag the bends (Vendor Setup) and save the draft first.",
    "DRAFT_STALE": "The CAD, drawing or bend candidates changed after the draft was saved; reload and confirm again.",
    "DRAFT_INVALID": "The Vendor Setup draft is invalid.",
    "DRAFT_REVISION_CHANGED": "The Vendor Setup draft changed in another session; reload before releasing.",
    "SEQUENCE_NOT_CONFIRMED": "Confirm the bending order before releasing.",
    "NO_ACTIVE_BENDS": "The draft has no active bend.",
    "UNCONFIRMED_ACTIVE_MARKS": "Every active bend mark must be review-confirmed.",
    "MARK_WITHOUT_LINE": "Every active bend mark needs a located bend line.",
    "MANUAL_MARK_MISSING_NOMINAL_ANGLE": "Manual bend marks need a nominal included angle.",
    "DUPLICATE_BEND_IDENTITY": "Two marks describe the same bend.",
    "CAD_MISSING": "Upload the folded STEP CAD.",
    "CAD_NOT_STEP": "Production recipes require a STEP CAD (.step / .stp).",
    "CAD_HASH_MISMATCH": "The CAD bytes do not match the catalog hash.",
    "DRAWING_MISSING": "Upload the part drawing (PDF or image).",
    "DRAWING_FORMAT_UNSUPPORTED": "The drawing format is not supported.",
    "DRAWING_HASH_MISMATCH": "The drawing bytes do not match the catalog hash.",
    "FLAT_MISSING": "The attached flat file is missing.",
    "FLAT_HASH_MISMATCH": "The attached flat file changed.",
    "FLAT_PARSE_FAILED": "The attached flat could not be parsed.",
    "FLAT_BINDING_UNCONFIRMED": "Confirm the flat's bend lines, or keep the flat unbound.",
    "FLAT_BINDING_STALE": "The flat binding was confirmed against other bend marks or CAD; confirm it again.",
    "FLAT_BINDING_UNKNOWN_BEND": "The flat binding names a bend that is not in the recipe.",
    "SKU_INVALID": "The part number is not a valid SKU.",
    "SKU_COLLISION": "Another part already uses this SKU (case and spacing are ignored).",
    "TOLERANCES_MISSING": "Set the default tolerances.",
    "TOLERANCE_INVALID": "A tolerance is invalid.",
    "OVERRIDE_UNKNOWN_BEND": "A per-bend override names a bend that is not active.",
    "OVERRIDE_UNKNOWN_MEASUREMENT": "A measurement override names an unknown measurement.",
    "MEASUREMENT_INVALID": "A general measurement is invalid.",
    "CONCURRENT_RELEASE": "Another release of this SKU happened at the same time; reload.",
    "RELEASE_DRAFT_REVISION_CHANGED": "The release settings changed in another session; reload.",
    "COMPILE_BLOCKED": "The onboarding compile is blocked; fix the listed issues.",
    "COMPILE_FAILED": "The onboarding compile failed.",
    "COMPILE_TIMEOUT": "The onboarding compile did not finish in time.",
    "COMPILE_INCONSISTENT": "The compile disagrees with the recipe (refused).",
    "CONFIRMATION_NOT_ALLOWED": "Only operator-confirmed recipes can be released here.",
    "MARK_NOT_IN_FLANGE_GRAPH": "A bend mark matches no bend in the CAD; remove it or correct its location.",
    "UNMARKED_BEND": "The CAD has a bend with no mark; add or confirm its mark.",
    "MARK_AMBIGUOUS": "A bend mark matches several CAD bends (or several marks one bend).",
    "DATUM_PANEL_UNDETERMINED": "The CAD flange graph has no usable datum panel.",
    "FLANGE_GRAPH_NOT_A_TREE": "The CAD flange graph is unusable.",
    "CAD_READ_FAILED": "The STEP file could not be read.",
    "RECIPE_INVALID": "The recipe did not validate in the compile.",
    "THICKNESS_INCONSISTENT": "The recipe thickness disagrees with the CAD.",
    "FLAT_BINDING_INCONSISTENT": "The confirmed flat binding disagrees with the CAD development.",
    "FLAT_DEVELOPED_LENGTH_MISMATCH": "The flat's developed lengths disagree with the CAD.",
    "STAGE_MODEL_FAILED": "A stage model could not be built.",
}

_TOLERANCE_LOC_NAMES = frozenset(
    {"plus", "minus", "angle_deg", "flange_mm", "flange_a_mm", "flange_b_mm", "bend_line_position_mm",
     "backgauge_mm", "general_mm", "surface_profile_mm", "default_tolerances", "tolerance"}
)
_STEP_EXTS = (".step", ".stp")
_DRAWING_FORMATS: dict[str, str] = {
    ".pdf": "PDF",
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".dxf": "DXF",
    ".dwg": "DWG",
}
_CHUNK = 1024 * 1024


class ReleaseGateError(Exception):
    """Release refused. ``codes`` are sorted unique :data:`RELEASE_GATE_CODES`;
    ``details`` maps a code to specifics; ``compile_report`` is the E1 report
    (issues, unmarked graph bends) when the compile ran."""

    def __init__(
        self,
        codes: Sequence[str] | Any,
        details: Mapping[str, Sequence[str]] | None = None,
        compile_report: Mapping[str, Any] | None = None,
    ) -> None:
        unique = tuple(sorted(set(codes)))
        unknown = set(unique) - RELEASE_GATE_CODES
        if unknown:
            raise ValueError(f"unknown release gate codes: {sorted(unknown)}")
        if not unique:
            raise ValueError("ReleaseGateError needs at least one code")
        self.codes = unique
        self.details: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in sorted((details or {}).items())}
        self.compile_report = dict(compile_report) if compile_report is not None else None
        super().__init__(", ".join(unique))

    @property
    def http_status(self) -> int:
        if "PART_NOT_FOUND" in self.codes:
            return 404
        if set(self.codes) & CONFLICT_CODES:
            return 409
        return 422

    def to_detail(self) -> dict[str, Any]:
        """The HTTP ``detail`` body: ``{codes, messages, details, compile?}``."""
        out: dict[str, Any] = {
            "codes": list(self.codes),
            "messages": {c: CODE_MESSAGES.get(c, c) for c in self.codes},
            "details": {k: list(v) for k, v in self.details.items()},
        }
        if self.compile_report is not None:
            out["compile"] = self.compile_report
        return out


class _Problems:
    def __init__(self) -> None:
        self.details: dict[str, list[str]] = {}

    def add(self, code: str, detail: str = "") -> None:
        bucket = self.details.setdefault(code, [])
        if detail and detail not in bucket:
            bucket.append(detail)

    def merge(self, other: Mapping[str, Sequence[str]]) -> None:
        for code, items in other.items():
            if not items:
                self.add(code)
            for item in items:
                self.add(code, item)

    def __bool__(self) -> bool:
        return bool(self.details)

    def error(self, compile_report: Mapping[str, Any] | None = None) -> ReleaseGateError:
        return ReleaseGateError(self.details.keys(), self.details, compile_report)


# ---------------------------------------------------------------------------
# Release listeners (e.g. the scan resolver's cache)
# ---------------------------------------------------------------------------

#: Listener references: a bound method is held through ``weakref.WeakMethod``
#: (its object may be collected), anything else strongly.
_listeners: list[Callable[[], Callable[[Any], None] | None]] = []
_listeners_lock = threading.Lock()


def _ref(callback: Callable[[Any], None]) -> Callable[[], Callable[[Any], None] | None]:
    if getattr(callback, "__self__", None) is not None and hasattr(callback, "__func__"):
        return weakref.WeakMethod(callback)  # type: ignore[arg-type]
    return lambda: callback


def _live_listeners() -> list[Callable[[Any], None]]:
    """Dereference the listeners, dropping collected ones (caller holds the lock)."""
    live: list[Callable[[Any], None]] = []
    kept = []
    for ref in _listeners:
        callback = ref()
        if callback is not None:
            live.append(callback)
            kept.append(ref)
    _listeners[:] = kept
    return live


def add_release_listener(callback: Callable[[Any], None]) -> None:
    """Call ``callback(recipe)`` after every release or retirement (exceptions are logged).

    A bound method is held weakly: the listener disappears with its object.
    """
    with _listeners_lock:
        if callback not in _live_listeners():
            _listeners.append(_ref(callback))


def remove_release_listener(callback: Callable[[Any], None]) -> None:
    with _listeners_lock:
        _live_listeners()
        _listeners[:] = [ref for ref in _listeners if ref() != callback]


def release_listener_count() -> int:
    """Live listeners (diagnostics / tests)."""
    with _listeners_lock:
        return len(_live_listeners())


def notify_release_listeners(recipe: Any) -> None:
    with _listeners_lock:
        callbacks = _live_listeners()
    for callback in callbacks:
        try:
            callback(recipe)
        except Exception:
            _LOG.exception("release listener failed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def candidate_angle_convention(cad_path: str) -> str:
    """NATIVE_SWEEP for STEP/IGES candidates, MESH_INCLUDED_FOLDED for mesh CAD."""
    ext = Path(cad_path).suffix.lower()
    return "NATIVE_SWEEP" if ext in (".step", ".stp", ".iges", ".igs") else "MESH_INCLUDED_FOLDED"


def drawing_format(path: str) -> str | None:
    """DrawingRef format of a drawing path (None when unsupported)."""
    return _DRAWING_FORMATS.get(Path(path).suffix.lower())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_root_for(settings: Any) -> Path:
    """The E1 root: the engine bundle when it holds a manifest, else the running source root."""
    bundle = Path(settings.engine_bundle_dir)
    return bundle if (bundle / "bundle_manifest.json").is_file() else Path(settings.repo_root)


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


def _freeze_file(src: Path, dest: Path, expected_sha: str) -> bool:
    """Copy ``src`` to the content-addressed ``dest`` (tmp + fsync + rename).

    Returns False when the bytes read do not hash to ``expected_sha`` (the
    source changed); an existing ``dest`` with the right hash is kept.
    """
    if dest.is_file() and _sha256_file(dest) == expected_sha:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    h = hashlib.sha256()
    try:
        with open(src, "rb") as fin, open(tmp, "xb") as fout:
            for chunk in iter(lambda: fin.read(_CHUNK), b""):
                h.update(chunk)
                fout.write(chunk)
            fout.flush()
            os.fsync(fout.fileno())
        if h.hexdigest() != expected_sha:
            tmp.unlink()
            return False
        os.replace(tmp, dest)
        _fsync_dir(dest.parent)
        return True
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _validation_codes(exc: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for err in exc.errors():
        loc = {str(p) for p in err.get("loc", ())}
        code = "TOLERANCE_INVALID" if loc & _TOLERANCE_LOC_NAMES else "DRAFT_INVALID"
        out.setdefault(code, []).append(f"release request {'.'.join(str(p) for p in err.get('loc', ()))}: "
                                        f"{err.get('msg', '')}"[:300])
    return out


def _principal_audit(principal: Any) -> Any:
    if principal is None:
        return None
    if hasattr(principal, "audit") and callable(principal.audit):
        return principal.audit()
    if isinstance(principal, Mapping):
        return {"kind": principal.get("kind", "user"), "id": principal.get("user_id", principal.get("id")),
                "role": principal.get("role")}
    return None


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


@dataclass
class CompileOutcome:
    """Result of one supervised E1 compile.

    ``status`` COMPILED / BLOCKED / FAILED. ``out_dir`` holds the compile
    output (``compiled_part.json``, ``stages/``, ``compile.log``) when one was
    written; ``staging_dir`` is the scratch directory to remove afterwards.
    """

    status: str
    codes: tuple[str, ...] = ()
    details: dict[str, list[str]] = field(default_factory=dict)
    compiled: Any = None
    out_dir: Path | None = None
    staging_dir: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> None:
        if self.staging_dir is not None:
            shutil.rmtree(self.staging_dir, ignore_errors=True)


def _compile_report(compiled: Any, recipe: Any) -> dict[str, Any]:
    """Wizard-facing summary of a CompiledPartV1: issues, unmarked graph bends, warnings."""
    labels = {b.bend_uid: b.label for b in recipe.bends}
    issues = []
    for issue in compiled.issues:
        issues.append(
            {
                "code": issue.code,
                "severity": issue.severity,
                "bend_uid": issue.bend_uid,
                "bend_label": labels.get(issue.bend_uid or ""),
                "graph_bend_id": issue.graph_bend_id,
                "detail": issue.detail,
            }
        )
    return {
        "status": compiled.status,
        "compiled_part_sha256": compiled.compiled_part_sha256,
        "compiler_version": compiled.compiler.version,
        "blocker_codes": list(compiled.blocker_codes),
        "warning_codes": sorted({i.code for i in compiled.issues if i.severity == "WARNING"}),
        "issues": issues,
        "unmarked_graph_bends": [g.model_dump(mode="json") for g in compiled.unmarked_graph_bends],
    }


@dataclass(frozen=True)
class CompileLauncher:
    """Runs the E1 compile CLI under the C02 supervisor (ENGINE_IO §3).

    ``engine_root`` is the subprocess cwd and ``PYTHONPATH`` root (the pinned
    engine bundle; in a dev checkout the source root). The budget is 120 s
    and 4 GiB; the thread caps are 1.
    """

    engine_root: Path
    python_exe: Path
    module: str = "domains.production.engine.compile"
    timeout_s: float = 120.0
    rss_limit_bytes: int = 4 * 1024**3
    shop_k: float | None = None
    config_path: Path | None = None
    poll_s: float = 0.25
    kill_grace_s: float = 3.0
    program_status_path: Path | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Any) -> "CompileLauncher":
        """Engine bundle when it holds a manifest, else the running source root (:func:`engine_root_for`)."""
        return cls(
            engine_root=engine_root_for(settings),
            python_exe=Path(settings.python_exe),
            rss_limit_bytes=min(int(settings.rss_limit_bytes), 4 * 1024**3),
            program_status_path=Path(settings.program_status_path),
        )

    def env(self) -> dict[str, str]:
        from apps.production_line.runner.runner import ENV_PASSTHROUGH, THREAD_ENV_VARS

        env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
        root = str(Path(self.engine_root))
        env["PYTHONPATH"] = os.pathsep.join([root, str(Path(root) / "backend")])
        for key in THREAD_ENV_VARS:
            env[key] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLBACKEND"] = "Agg"
        if self.program_status_path is not None:
            env["SHERMAN_QC_F1_PROGRAM_STATUS_PATH"] = str(self.program_status_path)
        env.update({str(k): str(v) for k, v in self.extra_env.items()})
        return env

    def compile(self, recipe: Any, *, cad_path: Path, flat_path: Path | None, staging_root: Path) -> CompileOutcome:
        """Compile ``recipe`` and classify the outcome (module docstring, step 10)."""
        from pydantic import ValidationError

        from apps.production_line.runner.supervisor import run_supervised
        from apps.production_line.runner.worker_protocol import (
            error_message,
            parse_protocol_line,
        )
        from contracts.production.canonical import canonical_json_bytes
        from contracts.production.compiled_part import (
            CompiledPartV1,
            check_compiled_against_recipe,
        )

        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / f".staging-{uuid.uuid4().hex}"
        staging.mkdir()
        recipe_path = staging / "recipe.json"
        recipe_path.write_bytes(canonical_json_bytes(recipe))
        out_dir = staging / "out"
        cmd = [str(self.python_exe), "-m", self.module, "--recipe", str(recipe_path), "--cad", str(cad_path),
               "--out", str(out_dir)]  # fmt: skip
        if flat_path is not None:
            cmd += ["--flat", str(flat_path)]
        if self.shop_k is not None:
            cmd += ["--shop-k", repr(float(self.shop_k))]
        if self.config_path is not None:
            cmd += ["--config", str(self.config_path)]
        tail: list[str] = []
        errors: list[dict[str, Any]] = []

        def _line(text: str) -> None:
            tag, payload = parse_protocol_line(text)
            if tag == "ERROR" and payload is not None:
                errors.append(payload)
            elif tag is None:
                tail.append(text)
                del tail[:-20]

        outcome = CompileOutcome(status="FAILED", staging_dir=staging)
        try:
            res = run_supervised(
                cmd,
                env=self.env(),
                cwd=self.engine_root,
                timeout_s=self.timeout_s,
                rss_limit_bytes=self.rss_limit_bytes,
                poll_s=self.poll_s,
                on_line=_line,
                log_path=staging / "compile_step.log",
                kill_grace_s=self.kill_grace_s,
                identity_tag=f"c05-compile-{staging.name}",
            )
        except BaseException:
            outcome.cleanup()  # e.g. the interpreter could not be spawned: leave no staging behind
            raise
        outcome.report = {
            "exit_code": res.exit_code,
            "duration_s": round(res.duration_s, 3),
            "peak_rss_bytes": res.peak_rss_bytes,
        }

        def failed(code: str, detail: str) -> CompileOutcome:
            outcome.status = "FAILED"
            outcome.codes = (code,)
            outcome.details = {code: [detail[:1000]]}
            outcome.report.update(status="FAILED", detail=detail[:1000], log_tail=tail[-10:])
            return outcome

        if res.timed_out:
            return failed("COMPILE_TIMEOUT", f"the compile exceeded {self.timeout_s:g} s")
        if res.rss_exceeded:
            return failed("COMPILE_FAILED", f"the compile exceeded its memory limit ({self.rss_limit_bytes} bytes)")
        compiled_json = out_dir / "compiled_part.json"
        if res.exit_code == 2 and not compiled_json.is_file():
            code, msg = None, ""
            try:
                import json

                data = json.loads((out_dir / "compile_error.json").read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    code, msg = error_message(data)
            except (OSError, ValueError):
                pass
            if code is None and errors:
                code, msg = error_message(errors[-1])
            outcome = failed("COMPILE_FAILED", f"E1 refused the input: {code or 'INVALID_INPUT'}: {msg}")
            if code in ("CAD_HASH_MISMATCH", "FLAT_HASH_MISMATCH"):
                outcome.codes = tuple(sorted({"COMPILE_FAILED", code}))
                outcome.details[code] = [msg or code]
            return outcome
        if res.exit_code not in (0, 2):
            detail = error_message(errors[-1])[1] if errors else " | ".join(tail[-5:])
            return failed("COMPILE_FAILED", f"the compile exited {res.exit_code}: {detail}")
        if not compiled_json.is_file():
            return failed("COMPILE_FAILED", "the compile wrote no compiled_part.json")
        try:
            compiled = CompiledPartV1.model_validate_json(compiled_json.read_bytes())
        except (ValidationError, ValueError) as exc:
            return failed("COMPILE_FAILED", f"compiled_part.json does not validate: {str(exc).splitlines()[0]}")
        expected = "COMPILED" if res.exit_code == 0 else "BLOCKED"
        if compiled.status != expected:
            return failed("COMPILE_FAILED", f"compile exit {res.exit_code} but status {compiled.status}")
        outcome.compiled = compiled
        outcome.out_dir = out_dir
        outcome.report.update(_compile_report(compiled, recipe))
        problems = check_compiled_against_recipe(compiled, recipe)
        if problems:
            outcome.status = "FAILED"
            outcome.codes = ("COMPILE_INCONSISTENT",)
            outcome.details = {"COMPILE_INCONSISTENT": list(problems)}
            outcome.report["status"] = "FAILED"
            outcome.report["consistency_problems"] = list(problems)
            return outcome
        for stage in compiled.stages:
            for ref in (stage.sample_cloud, stage.display_mesh):
                target = out_dir / ref.rel_path
                if not target.is_file() or _sha256_file(target) != ref.sha256:
                    return failed("COMPILE_FAILED", f"stage file {ref.rel_path} missing or not matching its sha256")
        if compiled.status == "BLOCKED":
            outcome.status = "BLOCKED"
            outcome.codes = tuple(sorted({"COMPILE_BLOCKED", *compiled.blocker_codes}))
            details: dict[str, list[str]] = {}
            for issue in outcome.report["issues"]:
                if issue["severity"] == "BLOCKER":
                    who = issue["bend_label"] or issue["bend_uid"] or issue["graph_bend_id"] or ""
                    details.setdefault(issue["code"], []).append(f"{who}: {issue['detail']}".strip(": "))
            outcome.details = details
            return outcome
        outcome.status = "COMPILED"
        return outcome


def compiled_history_rel_dir(recipe_sha256: str, compiled_part_sha256: str) -> str:
    """Data-root-relative directory of a kept, non-installed compile."""
    for value in (recipe_sha256, compiled_part_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("compile history paths are built from sha256 hex digests")
    return f"recipes/compiled_history/{recipe_sha256}/{compiled_part_sha256}"


def archive_compiled(src_dir: Path, data_root: Path, recipe_sha256: str, compiled_part_sha256: str) -> str:
    """Move a compile directory to :func:`compiled_history_rel_dir` (same file
    system: a rename); returns the data-root-relative ``compiled_part.json``
    path. When that compile is already archived, the archive is kept and
    ``src_dir`` is removed."""
    rel_dir = compiled_history_rel_dir(recipe_sha256, compiled_part_sha256)
    final = Path(data_root) / rel_dir
    final.parent.mkdir(parents=True, exist_ok=True)
    for path in Path(src_dir).rglob("*"):
        if path.is_symlink():
            raise ValueError(f"compile output contains a symlink: {path.name}")
    if (final / "compiled_part.json").is_file():
        shutil.rmtree(src_dir, ignore_errors=True)
    else:
        if final.exists():
            shutil.rmtree(final)
        os.replace(src_dir, final)
        _fsync_dir(final.parent)
    return f"{rel_dir}/compiled_part.json"


def install_compiled(out_dir: Path, data_root: Path, recipe_sha256: str, compiled_part_sha256: str) -> str:
    """Install a compile output at ``recipes/compiled/<recipe_sha256>/``; returns the
    data-root-relative ``compiled_part.json`` path. An existing install with the
    same ``compiled_part_sha256`` is kept; any other is replaced atomically."""
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.layout import compiled_part_rel_dir

    rel_dir = compiled_part_rel_dir(recipe_sha256)
    final = Path(data_root) / rel_dir
    final.parent.mkdir(parents=True, exist_ok=True)
    for path in out_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"compile output contains a symlink: {path.name}")
    with open(out_dir / "compiled_part.json", "rb") as fh:
        os.fsync(fh.fileno())
    if final.exists():
        try:
            existing = CompiledPartV1.model_validate_json((final / "compiled_part.json").read_bytes())
            if existing.compiled_part_sha256 == compiled_part_sha256:
                shutil.rmtree(out_dir, ignore_errors=True)
                return f"{rel_dir}/compiled_part.json"
        except (OSError, ValueError):
            pass
        old = final.with_name(f".{final.name}.old-{uuid.uuid4().hex}")
        os.replace(final, old)
        os.replace(out_dir, final)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(out_dir, final)
    _fsync_dir(final.parent)
    return f"{rel_dir}/compiled_part.json"


# ---------------------------------------------------------------------------
# Expectations and benchmark nominals (from the compile)
# ---------------------------------------------------------------------------


def _tol(t: Any) -> dict[str, Any] | None:
    return None if t is None else t.model_dump(mode="json")


def expectations_document(recipe: Any, compiled: Any) -> dict[str, Any]:
    """Nominal expectations the line will compare against (the compiled nominals).

    The compiled flange-graph values are authoritative (ENGINE_IO §2); the
    recipe's own ``nominal_angle`` is shown next to them so the wizard can
    flag a convention disagreement.
    """
    recipe_bends = {b.bend_uid: b for b in recipe.bends}
    bends = []
    for b in compiled.bends:
        rb = recipe_bends[b.bend_uid]
        bends.append(
            {
                "bend_uid": b.bend_uid,
                "label": b.label,
                "cad_label": rb.cad_label,
                "sequence_index": b.sequence_index,
                "machine_move_index": b.machine_move_index,
                "graph_bend_id": b.graph_bend_id,
                "nominal_included_deg": b.nominal_included_deg,
                "recipe_included_deg": rb.nominal_angle.included_deg,
                "recipe_angle_source": rb.nominal_angle.source,
                "sweep_deg": b.sweep_deg,
                "inner_radius_mm": b.inner_radius_mm,
                "thickness_mm": b.thickness_mm,
                "direction": b.direction,
                "supported": b.supported,
                "unsupported_feature": b.unsupported_feature,
                "oml_virtual_sharp": b.loci.oml_virtual_sharp.model_dump(mode="json"),
                "sides": [s.model_dump(mode="json") for s in b.sides],
            }
        )
    characteristics = [c.model_dump(mode="json") for c in compiled.characteristics]
    return {
        "schema": "sherman_recipe_expectations_v1",
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.recipe_sha256,
        "compiled_part_sha256": compiled.compiled_part_sha256,
        "algo_version": compiled.compiler.version,
        "conventions": {
            "angle": compiled.angle_convention,
            "flange_length": compiled.flange_length_convention,
            "bend_line": compiled.bend_line_convention,
            "frame": compiled.frame,
        },
        "thickness": None if compiled.thickness is None else compiled.thickness.model_dump(mode="json"),
        "k_factor": None if compiled.k_factor is None else compiled.k_factor.model_dump(mode="json"),
        "bends": bends,
        "characteristics": characteristics,
        "warnings": [
            {"code": i.code, "bend_uid": i.bend_uid, "detail": i.detail} for i in compiled.issues if i.severity == "WARNING"
        ],
    }


def benchmark_nominals(recipe: Any, compiled: Any) -> dict[str, Any]:
    """Per-bend nominals registered as the benchmark reference (plan §10 amendment 10)."""
    by_bend: dict[str, dict[str, Any]] = {}
    for c in compiled.characteristics:
        if c.bend_uid is None:
            continue
        slot = by_bend.setdefault(c.bend_uid, {"flange": {}, "backgauge": []})
        entry = {
            "characteristic_id": c.characteristic_id,
            "nominal": c.nominal,
            "unavailable_reason": c.unavailable_reason,
            "unit": c.unit,
            "convention": c.convention,
            "boundary_kind": c.boundary_kind,
            "tolerance": _tol(c.tolerance),
            "recipe_requirement": c.recipe_requirement,
        }
        if c.kind == "FLANGE_LENGTH":
            slot["flange"][c.side] = entry
        elif c.kind == "INCLUDED_ANGLE":
            slot["angle"] = entry
        elif c.kind == "BEND_LINE_POSITION":
            slot["bend_line_position"] = entry
        elif c.kind == "BACKGAUGE_LENGTH":
            slot["backgauge"].append({**entry, "stage_index": c.stage_index, "side": c.side})
    recipe_bends = {b.bend_uid: b for b in recipe.bends}
    bends = []
    for b in compiled.bends:
        rb = recipe_bends[b.bend_uid]
        slot = by_bend.get(b.bend_uid, {"flange": {}, "backgauge": []})
        bends.append(
            {
                "bend_uid": b.bend_uid,
                "sequence_index": b.sequence_index,
                "label": b.label,
                "cad_label": rb.cad_label,
                "machine_move_index": b.machine_move_index,
                "nominal_included_deg": b.nominal_included_deg,
                "recipe_nominal_angle": rb.nominal_angle.model_dump(mode="json"),
                "inner_radius_mm": b.inner_radius_mm,
                "thickness_mm": b.thickness_mm,
                "flange_inspection": rb.flange_inspection.model_dump(mode="json"),
                "flange_a": slot["flange"].get("A"),
                "flange_b": slot["flange"].get("B"),
                "angle": slot.get("angle"),
                "bend_line_position": slot.get("bend_line_position"),
                "backgauge": sorted(slot["backgauge"], key=lambda e: (e["stage_index"] or 0, e["side"] or "")),
                "supported": b.supported,
            }
        )
    general = [
        {
            "measurement_id": c.measurement_id,
            "nominal": c.nominal,
            "tolerance": _tol(c.tolerance),
            "recipe_requirement": c.recipe_requirement,
        }
        for c in compiled.characteristics
        if c.kind == "GENERAL_MEASUREMENT"
    ]
    return {
        "schema": "sherman_benchmark_reference_v1",
        "recipe_id": recipe.recipe_id,
        "sku": recipe.sku,
        "recipe_version": recipe.recipe_version,
        "recipe_sha256": recipe.recipe_sha256,
        "content_sha256": recipe.content_sha256,
        "compiled_part_sha256": compiled.compiled_part_sha256,
        "algo_version": compiled.compiler.version,
        "confirmation": recipe.confirmation,
        "inspection_stage": recipe.inspection_stage,
        "bends": bends,
        "general_measurements": general,
    }


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


@dataclass
class ReleaseOutcome:
    """``recipe`` (PartRecipeV1), ``outcome`` ('released' | 'unchanged'), the
    installed compile (``compiled``: CompiledRecord) and the compile report."""

    recipe: Any
    outcome: str
    compiled: CompiledRecord | None = None
    compile_report: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def __iter__(self):  # (recipe, outcome) unpacking, as the survey spec states it
        yield self.recipe
        yield self.outcome


_part_locks: dict[str, threading.Lock] = {}
_part_locks_guard = threading.Lock()


def _part_lock(part_id: str) -> threading.Lock:
    with _part_locks_guard:
        lock = _part_locks.get(part_id)
        if lock is None:
            lock = _part_locks[part_id] = threading.Lock()
        return lock


def _catalog_collisions(catalog: Any, sku: str, part_id: str) -> list[str]:
    from domains.production.recipe.canonical_sku import SkuError, canonical_sku

    others = []
    for other in catalog.list_parts(limit=100000):
        if str(other.get("id")) == part_id:
            continue
        try:
            if canonical_sku(str(other.get("part_number") or "")) == sku:
                others.append(str(other.get("part_number")))
        except SkuError:
            continue
    return others


def release(
    part_id: str,
    request: Any,
    principal: Any,
    *,
    catalog: Any,
    recipe_store: RecipeStore,
    data_root: Path,
    compiler: Any,
    line_store: Any = None,
    clock: Callable[[], datetime] = utc_now,
) -> ReleaseOutcome:
    """Release the part (module docstring). Raises :class:`ReleaseGateError`."""
    with _part_lock(part_id):
        return _release_locked(part_id, request, principal, catalog=catalog, store=recipe_store,
                               data_root=Path(data_root), compiler=compiler, line_store=line_store, clock=clock)


@dataclass
class PreparedRelease:
    """Everything the pre-compile checks established (:func:`prepare_release`)."""

    recipe: Any
    active: Any
    draft: Any
    cad_src: Path
    drawing_src: Path
    flat_path: Path | None


def prepare_release(
    part_id: str,
    request: Any,
    principal: Any,
    *,
    catalog: Any,
    store: RecipeStore,
    data_root: Path,
    clock: Callable[[], datetime] = utc_now,
) -> PreparedRelease:
    """Steps 1-7 of the gate (module docstring): read-only, no compile, no writes.

    Raises :class:`ReleaseGateError` with every applicable code. The onboarding
    checklist uses it as its dry run.
    """
    from pydantic import ValidationError

    from apps.production_line.recipes.catalog_port import DraftValidationError
    from contracts.production.part_recipe import (
        CadRef,
        DrawingRef,
        ReleasedBy,
        ReleaseRequestV1,
    )
    from domains.production.recipe.builder import RecipeBuildError, build_part_recipe
    from domains.production.recipe.canonical_sku import SkuError, canonical_sku

    problems = _Problems()
    part = catalog.get_part(part_id)
    if part is None:
        raise ReleaseGateError(["PART_NOT_FOUND"], {"PART_NOT_FOUND": [part_id]})

    # -- 1. request ------------------------------------------------------------
    req: ReleaseRequestV1 | None = None
    if isinstance(request, ReleaseRequestV1):
        req = request
    else:
        try:
            req = ReleaseRequestV1.model_validate(request)
        except ValidationError as exc:
            problems.merge(_validation_codes(exc))
    if req is not None and req.expected_release_draft_revision is not None:
        stored_draft = store.get_release_draft(part_id)
        current = stored_draft.revision if stored_draft is not None else 0
        if current != req.expected_release_draft_revision:
            problems.add("RELEASE_DRAFT_REVISION_CHANGED",
                         f"release draft is at revision {current}, not {req.expected_release_draft_revision}")

    # -- 2. SKU ------------------------------------------------------------------
    sku: str | None = None
    try:
        sku = canonical_sku(str(part.get("part_number") or ""))
    except SkuError:
        problems.add("SKU_INVALID", repr(part.get("part_number")))
    if sku is not None:
        owner = store.sku_owner(sku)
        if owner is not None and owner != part_id:
            problems.add("SKU_COLLISION", f"{sku} is registered to another part")
        registered = store.sku_for_part(part_id)
        if registered is not None and registered != sku:
            problems.add("SKU_COLLISION", f"this part is registered as {registered}; its part number now reads {sku}")
        for other in _catalog_collisions(catalog, sku, part_id):
            problems.add("SKU_COLLISION", f"catalog part {other!r} has the same canonical SKU {sku}")

    # -- 3. draft ------------------------------------------------------------------
    draft = catalog.get_vendor_setup_draft(part_id)
    spec_rows = catalog.get_bend_specs(part_id)
    if draft is not None:
        try:
            catalog.validate_vendor_draft(part_id, draft)
        except DraftValidationError as exc:
            for code in exc.codes:
                for msg in exc.messages:
                    problems.add(code, msg)

    # -- 4. CAD --------------------------------------------------------------------
    cad_ref: CadRef | None = None
    cad_src: Path | None = None
    cad_sha = part.get("cad_file_hash")
    cad_stored = part.get("cad_file_path")
    cad_ext = Path(str(cad_stored or "")).suffix.lower()
    if not cad_stored or not cad_sha:
        problems.add("CAD_MISSING", "no CAD uploaded")
    elif cad_ext not in _STEP_EXTS:
        problems.add("CAD_NOT_STEP", f"the catalog CAD is {cad_ext or 'untyped'}")
    else:
        cad_src = catalog.resolve_file(str(cad_stored))
        if not cad_src.is_file():
            problems.add("CAD_MISSING", "the catalog CAD file is missing")
            cad_src = None
        elif cached_sha256(cad_src) != cad_sha:
            problems.add("CAD_HASH_MISMATCH", "the CAD bytes do not hash to the catalog cad_file_hash")
            cad_src = None
        try:
            cad_ref = CadRef(
                sha256=str(cad_sha),
                original_filename=Path(str(cad_stored)).name[:255],
                stored_rel_path=f"recipes/cad/{cad_sha}{cad_ext}",
                candidate_generation=catalog.candidate_generation(part_id),
                candidate_angle_convention=candidate_angle_convention(str(cad_stored)),  # type: ignore[arg-type]
            )
        except ValidationError:
            problems.add("CAD_HASH_MISMATCH", "the catalog cad_file_hash is not a sha256")
            cad_ref = None

    # -- 5. drawing ------------------------------------------------------------------
    drawing_ref: DrawingRef | None = None
    drawing_src: Path | None = None
    d_stored, d_sha = part.get("drawing_file_path"), part.get("drawing_file_hash")
    if d_stored and d_sha:
        fmt = drawing_format(str(d_stored))
        d_ext = Path(str(d_stored)).suffix.lower()
        if fmt is None:
            problems.add("DRAWING_FORMAT_UNSUPPORTED", d_ext or "untyped")
        else:
            drawing_src = catalog.resolve_file(str(d_stored))
            if not drawing_src.is_file():
                problems.add("DRAWING_MISSING", "the catalog drawing file is missing")
                drawing_src = None
            elif cached_sha256(drawing_src) != d_sha:
                problems.add("DRAWING_HASH_MISMATCH", "the drawing bytes do not hash to the catalog drawing_file_hash")
                drawing_src = None
            try:
                drawing_ref = DrawingRef(
                    sha256=str(d_sha),
                    original_filename=Path(str(d_stored)).name[:255],
                    stored_rel_path=f"recipes/drawings/{d_sha}{d_ext}",
                    format=fmt,  # type: ignore[arg-type]
                )
            except ValidationError:
                problems.add("DRAWING_HASH_MISMATCH", "the catalog drawing_file_hash is not a sha256")
    # (no drawing: the builder reports DRAWING_MISSING)

    # -- 6. flat -----------------------------------------------------------------------
    flat_record = store.get_part_flat(part_id)
    flat_ref, flat_path, flat_problems = release_flat_ref(
        flat_record,
        draft=draft,
        cad_sha256=str(cad_sha) if cad_sha else None,
        confirmation=req.confirmation if req is not None else "OPERATOR_CONFIRMED",
        data_root=data_root,
    )
    problems.merge(flat_problems)

    # -- 7. build ------------------------------------------------------------------------
    recipe = None
    active = store.get_active(sku) if sku is not None else None
    try:
        released_by = ReleasedBy(**released_by_dict(principal))
    except (ValueError, ValidationError):
        released_by = None
        problems.add("DRAFT_INVALID", "the releasing principal has no user identity")
    if cad_ref is not None and req is not None and sku is not None and released_by is not None:
        version = store.max_version(sku) + 1
        try:
            recipe = build_part_recipe(
                part=part,
                spec_rows=spec_rows,
                draft=draft,
                release=req,
                cad=cad_ref,
                drawing=drawing_ref,
                version=version,
                supersedes_version=active.version if active is not None else None,
                released_by=released_by,
                released_at=iso_utc(clock()),
                flat=flat_ref,
                cad_units_verified=False,
            )
        except RecipeBuildError as exc:
            for code in exc.codes:
                problems.merge({code: list(exc.details.get(code, ()))})
    if problems:
        raise problems.error()
    assert recipe is not None and cad_src is not None and drawing_src is not None  # guaranteed by the checks above
    return PreparedRelease(recipe=recipe, active=active, draft=draft, cad_src=cad_src, drawing_src=drawing_src,
                           flat_path=flat_path)


def _release_locked(
    part_id: str,
    request: Any,
    principal: Any,
    *,
    catalog: Any,
    store: RecipeStore,
    data_root: Path,
    compiler: Any,
    line_store: Any,
    clock: Callable[[], datetime],
) -> ReleaseOutcome:
    from contracts.production.compiled_part import CompiledPartRefV1
    from domains.production.recipe.builder import draft_sha256

    prepared = prepare_release(part_id, request, principal, catalog=catalog, store=store, data_root=data_root,
                               clock=clock)
    recipe, active, draft = prepared.recipe, prepared.active, prepared.draft
    cad_src, drawing_src, flat_path = prepared.cad_src, prepared.drawing_src, prepared.flat_path

    # -- 8. unchanged ------------------------------------------------------------------------
    if active is not None and active.content_sha256 == recipe.content_sha256:
        return ReleaseOutcome(recipe=active.recipe(), outcome="unchanged",
                              compiled=store.get_compiled(active.recipe_sha256))

    # -- 9. freeze inputs ----------------------------------------------------------------
    cad_dest = data_root / recipe.cad.stored_rel_path
    if not _freeze_file(cad_src, cad_dest, recipe.cad.sha256):
        raise ReleaseGateError(["CAD_HASH_MISMATCH"], {"CAD_HASH_MISMATCH": ["the CAD changed while it was copied"]})
    drawing_dest = data_root / recipe.drawing.stored_rel_path
    if not _freeze_file(drawing_src, drawing_dest, recipe.drawing.sha256):
        raise ReleaseGateError(["DRAWING_HASH_MISMATCH"],
                               {"DRAWING_HASH_MISMATCH": ["the drawing changed while it was copied"]})

    # -- 10. compile -----------------------------------------------------------------------
    outcome = compiler.compile(recipe, cad_path=cad_dest, flat_path=flat_path,
                               staging_root=data_root / "recipes" / "compiled")
    try:
        _keep_compile_log(outcome, data_root, recipe.recipe_sha256)
        attempt = {
            "cad_sha256": recipe.cad.sha256,
            "draft_sha256": draft_sha256(draft) if draft is not None else None,
            "flat_sha256": recipe.flat.sha256 if recipe.flat is not None else None,
            "recipe_sha256": recipe.recipe_sha256,
            "content_sha256": recipe.content_sha256,
        }
        store.record_compile_attempt(part_id, status=outcome.status, codes=outcome.codes,
                                     report={**outcome.report, "recipe_id": recipe.recipe_id}, **attempt)
        if outcome.status != "COMPILED":
            # A refused release never reaches the line: no event, nothing stored but the attempt.
            raise ReleaseGateError(outcome.codes, outcome.details, outcome.report)
        compiled = outcome.compiled
        assert compiled is not None and outcome.out_dir is not None

        # -- 11. install + store --------------------------------------------------------------
        rel_path = install_compiled(outcome.out_dir, data_root, recipe.recipe_sha256, compiled.compiled_part_sha256)
        ref = CompiledPartRefV1(
            compiled_part_sha256=compiled.compiled_part_sha256,
            recipe_sha256=recipe.recipe_sha256,
            status="COMPILED",
            compiler_version=compiled.compiler.version,
            stored_rel_path=rel_path,
        )
        record = CompiledRecord(
            recipe_sha256=recipe.recipe_sha256,
            status="COMPILED",
            compiled_part_sha256=ref.compiled_part_sha256,
            compiler_version=ref.compiler_version,
            stored_rel_path=ref.stored_rel_path,
        )
        try:
            store.insert_recipe(
                recipe,
                compiled=record,
                expectations=expectations_document(recipe, compiled),
                benchmark_nominals=benchmark_nominals(recipe, compiled),
            )
        except RecipeStoreError as exc:
            shutil.rmtree(data_root / Path(rel_path).parent, ignore_errors=True)
            if exc.code in ("CONCURRENT_RELEASE", "SKU_COLLISION"):
                raise ReleaseGateError([exc.code], {exc.code: [exc.message]}) from None
            raise
    finally:
        outcome.cleanup()

    # -- 12. events + listeners ---------------------------------------------------------------
    payload = {
        "recipe_id": recipe.recipe_id,
        "sku": recipe.sku,
        "version": recipe.recipe_version,
        "recipe_sha256": recipe.recipe_sha256,
        "compiled_part_sha256": compiled.compiled_part_sha256,
    }
    _emit(line_store, "RECIPE_COMPILED", principal, {**payload, "blocker_codes": []})
    _emit(line_store, "RECIPE_RELEASED", principal, payload)
    notify_release_listeners(recipe)
    warnings = [f"{w['code']}: {w['detail']}" for w in outcome.report.get("issues", []) if w["severity"] == "WARNING"]
    return ReleaseOutcome(recipe=recipe, outcome="released", compiled=store.get_compiled(recipe.recipe_sha256),
                          compile_report=outcome.report, warnings=warnings)


def _keep_compile_log(outcome: CompileOutcome, data_root: Path, recipe_sha256: str) -> None:
    """Keep the step log (stdout/stderr) and E1's compile.log under ``logs/recipe_compile/``."""
    if outcome.staging_dir is None:
        return
    logs = data_root / "logs" / "recipe_compile"
    try:
        logs.mkdir(parents=True, exist_ok=True)
        step = outcome.staging_dir / "compile_step.log"
        if step.is_file():
            shutil.copyfile(step, logs / f"{recipe_sha256}.step.log")
        e1_log = (outcome.out_dir or outcome.staging_dir / "out") / "compile.log"
        if e1_log.is_file():
            shutil.copyfile(e1_log, logs / f"{recipe_sha256}.compile.log")
    except OSError:
        _LOG.exception("could not keep the compile logs for %s", recipe_sha256)


def _emit(line_store: Any, kind: str, principal: Any, payload: Mapping[str, Any]) -> None:
    if line_store is None:
        return
    try:
        line_store.append_event(kind, principal=_principal_audit(principal), payload=dict(payload))
    except Exception:
        _LOG.exception("could not append %s for %s (the release itself stands)", kind, payload.get("recipe_id"))
