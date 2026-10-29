"""Append-only benchmark case registry (hash-chained JSONL).

One row per benchmark case: one inspection of one physical (or synthetic) part,
bound to independent truth. Rows are never edited or deleted. A correction is
a new case id with a ``note`` that names the case it supersedes. The registry
holds the inputs of a frozen evaluation (accuracy protocol §5), so it must be
tamper-evident.

- Each row carries ``prev_row_sha256`` (the previous row's ``row_sha256``, or
  64 zeros for the first row) and ``row_sha256``, the sha256 of the row's
  canonical JSON without ``row_sha256`` (``contracts.production.canonical``).
  :meth:`CaseRegistry.rows` re-verifies the whole chain on every read and raises
  ``CHAIN_BROKEN`` on any edit, reorder, insertion or deletion of a row that has
  a successor. Removing rows from the **tail** leaves a valid shorter chain, so
  it is detected only against an externally recorded head
  (:meth:`CaseRegistry.rows` with ``expect_head``, CLI ``verify --expect-head``).
  The freeze records that head (accuracy protocol §5).
- Appends take an exclusive lock: ``fcntl.flock`` on POSIX, ``msvcrt.locking``
  on the first byte of a sibling ``<name>.lock`` file on Windows. The file is
  created 0600. Each row is written with one ``write`` and then fsynced.

Row (schema ``sherman_benchmark_case_v1``)::

    case_id         [A-Za-z0-9][A-Za-z0-9._:-]{0,127}, unique in the registry
    job_id          line_<32 hex> or null (a truth row may precede its job)
    recipe_sha256   64 hex
    scan_sha256     64 hex
    sku             canonical SKU text (informational)
    sku_family      the part-family label used for grouped reporting
    split_group     the grouping unit for splits: repeats and CAD variants of
                    one part share a group (roadmap M5)
    truth_source    synthetic | metrology_measured | operator_verdict
    truth           {bend_uid: {quantity_key: TruthQuantityV1}}
    registered_at   UTC ISO-8601 (``...Z``)
    registered_by   who registered it (user id or tool name)
    note            optional free text
    prev_row_sha256, row_sha256

Quantity keys (:data:`QUANTITY_KEYS`) are ``flange_a_mm``, ``flange_b_mm``,
``included_deg`` and ``bend_line_position_mm``. A ``TruthQuantityV1`` is
``{value?, verdict?, instrument, uncertainty?, nominal?, tol_plus?, tol_minus?}``:

- ``synthetic``: ``value`` is required (analytic truth).
- ``metrology_measured``: ``value`` and ``uncertainty`` > 0 are required.
  ``uncertainty`` is the instrument's expanded uncertainty (k = 2) in the
  quantity's unit.
- ``operator_verdict``: ``verdict`` IN|OUT is required and ``value`` is
  forbidden, because an operator verdict is not a measurement.
- ``nominal``, ``tol_plus`` and ``tol_minus`` are the released recipe's nominal
  and tolerance for the quantity (a snapshot taken when the case is
  registered). For ``synthetic`` and ``metrology_measured`` rows all three keys
  are **required to be present**, and ``nominal`` must be a number. The
  tolerance pair may be given explicitly as ``null`` when the recipe has no
  tolerance for that quantity; the truth class is then ``UNKNOWN``. The
  evaluator classifies truth from this snapshot only, never from the engine's
  result, so a failed or truncated run cannot move a case out of the
  denominators (accuracy protocol §5).

The survey's flat per-bend shape ``{flange_a_mm, flange_b_mm, included_deg,
instrument, uncertainty, nominal?, tolerance?}`` is also accepted on input and
normalised (:func:`normalise_bend_truth`). Its ``uncertainty`` is either a
number (all of the bend's quantities must then share one unit) or
``{"mm": u, "deg": u}``. ``nominal`` is ``{quantity_key: value}`` and
``tolerance`` is ``{quantity_key: [plus, minus] | null}``.

Custody: a registry may not live in the repository, in its research
directories (``apps.production_line.settings.forbidden_location_reason``), or
under any path component that names protected material (holdout, sealed,
EHO4, one-shot, authorization).

CLI::

    python -m domains.production.benchmark.case_registry verify --registry FILE [--expect-head SHA]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, ValidationError, field_validator, model_validator

from contracts.production.canonical import sha256_hex
from contracts.production.common import ContractModel

__all__ = [
    "REGISTRY_ROW_SCHEMA",
    "QUANTITY_KEYS",
    "TRUTH_SOURCES",
    "GENESIS_SHA256",
    "REGISTRY_ERROR_CODES",
    "RegistryError",
    "TruthQuantityV1",
    "BenchmarkCaseV1",
    "normalise_bend_truth",
    "registry_path_problem",
    "CaseRegistry",
    "main",
]

REGISTRY_ROW_SCHEMA = "sherman_benchmark_case_v1"
#: Truth quantity key -> (characteristic kind, flange side, unit).
QUANTITY_KEYS: dict[str, tuple[str, Optional[str], str]] = {
    "flange_a_mm": ("FLANGE_LENGTH", "A", "mm"),
    "flange_b_mm": ("FLANGE_LENGTH", "B", "mm"),
    "included_deg": ("INCLUDED_ANGLE", None, "deg"),
    "bend_line_position_mm": ("BEND_LINE_POSITION", None, "mm"),
}
TRUTH_SOURCES: tuple[str, ...] = ("synthetic", "metrology_measured", "operator_verdict")
GENESIS_SHA256 = "0" * 64
REGISTRY_ERROR_CODES: frozenset[str] = frozenset(
    {"PATH_FORBIDDEN", "ROW_INVALID", "DUPLICATE_CASE", "CHAIN_BROKEN", "HEAD_MISMATCH", "IO_ERROR"}
)
#: Path components (casefolded substrings) that name protected material (AGENTS.md custody).
PROTECTED_PATH_TOKENS: tuple[str, ...] = ("holdout", "sealed", "eho4", "one_shot", "one-shot", "oneshot", "authoriz")

_CASE_ID_RE = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_SHA_RE = r"^[0-9a-f]{64}$"
_JOB_ID_RE = r"^line_[0-9a-f]{32}$"
_BEND_UID_RE = re.compile(r"^bu_[0-9a-f]{20}$")
#: Recipe-snapshot keys that a measured (non-verdict) truth quantity must state explicitly.
_SNAPSHOT_KEYS: tuple[str, ...] = ("nominal", "tol_plus", "tol_minus")
_ISO_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$"


class RegistryError(Exception):
    """A refused registry operation. ``code`` is one of :data:`REGISTRY_ERROR_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        if code not in REGISTRY_ERROR_CODES:  # pragma: no cover - programming error
            raise ValueError(f"unknown registry error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _finite(value: Optional[float], name: str) -> Optional[float]:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class TruthQuantityV1(ContractModel):
    """Truth for one quantity of one bend (module docstring)."""

    value: Optional[float] = None
    verdict: Optional[Literal["IN", "OUT"]] = None
    instrument: str = Field(min_length=1, max_length=200)
    #: Expanded uncertainty (k = 2), same unit as the quantity.
    uncertainty: Optional[float] = Field(default=None, ge=0)
    nominal: Optional[float] = None
    tol_plus: Optional[float] = Field(default=None, ge=0)
    tol_minus: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "TruthQuantityV1":
        for name in ("value", "uncertainty", "nominal", "tol_plus", "tol_minus"):
            _finite(getattr(self, name), name)
        if self.value is None and self.verdict is None:
            raise ValueError("a truth quantity needs a value or a verdict")
        if (self.tol_plus is None) != (self.tol_minus is None):
            raise ValueError("tol_plus and tol_minus go together")
        if self.tol_plus is not None and self.tol_plus + (self.tol_minus or 0.0) <= 0:
            raise ValueError("tol_plus + tol_minus must be > 0")
        if self.tol_plus is not None and self.nominal is None:
            raise ValueError("a tolerance snapshot needs its nominal")
        return self


class BenchmarkCaseV1(ContractModel):
    """One registry row (module docstring)."""

    schema_: Literal["sherman_benchmark_case_v1"] = Field(default=REGISTRY_ROW_SCHEMA, alias="schema")
    case_id: str = Field(pattern=_CASE_ID_RE)
    job_id: Optional[str] = Field(default=None, pattern=_JOB_ID_RE)
    recipe_sha256: str = Field(pattern=_SHA_RE)
    scan_sha256: str = Field(pattern=_SHA_RE)
    sku: str = Field(min_length=1, max_length=64)
    sku_family: str = Field(min_length=1, max_length=64)
    split_group: str = Field(min_length=1, max_length=128)
    truth_source: Literal["synthetic", "metrology_measured", "operator_verdict"]
    truth: dict[str, dict[str, TruthQuantityV1]]
    registered_at: str = Field(pattern=_ISO_RE)
    registered_by: str = Field(min_length=1, max_length=128)
    note: Optional[str] = Field(default=None, max_length=2000)
    prev_row_sha256: str = Field(default=GENESIS_SHA256, pattern=_SHA_RE)
    row_sha256: Optional[str] = Field(default=None, pattern=_SHA_RE)

    @field_validator("truth", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {uid: normalise_bend_truth(bend) if isinstance(bend, Mapping) else bend for uid, bend in value.items()}

    @model_validator(mode="after")
    def _check(self) -> "BenchmarkCaseV1":
        if not self.truth:
            raise ValueError("truth must name at least one bend")
        for uid, bend in self.truth.items():
            if not _BEND_UID_RE.fullmatch(uid):
                raise ValueError(f"truth key {uid!r} is not a bend_uid (bu_<20 hex>)")
            if not bend:
                raise ValueError(f"truth for {uid} names no quantity")
            for key, q in bend.items():
                if key not in QUANTITY_KEYS:
                    raise ValueError(f"unknown truth quantity {key!r} (known: {sorted(QUANTITY_KEYS)})")
                if self.truth_source == "operator_verdict":
                    if q.verdict is None or q.value is not None:
                        raise ValueError(f"{uid}/{key}: an operator_verdict row carries a verdict and no value")
                else:
                    if q.value is None:
                        raise ValueError(f"{uid}/{key}: a {self.truth_source} row needs a value")
                    if self.truth_source == "metrology_measured" and not (q.uncertainty and q.uncertainty > 0):
                        raise ValueError(f"{uid}/{key}: a metrology_measured value needs an uncertainty > 0")
                    missing = [k for k in _SNAPSHOT_KEYS if k not in q.model_fields_set]
                    if missing or q.nominal is None:
                        raise ValueError(
                            f"{uid}/{key}: a {self.truth_source} row needs the recipe snapshot "
                            f"(nominal, and tol_plus/tol_minus given explicitly, null when the recipe has none); "
                            f"missing {missing or ['nominal']}"
                        )
        return self

    def payload(self) -> dict[str, Any]:
        """The row as JSON (aliases, no ``row_sha256``): the bytes the row hash covers."""
        doc = self.model_dump(mode="json", by_alias=True)
        doc.pop("row_sha256", None)
        return doc

    def computed_sha256(self) -> str:
        return sha256_hex(self.payload())


def normalise_bend_truth(bend: Mapping[str, Any]) -> dict[str, Any]:
    """Accept the nested shape ``{key: {value, instrument, ...}}`` or the survey's
    flat shape ``{flange_a_mm: 12.3, ..., instrument, uncertainty}``; return the
    nested shape. A flat ``uncertainty`` is a number (only when all quantities
    share one unit) or ``{"mm": u, "deg": u}``."""
    if "instrument" not in bend and "uncertainty" not in bend:
        return dict(bend)
    instrument = bend.get("instrument")
    unc = bend.get("uncertainty")
    nominals = bend.get("nominal")
    tolerances = bend.get("tolerance")
    if nominals is not None and not isinstance(nominals, Mapping):
        raise ValueError("a flat truth row's nominal must be {quantity_key: value}")
    if tolerances is not None and not isinstance(tolerances, Mapping):
        raise ValueError("a flat truth row's tolerance must be {quantity_key: [plus, minus] | null}")
    keys = [k for k in bend if k not in ("instrument", "uncertainty", "nominal", "tolerance")]
    units = {QUANTITY_KEYS[k][2] for k in keys if k in QUANTITY_KEYS}
    out: dict[str, Any] = {}
    for key in keys:
        value = bend[key]
        if key not in QUANTITY_KEYS:
            out[key] = value  # rejected by the model with a clear message
            continue
        unit = QUANTITY_KEYS[key][2]
        if isinstance(unc, Mapping):
            u = unc.get(unit)
        elif unc is None:
            u = None
        else:
            if len(units) > 1:
                raise ValueError("a flat truth row mixing mm and deg needs uncertainty {'mm': u, 'deg': u}")
            u = unc
        entry: dict[str, Any] = {"instrument": instrument, "uncertainty": u}
        if isinstance(value, str) and value in ("IN", "OUT"):
            entry["verdict"] = value
        else:
            entry["value"] = value
        if nominals is not None and key in nominals:
            entry["nominal"] = nominals[key]
        if tolerances is not None and key in tolerances:
            band = tolerances[key]
            if band is None:
                entry["tol_plus"] = entry["tol_minus"] = None
            elif isinstance(band, Sequence) and not isinstance(band, str) and len(band) == 2:
                entry["tol_plus"], entry["tol_minus"] = band[0], band[1]
            else:
                raise ValueError(f"tolerance for {key} must be [plus, minus] or null")
        out[key] = entry
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def registry_path_problem(path: Path, repo_root: Optional[Path] = None) -> Optional[str]:
    """Why ``path`` may not hold a registry, or ``None`` when it may."""
    from apps.production_line.settings import forbidden_location_reason

    p = Path(path)
    if not p.is_absolute():
        return "registry path must be absolute"
    for part in Path(os.path.realpath(p)).parts + p.parts:
        low = part.casefold()
        if any(tok in low for tok in PROTECTED_PATH_TOKENS):
            return f"path component {part!r} names protected material"
    reason = forbidden_location_reason(p, Path(repo_root) if repo_root is not None else _repo_root())
    return None if reason is None else f"registry {reason}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


try:  # POSIX advisory lock
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]
try:  # Windows byte-range lock on a sibling lock file
    import msvcrt as _msvcrt  # type: ignore[import-not-found]
except ImportError:
    _msvcrt = None  # type: ignore[assignment]


class _AppendLock:
    """Exclusive inter-process lock around one append (module docstring)."""

    def __init__(self, registry_fd: int, registry_path: Path) -> None:
        self._fd = registry_fd
        self._lock_path = registry_path.with_name(registry_path.name + ".lock")
        self._lock_fd: Optional[int] = None

    def __enter__(self) -> "_AppendLock":
        if _fcntl is not None:
            _fcntl.flock(self._fd, _fcntl.LOCK_EX)
        elif _msvcrt is not None:  # pragma: no cover - Windows
            self._lock_fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            os.lseek(self._lock_fd, 0, os.SEEK_SET)
            _msvcrt.locking(self._lock_fd, _msvcrt.LK_LOCK, 1)  # retries for ~10 s, then OSError
        else:  # pragma: no cover - no locking primitive at all: refuse rather than race
            raise RegistryError("IO_ERROR", "no file-locking primitive on this platform")
        return self

    def __exit__(self, *exc: object) -> None:
        if _fcntl is not None:
            try:
                _fcntl.flock(self._fd, _fcntl.LOCK_UN)
            except OSError:  # pragma: no cover
                pass
        elif self._lock_fd is not None:  # pragma: no cover - Windows
            try:
                os.lseek(self._lock_fd, 0, os.SEEK_SET)
                _msvcrt.locking(self._lock_fd, _msvcrt.LK_UNLCK, 1)
            finally:
                os.close(self._lock_fd)


class CaseRegistry:
    """Append-only registry at ``path`` (module docstring)."""

    def __init__(self, path: Path | str, *, repo_root: Optional[Path] = None) -> None:
        self.path = Path(path)
        problem = registry_path_problem(self.path, repo_root)
        if problem:
            raise RegistryError("PATH_FORBIDDEN", problem)

    # -- reading ---------------------------------------------------------------

    def _read_rows(self) -> list[BenchmarkCaseV1]:
        if not self.path.exists():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RegistryError("IO_ERROR", f"cannot read {self.path.name}: {exc}") from None
        rows: list[BenchmarkCaseV1] = []
        prev = GENESIS_SHA256
        seen: set[str] = set()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise RegistryError("CHAIN_BROKEN", f"line {lineno}: blank line in an append-only registry")
            try:
                row = BenchmarkCaseV1.model_validate(json.loads(line))
            except (ValueError, ValidationError) as exc:
                raise RegistryError("CHAIN_BROKEN", f"line {lineno}: unreadable row ({str(exc).splitlines()[0]})") from None
            if row.prev_row_sha256 != prev:
                raise RegistryError("CHAIN_BROKEN", f"line {lineno}: prev_row_sha256 does not follow the chain")
            if row.row_sha256 != row.computed_sha256():
                raise RegistryError("CHAIN_BROKEN", f"line {lineno}: row_sha256 does not match the row")
            if row.case_id in seen:
                raise RegistryError("CHAIN_BROKEN", f"line {lineno}: duplicate case_id {row.case_id}")
            seen.add(row.case_id)
            prev = row.row_sha256  # type: ignore[assignment]
            rows.append(row)
        if text and not text.endswith("\n"):
            raise RegistryError("CHAIN_BROKEN", "the last row is not newline-terminated (torn write)")
        return rows

    def rows(self, *, expect_head: Optional[str] = None) -> list[BenchmarkCaseV1]:
        """Every row, in order, after verifying the whole chain.

        ``expect_head`` is a previously recorded :meth:`head_sha256`. When given,
        the chain must still contain a row with that ``row_sha256``; a registry
        truncated below the recorded head raises ``HEAD_MISMATCH``. Rows
        appended after the recorded head are allowed (the registry only grows).
        """
        rows = self._read_rows()
        if expect_head is not None:
            if not re.fullmatch(_SHA_RE, expect_head):
                raise RegistryError("HEAD_MISMATCH", "expected head is not a sha256 hex digest")
            if expect_head != GENESIS_SHA256 and all(r.row_sha256 != expect_head for r in rows):
                raise RegistryError(
                    "HEAD_MISMATCH",
                    f"recorded head {expect_head[:12]}... is not in the chain ({len(rows)} rows): rows were removed",
                )
        return rows

    def get(self, case_id: str) -> Optional[BenchmarkCaseV1]:
        for row in self._read_rows():
            if row.case_id == case_id:
                return row
        return None

    def head_sha256(self) -> str:
        """``row_sha256`` of the last row (the value to record in a freeze), or the genesis hash."""
        rows = self._read_rows()
        return rows[-1].row_sha256 if rows else GENESIS_SHA256  # type: ignore[return-value]

    # -- appending -------------------------------------------------------------

    def append(self, row: BenchmarkCaseV1 | Mapping[str, Any]) -> BenchmarkCaseV1:
        """Validate, chain and append one row; returns the stored row."""
        doc = row.model_dump(mode="json", by_alias=True) if isinstance(row, BenchmarkCaseV1) else dict(row)
        doc.pop("row_sha256", None)
        doc.setdefault("registered_at", _now_iso())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        except OSError as exc:
            raise RegistryError("IO_ERROR", f"cannot open {self.path.name}: {exc.strerror}") from None
        try:
            with _AppendLock(fd, self.path):
                return self._append_locked(fd, doc)
        finally:
            os.close(fd)

    def _append_locked(self, fd: int, doc: dict[str, Any]) -> BenchmarkCaseV1:
        existing = self._read_rows()
        if any(r.case_id == doc.get("case_id") for r in existing):
            raise RegistryError("DUPLICATE_CASE", f"case_id {doc.get('case_id')!r} is already registered")
        doc["prev_row_sha256"] = existing[-1].row_sha256 if existing else GENESIS_SHA256
        try:
            draft = BenchmarkCaseV1.model_validate(doc)
        except ValidationError as exc:
            raise RegistryError("ROW_INVALID", str(exc).replace("\n", "; ")[:2000]) from None
        stored = draft.model_copy(update={"row_sha256": draft.computed_sha256()})
        line = json.dumps(
            stored.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ) + "\n"
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
        return stored


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m domains.production.benchmark.case_registry")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="verify the hash chain and print a summary")
    v.add_argument("--registry", type=Path, required=True)
    v.add_argument("--expect-head", default=None, metavar="SHA",
                   help="a recorded head_sha256 (e.g. from the freeze); detects tail truncation")
    args = parser.parse_args(argv)
    try:
        reg = CaseRegistry(args.registry.expanduser().absolute())
        rows = reg.rows(expect_head=args.expect_head)
    except RegistryError as exc:
        sys.stdout.write(json.dumps({"ok": False, "code": exc.code, "message": exc.message}) + "\n")
        return 2
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r.truth_source] = by_source.get(r.truth_source, 0) + 1
    sys.stdout.write(json.dumps({"ok": True, "rows": len(rows), "head_sha256": reg.head_sha256(),
                                 "by_truth_source": by_source}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
