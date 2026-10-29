"""Validated configuration and data-root layout for the Sherman QC production line.

The line is configured only through environment variables (``SHERMAN_LINE_*``) and
fails closed: :func:`load_settings` raises :class:`SettingsError` with a stable
``code`` instead of guessing.  The data root must be an absolute path *outside* the
repository (after resolving symlinks) and outside the repository's mutable research
directories (``data/``, ``output/``, ``outputs/``, ``artifacts/``, ``uploads/``,
``live_scans/``), even when one of those is a symlink that points elsewhere.

Environment (defaults in parentheses)::

    SHERMAN_LINE_ENABLED              must be truthy, else LINE_DISABLED
    SHERMAN_LINE_DATA_ROOT            absolute, outside the repo
    SHERMAN_LINE_STATIONS             JSON list of {station_id, inbox, sidecar_mode,
                                      source_policy, recursive}
    SHERMAN_LINE_STATION_ID           single-station shorthand (with SHERMAN_LINE_INBOX)
    SHERMAN_LINE_INBOX
    SHERMAN_LINE_ENGINE_BUNDLE_DIR    (<root>/engine/current)
    SHERMAN_LINE_PYTHON               (sys.executable)
    SHERMAN_LINE_JOB_TIMEOUT_S        (180; plan §10 amendment 1: E3 hard stop)
    SHERMAN_LINE_POSTPROCESS_TIMEOUT_S (300)
    SHERMAN_LINE_RSS_LIMIT_MB         (32768; ENGINE_IO-A5: 32 GiB; at most 32 GiB)
    SHERMAN_LINE_FILENAME_REGEX       optional, must contain a named group ``sku``
    SHERMAN_RELEASE_ROLES             (admin) comma-separated subset of admin,operator
    SHERMAN_ALLOW_SELF_REGISTER       (0)
    SHERMAN_PRODUCTION_HARDENING      (0)
    SHERMAN_LEGACY_LIVE_SCAN          (0: off whenever the line is enabled)
    SHERMAN_LEGACY_ENGINE             (0: the legacy analysis uploads that run the discarded
                                      inspection runtime are refused whenever the line is enabled)
    SHERMAN_REPORT_FONT_PATH          optional absolute font file
    SHERMAN_LINE_LOG_RETENTION_DAYS   (30)
    SHERMAN_LINE_PROGRAM_STATUS_PATH  (<root>/config/f1_program_status.json; may be
                                      absent, which downstream treats as fail-closed)

App copies (plan §10 amendment 10): servers run from an app copy built by
``apps.production_line.engine_bundle --profile app``.  There "the source root this
module runs from" is the copy, so every custody check also applies to the *dev repo*
the copy was built from, which the copy's ``bundle_manifest.json`` records as
``source_repo_root`` (:func:`custody_repo_roots`).  A copy whose manifest does not
name its source repo is refused (fail closed).

Wave-1 independence: this module duplicates the data-root directory list from
``contracts.production.layout`` (``DATA_ROOT_DIRS``/``FORBIDDEN_DATA_ROOT_PARENTS``)
instead of importing it; the integration parity test asserts the copies match.

:func:`ensure_layout` creates the layout with owner-only permissions (directories
0700) and :func:`get_secret` creates or loads 32-byte keys stored 0600 under
``secrets/``.
"""
from __future__ import annotations

import json
import os
import re
import secrets as _secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = [
    "SETTINGS_ERROR_CODES",
    "FORBIDDEN_REPO_SUBDIRS",
    "DATA_ROOT_DIRS",
    "STATION_ID_RE",
    "SIDECAR_MODES",
    "SOURCE_POLICIES",
    "KNOWN_ROLES",
    "MAX_RSS_LIMIT_MB",
    "SECRET_KEY_BYTES",
    "SettingsError",
    "StationConfig",
    "LinePaths",
    "LineSettings",
    "default_repo_root",
    "APP_MANIFEST_NAME",
    "APP_MANIFEST_SCHEMA",
    "source_repo_root_of",
    "custody_repo_roots",
    "is_inside",
    "forbidden_location_reason",
    "load_settings",
    "ensure_layout",
    "get_secret",
]

#: Stable error codes raised by :func:`load_settings` / :func:`get_secret`.
SETTINGS_ERROR_CODES: frozenset[str] = frozenset(
    {
        "LINE_DISABLED",
        "DATA_ROOT_MISSING",
        "DATA_ROOT_NOT_ABSOLUTE",
        "DATA_ROOT_INSIDE_REPO",
        "STATION_INVALID",
        "RSS_LIMIT_INVALID",
        "FILENAME_REGEX_INVALID",
        "VALUE_INVALID",
        "SECRET_INVALID",
    }
)

#: Repository directories that hold mutable research/custody material.  A data root,
#: station inbox or bundle input may never resolve under one of them.
#: Mirrors ``contracts.production.layout.FORBIDDEN_DATA_ROOT_PARENTS``.
FORBIDDEN_REPO_SUBDIRS: tuple[str, ...] = ("data", "output", "outputs", "artifacts", "uploads", "live_scans")

#: Directories created under the data root.  Mirrors ``contracts.production.layout.DATA_ROOT_DIRS``.
DATA_ROOT_DIRS: tuple[str, ...] = (
    "db",
    "recipes/cad",
    "recipes/drawings",
    "recipes/flats",
    "recipes/compiled",
    "archive/scans",
    "quarantine",
    "jobs",
    "engine",
    "logs",
    "secrets",
    "backups",
    "config",
)

STATION_ID_RE = re.compile(r"^[a-z0-9-]{1,32}$")
SIDECAR_MODES: tuple[str, ...] = ("off", "required")
SOURCE_POLICIES: tuple[str, ...] = ("move", "copy_leave")
KNOWN_ROLES: tuple[str, ...] = ("admin", "operator", "viewer")
#: Roles that may be granted ``recipe.release`` through ``SHERMAN_RELEASE_ROLES``.
RELEASABLE_ROLES: tuple[str, ...] = ("admin", "operator")
MAX_RSS_LIMIT_MB = 32 * 1024
SECRET_KEY_BYTES = 32
_SECRET_NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})
_STATION_KEYS = frozenset({"station_id", "inbox", "sidecar_mode", "source_policy", "recursive"})


class SettingsError(Exception):
    """Configuration refused.  ``code`` is one of :data:`SETTINGS_ERROR_CODES`."""

    def __init__(self, code: str, message: str = "") -> None:
        if code not in SETTINGS_ERROR_CODES:  # pragma: no cover - programming error
            raise ValueError(f"unknown settings error code {code!r}")
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {self.message}")


@dataclass(frozen=True)
class StationConfig:
    """One scan cell: its id and the watched inbox folder."""

    station_id: str
    inbox: Path
    sidecar_mode: str = "off"
    source_policy: str = "move"
    recursive: bool = False


@dataclass(frozen=True)
class LinePaths:
    """Absolute paths of the data-root layout (see ``contracts/production/layout.py``)."""

    data_root: Path
    db_dir: Path
    db_line: Path
    db_recipes: Path
    db_security: Path
    recipes_cad: Path
    recipes_drawings: Path
    recipes_flats: Path
    recipes_compiled: Path
    archive_scans: Path
    quarantine: Path
    jobs: Path
    engine: Path
    logs: Path
    secrets: Path
    backups: Path
    config: Path

    @classmethod
    def for_root(cls, data_root: Path) -> "LinePaths":
        root = Path(data_root)
        db = root / "db"
        return cls(
            data_root=root,
            db_dir=db,
            db_line=db / "line.sqlite3",
            db_recipes=db / "recipes.sqlite3",
            db_security=db / "security.sqlite3",
            recipes_cad=root / "recipes" / "cad",
            recipes_drawings=root / "recipes" / "drawings",
            recipes_flats=root / "recipes" / "flats",
            recipes_compiled=root / "recipes" / "compiled",
            archive_scans=root / "archive" / "scans",
            quarantine=root / "quarantine",
            jobs=root / "jobs",
            engine=root / "engine",
            logs=root / "logs",
            secrets=root / "secrets",
            backups=root / "backups",
            config=root / "config",
        )

    def layout_dirs(self) -> tuple[Path, ...]:
        """Every directory :func:`ensure_layout` creates, root first."""
        return (self.data_root,) + tuple(self.data_root / rel for rel in DATA_ROOT_DIRS)


@dataclass(frozen=True)
class LineSettings:
    """Validated line configuration.  Construct only through :func:`load_settings`."""

    enabled: bool
    repo_root: Path
    paths: LinePaths
    stations: tuple[StationConfig, ...]
    engine_bundle_dir: Path
    python_exe: Path
    job_timeout_s: int
    postprocess_timeout_s: int
    rss_limit_mb: int
    filename_regex: Optional[str]
    release_roles: frozenset[str]
    allow_self_register: bool
    production_hardening: bool
    legacy_live_scan_enabled: bool
    legacy_engine_enabled: bool
    report_font_path: Optional[Path]
    log_retention_days: int
    program_status_path: Path

    @property
    def data_root(self) -> Path:
        return self.paths.data_root

    @property
    def rss_limit_bytes(self) -> int:
        return self.rss_limit_mb * 1024 * 1024

    @property
    def station_ids(self) -> tuple[str, ...]:
        return tuple(s.station_id for s in self.stations)

    def station(self, station_id: str) -> Optional[StationConfig]:
        for s in self.stations:
            if s.station_id == station_id:
                return s
        return None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def default_repo_root() -> Path:
    """The source root this module runs from (the dev repo or a deployed app copy)."""
    return Path(__file__).resolve().parents[2]


#: Manifest written at the root of an engine bundle / app copy.  Mirrors
#: ``apps.production_line.engine_bundle.MANIFEST_NAME``/``MANIFEST_SCHEMA`` (parity-tested).
APP_MANIFEST_NAME = "bundle_manifest.json"
APP_MANIFEST_SCHEMA = "sherman_engine_bundle_manifest_v1"


def source_repo_root_of(root: Path) -> Optional[Path]:
    """The dev repo an app copy at ``root`` was built from, or ``None`` when ``root``
    is not a bundle (no ``bundle_manifest.json``: a dev checkout).

    Raises :class:`SettingsError` (``VALUE_INVALID``) when ``root`` carries a manifest
    that is unreadable, has the wrong schema, or does not record an absolute
    ``source_repo_root`` other than ``root`` itself: the dev repo is then unknown and
    custody cannot be checked.
    """
    real_root = _real(Path(root))
    manifest = real_root / APP_MANIFEST_NAME
    if not manifest.exists() and not manifest.is_symlink():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise SettingsError("VALUE_INVALID", f"{manifest} is unreadable; rebuild the app copy") from None
    if not isinstance(data, dict) or data.get("schema") != APP_MANIFEST_SCHEMA:
        raise SettingsError("VALUE_INVALID", f"{manifest} is not a {APP_MANIFEST_SCHEMA} manifest")
    src = data.get("source_repo_root")
    if not isinstance(src, str) or not src or not Path(src).is_absolute():
        raise SettingsError(
            "VALUE_INVALID", f"{manifest} does not record source_repo_root; rebuild the app copy with this version"
        )
    real_src = _real(Path(src))
    if real_src == real_root:
        raise SettingsError("VALUE_INVALID", f"{manifest} names the copy itself as its source repo")
    return real_src


def custody_repo_roots(repo_root: Path) -> tuple[Path, ...]:
    """Resolved roots that line data must stay out of: ``repo_root`` (the running
    source root) and, for an app copy, the dev repo it was built from.  Raises like
    :func:`source_repo_root_of`."""
    real = _real(Path(repo_root))
    src = source_repo_root_of(real)
    return (real,) if src is None else (real, src)


def _real(p: Path) -> Path:
    # os.path.realpath never raises for missing components (strict=False).
    return Path(os.path.realpath(os.fspath(p)))


def is_inside(child: Path, parent: Path) -> bool:
    """True when ``child`` equals or lies below ``parent`` (both already resolved)."""
    try:
        Path(child).relative_to(parent)
        return True
    except ValueError:
        return False


def forbidden_location_reason(path: Path, repo_root: Path) -> Optional[str]:
    """Why ``path`` may not hold line data, or ``None`` when it is acceptable.

    ``path`` is resolved through symlinks.  It is refused when it lies inside the
    repository, or inside the *resolved target* of any repository research directory
    (so a ``data -> /Volumes/x`` symlink also protects ``/Volumes/x``).  Only the
    directory entries themselves are resolved; nothing inside them is listed or read.

    When ``repo_root`` is an app copy, the dev repo recorded in its manifest is
    checked the same way; an app copy whose source repo is unknown refuses every
    path (fail closed).
    """
    real_path = _real(path)
    try:
        roots = custody_repo_roots(repo_root)
    except SettingsError as exc:
        return f"cannot be checked: {exc.message}"
    for real_repo in roots:
        if is_inside(real_path, real_repo):
            return f"resolves inside the repository {real_repo}"
        for name in FORBIDDEN_REPO_SUBDIRS:
            target = _real(real_repo / name)
            if is_inside(real_path, target):
                return f"resolves under the repository's {name}/ ({target})"
    return None


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------


def _get(env: Mapping[str, str], name: str) -> str:
    return str(env.get(name, "") or "").strip()


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name).lower()
    if raw == "" and name not in env:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise SettingsError("VALUE_INVALID", f"{name} must be a boolean (1/0/true/false), got {raw!r}")


def _int(env: Mapping[str, str], name: str, default: int, *, lo: int, hi: int, code: str = "VALUE_INVALID") -> int:
    raw = _get(env, name)
    if raw == "":
        return default
    try:
        value = int(raw, 10)
    except ValueError:
        raise SettingsError(code, f"{name} must be an integer, got {raw!r}") from None
    if not lo <= value <= hi:
        raise SettingsError(code, f"{name}={value} outside [{lo}, {hi}]")
    return value


def _abs_path(env: Mapping[str, str], name: str, default: Optional[Path]) -> Optional[Path]:
    raw = _get(env, name)
    if raw == "":
        return default
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        raise SettingsError("VALUE_INVALID", f"{name} must be an absolute path")
    return p


def _parse_station(obj: Any, index: int, repo_root: Path) -> StationConfig:
    if not isinstance(obj, dict):
        raise SettingsError("STATION_INVALID", f"station #{index} must be an object")
    unknown = set(obj) - _STATION_KEYS
    if unknown:
        raise SettingsError("STATION_INVALID", f"station #{index} has unknown keys {sorted(unknown)}")
    sid = obj.get("station_id")
    if not isinstance(sid, str) or not STATION_ID_RE.fullmatch(sid):
        raise SettingsError("STATION_INVALID", f"station #{index}: station_id must match [a-z0-9-]{{1,32}}")
    inbox_raw = obj.get("inbox")
    if not isinstance(inbox_raw, str) or not inbox_raw.strip():
        raise SettingsError("STATION_INVALID", f"station {sid}: inbox is required")
    inbox = Path(os.path.expanduser(inbox_raw.strip()))
    if not inbox.is_absolute():
        raise SettingsError("STATION_INVALID", f"station {sid}: inbox must be absolute")
    reason = forbidden_location_reason(inbox, repo_root)
    if reason:
        raise SettingsError("STATION_INVALID", f"station {sid}: inbox {reason}")
    sidecar = obj.get("sidecar_mode", "off")
    if sidecar not in SIDECAR_MODES:
        raise SettingsError("STATION_INVALID", f"station {sid}: sidecar_mode must be one of {SIDECAR_MODES}")
    policy = obj.get("source_policy", "move")
    if policy not in SOURCE_POLICIES:
        raise SettingsError("STATION_INVALID", f"station {sid}: source_policy must be one of {SOURCE_POLICIES}")
    recursive = obj.get("recursive", False)
    if not isinstance(recursive, bool):
        raise SettingsError("STATION_INVALID", f"station {sid}: recursive must be a JSON boolean")
    return StationConfig(station_id=sid, inbox=inbox, sidecar_mode=sidecar, source_policy=policy, recursive=recursive)


def _parse_stations(env: Mapping[str, str], repo_root: Path) -> tuple[StationConfig, ...]:
    raw = _get(env, "SHERMAN_LINE_STATIONS")
    single_id = _get(env, "SHERMAN_LINE_STATION_ID")
    single_inbox = _get(env, "SHERMAN_LINE_INBOX")
    if raw:
        if single_id or single_inbox:
            raise SettingsError(
                "STATION_INVALID",
                "set either SHERMAN_LINE_STATIONS or SHERMAN_LINE_STATION_ID/SHERMAN_LINE_INBOX, not both",
            )
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SettingsError("STATION_INVALID", f"SHERMAN_LINE_STATIONS is not valid JSON ({exc.msg})") from None
        if not isinstance(items, list):
            raise SettingsError("STATION_INVALID", "SHERMAN_LINE_STATIONS must be a JSON list")
        stations = tuple(_parse_station(obj, i, repo_root) for i, obj in enumerate(items))
    elif single_id or single_inbox:
        if not (single_id and single_inbox):
            raise SettingsError("STATION_INVALID", "SHERMAN_LINE_STATION_ID and SHERMAN_LINE_INBOX go together")
        stations = (_parse_station({"station_id": single_id, "inbox": single_inbox}, 0, repo_root),)
    else:
        stations = ()
    ids = [s.station_id for s in stations]
    if len(set(ids)) != len(ids):
        raise SettingsError("STATION_INVALID", "duplicate station_id")
    inboxes = [_real(s.inbox) for s in stations]
    for i, a in enumerate(inboxes):
        for j, b in enumerate(inboxes):
            if i != j and is_inside(a, b):
                raise SettingsError("STATION_INVALID", f"inbox of {ids[i]} overlaps inbox of {ids[j]}")
    return stations


def _validate_filename_regex(raw: str) -> Optional[str]:
    if raw == "":
        return None
    try:
        compiled = re.compile(raw)
    except re.error as exc:
        raise SettingsError("FILENAME_REGEX_INVALID", f"SHERMAN_LINE_FILENAME_REGEX does not compile ({exc})") from None
    if "sku" not in compiled.groupindex:
        raise SettingsError("FILENAME_REGEX_INVALID", "SHERMAN_LINE_FILENAME_REGEX needs a named group (?P<sku>...)")
    return raw


def _validate_roles(raw: str) -> frozenset[str]:
    roles = frozenset(r.strip().lower() for r in (raw or "admin").split(",") if r.strip())
    if not roles or not roles <= frozenset(RELEASABLE_ROLES):
        raise SettingsError("VALUE_INVALID", f"SHERMAN_RELEASE_ROLES must be a non-empty subset of {RELEASABLE_ROLES}")
    return roles


def load_settings(env: Optional[Mapping[str, str]] = None, repo_root: Optional[Path] = None) -> LineSettings:
    """Parse and validate the line configuration, or raise :class:`SettingsError`.

    ``env`` defaults to ``os.environ``; ``repo_root`` defaults to the source root this
    module is loaded from.  When that root is an app copy, the dev repo it was built
    from is guarded too (:func:`custody_repo_roots`).  Nothing is created on disk
    (see :func:`ensure_layout`).
    """
    env = os.environ if env is None else env
    repo = _real(Path(repo_root) if repo_root is not None else default_repo_root())
    custody_roots = custody_repo_roots(repo)  # raises VALUE_INVALID for an unverifiable copy

    if not _bool(env, "SHERMAN_LINE_ENABLED", False):
        raise SettingsError("LINE_DISABLED", "SHERMAN_LINE_ENABLED is not set")

    root_raw = _get(env, "SHERMAN_LINE_DATA_ROOT")
    if not root_raw:
        raise SettingsError("DATA_ROOT_MISSING", "SHERMAN_LINE_DATA_ROOT is not set")
    root_given = Path(os.path.expanduser(root_raw))
    if not root_given.is_absolute():
        raise SettingsError("DATA_ROOT_NOT_ABSOLUTE", "SHERMAN_LINE_DATA_ROOT must be an absolute path")
    reason = forbidden_location_reason(root_given, repo)
    if reason:
        raise SettingsError("DATA_ROOT_INSIDE_REPO", f"SHERMAN_LINE_DATA_ROOT {reason}")
    if any(is_inside(r, _real(root_given)) for r in custody_roots):
        raise SettingsError("DATA_ROOT_INSIDE_REPO", "SHERMAN_LINE_DATA_ROOT must not contain the repository")
    paths = LinePaths.for_root(_real(root_given))

    stations = _parse_stations(env, repo)
    for st in stations:
        if is_inside(_real(st.inbox), paths.data_root) or is_inside(paths.data_root, _real(st.inbox)):
            raise SettingsError("STATION_INVALID", f"station {st.station_id}: inbox overlaps the data root")

    rss = _int(env, "SHERMAN_LINE_RSS_LIMIT_MB", 32768, lo=256, hi=MAX_RSS_LIMIT_MB, code="RSS_LIMIT_INVALID")
    job_timeout = _int(env, "SHERMAN_LINE_JOB_TIMEOUT_S", 180, lo=10, hi=24 * 3600)
    post_timeout = _int(env, "SHERMAN_LINE_POSTPROCESS_TIMEOUT_S", 300, lo=10, hi=24 * 3600)
    retention = _int(env, "SHERMAN_LINE_LOG_RETENTION_DAYS", 30, lo=1, hi=3650)
    regex = _validate_filename_regex(_get(env, "SHERMAN_LINE_FILENAME_REGEX"))
    roles = _validate_roles(_get(env, "SHERMAN_RELEASE_ROLES"))

    bundle_dir = _abs_path(env, "SHERMAN_LINE_ENGINE_BUNDLE_DIR", paths.engine / "current")
    assert bundle_dir is not None
    python_exe = _abs_path(env, "SHERMAN_LINE_PYTHON", Path(sys.executable))
    assert python_exe is not None
    font = _abs_path(env, "SHERMAN_REPORT_FONT_PATH", None)
    status_path = _abs_path(env, "SHERMAN_LINE_PROGRAM_STATUS_PATH", paths.config / "f1_program_status.json")
    assert status_path is not None
    if forbidden_location_reason(bundle_dir, repo):
        raise SettingsError("VALUE_INVALID", "SHERMAN_LINE_ENGINE_BUNDLE_DIR must be outside the repository")
    if _research_dir_reason(status_path, repo):
        raise SettingsError("VALUE_INVALID", "SHERMAN_LINE_PROGRAM_STATUS_PATH must not be under a research directory")

    return LineSettings(
        enabled=True,
        repo_root=repo,
        paths=paths,
        stations=stations,
        engine_bundle_dir=bundle_dir,
        python_exe=python_exe,
        job_timeout_s=job_timeout,
        postprocess_timeout_s=post_timeout,
        rss_limit_mb=rss,
        filename_regex=regex,
        release_roles=roles,
        allow_self_register=_bool(env, "SHERMAN_ALLOW_SELF_REGISTER", False),
        production_hardening=_bool(env, "SHERMAN_PRODUCTION_HARDENING", False),
        legacy_live_scan_enabled=_bool(env, "SHERMAN_LEGACY_LIVE_SCAN", False),
        legacy_engine_enabled=_bool(env, "SHERMAN_LEGACY_ENGINE", False),
        report_font_path=font,
        log_retention_days=retention,
        program_status_path=status_path,
    )


def _research_dir_reason(path: Path, repo_root: Path) -> Optional[str]:
    """Like :func:`forbidden_location_reason` but only for the research directories."""
    real_path = _real(path)
    for root in custody_repo_roots(repo_root):
        for name in FORBIDDEN_REPO_SUBDIRS:
            target = _real(root / name)
            if is_inside(real_path, target):
                return f"resolves under the repository's {name}/"
    return None


# ---------------------------------------------------------------------------
# Layout and secrets
# ---------------------------------------------------------------------------


def ensure_layout(paths: LinePaths) -> LinePaths:
    """Create the data-root layout.  Every managed directory ends up mode 0700.

    Existing directories are tightened to 0700; symlinked layout directories are
    refused (a symlink could redirect line data into a forbidden location).
    """
    for d in paths.layout_dirs():
        if d.is_symlink():
            raise SettingsError("VALUE_INVALID", f"layout directory {d} is a symlink")
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not d.is_dir():
            raise SettingsError("VALUE_INVALID", f"layout path {d} is not a directory")
        os.chmod(d, 0o700)
    return paths


def _publish_secret_exclusive(target: Path, data: bytes, name: str) -> None:
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    except OSError as exc:
        raise SettingsError("SECRET_INVALID", f"cannot create secret {name} ({type(exc).__name__})") from None
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        try:
            os.unlink(target)
        except OSError:
            pass
        raise SettingsError("SECRET_INVALID", f"cannot write secret {name} ({type(exc).__name__})") from None


def get_secret(paths: LinePaths, name: str) -> bytes:
    """Create (once, atomically) or load the 32-byte key ``secrets/<name>.key`` (0600)."""
    if not _SECRET_NAME_RE.fullmatch(name):
        raise SettingsError("SECRET_INVALID", "secret name must match [a-z0-9_]{1,64}")
    paths.secrets.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(paths.secrets, 0o700)
    target = paths.secrets / f"{name}.key"
    if not target.exists():
        tmp = paths.secrets / f".{name}.{_secrets.token_hex(8)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(_secrets.token_bytes(SECRET_KEY_BYTES))
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(tmp, target)  # atomic publish; fails if another process won the race
            except FileExistsError:
                pass
            except OSError:
                # No hard links on this filesystem (exFAT, some SMB mounts): publish
                # with an exclusive create instead (still never overwrites a winner).
                _publish_secret_exclusive(target, tmp.read_bytes(), name)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
    if target.is_symlink():
        raise SettingsError("SECRET_INVALID", f"secret {name} is a symlink")
    data = target.read_bytes()
    if len(data) != SECRET_KEY_BYTES:
        raise SettingsError("SECRET_INVALID", f"secret {name} has {len(data)} bytes, expected {SECRET_KEY_BYTES}")
    os.chmod(target, 0o600)
    return data
