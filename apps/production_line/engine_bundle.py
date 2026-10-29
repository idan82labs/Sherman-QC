"""Pinned, immutable engine bundles and runnable app copies.

Why
    The line must never execute mutable research state from the dev repo.  Run from
    the repo, the legacy worker silently reads
    ``output/improvement_loop/best_config.latest.json`` (``backend/bend_inspection_pipeline.py:224-227``)
    and ``output/unary_models/latest.json`` (``domains/bend/services/structured_count_reporting_service.py:15``),
    and the legacy server writes ``data/``, ``uploads/`` and ``output/`` next to the
    source.  A bundle is a copy of exactly the source the line needs, with a sha256
    manifest that :func:`verify` re-checks.

Profiles
    ``engine`` (default)
        The import closure of the production engine: every module under
        ``domains/production/engine/**`` and ``contracts/production/**`` (E1 compile,
        E3 inspect and their contracts), the C06 entry points
        ``domains/production/inspection/postprocess_cli.py`` and
        ``domains/production/reporting/render_cli.py``, plus :data:`EXTRA_ALLOWLIST`
        (the CAD extractor V10 chain, which E1 may call and which imports
        ``scripts`` v7/v9 through ``from scripts import ...``) and any
        ``--extra-module``.  ``--include-legacy-worker`` adds
        ``backend/bend_analysis_worker.py`` for E4's control arm only.
        Only ``.py`` files are copied, plus font/licence resources under
        ``domains/production/**`` (the PDF's bundled Heebo) and the pinned zero-shot
        lane (:data:`ZEROSHOT_PIN_REL`: the pinned sources, the pin and the model).
    ``app`` (plan §10 amendment 10)
        A runnable app copy so E2E and shop-floor servers never run from the dev
        repo: the ``.py`` sources of ``backend/``, ``apps/``, ``domains/``,
        ``contracts/``, ``infrastructure/`` and ``scripts/`` (whole trees, because the legacy server
        launches workers with ``-m`` and pins some ``scripts/*.py`` by path), the
        import closure of ``apps/api/main.py``, ``apps/production_line/server.py`` and
        ``apps/production_line/service_main.py``, the same resources, and
        ``frontend/dist``.  The default frontend mode for this profile is ``build``:
        Vite builds the current ``frontend/react`` source straight into the copy, so
        the copy never serves a stale UI.  ``copy``/``auto`` copy an existing dist and
        warn ``FRONTEND_DIST_STALE`` when any frontend source file is newer than its
        ``index.html`` (``--strict-frontend`` turns that into a refusal).  The
        manifest records ``frontend.source_sha256`` over the frontend source.
        Start it with ``cwd=<copy>`` and ``PYTHONPATH=<copy>:<copy>/backend``
        (:func:`app_runtime_env`); its legacy state (``data/users.db``,
        ``data/qc_jobs.db``, ``uploads/``) then lives inside the copy.

Closure
    An AST scan from the entry files resolves ``import``/``from`` statements at any
    nesting level (so function-level lazy imports count), relative imports,
    ``importlib.import_module("literal")``/``__import__("literal")`` and
    ``[..., "-m", "pkg.mod"]`` argument lists, against the roots ``R`` and
    ``R/backend`` in that order (the runtime ``PYTHONPATH``).  Every package
    ``__init__.py`` on a resolved module's path is included and scanned.  Missing
    entries are skipped with a warning.

Source repo
    Every manifest records ``source_repo_root``: the dev repo the bundle was built
    from (for a bundle built from an app copy, the original dev repo that copy
    records).  ``apps.production_line.settings`` and ``bootstrap_admin`` use it so
    their custody checks guard the dev repo even when running from a copy; the
    builder itself refuses destinations and inputs inside either root.

Custody
    ``data/``, ``output/``, ``outputs/``, ``artifacts/``, ``uploads/``,
    ``live_scans/``, ``tests/``, ``__pycache__``, ``node_modules``, VCS and venv
    directories are excluded at any depth *before* any filesystem access, so the
    builder never lists, stats or opens them, even when an entry imports from one.
    ``research/**/*.json`` is never copied.  ``--engine-config`` may not come from a
    research directory.

Manifest (``bundle_manifest.json``, written last)
    ``bundle_id`` = sha256 over the sorted ``relpath:sha256`` lines, the canonical
    dependency-version JSON and the engine-config sha (or ``none``), first 16 hex.
    It is identical for identical inputs.  Other fields: ``schema``, ``profile``,
    ``source_repo_root``, ``frontend`` (app profile), ``created_at``, ``git_sha``, ``git_dirty_count`` (tracked changes under the
    bundled source trees only), ``python``, ``dependency_versions``, ``entries``,
    ``extra_modules``, ``files``, ``engine_config`` and ``warnings``.

Runtime config
    The runner (C02) sets ``BEND_INSPECTION_CONFIG`` to the bundled
    ``<bundle>/config/bend_runtime_config.json`` when one was pinned with
    ``--engine-config``, and otherwise to ``<bundle>/config/NONE`` (which does not
    exist, so the pipeline uses its built-in defaults).  The repo's
    ``output/improvement_loop`` and ``output/unary_models`` are never consulted
    because they are not in the bundle and the worker runs with
    ``cwd=<bundle>``.  :func:`engine_runtime_env` builds that environment.

CLI::

    python -m apps.production_line.engine_bundle build --repo-root R --dest D \\
        [--profile engine|app] [--engine-config FILE] [--extra-module MOD]... \\
        [--entry REL]... [--include-legacy-worker] [--frontend auto|copy|build|none] \\
        [--frontend-dist DIR] [--strict-frontend] [--force]
    python -m apps.production_line.engine_bundle --profile app --out D    # build shorthand
    python -m apps.production_line.engine_bundle verify --dest D [--check-dependencies]

``--force`` replaces an existing non-empty destination only when it already holds a
bundle (a ``bundle_manifest.json`` of this schema); anything else is refused with
``DEST_NOT_A_BUNDLE`` so a mistyped ``--out`` can never wipe a data root.

Verify reports, besides manifest mismatches, any native extension (``.so``,
``.pyd``, ``.dylib``), sourceless ``.pyc`` outside ``__pycache__`` and any
``__pycache__`` directory, in either profile, because the import system would
prefer them over the verified sources (a timestamp ``.pyc`` whose header matches a
verified source's mtime and size runs *instead of* that source).  Engines and app
copies therefore run with ``PYTHONDONTWRITEBYTECODE=1`` (``engine_runtime_env`` /
``app_runtime_env``); launching an app copy with another environment makes it write
a cache, which verify then reports, failing closed.

Exit codes: 0 ok, 1 verify mismatches, 2 refused/invalid input.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "ENGINE_CONFIG_REL",
    "NO_CONFIG_REL",
    "HARD_EXCLUDED_DIRS",
    "ENGINE_ENTRY_FILES",
    "ENGINE_SEED_TREES",
    "EXTRA_ALLOWLIST",
    "LEGACY_WORKER_ENTRY",
    "APP_ENTRY_FILES",
    "APP_SOURCE_TREES",
    "DEPENDENCY_DISTRIBUTIONS",
    "DEFAULT_FRONTEND_MODE",
    "recorded_source_repo",
    "frontend_source_fingerprint",
    "BundleError",
    "BuildResult",
    "compute_closure",
    "build_bundle",
    "verify",
    "read_manifest",
    "compute_bundle_id",
    "engine_runtime_env",
    "app_runtime_env",
    "main",
]

MANIFEST_NAME = "bundle_manifest.json"
MANIFEST_SCHEMA = "sherman_engine_bundle_manifest_v1"
ENGINE_CONFIG_REL = "config/bend_runtime_config.json"
NO_CONFIG_REL = "config/NONE"

#: Directory names never entered, at any depth.
HARD_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        "data",
        "output",
        "outputs",
        "artifacts",
        "uploads",
        "live_scans",
        "tests",
        "__pycache__",
        "node_modules",
        ".git",
        ".venv",
        ".venv-f1-structured",
        ".mypy_cache",
        ".pytest_cache",
    }
)
#: Repo research directories; inputs (``--engine-config``) may not come from them.
RESEARCH_DIRS: tuple[str, ...] = ("data", "output", "outputs", "artifacts", "uploads", "live_scans")

ENGINE_ENTRY_FILES: tuple[str, ...] = (
    "domains/production/inspection/postprocess_cli.py",
    "domains/production/reporting/render_cli.py",
)
#: Entry files of the zero-shot overlay steps (runner 3b); added to the engine profile when present.
ZEROSHOT_ENTRY_FILES: tuple[str, ...] = (
    "domains/production/zeroshot/lane_cli.py",
    "domains/production/zeroshot/overlay_cli.py",
)
ENGINE_SEED_TREES: tuple[str, ...] = ("domains/production/engine", "contracts/production")
#: Modules always added to the engine closure: the V10 CAD extractor chain
#: (plan §10 optional item; it imports ``scripts`` v7/v9 modules).
EXTRA_ALLOWLIST: tuple[str, ...] = ("scripts.extract_step_topology_bend_axes_v10",)
LEGACY_WORKER_ENTRY = "backend/bend_analysis_worker.py"

APP_ENTRY_FILES: tuple[str, ...] = (
    "apps/api/main.py",
    "apps/production_line/server.py",
    "apps/production_line/service_main.py",
)
APP_SOURCE_TREES: tuple[str, ...] = ("backend", "apps", "domains", "contracts", "infrastructure", "scripts")
FRONTEND_DIST_REL = "frontend/dist"

#: Frontend source whose hash is recorded and whose mtimes decide staleness.
FRONTEND_REACT_REL = "frontend/react"
FRONTEND_SOURCE_TREES: tuple[str, ...] = ("frontend/react/src", "frontend/react/public")
FRONTEND_SOURCE_FILES: tuple[str, ...] = (
    "frontend/react/index.html",
    "frontend/react/package.json",
    "frontend/react/package-lock.json",
    "frontend/react/vite.config.ts",
    "frontend/react/tsconfig.json",
    "frontend/react/tsconfig.node.json",
    "frontend/react/tailwind.config.js",
    "frontend/react/postcss.config.js",
)
#: Files the import system would load in preference to (or without) a verified source.
_CODE_ARTIFACT_SUFFIXES: tuple[str, ...] = (".so", ".pyd", ".dylib", ".pyc")

#: The pinned zero-shot lane (``domains/production/zeroshot``): its pin lists every source file the frozen
#: lane executes (``research/zeroshot/**`` and the modules it reuses) and the vendored selector model. Both
#: profiles copy exactly those files (``.py`` sources, the pin JSON and the model), so the runner's zs step can
#: re-verify the fingerprint inside the bundle; a pin that cannot be read adds nothing (``ZEROSHOT_PIN_UNREADABLE``).
ZEROSHOT_PIN_REL = "domains/production/zeroshot/assets/zs_lane_v1.pin.json"
RESOURCE_TREES: tuple[str, ...] = ("domains/production",)
RESOURCE_SUFFIXES: frozenset[str] = frozenset({".ttf", ".otf", ".woff", ".woff2"})
RESOURCE_NAMES_PREFIX: tuple[str, ...] = ("OFL", "LICENSE", "LICENCE")

DEPENDENCY_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "scipy",
    "open3d",
    "trimesh",
    "reportlab",
    "pymupdf",
    "cadquery-ocp",
    "ezdxf",
    "fastapi",
    "pydantic",
)

_PROFILES = ("engine", "app")
_FRONTEND_MODES = ("auto", "copy", "build", "none")
#: Frontend mode used when none is given: the app copy builds the current source.
DEFAULT_FRONTEND_MODE: Mapping[str, str] = {"app": "build", "engine": "none"}


class BundleError(Exception):
    """Build refused.  ``code`` is a stable identifier."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class BuildResult:
    dest: Path
    bundle_id: str
    profile: str
    files: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _excluded_parts(parts: Sequence[str]) -> bool:
    """True when a repo-relative *file* path (as parts) lies under a hard-excluded
    directory or is a research JSON file.  Pure string logic: no filesystem access.
    (Pass ``dir_parts + ("x",)`` to test a directory.)"""
    if any(p in HARD_EXCLUDED_DIRS for p in parts[:-1]):
        return True
    if parts and parts[0] == "research" and parts[-1].endswith(".json"):
        return True
    return False


def _rel_parts(rel: str) -> tuple[str, ...]:
    return tuple(PurePosixPath(rel).parts)


def _is_test_file(name: str) -> bool:
    return name == "conftest.py" or (name.startswith("test_") and name.endswith(".py"))


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _real(p: Path) -> Path:
    return Path(os.path.realpath(os.fspath(p)))


def _walk_files(repo: Path, tree_rel: str) -> Iterable[str]:
    """Repo-relative POSIX paths of regular files under ``tree_rel``, pruning
    excluded directories before descending (they are never listed)."""
    if _excluded_parts(_rel_parts(tree_rel) + ("x",)):
        return
    top = repo / tree_rel
    if not top.is_dir() or top.is_symlink():
        return
    for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in HARD_EXCLUDED_DIRS and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(repo).as_posix()
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}"
            if _excluded_parts(_rel_parts(rel)):
                continue
            yield rel


# ---------------------------------------------------------------------------
# Import closure
# ---------------------------------------------------------------------------


class _Resolver:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        # (filesystem root, repo-relative prefix parts) in runtime PYTHONPATH order.
        self.roots: tuple[tuple[Path, tuple[str, ...]], ...] = ((repo, ()), (repo / "backend", ("backend",)))

    def resolve(self, dotted: str) -> Optional[tuple[str, tuple[str, ...]]]:
        """``(relpath of the module file or "", repo-relative parts of the module)``
        for a module that exists in the repo, else None.  Namespace packages return
        an empty relpath.

        Mirrors ``importlib.machinery.PathFinder``: a regular package or module on
        *any* root beats a namespace portion (bare directory) on an earlier root,
        so a namespace result is returned only when no root has a regular one."""
        if not dotted or any(not p.isidentifier() for p in dotted.split(".")):
            return None
        parts = tuple(dotted.split("."))
        namespace_hit: Optional[tuple[str, tuple[str, ...]]] = None
        for root, prefix in self.roots:
            rel = prefix + parts
            if _excluded_parts(rel + ("__init__.py",)):
                continue  # never touch excluded directories
            pkg_init = root.joinpath(*parts, "__init__.py")
            if pkg_init.is_file():
                return "/".join(rel + ("__init__.py",)), rel
            mod = root.joinpath(*parts[:-1], parts[-1] + ".py")
            if mod.is_file() and not _excluded_parts(prefix + parts[:-1] + (parts[-1] + ".py",)):
                return "/".join(prefix + parts[:-1] + (parts[-1] + ".py",)), rel
            if namespace_hit is None and root.joinpath(*parts).is_dir():
                namespace_hit = ("", rel)
        return namespace_hit

    def package_inits(self, rel_parts: tuple[str, ...]) -> list[str]:
        """``__init__.py`` files of every package on a module's path."""
        out: list[str] = []
        for i in range(1, len(rel_parts)):
            init = self.repo.joinpath(*rel_parts[:i], "__init__.py")
            rel = "/".join(rel_parts[:i] + ("__init__.py",))
            if not _excluded_parts(_rel_parts(rel)) and init.is_file():
                out.append(rel)
        return out


def _module_name_of(rel: str) -> tuple[str, bool]:
    """(dotted module name relative to the repo root, is_package)."""
    parts = list(PurePosixPath(rel).parts)
    is_pkg = parts[-1] == "__init__.py"
    if is_pkg:
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts), is_pkg


def _literal_imports(tree: ast.AST) -> list[str]:
    """Module names named by string literals: importlib.import_module("x"),
    __import__("x"), and ``"-m", "x"`` pairs in list/tuple literals."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            fname = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            first = node.args[0]
            if fname in ("import_module", "__import__") and isinstance(first, ast.Constant) and isinstance(first.value, str):
                out.append(first.value)
        elif isinstance(node, (ast.List, ast.Tuple)):
            elts = node.elts
            for a, b in zip(elts, elts[1:]):
                if (
                    isinstance(a, ast.Constant)
                    and a.value == "-m"
                    and isinstance(b, ast.Constant)
                    and isinstance(b.value, str)
                ):
                    out.append(b.value)
    return out


def _imports_of(rel: str, source: str) -> list[str]:
    """Candidate absolute module names imported by a module (any nesting level)."""
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        raise BundleError("SOURCE_UNPARSEABLE", f"{rel}:{exc.lineno}: {exc.msg}") from None
    mod_name, is_pkg = _module_name_of(rel)
    package = mod_name if is_pkg else mod_name.rpartition(".")[0]
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".") if package else []
                if node.level - 1 > len(base_parts):
                    continue
                base_parts = base_parts[: len(base_parts) - (node.level - 1)] if node.level > 1 else base_parts
                base = ".".join(p for p in base_parts if p)
                target = ".".join(p for p in (base, node.module or "") if p)
            else:
                target = node.module or ""
            if target:
                out.append(target)
            for alias in node.names:
                if alias.name != "*" and target:
                    out.append(f"{target}.{alias.name}")
    out.extend(_literal_imports(tree))
    return out


def compute_closure(
    repo_root: Path,
    entries: Iterable[str],
    extra_modules: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """``(sorted repo-relative .py files, warnings)`` reachable from ``entries``
    (repo-relative files) and ``extra_modules`` (dotted names)."""
    repo = _real(Path(repo_root))
    resolver = _Resolver(repo)
    warnings: list[str] = []
    seen: set[str] = set()
    queue: list[str] = []

    def add(rel: str) -> None:
        if rel and rel not in seen and not _excluded_parts(_rel_parts(rel)):
            seen.add(rel)
            queue.append(rel)

    for rel in entries:
        if _excluded_parts(_rel_parts(rel)):
            warnings.append(f"ENTRY_EXCLUDED {rel}")
            continue
        path = repo / rel
        if not path.is_file():
            warnings.append(f"ENTRY_MISSING {rel}")
            continue
        add(rel)
        for init in resolver.package_inits(_rel_parts(rel)):
            add(init)
    for dotted in extra_modules:
        found = resolver.resolve(dotted)
        if found is None or not found[0]:
            warnings.append(f"EXTRA_MODULE_MISSING {dotted}")
            continue
        add(found[0])
        for init in resolver.package_inits(found[1]):
            add(init)

    while queue:
        rel = queue.pop()
        path = repo / rel
        real = _real(path)
        if not _inside(real, repo) or _excluded_parts(_rel_parts(real.relative_to(repo).as_posix())):
            warnings.append(f"SYMLINK_OUTSIDE_REPO {rel}")
            seen.discard(rel)
            continue
        source = path.read_text(encoding="utf-8", errors="surrogateescape")
        for name in _imports_of(rel, source):
            found = resolver.resolve(name)
            if found is None:
                continue
            mod_rel, mod_parts = found
            add(mod_rel)
            for init in resolver.package_inits(mod_parts):
                add(init)
    return sorted(seen), warnings


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------


def dependency_versions() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    for dist in DEPENDENCY_DISTRIBUTIONS:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


def compute_bundle_id(files: Mapping[str, str], deps: Mapping[str, Optional[str]], config_sha: Optional[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}:{files[rel]}\n".encode("utf-8"))
    h.update(json.dumps(dict(deps), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(b"\n")
    h.update((config_sha or "none").encode("ascii"))
    return h.hexdigest()[:16]


def _git_info(repo: Path, trees: Sequence[str]) -> tuple[Optional[str], Optional[int]]:
    git = shutil.which("git")
    if git is None:
        return None, None
    try:
        head = subprocess.run(
            [git, "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15
        )
        if head.returncode != 0:
            return None, None
        sha = head.stdout.strip() or None
        # Pathspec-limited and tracked-only, so git never enumerates research dirs.
        status = subprocess.run(
            [git, "-C", str(repo), "status", "--porcelain", "--untracked-files=no", "--", *trees],
            capture_output=True,
            text=True,
            timeout=60,
        )
        dirty = len([ln for ln in status.stdout.splitlines() if ln.strip()]) if status.returncode == 0 else None
        return sha, dirty
    except (OSError, subprocess.TimeoutExpired):
        return None, None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _collect_resources(repo: Path) -> list[str]:
    out: list[str] = []
    for tree in RESOURCE_TREES:
        for rel in _walk_files(repo, tree):
            name = PurePosixPath(rel).name
            if PurePosixPath(rel).suffix.lower() in RESOURCE_SUFFIXES or name.upper().startswith(RESOURCE_NAMES_PREFIX):
                out.append(rel)
    return out


def _zeroshot_files(repo: Path, warnings: list[str]) -> list[str]:
    """The files of the pinned zero-shot lane (:data:`ZEROSHOT_PIN_REL`): pinned sources, the pin, the model."""
    pin_path = repo / ZEROSHOT_PIN_REL
    if not pin_path.is_file():
        return []
    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
        rels = set(pin["code_fingerprint"]["files"]) | {ZEROSHOT_PIN_REL, str(pin["model"]["rel_path"])}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"ZEROSHOT_PIN_UNREADABLE {type(exc).__name__}")
        return []
    out: list[str] = []
    for rel in sorted(rels):
        parts = _rel_parts(rel)
        if PurePosixPath(rel).is_absolute() or ".." in parts or _excluded_parts(parts):
            warnings.append(f"ZEROSHOT_PIN_PATH_REFUSED {rel}")
            continue
        if not (repo / rel).is_file():
            warnings.append(f"ZEROSHOT_PIN_FILE_MISSING {rel}")
            continue
        out.append(rel)
    return out


def _collect_tree_sources(repo: Path, trees: Sequence[str]) -> list[str]:
    out: list[str] = []
    for tree in trees:
        for rel in _walk_files(repo, tree):
            name = PurePosixPath(rel).name
            if name.endswith(".py") and not _is_test_file(name):
                out.append(rel)
    return out


def _copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst, follow_symlinks=True)
    os.chmod(dst, 0o444)
    return _sha256_file(dst)


def recorded_source_repo(root: Path) -> Optional[Path]:
    """``source_repo_root`` recorded by a bundle at ``root``, or None when ``root``
    has no readable manifest of this schema that records one."""
    try:
        data = json.loads((Path(root) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        return None
    src = data.get("source_repo_root")
    if isinstance(src, str) and src and Path(src).is_absolute():
        return _real(Path(src))
    return None


def _custody_roots(repo: Path) -> tuple[Path, ...]:
    """The repo being bundled plus, when it is itself an app copy, its dev repo."""
    src = recorded_source_repo(repo)
    return (repo,) if src is None or src == repo else (repo, src)


def _research_input(path: Path, repo: Path) -> bool:
    real = _real(path)
    return any(_inside(real, _real(root / d)) for root in _custody_roots(repo) for d in RESEARCH_DIRS)


def _frontend_source_files(repo: Path) -> list[str]:
    out = [rel for tree in FRONTEND_SOURCE_TREES for rel in _walk_files(repo, tree)]
    out.extend(rel for rel in FRONTEND_SOURCE_FILES if (repo / rel).is_file() and not (repo / rel).is_symlink())
    return sorted(set(out))


def frontend_source_fingerprint(repo: Path) -> tuple[Optional[str], Optional[float]]:
    """``(sha256 over the sorted relpath:sha lines of the frontend source, newest
    mtime)``, or ``(None, None)`` when the repo has no frontend source."""
    rels = _frontend_source_files(repo)
    if not rels:
        return None, None
    h = hashlib.sha256()
    newest = 0.0
    for rel in rels:
        p = repo / rel
        h.update(f"{rel}:{_sha256_file(p)}\n".encode("utf-8"))
        newest = max(newest, p.stat().st_mtime)
    return h.hexdigest(), newest


def _frontend(
    mode: str,
    repo: Path,
    staging: Path,
    dist_src: Optional[Path],
    warnings: list[str],
    *,
    newest_source_mtime: Optional[float] = None,
    strict: bool = False,
) -> list[tuple[str, Path]]:
    """Returns (bundle relpath, absolute source) pairs of the frontend dist, building
    it into ``staging`` first when ``mode == 'build'``.  A copied dist older than the
    newest frontend source file warns ``FRONTEND_DIST_STALE`` (refused when
    ``strict``)."""
    if mode == "none":
        return []
    out_dir = staging / FRONTEND_DIST_REL
    if mode == "build":
        react = repo / "frontend" / "react"
        vite = react / "node_modules" / ".bin" / "vite"
        if not vite.exists():
            raise BundleError("FRONTEND_BUILD_FAILED", f"vite not found at {vite}")
        proc = subprocess.run(
            [str(vite), "build", "--outDir", str(out_dir), "--emptyOutDir"],
            cwd=str(react),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0 or not (out_dir / "index.html").is_file():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            raise BundleError("FRONTEND_BUILD_FAILED", " | ".join(tail)[:800])
        src_root = out_dir
    else:
        src_root = Path(dist_src) if dist_src is not None else repo / FRONTEND_DIST_REL
        if _research_input(src_root, repo):
            raise BundleError("FRONTEND_DIST_FORBIDDEN", "a frontend dist may not come from a research directory")
        if not (src_root / "index.html").is_file():
            if mode == "copy":
                raise BundleError("FRONTEND_DIST_MISSING", f"{src_root}/index.html not found")
            warnings.append(f"FRONTEND_DIST_MISSING {src_root}")
            return []
        if newest_source_mtime is not None and (src_root / "index.html").stat().st_mtime < newest_source_mtime:
            msg = f"{src_root} is older than the frontend source (rebuild with --frontend build)"
            if strict:
                raise BundleError("FRONTEND_DIST_STALE", msg)
            warnings.append(f"FRONTEND_DIST_STALE {msg}")
    pairs: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(src_root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in HARD_EXCLUDED_DIRS)
        for name in sorted(filenames):
            src = Path(dirpath) / name
            if src.is_symlink():
                warnings.append(f"FRONTEND_SYMLINK_SKIPPED {src.relative_to(src_root).as_posix()}")
                continue
            rel = f"{FRONTEND_DIST_REL}/{src.relative_to(src_root).as_posix()}"
            pairs.append((rel, src))
    return pairs


def build_bundle(
    repo_root: Path,
    dest: Path,
    *,
    profile: str = "engine",
    engine_config: Optional[Path] = None,
    extra_modules: Sequence[str] = (),
    entries: Optional[Sequence[str]] = None,
    include_legacy_worker: bool = False,
    frontend: Optional[str] = None,
    frontend_dist: Optional[Path] = None,
    strict_frontend: bool = False,
    force: bool = False,
    deps: Optional[Mapping[str, Optional[str]]] = None,
    now: Optional[datetime] = None,
) -> BuildResult:
    """Build a bundle at ``dest`` (atomically: staged next to it, then renamed).

    ``entries`` replaces the profile's default entry files (seed trees, the extra
    allowlist and the app source trees still apply).  ``deps`` overrides the
    dependency fingerprint (tests).  ``frontend`` defaults to
    :data:`DEFAULT_FRONTEND_MODE` for the profile (``build`` for ``app``).
    """
    if profile not in _PROFILES:
        raise BundleError("PROFILE_INVALID", f"profile must be one of {_PROFILES}")
    if frontend is None:
        frontend = DEFAULT_FRONTEND_MODE[profile]
    if frontend not in _FRONTEND_MODES:
        raise BundleError("FRONTEND_MODE_INVALID", f"frontend must be one of {_FRONTEND_MODES}")
    repo = _real(Path(repo_root))
    if not repo.is_dir():
        raise BundleError("REPO_ROOT_INVALID", f"{repo} is not a directory")
    dest = Path(dest)
    if not dest.is_absolute():
        raise BundleError("DEST_INVALID", "--dest must be an absolute path")
    real_dest = _real(dest)
    custody_roots = _custody_roots(repo)
    for root in custody_roots:
        if _inside(real_dest, root) or _inside(root, real_dest):
            raise BundleError("DEST_INSIDE_REPO", f"{real_dest} overlaps the repository {root}")
    if real_dest.exists() and not real_dest.is_dir():
        raise BundleError("DEST_INVALID", f"{real_dest} exists and is not a directory")
    if real_dest.exists() and any(real_dest.iterdir()):
        if not force:
            raise BundleError("DEST_NOT_EMPTY", f"{real_dest} exists and is not empty (use --force)")
        if not _is_bundle_dir(real_dest):
            raise BundleError(
                "DEST_NOT_A_BUNDLE",
                f"{real_dest} is not empty and holds no {MANIFEST_SCHEMA} manifest; --force only replaces a bundle",
            )
    source_repo_root = custody_roots[-1]

    warnings: list[str] = []
    config_bytes: Optional[bytes] = None
    if engine_config is not None:
        cfg = Path(engine_config)
        if _research_input(cfg, repo):
            raise BundleError("ENGINE_CONFIG_FORBIDDEN", "an engine config may not come from a research directory")
        try:
            config_bytes = cfg.read_bytes()
            parsed = json.loads(config_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise BundleError("ENGINE_CONFIG_INVALID", f"{cfg}: {type(exc).__name__}") from None
        if not isinstance(parsed, dict):
            raise BundleError("ENGINE_CONFIG_INVALID", "engine config must be a JSON object")

    # 1. Source selection.
    if profile == "engine":
        entry_files = list(entries) if entries is not None else (
            list(ENGINE_ENTRY_FILES) + [e for e in ZEROSHOT_ENTRY_FILES if (repo / e).is_file()])
        if include_legacy_worker:
            entry_files.append(LEGACY_WORKER_ENTRY)
        seed: list[str] = []
        for tree in ENGINE_SEED_TREES:
            found = _collect_tree_sources(repo, [tree])
            if not found:
                warnings.append(f"SEED_TREE_MISSING {tree}")
            seed.extend(found)
        extras = list(EXTRA_ALLOWLIST) + list(extra_modules)
        closure, w = compute_closure(repo, entry_files + seed, extras)
        warnings.extend(w)
        sources = closure
        trees_for_git = sorted({PurePosixPath(r).parts[0] for r in sources}) or ["."]
    else:
        entry_files = list(entries) if entries is not None else list(APP_ENTRY_FILES)
        tree_sources = _collect_tree_sources(repo, APP_SOURCE_TREES)
        closure, w = compute_closure(repo, entry_files, list(extra_modules))
        warnings.extend(w)
        sources = sorted(set(tree_sources) | set(closure))
        trees_for_git = list(APP_SOURCE_TREES) + ["frontend/react/src"]
    sources = sorted(set(sources) | set(_collect_resources(repo)) | set(_zeroshot_files(repo, warnings)))
    if not sources:
        raise BundleError("CLOSURE_EMPTY", "nothing to bundle (no entry files found)")

    # 2. Stage.
    parent = real_dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{real_dest.name}.staging-{secrets.token_hex(6)}"
    staging.mkdir(mode=0o755)
    try:
        files: dict[str, str] = {}
        for rel in sources:
            src = repo / rel
            real_src = _real(src)
            if not _inside(real_src, repo) or _excluded_parts(_rel_parts(real_src.relative_to(repo).as_posix())):
                warnings.append(f"SYMLINK_OUTSIDE_REPO {rel}")
                continue
            files[rel] = _copy_file(real_src, staging / rel)
        frontend_entry: Optional[dict[str, object]] = None
        if profile == "app":
            fe_sha, fe_newest = frontend_source_fingerprint(repo)
            stale_before = len(warnings)
            for rel, src in _frontend(
                frontend, repo, staging, frontend_dist, warnings, newest_source_mtime=fe_newest, strict=strict_frontend
            ):
                dst = staging / rel
                if _real(dst) == _real(src):  # built in place
                    os.chmod(dst, 0o444)
                    files[rel] = _sha256_file(dst)
                else:
                    files[rel] = _copy_file(src, dst)
            frontend_entry = {
                "mode": frontend,
                "source_sha256": fe_sha,
                "included": any(r.startswith(FRONTEND_DIST_REL + "/") for r in files),
                "stale": any(w.startswith("FRONTEND_DIST_STALE") for w in warnings[stale_before:]),
            }
        (staging / "config").mkdir(exist_ok=True)
        engine_config_entry: Optional[dict[str, str]] = None
        if config_bytes is not None:
            cfg_path = staging / ENGINE_CONFIG_REL
            cfg_path.write_bytes(config_bytes)
            os.chmod(cfg_path, 0o444)
            engine_config_entry = {"relpath": ENGINE_CONFIG_REL, "sha256": hashlib.sha256(config_bytes).hexdigest()}

        dep_versions = dict(deps) if deps is not None else dependency_versions()
        bundle_id = compute_bundle_id(files, dep_versions, engine_config_entry["sha256"] if engine_config_entry else None)
        git_sha, git_dirty = _git_info(repo, trees_for_git)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "profile": profile,
            "bundle_id": bundle_id,
            "source_repo_root": str(source_repo_root),
            "created_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "git_sha": git_sha,
            "git_dirty_count": git_dirty,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "dependency_versions": dep_versions,
            "entries": sorted(entry_files),
            "extra_modules": sorted(set((list(EXTRA_ALLOWLIST) if profile == "engine" else []) + list(extra_modules))),
            "files": dict(sorted(files.items())),
            "engine_config": engine_config_entry,
            "warnings": sorted(set(warnings)),
        }
        if frontend_entry is not None:
            manifest["frontend"] = frontend_entry
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o444)

        # 3. Publish.
        if real_dest.exists() and any(real_dest.iterdir()):
            old = parent / f".{real_dest.name}.old-{secrets.token_hex(6)}"
            os.rename(real_dest, old)
            os.rename(staging, real_dest)
            _rmtree(old)
        else:
            if real_dest.exists():
                real_dest.rmdir()
            os.rename(staging, real_dest)
    except BaseException:
        _rmtree(staging)
        raise
    return BuildResult(
        dest=real_dest, bundle_id=bundle_id, profile=profile, files=files, warnings=manifest["warnings"], manifest=manifest
    )


def _is_bundle_dir(path: Path) -> bool:
    """True when ``path`` holds a readable manifest of :data:`MANIFEST_SCHEMA`."""
    mpath = Path(path) / MANIFEST_NAME
    if mpath.is_symlink() or not mpath.is_file():
        return False
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(data, dict) and data.get("schema") == MANIFEST_SCHEMA


def _rmtree(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func, p, _exc):  # read-only files inside writable dirs
        os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(p)

    shutil.rmtree(path, onerror=_onerror)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def read_manifest(dest: Path) -> dict:
    return json.loads((Path(dest) / MANIFEST_NAME).read_text(encoding="utf-8"))


#: Directories a running app copy creates for itself (legacy state); ignored by verify.
_RUNTIME_DIRS = HARD_EXCLUDED_DIRS


def verify(dest: Path, *, check_dependencies: bool = False) -> list[str]:
    """Mismatches between ``dest`` and its manifest; an empty list means intact.

    Reports ``MANIFEST_MISSING``/``MANIFEST_INVALID``, ``MISSING <rel>``,
    ``MODIFIED <rel>``, ``UNEXPECTED <rel>`` (any extra file in an engine bundle;
    in an app copy an extra ``.py``/``.pth`` file; in both, a ``__pycache__/``
    directory, a native extension or a ``.pyc`` outside ``__pycache__``),
    ``CONFIG_MISSING``/``CONFIG_MODIFIED``/
    ``UNEXPECTED config/...``, ``BUNDLE_ID_MISMATCH`` and, with
    ``check_dependencies``, ``DEPENDENCY_MISMATCH <dist>``.
    """
    root = Path(dest)
    try:
        manifest = read_manifest(root)
        files: Mapping[str, str] = manifest["files"]
        profile = manifest.get("profile", "engine")
        deps = manifest.get("dependency_versions", {})
        cfg = manifest.get("engine_config")
    except FileNotFoundError:
        return ["MANIFEST_MISSING"]
    except (OSError, ValueError, KeyError, TypeError):
        return ["MANIFEST_INVALID"]
    problems: list[str] = []
    for rel in sorted(files):
        if _excluded_parts(_rel_parts(rel)) and not rel.startswith(FRONTEND_DIST_REL + "/"):
            problems.append(f"MANIFEST_INVALID {rel}")
            continue
        path = root / rel
        if not path.is_file() or path.is_symlink():
            problems.append(f"MISSING {rel}")
        elif _sha256_file(path) != files[rel]:
            problems.append(f"MODIFIED {rel}")

    cfg_path = root / ENGINE_CONFIG_REL
    if cfg:
        if not cfg_path.is_file():
            problems.append("CONFIG_MISSING")
        elif _sha256_file(cfg_path) != cfg.get("sha256"):
            problems.append("CONFIG_MODIFIED")
    elif cfg_path.exists():
        problems.append(f"UNEXPECTED {ENGINE_CONFIG_REL}")
    if (root / NO_CONFIG_REL).exists():
        problems.append(f"UNEXPECTED {NO_CONFIG_REL}")

    known = set(files) | {MANIFEST_NAME, ENGINE_CONFIG_REL}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if "__pycache__" in dirnames:
            # Engines and app copies run with PYTHONDONTWRITEBYTECODE, so a cache is
            # planted (or unverifiable) code that would shadow the verified sources.
            problems.append(f"UNEXPECTED {'' if rel_dir == '.' else rel_dir + '/'}__pycache__/")
        dirnames[:] = sorted(d for d in dirnames if d not in _RUNTIME_DIRS)
        for name in sorted(filenames):
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            if rel in known:
                continue
            in_dist = rel.startswith(FRONTEND_DIST_REL + "/")
            code_artifact = name.lower().endswith(_CODE_ARTIFACT_SUFFIXES) or ".so." in name.lower()
            if profile == "app" and not in_dist and not code_artifact and not (
                name.endswith(".py") or name.endswith(".pth")
            ):
                continue
            problems.append(f"UNEXPECTED {rel}")

    expected_id = compute_bundle_id(files, deps, cfg.get("sha256") if cfg else None)
    if expected_id != manifest.get("bundle_id"):
        problems.append("BUNDLE_ID_MISMATCH")
    if check_dependencies:
        current = dependency_versions()
        for dist in sorted(set(deps) | set(current)):
            if deps.get(dist) != current.get(dist):
                problems.append(f"DEPENDENCY_MISMATCH {dist} manifest={deps.get(dist)} current={current.get(dist)}")
    return problems


# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------


def engine_runtime_env(
    bundle_dir: Path,
    base_env: Optional[Mapping[str, str]] = None,
    *,
    program_status_path: Optional[Path] = None,
) -> dict[str, str]:
    """Environment for an engine subprocess launched with ``cwd=bundle_dir``."""
    b = Path(bundle_dir)
    env = dict(base_env if base_env is not None else os.environ)
    env["PYTHONPATH"] = f"{b}{os.pathsep}{b / 'backend'}"
    cfg = b / ENGINE_CONFIG_REL
    env["BEND_INSPECTION_CONFIG"] = str(cfg if cfg.is_file() else b / NO_CONFIG_REL)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    try:
        env["SHERMAN_ENGINE_BUNDLE_ID"] = str(read_manifest(b)["bundle_id"])
    except (OSError, ValueError, KeyError):
        env.pop("SHERMAN_ENGINE_BUNDLE_ID", None)
    if program_status_path is not None:
        env["SHERMAN_QC_F1_PROGRAM_STATUS_PATH"] = str(program_status_path)
    return env


def app_runtime_env(app_dir: Path, base_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Environment for servers started from an app copy with ``cwd=app_dir``.

    ``PYTHONDONTWRITEBYTECODE=1`` keeps the copy byte-identical to its manifest, so
    ``verify`` can treat any ``__pycache__`` as planted code."""
    a = Path(app_dir)
    env = dict(base_env if base_env is not None else os.environ)
    env["PYTHONPATH"] = f"{a}{os.pathsep}{a / 'backend'}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m apps.production_line.engine_bundle")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build an engine bundle or app copy")
    b.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    b.add_argument("--dest", "--out", dest="dest", type=Path, required=True)
    b.add_argument("--profile", choices=_PROFILES, default="engine")
    b.add_argument("--engine-config", type=Path)
    b.add_argument("--extra-module", action="append", default=[])
    b.add_argument("--entry", action="append", default=None, help="replace the profile's entry files")
    b.add_argument("--include-legacy-worker", action="store_true")
    b.add_argument(
        "--frontend", choices=_FRONTEND_MODES, default=None, help="default: build for --profile app, none for engine"
    )
    b.add_argument("--frontend-dist", type=Path)
    b.add_argument("--strict-frontend", action="store_true", help="refuse a copied dist older than its source")
    b.add_argument("--force", action="store_true")
    v = sub.add_parser("verify", help="verify a bundle against its manifest")
    v.add_argument("--dest", "--out", dest="dest", type=Path, required=True)
    v.add_argument("--check-dependencies", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in ("build", "verify", "-h", "--help"):
        args_list = ["build", *args_list]  # `--profile app --out DIR` shorthand
    args = _build_parser().parse_args(args_list)
    if args.command == "verify":
        problems = verify(args.dest, check_dependencies=args.check_dependencies)
        print(json.dumps({"dest": str(args.dest), "ok": not problems, "mismatches": problems}, indent=1))
        return 0 if not problems else 1
    try:
        res = build_bundle(
            args.repo_root,
            args.dest.expanduser().absolute(),
            profile=args.profile,
            engine_config=args.engine_config,
            extra_modules=args.extra_module,
            entries=args.entry,
            include_legacy_worker=args.include_legacy_worker,
            frontend=args.frontend,
            frontend_dist=args.frontend_dist,
            strict_frontend=args.strict_frontend,
            force=args.force,
        )
    except BundleError as exc:
        print(json.dumps({"ok": False, "code": exc.code, "message": exc.message}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "dest": str(res.dest),
                "profile": res.profile,
                "bundle_id": res.bundle_id,
                "file_count": len(res.files),
                "warnings": res.warnings,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
