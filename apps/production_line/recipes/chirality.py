"""Left/right (chirality) map of a released part's E1 compile: display only, no engine change.

Some sheet-metal parts are chiral: the mirror image is a different part. The
research chirality map (``research/chirality/chirality_map.py``,
``chirality_map_v0``) finds, from the CAD alone, whether a part is ACHIRAL,
CHIRAL_GROSS or CHIRAL_BY_FEATURES and which features distinguish it from its
mirror image. This module runs it for an installed compile and serves its
display summary to the onboarding wizard, Vendor Setup and the station. E3 does
not read it; nothing here changes an inspection result.

Storage (data-root relative, keyed by the compile's content hash, never inside
the content-addressed compile directory or ``compiled_part.json``)::

    recipes/chirality/<compiled_part_sha256>/
        chirality_map_v0.json, chirality_map_v0.npz   the research map (written by the worker)
        chirality_summary_v0.json                     display summary; its presence means READY
        chirality_failed.json                         {reason, detail, at} of the last failed run

The map is computed by :mod:`apps.production_line.recipes.chirality_worker` in
a supervised subprocess (thread caps 1, ``timeout_s``, ``rss_limit_bytes``)
from the source root, for the final sequence stage only by default. One run at
a time per process (a single worker thread): heavy geometry never runs in
parallel. A failure is recorded and shown as "not computed"; it never blocks
onboarding or release. A process restart forgets runs in flight; the next
wizard read schedules them again.

Status document (``sherman_chirality_status_v0``)::

    {schema, display_only: true, status: READY | PENDING | FAILED | NOT_COMPUTED, reason,
     detail, recipe_id, recipe_sha256, compiled_part_sha256, summary (chirality_summary_v0 | null),
     stages_computed, computed_at}
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "STATUS_SCHEMA",
    "SUMMARY_FILE",
    "FAILED_FILE",
    "ChiralityError",
    "ChiralityLauncher",
    "ChiralityService",
    "chirality_rel_dir",
    "final_stage_id",
    "status_document",
]

_LOG = logging.getLogger(__name__)

STATUS_SCHEMA = "sherman_chirality_status_v0"
SUMMARY_FILE = "chirality_summary_v0.json"
FAILED_FILE = "chirality_failed.json"
PART_CLASSES = ("ACHIRAL", "CHIRAL_GROSS", "CHIRAL_BY_FEATURES")
#: Worker exit code when the research module cannot be imported (e.g. an app copy without research/).
EXIT_MODULE_UNAVAILABLE = 3


class ChiralityError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail[:1000]
        super().__init__(f"{code}: {detail}")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def chirality_rel_dir(compiled_part_sha256: str) -> str:
    if not _is_sha(compiled_part_sha256):
        raise ValueError("chirality paths are built from a sha256 hex digest")
    return f"recipes/chirality/{compiled_part_sha256}"


def final_stage_id(compiled_part_json: Path) -> str | None:
    """The final sequence stage (SEQUENCE_PREFIX with the largest stage_index), else the last stage."""
    doc = json.loads(Path(compiled_part_json).read_text(encoding="utf-8"))
    stages = [s for s in doc.get("stages") or [] if isinstance(s, dict) and s.get("stage_model_id")]
    prefix = [s for s in stages if s.get("kind") == "SEQUENCE_PREFIX"]
    pick = max(prefix, key=lambda s: int(s.get("stage_index") or 0)) if prefix else (stages[-1] if stages else None)
    return None if pick is None else str(pick["stage_model_id"])


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, doc: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _valid_summary(doc: Any) -> bool:
    return (isinstance(doc, dict) and doc.get("schema") == "chirality_summary_v0"
            and doc.get("part_class") in PART_CLASSES and isinstance(doc.get("features"), list))


def status_document(status: str, *, reason: str | None = None, detail: str | None = None,
                    recipe: Any = None, compiled_part_sha256: str | None = None,
                    summary: Mapping[str, Any] | None = None, stages_computed: list[str] | None = None,
                    computed_at: str | None = None) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "display_only": True,
        "status": status,
        "reason": reason,
        "detail": detail,
        "recipe_id": getattr(recipe, "recipe_id", None),
        "recipe_sha256": getattr(recipe, "recipe_sha256", None),
        "compiled_part_sha256": compiled_part_sha256,
        "summary": dict(summary) if summary is not None else None,
        "stages_computed": list(stages_computed or []),
        "computed_at": computed_at,
    }


@dataclass(frozen=True)
class ChiralityLauncher:
    """Runs :mod:`chirality_worker` supervised from the source root (thread caps 1)."""

    source_root: Path
    python_exe: Path
    module: str = "apps.production_line.recipes.chirality_worker"
    timeout_s: float = 300.0
    rss_limit_bytes: int = 4 * 1024**3
    poll_s: float = 0.5
    kill_grace_s: float = 3.0

    @classmethod
    def from_settings(cls, settings: Any) -> "ChiralityLauncher":
        return cls(source_root=Path(settings.repo_root), python_exe=Path(settings.python_exe),
                   rss_limit_bytes=min(int(settings.rss_limit_bytes), 4 * 1024**3))

    def env(self) -> dict[str, str]:
        from apps.production_line.runner.runner import ENV_PASSTHROUGH, THREAD_ENV_VARS

        env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
        root = str(self.source_root)
        env["PYTHONPATH"] = os.pathsep.join([root, str(Path(root) / "backend")])
        for key in THREAD_ENV_VARS:
            env[key] = "1"
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
                   PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        return env

    def run(self, compiled_dir: Path, out_dir: Path, stages: list[str] | None) -> None:
        """Write the map and its summary into ``out_dir``; raises :class:`ChiralityError`."""
        from apps.production_line.runner.supervisor import run_supervised

        tail: list[str] = []

        def _line(text: str) -> None:
            tail.append(text)
            del tail[:-10]

        cmd = [str(self.python_exe), "-m", self.module, str(compiled_dir), "--out", str(out_dir)]
        if stages:
            cmd += ["--stages", ",".join(stages)]
        res = run_supervised(cmd, env=self.env(), cwd=self.source_root, timeout_s=self.timeout_s,
                             rss_limit_bytes=self.rss_limit_bytes, poll_s=self.poll_s, on_line=_line,
                             log_path=out_dir / "chirality_step.log", kill_grace_s=self.kill_grace_s,
                             identity_tag=f"chirality-{out_dir.name[:16]}")
        if res.timed_out:
            raise ChiralityError("CHIRALITY_TIMEOUT", f"the chirality map exceeded {self.timeout_s:g} s")
        if res.rss_exceeded:
            raise ChiralityError("CHIRALITY_FAILED", "the chirality map exceeded its memory limit")
        if res.exit_code == EXIT_MODULE_UNAVAILABLE:
            raise ChiralityError("CHIRALITY_UNAVAILABLE", " | ".join(tail[-3:]))
        if res.exit_code != 0:
            raise ChiralityError("CHIRALITY_FAILED", f"exit {res.exit_code}: {' | '.join(tail[-5:])}")


@dataclass
class ChiralityService:
    """Schedules and reads chirality maps under ``data_root`` (module docstring)."""

    data_root: Path
    runner: Callable[[Path, Path, list[str] | None], None]
    final_stage_only: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _inflight: set[str] = field(default_factory=set, repr=False)
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)

    def _dir(self, compiled_part_sha256: str) -> Path:
        return Path(self.data_root) / chirality_rel_dir(compiled_part_sha256)

    @staticmethod
    def _compile_of(store: Any, recipe: Any) -> tuple[str | None, Path | None, str | None]:
        """(compiled_part_sha256, compiled dir, reason when unusable) of a stored recipe."""
        compiled = store.get_compiled(recipe.recipe_sha256)
        if compiled is None or getattr(compiled, "status", None) != "COMPILED":
            return None, None, "COMPILE_NOT_INSTALLED"
        sha = getattr(compiled, "compiled_part_sha256", None)
        rel = getattr(compiled, "stored_rel_path", None)
        if not _is_sha(sha) or not rel:
            return None, None, "COMPILE_NOT_INSTALLED"
        return sha, Path(rel).parent, None

    def read(self, store: Any, recipe: Any) -> dict[str, Any]:
        """The status of ``recipe``'s compile without scheduling anything."""
        if recipe is None:
            return status_document("NOT_COMPUTED", reason="NO_RELEASED_RECIPE")
        sha, rel_dir, reason = self._compile_of(store, recipe)
        if sha is None:
            return status_document("NOT_COMPUTED", reason=reason, recipe=recipe)
        out = self._dir(sha)
        summary_path = out / SUMMARY_FILE
        if summary_path.is_file():
            try:
                doc = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = None
            if isinstance(doc, dict) and _valid_summary(doc.get("summary")):
                return status_document("READY", recipe=recipe, compiled_part_sha256=sha, summary=doc["summary"],
                                       stages_computed=doc.get("stages_computed"), computed_at=doc.get("computed_at"))
            return status_document("FAILED", reason="CHIRALITY_OUTPUT_INVALID", recipe=recipe, compiled_part_sha256=sha)
        with self._lock:
            if sha in self._inflight:
                return status_document("PENDING", recipe=recipe, compiled_part_sha256=sha)
        failed = out / FAILED_FILE
        if failed.is_file():
            try:
                doc = json.loads(failed.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = {}
            return status_document("FAILED", reason=str(doc.get("reason") or "CHIRALITY_FAILED"),
                                   detail=doc.get("detail"), recipe=recipe, compiled_part_sha256=sha,
                                   computed_at=doc.get("at"))
        return status_document("NOT_COMPUTED", reason="NOT_SCHEDULED", recipe=recipe, compiled_part_sha256=sha)

    def ensure(self, store: Any, recipe: Any, *, retry_failed: bool = False) -> dict[str, Any]:
        """:meth:`read`, scheduling a run when nothing is stored (or it failed and ``retry_failed``)."""
        doc = self.read(store, recipe)
        schedulable = doc["status"] == "NOT_COMPUTED" and doc["reason"] == "NOT_SCHEDULED"
        if retry_failed and doc["status"] == "FAILED":
            schedulable = True
        if not schedulable:
            return doc
        sha, rel_dir, _ = self._compile_of(store, recipe)
        assert sha is not None and rel_dir is not None
        compiled_dir = Path(self.data_root) / rel_dir
        with self._lock:
            if sha in self._inflight:
                return status_document("PENDING", recipe=recipe, compiled_part_sha256=sha)
            self._inflight.add(sha)
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chirality")
            executor = self._executor
        try:
            executor.submit(self._run, sha, compiled_dir)
        except RuntimeError:
            with self._lock:
                self._inflight.discard(sha)
            return status_document("NOT_COMPUTED", reason="NOT_SCHEDULED", recipe=recipe, compiled_part_sha256=sha)
        return status_document("PENDING", recipe=recipe, compiled_part_sha256=sha)

    def _run(self, sha: str, compiled_dir: Path) -> None:
        out = self._dir(sha)
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / FAILED_FILE).unlink(missing_ok=True)
            cpj = compiled_dir / "compiled_part.json"
            if not cpj.is_file():
                raise ChiralityError("COMPILE_NOT_INSTALLED", "compiled_part.json is missing")
            stages = None
            if self.final_stage_only:
                final = final_stage_id(cpj)
                if final is None:
                    raise ChiralityError("CHIRALITY_FAILED", "the compile has no stage models")
                stages = [final]
            self.runner(compiled_dir, out, stages)
            summary_path = out / SUMMARY_FILE
            doc = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
            if not (isinstance(doc, dict) and _valid_summary(doc.get("summary"))):
                summary_path.unlink(missing_ok=True)
                raise ChiralityError("CHIRALITY_OUTPUT_INVALID", "the worker wrote no valid summary")
        except Exception as exc:  # recorded and shown as "not computed"; never raised to a request
            code = exc.code if isinstance(exc, ChiralityError) else "CHIRALITY_FAILED"
            detail = exc.detail if isinstance(exc, ChiralityError) else f"{type(exc).__name__}: {exc}"[:1000]
            _LOG.warning("chirality map of %s failed: %s %s", sha[:12], code, detail)
            try:
                out.mkdir(parents=True, exist_ok=True)
                _write_json_atomic(out / FAILED_FILE, {"reason": code, "detail": detail, "at": _now()})
            except OSError:
                _LOG.exception("could not record the chirality failure of %s", sha[:12])
        finally:
            with self._lock:
                self._inflight.discard(sha)

    def wait_idle(self, timeout_s: float = 30.0) -> bool:
        """Block until no run is in flight (tests / shutdown)."""
        import time

        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            with self._lock:
                if not self._inflight:
                    return True
            time.sleep(0.02)
        return False
