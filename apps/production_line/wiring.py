"""Process wiring of the Sherman QC production line (C11).

One place builds every long-lived object of a line process from
:class:`~apps.production_line.settings.LineSettings`, so the API server
(:mod:`apps.production_line.server`) and the line service
(:mod:`apps.production_line.service_main`) share exactly one store, recipe
resolver and device-token store per process instead of each router caching its
own copy.

:func:`build_runtime` returns a :class:`LineRuntime`:

- ``line_store`` (C02 :class:`~apps.production_line.store.line_store.LineStore`
  on ``db/line.sqlite3``), ``recipe_store`` (C05
  :class:`~apps.production_line.recipes.store.RecipeStore` on
  ``db/recipes.sqlite3``) and ``device_store`` (C04 device tokens on
  ``db/security.sqlite3``);
- ``provider``: C05 :class:`~apps.production_line.recipes.resolver.LineRecipeResolver`
  over that recipe store. It is the runner's ``RecipeProvider`` *and*
  ``CompiledPartSink``, and C07's ``ActiveRecipeCatalog`` for quarantine
  assignment;
- ``resolver``: the hotfolder's ``ScanRecipeResolver``, the same resolver class
  given the part catalog so a catalogued but unreleased SKU is
  PART_NOT_READY instead of UNMATCHED. It is registered as a release listener;
  release listeners are weak references, so the runtime keeps it alive. The
  line service and the API server are separate processes, so a release in the
  API process never reaches the service's listener: the hotfolder therefore
  gets it wrapped in :class:`FreshOnMissResolver`, which re-resolves any
  non-MATCHED outcome once against a fresh snapshot, so a scan dropped right
  after a release is never quarantined from a stale cache;
- ``catalog``: the C05 catalog port;
- ``runner`` (one :class:`~apps.production_line.runner.runner.JobRunner`, stable
  instance id ``runner@<host>`` so a restarted service reclaims its own dead
  leases at once) and ``pollers`` (one
  :class:`~apps.production_line.ingest.hotfolder.HotfolderPoller` per configured
  station), only with ``workers=True`` (the line service);
- ``metrics``: the process metrics registry (C04
  :class:`~apps.production_line.telemetry.MetricsRegistry`).

:func:`runner_config_for` maps settings to the C02
:class:`~apps.production_line.runner.runner.RunnerConfig` (plan §10 amendment 1:
the engine step is ``python -m domains.production.engine.inspect`` with a 180 s
hard stop and 32 GiB (ENGINE_IO-A5), run from the pinned engine bundle; tightening is allowed,
loosening is not).

:func:`install_api_overrides` points the C04, C05 and C07 FastAPI dependency
functions at the runtime's instances (``app.dependency_overrides``), so every
route sees one settings object and one set of stores.

Nothing here imports the legacy server at import time. The default catalog for
the API process is C05's :class:`~apps.production_line.recipes.catalog_port.LegacyCatalogPort`
(resolved lazily on first use); the line service passes a catalog built on
``part_catalog.get_catalog()`` (the legacy catalog database, without importing
the legacy server).
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.production_line.settings import LineSettings

__all__ = [
    "LineRuntime",
    "build_runtime",
    "runner_config_for",
    "hotfolder_config_for",
    "install_api_overrides",
    "runner_instance_id",
    "bundle_id_of",
    "service_catalog_port",
    "FreshOnMissResolver",
]

_LOG = logging.getLogger(__name__)


def runner_instance_id(host: str | None = None) -> str:
    """Stable runner identity ``runner@<host>`` (C02: lets a restarted service
    reclaim the leases of its own dead predecessor immediately)."""
    name = (host or socket.gethostname() or "localhost").split(".")[0] or "localhost"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)[:48]
    return f"runner@{safe}"


def bundle_id_of(bundle_dir: Path) -> str | None:
    """The ``bundle_id`` of the engine bundle at ``bundle_dir``, or None when it has no
    readable manifest."""
    from apps.production_line.engine_bundle import read_manifest

    try:
        value = read_manifest(Path(bundle_dir)).get("bundle_id")
    except (OSError, ValueError, AttributeError):
        return None
    return str(value) if isinstance(value, str) and value else None


#: The runner's zero-shot step (enabled when the engine bundle contains it).
ZEROSHOT_MODULE = "domains.production.zeroshot.lane_cli"
ZEROSHOT_ENTRY_REL = "domains/production/zeroshot/lane_cli.py"


def runner_config_for(settings: LineSettings, **overrides: Any) -> Any:
    """C02 ``RunnerConfig`` for ``settings`` (module docstring).

    ``engine_root`` is the pinned engine bundle; engine limits come from the
    settings but never exceed the hard stop (180 s, 32 GiB by ENGINE_IO-A5); render
    and compile steps get the postprocess timeout and the same RSS cap. The
    step environment adds ``PYTHONNOUSERSITE``, the bundle id and the program
    status path (``SHERMAN_QC_F1_PROGRAM_STATUS_PATH``, fail-closed when absent).
    The zero-shot overlay step is enabled when the bundle carries
    ``domains/production/zeroshot/lane_cli.py`` (with the engine's RSS cap).
    ``overrides`` replace any field (tests, E2E).
    """
    from apps.production_line.runner.runner import RunnerConfig
    from apps.production_line.store.vocabulary import ENGINE_HARD_TIMEOUT_S, ENGINE_RSS_LIMIT_BYTES

    bundle = Path(settings.engine_bundle_dir)
    bundle_id = bundle_id_of(bundle)
    extra_env: dict[str, str] = {
        "PYTHONNOUSERSITE": "1",
        "SHERMAN_QC_F1_PROGRAM_STATUS_PATH": str(settings.program_status_path),
    }
    if bundle_id:
        extra_env["SHERMAN_ENGINE_BUNDLE_ID"] = bundle_id
    rss = min(int(settings.rss_limit_bytes), ENGINE_RSS_LIMIT_BYTES)
    values: dict[str, Any] = dict(
        engine_root=bundle,
        python_exe=Path(settings.python_exe),
        data_root=Path(settings.data_root),
        engine_timeout_s=float(min(int(settings.job_timeout_s), ENGINE_HARD_TIMEOUT_S)),
        engine_rss_limit_bytes=rss,
        engine_time_budget_s=float(max(5, min(int(settings.job_timeout_s), ENGINE_HARD_TIMEOUT_S) - 10)),
        render_timeout_s=float(settings.postprocess_timeout_s),
        render_rss_limit_bytes=rss,
        compile_rss_limit_bytes=rss,
        engine_bundle_id=bundle_id,
        extra_env=extra_env,
        # Zero-shot overlay step (runner 3b): on when the pinned bundle carries the lane; same RSS cap as E3.
        zeroshot_module=ZEROSHOT_MODULE if (bundle / ZEROSHOT_ENTRY_REL).is_file() else None,
        zeroshot_rss_limit_bytes=rss,
    )
    if "extra_env" in overrides:
        merged = dict(extra_env)
        merged.update({str(k): str(v) for k, v in dict(overrides.pop("extra_env")).items()})
        values["extra_env"] = merged
    values.update(overrides)
    return RunnerConfig(**values)


def hotfolder_config_for(settings: LineSettings, station: Any, **overrides: Any) -> Any:
    """C02 ``HotfolderConfig`` for one configured station."""
    from apps.production_line.ingest.hotfolder import HotfolderConfig

    values: dict[str, Any] = dict(
        station_id=station.station_id,
        inbox=Path(station.inbox),
        data_root=Path(settings.data_root),
        sidecar_mode=station.sidecar_mode,
        source_policy=station.source_policy,
        recursive=bool(station.recursive),
        repo_root=Path(settings.repo_root),
    )
    values.update(overrides)
    return HotfolderConfig(**values)


def service_catalog_port(settings: LineSettings) -> Any:
    """Catalog port for the line service: the legacy catalog database through
    ``part_catalog.get_catalog()`` (imports ``backend/part_catalog.py`` only, never
    the legacy server). Relative stored paths resolve against the app root."""
    from apps.production_line.recipes.catalog_port import CatalogManagerPort

    import part_catalog  # type: ignore[import-not-found]  # backend/ is on sys.path

    return CatalogManagerPort(part_catalog.get_catalog(), catalog_root=Path(settings.repo_root))


class FreshOnMissResolver:
    """``ScanRecipeResolver`` that never quarantines from a stale snapshot.

    MATCHED outcomes come from the inner resolver's cached snapshot; any other
    outcome is resolved once more after ``invalidate()`` (a fresh read of the
    recipe store and the catalog). Every other attribute is delegated.

    The inner resolver is shared by every station poller thread, so ``resolve``
    and ``invalidate`` run under one lock. A forced refresh re-reads the whole
    legacy catalog, so it happens at most once per ``min_refresh_s`` (default
    1 s): a burst of unmatched drops (a quarantine storm) costs one catalog read
    per second, not one per file. Inside that window the snapshot used for a
    miss is at most ``min_refresh_s`` old, so only a release published in the
    last second before the drop can still be missed; such a scan is quarantined
    with a resolvable reason and is assigned or re-resolved later, never lost.
    """

    def __init__(self, inner: Any, *, min_refresh_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if min_refresh_s < 0:
            raise ValueError("min_refresh_s must be >= 0")
        self._inner = inner
        self._min_refresh_s = float(min_refresh_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._last_refresh: float | None = None
        #: Number of forced snapshot refreshes (diagnostics and tests).
        self.refresh_count = 0

    def resolve(self, original_filename: str) -> Any:
        with self._lock:
            outcome = self._inner.resolve(original_filename)
            if getattr(outcome, "status", None) == "MATCHED":
                return outcome
            now = self._clock()
            if self._last_refresh is not None and now - self._last_refresh < self._min_refresh_s:
                return outcome  # the snapshot is already fresh (refreshed < min_refresh_s ago)
            self._inner.invalidate()
            self._last_refresh = now
            self.refresh_count += 1
            return self._inner.resolve(original_filename)

    def invalidate(self, *args: Any) -> None:
        with self._lock:
            self._inner.invalidate(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class LineRuntime:
    """The shared long-lived objects of one line process (module docstring)."""

    settings: LineSettings
    line_store: Any
    recipe_store: Any
    device_store: Any
    provider: Any
    resolver: Any
    catalog: Any
    metrics: Any
    runner: Any | None = None
    pollers: tuple[Any, ...] = ()
    #: Strong references to callbacks registered as (weakly held) release listeners.
    listeners: list[Callable[..., Any]] = field(default_factory=list)

    def close(self) -> None:
        """Unregister the release listeners this runtime installed."""
        from apps.production_line.recipes.release_service import remove_release_listener

        for cb in self.listeners:
            try:
                remove_release_listener(cb)
            except Exception:  # pragma: no cover - best effort
                _LOG.debug("release listener already removed")
        self.listeners.clear()


def build_runtime(
    settings: LineSettings,
    *,
    workers: bool = False,
    catalog: Any | None = None,
    runner_overrides: Mapping[str, Any] | None = None,
    hotfolder_overrides: Mapping[str, Any] | None = None,
    instance_id: str | None = None,
    metrics: Any | None = None,
    recover_on_start: bool = True,
) -> LineRuntime:
    """Build the runtime for ``settings`` (module docstring).

    ``workers=True`` also builds the job runner and one hotfolder poller per
    station (the line service). ``catalog`` defaults to C05's lazy
    ``LegacyCatalogPort`` rooted at the app root. Nothing is started here.
    """
    from apps.production_line.recipes.release_service import add_release_listener
    from apps.production_line.recipes.resolver import LineRecipeResolver, catalog_sku_source
    from apps.production_line.recipes.store import RecipeStore
    from apps.production_line.security.dependencies import device_token_store_for
    from apps.production_line.store.line_store import LineStore
    from apps.production_line.telemetry import MetricsRegistry

    line_store = LineStore(settings.paths.db_line)
    recipe_store = RecipeStore(settings.paths.db_recipes)
    device_store = device_token_store_for(settings)
    if catalog is None:
        from apps.production_line.recipes.catalog_port import LegacyCatalogPort

        catalog = LegacyCatalogPort(catalog_root=Path(settings.repo_root))
    provider = LineRecipeResolver(recipe_store, settings.filename_regex, data_root=Path(settings.data_root))
    resolver = LineRecipeResolver(
        recipe_store,
        settings.filename_regex,
        data_root=Path(settings.data_root),
        extra_skus=catalog_sku_source(catalog),
    )
    runtime = LineRuntime(
        settings=settings,
        line_store=line_store,
        recipe_store=recipe_store,
        device_store=device_store,
        provider=provider,
        resolver=resolver,
        catalog=catalog,
        metrics=metrics if metrics is not None else MetricsRegistry(),
    )
    for cb in (provider.invalidate, resolver.invalidate):
        add_release_listener(cb)
        runtime.listeners.append(cb)
    if workers:
        from apps.production_line.ingest.hotfolder import HotfolderPoller
        from apps.production_line.runner.runner import JobRunner

        runtime.runner = JobRunner(
            runner_config_for(settings, **dict(runner_overrides or {})),
            line_store,
            provider,
            instance_id=instance_id or runner_instance_id(),
            compiled_sink=provider,
            recover_on_start=recover_on_start,
        )
        runtime.pollers = tuple(
            HotfolderPoller(
                hotfolder_config_for(settings, st, **dict(hotfolder_overrides or {})),
                line_store,
                FreshOnMissResolver(resolver),
                instance_id=f"hotfolder@{st.station_id}",
            )
            for st in settings.stations
        )
    return runtime


def install_api_overrides(app: Any, runtime: LineRuntime) -> None:
    """Point the C04/C05/C07 dependency functions of ``app`` at ``runtime``.

    Overridden: ``get_line_settings`` (C04, so ``require_line_settings`` and every
    capability check see these settings), ``get_device_token_store`` (C04), the
    C05 ``get_recipe_store`` / ``get_line_store`` / ``get_catalog_port``, and the
    C07 ``get_line_store`` / ``get_recipe_provider`` / ``get_metrics_registry``.
    """
    import apps.api.routes.production_recipes as recipe_routes
    from apps.api.services import production_line_service as line_svc
    from apps.production_line.security import dependencies as deps

    settings = runtime.settings
    overrides: dict[Callable[..., Any], Callable[..., Any]] = {
        deps.get_line_settings: lambda: settings,
        deps.get_device_token_store: lambda: runtime.device_store,
        recipe_routes.get_recipe_store: lambda: runtime.recipe_store,
        recipe_routes.get_line_store: lambda: runtime.line_store,
        recipe_routes.get_catalog_port: lambda: runtime.catalog,
        line_svc.get_line_store: lambda: runtime.line_store,
        line_svc.get_recipe_provider: lambda: runtime.provider,
        line_svc.get_metrics_registry: lambda: runtime.metrics,
    }
    app.dependency_overrides.update(overrides)
