"""Structured logging and Prometheus-style metrics for the production line.

Logging
    :func:`configure_logging` writes JSON lines to ``<data_root>/logs/line.jsonl``
    (daily rotation, ``backupCount = retention_days``) plus stderr.  Each line is
    ``{ts, level, logger, msg, job_id, sku, scan_sha256, station_id, request_id}``
    (and ``exc`` when an exception is logged).  The correlation fields come from
    ``extra=`` on the log call or from the :func:`log_context` context variable, so a
    runner can wrap a job in ``with log_context(job_id=..., sku=...):``.  Messages are
    redacted: bearer tokens, ``sdt_`` device tokens, ``sst1.`` stream tokens and
    ``token=``/``password=`` query values never reach the log.

Metrics
    :class:`MetricsRegistry` holds thread-safe counters, gauges and histograms;
    :func:`render_prometheus` renders them (plus ad-hoc ``extra_gauges``) in the
    Prometheus text exposition format 0.0.4 for ``GET /api/line/metrics``.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import logging.handlers
import math
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, TextIO, Union

__all__ = [
    "CONTEXT_FIELDS",
    "DEFAULT_LOGGERS",
    "JsonLineFormatter",
    "redact",
    "log_context",
    "current_log_context",
    "configure_logging",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "DEFAULT_BUCKETS",
    "render_prometheus",
    "LINE_METRICS",
]

CONTEXT_FIELDS: tuple[str, ...] = ("job_id", "sku", "scan_sha256", "station_id", "request_id")
DEFAULT_LOGGERS: tuple[str, ...] = ("apps.production_line", "domains.production", "sherman_line")
LOG_FILE_NAME = "line.jsonl"

_CTX: contextvars.ContextVar[Mapping[str, Optional[str]]] = contextvars.ContextVar("sherman_line_log_ctx", default={})

_REDACTIONS: tuple[tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+"), r"\1 [REDACTED]"),
    (re.compile(r"\bsdt_[A-Za-z0-9_-]+"), "sdt_[REDACTED]"),
    (re.compile(r"\bsst1\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), "sst1.[REDACTED]"),
    # key=value forms, including access_token= / refresh_token= / client_secret=
    # (a lookbehind instead of \b, which does not fire between "_" and "t").
    (
        re.compile(
            r"(?i)(?<![A-Za-z0-9])((?:[A-Za-z0-9]+_)*(?:token|password|passwd|pwd|secret|api_key|apikey))=([\"']?)[^&\s\"',;]+"
        ),
        r"\1=\2[REDACTED]",
    ),
    # JSON / dict forms: "password": "x", 'access_token': 'x'.
    (
        re.compile(
            r"""(?i)(["'])((?:[A-Za-z0-9]+_)*(?:token|password|passwd|pwd|secret|api_key|apikey|authorization))\1(\s*:\s*)(["'])(?:\\.|(?!\4).)*\4"""
        ),
        r"\1\2\1\3\4[REDACTED]\4",
    ),
    # A bare JWT (header.payload.signature), e.g. "Authorization: eyJ...".
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"), "[REDACTED_JWT]"),
)


def redact(text: str) -> str:
    """Remove credentials from a log message."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def log_context(**fields: Optional[str]) -> Iterator[Mapping[str, Optional[str]]]:
    """Bind correlation fields for every log record emitted in this context."""
    unknown = set(fields) - set(CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"unknown log context fields {sorted(unknown)}")
    merged = dict(_CTX.get())
    merged.update({k: (None if v is None else str(v)) for k, v in fields.items()})
    token = _CTX.set(merged)
    try:
        yield merged
    finally:
        _CTX.reset(token)


def current_log_context() -> Mapping[str, Optional[str]]:
    return dict(_CTX.get())


class JsonLineFormatter(logging.Formatter):
    """One JSON object per record with the fixed correlation fields."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = _CTX.get()
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for name in CONTEXT_FIELDS:
            value = getattr(record, name, None)
            if value is None:
                value = ctx.get(name)
            payload[name] = None if value is None else str(value)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, sort_keys=False, default=str)


_HANDLER_MARK = "_sherman_line_handler"


class _PrivateTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Daily-rotating handler whose log file is always mode 0600.

    ``FileHandler._open`` creates the file under the process umask (normally 0644),
    both at start-up and for the fresh file after each midnight rollover; this opens
    it with ``O_CREAT`` mode 0600 and ``fchmod``s a pre-existing file, so the
    file-mode promise survives rotation (rotated backups keep the mode on rename).
    """

    def _open(self):  # type: ignore[override]
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except OSError:  # pragma: no cover - best effort on exotic filesystems
            pass
        return os.fdopen(fd, "a", encoding=self.encoding, errors=self.errors)


def configure_logging(
    paths: Any,
    level: Union[int, str] = "INFO",
    retention_days: int = 30,
    *,
    logger_names: Sequence[str] = DEFAULT_LOGGERS,
    stream: Optional[TextIO] = None,
) -> Path:
    """Attach the JSON-lines file handler and a stderr handler to ``logger_names``.

    ``paths`` is a :class:`~apps.production_line.settings.LinePaths` (its ``logs``
    directory is used) or a directory path.  Idempotent: handlers installed by an
    earlier call are replaced.  Returns the log file path.
    """
    logs_dir = Path(getattr(paths, "logs", paths))
    logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_file = logs_dir / LOG_FILE_NAME
    if not isinstance(retention_days, int) or retention_days < 1:
        raise ValueError("retention_days must be a positive integer")
    lvl = logging.getLevelName(level.upper()) if isinstance(level, str) else int(level)
    if not isinstance(lvl, int):
        raise ValueError(f"unknown log level {level!r}")

    formatter = JsonLineFormatter()
    file_handler = _PrivateTimedRotatingFileHandler(
        log_file, when="midnight", backupCount=retention_days, encoding="utf-8", utc=True, delay=False
    )
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_MARK, True)
    err_handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    err_handler.setFormatter(formatter)
    setattr(err_handler, _HANDLER_MARK, True)

    for name in logger_names:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if getattr(h, _HANDLER_MARK, False):
                lg.removeHandler(h)
                h.close()
        lg.addHandler(file_handler)
        lg.addHandler(err_handler)
        lg.setLevel(lvl)
        lg.propagate = False
    return log_file


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
DEFAULT_BUCKETS: tuple[float, ...] = (0.5, 1, 2.5, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900)

LabelKey = tuple[tuple[str, str], ...]


def _label_key(label_names: Sequence[str], labels: Mapping[str, Any]) -> LabelKey:
    if set(labels) != set(label_names):
        raise ValueError(f"expected labels {sorted(label_names)}, got {sorted(labels)}")
    return tuple((n, str(labels[n])) for n in label_names)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _fmt_labels(key: LabelKey, extra: Sequence[tuple[str, str]] = ()) -> str:
    items = list(key) + list(extra)
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{_escape_label(v)}"' for k, v in items) + "}"


def _fmt_value(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> None:
        if not _METRIC_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid metric name {name!r}")
        for ln in label_names:
            if not _LABEL_NAME_RE.fullmatch(ln) or ln.startswith("__") or ln == "le":
                raise ValueError(f"invalid label name {ln!r}")
        self.name = name
        self.help = help_text
        self.label_names = tuple(label_names)
        self._lock = threading.Lock()

    def header(self) -> list[str]:
        return [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} {self.kind}"]

    def samples(self) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, label_names)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        if amount < 0 or math.isnan(amount):
            raise ValueError("counters only go up")
        key = _label_key(self.label_names, labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: Any) -> float:
        with self._lock:
            return self._values.get(_label_key(self.label_names, labels), 0.0)

    def samples(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        if not items and not self.label_names:
            items = [((), 0.0)]
        return [f"{self.name}{_fmt_labels(k)} {_fmt_value(v)}" for k, v in items]


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, label_names)
        self._values: dict[LabelKey, float] = {}

    def set(self, value: float, **labels: Any) -> None:
        key = _label_key(self.label_names, labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        key = _label_key(self.label_names, labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: Any) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: Any) -> float:
        with self._lock:
            return self._values.get(_label_key(self.label_names, labels), 0.0)

    def samples(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        return [f"{self.name}{_fmt_labels(k)} {_fmt_value(v)}" for k, v in items]


class Histogram(_Metric):
    kind = "histogram"

    def __init__(
        self, name: str, help_text: str, label_names: Sequence[str] = (), buckets: Sequence[float] = DEFAULT_BUCKETS
    ) -> None:
        super().__init__(name, help_text, label_names)
        bs = sorted(float(b) for b in buckets if not math.isinf(b))
        if not bs or len(set(bs)) != len(bs):
            raise ValueError("histogram buckets must be non-empty and distinct")
        self.buckets: tuple[float, ...] = tuple(bs)
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = {}

    def observe(self, value: float, **labels: Any) -> None:
        if math.isnan(value):
            raise ValueError("cannot observe NaN")
        key = _label_key(self.label_names, labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            for i, b in enumerate(self.buckets):
                if value <= b:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
            self._sums[key] = self._sums.get(key, 0.0) + float(value)

    def count(self, **labels: Any) -> int:
        with self._lock:
            return sum(self._counts.get(_label_key(self.label_names, labels), []))

    def samples(self) -> list[str]:
        out: list[str] = []
        with self._lock:
            items = sorted(self._counts.items())
            sums = dict(self._sums)
        for key, counts in items:
            running = 0
            for b, c in zip(self.buckets, counts):
                running += c
                out.append(f"{self.name}_bucket{_fmt_labels(key, [('le', _fmt_value(b))])} {running}")
            running += counts[-1]
            out.append(f"{self.name}_bucket{_fmt_labels(key, [('le', '+Inf')])} {running}")
            out.append(f"{self.name}_sum{_fmt_labels(key)} {_fmt_value(sums.get(key, 0.0))}")
            out.append(f"{self.name}_count{_fmt_labels(key)} {running}")
        return out


class MetricsRegistry:
    """Named metrics; ``counter``/``gauge``/``histogram`` return the existing metric
    when called again with the same name and type."""

    def __init__(self) -> None:
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if type(existing) is not cls:
                    raise ValueError(f"metric {name} already registered as {existing.kind}")
                return existing
            metric = cls(name, *args, **kwargs)
            self._metrics[name] = metric
            return metric

    def counter(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> Counter:
        return self._get_or_create(Counter, name, help_text, label_names)

    def gauge(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> Gauge:
        return self._get_or_create(Gauge, name, help_text, label_names)

    def histogram(
        self, name: str, help_text: str, label_names: Sequence[str] = (), buckets: Sequence[float] = DEFAULT_BUCKETS
    ) -> Histogram:
        return self._get_or_create(Histogram, name, help_text, label_names, buckets)

    def metrics(self) -> list[_Metric]:
        with self._lock:
            return [self._metrics[k] for k in sorted(self._metrics)]


ExtraGauges = Union[Mapping[str, float], Iterable[tuple[str, float, str, Mapping[str, str]]]]


def render_prometheus(registry: MetricsRegistry, extra_gauges: Optional[ExtraGauges] = None) -> str:
    """Text exposition format 0.0.4.  ``extra_gauges`` is ``{name: value}`` or an
    iterable of ``(name, value, help, labels)`` computed at scrape time (queue depth,
    disk free, heartbeat ages ...)."""
    lines: list[str] = []
    seen: set[str] = set()
    for metric in registry.metrics():
        lines.extend(metric.header())
        lines.extend(metric.samples())
        seen.add(metric.name)
    if extra_gauges:
        grouped: dict[str, tuple[str, list[tuple[LabelKey, float]]]] = {}
        items: Iterable[Any] = extra_gauges.items() if isinstance(extra_gauges, Mapping) else extra_gauges
        for item in items:
            if isinstance(extra_gauges, Mapping):
                name, value = item
                help_text, labels = name, {}
            else:
                name, value, help_text, labels = item
            if not _METRIC_NAME_RE.fullmatch(name) or name in seen:
                raise ValueError(f"invalid or duplicate extra gauge {name!r}")
            for ln in labels:
                if not _LABEL_NAME_RE.fullmatch(ln):
                    raise ValueError(f"invalid label name {ln!r}")
            key = tuple(sorted((str(k), str(v)) for k, v in labels.items()))
            grouped.setdefault(name, (help_text, []))[1].append((key, float(value)))
        for name in sorted(grouped):
            help_text, samples = grouped[name]
            lines.append(f"# HELP {name} {_escape_help(help_text)}")
            lines.append(f"# TYPE {name} gauge")
            for key, value in sorted(samples):
                lines.append(f"{name}{_fmt_labels(key)} {_fmt_value(value)}")
    return "\n".join(lines) + "\n"


#: Standard line metric names (C02/C07 register them on their registry with these
#: names so dashboards stay stable).
LINE_METRICS: Mapping[str, tuple[str, str, tuple[str, ...]]] = {
    "sherman_line_scans_received_total": ("counter", "Scan files registered by the hotfolder", ("station_id",)),
    "sherman_line_scans_quarantined_total": ("counter", "Scan files quarantined", ("station_id", "reason")),
    "sherman_line_jobs_finished_total": ("counter", "Jobs that reached a terminal state", ("kind", "status")),
    "sherman_line_job_failures_total": ("counter", "Job failures by failure code", ("code",)),
    "sherman_line_job_duration_seconds": ("histogram", "Wall time of finished jobs", ("kind",)),
    "sherman_line_http_requests_total": ("counter", "HTTP requests served", ("method", "status")),
}
