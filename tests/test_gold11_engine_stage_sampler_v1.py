"""26 prospective sampler outcomes; source-only authored, no launch authority.

All inputs are invented code-location objects or bounded literal metadata. One
explicitly short self-owned thread/pipe control is prospective: 0.01 s interval,
1.0 s event wait and the unchanged 0.25 s finish join. It is not the production
30 s cadence. The remaining controls use deterministic event/thread/write fakes.
No backend, native geometry, physical input, arbitrary file or frame locals.
An exact independent protocol/PRE/archive must precede any import/collection/run.
"""
import ast
import builtins
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "scripts/gold11_engine_stage_sampler_v1.py"
TEST_SOURCE = "tests/test_gold11_engine_stage_sampler_v1.py"
TEST_INTERVAL_SECONDS = .01
TEST_EVENT_WAIT_SECONDS = 1.


@pytest.fixture(scope="module")
def sampler_module():
    prefix = "_invented_gold11_stage_sampler_v1"
    assert not any(n == prefix or n.startswith(prefix+".") for n in sys.modules)
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROOT/"scripts")]
    sys.modules[prefix] = package
    try:
        yield importlib.import_module(prefix+".gold11_engine_stage_sampler_v1")
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]


class Frame:
    """Only permitted location/link accesses exist; locals access is fatal."""
    def __init__(self, filename, function, line, previous=None, observed=None):
        self.code = types.SimpleNamespace(co_filename=filename, co_name=function)
        self.line, self.previous, self.observed = line, previous, observed

    @property
    def f_code(self):
        if self.observed is not None:
            self.observed.append(("code", self.line))
        return self.code

    @property
    def f_lineno(self):
        return self.line

    @property
    def f_back(self):
        if self.observed is not None:
            self.observed.append(("back", self.line))
        return self.previous

    @property
    def f_locals(self):
        raise AssertionError("FRAME_LOCALS_MUST_NEVER_BE_READ")


class Event:
    def __init__(self, stop_immediately=False):
        self.stopped, self.waits = stop_immediately, []

    def wait(self, seconds):
        self.waits.append(seconds)
        return self.stopped

    def set(self):
        self.stopped = True


class Thread:
    def __init__(self, *, target, name, daemon, fail_start=False, alive=False):
        self.target, self.name, self.daemon = target, name, daemon
        self.fail_start, self.alive = fail_start, alive
        self.ident, self.joins = None, []

    def start(self):
        if self.fail_start:
            raise RuntimeError("INVENTED_THREAD_START_FAILURE")
        self.ident = 12345

    def join(self, seconds):
        self.joins.append(seconds)

    def is_alive(self):
        return self.alive


def _sampler(m):
    return m.MainThreadStageSamplerV1(source_root="/invented", source_paths=("engine.py",), output_fd=91)


def _fake_run(m, monkeypatch, sampler, frame=None):
    sampler.stop = Event()
    monkeypatch.setattr(m.sys, "_current_frames", lambda: {sampler.main_thread: frame})
    monkeypatch.setattr(m.resource, "getrusage", lambda who: types.SimpleNamespace(
        ru_utime=1.25, ru_stime=.75, ru_maxrss=123456))
    monkeypatch.setattr(m.time, "monotonic", lambda: sampler.started_at+3.)


def test_static_source_and_fixed_observational_policy(sampler_module):
    m = sampler_module
    tree = ast.parse((ROOT/SOURCE).read_text(encoding="utf-8"))
    imports = {n.names[0].name for n in tree.body if isinstance(n, ast.Import)}
    imports |= {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert imports == {"json", "os", "pathlib", "resource", "sys", "threading", "time"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("f_locals", "f_globals") for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("open", "eval", "exec", "compile") for n in ast.walk(tree))
    assert (m.INTERVAL_SECONDS, m.MAX_SAMPLES, m.MAX_FRAMES, m.MAX_FRAME_WALK,
        m.MAX_SAMPLE_BYTES, m.MAX_WRITE_CALLS, m.FINISH_JOIN_SECONDS) == (30., 128, 12, 64, 4096, 8, .25)
    assert m.POLICY["maximum_sample_stream_bytes"] == 524288
    assert m.POLICY["interval_seconds"] == 30.
    assert m.POLICY["failure_policy"] == "KEEP_INFERENCE_UNCHANGED_REPORT_DIAGNOSTIC_FAILURE_SEPARATELY"


def test_source_scope_is_exact_lexical_allowlist_without_io(sampler_module, monkeypatch):
    m = sampler_module
    def forbidden(*args, **kwargs):
        raise AssertionError("NO_SOURCE_OR_INPUT_FILESYSTEM_ACCESS")
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(m.os, "open", forbidden)
        patch.setattr(m.os, "stat", forbidden)
        assert m._sources("/invented", ("backend/a.py", "scripts/b.py")) == {
            "/invented/backend/a.py": "backend/a.py", "/invented/scripts/b.py": "scripts/b.py"}
        assert m._sources("/", ("engine.py",)) == {"/engine.py": "engine.py"}
        names = tuple("f%d.py" % index for index in range(128))
        assert len(m._sources("/invented", names)) == 128


@pytest.mark.parametrize("kind", ["root", "container", "relative", "duplicates"])
def test_invalid_frozen_source_scope_is_rejected(sampler_module, kind):
    m = sampler_module
    if kind == "root":
        cases = [(value, ("x.py",)) for value in
            (None, 5, "relative", "/invented/", "/a/../b", "/a//b", "/a\x00b", "/caf\u00e9", "/"+"x"*2048)]
    elif kind == "container":
        cases = [("/invented", value) for value in ([], ["x.py"], (), tuple("x%d.py" % i for i in range(129)))]
    elif kind == "relative":
        cases = [("/invented", (value,)) for value in
            (None, 1, "", "a.txt", "/a.py", "../a.py", "a/../b.py", "a//b.py", "./a.py",
             "a\x00.py", "caf\u00e9.py", "x"*190+".py")]
    else:
        cases = [("/invented", ("x.py", "x.py"))]
    for root, paths in cases:
        with pytest.raises(ValueError):
            m._sources(root, paths)


def test_stack_names_only_exact_sources_and_never_locals(sampler_module):
    m = sampler_module
    tail = Frame("/invented/engine.py.extra", "unlisted_tail", 90)
    listed = Frame("/invented/engine.py", "n"*90, 42, tail)
    head = Frame("/elsewhere/engine.py", "unlisted_head", 3, listed)
    actual = m._stack(head, {"/invented/engine.py": "engine.py"})
    assert actual == dict(frames=[dict(source="engine.py", function="n"*64, line=42)],
        walked_frames=3, omitted_named_frames=0, stack_walk_truncated=False,
        complete_python_stack_claim=False, frame_locals_inspected=False)
    assert m._stack(None, {})["frames"] == []


def test_stack_has_exact_twelve_frame_and_sixtyfour_walk_caps(sampler_module):
    m = sampler_module
    observed, frame = [], None
    for line in range(70, 0, -1):
        frame = Frame("/invented/engine.py", "stage", line, frame, observed)
    result = m._stack(frame, {"/invented/engine.py": "engine.py"})
    assert [f["line"] for f in result["frames"]] == list(range(1, 13))
    assert result["walked_frames"] == 64 and result["omitted_named_frames"] == 52
    assert result["stack_walk_truncated"] is True
    assert observed == [(kind, line) for line in range(1, 65) for kind in ("code", "back")]
    assert not result["complete_python_stack_claim"] and not result["frame_locals_inspected"]


def test_full_line_byte_boundary_and_whole_overflow_refusal(sampler_module):
    m = sampler_module
    record = dict(schema=m.SCHEMA, sample_index=7, frames=[dict(source="engine.py", function="stage", line=1)],
        status="OBSERVED_STACK", geometry_or_stage_completion_evidence=False, padding="")
    encode = lambda value: (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("ascii")
    record["padding"] = "x"*(4096-len(encode(record)))
    assert len(m._wire(record)) == 4096 and m._wire(record) == encode(record)
    record["padding"] += "x"
    assert len(encode(record)) == 4097
    refused = json.loads(m._wire(record))
    assert refused == dict(schema=m.SCHEMA, sample_index=7, status="SAMPLE_TOO_LARGE_STACK_OMITTED_WHOLE",
        frames=[], geometry_or_stage_completion_evidence=False)
    assert record["frames"] and len(m._wire(record)) <= 4096
    assert m._wire(record).endswith(b"\n") and m._wire(record).count(b"\n") == 1
    record["padding"] = ""
    record["frames"] = [dict(source="x"*189+".py", function="\U0001f4a1"*64, line=i) for i in range(12)]
    assert json.loads(m._wire(record))["frames"] == []  # ASCII JSON expansion is part of the cap.


def test_partial_writes_complete_exactly_with_suffixes(sampler_module, monkeypatch):
    m = sampler_module
    seen = []
    def write(fd, view):
        seen.append((fd, bytes(view)))
        return min(2, len(view))
    monkeypatch.setattr(m.os, "write", write)
    assert m._write(91, b"abcdefgh") == 8
    assert seen == [(91, b"abcdefgh"), (91, b"cdefgh"), (91, b"efgh"), (91, b"gh")]


@pytest.mark.parametrize("count", [0, -1, True, 5])
def test_zero_negative_bool_or_excess_write_fails(sampler_module, monkeypatch, count):
    m = sampler_module
    calls = []
    monkeypatch.setattr(m.os, "write", lambda fd, view: calls.append(bytes(view)) or count)
    with pytest.raises(OSError, match="SHORT_OR_INVALID_WRITE"):
        m._write(91, b"1234")
    assert calls == [b"1234"]


def test_eight_write_call_cap_and_os_error_never_retry(sampler_module, monkeypatch):
    m = sampler_module
    calls = []
    monkeypatch.setattr(m.os, "write", lambda fd, view: calls.append(bytes(view)) or 1)
    with pytest.raises(OSError, match="WRITE_CALL_CAP"):
        m._write(91, b"123456789")
    assert len(calls) == 8 and calls[-1] == b"89"
    calls.clear()
    assert m._write(91, b"12345678") == 8 and len(calls) == 8
    def failed(fd, view):
        calls.append(bytes(view))
        raise OSError("INVENTED_IO_ERROR")
    monkeypatch.setattr(m.os, "write", failed)
    calls.clear()
    with pytest.raises(OSError, match="INVENTED_IO_ERROR"):
        m._write(91, b"data")
    assert calls == [b"data"]


@pytest.mark.parametrize("failure_stage", ["frame", "wire", "write"])
def test_attempt_precharge_and_incomplete_byte_accounting(sampler_module, monkeypatch, failure_stage):
    m = sampler_module
    sampler = _sampler(m)
    _fake_run(m, monkeypatch, sampler)
    def failure(*args):
        assert sampler.samples_attempted == 1
        raise RuntimeError("INVENTED_DIAGNOSTIC_FAILURE")
    chunks = []
    if failure_stage == "frame":
        monkeypatch.setattr(m.sys, "_current_frames", failure)
    elif failure_stage == "wire":
        monkeypatch.setattr(m, "_wire", failure)
    else:
        monkeypatch.setattr(m, "_wire", lambda record: b"1234567\n")
        def write(fd, view):
            assert sampler.samples_attempted == (1 if not chunks else 2)
            if not chunks:
                chunks.append(bytes(view))
                return len(view)
            if len(chunks) == 1:
                chunks.append(bytes(view[:2]))
                return 2
            raise OSError("INVENTED_ERROR_AFTER_TWO_PARTIAL_BYTES")
        monkeypatch.setattr(m.os, "write", write)
    sampler._run()
    receipt = sampler.finish()
    written = 1 if failure_stage == "write" else 0
    assert receipt["samples_attempted"] == written+1
    assert receipt["samples_written"] == written
    assert receipt["completely_written_sample_bytes"] == 8*written
    assert receipt["partial_sample_bytes_unknown"] is True
    assert receipt["failure_type"] == ("OSError" if written else "RuntimeError")
    assert receipt["sampler_thread_stopped"] and not receipt["inference_result_modified"]
    if written:
        assert chunks == [b"1234567\n", b"12"]
        assert sum(map(len, chunks)) == 10 > receipt["completely_written_sample_bytes"]


def test_exact_128_attempt_and_524288_full_line_stream_caps(sampler_module, monkeypatch):
    m = sampler_module
    sampler = _sampler(m)
    _fake_run(m, monkeypatch, sampler)
    records = []
    def wire(record):
        assert record["sample_index"] == len(records)
        assert sampler.samples_attempted == len(records)+1
        assert record["status"] == "MAIN_THREAD_NOT_OBSERVED"
        assert record["elapsed_seconds"] == 3. and record["process_cpu_seconds"] == 2.
        assert record["process_high_water_rss_bytes"] == 123456
        assert not record["geometry_or_stage_completion_evidence"]
        records.append(record)
        return b"x"*4095+b"\n"
    def write(fd, blob):
        assert fd == 91 and len(blob) == 4096 and sampler.samples_attempted == sampler.samples_written+1
        return len(blob)
    monkeypatch.setattr(m, "_wire", wire)
    monkeypatch.setattr(m, "_write", write)
    sampler._run()
    receipt = sampler.finish()
    assert len(records) == receipt["samples_attempted"] == receipt["samples_written"] == 128
    assert sampler.stop.waits == [30.]*128
    assert receipt["completely_written_sample_bytes"] == 524288
    assert receipt["sample_limit_reached"] and receipt["failure_type"] is None
    assert not receipt["partial_sample_bytes_unknown"]


def test_zero_samples_and_finish_before_start_are_honest_terminal_state(sampler_module, monkeypatch):
    m = sampler_module
    sampler = _sampler(m)
    sampler.stop = Event(stop_immediately=True)
    monkeypatch.setattr(m.sys, "_current_frames", lambda: pytest.fail("NO_SAMPLE_AFTER_STOP"))
    sampler._run()
    receipt = sampler.finish()
    assert receipt["samples_attempted"] == receipt["samples_written"] == receipt["completely_written_sample_bytes"] == 0
    assert not receipt["start_attempted"] and not receipt["sampler_thread_created"]
    assert receipt["sampler_thread_stopped"] and not receipt["partial_sample_bytes_unknown"]
    assert receipt["zero_samples_mean"] == "NO_TIMED_SAMPLE_NOT_PROOF_OF_NO_ENGINE_WORK"
    assert not receipt["sample_limit_reached"] and not receipt["inference_result_modified"]
    assert sampler.finish() == receipt
    with pytest.raises(ValueError, match="ALREADY_STARTED_OR_FINISHED"):
        sampler.start()


def test_start_finish_is_single_owned_daemon_and_policy_copy(sampler_module, monkeypatch):
    m = sampler_module
    made = []
    def factory(**kwargs):
        thread = Thread(**kwargs)
        made.append(thread)
        return thread
    monkeypatch.setattr(m.threading, "Thread", factory)
    sampler = _sampler(m)
    assert sampler.start() is sampler and len(made) == 1
    assert made[0].target == sampler._run and made[0].name == "gold11-stage-sampler-v1" and made[0].daemon is True
    with pytest.raises(ValueError):
        sampler.start()
    receipt = sampler.finish()
    assert made[0].joins == [.25] and sampler.stop.is_set()
    assert receipt["start_attempted"] and receipt["sampler_thread_created"] and receipt["sampler_thread_stopped"]
    assert receipt["last_sample_scope"] == "LAST_OBSERVED_STACK_ONLY_LATER_WORK_UNKNOWN"
    receipt["policy"]["interval_seconds"] = 0.
    assert m.POLICY["interval_seconds"] == 30.


@pytest.mark.parametrize("stage", ["construction", "start"])
def test_thread_start_failure_is_separate_precharged_and_not_retried(sampler_module, monkeypatch, stage):
    m = sampler_module
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        if stage == "construction":
            raise RuntimeError("INVENTED_THREAD_CONSTRUCTOR_FAILURE")
        return Thread(**kwargs, fail_start=True)
    monkeypatch.setattr(m.threading, "Thread", factory)
    sampler = _sampler(m)
    assert sampler.start() is sampler
    with pytest.raises(ValueError):
        sampler.start()
    receipt = sampler.finish()
    assert len(calls) == 1 and receipt["start_attempted"]
    assert receipt["sampler_thread_created"] is (stage == "start")
    assert receipt["failure_type"] == "RuntimeError" and receipt["sampler_thread_stopped"]
    assert receipt["samples_attempted"] == receipt["samples_written"] == 0
    assert receipt["partial_sample_bytes_unknown"] and not receipt["inference_result_modified"]


def test_main_thread_ownership_and_constructor_validation(sampler_module, monkeypatch):
    m = sampler_module
    sampler = _sampler(m)
    with monkeypatch.context() as patch:
        patch.setattr(m.threading, "get_ident", lambda: sampler.main_thread+1)
        for operation in (lambda: _sampler(m), sampler.start, sampler.finish):
            with pytest.raises(ValueError, match="MAIN_THREAD"):
                operation()
        assert not sampler.start_attempted and not sampler.finished
    for fd in (True, -1, 1.5, None):
        with pytest.raises(ValueError, match="DARWIN_LOG_FD"):
            m.MainThreadStageSamplerV1(source_root="/invented", source_paths=("a.py",), output_fd=fd)
    with monkeypatch.context() as patch:
        patch.setattr(m.sys, "platform", "invented-not-darwin")
        with pytest.raises(ValueError, match="DARWIN_LOG_FD"):
            _sampler(m)
    sampler.finish()


def test_still_live_thread_is_explicitly_incomplete_without_signals(sampler_module):
    m = sampler_module
    sampler = _sampler(m)
    sampler.thread = Thread(target=sampler._run, name="invented", daemon=True, alive=True)
    sampler.thread.ident = 12345
    sampler.start_attempted = True
    sampler.samples_attempted, sampler.samples_written, sampler.bytes_written = 2, 1, 81
    receipt = sampler.finish()
    assert sampler.thread.joins == [.25]
    assert receipt["samples_attempted"] == 2 and receipt["samples_written"] == 1
    assert receipt["completely_written_sample_bytes"] == 81
    assert receipt["partial_sample_bytes_unknown"] and not receipt["sampler_thread_stopped"]
    assert receipt["failure_type"] is None and not receipt["inference_result_modified"]


def test_one_short_self_owned_real_thread_control(sampler_module, monkeypatch):
    m = sampler_module
    assert m.INTERVAL_SECONDS == m.POLICY["interval_seconds"] == 30.
    assert threading.current_thread() is threading.main_thread()
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    os.set_blocking(write_fd, False)
    emitted = threading.Event()
    sampler = m.MainThreadStageSamplerV1(source_root=str(ROOT), source_paths=(TEST_SOURCE,), output_fd=write_fd)
    actual_write = m._write
    def write_once(fd, blob):
        count = actual_write(fd, blob)  # Actual nonblocking self-owned pipe; no file path.
        sampler.stop.set()  # Exactly one complete record, never fill the pipe.
        emitted.set()
        return count
    receipt = None
    try:
        monkeypatch.setattr(m, "INTERVAL_SECONDS", TEST_INTERVAL_SECONDS)
        monkeypatch.setattr(m, "_write", write_once)
        sampler.start()
        signaled = emitted.wait(TEST_EVENT_WAIT_SECONDS)
        receipt = sampler.finish()
        assert receipt["sampler_thread_stopped"], "REAL_CONTROL_THREAD_DID_NOT_STOP"
        assert signaled and receipt["failure_type"] is None
        blob = os.read(read_fd, 4097)
        assert 0 < len(blob) <= 4096 and blob.count(b"\n") == 1
        record = json.loads(blob)
        assert record["status"] == "OBSERVED_STACK" and record["sample_index"] == 0
        assert record["frames"] and all(f["source"] == TEST_SOURCE for f in record["frames"])
        assert not record["frame_locals_inspected"] and not record["geometry_or_stage_completion_evidence"]
        assert receipt["samples_attempted"] == receipt["samples_written"] == 1
        assert receipt["completely_written_sample_bytes"] == len(blob)
        assert not receipt["partial_sample_bytes_unknown"] and not receipt["inference_result_modified"]
        assert receipt["policy"]["interval_seconds"] == 30.  # Frozen production declaration, not test timing.
        summary = dict(record="ENGINE_STAGE_SAMPLER_SHORT_THREAD_CONTROL", test_interval_seconds=.01,
            test_event_wait_seconds=1., production_interval_seconds=30., finish_join_seconds=.25,
            samples=1, complete_bytes=len(blob), thread_stopped=True,
            scope="INVENTED_SHORT_THREAD_CONTROL_NOT_PRODUCTION_CADENCE_OR_ENGINE_STAGE_EVIDENCE")
        line = "ENGINE_STAGE_SAMPLER_SHORT_THREAD_JSON="+json.dumps(summary, sort_keys=True, separators=(",", ":"))
        assert len((line+"\n").encode()) <= 4096
        print(line)
    finally:
        if receipt is None:
            receipt = sampler.finish()
        if receipt["sampler_thread_stopped"]:
            os.close(write_fd)
            os.close(read_fd)
        # If the bounded finish cannot certify stop, do not close/reuse a live
        # writer's descriptors. Fail the test; outer worker custody must end it.
        assert receipt["sampler_thread_stopped"]
