"""Bounded, observational main-thread stack samples for a NEW runner only.

No frame locals, array values, input names, Gold, geometry or result changes.
Only exact caller-frozen Python source paths are named. This diagnostic can
localize the last observed work before CPU termination; it cannot prove that
later work did not occur. Source-only candidate, not yet execution-qualified.
"""
import json
import os
from pathlib import PurePath
import resource
import sys
import threading
import time


SCHEMA = "GOLD11_MAIN_THREAD_STAGE_SAMPLE_V1"
INTERVAL_SECONDS = 30.
MAX_SAMPLES = 128
MAX_SOURCE_FILES = 128
MAX_FRAMES = 12
MAX_FRAME_WALK = 64
MAX_SAMPLE_BYTES = 4096
MAX_WRITE_CALLS = 8
FINISH_JOIN_SECONDS = .25
POLICY = dict(schema=SCHEMA, interval_seconds=INTERVAL_SECONDS,
    maximum_samples=MAX_SAMPLES, maximum_named_frames=MAX_FRAMES,
    maximum_frame_walk=MAX_FRAME_WALK, maximum_sample_bytes=MAX_SAMPLE_BYTES,
    maximum_write_calls=MAX_WRITE_CALLS, finish_join_seconds=FINISH_JOIN_SECONDS,
    maximum_sample_stream_bytes=MAX_SAMPLES*MAX_SAMPLE_BYTES,
    scope="OBSERVATIONAL_LAST_SAMPLED_PYTHON_STACK_NOT_STAGE_COMPLETION_OR_GEOMETRY_EVIDENCE",
    failure_policy="KEEP_INFERENCE_UNCHANGED_REPORT_DIAGNOSTIC_FAILURE_SEPARATELY",
    io_limit="SYNCHRONOUS_WRITE_NOT_HARD_CANCELLED_BY_THREAD_OR_CLOCK")


def _sources(root, relative_paths):
    """Lexical source names only; no filesystem discovery or source/input read."""
    if (type(root) is not str or not root.startswith("/") or "\x00" in root
            or not root.isascii() or str(PurePath(root)) != root
            or ".." in PurePath(root).parts or len(root) > 2048
            or type(relative_paths) is not tuple
            or not 1 <= len(relative_paths) <= MAX_SOURCE_FILES):
        raise ValueError("INVALID_FROZEN_STAGE_SOURCE_SCOPE")
    result = {}
    for relative in relative_paths:
        if (type(relative) is not str or not relative.isascii() or "\x00" in relative
                or not 1 <= len(relative) <= 192 or not relative.endswith(".py")
                or relative.startswith("/") or ".." in PurePath(relative).parts
                or str(PurePath(relative)) != relative or relative in result.values()):
            raise ValueError("INVALID_FROZEN_STAGE_SOURCE_PATH")
        result[str(PurePath(root)/relative)] = relative
    return result


def _stack(frame, sources):
    """Copy bounded code locations, never retain frames or examine f_locals."""
    frames, walked, omitted = [], 0, 0
    while frame is not None and walked < MAX_FRAME_WALK:
        walked += 1
        code = frame.f_code
        name = sources.get(code.co_filename)
        if name is not None:
            if len(frames) < MAX_FRAMES:
                frames.append(dict(source=name, function=code.co_name[:64], line=int(frame.f_lineno)))
            else:
                omitted += 1
        frame = frame.f_back
    return dict(frames=frames, walked_frames=walked, omitted_named_frames=omitted,
        stack_walk_truncated=frame is not None,
        complete_python_stack_claim=False, frame_locals_inspected=False)


def _wire(record):
    blob = (json.dumps(record, sort_keys=True, separators=(",", ":"),
        allow_nan=False)+"\n").encode("ascii")
    if len(blob) > MAX_SAMPLE_BYTES:
        record = dict(schema=SCHEMA, sample_index=record["sample_index"],
            status="SAMPLE_TOO_LARGE_STACK_OMITTED_WHOLE", frames=[],
            geometry_or_stage_completion_evidence=False)
        blob = (json.dumps(record, sort_keys=True, separators=(",", ":"))+"\n").encode("ascii")
    return blob


def _write(fd, blob):
    offset = 0
    for _ in range(MAX_WRITE_CALLS):
        if offset == len(blob):
            return offset
        count = os.write(fd, memoryview(blob)[offset:])
        if type(count) is not int or count <= 0 or count > len(blob)-offset:
            raise OSError("STAGE_SAMPLE_SHORT_OR_INVALID_WRITE")
        offset += count
    if offset != len(blob):
        raise OSError("STAGE_SAMPLE_WRITE_CALL_CAP")
    return offset


class MainThreadStageSamplerV1:
    """One self-owned daemon thread; finish cannot signal or block indefinitely.

    Output shares the worker's existing bounded log. Successful inference does
    not depend on successful diagnostic sampling. A still-writing daemon or a
    partial log remains explicitly incomplete, not a certified stage receipt.
    """
    def __init__(self, *, source_root, source_paths, output_fd):
        if sys.platform != "darwin" or type(output_fd) is not int or output_fd < 0:
            raise ValueError("DECLARED_DARWIN_LOG_FD_REQUIRED")
        if threading.get_ident() != threading.main_thread().ident:
            raise ValueError("STAGE_SAMPLER_MUST_BE_OWNED_BY_MAIN_THREAD")
        self.sources = _sources(source_root, source_paths)
        self.fd = output_fd
        self.main_thread = threading.get_ident()
        self.started_at = time.monotonic()
        self.stop = threading.Event()
        self.thread = None
        self.start_attempted = self.finished = False
        self.samples_attempted = self.samples_written = self.bytes_written = 0
        self.failure = None
        self.limit_reached = False

    def start(self):
        if self.start_attempted or self.finished:
            raise ValueError("STAGE_SAMPLER_ALREADY_STARTED_OR_FINISHED")
        if threading.get_ident() != self.main_thread:
            raise ValueError("STAGE_SAMPLER_MUST_BE_STARTED_BY_MAIN_THREAD")
        self.start_attempted = True
        try:
            self.thread = threading.Thread(target=self._run, name="gold11-stage-sampler-v1", daemon=True)
            self.thread.start()
        except Exception as error:
            self.failure = type(error).__name__
        return self

    def _run(self):
        try:
            for index in range(MAX_SAMPLES):
                if self.stop.wait(INTERVAL_SECONDS):
                    return
                self.samples_attempted += 1  # Precharge; no failed-sample refund.
                frame = sys._current_frames().get(self.main_thread)
                try:
                    stack = _stack(frame, self.sources)
                    present = frame is not None
                finally:
                    del frame
                usage = resource.getrusage(resource.RUSAGE_SELF)
                record = dict(schema=SCHEMA, sample_index=index,
                    status="OBSERVED_STACK" if present else "MAIN_THREAD_NOT_OBSERVED",
                    elapsed_seconds=time.monotonic()-self.started_at,
                    process_cpu_seconds=usage.ru_utime+usage.ru_stime,
                    process_high_water_rss_bytes=usage.ru_maxrss,
                    geometry_or_stage_completion_evidence=False, **stack)
                blob = _wire(record)
                self.bytes_written += _write(self.fd, blob)
                self.samples_written += 1
            self.limit_reached = True
        except Exception as error:
            self.failure = type(error).__name__

    def finish(self):
        if threading.get_ident() != self.main_thread:
            raise ValueError("STAGE_SAMPLER_MUST_BE_FINISHED_BY_MAIN_THREAD")
        self.finished = True
        self.stop.set()
        thread = self.thread
        if thread is not None and thread.ident is not None:
            thread.join(FINISH_JOIN_SECONDS)
        alive = thread is not None and thread.is_alive()
        return dict(schema="GOLD11_MAIN_THREAD_STAGE_SAMPLER_RECEIPT_V1", policy=dict(POLICY),
            start_attempted=self.start_attempted, sampler_thread_created=thread is not None,
            samples_attempted=self.samples_attempted, samples_written=self.samples_written,
            completely_written_sample_bytes=self.bytes_written,
            partial_sample_bytes_unknown=self.failure is not None or alive,
            sampler_thread_stopped=not alive, sample_limit_reached=self.limit_reached,
            failure_type=self.failure, inference_result_modified=False,
            zero_samples_mean="NO_TIMED_SAMPLE_NOT_PROOF_OF_NO_ENGINE_WORK",
            last_sample_scope="LAST_OBSERVED_STACK_ONLY_LATER_WORK_UNKNOWN")
