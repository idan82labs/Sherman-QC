"""Unqualified macOS generation-aware V3 successor; never imports a runner.

One owned child, no retry. Native PPID/group censuses replace recurring ps
children. Lifetime identity is (PID, uniqueid); exec's pidversion is refreshed
only after coherent identity/ownership checks. Signals use the kernel's audit-
token operation, never numeric kill/killpg. No library is loaded at import.

The caller remains the sole waiter for its direct child and must leave SIGCHLD
at SIG_DFL. That child is kept unreaped until group cleanup ends, including as
a zombie: its PID cannot be recycled while supplying the owned-group anchor.
Sampling cannot observe a descendant born, escaped and reparented between
censuses. Kernel calls, launch/stdin/log I/O are not cancellable hard deadlines;
the caller must contain the whole invocation within its own absolute budget.

ABI/semantics sources (Apple XNU, source reviewed 2026-09-12):
https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info.h
https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info_private.h
https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/proc_info.c
https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_proc.c
https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_prot.c
https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_exit.c
https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.c
Private interfaces require separate host qualification; no unsupported fallback.
"""

import ctypes
import errno
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time


MAX_PIDS_PER_LIST = 4096
MAX_TRACKED_IDENTITIES = 8192
MAX_ADMITTED_IDENTITIES = 1_000_000
MAX_WORK_SAMPLES = 3_000_000
MAX_WORK_NATIVE_CALLS = 100_000_000
MAX_CLEANUP_SAMPLES = 1024
MAX_CLEANUP_NATIVE_CALLS = 262_144
MAX_NATIVE_CALLS_PER_SAMPLE = 65_536
MAX_IDENTITY_EVENTS = 16
COHERENCE_ATTEMPTS = 2
EXIT_TRANSITION_YIELD_SECONDS = .001
PROC_FLAG_INEXIT = 4
SZOMB = 5


def _require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


class _ExitTransitionDeadline(TimeoutError):
    """A previously observed exit transition did not resolve in this sample."""
    def __init__(self, cause):
        self.cause = str(cause)[:160]
        super().__init__("EXIT_TRANSITION_UNRESOLVED_DEADLINE:"+self.cause)


class _BSDInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid",
        "svuid", "svgid", "reserved")]
    _fields_ += [("comm", ctypes.c_char*16), ("name", ctypes.c_char*32)]
    _fields_ += [(name, ctypes.c_uint32) for name in ("nfiles", "pgid", "pjobc", "tdev", "tpgid")]
    _fields_ += [("nice", ctypes.c_int32), ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]


class _UniqueInfo(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8*16), ("uniqueid", ctypes.c_uint64),
        ("parent_uniqueid", ctypes.c_uint64), ("idversion", ctypes.c_int32),
        ("original_parent_version", ctypes.c_int32), ("reserve2", ctypes.c_uint64),
        ("reserve3", ctypes.c_uint64)]


class _BSDWithUnique(ctypes.Structure):
    _fields_ = [("bsd", _BSDInfo), ("unique", _UniqueInfo)]


class _TaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "virtual_size", "resident_size", "total_user", "total_system", "threads_user", "threads_system")]
    _fields_ += [(name, ctypes.c_int32) for name in (
        "policy", "faults", "pageins", "cow_faults", "messages_sent", "messages_received",
        "syscalls_mach", "syscalls_unix", "csw", "threadnum", "numrunning", "priority")]


class _AuditToken(ctypes.Structure):
    _fields_ = [("val", ctypes.c_uint32*8)]


def _lifetime(row):
    return row["pid"], row["uniqueid"]


def _coherent_key(row):
    # Runnable/sleeping transitions and mutable RSS are not identity changes.
    return tuple(row[key] for key in ("pid", "uniqueid", "idversion", "ppid", "parent_uniqueid",
        "pgid", "uid", "gid", "ruid", "rgid", "start_sec", "start_usec")) + (row["status"] == SZOMB,)


class _MacOSProcessAPI:
    """Narrow native adapter, independently mockable without loading libproc.

    Returned errno from proc_signal_with_audittoken is the function result;
    proc_pidinfo/proc_listpids instead return zero and set errno on failure.
    Flavor18 arg=1 includes zombies. Only its explicit ESRCH means absence;
    short records, permission failures and missing symbols are never absence.
    """
    def __init__(self):
        _require(sys.platform == "darwin" and sys.byteorder == "little"
            and ctypes.sizeof(ctypes.c_void_p) == 8, "UNSUPPORTED_PROCESS_IDENTITY_PLATFORM")
        _require([ctypes.sizeof(kind) for kind in (_BSDInfo, _UniqueInfo, _BSDWithUnique, _TaskInfo, _AuditToken)]
            == [136, 56, 192, 96, 32], "UNSUPPORTED_LIBPROC_LAYOUT")
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                          ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_pidinfo.restype = ctypes.c_int
        self.lib.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_listpids.restype = ctypes.c_int
        self.lib.proc_signal_with_audittoken.argtypes = [ctypes.POINTER(_AuditToken), ctypes.c_int]
        self.lib.proc_signal_with_audittoken.restype = ctypes.c_int
        self.phase = "work"
        self.deadline = 0.
        self.in_sample = 0
        self.work = dict(work_samples=0, cleanup_samples=0, work_native_calls=0,
            cleanup_native_calls=0, native_calls_attempted=0, list_calls=0,
            identity_calls=0, rss_calls=0, signal_calls=0, coherence_retries=0,
            explicit_esrch_observations=0, token_stale_or_exit=0,
            task_rss_esrch_observations=0, exit_transition_entries=0,
            exit_transition_waits=0, exit_transition_wait_seconds=0.,
            exit_transition_resolved_rss=0, exit_transition_resolved_zombie=0,
            exit_transition_resolved_absence=0, exit_transition_unresolved=0,
            exit_transition_diagnostics_discarded=0)
        self.exit_transition_diagnostics = []
        self.exit_transition_first_failure = None

    def begin_sample(self, end):
        _require(type(end) in (int, float) and math.isfinite(end), "INVALID_NATIVE_SAMPLE_DEADLINE")
        self.deadline = min(end, time.monotonic()+1.)
        key = self.phase+"_samples"
        self.work[key] += 1
        maximum = MAX_WORK_SAMPLES if self.phase == "work" else MAX_CLEANUP_SAMPLES
        _require(self.work[key] <= maximum, "PROCESS_SAMPLE_COUNT_LIMIT")
        self.in_sample = 0
        self.check_time()

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("NATIVE_PROCESS_SAMPLE_DEADLINE")

    def _call(self, kind, function, *args):
        self.check_time()
        self.in_sample += 1
        self.work["native_calls_attempted"] += 1
        self.work[kind+"_calls"] += 1
        key = self.phase+"_native_calls"
        self.work[key] += 1
        maximum = MAX_WORK_NATIVE_CALLS if self.phase == "work" else MAX_CLEANUP_NATIVE_CALLS
        _require(self.in_sample <= MAX_NATIVE_CALLS_PER_SAMPLE and self.work[key] <= maximum,
                 "NATIVE_PROCESS_CALL_LIMIT")
        ctypes.set_errno(0)
        result = function(*args)
        error = ctypes.get_errno()
        self.check_time()  # A late return is failure, not a hard-cancellation claim.
        return result, error

    def list_pids(self, kind, identifier):
        _require(type(kind) is int and kind in (2, 6) and type(identifier) is int
            and 0 < identifier < 2**31, "INVALID_SCOPED_PID_ENUMERATION")
        # One overflow slot distinguishes the exact cap from a saturated buffer.
        slots = (ctypes.c_int32*(MAX_PIDS_PER_LIST+1))()
        size, error = self._call("list", self.lib.proc_listpids, kind, identifier,
                                 ctypes.byref(slots), ctypes.sizeof(slots))
        _require(size >= 0 and size % 4 == 0 and size <= ctypes.sizeof(slots)
            and not (size == 0 and error), "PID_ENUMERATION_UNAVAILABLE_OR_MALFORMED")
        _require(size//4 <= MAX_PIDS_PER_LIST, "PID_ENUMERATION_CAP_OR_TRUNCATION")
        rows = tuple(int(slots[index]) for index in range(size//4))
        _require(all(0 < pid < 2**31 for pid in rows) and len(set(rows)) == len(rows),
                 "INVALID_OR_DUPLICATE_ENUMERATED_PID")
        return tuple(sorted(rows))

    def _bsd(self, pid):
        _require(type(pid) is int and 0 < pid < 2**31, "INVALID_IDENTITY_PID")
        record = _BSDWithUnique()
        size, error = self._call("identity", self.lib.proc_pidinfo, pid, 18, 1,
                                 ctypes.byref(record), ctypes.sizeof(record))
        if size == 0 and error == errno.ESRCH:
            self.work["explicit_esrch_observations"] += 1
            return None
        _require(size == ctypes.sizeof(record), "PROCESS_IDENTITY_UNAVAILABLE_OR_SHORT")
        bsd, unique = record.bsd, record.unique
        answer = {key: int(getattr(bsd, key)) for key in ("pid", "ppid", "pgid", "status", "flags",
                  "uid", "gid", "ruid", "rgid", "start_sec", "start_usec")}
        answer.update(uniqueid=int(unique.uniqueid), parent_uniqueid=int(unique.parent_uniqueid),
                      idversion=int(unique.idversion) & 0xffffffff)
        _require(answer["pid"] == pid and answer["uniqueid"] > 0 and answer["idversion"] > 0
            and 0 < answer["pgid"] < 2**31 and 0 <= answer["ppid"] < 2**31
            and 1 <= answer["status"] <= SZOMB and answer["start_usec"] < 1_000_000,
            "MALFORMED_PROCESS_IDENTITY_FIELDS")
        return answer

    def _transition_diagnostic(self, outcome, started, first, second, task_size, task_errno,
                               cause=None, entry=None):
        def view(row):
            return None if row is None else {key: row[key] for key in (
                "pid", "uniqueid", "idversion", "ppid", "parent_uniqueid", "pgid",
                "uid", "gid", "ruid", "rgid", "start_sec", "start_usec", "status", "flags")}
        event = dict(outcome=outcome, phase=self.phase, before=view(first), after=view(second),
            task_size=task_size, task_errno=task_errno,
            elapsed_seconds=max(0., time.monotonic()-started), sample_native_calls=self.in_sample,
            cause=None if cause is None else str(cause)[:160], entry=entry)
        # Routine resolved transitions cannot displace the first causal failure.
        if outcome == "UNRESOLVED" and self.exit_transition_first_failure is None:
            self.exit_transition_first_failure = dict(event)
        if len(self.exit_transition_diagnostics) < MAX_IDENTITY_EVENTS:
            self.exit_transition_diagnostics.append(event)
        else:
            self.work["exit_transition_diagnostics_discarded"] += 1
        return event

    def inspect(self, pid, *, rss=False):
        _require(type(rss) is bool, "INVALID_RSS_QUERY_SELECTION")
        mismatches, transition_started, transition_entry = 0, None, None
        first = second = task_size = task_errno = None

        def resolved(kind):
            if transition_started is not None:
                self.work["exit_transition_resolved_"+kind] += 1
                self._transition_diagnostic("RESOLVED_"+kind.upper(), transition_started,
                    first, second, task_size, task_errno, entry=transition_entry)

        try:
            while mismatches < COHERENCE_ATTEMPTS:
                first = second = task_size = task_errno = None
                first = self._bsd(pid)
                if first is None:
                    resolved("absence")
                    return None
                resident, task_missing = None, False
                if rss and first["status"] != SZOMB:
                    task = _TaskInfo()
                    task_size, task_errno = self._call("rss", self.lib.proc_pidinfo, pid, 4, 0,
                                             ctypes.byref(task), ctypes.sizeof(task))
                    task_missing = task_size == 0 and task_errno == errno.ESRCH
                    if task_missing:
                        self.work["task_rss_esrch_observations"] += 1
                    _require(task_missing or task_size == ctypes.sizeof(task),
                             "TASK_RSS_UNAVAILABLE_OR_SHORT")
                    if not task_missing:
                        resident = int(task.resident_size)
                second = self._bsd(pid)
                if second is None:
                    resolved("absence")
                    return None
                coherent = _coherent_key(first) == _coherent_key(second)
                if coherent and not task_missing:
                    second["rss"] = resident
                    resolved("zombie" if second["status"] == SZOMB else "rss")
                    return second
                if (coherent and task_missing and first["status"] != SZOMB
                        and second["status"] != SZOMB and second["flags"] & PROC_FLAG_INEXIT):
                    # Positive exit-in-progress plus coherent identity permits
                    # only a bounded observation wait, never RSS/exit acceptance.
                    # No begin_sample, deadline renewal or mismatch-count reset.
                    if transition_started is None:
                        transition_started = time.monotonic()
                        self.work["exit_transition_entries"] += 1
                        transition_entry = self._transition_diagnostic("ENTERED", transition_started,
                            first, second, task_size, task_errno)
                    self.check_time()
                    remaining = self.deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("NATIVE_PROCESS_SAMPLE_DEADLINE")
                    wait_started = time.monotonic()
                    self.work["exit_transition_waits"] += 1
                    try:
                        time.sleep(min(EXIT_TRANSITION_YIELD_SECONDS, remaining))
                    finally:
                        self.work["exit_transition_wait_seconds"] += max(
                            0., time.monotonic()-wait_started)
                    self.check_time()
                    continue
                # Preserve the original allowance for genuinely incoherent or
                # otherwise unresolved reads, even across transition waits.
                mismatches += 1
                self.work["coherence_retries"] += 1
            raise RuntimeError("PROCESS_IDENTITY_OR_RSS_DID_NOT_STABILIZE")
        except Exception as error:
            if transition_started is not None:
                self.work["exit_transition_unresolved"] += 1
                self._transition_diagnostic("UNRESOLVED", transition_started,
                    first, second, task_size, task_errno, cause=error, entry=transition_entry)
                if isinstance(error, TimeoutError):
                    raise _ExitTransitionDeadline(error) from error
            raise

    def signal_owned(self, expected):
        """Only SIGKILL; never signal0 or a numeric-PID fallback.

        The frozen kernel lookup consumes token slots5/7 for PID/pidversion,
        then holds/validates a proc_ident through psignal and uses the real
        caller's credentials. This lookup token is NOT a credential attestation.
        A late exec or replacement yields ESRCH instead of signaling its new
        version; the caller may only retry after a new coherent ownership read.
        """
        current = self.inspect(expected["pid"])
        if current is None:
            return dict(status="VERIFIED_ABSENT", current=None)
        if _lifetime(current) != _lifetime(expected):
            return dict(status="REPLACEMENT_NOT_SIGNALED", current=current)
        if current["pgid"] != expected["pgid"]:
            return dict(status="GROUP_CHANGE_NOT_SIGNALED", current=current)
        if current["status"] == SZOMB:
            return dict(status="ZOMBIE_NOT_SIGNALED", current=current)
        token = _AuditToken()
        token.val[1], token.val[2] = current["uid"], current["gid"]
        token.val[3], token.val[4] = current["ruid"], current["rgid"]
        token.val[5], token.val[7] = current["pid"], current["idversion"]
        result, _ = self._call("signal", self.lib.proc_signal_with_audittoken,
                                ctypes.byref(token), int(signal.SIGKILL))
        if result == errno.ESRCH:
            self.work["token_stale_or_exit"] += 1
            return dict(status="TOKEN_STALE_OR_EXIT_NOT_RETIREMENT_PROOF", current=current)
        _require(result == 0, "ATOMIC_OWNED_SIGNAL_DENIED_OR_UNAVAILABLE")
        return dict(status="ATOMIC_SIGNAL_ACCEPTED", current=current)


class _Custody:
    def __init__(self, api, child_pid, parent, outer, receipt, stop):
        self.api, self.parent, self.child_pid = api, parent, child_pid
        self.outer, self.receipt, self.stop = outer, receipt, stop
        self.expected_group = child_pid if outer else parent["pgid"]
        self.known, self.unknown = {}, set()
        self.events, self.error_events = [], []
        self.counters = dict(admitted=0, tracked_peak=0, verified_absence_retirements=0,
            replacement_retirements=0, exec_version_updates=0, reparent_updates=0,
            candidate_membership_races=0, events_discarded=0, error_events_discarded=0)
        self.reaped = self.done = False
        root = api.inspect(child_pid)
        _require(root is not None and root["ppid"] == parent["pid"]
            and root["parent_uniqueid"] == parent["uniqueid"]
            and root["pgid"] == self.expected_group, "DIRECT_CHILD_IDENTITY_NOT_AUTHENTICATED")
        self.root = dict(root)
        self._admit(root)

    def _event(self, kind, before, after):
        def view(row):
            return None if row is None else {key: row[key] for key in ("pid", "uniqueid", "idversion", "ppid", "pgid")}
        event = dict(kind=kind, before=view(before), after=view(after))
        # Routine short-lived children must not consume the only evidence slot
        # for a later causal identity failure. Both histories are bounded; the
        # critical history preserves its first actual failure independently.
        if kind in ("SAME_GENERATION_GROUP_CHANGE", "NEW_AUTHENTICATED_FOREIGN_DESCENDANT",
                    "OWNED_IDENTITY_QUERY_UNAVAILABLE", "OWNED_SIGNAL_IDENTITY_OR_OPERATION_UNAVAILABLE",
                    "GROUP_CHANGE_NOT_SIGNALED"):
            if len(self.error_events) < MAX_IDENTITY_EVENTS:
                self.error_events.append(event)
            else:
                self.counters["error_events_discarded"] += 1
        if len(self.events) < MAX_IDENTITY_EVENTS:
            self.events.append(event)
        else:
            self.counters["events_discarded"] += 1

    def _mark_unknown(self, pid):
        self.unknown.add(pid)
        self.receipt["observed_identity_unknown_pid_count_peak"] = max(
            self.receipt["observed_identity_unknown_pid_count_peak"], len(self.unknown))

    def _admit(self, row):
        _require(row["pid"] != self.parent["pid"], "SUPERVISOR_PRESENT_IN_CHILD_CUSTODY")
        if row["pid"] in self.known:
            _require(_lifetime(self.known[row["pid"]]) == _lifetime(row), "UNRECONCILED_PID_GENERATION")
            self._update(row)
            return
        self.counters["admitted"] += 1
        _require(len(self.known) < MAX_TRACKED_IDENTITIES
            and self.counters["admitted"] <= MAX_ADMITTED_IDENTITIES, "OWNED_IDENTITY_TRACKING_CAP")
        self.known[row["pid"]] = dict(row)
        self.counters["tracked_peak"] = max(self.counters["tracked_peak"], len(self.known))
        if row["pgid"] != self.expected_group:
            self._escape(row, None)

    def _escape(self, row, before):
        self.receipt["observed_foreign_group_escape_detected"] = True
        if before is not None:
            self.receipt["observed_process_identity_change_detected"] = True
            self._event("SAME_GENERATION_GROUP_CHANGE", before, row)
        else:
            self._event("NEW_AUTHENTICATED_FOREIGN_DESCENDANT", None, row)
        self.stop("OBSERVED_DESCENDANT_OUTSIDE_OWNED_GROUP")

    def _update(self, row):
        before = self.known[row["pid"]]
        if row["pgid"] != before["pgid"]:
            self._escape(row, before)
        if row["idversion"] != before["idversion"]:
            self.counters["exec_version_updates"] += 1
        if row["ppid"] != before["ppid"]:
            self.counters["reparent_updates"] += 1
        self.known[row["pid"]] = dict(row)

    def _retire(self, pid, replacement):
        before = self.known.pop(pid)
        kind = "VERIFIED_KERNEL_ABSENCE" if replacement is None else "VERIFIED_NEW_GENERATION"
        self.counters["verified_absence_retirements" if replacement is None else "replacement_retirements"] += 1
        self._event(kind, before, replacement)
        self.unknown.discard(pid)

    def snapshot(self, end):
        self.api.begin_sample(end)
        cache = {}

        def inspect(pid):
            if pid not in cache:
                _require(len(cache) < 2*MAX_TRACKED_IDENTITIES+1, "PROCESS_SAMPLE_ROW_CAP")
                try:
                    cache[pid] = self.api.inspect(pid, rss=True)
                except Exception:
                    if pid in self.known:
                        self._mark_unknown(pid)
                        self._event("OWNED_IDENTITY_QUERY_UNAVAILABLE", self.known[pid], None)
                    raise
            return cache[pid]

        parent = inspect(self.parent["pid"])
        _require(parent is not None and _lifetime(parent) == _lifetime(self.parent)
            and type(parent["rss"]) is int, "SUPERVISOR_IDENTITY_OR_RSS_UNAVAILABLE")
        root = inspect(self.child_pid)
        _require(root is not None and _lifetime(root) == _lifetime(self.root)
            and not self.reaped, "UNREAPED_CHILD_GENERATION_ANCHOR_LOST")
        self.done = root["status"] == SZOMB
        for pid in tuple(self.known):
            row = inspect(pid)
            if row is None or _lifetime(row) != _lifetime(self.known[pid]):
                _require(pid != self.child_pid, "DIRECT_CHILD_LIFETIME_LOST_BEFORE_REAP")
                self._retire(pid, row)
            else:
                self.unknown.discard(pid)
                self._update(row)
        # The unreaped root anchors the numeric group identifier. A candidate
        # that vanishes/changes membership before identity capture never gains
        # signal authority. It is an explicitly unobserved census race.
        if self.outer:
            for pid in self.api.list_pids(2, self.child_pid):
                row = inspect(pid)
                if row is not None and row["pgid"] == self.expected_group:
                    self._admit(row)
                else:
                    self.counters["candidate_membership_races"] += 1
        queue, visited = list(self.known), set()
        while queue:
            pid = queue.pop()
            row = self.known.get(pid)
            if row is None or _lifetime(row) in visited:
                continue
            _require(len(visited) < MAX_TRACKED_IDENTITIES, "DESCENDANT_ENUMERATION_CAP")
            visited.add(_lifetime(row))
            if row["status"] == SZOMB:
                continue
            for candidate in self.api.list_pids(6, pid):
                member = inspect(candidate)
                # p_puniqueid is not rewritten on ordinary reparenting. Use it
                # for positive new-child ancestry, never to discard a known
                # surviving generation whose present PPID changed.
                if member is not None and member["ppid"] == pid and member["parent_uniqueid"] == row["uniqueid"]:
                    was_known = candidate in self.known
                    self._admit(member)
                    if not was_known:
                        queue.append(candidate)
                else:
                    self.counters["candidate_membership_races"] += 1
        foreign = {pid for pid, row in self.known.items() if row["pgid"] != self.expected_group}
        self.receipt["observed_foreign_pid_count_peak"] = max(self.receipt["observed_foreign_pid_count_peak"], len(foreign))
        self.receipt["observed_foreign_group_escape_unconfirmed_after_cleanup"] = bool(foreign)
        live_rss = [row["rss"] for row in self.known.values() if type(row.get("rss")) is int]
        group = {pid for pid, row in self.known.items() if row["pgid"] == self.expected_group}
        if self.done:
            group.discard(self.child_pid)
        return dict(done=self.done, group=group, retained=set(self.known), foreign=foreign,
            rss=sum(live_rss), parent_rss=parent["rss"], any_rss_sample=bool(live_rss),
            child_rss_sample=type(self.known[self.child_pid].get("rss")) is int)

    def publish(self):
        self.receipt["identity_counters"] = dict(self.counters)
        self.receipt["identity_events"] = list(self.events)
        self.receipt["identity_error_events"] = list(self.error_events)
        self.receipt["observed_identity_unknown_still_present"] = bool(self.unknown)
        self.receipt["observed_identity_unknown_pid_count_peak"] = max(
            self.receipt["observed_identity_unknown_pid_count_peak"], len(self.unknown))
        self.receipt["observed_foreign_identity_changed_or_unknown"] = bool(self.unknown)

    def signal_current_owned(self):
        # Stop the direct producer first, then its authenticated descendants.
        # Independently preserve the unreaped anchor even when an earlier full
        # census failed on another row. That failure never relaxes this check.
        anchor = self.api.inspect(self.child_pid)
        _require(anchor is not None and _lifetime(anchor) == _lifetime(self.root)
            and not self.reaped, "UNREAPED_CHILD_SIGNAL_ANCHOR_LOST")
        pids = [self.child_pid] + sorted(set(self.known)-{self.child_pid}, reverse=True)
        errors = []
        for pid in pids:
            self.api.check_time()
            expected = self.known.get(pid)
            if expected is None or expected["pgid"] != self.expected_group or expected["status"] == SZOMB:
                continue
            try:
                outcome = self.api.signal_owned(expected)
            except Exception as error:
                self._mark_unknown(pid)
                self._event("OWNED_SIGNAL_IDENTITY_OR_OPERATION_UNAVAILABLE", expected, None)
                self.stop("OWNED_PROCESS_SIGNAL_UNCONFIRMED")
                if len(errors) < MAX_IDENTITY_EVENTS:
                    errors.append(type(error).__name__+":"+str(error)[:160])
                if pid == self.child_pid:
                    raise  # Do not continue past a lost direct anchor.
                continue
            self._event(outcome["status"], expected, outcome["current"])
            if outcome["status"] in ("VERIFIED_ABSENT", "REPLACEMENT_NOT_SIGNALED"):
                if pid == self.child_pid:
                    raise RuntimeError("DIRECT_CHILD_ANCHOR_LOST_DURING_SIGNAL")
                self._retire(pid, outcome["current"])
            elif outcome["status"] == "GROUP_CHANGE_NOT_SIGNALED":
                self._update(outcome["current"])
            # ESRCH from atomic signaling is ambiguous between exit and exec;
            # it is NOT retirement evidence. The next snapshot must resolve it.
        return errors

def clean_environment(cache):
    return dict(PATH="/usr/bin:/bin", HOME=str(cache), TMPDIR=str(cache),
                XDG_CACHE_HOME=str(cache), PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def memory_budget(tree_rss, parent_rss, rss_limit, *, include_parent=False, parent_rss_limit=None):
    if any(type(value) is not int or value < 0 for value in (tree_rss, parent_rss, rss_limit)):
        raise ValueError("INVALID_RSS_SAMPLE")
    measured = tree_rss + parent_rss if include_parent else tree_rss
    if measured > rss_limit:
        return measured, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT" if include_parent else "TREE_RSS_LIMIT"
    if parent_rss_limit is not None and parent_rss > parent_rss_limit:
        return measured, "COORDINATOR_RSS_LIMIT"
    return measured, None


def supervise_process(command, request, *, cache, log_path, wall_seconds, rss_limit,
                      log_limit, new_session=False, parent_rss_limit=None):
    """One child; V1 call shape, generation-aware observation and cleanup.

    Work has the supplied deadline; failed-only cleanup reserves five additional
    seconds. All time is returned, but native calls and synchronous I/O cannot
    be forcibly interrupted here. The caller remains the sole child waiter and
    owns an enclosing deadline. There is no process-group numeric signal.
    A cleanup failure returns explicit attention/uncertainty, never permission
    to signal a PID from this receipt later. Sampling is not OS containment.
    """
    if (type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or wall_seconds <= 0
            or any(type(value) is not int or value <= 0 for value in (rss_limit, log_limit))
            or parent_rss_limit is not None and (type(parent_rss_limit) is not int or parent_rss_limit <= 0)
            or type(new_session) is not bool):
        raise ValueError("INVALID_SUPERVISION_BUDGET")
    started = time.monotonic()
    work_end = started+wall_seconds
    child, api, state, parent = None, None, None, None
    selector = selectors.DefaultSelector()
    receipt = dict(exit_code=None, budget_stop=None, child_pid=None,
        sampled_tree_peak_rss_bytes=0, sampled_parent_peak_rss_bytes=0,
        sampled_budget_peak_rss_bytes=0, forwarded_log_bytes=0, stored_log_bytes=0,
        log_write_attempted_bytes=0, log_write_accepted_bytes=0,
        log_storage_bytes_uncertain=False, log_storage_status="NOT_CREATED",
        log_created=False, owned_group_id=None, owned_group_empty_confirmed=None,
        observed_foreign_group_escape_detected=False,
        observed_foreign_pid_count_peak=0,
        observed_foreign_group_escape_unconfirmed_after_cleanup=False,
        observed_foreign_identity_changed_or_unknown=False,
        observed_process_identity_change_detected=False,
        observed_identity_unknown_pid_count_peak=0,
        observed_identity_unknown_still_present=False,
        cleanup_errors=[], control_close_errors=[], direct_child_reaped=False,
        child_rss_sample_count=0, child_tree_rss_sample_count=0,
        child_rss_observation_status="NOT_YET_OBSERVED",
        rss_budget_scope="OWNED_GROUP_AND_DESCENDANTS_PLUS_OUTER_SUPERVISOR" if new_session else "OBSERVED_LEAF_TREE",
        inner_custody_limit="DIRECT_CHILD_AND_OBSERVED_DESCENDANTS_NOT_UNOBSERVED_NATIVE_REPARENTING",
        supervisor_schema="GOLD11_OWNED_PROCESS_SUPERVISION_V3",
        process_identity_policy="COHERENT_PID_UNIQUEID_LIFETIME_WITH_REFRESHED_EXEC_PIDVERSION",
        atomic_signal_scope="FRESH_OWNED_GENERATION_AUDIT_TOKEN_NOT_ATOMIC_GROUP_MEMBERSHIP",
        requires_process_attention=False, identity_counters={}, identity_events=[], identity_error_events=[],
        native_process_work={}, supervisor_failure_reason=None,
        exit_transition_diagnostics=[], exit_transition_first_failure=None,
        cleanup_error_details=[], cleanup_errors_discarded=0,
        process_api_limits=dict(pids_per_list=MAX_PIDS_PER_LIST,
            tracked_identities=MAX_TRACKED_IDENTITIES, admitted_identities=MAX_ADMITTED_IDENTITIES,
            work_samples=MAX_WORK_SAMPLES, work_native_calls=MAX_WORK_NATIVE_CALLS,
            cleanup_samples=MAX_CLEANUP_SAMPLES, cleanup_native_calls=MAX_CLEANUP_NATIVE_CALLS,
            native_calls_per_sample=MAX_NATIVE_CALLS_PER_SAMPLE,
            identity_events=MAX_IDENTITY_EVENTS, identity_error_events=MAX_IDENTITY_EVENTS,
            coherence_attempts=COHERENCE_ATTEMPTS,
            exit_transition_yield_seconds=EXIT_TRANSITION_YIELD_SECONDS,
            exit_transition_diagnostics=MAX_IDENTITY_EVENTS))

    def stop(reason):
        if receipt["budget_stop"] is None:
            receipt["budget_stop"] = reason

    def cleanup_error(error):
        if len(receipt["cleanup_errors"]) < MAX_IDENTITY_EVENTS:
            receipt["cleanup_errors"].append(type(error).__name__)
            receipt["cleanup_error_details"].append(str(error)[:160])
        else:
            receipt["cleanup_errors_discarded"] += 1
        stop("OWNED_PROCESS_CLEANUP_UNCONFIRMED")

    def sample(end):
        observed = state.snapshot(end)
        rss, parent_rss = observed["rss"], observed["parent_rss"]
        receipt["child_rss_sample_count"] += int(observed["child_rss_sample"])
        receipt["child_tree_rss_sample_count"] += int(observed["any_rss_sample"])
        receipt["sampled_tree_peak_rss_bytes"] = max(receipt["sampled_tree_peak_rss_bytes"], rss)
        receipt["sampled_parent_peak_rss_bytes"] = max(receipt["sampled_parent_peak_rss_bytes"], parent_rss)
        measured, reason = memory_budget(rss, parent_rss, rss_limit,
            include_parent=new_session, parent_rss_limit=parent_rss_limit)
        receipt["sampled_budget_peak_rss_bytes"] = max(receipt["sampled_budget_peak_rss_bytes"], measured)
        if reason:
            stop(reason)
        state.publish()
        return observed

    def reap_and_confirm(end, observed):
        # No poll/wait has previously released the root's generation anchor.
        _require(observed["done"] and not observed["group"], "REAP_BEFORE_OWNED_GROUP_QUIESCENCE")
        left = end-time.monotonic()
        if left <= 0:
            raise TimeoutError("CHILD_REAP_DEADLINE")
        child.wait(timeout=left)
        state.reaped = True
        receipt["direct_child_reaped"] = True
        if new_session:
            # Read-only final census after reap; no signal may follow anchor
            # release, even if a newly reused numeric PGID now appears.
            api.begin_sample(end)
            _require(not api.list_pids(2, child.pid), "OWNED_GROUP_NOT_EMPTY_AFTER_REAP")
            receipt["owned_group_empty_confirmed"] = True

    def cleanup():
        nonlocal state
        if child is None or receipt["direct_child_reaped"]:
            return
        end = time.monotonic()+5.
        if api is None or parent is None:
            cleanup_error(RuntimeError("NO_AUTHENTICATED_NATIVE_CLEANUP_ADAPTER"))
            return
        api.phase = "cleanup"  # Separate bounded reserve after a work-call cap.
        while time.monotonic() < end:
            try:
                if state is None:
                    api.begin_sample(end)
                    state = _Custody(api, child.pid, parent, new_session, receipt, stop)
                observed = sample(end)
                if observed["done"] and not observed["group"]:
                    reap_and_confirm(end, observed)
                    return
            except Exception as error:
                cleanup_error(error)
                if receipt["direct_child_reaped"]:
                    return  # Never use old group ownership after reap.
            if state is not None and not receipt["direct_child_reaped"]:
                try:
                    # Even after a failed census these fixed, already observed
                    # lifetimes need independent fresh native authorization.
                    api.begin_sample(end)
                    for detail in state.signal_current_owned():
                        cleanup_error(RuntimeError(detail))
                except Exception as error:
                    cleanup_error(error)
            time.sleep(min(.02, max(0., end-time.monotonic())))
        cleanup_error(TimeoutError("OWNED_PROCESS_CLEANUP_DEADLINE"))

    try:
        request_blob = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(request_blob) > 16384:
            raise ValueError("WORKER_DISPATCH_BYTE_BUDGET")
        _require(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL,
                 "SIGCHLD_MUST_RETAIN_UNREAPED_CHILD_ANCHOR")
        api = _MacOSProcessAPI()
        api.begin_sample(work_end)
        parent = api.inspect(os.getpid(), rss=True)
        _require(parent is not None and type(parent["rss"]) is int,
                 "SUPERVISOR_IDENTITY_OR_RSS_UNAVAILABLE")
        child = subprocess.Popen(command, cwd=cache, env=clean_environment(cache), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=new_session)
        receipt["child_pid"] = child.pid
        if new_session:
            receipt.update(owned_group_id=child.pid, owned_group_empty_confirmed=False)
        api.begin_sample(work_end)
        state = _Custody(api, child.pid, parent, new_session, receipt, stop)
        child.stdin.write(request_blob)
        child.stdin.close()
        selector.register(child.stdout, selectors.EVENT_READ)
        with log_path.open("xb") as log:
            receipt["log_created"] = True
            receipt["log_storage_status"] = "CREATED_NO_WRITE_ATTEMPTS"
            while True:
                left = work_end-time.monotonic()
                if left <= 0:
                    stop("WALL_LIMIT")
                    break
                for key, _ in selector.select(timeout=min(.05, left)):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        # Preserve V1's precharged, possibly uncertain log I/O
                        # counters. Flush is not a durability attestation.
                        receipt["forwarded_log_bytes"] += len(block)
                        keep = max(0, log_limit-receipt["log_write_attempted_bytes"])
                        retained = block[:keep]
                        receipt["log_write_attempted_bytes"] += len(retained)
                        try:
                            written = log.write(retained)
                            if type(written) is not int or not 0 <= written <= len(retained):
                                raise OSError("INVALID_LOG_WRITE_RESULT")
                            receipt["log_write_accepted_bytes"] += written
                            receipt["stored_log_bytes"] += written
                            if written != len(retained):
                                raise OSError("SHORT_LOG_WRITE")
                            log.flush()
                            receipt["log_storage_status"] = "WRITE_AND_FLUSH_RETURNED_NOT_DURABILITY_ATTESTATION"
                        except BaseException:
                            receipt["stored_log_bytes"] = None
                            receipt["log_storage_bytes_uncertain"] = True
                            receipt["log_storage_status"] = "WRITE_OR_FLUSH_FAILED_STORED_BYTE_COUNT_UNKNOWN"
                            raise
                if receipt["forwarded_log_bytes"] > log_limit:
                    stop("LOG_BYTE_LIMIT")
                observed = sample(work_end)
                if observed["done"] and (observed["group"] or observed["foreign"]):
                    stop("OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT")
                if receipt["budget_stop"]:
                    break
                if observed["done"] and not selector.get_map():
                    reap_and_confirm(work_end, observed)
                    break
    except Exception as error:
        receipt["supervisor_failure_reason"] = str(error)[:160]
        stop("SUPERVISOR_FAILURE:"+type(error).__name__)
    finally:
        try:
            if child is not None:
                # A failed initial dispatch must not strand a child waiting on
                # our own stdin. Closing it confers no process signal authority.
                if child.stdin is not None and not child.stdin.closed:
                    try:
                        child.stdin.close()
                    except Exception as error:
                        receipt["control_close_errors"].append("STDIN:"+type(error).__name__)
                        stop("SUPERVISOR_CONTROL_CLOSE_FAILURE")
                if not receipt["direct_child_reaped"]:
                    cleanup()
                receipt["exit_code"] = child.returncode
        finally:
            try:
                selector.close()
            except Exception as error:
                receipt["control_close_errors"].append("SELECTOR:"+type(error).__name__)
                stop("SUPERVISOR_CONTROL_CLOSE_FAILURE")
            if child is not None and child.stdout is not None:
                try:
                    child.stdout.close()
                except Exception as error:
                    receipt["control_close_errors"].append("STDOUT:"+type(error).__name__)
                    stop("SUPERVISOR_CONTROL_CLOSE_FAILURE")
            if state is not None:
                state.publish()
            if api is not None:
                receipt["native_process_work"] = dict(api.work)
                receipt["exit_transition_diagnostics"] = list(api.exit_transition_diagnostics)
                receipt["exit_transition_first_failure"] = api.exit_transition_first_failure
        receipt["requires_process_attention"] = bool(child is not None and (
            not receipt["direct_child_reaped"]
            or new_session and not receipt["owned_group_empty_confirmed"]
            or receipt["observed_foreign_group_escape_unconfirmed_after_cleanup"]
            or receipt["observed_identity_unknown_still_present"]))
        receipt["wall_seconds"] = time.monotonic()-started
        receipt["child_rss_observation_status"] = (
            "SAMPLED_NOT_CONTINUOUS_OR_OS_ALLOCATION_CAP" if receipt["child_tree_rss_sample_count"]
            else "UNOBSERVED_CHILD_TREE_PEAK_NOT_ZERO_REQUIRES_SEPARATE_HIGHWATER_EVIDENCE")
        if not receipt["child_tree_rss_sample_count"]:
            receipt["sampled_tree_peak_rss_bytes"] = None
        if receipt["budget_stop"] is None and receipt["wall_seconds"] > wall_seconds:
            stop("WALL_LIMIT_AFTER_FINAL_PROCESS_SAMPLE")
    return receipt
