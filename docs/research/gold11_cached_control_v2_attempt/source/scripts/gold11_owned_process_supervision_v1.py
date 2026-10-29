"""Bounded owned-process supervision; stdlib only, no repository/data access.

The caller verifies this source buffer, authorizes the command and supplies all
limits. Python/native execution and audit hooks are caller-owned. An outer
new-session child owns one process group: every sample includes that group even
after its leader exits, and successful return requires the group observed empty.
An inner child shares its coordinator's group: never kill that whole group or
mistake its unrelated members for this leaf's descendants. Inner custody covers
the direct child and observed descendants, not native create/reparent-before-
observation, hostile group escape or OS-level containment. Sampling is not an OS
allocation cap; PID/PGID observations are not cryptographic process identities.
"""
import json
import math
import os
import selectors
import signal
import subprocess
import time


def process_table(timeout=1):
    if type(timeout) not in (int, float) or not 0 < timeout <= 1:
        raise ValueError("INVALID_PROCESS_SAMPLE_TIMEOUT")
    ps = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
                        capture_output=True, text=True, timeout=timeout)
    if ps.returncode != 0 or not ps.stdout.strip():
        raise RuntimeError("RSS_MONITOR_UNAVAILABLE")
    table = {}
    for line in ps.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4 or not all(value.isdigit() for value in fields):
            raise RuntimeError("RSS_MONITOR_MALFORMED")
        pid, ppid, pgid, kib = map(int, fields)
        if pid in table:
            raise RuntimeError("RSS_MONITOR_DUPLICATE_PID")
        table[pid] = dict(ppid=ppid, pgid=pgid, rss=kib*1024)
    return table


def descendants(table, root_pid):
    found = {root_pid} if root_pid in table else set()
    while True:
        new = {pid for pid, row in table.items() if row["ppid"] in found} - found
        if not new:
            return found
        found.update(new)


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
    """One child, no retry. Failed cleanup/uncertain emptiness never succeeds.

    Work ends at the supplied wall limit. Failed-only kill/reap confirmation is
    bounded to five additional seconds, charged in returned wall time; callers
    must reserve it within their own absolute deadline. No cache is deleted.
    Popen, initial stdin delivery and synchronous log I/O rely on that caller
    deadline; this helper is not a self-contained hard-wall/OS supervisor.
    """
    if (type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or wall_seconds <= 0
            or any(type(value) is not int or value <= 0 for value in (rss_limit, log_limit))
            or parent_rss_limit is not None and (type(parent_rss_limit) is not int or parent_rss_limit <= 0)
            or type(new_session) is not bool):
        raise ValueError("INVALID_SUPERVISION_BUDGET")
    started = time.monotonic()
    work_end = started+wall_seconds
    child, selector = None, selectors.DefaultSelector()
    known = {}  # Retain observed PID/PGID membership across reparenting.
    identity_changed = set()
    foreign_seen = {}  # First observed foreign PGID; never rewrite identity.
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
        inner_custody_limit="DIRECT_CHILD_AND_OBSERVED_DESCENDANTS_NOT_UNOBSERVED_NATIVE_REPARENTING")

    def stop(reason):
        if receipt["budget_stop"] is None:
            receipt["budget_stop"] = reason

    def sample(end):
        left = end-time.monotonic()
        if left <= 0:
            raise TimeoutError("PROCESS_SAMPLE_DEADLINE")
        done = child.poll() is not None
        table = process_table(timeout=min(1, left))
        if not done and child.pid not in table:
            # A second poll may establish the ordinary exit-between-samples.
            if child.poll() is None:
                raise RuntimeError("ACTIVE_WORKER_RSS_MISSING")
            done = True
        rooted = descendants(table, child.pid)
        group = {pid for pid, row in table.items() if row["pgid"] == child.pid} if new_session else set()
        if child.poll() is not None:
            done = True
            # poll has reaped our direct child; a ps row captured immediately
            # before that exit is not evidence of an orphaned live leader.
            group.discard(child.pid)
        # Preserve the first identity. A known PID with another PGID may be an
        # escape or PID reuse: neither establishes that our old member is gone.
        # This applies to formerly OWNED as well as already-foreign members.
        changed = {pid for pid, pgid in known.items() if pid in table and table[pid]["pgid"] != pgid}
        identity_changed.update(changed)
        unknown_present = identity_changed.intersection(table)
        receipt["observed_process_identity_change_detected"] = bool(identity_changed)
        receipt["observed_identity_unknown_still_present"] = bool(unknown_present)
        receipt["observed_identity_unknown_pid_count_peak"] = max(
            receipt["observed_identity_unknown_pid_count_peak"], len(unknown_present))
        if changed:
            stop("OBSERVED_PROCESS_IDENTITY_CHANGED_OR_REUSED")
        if new_session and child.pid in table and table[child.pid]["pgid"] != child.pid:
            raise RuntimeError("OWNED_GROUP_LEADER_IDENTITY_MISMATCH")
        for pid in rooted | group:
            known.setdefault(pid, table[pid]["pgid"])
        retained = {pid for pid, pgid in known.items() if pid in table and table[pid]["pgid"] == pgid}
        if new_session:
            for pid in rooted | retained:
                if table[pid]["pgid"] != child.pid:
                    foreign_seen.setdefault(pid, table[pid]["pgid"])
        foreign_present = set(foreign_seen).intersection(table)
        # Include uncertain still-present rows conservatively in sampled RSS,
        # but never use that accounting inclusion as signal authority.
        measured_pids = rooted | group | retained | unknown_present | foreign_present
        parent_rss = table.get(os.getpid(), {}).get("rss")
        if parent_rss is None:
            raise RuntimeError("PARENT_RSS_MISSING")
        # Never double-count the outer PID if malformed/escaped ownership puts
        # it into a child set. Treat that as a custody failure, not a smaller sum.
        if os.getpid() in measured_pids:
            raise RuntimeError("SUPERVISOR_PRESENT_IN_CHILD_CUSTODY")
        rss = sum(table[pid]["rss"] for pid in measured_pids)
        if child.pid in table:
            receipt["child_rss_sample_count"] += 1
        if measured_pids:
            receipt["child_tree_rss_sample_count"] += 1
        receipt["sampled_tree_peak_rss_bytes"] = max(receipt["sampled_tree_peak_rss_bytes"], rss)
        receipt["sampled_parent_peak_rss_bytes"] = max(receipt["sampled_parent_peak_rss_bytes"], parent_rss)
        measured, reason = memory_budget(rss, parent_rss, rss_limit,
            include_parent=new_session, parent_rss_limit=parent_rss_limit)
        receipt["sampled_budget_peak_rss_bytes"] = max(receipt["sampled_budget_peak_rss_bytes"], measured)
        if reason:
            stop(reason)
        if new_session:
            receipt["owned_group_empty_confirmed"] = not group and done
            receipt["observed_foreign_pid_count_peak"] = max(receipt["observed_foreign_pid_count_peak"], len(foreign_present))
            receipt["observed_foreign_group_escape_unconfirmed_after_cleanup"] = bool(foreign_present)
            if any(table[pid]["pgid"] != foreign_seen[pid] for pid in foreign_present):
                receipt["observed_foreign_identity_changed_or_unknown"] = True
            if foreign_present:
                receipt["observed_foreign_group_escape_detected"] = True
                stop("OBSERVED_DESCENDANT_OUTSIDE_OWNED_GROUP")
        return done, group, retained

    def cleanup():
        if child is None:
            return
        end = time.monotonic()+5
        try:
            if new_session:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                # Only known leaf PIDs; the coordinator and sibling leaves share
                # this PGID and must not be signaled by a leaf failure.
                for pid in sorted(set(known) | {child.pid}, reverse=True):
                    try:
                        expected = known.get(pid, os.getpgrp())
                        if os.getpgid(pid) == expected == os.getpgrp():
                            os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            child.wait(timeout=max(.001, end-time.monotonic()))
            receipt["direct_child_reaped"] = True
            while True:
                _, group, retained = sample(end)
                outstanding = group if new_session else retained-{child.pid}
                if not outstanding:
                    # Only the owned PGID is killed/confirmed. A detected foreign
                    # escape keeps its separate outstanding flag and failure;
                    # group_empty must never be read as all descendants removed.
                    return
                if time.monotonic() >= end:
                    raise TimeoutError("OWNED_PROCESS_CLEANUP_DEADLINE")
                time.sleep(min(.01, max(0, end-time.monotonic())))
        except Exception as error:
            receipt["cleanup_errors"].append(type(error).__name__)
            if new_session:
                receipt["owned_group_empty_confirmed"] = False
            stop("OWNED_PROCESS_CLEANUP_UNCONFIRMED")

    try:
        request_blob = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(request_blob) > 16384:
            raise ValueError("WORKER_DISPATCH_BYTE_BUDGET")
        child = subprocess.Popen(command, cwd=cache, env=clean_environment(cache), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=new_session)
        receipt["child_pid"] = child.pid
        if new_session:
            receipt.update(owned_group_id=child.pid, owned_group_empty_confirmed=False)
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
                        # Observed bytes are charged before any fallible write.
                        # Accepted write bytes are not yet a flushed/durable
                        # file-length observation; a failed/short write or flush
                        # makes stored length unknown rather than exact zero.
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
                done, group, retained = sample(work_end)
                if done and (group if new_session else retained-{child.pid}):
                    stop("OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT")
                if receipt["budget_stop"]:
                    break
                if done and not selector.get_map():
                    receipt["direct_child_reaped"] = True
                    break
    except Exception as error:
        stop("SUPERVISOR_FAILURE:"+type(error).__name__)
    finally:
        try:
            if child is not None:
                try:
                    running = child.poll() is None
                except Exception as error:
                    running = True
                    stop("FINAL_CHILD_POLL_FAILURE:"+type(error).__name__)
                if receipt["budget_stop"] or running or (new_session and not receipt["owned_group_empty_confirmed"]):
                    cleanup()
                receipt["exit_code"] = child.returncode
        finally:
            # Control-resource failures cannot bypass owned-process cleanup.
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
        receipt["wall_seconds"] = time.monotonic()-started
        receipt["child_rss_observation_status"] = (
            "SAMPLED_NOT_CONTINUOUS_OR_OS_ALLOCATION_CAP" if receipt["child_tree_rss_sample_count"]
            else "UNOBSERVED_CHILD_TREE_PEAK_NOT_ZERO_REQUIRES_SEPARATE_HIGHWATER_EVIDENCE")
        if not receipt["child_tree_rss_sample_count"]:
            receipt["sampled_tree_peak_rss_bytes"] = None
        if receipt["budget_stop"] is None and receipt["wall_seconds"] > wall_seconds:
            stop("WALL_LIMIT_AFTER_FINAL_PROCESS_SAMPLE")
    return receipt
