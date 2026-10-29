"""Independent, prospective cached-runner custody controls; invented inputs only.

No runner module top-level, real BINDINGS, cached prediction, catalog, Gold or
native geometry is executed/read. Named function ASTs run in an invented namespace.
The one opt-in fresh stdlib child tests actual audit delivery, not physical access.
Root owns first execution:60 monitored-wall/45 worker-CPU/1GiB group/1MiB stdout.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"scripts/run_gold11_cached_control_v1.py"
SUPPORT_SOURCE = ROOT/"scripts/run_gold11_mesh_development_v2.py"
RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
    process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
SMOKE_OPT_IN = "SHERMAN_RUN_CACHED_CONTROL_GUARD_SMOKE_V2"
CHILD_LIMITS = dict(wall_seconds=8, cpu_seconds=5, output_bytes=65536)
FUNCTIONS = ("digest", "encoded", "decode", "read_fixed", "publish_bytes", "publish_json",
    "support_module", "authorize_worker", "install_input_guard", "scientific_artifact", "selected_references",
    "unavailable_cases", "verify_retained_artifacts", "audit_final_sources",
    "final_budget_evidence", "budgets_met", "retained_scientific_usage", "execute")


def _definitions(path, names):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names) and {node.name for node in selected} == set(names)
    return ast.Module(body=selected, type_ignores=[])


@pytest.fixture
def runner(tmp_path):
    module = types.ModuleType("_invented_cached_control_functions")
    module.__dict__.update(ast=ast, hashlib=hashlib, json=json, os=os, Path=Path,
        resource=resource, stat=stat, sys=sys, importlib=types.SimpleNamespace(util=importlib.util),
        secrets=types.SimpleNamespace(token_hex=lambda n: "a"*(2*n)),
        ROOT=tmp_path/"invented_root", RUN=tmp_path/"invented_attempt",
        RUNNER="source_runner.py", SUPPORT="support.py", SUPPORT_SHA="a"*64,
        CATALOG="fixed_catalog.json", CATALOG_SHA="b"*64, SELECTED_SHA="c"*64,
        SOURCES=("source_a.py", "source_b.py"),
        VARIANTS=("scan_only_a0", "cad_only_gated", "intersection", "union"))
    tree = ast.parse(SOURCE.read_text())
    limits, = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "LIMITS" for target in node.targets)]
    exec(compile(ast.Module(body=[limits], type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    exec(compile(_definitions(SOURCE, FUNCTIONS), str(SOURCE), "exec"), module.__dict__)
    return module


class _DescriptorOS:
    O_RDONLY, O_DIRECTORY, O_NOFOLLOW = os.O_RDONLY, os.O_DIRECTORY, os.O_NOFOLLOW
    O_NONBLOCK, O_CLOEXEC = os.O_NONBLOCK, os.O_CLOEXEC
    def __init__(self, blob=b"abc", *, mode=stat.S_IFREG, size=None, drift=None, fail_open=None):
        self.blob, self.offset = blob, 0
        self.mode, self.size = mode, len(blob) if size is None else size
        self.drift, self.fail_open = drift, fail_open
        self.opens, self.closed, self.read_sizes, self.stat_calls = [], [], [], 0
    def open(self, path, flags, **kwargs):
        self.opens.append((path, flags, kwargs))
        if len(self.opens) == self.fail_open:
            raise OSError("INVENTED_OPEN_FAILURE")
        return 100+len(self.opens)
    def close(self, fd):
        self.closed.append(fd)
    def fstat(self, fd):
        self.stat_calls += 1
        values = dict(st_mode=self.mode, st_dev=7, st_ino=8, st_size=self.size,
            st_mtime_ns=9, st_ctime_ns=10)
        if self.stat_calls == 2 and self.drift:
            values[self.drift] += 1
        return types.SimpleNamespace(**values)
    def read(self, fd, count):
        self.read_sizes.append(count)
        result = self.blob[self.offset:self.offset+count]
        self.offset += len(result)
        return result


@pytest.mark.parametrize("path", (None, Path("a"), "", "/absolute", "a/../b", "a//b", "./a", "a/"))
def test_noncanonical_relative_path_refuses_before_any_descriptor_open(runner, monkeypatch, path):
    machine = _DescriptorOS()
    monkeypatch.setattr(runner, "os", machine)
    with pytest.raises(ValueError, match="FIXED_RELATIVE_PATH_REQUIRED"):
        runner.read_fixed(runner.ROOT, path, 10)
    assert machine.opens == []


def test_descriptor_relative_nofollow_components_bounded_reads_and_exact_hash(runner, monkeypatch):
    machine = _DescriptorOS()
    monkeypatch.setattr(runner, "os", machine)
    assert runner.read_fixed(runner.ROOT, "a/b/c.json", 3, runner.digest(b"abc")) == b"abc"
    assert [row[0] for row in machine.opens] == [runner.ROOT, "a", "b", "c.json"]
    for index, (_, flags, kwargs) in enumerate(machine.opens):
        assert flags & os.O_NOFOLLOW
        assert bool(flags & os.O_DIRECTORY) == (index < 3)
        assert kwargs == ({} if index == 0 else {"dir_fd": 100+index})
    assert machine.opens[-1][1] & os.O_NONBLOCK
    assert machine.closed == [101, 102, 104, 103]
    assert machine.read_sizes == [4, 1]


@pytest.mark.parametrize("allow", (False, True))
def test_empty_read_is_only_allowed_by_explicit_option(runner, monkeypatch, allow):
    machine = _DescriptorOS(b"")
    monkeypatch.setattr(runner, "os", machine)
    if allow:
        assert runner.read_fixed(runner.ROOT, "empty", 0, allow_empty=True) == b""
    else:
        with pytest.raises(ValueError, match="NONREGULAR_OR_BYTE_BUDGET"):
            runner.read_fixed(runner.ROOT, "empty", 0)
    assert machine.closed == [102, 101]


@pytest.mark.parametrize("mode", (stat.S_IFDIR, stat.S_IFIFO, stat.S_IFLNK))
def test_nonregular_leaf_is_never_read(runner, monkeypatch, mode):
    machine = _DescriptorOS(mode=mode)
    monkeypatch.setattr(runner, "os", machine)
    with pytest.raises(ValueError, match="NONREGULAR_OR_BYTE_BUDGET"):
        runner.read_fixed(runner.ROOT, "leaf", 3)
    assert machine.read_sizes == [] and machine.closed == [102, 101]


@pytest.mark.parametrize("kind", ("stated_size", "growing_stream"))
def test_size_and_live_stream_overflow_both_refuse(runner, monkeypatch, kind):
    machine = _DescriptorOS(b"abcd", size=4 if kind == "stated_size" else 3)
    monkeypatch.setattr(runner, "os", machine)
    with pytest.raises(ValueError, match="BYTE_BUDGET"):
        runner.read_fixed(runner.ROOT, "leaf", 3)
    assert machine.closed == [102, 101]
    assert machine.offset == (0 if kind == "stated_size" else 4)


@pytest.mark.parametrize("field", ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
def test_every_descriptor_identity_field_is_rechecked_after_read(runner, monkeypatch, field):
    machine = _DescriptorOS(drift=field)
    monkeypatch.setattr(runner, "os", machine)
    with pytest.raises(ValueError, match="SOURCE_CHANGED_DURING_READ"):
        runner.read_fixed(runner.ROOT, "leaf", 3)
    assert machine.stat_calls == 2 and machine.closed == [102, 101]


def test_wrong_declared_hash_refuses_complete_bytes(runner, monkeypatch):
    machine = _DescriptorOS()
    monkeypatch.setattr(runner, "os", machine)
    with pytest.raises(ValueError, match="DECLARED_HASH_MISMATCH"):
        runner.read_fixed(runner.ROOT, "leaf", 3, "0"*64)
    assert machine.closed == [102, 101]


@pytest.mark.parametrize("failure", (2, 3))
def test_failed_component_open_closes_only_owned_live_descriptors(runner, monkeypatch, failure):
    machine = _DescriptorOS(fail_open=failure)
    monkeypatch.setattr(runner, "os", machine)
    with pytest.raises(OSError):
        runner.read_fixed(runner.ROOT, "a/leaf", 3)
    assert machine.closed == ([101] if failure == 2 else [101, 102])


def _guard_fixture(runner, monkeypatch, *, event_change=None, deliver=True, nested=False):
    hooks, base_events, closed, opened = [], [], [], []
    def base(event, args):
        base_events.append((event, args))
        raise PermissionError("BASE_DENIAL")
    def fake_open(path, flags, **kwargs):
        opened.append((path, flags, kwargs))
        args = (os.fsdecode(path), None, flags | os.O_CLOEXEC)
        if event_change:
            args = event_change(args)
        if deliver:
            hooks[0]("open", args)
        if nested:
            with pytest.raises(PermissionError, match="BASE_DENIAL"):
                hooks[0]("open", args)
        return 77
    fake_os = types.SimpleNamespace(open=fake_open, close=closed.append, fsdecode=os.fsdecode,
        O_CLOEXEC=os.O_CLOEXEC, O_RDONLY=os.O_RDONLY)
    monkeypatch.setattr(runner, "os", fake_os)
    monkeypatch.setattr(runner, "sys", types.SimpleNamespace(addaudithook=hooks.append))
    def read(root, relative, cap, expected, *, opener):
        assert root == runner.ROOT
        return opener(relative, os.O_RDONLY, dir_fd=31)
    monkeypatch.setattr(runner, "read_fixed", read)
    support = types.SimpleNamespace(worker_audit_hook=lambda *args: base)
    read, returned = runner.install_input_guard(support, runner.RUN/"work",
        [dict(a0_path="owned/a.json", cad_path="owned/c.json")])
    assert returned is base and len(hooks) == 1
    return read, hooks[0], opened, base_events, closed


@pytest.mark.parametrize("relative", ("owned/a.json", "owned/c.json", "fixed_catalog.json"))
def test_exact_input_component_grant_is_delivered_once(runner, monkeypatch, relative):
    read, guard, opened, base, closed = _guard_fixture(runner, monkeypatch)
    assert read(relative, 10, "hash") == 77
    assert opened == [(relative, os.O_RDONLY, {"dir_fd": 31})] and base == [] and closed == []
    with pytest.raises(PermissionError, match="BASE_DENIAL"):
        guard("open", (relative, None, os.O_RDONLY))


@pytest.mark.parametrize("relative", ("another.json", "owned/other.json", "../owned/a.json", None))
def test_unlisted_input_has_no_path_or_component_grant(runner, monkeypatch, relative):
    read, _, opened, _, _ = _guard_fixture(runner, monkeypatch)
    with pytest.raises(PermissionError, match="UNDECLARED_INPUT_PATH"):
        read(relative, 10, "hash")
    assert opened == []


def test_nested_audit_open_cannot_borrow_consumed_component_grant(runner, monkeypatch):
    read, _, _, base, _ = _guard_fixture(runner, monkeypatch, nested=True)
    assert read("owned/a.json", 10, "hash") == 77
    assert len(base) == 1


@pytest.mark.parametrize("fault", ("path", "mode", "flags"))
def test_component_audit_mismatch_clears_grant_before_later_events(runner, monkeypatch, fault):
    def changed(args):
        path, mode, flags = args
        return (("wrong" if fault == "path" else path), ("r" if fault == "mode" else mode),
                flags | os.O_WRONLY if fault == "flags" else flags)
    read, guard, _, base, _ = _guard_fixture(runner, monkeypatch, event_change=changed)
    with pytest.raises(PermissionError, match="FIXED_INPUT_COMPONENT_GRANT_MISMATCH"):
        read("owned/a.json", 10, "hash")
    with pytest.raises(PermissionError, match="BASE_DENIAL"):
        guard("open", ("owned/a.json", None, os.O_RDONLY))
    assert len(base) == 1


def test_missing_actual_audit_delivery_closes_fd_and_refuses(runner, monkeypatch):
    read, _, _, _, closed = _guard_fixture(runner, monkeypatch, deliver=False)
    with pytest.raises(PermissionError, match="FIXED_INPUT_AUDIT_GRANT_NOT_CONSUMED"):
        read("owned/a.json", 10, "hash")
    assert closed == [77]


@pytest.mark.parametrize("frozen", (False, True))
def test_support_executes_verified_buffer_once_and_never_loader_reopens(runner, monkeypatch, frozen):
    blob = b"verified_buffer_marker = 731\n"
    reads, loaders = [], []
    def read(*args):
        reads.append(args)
        return blob
    class Loader:
        def exec_module(self, module):
            loaders.append(True)
            pytest.fail("Verified source must not be reopened through the loader")
    utility = types.SimpleNamespace(
        spec_from_file_location=lambda *args: types.SimpleNamespace(loader=Loader()),
        module_from_spec=lambda spec: types.ModuleType("invented_support"))
    monkeypatch.setattr(runner, "read_fixed", read)
    monkeypatch.setattr(runner, "importlib", types.SimpleNamespace(util=utility))
    monkeypatch.setattr(runner, "sys", types.SimpleNamespace(dont_write_bytecode=False))
    result = runner.support_module(frozen=frozen)
    assert result.verified_buffer_marker == 731 and loaders == []
    assert reads == [(runner.RUN/"source" if frozen else runner.ROOT,
                      runner.SUPPORT, runner.LIMITS["source_file_bytes"], runner.SUPPORT_SHA)]


@pytest.mark.parametrize("over", (False, True))
def test_scientific_artifact_precharges_exact_limit_before_publication(runner, monkeypatch, tmp_path, over):
    maximum = runner.LIMITS["scientific_artifact_bytes"]
    budget, observed = {"bytes": maximum-3+int(over)}, []
    monkeypatch.setattr(runner, "publish_bytes", lambda path, blob: observed.append((budget["bytes"], path, blob)))
    if over:
        with pytest.raises(ValueError, match="AGGREGATE_SCIENTIFIC_ARTIFACT_BYTE_BUDGET"):
            runner.scientific_artifact(tmp_path, "fixed.json", b"abc", budget)
        assert observed == [] and budget["bytes"] == maximum-2
    else:
        runner.scientific_artifact(tmp_path, "fixed.json", b"abc", budget)
        assert observed == [(maximum, tmp_path/"fixed.json", b"abc")]


@pytest.mark.parametrize("failure", (OSError, PermissionError, FileExistsError))
def test_failed_scientific_publication_keeps_full_attempted_charge(runner, monkeypatch, tmp_path, failure):
    budget = {"bytes": 7}
    def fail(path, blob):
        assert budget["bytes"] == 10
        raise failure("INVENTED_PARTIAL_PUBLICATION")
    monkeypatch.setattr(runner, "publish_bytes", fail)
    with pytest.raises(failure):
        runner.scientific_artifact(tmp_path, "nested/fixed.json", b"abc", budget)
    assert budget["bytes"] == 10


def _invented_cases(runner):
    return [dict(case_id=f"invented_{i}", family_id=f"family_{i%6}", scan_sha256=f"{i+1:064x}",
        reference_id=f"reference_{i}", reference_scan_path=f"not_opened/scan_{i}",
        a0_path=f"original_a/{i}.json", cad_path=f"original_c/{i}.json") for i in range(11)]


def _reference_fixture(runner):
    cases = _invented_cases(runner)
    selected = {case["case_id"]: dict(case_id=case["reference_id"], scan_sha256=case["scan_sha256"],
        scan_path=case["reference_scan_path"], part_id=case["family_id"],
        bends=[{"invented_reference_row": j} for j in range(6 if i < 6 else 7)])
        for i, case in enumerate(cases)}
    # These 71 marker objects test selection custody only, not bend geometry.
    runner.SELECTED_SHA = runner.digest(runner.encoded(selected))
    return cases, selected


def test_fixed_reference_selection_uses_exact_identity_and_ignores_unselected_fields(runner):
    cases, selected = _reference_fixture(runner)
    catalog = dict(cases=[dict(scan_sha256="unselected", bends="DO_NOT_INTERPRET", scan_path="DO_NOT_OPEN"),
                          *reversed(list(selected.values()))])
    answer = runner.selected_references(runner.encoded(catalog), cases)
    assert answer == selected and sum(len(row["bends"]) for row in answer.values()) == 71


@pytest.mark.parametrize("fault", ("duplicate", "missing", "reference_id", "scan_path", "family", "count", "hash"))
def test_selected_reference_faults_cannot_shrink_or_change_the_fixed_denominator(runner, fault):
    cases, selected = _reference_fixture(runner)
    rows = deepcopy(list(selected.values()))
    if fault == "duplicate": rows.append(deepcopy(rows[0]))
    elif fault == "missing": rows.pop()
    elif fault == "reference_id": rows[-1]["case_id"] = "wrong"
    elif fault == "scan_path": rows[-1]["scan_path"] = "wrong"
    elif fault == "family": rows[-1]["part_id"] = "wrong"
    elif fault == "count": rows[-1]["bends"].pop()
    else: rows[-1]["bends"][-1]["invented_reference_row"] = -1
    with pytest.raises(ValueError):
        runner.selected_references(runner.encoded(dict(cases=rows)), cases)


def _retained_fixture(runner, monkeypatch):
    cases, selected = _reference_fixture(runner)
    blobs, checks, reads = {}, [], []
    for case in cases:
        for role in ("a0", "cad"):
            blob = runner.encoded(dict(invented_case=case["case_id"], role=role))
            case[role+"_sha256"] = runner.digest(blob)
            snapshot = f"inputs/{role}/{case['case_id']}.json"
            blobs[snapshot] = blob
            check = dict(case_id=case["case_id"], role=role, path=case[role+"_path"],
                expected_sha256=case[role+"_sha256"], scan_sha256=case["scan_sha256"],
                status="BYTE_VERIFIED_AND_SNAPSHOTTED", bytes=len(blob), snapshot=snapshot)
            checks.append(check)
            blobs[f"checks/{case['case_id']}_{role}.json"] = runner.encoded(check)
    bundle = dict(cases=cases)
    blobs["SELECTED_REFERENCES.json"] = runner.encoded(selected)
    ledger = dict(status="ALL22_CACHED_BYTES_AND_EXACT71_REFERENCES_VERIFIED", checks=checks,
        catalog_path=runner.CATALOG, catalog_sha256=runner.CATALOG_SHA, selected_reference_sha256=runner.SELECTED_SHA,
        case_count=11, family_count=6, reference_line_count=71,
        raw_scan_or_cad_opened=False, historical_reference_identity="UNESTABLISHED")
    blobs["INPUT_PREFLIGHT.json"] = runner.encoded(ledger)
    report = dict(schema="GOLD11_CACHED_FUSION_GEOMETRY_DIAGNOSTIC_V1", status="COMPUTED_DEVELOPMENT_DIAGNOSTIC",
        case_count=11, family_count=6, reference_line_count=71,
        variants={variant: dict(case_dispositions={case["case_id"]: {"family_id": case["family_id"]} for case in cases},
            families={case["family_id"]: {} for case in cases}, geometry={"case_count": 11, "reference_line_count": 71})
            for variant in runner.VARIANTS},
        reconstruction=dict(schema="GOLD11_RETAINED_FUSION_RECONSTRUCTION_V1", cases={case["case_id"]:
            {"variants": {variant: [] for variant in runner.VARIANTS}} for case in cases}),
        runner_provenance=dict(bundle_sha256=runner.digest(runner.encoded(bundle)),
            input_preflight_sha256=runner.digest(blobs["INPUT_PREFLIGHT.json"]),
            selected_reference_sha256=runner.SELECTED_SHA,
            case_scan_sha256={case["case_id"]: case["scan_sha256"] for case in cases}),
        inference_execution="NOT_PERFORMED", variant_selection="NONE_ALL_FOUR_REPORTED",
        observed_bend_count=None, production_use_allowed=False, historical_score_comparability="UNESTABLISHED")
    receipt = dict(status="COMPUTED_CACHED_CONTROL_NOT_INFERENCE", case_count=11, variant_count=4,
        worker_cpu_seconds=.01, worker_peak_rss_bytes=1024, observed_bend_count=None, production_use_allowed=False)
    def refresh():
        blobs["SCORER_REPORT.json"] = runner.encoded(report)
        receipt.update(report_sha256=runner.digest(blobs["SCORER_REPORT.json"]),
            scientific_artifact_bytes=sum(map(len, blobs.values())))
    refresh()
    def read(root, relative, cap, expected=None, **kwargs):
        assert root == runner.RUN/"work", "Verification must never reopen originals"
        reads.append((relative, cap, expected))
        blob = blobs[relative]
        assert len(blob) <= cap
        if expected is not None and runner.digest(blob) != expected:
            raise ValueError("DECLARED_HASH_MISMATCH")
        return blob
    monkeypatch.setattr(runner, "read_fixed", read)
    return bundle, receipt, blobs, ledger, report, reads, refresh


def test_retained_verification_binds_all22_owned_inputs_and_full_four_variant_report(runner, monkeypatch):
    bundle, receipt, blobs, _, _, reads, _ = _retained_fixture(runner, monkeypatch)
    result = runner.verify_retained_artifacts(bundle, receipt)
    assert result["report_sha256"] == receipt["report_sha256"]
    assert result["scientific_artifact_bytes"] == sum(map(len, blobs.values()))
    assert len(reads) == 47 and len({row[0] for row in reads}) == 47
    assert sum(relative.startswith("inputs/") for relative, _, _ in reads) == 22
    assert reads[-1][0] == "SCORER_REPORT.json" and reads[-1][2] == receipt["report_sha256"]


@pytest.mark.parametrize("fault", ("hash_missing", "hash_none", "hash_dict", "hash_upper", "status", "case_bool",
    "variant_count", "byte_bool", "cpu_nan", "cpu_over", "rss_over", "observed", "production"))
def test_malformed_worker_receipt_refuses_before_any_owned_read(runner, monkeypatch, fault):
    bundle, receipt, _, _, _, reads, _ = _retained_fixture(runner, monkeypatch)
    if fault == "hash_missing": receipt.pop("report_sha256")
    elif fault == "hash_none": receipt["report_sha256"] = None
    elif fault == "hash_dict": receipt["report_sha256"] = {"hash": "f"*64}
    elif fault == "hash_upper": receipt["report_sha256"] = "F"*64
    elif fault == "status": receipt["status"] = "COMPLETE"
    elif fault == "case_bool": receipt["case_count"] = True
    elif fault == "variant_count": receipt["variant_count"] = 3
    elif fault == "byte_bool": receipt["scientific_artifact_bytes"] = True
    elif fault == "cpu_nan": receipt["worker_cpu_seconds"] = float("nan")
    elif fault == "cpu_over": receipt["worker_cpu_seconds"] = runner.LIMITS["worker_cpu_seconds"]+1
    elif fault == "rss_over": receipt["worker_peak_rss_bytes"] = runner.LIMITS["global_tree_rss_bytes"]+1
    elif fault == "observed": receipt["observed_bend_count"] = 0
    else: receipt["production_use_allowed"] = True
    with pytest.raises(ValueError, match="WORKER_COMPLETION_CONTRACT"):
        runner.verify_retained_artifacts(bundle, receipt)
    assert reads == []


@pytest.mark.parametrize("fault", ("late_snapshot", "late_check", "selected", "ledger", "report_hash", "byte_accounting"))
def test_late_retained_corruption_never_returns_a_verified_prefix(runner, monkeypatch, fault):
    bundle, receipt, blobs, ledger, _, reads, _ = _retained_fixture(runner, monkeypatch)
    last = bundle["cases"][-1]["case_id"]
    if fault == "late_snapshot": blobs[f"inputs/cad/{last}.json"] += b" "
    elif fault == "late_check": blobs[f"checks/{last}_cad.json"] += b" "
    elif fault == "selected": blobs["SELECTED_REFERENCES.json"] += b" "
    elif fault == "ledger":
        ledger["checks"][-1]["status"] = "NOT_CHECKED"
        blobs["INPUT_PREFLIGHT.json"] = runner.encoded(ledger)
    elif fault == "report_hash": blobs["SCORER_REPORT.json"] += b" "
    else: receipt["scientific_artifact_bytes"] -= 1
    with pytest.raises(ValueError):
        runner.verify_retained_artifacts(bundle, receipt)
    assert len(reads) >= 43


@pytest.mark.parametrize("fault", ("lane_missing", "case_missing", "family_missing", "wrong_family", "lane_reference_count",
    "reconstruction_missing", "reconstruction_variant", "bundle", "selected", "preflight", "inference", "selection", "comparability"))
def test_hash_bound_but_semantically_incomplete_report_is_not_accepted(runner, monkeypatch, fault):
    bundle, receipt, _, _, report, reads, refresh = _retained_fixture(runner, monkeypatch)
    variant, case = runner.VARIANTS[-1], bundle["cases"][-1]["case_id"]
    if fault == "lane_missing": report["variants"].pop(variant)
    elif fault == "case_missing": report["variants"][variant]["case_dispositions"].pop(case)
    elif fault == "family_missing": report["variants"][variant]["families"].pop("family_5")
    elif fault == "wrong_family": report["variants"][variant]["case_dispositions"][case]["family_id"] = "wrong"
    elif fault == "lane_reference_count": report["variants"][variant]["geometry"]["reference_line_count"] = 70
    elif fault == "reconstruction_missing": report.pop("reconstruction")
    elif fault == "reconstruction_variant": report["reconstruction"]["cases"][case]["variants"].pop(variant)
    elif fault in ("bundle", "selected", "preflight"):
        key = {"bundle": "bundle_sha256", "selected": "selected_reference_sha256", "preflight": "input_preflight_sha256"}[fault]
        report["runner_provenance"][key] = "0"*64
    elif fault == "inference": report["inference_execution"] = "RERUN"
    elif fault == "selection": report["variant_selection"] = "BEST_F1"
    else: report["historical_score_comparability"] = "ESTABLISHED"
    refresh()  # Correct content hash and exact bytes cannot replace semantic checks.
    with pytest.raises(ValueError):
        runner.verify_retained_artifacts(bundle, receipt)
    assert len(reads) == 47


@pytest.mark.parametrize("fault", (None, "current_drift", "frozen_error", "timeout"))
def test_all_source_paths_are_audited_with_partial_error_evidence(runner, monkeypatch, fault):
    blobs = {name: name.encode() for name in runner.SOURCES}
    bundle = dict(source_sha256={name: runner.digest(blob) for name, blob in blobs.items()}, runtime_metadata={"invented": 1})
    audit, seen = {"independent": True}, []
    def read(base, name, cap, expected=None):
        seen.append((base, name))
        if name == "FROZEN.json": return runner.encoded(bundle)
        if name == "PRE_RUN_AUDIT.json": return runner.encoded(audit)
        if fault == "timeout" and base == runner.ROOT: raise TimeoutError("STOP")
        if fault == "frozen_error" and base == runner.RUN/"source" and name == runner.SOURCES[0]: raise OSError("STOP")
        if fault == "current_drift" and base == runner.ROOT and name == runner.SOURCES[0]: return b"changed"
        return blobs[name]
    monkeypatch.setattr(runner, "read_fixed", read)
    monkeypatch.setattr(runner, "runtime_metadata", lambda support: bundle["runtime_metadata"], raising=False)
    evidence = {}
    if fault == "timeout":
        with pytest.raises(TimeoutError): runner.audit_final_sources(bundle, audit, object(), evidence)
        assert evidence["complete"] is evidence["unchanged"] is False
    else:
        runner.audit_final_sources(bundle, audit, object(), evidence)
        assert evidence["complete"] is True and evidence["unchanged"] is (fault is None)
        assert len(seen) == 6
        if fault == "frozen_error":
            assert evidence["errors"] == {"frozen:source_a.py": "OSError"}
            assert "source_b.py" in evidence["frozen"]


class _Entries:
    def __init__(self, rows): self.rows = iter(rows)
    def __enter__(self): return self
    def __iter__(self): return self
    def __next__(self): return next(self.rows)
    def __exit__(self, *args):
        self.rows = iter(())
        return False


def _entry(name, size, mode=stat.S_IFREG):
    def info(*, follow_symlinks):
        assert follow_symlinks is False
        return types.SimpleNamespace(st_mode=mode, st_size=size)
    return types.SimpleNamespace(name=name, stat=info)


@pytest.mark.parametrize("fault", (None, "unknown_directory", "symlink", "entries"))
def test_retained_usage_counts_partials_without_traversing_unlisted_directories(runner, monkeypatch, fault):
    rows = {
        "": [_entry("inputs", 0, stat.S_IFDIR), _entry("checks", 0, stat.S_IFDIR),
             _entry("LIVE_OUTPUT.txt", 999), _entry(".WORKER_COMPLETED.json.partial-x", 999),
             _entry("SCORER_REPORT.json", 7), _entry(".SCORER_REPORT.json.partial-x", 7)],
        "inputs": [_entry("a0", 0, stat.S_IFDIR), _entry("cad", 0, stat.S_IFDIR)],
        "inputs/a0": [_entry("invented.json", 3)], "inputs/cad": [], "checks": [_entry("check.json", 5)]}
    if fault == "unknown_directory": rows[""] += [_entry("undeclared", 0, stat.S_IFDIR)]
    if fault == "symlink": rows["inputs/a0"] = [_entry("link", 3, stat.S_IFLNK)]
    if fault == "entries": rows["inputs/cad"] = [_entry(f"invented_{i}.json", 1) for i in range(129)]
    visited = []
    def scan(path):
        relative = str(path.relative_to(runner.RUN/"work"))
        relative = "" if relative == "." else relative
        visited.append(relative)
        return _Entries(rows[relative])
    monkeypatch.setattr(runner, "os", types.SimpleNamespace(scandir=scan))
    if fault:
        with pytest.raises(ValueError): runner.retained_scientific_usage(runner.RUN/"work")
    else:
        result = runner.retained_scientific_usage(runner.RUN/"work")
        assert result["logical_bytes_including_partials"] == 22
        assert result["partial_hardlinks_counted_conservatively_twice"] is True
        assert len(result["files"]) == 4
        assert visited == ["", "inputs", "inputs/a0", "inputs/cad", "checks"]
    assert set(visited) <= {"", "inputs", "inputs/a0", "inputs/cad", "checks"}


def test_retained_usage_missing_owned_directories_is_partial_evidence_not_zero_score(runner, monkeypatch):
    def absent(path): raise FileNotFoundError("INVENTED_ABSENT")
    monkeypatch.setattr(runner, "os", types.SimpleNamespace(scandir=absent))
    assert runner.retained_scientific_usage(runner.RUN/"work")["logical_bytes_including_partials"] == 0
    # A zero byte count is not a verified report; execute separately checks that.


_BUDGET_FLAGS = ("wall_budget_met", "cpu_budget_met", "parent_rss_budget_met", "reaped_child_rss_budget_met")


@pytest.mark.parametrize("over", (None, "wall", "cpu", "parent", "children"))
def test_final_resource_observations_include_reaped_child_highwater_and_exact_caps(runner, monkeypatch, over):
    values = dict(wall=90, cpu=60, parent=1024**3, children=1024**3)
    if over: values[over] += 1
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(monotonic=lambda: 100+values["wall"]), raising=False)
    monkeypatch.setattr(runner, "aggregate_cpu", lambda: 200+values["cpu"], raising=False)
    monkeypatch.setattr(runner, "resource", types.SimpleNamespace(RUSAGE_SELF=0, RUSAGE_CHILDREN=1,
        getrusage=lambda who: types.SimpleNamespace(ru_maxrss=values["parent" if who == 0 else "children"])))
    result = runner.final_budget_evidence(100, 200)
    assert result["parent_wall_seconds"] == values["wall"]
    assert result["aggregate_cpu_seconds"] == values["cpu"]
    assert result["parent_peak_rss_bytes"] == values["parent"]
    assert result["reaped_children_peak_rss_bytes"] == values["children"]
    assert runner.budgets_met(result) is (over is None)


@pytest.mark.parametrize("flag", _BUDGET_FLAGS)
def test_success_budget_flags_require_true_not_integer_one(runner, flag):
    evidence = dict.fromkeys(_BUDGET_FLAGS, True)
    evidence[flag] = 1
    assert runner.budgets_met(evidence) is False


def _execution_fixture(runner, monkeypatch, fault=None):
    """No process, original source/data or real receipt is touched by these mocks."""
    runner.RUN.mkdir()
    state = dict(events=[], publications={}, byte_publications={}, active=0, deadlines=[],
        worker_calls=0, audit_calls=0, verification_calls=0, budget_calls=0, print_calls=0)
    bundle, audit = dict(cases=_invented_cases(runner), runtime_metadata={}), {"invented_independent": True}
    output = b"invented worker completed\n"
    artifacts = dict(report_sha256="d"*64, scientific_artifact_bytes=100,
        selected_reference_sha256="e"*64, input_preflight_sha256="f"*64)
    @contextmanager
    def deadline(seconds):
        assert 0 < seconds <= 90
        state["deadlines"].append(seconds)
        state["active"] += 1
        try: yield
        finally: state["active"] -= 1
    def supervise(command, request, **kwargs):
        assert state["active"] == 1
        state["worker_calls"] += 1
        state["events"].append("worker")
        assert command[-1] == "--worker" and command[1:5] == ["-I", "-S", "-B", "-u"]
        assert kwargs == dict(cache=runner.RUN/"work", log_path=runner.RUN/"work/LIVE_OUTPUT.txt",
            wall_seconds=80, rss_limit=1024**3, log_limit=1024**2, new_session=True)
        assert request["parent_pid"] == os.getpid() and len(request["token"]) == 64
        if fault == "worker_exception": raise RuntimeError("INVENTED_WORKER_FAILURE")
        return dict(exit_code=1 if fault == "worker_false" else 0,
            budget_stop="CPU" if fault == "worker_budget" else None,
            forwarded_log_bytes=len(output)+(1 if fault == "log_mismatch" else 0))
    support = types.SimpleNamespace(deadline=deadline, cache_identity=lambda path: {"invented": 1},
        supervise_process=supervise)
    def get_support():
        state["events"].append("support")
        return support
    def verify_source(support):
        state["events"].append("verify_bundle")
        assert state["active"] == 1
        if fault == "preflight": raise ValueError("INVENTED_PREFLIGHT_FAILURE")
        return bundle
    def publish(path, payload):
        assert state["active"] == 1 or path.name == "DISPATCH.json"
        name = path.name
        state["events"].append(name)
        if name == "STARTED.json" and fault == "started_exclusive": raise FileExistsError("OTHER_OWNER")
        # Simulate durable link followed by fsync failure: retain the publication.
        state["publications"][name] = deepcopy(payload)
        if name == "STARTED.json" and fault == "started_fsync": raise OSError("INVENTED_POST_LINK_FSYNC")
        if name == "COMPLETED.json" and fault == "completed_publish": raise OSError("INVENTED_FSYNC")
    def publish_blob(path, blob):
        assert state["active"] == 1
        if fault == "output_publish": raise OSError("INVENTED_OUTPUT_FAILURE")
        state["byte_publications"][path.name] = blob
    def read(base, relative, cap, expected=None, *, allow_empty=False):
        assert base == runner.RUN/"work"
        if relative == "WORKER_COMPLETED.json": return runner.encoded({"invented_worker_receipt": True})
        assert relative == "LIVE_OUTPUT.txt" and allow_empty is True and cap == 1024**2
        if fault == "log_missing": raise FileNotFoundError("INVENTED_MISSING_LOG")
        return output
    def verify_artifacts(b, receipt):
        state["verification_calls"] += 1
        assert b is bundle and receipt == {"invented_worker_receipt": True}
        if fault == "artifact_corruption": raise ValueError("INVENTED_LAST_ARTIFACT_DRIFT")
        return artifacts
    def audit_sources(b, a, s, evidence):
        state["audit_calls"] += 1
        state["events"].append("source_audit")
        evidence.update(complete=True, unchanged=fault != "source_drift", current={}, frozen={})
    def budget(start, cpu):
        state["budget_calls"] += 1
        assert (start, cpu) == (100.0, 200.0)
        result = dict.fromkeys(_BUDGET_FLAGS, True)
        trigger = {"pre_terminal_budget": 1, "post_completed_budget": 2,
            "post_final_gate_budget": 3, "post_stdout_budget": 4}.get(fault)
        if trigger is not None and state["budget_calls"] >= trigger: result["wall_budget_met"] = False
        return result
    def printed(value, *, flush):
        state["print_calls"] += 1
        assert flush is True and state["active"] == 1
        assert json.loads(value)["release_allowed"] is False
        if fault == "print_timeout": raise TimeoutError("INVENTED_BLOCKED_FINAL_STDOUT")
    def clock():
        state["events"].append("clock")
        return 100.0
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(monotonic=clock), raising=False)
    monkeypatch.setattr(runner, "aggregate_cpu", lambda: 200.0, raising=False)
    monkeypatch.setattr(runner, "support_module", get_support)
    monkeypatch.setattr(runner, "verify_bundle", verify_source, raising=False)
    monkeypatch.setattr(runner, "pre_run_audit", lambda b: audit, raising=False)
    monkeypatch.setattr(runner, "publish_json", publish)
    monkeypatch.setattr(runner, "publish_bytes", publish_blob)
    monkeypatch.setattr(runner, "read_fixed", read)
    monkeypatch.setattr(runner, "verify_retained_artifacts", verify_artifacts)
    monkeypatch.setattr(runner, "audit_final_sources", audit_sources)
    monkeypatch.setattr(runner, "retained_scientific_usage", lambda path: dict(
        logical_bytes_including_partials=101 if fault == "partial_bytes" else 100))
    monkeypatch.setattr(runner, "final_budget_evidence", budget)
    monkeypatch.setattr(runner, "print", printed, raising=False)
    return state


def test_execute_success_is_scalar_hash_bound_and_still_pending_independent_release(runner, monkeypatch):
    state = _execution_fixture(runner, monkeypatch)
    assert runner.execute() == 0
    published = state["publications"]
    assert state["events"].index("clock") < state["events"].index("support") < state["events"].index("STARTED.json")
    assert state["deadlines"][0] == 20 and state["worker_calls"] == state["audit_calls"] == 1
    assert state["verification_calls"] == state["print_calls"] == 1 and state["budget_calls"] == 4
    assert "UNAVAILABLE.json" not in published and "FINALIZATION_FAILED.json" not in published
    completed, gate = published["COMPLETED.json"], published["FINAL_GATE.json"]
    assert completed["status"] == "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT" and completed["failures"] == []
    assert completed["observed_bend_count"] is None and completed["production_use_allowed"] is False
    assert completed["inference_execution"] == "NOT_PERFORMED" and completed["historical_comparability"] == "UNESTABLISHED"
    assert gate["report_sha256"] == "d"*64 and type(gate["report_sha256"]) is str
    assert gate["completed_sha256"] == runner.digest(runner.encoded(completed))
    assert gate["output_sha256"] == runner.digest(state["byte_publications"]["OUTPUT.txt"])


@pytest.mark.parametrize("fault", ("worker_false", "worker_budget", "worker_exception", "artifact_corruption",
    "source_drift", "partial_bytes", "log_missing", "log_mismatch", "output_publish", "completed_publish",
    "pre_terminal_budget", "post_completed_budget", "post_final_gate_budget", "print_timeout", "post_stdout_budget"))
def test_every_failed_worker_or_late_terminal_seam_retains_veto_not_empty_success(runner, monkeypatch, fault):
    state = _execution_fixture(runner, monkeypatch, fault)
    assert runner.execute() == 1
    published = state["publications"]
    assert state["worker_calls"] == state["audit_calls"] == 1
    assert published["FINALIZATION_FAILED.json"]["status"] == "TERMINAL_VETO_NO_RELEASE"
    assert published["FINALIZATION_FAILED.json"]["failures"]
    for lane in published["FINALIZATION_FAILED.json"]["variants"].values():
        assert len(lane) == 11
        for row in lane.values():
            assert all(row[key] is None for key in ("precision", "recall", "f1", "true_positive", "false_positive", "false_negative", "observed_bend_count"))
    if fault in ("worker_false", "worker_budget", "worker_exception"):
        assert state["verification_calls"] == 0
    if fault in ("log_missing", "log_mismatch"):
        assert runner.decode(state["byte_publications"]["OUTPUT.txt"])["status"] == "LOG_UNAVAILABLE_NOT_EMPTY_SUCCESS"
        assert any(row["stage"] == "LOG_READ_OR_BINDING" for row in published["COMPLETED.json"]["failures"])
    if fault in ("post_final_gate_budget", "print_timeout", "post_stdout_budget"):
        assert "FINAL_GATE.json" in published  # A later additive veto dominates it.
    else:
        assert "FINAL_GATE.json" not in published
    if fault == "print_timeout":
        assert state["print_calls"] == 1 and state["deadlines"][-1] == 1


def test_post_link_started_fsync_failure_records_bounded_veto_without_dispatch(runner, monkeypatch):
    state = _execution_fixture(runner, monkeypatch, "started_fsync")
    assert runner.execute() == 1
    assert state["worker_calls"] == state["audit_calls"] == 0
    assert set(state["publications"]) == {"STARTED.json", "FINALIZATION_FAILED.json"}
    veto = state["publications"]["FINALIZATION_FAILED.json"]
    assert veto["worker_dispatched"] is False and veto["no_resume"] is True
    assert veto["failures"] == [dict(stage="STARTED_PUBLICATION", error_type="OSError")]
    assert state["deadlines"] == [20, 1] and not (runner.RUN/"work").exists()


@pytest.mark.parametrize("fault", ("STARTED.json", "COMPLETED.json", "FINAL_GATE.json", "FINALIZATION_FAILED.json",
    "UNAVAILABLE.json", "DISPATCH.json", "WORKER_CLAIMED.json", "started_exclusive", "preflight"))
def test_prior_attempt_and_exclusive_loser_never_write_a_veto_into_other_owner(runner, monkeypatch, fault):
    state = _execution_fixture(runner, monkeypatch, fault)
    if fault.endswith(".json"): (runner.RUN/fault).write_bytes(b"invented other owner")
    with pytest.raises((ValueError, FileExistsError)):
        runner.execute()
    assert state["publications"] == {} and state["worker_calls"] == 0
    assert not (runner.RUN/"work").exists()


@pytest.mark.parametrize("terminal", (None, "COMPLETED.json", "FINAL_GATE.json", "FINALIZATION_FAILED.json", "UNAVAILABLE.json"))
def test_worker_authorization_refuses_every_terminal_veto_before_exclusive_claim(runner, monkeypatch, terminal):
    cache, bundle, audit = runner.RUN/"work", {"invented": "bundle"}, {"invented": "audit"}
    token, parent, pid = "a"*64, 123, 456
    request = dict(token=token, parent_pid=parent)
    expected = dict(parent_pid=parent, token_sha256=runner.digest(token.encode()),
        bundle_sha256=runner.digest(runner.encoded(bundle)), cache_identity={"invented": "cache"})
    started = dict(status="STARTED_NO_RETRY", limits=runner.LIMITS,
        bundle_sha256=runner.digest(runner.encoded(bundle)), audit_sha256=runner.digest(runner.encoded(audit)))
    claims, checked = [], []
    class OwnedPath:
        def __new__(cls, *args): return Path(*args)
        @staticmethod
        def cwd(): return cache
    def exists(path):
        checked.append(path.name)
        return path.name == terminal
    monkeypatch.setattr(runner, "Path", OwnedPath)
    monkeypatch.setattr(runner, "__file__", str(runner.RUN/"source"/runner.RUNNER), raising=False)
    monkeypatch.setattr(runner, "sys", types.SimpleNamespace(stdin=types.SimpleNamespace(buffer=io.BytesIO(runner.encoded(request)))))
    monkeypatch.setattr(runner, "os", types.SimpleNamespace(getppid=lambda: parent, getpid=lambda: pid,
        getpgrp=lambda: pid, environ={"TMPDIR": str(cache)}, path=types.SimpleNamespace(lexists=exists)))
    monkeypatch.setattr(runner, "secrets", types.SimpleNamespace(compare_digest=lambda a, b: a == b))
    monkeypatch.setattr(runner, "verify_bundle", lambda support: bundle, raising=False)
    monkeypatch.setattr(runner, "pre_run_audit", lambda b: audit, raising=False)
    def read(base, relative, cap):
        assert base == runner.RUN and cap == 16384
        return runner.encoded({"DISPATCH.json": expected, "STARTED.json": started}[relative])
    monkeypatch.setattr(runner, "read_fixed", read)
    monkeypatch.setattr(runner, "publish_json", lambda path, value: claims.append((path, value)))
    support = types.SimpleNamespace(is_hex=lambda value, length: type(value) is str and len(value) == length,
        cache_identity=lambda path: {"invented": "cache"})
    if terminal:
        with pytest.raises(ValueError, match="WORKER_PARENT_FROZEN_SOURCE_OR_CONTROL_BINDING"):
            runner.authorize_worker(support)
        assert claims == [] and checked[-1] == terminal
    else:
        assert runner.authorize_worker(support) == (bundle, cache)
        assert len(claims) == 1 and claims[0][0].name == "WORKER_CLAIMED.json"
        assert claims[0][1]["no_retry"] is True
        assert checked == ["COMPLETED.json", "FINAL_GATE.json", "FINALIZATION_FAILED.json", "UNAVAILABLE.json"]


def _called(node, name):
    return [call for call in ast.walk(node) if isinstance(call, ast.Call)
        and ((isinstance(call.func, ast.Name) and call.func.id == name)
             or (isinstance(call.func, ast.Attribute) and call.func.attr == name))]


def test_worker_source_orders_resource_guard_all22_preflight_catalog_and_locked_scorer():
    worker, = _definitions(SOURCE, ("worker",)).body
    cpu, = _called(worker, "setrlimit")
    support, = _called(worker, "support_module")
    guard, = _called(worker, "install_input_guard")
    lock, = _called(worker, "addaudithook")
    imports = [node for node in ast.walk(worker) if isinstance(node, ast.ImportFrom)]
    assert len(imports) == 1 and imports[0].module == "scripts.gold11_cached_fusion_scoring_v1"
    catalog, = [call for call in _called(worker, "input_read")
        if isinstance(call.args[0], ast.Name) and call.args[0].id == "CATALOG"]
    preflight, = [node for node in ast.walk(worker) if isinstance(node, ast.Constant)
        and node.value == "CACHED_INPUT_PREFLIGHT_INCOMPLETE_NO_SCORING_OR_CATALOG_READ"]
    assert cpu.lineno < support.lineno < guard.lineno < preflight.lineno < catalog.lineno < lock.lineno < imports[0].lineno
    assert ast.unparse(lock.args[0]) == "locked_guard"
    assert "len(checks) != 22" in ast.unparse(worker)
    calls = {ast.unparse(call.func) for call in ast.walk(worker) if isinstance(call, ast.Call)}
    assert not any("inference" in name or "evaluate_mesh" in name for name in calls)


def test_execute_final_print_is_deadline_nested_and_budget_rechecked_without_cleanup():
    execute, = _definitions(SOURCE, ("execute",)).body
    printed, = _called(execute, "print")
    wrappers = [node for node in ast.walk(execute) if isinstance(node, ast.With)
        and any(child is printed for child in ast.walk(node))]
    assert len(wrappers) == 1
    context = wrappers[0].items[0].context_expr
    assert ast.unparse(context) == "support.deadline(remaining())"
    later_budget_checks = [call for call in _called(execute, "final_budget_evidence") if call.lineno > printed.lineno]
    assert later_budget_checks
    assert not _called(execute, "rmtree") and not _called(execute, "unlink")


_SMOKE_IMPORTS = """import os, sys, json, hashlib, stat, fcntl, resource
from pathlib import Path
from types import SimpleNamespace
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
"""
_SMOKE_BODY = """
cache = RUN / 'work'
(RUN / 'source/scripts').mkdir(parents=True)
cache.mkdir()
ROOT.mkdir()
CATALOG = 'invented_catalog.json'
payload = b'invented fixture bytes'
for name in ('invented_a0.json', 'invented_cad.json', CATALOG, 'undeclared.json'):
    (ROOT / name).write_bytes(payload)
os.chdir(cache)
def runtime_roots():
    return ()  # No imports/native loads after guard; exact base-hook functions.
support = SimpleNamespace(worker_audit_hook=worker_audit_hook)
read_input, locked = install_input_guard(support, cache, [
    dict(a0_path='invented_a0.json', cad_path='invented_cad.json')])
def denied(call):
    try:
        call()
    except PermissionError:
        return True
    raise AssertionError('INVENTED_SCOPE_MUST_BE_DENIED')
assert denied(lambda: (ROOT / 'invented_a0.json').read_bytes())
assert read_input('invented_a0.json', len(payload), digest(payload)) == payload
assert read_input('invented_cad.json', len(payload), digest(payload)) == payload
assert read_input(CATALOG, len(payload), digest(payload)) == payload
assert denied(lambda: read_input('undeclared.json', len(payload), digest(payload)))
sys.addaudithook(locked)
assert denied(lambda: read_input('invented_a0.json', len(payload), digest(payload)))
assert denied(lambda: (ROOT / CATALOG).read_bytes())
owned = cache / 'tiny_owned.txt'
owned.write_bytes(b'owned')
assert owned.read_bytes() == b'owned'
owned.unlink()
print(json.dumps(dict(status='INVENTED_AUDIT_GUARD_SMOKE_COMPLETE',
    granted_components_observed=True, original_locked_after_preflight=True,
    physical_inputs_opened=False, inference_performed=False)), flush=True)
"""


def _guard_program(owned):
    runner_tree = _definitions(SOURCE, ("digest", "read_fixed", "install_input_guard"))
    support_tree = _definitions(SUPPORT_SOURCE, ("contained", "allowed_native_library", "worker_audit_hook"))
    definitions = ast.unparse(ast.Module(body=runner_tree.body+support_tree.body, type_ignores=[]))
    return (_SMOKE_IMPORTS + f"ROOT = Path({str(owned/'inputs')!r})\nRUN = Path({str(owned/'run')!r})\n"
        + definitions + "\n" + _SMOKE_BODY)


def test_smoke_source_has_fixed_extraction_scope_no_cli_or_native_and_limits_before_hook(tmp_path):
    program = _guard_program(tmp_path/"fresh_owned_smoke")
    tree = ast.parse(program)
    top_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert top_functions == {"digest", "read_fixed", "install_input_guard", "contained", "allowed_native_library", "worker_audit_hook", "runtime_roots", "denied"}
    top_imports = {alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert top_imports == {"os", "sys", "json", "hashlib", "stat", "fcntl", "resource", "Path", "SimpleNamespace"}
    assert not {"worker", "main", "execute", "freeze", "authorize_worker"} & top_functions
    assert "BINDINGS" not in program and "data/bend_corpus" not in program
    limits = _called(tree, "setrlimit")
    guard, = _called(tree, "install_input_guard")
    assert len(limits) == 2 and all(call.lineno < guard.lineno for call in limits)
    assert RESOURCE_LIMITS == dict(wall_seconds=60, worker_cpu_seconds=45,
        process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
    assert CHILD_LIMITS == dict(wall_seconds=8, cpu_seconds=5, output_bytes=65536)
    assert SMOKE_OPT_IN == "SHERMAN_RUN_CACHED_CONTROL_GUARD_SMOKE_V2"


@pytest.mark.skipif(os.environ.get(SMOKE_OPT_IN) != "1", reason="Requires separately frozen invented subprocess release")
def test_opt_in_fresh_same_group_stdlib_child_exercises_actual_audit_delivery(tmp_path):
    owned = tmp_path/"fresh_owned_smoke"
    owned.mkdir()
    program = _guard_program(owned)
    log = owned/"GUARD_OUTPUT.txt"
    # Never start a new process group: the external 1GiB tree monitor retains it.
    with log.open("xb") as stream:
        child = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-u", "-c", program],
            cwd=owned, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=False, env={"PATH": "/usr/bin:/bin", "HOME": str(owned),
                "TMPDIR": str(owned), "PYTHONDONTWRITEBYTECODE": "1"})
        timed_out = False
        try:
            exit_code = child.wait(timeout=CHILD_LIMITS["wall_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            exit_code = child.wait(timeout=1)
    with log.open("rb") as stream:
        output = stream.read(CHILD_LIMITS["output_bytes"]+1)
    # Forward the complete bounded invented-child log BEFORE assertions. The
    # outer harness retains it even if pytest later removes its owned tmp tree.
    print(output.decode("utf-8", errors="replace"), flush=True)
    assert len(output) <= CHILD_LIMITS["output_bytes"] and not timed_out and exit_code == 0
    record = json.loads(output)
    assert record == dict(status="INVENTED_AUDIT_GUARD_SMOKE_COMPLETE", granted_components_observed=True,
        original_locked_after_preflight=True, physical_inputs_opened=False, inference_performed=False)



def test_scandir_fixture_iterable_and_context_identity_match_calling_protocol():
    first, second = object(), object()
    entries = _Entries([first, second])
    assert iter(entries) is entries
    with entries as entered:
        assert entered is entries and iter(entered) is entries
        # The actual helper loops over the original object, not __enter__'s
        # return value. Both names must address this same consuming iterator.
        assert next(entries) is first
        assert next(entered) is second
        with pytest.raises(StopIteration):
            next(entries)
    assert list(entries) == []


def test_scandir_fixture_repeated_iteration_never_restarts_or_duplicates_rows():
    rows = [object(), object(), object()]
    entries = _Entries(rows)
    first_iterator, second_iterator = iter(entries), iter(entries)
    assert first_iterator is second_iterator is entries
    assert next(first_iterator) is rows[0]
    assert next(second_iterator) is rows[1]
    assert list(entries) == [rows[2]]
    assert list(first_iterator) == list(second_iterator) == list(entries) == []
    with _Entries([]) as empty:
        assert list(empty) == []


def test_scandir_fixture_early_context_exit_closes_and_does_not_suppress_failure():
    entries = _Entries([object(), object()])
    with pytest.raises(RuntimeError, match="INVENTED_CONTEXT_FAILURE"):
        with entries as entered:
            assert entered is entries
            next(entries)
            raise RuntimeError("INVENTED_CONTEXT_FAILURE")
    assert list(entries) == []
    with pytest.raises(StopIteration):
        next(entries)


def test_real_scandir_accounts_only_fresh_invented_files_and_retained_partial_hardlinks(runner, tmp_path):
    cache = tmp_path/"invented_scandir_cache"
    for relative in ("inputs/a0", "inputs/cad", "checks"):
        (cache/relative).mkdir(parents=True, exist_ok=True)
    report, partial = cache/"SCORER_REPORT.json", cache/".SCORER_REPORT.json.partial-invented"
    report.write_bytes(b"report!")
    os.link(report, partial)
    (cache/"inputs/a0/invented.json").write_bytes(b"abc")
    (cache/"checks/invented.json").write_bytes(b"check")
    (cache/"LIVE_OUTPUT.txt").write_bytes(b"separately bounded invented log")
    (cache/"WORKER_COMPLETED.json").write_bytes(b"invented control")
    assert runner.os is os and report.stat().st_ino == partial.stat().st_ino
    result = runner.retained_scientific_usage(cache)
    assert result["files"] == {
        "SCORER_REPORT.json": 7, ".SCORER_REPORT.json.partial-invented": 7,
        "inputs/a0/invented.json": 3, "checks/invented.json": 5}
    assert result["logical_bytes_including_partials"] == 22
    assert result["partial_hardlinks_counted_conservatively_twice"] is True


def test_qualification_v2_preserves_frozen_v1_except_exact_iterator_repair_and_optin():
    predecessor = ROOT/"tests/test_gold11_cached_control_v1.py"
    blob = predecessor.read_bytes()  # Named frozen source only; never import it.
    assert hashlib.sha256(blob).hexdigest() == "67ec0c5dccebfd58298818e6c43df2651eeee2492c636427f15d4776241dd5e6"
    old, current = ast.parse(blob), ast.parse(Path(__file__).read_text())
    additions = {
        "test_scandir_fixture_iterable_and_context_identity_match_calling_protocol",
        "test_scandir_fixture_repeated_iteration_never_restarts_or_duplicates_rows",
        "test_scandir_fixture_early_context_exit_closes_and_does_not_suppress_failure",
        "test_real_scandir_accounts_only_fresh_invented_files_and_retained_partial_hardlinks",
        "test_qualification_v2_preserves_frozen_v1_except_exact_iterator_repair_and_optin",
    }
    added = [node for node in current.body if isinstance(node, ast.FunctionDef) and node.name in additions]
    assert len(added) == len(additions) and {node.name for node in added} == additions
    current.body = [node for node in current.body if node not in added]
    original_entries, = [node for node in old.body if isinstance(node, ast.ClassDef) and node.name == "_Entries"]
    repaired_entries, = [node for node in current.body if isinstance(node, ast.ClassDef) and node.name == "_Entries"]
    exact_repair = ast.parse("""class _Entries:
    def __init__(self, rows): self.rows = iter(rows)
    def __enter__(self): return self
    def __iter__(self): return self
    def __next__(self): return next(self.rows)
    def __exit__(self, *args):
        self.rows = iter(())
        return False
""").body[0]
    assert ast.dump(repaired_entries, include_attributes=False) == ast.dump(exact_repair, include_attributes=False)
    current.body[current.body.index(repaired_entries)] = deepcopy(original_entries)
    class RestoreOptIn(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "SHERMAN_RUN_CACHED_CONTROL_GUARD_SMOKE_V2":
                return ast.copy_location(ast.Constant(value="SHERMAN_RUN_CACHED_CONTROL_GUARD_SMOKE_V1"), node)
            return node
    current = RestoreOptIn().visit(current)
    assert ast.dump(current, include_attributes=False) == ast.dump(old, include_attributes=False)
