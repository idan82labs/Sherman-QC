"""Pure mocked protocol/supervisor tests; no native geometry or corpus reads."""

import ast
import copy
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import run_vendor_unfold_runtime_smoke_v2 as smoke
from backend.vendor_unfold_contract_v2 import unavailable


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "_pins", lambda: {"authored-source-only": "a" * 64})
    monkeypatch.setattr(smoke, "_review", lambda *args: {"mocked_independent_pre": True})
    monkeypatch.setattr(smoke.worker, "ADMISSION_LOCK_PATH", str(tmp_path / "authored-admission.lock"))
    monkeypatch.setattr(smoke.worker, "_cache", {})
    directory, record = tmp_path / "generated", tmp_path / "generation.jsonl"
    frozen = smoke.freeze_generation(directory, record)

    def supervise(folder, request, lock, started=None):
        assert smoke.worker.strict_json((folder / "request.json").read_bytes()) == request
        cases = []
        for case_id in smoke.CASE_IDS:
            raw = ("fresh authored mock " + case_id).encode()
            case = smoke.case_definition(case_id)
            target = Path(request["freeze"]["directory"]) / case["step_filename"]
            with target.open("xb") as source:
                source.write(raw)
            cases.append({**case, "source_sha256": smoke._hash(raw)})
        smoke.save_new(Path(request["freeze"]["directory"]) / "generated-cases.json", cases)
        return {"returncode": 0, "peak_process_group_rss_kib": 100, "elapsed_seconds": .5}

    monkeypatch.setattr(smoke, "_supervise_generation", supervise)
    return directory, record, frozen


def generated(prepared):
    directory, record, frozen = prepared
    ledger = smoke.generate(frozen, directory, record, "unused-mocked-pre", "unused-mocked-archive")
    assert ledger["status"] == "COMPLETE_SYNTHETIC_GENERATION"
    return directory, record, ledger


def test_module_has_no_top_level_native_imports():
    source = Path(smoke.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module for node in imports if isinstance(node, ast.ImportFrom)]
    assert not any(name.split(".")[0] in {"OCP", "numpy", "FreeCAD"} for name in names)


def test_fixed_two_case_oracles_are_pose_invariant():
    first, posed = [smoke.case_definition(case_id) for case_id in smoke.CASE_IDS]
    assert first["oracle"] == posed["oracle"]
    assert first["oracle"]["flat_area_mm2"] == 40 * (35 + smoke.math.pi)
    assert first["oracle"]["sorted_flat_bbox_mm"] == sorted([40, 35 + smoke.math.pi])
    assert posed["construction"]["rigid_pose"] == smoke.POSE
    a, b = posed["candidates"][0]["line_start"], posed["candidates"][0]["line_end"]
    assert smoke.math.isclose(smoke.math.dist(a, b), 40, abs_tol=1e-12)
    assert first["candidates"] != posed["candidates"]
    assert first["oracle"]["manufacturing_ready"] is False


def test_generation_is_exclusive_append_only_and_provenance_bound(prepared, tmp_path):
    directory, record, ledger = generated(prepared)
    events = [json.loads(line) for line in record.read_bytes().splitlines()]
    assert [event["event"] for event in events] == ["START", "TERMINAL"]
    assert events[-1] == ledger
    evaluation_record = tmp_path / "evaluation.jsonl"
    frozen = smoke.freeze_evaluation(directory, record, evaluation_record)
    assert frozen["cases"] == ledger["cases"]
    assert frozen["selection"] == "ALL_GENERATED_CASES_IN_RETAINED_ORDER"
    assert frozen["checks"] == list(smoke.CHECKS)
    assert frozen["generation_provenance"]["record_sha256"] == smoke._hash(record.read_bytes())
    with pytest.raises(FileExistsError):
        smoke.generate(events[0]["freeze"], directory, record, "unused", "unused")


def test_protocol_publication_never_overwrites(tmp_path):
    protocol = tmp_path / "authored-protocol.json"
    smoke.save_new(protocol, {"first": True})
    original = protocol.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.save_new(protocol, {"second": True})
    assert protocol.read_bytes() == original


def test_generation_freeze_rejects_runner_binding_change(prepared):
    directory, record, frozen = prepared
    frozen["runner_path"] = "/changed-authored-runner.py"
    with pytest.raises(ValueError, match="GENERATION_FREEZE_MISMATCH"):
        smoke.generate(frozen, directory, record, "unused", "unused")
    assert not record.exists() and not directory.exists()


def test_freeze_does_not_alias_operational_limits(prepared):
    directory, record, frozen = prepared
    frozen["limits"]["wall_seconds"] = 999
    assert smoke.GEN_LIMITS["wall_seconds"] == 60
    with pytest.raises(ValueError, match="GENERATION_FREEZE_MISMATCH"):
        smoke.generate(frozen, directory, record, "unused", "unused")


@pytest.mark.parametrize("change", ["subset", "oracle", "hash", "extra", "symlink", "budget"])
def test_generated_cohort_cannot_be_reduced_rewritten_or_redirected(prepared, change, monkeypatch):
    directory, _, ledger = generated(prepared)
    cases = copy.deepcopy(ledger["cases"])
    metadata = directory / "generated-cases.json"
    if change == "subset":
        cases = cases[:1]
    elif change == "oracle":
        cases[1]["oracle"]["flat_area_mm2"] += 1
    elif change == "hash":
        cases[0]["source_sha256"] = "0" * 64
    elif change == "extra":
        (directory / "unselected-authored-file").write_bytes(b"mock")
    elif change == "symlink":
        target = directory / cases[0]["step_filename"]
        target.unlink()
        target.symlink_to(directory / cases[1]["step_filename"])
    else:
        monkeypatch.setattr(smoke, "MAX_GENERATED_BYTES", 1)
    # Test-only authored mutation of private mock metadata, not a runtime helper.
    metadata.write_bytes(smoke._json(cases))
    with pytest.raises(ValueError):
        smoke.load_cases(directory)


def test_failed_generation_retains_terminal_record_and_does_not_retry(prepared, monkeypatch):
    directory, record, frozen = prepared
    monkeypatch.setattr(smoke, "_supervise_generation", lambda *args: (_ for _ in ()).throw(ValueError("MOCK_FAILURE")))
    ledger = smoke.generate(frozen, directory, record, "unused", "unused")
    assert ledger["status"] == "FAILED" and ledger["reason"] == "MOCK_FAILURE"
    assert len(record.read_bytes().splitlines()) == 2
    with pytest.raises(FileExistsError):
        smoke.generate(frozen, directory, record, "unused", "unused")


def test_aggregate_generation_clock_includes_post_child_source_checks(prepared, monkeypatch):
    directory, record, frozen = prepared
    clock = {"now": 0., "pin_calls": 0}
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock["now"])
    original_supervisor = smoke._supervise_generation
    def supervise(folder, request, lock, started=None):
        result = original_supervisor(folder, request, lock, started)
        clock["now"] = 59.
        return result
    def pins():
        clock["pin_calls"] += 1
        if clock["pin_calls"] > 1:
            clock["now"] = 61.
        return {"authored-source-only": "a" * 64}
    monkeypatch.setattr(smoke, "_supervise_generation", supervise)
    monkeypatch.setattr(smoke, "_pins", pins)
    ledger = smoke.generate(frozen, directory, record, "unused", "unused")
    assert ledger["status"] == "FAILED"
    assert ledger["reason"] == "GENERATION_AGGREGATE_TIME_BUDGET_EXCEEDED"
    assert ledger["elapsed_seconds"] == 61.


def test_generation_quarantine_uses_shared_v2_journal(prepared):
    directory, record, frozen = prepared
    lock = Path(smoke.worker.ADMISSION_LOCK_PATH)
    lock.write_text('{"state":"QUARANTINED","pgid":12345}')
    original = lock.read_bytes()
    ledger = smoke.generate(frozen, directory, record, "unused", "unused")
    assert ledger["status"] == "FAILED" and "QUARANTINED" in ledger["reason"]
    assert not directory.exists() and lock.read_bytes() == original


@pytest.mark.parametrize("mode,reason", [("rss", "RSS_BUDGET"), ("monitor", "RSS_MONITOR"),
    ("descendants", "LIVE_DESCENDANTS"), ("crash", "CHILD_FAILED"), ("time", "TIME_BUDGET"),
    ("log", "LOG_BUDGET")])
def test_generation_supervisor_caps_and_cleanup_are_mandatory(tmp_path, monkeypatch, mode, reason):
    captured, stopped = {}, []
    process = SimpleNamespace(pid=12345, returncode=1 if mode == "crash" else 0,
        poll=lambda: None if mode in {"rss", "monitor", "time", "log"} else 0)
    def popen(args, **kwargs):
        captured.update(kwargs)
        if mode == "log":
            kwargs["stdout"].write(b"x" * (smoke.GEN_LIMITS["combined_log_bytes"] + 1))
        return process
    monkeypatch.setattr(smoke.subprocess, "Popen", popen)
    rows = [] if mode in {"monitor", "crash", "time", "log"} else [(12346,1)] if mode == "descendants" else [(12345,smoke.GEN_LIMITS["process_group_rss_kib"]+1)]
    if mode == "log": rows = [(12345, 1)]
    monkeypatch.setattr(smoke.worker, "_group_rows", lambda _: rows)
    monkeypatch.setattr(smoke.worker, "_stop_group", lambda p: stopped.append(p.pid))
    monkeypatch.setattr(smoke, "_check_generation_output", lambda _: None)
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "must-not-inherit")
    if mode == "time":
        ticks = iter([0., 0., 61.])  # Start, pre-spawn admission, post-spawn expiry.
        monkeypatch.setattr(smoke.time, "monotonic", lambda: next(ticks))
    with (tmp_path / "authored.lock").open("w+b") as lock:
        with pytest.raises(ValueError, match=reason):
            smoke._supervise_generation(tmp_path, {"nonce": "authored", "freeze": {"directory": str(tmp_path)}}, lock)
        lock.seek(0)
        assert json.loads(lock.read()) == {"state": "QUIESCENT", "pgid": None}
    assert stopped == [12345]
    assert captured["start_new_session"] is True
    assert "DYLD_LIBRARY_PATH" not in captured["env"]
    assert captured["env"]["HOME"] == str(tmp_path)
    assert captured["env"]["OMP_NUM_THREADS"] == "1"


def test_failed_cleanup_quarantines_instead_of_marking_quiescent(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=12345, returncode=1, poll=lambda: 0)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(smoke.worker, "_group_rows", lambda _: [])
    monkeypatch.setattr(smoke.worker, "_stop_group", lambda _: (_ for _ in ()).throw(ValueError("MOCK_CLEANUP_FAILED")))
    monkeypatch.setattr(smoke, "_check_generation_output", lambda _: None)
    with (tmp_path / "authored.lock").open("w+b") as lock:
        with pytest.raises(ValueError, match="CLEANUP_FAILED"):
            smoke._supervise_generation(tmp_path, {"nonce": "authored", "freeze": {"directory": str(tmp_path)}}, lock)
        lock.seek(0)
        assert json.loads(lock.read()) == {"state": "QUARANTINED", "pgid": 12345}


def test_expired_aggregate_generation_admission_never_spawns_child(tmp_path, monkeypatch):
    spawned, stopped = [], []
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 61.)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    monkeypatch.setattr(smoke.worker, "_stop_group", lambda process: stopped.append(process))
    with (tmp_path / "authored.lock").open("w+b") as lock:
        with pytest.raises(ValueError, match="GENERATION_AGGREGATE_TIME_BUDGET_EXCEEDED"):
            smoke._supervise_generation(tmp_path, {"nonce": "authored"}, lock, started=0.)
        lock.seek(0)
        assert json.loads(lock.read()) == {"state": "QUIESCENT", "pgid": None}
    assert not spawned and not stopped


def test_supervisor_success_reports_peak_and_quiescence(tmp_path, monkeypatch):
    polls = iter([None, 0])
    process = SimpleNamespace(pid=12345, returncode=0, poll=lambda: next(polls))
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(smoke.worker, "_group_rows", lambda _: [(12345, 100)])
    monkeypatch.setattr(smoke.worker, "_stop_group", lambda _: None)
    monkeypatch.setattr(smoke.time, "sleep", lambda _: None)
    generated_directory = tmp_path / "authored-generated"
    generated_directory.mkdir()
    with (tmp_path / "authored.lock").open("w+b") as lock:
        result = smoke._supervise_generation(tmp_path,
            {"nonce": "authored", "freeze": {"directory": str(generated_directory)}}, lock)
        lock.seek(0)
        assert json.loads(lock.read()) == {"state": "QUIESCENT", "pgid": None}
    assert result["returncode"] == 0 and result["peak_process_group_rss_kib"] == 100


@pytest.mark.parametrize("failure", ["step-budget", "metadata-budget", "extra", "directory", "symlink"])
def test_generation_output_budget_is_checked_while_child_runs(tmp_path, failure):
    if failure == "step-budget":
        (tmp_path / "L90.step").write_bytes(b"x" * (smoke.MAX_GENERATED_BYTES + 1))
    elif failure == "metadata-budget":
        (tmp_path / "generated-cases.json").write_bytes(b"x" * 100_001)
    elif failure == "extra":
        (tmp_path / "not-selected.step").write_bytes(b"mock")
    elif failure == "directory":
        (tmp_path / "L90.step").mkdir()
    else:
        (tmp_path / "L90.step").symlink_to(tmp_path / "mock-target")
    with pytest.raises(ValueError):
        smoke._check_generation_output(tmp_path)


def test_native_entrypoint_rejects_unsupervised_request_before_native_import(tmp_path, monkeypatch):
    path = tmp_path / "authored-request.json"
    smoke.save_new(path, {"nonce": "authored", "freeze": {}})
    monkeypatch.delenv("SHERMAN_VENDOR_SMOKE_GENERATION_V2", raising=False)
    with pytest.raises(ValueError, match="SUPERVISED_GENERATION_ENTRYPOINT_REQUIRED"):
        smoke._generate_native(path)


def make_archive(path, source_pins, raw_by_path):
    with tarfile.open(path, "x:") as archive:
        for name in source_pins:
            raw = raw_by_path[name]
            member = tarfile.TarInfo("sources/" + smoke._hash(name.encode()))
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def test_source_archive_requires_exact_member_payloads(tmp_path):
    raw = {"/authored/source.py": b"pure synthetic source"}
    pins = {name: smoke._hash(value) for name, value in raw.items()}
    archive = tmp_path / "authored-source.tar"
    make_archive(archive, pins, raw)
    assert smoke._verify_archive(archive, pins) == smoke._hash(archive.read_bytes())
    with pytest.raises(ValueError, match="PIN_MISMATCH"):
        smoke._verify_archive(archive, {"/authored/source.py": "0" * 64})
    with pytest.raises(ValueError, match="INCOMPLETE"):
        smoke._verify_archive(archive, {**pins, "/authored/missing.py": "1" * 64})


def test_empty_regular_pinned_source_member_is_losslessly_validated(tmp_path):
    raw = {"/authored/empty/__init__.py": b""}
    pins = {name: smoke._hash(value) for name, value in raw.items()}
    archive = tmp_path / "authored-empty-source.tar"
    make_archive(archive, pins, raw)
    assert smoke._verify_archive(archive, pins) == smoke._hash(archive.read_bytes())
    with pytest.raises(ValueError, match="SOURCE_ARCHIVE_PIN_MISMATCH"):
        smoke._verify_archive(archive, {"/authored/empty/__init__.py": "0" * 64})


@pytest.mark.parametrize("failure", ["duplicate", "unexpected", "symlink", "hardlink", "directory"])
def test_source_archive_refuses_extra_or_nonregular_members(tmp_path, failure):
    source_path, raw = "/authored/source.py", b"pure synthetic source"
    pins = {source_path: smoke._hash(raw)}
    archive = tmp_path / "authored-invalid.tar"
    with tarfile.open(archive, "x:") as stream:
        member = tarfile.TarInfo("sources/" + smoke._hash(source_path.encode()))
        member.size = len(raw)
        stream.addfile(member, io.BytesIO(raw))
        bad = tarfile.TarInfo(member.name if failure == "duplicate" else "unselected")
        if failure in {"symlink", "hardlink", "directory"}:
            bad.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "directory": tarfile.DIRTYPE}[failure]
            bad.linkname = "/authored/elsewhere.py"
            stream.addfile(bad)
        else:
            bad.size = len(raw)
            stream.addfile(bad, io.BytesIO(raw))
    with pytest.raises(ValueError, match="MEMBER_INVALID"):
        smoke._verify_archive(archive, pins)


def test_exact_pre_is_bound_to_freeze_source_archive_and_non_author(tmp_path):
    raw = {"/authored/source.py": b"pure synthetic source"}
    pins = {name: smoke._hash(value) for name, value in raw.items()}
    frozen = {"source_pins": pins, "schema": "authored-protocol"}
    archive = tmp_path / "authored-source.tar"
    make_archive(archive, pins, raw)
    review = {"schema": "SHERMAN_VENDOR_UNFOLD_RUNTIME_PRE_V2", "disposition": "GREEN",
              "author": "author", "reviewer": "independent-reviewer", "blocking_findings": [],
              "freeze_sha256": smoke._hash(smoke._json(frozen)),
              "source_bundle_sha256": smoke._hash(smoke._json(pins)),
              "source_archive_sha256": smoke._hash(archive.read_bytes())}
    path = tmp_path / "authored-review.json"
    smoke.save_new(path, review)
    assert smoke._review(frozen, path, archive)["review_sha256"] == smoke._hash(path.read_bytes())
    for changes in ({"reviewer": "author"}, {"freeze_sha256": "0" * 64},
                    {"source_bundle_sha256": "0" * 64}, {"blocking_findings": ["mock finding"]},
                    {"source_archive_sha256": "0" * 64}):
        path.write_bytes(smoke._json({**review, **changes}))
        with pytest.raises(ValueError):
            smoke._review(frozen, path, archive)


def test_evaluation_keeps_both_cases_with_cold_cache_mutation_and_hash_refusal(prepared, tmp_path, monkeypatch):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / "evaluation.jsonl"
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    calls, cached = [], {}
    def bounded(path, expected, candidates):
        raw = Path(path).read_bytes()
        calls.append((Path(path).name, expected, candidates))
        if smoke._hash(raw) != expected:
            return unavailable("CAD_HASH_MISMATCH")
        if expected in cached:
            result = copy.deepcopy(cached[expected])
            result["diagnostics"]["cache_hit"] = True
            return result
        result = {"vertices": [0., 0., 0.], "diagnostics": {"cache_hit": False, "limits": dict(smoke.worker.LIMITS)}}
        cached[expected] = copy.deepcopy(result)
        return result
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", bounded)
    monkeypatch.setattr(smoke, "assert_preview", lambda *args: {"mocked_oracle": True})
    ledger = smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert ledger["status"] == "COMPLETE_SYNTHETIC_RUNTIME_SMOKE"
    assert [row["case_id"] for row in ledger["cases"]] == list(smoke.CASE_IDS)
    assert len(calls) == 6
    assert all([check["check"] for check in row["checks"]] == list(smoke.CHECKS) for row in ledger["cases"])
    assert all(check["status"] == "PASSED" for row in ledger["cases"] for check in row["checks"])
    receipts = [check for row in ledger["cases"] for check in row["checks"]]
    assert ledger["total_receipt_bytes"] == sum(check["receipt_bytes"] for check in receipts)
    for check in receipts:
        raw = Path(check["receipt_path"]).read_bytes()
        assert smoke._hash(raw) == check["receipt_sha256"]
        if check["check"] == "cold":
            assert json.loads(raw)["vertices"][0] == 0
            assert json.loads(raw)["diagnostics"]["limits"]["wall_seconds"] == 60
    assert [json.loads(line)["event"] for line in record.read_bytes().splitlines()] == ["START", "CASE", "CASE", "TERMINAL"]


def test_evaluation_failure_does_not_reduce_selected_cohort(prepared, tmp_path, monkeypatch):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / "evaluation-failure.jsonl"
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: unavailable("MOCK_NATIVE_FAILURE"))
    ledger = smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert ledger["status"] == "FAILED"
    assert [row["case_id"] for row in ledger["cases"]] == list(smoke.CASE_IDS)
    assert all(row["passed"] is False for row in ledger["cases"])
    for row in ledger["cases"]:
        assert [check["status"] for check in row["checks"]] == ["FAILED", "NOT_EXECUTED_AFTER_FAILURE", "NOT_EXECUTED_AFTER_FAILURE"]
        cold_receipt = Path(row["checks"][0]["receipt_path"])
        assert json.loads(cold_receipt.read_bytes())["reason"] == "MOCK_NATIVE_FAILURE"


def test_slow_post_reservation_source_pinning_cannot_launch_any_wrapper(prepared, tmp_path, monkeypatch):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / "evaluation-slow-pinning.jsonl"
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    clock, pin_calls, wrapper_calls = {"now": 0.}, [], []
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock["now"])
    def pins():
        pin_calls.append(True)
        # The first two calls reconstruct/compare the evaluation freeze before
        # its clock starts. The third is the case's post-reservation pin check.
        if len(pin_calls) >= 3:
            clock["now"] = 500.
        return {"authored-source-only": "a" * 64}
    monkeypatch.setattr(smoke, "_pins", pins)
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: wrapper_calls.append(args))
    ledger = smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert not wrapper_calls and ledger["status"] == "FAILED"
    assert ledger["cases"][0]["reason"] == "EVALUATION_WRAPPER_ADMISSION_BUDGET_EXHAUSTED"
    assert [row["case_id"] for row in ledger["cases"]] == list(smoke.CASE_IDS)
    assert all(check["status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in ledger["cases"] for check in row["checks"])
    assert ledger["total_receipt_bytes"] == 0


@pytest.mark.parametrize("after_call", [1, 2])
def test_between_call_budget_exhaustion_prevents_next_wrapper(prepared, tmp_path, monkeypatch, after_call):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / f"evaluation-between-calls-{after_call}.jsonl"
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    clock, calls, cache = {"now": 0.}, [], {}
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock["now"])
    def bounded(path, expected, candidates):
        calls.append((Path(path).name, clock["now"]))
        if expected in cache:
            result = copy.deepcopy(cache[expected])
            result["diagnostics"]["cache_hit"] = True
        else:
            result = {"vertices": [0.,0.,0.], "diagnostics": {"cache_hit": False, "limits": dict(smoke.worker.LIMITS)}}
            cache[expected] = copy.deepcopy(result)
        if len(calls) == after_call:
            clock["now"] = 361.
        return result
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", bounded)
    monkeypatch.setattr(smoke, "assert_preview", lambda *args: {"mocked_oracle": True})
    ledger = smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert len(calls) == after_call and ledger["status"] == "FAILED"
    assert ledger["cases"][0]["reason"] == "EVALUATION_WRAPPER_ADMISSION_BUDGET_EXHAUSTED"
    assert [check["status"] for check in ledger["cases"][0]["checks"]] == ["PASSED"] * after_call + ["NOT_EXECUTED_AFTER_FAILURE"] * (3-after_call)
    assert all(check["status"] == "NOT_EXECUTED_AFTER_FAILURE" for check in ledger["cases"][1]["checks"])
    assert all(Path(check["receipt_path"]).is_file() for check in ledger["cases"][0]["checks"][:after_call])


@pytest.mark.parametrize("now,admitted", [(360., True), (360.000001, False)])
def test_wrapper_admission_requires_at_least_full_sixty_seconds(monkeypatch, now, admitted):
    calls, check = [], {"status": "NOT_EXECUTED"}
    monkeypatch.setattr(smoke.time, "monotonic", lambda: now)
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: calls.append(args))
    if admitted:
        smoke._call_wrapper_admitted(0., check, "authored-source", "a"*64, [])
        assert len(calls) == 1 and check["status"] == "RUNNING"
    else:
        with pytest.raises(ValueError, match="EVALUATION_WRAPPER_ADMISSION_BUDGET_EXHAUSTED"):
            smoke._call_wrapper_admitted(0., check, "authored-source", "a"*64, [])
        assert not calls and check["status"] == "NOT_EXECUTED"


def test_cache_alias_failure_retains_corrupted_cache_and_unchanged_cold_receipts(prepared, tmp_path, monkeypatch):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / "evaluation-alias-failure.jsonl"
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    cache = {}
    def aliased(path, expected, candidates):
        if expected in cache:
            cache[expected]["diagnostics"]["cache_hit"] = True
            return cache[expected]
        result = {"vertices": [0.,0.,0.], "diagnostics": {"cache_hit": False, "limits": dict(smoke.worker.LIMITS)}}
        cache[expected] = result  # Deliberately broken mocked caller alias.
        return result
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", aliased)
    monkeypatch.setattr(smoke, "assert_preview", lambda *args: {"mocked_oracle": True})
    ledger = smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert ledger["status"] == "FAILED"
    for row in ledger["cases"]:
        assert [check["status"] for check in row["checks"]] == ["PASSED", "FAILED", "NOT_EXECUTED_AFTER_FAILURE"]
        cold, cached, _ = row["checks"]
        assert json.loads(Path(cold["receipt_path"]).read_bytes())["vertices"][0] == 0
        assert json.loads(Path(cached["receipt_path"]).read_bytes())["vertices"][0] == 99


def test_receipt_budget_failure_retains_observed_hash_and_all_dispositions(prepared, tmp_path, monkeypatch):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / "evaluation-budget-failure.jsonl"
    limits = {**smoke.EVAL_LIMITS, "total_receipt_bytes": 1}
    monkeypatch.setattr(smoke, "EVAL_LIMITS", limits)
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: unavailable("MOCK_FAILURE"))
    ledger = smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert ledger["status"] == "FAILED" and ledger["total_receipt_bytes"] == 0
    for row in ledger["cases"]:
        cold = row["checks"][0]
        assert cold["status"] == "FAILED" and cold["result_observed"] is True
        assert cold["reason"] == "EVALUATION_RECEIPT_BYTE_BUDGET_EXCEEDED"
        assert len(cold["observed_payload_sha256"]) == 64
        assert "receipt_path" not in cold


def test_existing_derived_receipt_is_never_overwritten(prepared, tmp_path, monkeypatch):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / "evaluation-collision.jsonl"
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    prior = record.with_name(record.stem + "-L90-cold.json")
    prior.write_bytes(b"authored prior immutable receipt")
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: unavailable("MOCK_FAILURE"))
    ledger = smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert ledger["status"] == "FAILED"
    assert ledger["cases"][0]["checks"][0]["exception_type"] == "FileExistsError"
    assert prior.read_bytes() == b"authored prior immutable receipt"


def test_evaluation_refuses_protocol_selection_or_oracle_change(prepared, tmp_path):
    directory, generation_record, _ = generated(prepared)
    record = tmp_path / "evaluation-refused.jsonl"
    frozen = smoke.freeze_evaluation(directory, generation_record, record)
    frozen["cases"] = frozen["cases"][:1]
    with pytest.raises(ValueError, match="EVALUATION_FREEZE_MISMATCH"):
        smoke.evaluate(frozen, directory, record, generation_record, "unused", "unused")
    assert not record.exists()


def visual_receipt(case, bundle):
    # Pure authored transport geometry: one rectangle subdivided into strips.
    # This exercises numerical/envelope checking, not native development.
    from tests.test_vendor_unfold_contract_v2 import receipt

    result = receipt()
    height = 35 + smoke.math.pi
    vertices = [0.,0.,0.,20.,0.,0.,0.,height,0.,20.,height,0.,40.,0.,0.,40.,height,0.]
    result.update(source_sha256=case["source_sha256"], vertices=vertices)
    result["bend_lines"][0]["feature_id"] = case["candidates"][0]["feature_id"]
    result["flange_surfaces"][0].update(feature_id=case["candidates"][0]["feature_id"], vertices=list(vertices))
    result["diagnostics"].update(transport_version=2, source_bundle_sha256=bundle,
        candidate_binding=copy.deepcopy(case["candidates"]),
        candidate_input_sha256=smoke._hash(smoke._json(case["candidates"])),
        limits=dict(smoke.worker.LIMITS), elapsed_seconds=.2, peak_process_group_rss_kib=100,
        cache_hit=False)
    return result


def test_actual_contract_and_analytical_oracle_are_checked_without_native_geometry():
    case = {**smoke.case_definition("L90"), "source_sha256": "a" * 64}
    result = visual_receipt(case, "b" * 64)
    observed = smoke.assert_preview(case, result, "b" * 64)
    assert observed["flat_area_mm2"] == case["oracle"]["flat_area_mm2"]
    assert observed["flat_bbox_mm"] == case["oracle"]["sorted_flat_bbox_mm"]
    assert observed["bend_count"] == 1


@pytest.mark.parametrize("failure", ["bbox-area", "source", "bundle", "candidate", "limits", "elapsed", "rss", "release"])
def test_actual_contract_oracles_reject_adversarial_receipts(failure):
    case = {**smoke.case_definition("L90"), "source_sha256": "a" * 64}
    result = visual_receipt(case, "b" * 64)
    if failure == "bbox-area":
        for geometry in (result["vertices"], result["flange_surfaces"][0]["vertices"]):
            for index in (7, 10, 16):
                geometry[index] += .1
    elif failure == "source":
        result["source_sha256"] = "c" * 64
    elif failure == "bundle":
        result["diagnostics"]["source_bundle_sha256"] = "c" * 64
    elif failure == "candidate":
        result["diagnostics"]["candidate_input_sha256"] = "c" * 64
    elif failure == "limits":
        result["diagnostics"]["limits"]["wall_seconds"] = 999
    elif failure == "elapsed":
        result["diagnostics"]["elapsed_seconds"] = smoke.math.inf
    elif failure == "rss":
        result["diagnostics"]["peak_process_group_rss_kib"] = 0
    else:
        result["manufacturing_ready"] = True
    with pytest.raises(ValueError):
        smoke.assert_preview(case, result, "b" * 64)
