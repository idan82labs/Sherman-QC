"""Source-only mocked five-part custody; all JSON is authored under tmp_path.

Native imports are blocked by the reported harness. Live dependency discovery,
worker execution and original raw-root operations are always forbidden here.
Numeric charts test composed admission, not native or physical qualification.
"""
import ast
import copy
import json
from pathlib import Path

import pytest

from backend import vendor_unfold_preview_v4 as engine
from backend.vendor_unfold_contract_v3 import PreviewContractError
from scripts import run_vendor_unfold_five_part_candidate_v4 as runner
from tests.test_vendor_unfold_preview_v4 import quarter


RUNTIME = {"/authored/v4-wrapper-source": "b"*64}
REAL_REVIEW = runner._review


@pytest.fixture(autouse=True)
def custody(monkeypatch):
    monkeypatch.setattr(runner.worker, "_source_pins", lambda: RUNTIME.copy())
    monkeypatch.setattr(runner.worker, "_cache", {})
    monkeypatch.setattr(runner.worker, "_cache_bytes", 0)
    monkeypatch.setattr(runner.worker, "build_unfold_preview_bounded",
                        lambda *args: pytest.fail("Only an explicitly mocked wrapper may run"))
    # Guard lexical raw-root access; the worker mock may receive its path but
    # must not open/stat/resolve/search anything there.
    for name in ("open", "stat", "lstat", "resolve", "iterdir", "glob", "rglob"):
        original = getattr(Path, name)
        def guard(path, *args, _original=original, **kwargs):
            if path.is_relative_to(runner.RAW_ROOT):
                pytest.fail("Original CAD operations are forbidden")
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, name, guard)


def _post(bundle):
    return {"schema": runner.POST_SCHEMA, "disposition": runner.POST_DISPOSITION,
        "author": "authored-main-agent", "reviewer": "authored-independent-agent", "blocking_findings": [],
        "evaluation_freeze_sha256": "a"*64, "evaluation_record_sha256": "c"*64,
        "wrapper_source_bundle_sha256": bundle, "source_archive_sha256": "d"*64,
        "case_count": 2, "checks_passed": 6, "qualification_scope": runner.POST_SCOPE}


def _write(path, payload):
    path.write_bytes(runner.helpers._json(payload)+b"\n")
    return runner.helpers._hash(path.read_bytes())


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    fixtures = tmp_path/"authored-derived-fixtures"
    fixtures.mkdir()
    monkeypatch.setattr(runner, "FIXTURE_ROOT", fixtures)
    _, _, candidates = quarter()
    for number, _, digest in runner.APPROVED:
        fixture = {"part_number": number, "cad_sha256": digest,
            "candidates": copy.deepcopy(candidates), "geometry_source": "AUTHORED_NUMERIC_MOCK",
            "unfold_preview": runner.worker.unavailable("AUTHORED_BASELINE_UNKNOWN", digest),
            "original_authored_sentinel": number}
        _write(fixtures/(number+".json"), fixture)
    attempt = tmp_path/"new-candidate-attempt"
    attempt.mkdir()
    post = tmp_path/"authored-new-post.json"
    post_hash = _write(post, _post(runner.helpers._hash(runner.helpers._json(RUNTIME))))
    frozen = runner.freeze(attempt, post, post_hash)
    monkeypatch.setattr(runner, "_review", lambda *args: {"authored_mock_pre": True})
    return frozen, attempt, post, post_hash


def _execute(prepared):
    frozen, attempt, post, digest = prepared
    return runner.execute(frozen, attempt, post, digest, attempt/"authored-pre.json", attempt/"authored-source.tar")


def _available(case, bundle):
    payload = engine._develop_faces(*quarter(), case["raw_sha256"])
    payload["diagnostics"].update(transport_version=3, cache_profile=runner.worker.PROFILE,
        v4_engine_source_sha256=runner.worker.ENGINE_SHA256,
        candidate_binding=copy.deepcopy(case["candidates"]), candidate_input_sha256=case["candidate_sha256"],
        source_bundle_sha256=bundle, limits=dict(runner.worker.LIMITS),
        peak_process_group_rss_kib=100, elapsed_seconds=.2, cache_hit=False)
    return payload


def _wrapper(monkeypatch, frozen, *, available=False, mutate=None, advance=None):
    calls = []
    by_hash = {case["raw_sha256"]: case for case in frozen["cases"]}
    def mocked(path, digest, candidates):
        case = by_hash[digest]
        assert path == runner.RAW_ROOT/case["raw_filename"]
        assert candidates == case["candidates"]
        calls.append(case["part_number"])
        payload = (_available(case, frozen["wrapper_source_bundle_sha256"]) if available
                   else runner.worker.unavailable("AUTHORED_OPERATIONAL_UNAVAILABLE", digest))
        if mutate is not None:
            mutate(payload, len(calls))
        if advance is not None:
            advance(len(calls))
        return payload
    monkeypatch.setattr(runner.worker, "build_unfold_preview_bounded", mocked)
    return calls


def _events(frozen):
    return [json.loads(line) for line in (Path(frozen["results_directory"])/"execution.jsonl").read_text().splitlines()]


def test_import_has_no_native_or_runtime_discovery_entrypoint():
    tree = ast.parse(Path(runner.__file__).read_text())
    imports = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    imports += [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert not any(name.split(".")[0] in {"OCP", "FreeCAD", "Part", "numpy"} for name in imports)
    assert not hasattr(runner, "generate")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {"_source_pins", "build_unfold_preview_bounded"} for node in tree.body)


def test_freeze_exact_five_and_actual_wrapper_subset(prepared):
    frozen, _, _, _ = prepared
    assert [(row["part_number"], row["raw_filename"], row["raw_sha256"]) for row in frozen["cases"]] == list(runner.APPROVED)
    assert frozen["wrapper_source_pins"] == RUNTIME
    assert frozen["wrapper_source_bundle_sha256"] == runner.helpers._hash(runner.helpers._json(RUNTIME))
    assert frozen["source_pins"] != RUNTIME
    assert set(frozen["frontend_source_identities"]) == set(runner.FE_SOURCES)
    assert frozen["physical_inputs_read_during_freeze"] is False
    assert frozen["frontend_validator_executed"] is False
    assert frozen["expected_gold"] is None
    assert frozen["site_or_default_activation_authorized"] is False
    assert all(case["baseline"]["bend_count"] == case["baseline"]["triangle_count"] == 0 for case in frozen["cases"])


@pytest.mark.parametrize("change", [
    lambda p: p.update(schema="SHERMAN_VENDOR_UNFOLD_V2_RUNTIME_POST_V1"),
    lambda p: p.update(accepted=True), lambda p: p.pop("evaluation_record_sha256"),
    lambda p: p.update(reviewer=p["author"]), lambda p: p.update(author="  "),
    lambda p: p.update(blocking_findings=["block"]), lambda p: p.update(case_count=True),
    lambda p: p.update(checks_passed=True), lambda p: p.update(case_count=5),
    lambda p: p.update(wrapper_source_bundle_sha256="f"*64),
    lambda p: p.update(qualification_scope="GENERAL_CAD_SUPPORT"),
    lambda p: p.update(source_archive_sha256="not-a-hash"),
])
def test_qualification_post_closed_typed_and_current(tmp_path, change):
    bundle = runner.helpers._hash(runner.helpers._json(RUNTIME))
    post = _post(bundle)
    change(post)
    path = tmp_path/"authored-post.json"
    digest = _write(path, post)
    with pytest.raises(ValueError, match="CURRENT_V4_NATIVE_RUNTIME_POST_REQUIRED"):
        runner._qualification_post(path, digest, bundle)


def test_post_hash_verified_before_parsing(tmp_path):
    path = tmp_path/"authored-post.json"
    path.write_bytes(b"not JSON")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        runner._qualification_post(path, "a"*64, "b"*64)


@pytest.mark.parametrize("change", [
    lambda p: p.update(part_number="not-approved"), lambda p: p.update(cad_sha256="f"*64),
    lambda p: p.update(candidates=[]),
    lambda p: p["unfold_preview"].update(manufacturing_ready=True),
    lambda p: p["unfold_preview"].update(vertices=[0., 0., 0.]),
    lambda p: p["unfold_preview"].update(reason=""),
])
def test_released_fixture_identity_and_unknown_baseline(prepared, change):
    frozen, _, _, _ = prepared
    case = frozen["cases"][0]
    path = Path(case["fixture_path"])
    payload = json.loads(path.read_bytes())
    change(payload)
    _write(path, payload)
    with pytest.raises((ValueError, PreviewContractError)):
        runner._fixture(case["part_number"], case["raw_sha256"])


def test_unknown_execution_is_complete_not_support_and_fixtures_stay_identical(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    before = {row["fixture_path"]: Path(row["fixture_path"]).read_bytes() for row in frozen["cases"]}
    calls = _wrapper(monkeypatch, frozen)
    result = _execute(prepared)
    assert result["status"] == "COMPLETE_EXPLORATORY_EXECUTION"
    assert result["wrapper_cases_completed"] == 5
    assert result["geometry_available_cases"] == result["native_receipt_verified_cases"] == 0
    assert result["accuracy"] is None and result["manufacturing_ready"] is False
    assert result["physical_inputs_read"] is None and result["worker_source_access_attempted"] is True
    assert calls == [row[0] for row in runner.APPROVED]
    for row in result["cases"]:
        assert row["disposition"]["bend_count"] == row["disposition"]["triangle_count"] == 0
        assert row["disposition"]["support_status"] == "UNKNOWN_OR_UNAVAILABLE_NOT_A_DEFECT"
        assert runner.helpers._hash(Path(row["receipt"]["path"]).read_bytes()) == row["receipt"]["payload_sha256"]
        proposed = json.loads(Path(row["proposal"]["path"]).read_bytes())
        assert proposed["original_authored_sentinel"] == row["part_number"]
    assert {path: Path(path).read_bytes() for path in before} == before
    assert [event["event"] for event in _events(frozen)] == ["START", *(["CASE"]*5), "TERMINAL"]


def test_pure_available_composed_receipts_do_not_claim_accuracy(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    _wrapper(monkeypatch, frozen, available=True)
    result = _execute(prepared)
    assert result["status"] == "COMPLETE_EXPLORATORY_EXECUTION"
    assert result["geometry_available_cases"] == result["native_receipt_verified_cases"] == 5
    assert all(row["disposition"]["candidate_count"] == 1 and row["disposition"]["accuracy"] is None for row in result["cases"])
    assert result["site_or_default_activation_authorized"] is False


@pytest.mark.parametrize("change", [
    lambda p: p.update(source_sha256="f"*64),
    lambda p: p.update(manufacturing_ready=True),
    lambda p: p["diagnostics"].update(mechanism_version=3),
    lambda p: p["diagnostics"].update(transport_version=True),
    lambda p: p["diagnostics"].update(v4_engine_source_sha256="f"*64),
    lambda p: p["diagnostics"].update(candidate_input_sha256="f"*64),
    lambda p: p["diagnostics"].update(source_bundle_sha256="f"*64),
    lambda p: p["diagnostics"].update(cache_hit=True),
    lambda p: p["diagnostics"].update(peak_process_group_rss_kib=True),
    lambda p: p["diagnostics"].update(peak_process_group_rss_kib=runner.worker.MAX_RSS_KIB+1),
    lambda p: p["diagnostics"].update(elapsed_seconds=60.),
    lambda p: p["diagnostics"]["limits"].update(wall_seconds=61),
    lambda p: p["bend_lines"][0].update(feature_id="unknown"),
    lambda p: p["diagnostics"].pop("candidate_binding"),
])
def test_malformed_available_receipt_retained_before_failure(prepared, monkeypatch, change):
    frozen, _, _, _ = prepared
    _wrapper(monkeypatch, frozen, available=True, mutate=lambda payload, _: change(payload))
    result = _execute(prepared)
    assert result["status"] == "FAILED"
    assert all(row["execution_status"] == "FAILED" and row["proposal"] is None for row in result["cases"])
    for row in result["cases"]:
        raw = Path(row["receipt"]["path"]).read_bytes()
        assert runner.helpers._hash(raw) == row["receipt"]["payload_sha256"]
        assert len(raw) == row["receipt"]["payload_bytes"]
    assert len(_events(frozen)) == 7


def test_partial_native_unavailable_is_not_operational_success(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    _wrapper(monkeypatch, frozen, mutate=lambda payload, _: payload["diagnostics"].update(source_bundle_sha256="b"*64))
    result = _execute(prepared)
    assert all(row["reason"] == "INCOMPLETE_PHYSICAL_NATIVE_RECEIPT_BINDINGS" for row in result["cases"])


def test_overbudget_actual_receipt_keeps_observed_hash(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    _wrapper(monkeypatch, frozen)
    monkeypatch.setitem(runner.LIMITS, "receipt_bytes_per_case", 1)
    # Keep frozen comparison stable while testing only the retention bound.
    monkeypatch.setattr(runner, "freeze", lambda *args: frozen)
    result = _execute(prepared)
    assert all(row["observed_receipt"]["payload_bytes"] > 1 and "receipt" not in row for row in result["cases"])
    assert result["total_receipt_bytes"] == 0


def test_clock_exhausted_in_pins_before_first_admission(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original = runner._pin_sets
    def delayed_pins():
        value = original()
        clock[0] = 500.
        return value
    monkeypatch.setattr(runner, "_pin_sets", delayed_pins)
    result = _execute(prepared)
    assert result["status"] == "FAILED" and result["physical_inputs_read"] is False
    assert all(row["execution_status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in result["cases"])
    assert [event["event"] for event in _events(frozen)] == ["START", *(["CASE"]*5), "TERMINAL"]


def test_between_case_exhaustion_never_calls_later_cases(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    calls = _wrapper(monkeypatch, frozen, advance=lambda _: clock.__setitem__(0, 361.))
    result = _execute(prepared)
    assert len(calls) == 1
    assert result["cases"][0]["execution_status"] == "COMPLETED"
    assert all(row["execution_status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in result["cases"][1:])
    assert result["status"] == "FAILED"


def test_exact_reservation_boundary_360_seconds_admits(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original = runner._pin_sets
    def delayed_pins():
        value = original()
        clock[0] = 360.
        return value
    monkeypatch.setattr(runner, "_pin_sets", delayed_pins)
    calls = _wrapper(monkeypatch, frozen)
    assert _execute(prepared)["status"] == "COMPLETE_EXPLORATORY_EXECUTION"
    assert len(calls) == 5


def test_final_clock_420_fails_even_after_five_completed_receipts(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    _wrapper(monkeypatch, frozen, advance=lambda count: clock.__setitem__(0, 420. if count == 5 else 0.))
    result = _execute(prepared)
    assert result["wrapper_cases_completed"] == 5
    assert result["status"] == "FAILED"
    assert _events(frozen)[-1]["elapsed_seconds"] == 420.


def test_interrupt_retains_five_dispositions_and_terminal(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    def interrupted(*args):
        raise KeyboardInterrupt("authored interrupt")
    monkeypatch.setattr(runner.worker, "build_unfold_preview_bounded", interrupted)
    result = _execute(prepared)
    assert result["cases"][0]["execution_status"] == "FAILED"
    assert all(row["execution_status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in result["cases"][1:])
    assert [event["event"] for event in _events(frozen)] == ["START", *(["CASE"]*5), "TERMINAL"]


def test_terminal_attempt_cannot_retry_or_overwrite(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    calls = _wrapper(monkeypatch, frozen)
    _execute(prepared)
    before = (Path(frozen["results_directory"])/"execution.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        _execute(prepared)
    assert len(calls) == 5
    assert (Path(frozen["results_directory"])/"execution.jsonl").read_bytes() == before


def test_freeze_mismatch_stops_before_output_or_wrapper(prepared):
    frozen, attempt, post, digest = prepared
    malformed = copy.deepcopy(frozen)
    malformed["cases"] = malformed["cases"][:-1]
    with pytest.raises(ValueError, match="FREEZE_MISMATCH"):
        runner.execute(malformed, attempt, post, digest, attempt/"pre", attempt/"archive")
    assert not Path(frozen["results_directory"]).exists()


def test_fixture_mutation_during_preparation_refuses_exact_case(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    original = runner._fixture
    calls = [0]
    def changed(number, digest, **kwargs):
        values = original(number, digest, **kwargs)
        fixture, metadata = values[:2]
        calls[0] += 1
        if calls[0] > 5:
            metadata["fixture_sha256"] = "f"*64
        return (fixture, metadata, values[2]) if kwargs.get("include_raw") else (fixture, metadata)
    monkeypatch.setattr(runner, "_fixture", changed)
    result = _execute(prepared)
    assert all(row["execution_status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in result["cases"])
    assert result["physical_inputs_read"] is False


def test_source_mutation_after_call_fails_terminal_without_dropping_receipt(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    original = runner._pin_sets
    calls = [0]
    def pins():
        union, runtime = original()
        calls[0] += 1
        if calls[0] >= 7:
            union["/authored/new-source"] = "f"*64
        return union, runtime
    monkeypatch.setattr(runner, "_pin_sets", pins)
    _wrapper(monkeypatch, frozen)
    result = _execute(prepared)
    assert result["wrapper_cases_completed"] == 5 and result["status"] == "FAILED"
    assert result["reason"] == "SOURCE_CHANGED_DURING_PHYSICAL_CANDIDATE"


def test_raw_or_fixture_output_destination_refused_before_raw_access(tmp_path, monkeypatch):
    for path in (runner.RAW_ROOT/"never-created", runner.FIXTURE_ROOT/"never-created"):
        with pytest.raises(ValueError, match="SEPARATE_NEW_ATTEMPT"):
            runner._attempt_directory(path)
    link = tmp_path/"output-alias"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="SYMLINK_ATTEMPT"):
        runner._attempt_directory(link/"new-output")


def test_pre_exact_eight_keys_and_archive_binding(prepared, tmp_path, monkeypatch):
    frozen, _, _, _ = prepared
    monkeypatch.setattr(runner, "_review", REAL_REVIEW)
    monkeypatch.setattr(runner.helpers, "_verify_archive", lambda *args: "d"*64)
    review = {"schema": runner.PRE_SCHEMA, "disposition": "GREEN", "author": "authored-main-agent",
        "reviewer": "authored-independent-agent", "blocking_findings": [],
        "freeze_sha256": runner.helpers._hash(runner.helpers._json(frozen)),
        "source_bundle_sha256": runner.helpers._hash(runner.helpers._json(frozen["source_pins"])),
        "source_archive_sha256": "d"*64}
    path = tmp_path/"authored-pre.json"
    _write(path, review)
    assert runner._review(frozen, path, tmp_path/"never-read-archive")["source_archive_sha256"] == "d"*64
    review["reviewer"] = review["author"]
    _write(path, review)
    with pytest.raises(ValueError, match="EXACT_INDEPENDENT_FIVE_PART_PRE"):
        runner._review(frozen, path, tmp_path/"never-read-archive")


@pytest.mark.parametrize("change", [
    lambda p: p.update(extra="not-closed"), lambda p: p.pop("source_archive_sha256"),
    lambda p: p.update(schema="OLD_PRE"), lambda p: p.update(disposition="ACCEPTED"),
    lambda p: p.update(author=""), lambda p: p.update(reviewer=None),
    lambda p: p.update(blocking_findings=["blocking"]),
    lambda p: p.update(freeze_sha256="f"*64), lambda p: p.update(source_bundle_sha256="f"*64),
    lambda p: p.update(source_archive_sha256="f"*64),
])
def test_pre_closed_schema_all_bindings(prepared, tmp_path, monkeypatch, change):
    frozen, _, _, _ = prepared
    monkeypatch.setattr(runner, "_review", REAL_REVIEW)
    monkeypatch.setattr(runner.helpers, "_verify_archive", lambda *args: "d"*64)
    review = {"schema": runner.PRE_SCHEMA, "disposition": "GREEN", "author": "authored-main-agent",
        "reviewer": "authored-independent-agent", "blocking_findings": [],
        "freeze_sha256": runner.helpers._hash(runner.helpers._json(frozen)),
        "source_bundle_sha256": runner.helpers._hash(runner.helpers._json(frozen["source_pins"])),
        "source_archive_sha256": "d"*64}
    change(review)
    path = tmp_path/"authored-malformed-pre.json"
    _write(path, review)
    with pytest.raises(ValueError):
        runner._review(frozen, path, tmp_path/"never-read-archive")


def test_native_unavailable_still_verifies_transport_and_counts_zero(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    def unavailable(payload, _):
        payload.update(status="UNAVAILABLE", reason="AUTHORED_NATIVE_SUPPORT_UNKNOWN",
                       vertices=[], triangles=[], bend_lines=[], flange_surfaces=[])
    _wrapper(monkeypatch, frozen, available=True, mutate=unavailable)
    result = _execute(prepared)
    assert result["status"] == "COMPLETE_EXPLORATORY_EXECUTION"
    assert result["geometry_available_cases"] == 0 and result["native_receipt_verified_cases"] == 5
    assert all(row["disposition"]["bend_count"] == row["disposition"]["triangle_count"] == 0 for row in result["cases"])


@pytest.mark.parametrize("change", [
    lambda p: p.update(source_sha256="f"*64),
    lambda p: p.update(vertices=[0., 0., 0.]),
    lambda p: p["diagnostics"].update(cache_profile="OLD_PROFILE"),
    lambda p: p["diagnostics"].update(mechanism_version=True),
    lambda p: p["diagnostics"].update(transport_version=4),
])
def test_operational_unavailable_closed_and_typed(prepared, monkeypatch, change):
    frozen, _, _, _ = prepared
    _wrapper(monkeypatch, frozen, mutate=lambda payload, _: change(payload))
    result = _execute(prepared)
    assert result["status"] == "FAILED" and result["wrapper_cases_completed"] == 0
    assert all("receipt" in row and row["execution_status"] == "FAILED" for row in result["cases"])


def test_fixture_helper_cannot_read_a_sixth_path(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.helpers, "_read", lambda *args: pytest.fail("Unapproved read attempted"))
    with pytest.raises(ValueError, match="ONLY_EXACT_APPROVED"):
        runner._fixture("../unapproved", "a"*64)


def test_auxiliary_gates_never_read_raw_or_released_payloads():
    with pytest.raises(ValueError, match="SEPARATE_NEW_ATTEMPT"):
        runner._qualification_post(runner.RAW_ROOT/runner.APPROVED[0][1], "a"*64, "b"*64)


def test_pins_pin_exact_approved_source_constants_without_importing_legacy(prepared):
    pins, runtime = runner._pin_sets()
    assert runtime == RUNTIME
    assert set(runtime).issubset(pins)
    expected = (*runner.OWN_SOURCES, runner.HELPER_SOURCE, runner.APPROVAL_SOURCE, *runner.FE_SOURCES)
    for name in expected:
        path = (runner.ROOT/name).resolve()
        assert pins[str(path)] == runner.helpers._hash(path.read_bytes())


def test_fresh_cache_is_mandatory_before_outputs(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    monkeypatch.setattr(runner.worker, "_cache", {"authored": b"cached"})
    with pytest.raises(ValueError, match="FRESH_PROCESS_CACHE"):
        _execute(prepared)
    assert not Path(frozen["results_directory"]).exists()


def test_post_pin_verification_is_in_final_elapsed(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original = runner._pin_sets
    seen = [0]
    def pins():
        seen[0] += 1
        value = original()
        if seen[0] == 7:
            clock[0] = 420.
        return value
    monkeypatch.setattr(runner, "_pin_sets", pins)
    calls = _wrapper(monkeypatch, frozen)
    result = _execute(prepared)
    assert len(calls) == 5 and result["wrapper_cases_completed"] == 5
    assert result["status"] == "FAILED" and result["elapsed_seconds"] == 420.


def test_proposal_limit_preserves_actual_receipt(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    monkeypatch.setitem(runner.LIMITS, "proposal_bytes_per_case", 1)
    monkeypatch.setattr(runner, "freeze", lambda *args: frozen)
    _wrapper(monkeypatch, frozen)
    result = _execute(prepared)
    assert result["total_proposal_bytes"] == 0
    assert all("receipt" in row and row["proposal"] is None and row["execution_status"] == "FAILED" for row in result["cases"])


def test_exclusive_protocol_never_overwrites(tmp_path):
    path = tmp_path/"authored-freeze.json"
    runner.helpers.save_new(path, {"authored": "first"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.helpers.save_new(path, {"authored": "second"})
    assert path.read_bytes() == before


def test_protocol_and_ledger_have_explicit_byte_budgets(prepared, monkeypatch, tmp_path):
    _, attempt, post, digest = prepared
    monkeypatch.setitem(runner.LIMITS, "protocol_bytes", 1)
    with pytest.raises(ValueError, match="PROTOCOL_BYTE_BUDGET"):
        runner.freeze(attempt, post, digest)
    monkeypatch.setitem(runner.LIMITS, "ledger_bytes_per_event", 1)
    path = tmp_path/"authored-ledger.jsonl"
    with path.open("xb") as stream:
        with pytest.raises(ValueError, match="LEDGER_BYTE_BUDGET"):
            runner._append_event(stream, {"event": "AUTHORED"})
    assert path.read_bytes() == b""


def test_all_five_exact_derived_inputs_retained_before_any_call(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    before = {case["part_number"]: Path(case["fixture_path"]).read_bytes() for case in frozen["cases"]}
    # Noncanonical whitespace must be kept byte-for-byte, not normalized.
    first = Path(frozen["cases"][0]["fixture_path"])
    first.write_bytes(b" \n"+first.read_bytes()+b" \n")
    frozen = runner.freeze(prepared[1], prepared[2], prepared[3])
    prepared = (frozen, *prepared[1:])
    before[frozen["cases"][0]["part_number"]] = first.read_bytes()
    folder = Path(frozen["results_directory"])
    calls = []
    def mocked(path, digest, candidates):
        for number, raw in before.items():
            assert (folder/(number+".released-input.json")).read_bytes() == raw
        calls.append(digest)
        return runner.worker.unavailable("AUTHORED_UNKNOWN", digest)
    monkeypatch.setattr(runner.worker, "build_unfold_preview_bounded", mocked)
    result = _execute(prepared)
    assert len(calls) == 5
    assert result["total_retained_input_bytes"] == sum(map(len, before.values()))
    for row in result["cases"]:
        snapshot = row["input_snapshot"]
        raw = Path(snapshot["path"]).read_bytes()
        assert raw == before[row["part_number"]]
        assert runner.helpers._hash(raw) == snapshot["sha256"]


def test_failed_case_retains_original_derived_input_even_without_proposal(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    _wrapper(monkeypatch, frozen, mutate=lambda payload, _: payload.update(manufacturing_ready=True))
    result = _execute(prepared)
    assert all(row["proposal"] is None and row["execution_status"] == "FAILED" for row in result["cases"])
    for case, row in zip(frozen["cases"], result["cases"]):
        assert Path(row["input_snapshot"]["path"]).read_bytes() == Path(case["fixture_path"]).read_bytes()


def test_snapshot_failure_refuses_all_calls_and_retains_partial_input_truth(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    original = runner._fixture
    def changed(number, digest, **kwargs):
        result = original(number, digest, **kwargs)
        if kwargs.get("include_raw") and number == frozen["cases"][2]["part_number"]:
            result[1]["fixture_sha256"] = "f"*64
        return result
    monkeypatch.setattr(runner, "_fixture", changed)
    result = _execute(prepared)
    assert result["physical_inputs_read"] is False and result["worker_source_access_attempted"] is False
    assert all(row["execution_status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in result["cases"])
    assert all(row["input_snapshot"] is not None for row in result["cases"][:2])
    assert all(row["input_snapshot"] is None for row in result["cases"][2:])
    assert len(_events(frozen)) == 7


def test_snapshot_time_is_in_campaign_and_prevents_late_call(prepared, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original = runner._retain_inputs
    def delayed(*args):
        original(*args)
        clock[0] = 500.
    monkeypatch.setattr(runner, "_retain_inputs", delayed)
    result = _execute(prepared)
    assert result["physical_inputs_read"] is False
    assert all(row["input_snapshot"] is not None and row["execution_status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in result["cases"])


def test_snapshot_budget_prevents_any_cad_admission(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    monkeypatch.setitem(runner.LIMITS, "retained_input_bytes_per_case", 1)
    monkeypatch.setattr(runner, "freeze", lambda *args: frozen)
    result = _execute(prepared)
    assert result["physical_inputs_read"] is False and result["total_retained_input_bytes"] == 0
    assert result["status"] == "FAILED"


def test_snapshot_write_failure_records_attempt_and_blocks_all_calls(prepared, monkeypatch):
    frozen, _, _, _ = prepared
    original = Path.open
    def denied(path, *args, **kwargs):
        if path.name == frozen["cases"][0]["part_number"]+".released-input.json":
            raise OSError("authored write denial")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", denied)
    result = _execute(prepared)
    assert result["worker_source_access_attempted"] is False
    first = result["cases"][0]
    assert first["input_snapshot"] is None
    assert first["input_snapshot_write_attempt"]["status"] == "WRITE_STARTED"
    assert first["input_snapshot_write_attempt"]["expected_sha256"] == frozen["cases"][0]["fixture_sha256"]
    assert len(_events(frozen)) == 7
