"""Two-case diagnostic of the terminal V11 40PASS/1FAIL gate, not qualification.

Source checks never call geometry. The explicit opt-in records exactly the two
original50-row cases, with the original .05mm declaration/policy/bank2 and one
public initializer call each. Pass-through observers do not change an argument,
returned object, query order or engine budget. All context/query records and the
full returned answer are retained, including rejected/absent slots. The original
positive assertions run only after inference, against that same answer; their
failure remains a separate FAIL, even when diagnostic recording succeeds.

Root owns prospective60wall/45workerCPU/1GiB/1MiB supervision and a fresh owned
artifact directory. Two full JSON artifacts, <=16MiB each; no truncation, partial
success, cleanup, second inference, native imports or physical inputs. Known pose
is used only in post-evaluation diagnostics, never in the initializer call.
"""

import ast
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "tests/test_cad_guided_plane_initialization_v11.py"
LEGACY = "tests/test_cad_guided_plane_initialization_v10.py"
RUNTIME = "backend/cad_guided_plane_initialization_v11.py"
SOURCE_HASHES = {
    ORIGINAL: "16a646404b4cf35975956b717a9143b52d8bca1e7ec5467f40096aefbc0edf1e",
    LEGACY: "4882b701af3806fd03ddde9bb0b1af823bd0124e859e6655207f683db1563eb0",
    RUNTIME: "31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e",
}
ORIGINAL_TEST = "test_actual_fifty_row_two_plane_point_zero_five_lane_required_positive"
OPT_IN = "SHERMAN_RUN_TWO_PLANE_V11_STAGE_DIAGNOSTIC"
ARTIFACT_ENV = "SHERMAN_TWO_PLANE_V11_STAGE_ARTIFACT_DIRECTORY"
RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
    process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
CASES = ("original_fifty_row_clean", "original_fifty_row_balanced_normal_noise")
MAX_SOURCE_BYTES = 256*1024
MAX_ARTIFACT_BYTES = 16*1024**2
MAX_TREE_NODES = 1_000_000
MAX_OBSERVATIONS_PER_CHANNEL = 128  # 64 original slots times two supports.
MAX_ROWS_PER_CHANNEL = 3200  # 64 times two times all25 original rows.


def _source(relative):
    with (ROOT/relative).open("rb") as stream:
        blob = stream.read(MAX_SOURCE_BYTES+1)
    assert len(blob) <= MAX_SOURCE_BYTES
    assert hashlib.sha256(blob).hexdigest() == SOURCE_HASHES[relative]
    return blob


def _function(tree, name):
    rows = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(rows) == 1
    return rows[0]


def _oracle_ast(tree):
    original = _function(tree, ORIGINAL_TEST)
    assert isinstance(original.body[1], ast.Assign)
    assert ast.unparse(original.body[1].targets[0]) == "answer"
    # Keep EVERY original post-inference assertion, in its original order.
    arguments = ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
        ("answer", "inputs", "nominal", "clean_observed", "legacy")],
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    return ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name="oracle",
        args=arguments, body=original.body[2:], decorator_list=[])], type_ignores=[]))


def _reviewed_capsules():
    # Called only inside the explicitly enabled diagnostic, never at collection.
    # Import the reviewed namespace modules, not backend/__init__/application.
    from backend import cad_guided_plane_initialization_v11 as runtime
    from backend import cad_guided_plane_initialization_v10 as prior_runtime

    blobs = {path: _source(path) for path in SOURCE_HASHES}
    old_tree, original_tree = ast.parse(blobs[LEGACY]), ast.parse(blobs[ORIGINAL])
    names = ("_rectangle", "_policy", "_assert_scope", "_assert_complete_prefilter",
             "_assert_complete_box_supports", "_action_error")
    old_namespace = dict(np=np, json=json, v5=prior_runtime)
    old_nodes = [_function(old_tree, name) for name in names]
    exec(compile(ast.Module(body=old_nodes, type_ignores=[]), str(ROOT/LEGACY), "exec"), old_namespace)
    legacy = SimpleNamespace(**{name: old_namespace[name] for name in names})
    namespace = dict(np=np, v11=runtime)
    constants = {"SPECS", "ROTATION", "TRANSLATION", "UNIT"}
    nodes = [node for node in original_tree.body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in constants]
    assert len(nodes) == 4
    nodes.append(_function(original_tree, "_two_rectangles"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/ORIGINAL), "exec"), namespace)
    exec(compile(_oracle_ast(original_tree), str(ROOT/ORIGINAL), "exec"), namespace)
    return runtime, legacy, namespace


class _Tree:
    """Bounded full-value JSON representation, not an array-descriptor projection.

    Ordinary dict/list metadata is unchanged; tuples become JSON lists. NumPy
    arrays carry all values, dtype, shape and raw-byte digest. Integer-key maps
    use explicit key/value entries to avoid silently changing their identities.
    This is diagnostic transport, not a claim to reconstruct arbitrary Python.
    """
    def __init__(self, maximum=MAX_TREE_NODES):
        self.maximum, self.nodes = maximum, 0

    def convert(self, value, depth=0):
        self.nodes += 1
        if self.nodes > self.maximum or depth > 64:
            raise ValueError("DIAGNOSTIC_TREE_BUDGET_EXCEEDED")
        if type(value) is np.ndarray:
            if value.dtype.kind not in "biufU" or value.size > 1_000_000:
                raise ValueError("UNSUPPORTED_DIAGNOSTIC_ARRAY")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError("NONFINITE_DIAGNOSTIC_ARRAY")
            return dict(array_dtype=value.dtype.str, array_shape=list(value.shape),
                array_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                array_values=self.convert(value.tolist(), depth+1))
        if type(value) in (np.float64, np.float32):
            value = float(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif type(value) is np.bool_:
            value = bool(value)
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("NONFINITE_DIAGNOSTIC_SCALAR")
            return value
        if type(value) in (list, tuple):
            return [self.convert(row, depth+1) for row in value]
        if type(value) is dict:
            if all(type(key) is str for key in value):
                return {key: self.convert(row, depth+1) for key, row in value.items()}
            if all(type(key) in (str, int) for key in value):
                return dict(integer_or_mixed_key_entries=[
                    [key, self.convert(row, depth+1)] for key, row in value.items()])
        raise ValueError("UNSUPPORTED_DIAGNOSTIC_VALUE")


def _error(error):
    try:
        message = str(error)[:4096]
    except Exception:
        message = "EXCEPTION_MESSAGE_UNAVAILABLE"
    return dict(type=type(error).__name__, message=message)


class _Observations:
    """Pass-through attempted/returned/raised ledger, separate from engine work.

    A requested dispatch can fail preflight without a charged runtime call. A
    raised method never returns its internal primitive payload: retain the
    attempt/exception/counters, but do not claim full nested evidence exists.
    Recording failures cannot replace an ordinary engine result or exception.
    """
    KEYS = {
        "box": ("box_screen_calls", "box_screen_rows", "completed_box_screen_calls",
                "completed_box_screen_rows"),
        "finite_query": ("query_calls", "query_rows", "completed_query_calls", "completed_query_rows"),
    }

    def __init__(self, tree, maximum_calls=MAX_OBSERVATIONS_PER_CHANNEL,
                 maximum_rows=MAX_ROWS_PER_CHANNEL):
        self.tree, self.maximum_calls, self.maximum_rows = tree, maximum_calls, maximum_rows
        self.records = {key: [] for key in self.KEYS}
        self.counts = {key: dict(attempted_calls=0, attempted_rows=0, returned_calls=0,
            returned_rows=0, raised_calls=0, raised_rows=0, runtime_charged_calls=0,
            runtime_charged_rows=0, runtime_completed_calls=0, runtime_completed_rows=0)
            for key in self.KEYS}
        self.errors, self.error_count = [], 0

    def fault(self, error):
        self.error_count += 1
        if len(self.errors) < 8:
            self.errors.append(_error(error))

    def snapshot(self, owner, channel):
        try:
            values = [owner.statistics[key] for key in self.KEYS[channel]]
            if not all(type(value) is int and value >= 0 for value in values):
                raise ValueError("INVALID_RUNTIME_OBSERVATION_COUNTER")
            return values
        except Exception as error:
            self.fault(error)
            return None

    def begin(self, channel, owner, face, points, gate):
        count, stats = len(points), self.counts[channel]
        stats["attempted_calls"] += 1
        stats["attempted_rows"] += count
        before, record = self.snapshot(owner, channel), None
        try:
            if stats["attempted_calls"] > self.maximum_calls:
                raise ValueError("DIAGNOSTIC_OBSERVATION_COUNT_EXCEEDED")
            if stats["attempted_rows"] > self.maximum_rows:
                raise ValueError("DIAGNOSTIC_OBSERVATION_ROWS_EXCEEDED")
            record = dict(attempt_index=stats["attempted_calls"]-1,
                request=self.tree.convert(dict(cad_face_id=face, points=points, distance_gate_mm=gate)),
                runtime_before=before, runtime_after=None, disposition="ATTEMPTED",
                answer=None, exception=None, nested_primitive_evidence="NOT_RETURNED_YET")
            self.records[channel].append(record)
        except Exception as error:
            self.fault(error)
        return count, before, record

    def finish(self, channel, owner, token, *, answer=None, error=None):
        count, before, record = token
        stats = self.counts[channel]
        after = self.snapshot(owner, channel)
        event = "raised" if error is not None else "returned"
        stats[event+"_calls"] += 1
        stats[event+"_rows"] += count
        try:
            if before is None or after is None:
                raise ValueError("RUNTIME_OBSERVATION_COUNTER_UNAVAILABLE")
            delta = [a-b for a, b in zip(after, before)]
            if (delta[0] not in (0, 1) or delta[2] not in (0, 1)
                    or delta[1] != count*delta[0] or delta[3] != count*delta[2]
                    or delta[2] > delta[0]):
                raise ValueError("INVALID_RUNTIME_OBSERVATION_COUNTER_DELTA")
            for name, value in zip(("runtime_charged_calls", "runtime_charged_rows",
                                   "runtime_completed_calls", "runtime_completed_rows"), delta):
                stats[name] += value
            if record is not None:
                record["runtime_after"] = after
            if error is not None:
                if record is not None:
                    record.update(disposition="RAISED", exception=_error(error),
                        nested_primitive_evidence="UNAVAILABLE_METHOD_RAISED_BEFORE_RETURN")
                # Retained exception/counters are not the missing primitive result.
                self.fault(ValueError("RAISED_ATTEMPT_HAS_INCOMPLETE_NESTED_EVIDENCE"))
            elif record is not None:
                record.update(disposition="RETURNED", answer=self.tree.convert(answer),
                    nested_primitive_evidence="FULL_RETURNED_METHOD_PAYLOAD")
            if error is None and (type(answer) is not dict or answer.get("status") != "COMPLETE"):
                self.fault(ValueError("RETURNED_METHOD_PAYLOAD_NOT_COMPLETE"))
        except Exception as capture_error:
            self.fault(capture_error)
            # Do not leave a raised attempt looking merely pending if serialization
            # or telemetry failed. This metadata does not replace its exception.
            if record is not None:
                record.update(disposition="RAISED" if error is not None else "RETURNED_CAPTURE_INCOMPLETE",
                    exception=_error(error) if error is not None else None,
                    nested_primitive_evidence="INCOMPLETE_CAPTURE")

    def invoke(self, channel, original, owner, face, points, gate=None):
        token = self.begin(channel, owner, face, points, gate)
        try:
            answer = (original(owner, face, points, gate) if channel == "box"
                      else original(owner, face, points))
        except BaseException as error:
            self.finish(channel, owner, token, error=error)
            raise  # Preserve the identical original object, never a replacement.
        self.finish(channel, owner, token, answer=answer)
        return answer  # Preserve the identical original return object.

    def reconcile(self, runtime_statistics):
        details = {}
        for channel, keys in self.KEYS.items():
            stats = self.counts[channel]
            expected = [stats[key] for key in ("runtime_charged_calls", "runtime_charged_rows",
                                              "runtime_completed_calls", "runtime_completed_rows")]
            try:
                actual = [runtime_statistics[key] for key in keys]
                matched = (all(type(value) is int and value >= 0 for value in actual)
                    and actual == expected
                    and stats["attempted_calls"] == stats["returned_calls"]+stats["raised_calls"]
                    and stats["attempted_rows"] == stats["returned_rows"]+stats["raised_rows"]
                    and len(self.records[channel]) == stats["attempted_calls"]
                    and stats["returned_calls"] == stats["runtime_completed_calls"]
                    and stats["returned_rows"] == stats["runtime_completed_rows"])
            except Exception as error:
                self.fault(error)
                actual, matched = None, False
            details[channel] = dict(runtime_counter_keys=list(keys), observed_deltas=expected,
                runtime_final=actual, matched=matched)
            if not matched:
                self.fault(ValueError("DIAGNOSTIC_RUNTIME_COUNTER_RECONCILIATION_FAILED"))
        return dict(counts=self.counts, reconciliation=details, errors=self.errors,
            error_count=self.error_count, complete=self.error_count == 0)


def _pose_errors(answer, context, clean_observed, nominal, legacy):
    """Post-evaluation only; no truth-driven pose creation or inference retry."""
    helper = answer.get("two_plane_context") or {}
    groups = (("helper_candidate", helper.get("candidates", []), "rotation", "translation_mm"),
        ("generated_hypothesis", (context or {}).get("hypotheses", []), "rotation", "translation_mm"),
        ("retained_hypothesis", answer.get("hypotheses", []), "rotation", "translation_mm"),
        ("all_passing_group", answer.get("all_passing_start_groups", []),
         "initialized_rotation", "initialized_translation_mm"),
        ("selected_start_group", answer.get("start_groups", []),
         "initialized_rotation", "initialized_translation_mm"))
    output = []
    for origin, rows, rk, tk in groups:
        for index, row in enumerate(rows):
            entry = dict(origin=origin, index=index, pose_present=row.get(rk) is not None
                and row.get(tk) is not None, action_error_mm=None)
            if entry["pose_present"]:
                try:
                    entry["action_error_mm"] = legacy._action_error(
                        clean_observed, nominal, row[rk], row[tk])
                except Exception as error:
                    entry["error"] = _error(error)
            output.append(entry)
    return output


def _record_case(noisy, runtime, legacy, capsule, monkeypatch):
    inputs, nominal, clean_observed = capsule["_two_rectangles"](legacy, noisy)
    # The immutable source of this constructor and call is pinned above.
    policy = legacy._policy(50)
    search = runtime._search.PlaneSearchBudgetsV6(max_bank_planes=2)
    tree = _Tree()
    observations, prepared = _Observations(tree), []
    preparation_calls = 0
    original_prepare = runtime._prepare_triplets
    original_screen, original_query = runtime._FaceQueries.screen, runtime._FaceQueries.__call__

    def prepare(*args, **kwargs):
        nonlocal preparation_calls
        preparation_calls += 1
        context = original_prepare(*args, **kwargs)
        if len(prepared) < 1:
            prepared.append(context)  # Full generated arrays/records, no projection.
        return context

    def screen(self, face, points, gate):
        return observations.invoke("box", original_screen, self, face, points, gate)

    def query(self, face, points):
        return observations.invoke("finite_query", original_query, self, face, points)

    with monkeypatch.context() as observer:
        observer.setattr(runtime, "_prepare_triplets", prepare)
        observer.setattr(runtime._FaceQueries, "screen", screen)
        observer.setattr(runtime._FaceQueries, "__call__", query)
        # EXACT original public call, not _initialize or an alternate policy.
        answer = runtime.prefilter_plane_initializations_v11(*inputs,
            position_error_bounds_mm=.05, registration_policy=policy, unit_assumption=capsule["UNIT"],
            search_budgets=search)
    # Known construction and all oracle evaluations are strictly after the call.
    oracle = dict(status="NOT_EVALUATED", original_assertion_body_unchanged=True,
                  original_start_error_ceiling_mm=1e-7)
    try:
        capsule["oracle"](answer, inputs, nominal, clean_observed, legacy)
        oracle["status"] = "PASS"
    except Exception as error:
        oracle.update(status="FAIL", first_failure=_error(error))
    context = prepared[0] if prepared else None
    post = dict(nominal=nominal, clean_observed=clean_observed,
        construction_rotation=capsule["ROTATION"], construction_translation_mm=capsule["TRANSLATION"],
        expected_scan_to_cad_rotation=capsule["ROTATION"].T,
        expected_scan_to_cad_translation_mm=-capsule["TRANSLATION"]@capsule["ROTATION"],
        pose_errors=_pose_errors(answer, context, clean_observed, nominal, legacy))
    full = tree.convert(dict(inputs=dict(zip(("scan_points", "scan_triangle_raw_rows",
        "cad_triangles", "cad_face_ids", "cad_source_triangle_ids"), inputs)),
        full_position_error_bounds_mm=np.full(50, .05), registration_policy=asdict(policy),
        search_budgets=asdict(search), prepared_context=context, returned_answer=answer,
        post_evaluation=post))
    reconciliation = observations.reconcile(answer.get("statistics", {}))
    return dict(schema="two_plane_v11_stage_diagnostic_v1", case_name=CASES[int(noisy)],
        diagnostic_only=True, qualifies_engine=False, native_executed=False,
        observed_bend_count=None, production_use_allowed=False, source_hashes=SOURCE_HASHES,
        original_positive_oracle=oracle, full_evidence=full, observations=observations.records,
        capture=dict(preparation_calls=preparation_calls, captured_context_count=len(prepared),
            channel_accounting=reconciliation, tree_nodes=tree.nodes,
            complete=preparation_calls == 1 and len(prepared) == 1 and reconciliation["complete"]))


def _publish(directory, index, record, maximum=MAX_ARTIFACT_BYTES):
    assert type(index) is int and index in (0, 1)
    assert type(maximum) is int and 0 <= maximum <= MAX_ARTIFACT_BYTES
    payload = bytearray()
    for chunk in json.JSONEncoder(allow_nan=False, ensure_ascii=True, separators=(",", ":")).iterencode(record):
        part = chunk.encode("utf-8")
        if len(payload)+len(part) > maximum:
            raise ValueError("COMPLETE_CASE_ARTIFACT_BUDGET_EXCEEDED")
        payload.extend(part)
    path = directory/f"case_{index:02d}.json"
    with path.open("xb") as stream:
        assert stream.write(payload) == len(payload)
        stream.flush()
        os.fsync(stream.fileno())
    with path.open("rb") as stream:
        retained = stream.read(maximum+1)
    assert retained == payload
    return dict(name=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("relative", tuple(SOURCE_HASHES))
def test_exact_original_source_bindings(relative):
    assert _source(relative)


def test_original_oracle_is_every_unmodified_post_call_statement():
    tree = ast.parse(_source(ORIGINAL))
    expected = _function(tree, ORIGINAL_TEST).body[2:]
    assert [ast.dump(x, include_attributes=False) for x in _oracle_ast(tree).body[0].body] == [
        ast.dump(x, include_attributes=False) for x in expected]
    assert len(expected) == 14


def test_fixture_is_original_source_not_easier_new_geometry():
    tree = ast.parse(_source(ORIGINAL))
    fixture = ast.unparse(_function(tree, "_two_rectangles"))
    assert "legacy._rectangle(spec, 5)" in fixture and "legacy._rectangle(spec, 2)" in fixture
    assert "displacement = 0.0005 * np.outer(e, e).ravel()" in fixture
    assert "inset=" not in fixture
    assert "len(observed) == 50" in fixture and "len(raw) == 64" in fixture


def test_diagnostic_has_one_public_call_no_native_or_second_inference():
    tree = ast.parse(Path(__file__).read_text())
    node = _function(tree, "_record_case")
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "prefilter_plane_initializations_v11"]
    assert len(calls) == 1
    assert {kw.arg for kw in calls[0].keywords} == {
        "position_error_bounds_mm", "registration_policy", "unit_assumption", "search_budgets"}
    assert ast.literal_eval(calls[0].keywords[0].value) == .05
    assert "max_bank_planes=2" in ast.unparse(node)
    assert {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        and n.module == "backend"} == {
            "cad_guided_plane_initialization_v11", "cad_guided_plane_initialization_v10"}


def test_fixed_case_and_resource_accounting():
    assert len(CASES) == 2 and MAX_ARTIFACT_BYTES == 16*1024**2
    assert MAX_OBSERVATIONS_PER_CHANNEL == 64*2 and MAX_ROWS_PER_CHANNEL == 64*2*25
    assert RESOURCE_LIMITS == dict(wall_seconds=60, worker_cpu_seconds=45,
        process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)


def test_full_array_and_integer_identity_transport():
    points = np.array([[0., -0., .002], [1., 2., 3.]], dtype=np.float64)
    converted = _Tree().convert({7: points})["integer_or_mixed_key_entries"]
    assert converted[0][0] == 7
    row = converted[0][1]
    restored = np.asarray(row["array_values"], dtype=np.dtype(row["array_dtype"]))
    assert restored.tobytes() == points.tobytes()
    assert row["array_sha256"] == hashlib.sha256(points.tobytes()).hexdigest()


@pytest.mark.parametrize("value", (float("nan"), float("inf"), object()))
def test_nonrepresentable_evidence_is_not_silently_truncated(value):
    with pytest.raises(ValueError):
        _Tree().convert(value)


def test_tree_budget_refuses_whole_conversion():
    with pytest.raises(ValueError, match="DIAGNOSTIC_TREE_BUDGET_EXCEEDED"):
        _Tree(maximum=2).convert([1, 2])


def test_complete_artifact_real_owned_publication_and_existing_target_refusal(tmp_path):
    record = dict(original_positive_oracle=dict(status="FAIL"), full=[0., -0., 1.])
    receipt = _publish(tmp_path, 0, record)
    assert receipt["sha256"] == hashlib.sha256((tmp_path/receipt["name"]).read_bytes()).hexdigest()
    assert json.loads((tmp_path/receipt["name"]).read_bytes()) == record
    with pytest.raises(FileExistsError):
        _publish(tmp_path, 0, dict(full="different"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["case_00.json"]


def test_artifact_overflow_writes_no_partial_success(tmp_path):
    with pytest.raises(ValueError, match="COMPLETE_CASE_ARTIFACT_BUDGET_EXCEEDED"):
        _publish(tmp_path, 1, dict(full="retained entirely or refused"), maximum=4)
    assert list(tmp_path.iterdir()) == []


def _fake_observation_runtime(channel, points, response, *, failure=None, charge=True,
                              before_call=None):
    """Invented counter-only runtime: no surface, initializer or geometry calls."""
    owner = SimpleNamespace(statistics={key: 0 for keys in _Observations.KEYS.values() for key in keys})
    calls = []

    def original(actual_owner, face, actual_points, *gate):
        assert actual_owner is owner and actual_points is points
        assert face == 11 and gate == ((.5,) if channel == "box" else ())
        calls.append((actual_owner, face, actual_points, gate))
        if before_call is not None:
            before_call()
        names = _Observations.KEYS[channel]
        if charge:
            owner.statistics[names[0]] += 1
            owner.statistics[names[1]] += len(points)
        if failure is not None:
            raise failure
        owner.statistics[names[2]] += 1
        owner.statistics[names[3]] += len(points)
        return response
    return owner, original, calls


@pytest.mark.parametrize("channel", ("box", "finite_query"))
def test_observer_records_attempt_before_call_and_preserves_objects(channel):
    points, response = np.array([[1., 2., 3.]]), dict(status="COMPLETE", payload=[17])
    ledger = _Observations(_Tree())

    def before_call():
        row, = ledger.records[channel]
        assert row["disposition"] == "ATTEMPTED" and row["answer"] is None
        assert row["request"]["cad_face_id"] == 11
        assert row["request"]["points"]["array_values"] == points.tolist()
    owner, original, calls = _fake_observation_runtime(channel, points, response, before_call=before_call)
    returned = ledger.invoke(channel, original, owner, 11, points, .5)
    assert returned is response and calls[0][2] is points
    row, = ledger.records[channel]
    assert row["disposition"] == "RETURNED" and row["answer"] == response
    assert row["runtime_before"] == [0, 0, 0, 0] and row["runtime_after"] == [1, 1, 1, 1]
    assert ledger.reconcile(owner.statistics)["complete"] is True


@pytest.mark.parametrize("channel", ("box", "finite_query"))
@pytest.mark.parametrize("charge", (False, True))
def test_raised_attempt_preserves_original_exception_and_distinguishes_preflight_charge(channel, charge):
    points, failure = np.array([[1., 2., 3.]]), RuntimeError("INVENTED_ORIGINAL_FAILURE")
    ledger = _Observations(_Tree())
    owner, original, calls = _fake_observation_runtime(channel, points, None, failure=failure, charge=charge)
    with pytest.raises(RuntimeError) as caught:
        ledger.invoke(channel, original, owner, 11, points, .5)
    assert caught.value is failure and len(calls) == 1
    row, = ledger.records[channel]
    assert row["disposition"] == "RAISED" and row["answer"] is None
    assert row["exception"] == dict(type="RuntimeError", message="INVENTED_ORIGINAL_FAILURE")
    assert row["nested_primitive_evidence"] == "UNAVAILABLE_METHOD_RAISED_BEFORE_RETURN"
    assert row["runtime_after"] == [int(charge), int(charge), 0, 0]
    result = ledger.reconcile(owner.statistics)
    assert result["reconciliation"][channel]["matched"] is True
    assert result["complete"] is False
    assert result["counts"][channel]["attempted_calls"] == result["counts"][channel]["raised_calls"] == 1
    assert result["counts"][channel]["runtime_charged_calls"] == int(charge)


@pytest.mark.parametrize("channel", ("box", "finite_query"))
def test_baseexception_is_retained_and_reraised_by_identity(channel):
    points, failure = np.array([[1., 2., 3.]]), SystemExit(19)
    ledger = _Observations(_Tree())
    owner, original, _ = _fake_observation_runtime(channel, points, None, failure=failure)
    with pytest.raises(SystemExit) as caught:
        ledger.invoke(channel, original, owner, 11, points, .5)
    assert caught.value is failure
    assert ledger.records[channel][0]["exception"]["type"] == "SystemExit"
    assert ledger.reconcile(owner.statistics)["complete"] is False


@pytest.mark.parametrize("channel", ("box", "finite_query"))
def test_noncomplete_return_payload_is_retained_but_not_full_completed_capture(channel):
    points = np.array([[1., 2., 3.]])
    response = dict(status="UNAVAILABLE", reason="INVENTED_PRIMITIVE_REFUSAL", partial={"rows": 1})
    ledger = _Observations(_Tree())
    owner, original, _ = _fake_observation_runtime(channel, points, response)
    assert ledger.invoke(channel, original, owner, 11, points, .5) is response
    assert ledger.records[channel][0]["answer"] == response
    assert ledger.reconcile(owner.statistics)["complete"] is False


@pytest.mark.parametrize("channel", ("box", "finite_query"))
def test_unobserved_runtime_counter_drift_vetoes_capture_completion(channel):
    points, response = np.array([[1., 2., 3.]]), dict(status="COMPLETE")
    ledger = _Observations(_Tree())
    owner, original, _ = _fake_observation_runtime(channel, points, response)
    assert ledger.invoke(channel, original, owner, 11, points, .5) is response
    owner.statistics[_Observations.KEYS[channel][0]] += 1
    result = ledger.reconcile(owner.statistics)
    assert result["complete"] is False and result["reconciliation"][channel]["matched"] is False


@pytest.mark.parametrize("channel", ("box", "finite_query"))
@pytest.mark.parametrize("kind", ("calls", "rows"))
def test_late_channel_overflow_cannot_stop_original_call_or_claim_success(channel, kind):
    points, response = np.array([[1., 2., 3.]]), dict(status="COMPLETE")
    ledger = _Observations(_Tree(), maximum_calls=1 if kind == "calls" else 128,
                           maximum_rows=1 if kind == "rows" else 3200)
    owner, original, calls = _fake_observation_runtime(channel, points, response)
    assert ledger.invoke(channel, original, owner, 11, points, .5) is response
    assert ledger.invoke(channel, original, owner, 11, points, .5) is response
    assert len(calls) == 2 and len(ledger.records[channel]) == 1
    result = ledger.reconcile(owner.statistics)
    assert result["complete"] is False
    assert result["counts"][channel]["attempted_calls"] == result["counts"][channel]["returned_calls"] == 2
    assert result["counts"][channel]["runtime_completed_calls"] == 2
    assert result["reconciliation"][channel]["matched"] is False


@pytest.mark.parametrize("channel", ("box", "finite_query"))
def test_response_serialization_failure_preserves_return_and_recorded_attempt(channel):
    points, response = np.array([[1., 2., 3.]]), dict(status="COMPLETE", unsupported=object())
    ledger = _Observations(_Tree())
    owner, original, _ = _fake_observation_runtime(channel, points, response)
    assert ledger.invoke(channel, original, owner, 11, points, .5) is response
    row, = ledger.records[channel]
    assert row["disposition"] == "RETURNED_CAPTURE_INCOMPLETE" and row["answer"] is None
    assert row["request"]["points"]["array_values"] == points.tolist()
    assert ledger.reconcile(owner.statistics)["complete"] is False


@pytest.mark.skipif(os.environ.get(OPT_IN) != "1", reason="Explicit fresh two-case diagnostic release required")
def test_record_both_original_cases_without_reclassifying_failed_positives(monkeypatch):
    raw = os.environ.get(ARTIFACT_ENV)
    assert raw and Path(raw).is_absolute()
    directory = Path(raw)
    directory.mkdir(mode=0o700, exist_ok=False)
    runtime, legacy, capsule = _reviewed_capsules()
    dispositions = []
    for index, noisy in enumerate((False, True)):
        try:
            record = _record_case(noisy, runtime, legacy, capsule, monkeypatch)
            artifact = _publish(directory, index, record)
            dispositions.append(dict(case_name=CASES[index], artifact=artifact,
                diagnostic_recording_complete=record["capture"]["complete"],
                original_positive_status=record["original_positive_oracle"]["status"],
                start_group_count=len(record["full_evidence"]["returned_answer"]["start_groups"])))
        except Exception as error:
            dispositions.append(dict(case_name=CASES[index], diagnostic_recording_complete=False,
                original_positive_status="UNKNOWN_IF_NOT_DURABLY_RETAINED", error=_error(error)))
    summary = dict(schema="two_plane_v11_stage_diagnostic_summary_v1", qualifies_engine=False,
        cases=dispositions, diagnostic_recording_complete=all(row["diagnostic_recording_complete"]
            for row in dispositions), original_positive_count=sum(
                row["original_positive_status"] == "PASS" for row in dispositions))
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    assert summary["diagnostic_recording_complete"], summary
    assert len(dispositions) == 2
