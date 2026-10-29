"""C10 Jev harness: dry-run client, live gating, retries, cache and CLI exit codes.

``httpx.Client.send`` is patched to raise for every test, so no request can leave the
machine. Live-path logic is exercised with an in-memory fake transport.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest

from research.jev import cli
from research.jev import client_v1 as C
from research.jev.request_v1 import body_sha256, validate_request

FAKE_KEY = "fake-test-key-not-a-secret-0000"


class NetworkBlocked(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    calls = []

    def _refuse(self, request, *a, **k):
        calls.append(str(request.url))
        raise NetworkBlocked("network access is forbidden in C10 tests")

    monkeypatch.setattr(httpx.Client, "send", _refuse)
    return calls


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    out = tmp_path_factory.mktemp("jevws") / "ws"
    paths = cli.write_synthetic_workspace(out)
    return {k: Path(v) for k, v in paths.items()} | {"root": out}


class FakeTransport:
    def __init__(self, script=None):
        self.script = list(script or [])
        self.bodies: list[bytes] = []

    def post(self, body: bytes):
        self.bodies.append(body)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if item is not None:
                return item
        req = json.loads(body)
        answers = {}
        for qid, q in req["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.7}
            elif q["type"] == "choice":
                opts = list(q["criteria"])
                answers[qid] = {"type": "choice", "choice": opts[0],
                                "probabilities": {o: (1.0 if i == 0 else 0.0) for i, o in enumerate(opts)},
                                "confidence": 1.0}
        payload = {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 5}}
        return 200, {}, json.dumps(payload).encode()


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------


def test_dry_run_writes_requests_and_summary_without_network(ws, tmp_path, _no_network):
    out = tmp_path / "dry"
    summary = C.dry_run(ws["onboarding_states"], out, question_names=["is_real_bend", "candidate_kind"])
    files = sorted((out / "requests").glob("*.json"))
    states = [json.loads(line) for line in ws["onboarding_states"].read_text().splitlines()]
    assert summary["count"] == len(files) == len(states)
    assert {f.stem for f in files} == {s["record_id"] for s in states}
    total = 0
    for f in files:
        body = json.loads(f.read_bytes())
        validate_request(body)
        assert body["model"] == "jev-1.13.0"
        assert set(body["questions"]) == {"is_real_bend", "candidate_kind"}
        total += math.ceil(len(f.read_bytes().decode()) / 3)
    assert summary["est_tokens"] == total
    assert summary["est_cost_usd"] == pytest.approx(total * 0.042 / 1e6)
    assert summary["network"] == "none"
    assert json.loads((out / "summary.json").read_text()) == summary
    assert _no_network == []


def test_dry_run_refuses_a_non_empty_out_dir(ws, tmp_path):
    out = tmp_path / "dry"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(C.ClientError, match="not empty"):
        C.dry_run(ws["onboarding_states"], out)


def test_dry_run_with_splits_filters_like_live(ws, tmp_path):
    s1 = C.dry_run(ws["onboarding_states"], tmp_path / "a", splits=ws["splits"], labels=ws["onboarding_labels"])
    s2 = C.dry_run(ws["onboarding_states"], tmp_path / "b", splits=ws["splits"], labels=ws["onboarding_labels"],
                   include_eval=True)
    assert s1["skipped_by_split"] == {"eval": 8}
    assert s2["count"] == s1["count"] + 8 and s2["skipped_by_split"] == {}


def test_cad_free_dry_run_drops_the_cad_block(ws, tmp_path):
    C.dry_run(ws["onboarding_states"], tmp_path / "cf", cad_free=True)
    for f in (tmp_path / "cf" / "requests").glob("*.json"):
        assert "cad_relation" not in json.loads(f.read_bytes())["state"]


@pytest.mark.parametrize("model", ["jev-latest", "jev-preview", "jev", "gpt-4", "jev-1.13"])
def test_aliases_and_unversioned_models_are_refused(ws, tmp_path, model):
    with pytest.raises(C.ModelNotVersioned):
        C.dry_run(ws["onboarding_states"], tmp_path / "x", model=model)


def test_alias_allowed_only_explicitly_and_recorded(ws, tmp_path):
    summary = C.dry_run(ws["onboarding_states"], tmp_path / "x", model="jev-latest", allow_alias=True)
    assert summary["model_is_alias"] is True and summary["model"] == "jev-latest"


# --------------------------------------------------------------------------
# Live gating
# --------------------------------------------------------------------------


def test_live_needs_the_flag_and_the_key(ws, tmp_path):
    kwargs = dict(splits=ws["splits"], labels=ws["onboarding_labels"], transport=FakeTransport())
    with pytest.raises(C.LiveNotAuthorized, match="--live"):
        C.live_run(ws["onboarding_states"], tmp_path / "l", live=False, env={C.API_KEY_ENV: FAKE_KEY}, **kwargs)
    with pytest.raises(C.LiveNotAuthorized, match=C.API_KEY_ENV):
        C.live_run(ws["onboarding_states"], tmp_path / "l", live=True, env={}, **kwargs)
    assert not (tmp_path / "l").exists()


def test_cli_exit_codes(ws, tmp_path, monkeypatch):
    monkeypatch.delenv(C.API_KEY_ENV, raising=False)
    assert cli.main(["dry-run", "--states", str(ws["onboarding_states"]), "--out", str(tmp_path / "d")]) == 0
    assert (tmp_path / "d" / "summary.json").is_file()
    live = ["live", "--states", str(ws["onboarding_states"]), "--out", str(tmp_path / "l"),
            "--splits", str(ws["splits"]), "--labels", str(ws["onboarding_labels"])]
    assert cli.main(live) == 2  # no --live
    assert cli.main([*live, "--live"]) == 2  # no key
    assert cli.main(["dry-run", "--states", str(ws["onboarding_states"]), "--out", str(tmp_path / "e"),
                     "--model", "jev-latest"]) == 2
    assert cli.main(["export-onboarding", "--input", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o")]) == 2


def test_real_transport_goes_through_httpx_send_and_never_leaks_the_key(ws, tmp_path, _no_network):
    out = tmp_path / "live"
    with pytest.raises(NetworkBlocked):
        C.live_run(ws["onboarding_states"], out, live=True, env={C.API_KEY_ENV: FAKE_KEY},
                   splits=ws["splits"], labels=ws["onboarding_labels"], max_records=1)
    assert _no_network == [C.ENDPOINT]
    transport = C.HttpxTransport(FAKE_KEY)
    assert FAKE_KEY not in repr(transport)
    for path in out.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(errors="replace")


# --------------------------------------------------------------------------
# Live logic with a fake transport
# --------------------------------------------------------------------------


def _live(ws, out, transport, **kw):
    sleeps: list[float] = []
    summary = C.live_run(ws["onboarding_states"], out, live=True, env={C.API_KEY_ENV: FAKE_KEY},
                         splits=ws["splits"], labels=ws["onboarding_labels"], transport=transport,
                         sleep=sleeps.append, **kw)
    return summary, sleeps


def test_live_sends_dev_calib_only_caches_and_records_models(ws, tmp_path):
    t = FakeTransport()
    summary, _ = _live(ws, tmp_path / "l", t)
    assert summary["sent"] == len(t.bodies) == summary["count"] == 16  # 8 dev + 8 calib, eval held back
    assert summary["skipped_by_split"] == {"eval": 8}
    rows = [json.loads(x) for x in (tmp_path / "l" / "answers.jsonl").read_text().splitlines()]
    assert all(r["response_model"] == "jev-1.13.0" and r["usage"]["input_tokens"] == 100 for r in rows)
    assert all(r["body_sha256"] == body_sha256(json.loads(b)) for r, b in zip(rows, t.bodies))
    # Re-run: everything comes from the cache, nothing is sent.
    t2 = FakeTransport()
    summary2, _ = _live(ws, tmp_path / "l", t2)
    assert t2.bodies == [] and summary2["cached"] == 16 and summary2["sent"] == 0
    # eval only with include_eval
    t3 = FakeTransport()
    summary3, _ = _live(ws, tmp_path / "l", t3, include_eval=True)
    assert len(t3.bodies) == 8 and summary3["cached"] == 16


def test_live_retries_429_and_529_honouring_retry_after(ws, tmp_path):
    t = FakeTransport([(429, {"Retry-After": "7"}, b"slow down"), (529, {}, b"overloaded"),
                       (503, {}, b"unavailable")])
    summary, sleeps = _live(ws, tmp_path / "l", t, max_records=1)
    assert sleeps == [7.0, 2.0, 4.0]  # Retry-After, then exponential backoff 1 -> 2 -> 4 (first used by 429)
    assert summary["answered"] == 1 and len(t.bodies) == 4


@pytest.mark.parametrize("status", [400, 401, 413, 415, 422])
def test_live_fails_fast_on_client_errors(ws, tmp_path, status):
    t = FakeTransport([(status, {}, b'{"error":"bad"}')])
    with pytest.raises(C.ApiError) as err:
        _live(ws, tmp_path / "l", t)
    assert err.value.status == status and err.value.exit_code == 3
    assert len(t.bodies) == 1
    summary = json.loads((tmp_path / "l" / "summary.json").read_text())
    assert summary["answered"] == 0 and str(status) in summary["error"]


def test_live_retries_transport_failures_then_succeeds(ws, tmp_path):
    t = FakeTransport([httpx.ReadTimeout("timed out"), httpx.ConnectError("refused"), C.TransportFailure("x")])
    summary, sleeps = _live(ws, tmp_path / "l", t, max_records=1)
    assert sleeps == [1.0, 2.0, 4.0] and summary["answered"] == 1 and len(t.bodies) == 4


def test_live_transport_failures_exhausted_give_exit_3_and_write_outputs(ws, tmp_path):
    """A timeout storm ends as ApiError (exit 3) with answers.jsonl and summary.json written."""
    t = FakeTransport([None, None, *[httpx.ReadTimeout("timed out")] * 10])
    with pytest.raises(C.ApiError, match="gave up") as err:
        _live(ws, tmp_path / "l", t, max_records=3)
    assert err.value.exit_code == 3 and err.value.status == C.TRANSPORT_FAILURE_STATUS
    assert len(t.bodies) == 2 + C.MAX_ATTEMPTS
    rows = (tmp_path / "l" / "answers.jsonl").read_text().splitlines()
    summary = json.loads((tmp_path / "l" / "summary.json").read_text())
    assert len(rows) == 2 and summary["answered"] == 2 and "ReadTimeout" in summary["error"]


def test_cli_maps_a_timeout_storm_to_exit_3(ws, tmp_path, monkeypatch):
    monkeypatch.setenv(C.API_KEY_ENV, FAKE_KEY)
    monkeypatch.setattr(C.HttpxTransport, "post", lambda self, body: (_ for _ in ()).throw(C.TransportFailure("t")))
    monkeypatch.setattr(C.time, "sleep", lambda _s: None)
    rc = cli.main(["live", "--live", "--states", str(ws["onboarding_states"]), "--out", str(tmp_path / "l"),
                   "--splits", str(ws["splits"]), "--labels", str(ws["onboarding_labels"]), "--max-records", "1"])
    assert rc == 3
    assert (tmp_path / "l" / "summary.json").is_file()


def test_httpx_transport_maps_transport_errors(monkeypatch):
    def _timeout(self, request, *a, **k):
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(httpx.Client, "send", _timeout)
    transport = C.HttpxTransport(FAKE_KEY)
    with pytest.raises(C.TransportFailure) as err:
        transport.post(b"{}")
    assert FAKE_KEY not in str(err.value)


def test_answer_rows_carry_a_rebuildable_request_identity(ws, tmp_path):
    t = FakeTransport()
    _live(ws, tmp_path / "l", t, max_records=2, cad_free=True, question_names=["is_real_bend", "candidate_kind"])
    states = dict(C.load_states(ws["onboarding_states"]))
    for line in (tmp_path / "l" / "answers.jsonl").read_text().splitlines():
        row = json.loads(line)
        ident = C.request_identity(states[row["record_id"]], model=row["request_model"],
                                   variant=row["question_variant"], question_names=row["question_ids"],
                                   cad_free=row["ablation"]["cad_free"], context_free=row["ablation"]["context_free"])
        assert {k: row[k] for k in ident} == ident
        assert row["ablation"] == {"cad_free": True, "context_free": False}


def test_live_gives_up_after_bounded_retries(ws, tmp_path):
    t = FakeTransport([(429, {}, b"x")] * 10)
    with pytest.raises(C.ApiError, match="gave up"):
        _live(ws, tmp_path / "l", t, max_records=1)
    assert len(t.bodies) == C.MAX_ATTEMPTS


def test_invalid_responses_are_refused(ws, tmp_path):
    bad = {"model": "jev-latest", "answers": {"is_real_bend": {"type": "noul", "noul": 0.4}}, "usage": {}}
    t = FakeTransport([(200, {}, json.dumps(bad).encode())])
    with pytest.raises(C.ApiError, match="versioned"):
        _live(ws, tmp_path / "l", t, max_records=1)
    q = {"is_real_bend": {"type": "noul"}}
    for payload in (
        {"model": "jev-1.13.0", "answers": {}, "usage": {}},
        {"model": "jev-1.13.0", "answers": {"is_real_bend": {"type": "noul", "noul": 1.5}}, "usage": {}},
        {"model": "jev-1.13.0", "answers": {"is_real_bend": {"type": "choice", "noul": 0.5}}, "usage": {}},
        {"model": "jev-1.13.0", "answers": {"is_real_bend": {"type": "noul", "noul": 0.5}}},
    ):
        with pytest.raises(C.ApiError):
            C.validate_response(payload, q)


def test_parse_retry_after_forms():
    assert C.parse_retry_after("3") == 3.0
    assert C.parse_retry_after("100000") == C.RETRY_AFTER_CAP_S
    assert C.parse_retry_after("Wed, 21 Oct 2015 07:28:10 GMT", now=1445412480.0) == pytest.approx(10.0)
    assert C.parse_retry_after("soon") is None and C.parse_retry_after(None) is None
