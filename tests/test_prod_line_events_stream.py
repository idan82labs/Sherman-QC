"""C07 SSE event stream and stream tokens.

``GET /api/line/events/stream?station_id&token`` replays events after
``Last-Event-ID`` (or ``after_id``), then polls the store and sends
``: heartbeat`` comments. The station-bound stream token is the only auth.

TestClient buffers a streaming body until the generator ends, so these tests
use a short ``sse_max_lifetime_s``. One test runs a real uvicorn server to
prove frames are delivered incrementally, before the stream ends.
"""
from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from apps.api.services import production_line_service as svc
from apps.production_line.security import stream_tokens
from apps.production_line.security.principals import Principal
from tests.test_prod_line_routes import OTHER, STATION, LineHarness, build_harness, h


@pytest.fixture()
def harness(tmp_path: Path) -> Iterator[LineHarness]:
    hs = build_harness(tmp_path)
    yield hs
    hs.client.close()
    svc.reset_line_api_caches()


def _five_events(hs: LineHarness, station: str = STATION) -> list[int]:
    ids = [
        hs.store.append_event("SCAN_RECEIVED", station_id=station, payload={"n": i, "original_name": f"s{i}.ply"})
        for i in range(1, 6)
    ]
    assert ids == sorted(ids)
    return ids


def _token(hs: LineHarness, auth: str = "tok-viewer", station: str = STATION) -> str:
    r = hs.client.post("/api/line/stream-token", json={"station_id": station}, headers=h(auth))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["station_id"] == station and body["expires_at"].endswith("Z")
    return body["token"]


def parse_sse(text: str) -> list[dict[str, Any]]:
    """SSE text -> blocks: ``{"comment": str}``, ``{"retry": int}`` or ``{"id", "event", "data"}``."""
    out: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        if all(line.startswith(":") for line in lines):
            out.append({"comment": lines[0][1:].strip()})
            continue
        frame: dict[str, Any] = {}
        for line in lines:
            key, _, value = line.partition(": ")
            frame[key] = value
        if "retry" in frame:
            frame["retry"] = int(frame["retry"])
        if "id" in frame:
            frame["id"] = int(frame["id"])
        out.append(frame)
    return out


def test_stream_replays_after_last_event_id_then_heartbeat(harness: LineHarness) -> None:
    ids = _five_events(harness)
    harness.store.append_event("SCAN_RECEIVED", station_id=OTHER, payload={"n": 99})
    token = _token(harness)
    t0 = time.monotonic()
    r = harness.client.get(
        f"/api/line/events/stream?station_id={STATION}&token={token}&after_id=0",
        headers={"Last-Event-ID": str(ids[2])},
    )
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"
    blocks = parse_sse(r.text)
    assert blocks[0] == {"retry": harness.config.sse_retry_ms}
    frames = [b for b in blocks if "id" in b]
    # Last-Event-ID wins over after_id; only this station's events; in order.
    assert [f["id"] for f in frames] == ids[3:]
    assert all(f["event"] == "SCAN_RECEIVED" for f in frames)
    first_beat = next(i for i, b in enumerate(blocks) if b.get("comment") == "heartbeat")
    last_frame = max(i for i, b in enumerate(blocks) if "id" in b)
    assert first_beat > last_frame, "replayed events come before the first heartbeat"
    # Heartbeats keep coming at the configured interval until the lifetime ends.
    beats = sum(1 for b in blocks if b.get("comment") == "heartbeat")
    assert 2 <= beats <= int(harness.config.sse_max_lifetime_s / harness.config.sse_heartbeat_s) + 1
    assert elapsed < harness.config.sse_max_lifetime_s + 2.0


def test_stream_after_id_query_and_default_is_live_only(harness: LineHarness) -> None:
    ids = _five_events(harness)
    token = _token(harness)
    r = harness.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}&after_id={ids[0]}")
    assert [b["id"] for b in parse_sse(r.text) if "id" in b] == ids[1:]
    # An unparsable Last-Event-ID is ignored in favour of after_id.
    r = harness.client.get(
        f"/api/line/events/stream?station_id={STATION}&token={token}&after_id={ids[3]}", headers={"Last-Event-ID": "abc"}
    )
    assert [b["id"] for b in parse_sse(r.text) if "id" in b] == ids[4:]
    # No cursor: only events after the connection (none here).
    r = harness.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}")
    assert [b for b in parse_sse(r.text) if "id" in b] == []


def test_stream_data_is_line_event_json(harness: LineHarness) -> None:
    import json

    job = harness.queue_job(STATION)
    harness.client.post(f"/api/line/jobs/{job['job_id']}/cancel", headers=h("tok-operator"))
    token = _token(harness)
    r = harness.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}&after_id=0")
    frames = [b for b in parse_sse(r.text) if "id" in b]
    kinds = [f["event"] for f in frames]
    assert "JOB_QUEUED" in kinds and "JOB_CANCEL_REQUESTED" in kinds and "JOB_CANCELLED" in kinds
    for f in frames:
        data = json.loads(f["data"])
        assert data["id"] == f["id"] and data["kind"] == f["event"] and data["schema"] == "sherman_line_event_v1"
    cancel = next(json.loads(f["data"]) for f in frames if f["event"] == "JOB_CANCEL_REQUESTED")
    assert cancel["principal"]["id"] == "2"


def test_device_stream_hides_principals(harness: LineHarness) -> None:
    import json

    job = harness.queue_job(STATION)
    harness.client.post(f"/api/line/jobs/{job['job_id']}/cancel", headers=h("tok-operator"))
    _, device = harness.device_token(STATION)
    token = _token(harness, device)
    r = harness.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}&after_id=0")
    frames = [json.loads(b["data"]) for b in parse_sse(r.text) if "id" in b]
    assert frames and all(f["principal"] is None for f in frames)


@pytest.mark.parametrize("case", ["missing", "tampered", "expired", "wrong_station", "garbage", "wrong_secret"])
def test_stream_token_refusals(harness: LineHarness, case: str) -> None:
    _five_events(harness)
    secret = svc.get_secret(harness.settings.paths, "stream_token")
    viewer = Principal.for_user(3, "vera", "viewer")
    good = _token(harness)
    if case == "missing":
        url = f"/api/line/events/stream?station_id={STATION}"
    else:
        if case == "tampered":
            head, body, mac = good.split(".")
            token = f"{head}.{body[:-2]}{'A' if body[-2] != 'A' else 'B'}{body[-1]}.{mac}"
        elif case == "expired":
            token = stream_tokens.mint(viewer, STATION, secret, ttl_s=60, now=time.time() - 120)
        elif case == "wrong_station":
            token = stream_tokens.mint(viewer, OTHER, secret)
        elif case == "garbage":
            token = "sst1.x.y"
        else:
            token = stream_tokens.mint(viewer, STATION, b"another-secret-of-32-bytes-xxxxx")
        url = f"/api/line/events/stream?station_id={STATION}&token={token}&after_id=0"
    r = harness.client.get(url)
    assert r.status_code == 401, case
    assert r.headers["x-sherman-error-code"] in ("STREAM_TOKEN_INVALID", "STREAM_TOKEN_REQUIRED")
    assert "www-authenticate" in r.headers
    assert "SCAN_RECEIVED" not in r.text


def test_stream_ignores_bearer_header_without_token(harness: LineHarness) -> None:
    r = harness.client.get(f"/api/line/events/stream?station_id={STATION}", headers=h("tok-admin"))
    assert r.status_code == 401


def test_revoked_device_cannot_open_stream(harness: LineHarness) -> None:
    token_id, device = harness.device_token(STATION)
    token = _token(harness, device)
    from apps.production_line.security.dependencies import device_token_store_for

    assert device_token_store_for(harness.settings).revoke(token_id, by="test")
    r = harness.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}")
    assert r.status_code == 401


def test_stream_token_mint_scoping(harness: LineHarness) -> None:
    _, device = harness.device_token(STATION)
    assert harness.client.post("/api/line/stream-token", json={"station_id": OTHER}, headers=h(device)).status_code == 403
    assert harness.client.post("/api/line/stream-token", json={"station_id": STATION}).status_code == 401
    r = harness.client.post("/api/line/stream-token", json={"station_id": "cell-9"}, headers=h("tok-viewer"))
    assert r.status_code == 404
    r = harness.client.post("/api/line/stream-token", json={"station_id": "BAD ID"}, headers=h("tok-viewer"))
    assert r.status_code == 422
    r = harness.client.post("/api/line/stream-token", json={"station_id": STATION}, headers=h("tok-viewer"))
    assert r.headers["cache-control"] == "no-store"
    claims = stream_tokens.verify(r.json()["token"], svc.get_secret(harness.settings.paths, "stream_token"))
    assert claims is not None and claims.expires_at - claims.issued_at == 300 and claims.station_id == STATION


class _FakeRequest:
    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def test_generator_stops_when_device_revoked_and_on_disconnect(harness: LineHarness) -> None:
    import anyio

    _five_events(harness)
    cfg = svc.LineApiConfig(sse_heartbeat_s=0.05, sse_poll_s=0.01, sse_max_lifetime_s=30)
    allowed = {"ok": True}

    async def run(req: _FakeRequest, stop_after_beats: int | None) -> list[str]:
        out: list[str] = []
        beats = 0
        gen = svc.sse_event_stream(
            req, harness.store, STATION, 0, cfg, principal_kind="device", still_authorized=lambda: allowed["ok"]
        )
        async for chunk in gen:
            out.append(chunk)
            if chunk.startswith(": heartbeat"):
                beats += 1
                if stop_after_beats is not None and beats >= stop_after_beats:
                    allowed["ok"] = False  # revoke: the next heartbeat check closes the stream
        return out

    t0 = time.monotonic()
    chunks = anyio.run(run, _FakeRequest(), 2)
    assert time.monotonic() - t0 < 5, "revocation must close the stream long before its lifetime"
    assert sum(c.startswith(": heartbeat") for c in chunks) == 2
    assert sum(c.startswith("id: ") for c in chunks) == 5

    req = _FakeRequest()
    req.disconnected = True
    assert anyio.run(run, req, None) == [f"retry: {cfg.sse_retry_ms}\n\n"]


def test_generator_replays_large_backlog_in_batches(harness: LineHarness) -> None:
    import anyio

    for i in range(25):
        harness.store.append_event("SCAN_RECEIVED", station_id=STATION, payload={"n": i})
    cfg = svc.LineApiConfig(sse_batch=7, sse_heartbeat_s=1, sse_poll_s=0.01, sse_max_lifetime_s=0.05)

    async def run() -> list[str]:
        return [c async for c in svc.sse_event_stream(_FakeRequest(), harness.store, STATION, 0, cfg, principal_kind="user")]

    ids = [int(c.split("\n", 1)[0][4:]) for c in anyio.run(run) if c.startswith("id: ")]
    assert ids == sorted(ids) and len(ids) == 25


def test_stream_cursor_ahead_of_store_sends_reset_then_continues(harness: LineHarness) -> None:
    import anyio
    import json

    ids = _five_events(harness)
    token = _token(harness)
    # Route: a Last-Event-ID from before a DB restore (ids restarted lower).
    r = harness.client.get(
        f"/api/line/events/stream?station_id={STATION}&token={token}", headers={"Last-Event-ID": str(ids[-1] + 5000)}
    )
    blocks = parse_sse(r.text)
    assert blocks[1]["event"] == svc.SSE_RESET_EVENT and blocks[1]["id"] == ids[-1]
    data = json.loads(blocks[1]["data"])
    assert data == {"kind": "RESET", "reason": "CURSOR_AHEAD_OF_STORE", "last_id": ids[-1], "requested_after_id": ids[-1] + 5000}
    assert not [b for b in blocks if "id" in b and b.get("event") != "RESET"], "no backlog replay after a reset"

    # Generator: events appended after the reset are delivered (the stream is not starved).
    cfg = svc.LineApiConfig(sse_heartbeat_s=0.05, sse_poll_s=0.01, sse_max_lifetime_s=0.5)

    async def run() -> list[str]:
        out: list[str] = []
        async for chunk in svc.sse_event_stream(_FakeRequest(), harness.store, STATION, ids[-1] + 5000, cfg, principal_kind="user"):
            out.append(chunk)
            if "event: RESET" in chunk:
                harness.store.append_event("JOB_QUEUED", station_id=STATION, payload={})
        return out

    chunks = anyio.run(run)
    live = [c for c in chunks if c.startswith(f"id: {ids[-1] + 1}\n")]
    assert len(live) == 1 and "event: JOB_QUEUED" in live[0]
    # A cursor at the store's newest id is not a reset.
    r = harness.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}&after_id={ids[-1] + 1}")
    assert "RESET" not in r.text


def test_stream_closes_when_its_token_expires(tmp_path: Path) -> None:
    cfg = svc.LineApiConfig(min_free_bytes=0, sse_heartbeat_s=0.2, sse_poll_s=0.02, sse_max_lifetime_s=30, readiness_ttl_s=0)
    hs = build_harness(tmp_path, cfg)
    try:
        secret = svc.get_secret(hs.settings.paths, "stream_token")
        token = stream_tokens.mint(Principal.for_user(3, "vera", "viewer"), STATION, secret, ttl_s=2)
        t0 = time.monotonic()
        r = hs.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}")
        elapsed = time.monotonic() - t0
        assert r.status_code == 200
        assert elapsed < 5, f"a user stream must end at token expiry, not after sse_max_lifetime_s ({elapsed:.1f} s)"
    finally:
        hs.client.close()
        svc.reset_line_api_caches()


def test_stream_payloads_are_redacted(harness: LineHarness) -> None:
    import json

    root = str(harness.settings.data_root)
    harness.failed_job(STATION, message=f"crash in {root}/jobs/x Bearer abcdefSECRET123")
    harness.store.append_event("SCAN_QUARANTINED", station_id=STATION, payload={"detail": f"{root}/inbox/a.ply"})
    _, device = harness.device_token(STATION)
    for auth in (device, "tok-viewer"):
        token = _token(harness, auth)
        r = harness.client.get(f"/api/line/events/stream?station_id={STATION}&token={token}&after_id=0")
        frames = [b for b in parse_sse(r.text) if "id" in b]
        text = "\n".join(f["data"] for f in frames)
        assert frames and root not in text and "SECRET123" not in text
        assert "<data_root>" in text
        assert all(json.loads(f["data"])["schema"] == "sherman_line_event_v1" for f in frames)


def test_sse_frame_shape() -> None:
    frame = svc.sse_frame({"id": 7, "kind": "JOB_FAILED\nevent: x", "payload": {"m": "a\nb"}})
    lines = frame.split("\n")
    assert lines[0] == "id: 7" and lines[1] == "event: JOB_FAILED" and lines[2].startswith("data: {")
    assert frame.endswith("\n\n") and frame.count("\n") == 4, "data stays on one line"


def test_parse_last_event_id() -> None:
    assert svc.parse_last_event_id("12", 3) == 12
    assert svc.parse_last_event_id(" 12 ", None) == 12
    assert svc.parse_last_event_id("-1", 3) == 3
    assert svc.parse_last_event_id("x", None) is None
    assert svc.parse_last_event_id(None, None) is None


# --------------------------------------------------------------------------- real server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_real_server_delivers_frames_incrementally(tmp_path: Path) -> None:
    import uvicorn

    cfg = svc.LineApiConfig(min_free_bytes=0, sse_heartbeat_s=0.3, sse_poll_s=0.05, sse_max_lifetime_s=20)
    hs = build_harness(tmp_path, cfg)
    ids = _five_events(hs)
    token = _token(hs)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(hs.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        url = f"http://127.0.0.1:{port}/api/line/events/stream?station_id={STATION}&token={token}"
        got: list[str] = []
        t0 = time.monotonic()
        with httpx.Client(timeout=10) as client:
            with client.stream("GET", url, headers={"Last-Event-ID": str(ids[2])}) as r:
                assert r.status_code == 200
                assert r.headers["x-accel-buffering"] == "no" and r.headers["cache-control"] == "no-cache"
                buf = ""
                live_sent = False
                for text in r.iter_text():
                    buf += text
                    while "\n\n" in buf:
                        block, buf = buf.split("\n\n", 1)
                        got.append(block)
                        if block.startswith(": heartbeat") and not live_sent:
                            # A live event appended now must arrive on the open stream.
                            live_id = hs.store.append_event("JOB_QUEUED", station_id=STATION, payload={})
                            live_sent = True
                    if any(b.startswith(f"id: {ids[-1] + 1}") for b in got) or time.monotonic() - t0 > 8:
                        break
        elapsed = time.monotonic() - t0
        frames = [int(b.split("\n", 1)[0][4:]) for b in got if b.startswith("id: ")]
        assert frames[:2] == ids[3:]
        assert live_sent and frames[-1] == live_id
        first_beat = next(i for i, b in enumerate(got) if b.startswith(": heartbeat"))
        assert first_beat > got.index(next(b for b in got if b.startswith(f"id: {ids[4]}")))
        assert elapsed < cfg.sse_max_lifetime_s, "frames arrived while the stream was still open"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        hs.client.close()
        svc.reset_line_api_caches()
