"""Fast end-to-end test of the production line (C11), always on.

One synthetic part (an E2 part set: a real STEP BRep, its flat, a drawing and
scans of the same part) goes through the whole line in one process, wired by
:mod:`apps.production_line.wiring` exactly as the line service and the API
server wire it:

1. tmp data root, tmp station inbox, tmp legacy ``PartCatalogManager``;
2. the part is onboarded into the catalog (CAD + drawing "uploads", CAD bend
   candidates, a confirmed Vendor Setup draft with the bending order);
3. it is **released through the C05 router** (real E1 compile, benchmark row);
4. ZEISS-style drops into the inbox (write to a temp name, then rename) are
   picked up by the real C02 hotfolder;
5. the real C02 runner runs each job. The engine step is the E4 **truth
   oracle** (``domains.production.engine.eval.oracle``: an E3-compatible CLI
   that reports E2 truth, marked ``E4_TRUTH_ORACLE_NOT_AN_ENGINE``; it is a
   test double, never evidence about E3). The C06 render step (viewer assets,
   PDF) is **real**;
6. the C07 station feed, result and artifact routes are read through the same
   ASGI stack the server runs (access log + route policy + overrides).

Asserted: the release registers a benchmark reference with per-bend nominals;
the error scan (+1.5 mm on a B2 flange, +2 deg on B3) shows exactly those
characteristics FAIL with actuals equal to truth and every other required
characteristic PASS; the PDF downloads and names the SKU; a duplicate drop is
idempotent; the incomplete scan greys its unformed bends (NOT_YET_FORMED, never
FAIL) and is never green; an unknown-SKU drop is quarantined, assigned by SKU and
runs; a runner killed mid-job is recovered by its restarted successor; the
access log never contains a stream token. The real-engine variant is
``tests/test_prod_e2e_engine.py`` (gated).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
STATION = "cell-1"
ORACLE_MODULE = "domains.production.engine.eval.oracle"
USERS = {"tok-admin": ("1", "ada", "admin"), "tok-operator": ("2", "otto", "operator"), "tok-viewer": ("3", "vera", "viewer")}


class StubVerifier:
    def verify(self, token: str):
        from apps.production_line.security.principals import Principal

        row = USERS.get(token)
        return None if row is None else Principal.for_user(*row)


def h(token: str = "tok-admin") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def drop_scan(inbox: Path, src: Path, name: str, *, extra: bytes = b"") -> Path:
    """ZEISS-like export: stream into a temp name in chunks, then rename."""
    data = src.read_bytes() + extra
    tmp = inbox / f".{name}.part"
    with open(tmp, "wb") as fh:
        for i in range(0, len(data), 64 * 1024):
            fh.write(data[i:i + 64 * 1024])
            fh.flush()
    final = inbox / name
    os.replace(tmp, final)
    old = time.time() - 5
    os.utime(final, (old, old))
    return final


def poll_until(line: SimpleNamespace, predicate, *, polls: int = 20) -> list[Any]:
    seen: list[Any] = []
    for _ in range(polls):
        seen.extend(line.poller.poll_once())
        if predicate(seen):
            return seen
        time.sleep(0.02)
    raise AssertionError(f"hotfolder outcome not reached; saw {[(o.action, o.reason) for o in seen]}")


def ply_with_comment(src: Path, comment: str) -> bytes:
    """The same scan bytes with one extra header comment (new sha, same geometry)."""
    raw = src.read_bytes()
    head, sep, rest = raw.partition(b"\n")
    assert head.strip() == b"ply" and sep
    return head + b"\n" + f"comment {comment}".encode("ascii") + b"\n" + rest


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    from part_catalog import PartCatalogManager

    from apps.production_line.recipes.catalog_port import CatalogManagerPort
    from apps.production_line.recipes.synthetic_onboarding import onboard_case, release_request, vendor_setup_loader
    from apps.production_line.security import dependencies as deps
    from apps.production_line.server import build_asgi
    from apps.production_line.settings import ensure_layout, load_settings
    from apps.production_line.wiring import build_runtime
    from domains.production.engine.synthetic.cases import generate_part_set

    t0 = time.monotonic()
    base = tmp_path_factory.mktemp("e2e_fast")
    meta = generate_part_set("HAT", "DEV", base / "e2", target_points=30000, ray_threads=1)
    set_dir = base / "e2" / meta["set_id"]
    part_set = json.loads((set_dir / "part_set.json").read_text(encoding="utf-8"))
    sku = part_set["sku"]
    # The oracle finds a scan's truth by its sha256 under SHERMAN_E4_ORACLE_CASES. Re-exported
    # variants (new bytes, same geometry) get their own case directory there (oracle_variant).
    oracle_root = base / "oracle_cases"
    for scan_dir in sorted((set_dir / "scans").iterdir()):
        shutil.copytree(scan_dir, oracle_root / scan_dir.name)
    data_root, inbox = base / "line", base / "inbox"
    inbox.mkdir()
    env = {
        "SHERMAN_LINE_ENABLED": "1",
        "SHERMAN_LINE_DATA_ROOT": str(data_root),
        "SHERMAN_LINE_STATION_ID": STATION,
        "SHERMAN_LINE_INBOX": str(inbox),
    }
    settings = load_settings(env, repo_root=REPO)
    ensure_layout(settings.paths)

    catalog = PartCatalogManager(str(base / "catalog.sqlite"))
    ob = onboard_case(catalog, set_dir, part_number=sku, upload_dir=base / "uploads",
                      revision=part_set["revision"], part_name=part_set["part_name"])
    port = CatalogManagerPort(catalog, vendor_setup_loader=vendor_setup_loader(catalog))
    runtime = build_runtime(
        settings,
        workers=True,
        catalog=port,
        instance_id="runner@e2e-fast",
        runner_overrides={
            "engine_root": REPO,
            "inspect_module": ORACLE_MODULE,
            "extra_env": {"SHERMAN_E4_ORACLE_CASES": str(oracle_root)},
            "lease_ttl_s": 30.0,
            "heartbeat_s": 1.0,
        },
        hotfolder_overrides={"poll_interval_s": 0.05, "stable_polls": 1, "min_age_s": 0.0, "min_size_bytes": 0},
    )
    app = FastAPI()
    import apps.api.routes.production_line as line_routes
    import apps.api.routes.production_recipes as recipe_routes

    app.include_router(recipe_routes.router)
    app.include_router(line_routes.router)
    asgi = build_asgi(app, settings, runtime)
    app.dependency_overrides[deps.get_user_token_verifier] = lambda: StubVerifier()
    client = TestClient(asgi)
    r = client.post(f"/api/production/parts/{ob.part_id}/recipe/release", json=release_request(ob), headers=h())
    assert r.status_code == 201, r.text
    release = r.json()
    yield SimpleNamespace(
        settings=settings, runtime=runtime, client=client, app=app, ob=ob, sku=sku, set_dir=set_dir,
        part_set=part_set, inbox=inbox, data_root=data_root, release=release, poller=runtime.pollers[0],
        runner=runtime.runner, store=runtime.line_store, setup_s=time.monotonic() - t0, oracle_root=oracle_root,
    )
    client.close()
    runtime.close()


def scan_entry(line: SimpleNamespace, name: str) -> dict[str, Any]:
    return next(s for s in line.part_set["scans"] if s["name"] == name)


def run_drop(line: SimpleNamespace, name: str, *, file_name: str | None = None) -> dict[str, Any]:
    """Drop one part-set scan, ingest it, run its job; return the finished job row."""
    entry = scan_entry(line, name)
    drop_scan(line.inbox, line.set_dir / entry["dir"] / "scan.ply", file_name or entry["zeiss_filename"])
    outcomes = poll_until(line, lambda seen: any(o.action == "enqueued" for o in seen))
    job_id = next(o.job_id for o in outcomes if o.action == "enqueued")
    assert line.runner.run_once() == job_id
    job = line.store.get_job(job_id)
    assert job["execution_status"] == "completed", job.get("failure")
    return job


def oracle_variant(line: SimpleNamespace, name: str, comment: str) -> Path:
    """A re-export of a part-set scan (one extra PLY header comment: new sha256, same
    points) registered with the oracle under its own case directory; returns the scan."""
    src = line.oracle_root / name
    dst = line.oracle_root / f"{name}-{comment}"
    shutil.copytree(src, dst)
    data = ply_with_comment(src / "scan.ply", comment)
    (dst / "scan.ply").write_bytes(data)
    case = json.loads((dst / "case.json").read_text(encoding="utf-8"))
    case["files"]["scan.ply"] = hashlib.sha256(data).hexdigest()
    (dst / "case.json").write_text(json.dumps(case), encoding="utf-8")
    return dst / "scan.ply"


def latest(line: SimpleNamespace) -> dict[str, Any]:
    r = line.client.get(f"/api/line/stations/{STATION}/latest", headers=h("tok-viewer"))
    assert r.status_code == 200, r.text
    return r.json()


def char(row: dict[str, Any], kind: str, side: str | None = None) -> dict[str, Any]:
    return next(c for c in row["characteristics"] if c["kind"] == kind and c.get("side") == side)


# ---------------------------------------------------------------------------- release


def test_release_registers_benchmark_reference_with_per_bend_nominals(line) -> None:
    rel = line.release
    assert rel["outcome"] == "released" and rel["recipe_id"] == f"{line.sku}@v1"
    bench = line.runtime.recipe_store.get_benchmark_reference(rel["recipe_id"])
    assert bench is not None and bench["status"] == "REGISTERED"
    nominals = bench.get("nominals") or {}
    text = json.dumps(nominals)
    bends = [b.bend_uid for b in line.runtime.recipe_store.get(rel["recipe_id"]).recipe().bends]
    assert nominals and bends, "benchmark reference without nominals"
    for uid in bends:
        assert uid in text, f"benchmark nominals miss bend {uid}"
    # The scan resolver sees the release (provider cache invalidated by the release listener).
    assert line.runtime.resolver.resolve(scan_entry(line, "nominal")["zeiss_filename"]).status == "MATCHED"


# ---------------------------------------------------------------------------- inspect: injected errors


def test_error_scan_reaches_station_feed_with_truthful_verdicts_and_pdf(line) -> None:
    job = run_drop(line, "errors")
    feed = latest(line)
    assert feed["latest_completed"]["job"]["job_id"] == job["job_id"]
    result = feed["latest_completed"]["result"]
    assert result["summary"]["state"] == "OUT_OF_TOLERANCE_FOUND"
    assert result["context"]["station_id"] == STATION and result["sku"] == line.sku
    assert "E4_TRUTH_ORACLE_NOT_AN_ENGINE" in result["warnings"]  # the double is labelled as such
    injected = scan_entry(line, "errors")["expected"]["injected"]
    never_pass = set(scan_entry(line, "errors")["expected"]["characteristics_never_pass"])
    assert len(never_pass) == 2
    rows = {r["label"]: r for r in result["bends"]}
    assert [r["sequence_index"] for r in result["bends"]] == list(range(1, len(rows) + 1))
    failed: set[str] = set()
    for row in result["bends"]:
        for c in row["characteristics"]:
            key = f"{c['bend_uid']}/{c['kind']}" + (f"/{c['side']}" if c.get("side") else "")
            if c["decision"] == "FAIL":
                failed.add(key)
            if key in injected and c["state"] == "MEASURED":
                assert abs(c["actual"] - injected[key]["truth"]) <= (0.2 if c["unit"] == "mm" else 0.3)
            if c["required"] and key not in never_pass and c["state"] == "MEASURED":
                assert c["decision"] in ("PASS", "REVIEW"), (key, c["decision"])
    assert failed == never_pass
    b2, b3 = rows["B2"], rows["B3"]
    assert any(c["decision"] == "FAIL" and c["kind"] == "FLANGE_LENGTH" for c in b2["characteristics"])
    assert char(b3, "INCLUDED_ANGLE")["decision"] == "FAIL"
    assert abs(char(b3, "INCLUDED_ANGLE")["deviation"] - 2.0) <= 0.3
    # Artifacts: the real C06 render wrote the PDF and the viewer assets.
    arts = feed["latest_completed"]["artifacts"]
    assert {"report.pdf", "viewer_scan.ply"} <= set(arts)
    pdf = line.client.get(arts["report.pdf"], headers=h("tok-viewer"))
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    import fitz

    with fitz.open(stream=pdf.content, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    assert line.sku in text
    r = line.client.get(f"/api/line/jobs/{job['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 200 and r.json()["summary"] == result["summary"]


def test_duplicate_drop_is_idempotent(line) -> None:
    entry = scan_entry(line, "nominal")
    src = line.set_dir / entry["dir"] / "scan.ply"
    before = len(line.store.list_jobs(limit=500))
    drop_scan(line.inbox, src, entry["zeiss_filename"])
    first = poll_until(line, lambda seen: any(o.action == "enqueued" for o in seen))
    drop_scan(line.inbox, src, entry["zeiss_filename"].replace(" 1.ply", " 2.ply"))
    second = poll_until(line, lambda seen: any(o.action in ("duplicate", "enqueued") for o in seen))
    assert [o.action for o in second if o.action in ("duplicate", "enqueued")] == ["duplicate"]
    assert len(line.store.list_jobs(limit=500)) == before + 1
    assert line.runner.run_once() == next(o.job_id for o in first if o.action == "enqueued")
    assert latest(line)["latest_completed"]["result"]["summary"]["state"] == "ALL_MEASURED_IN_TOLERANCE"
    assert not any(line.inbox.iterdir()), "the inbox must be emptied (source policy move)"


# ---------------------------------------------------------------------------- inspect: incomplete part


def test_incomplete_scan_greys_unformed_bends_and_is_never_green(line) -> None:
    run_drop(line, "stage_k")
    result = latest(line)["latest_completed"]["result"]
    expected = scan_entry(line, "stage_k")["expected"]["bend_states_contract"]
    states = {r["bend_uid"]: r["state"] for r in result["bends"]}
    assert states == expected
    assert "NOT_YET_FORMED" in states.values()
    assert result["summary"]["state"] not in ("ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE")
    assert result["stage_decision"]["status"] == "FINAL_INCOMPLETE"
    for row in result["bends"]:
        if row["state"] == "NOT_YET_FORMED":
            assert all(c["decision"] != "FAIL" for c in row["characteristics"]), row["label"]
            assert all(c["actual"] is None for c in row["characteristics"] if c["state"] != "MEASURED")


# ---------------------------------------------------------------------------- quarantine


def test_unknown_sku_drop_is_quarantined_then_assigned_and_runs(line) -> None:
    entry = scan_entry(line, "unknown_sku")
    # E2's unknown-SKU scan is the nominal scan byte for byte (a duplicate once that ran, and its
    # truth.json names the nominal case, which the oracle refuses), so drop a re-export of the
    # nominal scan under the unknown-SKU name.
    drop_scan(line.inbox, oracle_variant(line, "nominal", "quarantine"), entry["zeiss_filename"])
    poll_until(line, lambda seen: any(o.action == "quarantined" for o in seen))
    r = line.client.get("/api/line/quarantine", headers=h("tok-viewer"))
    assert r.status_code == 200
    item = next(i for i in r.json()["items"] if i["scan_file"]["original_name"] == entry["zeiss_filename"])
    assert item["quarantine_reason"] == "UNMATCHED_SKU" and item["resolvable"] is True
    assert item["message"]
    scan_file_id = item["scan_file"]["scan_file_id"]
    assert line.client.post(f"/api/line/quarantine/{scan_file_id}/assign", json={"sku": line.sku},
                            headers=h("tok-viewer")).status_code == 403  # viewers cannot operate
    r = line.client.post(f"/api/line/quarantine/{scan_file_id}/assign", json={"sku": line.sku},
                         headers=h("tok-operator"))
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert line.runner.run_once() == job_id
    finished = line.store.get_job(job_id)
    assert finished["execution_status"] == "completed", finished.get("failure")
    assert latest(line)["latest_completed"]["job"]["job_id"] == job_id


# ---------------------------------------------------------------------------- recovery


def test_runner_killed_mid_job_is_recovered_by_its_restarted_successor(line) -> None:
    from apps.production_line.runner.runner import JobRunner, lease_owner_for
    from apps.production_line.wiring import runner_config_for

    entry = scan_entry(line, "nominal")
    drop_scan(line.inbox, oracle_variant(line, "nominal", "recovery"), entry["zeiss_filename"].replace(" 1.ply", " 3.ply"))
    outcomes = poll_until(line, lambda seen: any(o.action == "enqueued" for o in seen))
    job_id = next(o.job_id for o in outcomes if o.action == "enqueued")
    # A predecessor process of this runner leased the job and died mid-job. Its lease owner
    # has the runner's own shape (C02: '<instance_id>@<host>:<pid>'), so the successor
    # recognises it as its dead predecessor on this host.
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead_pid = int(dead.stdout.strip())
    leased = line.store.lease_next(lease_owner_for("runner@e2e-fast", pid=dead_pid), 30.0, None)
    assert leased is not None and leased["job_id"] == job_id
    assert line.store.get_job(job_id)["execution_status"] == "running"
    successor = JobRunner(
        runner_config_for(line.settings, engine_root=REPO, inspect_module=ORACLE_MODULE,
                          extra_env={"SHERMAN_E4_ORACLE_CASES": str(line.oracle_root)}),
        line.store, line.runtime.provider, instance_id="runner@e2e-fast", compiled_sink=line.runtime.provider,
    )
    assert line.store.get_job(job_id)["execution_status"] == "queued"  # reclaimed at once, no TTL wait
    assert successor.run_once() == job_id
    job = line.store.get_job(job_id)
    assert job["execution_status"] == "completed" and job["attempts"] == 2
    kinds = [e["kind"] for e in line.store.events_after(0, limit=1000) if e.get("job_id") == job_id]
    assert "JOB_RECOVERED" in kinds and kinds[-1] == "JOB_COMPLETED"


def test_runner_loads_the_compile_the_job_was_enqueued_with(line) -> None:
    from apps.production_line.runner.runner import JobRunner
    from apps.production_line.wiring import runner_config_for

    calls: list[tuple[str, str | None]] = []
    real = line.runtime.provider

    class SpyProvider:
        def load(self, recipe_id: str, *, compiled_part_sha256: str | None = None):
            calls.append((recipe_id, compiled_part_sha256))
            return real.load(recipe_id, compiled_part_sha256=compiled_part_sha256)

    entry = scan_entry(line, "nominal")
    drop_scan(line.inbox, oracle_variant(line, "nominal", "pinned"), entry["zeiss_filename"].replace(" 1.ply", " 5.ply"))
    outcomes = poll_until(line, lambda seen: any(o.action == "enqueued" for o in seen))
    job_id = next(o.job_id for o in outcomes if o.action == "enqueued")
    job = line.store.get_job(job_id)
    runner = JobRunner(
        runner_config_for(line.settings, engine_root=REPO, inspect_module=ORACLE_MODULE,
                          extra_env={"SHERMAN_E4_ORACLE_CASES": str(line.oracle_root)}),
        line.store, SpyProvider(), instance_id="runner@e2e-pinned",
    )
    assert runner.run_once() == job_id
    assert calls == [(job["recipe_id"], job["compiled_part_sha256"])]
    assert line.store.get_job(job_id)["execution_status"] == "completed"


def _enqueue_variant(line: SimpleNamespace, comment: str, counter: int) -> str:
    entry = scan_entry(line, "nominal")
    drop_scan(line.inbox, oracle_variant(line, "nominal", comment),
              entry["zeiss_filename"].replace(" 1.ply", f" {counter}.ply"))
    outcomes = poll_until(line, lambda seen: any(o.action == "enqueued" for o in seen))
    return next(o.job_id for o in outcomes if o.action == "enqueued")


def test_job_runs_against_its_pinned_compile_after_a_recompute_supersedes_it(line, tmp_path) -> None:
    """The case the runner's pinned load exists for (C05 compiled_history): two jobs are
    enqueued against compile X; a recompute installs compile Y for the same recipe. The
    first job still runs against X (now in history). With X's history copy gone, the second
    job fails closed on the compiled-part check instead of silently running against Y."""
    from apps.production_line.runner.runner import JobRunner
    from apps.production_line.wiring import runner_config_for
    from contracts.production.compiled_part import CompiledPartV1

    provider = line.runtime.provider
    recipe_id = line.release["recipe_id"]
    before = provider.load(recipe_id)
    orig_sha = before.compiled_part_sha256
    orig_copy = tmp_path / "orig_compile"
    shutil.copytree(before.compiled_dir, orig_copy)
    job_a = _enqueue_variant(line, "pin-a", 6)
    job_b = _enqueue_variant(line, "pin-b", 7)
    assert line.store.get_job(job_a)["compiled_part_sha256"] == orig_sha
    assert line.store.get_job(job_b)["compiled_part_sha256"] == orig_sha

    # A recompute for the same recipe: same geometry, another compiler version, so a new hash.
    recompute = tmp_path / "recompute"
    shutil.copytree(before.compiled_dir, recompute)
    doc = json.loads((recompute / "compiled_part.json").read_text(encoding="utf-8"))
    doc["compiler"]["version"] = doc["compiler"]["version"] + "+recompute"
    doc.pop("compiled_part_sha256")
    new_part = CompiledPartV1.finalize(**doc)
    assert new_part.compiled_part_sha256 != orig_sha
    (recompute / "compiled_part.json").write_text(new_part.model_dump_json(by_alias=True), encoding="utf-8")
    recipe_sha = line.store.get_job(job_a)["recipe_sha256"]
    provider.store_compiled(recipe_sha, recompute, status="COMPILED",
                            compiled_part_sha256=new_part.compiled_part_sha256, blocker_codes=())
    try:
        assert provider.load(recipe_id).compiled_part_sha256 == new_part.compiled_part_sha256
        runner = JobRunner(
            runner_config_for(line.settings, engine_root=REPO, inspect_module=ORACLE_MODULE,
                              extra_env={"SHERMAN_E4_ORACLE_CASES": str(line.oracle_root)}),
            line.store, provider, instance_id="runner@e2e-history", compiled_sink=provider,
        )
        assert runner.run_once() == job_a
        done = line.store.get_job(job_a)
        assert done["execution_status"] == "completed", done.get("failure")
        res = line.client.get(f"/api/line/jobs/{job_a}/result", headers=h("tok-viewer"))
        assert res.status_code == 200, res.text
        assert res.json()["engine"]["compiled_part_sha256"] == orig_sha  # ran against X, not Y

        # Mutate: X is gone from history -> the pinned job must not run against Y.
        history = line.data_root / "recipes" / "compiled_history" / recipe_sha / orig_sha
        assert history.is_dir(), "the superseded compile was not archived to compiled_history"
        shutil.rmtree(history)
        assert runner.run_once() == job_b
        failed = line.store.get_job(job_b)
        assert failed["execution_status"] == "failed"
        assert failed["failure"]["code"] in ("COMPILED_PART_HASH_MISMATCH", "COMPILED_PART_MISSING"), failed["failure"]
    finally:
        # Restore X as the installed compile for the remaining tests in this module.
        provider.store_compiled(recipe_sha, orig_copy, status="COMPILED", compiled_part_sha256=orig_sha, blocker_codes=())
        # Drain anything this test left queued (after an early failure) so later tests stay independent.
        for _ in range(4):
            if line.runner.run_once() is None:
                break
        assert provider.load(recipe_id).compiled_part_sha256 == orig_sha


def test_runner_still_supports_a_provider_without_the_pinned_keyword(line) -> None:
    """A provider whose load() has no compiled_part_sha256 keyword is called as before
    (mutation: always passing the keyword would raise TypeError and fail the job)."""
    from apps.production_line.runner.runner import JobRunner
    from apps.production_line.wiring import runner_config_for

    real = line.runtime.provider
    calls: list[str] = []

    class LegacyProvider:
        def load(self, recipe_id: str):
            calls.append(recipe_id)
            return real.load(recipe_id)

    job_id = _enqueue_variant(line, "legacy-provider", 8)
    runner = JobRunner(
        runner_config_for(line.settings, engine_root=REPO, inspect_module=ORACLE_MODULE,
                          extra_env={"SHERMAN_E4_ORACLE_CASES": str(line.oracle_root)}),
        line.store, LegacyProvider(), instance_id="runner@e2e-legacy",
    )
    assert runner.run_once() == job_id
    assert calls == [line.store.get_job(job_id)["recipe_id"]]
    assert line.store.get_job(job_id)["execution_status"] == "completed", line.store.get_job(job_id).get("failure")


def test_invalid_engine_output_fails_the_job_and_never_replaces_the_shown_verdict(line) -> None:
    from apps.production_line.runner.runner import JobRunner
    from apps.production_line.wiring import runner_config_for

    shown = latest(line)["latest_completed"]["job"]["job_id"]
    entry = scan_entry(line, "nominal")
    drop_scan(line.inbox, oracle_variant(line, "nominal", "garbage"), entry["zeiss_filename"].replace(" 1.ply", " 4.ply"))
    outcomes = poll_until(line, lambda seen: any(o.action == "enqueued" for o in seen))
    job_id = next(o.job_id for o in outcomes if o.action == "enqueued")
    bad = JobRunner(
        runner_config_for(line.settings, engine_root=REPO, inspect_module=ORACLE_MODULE,
                          extra_env={"SHERMAN_E4_ORACLE_CASES": str(line.oracle_root),
                                     "SHERMAN_E4_ORACLE_MODE": "garbage"}),
        line.store, line.runtime.provider, instance_id="runner@e2e-garbage", compiled_sink=line.runtime.provider,
    )
    assert bad.run_once() == job_id
    job = line.store.get_job(job_id)
    assert job["execution_status"] == "failed" and job["failure"]["code"] == "ENGINE_OUTPUT_INVALID"
    feed = latest(line)
    assert feed["latest_completed"]["job"]["job_id"] == shown  # the failed job never becomes the verdict
    assert feed["latest_issue"]["kind"] == "JOB_FAILED" and feed["latest_issue"]["reason_code"] == "ENGINE_OUTPUT_INVALID"
    assert line.client.get(f"/api/line/jobs/{job_id}/result", headers=h("tok-viewer")).status_code in (404, 409)


# ---------------------------------------------------------------------------- server stack


def test_access_log_redacts_stream_token_and_tags_request_id(line, caplog) -> None:
    caplog.set_level(logging.INFO, logger="sherman_line.access")
    r = line.client.get(f"/api/line/events/stream?station_id={STATION}&token=sst1.SECRETPART.SIG",
                        headers={"X-Request-ID": "req-e2e-0001"})
    assert r.status_code in (401, 403)
    assert r.headers.get("x-request-id") == "req-e2e-0001"
    access = [rec.getMessage() for rec in caplog.records if rec.name == "sherman_line.access"]
    assert access and any("/api/line/events/stream" in m for m in access)
    assert all("SECRETPART" not in m for m in access)
    assert any("token=%5BREDACTED%5D" in m or "token=[REDACTED]" in m for m in access)


def test_unconfigured_line_answers_503_through_the_same_stack() -> None:
    from apps.production_line.server import build_asgi
    from apps.production_line.security import dependencies as deps

    app = FastAPI()
    import apps.api.routes.production_line as line_routes

    app.include_router(line_routes.router)
    app.dependency_overrides[deps.get_line_settings] = lambda: None
    with TestClient(build_asgi(app, None)) as client:
        assert client.get(f"/api/line/stations/{STATION}/latest", headers=h()).status_code == 503


def test_fast_e2e_budget(line) -> None:
    # Setup (part set + E1 release) must leave room for the jobs inside the 90 s budget.
    assert line.setup_s < 60.0, f"fast E2E setup took {line.setup_s:.1f} s"


# ---------------------------------------------------------------------------- entry points (refusals)

_LINE_ENV = ("SHERMAN_LINE_ENABLED", "SHERMAN_LINE_DATA_ROOT", "SHERMAN_LINE_STATION_ID", "SHERMAN_LINE_INBOX",
             "SHERMAN_LINE_STATIONS", "SHERMAN_LINE_ENGINE_BUNDLE_DIR", "SHERMAN_PRODUCTION_HARDENING", "SECRET_KEY",
             "WEB_CONCURRENCY", "UVICORN_WORKERS", "PRODUCTION")


@pytest.fixture()
def clean_env(monkeypatch):
    """No line variables; the loggers that ``main`` configures are restored afterwards."""
    from apps.production_line.telemetry import DEFAULT_LOGGERS

    for name in _LINE_ENV:
        monkeypatch.delenv(name, raising=False)
    saved = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).propagate, logging.getLogger(n).level)
             for n in DEFAULT_LOGGERS}
    yield monkeypatch
    for n, (handlers, propagate, level) in saved.items():
        lg = logging.getLogger(n)
        for h_ in list(lg.handlers):
            if h_ not in handlers:
                lg.removeHandler(h_)
                h_.close()
        lg.propagate, lg.level = propagate, level


def test_server_refuses_before_importing_the_legacy_server(clean_env, tmp_path, capsys) -> None:
    from apps.production_line import server

    legacy_loaded = "_server_legacy_impl" in sys.modules
    clean_env.setenv("WEB_CONCURRENCY", "2")
    assert server.main(["--port", "1"]) == 2
    assert "MULTIPLE_WORKERS" in capsys.readouterr().err
    clean_env.delenv("WEB_CONCURRENCY")
    clean_env.setenv("SHERMAN_LINE_ENABLED", "1")  # enabled but no data root: never half-start
    assert server.main(["--port", "1"]) == 2
    assert "DATA_ROOT_MISSING" in capsys.readouterr().err
    clean_env.setenv("SHERMAN_LINE_DATA_ROOT", str(tmp_path / "root"))
    clean_env.setenv("SHERMAN_PRODUCTION_HARDENING", "1")
    assert server.main(["--port", "1"]) == 2  # SECRET_KEY unset
    err = capsys.readouterr().err
    assert "SECRET_KEY_MISSING_OR_DEFAULT" in err and "refusing to start" in err
    assert ("_server_legacy_impl" in sys.modules) == legacy_loaded, "a refused start imported the legacy server"
    assert not (tmp_path / "root").exists(), "a refused start created the data root"


def test_service_refusals_name_their_code(clean_env, tmp_path, capsys) -> None:
    from apps.production_line import service_main

    assert service_main.main(["--run-for", "0"]) == 2
    assert "LINE_DISABLED" in capsys.readouterr().err
    clean_env.setenv("SHERMAN_LINE_ENABLED", "1")
    clean_env.setenv("SHERMAN_LINE_DATA_ROOT", str(tmp_path / "root"))
    clean_env.setenv("SHERMAN_LINE_STATION_ID", STATION)
    clean_env.setenv("SHERMAN_LINE_INBOX", str(tmp_path / "missing-inbox"))
    assert service_main.main(["--run-for", "0"]) == 2
    assert "ENGINE_BUNDLE_MISSING" in capsys.readouterr().err
    # A bundle that does not verify is refused, never run.
    bundle = tmp_path / "root" / "engine" / "current"
    bundle.mkdir(parents=True)
    (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    assert service_main.main(["--run-for", "0"]) == 2
    assert "ENGINE_BUNDLE_INVALID" in capsys.readouterr().err


def test_service_builds_a_verified_bundle_then_refuses_a_missing_inbox(clean_env, tmp_path, capsys) -> None:
    from apps.production_line import service_main
    from apps.production_line.engine_bundle import verify

    clean_env.setenv("SHERMAN_LINE_ENABLED", "1")
    clean_env.setenv("SHERMAN_LINE_DATA_ROOT", str(tmp_path / "root"))
    clean_env.setenv("SHERMAN_LINE_STATION_ID", STATION)
    clean_env.setenv("SHERMAN_LINE_INBOX", str(tmp_path / "missing-inbox"))
    assert service_main.main(["--build-bundle", "--run-for", "0"]) == 2
    assert "STATION_INBOX_MISSING" in capsys.readouterr().err
    bundle = tmp_path / "root" / "engine" / "current"
    assert verify(bundle) == []
    assert (bundle / "domains" / "production" / "engine" / "inspect" / "__main__.py").is_file()
    assert not (bundle / "backend" / "_server_legacy_impl.py").exists(), "the engine bundle must not carry the legacy server"


def test_run_service_starts_and_stops_its_workers(tmp_path) -> None:
    import threading

    from apps.production_line.service_main import run_service
    from apps.production_line.settings import ensure_layout, load_settings
    from apps.production_line.wiring import build_runtime

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    settings = load_settings({"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root"),
                              "SHERMAN_LINE_STATION_ID": STATION, "SHERMAN_LINE_INBOX": str(inbox)}, repo_root=REPO)
    ensure_layout(settings.paths)
    runtime = build_runtime(settings, workers=True, catalog=SimpleNamespace(list_parts=lambda limit=0: []),
                            instance_id="runner@e2e-svc",
                            runner_overrides={"idle_poll_s": 0.05, "heartbeat_s": 0.05},
                            hotfolder_overrides={"poll_interval_s": 0.05})
    stop = threading.Event()
    t = threading.Timer(1.0, stop.set)
    t.start()
    t0 = time.monotonic()
    assert run_service(runtime, stop, heartbeat_s=0.1, grace_s=10.0) is True
    assert time.monotonic() - t0 < 10.0
    components = {row["component"] for row in runtime.line_store.get_heartbeats()}
    assert {"service", "runner", f"hotfolder:{STATION}"} <= components
    runtime.close()


def test_runner_config_never_loosens_the_engine_hard_stop(tmp_path) -> None:
    from apps.production_line.settings import load_settings
    from apps.production_line.wiring import runner_config_for, runner_instance_id

    settings = load_settings({"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root"),
                              "SHERMAN_LINE_JOB_TIMEOUT_S": "900", "SHERMAN_LINE_RSS_LIMIT_MB": "32768"}, repo_root=REPO)
    cfg = runner_config_for(settings)
    assert cfg.engine_timeout_s == 180.0 and cfg.engine_rss_limit_bytes == 32 * 1024**3  # ENGINE_IO-A5
    assert cfg.inspect_module == "domains.production.engine.inspect"
    assert cfg.engine_root == settings.engine_bundle_dir and cfg.engine_time_budget_s < cfg.engine_timeout_s
    assert cfg.extra_env["SHERMAN_QC_F1_PROGRAM_STATUS_PATH"] == str(settings.program_status_path)
    assert runner_instance_id("shop-pc.local") == "runner@shop-pc"


def test_redact_query_blanks_credentials_by_key() -> None:
    from apps.production_line.server import redact_query

    out = redact_query("station_id=cell-1&token=sst1.abc.def&access_token=xyz&after_id=3")
    assert "sst1" not in out and "xyz" not in out
    assert "station_id=cell-1" in out and "after_id=3" in out
    assert redact_query("") == ""


def test_fresh_on_miss_resolver_rechecks_only_misses() -> None:
    from apps.production_line.wiring import FreshOnMissResolver

    class Inner:
        def __init__(self, stale: bool) -> None:
            self.stale, self.invalidations, self.calls = stale, 0, 0
            self.data_root = Path("/x")

        def resolve(self, name: str):
            self.calls += 1
            return SimpleNamespace(status="UNMATCHED" if self.stale else "MATCHED")

        def invalidate(self, *_: Any) -> None:
            self.invalidations += 1
            self.stale = False

    stale = Inner(stale=True)  # released in another process after the snapshot was cached
    assert FreshOnMissResolver(stale).resolve("SKU_A_FULL_SCAN 1.ply").status == "MATCHED"
    assert (stale.invalidations, stale.calls) == (1, 2)
    fresh = Inner(stale=False)
    wrapped = FreshOnMissResolver(fresh)
    assert wrapped.resolve("SKU_A_FULL_SCAN 1.ply").status == "MATCHED"
    assert (fresh.invalidations, fresh.calls) == (0, 1)  # hits stay cached
    assert wrapped.data_root == Path("/x")  # everything else is delegated


def test_fresh_on_miss_resolver_rate_limits_forced_refreshes() -> None:
    from apps.production_line.wiring import FreshOnMissResolver

    class Inner:
        invalidations = 0

        def resolve(self, name: str):
            return SimpleNamespace(status="UNMATCHED")

        def invalidate(self, *_: Any) -> None:
            self.invalidations += 1

    now = [100.0]
    inner = Inner()
    wrapped = FreshOnMissResolver(inner, min_refresh_s=1.0, clock=lambda: now[0])
    for _ in range(50):  # a quarantine storm inside one second: one catalog re-read
        assert wrapped.resolve("UNKNOWN_A_FULL_SCAN 1.ply").status == "UNMATCHED"
        now[0] += 0.01
    assert inner.invalidations == 1 and wrapped.refresh_count == 1
    now[0] += 1.0
    wrapped.resolve("UNKNOWN_A_FULL_SCAN 2.ply")
    assert inner.invalidations == 2  # the window elapsed: the next miss refreshes again
    with pytest.raises(ValueError):
        FreshOnMissResolver(inner, min_refresh_s=-1)


def test_fresh_on_miss_resolver_serialises_station_threads() -> None:
    import threading

    from apps.production_line.wiring import FreshOnMissResolver

    class Inner:
        def __init__(self) -> None:
            self.inside, self.max_inside = 0, 0
            self.guard = threading.Lock()

        def _enter(self) -> None:
            with self.guard:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.002)
            with self.guard:
                self.inside -= 1

        def resolve(self, name: str):
            self._enter()
            return SimpleNamespace(status="UNMATCHED")

        def invalidate(self, *_: Any) -> None:
            self._enter()

    inner = Inner()
    wrapped = FreshOnMissResolver(inner, min_refresh_s=0.0)
    threads = [threading.Thread(target=lambda: [wrapped.resolve("X_A_FULL_SCAN 1.ply") for _ in range(20)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inner.max_inside == 1  # the shared inner resolver is never entered concurrently


def test_service_exit_code_distinguishes_an_abandoned_job(clean_env, tmp_path, monkeypatch, capsys) -> None:
    import signal as _signal

    from apps.production_line import service_main, wiring

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    monkeypatch.setenv("SHERMAN_LINE_ENABLED", "1")
    monkeypatch.setenv("SHERMAN_LINE_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("SHERMAN_LINE_STATION_ID", STATION)
    monkeypatch.setenv("SHERMAN_LINE_INBOX", str(inbox))
    monkeypatch.setattr(service_main, "prepare_engine_bundle", lambda settings, build=False: "bundle-test")
    monkeypatch.setattr(wiring, "build_runtime", lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(wiring, "service_catalog_port", lambda settings: None)
    monkeypatch.setattr(_signal, "signal", lambda *a, **k: None)
    for clean, code in ((True, 0), (False, service_main.EXIT_GRACE_ELAPSED)):
        monkeypatch.setattr(service_main, "run_service", lambda *a, _c=clean, **k: _c)
        assert service_main.main(["--run-for", "0"]) == code
    assert service_main.EXIT_GRACE_ELAPSED not in (0, 2)
    assert "GRACE_ELAPSED" in capsys.readouterr().err
