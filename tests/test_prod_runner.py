"""C02 job runner: preflight hash checks, E3 + C06 supervised steps, failure
mapping, cancel, lease loss, durability, recipe-compile jobs.

The engine bundle is a tmp directory holding fake ``fake_inspect`` /
``fake_render`` / ``fake_compile`` modules (script strings below) that speak
the step protocol. The recipe, compiled part and engine result are the C01
SYN-HAT-1 examples re-finalised around synthetic CAD and stage bytes, so the
runner's real contract validation and ``check_result_against_compiled`` run
on them. No real scans or CAD are read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.production_line.ingest.ports import ResolvedRecipe
from apps.production_line.runner.runner import (
    CONTEXT_FILLED_WARNING,
    JobRunner,
    RunnerConfig,
    build_step_env,
    lease_owner_for,
    lease_owner_prefix,
)
from apps.production_line.store.line_store import LineStore, scan_file_id_for

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "contracts" / "production" / "examples"
ADMIN = {"kind": "user", "id": "admin", "role": "admin"}
SCAN_NAME = "SYN-HAT-1_A_FULL_SCAN 1.ply"

FAKE_INSPECT = r'''
import argparse, json, os, pathlib, sys, time
ap = argparse.ArgumentParser()
# No --context (ENGINE_IO §4): argparse exits 2 if the runner ever passes job identity again.
for a in ["--compiled", "--recipe", "--scan", "--out", "--scan-sha256", "--compiled-sha256", "--recipe-sha256",
          "--time-budget-s", "--declared-stage", "--declared-stage-source", "--engine-bundle-id", "--config"]:
    ap.add_argument(a)
args = ap.parse_args()
mode = os.environ.get("FAKE_ENGINE_MODE", "ok")
d = pathlib.Path(os.environ["FAKE_DIR"])
old_alive = None
if os.environ.get("FAKE_CHECK_PGID"):
    try:
        os.killpg(int(os.environ["FAKE_CHECK_PGID"]), 0)
        old_alive = True
    except ProcessLookupError:
        old_alive = False
(d / "engine_marker.json").write_text(json.dumps({
    "argv": sys.argv[1:], "cwd": os.getcwd(), "secret": "TYPESAFE_API_KEY" in os.environ,
    "omp": os.environ.get("OMP_NUM_THREADS"), "pythonpath": os.environ.get("PYTHONPATH"),
    "out_existed": os.path.exists(args.out), "pgid": os.getpgid(0), "old_alive": old_alive}))
print("PROGRESS\t" + json.dumps({"stage": "registration", "progress": 50}), flush=True)
out = pathlib.Path(args.out)
if mode == "sleep":
    time.sleep(120)
if mode == "crash":
    print("Traceback (most recent call last): fake segfault", flush=True)
    sys.exit(1)
if mode == "error3":
    print("ERROR\t" + json.dumps({"code": "INTERNAL", "message": "engine bug"}), flush=True)
    sys.exit(3)
if mode == "exit3silent":
    print("internal state dump", flush=True)
    sys.exit(3)
if mode == "input2":
    out.mkdir(parents=True)
    (out / "engine_error.json").write_text(json.dumps({"code": "SCAN_HASH_MISMATCH", "message": "scan bytes differ"}))
    sys.exit(2)
if mode == "noresult":
    sys.exit(0)
result = json.loads((d / "template_result.json").read_text())
result["context"] = {}  # E3 writes an empty ResultContextV1
result["scan"]["sha256"] = args.scan_sha256
if mode == "inconsistent":
    result["engine"]["compiled_part_sha256"] = "0" * 64
if mode == "badcontext":
    result["context"] = {"serial": "SN0042"}  # identity leaked into the engine
if mode == "wrongscan":
    result["scan"]["sha256"] = "0" * 64
out.mkdir(parents=True)
if mode == "garbage":
    (out / "result.json").write_text("{not json")
else:
    (out / "result.json").write_text(json.dumps(result))
claimed = "ALL_MEASURED_IN_TOLERANCE" if mode == "lie" else result["summary"]["state"]
print("RESULT\t" + json.dumps({"summary_state": claimed, "inspection_validity": "VALID", "result": "result.json"}), flush=True)
'''

FAKE_RENDER = r'''
import argparse, json, os, pathlib, sys, time
ap = argparse.ArgumentParser()
for a in ["--job-dir", "--result", "--out", "--engine-bundle-id"]:
    ap.add_argument(a)
args = ap.parse_args()
mode = os.environ.get("FAKE_RENDER_MODE", "ok")
d = pathlib.Path(os.environ["FAKE_DIR"])
(d / "render_marker.json").write_text(json.dumps({"argv": sys.argv[1:]}))
out = pathlib.Path(args.out)
if mode == "sleep":
    time.sleep(120)
if mode == "exit2":
    (out / "postprocess_error.json").write_text(json.dumps({"code": "VIEWER_ASSETS_MISSING", "message": "registered_scan.ply absent"}))
    print("ERROR\t" + json.dumps({"code": "VIEWER_ASSETS_MISSING", "message": "registered_scan.ply absent"}), flush=True)
    sys.exit(2)
if mode == "nooutput":
    sys.exit(0)
result = json.loads(pathlib.Path(args.result).read_text())
result["warnings"] = list(result.get("warnings", [])) + ["PDF_FONT_FALLBACK"]
if mode != "nocontext":  # C06 fills the job identity from input/context.json (ENGINE_IO §5)
    result["context"] = json.loads((pathlib.Path(args.job_dir) / "input" / "context.json").read_text())
if mode == "alter":
    result["part_name"] = "a different part"
if mode == "wrongcontext":
    result["context"]["serial"] = "SOMEONE-ELSE"
(out / "shopfloor_result.json").write_text(json.dumps(result))
print("PROGRESS\t" + json.dumps({"stage": "report", "progress": 90}), flush=True)
print("RESULT\t" + json.dumps({"shopfloor_result": "shopfloor_result.json"}), flush=True)
'''

FAKE_COMPILE = r'''
import argparse, json, os, pathlib, shutil, sys
ap = argparse.ArgumentParser()
for a in ["--recipe", "--cad", "--flat", "--out", "--config", "--shop-k"]:
    ap.add_argument(a)
args = ap.parse_args()
mode = os.environ.get("FAKE_COMPILE_MODE", "ok")
d = pathlib.Path(os.environ["FAKE_DIR"])
(d / "compile_marker.json").write_text(json.dumps({"argv": sys.argv[1:]}))
out = pathlib.Path(args.out)
if mode == "badinput":
    out.mkdir(parents=True)
    (out / "compile_error.json").write_text(json.dumps({"code": "CAD_HASH_MISMATCH", "message": "cad bytes differ"}))
    sys.exit(2)
shutil.copytree(d / "compiled_src", out)
print("RESULT\t" + json.dumps({"status": "COMPILED", "out": str(out)}), flush=True)
'''


def ply_bytes(seed: int = 0) -> bytes:
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    return head.encode() + b"".join(struct.pack("<fff", i + seed, i, 0.0) for i in range(3))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 17, 7, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class Provider:
    def __init__(self) -> None:
        self.recipes: dict[str, ResolvedRecipe] = {}

    def load(self, recipe_id: str) -> ResolvedRecipe:
        return self.recipes[recipe_id]


class Sink:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def store_compiled(self, recipe_sha256, compile_out_dir, *, status, compiled_part_sha256, blocker_codes):
        self.calls.append(
            {
                "recipe_sha256": recipe_sha256,
                "files": sorted(str(p.relative_to(compile_out_dir)) for p in Path(compile_out_dir).rglob("*") if p.is_file()),
                "status": status,
                "compiled_part_sha256": compiled_part_sha256,
                "blocker_codes": blocker_codes,
            }
        )


@dataclass
class Line:
    root: Path
    engine_root: Path
    fake_dir: Path
    store: LineStore
    provider: Provider
    recipe: object
    compiled: object
    cad_path: Path
    compiled_dir: Path
    scan_bytes: bytes
    archived: Path
    job_id: str

    def config(self, **kw) -> RunnerConfig:
        env = {"FAKE_DIR": str(self.fake_dir), **kw.pop("extra_env", {})}
        defaults = dict(
            engine_root=self.engine_root,
            python_exe=Path(sys.executable),
            data_root=self.root,
            inspect_module="fake_inspect",
            render_module="fake_render",
            compile_module="fake_compile",
            supervisor_poll_s=0.1,
            lease_ttl_s=30,
            heartbeat_s=0.5,
            kill_grace_s=2,
            engine_bundle_id="bundle-test",
            extra_env=env,
        )
        defaults.update(kw)
        return RunnerConfig(**defaults)

    def runner(self, sink=None, **kw) -> JobRunner:
        return JobRunner(self.config(**kw), self.store, self.provider, compiled_sink=sink)

    def ws(self) -> Path:
        return self.root / "jobs" / self.job_id


def build_fixture_docs(cad: bytes):
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.inspection_result import ShopfloorInspectionResultV1
    from contracts.production.part_recipe import PartRecipeV1

    cad_sha = sha(cad)
    r = json.loads((EXAMPLES / "part_recipe_v1.example.json").read_text())
    r["cad"]["sha256"] = cad_sha
    r["cad"]["stored_rel_path"] = f"recipes/cad/{cad_sha}.step"
    for key in ("content_sha256", "recipe_sha256"):
        r.pop(key)
    recipe = PartRecipeV1.finalize(**r)
    c = json.loads((EXAMPLES / "compiled_part_v1.example.json").read_text())
    c["recipe"].update(recipe_sha256=recipe.recipe_sha256, content_sha256=recipe.content_sha256)
    c["cad_sha256"] = cad_sha
    stage_files: dict[str, bytes] = {}
    for stage in c["stages"]:
        for key in ("sample_cloud", "display_mesh"):
            data = f"{stage['stage_model_id']}:{key}".encode()
            stage_files[stage[key]["rel_path"]] = data
            stage[key]["sha256"] = sha(data)
    c.pop("compiled_part_sha256")
    compiled = CompiledPartV1.finalize(**c)
    res = json.loads((EXAMPLES / "shopfloor_result_v1.example.json").read_text())
    res["recipe"].update(recipe_sha256=recipe.recipe_sha256, content_sha256=recipe.content_sha256)
    res["engine"]["compiled_part_sha256"] = compiled.compiled_part_sha256
    ShopfloorInspectionResultV1.model_validate(res)
    return recipe, compiled, stage_files, res


def write_compiled(directory: Path, compiled, stage_files: dict[str, bytes]) -> None:
    from contracts.production.canonical import canonical_json_bytes

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "compiled_part.json").write_bytes(canonical_json_bytes(compiled))
    for rel, data in stage_files.items():
        (directory / rel).parent.mkdir(parents=True, exist_ok=True)
        (directory / rel).write_bytes(data)


@pytest.fixture()
def line(tmp_path) -> Line:
    from contracts.production.canonical import canonical_json_bytes

    root = tmp_path / "data_root"
    engine_root = tmp_path / "engine_bundle"
    fake_dir = tmp_path / "fake"
    for d in (root, engine_root, fake_dir):
        d.mkdir()
    (engine_root / "fake_inspect.py").write_text(textwrap.dedent(FAKE_INSPECT))
    (engine_root / "fake_render.py").write_text(textwrap.dedent(FAKE_RENDER))
    (engine_root / "fake_compile.py").write_text(textwrap.dedent(FAKE_COMPILE))

    cad = b"ISO-10303-21;\nHEADER; synthetic SYN-HAT-1 test CAD\n"
    recipe, compiled, stage_files, result = build_fixture_docs(cad)
    cad_path = root / recipe.cad.stored_rel_path
    cad_path.parent.mkdir(parents=True)
    cad_path.write_bytes(cad)
    compiled_dir = root / "recipes" / "compiled" / recipe.recipe_sha256
    write_compiled(compiled_dir, compiled, stage_files)
    write_compiled(fake_dir / "compiled_src", compiled, stage_files)
    (fake_dir / "template_result.json").write_text(json.dumps(result))

    provider = Provider()
    provider.recipes[recipe.recipe_id] = ResolvedRecipe(
        recipe_id=recipe.recipe_id,
        recipe_version=recipe.recipe_version,
        recipe_sha256=recipe.recipe_sha256,
        sku=recipe.sku,
        recipe_json_bytes=canonical_json_bytes(recipe),
        cad_path=cad_path,
        cad_sha256=recipe.cad.sha256,
        part_name=recipe.part_name,
        compiled_dir=compiled_dir,
        compiled_part_sha256=compiled.compiled_part_sha256,
    )

    store = LineStore(root / "db" / "line.sqlite3")
    scan = ply_bytes()
    scan_sha = sha(scan)
    rel = f"archive/scans/2026/10/{scan_sha}.ply"
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_bytes(scan)
    row, _ = store.register_scan(
        sha256=scan_sha,
        original_name=SCAN_NAME,
        size_bytes=len(scan),
        station_id="cell-1",
        stored_rel_path=rel,
        parse={"status": "OK", "sku": recipe.sku, "serial": "SN0042"},
        match={"status": "MATCHED", "recipe_id": recipe.recipe_id, "recipe_version": 1, "recipe_sha256": recipe.recipe_sha256},
        format="PLY_BINARY_LE",
        vertex_count=3,
        face_count=0,
    )
    job_id, _ = store.match_and_enqueue(
        row["scan_file_id"],
        match={"status": "MATCHED", "recipe_id": recipe.recipe_id, "recipe_version": 1, "recipe_sha256": recipe.recipe_sha256},
        sku=recipe.sku,
        compiled_part_sha256=compiled.compiled_part_sha256,
    )
    return Line(root, engine_root, fake_dir, store, provider, recipe, compiled, cad_path, compiled_dir, scan, root / rel, job_id)


def failure_of(line: Line) -> dict:
    job = line.store.get_job(line.job_id)
    assert job["execution_status"] == "failed", job
    return job["failure"]


# ---------------------------------------------------------------- happy path
def test_completed_job_carries_summary_and_workspace(line, monkeypatch):
    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.line_job import LineJobV1

    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test-never-forwarded")
    runner = line.runner()
    assert runner.run_once() == line.job_id
    job = line.store.get_job(line.job_id)
    assert job["execution_status"] == "completed", job["failure"]
    template = json.loads((line.fake_dir / "template_result.json").read_text())
    assert job["result_summary"] == {
        "inspection_validity": template["inspection_validity"]["state"],
        "summary_state": template["summary"]["state"],
        "bends_total": template["summary"]["bends_total"],
        "required_fail": template["summary"]["required_fail"],
        "required_review": template["summary"]["required_review"],
        "required_unknown": template["summary"]["required_unknown"],
        "stage_status": template["stage_decision"]["status"],
    }
    assert job["engine_bundle_id"] == "bundle-test" and job["attempts"] == 1
    LineJobV1.model_validate(job)

    ws = line.ws()
    assert (ws / "recipe.json").read_bytes() == canonical_json_bytes(line.recipe)
    assert sha((ws / "input" / "scan.ply").read_bytes()) == job["scan_sha256"]
    assert json.loads((ws / "job.json").read_text())["job_id"] == line.job_id
    context = json.loads((ws / "input" / "context.json").read_text())
    assert context["scan_file_name"] == SCAN_NAME and context["serial"] == "SN0042"
    # The studio (CAD) bend labels ride in the context for display (station + PDF).
    assert context["bend_display_labels"] == {b.bend_uid: b.cad_label for b in line.recipe.bends}
    assert (ws / "compiled" / "compiled_part.json").is_file() and (ws / "compiled" / "stages" / "S4.npz").is_file()
    final = json.loads((ws / "result" / "shopfloor_result.json").read_text())
    assert final["context"] == context  # C06 filled the identity; the engine never had it
    assert CONTEXT_FILLED_WARNING not in final["warnings"]  # no runner fallback was needed
    assert job["lease_owner"] is None or job["lease_owner"].startswith(lease_owner_prefix(runner.instance_id))
    assert (ws / "logs" / "attempt-1" / "engine_step.log").is_file()  # logs are kept per attempt
    assert not (ws / "logs" / "active_step.json").exists()  # no step left running

    marker = json.loads((line.fake_dir / "engine_marker.json").read_text())
    argv = marker["argv"]

    def arg(name: str) -> str:
        return argv[argv.index(name) + 1]

    assert arg("--scan-sha256") == job["scan_sha256"]
    assert arg("--compiled-sha256") == line.compiled.compiled_part_sha256
    assert arg("--recipe-sha256") == line.recipe.recipe_sha256
    assert arg("--scan").endswith("input/scan.ply") and arg("--out").endswith("/engine")
    assert "--context" not in argv and arg("--engine-bundle-id") == "bundle-test"
    assert str(ws / "input" / "context.json") not in argv  # the engine never sees job identity
    assert float(arg("--time-budget-s")) <= 175
    assert not any("FULL_SCAN" in a or "SN0042" in a for a in argv)  # engine blindness: no filename tokens
    assert marker["secret"] is False  # whitelisted env: secrets never reach a step
    assert marker["omp"] and marker["out_existed"] is False
    assert Path(marker["cwd"]).resolve() == line.engine_root.resolve()
    assert marker["pythonpath"].split(os.pathsep)[0] == str(line.engine_root)
    render = json.loads((line.fake_dir / "render_marker.json").read_text())["argv"]
    assert render[render.index("--out") + 1].endswith("/result")

    kinds = [e["kind"] for e in line.store.events_after(0, limit=500)]
    assert kinds[-1] == "JOB_COMPLETED" and "JOB_STARTED" in kinds and "JOB_PROGRESS" in kinds
    assert runner.run_once() is None  # queue empty


def test_step_env_is_whitelisted(line, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    env = build_step_env(line.config(thread_env={"OMP_NUM_THREADS": "1"}))
    assert "TYPESAFE_API_KEY" not in env and "AWS_SECRET_ACCESS_KEY" not in env
    assert env["OMP_NUM_THREADS"] == "1" and env["FAKE_DIR"] == str(line.fake_dir)
    assert env["PYTHONPATH"] == os.pathsep.join([str(line.engine_root), str(line.engine_root / "backend")])


# ---------------------------------------------------------------- preflight
def test_cad_hash_mismatch_never_spawns(line):
    line.cad_path.write_bytes(b"tampered CAD")
    line.runner().run_once()
    assert failure_of(line)["code"] == "RECIPE_CAD_HASH_MISMATCH"
    assert not (line.fake_dir / "engine_marker.json").exists()


def _tamper_recipe(line):
    from dataclasses import replace

    data = json.loads(line.provider.recipes[line.recipe.recipe_id].recipe_json_bytes)
    data["part_name"] = "edited after release"
    line.provider.recipes[line.recipe.recipe_id] = replace(
        line.provider.recipes[line.recipe.recipe_id], recipe_json_bytes=json.dumps(data).encode()
    )


def _tamper_stage(line):
    (line.compiled_dir / "stages" / "S2.ply").write_bytes(b"swapped mesh")


def _tamper_compiled_json(line):
    p = line.compiled_dir / "compiled_part.json"
    data = json.loads(p.read_text())
    data["compiler"]["version"] = "edited"
    p.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda line: line.provider.recipes.clear(), "RECIPE_MISSING"),
        (_tamper_recipe, "RECIPE_SHA_MISMATCH"),
        (lambda line: line.cad_path.unlink(), "RECIPE_CAD_MISSING"),
        (lambda line: (line.compiled_dir / "compiled_part.json").unlink(), "COMPILED_PART_MISSING"),
        (_tamper_compiled_json, "COMPILED_PART_HASH_MISMATCH"),
        (_tamper_stage, "COMPILED_PART_HASH_MISMATCH"),
        (lambda line: line.archived.unlink(), "SCAN_MISSING"),
        (lambda line: line.archived.write_bytes(ply_bytes(seed=7)), "SCAN_HASH_MISMATCH"),
    ],
)
def test_preflight_failures_never_spawn(line, mutate, code):
    mutate(line)
    line.runner().run_once()
    assert failure_of(line)["code"] == code
    assert not (line.fake_dir / "engine_marker.json").exists()


def test_job_pinned_to_other_compile_fails(line):
    line.store.request_cancel(line.job_id, ADMIN)  # retire the fixture job
    scan2 = ply_bytes(seed=3)
    s2 = sha(scan2)
    rel = f"archive/scans/2026/10/{s2}.ply"
    (line.root / rel).write_bytes(scan2)
    match = {"status": "MATCHED", "recipe_id": line.recipe.recipe_id, "recipe_version": 1, "recipe_sha256": line.recipe.recipe_sha256}
    row, _ = line.store.register_scan(
        sha256=s2, original_name="x.ply", size_bytes=len(scan2), station_id="cell-1", stored_rel_path=rel,
        parse={"status": "OK"}, match=match, format="PLY_BINARY_LE",
    )  # fmt: skip
    job_id, _ = line.store.match_and_enqueue(row["scan_file_id"], match=match, sku=line.recipe.sku, compiled_part_sha256="9" * 64)
    line.runner().run_once()
    assert line.store.get_job(job_id)["failure"]["code"] == "COMPILED_PART_HASH_MISMATCH"


# ---------------------------------------------------------------- engine step mapping
@pytest.mark.parametrize(
    ("mode", "code", "needle"),
    [
        ("crash", "WORKER_CRASH", "fake segfault"),
        ("error3", "WORKER_ERROR", "engine bug"),
        ("exit3silent", "WORKER_ERROR", "internal error"),
        ("input2", "SCAN_HASH_MISMATCH", "scan bytes differ"),
        ("noresult", "WORKER_NO_RESULT", "RESULT"),
        ("inconsistent", "ENGINE_OUTPUT_INVALID", "COMPILED_PART_REF_MISMATCH"),
        ("lie", "ENGINE_OUTPUT_INVALID", "RESULT line says"),
        ("garbage", "ENGINE_OUTPUT_INVALID", "does not validate"),
        ("badcontext", "ENGINE_OUTPUT_INVALID", "context"),
        ("wrongscan", "ENGINE_OUTPUT_INVALID", "result.scan.sha256"),
    ],
)
def test_engine_failure_mapping(line, mode, code, needle):
    line.runner(extra_env={"FAKE_ENGINE_MODE": mode}).run_once()
    failure = failure_of(line)
    assert failure["code"] == code and needle in failure["message"], failure
    assert line.store.get_job(line.job_id)["result_summary"] is None
    assert not (line.fake_dir / "render_marker.json").exists()  # nothing rendered from a bad engine run


def test_engine_timeout(line):
    started = time.monotonic()
    line.runner(engine_timeout_s=1.0, extra_env={"FAKE_ENGINE_MODE": "sleep"}).run_once()
    assert failure_of(line)["code"] == "WORKER_TIMEOUT"
    assert time.monotonic() - started < 1.0 + 12


def test_engine_limits_cannot_be_loosened(line):
    with pytest.raises(ValueError):
        line.runner(engine_timeout_s=600)
    with pytest.raises(ValueError):
        line.runner(engine_rss_limit_bytes=33 * 1024**3)


def test_cancel_during_engine_ends_within_three_seconds(line):
    runner = line.runner(extra_env={"FAKE_ENGINE_MODE": "sleep"})
    done = threading.Event()
    t = threading.Thread(target=lambda: (runner.run_once(), done.set()))
    t.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (line.fake_dir / "engine_marker.json").exists():
        time.sleep(0.05)
    assert (line.fake_dir / "engine_marker.json").exists()
    asked = time.monotonic()
    assert line.store.request_cancel(line.job_id, ADMIN) == "cancel_requested"
    assert done.wait(10)
    assert time.monotonic() - asked < 3.0
    t.join()
    job = line.store.get_job(line.job_id)
    assert job["execution_status"] == "cancelled" and job["failure"] is None


def test_lease_loss_kills_step_and_writes_nothing(line):
    runner = line.runner(extra_env={"FAKE_ENGINE_MODE": "sleep"}, lease_ttl_s=3, heartbeat_s=0.2)
    t = threading.Thread(target=runner.run_once)
    t.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (line.fake_dir / "engine_marker.json").exists():
        time.sleep(0.05)
    conn = sqlite3.connect(line.root / "db" / "line.sqlite3")
    conn.execute("UPDATE jobs SET lease_owner='other-runner' WHERE job_id=?", (line.job_id,))
    conn.commit()
    conn.close()
    t.join(15)
    assert not t.is_alive()
    job = line.store.get_job(line.job_id)
    assert job["execution_status"] == "running" and job["lease_owner"] == "other-runner"


# ---------------------------------------------------------------- render step mapping
def test_render_exit2_carries_postprocess_error_message(line):
    line.runner(extra_env={"FAKE_RENDER_MODE": "exit2"}).run_once()
    failure = failure_of(line)
    assert failure["code"] == "POSTPROCESS_ERROR"
    assert failure["message"] == "VIEWER_ASSETS_MISSING: registered_scan.ply absent"


@pytest.mark.parametrize(
    ("mode", "code", "needle"),
    [
        ("alter", "POSTPROCESS_ERROR", "RESULT_ALTERED"),
        ("wrongcontext", "POSTPROCESS_ERROR", "result.context is not input/context.json"),
        ("nooutput", "POSTPROCESS_ERROR", "RESULT_MISSING"),
    ],
)
def test_render_must_not_alter_or_skip(line, mode, code, needle):
    line.runner(extra_env={"FAKE_RENDER_MODE": mode}).run_once()
    failure = failure_of(line)
    assert failure["code"] == code and needle in failure["message"]
    assert line.store.get_job(line.job_id)["result_summary"] is None


def test_runner_fills_context_when_render_leaves_it_empty(line):
    """ENGINE_IO §5: "C06 (or the runner)" adds the job identity after the engine."""
    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.inspection_result import ShopfloorInspectionResultV1

    line.runner(extra_env={"FAKE_RENDER_MODE": "nocontext"}).run_once()
    job = line.store.get_job(line.job_id)
    assert job["execution_status"] == "completed", job["failure"]
    ws = line.ws()
    raw = (ws / "result" / "shopfloor_result.json").read_bytes()
    final = ShopfloorInspectionResultV1.model_validate_json(raw)
    assert raw == canonical_json_bytes(final)  # rewritten canonically
    assert final.context.model_dump(mode="json") == json.loads((ws / "input" / "context.json").read_text())
    assert final.context.job_id == line.job_id and final.context.serial == "SN0042"
    engine = json.loads((ws / "engine" / "result.json").read_text())
    assert engine["context"] == {} or all(v is None for v in engine["context"].values())
    # nothing but context changed relative to what the render step wrote
    assert final.summary.state == engine["summary"]["state"] and "PDF_FONT_FALLBACK" in final.warnings
    # the fallback is visible in the result: the PDF rendered before it has no job identity
    assert final.warnings[-1] == CONTEXT_FILLED_WARNING and final.warnings.count(CONTEXT_FILLED_WARNING) == 1


def test_render_timeout(line):
    line.runner(render_timeout_s=1.0, extra_env={"FAKE_RENDER_MODE": "sleep"}).run_once()
    assert failure_of(line)["code"] == "POSTPROCESS_TIMEOUT"


# ---------------------------------------------------------------- durability
def test_durability_recover_requeue_then_interrupted(tmp_path):
    clock = FakeClock()
    db = tmp_path / "root" / "db" / "line.sqlite3"
    store = LineStore(db, clock=clock)
    scan_sha = "a" * 64
    match = {"status": "MATCHED", "recipe_id": "P@v1", "recipe_version": 1, "recipe_sha256": "b" * 64}
    row, _ = store.register_scan(
        sha256=scan_sha, original_name="P_A.ply", size_bytes=10, station_id="cell-1",
        stored_rel_path=f"archive/scans/2026/10/{scan_sha}.ply", parse={"status": "OK"}, match=match,
    )  # fmt: skip
    job_id, _ = store.match_and_enqueue(row["scan_file_id"], match=match, sku="P", compiled_part_sha256="c" * 64)
    store.lease_next("crashed-runner", 60)
    del store  # crash: only the file remains
    clock.advance(61)
    cfg = RunnerConfig(engine_root=tmp_path, python_exe=Path(sys.executable), data_root=tmp_path / "root")
    store2 = LineStore(db, clock=clock)
    JobRunner(cfg, store2, Provider())  # startup recovers
    job = store2.get_job(job_id)
    assert job["execution_status"] == "queued" and job["attempts"] == 1
    store2.lease_next("crashed-again", 60)
    clock.advance(61)
    store3 = LineStore(db, clock=clock)
    JobRunner(cfg, store3, Provider())
    job = store3.get_job(job_id)
    assert job["execution_status"] == "failed" and job["failure"]["code"] == "WORKER_INTERRUPTED"
    assert [e["payload"]["action"] for e in store3.events_after(0) if e["kind"] == "JOB_RECOVERED"] == ["requeued", "failed"]
    again, created = store3.match_and_enqueue(scan_file_id_for(scan_sha), match=match, sku="P", compiled_part_sha256="c" * 64)
    assert (again, created) == (job_id, False)  # dedupe: one job per scan + recipe


def test_restart_with_stable_instance_id_reclaims_dead_lease_at_once(line):
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead_pid = int(dead.stdout)  # a pid that has exited
    line.store.lease_next(lease_owner_for("runner", dead_pid), 600)  # lease far from expiry
    foreign = line.store.enqueue_job("recipe_compile", sku="P", recipe_id="P@v1", recipe_version=1, recipe_sha256="f" * 64)[0]
    line.store.lease_next(lease_owner_for("other", dead_pid), 600)
    runner = JobRunner(line.config(), line.store, line.provider, instance_id="runner")
    assert line.store.get_job(line.job_id)["execution_status"] == "queued"  # reclaimed without the TTL
    assert line.store.get_job(foreign)["execution_status"] == "running"  # someone else's lease is untouched
    assert runner.run_once() == line.job_id
    assert line.store.get_job(line.job_id)["execution_status"] == "completed"


def test_same_instance_id_on_another_host_is_never_reclaimed_early(line):
    """A pid only means something on the host that issued it: a lease held by
    the same instance_id on another host waits for the TTL, even when that pid
    is dead (or reused) here."""
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead_pid = int(dead.stdout)
    remote = lease_owner_for("runner", dead_pid, host="another-host.invalid")
    line.store.lease_next(remote, 600)
    legacy = line.store.enqueue_job("recipe_compile", sku="P", recipe_id="P@v1", recipe_version=1, recipe_sha256="f" * 64)[0]
    line.store.lease_next(f"runner:{dead_pid}", 600)  # host-less owner: never probed
    JobRunner(line.config(), line.store, line.provider, instance_id="runner")
    assert line.store.get_job(line.job_id)["execution_status"] == "running"
    assert line.store.get_job(line.job_id)["lease_owner"] == remote
    assert line.store.get_job(legacy)["execution_status"] == "running"
    assert lease_owner_for("runner", 7, host="h") == "runner@h:7" and lease_owner_prefix("runner", "h") == "runner@h:"


def test_retry_after_failure_runs_again(line):
    line.cad_path.write_bytes(b"tampered")
    runner = line.runner()
    runner.run_once()
    assert failure_of(line)["code"] == "RECIPE_CAD_HASH_MISMATCH"
    line.cad_path.write_bytes(b"ISO-10303-21;\nHEADER; synthetic SYN-HAT-1 test CAD\n")
    retry = line.store.request_retry(line.job_id, ADMIN)
    assert runner.run_once() == retry
    assert line.store.get_job(retry)["execution_status"] == "completed"


# ---------------------------------------------------------------- recipe compile jobs
def _enqueue_compile(line) -> str:
    job_id, _ = line.store.enqueue_job(
        "recipe_compile",
        sku=line.recipe.sku,
        recipe_id=line.recipe.recipe_id,
        recipe_version=line.recipe.recipe_version,
        recipe_sha256=line.recipe.recipe_sha256,
        dedupe_key=f"compile:{line.recipe.recipe_sha256}",
    )
    line.store.request_cancel(line.job_id, ADMIN)  # keep the inspect fixture job out of the way
    return job_id


def test_recipe_compile_job(line):
    job_id = _enqueue_compile(line)
    sink = Sink()
    assert line.runner(sink=sink).run_once() == job_id
    job = line.store.get_job(job_id)
    assert job["execution_status"] == "completed" and job["result_summary"] is None
    assert sink.calls == [
        {
            "recipe_sha256": line.recipe.recipe_sha256,
            "files": sorted(["compiled_part.json"] + [f"stages/S{k}.{e}" for k in range(5) for e in ("npz", "ply")]),
            "status": "COMPILED",
            "compiled_part_sha256": line.compiled.compiled_part_sha256,
            "blocker_codes": (),
        }
    ]
    argv = json.loads((line.fake_dir / "compile_marker.json").read_text())["argv"]
    assert argv[argv.index("--cad") + 1] == str(line.cad_path)
    assert "RECIPE_COMPILED" in [e["kind"] for e in line.store.events_after(0, limit=500)]


def test_recipe_compile_input_error_and_missing_sink(line):
    job_id = _enqueue_compile(line)
    line.runner(sink=Sink(), extra_env={"FAKE_COMPILE_MODE": "badinput"}).run_once()
    assert line.store.get_job(job_id)["failure"]["code"] == "RECIPE_CAD_HASH_MISMATCH"
    assert "RECIPE_COMPILE_FAILED" in [e["kind"] for e in line.store.events_after(0, limit=500)]
    retry = line.store.request_retry(job_id, ADMIN)
    line.runner(sink=None).run_once()
    assert line.store.get_job(retry)["failure"]["code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------- import purity (acceptance)
def test_backbone_imports_stay_light():
    code = (
        "import sys, apps.production_line.store.line_store, apps.production_line.ingest.hotfolder, "
        "apps.production_line.runner.runner; "
        "bad=[m for m in sys.modules if m.startswith(('_server_legacy_impl','apps.api','open3d','numpy',"
        "'contracts.production','domains.production'))]; assert not bad, bad"
    )
    env = {**os.environ, "PYTHONPATH": ".:backend"}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO, env=env)


# ---------------------------------------------------------------- review fixes (fix stage)
def _enqueue_second_scan(line, compiled_sha: str) -> str:
    """Retire the fixture job and enqueue a fresh scan pinned to ``compiled_sha``."""
    line.store.request_cancel(line.job_id, ADMIN)
    scan2 = ply_bytes(seed=5)
    s2 = sha(scan2)
    rel = f"archive/scans/2026/10/{s2}.ply"
    (line.root / rel).write_bytes(scan2)
    match = {"status": "MATCHED", "recipe_id": line.recipe.recipe_id, "recipe_version": 1, "recipe_sha256": line.recipe.recipe_sha256}
    row, _ = line.store.register_scan(
        sha256=s2, original_name="SYN-HAT-1_A_SCAN_2 1.ply", size_bytes=len(scan2), station_id="cell-1",
        stored_rel_path=rel, parse={"status": "OK"}, match=match, format="PLY_BINARY_LE",
    )  # fmt: skip
    job_id, _ = line.store.match_and_enqueue(row["scan_file_id"], match=match, sku=line.recipe.sku, compiled_part_sha256=compiled_sha)
    return job_id


def test_blocked_compiled_part_fails_preflight_without_spawning(line):
    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.compiled_part import CompiledPartV1

    data = line.compiled.model_dump(mode="json", by_alias=True)
    data.pop("compiled_part_sha256")
    data["status"] = "BLOCKED"
    data["issues"] = list(data["issues"]) + [{"code": "STAGE_MODEL_FAILED", "severity": "BLOCKER", "detail": "synthetic"}]
    blocked = CompiledPartV1.finalize(**data)
    (line.compiled_dir / "compiled_part.json").write_bytes(canonical_json_bytes(blocked))
    job_id = _enqueue_second_scan(line, blocked.compiled_part_sha256)
    line.runner().run_once()
    failure = line.store.get_job(job_id)["failure"]
    assert failure["code"] == "COMPILE_BLOCKED" and "STAGE_MODEL_FAILED" in failure["message"]
    assert not (line.fake_dir / "engine_marker.json").exists()


def test_render_over_rss_ceiling_is_postprocess_error(line):
    # 8 MiB is below any Python interpreter's RSS: the render step breaches it at once.
    line.runner(render_rss_limit_bytes=8 * 1024**2, extra_env={"FAKE_RENDER_MODE": "sleep"}).run_once()
    failure = failure_of(line)
    assert failure["code"] == "POSTPROCESS_ERROR" and "RSS" in failure["message"]


def test_run_forever_survives_store_errors(line, monkeypatch):
    calls = {"beat": 0, "recover": 0}
    real_beat = line.store.put_heartbeat

    def flaky_beat(*a, **kw):
        calls["beat"] += 1
        if calls["beat"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_beat(*a, **kw)

    runner = line.runner(lease_ttl_s=0.2, idle_poll_s=0.05)
    real_recover = runner.recover

    def flaky_recover():
        calls["recover"] += 1
        if calls["recover"] == 1:
            raise sqlite3.OperationalError("disk I/O error")
        return real_recover()

    monkeypatch.setattr(line.store, "put_heartbeat", flaky_beat)
    monkeypatch.setattr(runner, "recover", flaky_recover)
    stop = threading.Event()
    t = threading.Thread(target=runner.run_forever, args=(stop,), daemon=True)
    t.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (
        line.store.get_job(line.job_id)["execution_status"] == "completed" and calls["recover"] >= 2 and calls["beat"] >= 2
    ):
        time.sleep(0.05)
    assert line.store.get_job(line.job_id)["execution_status"] == "completed"
    assert calls["beat"] >= 2 and calls["recover"] >= 2  # each failed once, and the loop called them again
    assert t.is_alive()  # neither the failed heartbeat nor the failed recovery ended the loop
    stop.set()
    t.join(10)
    assert not t.is_alive()


def test_compile_events_share_the_final_transaction(line):
    job_id = _enqueue_compile(line)
    assert line.runner(sink=Sink()).run_once() == job_id
    events = line.store.events_after(0, limit=500)  # every station, in id order
    i = [e["kind"] for e in events].index("RECIPE_COMPILED")
    # written in complete_job's transaction: immediately followed by JOB_COMPLETED
    assert events[i]["job_id"] == job_id and events[i + 1]["kind"] == "JOB_COMPLETED"
    assert events[i + 1]["job_id"] == job_id and events[i + 1]["id"] == events[i]["id"] + 1
    assert [e["kind"] for e in events].count("RECIPE_COMPILED") == 1


def test_lost_lease_never_announces_compile_outcome(line):
    job_id = _enqueue_compile(line)
    line.store.lease_next("runner-a", 30)
    before = line.store.last_event_id()
    assert not line.store.complete_job(job_id, "runner-b", None, extra_event=("RECIPE_COMPILED", {"recipe_id": "x"}))
    assert not line.store.fail_job(job_id, "runner-b", "WORKER_CRASH", "x", extra_event=("RECIPE_COMPILE_FAILED", {}))
    assert line.store.last_event_id() == before  # nothing announced for a lease we do not hold
    assert line.store.fail_job(job_id, "runner-a", "WORKER_CRASH", "x", extra_event=("RECIPE_COMPILE_FAILED", {"code": "WORKER_CRASH"}))
    tail = [e["kind"] for e in line.store.events_after(before, limit=10)]
    assert tail == ["RECIPE_COMPILE_FAILED", "JOB_FAILED"]


def test_attempt_logs_are_kept_across_retries(line):
    line.runner(extra_env={"FAKE_ENGINE_MODE": "crash"}).run_once()
    assert failure_of(line)["code"] == "WORKER_CRASH"
    retry = line.store.request_retry(line.job_id, ADMIN)
    line.runner().run_once()
    assert line.store.get_job(retry)["execution_status"] == "completed"
    first = line.ws() / "logs" / "attempt-1" / "engine_step.log"
    assert first.is_file() and "fake segfault" in first.read_text()  # the crashed attempt's log survives
    assert (line.root / "jobs" / retry / "logs" / "attempt-1" / "engine_step.log").is_file()


DRIVER = r'''
import base64, json, sys, threading
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
from apps.production_line.ingest.ports import ResolvedRecipe
from apps.production_line.runner import supervisor
from apps.production_line.runner.runner import JobRunner, RunnerConfig
from apps.production_line.store.line_store import LineStore
if spec["no_sentinel"]:
    supervisor._start_sentinel = lambda pgid, tag: None
r = spec["recipe"]
resolved = ResolvedRecipe(
    recipe_id=r["recipe_id"], recipe_version=r["recipe_version"], recipe_sha256=r["recipe_sha256"], sku=r["sku"],
    recipe_json_bytes=base64.b64decode(r["recipe_b64"]), cad_path=Path(r["cad_path"]), cad_sha256=r["cad_sha256"],
    part_name=r["part_name"], compiled_dir=Path(r["compiled_dir"]), compiled_part_sha256=r["compiled_part_sha256"],
)
class P:
    def load(self, recipe_id):
        if recipe_id != resolved.recipe_id:
            raise KeyError(recipe_id)
        return resolved
c = spec["config"]
cfg = RunnerConfig(engine_root=Path(c["engine_root"]), python_exe=Path(sys.executable), data_root=Path(c["data_root"]),
                   inspect_module="fake_inspect", render_module="fake_render", supervisor_poll_s=0.1, lease_ttl_s=30,
                   heartbeat_s=0.5, kill_grace_s=1, extra_env=c["extra_env"])
runner = JobRunner(cfg, LineStore(c["db"]), P(), instance_id="runner@test")
runner.run_once()
'''


def _spawn_runner_process(line, tmp_path, *, no_sentinel: bool) -> subprocess.Popen:
    import base64

    resolved = line.provider.recipes[line.recipe.recipe_id]
    spec = {
        "no_sentinel": no_sentinel,
        "recipe": {
            "recipe_id": resolved.recipe_id, "recipe_version": resolved.recipe_version,
            "recipe_sha256": resolved.recipe_sha256, "sku": resolved.sku,
            "recipe_b64": base64.b64encode(resolved.recipe_json_bytes).decode(), "cad_path": str(resolved.cad_path),
            "cad_sha256": resolved.cad_sha256, "part_name": resolved.part_name,
            "compiled_dir": str(resolved.compiled_dir), "compiled_part_sha256": resolved.compiled_part_sha256,
        },
        "config": {
            "engine_root": str(line.engine_root), "data_root": str(line.root), "db": str(line.root / "db" / "line.sqlite3"),
            "extra_env": {"FAKE_DIR": str(line.fake_dir), "FAKE_ENGINE_MODE": "sleep"},
        },
    }  # fmt: skip
    spec_path = tmp_path / "driver_spec.json"
    spec_path.write_text(json.dumps(spec))
    (tmp_path / "driver.py").write_text(DRIVER)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(REPO / "backend")]), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen([sys.executable, str(tmp_path / "driver.py"), str(spec_path)], cwd=REPO, env=env)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (line.fake_dir / "engine_marker.json").exists():
        assert proc.poll() is None, "runner process exited early"
        time.sleep(0.05)
    assert (line.fake_dir / "engine_marker.json").exists()
    return proc


def _group_alive(pgid: int) -> bool:
    """Live (non-zombie) members exist; macOS answers killpg(pgid, 0) with EPERM
    for a group of zombies awaiting their reaper, so ps decides."""
    from apps.production_line.runner.supervisor import group_members

    members = group_members(pgid)
    assert members is not None, "ps unavailable"
    return bool(members)


def _wait_dead(pgid: int, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if not _group_alive(pgid):
            return True
        time.sleep(0.05)
    return not _group_alive(pgid)


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_sigkilled_runner_takes_its_engine_with_it(line, tmp_path):
    """Parent-death sentinel: the engine never outlives a SIGKILLed runner."""
    proc = _spawn_runner_process(line, tmp_path, no_sentinel=False)
    record = json.loads((line.ws() / "logs" / "active_step.json").read_text())
    pgid = json.loads((line.fake_dir / "engine_marker.json").read_text())["pgid"]
    assert record["pgid"] == pgid and record["tag"] == line.job_id and record["attempt"] == 1
    proc.kill()
    proc.wait(10)
    assert _wait_dead(pgid, 8), "the engine group outlived its runner"


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_restart_kills_recorded_orphan_before_the_new_engine_starts(line, tmp_path):
    """Without the sentinel (e.g. it failed to start) the durable record still
    lets the restarted runner kill the orphan before it clears engine/."""
    proc = _spawn_runner_process(line, tmp_path, no_sentinel=True)
    pgid = json.loads((line.fake_dir / "engine_marker.json").read_text())["pgid"]
    proc.kill()
    proc.wait(10)
    time.sleep(0.5)
    assert _group_alive(pgid)  # orphaned: nothing supervises it any more
    try:
        (line.fake_dir / "engine_marker.json").unlink()
        runner = JobRunner(
            line.config(extra_env={"FAKE_CHECK_PGID": str(pgid)}), line.store, line.provider, instance_id="runner@test"
        )
        assert not _group_alive(pgid)  # reclaiming the dead runner's lease killed the orphan at once
        assert line.store.get_job(line.job_id)["execution_status"] == "queued"
        assert runner.run_once() == line.job_id
        job = line.store.get_job(line.job_id)
        assert job["execution_status"] == "completed" and job["attempts"] == 2, job["failure"]
        marker = json.loads((line.fake_dir / "engine_marker.json").read_text())
        assert marker["old_alive"] is False and marker["out_existed"] is False
        assert not (line.ws() / "logs" / "active_step.json").exists()
        assert (line.ws() / "logs" / "attempt-1").is_dir() and (line.ws() / "logs" / "attempt-2").is_dir()
    finally:
        if _group_alive(pgid):
            os.killpg(pgid, 9)


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_recorded_group_with_recycled_pgid_is_left_alone(line):
    """A record whose pgid now belongs to an unrelated group is never signalled."""
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], process_group=0)
    try:
        record = line.ws() / "logs" / "active_step.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"pgid": stranger.pid, "tag": line.job_id, "host": __import__("socket").gethostname()}))
        line.runner().run_once()
        assert stranger.poll() is None  # untouched
        assert line.store.get_job(line.job_id)["execution_status"] == "completed"
        assert not record.exists()
    finally:
        stranger.kill()
        stranger.wait(5)
