"""Runner step 3b: the zs step alongside E3 and the overlay merge, with containment (synthetic only).

Reuses the C02 runner fixture (``tests/test_prod_runner.py``: fake E3 / C06 modules, SYN-HAT-1 recipe, compiled
part and result). The fake engine bundle additionally links the repo's ``domains`` and ``contracts`` packages so
the REAL merge step (``domains.production.zeroshot.overlay_cli``) runs; the zs step itself is a fake module
(``fake_zs``) that writes synthetic raw-frame lines, crashes, hangs or prints garbage. No scan, CAD or model.
"""
from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from tests.test_prod_runner import REPO, Line, _wait_dead, line, sha  # noqa: F401  (the ``line`` fixture)

FAKE_ZS = r'''
import argparse, hashlib, json, os, pathlib, sys, time
ap = argparse.ArgumentParser()
for a in ["--scan", "--compiled", "--out"]:
    ap.add_argument(a)
args = ap.parse_args()
mode = os.environ.get("FAKE_ZS_MODE", "ok")
d = pathlib.Path(os.environ["FAKE_DIR"])
(d / "zs_marker.json").write_text(json.dumps({"argv": sys.argv[1:], "pid": os.getpid(), "pgid": os.getpgid(0)}))
if mode == "crash":
    raise SystemExit("Traceback: fake zs segfault")
if mode == "sleep":
    time.sleep(120)
out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
if mode == "garbage":
    out.write_text("{not json")
    sys.exit(0)
lines = json.loads((d / "zs_lines_in.json").read_text())
doc = {"schema": "sherman_zeroshot_lines_v1", "status": "OK", "lane": "zs_lane_v1", "variant": "zs",
       "runtime_version": "zs-prod-1.0.0", "code_fingerprint": "a" * 64, "model_sha256": "b" * 64,
       "scan_sha256": hashlib.sha256(pathlib.Path(args.scan).read_bytes()).hexdigest(),
       "priors": {"n_bends": 4, "t_mm": 2.0, "K": 1, "max_families": 3},
       "lines": [{"line_id": f"ZS{i + 1:02d}", "start_mm": a, "end_mm": b, "incl_deg": 90.0, "score": 0.9}
                 for i, (a, b) in enumerate(lines)]}
if mode == "otherscan":
    doc["scan_sha256"] = "0" * 64
out.write_text(json.dumps(doc))
print("zs: OK", flush=True)
'''


def zs_lines(line: Line, *, extra: bool) -> list:
    from contracts.production.inspection_result import ShopfloorInspectionResultV1
    from domains.production.engine.eval.engine_output import decoded_stage, frame_to_raw, stage_locus

    res = ShopfloorInspectionResultV1.model_validate_json((line.fake_dir / "template_result.json").read_bytes())
    st = decoded_stage(res, line.compiled)
    out = [frame_to_raw(res, stage_locus(line.compiled, st, b.bend_uid, "outer_apex")).tolist() for b in line.compiled.bends]
    if extra:  # a bend the recipe does not have
        out.append(frame_to_raw(res, np.asarray([[20.0, 0.0, 40.0], [20.0, 100.0, 40.0]])).tolist())
    return out


def zs_runner(line: Line, mode: str = "ok", *, extra: bool = True, **kw):
    for pkg in ("domains", "contracts"):
        link = line.engine_root / pkg
        if not link.exists():
            os.symlink(REPO / pkg, link)
    (line.engine_root / "fake_zs.py").write_text(textwrap.dedent(FAKE_ZS))
    (line.fake_dir / "zs_lines_in.json").write_text(json.dumps(zs_lines(line, extra=extra)))
    env = {"FAKE_ZS_MODE": mode, **kw.pop("extra_env", {})}
    return line.runner(zeroshot_module="fake_zs", extra_env=env, **kw)


def expected_final(line: Line) -> dict:
    """What the fake C06 writes for the job when nothing else interferes."""
    template = json.loads((line.fake_dir / "template_result.json").read_text())
    template["scan"]["sha256"] = sha(line.scan_bytes)
    template["warnings"] = list(template.get("warnings", [])) + ["PDF_FONT_FALLBACK"]
    template["context"] = json.loads((line.ws() / "input" / "context.json").read_text())
    return template


def summary_of(line: Line) -> dict:
    job = line.store.get_job(line.job_id)
    assert job["execution_status"] == "completed", job["failure"]
    return job["result_summary"]


def test_overlay_written_next_to_an_unchanged_result(line):
    runner = zs_runner(line)
    assert runner.run_once() == line.job_id
    ws = line.ws()
    summary = summary_of(line)
    template = json.loads((line.fake_dir / "template_result.json").read_text())
    assert summary["summary_state"] == template["summary"]["state"]  # E3's state, untouched
    assert json.loads((ws / "result" / "shopfloor_result.json").read_text()) == expected_final(line)
    ov = json.loads((ws / "zeroshot" / "zeroshot_overlay_v1.json").read_text())
    assert ov["schema"] == "sherman_zeroshot_overlay_v1" and ov["scan_sha256"] == sha(line.scan_bytes)
    assert ov["e3_summary_state"] == ov["effective_summary_state"] == "OUT_OF_TOLERANCE_FOUND"  # FAIL never moves
    assert [ln["kind"] for ln in ov["lines"]] == ["UNEXPECTED_BEND"] and ov["review_reasons"] == ["UNEXPECTED_FEATURE_REVIEW"]
    status = json.loads((ws / "zeroshot" / "overlay_status.json").read_text())
    assert status["overlay_written"] is True and status["zs"]["exit_code"] == 0
    # blindness: the zs step only ever sees the staged scan and the compiled part
    argv = json.loads((line.fake_dir / "zs_marker.json").read_text())["argv"]
    assert argv[argv.index("--scan") + 1].endswith("input/scan.ply")
    assert argv[argv.index("--compiled") + 1].endswith("compiled/compiled_part.json")
    assert not any("FULL_SCAN" in a or "SN0042" in a or "context" in a or line.job_id in Path(a).name for a in argv)
    assert not (ws / "logs" / "active_step_zs.json").exists() and not (ws / "logs" / "active_step.json").exists()
    assert (ws / "logs" / "attempt-1" / "zeroshot_step.log").is_file()


@pytest.mark.parametrize("mode", ["crash", "garbage", "otherscan"])
def test_zs_failure_leaves_the_job_unchanged(line, mode):
    runner = zs_runner(line, mode)
    assert runner.run_once() == line.job_id
    ws = line.ws()
    summary = summary_of(line)
    template = json.loads((line.fake_dir / "template_result.json").read_text())
    assert summary["summary_state"] == template["summary"]["state"]
    assert summary["inspection_validity"] == template["inspection_validity"]["state"]
    assert json.loads((ws / "result" / "shopfloor_result.json").read_text()) == expected_final(line)
    assert not (ws / "zeroshot" / "zeroshot_overlay_v1.json").exists()
    status = json.loads((ws / "zeroshot" / "overlay_status.json").read_text())
    assert status["overlay_written"] is False


def test_zs_hang_is_cut_at_its_own_limit_and_the_job_completes(line):
    runner = zs_runner(line, "sleep", zeroshot_timeout_s=3.0)
    t0 = time.monotonic()
    assert runner.run_once() == line.job_id
    assert time.monotonic() - t0 < 30
    summary_of(line)
    status = json.loads((line.ws() / "zeroshot" / "overlay_status.json").read_text())
    assert status["zs"]["timed_out"] is True and status["overlay_written"] is False
    pgid = json.loads((line.fake_dir / "zs_marker.json").read_text())["pgid"]
    assert _wait_dead(pgid, 5.0)


def test_engine_failure_stops_the_zs_step(line):
    runner = zs_runner(line, "sleep", extra_env={"FAKE_ENGINE_MODE": "crash"})
    t0 = time.monotonic()
    assert runner.run_once() == line.job_id
    assert time.monotonic() - t0 < 30
    job = line.store.get_job(line.job_id)
    assert job["execution_status"] == "failed" and job["failure"]["code"] == "WORKER_CRASH"
    deadline = time.monotonic() + 10
    pgid = None
    while time.monotonic() < deadline and pgid is None:
        marker = line.fake_dir / "zs_marker.json"
        pgid = json.loads(marker.read_text())["pgid"] if marker.exists() else None
        time.sleep(0.1)
    if pgid is not None:
        assert _wait_dead(pgid, 5.0)
    assert not (line.ws() / "logs" / "active_step_zs.json").exists()


def test_zero_shot_disabled_by_default(line):
    runner = line.runner()
    assert runner.config.zeroshot_module is None
    assert runner.run_once() == line.job_id
    summary_of(line)
    assert not (line.ws() / "zeroshot").exists()


def test_zero_shot_limits_cannot_exceed_the_engine(line):
    with pytest.raises(ValueError):
        line.runner(zeroshot_module="fake_zs", zeroshot_rss_limit_bytes=33 * 1024**3)
    with pytest.raises(ValueError):
        line.runner(zeroshot_module="fake_zs", zeroshot_timeout_s=600)


# ------------------------------------------------------------------ station API (C07 service)


def test_station_api_serves_only_a_consistent_overlay(line):
    from types import SimpleNamespace

    from apps.api.services.production_line_service import load_zeroshot_overlay

    runner = zs_runner(line)
    assert runner.run_once() == line.job_id
    job = line.store.get_job(line.job_id)
    settings = SimpleNamespace(paths=SimpleNamespace(jobs=line.root / "jobs"))
    loaded = load_zeroshot_overlay(settings, job)
    on_disk = json.loads((line.ws() / "zeroshot" / "zeroshot_overlay_v1.json").read_text())
    assert loaded.code is None and loaded.overlay == on_disk and loaded.etag
    # another summary state than the job's: refused, never shown
    other = {**job, "result_summary": {**job["result_summary"], "summary_state": "ALL_MEASURED_IN_TOLERANCE"}}
    assert load_zeroshot_overlay(settings, other).code == "OVERLAY_INCONSISTENT"
    other = {**job, "scan_sha256": "0" * 64}
    assert load_zeroshot_overlay(settings, other).code == "OVERLAY_INCONSISTENT"
    path = line.ws() / "zeroshot" / "zeroshot_overlay_v1.json"
    path.write_text(json.dumps({**on_disk, "effective_summary_state": "ALL_MEASURED_IN_TOLERANCE"}))
    assert load_zeroshot_overlay(settings, job).code == "OVERLAY_INVALID"
    path.unlink()
    assert load_zeroshot_overlay(settings, job).code == "OVERLAY_MISSING"


def test_router_has_the_overlay_route():
    from apps.api.routes.production_line import router

    assert "/api/line/jobs/{job_id}/zeroshot-overlay" in {r.path for r in router.routes}  # type: ignore[attr-defined]
