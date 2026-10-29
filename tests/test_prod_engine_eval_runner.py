"""E4a tiered runner and process monitor: E1 compile + E3-CLI subprocess driving,
time/RSS capture, hard limits, failure classification, resume and blindness.

The engine command is the E4 truth oracle's E3-compatible CLI (E3 is built
concurrently); E1 is the real compile CLI.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from domains.production.engine.eval import freeze as fz
from domains.production.engine.eval.plyio import sha256_file
from domains.production.engine.eval.procmon import group_rss_bytes, run_monitored
from domains.production.engine.eval.runner import (
    RunnerConfig,
    RunnerPolicyError,
    blindness_check,
    code_fingerprint,
    engine_env,
    rescore_run,
    run_tier,
)
from domains.production.engine.synthetic.cases import generate_tier

PY = sys.executable
ORACLE = (PY, "-m", "domains.production.engine.eval.oracle")
CASES = ("E2-DEV-LB-000", "E2-DEV-UC-003")


@pytest.fixture(scope="module")
def tier(tmp_path_factory):
    root = tmp_path_factory.mktemp("e4a_runner")
    generate_tier("regression", "DEV", root / "tier", case_ids=CASES)
    return root


def _cfg(tier: Path, out: str, **kw) -> RunnerConfig:
    mode = kw.pop("mode", "")
    # The engine here is the oracle, so its code identity is the eval package. E1/E3 code is edited
    # concurrently by other chunks, which (correctly) invalidates resume; pin the roots for determinism.
    base = RunnerConfig(tier_dir=tier / "tier", out_dir=tier / out, engine_cmd=ORACLE, bootstrap=False,
                        extra_env={"SHERMAN_E4_ORACLE_CASES": str(tier / "tier"), "SHERMAN_E4_ORACLE_MODE": mode},
                        engine_code_roots=("domains/production/engine/eval",),
                        compile_code_roots=("domains/production/engine/eval",))
    return replace(base, **kw)


# ---------------------------------------------------------------------------
# process monitor
# ---------------------------------------------------------------------------


def test_monitor_parses_protocol_records_and_measures(tmp_path):
    code = "import sys,json; print('PROGRESS\\t'+json.dumps({'progress':50})); print('RESULT\\t'+json.dumps({'ok':1})); print('noise')"
    out = run_monitored([PY, "-c", code], timeout_s=30, rss_limit_bytes=2 * 1024**3, stderr_path=tmp_path / "e.log")
    assert out.returncode == 0 and out.signal is None and not out.timed_out and not out.rss_exceeded
    assert out.last("RESULT") == {"ok": 1} and out.last("PROGRESS") == {"progress": 50}
    assert out.wall_s > 0 and out.peak_rss_bytes and out.peak_rss_bytes > 1_000_000


def test_monitor_enforces_the_hard_timeout_on_the_whole_group(tmp_path):
    # the child spawns a grandchild; both must be gone after the hard stop
    code = ("import subprocess,sys,time,json; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
            "print('RESULT\\t'+json.dumps({'grandchild':p.pid}),flush=True); time.sleep(60)")
    out = run_monitored([PY, "-c", code], timeout_s=1.5, rss_limit_bytes=2 * 1024**3, kill_grace_s=0.5)
    assert out.timed_out and not out.rss_exceeded and out.wall_s < 10
    assert out.signal is not None and out.returncode is None
    gpid = out.last("RESULT")["grandchild"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(gpid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("grandchild survived the group kill")
    assert group_rss_bytes(-12345) == 0  # an empty group sums to zero


def test_monitor_kills_on_rss_breach(tmp_path):
    code = "b=bytearray(400*1024*1024)\nfor i in range(0,len(b),4096): b[i]=1\nimport time; time.sleep(10)"
    out = run_monitored([PY, "-c", code], timeout_s=30, rss_limit_bytes=150 * 1024**2, sample_interval_s=0.1)
    assert out.rss_exceeded and not out.timed_out and out.signal == 9
    assert out.peak_rss_bytes > 150 * 1024**2


def test_engine_env_caps_threads_and_strips_harness_switches(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERMAN_E4_ORACLE_MODE", "crash")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    env = engine_env(RunnerConfig(tier_dir=tmp_path, out_dir=tmp_path))
    assert env["OMP_NUM_THREADS"] == "1" and env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert "SHERMAN_E4_ORACLE_MODE" not in env and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPATH"].split(":")[1].endswith("backend")


def test_code_fingerprint_changes_with_code(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    a = code_fingerprint(tmp_path, ("pkg", "missing"))
    (tmp_path / "pkg" / "a.py").write_text("x = 2\n")
    assert code_fingerprint(tmp_path, ("pkg", "missing")) != a


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def test_run_tier_compiles_inspects_scores_and_resumes(tier):
    seen = []
    card = run_tier(_cfg(tier, "run_ok"), progress=lambda i, n, d: seen.append(d))
    out = tier / "run_ok"
    assert [d["status"] for d in seen] == ["OK", "OK"]
    assert all(d["wall_s"] > 0 and d["peak_rss_bytes"] > 0 for d in seen)
    for name in ("run_manifest.json", "scorecard.json", "scorecard.md", "rows/chars.jsonl", "rows/bends.jsonl"):
        assert (out / name).is_file(), name
    assert card["counts"]["run_status"] == {"OK": 2}
    assert next(g for g in card["gates"] if g["id"] == "G1.ENGINE_FAILURES")["verdict"] == "PASS"
    assert "ORACLE_RESULTS" in card["not_qualifying_reasons"]
    rec = json.loads((out / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["compile"]["status"] == "COMPILED" and rec["engine_output"]["status"] == "OK"
    # the harness's run conditions are recorded and (oracle command, extra env) never qualify
    assert card["provenance"]["run_conditions"]["engine_argv_tail"] == ["-m", "domains.production.engine.eval.oracle"]
    assert {"NON_STANDARD_ENGINE_CMD", "NON_STANDARD_ENGINE_ENV"} <= set(card["not_qualifying_reasons"])
    # inputs are read-only copies, never hard links into the E2 tier
    staged = Path(rec["engine_output"]["dir"]).parent / "input" / "scan.ply"
    assert os.stat(staged).st_nlink == 1 and not os.stat(staged).st_mode & stat.S_IWUSR
    assert os.stat(tier / "tier" / CASES[0] / "scan.ply").st_nlink == 1
    argv = rec["process"]["argv"]
    # blindness: no case id, family code or ZEISS-like name reaches the engine command line
    joined = " ".join(argv)
    assert CASES[0] not in joined and "E2-DEV" not in joined and "_SCAN" not in joined
    assert argv[argv.index("--scan") + 1].endswith("/input/scan.ply")
    for flag in ("--scan-sha256", "--compiled-sha256", "--recipe-sha256"):
        assert flag in argv
    # a rerun with unchanged identity skips every case and gives the same scorecard
    again = []
    card2 = run_tier(_cfg(tier, "run_ok"), progress=lambda i, n, d: again.append(d))
    assert all(d.get("skipped") for d in again)
    assert card2["gates"] == card["gates"]
    # another engine configuration is another identity: rerun, compile cache reused
    cfg = tier / "engine_cfg.json"
    cfg.write_text("{}\n")
    third = []
    run_tier(_cfg(tier, "run_ok", engine_config=cfg), progress=lambda i, n, d: third.append(d))
    assert not any(d.get("skipped") for d in third)
    rec = json.loads((out / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["compile"]["reused"] is True
    # a change in the engine's code identity also invalidates resume
    fourth = []
    run_tier(_cfg(tier, "run_ok", engine_config=cfg, engine_code_roots=("domains/production/engine/eval/scorer.py",)),
             progress=lambda i, n, d: fourth.append(d))
    assert not any(d.get("skipped") for d in fourth)
    # a change in the eval code re-scores the stored engine outputs without rerunning the engine
    walls = {c: json.loads((out / "cases" / f"{c}.json").read_text())["run"]["wall_s"] for c in CASES}
    fifth = []
    run_tier(_cfg(tier, "run_ok", engine_config=cfg, engine_code_roots=("domains/production/engine/eval/scorer.py",),
                  eval_code_roots=("domains/production/engine/eval/stats.py",)), progress=lambda i, n, d: fifth.append(d))
    assert all(d.get("skipped") and d.get("rescored") for d in fifth)
    assert {c: json.loads((out / "cases" / f"{c}.json").read_text())["run"]["wall_s"] for c in CASES} == walls
    # `score` rebuilds the scorecard with the run's provenance (configs, limits, run conditions)
    rescored = rescore_run(out, bootstrap=False)
    prov = rescored["provenance"]
    assert prov["engine_config_sha256"] == sha256_file(cfg) and prov["limits"]["engine_timeout_s"] == 180.0
    assert prov["run_conditions"]["engine_extra_env_keys"] == ["SHERMAN_E4_ORACLE_CASES", "SHERMAN_E4_ORACLE_MODE"]
    assert rescored["gates"] == run_tier(_cfg(tier, "run_ok", engine_config=cfg, engine_code_roots=(
        "domains/production/engine/eval/scorer.py",), eval_code_roots=("domains/production/engine/eval/stats.py",)))["gates"]
    cli = subprocess.run([PY, "-m", "domains.production.engine.eval", "score", "--run-dir", str(out)],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert cli.returncode == 0, cli.stderr


@pytest.mark.parametrize("mode,status", [
    ("crash", "ENGINE_CRASH"),
    ("exit2", "ENGINE_EXIT_2"),
    ("exit3", "ENGINE_EXIT_3"),
    ("garbage", "RESULT_INVALID"),
    ("no_result", "RESULT_MISSING"),
])
def test_engine_failures_are_classified_and_scored_as_failures(tier, mode, status):
    out = f"run_{mode}"
    card = run_tier(_cfg(tier, out, mode=mode, case_ids=(CASES[0],)))
    rec = json.loads((tier / out / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["run"]["status"] == status
    if status == "ENGINE_EXIT_2":
        assert rec["run"]["detail"] == "COMPILE_BLOCKED"
    gate = next(g for g in card["gates"] if g["id"] == "G1.ENGINE_FAILURES")
    assert gate["verdict"] == "FAIL" and gate["details"]["by_status"] == {status: 1}


def test_hard_timeout_and_rss_limit_are_runner_failures(tier):
    card = run_tier(_cfg(tier, "run_slow", mode="sleep=30", case_ids=(CASES[0],), engine_timeout_s=2.0))
    rec = json.loads((tier / "run_slow" / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["run"]["status"] == "ENGINE_TIMEOUT" and rec["process"]["timed_out"]
    assert 2.0 <= rec["run"]["wall_s"] < 15
    card = run_tier(_cfg(tier, "run_fat", mode="alloc_mb=500,hold_s=5", case_ids=(CASES[0],),
                         engine_rss_limit_bytes=200 * 1024**2))
    rec = json.loads((tier / "run_fat" / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["run"]["status"] == "ENGINE_RSS_LIMIT" and rec["run"]["peak_rss_bytes"] > 200 * 1024**2
    assert card["counts"]["run_status"] == {"ENGINE_RSS_LIMIT": 1}


def test_compile_failures_stay_in_the_denominators(tier):
    card = run_tier(_cfg(tier, "run_nocompile", compile_module="domains.production.engine.eval.no_such_module",
                         case_ids=(CASES[0],)))
    rec = json.loads((tier / "run_nocompile" / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["compile"]["status"] == "COMPILE_FAILED" and rec["run"]["status"] == "COMPILE_FAILED"
    gate = next(g for g in card["gates"] if g["id"] == "G1.ENGINE_FAILURES")
    # nothing reached the engine: the engine gate cannot be evaluated (never PASS) ...
    assert gate["verdict"] == "NOT_EVALUABLE" and gate["n"] == 0 and gate["details"]["compile_failures"]
    # ... and the uncompiled case still costs coverage
    cov = next(g for g in card["gates"] if g["id"] == "G1.COVERAGE")
    assert cov["n"] > 0 and cov["value"] == 0.0 and cov["verdict"] == "FAIL"
    assert card["verdict"] == "FAIL"


def test_blindness_check_passes_for_a_blind_engine_and_fails_for_a_name_reader(tier):
    ok = blindness_check(_cfg(tier, "blind_ok"), [CASES[0]])
    assert ok["verdict"] == "PASS" and ok["cases"][0]["identical"] is True
    bad = blindness_check(_cfg(tier, "blind_bad", mode="name_leak"), [CASES[0]])
    assert bad["verdict"] == "FAIL" and "warnings" in bad["cases"][0]["differing_keys"]



def test_an_engine_that_writes_to_its_input_is_caught_and_the_tier_is_untouched(tier):
    before = sha256_file(tier / "tier" / CASES[0] / "scan.ply")
    card = run_tier(_cfg(tier, "run_modify", mode="modify_input", case_ids=(CASES[0],)))
    rec = json.loads((tier / "run_modify" / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["run"]["status"] == "RESULT_INCONSISTENT" and rec["run"]["detail"] == "ENGINE_MODIFIED_INPUT:scan"
    assert next(g for g in card["gates"] if g["id"] == "G1.ENGINE_FAILURES")["verdict"] == "FAIL"
    assert sha256_file(tier / "tier" / CASES[0] / "scan.ply") == before


def test_transient_compile_failures_are_never_reused_from_the_cache(tier):
    run_tier(_cfg(tier, "run_transient", case_ids=(CASES[0],), compile_timeout_s=0.05))
    rec = json.loads((tier / "run_transient" / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["compile"]["status"] == "COMPILE_FAILED" and rec["compile"]["detail"] == "timeout"
    run_tier(_cfg(tier, "run_transient", case_ids=(CASES[0],)))
    rec = json.loads((tier / "run_transient" / "cases" / f"{CASES[0]}.json").read_text())
    assert rec["compile"]["status"] == "COMPILED" and rec["compile"]["reused"] is False


def test_qual_seeds_need_the_final_run_opt_in_and_every_run_is_logged(tier, tmp_path):
    qual = tmp_path / "qual_tier"
    shutil.copytree(tier / "tier", qual)
    manifest = json.loads((qual / "manifest.json").read_text())
    manifest["seedset"] = "QUAL"  # a stand-in: the policy reads the tier manifest (no QUAL case is generated)
    (qual / "manifest.json").write_text(json.dumps(manifest))
    base = replace(_cfg(tier, "run_qual", case_ids=(CASES[0],)), tier_dir=qual,
                   extra_env={"SHERMAN_E4_ORACLE_CASES": str(qual), "SHERMAN_E4_ORACLE_MODE": ""})
    with pytest.raises(RunnerPolicyError):
        run_tier(base)
    with pytest.raises(RunnerPolicyError):
        run_tier(replace(base, final_qual_run=True))  # no ledger to log it in
    hidden = tmp_path / "hidden_qual"  # a DEV manifest cannot hide a QUAL case
    shutil.copytree(tier / "tier", hidden)
    case = json.loads((hidden / CASES[0] / "case.json").read_text())
    case["spec"]["seedset"] = "QUAL"
    (hidden / CASES[0] / "case.json").write_text(json.dumps(case))
    with pytest.raises(RunnerPolicyError):
        run_tier(replace(base, tier_dir=hidden))
    ledger = tmp_path / "ledger.jsonl"
    card = run_tier(replace(base, final_qual_run=True, ledger=ledger))
    runs = [e for e in fz.read_ledger(ledger) if e["kind"] == "RUN"]
    assert len(runs) == 1 and runs[0]["items"]["seedset"] == "QUAL" and runs[0]["items"]["cases"] == 1
    lc = card["provenance"]["ledger_check"]
    assert len(lc["qual_runs"]) == 1 and not lc["qual_seeds_reused"]
    cfg = tmp_path / "other_engine_config.json"
    cfg.write_text("{}\n")
    card = run_tier(replace(base, final_qual_run=True, ledger=ledger, engine_config=cfg))  # a second engine identity
    assert card["provenance"]["ledger_check"]["qual_seeds_reused"]
    assert "QUAL_SEEDS_REUSED" in card["not_qualifying_reasons"]
