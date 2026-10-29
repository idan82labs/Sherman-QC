"""Run-scoped sha256 allowlist (research/zeroshot/common/run_allowlist.py), final eval FE1 2026-10-28.

Synthetic only: a copy of the freeze ledger (so seq 35 is the real authorization NOTE) plus a temp allowlist of 28
synthetic files under a guard-refused directory name. No scan, manifest or label is opened.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from domains.production.engine.eval import freeze
from research.bench_labeling.custody import CustodyError
from research.zeroshot.common import run_allowlist as RA

REAL_LEDGER = RA.FREEZE_LEDGER


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def env(tmp_path):
    ledger = tmp_path / "LEDGER.json"
    shutil.copyfile(REAL_LEDGER, ledger)
    lines = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    ledger.write_text("".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in lines[:RA.AUTH_SEQ]))
    auth = lines[RA.AUTH_SEQ - 1]
    guarded = tmp_path / "fe_heldout_copies"
    guarded.mkdir()
    files = []
    for i in range(RA.N_ITEMS):
        f = guarded / f"gb_{i:06x}.ply"
        f.write_bytes(f"synthetic scan {i}\n".encode())
        files.append(f)
    allow = tmp_path / "RUN_ALLOWLIST.json"
    allow.write_text(json.dumps({
        "schema": RA.SCHEMA, "authorization": {"seq": RA.AUTH_SEQ, "entry_id": RA.AUTH_ENTRY_ID},
        "set_manifest": {"sha256": auth["items"]["set_manifest"]["sha256"]},
        "sha256": sorted(_sha(f) for f in files)}, indent=1))
    freeze.append_entry(ledger, kind="NOTE", recorded_by="test", label="test allowlist binding",
                        items={"run_allowlist": {"authorization_seq": RA.AUTH_SEQ, "allowlist_sha256": _sha(allow)}})
    log = tmp_path / "admissions.jsonl"
    kw = {"allowlist": allow, "ledger": ledger, "log": log}
    return {"files": files, "allow": allow, "ledger": ledger, "log": log, "kw": kw, "dir": guarded, "tmp": tmp_path}


def test_allowed_sha_passes_and_is_logged(env):
    f = env["files"][3]
    with pytest.raises(CustodyError):            # the unchanged name rules refuse it on their own
        RA._name_check(f)
    assert RA.check_path(f, by="test", **env["kw"]) == f.resolve()
    recs = [json.loads(x) for x in env["log"].read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["event"] == "ADMIT" and recs[0]["sha256"] == _sha(f)
    assert recs[0]["authorization_seq"] == RA.AUTH_SEQ


def test_wrong_sha_with_guarded_name_refused(env):
    f = env["dir"] / "gb_ffffff.ply"
    f.write_bytes(b"not an authorized scan\n")
    with pytest.raises(CustodyError, match="not authorized"):
        RA.check_path(f, **env["kw"])
    assert not env["log"].exists()


def test_tampered_allowlist_refused(env):
    doc = json.loads(env["allow"].read_text())
    extra = env["dir"] / "gb_eeeeee.ply"
    extra.write_bytes(b"smuggled\n")
    doc["sha256"][0] = _sha(extra)
    env["allow"].write_text(json.dumps(doc, indent=1))
    for f in (extra, env["files"][1]):
        with pytest.raises(CustodyError, match="differs from the ledger NOTE"):
            RA.check_path(f, **env["kw"])
    assert not env["log"].exists()


def test_unbound_allowlist_refused(env):
    lines = env["ledger"].read_text().splitlines()
    env["ledger"].write_text("\n".join(lines[:-1]) + "\n")     # drop the binding NOTE
    with pytest.raises(CustodyError, match="no ledger NOTE"):
        RA.check_path(env["files"][0], **env["kw"])


def test_unrelated_guarded_paths_still_refused(env):
    for sub in ("x_holdout10", "N003", "eho4", "foo_SEALED", "rhino-preview", "38168001_02", "one_shot"):
        d = env["tmp"] / sub
        d.mkdir()
        f = d / "scan.ply"
        f.write_bytes(f"unrelated {sub}\n".encode())
        with pytest.raises(CustodyError):
            RA.check_path(f, **env["kw"])
    assert not env["log"].exists()


def test_name_clean_path_unchanged(env):
    d = env["tmp"] / "plain"
    d.mkdir()
    f = d / "scan.ply"
    f.write_bytes(b"ordinary\n")
    assert RA.check_path(f, **env["kw"]) == f.resolve()
    assert not env["log"].exists()


def test_stage_makes_name_clean_byte_copy(env):
    f = env["files"][5]
    dst = RA.stage(f, env["tmp"] / "stage" / "scans", **env["kw"])
    assert _sha(dst) == _sha(f) and RA._name_check(dst) == dst
    with pytest.raises(CustodyError):
        RA.stage(f, env["tmp"] / "stage_heldout", **env["kw"])
