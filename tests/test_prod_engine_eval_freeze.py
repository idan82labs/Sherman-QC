"""E4a freeze ledger: hash-chained entries, tamper detection, the qualifying check,
and the evaluation CLI entry points."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from domains.production.engine.eval import freeze as fz
from domains.production.engine.eval.scorer import build_scorecard
from domains.production.engine.eval.thresholds import ALL_THRESHOLDS, G1_THRESHOLDS, thresholds_bundle, thresholds_sha256
from domains.production.engine.synthetic.cases import SeedsetLockedError, plan_cases

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "docs" / "production" / "runlog" / "scanner_model_card.json"


def _items(tmp_path: Path) -> dict:
    cl = tmp_path / "g2_cases.json"
    cl.write_text(json.dumps({"cases": ["c1", "c2", "c3"]}))
    cfg = tmp_path / "engine_config.json"
    cfg.write_text('{"k": 1}\n')
    recipe = tmp_path / "r.json"
    recipe.write_text("{}\n")
    return fz.freeze_items(seeds=[("regression", "DEV")], recipes=[recipe], case_lists={"G2": cl},
                           engine_config=cfg, scanner_model_card=CARD,
                           code_roots={"eval": ["domains/production/engine/eval"]})


def test_items_hash_everything_that_is_frozen(tmp_path):
    items = _items(tmp_path)
    assert set(items["thresholds"]) == set(ALL_THRESHOLDS)
    assert items["thresholds"]["G1"]["sha256"] == thresholds_sha256(G1_THRESHOLDS)
    seeds = items["seeds"][0]
    assert seeds["count"] == len(plan_cases("regression", "DEV")) and len(seeds["seeds_sha256"]) == 64
    assert items["case_lists"]["G2"]["count"] == 3
    assert items["scanner_model_card"]["provenance"] == "MODEL_CARD"
    assert len(items["code_fingerprints"]["eval"]) == 64
    with pytest.raises(SeedsetLockedError):
        fz.freeze_items(seeds=[("full", "QUAL")])  # QUAL seeds need the explicit opt-in
    assert fz.freeze_items(seeds=[("full", "QUAL")], allow_qual=True)["seeds"][0]["seedset"] == "QUAL"
    assert set(thresholds_bundle()) == set(ALL_THRESHOLDS)


def test_ledger_chain_detects_tampering(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    e1 = fz.append_entry(ledger, label="pre-A1 freeze", items=_items(tmp_path))
    e2 = fz.append_entry(ledger, label="note", items={"text": "x"}, kind="NOTE")
    assert e2["previous_sha256"] == e1["entry_sha256"] and fz.verify_ledger(ledger)["ok"]
    lines = ledger.read_text().splitlines()
    doc = json.loads(lines[0])
    doc["items"]["thresholds"]["G1"]["sha256"] = "0" * 64
    ledger.write_text("\n".join([json.dumps(doc, sort_keys=True), lines[1]]) + "\n")
    check = fz.verify_ledger(ledger)
    assert not check["ok"] and "1:HASH" in check["problems"]
    with pytest.raises(ValueError):
        fz.append_entry(ledger, label="after tamper", items={})
    ledger.write_text("\n".join([lines[1], lines[0]]) + "\n")  # reordered
    assert {"1:SEQUENCE", "1:CHAIN"} <= set(fz.verify_ledger(ledger)["problems"])
    with pytest.raises(ValueError):
        fz.append_entry(tmp_path / "x.jsonl", label="bad", items={}, kind="EDIT")


def test_check_against_ledger_and_qualifying_scorecards(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    assert fz.check_against_ledger(ledger)["mismatches"] == ["NO_FREEZE_ENTRY"]
    fz.append_entry(ledger, label="freeze", items=_items(tmp_path))
    ids = [s.case_id for s in plan_cases("regression", "DEV")]
    ok = fz.check_against_ledger(ledger, case_ids=ids, tier="regression", seedset="DEV",
                                 engine_config=tmp_path / "engine_config.json")
    assert ok["ok"], ok
    assert "CASE_LIST_DIFFERS_FROM_FROZEN" in fz.check_against_ledger(ledger, case_ids=ids[:-1], tier="regression",
                                                                      seedset="DEV")["mismatches"]
    assert "SCANNER_MODEL_DIFFERS_FROM_FROZEN" in fz.check_against_ledger(
        ledger, scanner_model_sha256="f" * 64)["mismatches"]
    (tmp_path / "engine_config.json").write_text('{"k": 2}\n')
    assert "ENGINE_CONFIG_DIFFERS_FROM_FROZEN" in fz.check_against_ledger(
        ledger, engine_config=tmp_path / "engine_config.json")["mismatches"]
    changed = json.loads(json.dumps(ALL_THRESHOLDS))
    changed["G1"]["coverage_min"] = 0.5  # a threshold moved mid-run
    monkeypatch.setattr(fz, "ALL_THRESHOLDS", changed)
    bad = fz.check_against_ledger(ledger)
    assert not bad["ok"] and "THRESHOLDS_CHANGED:G1" in bad["mismatches"]
    manifest = {"tier": "full", "seedset": "QUAL", "scanner_model": {"provenance": "MODEL_CARD"}}
    card = build_scorecard([], tier="full", manifest=manifest, ledger_check=bad, bootstrap=False)
    assert "FREEZE_LEDGER_MISMATCH" in card["not_qualifying_reasons"] and not card["qualifying"]


def test_cli_entry_points(tmp_path):
    env_py = [sys.executable, "-m", "domains.production.engine.eval"]
    help_out = subprocess.run(env_py + ["--help"], capture_output=True, text=True, cwd=REPO)
    assert help_out.returncode == 0 and "freeze" in help_out.stdout and "packet" in help_out.stdout
    ledger = tmp_path / "l.jsonl"
    out = subprocess.run(env_py + ["freeze", "--ledger", str(ledger), "--label", "cli", "--seeds", "smoke:DEV",
                                   "--scanner-model-card", str(CARD)], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    ver = subprocess.run(env_py + ["verify-ledger", "--ledger", str(ledger)], capture_output=True, text=True, cwd=REPO)
    assert ver.returncode == 0 and json.loads(ver.stdout)["ok"]
    ledger.write_text(ledger.read_text().replace('"cli"', '"edited"'))
    ver = subprocess.run(env_py + ["verify-ledger", "--ledger", str(ledger)], capture_output=True, text=True, cwd=REPO)
    assert ver.returncode == 2
    missing = subprocess.run(env_py + ["score", "--run-dir", str(tmp_path / "nope")], capture_output=True, text=True,
                             cwd=REPO)
    assert missing.returncode == 2



# ---------------------------------------------------------------------------
# fix-stage: re-freeze, frozen code, anchor, QUAL runs, waivers
# ---------------------------------------------------------------------------


def test_a_refreeze_after_a_threshold_change_is_detected(tmp_path, monkeypatch):
    """FREEZE(pre-A1), lower G1 coverage_min, FREEZE(re-freeze): the ledger still verifies,
    but the check must not accept the moved threshold."""
    ledger = tmp_path / "ledger.jsonl"
    fz.append_entry(ledger, label="pre-A1", items=_items(tmp_path))
    changed = json.loads(json.dumps(ALL_THRESHOLDS))
    changed["G1"]["coverage_min"] = 0.5
    monkeypatch.setattr(fz, "ALL_THRESHOLDS", changed)
    fz.append_entry(ledger, label="re-freeze", items=fz.freeze_items())
    assert fz.verify_ledger(ledger)["ok"]
    check = fz.check_against_ledger(ledger)
    assert not check["ok"]
    assert {"THRESHOLDS_REFROZEN:G1", "THRESHOLDS_CHANGED:G1"} <= set(check["mismatches"])
    assert check["entry_id"] != check["latest_freeze_entry_id"]  # the first freeze stays authoritative


def test_frozen_eval_code_must_not_change_unless_the_owner_waives_it(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    items = fz.freeze_items(code_roots={k: list(fz.CODE_ROOTS[k]) for k in fz.FROZEN_CODE})
    fz.append_entry(ledger, label="pre-A1", items=items)
    current = fz.current_code_fingerprints(REPO, fz.FROZEN_CODE)
    assert fz.check_against_ledger(ledger, code_fingerprints=current)["ok"]
    moved = dict(current, eval="0" * 64)  # the scorer changed after the freeze
    check = fz.check_against_ledger(ledger, code_fingerprints=moved)
    assert not check["ok"] and check["mismatches"] == ["EVAL_CODE_CHANGED_SINCE_FREEZE"]
    with pytest.raises(ValueError):
        fz.append_entry(ledger, label="w", kind="WAIVER", items={"scope": "EVAL_CODE_CHANGED_SINCE_FREEZE",
                                                                 "fingerprint": "0" * 64, "reason": "x"})  # no owner
    with pytest.raises(ValueError):
        fz.append_entry(ledger, label="w", kind="WAIVER", items={"scope": "THRESHOLDS_CHANGED", "approved_by": "o",
                                                                 "reason": "x"})  # never waivable
    fz.append_entry(ledger, label="owner waiver", kind="WAIVER", items={
        "scope": "EVAL_CODE_CHANGED_SINCE_FREEZE", "fingerprint": "0" * 64, "approved_by": "owner", "reason": "bug fix"})
    check = fz.check_against_ledger(ledger, code_fingerprints=moved)
    assert check["ok"] and check["waived"] == ["EVAL_CODE_CHANGED_SINCE_FREEZE"]
    other = dict(current, eval="1" * 64)  # a waiver covers exactly the fingerprint it names
    assert fz.check_against_ledger(ledger, code_fingerprints=other)["mismatches"] == ["EVAL_CODE_CHANGED_SINCE_FREEZE"]
    assert fz.check_against_ledger(tmp_path / "none.jsonl", code_fingerprints=current)["mismatches"] == [
        "NO_FREEZE_ENTRY"]


def test_the_external_anchor_detects_a_rewritten_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    first = fz.append_entry(ledger, label="pre-A1", items=_items(tmp_path))
    check = fz.check_against_ledger(ledger, anchor_sha256=first["entry_sha256"])
    assert check["ok"] and check["anchor_checked"] and check["first_entry_sha256"] == first["entry_sha256"]
    assert not fz.check_against_ledger(ledger)["anchor_checked"]
    ledger.unlink()  # someone rebuilds the ledger from scratch: it verifies, the anchor does not
    fz.append_entry(ledger, label="pre-A1 (rewritten)", items=fz.freeze_items())
    assert fz.verify_ledger(ledger)["ok"]
    assert "LEDGER_ANCHOR_MISMATCH" in fz.check_against_ledger(ledger, anchor_sha256=first["entry_sha256"])["mismatches"]


def test_qual_run_entries_flag_a_second_engine_on_the_qual_seeds(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    fz.append_entry(ledger, label="pre-A1", items=_items(tmp_path))
    a = {"engine_fingerprint": "a" * 64, "engine_config_sha256": None}
    run = {"tier": "full", "seedset": "QUAL", "case_ids_sha256": "c" * 64, **a}
    fz.append_entry(ledger, label="QUAL full run", kind="RUN", items=run)
    check = fz.check_against_ledger(ledger, tier="full", engine_identity=a)
    assert len(check["qual_runs"]) == 1 and not check["qual_seeds_reused"]  # a resume of the same final run
    b = dict(a, engine_fingerprint="b" * 64)
    fz.append_entry(ledger, label="QUAL full run", kind="RUN", items=dict(run, **b))
    check = fz.check_against_ledger(ledger, tier="full", engine_identity=b)
    assert len(check["qual_runs"]) == 2 and check["qual_seeds_reused"]
    with pytest.raises(ValueError):
        fz.append_entry(ledger, label="bad run", kind="RUN", items={"tier": "full"})


def test_waive_cli_requires_the_owner_fields(tmp_path):
    env_py = [sys.executable, "-m", "domains.production.engine.eval"]
    ledger = tmp_path / "l.jsonl"
    out = subprocess.run(env_py + ["waive", "--ledger", str(ledger), "--scope", "SCANNER_MODEL_PARTIAL",
                                   "--approved-by", "owner", "--reason", "dropout not measurable from exports"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 2  # no card / fields named
    out = subprocess.run(env_py + ["waive", "--ledger", str(ledger), "--scope", "SCANNER_MODEL_PARTIAL",
                                   "--approved-by", "owner", "--reason", "dropout not measurable from exports",
                                   "--scanner-model-card", str(CARD), "--field", "grazing_start_deg"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    entry = fz.read_ledger(ledger)[0]
    assert entry["kind"] == "WAIVER" and entry["items"]["fields"] == ["grazing_start_deg"]
    assert entry["recorded_by"] == "owner-waiver-cli"
