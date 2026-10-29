"""E4b pre-A1 freeze ledger: verbatim plan thresholds, the FREEZE entry, the anchor, drift detection.

The ledger is written into a temporary freeze directory with E4a's hash-chained
ledger; the plan, scanner card and code fingerprints are the real ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from domains.production.engine.eval import freeze as e4a
from domains.production.engine.eval.freeze import ledger as lg

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "docs/production/PRODUCTION_LINE_PLAN.md"


def test_plan_sections_are_verbatim():
    text = PLAN.read_text(encoding="utf-8")
    secs = {s["section"]: s for s in lg.plan_sections(text)}
    assert secs["5"]["text"].startswith("## 5. Accuracy gates") and secs["5"]["text"] in text
    assert secs["10"]["text"].startswith("## 10. Binding amendments") and secs["10"]["text"] in text
    assert "line-F1 ≥ 0.80" in secs["5"]["text"] and "DEV-CHECK subset" in secs["10"]["text"]
    assert "## 6." not in secs["5"]["text"] and "## 11." not in secs["10"]["text"]
    with pytest.raises(ValueError):
        lg.plan_sections(text.replace("## 10. Binding", "## X. Binding"))


def _freeze_dir(tmp: Path) -> tuple[Path, Path]:
    fd, runlog = tmp / "runlog" / "freeze", tmp / "runlog"
    (fd / "recipes").mkdir(parents=True)
    (fd / "recipes_devcheck_SEALED").mkdir()
    (fd / "recipes" / "DEV-AAAAAAAAAA.recipe.json").write_text('{"r": 1}', encoding="utf-8")
    (fd / "recipes_devcheck_SEALED" / "DEV-BBBBBBBBBB.recipe.json").write_text('{"r": 2}', encoding="utf-8")
    for name in ("cases_A1", "g2_complete_A1", "g3_staged_A1", "known_good_A1", "repeatability_A1", "gold_g2a_A1",
                 "gold_g2a_relabel_A1", "gold_broad_A1", "broad_A1", "broad_staged_A1", "devcheck_SEALED",
                 "broad_reference_lanes_A1", "broad_reference_lanes_devcheck_SEALED"):
        (fd / f"{name}.json").write_text(json.dumps({"cases": [name]}), encoding="utf-8")
    for name in ("inventory", "gold_transforms", "compile_report", "gold_transforms_devcheck_SEALED"):
        (fd / f"{name}.json").write_text("{}", encoding="utf-8")
    (fd / "README.md").write_text("freeze\n", encoding="utf-8")
    return fd, runlog


def test_ledger_entry_anchor_and_clean_verification(tmp_path):
    fd, runlog = _freeze_dir(tmp_path)
    out = lg.write_ledger(fd, runlog=runlog)
    entry = e4a.first_freeze(fd / lg.LEDGER_NAME)
    assert entry["entry_sha256"] == out["entry_sha256"] and entry["recorded_by"] == "e4b"
    items = entry["items"]
    assert set(items["thresholds"]) == {"G1", "G2_LINE", "G2_PHYSICAL", "G3", "G4"}
    assert {(s["tier"], s["seedset"]) for s in items["seeds"]} == set(lg.FROZEN_SEEDS)
    assert items["recipes"]["count"] == 1 and "DEVCHECK_SEALED" not in items["case_lists"]
    assert {"BROAD_A1", "BROAD_STAGED_A1", "GOLD_BROAD_A1", "GOLD_G2A_RELABEL_A1"} <= set(items["case_lists"])
    sealed = items["devcheck_sealed"]
    assert set(sealed) == {"devcheck_SEALED", "gold_transforms_devcheck_SEALED", "broad_reference_lanes_devcheck_SEALED",
                           "recipes_devcheck_SEALED_aggregate"}
    assert all(set(v) <= {"sha256", "definition"} for v in sealed.values())  # sha only: no path, no count
    assert "gold_transforms_devcheck_SEALED" not in json.dumps(items["files"])  # the A1 part is the recorded file
    dec = items["decisions"]
    assert dec["compile_rate_gate_4d"]["rule_version"] == "E4B-4D-GATE-2"
    assert dec["compile_rate_gate_4d"]["gated_variant"] == "dev"
    assert "run_physical_gate" in dec["amendment_7_order"] and "builder_SEALED" in dec["work_directories"]["builder"]
    assert any("E4B-4D-GATE-2" in x for x in items["pre_a1"]["open_items"])
    ledger_text = (fd / lg.LEDGER_NAME).read_text(encoding="utf-8")
    assert "DEV-BBBBBBBBBB" not in ledger_text and "DEV-AAAAAAAAAA" in ledger_text
    assert (fd / "owner_directions.json").is_file() and "owner_directions" in items["files"]
    assert items["scanner_model_card"]["model_sha256"] == "cf55a20c23f37e9076426f0be559584fa9f00941db5d856ddc109bac4cabf8ce"
    assert set(items["code_fingerprints"]) == {"engine", "compile", "eval", "contracts"}
    assert "engine_config" not in items  # A1's knob: frozen later by the latest FREEZE that defines one
    assert (fd / lg.ANCHOR_NAME).read_text().split()[0] == out["entry_sha256"]
    verbatim = (fd / "thresholds_plan_verbatim.md").read_text(encoding="utf-8")
    assert "## 5. Accuracy gates" in verbatim and "## 10. Binding amendments" in verbatim
    ver = lg.verify_freeze(fd)
    assert ver["ok"], ver
    assert all(v["same"] for v in ver["code_fingerprints"].values())
    # the E4a check a qualifying run performs: anchored, no mismatch
    check = e4a.check_against_ledger(fd / lg.LEDGER_NAME, anchor_sha256=out["entry_sha256"],
                                     code_fingerprints=e4a.current_code_fingerprints(REPO, e4a.FROZEN_CODE),
                                     engine_config_sha256=None)
    assert check["ok"] and check["anchor_checked"]


def test_drift_and_rewrites_are_detected(tmp_path, monkeypatch):
    fd, runlog = _freeze_dir(tmp_path)
    out = lg.write_ledger(fd, runlog=runlog)
    (fd / "g2_complete_A1.json").write_text(json.dumps({"cases": ["moved"]}), encoding="utf-8")
    ver = lg.verify_freeze(fd)
    assert not ver["ok"] and {"item": "case_lists:G2_COMPLETE_A1", "problem": "CHANGED"} in ver["file_drift"]
    (fd / "g2_complete_A1.json").write_text(json.dumps({"cases": ["g2_complete_A1"]}), encoding="utf-8")
    assert lg.verify_freeze(fd)["ok"]
    # sealed material is re-hashed only on request (A1 never reads a sealed path)
    (fd / "recipes_devcheck_SEALED" / "DEV-BBBBBBBBBB.recipe.json").write_text('{"r": 9}', encoding="utf-8")
    assert lg.verify_freeze(fd)["ok"]
    sealed_ver = lg.verify_freeze(fd, include_sealed=True)
    assert not sealed_ver["ok"] and sealed_ver["file_drift"] == [
        {"item": "devcheck_sealed:recipes_devcheck_SEALED_aggregate", "problem": "CHANGED_OR_MISSING"}]
    (fd / "recipes_devcheck_SEALED" / "DEV-BBBBBBBBBB.recipe.json").write_text('{"r": 2}', encoding="utf-8")
    assert lg.verify_freeze(fd, include_sealed=True)["ok"]
    changed = {k: dict(v) for k, v in e4a.ALL_THRESHOLDS.items()}
    changed["G2_PHYSICAL"]["compile_rate_min"] = 0.5
    monkeypatch.setattr(e4a, "ALL_THRESHOLDS", changed)
    ver = lg.verify_freeze(fd)
    assert not ver["ok"] and "THRESHOLDS_CHANGED:G2_PHYSICAL" in ver["check_against_ledger"]["mismatches"]
    monkeypatch.undo()
    with pytest.raises(FileExistsError):
        lg.write_ledger(fd, runlog=runlog)
    (fd / "recipes" / "DEV-AAAAAAAAAA.recipe.json").write_text('{"r": 3}', encoding="utf-8")  # a pre-A1 fix
    assert not lg.verify_freeze(fd)["ok"]
    again = lg.write_ledger(fd, runlog=runlog, replace_before_a1=True)
    assert again["entry_sha256"] != out["entry_sha256"]
    ver = lg.verify_freeze(fd)
    assert ver["ok"] and ver["anchor"] == again["entry_sha256"]
    # a stale anchor (the rewritten ledger's old first entry) is caught
    (fd / lg.ANCHOR_NAME).write_text(out["entry_sha256"] + "\n", encoding="utf-8")
    assert not lg.verify_freeze(fd)["ok"]
    # once a QUAL run has been logged the ledger can no longer be replaced
    e4a.append_entry(fd / lg.LEDGER_NAME, kind="RUN", label="QUAL full run",
                     items={"tier": "full", "seedset": "QUAL", "case_ids_sha256": "0" * 64})
    with pytest.raises(PermissionError):
        lg.write_ledger(fd, runlog=runlog, replace_before_a1=True)
