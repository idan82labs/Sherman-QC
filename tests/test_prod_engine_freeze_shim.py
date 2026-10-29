"""E4b package shim: ``domains.production.engine.eval.freeze`` stays E4a's freeze ledger.

The E4b subpackage directory ``eval/freeze/`` shadows E4a's ``eval/freeze.py`` by
Python's import rules; its ``__init__`` executes E4a's module in place. These
tests pin that nothing about E4a's module changes: same file, same names,
monkeypatching and reload semantics, relative imports, and the E4a CLI.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_freeze_package_is_e4a_ledger_module():
    from domains.production.engine.eval import freeze as fz

    assert Path(fz.__file__).resolve() == REPO / "domains/production/engine/eval/freeze.py"
    assert fz.__package__ == "domains.production.engine.eval"
    assert fz.__spec__.origin == str(REPO / "domains/production/engine/eval/freeze.py")
    assert [Path(p).resolve() for p in fz.__path__] == [REPO / "domains/production/engine/eval/freeze"]
    for name in ("append_entry", "check_against_ledger", "freeze_items", "verify_ledger", "read_ledger",
                 "current_code_fingerprints", "LEDGER_SCHEMA", "CODE_ROOTS", "FROZEN_CODE", "WAIVABLE_SCOPES"):
        assert hasattr(fz, name), name
    assert fz.__doc__.startswith("Freeze ledger (plan §10 amendment 4")


def test_e4a_functions_see_monkeypatched_module_globals(tmp_path, monkeypatch):
    from domains.production.engine.eval import freeze as fz

    items = fz.freeze_items()
    monkeypatch.setattr(fz, "ALL_THRESHOLDS", {"ONLY": {"version": "x", "a": 1}})
    patched = fz.freeze_items()
    assert set(items["thresholds"]) == {"G1", "G2_LINE", "G2_PHYSICAL", "G3", "G4"}
    assert set(patched["thresholds"]) == {"ONLY"}  # the functions read this module's globals


def test_reload_keeps_relative_imports_working_without_warnings(tmp_path):
    from domains.production.engine.eval import freeze as fz

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.reload(fz)
        fps = fz.current_code_fingerprints(REPO, ("eval",))  # a function-level relative import (.runner)
    assert len(fps["eval"]) == 64
    led = tmp_path / "L.jsonl"
    fz.append_entry(led, label="t", items=fz.freeze_items())
    assert fz.verify_ledger(led)["ok"]


def test_e4b_submodules_import_and_eval_fingerprint_covers_them():
    from domains.production.engine.eval import freeze as fz
    from domains.production.engine.eval.freeze import cases, common, gold, ledger, legacy, physical, prefreeze  # noqa

    assert common.REPO_ROOT == REPO
    assert sys.modules["domains.production.engine.eval.freeze"] is fz
    from domains.production.engine.eval.runner import code_fingerprint

    roots = fz.CODE_ROOTS["eval"]
    files = sorted(p for p in (REPO / roots[0]).rglob("*.py") if "__pycache__" not in p.parts)
    assert REPO / "domains/production/engine/eval/freeze/common.py" in files
    assert len(code_fingerprint(REPO, roots)) == 64


def test_e4a_cli_still_reaches_the_ledger(tmp_path):
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'backend'}", "PYTHONDONTWRITEBYTECODE": "1"}
    ledger = tmp_path / "L.jsonl"
    out = subprocess.run([sys.executable, "-m", "domains.production.engine.eval", "freeze", "--ledger", str(ledger),
                          "--label", "shim"], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    doc = json.loads(out.stdout)
    assert doc["seq"] == 1
    ver = subprocess.run([sys.executable, "-m", "domains.production.engine.eval", "verify-ledger", "--ledger",
                          str(ledger)], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert ver.returncode == 0 and json.loads(ver.stdout)["ok"]
