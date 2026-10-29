from __future__ import annotations

import json
import ast
import os
from pathlib import Path
from typing import Any, Dict

from domains.bend.services.promotion_readiness import load_program_status

# Execute only these two exact orchestration helpers, not native worker imports.
_source = Path(__file__).resolve().parents[1] / "bend_analysis_worker.py"
_tree = ast.parse(_source.read_text())
_functions = [node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name in {
    "_guard_table_text", "_load_qualification_context",
}]
assert len(_functions) == 2
_namespace = {"__file__": str(_source), "Path": Path, "os": os, "Any": Any,
              "Dict": Dict, "load_program_status": load_program_status}
exec(compile(ast.Module(body=_functions, type_ignores=[]), str(_source), "exec"), _namespace)
_guard_table_text = _namespace["_guard_table_text"]
_load_qualification_context = _namespace["_load_qualification_context"]


def test_worker_readiness_missing_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERMAN_QC_F1_PROGRAM_STATUS_PATH", str(tmp_path / "missing.json"))
    result = _load_qualification_context()
    assert result["operating_mode"] == "RESEARCH_ONLY"
    assert result["production_use_allowed"] is False


def test_worker_readiness_requires_explicit_promotion_true(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2",
        "production_promotion_authorized": False,
        "active_blockers": ["NEEDS_UNTOUCHED_CORPUS"],
    }))
    monkeypatch.setenv("SHERMAN_QC_F1_PROGRAM_STATUS_PATH", str(path))
    blocked = _load_qualification_context()
    assert blocked["promotion_status"] == "PROMOTION_BLOCKED"
    assert len(blocked["program_status_sha256"]) == 64
    path.write_text(json.dumps({
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2",
        "production_promotion_authorized": True,
        "physical_generalization_progress": True,
        "promotion_evidence": {
            "status": "PASS",
            "candidate_lock_sha256": "a" * 64,
            "qualification_report_sha256": "b" * 64,
            "dual_review_truth_sha256": "c" * 64,
        },
    }))
    promoted = _load_qualification_context()
    assert promoted["operating_mode"] == "RESEARCH_ONLY"
    assert promoted["production_use_allowed"] is False
    assert promoted["promotion_status"] == "PROMOTION_EVIDENCE_INVALID_FAIL_CLOSED"


def test_table_export_is_marked_even_with_unverified_promotion_claim():
    blocked = _guard_table_text("B1 PASS", {"production_use_allowed": False})
    assert blocked.startswith("RESEARCH ONLY - NOT QUALIFIED")
    assert blocked.endswith("B1 PASS")
    assert _guard_table_text("B1 PASS", {"production_use_allowed": True}) == blocked
