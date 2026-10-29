"""Synthetic generation custody checks; never launch native work or access CAD."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def fixture_generator(tmp_path, monkeypatch, generation):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("synthetic_vendor_unfold_generator", root / "scripts/build_vendor_unfold_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    (tmp_path / "backend").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "backend/vendor_unfold_preview.py").write_text("synthetic engine\n")
    (tmp_path / "backend/vendor_unfold_worker.py").write_text("synthetic supervisor\n")
    (tmp_path / "scripts/extract_step_topology_bend_axes.py").write_bytes((root / "scripts/extract_step_topology_bend_axes.py").read_bytes())
    folder = tmp_path / "sites/vendor-tagging-demo/public/vendor-corpus"
    folder.mkdir(parents=True)
    for number, _, sha in module.SOURCES:
        (folder / f"{number}.json").write_text(json.dumps({"part_number": number, "cad_sha256": sha, "candidates": []}))
    calls = []
    def no_native(path, sha, candidates):
        calls.append((path, sha, candidates))
        return {"status": "UNAVAILABLE", "reason": "SYNTHETIC_ONLY", "vertices": [], "triangles": [], "bend_lines": [], "diagnostics": {}}
    module.build_unfold_preview_bounded = no_native
    monkeypatch.setattr(sys, "argv", ["generator", "--engine-sha256", hashlib.sha256(b"synthetic engine\n").hexdigest(),
        "--supervisor-sha256", hashlib.sha256(b"synthetic supervisor\n").hexdigest(), "--generation", generation])
    return module, folder, calls


@pytest.mark.parametrize("generation", ["unfold-generation-v2", "unfold-generation-v3"])
def test_all_selected_cases_retained_and_existing_run_is_terminal(tmp_path, monkeypatch, generation):
    module, folder, calls = fixture_generator(tmp_path, monkeypatch, generation)
    module.main()
    report = json.loads((folder / f"{generation}.json").read_text())
    assert report["status"] == "COMPLETE"
    assert len(calls) == len(report["dispositions"]) == 5
    assert all(row["reason"] == "SYNTHETIC_ONLY" for row in report["dispositions"])
    assert report["limits"]["wall_seconds"] == 60
    assert report["limits"]["rss_gib"] == 4
    with pytest.raises(ValueError, match="terminal"):
        module.main()
    assert len(calls) == 5


@pytest.mark.parametrize("generation", ["unfold-generation-v2", "unfold-generation-v3"])
def test_failed_input_binding_retains_not_started_cases(tmp_path, monkeypatch, generation):
    module, folder, calls = fixture_generator(tmp_path, monkeypatch, generation)
    number = module.SOURCES[2][0]
    fixture = json.loads((folder / f"{number}.json").read_text())
    fixture["cad_sha256"] = "wrong"
    (folder / f"{number}.json").write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="identity mismatch"):
        module.main()
    report = json.loads((folder / f"{generation}.json").read_text())
    assert report["status"] == "FAILED"
    assert len(report["dispositions"]) == 5
    assert len(calls) == 2
    assert [row["status"] for row in report["dispositions"]] == ["UNAVAILABLE", "UNAVAILABLE", "NOT_STARTED", "NOT_STARTED", "NOT_STARTED"]
