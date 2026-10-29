from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_exp_cad_0030_cad_only_preflight.py"
PREREG = ROOT / "docs" / "research" / "2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json"
CAD = ROOT / "data" / "bend_corpus" / "incoming" / "2026-08-17_20_shaked_new_good_bad_pairs" / "authenticated_source" / "cad" / "42637000_B.SLDPRT"
MESH_ROOT = ROOT / "output" / "research" / "autoresearch-loop" / "exp-cad-0015-new-physical-intake-v1" / "cad_meshes"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("exp30_cad_wrapper", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_exp30_wrapper_bootstraps_outside_project_cwd() -> None:
    completed = subprocess.run(
        [sys.executable, str(WRAPPER), "--help"],
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "EXP_CAD_0030" in completed.stdout
    assert "EXP_CAD_0029" not in completed.stdout


def test_exp30_identity_rejects_exp29_schema_before_geometry(tmp_path: Path) -> None:
    wrapper = load_wrapper()
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    payload["schema"] = "EXP_CAD_0029_CFPDG_PREREGISTRATION_V2_PRE_SCAN_AUDIT_AMENDMENT"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        wrapper.BASE.execute_preflight(
            preregistration_path=invalid,
            source_cad_path=CAD,
            mesh_path=MESH_ROOT / "42637000_B.ply",
            mesh_report_path=MESH_ROOT / "42637000_B.report.json",
            output_root=tmp_path / "must-not-exist",
        )


def test_exp30_emitted_identity_and_wrapper_binding_are_exact(tmp_path: Path) -> None:
    wrapper = load_wrapper()
    result = wrapper.BASE.execute_preflight(
        preregistration_path=PREREG,
        source_cad_path=CAD,
        mesh_path=MESH_ROOT / "42637000_B.ply",
        mesh_report_path=MESH_ROOT / "42637000_B.report.json",
        output_root=tmp_path / "cad",
    )
    assert result["schema"] == "EXP_CAD_0030_CFPDG_V3_CAD_ONLY_PREFLIGHT_V1"
    assert result["experiment_id"] == "EXP_CAD_0030"
    assert result["architecture"] == "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
    assert result["identity_wrapper_sha256"] == wrapper.BASE.CORE.sha256_path(WRAPPER)
    assert result["implementation_sha256"] == wrapper.BASE.CORE.sha256_path(
        Path(wrapper.BASE.__file__)
    )
    assert result["core_implementation_sha256"] == wrapper.BASE.CORE.sha256_path(
        Path(wrapper.BASE.CORE.__file__)
    )
    assert result["physical_scan_open_call_count"] == 0
