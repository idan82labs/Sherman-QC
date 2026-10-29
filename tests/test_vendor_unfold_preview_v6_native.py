"""Full authored L: native V3 candidates to V6 flat and discrete oracle."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SHERMAN_AUTHORED_NATIVE_PHASE") != "1",
    reason="root-coordinated bounded native phase required")


def test_authored_l_v3_candidate_v6_unfold_oracle(tmp_path):
    source = Path(__file__).with_name("test_extract_step_topology_bend_axes_v3_native.py")
    spec = importlib.util.spec_from_file_location("authored_l", source)
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    from scripts.extract_step_topology_bend_axes_v3 import extract_step_bend_axes
    from backend.vendor_cad_bend_candidates_v3 import axis_rows_v3
    from backend.vendor_unfold_preview_v6 import build_unfold_preview
    path = tmp_path / "authored.step"
    fixture.write_step(fixture.authored_l_bend(), path)
    extraction = extract_step_bend_axes(path)
    candidates = axis_rows_v3(extraction["bend_axes"])
    result = build_unfold_preview(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidates)
    assert result["status"] == "AVAILABLE", json.dumps(result, sort_keys=True)
    assert len(result["bend_lines"]) == 1
    oracle = result["diagnostics"]["geometry_oracle"]
    assert oracle["status"] == "VERIFIED_UNDER_DECLARED_GEOMETRY_MODEL", oracle
    assert oracle["bend_count"] == 1
