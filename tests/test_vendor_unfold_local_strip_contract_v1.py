"""Independent synthetic finite-strip ledgers; no CAD or physical inputs."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from backend import vendor_unfold_local_strip_contract_v1 as contract
from backend.vendor_unfold_local_strip_guard_v1 import LocalStripBudget, certify_local_finite_strip


def _certificate(side, budget, width=2.):
    polarity = -1. if side == "parent" else 1.
    vertices = [[0., 0., 0.], [10., 0., 0.], [10., 5.*polarity, 0.], [0., 5.*polarity, 0.]]
    result = certify_local_finite_strip(vertices, [[0, 1, 2], [0, 2, 3]],
        cylinder_origin=[0., 0., 0.], cylinder_axis=[1., 0., 0.], signed_tangent=[0., 1., 0.],
        seam_endpoints=[[0., 0., 0.], [10., 0., 0.]], axial_min=0., axial_max=10.,
        developed_strip_width=width, side=side, budget=budget)
    result.update(radius_mm=2., sweep_radians=1.)
    return result


def _payload(n=1, width=2.):
    budget = LocalStripBudget()
    names = ["z-first", "a-second"][:n]
    proofs, candidates, lines = [], [], []
    for index, ident in enumerate(names):
        proofs.append({"feature_id": ident, "cylinder_source_face_index": n+1+index,
            "parent_source_face_index": index, "child_source_face_index": index+1,
            "parent": _certificate("parent", budget, width), "child": _certificate("child", budget, width)})
        candidates.append({"feature_id": ident, "line_start": [0., float(index), 0.],
                           "line_end": [10., float(index), 0.]})
        lines.append({"feature_id": ident, "sweep_deg": math.degrees(1.)})
    diag = {"mechanism_version": 5, "transport_version": 4, "finite_width_model": contract.MODEL,
        "local_strip_source_sha256": contract.LOCAL_STRIP_SOURCE_SHA256,
        "v4_component_source_sha256": contract.V4_COMPONENT_SOURCE_SHA256,
        "candidate_count": n, "native_face_count": 2*(2*n+1),
        "selected_source_face_indices": list(range(2*n+1)),
        "selected_cylinder_face_indices": list(range(n+1, 2*n+1)), "root_source_face_index": 0,
        "selected_skin_rule": "LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN",
        "finite_width_proofs": sorted(proofs, key=lambda row: row["feature_id"]),
        "finite_width_work": budget.snapshot(), "limitations": sorted(contract.LIMITATIONS),
        **{key: True for key in contract.OBLIGATIONS}}
    return {"diagnostics": diag, "bend_lines": lines}, candidates


def test_actual_helper_one_bend_ledger_admits_without_mutation():
    payload, candidates = _payload()
    saved = copy.deepcopy(payload)
    assert contract.validate_local_strip_diagnostics(payload, candidates) is None
    assert payload == saved


def test_feature_sorted_two_bend_rows_reconstruct_one_budget_and_tree():
    payload, candidates = _payload(2)
    assert payload["diagnostics"]["finite_width_proofs"][0]["feature_id"] == "a-second"
    assert contract.validate_local_strip_diagnostics(payload, candidates) is None


@pytest.mark.parametrize("side", ["parent", "child"])
@pytest.mark.parametrize("field,value", [
    ("model", "NATIVE_TESSELLATION_FINITE_AXIAL_SLAB"), ("mechanism_version", True),
    ("geometry_mutated", True), ("manufacturing_ready", True), ("exact_binary_clipping", False),
    ("tolerance_mm", 1e-4), ("radius_mm", True), ("radius_mm", 0.), ("radius_mm", 1e7),
    ("sweep_radians", math.pi), ("sweep_radians", float("inf")),
    ("developed_strip_width_mm", 3.), ("charged_uncertainty_mm", 1e-3),
    ("intersecting_axial_triangles", 0), ("retained_remote_side_triangles", 99),
    ("native_seam_boundary_interval_count", 0), ("native_seam_local_material_side_checked", False),
    ("whole_supplied_mesh_finite_strip_checked", False), ("required_external_gates", []),
    ("certified_strip_intrusion_bounds", [0., 2.1]), ("axial_bounds", [0., 0.]),
])
def test_corrupt_certificate_refuses(side, field, value):
    payload, candidates = _payload()
    payload["diagnostics"]["finite_width_proofs"][0][side][field] = value
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)


@pytest.mark.parametrize("field,value", [
    ("mechanism_version", 4), ("mechanism_version", True), ("finite_width_model", "legacy"),
    ("local_strip_source_sha256", "0"*64), ("v4_component_source_sha256", "0"*64),
    ("native_face_count", True), ("selected_cylinder_face_indices", [1, 2]),
    ("root_source_face_index", 2), ("candidate_count", True), ("limitations", []),
    ("seam_closure_checked", False), ("whole_unchanged_mesh_overlap_checked", False),
])
def test_corrupt_mechanism_or_external_obligation_refuses(field, value):
    payload, candidates = _payload()
    payload["diagnostics"][field] = value
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)


def test_old_certificate_schema_cannot_be_forged_as_local_model():
    payload, candidates = _payload()
    payload["diagnostics"]["finite_width_proofs"][0]["parent"]["expanded_axial_bounds"] = [-1e-5, 10.00001]
    with pytest.raises(contract.LocalStripContractError, match="SCHEMA"):
        contract.validate_local_strip_diagnostics(payload, candidates)


@pytest.mark.parametrize("key", list(contract.WORK_CAPS))
def test_shared_work_counter_reset_refuses(key):
    payload, candidates = _payload(2)
    payload["diagnostics"]["finite_width_proofs"][0]["parent"]["shared_work"][key] -= 1
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)


@pytest.mark.parametrize("key", list(contract.WORK_CAPS))
def test_work_bool_and_over_budget_refuse(key):
    payload, candidates = _payload()
    payload["diagnostics"]["finite_width_work"][key] = True
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)
    payload["diagnostics"]["finite_width_work"][key] = contract.WORK_CAPS[key]+1
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)


def test_wrong_child_or_duplicate_cylinder_refuses_tree_bijection():
    payload, candidates = _payload(2)
    row = payload["diagnostics"]["finite_width_proofs"][0]
    row["child_source_face_index"] = 1
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)
    payload, candidates = _payload(2)
    row = payload["diagnostics"]["finite_width_proofs"][0]
    row["cylinder_source_face_index"] = payload["diagnostics"]["finite_width_proofs"][1]["cylinder_source_face_index"]
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)


def test_controlled_width_product_allows_only_one_serialized_ulp():
    payload, candidates = _payload(width=math.nextafter(2., math.inf))
    assert contract.validate_local_strip_diagnostics(payload, candidates) is None
    payload, candidates = _payload(width=math.nextafter(math.nextafter(2., math.inf), math.inf))
    with pytest.raises(contract.LocalStripContractError, match="WIDTH_MISMATCH"):
        contract.validate_local_strip_diagnostics(payload, candidates)


def test_parent_child_width_and_actual_exported_sweep_are_bound():
    payload, candidates = _payload()
    payload["diagnostics"]["finite_width_proofs"][0]["child"]["sweep_radians"] = math.nextafter(1., math.inf)
    with pytest.raises(contract.LocalStripContractError, match="PARENT_CHILD_BINDING"):
        contract.validate_local_strip_diagnostics(payload, candidates)
    payload, candidates = _payload()
    payload["bend_lines"][0]["sweep_deg"] += .001
    with pytest.raises(contract.LocalStripContractError, match="EXPORTED_SWEEP"):
        contract.validate_local_strip_diagnostics(payload, candidates)


def test_parent_child_events_must_be_adjacent_and_parent_previously_developed():
    payload, candidates = _payload(2)
    rows = payload["diagnostics"]["finite_width_proofs"]
    # Change source graph without changing coherent arithmetic counters.
    rows[1]["parent_source_face_index"] = 2
    with pytest.raises(contract.LocalStripContractError, match="TREE_CHRONOLOGY"):
        contract.validate_local_strip_diagnostics(payload, candidates)


def test_no_addon_admission_for_missing_candidates_or_line_sweeps():
    payload, candidates = _payload()
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, [])
    payload["bend_lines"][0].pop("sweep_deg")
    with pytest.raises(contract.LocalStripContractError):
        contract.validate_local_strip_diagnostics(payload, candidates)


def test_browser_exact_dyadic_mirror_matches_python_on_authored_receipts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for cross-language contract verification")
    fixtures = []
    for n in (1, 2):
        payload, candidates = _payload(n)
        fixtures.append((payload, candidates))
    for radius, sweep in ((2., 1e-4), (.5, 2.9), (1e6, .1), (3.5, math.radians(175))):
        payload, candidates = _payload(width=radius*sweep)
        for side in ("parent", "child"):
            payload["diagnostics"]["finite_width_proofs"][0][side].update(radius_mm=radius, sweep_radians=sweep)
        payload["bend_lines"][0]["sweep_deg"] = math.degrees(sweep)
        fixtures.append((payload, candidates))
    mutations = [("model", "legacy"), ("mechanism_version", True), ("radius_mm", True),
        ("developed_strip_width_mm", 2.01), ("sweep_radians", math.pi),
        ("charged_uncertainty_mm", .001), ("native_seam_boundary_interval_count", False),
        ("certified_strip_intrusion_bounds", [1e-5, 2.01]), ("geometry_mutated", True),
        ("required_external_gates", []), ("native_seam_local_material_side_checked", False)]
    for field, value in mutations:
        payload, candidates = _payload()
        payload["diagnostics"]["finite_width_proofs"][0]["parent"][field] = value
        fixtures.append((payload, candidates))
    for key in contract.WORK_CAPS:
        payload, candidates = _payload(2)
        payload["diagnostics"]["finite_width_proofs"][0]["parent"]["shared_work"][key] -= 1
        fixtures.append((payload, candidates))
    for width in (math.nextafter(2., math.inf), math.nextafter(math.nextafter(2., math.inf), math.inf)):
        fixtures.append(_payload(width=width))
    expected = []
    for payload, candidates in fixtures:
        try:
            contract.validate_local_strip_diagnostics(payload, candidates)
            expected.append(True)
        except contract.LocalStripContractError:
            expected.append(False)
    source = """
import fs from 'node:fs';
import ts from 'typescript';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const source=fs.readFileSync('src/features/part-catalog/pages/vendorUnfoldLocalStrip.ts','utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2020,module:ts.ModuleKind.ESNext}}).outputText;
const module=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
console.log(JSON.stringify(input.map(([p,c])=>module.validUnfoldLocalStripDiagnostics(p.diagnostics,c,p.bend_lines))));
"""
    frontend = Path(__file__).resolve().parents[1]/"frontend/react"
    result = subprocess.run([node, "--input-type=module", "-e", source], cwd=frontend,
                            input=json.dumps(fixtures), capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout) == expected
    assert expected.count(True) >= 7 and expected.count(False) >= 10
