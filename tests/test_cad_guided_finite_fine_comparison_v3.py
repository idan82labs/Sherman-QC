"""One frozen invented candidate-level V2/V3 mechanism comparison, not global fit.

Reason: the retained full V2 connected matrix is1/3 and its best generated
nominal candidate20 remains1.94mm from truth despite adequate local rank. This
separate diagnostic fixes that candidate index prospectively; it must NOT be
used as a production start selector or called a successful24-start workflow.

Generate one original nominal STEP/analytic PLY, one32768-sample finite export,
one unchanged V2 coarse call from generated PCA start20. Run one V2 fine control
and one V3 fine challenger from identical copies of that coarse result, each
with the unchanged20-iteration policy and its own default cumulative budgets.
All original geometric gates remain; no pose or Gold is supplied to either fine
call. Apply oracle only after BOTH arms terminate. Retain both failures. No
iteration/threshold/sample sweep or resume. Prior terminal attempts stay intact.

Prospective external limits60s wall/45sCPU/1GiB/1MiB; one serial heavy job. No
OCP at collection. The pytest success condition is complete diagnostic custody,
not a favorable geometry outcome. Advancement requires the reported explicit
criterion PLUS a later independently audited all24-start three-case workflow.
"""

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name("test_cad_guided_mesh_connected_native_dense_v1.py")
CANDIDATE_INDEX = 20
NATIVE_OPT_IN = "SHERMAN_RUN_FIXED_FINITE_FINE_COMPARISON_V3"
LIMITS = dict(wall_seconds=60, cpu_seconds=45, process_tree_rss_bytes=1024**3, output_bytes=1024**2)
ARMS = ("V2_FIXED_POINT_FINE_CONTROL", "V3_FINITE_GAUSS_NEWTON_FINE_CHALLENGER")


def _helpers():
    spec = importlib.util.spec_from_file_location("_fixed_fine_dense_fixture_v3", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inference(scan_points, surface, policy):
    """No known pose, original/raw identities or expected geometry are inputs."""
    from backend import cad_guided_global_registration_v1 as v1
    from backend import cad_guided_global_registration_v2 as v2
    from backend.cad_guided_finite_gauss_newton_v3 import refine_finite_surface_v3

    scan = v1._canonical_scan(scan_points)
    cad = v1._canonical_scan(surface["points"])
    coarse, refine = v1._spatial_sample(scan, policy.coarse_points), v1._spatial_sample(scan, policy.refine_points)
    cad_pca = v1._spatial_sample(cad, policy.refine_points)
    starts = list(v1._starts(coarse, cad_pca))
    assert len(starts) == 24
    index = v2.build_triangle_surface(surface["triangles"], surface["source_face_indices"])
    assert index["status"] == "COMPLETE", index["reason"]
    def queries():
        return v2._Queries(index["index"], v2.SurfaceRegistrationBudgetsV2(), v2.TriangleSurfaceBudgets())
    diagonal = float(v2._norm(np.ptp(surface["triangles"].reshape(-1, 3), axis=0)))
    coarse_query = queries()
    r, t = starts[CANDIDATE_INDEX]
    r, t, count, state, last = v2._refine(coarse, coarse_query, r, t,
        diagonal=diagonal, policy=policy, coarse=True)
    receipt = dict(candidate_index=CANDIDATE_INDEX, generated_start_count=len(starts),
        executed_start_count=1, policy=asdict(policy),
        coarse=dict(rotation=r.tolist(), translation=t.tolist(), iterations=count,
                    state=state, last=last, query_resources=dict(coarse_query.statistics)), arms=[])
    for arm in ARMS:
        query = queries()
        record = dict(arm=arm, status="FAILED", phase="FINE", rotation=None, translation=None)
        try:
            if arm == ARMS[0]:
                nr, nt, n, reason, details = v2._refine(refine, query, r.copy(), t.copy(),
                    diagonal=diagonal, policy=policy, coarse=False)
            else:
                nr, nt, n, reason, details = refine_finite_surface_v3(
                    refine, query, r.copy(), t.copy(), policy=policy)
            record.update(rotation=nr.tolist(), translation=nt.tolist(), iterations=n,
                          convergence_state=reason, last_step=details, phase="FINAL_UNCHANGED_DIAGNOSIS")
            record["diagnosis"] = v2._diagnose(scan, cad, refine, query, nr, nt, policy)
            record.update(status="TERMINAL_DIAGNOSTIC", phase="COMPLETE")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:200])
        record["query_resources"] = dict(query.statistics)
        receipt["arms"].append(record)
    return receipt


def _attach_oracle_and_decision(receipt, raw, cad_frame):
    """Post-inference only; never feeds any geometry back to the candidate."""
    for row in receipt["arms"]:
        row["post_inference_action_max_mm"] = None
        row["post_inference_action_rms_mm"] = None
        if row["rotation"] is not None and row["translation"] is not None:
            error = np.linalg.norm(raw@np.asarray(row["rotation"]).T+row["translation"]-cad_frame, axis=1)
            row.update(post_inference_action_max_mm=float(error.max()),
                       post_inference_action_rms_mm=float(np.sqrt(np.mean(error**2))))
    control, challenger = receipt["arms"]
    good = all(row["status"] == "TERMINAL_DIAGNOSTIC" for row in receipt["arms"])
    receipt["candidate_level_positive_criterion"] = bool(good and
        not challenger["diagnosis"]["reasons"] and challenger["post_inference_action_max_mm"] <= .20
        and challenger["post_inference_action_max_mm"] < control["post_inference_action_max_mm"])
    receipt["criterion_scope"] = "ONE_FIXED_SYNTHETIC_START_NOT_GLOBAL_REGISTRATION_OR_MEASUREMENT"
    receipt["production_use_allowed"] = False


def run_fixed_fine_comparison_v3():
    from backend.cad_guided_mesh_offline_v1 import _read_step_owned_snapshot
    from backend.cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
    from backend.cad_guided_cad_surface_v2 import sample_native_cad_surface_v2

    receipt = dict(protocol="FIXED_CANDIDATE_FINITE_FINE_COMPARISON_V3", resource_limits=LIMITS,
        status="FAILED", expected_arms=list(ARMS), arms=[], phase="INVENTED_FIXTURE")
    try:
        dense = _helpers()
        original = dense._original()
        policies = dense._policies("v1")
        wp = policies["workflow_policy"]
        assert wp.cad_sample_count == 32768
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
        raw, faces, cad_frame = original._analytic_mesh_scan(False)
        scan_blob = original._ply_bytes(raw, faces)
        scan = prepare_mesh_scan_v1(scan_bytes=scan_blob, max_engine_rows=wp.max_registration_scan_rows)
        assert scan["status"] == "AVAILABLE", scan.get("reasons")
        shape, units = _read_step_owned_snapshot(cad_blob)
        surface = sample_native_cad_surface_v2(shape, allow_ocp=True, sample_count=wp.cad_sample_count,
            linear_deflection_mm=wp.cad_linear_deflection_mm,
            angular_deflection_radians=wp.cad_angular_deflection_radians)
        assert surface["status"] == "AVAILABLE", surface.get("reasons")
        receipt.update(cad_sha256=hashlib.sha256(cad_blob).hexdigest(),
            scan_sha256=hashlib.sha256(scan_blob).hexdigest(), source_units=units,
            surface_digests=surface["digests"], triangle_count=len(surface["triangles"]),
            phase="GEOMETRY_ONLY_PAIRED_INFERENCE")
        receipt.update(_inference(scan["points"], surface, policies["registration_policy"]))
        _attach_oracle_and_decision(receipt, raw, cad_frame)
        receipt.update(status="COMPLETE_DIAGNOSTIC", phase="COMPLETE")
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error)[:300])
    while len(receipt["arms"]) < len(ARMS):
        receipt["arms"].append(dict(arm=ARMS[len(receipt["arms"])], status="NOT_ATTEMPTED",
            reason="PARENT_STAGE_FAILURE_RETAINED", rotation=None, translation=None))
    return receipt


def test_fixed_diagnostic_contract_is_not_global_inference_or_favorable_result_gate():
    assert CANDIDATE_INDEX == 20 and len(ARMS) == 2
    assert LIMITS == dict(wall_seconds=60, cpu_seconds=45, process_tree_rss_bytes=1024**3, output_bytes=1024**2)


def test_post_inference_decision_requires_both_arms_and_all_original_pose_gates():
    raw = np.array([[1., 2., 3.], [4., 5., 6.]])
    receipt = dict(arms=[dict(arm=arm, status="TERMINAL_DIAGNOSTIC", rotation=np.eye(3).tolist(),
        translation=[.1 if index else .4, 0., 0.], diagnosis={"reasons": []}) for index, arm in enumerate(ARMS)])
    _attach_oracle_and_decision(receipt, raw, raw)
    assert receipt["candidate_level_positive_criterion"]
    receipt["arms"][1]["diagnosis"]["reasons"] = ["LOCAL_POSE_UNOBSERVABLE"]
    _attach_oracle_and_decision(receipt, raw, raw)
    assert not receipt["candidate_level_positive_criterion"]


def test_fixed_native_fine_comparison_v3():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Requires the separate60/45s1GiB root-supervised invented candidate diagnostic")
    receipt = run_fixed_fine_comparison_v3()
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    assert [row["arm"] for row in receipt["arms"]] == list(ARMS)
    assert receipt["status"] == "COMPLETE_DIAGNOSTIC", receipt
    assert all(row["status"] == "TERMINAL_DIAGNOSTIC" for row in receipt["arms"]), receipt
