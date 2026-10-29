"""Frozen diagnostic of dense synthetic registration candidate 20; not a retry.

The retained 32,768-sample matrix ``mesh_connected_native_dense_v1-j5p3bnlr``
remains negative (1/3). This separate diagnostic traces ONLY its nominal v1
candidate 20 on the exact original invented geometry and strict policies. It
does not optimize a policy, select another candidate, or assert that registration
should pass. Diagnostic completion is not engine improvement or qualification.

Four boundary snapshots are frozen: generated PCA start; after the coarse call
under the coarse gate; before the fine call under the fine gate; after the fine
call under the fine gate. The existing ``_refine`` is called once per phase with
its original 20-iteration maximum and unchanged natural early termination. No
runtime monkeypatch, forced pose, supplied correspondence, or iteration-until-
pass is used. After-phase sets are NEXT selections at the returned pose, NOT
claims about the preceding last optimizer step. In particular the after-coarse
and before-fine poses are identical, but their correspondence gates differ.

All ranks are diagnostics of the normalized point-to-plane Jacobian. The coarse
solver itself remains the source's point-to-point Kabsch algorithm. Direction
counts use the invented CAD frame: only its remote end caps have z-normal
support. They never affect inference. Oracle action errors and original raw-row
identities are attached only after both refinement calls have returned.

Root must separately authorize execution under the fixed external 60 s wall /
45 s CPU / 1 GiB process-tree RSS budget, after the physical job is terminal.
Only one constructed STEP and one analytic nominal mesh PLY are generated in
owned temporary storage. No physical/raw/Gold/held-out/retained output file is
opened. No native work occurs at import or ordinary source-test collection.
The source-only controls are authored here, not claimed executed.
"""

import ast
import builtins
import copy
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest


ORIGINAL_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_native_v1.py")
ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_SOURCE = ROOT / "backend/cad_guided_global_registration_v1.py"
PREDECESSOR_RECEIPT = "mesh_connected_native_dense_v1-j5p3bnlr"
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_REGISTRATION_STAGE_TRACE_V1"
CASE_NAME = "dense_nominal_candidate_20_stage_trace"
CANDIDATE_INDEX = 20
CAD_SAMPLE_COUNT = 32768
RESOURCE_LIMITS = {"wall_seconds": 60, "cpu_seconds": 45, "process_tree_rss_bytes": 1024**3}
STAGES = ("PCA_START", "AFTER_COARSE_NEXT_SELECTION", "BEFORE_FIRST_FINE_SELECTION",
          "AFTER_FINE_NEXT_SELECTION")
SVD_RCOND = 1e-10  # Same numerical cutoff as the unchanged fine least-squares call.
NORMAL_COMPONENT_EPSILON = 1e-8  # Diagnostic binning only; never an inference gate.


@lru_cache(maxsize=1)
def _original():
    spec = importlib.util.spec_from_file_location("_stage_trace_original_mesh_fixture", ORIGINAL_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policies():
    request = _original()._policies("v1")
    assert request["workflow_policy"].cad_sample_count == 8192
    return {**request, "workflow_policy": replace(request["workflow_policy"],
                                                 cad_sample_count=CAD_SAMPLE_COUNT)}


def _source_digests():
    paths = (Path(__file__).resolve(), ORIGINAL_SOURCE.resolve(), REGISTRATION_SOURCE,
             ROOT / "backend/cad_guided_mesh_offline_v1.py",
             ROOT / "backend/cad_guided_mesh_input_v1.py",
             ROOT / "backend/cad_guided_cad_sampling_v1.py",
             ROOT / "backend/cad_guided_native_features_v1.py")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _exact_row_ids(reference, selected):
    """Identity lookup between actual rows, not nearest-neighbor correspondence."""
    lookup = {tuple(row): index for index, row in enumerate(reference)}
    assert len(lookup) == len(reference), "FIXED_FIXTURE_REQUIRES_UNIQUE_IDENTITY_ROWS"
    ids = np.asarray([lookup[tuple(row)] for row in selected], dtype=np.int64)
    assert np.array_equal(reference[ids], selected)
    return ids


def _svd_evidence(jacobian):
    if len(jacobian) == 0:
        return dict(status="NO_ROWS", normalized_singular_values=[0.] * 6,
                    rank_at_solver_rcond=0, solver_rcond=SVD_RCOND,
                    sigma_min_over_sigma_max=0., smallest_right_singular_vector=None)
    _, values, right = np.linalg.svd(jacobian, full_matrices=len(jacobian) < 6)
    values = np.pad(values, (0, 6 - len(values))) / math.sqrt(len(jacobian))
    vector = right[-1].copy()
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0:
        vector *= -1  # Sign-only canonicalization; no geometric information added.
    cutoff = float(values[0] * SVD_RCOND)
    return dict(status="AVAILABLE", normalized_singular_values=values.tolist(),
                rank_at_solver_rcond=int(np.count_nonzero(values > cutoff)),
                solver_rcond=SVD_RCOND, normalized_rank_cutoff=cutoff,
                sigma_min_over_sigma_max=float(values[-1] / values[0]) if values[0] else 0.,
                smallest_right_singular_vector=vector.tolist(),
                smallest_singular_value_is_float_zero=bool(values[-1] == 0.),
                null_status="FLOAT64_DIAGNOSTIC_NOT_EXACT_NULLSPACE_CERTIFICATION",
                coordinate_order=["L_omega_x", "L_omega_y", "L_omega_z", "t_x", "t_y", "t_z"])


def _subset_evidence(registration, points, normals, nearest, distance, residual,
                     rows, canonical_ids, policy):
    selected_normals = normals[nearest[rows]]
    absolute = np.abs(selected_normals)
    dominant = np.argmax(absolute, axis=1) if len(rows) else np.empty(0, dtype=int)
    caps = absolute[:, 2] >= 1 - NORMAL_COMPONENT_EPSILON
    out_of_xy = absolute[:, 2] > NORMAL_COMPONENT_EPSILON
    record = dict(count=len(rows), optimizer_local_row_ids=rows.tolist(),
                  canonical_scan_row_ids=canonical_ids[rows].tolist(),
                  matched_canonical_cad_row_ids=nearest[rows].tolist(),
                  nearest_sample_distance_mm=distance[rows].tolist(),
                  signed_plane_residual_mm=residual[rows].tolist(),
                  normal_count_semantics="SCAN_TO_CAD_NORMAL_CORRESPONDENCES_NOT_DISTINCT_CONSTRAINTS",
                  jacobian_chart="SUBSET_CENTER_AND_RMS_LENGTH_RECOMPUTED_PER_SUBSET",
                  physical_twist_conversion="omega=z_rotation/L; v=z_translation-cross(omega,center)",
                  normal_direction_counts=dict(
                      dominant_axis={axis: int(np.count_nonzero(dominant == index))
                                     for index, axis in enumerate(("x", "y", "z"))},
                      axis_aligned={axis: int(np.count_nonzero(absolute[:, index] >= 1 - NORMAL_COMPONENT_EPSILON))
                                    for index, axis in enumerate(("x", "y", "z"))},
                      xy_plane_normals=int(np.count_nonzero(~out_of_xy)),
                      nonzero_z_normals=int(np.count_nonzero(out_of_xy)),
                      cap_z_normals=int(np.count_nonzero(caps)),
                      component_epsilon=NORMAL_COMPONENT_EPSILON),
                  cap_canonical_scan_row_ids=canonical_ids[rows[caps]].tolist())
    if not len(rows):
        record.update(jacobian_center=None, jacobian_length_scale_mm=None,
                      unweighted=_svd_evidence(np.empty((0, 6))),
                      huber_weighted=_svd_evidence(np.empty((0, 6))))
        return record
    jac, center, scale = registration._jacobian(points[rows], selected_normals)
    weights = np.minimum(1., policy.huber_delta_mm / np.maximum(np.abs(residual[rows]), 1e-15))
    record.update(jacobian_center=center.tolist(), jacobian_length_scale_mm=float(scale),
                  unweighted=_svd_evidence(jac),
                  huber_weighted=_svd_evidence(jac * np.sqrt(weights)[:, None]))
    return record


def _snapshot(registration, scan, optimizer, refine, cad, normals, tree, rotation,
              translation, *, stage, policy, coarse):
    points = optimizer @ rotation.T + translation
    distance, nearest = tree.query(points, k=1, workers=1)
    gate = (max(policy.max_correspondence_distance_mm,
                policy.coarse_distance_fraction * float(np.linalg.norm(np.ptp(cad, axis=0))))
            if coarse else policy.max_correspondence_distance_mm)
    admissible = np.flatnonzero(distance <= gate)
    trimmed = registration._trim_indices(distance, gate, policy.trim_fraction)
    residual = np.einsum("ij,ij->i", points - cad[nearest], normals[nearest])
    canonical_ids = _exact_row_ids(scan, optimizer)
    return dict(stage=stage, rotation=rotation.tolist(), translation=translation.tolist(),
                selection_role="NEXT_SELECTION_AT_RECORDED_POSE_NOT_PREVIOUS_LAST_STEP",
                selection_gate="COARSE" if coarse else "FINE",
                correspondence_distance_gate_mm=gate, optimizer_scan_count=len(optimizer),
                optimizer_canonical_scan_row_ids=canonical_ids.tolist(),
                point_to_plane_evidence_role="DIAGNOSTIC_ONLY_COARSE_SOLVER_REMAINS_POINT_TO_POINT"
                if coarse else "UNCHANGED_FINE_SOLVER_JACOBIAN_AND_HUBER_WEIGHTS",
                admissible=_subset_evidence(registration, points, normals, nearest, distance,
                    residual, admissible, canonical_ids, policy),
                trimmed=_subset_evidence(registration, points, normals, nearest, distance,
                    residual, trimmed, canonical_ids, policy),
                unchanged_full_diagnostic=registration._diagnose(
                    scan, cad, normals, tree, rotation, translation, policy,
                    optimization_points=refine))


def _trace_inference(scan_points, cad_points, cad_normals, *, policy):
    """Geometry-only candidate execution; no oracle or fixture labels accepted."""
    from backend import cad_guided_global_registration_v1 as registration

    scan = registration._canonical_scan(scan_points)
    cad, normals = registration._canonical_cad(cad_points, cad_normals)
    coarse = registration._spatial_sample(scan, policy.coarse_points)
    refine = registration._spatial_sample(scan, policy.refine_points)
    cad_pca = registration._spatial_sample(cad, policy.refine_points)
    scan_center, scan_basis = registration._pca(coarse)
    cad_center, cad_basis = registration._pca(cad_pca)
    starts = list(registration._starts(coarse, cad_pca))
    assert len(starts) == 24
    rotation, translation = starts[CANDIDATE_INDEX]
    tree = registration.cKDTree(cad)
    snapshots = [_snapshot(registration, scan, coarse, refine, cad, normals, tree,
        rotation, translation, stage=STAGES[0], policy=policy, coarse=True)]
    rotation, translation, nc, coarse_reason = registration._refine(
        coarse, cad, normals, tree, rotation, translation, policy=policy, coarse=True)
    snapshots.append(_snapshot(registration, scan, coarse, refine, cad, normals, tree,
        rotation, translation, stage=STAGES[1], policy=policy, coarse=True))
    snapshots.append(_snapshot(registration, scan, refine, refine, cad, normals, tree,
        rotation, translation, stage=STAGES[2], policy=policy, coarse=False))
    rotation, translation, nf, fine_reason = registration._refine(
        refine, cad, normals, tree, rotation, translation, policy=policy, coarse=False)
    snapshots.append(_snapshot(registration, scan, refine, refine, cad, normals, tree,
        rotation, translation, stage=STAGES[3], policy=policy, coarse=False))
    return dict(candidate_index=CANDIDATE_INDEX, generated_pca_start_count=len(starts),
                executed_candidate_count=1, snapshots=snapshots,
                policy=asdict(policy), scan_unique_rows=len(scan), cad_unique_rows=len(cad),
                coarse_scan_rows=len(coarse), refined_scan_rows=len(refine), cad_pca_rows=len(cad_pca),
                pca=dict(scan_center=scan_center.tolist(), scan_basis_columns=scan_basis.tolist(),
                         cad_center=cad_center.tolist(), cad_basis_columns=cad_basis.tolist()),
                coarse_iterations=nc, refine_iterations=nf,
                coarse_convergence_state=coarse_reason, refine_convergence_state=fine_reason), scan


def _attach_post_inference_oracle(trace, canonical_scan, raw, cad_frame):
    """No result from this function is passed back to the registration algorithm."""
    canonical_to_raw = _exact_row_ids(raw, canonical_scan)
    fixture_caps = set(np.flatnonzero((cad_frame[:, 2] == 0.) | (cad_frame[:, 2] == 30.)).tolist())
    trace["post_inference_fixture_cap_identity"] = dict(
        role="ANALYTIC_FIXTURE_ORACLE_ONLY_NOT_CAD_MATCH_IDENTITY_OR_INFERENCE_INPUT",
        original_raw_row_ids=sorted(fixture_caps), count=len(fixture_caps))
    for snapshot in trace["snapshots"]:
        rotation = np.asarray(snapshot["rotation"])
        translation = np.asarray(snapshot["translation"])
        error = np.linalg.norm(raw @ rotation.T + translation - cad_frame, axis=1)
        snapshot["post_inference_oracle"] = dict(
            role="EVALUATION_ONLY_NOT_INFERENCE_INPUT", evaluated_original_raw_rows=len(raw),
            pose_action_max_mm=float(np.max(error)),
            pose_action_rms_mm=float(np.sqrt(np.mean(error ** 2))))
        snapshot["optimizer_original_raw_row_ids"] = canonical_to_raw[
            snapshot["optimizer_canonical_scan_row_ids"]].tolist()
        snapshot["post_inference_fixture_cap_optimizer_raw_row_ids"] = sorted(
            fixture_caps.intersection(snapshot["optimizer_original_raw_row_ids"]))
        for name in ("admissible", "trimmed"):
            group = snapshot[name]
            group["original_raw_row_ids"] = canonical_to_raw[group["canonical_scan_row_ids"]].tolist()
            group["cap_original_raw_row_ids"] = canonical_to_raw[group["cap_canonical_scan_row_ids"]].tolist()
            group["post_inference_fixture_cap_raw_row_ids"] = sorted(
                fixture_caps.intersection(group["original_raw_row_ids"]))


def _assert_complete_diagnostic(record):
    """Checks receipt integrity, NEVER that the candidate meets accuracy gates."""
    trace = record["trace"]
    assert trace["candidate_index"] == CANDIDATE_INDEX and trace["executed_candidate_count"] == 1
    assert trace["generated_pca_start_count"] == 24
    assert [row["stage"] for row in trace["snapshots"]] == list(STAGES)
    assert trace["policy"]["coarse_iterations"] == trace["policy"]["refine_iterations"] == 20
    assert 0 <= trace["coarse_iterations"] <= 20 and 0 <= trace["refine_iterations"] <= 20
    after_coarse, before_fine = trace["snapshots"][1:3]
    assert after_coarse["rotation"] == before_fine["rotation"]
    assert after_coarse["translation"] == before_fine["translation"]
    for snapshot in trace["snapshots"]:
        assert snapshot["selection_role"] == "NEXT_SELECTION_AT_RECORDED_POSE_NOT_PREVIOUS_LAST_STEP"
        rotation = np.asarray(snapshot["rotation"])
        assert np.linalg.norm(rotation.T @ rotation - np.eye(3)) <= 1e-8
        assert abs(float(np.linalg.det(rotation)) - 1.) <= 1e-8
        assert set(snapshot["trimmed"]["original_raw_row_ids"]) <= set(snapshot["admissible"]["original_raw_row_ids"])
        for group in (snapshot["admissible"], snapshot["trimmed"]):
            assert group["count"] == len(group["original_raw_row_ids"])
            assert sum(group["normal_direction_counts"]["dominant_axis"].values()) == group["count"]
            assert len(group["unweighted"]["normalized_singular_values"]) == 6
            assert len(group["huber_weighted"]["normalized_singular_values"]) == 6
        if snapshot["selection_gate"] == "FINE":
            full = snapshot["unchanged_full_diagnostic"]
            assert snapshot["trimmed"]["count"] == full["fit_subset_count"]
            assert np.allclose(snapshot["trimmed"]["unweighted"]["normalized_singular_values"],
                               full["fit_observability_singular_values"], rtol=1e-12, atol=1e-15)
            assert np.allclose(snapshot["trimmed"]["huber_weighted"]["normalized_singular_values"],
                               full["weighted_fit_observability_singular_values"], rtol=1e-12, atol=1e-15)
    assert record["source_digests_before"] == record["source_digests_after"]
    json.dumps(record, allow_nan=False)


def run_invented_registration_stage_trace_checks():
    """One root-supervised native diagnostic; all failures retain the case slot."""
    record = dict(case=CASE_NAME, passed=False, phase="fixture",
                  pass_semantics="DIAGNOSTIC_COMPLETENESS_NOT_REGISTRATION_SUCCESS")
    try:
        from backend.cad_guided_mesh_offline_v1 import _read_step_owned_snapshot
        from backend.cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from backend.cad_guided_cad_sampling_v1 import sample_native_cad_surface_v1
        from backend.cad_guided_native_features_v1 import extract_ocp_guided_features_v1

        record["source_digests_before"] = _source_digests()
        original, policies = _original(), _policies()
        workflow = policies["workflow_policy"]
        record["policies"] = {name: asdict(value) for name, value in policies.items()}
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
        raw, faces, cad_frame = original._analytic_mesh_scan(False)
        scan_blob = original._ply_bytes(raw, faces)
        record.update(cad_sha256=hashlib.sha256(cad_blob).hexdigest(),
                      scan_sha256=hashlib.sha256(scan_blob).hexdigest(), raw_scan_rows=len(raw))
        record["phase"] = "input_preparation"
        prepared = prepare_mesh_scan_v1(scan_bytes=scan_blob,
                                        max_engine_rows=workflow.max_registration_scan_rows)
        record["input_preparation"] = {key: prepared[key] for key in (
            "status", "reasons", "source_bindings", "coverage")}
        assert prepared["status"] == "AVAILABLE"
        assert np.array_equal(prepared["raw_points"], raw)
        record["phase"] = "native_import_and_sampling"
        shape, record["cad_units"] = _read_step_owned_snapshot(cad_blob)
        # Preserve original workflow call order and shared native handle.
        features = extract_ocp_guided_features_v1(shape, policy=policies["measurement_policy"],
                                                unit_status="DECLARED_MM", allow_ocp=True)
        record["native_features"] = features["metadata"]
        cad = sample_native_cad_surface_v1(shape, allow_ocp=True, native_binding="ocp",
            sample_count=workflow.cad_sample_count,
            linear_deflection_mm=workflow.cad_linear_deflection_mm,
            angular_deflection_radians=workflow.cad_angular_deflection_radians)
        record["cad_surface_sampling"] = {key: value for key, value in cad.items()
                                           if not isinstance(value, np.ndarray)}
        assert cad["status"] == "AVAILABLE"
        assert cad["coverage"]["sample_count"] == CAD_SAMPLE_COUNT
        record["sample_array_sha256"] = {name: hashlib.sha256(
            np.asarray(value, dtype="<f8").tobytes()).hexdigest()
            for name, value in (("scan_points", prepared["points"]),
                                ("cad_points", cad["points"]), ("cad_normals", cad["normals"]))}
        record["phase"] = "fixed_candidate_stage_trace"
        record["trace"], canonical_scan = _trace_inference(
            prepared["points"], cad["points"], cad["normals"], policy=policies["registration_policy"])
        record["phase"] = "post_inference_diagnostics"
        _attach_post_inference_oracle(record["trace"], canonical_scan, raw, cad_frame)
        record["source_digests_after"] = _source_digests()
        _assert_complete_diagnostic(record)
        record.update(passed=True, phase="complete")
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error)[:1000])
    return dict(protocol="CAD_GUIDED_REGISTRATION_STAGE_TRACE_V1", cases=[record], expected_case_count=1,
                predecessor_receipt=PREDECESSOR_RECEIPT, fixed_candidate_index=CANDIDATE_INDEX,
                external_resource_limits=RESOURCE_LIMITS, observed_bend_count=None,
                conformance="INDETERMINATE", production_use_allowed=False,
                evidence_class="INVENTED_DIAGNOSTIC_NOT_ENGINE_SUCCESS_OR_PHYSICAL_GENERALIZATION")


def test_stage_trace_keeps_original_dense_policy_without_native_import(monkeypatch):
    real_import, calls = builtins.__import__, []

    def guard(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise AssertionError("SOURCE_CONTROL_MUST_NOT_IMPORT_NATIVE")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    original = _original()
    before = {name: asdict(value) for name, value in original._policies("v1").items()}
    globals_before = copy.deepcopy((original.CONNECTED_CASES, original.CRITERIA, original.RESOURCE_LIMITS))
    expected = copy.deepcopy(before)
    expected["workflow_policy"]["cad_sample_count"] = 32768
    assert {name: asdict(value) for name, value in _policies().items()} == expected
    assert {name: asdict(value) for name, value in original._policies("v1").items()} == before
    assert (original.CONNECTED_CASES, original.CRITERIA, original.RESOURCE_LIMITS) == globals_before
    assert calls == []


def test_stage_trace_has_fixed_budget_and_single_candidate_only():
    assert RESOURCE_LIMITS == _original().RESOURCE_LIMITS
    assert RESOURCE_LIMITS == {"wall_seconds": 60, "cpu_seconds": 45, "process_tree_rss_bytes": 1024**3}
    assert CANDIDATE_INDEX == 20 and CAD_SAMPLE_COUNT == 32768 and len(STAGES) == 4
    node = next(item for item in ast.parse(Path(__file__).read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name == "_trace_inference")
    calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)
             and isinstance(item.func, ast.Attribute) and item.func.attr == "_refine"]
    assert len(calls) == 2
    assert [next(key.value.value for key in call.keywords if key.arg == "coarse") for call in calls] == [True, False]
    assert not any(isinstance(item, (ast.For, ast.While, ast.AsyncFor)) for item in ast.walk(node))
    assert not any(isinstance(item, ast.Attribute) and item.attr == "register_whole_part_v1"
                   for item in ast.walk(node))


def test_stage_trace_oracle_is_not_inference_input_and_no_runtime_patch():
    tree = ast.parse(Path(__file__).read_text())
    inference = next(item for item in tree.body
                     if isinstance(item, ast.FunctionDef) and item.name == "_trace_inference")
    assert [arg.arg for arg in inference.args.args] == ["scan_points", "cad_points", "cad_normals"]
    assert [arg.arg for arg in inference.args.kwonlyargs] == ["policy"]
    assert not any(isinstance(item, ast.Name) and any(token in item.id for token in ("oracle", "cad_frame", "raw"))
                   for item in ast.walk(inference))
    worker = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                  and item.name == "run_invented_registration_stage_trace_checks")
    named_calls = [item for item in ast.walk(worker) if isinstance(item, ast.Call)
                   and isinstance(item.func, ast.Name)]
    trace_call = next(item for item in named_calls if item.func.id == "_trace_inference")
    oracle_call = next(item for item in named_calls if item.func.id == "_attach_post_inference_oracle")
    assert trace_call.lineno < oracle_call.lineno
    assert not any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                   and item.func.id in ("setattr", "exec", "eval") for item in ast.walk(worker))
    assert not any(isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                   and any(isinstance(target, ast.Attribute) for target in
                           (item.targets if isinstance(item, ast.Assign) else [item.target]))
                   for item in ast.walk(inference))


@pytest.mark.parametrize("last,expected_rank", ((0., 5), (6., 6)))
def test_diagnostic_svd_rank_and_null_vector_have_explicit_scale(last, expected_rank):
    diagonal = np.asarray([1., 2., 3., 4., 5., last])
    evidence = _svd_evidence(np.diag(diagonal))
    assert evidence["rank_at_solver_rcond"] == expected_rank
    assert np.allclose(evidence["normalized_singular_values"], np.sort(diagonal)[::-1] / math.sqrt(6))
    if last == 0.:
        assert evidence["smallest_singular_value_is_float_zero"]
        assert evidence["smallest_right_singular_vector"] == [0., 0., 0., 0., 0., 1.]
    else:
        assert not evidence["smallest_singular_value_is_float_zero"]


def test_fixed_invented_registration_stage_trace():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Root must opt in after the physical job is terminal under the fixed external budget")
    receipt = run_invented_registration_stage_trace_checks()
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    print(json.dumps({"registration_stage_trace_compact": [
        {"case": row["case"], "passed": row["passed"], "phase": row["phase"],
         "error_type": row.get("error_type"), "error": row.get("error"),
         "stages": [{"stage": stage["stage"], "oracle": stage.get("post_inference_oracle"),
                     "admissible_count": stage["admissible"]["count"],
                     "trimmed_count": stage["trimmed"]["count"],
                     "admissible_normals": stage["admissible"]["normal_direction_counts"],
                     "trimmed_normals": stage["trimmed"]["normal_direction_counts"],
                     "admissible_rank": stage["admissible"]["unweighted"]["rank_at_solver_rcond"],
                     "trimmed_rank": stage["trimmed"]["unweighted"]["rank_at_solver_rcond"]}
                    for stage in row.get("trace", {}).get("snapshots", [])]}
        for row in receipt["cases"]]}, sort_keys=True, allow_nan=False), flush=True)
    assert len(receipt["cases"]) == 1
    assert receipt["cases"][0]["passed"], receipt
