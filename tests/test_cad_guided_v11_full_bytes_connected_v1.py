"""Full-bytes real-connected V11 acceptance (no mocked seam): STEP bytes -> REAL owned STEP reader -> REAL native
factory (v1 branch) -> REAL native feature bridge V2 (kernel-V1 -> kernel-V3) -> REAL mesh-PLY scan preparation,
triangle topology and CAD surface sampling -> REAL initializer V16 (completion lane V1) + registration V5 (frozen
V11 CONFIG registration/search/noisy/patch/completion/surface/context policies and budgets; the CPU allowance is
the fixture's own) -> REAL context surface roles -> REAL context measurement V4 / kernel V3 / support cover V5 /
publication validator -> predictions, through the ACTUAL successor dispatch `inspect_cad_guided_mesh_v18`.

Scan: independently constructed from declared constants, never from the CAD tessellation or the extracted feature.
A two-skin bend whose inner surface radius is the NON-NOMINAL 5.06 mm (the CAD nominal is 5.0 mm), skins 2.5 mm
apart, SHEARED (35 deg) cut surfaces at both flange ends inside the feature span (between-skin rows whose incident
mesh triangles are not axis-normal, so the real surface-role filter retains them; the first fixture revision used
square axis-normal cut patches and the real filter removed all 558-567 of them before the kernel, unlike V10's
physical ROIs with zero cap-role exclusions), bounded PER-COORDINATE noise |e_i| <= 0.005 mm (row norm <= sqrt(3) *
0.005 mm) from a seed independent of every deformation, under a declared oblique rigid pose. The strip is symmetric
under S: (x, y, z) -> (y, x, HEIGHT - z) (a proper rotation), so the declared pose truth is the pose family
{truth, S o truth}; registration is bound to that family for EVERY context that feeds an emitted group.

Expectations from the constants only: pose action within 0.5 mm on every raw row (declared after observing 0.085 mm
without and 0.32 mm with the between-skin rows inside the frozen 2 mm correspondence distance; the frozen admission
gate is p95 <= 1.0 mm; the measurement runs on raw rows and is pose-independent within the crop margins), measured
radius within 0.02 mm of 5.06 mm and more than 0.04 mm from the CAD nominal, the carrier line within 0.10 mm / 1 deg
of the transported mid-angle generator and at least 20 mm long (a line-distance / length check, NOT exact finite
endpoint truth), per-role kept / excluded accounting (the sheared rows as UNASSIGNED between-skin rows within the interior/possible
interval of the construction — the recovered context crop is not published row by row, so the sheared identity is
claimed only as: never retained support, and an aggregate count between the rows certainly inside and the rows
possibly inside under the pose bound — nothing outside the sandwich), zero pre-kernel cap-role exclusions, nonempty
retained raw support for every role.

Controls through the same chain with a DECLARED SPATIALLY CONNECTED patch of ORIGINAL +x inner-flange rows
(flange distance 4.5-9.0 mm, height 8-18 mm; background noise unchanged) displaced 0.2 mm: toward the air side the
patch stays in its group (outside-sandwich count == the patch, in whichever role the pose maps it to) and the
unchanged gate refuses NOT_PLANAR without any prediction (each refused context carries its own pose-family
evidence, so the refusal is attributable to the deformation); toward the outer skin it is an exact counted
unverified exclusion (excluded unassigned rows == patch + the sheared interval for that role), coverage
NOT_ESTABLISHED, and no patch row appears in any emitted group's retained raw support (all roles, original IDs).

Attribution: the frozen admission gate (p95 <= 1.0 mm) also admits contexts whose pose acts several millimetres off
at the far rows (observed 4.0-4.5 mm members of the S family); the kernel measures those raw rows too, but their crop
boundaries are not known to the construction, so every exact/interval claim below is asserted only for windows whose
own context pose is within the declared family bound (at least one such window is required), and out-of-bound windows
carry only the invariant accounting (kept + excluded == crop, identities, zero cap exclusions) and are listed in the
receipt as unattributable. Emitted groups are attributed through every member context's pose.

Execution contract (FRESH_PROCESS_ENVELOPE): the V5 CPU guard is ABSOLUTE process CPU, so every case must run in
its own fresh process under the declared wall / CPU / OMP envelope with a 4 GiB RSS BUDGET checked against the final
high-water mark reported by /usr/bin/time -l and ps samples (not continuous enforcement); a case whose registration expired or
omitted candidates by the guard is an execution-envelope failure, never a geometry result, and is reported as such.
Synthetic algorithm evidence only: no physical input, Gold, calibration, sheet identity or manufacturing claim.
"""
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest

from .test_cad_guided_native_feature_bridge_v1_factory_to_measurement_v18 import _invented_bend, _owned_step_bytes
from .test_cad_guided_mesh_connected_native_v1 import _ply_bytes, _oracle_pose

ROOT = Path(__file__).resolve().parents[1]
CAD_RADIUS_MM, SCAN_RADIUS_MM, THICKNESS_MM, HEIGHT_MM, FLANGE_MM = 5.0, 5.06, 2.5, 30., 15.
NOISE_MM = .005                                   # per coordinate; row norm <= sqrt(3) * NOISE_MM
NOISE_SEED = 17                                   # independent of every deformation
DENT_PATCH = dict(flange_distance_mm=(4.5, 9.0), height_mm=(8., 18.))       # connected rectangle of the +x inner flange grid
CRITERIA = dict(pose_action_error_mm=.5, radius_error_mm=.02, nominal_separation_mm=.04, line_distance_mm=.10,
                direction_error_deg=1., minimum_line_length_mm=20.)
FRESH_PROCESS_ENVELOPE = dict(one_case_per_process=True, wall_seconds=1200, cpu_seconds_rlimit=900,
    allowance_cpu_budget_seconds=900., allowance_cpu_guard_seconds=720., rss_budget_bytes=4*1024**3,
    rss_check="FINAL_HWM_FROM_usr_bin_time_l_PLUS_ps_SAMPLES_NOT_CONTINUOUS_ENFORCEMENT",
    omp_num_threads=1, process_cpu_at_start_max_seconds=60.,
    launch="per case: ulimit -t 900; /usr/bin/time -l perl -e 'alarm 1200; exec @ARGV' -- env OMP_NUM_THREADS=1 "
           "PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null "
           "--noconftest -p no:cacheprovider tests/test_cad_guided_v11_full_bytes_connected_v1.py -q -s -k <case>")
SYMMETRY_ROTATION = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])   # S: (x, y, z) -> (y, x, HEIGHT - z)
# Tagged sheared rows per flange crop: 136 under the nominal frame; eroding every crop bound by the 0.5 mm pose bound
# leaves 15 flange columns (d 3.3..11.7) x 3 shear levels (s >= 0.9, z >= 0.63 mm) x 2 ends = 90 rows certainly
# inside; dilating leaves 152 rows possibly inside. The per-coordinate 0.005 mm noise crosses none of these bounds.
SHEARED_NOMINAL, SHEARED_INTERIOR, SHEARED_POSSIBLE = 136, 90, 152


@pytest.fixture(scope="module")
def m():
    saved = {k: v for k, v in sys.modules.items() if k == "backend" or k.startswith("backend.") or k == "scripts" or k.startswith("scripts.")}
    for k in saved:
        del sys.modules[k]
    for name, folder in (("backend", "backend"), ("scripts", "scripts")):
        package = types.ModuleType(name); package.__path__ = [str(ROOT/folder)]; sys.modules[name] = package
    mapping = dict(workflow="cad_guided_mesh_offline_v18", bridge="cad_guided_native_feature_bridge_v2",
        kernel1="cad_guided_measurement", kernel="cad_guided_measurement_v3", context="cad_guided_context_measurement_v4",
        cover="cad_guided_support_cover_measurement_v5", native="cad_guided_native_features",
        native1="cad_guided_native_features_v1", registration1="cad_guided_global_registration_v1",
        instances="cad_guided_instances", noisy="cad_guided_noisy_mesh_planes_v5", patch="cad_guided_mesh_plane_proposals_v4",
        search="cad_guided_plane_search_v6", surface="cad_guided_triangle_surface_v4",
        registration4="cad_guided_global_registration_v4", registration5="cad_guided_global_registration_v5",
        init16="cad_guided_plane_initialization_v16", init12="cad_guided_plane_initialization_v12",
        two_plane="cad_guided_two_plane_context_v4")
    try:
        ns = types.SimpleNamespace(**{name: importlib.import_module("backend."+path) for name, path in mapping.items()})
        spec = importlib.util.spec_from_file_location("_v11_runner_config_for_full_bytes_fixture",
                                                      ROOT/"scripts"/"run_gold11_mesh_development_v11.py")
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        ns.runner = runner
        yield ns
    finally:
        for k in tuple(sys.modules):
            if k == "backend" or k.startswith("backend.") or k == "scripts" or k.startswith("scripts."):
                del sys.modules[k]
        sys.modules.update(saved)


def _grid(rows):
    """A rectangular sample grid (list of rows of xyz) -> (points, triangle faces over that grid)."""
    patch = np.asarray(rows, float)
    first, second = patch.shape[:2]
    faces = []
    for i in range(first-1):
        for j in range(second-1):
            a = i*second+j; b, c, d = a+second, a+1, a+second+1
            faces.extend(((a, b, c), (c, b, d)))
    return patch.reshape((-1, 3)), np.asarray(faces, dtype=np.int64)


def _cad_frame_mesh(*, dent=None):
    """Two-skin bend mesh in the CAD frame from the declared constants -> (points, faces, tags, dent row indices).

    Inner surface at SCAN_RADIUS_MM (arc 0..90 deg about +z, tangent flanges along -y at the +x contact and along
    -x at the +y contact), outer skin at + THICKNESS_MM, sheared (35 deg) cut surfaces at both axial ends of both
    flanges spanning the thickness (between-skin rows inside the feature span whose incident triangles are not
    cap-like). `dent=offset_mm` displaces the DECLARED CONNECTED PATCH of original +x inner-flange rows along +x
    (toward the outer skin when positive, toward the air side when negative). The background noise is drawn from
    NOISE_SEED before any displacement and is identical for every variant."""
    zs = np.linspace(.5, HEIGHT_MM-.5, 30)
    angles = np.linspace(0., math.pi/2, 47)
    ds = np.arange(.3, FLANGE_MM-.3+1e-9, .3)
    patches, tags, offset = [], {}, 0
    def add(rows, tag):
        nonlocal offset
        points, faces = _grid(rows)
        tags.setdefault(tag, []).append(np.arange(offset, offset+len(points)))
        patches.append((points, faces+offset)); offset += len(points)
    for r, skin in ((SCAN_RADIUS_MM, "inner"), (SCAN_RADIUS_MM+THICKNESS_MM, "outer")):
        add([[[r*math.cos(a), r*math.sin(a), z] for z in zs] for a in angles], "arc_"+skin)
        add([[[r, -d, z] for z in zs] for d in ds], "flange_x_"+skin)                             # +x contact flange
        add([[[-d, r, z] for z in zs] for d in ds], "flange_y_"+skin)                             # +y contact flange
    ss, ds_edge, shear = np.array([.45, .9, 1.35, 1.8]), ds[::2], math.tan(math.radians(35.))
    for z_end, sign in ((0., 1.), (HEIGHT_MM, -1.)):
        add([[[SCAN_RADIUS_MM+s, -d, z_end+sign*s*shear] for s in ss] for d in ds_edge], "shear_x")   # sheared cut, +x flange
        add([[[-d, SCAN_RADIUS_MM+s, z_end+sign*s*shear] for s in ss] for d in ds_edge], "shear_y")   # sheared cut, +y flange
    points = np.vstack([p for p, _ in patches]); faces = np.vstack([f for _, f in patches])
    tags = {k: np.concatenate(v) for k, v in tags.items()}
    noise = np.random.default_rng(NOISE_SEED).uniform(-NOISE_MM, NOISE_MM, points.shape)
    flange = tags["flange_x_inner"]
    lo_d, hi_d = DENT_PATCH["flange_distance_mm"]; lo_z, hi_z = DENT_PATCH["height_mm"]
    inside = (-points[flange, 1] >= lo_d) & (-points[flange, 1] <= hi_d) & (points[flange, 2] >= lo_z) & (points[flange, 2] <= hi_z)
    chosen = np.sort(flange[inside])
    if dent is not None:
        points[chosen, 0] += float(dent)
    return points+noise, faces, tags, chosen


def _crop_rows(cad_points, rows, role, margin=0.):
    """How many of `rows` fall in the kernel's flange crop for `role` under the CAD-nominal feature frame, with every
    crop bound moved OUTWARD by `margin` (positive: rows that could enter under a pose error up to `margin`) or INWARD
    (negative: rows that stay inside under any pose error up to |margin|). Exact identity claims are made only for
    rows whose distance to every bound exceeds the declared pose bound; other rows get interval claims."""
    p = cad_points[rows]
    along = -p[:, 1] if role == 0 else -p[:, 0]
    lateral = (p[:, 0] if role == 0 else p[:, 1])-CAD_RADIUS_MM
    return int(np.sum((p[:, 2] >= -margin) & (p[:, 2] <= HEIGHT_MM+margin) & (along >= 2.5-margin) & (along <= 12.5+margin)
                      & (np.abs(lateral) <= 3.+margin)))


def _sheared_bounds(run):
    """[interior, possible] counts of the tagged sheared rows per flange under the declared pose bound: rows farther than
    the bound from every crop boundary are certainly inside; rows within the bound of a boundary may fall either way
    (the actual recovered context crop is not published row by row, so only these bounds are claimed)."""
    bound = CRITERIA["pose_action_error_mm"]
    return [(_crop_rows(run.cad, run.tags[tag], role, -bound), _crop_rows(run.cad, run.tags[tag], role, bound))
            for tag, role in (("shear_x", 0), ("shear_y", 1))]


def _policies(m):
    """The frozen V11 CONFIG policy objects (control arm) with only the fixture's own CPU allowance."""
    C, caps, ctx = m.runner.CONFIG, m.runner.CERTIFICATE_CAPS, m.runner.CONTEXT_V4_REQUEST
    noisy = dict(C["noisy_policy"]); noisy["tile_triangles"] = tuple(noisy["tile_triangles"])
    measurement = dict(C["measurement"]); measurement["flange_window_mm"] = tuple(measurement["flange_window_mm"])
    allowance = dict(C["refinement_allowance"], cpu_budget_seconds=FRESH_PROCESS_ENVELOPE["allowance_cpu_budget_seconds"],
                     cpu_guard_seconds=FRESH_PROCESS_ENVELOPE["allowance_cpu_guard_seconds"])
    return dict(
        registration_policy=m.registration1.WholePartRegistrationPolicyV1(**C["registration"]),
        measurement_policy=m.native.NativeGuidedPolicy(**measurement),
        instance_policy=m.instances.SupportPolicy(**C["support"]), allow_ocp=True,
        position_error_bound_mm=C["position_error_bound_mm"],
        noisy_policy=m.noisy.NoisyMeshPlanePolicyV5(**noisy), noisy_budgets=m.noisy.NoisyMeshPlaneBudgetsV5(**C["noisy_budgets"]),
        cad_patch_policy=m.patch.CoplanarPatchPolicyV4(**C["cad_patch_policy"]),
        cad_patch_budgets=m.patch.CoplanarPatchBudgetsV4(**C["cad_patch_budgets"]),
        search_policy=m.search.PlaneSearchPolicyV6(**C["search_policy"]), search_budgets=m.search.PlaneSearchBudgetsV6(**C["search_budgets"]),
        surface_budgets=m.surface.TriangleSurfaceBudgetsV4(**C["surface_budgets"], max_certificate_pairs=caps["surface_budgets"]),
        registration_budgets=m.registration4.SurfaceRegistrationBudgetsV4(**C["registration_budgets"], max_certificate_pairs=caps["registration_budgets"]),
        workflow_policy=m.workflow.MeshWorkflowPolicyV18(**C["workflow"], native_adapter_version="v1"),
        refinement_allowance=m.registration5.RefinementAllowanceV1(**allowance),
        initialization_budgets=m.init12.PlaneInitializationBudgetsV12(**C["initialization_budgets"], max_certificate_pairs=caps["initialization_budgets"]),
        completion_policy=m.init16.CompletionLanePolicyV1(**C["completion_policy"]),
        completion_budgets=m.init12.PlaneInitializationBudgetsV12(**C["completion_budgets"], max_certificate_pairs=caps["initialization_budgets"]),
        layer_policy=m.kernel.LayerSelectionPolicyV2(**C["layer_policy"]),
        two_plane_enabled=True, connected_measurement_contexts=True, two_plane_context_mode="v4",
        two_plane_context_budgets=m.two_plane.TwoPlaneContextBudgetsV4(**ctx["budgets"]),
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None)


def _fresh_process_guard():
    cpu = time.process_time()
    assert cpu < FRESH_PROCESS_ENVELOPE["process_cpu_at_start_max_seconds"], (
        "FRESH_PROCESS_REQUIRED_THE_V5_CPU_GUARD_IS_ABSOLUTE_PROCESS_CPU: %.1f s already consumed" % cpu)


def _run(m, *, dent=None):
    _fresh_process_guard()
    cad_points, faces, tags, chosen = _cad_frame_mesh(dent=dent)
    rotation, translation = _oracle_pose()
    raw = cad_points @ rotation.T + translation                       # raw = R cad + t  (declared truth)
    scan_bytes = _ply_bytes(raw, faces)
    cad_bytes = _owned_step_bytes(_invented_bend(thickness=THICKNESS_MM, flange_length=FLANGE_MM))
    started = time.process_time(); wall = time.monotonic()
    answer = m.workflow.inspect_cad_guided_mesh_v18(scan_bytes=scan_bytes, cad_bytes=cad_bytes, **_policies(m))
    cpu, elapsed = time.process_time()-started, time.monotonic()-wall
    return types.SimpleNamespace(answer=answer, raw=raw, cad=cad_points, faces=faces, tags=tags, chosen=chosen,
                                 rotation=rotation, translation=translation, cpu_seconds=cpu, wall_seconds=elapsed,
                                 scan_sha256=hashlib.sha256(scan_bytes).hexdigest(), cad_sha256=hashlib.sha256(cad_bytes).hexdigest())


def _envelope_guard(answer):
    """A guard-expired or omitted candidate is an execution-envelope failure of this fixture, never a geometry result."""
    evidence = (answer.get("registration") or {}).get("evidence") or {}
    ledger = evidence.get("candidate_disposition_ledger") or {}
    assert ledger and not ledger.get("expired") and not ledger.get("omitted") and not ledger.get("not_attempted"), (
        "EXECUTION_ENVELOPE_CPU_GUARD_OR_QUOTA_TRIPPED_NOT_A_GEOMETRY_RESULT", ledger, answer.get("phase"), answer.get("reasons"))


def _line_distance(points, start, end):
    direction = (end-start)/np.linalg.norm(end-start)
    rel = points-start
    return np.linalg.norm(rel-np.outer(rel @ direction, direction), axis=1)


def _pose_action_error(run, candidate):
    """max over ALL raw rows of |R raw + t - truth| for the declared pose family {truth, S o truth} (mm)."""
    R, t = np.asarray(candidate["rotation"], float), np.asarray(candidate["translation"], float)
    estimate = run.raw @ R.T + t
    symmetric = run.cad @ SYMMETRY_ROTATION.T + np.array([0., 0., HEIGHT_MM])
    plain = float(np.linalg.norm(estimate-run.cad, axis=1).max())
    mirrored = float(np.linalg.norm(estimate-symmetric, axis=1).max())
    return min(plain, mirrored), ("TRUTH" if plain <= mirrored else "SYMMETRIC_TRUTH")


def _windows(answer):
    return answer["measurements"]["source_context_result"]["context_windows"]


def _window_pose_error(run, window):
    """The pose-family action error of the context that produced `window` (attribution evidence for its accounting)."""
    candidates = {c["index"]: c for c in run.answer["registration"]["evidence"]["candidates"]}
    return _pose_action_error(run, candidates[window["context_index"]])


def _measured_groups(answer):
    return [g for g in answer["measurements"]["support_groups"] if g["status"] == "MEASURED_CONDITIONAL"]


def _group_contexts(answer, group):
    """Context windows (and their candidates) that feed one emitted group, by member record id."""
    windows = [w for w in _windows(answer) if w["support_record_id"] in set(group["member_record_ids"])]
    assert windows, group["member_record_ids"]
    candidates = {c["index"]: c for c in answer["registration"]["evidence"]["candidates"]}
    return [(w, candidates[w["context_index"]]) for w in windows]


def _assert_group_pose_and_support(run, group, *, forbidden_rows=()):
    """Every context feeding an emitted group is a pose of the declared family within the bound; the group's retained
    raw support is nonempty for the arc and BOTH flange roles and (by original row id) disjoint from `forbidden_rows`."""
    errors = []
    for window, candidate in _group_contexts(run.answer, group):
        assert (candidate.get("conditional_context_support") or {}).get("status") == "SUPPORTED_CONTEXT"
        error, which = _pose_action_error(run, candidate)
        errors.append((window["context_index"], round(error, 4), which))
        assert error <= CRITERIA["pose_action_error_mm"], errors
    support = group["raw_support"]
    flange_rows = [np.asarray(rows, dtype=np.int64) for rows in support["flange_rows"]]
    arc_rows = np.asarray(support["arc_rows"], dtype=np.int64)
    assert len(flange_rows) == 2 and all(len(rows) > 100 for rows in flange_rows) and len(arc_rows) > 100
    forbidden = set(np.asarray(forbidden_rows, dtype=np.int64).tolist())
    sheared = set(run.tags["shear_x"].tolist()) | set(run.tags["shear_y"].tolist())
    for rows in (*flange_rows, arc_rows):
        assert rows.min() >= 0 and rows.max() < len(run.raw) and len(np.unique(rows)) == len(rows)
        assert not (set(rows.tolist()) & forbidden)
        assert not (set(rows.tolist()) & sheared)                     # a tagged sheared row is never retained support
    return errors, [int(len(rows)) for rows in flange_rows], int(len(arc_rows))


def _receipt(name, run, **extra):
    print(json.dumps(dict(receipt=name, fixture_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scan_sha256=run.scan_sha256, cad_sha256=run.cad_sha256, raw_rows=int(len(run.raw)), faces=int(len(run.faces)),
        cpu_seconds=round(run.cpu_seconds, 2), wall_seconds=round(run.wall_seconds, 2),
        process_cpu_seconds_total=round(time.process_time(), 2), envelope=FRESH_PROCESS_ENVELOPE, **extra), sort_keys=True), flush=True)


def test_full_bytes_real_chain_measures_the_non_nominal_scan_radius_and_transports_the_carrier(m):
    run = _run(m)
    answer = run.answer
    _envelope_guard(answer)
    assert answer["status"] == "PARTIAL" and answer["phase"] == "COMPLETE", (answer["phase"], answer["reasons"])
    assert answer["implementation"] == m.workflow.WORKFLOW_IMPLEMENTATION and "NATIVE_BRIDGE_V2" in answer["implementation"]
    assert answer["cad_sha256"] == run.cad_sha256
    bridge = answer["native_feature_bridge"]                          # native-to-V3 type identity through the REAL bridge
    assert bridge["feature_count"] == 1 and "KERNEL_V1_TO_KERNEL_V3" in json.dumps(bridge)
    assert answer["kernel_implementation"] == m.workflow.KERNEL_IMPLEMENTATION
    assert answer["registration"]["transform_scan_to_cad"] is None          # conditional contexts, never an accepted pose
    groups = _measured_groups(answer)
    assert groups and answer["predictions"] and len(answer["predictions"]) == len(groups)
    pose_receipt, support_receipt = [], []
    for group in groups:
        errors, flange_counts, arc_count = _assert_group_pose_and_support(run, group)
        pose_receipt.append(errors); support_receipt.append(dict(flange_rows=flange_counts, arc_rows=arc_count))
        radius = group["measurements"]["measured_radius_mm"]
        assert abs(radius-SCAN_RADIUS_MM) <= CRITERIA["radius_error_mm"] and abs(radius-CAD_RADIUS_MM) > CRITERIA["nominal_separation_mm"], radius
        assert abs(abs(group["measurements"]["measured_sweep_deg"])-90.) < .5
        assert group["inner_face_claim_eligible"] is False and group["measurements"]["physical_trim_endpoints"] is None
    assert answer["conformance"] == "INDETERMINATE" and answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
    # Carrier transport: the constructed mid-angle generator (radius 5.06, 45 deg, along the axis) in the raw frame; the
    # generator is invariant under S. Line distance / direction / length, not exact finite endpoint truth.
    expected = np.array([[SCAN_RADIUS_MM/math.sqrt(2), SCAN_RADIUS_MM/math.sqrt(2), z] for z in (0., HEIGHT_MM)]) @ run.rotation.T + run.translation
    line_errors = []
    for prediction in answer["predictions"]:
        endpoints = np.array([prediction["line_start"], prediction["line_end"]], float)
        line_errors.append(float(_line_distance(endpoints, expected[0], expected[1]).max()))
        direction, want = endpoints[1]-endpoints[0], expected[1]-expected[0]
        angle = math.degrees(math.acos(min(1., abs(direction @ want)/np.linalg.norm(direction)/np.linalg.norm(want))))
        assert angle <= CRITERIA["direction_error_deg"] and np.linalg.norm(direction) >= CRITERIA["minimum_line_length_mm"]
        assert prediction["carrier_role"] == m.kernel.GENERIC_CARRIER_ROLE and prediction["physical_trim_endpoints"] is None
    assert max(line_errors) <= CRITERIA["line_distance_mm"], line_errors
    # Exact retained raw-row accounting on every measured window: both skins separated, the sheared cut rows excluded as
    # UNASSIGNED between-skin rows (exactly the constructed count per role), nothing outside the sandwich, every crop row
    # accounted, zero pre-kernel cap-role exclusions.
    bounds = _sheared_bounds(run)                                     # the two flanges are constructed alike
    assert bounds[0] == bounds[1] and bounds[0] == (SHEARED_INTERIOR, SHEARED_POSSIBLE)
    interior, possible = bounds[0]
    measured_windows = [w for w in _windows(answer) if w["status"] == "MEASURED_CONTEXT"]
    assert measured_windows
    accounting, attributable = [], 0
    for window in measured_windows:
        error, which = _window_pose_error(run, window)                # this window's own attribution evidence
        in_bound = error <= CRITERIA["pose_action_error_mm"]
        attributable += int(in_bound)
        assert window["excluded_cap_like_roi_rows"] == 0
        support = window["diagnostics"]["flange_support"]
        assert support["semantics"].endswith("COVERAGE_NOT_ESTABLISHED")
        for role in support["per_role"]:
            assert role["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
            assert role["kept_rows"]+role["excluded_opposite_skin_rows"]+role["excluded_bridging_rows"] == role["crop_rows"]
            assert role["excluded_opposite_skin_rows"] > 0 and role["outside_sandwich_member_rows"] == 0
            if in_bound:
                assert interior <= role["excluded_bridging_rows"] <= possible, (window["context_index"], role)
        for flange in window["diagnostics"]["flanges"]:
            selection = flange["layer_selection"]
            assert selection["refit_converged"] is True and abs(selection["observed_layer_gap_mm"]-THICKNESS_MM) < .05
        accounting.append(dict(context=window["context_index"], pose_family_error_mm=round(error, 4), pose_family_member=which,
                               attributable=in_bound, pre_role_filter_roi_rows=window["pre_role_filter_roi_rows"],
                               local_input_rows=window["local_input_rows"], per_role=support["per_role"]))
    assert attributable >= 1, accounting
    assert answer["support_cover_publication_validation"]["cover_plan_revalidated"] is True
    _receipt("V11_FULL_BYTES_REAL_CONNECTED_POSITIVE", run, supported_candidates=len([c for c in answer["registration"]["evidence"]["candidates"]
             if (c.get("conditional_context_support") or {}).get("status") == "SUPPORTED_CONTEXT"]),
             group_context_pose_errors=pose_receipt, retained_support=support_receipt,
             measured_radius_mm=[g["measurements"]["measured_radius_mm"] for g in groups], line_distance_mm=line_errors,
             predictions=len(answer["predictions"]), measured_windows=len(measured_windows), attributable_windows=attributable,
             windows=len(_windows(answer)), accounting=accounting)


def test_full_bytes_real_chain_refuses_an_air_side_dent_of_original_rows_without_trimming(m):
    """Every context within the declared pose-family bound refuses the dented flange NOT_PLANAR with the whole patch
    kept in its group (outside-sandwich count == patch in the role the pose maps it to, 0 in the other) and the
    sheared interval intact; no prediction can come from an attributable context. Out-of-bound contexts (admitted by
    the frozen p95 gate) are reported, never attributed: a refusal there is listed, a measurement there is listed as
    unattributable, and any emitted group must consist of such contexts only."""
    run = _run(m, dent=-.2)
    answer = run.answer
    _envelope_guard(answer)
    assert answer["conformance"] == "INDETERMINATE" and answer["phase"] in ("COMPLETE", "MEASUREMENT_REFUSAL"), (answer["phase"], answer["reasons"])
    windows = _windows(answer)
    assert windows
    patch = int(len(run.chosen)); bound = CRITERIA["pose_action_error_mm"]
    assert patch > 100 and _crop_rows(run.cad, run.chosen, 0, -bound) == patch          # the patch is interior under any in-bound pose
    assert _sheared_bounds(run)[0] == (SHEARED_INTERIOR, SHEARED_POSSIBLE)
    interior, possible = SHEARED_INTERIOR, SHEARED_POSSIBLE
    accounting, attributable = [], 0
    for window in windows:
        error, which = _window_pose_error(run, window)
        in_bound = error <= bound
        assert window["excluded_cap_like_roi_rows"] == 0
        per_role = window["diagnostics"]["flange_support"]["per_role"]
        for role in per_role:
            assert role["kept_rows"]+role["excluded_opposite_skin_rows"]+role["excluded_bridging_rows"] == role["crop_rows"]
        entry = dict(context=window["context_index"], pose_family_error_mm=round(error, 4), pose_family_member=which,
                     attributable=in_bound, status=window["status"], reasons=window["reasons"][:2],
                     failure_phase=window["diagnostics"].get("failure_phase"), per_role=per_role)
        if in_bound:
            attributable += 1
            assert window["status"] != "MEASURED_CONTEXT" and window["reasons"][0] == "FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL", entry
            phase = window["diagnostics"]["failure_phase"]
            assert phase in ("PLANE_FIT_ROLE_0", "PLANE_FIT_ROLE_1")                  # the pose family maps the patch to either role
            dented_role = int(phase[-1])
            for role in per_role:
                assert role["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
                assert interior <= role["excluded_bridging_rows"] <= possible, (window["context_index"], role)
                assert role["outside_sandwich_member_rows"] == (patch if role["role"] == dented_role else 0), (window["context_index"], role)
            selection = window["diagnostics"]["flanges"][dented_role]["layer_selection"]
            assert selection["refit_converged"] is True and selection["kept_rows"] == max(selection["group_rows"])
            entry["dented_role"] = dented_role
        accounting.append(entry)
    assert attributable >= 1, accounting
    in_bound_contexts = {row["context"] for row in accounting if row["attributable"]}
    for group in _measured_groups(answer):                            # only unattributable contexts may ever emit here
        contexts = {w["context_index"] for w, _ in _group_contexts(answer, group)}
        assert not (contexts & in_bound_contexts), (contexts, in_bound_contexts)
    _receipt("V11_FULL_BYTES_REAL_CONNECTED_AIR_SIDE_DENT", run, windows=len(windows), attributable_windows=attributable,
             predictions_from_unattributable_contexts=len(answer["predictions"]), patch_rows=patch, accounting=accounting)


def test_full_bytes_real_chain_publishes_a_between_skin_dent_as_an_exact_unverified_exclusion(m):
    """Window level (the changed mechanism, asserted unconditionally on EVERY context window): the connected patch
    displaced between the skins is an EXACT counted unverified exclusion on top of the sheared interval, coverage
    NOT_ESTABLISHED, no patch row retained. Cover level: the whole-component raw-support cover rule of cover V5
    (`NARY_EXISTING_EXACT_RAW_SUPPORT_COVER`) either finds a member whose retained support covers the union of the
    pose family's contexts (predictions; retained raw support disjoint from the patch and the sheared rows) or
    abstains `NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER` after the cross-context arc-ownership rule invalidated
    every record (the declared, retained negative outcome observed in the diagnostic: eight pose-family contexts whose
    crop boundaries differ by more than one member's support). Both branches carry full accounting; the receipt
    names the branch. Neither branch forces a prediction or weakens a gate."""
    run = _run(m, dent=.2)
    answer = run.answer
    _envelope_guard(answer)
    assert answer["phase"] == "COMPLETE" and answer["conformance"] == "INDETERMINATE", (answer["phase"], answer["reasons"])
    patch = int(len(run.chosen)); bound = CRITERIA["pose_action_error_mm"]
    assert patch > 100 and _crop_rows(run.cad, run.chosen, 0, -bound) == patch
    assert _sheared_bounds(run)[0] == (SHEARED_INTERIOR, SHEARED_POSSIBLE)
    interior, possible = SHEARED_INTERIOR, SHEARED_POSSIBLE
    windows = _windows(answer)
    assert windows
    accounting, attributable = [], 0
    for window in windows:
        error, which = _window_pose_error(run, window)
        in_bound = error <= bound
        assert window["excluded_cap_like_roi_rows"] == 0
        support = window["diagnostics"]["flange_support"]
        for role in support["per_role"]:
            assert role["kept_rows"]+role["excluded_opposite_skin_rows"]+role["excluded_bridging_rows"] == role["crop_rows"]
        entry = dict(context=window["context_index"], pose_family_error_mm=round(error, 4), pose_family_member=which,
                     attributable=in_bound, status=window["status"], reasons=window["reasons"][:2], per_role=support["per_role"])
        if in_bound:
            attributable += 1
            assert window["status"] == "MEASURED_CONTEXT", entry
            assert support["semantics"].endswith("COVERAGE_NOT_ESTABLISHED")
            excluded = sorted(role["excluded_bridging_rows"] for role in support["per_role"])
            # The patch is an EXACT counted exclusion on top of the sheared rows of its role; the sheared rows themselves
            # carry only the interior/possible interval claim.
            assert interior <= excluded[0] <= possible and interior+patch <= excluded[1] <= possible+patch, (excluded, interior, possible, patch)
            for role in support["per_role"]:
                assert role["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
                assert role["outside_sandwich_member_rows"] == 0
            for flange in window["diagnostics"]["flanges"]:
                assert flange["layer_selection"]["refit_converged"] is True and flange["after_selection"]["status"] == "PASS"
        accounting.append(entry)
    assert attributable >= 1, accounting
    measured_windows = [w for w in windows if w["status"] == "MEASURED_CONTEXT"]
    # No patch row and no tagged sheared row is retained support in ANY context record (published whether or not the
    # record survives the cross-context ownership rule).
    source = answer["measurements"]["source_context_result"]
    forbidden = set(run.chosen.tolist()); sheared_tags = set(run.tags["shear_x"].tolist()) | set(run.tags["shear_y"].tolist())
    assert source["support_records"]
    for record in source["support_records"]:
        rows = set(record["raw_support"]["arc_rows"])
        for ids in record["raw_support"]["flange_rows"]:
            rows |= set(ids)
        assert rows and not (rows & forbidden) and not (rows & sheared_tags)
    groups = _measured_groups(answer)
    if groups:
        branch = "PREDICTIONS_WITH_A_WHOLE_COMPONENT_COVER"
        assert answer["status"] == "PARTIAL" and len(answer["predictions"]) == len(groups)
        pose_receipt, support_receipt = [], []
        for group in groups:
            errors, flange_counts, arc_count = _assert_group_pose_and_support(run, group, forbidden_rows=run.chosen)
            pose_receipt.append(errors); support_receipt.append(dict(flange_rows=flange_counts, arc_rows=arc_count))
            assert abs(group["measurements"]["measured_radius_mm"]-SCAN_RADIUS_MM) <= CRITERIA["radius_error_mm"]
            assert group["measurements"]["physical_trim_endpoints"] is None and group["inner_face_claim_eligible"] is False
        cover = dict(group_context_pose_errors=pose_receipt, retained_support=support_receipt)
    else:
        branch = "ABSTAINED_NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER"
        assert answer["status"] == "UNAVAILABLE" and answer["predictions"] == []
        assert answer["reasons"] == ["NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS"]
        assert source["reasons"] == ["NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS"]
        assert all(r["status"] == "UNAVAILABLE" and r["reasons"] == ["AMBIGUOUS_SHARED_RAW_OR_DUPLICATED_POSITION_SUPPORT"]
                   for r in source["support_records"])
        plan_groups = answer["measurements"]["support_cover_plan"]["groups"]
        assert plan_groups and all(g["status"] == "UNAVAILABLE" and g["reasons"] == ["NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER"]
                                   for g in plan_groups)
        # The abstention is verified from the published supports with the planner's own structure: ONE component
        # holds every record, and every member's ARC support misses rows of that component's arc union, so no member
        # can be a whole-component cover whatever flange permutation the planner tries (arc failure alone suffices;
        # no flange-permutation oracle is needed or claimed).
        records = source["support_records"]
        assert len(plan_groups) == 1 and sorted(plan_groups[0]["member_record_ids"]) == sorted(r["record_id"] for r in records)
        union_arc = set().union(*[set(r["raw_support"]["arc_rows"]) for r in records])
        arc_missing = [len(union_arc - set(r["raw_support"]["arc_rows"])) for r in records]
        assert len(records) >= 2 and all(missing > 0 for missing in arc_missing), arc_missing
        cover = dict(records=len(records), arc_union=len(union_arc), arc_missing_per_record=arc_missing,
                     record_supports=[[len(x) for x in r["raw_support"]["flange_rows"]]+[len(r["raw_support"]["arc_rows"])] for r in records])
    _receipt("V11_FULL_BYTES_REAL_CONNECTED_BETWEEN_SKIN_DENT", run, branch=branch, patch_rows=patch, measured_windows=len(measured_windows),
             attributable_windows=attributable, windows=len(windows), accounting=accounting, cover=cover)
