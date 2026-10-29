"""Additive fine-update experiment after the retained V2 connected failure.

Coarse V2, PCA starts, global trim and final observability/acceptance stay outside
this helper and unchanged. This is not a public registration or measured-pose
entrypoint. It accepts the current iterate only from its registration caller.

FACE interiors use a local point-to-plane linearization. Finite EDGE/VERTEX rows
use a fixed-nearest-point vector surrogate, not an infinite facet plane or an
exact edge projector. A single Euclidean Huber weight belongs to each physical
point, shared by all three boundary residual components. The expanded optimizer
matrix is NEVER an observability or uncertainty certificate.

Every trial re-queries finite surface distances on the SAME retained point IDs;
it cannot improve by dropping/retrimming/gating difficult trial points. Actual
distance acceptance is needed because a FACE plane is only a local model. This
gives conditional same-set descent within floating-point slack, not a global
convergence, correct-pose, acquisition-side or calibration guarantee.
"""

import math

import numpy as np

from . import cad_guided_global_registration_v1 as _v1
from . import cad_guided_global_registration_v2 as _v2


def _mixed_system(points, targets, normals, regions, huber_delta):
    """Return weighted A,b, center,L and one weight per actual point.

    Solved coordinates are [L*omega, center_displacement], never scale.
    FACE contributes one scalar equation; boundary contributes three coupled
    equations with the same robust weight. No division by expanded row count.
    """
    if (points.ndim != 2 or points.shape[1] != 3 or not 36 <= len(points) <= 20000
            or targets.shape != points.shape or normals.shape != points.shape
            or regions.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(targets).all()
            or not np.isfinite(normals).all()
            or not np.isin(regions, ("FACE", "EDGE", "VERTEX")).all()
            or not math.isfinite(huber_delta) or huber_delta <= 0
            or not np.allclose(_v2._norm(normals), 1., rtol=0, atol=1e-8)):
        raise ValueError("INVALID_MIXED_FINITE_LINEARIZATION")
    center = np.mean(points, axis=0)
    centered = points - center
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if not math.isfinite(scale) or scale <= 1e-10:
        raise ValueError("DEGENERATE_WHOLE_PART_EXTENT")
    error = points - targets
    distance = _v2._norm(error)
    weights = np.minimum(1., huber_delta / np.maximum(distance, 1e-15))
    root = np.sqrt(weights)
    face = regions == "FACE"
    face_j = np.column_stack((np.cross(centered[face], normals[face])/scale, normals[face]))
    face_r = np.einsum("ij,ij->i", error[face], normals[face])
    basis = np.broadcast_to(np.eye(3), (np.count_nonzero(~face), 3, 3))
    boundary_j = np.concatenate((np.cross(centered[~face, None, :], basis)/scale, basis), axis=2)
    matrix = np.concatenate((face_j*root[face, None],
                             (boundary_j*root[~face, None, None]).reshape(-1, 6)))
    rhs = -np.concatenate((face_r*root[face], (error[~face]*root[~face, None]).reshape(-1)))
    return matrix, rhs, center, scale, weights


def refine_finite_surface_v3(scan, query_surface, rotation, translation, *, policy):
    """One bounded fine stage; return the V2 five-field refinement contract.

    Fatal finite-query/budget failures propagate to the registration owner. The
    query object must include every trial in its existing cumulative budgets.
    Final registration diagnostics must run separately; this function cannot
    select a pose or publish a measurement. Coarse fitting is not called here.
    """
    if type(policy) is not _v1.WholePartRegistrationPolicyV1:
        raise ValueError("INVALID_WHOLE_PART_REGISTRATION_POLICY")
    # Recheck a bypass-mutated frozen policy before any query or iteration.
    from dataclasses import asdict
    _v1.WholePartRegistrationPolicyV1(**asdict(policy))
    scan = _v1._positions(scan, 20000)
    r, t = np.asarray(rotation, dtype=float).copy(), np.asarray(translation, dtype=float).copy()
    if (r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all()
            or not np.allclose(r.T@r, np.eye(3), rtol=0, atol=1e-8)
            or abs(float(np.linalg.det(r))-1.) > 1e-8):
        raise ValueError("INVALID_FINE_STAGE_RIGID_ITERATE")
    last = {}
    for iteration in range(policy.refine_iterations):
        transformed = scan@r.T+t
        query = query_surface(transformed)
        chosen = _v2._chosen(query, policy.max_correspondence_distance_mm, policy.trim_fraction)
        last = dict(admissible_rows=int(np.count_nonzero(
            (query["distances"] <= policy.max_correspondence_distance_mm) & _v2._unambiguous(query))),
            retained_rows=len(chosen), ambiguous_rows=int(np.count_nonzero(~_v2._unambiguous(query))),
            boundary_rows=int(np.count_nonzero(query["closest_regions"] != "FACE")),
            update_model="FACE_NORMAL_BOUNDARY_FIXED_POINT_GAUSS_NEWTON_V3",
            line_search="ACTUAL_FINITE_DISTANCE_SAME_RETAINED_ROWS_NO_RESELECTION",
            boundary_optimizer_rank_is_observability=False)
        if len(chosen) < 36:
            return r, t, iteration, "INSUFFICIENT_REFINEMENT_SUPPORT", last
        p, q = transformed[chosen], query["closest_points"][chosen]
        matrix, rhs, center, scale, weights = _mixed_system(
            p, q, query["oriented_normals"][chosen], query["closest_regions"][chosen], policy.huber_delta_mm)
        step = np.linalg.lstsq(matrix, rhs, rcond=1e-10)[0]
        if not np.isfinite(step).all():
            raise ValueError("NONFINITE_FINITE_GAUSS_NEWTON_STEP")
        omega, displacement = step[:3]/scale, step[3:]
        angle, shift = float(_v2._norm(omega)), float(_v2._norm(displacement))
        factor = min(1., policy.max_rotation_step_radians/max(angle, 1e-15),
                     policy.max_translation_step_mm/max(shift, 1e-15))
        previous = _v2._huber_loss(query["distances"][chosen], policy.huber_delta_mm)
        accepted, nr, nt = False, r, t
        for backtrack in range(8):
            fraction = factor*.5**backtrack
            increment = _v1._rodrigues(fraction*omega)
            trial_r = increment@r
            trial_t = increment@(t-center)+center+fraction*displacement
            trial = scan[chosen]@trial_r.T+trial_t
            trial_query = query_surface(trial)
            # Ambiguity does not delete a row from the objective. It is handled
            # in the next selection and the unchanged final pose-admission gate.
            loss = _v2._huber_loss(trial_query["distances"], policy.huber_delta_mm)
            if loss <= previous+1e-14:
                nr, nt, accepted = trial_r, trial_t, True
                last.update(fixed_rows_finite_huber_before=previous, fixed_rows_finite_huber_after=loss,
                    trial_query_rows=len(trial), expanded_linear_equations=len(rhs),
                    huber_weight_min=float(weights.min()), update_center=center.tolist(),
                    update_fraction=fraction, applied_rotation_radians=fraction*angle,
                    applied_center_displacement_mm=fraction*shift, backtracking_trials=backtrack+1)
                break
        if not accepted:
            return r, t, iteration+1, "NO_IMPROVING_FINITE_SURFACE_STEP", last
        motion = float(np.max(_v2._norm(scan@(nr-r).T+nt-t)))
        r, t = nr, nt
        if motion <= 1e-8:
            return r, t, iteration+1, "STATIONARY", last
    return r, t, policy.refine_iterations, "ITERATION_LIMIT", last
