"""V11 multi-scan healing seam with a bounded provisional-pose gate.

The terminal V11 run sometimes produced a finite registration candidate but no
accepted measurement context.  This successor may use that candidate only after
an independent, label-free gate on the candidate's own finite-distance evidence:
proper rigid transform, p95 proximity no worse than the retained 2 mm proximity
budget, and at least 55% represented scan coverage.  Accepted contexts keep
their original identity; rejected contexts are explicitly labelled provisional
and never become a verified datum.

Native STEP axes remain topology proposals.  Every emitted line is trimmed from
raw scan support by the V1 recovery kernel.  CAD nominal radii/sweeps are
reference metadata only.  Gold, expected counts, and case identities are not
accepted by this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cad_guided_v11_topology_recovery_v1 import (
    SUPPORT_RADIUS_MM,
    _duplicate,
    _finite_line,
    _raw_from_cad,
    recover_scan_supported_axes_v1,
)


IMPLEMENTATION = "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING"
MAX_CANDIDATES = 64
MAX_PROVISIONAL_P95_MM = 2.0
MIN_PROVISIONAL_COVERAGE = 0.55


def _proper_rigid(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(candidate.get("rotation"), dtype=np.float64)
    translation = np.asarray(candidate.get("translation"), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("PROVISIONAL_POSE_SHAPE")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("PROVISIONAL_POSE_NONFINITE")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("PROVISIONAL_POSE_NOT_RIGID")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("PROVISIONAL_POSE_REFLECTION")
    return rotation, translation


def _finite_context_quality(candidate: dict[str, Any]) -> dict[str, float] | None:
    support = candidate.get("conditional_context_support")
    metrics = support.get("finite_distance_metrics") if isinstance(support, dict) else None
    coverage = support.get("geometric_scan_coverage_lower") if isinstance(support, dict) else None
    if not isinstance(metrics, dict) or not isinstance(coverage, (int, float)):
        return None
    p95 = metrics.get("p95_upper")
    rms = metrics.get("rms_upper")
    if not isinstance(p95, (int, float)) or not isinstance(rms, (int, float)):
        return None
    values = (float(p95), float(rms), float(coverage))
    if not np.isfinite(values).all():
        return None
    return {"p95_upper_mm": values[0], "rms_upper_mm": values[1],
            "coverage_lower": values[2]}


def select_pose_v2(registration_evidence: dict[str, Any]) -> dict[str, Any]:
    """Select one retained pose or a gated provisional pose deterministically.

    The selected candidate is ordered by p95 distance, RMS, descending
    coverage, then its published index.  No reference-line or count data enters
    this decision.  A rejected V11 context is never relabelled as accepted.
    """
    if not isinstance(registration_evidence, dict):
        return {"status": "UNAVAILABLE", "reason": "REGISTRATION_EVIDENCE_UNAVAILABLE"}
    candidates = registration_evidence.get("candidates")
    indices = registration_evidence.get("measurement_context_indices")
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES:
        return {"status": "UNAVAILABLE", "reason": "REGISTRATION_CANDIDATE_BUDGET"}
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        return {"status": "UNAVAILABLE", "reason": "REGISTRATION_CONTEXT_INDEX_LEDGER"}

    # A context that V11 itself admitted has priority and preserves its exact
    # published pose.  The same rigid/finite checks still apply.
    if any(index < 0 or index >= len(candidates) for index in indices):
        return {"status": "UNAVAILABLE", "reason": "REGISTRATION_CONTEXT_INDEX_RANGE"}
    accepted = list(indices)
    if accepted:
        index = accepted[0]
        candidate = candidates[index]
        try:
            rotation, translation = _proper_rigid(candidate)
        except ValueError as error:
            return {"status": "UNAVAILABLE", "reason": str(error)}
        quality = _finite_context_quality(candidate)
        return {
            "status": "AVAILABLE",
            "pose_source": "RETAINED_V11_ACCEPTED_CONTEXT",
            "registration_context_accepted": True,
            "candidate_index": index,
            "rotation": rotation,
            "translation": translation,
            "quality": quality,
        }

    provisional = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        quality = _finite_context_quality(candidate)
        if quality is None:
            continue
        if (quality["p95_upper_mm"] > MAX_PROVISIONAL_P95_MM
                or quality["coverage_lower"] < MIN_PROVISIONAL_COVERAGE):
            continue
        try:
            rotation, translation = _proper_rigid(candidate)
        except ValueError:
            continue
        provisional.append((quality["p95_upper_mm"], quality["rms_upper_mm"],
                            -quality["coverage_lower"], index, rotation, translation, quality))
    if not provisional:
        return {
            "status": "UNAVAILABLE",
            "reason": "NO_GATED_PROVISIONAL_V11_POSE",
            "gate": {"max_p95_upper_mm": MAX_PROVISIONAL_P95_MM,
                     "min_coverage_lower": MIN_PROVISIONAL_COVERAGE},
        }
    p95, rms, neg_coverage, index, rotation, translation, quality = min(
        provisional, key=lambda row: row[:4])
    return {
        "status": "AVAILABLE",
        "pose_source": "RETAINED_V11_REJECTED_CONTEXT_GATED_PROVISIONAL",
        "registration_context_accepted": False,
        "candidate_index": index,
        "rotation": rotation,
        "translation": translation,
        "quality": quality,
        "gate": {"max_p95_upper_mm": MAX_PROVISIONAL_P95_MM,
                 "min_coverage_lower": MIN_PROVISIONAL_COVERAGE},
    }


def recover_scan_supported_axes_v2(
    scan_points: np.ndarray,
    cad_path,
    registration_evidence: dict[str, Any],
    *,
    support_radius_mm: float = SUPPORT_RADIUS_MM,
) -> dict[str, Any]:
    """Recover finite scan-supported native axes using a gated retained pose."""
    selection = select_pose_v2(registration_evidence)
    base = {
        "schema": "CAD_GUIDED_V11_TOPOLOGY_RECOVERY_V2",
        "implementation": IMPLEMENTATION,
        "research_only": True,
        "gold_or_expected_count_used": False,
        "observed_bend_count": None,
        "observed_bend_count_status": "MANUFACTURING_EVENT_IDENTITY_UNKNOWN",
        "pose_selection": {key: value for key, value in selection.items()
                           if key not in {"rotation", "translation"}},
        "prediction_geometry_frame": "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
    }
    if selection["status"] != "AVAILABLE":
        return {**base, "status": "UNAVAILABLE", "predictions": [], "records": [],
                "accepted_count": 0, "native_axis_count": None,
                "reason": selection["reason"]}
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = selection["rotation"]
    matrix[:3, 3] = selection["translation"]
    recovered = recover_scan_supported_axes_v1(
        scan_points, cad_path, matrix, support_radius_mm=support_radius_mm)
    return {
        **base,
        **recovered,
        "implementation": IMPLEMENTATION,
        "pose_selection": base["pose_selection"],
        "prediction_semantics": (
            "SCAN_SUPPORTED_CONDITIONAL_CONTACT_LINES; PROVISIONAL_REJECTED_CONTEXT_POSE"
            if not selection["registration_context_accepted"]
            else "SCAN_SUPPORTED_CONDITIONAL_CONTACT_LINES; RETAINED_ACCEPTED_CONTEXT_POSE"
        ),
    }
