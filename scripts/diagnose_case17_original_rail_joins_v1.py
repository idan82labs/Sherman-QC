"""Pinned derived-only endpoint diagnostic for case17 faces22/80. No native imports."""
import hashlib
import json
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "artifacts/cad-reference-evidence-v1-2026-09-30-attempt1/case-17/EVIDENCE.json"
PIN = "4f9766af5c8dd0974090da8a518134414f2c8662f9885068e0c06e97804efb8f"


def endpoints(occurrence, radius):
    curve = occurrence["pcurve"]
    first, last = curve["actual_trim_parameter_interval"]
    if "line" in curve:
        line = curve["line"]
        points = [[line["origin"][j] + t * line["direction"][j] for j in range(2)] for t in (first,last)]
        poles = points
        proof = "ANALYTIC_LINE_AT_ACTUAL_PARAMETERS"
    elif "bspline" in curve:
        spline = curve["bspline"]
        degree = spline["degree"]
        if spline["periodic"] or [first,last] != spline["original_parameter_domain"] or [first,last] != [spline["knots"][0],spline["knots"][-1]] or spline["multiplicities"][0] != degree+1 or spline["multiplicities"][-1] != degree+1:
            raise ValueError("EXACT_ORIGINAL_CLAMPED_ENDPOINT_PROOF_BLOCKED")
        poles = spline["poles"]
        points = [poles[0],poles[-1]]
        proof = "EXACT_ACTIVE_ORIGINAL_CLAMPED_BSPLINE_ENDPOINT_POLES"
    else:
        raise ValueError("UNSUPPORTED_ORIGINAL_PCURVE")
    points = [[radius*p[0],p[1]] for p in points]
    if occurrence["orientation"].endswith("TopAbs_REVERSED"):
        points.reverse()
    elif not occurrence["orientation"].endswith("TopAbs_FORWARD"):
        raise ValueError("UNSUPPORTED_ORIENTATION")
    return points, radius*(max(p[0] for p in poles)-min(p[0] for p in poles)), proof


def diagnose():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError("RELEASED_EVIDENCE_PIN_MISMATCH")
    evidence = json.loads(raw)
    results = []
    for face in evidence["faces"]:
        if face["id"] not in (22,80):
            continue
        radius, tol = face["cylinder"]["radius"],face["tolerance"]
        records = []
        for occurrence in face["wires"][0]["edge_occurrences"]:
            points, band, proof = endpoints(occurrence,radius)
            records.append({"edge_id":occurrence["edge_id"],"orientation":occurrence["orientation"],
                            "oriented_start_metric":points[0],"oriented_end_metric":points[1],
                            "u_coefficient_band_metric":band,"rail_by_face_tolerance":band<=tol,"endpoint_proof":proof})
        scale = max(1.,*(abs(v) for r in records for p in (r["oriented_start_metric"],r["oriented_end_metric"]) for v in p))
        guard = 64*math.ulp(scale)
        joins = []
        for i,previous in enumerate(records):
            following = records[(i+1)%len(records)]
            if previous["rail_by_face_tolerance"] and following["rail_by_face_tolerance"]:
                sign = 1 if previous["oriented_end_metric"][1]>previous["oriented_start_metric"][1] else -1
                delta = [following["oriented_start_metric"][j]-previous["oriented_end_metric"][j] for j in range(2)]
                joins.append({"from_edge":previous["edge_id"],"to_edge":following["edge_id"],"axial_direction_sign":sign,
                              "delta_metric":delta,"signed_axial_step":sign*delta[1],"full_chart_gap":math.hypot(*delta),
                              "backtrack_exceeds_64ulp":sign*delta[1]<-guard,
                              "gap_within_face_tolerance":math.hypot(*delta)<=tol})
        results.append({"face_id":face["id"],"radius":radius,"face_tolerance":tol,"coordinate_scale":scale,
                        "guard_64ulp":guard,"ordered_edges":records,"internal_rail_joins":joins,
                        "native_vertex_correspondence":"UNAVAILABLE_IN_RELEASED_NEUTRAL_EVIDENCE; ordered edge occurrence adjacency only"})
    return {"schema":"case17-original-rail-join-diagnostic-v1","source_sha256":PIN,"faces":results,
            "scope":"Derived original pcurves only; no native, raw CAD, snapping, threshold or engine changes"}


if __name__ == "__main__":
    print(json.dumps(diagnose(),indent=2,sort_keys=True,allow_nan=False))
