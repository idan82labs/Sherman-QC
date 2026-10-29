from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import runpy

import numpy as np
import pytest

import backend.f1_boundary_rejecting_multiscale_variation_track_composition_v70 as v70
import backend.f1_candidate_local_exact_closure_multiscale_variation_track_v71 as v71


HELPERS = runpy.run_path(str(Path(__file__).with_name("test_f1_boundary_rejecting_multiscale_variation_track_composition_v70.py")))
S = HELPERS["S"]
X = HELPERS["X"]


def _row(name: str, **kwargs):
    return HELPERS["_row"](name, **kwargs)


def _run(vertices, faces, rows):
    return v71.apply_clec_mvt_v1({"predictions": rows}, vertices, faces)


def _positive_case(name: str):
    if name == "P01": return HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0)), [_row(name)]
    if name == "P02": return HELPERS["_extrude"](HELPERS["_sharp_profile"](12.0)), [_row(name)]
    if name == "P03": return HELPERS["_extrude"](HELPERS["_rounded_profile"](90.0, 12.0)), [_row(name)]
    if name == "P04": return HELPERS["_extrude"](HELPERS["_faceted_profile"](60.0, 3)), [_row(name)]
    if name == "P05": return HELPERS["_extrude"](HELPERS["_rounded_profile"](170.0, 4.0)), [_row(name)]
    if name == "P06":
        profiles = [HELPERS["_rounded_profile"](90.0, 12.0) if 20 <= x < 44 else np.column_stack((S, np.zeros_like(S))) for x in X]
        return HELPERS["_extrude_profiles"](profiles), [_row(name)]
    if name == "P07":
        vertices, _faces = HELPERS["_extrude"](HELPERS["_rounded_profile"](90.0, 12.0))
        keep = np.ones(len(vertices), dtype=bool); width = len(S)
        for x in list(range(12, 16)) + list(range(31, 35)): keep[x * width:(x + 1) * width] = False
        for x in range(47, 55): keep[x * width + np.flatnonzero(S < 0)] = False
        return (vertices[keep], None), [_row(name)]
    if name in {"P08", "P09"}:
        separation = 8.0 if name == "P08" else 3.5
        profile, first, second = HELPERS["_opposing_fold_profile"](-separation / 2.0, separation / 2.0)
        rows = [_row(name + "-A", start=(0, *first), end=(64, *first)), _row(name + "-B", start=(0, *second), end=(64, *second))]
        return HELPERS["_extrude"](profile), rows
    if name == "P10":
        profiles = [HELPERS["_sharp_profile"](60.0, 2.0 * math.sin(2.0 * math.pi * x / 64.0)) for x in X]
        return HELPERS["_extrude_profiles"](profiles), [_row(name)]
    raise KeyError(name)


@pytest.mark.parametrize("name", [f"P{index:02d}" for index in range(1, 11)])
def test_v70_positive_fixture_bytes_remain_keep_track(name: str):
    (vertices, faces), rows = _positive_case(name)
    payload, evaluations = _run(vertices, faces, rows)
    assert payload["clec_mvt_v1_audit"]["status"] == "COMPLETE"
    assert all(value.decision == "KEEP_TRACK" for value in evaluations)


def _negative_case(name: str):
    flat = np.column_stack((S, np.zeros_like(S)))
    if name in {"N01", "N06"}:
        vertices, faces = HELPERS["_extrude"](flat)
        if name == "N06": faces = np.vstack((faces, faces[:, ::-1]))
        return vertices, faces
    if name in {"N02", "N03", "N04"}:
        profile = np.column_stack((np.arange(73, dtype=np.float64), np.zeros(73)))
        vertices, faces = HELPERS["_extrude"](profile)
        if name == "N04": vertices[:, 2] += 0.05 * np.sin(np.arange(len(vertices)) * 1.61803398875)
        return vertices, faces
    if name == "N05": return HELPERS["_cylinder"](80.0, complete=False)
    if name == "N07":
        profiles = []
        for x in X:
            angle = 90.0 * math.sin(math.pi * (x - 29.0) / 6.0) ** 2 if 29 <= x <= 35 else 0.0
            profiles.append(HELPERS["_sharp_profile"](angle))
        return HELPERS["_extrude_profiles"](profiles)
    if name == "N09": return HELPERS["_cylinder"](12.0, complete=True)
    raise KeyError(name)


@pytest.mark.parametrize("name", ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N09"])
def test_v70_negative_fixture_bytes_remain_complete_veto(name: str):
    vertices, faces = _negative_case(name)
    payload, evaluations = _run(vertices, faces, [_row(name)])
    assert payload["clec_mvt_v1_audit"]["status"] == "COMPLETE"
    assert evaluations[0].decision == "VETO_COMPLETE_NO_TRACK"


def test_v70_duplicate_fixture_has_one_geometry_owner():
    vertices, faces = HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0))
    rows = [_row("N08-A"), _row("N08-B", start=(0, 0.75, 0), end=(64, 0.75, 0))]
    _payload, evaluations = _run(vertices, faces, rows)
    assert sum(value.decision == "KEEP_TRACK" for value in evaluations) == 1
    assert sum(value.decision == "SUPPRESS_DUPLICATE" for value in evaluations) == 1


def test_v70_short_crossing_fixture_remains_two_complete_vetoes():
    profiles = []
    for x in X:
        angle = 90.0 * math.sin(math.pi * (x - 29.0) / 6.0) ** 2 if 29 <= x <= 35 else 0.0
        profiles.append(HELPERS["_sharp_profile"](angle))
    vertices, faces = HELPERS["_extrude_profiles"](profiles)
    rows = [_row("N10-A"), _row("N10-B", start=(32, -32, 0), end=(32, 32, 0))]
    _payload, evaluations = _run(vertices, faces, rows)
    assert [value.decision for value in evaluations] == ["VETO_COMPLETE_NO_TRACK", "VETO_COMPLETE_NO_TRACK"]


def test_v70_comparator_constants_and_semantic_diff_are_exact():
    assert v71.frozen_configuration()["changed_numeric_constants"] == []
    assert v71.frozen_configuration()["changed_field_formulas"] == []
    assert v70.WEAK_THRESHOLD == 0.20 and v70.STRONG_THRESHOLD == 0.55
    assert v70.HYSTERESIS_DISTANCE == 8.0 and v70.TRACK_STRENGTH == 0.35


def test_forbidden_fields_do_not_change_v71_output_hash():
    vertices, faces = HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0))
    first = _row("SAFE", score=0.0)
    second = _row("SAFE", score=999.0); second["selection_route"] = "different-forbidden-route"
    a, _ = _run(vertices, faces, [first]); b, _ = _run(vertices, faces, [second])
    assert a["clec_mvt_v1_audit"]["output_sha256"] == b["clec_mvt_v1_audit"]["output_sha256"]


def test_endpoint_source_and_winding_permutation_invariance():
    vertices, faces = HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0))
    baseline, baseline_eval = _run(vertices, faces, [_row("INV")])
    rng = np.random.default_rng(7107); permutation = rng.permutation(len(vertices)); inverse = np.empty(len(vertices), np.int64); inverse[permutation] = np.arange(len(vertices))
    changed, changed_eval = _run(vertices[permutation], inverse[faces][:, ::-1], [_row("INV", start=(64,0,0), end=(0,0,0))])
    assert baseline_eval[0].decision == changed_eval[0].decision == "KEEP_TRACK"
    assert baseline["clec_mvt_v1_audit"]["output_sha256"] == changed["clec_mvt_v1_audit"]["output_sha256"]


def test_exact_refinement_preserves_v71_positive_decision():
    vertices, faces = HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0))
    baseline, baseline_eval = _run(vertices, faces, [_row("REFINE")])
    output = [point.copy() for point in vertices]; midpoint = {}
    def mid(a,b):
        key=tuple(sorted((int(a),int(b))))
        if key not in midpoint: midpoint[key]=len(output); output.append(0.5*(vertices[key[0]]+vertices[key[1]]))
        return midpoint[key]
    refined=[]
    for a,b,c in faces:
        ab,bc,ca=mid(a,b),mid(b,c),mid(c,a);refined.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
    changed, changed_eval = _run(np.asarray(output), np.asarray(refined), [_row("REFINE")])
    assert baseline_eval[0].decision == changed_eval[0].decision == "KEEP_TRACK"


def test_l01_dense_sparse_field_equality_on_q0():
    vertices, faces = HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0)); started = __import__("time").perf_counter()
    graph = v71._build_source_graph(vertices, faces, started); row = _row("L01")
    candidate = v70._candidate({"candidate_opaque_hash": v71._sha(str(row["route_hash"]).encode()), "line_start": row["line_start"], "line_end": row["line_end"]}, 0)
    domain, _ = v71._domain(candidate, graph); sparse = v71._build_sparse_field(graph, [domain], started)
    all_neighbors=[]
    for begin in range(0,len(graph.source.points),256): all_neighbors.append(v71._query_exact(graph.tree,graph.source.points,np.arange(begin,min(len(graph.source.points),begin+256),dtype=np.int32),1024))
    dense_sigma=v71._sv_batch(graph.source.points,np.arange(len(graph.source.points),dtype=np.int32),np.vstack(all_neighbors))
    assert all(np.array_equal(sparse.sigma[int(q)], dense_sigma[int(q)]) for q in sparse.q0)


def test_l02_remote_disconnected_geometry_does_not_change_decision():
    vertices, faces = HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0)); base, base_eval = _run(vertices, faces, [_row("L02")])
    remote = vertices[:1300] + np.asarray((100000.0,100000.0,100000.0)); augmented = np.vstack((vertices,remote))
    changed, changed_eval = _run(augmented, faces, [_row("L02")])
    assert base_eval[0].decision == changed_eval[0].decision == "KEEP_TRACK"
    assert base["clec_mvt_v1_audit"]["candidates"][0]["core_sha256"] == changed["clec_mvt_v1_audit"]["candidates"][0]["core_sha256"]


def _tiny_graph(strong: bool, closed: bool):
    points=np.asarray([[0.,0,0],[1.,0,0],[2.,0,0],[3.,0,0]])
    source=v70.CanonicalSource(points,np.empty((0,3),np.int32),np.empty((0,2),np.int32),np.zeros(4,bool),"tiny")
    offsets=np.asarray([0,1,3,5,6]);neighbors=np.asarray([1,0,2,1,3,2],np.int32);lengths=np.ones(6)
    graph=v71.SourceGraph(source,None,np.ones(4),1.,offsets,neighbors,lengths,3)
    candidate=v70._candidate({"candidate_opaque_hash":"tiny","line_start":[0,0,0],"line_end":[1,0,0]},0)
    outer=np.asarray([0,1,2,3] if closed else [0,1,2],np.int32); interior=outer.copy() if closed else np.asarray([0,1],np.int32)
    domain=v71.CandidateDomain(candidate,1.,1.,np.asarray([0],np.int32),interior,outer,"c","i","e")
    q0=outer;weights=np.full(len(q0),0.3);strongs=np.zeros(len(q0),bool)
    if strong: strongs[-1]=True;weights[-1]=0.8
    sparse=v71.SparseField(q0,q0,"q0","q1",{},weights,strongs,np.ones(len(q0),bool),np.zeros(len(q0)),0)
    return graph,domain,sparse


def test_l03_external_strong_seed_path_abstains_not_vetoes():
    graph,domain,sparse=_tiny_graph(False,False);_ridge,unresolved,*_=v71._candidate_field(graph,sparse,domain);assert unresolved


def test_l04_closed_weak_component_permits_complete_negative():
    graph,domain,sparse=_tiny_graph(False,True);_ridge,unresolved,*_=v71._candidate_field(graph,sparse,domain);assert not unresolved


def test_l05_shell_comparator_is_inclusive():
    graph,domain,sparse=_tiny_graph(False,True);inside,_=v71._graph_domain(graph,np.asarray([0],np.int32),1.0);outside,_=v71._graph_domain(graph,np.asarray([0],np.int32),np.nextafter(1.0,math.inf));assert set(inside)=={0,1};assert set(outside)=={0,1}


def test_l06_overlapping_candidate_domain_order_is_canonical():
    vertices,faces=HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0));rows=[_row("L06-A"),_row("L06-B",start=(0,.75,0),end=(64,.75,0))]
    a,_=_run(vertices,faces,rows);b,_=_run(vertices,faces,list(reversed(rows)));assert a["clec_mvt_v1_audit"]["q0_sha256"]==b["clec_mvt_v1_audit"]["q0_sha256"];assert a["clec_mvt_v1_audit"]["q1_sha256"]==b["clec_mvt_v1_audit"]["q1_sha256"]


def test_l07_reflection_and_endpoint_reversal_preserve_decision():
    vertices,faces=HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0));a,ae=_run(vertices,faces,[_row("L07")]);ref=vertices.copy();ref[:,2]*=-1;b,be=_run(ref,faces[:,::-1],[_row("L07",start=(64,0,0),end=(0,0,0))]);assert ae[0].decision==be[0].decision=="KEEP_TRACK"


def test_l08_cap_equalities_are_frozen_in_configuration():
    assert v71.MAX_Q0==32768 and v71.MAX_Q1==65536 and v71.MAX_DOMAIN_MEMBERSHIP==262144 and v71.MAX_GRAPH_EDGE_VISITS==4194304


def test_l09_candidate_cap_plus_one_aborts_atomically():
    vertices,faces=HELPERS["_extrude"](HELPERS["_sharp_profile"](90.0));rows=[_row(f"CAP-{i}") for i in range(129)];payload,evaluations=_run(vertices,faces,rows);assert evaluations==();assert payload["clec_mvt_v1_audit"]["status"]=="SCAN_ATOMIC_ABORT";assert payload["clec_mvt_v1_audit"]["ordinary_nms_invocation_count"]==0


def test_l10_stress_contract_and_audit_exclusion():
    # The full host stress is executed by the locked runner. This unit fixture
    # proves the exact cap contract and coordinate-exclusion schema before it.
    config=v71.frozen_configuration();assert config["caps"]["q0"]==32768;assert config["caps"]["q1"]==65536
    assert "coordinate" not in v71.SCHEMA and "filename" not in v71.SCHEMA
