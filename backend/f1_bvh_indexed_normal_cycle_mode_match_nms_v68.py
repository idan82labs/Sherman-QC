"""V68: BVH-indexed early-stage NCMM deduplicator.

V68 preserves every V67 geometric predicate and exact corridor clip.  Its only
mechanism change is a deterministic atom-AABB BVH, allowing the duplicate-rich
pre-NMS candidate stream to query local normal-cycle atoms without rescanning
the complete mesh measure for every pair.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

import backend.f1_similarity_invariant_normal_cycle_mode_match_nms_v67 as base


PROFILE = "V68_BVH_INDEXED_EARLY_NORMAL_CYCLE_MODE_MATCH_NMS_V1"
MECHANISM_LOCK = "V68_BVH_INDEXED_EARLY_NCMM_NMS_V1_LOCKED"
MAX_PRIMARY_SECONDS = 60.0
LEAF_SIZE = 64


@dataclass
class _Node:
    low: np.ndarray
    high: np.ndarray
    left: int
    right: int
    indices: tuple[int, ...]


class AtomBVH:
    def __init__(self, atoms: Sequence[base.Atom]):
        self.atoms = atoms
        count = len(atoms)
        self.low = np.empty((count, 3), dtype=np.float64)
        self.high = np.empty((count, 3), dtype=np.float64)
        self.center = np.empty((count, 3), dtype=np.float64)
        for index, atom in enumerate(atoms):
            self.low[index] = np.minimum(atom.start, atom.end)
            self.high[index] = np.maximum(atom.start, atom.end)
            self.center[index] = 0.5 * (atom.start + atom.end)
        self.nodes: list[_Node] = []
        self.root = self._build(tuple(range(count))) if count else -1

    def _build(self, indices: tuple[int, ...]) -> int:
        low = np.min(self.low[list(indices)], axis=0)
        high = np.max(self.high[list(indices)], axis=0)
        node_index = len(self.nodes)
        self.nodes.append(_Node(low, high, -1, -1, ()))
        if len(indices) <= LEAF_SIZE:
            self.nodes[node_index] = _Node(low, high, -1, -1, tuple(sorted(indices)))
            return node_index
        centers = self.center[list(indices)]
        axis = int(np.argmax(np.max(centers, axis=0) - np.min(centers, axis=0)))
        ordered = sorted(indices, key=lambda idx: (base.q40_int(self.center[idx, axis]), idx))
        midpoint = len(ordered) // 2
        left = self._build(tuple(ordered[:midpoint]))
        right = self._build(tuple(ordered[midpoint:]))
        self.nodes[node_index] = _Node(low, high, left, right, ())
        return node_index

    def query(self, low: np.ndarray, high: np.ndarray) -> list[base.Atom]:
        if self.root < 0:
            return []
        output: list[int] = []
        stack = [self.root]
        while stack:
            node = self.nodes[stack.pop()]
            if np.any(node.high < low) or np.any(node.low > high):
                continue
            if node.indices:
                output.extend(index for index in node.indices if not (np.any(self.high[index] < low) or np.any(self.low[index] > high)))
            else:
                stack.append(node.right)
                stack.append(node.left)
        return [self.atoms[index] for index in sorted(set(output))]


def _corridor_bounds(frame: Mapping[str, Any], xi0: float, xi1: float) -> tuple[np.ndarray, np.ndarray]:
    radius = max(frame["distance"] / 2.0, frame["length"] / 128.0)
    points = []
    for z in (frame["z0"], frame["z1"]):
        origin = frame["first_point"](z)
        for xi in (xi0, xi1):
            for eta in (-radius, radius):
                points.append(origin + xi * frame["distance"] * frame["e"] + eta * frame["n"])
    array = np.asarray(points, dtype=np.float64)
    return np.min(array, axis=0), np.max(array, axis=0)


def _indexed_pair(
    first: base.Candidate,
    second: base.Candidate,
    measure: base.MeshMeasure,
    hinge_bvh: AtomBVH,
    veto_bvh: AtomBVH,
    visit_counter: list[int],
) -> base.PairEvaluation:
    if base._exact_coincident(first, second):
        return base._evaluate_pair(first, second, measure, visit_counter)
    frame = base._pair_frame(first, second)
    if frame is None:
        return base._evaluate_pair(first, second, base.MeshMeasure([], [], measure.face_count, measure.graph_edge_count, measure.orientable_component_count, measure.nonorientable_component_count), visit_counter)
    frame["first_point"] = lambda z: base._point_on_line(first, frame["axis"], z)
    hinge_low, hinge_high = _corridor_bounds(frame, -0.5, 1.5)
    veto_low, veto_high = _corridor_bounds(frame, 0.25, 0.75)
    local = base.MeshMeasure(
        hinge_bvh.query(hinge_low, hinge_high),
        veto_bvh.query(veto_low, veto_high),
        measure.face_count,
        measure.graph_edge_count,
        measure.orientable_component_count,
        measure.nonorientable_component_count,
    )
    return base._evaluate_pair(first, second, local, visit_counter)


def apply_indexed_ncmm_nms_v1(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    vertices: np.ndarray | None,
    faces: np.ndarray | None,
    *,
    variant: bool = False,
) -> tuple[dict[str, Any], list[base.PairEvaluation]]:
    baseline = deepcopy(dict(source)) if isinstance(source, Mapping) else {}
    rows = [deepcopy(dict(row)) for row in (source.get("predictions") or ())] if isinstance(source, Mapping) else [deepcopy(dict(row)) for row in source]
    if vertices is None or faces is None or len(np.asarray(faces)) == 0:
        noop = base._atomic_noop(rows, "POINT_ONLY_KEEP_ALL", "ORIENTATION_UNAVAILABLE")
        baseline.update(noop)
        return baseline, []
    started = time.perf_counter()
    try:
        candidates = base._candidates(rows)
        if len(candidates) * (len(candidates) - 1) // 2 > base.MAX_TOTAL_PAIRS:
            raise base.NCMMResourceAbort("PAIR_CAP")
        measure = base.build_mesh_measure(np.asarray(vertices), np.asarray(faces), variant=variant)
        hinge_bvh, veto_bvh = AtomBVH(measure.hinge_atoms), AtomBVH(measure.veto_atoms)
        pairs: list[base.PairEvaluation] = []
        visits = [0]
        eligible = 0
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1 :]:
                pair = _indexed_pair(first, second, measure, hinge_bvh, veto_bvh, visits)
                if pair.reason != "PAIR_GEOMETRY":
                    eligible += 1
                    if eligible > base.MAX_ELIGIBLE_PAIRS:
                        raise base.NCMMResourceAbort("ELIGIBLE_PAIR_CAP")
                pairs.append(pair)
                if not variant and time.perf_counter() - started > MAX_PRIMARY_SECONDS:
                    raise base.NCMMResourceAbort("PRIMARY_WALL_TIME_CAP")

        positives = [pair for pair in pairs if pair.state in ("EXACT_COINCIDENT", "ONE_FOLD_MODE")]
        positives.sort(key=lambda pair: (0 if pair.state == "EXACT_COINCIDENT" else 1, tuple(-value for value in (pair.confidence_q40 or ())), pair.first_hash, pair.second_hash))
        matched: set[str] = set()
        suppressed: set[str] = set()
        selected = []
        for pair in positives:
            if pair.first_hash in matched or pair.second_hash in matched:
                continue
            matched.update((pair.first_hash, pair.second_hash))
            suppressed.add(str(pair.proposed_loser))
            selected.append({"pair": [pair.first_hash, pair.second_hash], "winner": pair.proposed_winner, "loser": pair.proposed_loser, "state": pair.state})
        survivors = [deepcopy(row) for row in rows if base._candidate_hash(row) not in suppressed]
        audit = {
            "profile": PROFILE,
            "mechanism_lock": MECHANISM_LOCK,
            "fixture_lock": base.FIXTURE_LOCK,
            "status": "OK",
            "source_hash": hashlib.sha256(np.asarray(vertices, dtype=">f8").tobytes() + np.asarray(faces, dtype=">i8").tobytes()).hexdigest(),
            "candidate_input_hash": base.sha256_json([base._candidate_hash(row) for row in rows]),
            "mesh_face_count": measure.face_count,
            "hinge_atom_count": len(measure.hinge_atoms),
            "veto_atom_count": len(measure.veto_atoms),
            "candidate_count": len(candidates),
            "pair_count": len(pairs),
            "geometry_or_exact_pair_count": eligible,
            "clip_visit_count": visits[0],
            "bvh_node_count": len(hinge_bvh.nodes) + len(veto_bvh.nodes),
            "pair_records": [base._pair_audit(pair) for pair in pairs if pair.reason != "PAIR_GEOMETRY" or pair.state == "EXACT_COINCIDENT"],
            "selected_matching": selected,
            "suppressed_hashes": sorted(suppressed),
            "survivor_hashes": [base._candidate_hash(row) for row in survivors],
        }
        encoded = base._encode_audit(audit)
        if len(encoded) > base.MAX_AUDIT_BYTES:
            raise base.NCMMResourceAbort("AUDIT_CAP")
        audit["semantic_sha256"] = hashlib.sha256(encoded).hexdigest()
        baseline.update({"schema_version": 1, "kind": "sherman_qc_f1_v68_bvh_indexed_early_ncmm_result", "profile": PROFILE, "predictions": survivors, "candidate_count": len(survivors), "ncmm_nms_v1_audit": audit, "source_only": True, "research_only": True, "expected_count_used": False, "labels_used": False, "cad_used": False, "part_identity_used": False, "filename_identity_used": False})
        return baseline, pairs
    except (base.NCMMInputError, base.NCMMResourceAbort) as error:
        status = "RESOURCE_ABORT" if isinstance(error, base.NCMMResourceAbort) else "NUMERIC_OR_INPUT_ABORT"
        noop = base._atomic_noop(rows, status, str(error))
        noop["ncmm_nms_v1_audit"].update({"profile": PROFILE, "mechanism_lock": MECHANISM_LOCK})
        baseline.update(noop)
        return baseline, []


__all__ = ["AtomBVH", "MECHANISM_LOCK", "PROFILE", "apply_indexed_ncmm_nms_v1"]
