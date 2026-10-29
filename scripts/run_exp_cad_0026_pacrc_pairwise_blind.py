#!/usr/bin/env python3
"""Blind pairwise CAD-atlas coherent residual contrast (PACRC-V1).

This source-only runner derives opaque scan pairs from equal authenticated CAD
hashes, admits only pairs whose two registrations already passed frozen native
visual review, and compares each pair symmetrically over the accepted
face-owned CAD mesh.  It does not open evaluator roles, labels, gold, expected
counts, filenames-as-quality, or bend-line truth.

The output is a pre-truth anomaly ranking with explicit abstention.  It is not
bend F1 and it is not a production tolerance decision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP25_PATH = ROOT / "scripts" / "run_exp_cad_0025_analytic_brep_carrier_orientation.py"
EXP21_PATH = ROOT / "scripts" / "run_exp_cad_0021_protected_registration_selector.py"
SPEC = importlib.util.spec_from_file_location("exp25_orientation", EXP25_PATH)
EXP25 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP25)
EXP21 = EXP25.EXP21


TARGET_CELL_AREA_MM2 = 1.125
COMPONENT_CONNECT_RADIUS_MM = 3.0
MIN_COMPONENT_CELL_COUNT = 6
MIN_COMPONENT_AREA_MM2 = 9.0
MIN_COMPONENT_OBSERVABLE_AREA_FRACTION = 0.001
MIN_COMPONENT_MEDIAN_EXCESS_MM = 0.25
MIN_ABS_PAIR_SCORE = 0.60
MIN_ANCHOR_AREA_FRACTION = 0.20
MIN_ANCHOR_CELL_COUNT = 100
MAX_ONE_SIDED_SUPPORT_AREA_FRACTION = 0.02
PAIR_CAP_MULTIPLIER = 6.0
PAIR_CAP_MIN_MM = 8.0
ANCHOR_MULTIPLIER = 0.6
ANCHOR_MIN_MM = 1.0
TAU_MIN_MM = 0.5
TAU_MAD_MULTIPLIER = 3.0 * 1.4826
TRANSLATION_UNCERTAINTY_THICKNESS_FRACTION = 0.125
ROTATION_UNCERTAINTY_DEG = 0.25
EXPECTED_EVALUATOR_MANIFEST_SHA256 = "d39c40da2868b3e6220940ce63a2b673a28a866f6c53aefd4818f30924cd47e0"
EXPECTED_SOURCE_MANIFEST_SHA256 = "e62cba70e0635e5d4b15710d26a60f9cfc3c4565ebd38e75f3f78ed2b8c1abc9"
EXPECTED_EXP15_VISUAL_REVIEW_SHA256 = "e671ad465bc56b956bfe964a9af997cb9b1a01b75d39efdfe7903ff78704ef87"
EXPECTED_NATIVE_AVAILABILITY_AUDIT_SHA256 = "77d209544acf6551bb71e3dd7cc8bbe4c913d11988c9e103d21ff179b2be25d5"
EXPECTED_EXP25_RESULT_SHA256 = "777f1b42d80d44e5e828e3f6b9e87a3cec6f4319387310e4141e7a5d456d70f3"
EXPECTED_ELIGIBLE_CASE_TOKENS = {"N005", "N006", "N009", "N010", "N011", "N012"}
MESH_WELD_TOLERANCE_MM = 1e-8
EXPECTED_EXP21_IMPLEMENTATION_SHA256 = "9587e920cd1b3bf5a0e109b2d520e772592c10d671a368f8e5f0f517d91056e0"
EXPECTED_EXP25_IMPLEMENTATION_SHA256 = "09c28f30d8653b00fd77639245718e1b81243dbd4ab0bc8378a5ec0daa295cc7"
EXPECTED_TRANSFORM_SEMANTIC_SHA256 = {
    "N005": "22397a4f1be38d4ecf2dd151ed049862be44a0b51d2d90e0a15ac1f1defee89d",
    "N006": "921eb64770d3c106effda1b6ca11b2d4804eaaa5a95b3d4811b4f420de784a02",
    "N009": "02e85cc8bc4e58c0ef0f9ddfb5eca097d26d3d3492378781babf42b65da67aca",
    "N010": "766aef6c8d286798b793f32c05405ca00b2bd67ca9656fab2f1daf360b372b14",
    "N011": "abc21a18863a10ad10059049196c13a6e23bc6146bc93b78c0c8ce43c34b8173",
    "N012": "55d0f9db0f6e6860bbec4381565ba26d7dbf29c375fb1d493d7a186ed9b29ba3",
}
EXPECTED_CAD_ARTIFACTS = {
    "7521046532d251b80a9cd94e25399a069997de655ebeda1687dab8e9919c236c": {
        "sheet_thickness_mm": 1.5,
        "mesh_sha256": "c3484352b03b285c1871574705d407809714318ec02c2c24420790656ff0b880",
        "report_sha256": "b25460c2af4a4c663cb17b35d3da5a604fadab525044f83f0c15decc5ecddd22",
    },
    "fdb2b5ca0a9a572f5320a3644543fca595330b0800d0e0d2ecd04e41f313e571": {
        "sheet_thickness_mm": 1.5,
        "mesh_sha256": "9e6571704780383e99b6fa4f2800fa69a50a3e187467698fe5dd7414d9d2a716",
        "report_sha256": "01b60ca43635854a04df68a536c5daf15cf8650decb743caf8bb7ae593535fbf",
    },
    "b05649ff714e718962194331d5cfa1f992db0764ef3c883246a722db9a144791": {
        "sheet_thickness_mm": 2.0,
        "mesh_sha256": "ec98b0180e9b83d54662dae604aec1c4f89544952b074beb565b9b246a34cdc3",
        "report_sha256": "01e465a2e475eeff18988d0dae1f2def48e92ee95dce8f07d6e97ff5a4109985",
    },
}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(values) or len(values) != len(weights) or not np.all(weights >= 0):
        raise ValueError("invalid weighted quantile input")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("zero weighted quantile mass")
    order = np.lexsort((np.arange(len(values)), values))
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, float(quantile) * total, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def radical_inverse_base2(index: int) -> float:
    result = 0.0
    fraction = 0.5
    value = int(index)
    while value:
        if value & 1:
            result += fraction
        fraction *= 0.5
        value >>= 1
    return result


def deterministic_equal_area_atlas(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    target_cell_area_mm2: float = TARGET_CELL_AREA_MM2,
) -> dict[str, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("invalid triangle mesh")
    centroids: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    for parent_index, face in enumerate(triangles):
        triangle = vertices[face]
        area = 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
        if not np.isfinite(area) or area <= 1e-12:
            continue
        count = max(1, int(math.ceil(area / target_cell_area_mm2)))
        if count == 1:
            barycentric = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        else:
            indices = np.arange(count, dtype=np.int64)
            u = (indices.astype(np.float64) + 0.5) / float(count)
            v = np.asarray(
                [radical_inverse_base2(int(index) + 1) for index in indices],
                dtype=np.float64,
            )
            root = np.sqrt(u)
            barycentric = np.column_stack((1.0 - root, root * (1.0 - v), root * v))
        centroids.append(barycentric @ triangle)
        weights.append(np.full(count, area / count, dtype=np.float64))
        parents.append(np.full(count, parent_index, dtype=np.int64))
    if not centroids:
        raise ValueError("empty deterministic CAD atlas")
    points = np.vstack(centroids)
    area_weights = np.concatenate(weights)
    parent_faces = np.concatenate(parents)
    return {
        "points": points,
        "area_weights_mm2": area_weights,
        "parent_faces": parent_faces,
    }


def face_adjacency_set(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    weld_tolerance_mm: float = MESH_WELD_TOLERANCE_MM,
) -> set[tuple[int, int]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if weld_tolerance_mm <= 0.0:
        raise ValueError("mesh weld tolerance must be positive")
    vertex_union = UnionFind(len(vertices))
    close_pairs = cKDTree(vertices).query_pairs(weld_tolerance_mm, output_type="ndarray")
    for first, second in close_pairs:
        vertex_union.union(int(first), int(second))
    welded_vertex_ids = np.asarray(
        [vertex_union.find(index) for index in range(len(vertices))], dtype=np.int64
    )
    edge_owners: dict[tuple[int, int], list[int]] = {}
    for face_index, triangle in enumerate(triangles):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(
                sorted(
                    (
                        int(welded_vertex_ids[int(triangle[first])]),
                        int(welded_vertex_ids[int(triangle[second])]),
                    )
                )
            )
            if edge[0] == edge[1]:
                continue
            edge_owners.setdefault(edge, []).append(face_index)
    adjacency: set[tuple[int, int]] = set()
    for owners in edge_owners.values():
        for owner_index, first_owner in enumerate(owners):
            for second_owner in owners[owner_index + 1 :]:
                if first_owner != second_owner:
                    adjacency.add(tuple(sorted((first_owner, second_owner))))
    return adjacency


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        while parent != int(self.parent[parent]):
            parent = int(self.parent[parent])
        while value != parent:
            next_value = int(self.parent[value])
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, first: int, second: int) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def coherent_components(
    atlas_points: np.ndarray,
    parent_faces: np.ndarray,
    area_weights: np.ndarray,
    excess: np.ndarray,
    active_mask: np.ndarray,
    *,
    face_adjacency: set[tuple[int, int]],
    observable_area_mm2: float,
    sign: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    active_indices = np.flatnonzero(active_mask)
    retained = np.zeros(len(atlas_points), dtype=bool)
    if not len(active_indices):
        return [], retained
    tree = cKDTree(atlas_points[active_indices])
    pairs = tree.query_pairs(COMPONENT_CONNECT_RADIUS_MM, output_type="ndarray")
    union = UnionFind(len(active_indices))
    for local_first, local_second in pairs:
        first = int(active_indices[int(local_first)])
        second = int(active_indices[int(local_second)])
        face_first, face_second = int(parent_faces[first]), int(parent_faces[second])
        if face_first == face_second or tuple(sorted((face_first, face_second))) in face_adjacency:
            union.union(int(local_first), int(local_second))
    groups: dict[int, list[int]] = {}
    for local_index, global_index in enumerate(active_indices):
        groups.setdefault(union.find(local_index), []).append(int(global_index))
    minimum_area = max(
        MIN_COMPONENT_AREA_MM2,
        MIN_COMPONENT_OBSERVABLE_AREA_FRACTION * observable_area_mm2,
    )
    rows = []
    for indices in groups.values():
        selected = np.asarray(indices, dtype=np.int64)
        area = float(np.sum(area_weights[selected]))
        median_excess = weighted_quantile(excess[selected], area_weights[selected], 0.5)
        accepted = bool(
            len(selected) >= MIN_COMPONENT_CELL_COUNT
            and area >= minimum_area
            and median_excess >= MIN_COMPONENT_MEDIAN_EXCESS_MM
        )
        if accepted:
            retained[selected] = True
        rows.append(
            {
                "sign": sign,
                "cell_count": int(len(selected)),
                "area_mm2": area,
                "median_excess_mm": median_excess,
                "mass_mm3": float(np.sum(area_weights[selected] * excess[selected])) if accepted else 0.0,
                "accepted": accepted,
                "atlas_indices": selected.tolist() if accepted else [],
            }
        )
    rows.sort(key=lambda row: (-float(row["mass_mm3"]), -float(row["area_mm2"]), -int(row["cell_count"])))
    return rows, retained


def score_pair_core(
    atlas: Mapping[str, np.ndarray],
    face_adjacency: set[tuple[int, int]],
    scan_first: np.ndarray,
    scan_second: np.ndarray,
    *,
    sheet_thickness_mm: float,
    retain_arrays: bool,
) -> dict[str, Any]:
    points = np.asarray(atlas["points"], dtype=np.float64)
    weights = np.asarray(atlas["area_weights_mm2"], dtype=np.float64)
    parents = np.asarray(atlas["parent_faces"], dtype=np.int64)
    first_distances, _ = cKDTree(scan_first).query(points, k=1, workers=1)
    second_distances, _ = cKDTree(scan_second).query(points, k=1, workers=1)
    cap = max(PAIR_CAP_MIN_MM, PAIR_CAP_MULTIPLIER * sheet_thickness_mm)
    anchor_threshold = max(ANCHOR_MIN_MM, ANCHOR_MULTIPLIER * sheet_thickness_mm)
    anchors = (first_distances <= anchor_threshold) & (second_distances <= anchor_threshold)
    anchor_count = int(np.count_nonzero(anchors))
    anchor_area = float(np.sum(weights[anchors]))
    total_area = float(np.sum(weights))
    anchor_area_fraction = anchor_area / total_area
    capped_first = np.minimum(first_distances, cap)
    capped_second = np.minimum(second_distances, cap)
    if anchor_count:
        raw_difference = capped_first[anchors] - capped_second[anchors]
        bias = weighted_quantile(raw_difference, weights[anchors], 0.5)
        centered_anchor = raw_difference - bias
        mad = weighted_quantile(np.abs(centered_anchor), weights[anchors], 0.5)
    else:
        bias = 0.0
        mad = math.inf
    tau = max(TAU_MIN_MM, TAU_MAD_MULTIPLIER * mad) if np.isfinite(mad) else math.inf
    delta = capped_first - capped_second - bias
    observable = (first_distances < cap) & (second_distances < cap)
    observable_area = float(np.sum(weights[observable]))
    one_sided = (first_distances >= cap) ^ (second_distances >= cap)
    one_sided_area_fraction = float(np.sum(weights[one_sided])) / total_area
    positive_excess = np.maximum(delta - tau, 0.0)
    negative_excess = np.maximum(-delta - tau, 0.0)
    positive_rows, retained_positive = coherent_components(
        points, parents, weights, positive_excess, observable & (positive_excess > 0.0),
        face_adjacency=face_adjacency, observable_area_mm2=observable_area, sign="FIRST_MORE_ANOMALOUS",
    )
    negative_rows, retained_negative = coherent_components(
        points, parents, weights, negative_excess, observable & (negative_excess > 0.0),
        face_adjacency=face_adjacency, observable_area_mm2=observable_area, sign="SECOND_MORE_ANOMALOUS",
    )
    positive_mass = float(np.sum(weights[retained_positive] * positive_excess[retained_positive]))
    negative_mass = float(np.sum(weights[retained_negative] * negative_excess[retained_negative]))
    denominator = positive_mass + negative_mass
    score = (positive_mass - negative_mass) / denominator if denominator > 0.0 else 0.0
    baseline_first_p90 = weighted_quantile(first_distances, weights, 0.90)
    baseline_second_p90 = weighted_quantile(second_distances, weights, 0.90)
    result: dict[str, Any] = {
        "cap_mm": cap,
        "anchor_threshold_mm": anchor_threshold,
        "anchor_count": anchor_count,
        "anchor_area_fraction": anchor_area_fraction,
        "pair_bias_mm": bias,
        "anchor_mad_mm": mad,
        "tau_mm": tau,
        "observable_area_mm2": observable_area,
        "one_sided_support_area_fraction": one_sided_area_fraction,
        "positive_components": positive_rows,
        "negative_components": negative_rows,
        "positive_excess_mass_mm3": positive_mass,
        "negative_excess_mass_mm3": negative_mass,
        "signed_pair_score": float(score),
        "baseline_first_cad_to_scan_p90_mm": baseline_first_p90,
        "baseline_second_cad_to_scan_p90_mm": baseline_second_p90,
    }
    if retain_arrays:
        result["_arrays"] = {
            "first_distances": first_distances,
            "second_distances": second_distances,
            "delta": delta,
            "anchors": anchors,
            "one_sided": one_sided,
            "retained_positive": retained_positive,
            "retained_negative": retained_negative,
        }
    return result


def score_sign(score: float) -> int:
    return 1 if score > 0.0 else (-1 if score < 0.0 else 0)


def central_rank_gate_reasons(result: Mapping[str, Any]) -> list[str]:
    reasons = []
    if abs(float(result["signed_pair_score"])) < MIN_ABS_PAIR_SCORE:
        reasons.append("PAIR_SCORE_MAGNITUDE_GATE")
    if int(result["anchor_count"]) < MIN_ANCHOR_CELL_COUNT:
        reasons.append("ANCHOR_COUNT_GATE")
    if float(result["anchor_area_fraction"]) < MIN_ANCHOR_AREA_FRACTION:
        reasons.append("ANCHOR_AREA_GATE")
    if float(result["one_sided_support_area_fraction"]) >= MAX_ONE_SIDED_SUPPORT_AREA_FRACTION:
        reasons.append("ONE_SIDED_SUPPORT_GATE")
    if score_sign(float(result["signed_pair_score"])) == 0:
        reasons.append("NO_SIGNED_RANK")
    return reasons


def apply_small_rotation(points: np.ndarray, center: np.ndarray, axis_index: int, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = 1.0
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    rotation = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return (points - center) @ rotation.T + center


def uncertainty_variants(points: np.ndarray, center: np.ndarray, thickness: float) -> list[tuple[str, np.ndarray]]:
    rows = []
    translation = TRANSLATION_UNCERTAINTY_THICKNESS_FRACTION * thickness
    for axis in range(3):
        for sign in (-1.0, 1.0):
            vector = np.zeros(3, dtype=np.float64)
            vector[axis] = sign * translation
            rows.append((f"T{axis}_{sign:+.0f}", points + vector))
            rows.append(
                (
                    f"R{axis}_{sign:+.0f}",
                    apply_small_rotation(points, center, axis, sign * ROTATION_UNCERTAINTY_DEG),
                )
            )
    return rows


def source_transform(
    case_token: str,
    *,
    source_record: Mapping[str, Any],
    registration_root: Path,
    exp25_root: Path,
    exp25_result: Mapping[str, Any],
) -> tuple[np.ndarray, str, str]:
    exp25_case = exp25_root / "cases" / case_token / "semantic_result.json"
    if exp25_case.exists():
        if exp25_result.get("post_visual_absolute_registration_orientation_authorized_for_bound_cases") is not True:
            raise ValueError("EXP25 post-visual authorization unavailable")
        semantic_sha = EXP21.sha256_path(exp25_case)
        if semantic_sha != EXPECTED_TRANSFORM_SEMANTIC_SHA256.get(case_token):
            raise ValueError("EXP25 case is not the frozen reviewed semantic artifact")
        if semantic_sha != exp25_result.get("semantic_sha256_primary_and_replay", {}).get(case_token):
            raise ValueError("EXP25 result does not authorize this exact semantic artifact")
        payload = json.loads(exp25_case.read_text(encoding="utf-8"))
        if payload.get("case_token") != case_token or payload.get("scan_sha256") != source_record["scan_sha256"]:
            raise ValueError("EXP25 case binding mismatch")
        if not payload.get("orientation_gate_pass") or payload.get("evaluator_data_opened") is not False:
            raise ValueError("EXP25 case is not source-only actionable")
        transform = np.asarray(payload["selected_absolute_transform"], dtype=np.float64)
        return transform, "EXP25_POST_VISUAL_PROPER_ORIENTATION", semantic_sha

    path = registration_root / "cases" / case_token / "semantic_result.json"
    semantic_sha = EXP21.sha256_path(path)
    if semantic_sha != EXPECTED_TRANSFORM_SEMANTIC_SHA256.get(case_token):
        raise ValueError("EXP15 case is not the frozen visually reviewed semantic artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "EXP_CAD_0015_NEW_PHYSICAL_REGISTRATION_CASE_V2":
        raise ValueError("unexpected EXP15 registration schema")
    if payload.get("case_token") != case_token or payload.get("scan_sha256") != source_record["scan_sha256"]:
        raise ValueError("EXP15 case binding mismatch")
    if payload.get("source_cad_sha256") != source_record["cad_sha256"]:
        raise ValueError("EXP15 CAD binding mismatch")
    if payload.get("alignment_abstained") or payload.get("evaluator_data_opened") is not False:
        raise ValueError("EXP15 registration is not source-only actionable")
    if payload.get("gold_or_expected_counts_used") is not False or payload.get("generalization_evidence") is not False:
        raise ValueError("EXP15 registration used forbidden evidence")
    transform = np.asarray(payload["raw_scan_to_reference_transform"], dtype=np.float64)
    return transform, "EXP15_EXACT_V2_PROPER", semantic_sha


def render_pair(
    path: Path,
    *,
    pair_id: str,
    first_token: str,
    second_token: str,
    atlas: Mapping[str, np.ndarray],
    scan_first: np.ndarray,
    scan_second: np.ndarray,
    score: Mapping[str, Any],
) -> None:
    points = np.asarray(atlas["points"], dtype=np.float64)
    weights = np.asarray(atlas["area_weights_mm2"], dtype=np.float64)
    arrays = score["_arrays"]
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * np.sqrt(weights[:, None])).T @ (centered * np.sqrt(weights[:, None]))
    _values, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    projected = centered @ basis
    first_projected = (scan_first - center) @ basis
    second_projected = (scan_second - center) @ basis
    scan_step_first = max(1, len(scan_first) // 80000)
    scan_step_second = max(1, len(scan_second) // 80000)
    delta = np.asarray(arrays["delta"])
    retained_positive = np.asarray(arrays["retained_positive"])
    retained_negative = np.asarray(arrays["retained_negative"])
    one_sided = np.asarray(arrays["one_sided"])
    tau = float(score["tau_mm"])
    color_scale = max(2.0 * tau, 1.0)
    figure, axes = plt.subplots(3, 2, figsize=(30, 24), dpi=100, facecolor="#0b1016")

    for column, (token, cloud) in enumerate(
        ((first_token, first_projected), (second_token, second_projected))
    ):
        axis = axes[0, column]
        axis.set_facecolor("#0b1016")
        axis.scatter(projected[:, 0], projected[:, 1], color="#8b94a3", s=0.35, alpha=0.18, linewidths=0)
        step = scan_step_first if column == 0 else scan_step_second
        axis.scatter(cloud[::step, 0], cloud[::step, 1], color="#46d9ff" if column == 0 else "#ff66c4", s=0.65, alpha=0.32, linewidths=0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
        axis.set_title(f"{token} transformed scan over accepted CAD atlas", color="white", fontsize=15)

    projections = ((0, 1, "primary atlas view"), (0, 2, "secondary atlas view"))
    for column, (horizontal, vertical, title) in enumerate(projections):
        axis = axes[1, column]
        axis.set_facecolor("#0b1016")
        colors = plt.get_cmap("coolwarm")(
            np.clip((delta + color_scale) / (2.0 * color_scale), 0.0, 1.0)
        )
        axis.scatter(projected[:, horizontal], projected[:, vertical], c=colors, s=0.55, alpha=0.72, linewidths=0)
        axis.scatter(projected[retained_positive, horizontal], projected[retained_positive, vertical], color="#ff2d2d", s=4.0, alpha=0.95, linewidths=0)
        axis.scatter(projected[retained_negative, horizontal], projected[retained_negative, vertical], color="#2775ff", s=4.0, alpha=0.95, linewidths=0)
        axis.scatter(projected[one_sided, horizontal], projected[one_sided, vertical], facecolors="none", edgecolors="#ffe066", s=9.0, linewidths=0.5)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
        axis.set_title(
            f"{title} — red: {first_token} worse; blue: {second_token} worse; yellow ring: one-sided",
            color="white", fontsize=14,
        )

    for column, (mask, title, color) in enumerate(
        ((retained_positive, f"dominant {first_token}-worse region", "#ff2d2d"), (retained_negative, f"dominant {second_token}-worse region", "#2775ff"))
    ):
        axis = axes[2, column]
        axis.set_facecolor("#0b1016")
        if np.any(mask):
            selected = projected[mask, :2]
            low = np.min(selected, axis=0)
            high = np.max(selected, axis=0)
            padding = np.maximum((high - low) * 0.35, 5.0)
            in_view = (
                (projected[:, 0] >= low[0] - padding[0])
                & (projected[:, 0] <= high[0] + padding[0])
                & (projected[:, 1] >= low[1] - padding[1])
                & (projected[:, 1] <= high[1] + padding[1])
            )
            axis.scatter(projected[in_view, 0], projected[in_view, 1], color="#667080", s=0.6, alpha=0.25, linewidths=0)
            axis.scatter(selected[:, 0], selected[:, 1], color=color, s=5.0, alpha=0.95, linewidths=0)
            axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
            axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
            axis.set_aspect("equal", adjustable="box")
            axis.set_axis_off()
        else:
            axis.text(0.5, 0.5, "NO RETAINED COMPONENT", transform=axis.transAxes, ha="center", va="center", color="#c7ced8", fontsize=18)
            axis.set_axis_off()
        axis.set_title(title, color="white", fontsize=15)

    figure.suptitle(
        f"{pair_id} — blind PACRC-V1\n"
        f"D={float(score['signed_pair_score']):+.3f}; tau={tau:.3f} mm; "
        f"anchor area={float(score['anchor_area_fraction']):.1%}; one-sided={float(score['one_sided_support_area_fraction']):.2%}",
        color="white", fontsize=20, y=0.995,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def execute_pair(
    *,
    pair_id: str,
    first_record: Mapping[str, Any],
    second_record: Mapping[str, Any],
    sheet_thickness_mm: float,
    mesh_path: Path,
    mesh_report_path: Path,
    scan_root: Path,
    registration_root: Path,
    exp25_root: Path,
    exp25_result: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    expected_cad = EXPECTED_CAD_ARTIFACTS.get(str(first_record["cad_sha256"]))
    if expected_cad is None or first_record["cad_sha256"] != second_record["cad_sha256"]:
        raise ValueError("pair CAD is outside the frozen eligible artifact set")
    if not math.isclose(sheet_thickness_mm, float(expected_cad["sheet_thickness_mm"]), abs_tol=1e-12):
        raise ValueError("sheet thickness differs from frozen CAD binding")
    if EXP21.sha256_path(mesh_path) != expected_cad["mesh_sha256"]:
        raise ValueError("CAD mesh differs from frozen source-only artifact")
    if EXP21.sha256_path(mesh_report_path) != expected_cad["report_sha256"]:
        raise ValueError("CAD mesh report differs from frozen source-only artifact")
    mesh_report = json.loads(mesh_report_path.read_text(encoding="utf-8"))
    if mesh_report.get("status") != "SOURCE_ONLY_ENGINE_MESH_READY":
        raise ValueError("CAD mesh report is not source-only ready")
    if mesh_report.get("evaluator_data_opened") is not False or mesh_report.get("labels_or_expected_counts_used") is not False:
        raise ValueError("CAD mesh report used forbidden evidence")
    if mesh_report["source"]["sha256"] != first_record["cad_sha256"] or first_record["cad_sha256"] != second_record["cad_sha256"]:
        raise ValueError("pair CAD binding mismatch")
    if EXP21.sha256_path(mesh_path) != mesh_report["mesh"]["sha256"]:
        raise ValueError("accepted CAD mesh hash mismatch")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    atlas = deterministic_equal_area_atlas(vertices, triangles)
    face_adjacency = face_adjacency_set(vertices, triangles)

    transformed_scans = []
    transform_bindings = []
    for record in (first_record, second_record):
        token = str(record["opaque_id"])
        scan_path = scan_root / f"{token}.ply"
        if EXP21.sha256_path(scan_path) != record["scan_sha256"]:
            raise ValueError("scan hash mismatch")
        transform, source, semantic_sha = source_transform(
            token,
            source_record=record,
            registration_root=registration_root,
            exp25_root=exp25_root,
            exp25_result=exp25_result,
        )
        expected_transform_source = (
            "EXP25_POST_VISUAL_PROPER_ORIENTATION"
            if token in {"N011", "N012"}
            else "EXP15_EXACT_V2_PROPER"
        )
        if source != expected_transform_source:
            raise ValueError(f"unexpected frozen transform source for {token}")
        stats = EXP21.absolute_similarity_stats(transform)
        if not stats["proper_similarity_transform"] or abs(float(stats["uniform_scale"]) - 1.0) > 1e-6:
            raise ValueError("non-proper or scaled transform forbidden")
        raw = np.asarray(o3d.io.read_point_cloud(str(scan_path)).points, dtype=np.float64)
        capped, _normals = EXP21.density_capped_points_and_normals(raw)
        transformed_scans.append(EXP21.apply_transform(capped, transform))
        transform_bindings.append(
            {
                "case_token": token,
                "source": source,
                "semantic_sha256": semantic_sha,
                "absolute_transform_stats": stats,
                "density_capped_point_count": int(len(capped)),
            }
        )
    first_scan, second_scan = transformed_scans
    central = score_pair_core(
        atlas, face_adjacency, first_scan, second_scan,
        sheet_thickness_mm=sheet_thickness_mm, retain_arrays=True,
    )
    central_sign = score_sign(float(central["signed_pair_score"]))

    partition_checks = []
    for name, point_slice in (("EVEN", slice(0, None, 2)), ("ODD", slice(1, None, 2))):
        result = score_pair_core(
            atlas, face_adjacency, first_scan[point_slice], second_scan[point_slice],
            sheet_thickness_mm=sheet_thickness_mm, retain_arrays=False,
        )
        variant_reasons = central_rank_gate_reasons(result)
        partition_checks.append(
            {
                "partition": name,
                "signed_pair_score": result["signed_pair_score"],
                "variant_rank_gate_reasons": variant_reasons,
                "variant_rank_authorized": not variant_reasons,
                "winner_sign_agrees": (
                    not variant_reasons
                    and score_sign(float(result["signed_pair_score"])) == central_sign
                ),
            }
        )

    atlas_center = np.average(
        atlas["points"], axis=0, weights=atlas["area_weights_mm2"]
    )
    uncertainty_checks = []
    for target_index, label in ((0, "FIRST"), (1, "SECOND")):
        base_scan = transformed_scans[target_index]
        for variant_name, variant_scan in uncertainty_variants(base_scan, atlas_center, sheet_thickness_mm):
            scans = [first_scan, second_scan]
            scans[target_index] = variant_scan
            result = score_pair_core(
                atlas, face_adjacency, scans[0], scans[1],
                sheet_thickness_mm=sheet_thickness_mm, retain_arrays=False,
            )
            variant_reasons = central_rank_gate_reasons(result)
            uncertainty_checks.append(
                {
                    "perturbed_scan": label,
                    "variant": variant_name,
                    "signed_pair_score": result["signed_pair_score"],
                    "variant_rank_gate_reasons": variant_reasons,
                    "variant_rank_authorized": not variant_reasons,
                    "winner_sign_agrees": (
                        not variant_reasons
                        and score_sign(float(result["signed_pair_score"])) == central_sign
                    ),
                }
            )

    reasons = central_rank_gate_reasons(central)
    if not all(row["winner_sign_agrees"] for row in partition_checks):
        reasons.append("POINT_PARTITION_STABILITY_GATE")
    if not all(row["winner_sign_agrees"] for row in uncertainty_checks):
        reasons.append("REGISTRATION_UNCERTAINTY_STABILITY_GATE")
    numeric_rank_authorized = not reasons
    first_token, second_token = str(first_record["opaque_id"]), str(second_record["opaque_id"])
    numeric_winner = first_token if central_sign > 0 else (second_token if central_sign < 0 else None)
    baseline_winner = (
        first_token
        if float(central["baseline_first_cad_to_scan_p90_mm"]) > float(central["baseline_second_cad_to_scan_p90_mm"])
        else second_token
    )

    arrays = central.pop("_arrays")
    evidence_dtype = np.dtype(
        [
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("area_mm2", "<f8"),
            ("parent_face", "<i8"), ("first_distance_mm", "<f8"),
            ("second_distance_mm", "<f8"), ("delta_mm", "<f8"),
            ("anchor", "u1"), ("one_sided", "u1"),
            ("retained_positive", "u1"), ("retained_negative", "u1"),
        ]
    )
    evidence = np.empty(len(atlas["points"]), dtype=evidence_dtype)
    evidence["x"], evidence["y"], evidence["z"] = atlas["points"].T
    evidence["area_mm2"] = atlas["area_weights_mm2"]
    evidence["parent_face"] = atlas["parent_faces"]
    evidence["first_distance_mm"] = arrays["first_distances"]
    evidence["second_distance_mm"] = arrays["second_distances"]
    evidence["delta_mm"] = arrays["delta"]
    evidence["anchor"] = arrays["anchors"]
    evidence["one_sided"] = arrays["one_sided"]
    evidence["retained_positive"] = arrays["retained_positive"]
    evidence["retained_negative"] = arrays["retained_negative"]
    pair_root = output_root / "pairs" / pair_id
    evidence_path = pair_root / "atlas_evidence.npy"
    atomic_npy(evidence_path, evidence)
    render_path = pair_root / "pacrc_native_visual.png"
    central["_arrays"] = arrays
    render_pair(
        render_path,
        pair_id=pair_id,
        first_token=first_token,
        second_token=second_token,
        atlas=atlas,
        scan_first=first_scan,
        scan_second=second_scan,
        score=central,
    )
    central.pop("_arrays")
    for component in central["positive_components"] + central["negative_components"]:
        component.pop("atlas_indices", None)
    payload = {
        "schema": "EXP_CAD_0026_PACRC_PAIR_CASE_V4",
        "experiment_id": "EXP_CAD_0026",
        "pair_id": pair_id,
        "case_tokens": [first_token, second_token],
        "source_cad_sha256": first_record["cad_sha256"],
        "sheet_thickness_mm": sheet_thickness_mm,
        "mesh_report_sha256": EXP21.sha256_path(mesh_report_path),
        "accepted_cad_mesh_sha256": EXP21.sha256_path(mesh_path),
        "atlas_cell_count": int(len(atlas["points"])),
        "atlas_total_area_mm2": float(np.sum(atlas["area_weights_mm2"])),
        "mesh_weld_tolerance_mm": MESH_WELD_TOLERANCE_MM,
        "welded_face_adjacency_edge_count": int(len(face_adjacency)),
        "transform_bindings": transform_bindings,
        "central_metrics": central,
        "partition_checks": partition_checks,
        "registration_uncertainty_checks": uncertainty_checks,
        "numeric_rank_authorized": numeric_rank_authorized,
        "numeric_rank_reasons": reasons,
        "numeric_anomaly_winner_token": numeric_winner if numeric_rank_authorized else None,
        "baseline_higher_cad_p90_winner_token": baseline_winner,
        "evidence": {"path": str(evidence_path.relative_to(output_root)), "sha256": EXP21.sha256_path(evidence_path)},
        "render": {"path": str(render_path.relative_to(output_root)), "sha256": EXP21.sha256_path(render_path)},
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "final_rank_authorized": False,
        "evaluator_manifest_opened": False,
        "labels_or_roles_used": False,
        "bend_line_f1_authorized": False,
    }
    semantic_path = pair_root / "semantic_result.json"
    EXP21.atomic_json(semantic_path, payload)
    return {
        "pair_id": pair_id,
        "case_tokens": [first_token, second_token],
        "numeric_rank_authorized": numeric_rank_authorized,
        "numeric_anomaly_winner_token": payload["numeric_anomaly_winner_token"],
        "signed_pair_score": central["signed_pair_score"],
        "baseline_higher_cad_p90_winner_token": baseline_winner,
        "semantic_sha256": EXP21.sha256_path(semantic_path),
        "render_sha256": EXP21.sha256_path(render_path),
        "evidence_sha256": EXP21.sha256_path(evidence_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--registration-root", required=True, type=Path)
    parser.add_argument("--exp15-visual-review", required=True, type=Path)
    parser.add_argument("--native-availability-audit", required=True, type=Path)
    parser.add_argument("--exp25-root", required=True, type=Path)
    parser.add_argument("--exp25-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    exact_bindings = {
        args.source_manifest: EXPECTED_SOURCE_MANIFEST_SHA256,
        args.exp15_visual_review: EXPECTED_EXP15_VISUAL_REVIEW_SHA256,
        args.native_availability_audit: EXPECTED_NATIVE_AVAILABILITY_AUDIT_SHA256,
        args.exp25_result: EXPECTED_EXP25_RESULT_SHA256,
        EXP21_PATH: EXPECTED_EXP21_IMPLEMENTATION_SHA256,
        EXP25_PATH: EXPECTED_EXP25_IMPLEMENTATION_SHA256,
    }
    for path, expected_sha256 in exact_bindings.items():
        if EXP21.sha256_path(path) != expected_sha256:
            raise ValueError(f"frozen source-only binding mismatch: {path.name}")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("evaluator_only_manifest_opened") is not False:
        raise ValueError("source manifest says evaluator roles were opened")
    visual_review = json.loads(args.exp15_visual_review.read_text(encoding="utf-8"))
    if visual_review.get("evaluator_data_opened") is not False or visual_review.get("gold_or_expected_counts_used") is not False:
        raise ValueError("EXP15 review used forbidden evidence")
    decisions = {row["case_token"]: row for row in visual_review["decisions"]}
    availability = json.loads(args.native_availability_audit.read_text(encoding="utf-8"))
    if availability.get("evaluator_data_opened") is not False or availability.get("labels_or_expected_counts_used") is not False:
        raise ValueError("native availability audit used forbidden evidence")
    thickness_by_cad = {
        row["source_sha256"]: float(row["source_sheet_thickness_property"])
        for row in availability["records"]
        if row.get("source_sheet_thickness_property") not in (None, "")
        and str(row.get("source_sheet_thickness_property")).replace(".", "", 1).isdigit()
    }
    exp25_result = json.loads(args.exp25_result.read_text(encoding="utf-8"))

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in source_manifest["records"]:
        grouped.setdefault(record["cad_sha256"], []).append(record)
    if len(grouped) != 7 or any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("authenticated source does not contain seven two-scan CAD pairs")
    declared_pairs = []
    eligible_pairs = []
    for cad_sha, records in sorted(grouped.items(), key=lambda item: sorted(row["opaque_id"] for row in item[1])):
        records = sorted(records, key=lambda row: row["opaque_id"])
        tokens = [row["opaque_id"] for row in records]
        pair_id = f"PAIR_{tokens[0]}_{tokens[1]}"
        pair_decisions = [decisions.get(token, {}) for token in tokens]
        actionable = all(str(row.get("decision", "")).startswith("PASS_ACTIONABLE") for row in pair_decisions)
        reason = None if actionable else "EXP15_SOURCE_ONLY_NATIVE_REGISTRATION_REVIEW_NOT_PAIR_ACTIONABLE"
        declared_pairs.append(
            {
                "pair_id": pair_id,
                "case_tokens": tokens,
                "source_cad_sha256": cad_sha,
                "pretruth_status": "ELIGIBLE" if actionable else "ABSTAIN",
                "pretruth_reason": reason,
                "registration_visual_decisions": [row.get("decision") for row in pair_decisions],
            }
        )
        if actionable:
            eligible_pairs.append((pair_id, records))
    if len(eligible_pairs) != 3:
        raise ValueError("pretruth eligibility denominator differs from frozen 3/7")
    eligible_tokens = {
        str(record["opaque_id"])
        for _pair_id, records in eligible_pairs
        for record in records
    }
    if eligible_tokens != EXPECTED_ELIGIBLE_CASE_TOKENS:
        raise ValueError("pretruth eligible case identities differ from frozen contract")

    pair_records = []
    for pair_id, records in eligible_pairs:
        first = records[0]
        cad_path = Path(first["cad_path"])
        name = cad_path.stem
        if first["cad_sha256"] not in thickness_by_cad:
            raise ValueError(f"source thickness unavailable for {pair_id}")
        print(pair_id, flush=True)
        pair_records.append(
            execute_pair(
                pair_id=pair_id,
                first_record=records[0],
                second_record=records[1],
                sheet_thickness_mm=thickness_by_cad[first["cad_sha256"]],
                mesh_path=(args.mesh_root / f"{name}.ply").resolve(),
                mesh_report_path=(args.mesh_root / f"{name}.report.json").resolve(),
                scan_root=args.scan_root.resolve(),
                registration_root=args.registration_root.resolve(),
                exp25_root=args.exp25_root.resolve(),
                exp25_result=exp25_result,
                output_root=args.output_root.resolve(),
            )
        )
    config = {
        "target_cell_area_mm2": TARGET_CELL_AREA_MM2,
        "component_connect_radius_mm": COMPONENT_CONNECT_RADIUS_MM,
        "minimum_component_cell_count": MIN_COMPONENT_CELL_COUNT,
        "minimum_component_area_mm2": MIN_COMPONENT_AREA_MM2,
        "minimum_component_observable_area_fraction": MIN_COMPONENT_OBSERVABLE_AREA_FRACTION,
        "minimum_component_median_excess_mm": MIN_COMPONENT_MEDIAN_EXCESS_MM,
        "minimum_abs_pair_score": MIN_ABS_PAIR_SCORE,
        "minimum_anchor_area_fraction": MIN_ANCHOR_AREA_FRACTION,
        "minimum_anchor_cell_count": MIN_ANCHOR_CELL_COUNT,
        "maximum_one_sided_support_area_fraction": MAX_ONE_SIDED_SUPPORT_AREA_FRACTION,
        "translation_uncertainty_thickness_fraction": TRANSLATION_UNCERTAINTY_THICKNESS_FRACTION,
        "rotation_uncertainty_deg": ROTATION_UNCERTAINTY_DEG,
        "mesh_coordinate_weld_tolerance_mm": MESH_WELD_TOLERANCE_MM,
        "frozen_transform_semantic_sha256": EXPECTED_TRANSFORM_SEMANTIC_SHA256,
        "frozen_cad_artifacts": EXPECTED_CAD_ARTIFACTS,
        "imported_exp21_implementation_sha256": EXPECTED_EXP21_IMPLEMENTATION_SHA256,
        "imported_exp25_implementation_sha256": EXPECTED_EXP25_IMPLEMENTATION_SHA256,
        "evaluator_manifest_expected_sha256_for_later_isolated_unblinder": EXPECTED_EVALUATOR_MANIFEST_SHA256,
        "evaluator_manifest_not_accessed_by_source_runner": True,
    }
    summary = {
        "schema": "EXP_CAD_0026_PACRC_SOURCE_ONLY_SUMMARY_V4",
        "experiment_id": "EXP_CAD_0026",
        "architecture": "PAIRWISE_CAD_ATLAS_COHERENT_RESIDUAL_CONTRAST_V1",
        "implementation_sha256": EXP21.sha256_path(Path(__file__)),
        "invalidated_predecessor_output_roots": [
            "exp-cad-0026-pacrc-blind-v1",
            "exp-cad-0026-pacrc-blind-v2",
            "exp-cad-0026-pacrc-blind-v3",
        ],
        "source_manifest_sha256": EXP21.sha256_path(args.source_manifest),
        "exp15_visual_review_sha256": EXP21.sha256_path(args.exp15_visual_review),
        "native_availability_audit_sha256": EXP21.sha256_path(args.native_availability_audit),
        "exp25_result_sha256": EXP21.sha256_path(args.exp25_result),
        "config": config,
        "declared_pair_count": len(declared_pairs),
        "pretruth_eligible_pair_count": len(eligible_pairs),
        "pretruth_coverage_ceiling": len(eligible_pairs) / len(declared_pairs),
        "declared_pairs": declared_pairs,
        "pair_records": pair_records,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "final_rank_authorized_count": 0,
        "generalization_progress_indicator": "GPI-C_PENDING_BLINDED_PHYSICAL_PAIRWISE_VISUAL_GATE",
        "evaluator_manifest_opened": False,
        "labels_or_roles_used": False,
        "bend_line_f1_authorized": False,
        "generalization_evidence": False,
        "production_promotion_authorized": False,
    }
    EXP21.atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps(pair_records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
