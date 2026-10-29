"""Shared deterministic protocol utilities for the locked V66 one-shot lane."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_cylindrical_consensus_recovery_v52 import compose_v52, _represented
from backend.f1_transverse_face_turn_ownership_certificate_v66 import canonical_bytes
from scripts.run_f1_v41_v16_real_scan_compatibility import load_bounded_source


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/bend_corpus/incoming/2026-08-13_shaked_new_parts_blind/opaque"
POOL_ROOT = ROOT / "output/research/f1-new-parts-20260814/aug13_candidate_pool_mesh_v3"
V51_ROOT = ROOT / "output/research/f1-v51-single-axis-parallel-return-recovery-v1"

CASE_BINDINGS: tuple[tuple[str, str], ...] = (
    ("U01", "01_U01.json"),
    ("U02", "02_U02.json"),
    ("U03", "03_U03.json"),
    ("U04", "04_U04.json"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    temporary.replace(path)


def source_paths(case_id: str, pool_name: str) -> tuple[Path, Path, Path]:
    return SOURCE_ROOT / f"{case_id}.ply", POOL_ROOT / pool_name, V51_ROOT / f"{case_id}.json"


def opaque_source_id(source_path: Path) -> str:
    return "S" + sha256_file(source_path)[:20].upper()


def load_identity(case_id: str, pool_name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    source_path, pool_path, v51_path = source_paths(case_id, pool_name)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    v51 = json.loads(v51_path.read_text(encoding="utf-8"))
    bounded, replay = load_bounded_source(source_path, pool)
    vertices = np.asarray(bounded["positions"], dtype=np.float64)
    faces = np.asarray(bounded["triangles"], dtype=np.int64)
    return vertices, faces, pool, {"v51": v51, "bounded_replay": replay}


def frozen_v52(case_id: str, pool_name: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]:
    vertices, faces, pool, loaded = load_identity(case_id, pool_name)
    result = compose_v52(deepcopy(loaded["v51"]), pool)
    rows = []
    for ordinal, original in enumerate(result.get("predictions") or ()):
        row = deepcopy(dict(original))
        bend_id = str(row.get("bend_id") or f"ROW-{ordinal}")
        row["route_hash"] = hashlib.sha256(bend_id.encode("utf-8")).hexdigest()[:32]
        row["better_first_key"] = [-float(row.get("selector_score") or 0.0), bend_id]
        rows.append(row)
    result["predictions"] = rows
    result["candidate_count"] = len(rows)
    result["v66_protocol_metadata"] = {
        "route_hash": "sha256_existing_bend_id_prefix_128",
        "better_first_key": "negative_existing_selector_score_then_existing_bend_id",
        "labels_used": False,
        "expected_count_used": False,
        "identity_used_by_decision": False,
    }
    return result, vertices, faces, loaded["bounded_replay"]


def mesh_semantic_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    vertex_rows = sorted(hashlib.sha256(np.asarray(vertex, dtype=">f8").tobytes()).hexdigest() for vertex in np.asarray(vertices))
    face_rows = []
    for face in np.asarray(faces, dtype=np.int64):
        keys = sorted(vertex_rows_by_index(vertices, face))
        face_rows.append("".join(keys))
    return hashlib.sha256(canonical_json_bytes({"vertices": vertex_rows, "faces": sorted(face_rows)})).hexdigest()


def vertex_rows_by_index(vertices: np.ndarray, face: Sequence[int]) -> list[str]:
    return [hashlib.sha256(np.asarray(vertices[int(index)], dtype=">f8").tobytes()).hexdigest() for index in face]


def transform_candidates(rows: Sequence[Mapping[str, Any]], matrix: np.ndarray, translation: np.ndarray) -> list[dict[str, Any]]:
    transformed = []
    for original in rows:
        row = deepcopy(dict(original))
        endpoints = np.asarray((row["line_start"], row["line_end"]), dtype=np.float64)
        mapped = endpoints @ matrix.T + translation
        row["line_start"] = mapped[0].tolist()
        row["line_end"] = mapped[1].tolist()
        transformed.append(row)
    return transformed


def similarity_variant(vertices: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    q = np.asarray(((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)), dtype=np.float64)
    matrix = 0.8 * q
    diagonal = float(np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0)))
    translation = np.asarray((8.0 * diagonal, -4.0 * diagonal, 2.0 * diagonal), dtype=np.float64)
    return vertices @ matrix.T + translation, matrix, transform_candidates(rows, matrix, translation)


def _coordinate_hash(point: np.ndarray) -> str:
    clean = np.asarray(point, dtype=np.float64).copy()
    clean[clean == 0.0] = 0.0
    return hashlib.sha256(clean.astype(">f8", copy=False).tobytes()).hexdigest()


def order_winding_variant(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertex_hashes = np.asarray([_coordinate_hash(point) for point in vertices])
    permutation = np.argsort(vertex_hashes, kind="mergesort")
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
    reordered_vertices = np.asarray(vertices[permutation], dtype=np.float64)
    remapped = inverse[np.asarray(faces, dtype=np.int64)]
    records = []
    for face in remapped:
        keys = sorted(_coordinate_hash(reordered_vertices[int(index)]) for index in face)
        face_hash = hashlib.sha256("".join(keys).encode("ascii")).hexdigest()
        current = face[::-1] if int(face_hash[-1], 16) & 1 else face
        records.append((face_hash, tuple(int(value) for value in current)))
    records.sort(key=lambda item: (item[0], item[1]))
    return reordered_vertices, np.asarray([item[1] for item in records], dtype=np.int64)


def pl_refinement_variant(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    records = []
    for ordinal, face in enumerate(np.asarray(faces, dtype=np.int64)):
        keys = sorted(_coordinate_hash(vertices[int(index)]) for index in face)
        records.append((hashlib.sha256("".join(keys).encode("ascii")).hexdigest(), ordinal))
    records.sort()
    count = min(4096, len(faces) // 128)
    selected = {ordinal for _key, ordinal in records[:count]}
    output_vertices = [np.asarray(vertex, dtype=np.float64) for vertex in vertices]
    output_faces: list[tuple[int, int, int]] = []
    for ordinal, face in enumerate(np.asarray(faces, dtype=np.int64)):
        a, b, c = (int(value) for value in face)
        if ordinal not in selected:
            output_faces.append((a, b, c))
            continue
        barycenter = (vertices[a] + vertices[b] + vertices[c]) / 3.0
        center = len(output_vertices)
        output_vertices.append(barycenter)
        output_faces.extend(((a, b, center), (b, c, center), (c, a, center)))
    return np.asarray(output_vertices, dtype=np.float64), np.asarray(output_faces, dtype=np.int64), count


def ordinary_final_nms(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]]]:
    ordered = sorted((deepcopy(dict(row)) for row in rows), key=lambda row: (tuple(row["better_first_key"]), str(row["route_hash"])))
    accepted: list[dict[str, Any]] = []
    audit = []
    for row in ordered:
        duplicate = next((prior for prior in accepted if _represented(row, [prior])), None)
        audit.append({"route_hash": row["route_hash"], "accepted": duplicate is None, "duplicate_of": None if duplicate is None else duplicate["route_hash"]})
        if duplicate is None:
            accepted.append(row)
    return accepted, audit


def tftoc_semantic_signature(result: Mapping[str, Any], pairs: Sequence[Any]) -> Mapping[str, Any]:
    audit = result["tftoc_v1_audit"]
    return {
        "status": audit["status"],
        "pairs": [
            {
                "route_hashes": list(pair.route_hashes),
                "class": pair.pair_class,
                "same": pair.same_count,
                "split": pair.split_count,
            }
            for pair in pairs
        ],
        "cliques": [
            {
                "members": row["members"],
                "representative": row["representative"],
                "suppressed": row["suppressed"],
            }
            for row in audit.get("cliques") or ()
        ],
        "suppressed_hashes": audit.get("suppressed_hashes") or [],
        "survivor_hashes": [row["route_hash"] for row in result.get("predictions") or ()],
    }


def pairwise_line_metrics(first: Mapping[str, Any], second: Mapping[str, Any]) -> tuple[float, float, float]:
    a0, a1 = np.asarray(first["line_start"], dtype=np.float64), np.asarray(first["line_end"], dtype=np.float64)
    b0, b1 = np.asarray(second["line_start"], dtype=np.float64), np.asarray(second["line_end"], dtype=np.float64)
    ua, ub = a1 - a0, b1 - b0
    la, lb = float(np.linalg.norm(ua)), float(np.linalg.norm(ub))
    ua, ub = ua / la, ub / lb
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(ua, ub)), -1.0, 1.0))))
    axis = ua if float(np.dot(ua, ub)) >= 0 else -ua
    delta = 0.5 * (b0 + b1) - 0.5 * (a0 + a1)
    transverse = float(np.linalg.norm(delta - np.dot(delta, axis) * axis))
    along = abs(float(np.dot(delta, axis)))
    overlap = max(0.0, 0.5 * (la + lb) - along) / min(la, lb)
    return angle, transverse, overlap
