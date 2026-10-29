#!/usr/bin/env python3
"""Build the complete frozen V34 source-only one-shot fixture schedule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research" / "f1_v34_opgt_v1_lock.json"
V33_FIX = ROOT / "output" / "research" / "f1-v33-pnfm-v1" / "source_free_fixtures"
OUT = ROOT / "output" / "research" / "f1-v34-opgt-v1"
FIX = OUT / "source_free_fixtures"
BIND = OUT / "triangle_bindings"
MANIFEST = OUT / "fixture_manifest.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_parts(key: int) -> tuple[int, int, int]:
    skin = key // 10_000_000
    remainder = key - skin * 10_000_000 - 1
    return skin, remainder // 10_000, remainder % 10_000


def faces_from_keys(keys: list[int]) -> list[list[int]]:
    by = {key_parts(int(key)): index for index, key in enumerate(keys)}
    faces: list[list[int]] = []
    for (skin, yi, ci), a in sorted(by.items()):
        b = by.get((skin, yi, ci + 1))
        c = by.get((skin, yi + 1, ci))
        d = by.get((skin, yi + 1, ci + 1))
        if b is not None and c is not None and d is not None:
            faces.extend(([a, c, b], [b, c, d]))
    return faces


def expected_pair(value: dict[str, Any]) -> list[int]:
    if "pre_nms" in value:
        return [int(value["pre_nms"]), int(value["final"])]
    rescues = int(value["rescues"])
    return [rescues, rescues]


def descriptor(
    fixture_id: str,
    source_path: Path,
    triangle_path: Path,
    candidates: list[dict[str, Any]],
    expected: list[int],
    construction: str,
    introduced_version: int,
    original_fixture_sha256: str,
    category: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "kind": "v34_opgt_frozen_fixture_binding",
        "fixture_id": fixture_id,
        "source_payload_path": str(source_path.relative_to(ROOT)),
        "source_payload_sha256": sha(source_path),
        "triangle_binding_path": str(triangle_path.relative_to(ROOT)),
        "triangle_binding_sha256": sha(triangle_path),
        "candidate_records": candidates,
        "expected_pre_final": expected,
        "construction": construction,
        "introduced_version": introduced_version,
        "original_fixture_sha256": original_fixture_sha256,
        "category": category,
    }
    if metadata:
        value.update(metadata)
    return value


def primary_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fixture_id in lock["ordered_primary_fixture_ids"]:
        prior_path = V33_FIX / f"{fixture_id}.json"
        prior = load(prior_path)
        triangle_path = ROOT / prior["triangle_binding_path"]
        row = descriptor(
            fixture_id,
            ROOT / prior["source_payload_path"],
            triangle_path,
            prior["candidate_records"],
            lock["primary_expected_pre_final"][fixture_id],
            prior["construction"],
            34,
            prior["source_payload_sha256"],
            "primary",
        )
        write_json(FIX / f"{fixture_id}.json", row)
        rows.append(row)
    return rows


def raw_mesh(version: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    source = payload.get("source", {})
    if version == 30:
        coordinates = source.get("coordinates")
        triangles = source.get("triangles")
        if coordinates is None or not triangles:
            return None
        return {"positions": coordinates, "triangles": triangles}
    positions = source.get("positions")
    keys = source.get("stable_source_keys")
    if positions is None or keys is None:
        return None
    faces = faces_from_keys(keys)
    if not faces:
        return None
    return {"positions": positions, "triangles": faces}


def candidate_schema_compatible(payload: dict[str, Any]) -> bool:
    required = {
        "candidate_id",
        "frozen_v16_order",
        "center_xyz",
        "unit_axis_xyz",
        "segment_length_L",
        "local_scale_s",
        "signed_turn_alpha",
        "parent_A",
        "parent_B",
        "render_line_start",
        "render_line_end",
    }
    records = payload.get("candidate_records")
    return bool(records) and all(required <= set(record) for record in records)


def historical_rows(primary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = {
        30: (
            ROOT / "output/research/f1-v30-amfc-v1/gate1_harness_manifest.json",
            ROOT / "research/f1_v30_amfc_v1_lock.json",
        ),
        31: (
            ROOT / "output/research/f1-v31-dhff-v1/gate1_harness_manifest.json",
            ROOT / "research/f1_v31_dhff_v1_lock.json",
        ),
        32: (
            ROOT / "output/research/f1-v32-avhfm-v1/preinvoke_harness_manifest.json",
            ROOT / "research/f1_v32_avhfm_v1_lock.json",
        ),
    }
    audit: list[dict[str, Any]] = [
        {"version": version, "status": "NO_FROZEN_SOURCE_FREE_MANIFEST_PRESENT"}
        for version in range(17, 30)
    ]
    seen: dict[str, tuple[list[int], str]] = {
        row["original_fixture_sha256"]: (row["expected_pre_final"], row["fixture_id"])
        for row in primary
    }
    rows: list[dict[str, Any]] = []
    for version in (30, 31, 32):
        manifest_path, lock_path = sources[version]
        manifest, version_lock = load(manifest_path), load(lock_path)
        expected = version_lock["expected_results"]
        included = excluded = deduped = 0
        for entry in sorted(
            manifest["fixture_rows"],
            key=lambda value: (str(value["fixture_id"]), str(value["sha256"])),
        ):
            fixture_id = str(entry["fixture_id"])
            if fixture_id.startswith("STRESS_"):
                excluded += 1
                continue
            source_path = ROOT / entry["path"]
            source_sha = sha(source_path)
            pair = expected_pair(expected[fixture_id])
            if source_sha in seen:
                if seen[source_sha][0] != pair:
                    raise SystemExit(
                        f"conflicting frozen outcome for {source_sha}: {seen[source_sha]} vs {(pair, fixture_id)}"
                    )
                deduped += 1
                continue
            payload = load(source_path)
            raw = raw_mesh(version, payload)
            if raw is None or not candidate_schema_compatible(payload):
                excluded += 1
                continue
            stable_id = f"H{version}_{fixture_id}_{source_sha[:12]}"
            triangle_path = BIND / f"{stable_id}.json"
            write_json(triangle_path, raw)
            row = descriptor(
                stable_id,
                source_path,
                triangle_path,
                payload["candidate_records"],
                pair,
                payload.get("construction", "frozen historical source-free fixture"),
                version,
                source_sha,
                "historical_union",
            )
            write_json(FIX / f"{stable_id}.json", row)
            rows.append(row)
            seen[source_sha] = (pair, stable_id)
            included += 1
        audit.append(
            {
                "version": version,
                "status": "FROZEN_MANIFEST_CONSUMED",
                "manifest_path": str(manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha(manifest_path),
                "lock_path": str(lock_path.relative_to(ROOT)),
                "lock_sha256": sha(lock_path),
                "included": included,
                "deduplicated": deduped,
                "source_incompatible_excluded": excluded,
            }
        )
    rows.sort(key=lambda row: (row["introduced_version"], row["fixture_id"], row["original_fixture_sha256"]))
    return rows, audit


def _transform_candidate(candidate: dict[str, Any], matrix: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
    value = json.loads(json.dumps(candidate))
    for field in ("center_xyz", "render_line_start", "render_line_end"):
        value[field] = (matrix @ np.asarray(value[field], dtype=np.float64) + translation).tolist()
    value["unit_axis_xyz"] = (matrix @ np.asarray(value["unit_axis_xyz"], dtype=np.float64)).tolist()
    for parent_key in ("parent_A", "parent_B"):
        parent = value[parent_key]
        normal = matrix @ np.asarray(parent["unit_normal_xyz"], dtype=np.float64)
        parent["unit_normal_xyz"] = normal.tolist()
        parent["centroid_xyz"] = (
            matrix @ np.asarray(parent["centroid_xyz"], dtype=np.float64) + translation
        ).tolist()
        parent["offset"] = float(parent["offset"] + np.dot(normal, translation))
    return value


def _write_generated_row(
    fixture_id: str,
    raw: dict[str, Any],
    candidates: list[dict[str, Any]],
    expected: list[int] | None,
    construction: str,
    category: str,
    original_fixture_sha256: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    triangle_path = BIND / f"{fixture_id}.json"
    write_json(triangle_path, raw)
    row = descriptor(
        fixture_id,
        triangle_path,
        triangle_path,
        candidates,
        expected,
        construction,
        34,
        original_fixture_sha256,
        category,
        metadata,
    )
    write_json(FIX / f"{fixture_id}.json", row)
    return row


def post_historical_rows(primary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base = next(row for row in primary if row["fixture_id"] == "SF01_FOLD60_MESH_C2")
    base_raw = load(ROOT / base["triangle_binding_path"])
    base_candidates = base["candidate_records"]
    base_sha = base["original_fixture_sha256"]
    rows: list[dict[str, Any]] = []

    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.asarray([7.0, -11.0, 13.0], dtype=np.float64) * float(base_candidates[0]["local_scale_s"])
    rigid_raw = {
        "positions": (np.asarray(base_raw["positions"], dtype=np.float64) @ rotation.T + translation).tolist(),
        "triangles": base_raw["triangles"],
    }
    rows.append(_write_generated_row(
        "META_RIGID_TRANSFORM", rigid_raw,
        [_transform_candidate(candidate, rotation, translation) for candidate in base_candidates],
        [1, 1], "proper rigid transform of frozen clean fold", "metamorphic", base_sha,
        {"metamorphic_transform": "rigid_transform", "semantic_reference_fixture_id": base["fixture_id"]},
    ))

    reversed_candidates = json.loads(json.dumps(base_candidates))
    for candidate in reversed_candidates:
        candidate["unit_axis_xyz"] = (-np.asarray(candidate["unit_axis_xyz"], dtype=np.float64)).tolist()
        candidate["render_line_start"], candidate["render_line_end"] = candidate["render_line_end"], candidate["render_line_start"]
        candidate["signed_turn_alpha"] = -float(candidate["signed_turn_alpha"])
    rows.append(_write_generated_row(
        "META_CANDIDATE_AXIS_REVERSAL", base_raw, reversed_candidates, [1, 1],
        "frozen candidate axis and endpoint reversal", "metamorphic", base_sha,
        {"metamorphic_transform": "candidate_axis_reversal", "semantic_reference_fixture_id": base["fixture_id"]},
    ))

    reflection = np.diag([-1.0, 1.0, 1.0])
    reflected_raw = {
        "positions": (np.asarray(base_raw["positions"], dtype=np.float64) @ reflection.T).tolist(),
        "triangles": base_raw["triangles"],
    }
    rows.append(_write_generated_row(
        "META_TRANSVERSE_FRAME_REVERSAL", reflected_raw,
        [_transform_candidate(candidate, reflection, np.zeros(3)) for candidate in base_candidates],
        [1, 1], "improper rigid reflection forcing the canonical transverse frame branch", "metamorphic", base_sha,
        {"metamorphic_transform": "transverse_frame_reversal", "semantic_reference_fixture_id": base["fixture_id"]},
    ))

    winding_raw = {"positions": base_raw["positions"], "triangles": [list(reversed(face)) for face in base_raw["triangles"]]}
    rows.append(_write_generated_row(
        "META_FACE_WINDING_INVERSION", winding_raw, base_candidates, [1, 1],
        "every triangle winding inverted", "metamorphic", base_sha,
        {"metamorphic_transform": "face_winding_inversion", "semantic_reference_fixture_id": base["fixture_id"]},
    ))

    face_order = sorted(
        range(len(base_raw["triangles"])),
        key=lambda index: hashlib.sha256(f"v34-face-{index}".encode("ascii")).digest(),
    )
    permutation_raw = {"positions": base_raw["positions"], "triangles": [base_raw["triangles"][index] for index in face_order]}
    rows.append(_write_generated_row(
        "META_FACE_PERMUTATION", permutation_raw, base_candidates, [1, 1],
        "deterministic hash permutation of triangle order", "metamorphic", base_sha,
        {"metamorphic_transform": "face_permutation", "semantic_reference_fixture_id": base["fixture_id"]},
    ))

    for source_id, fixture_id in (
        ("SF_PNFM_FRAG_COMPLEMENTARY_SKINS", "META_DELETION_COMPLEMENTARY_SKINS"),
        ("SF_PNFM_FRAG_PHASE_WINDOWS", "META_DELETION_PHASE_WINDOWS"),
    ):
        source = next(row for row in primary if row["fixture_id"] == source_id)
        raw = load(ROOT / source["triangle_binding_path"])
        rows.append(_write_generated_row(
            fixture_id, raw, source["candidate_records"], [1, 1],
            f"frozen deletion transform from {source_id}", "metamorphic", source["original_fixture_sha256"],
            {"metamorphic_transform": "frozen_deletion_transforms", "semantic_reference_fixture_id": base["fixture_id"]},
        ))

    replay = _write_generated_row(
        "REPLAY_SF01_FOLD60_MESH_C2", base_raw, base_candidates, [1, 1],
        "byte replay clone of frozen clean fold", "replay", base_sha,
        {"replay_reference_fixture_id": base["fixture_id"]},
    )

    used = np.unique(np.asarray(base_raw["triangles"], dtype=np.int64).reshape(-1))
    all_positions = np.asarray(base_raw["positions"], dtype=np.float64)
    all_faces = np.asarray(base_raw["triangles"], dtype=np.int64)
    start = np.asarray(base_candidates[0]["render_line_start"], dtype=np.float64)
    end = np.asarray(base_candidates[0]["render_line_end"], dtype=np.float64)
    direction = end - start
    denominator = float(np.dot(direction, direction))
    centroids = np.mean(all_positions[all_faces], axis=1)
    parameter = np.clip(((centroids - start) @ direction) / denominator, 0.0, 1.0)
    segment_distance = np.linalg.norm(centroids - (start + parameter[:, None] * direction), axis=1)
    selected_faces = all_faces[np.argsort(segment_distance, kind="stable")[:64]]
    selected_used = np.unique(selected_faces.reshape(-1))
    selected_positions = np.asarray(base_raw["positions"], dtype=np.float64)[selected_used]
    remap = {int(old): new for new, old in enumerate(selected_used)}
    selected_faces = [[remap[int(index)] for index in face] for face in selected_faces]
    padding_count = 200_000 - len(selected_positions)
    pad_index = np.arange(padding_count, dtype=np.float64)
    padding = np.column_stack((20_000.0 + 0.01 * pad_index, np.mod(pad_index, 997.0), 5_000.0 + np.floor(pad_index / 997.0)))
    stress_raw = {"positions": np.vstack((selected_positions, padding)).tolist(), "triangles": selected_faces}
    stress_candidates: list[dict[str, Any]] = []
    for index in range(128):
        candidate = json.loads(json.dumps(base_candidates[0]))
        candidate["candidate_id"] = 9_000_000 + index
        candidate["frozen_v16_order"] = index
        stress_candidates.append(candidate)
    stress = _write_generated_row(
        "STRESS_200K_128", stress_raw, stress_candidates, None,
        "exact 200000-vertex, 128-candidate bounded resource stress", "stress", base_sha,
        {"resource_contract": {"vertices": 200000, "candidates": 128, "seconds_max": 60.0}},
    )
    stress["unused_base_vertex_count"] = int(len(used) - len(selected_used))
    write_json(FIX / "STRESS_200K_128.json", stress)
    return rows, [replay], stress


def main() -> None:
    lock = load(LOCK)
    primary = primary_rows(lock)
    historical, audit = historical_rows(primary)
    metamorphic, replay, stress = post_historical_rows(primary)
    rows = primary + historical + metamorphic + replay + [stress]
    manifest = {
        "schema_version": 1,
        "kind": "v34_opgt_fixture_manifest",
        "lock_sha256": sha(LOCK),
        "fixture_count": len(rows),
        "primary_count": len(primary),
        "historical_count": len(historical),
        "metamorphic_count": len(metamorphic),
        "replay_count": len(replay),
        "stress_count": 1,
        "ordered_fixture_ids": [row["fixture_id"] for row in rows],
        "historical_manifest_audit": audit,
        "rows": [
            {
                "fixture_id": row["fixture_id"],
                "descriptor_path": str((FIX / f"{row['fixture_id']}.json").relative_to(ROOT)),
                "descriptor_sha256": sha(FIX / f"{row['fixture_id']}.json"),
                "source_payload_sha256": row["source_payload_sha256"],
                "triangle_binding_sha256": row["triangle_binding_sha256"],
                "expected_pre_final": row["expected_pre_final"],
                "category": row["category"],
            }
            for row in rows
        ],
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({"fixtures": len(rows), "primary": len(primary), "historical": len(historical), "metamorphic": len(metamorphic), "replay": len(replay), "stress": 1, "manifest": str(MANIFEST), "sha256": sha(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
