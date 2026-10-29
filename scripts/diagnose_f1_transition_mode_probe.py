#!/usr/bin/env python3
"""Emit a bounded, inference-safe TRANSITION_MODE_PROBE census.

The scan is hashed but never parsed.  Candidate pool, typed graph, and frozen
A0 solution are read before any gold artifact exists; no gold/CAD option is
accepted by this command.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_structured_birth_diagnostics import (  # noqa: E402
    diagnose_transition_mode_probe,
)


A0_PARENT_PROJECTION_PROFILE = "A0_V1_CENTROID_HALF_OPEN_V2_PARENT_V1"
MAX_LEGACY_SUPPORT_ATOMS = 20_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def _transition_payload(candidate_pool: Mapping[str, Any]) -> Mapping[str, Any]:
    if candidate_pool.get("support_representation_profile") == "TRANSITION_ATOM_V2":
        return candidate_pool
    for key in ("transition_atom_graph_v2", "transition_support", "compact_transition_graph"):
        value = candidate_pool.get(key)
        if isinstance(value, Mapping):
            return value
    raise RuntimeError(
        "candidate pool lacks compact_transition_graph_v2() output; build "
        "TRANSITION_ATOM_V2 before running TRANSITION_MODE_PROBE_V1"
    )


def _legacy_owner_lookup(a0_solution: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = a0_solution.get("atom_owners")
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        records = []
        for atom_id, owner in raw.items():
            if isinstance(owner, Mapping):
                record = dict(owner)
                record.setdefault("atom_id", atom_id)
            else:
                record = {"atom_id": atom_id, "owner_kind": owner}
            records.append(record)
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("A0 solution must provide atom_owners for V1-to-V2 projection")
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("A0 atom owner records must be objects")
        atom_id = str(raw_record.get("atom_id") or "")
        if not atom_id or atom_id in result:
            raise ValueError("A0 atom owners require unique atom_id values")
        owner_kind = str(raw_record.get("owner_kind") or raw_record.get("owner") or "UNKNOWN").upper()
        path_id = raw_record.get("latent_path_id") or raw_record.get("path_id")
        semantic_label = raw_record.get("semantic_label")
        if owner_kind in {"RESIDUAL", "RESIDUAL_NONCOUNTED"}:
            owner_kind = "RESIDUAL"
            semantic_label = "RESIDUAL_NONCOUNTED"
            path_id = None
        result[atom_id] = {
            "owner_kind": owner_kind,
            "semantic_label": None if semantic_label is None else str(semantic_label),
            "latent_path_id": None if path_id is None else str(path_id),
        }
    return result


def project_a0_v1_owners_to_transition_parents(
    candidate_pool: Mapping[str, Any],
    transition_payload: Mapping[str, Any],
    a0_solution: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project ownership by exact half-open V2 voxel containment only.

    This intentionally uses compact V1 atom centroids rather than source
    samples or nearest-neighbor assignment.  Any ambiguity or absent owner is
    preserved as a fail-closed physical-parent owner.
    """

    legacy_atoms = candidate_pool.get("support_atom_table")
    parents = transition_payload.get("support_parent_table")
    if not isinstance(legacy_atoms, list):
        raise ValueError("candidate pool must expose compact legacy support_atom_table")
    if not isinstance(parents, list):
        raise ValueError("TRANSITION_ATOM_V2 payload must expose support_parent_table")
    if len(legacy_atoms) > MAX_LEGACY_SUPPORT_ATOMS:
        raise ValueError(
            f"legacy support atoms exceed projection cap: {len(legacy_atoms)} > "
            f"{MAX_LEGACY_SUPPORT_ATOMS}"
        )
    owner_by_atom = _legacy_owner_lookup(a0_solution)
    atom_by_id: dict[str, Mapping[str, Any]] = {}
    for atom in legacy_atoms:
        if not isinstance(atom, Mapping):
            raise ValueError("legacy support_atom_table records must be objects")
        atom_id = str(atom.get("atom_id") or "")
        centroid = atom.get("centroid")
        if not atom_id or atom_id in atom_by_id:
            raise ValueError("legacy support atoms require unique atom_id values")
        if (
            not isinstance(centroid, list)
            or len(centroid) != 3
            or not all(math.isfinite(float(value)) for value in centroid)
        ):
            raise ValueError("legacy support atom centroid must be a finite three-vector")
        atom_by_id[atom_id] = atom

    parent_by_voxel: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    parent_size: float | None = None
    for parent in parents:
        if not isinstance(parent, Mapping):
            raise ValueError("transition parent records must be objects")
        parent_id = str(parent.get("parent_atom_id") or "")
        voxel_key = parent.get("voxel_key")
        voxel_size = float(parent.get("voxel_size_mm") or 0.0)
        if (
            not parent_id
            or not isinstance(voxel_key, list)
            or len(voxel_key) != 3
            or not math.isfinite(voxel_size)
            or voxel_size <= 0.0
        ):
            raise ValueError("transition parent requires ID, three-key voxel, and positive voxel size")
        if parent_size is None:
            parent_size = voxel_size
        elif not math.isclose(parent_size, voxel_size, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("A0 projection requires one uniform V2 parent voxel size")
        key = tuple(int(value) for value in voxel_key)
        if key in parent_by_voxel:
            raise ValueError("transition parent voxel keys must be unique")
        parent_by_voxel[key] = parent
    if parent_size is None:
        raise ValueError("TRANSITION_ATOM_V2 support_parent_table is empty")

    enclosed_by_parent: dict[str, list[str]] = {
        str(parent["parent_atom_id"]): [] for parent in parents
    }
    unmapped_legacy_atoms = 0
    for atom_id in sorted(atom_by_id):
        centroid = atom_by_id[atom_id]["centroid"]
        key = tuple(math.floor(float(value) / parent_size) for value in centroid)
        parent = parent_by_voxel.get(key)
        if parent is None:
            unmapped_legacy_atoms += 1
            continue
        enclosed_by_parent[str(parent["parent_atom_id"])].append(atom_id)

    records: list[dict[str, Any]] = []
    counters = {
        "parent_count": len(parents),
        "legacy_atom_count": len(legacy_atoms),
        "legacy_owner_count": len(owner_by_atom),
        "mapped_legacy_atom_count": len(legacy_atoms) - unmapped_legacy_atoms,
        "unmapped_legacy_atom_count": unmapped_legacy_atoms,
        "residual_parent_count": 0,
        "unanimous_nonresidual_parent_count": 0,
        "mixed_parent_count": 0,
        "unknown_parent_count": 0,
        "maximum_enclosed_legacy_atoms": 0,
    }
    for parent_id in sorted(enclosed_by_parent):
        atom_ids = sorted(enclosed_by_parent[parent_id])
        counters["maximum_enclosed_legacy_atoms"] = max(
            counters["maximum_enclosed_legacy_atoms"], len(atom_ids)
        )
        known_owners = [owner_by_atom.get(atom_id) for atom_id in atom_ids]
        if not atom_ids or any(owner is None for owner in known_owners):
            counters["unknown_parent_count"] += 1
            records.append(
                {
                    "parent_atom_id": parent_id,
                    "owner_kind": "UNKNOWN",
                    "semantic_label": "UNKNOWN_A0_OWNER",
                    "latent_path_id": None,
                    "enclosed_legacy_atom_count": len(atom_ids),
                }
            )
            continue
        owners = [dict(owner) for owner in known_owners if owner is not None]
        if all(owner["owner_kind"] == "RESIDUAL" for owner in owners):
            counters["residual_parent_count"] += 1
            records.append(
                {
                    "parent_atom_id": parent_id,
                    "owner_kind": "RESIDUAL",
                    "semantic_label": "RESIDUAL_NONCOUNTED",
                    "latent_path_id": None,
                    "enclosed_legacy_atom_count": len(atom_ids),
                }
            )
            continue
        identities = {
            (
                str(owner["owner_kind"]),
                owner.get("semantic_label"),
                owner.get("latent_path_id"),
            )
            for owner in owners
        }
        if len(identities) == 1:
            owner_kind, semantic_label, path_id = next(iter(identities))
            counters["unanimous_nonresidual_parent_count"] += 1
            records.append(
                {
                    "parent_atom_id": parent_id,
                    "owner_kind": owner_kind,
                    "semantic_label": semantic_label or owner_kind,
                    "latent_path_id": path_id,
                    "enclosed_legacy_atom_count": len(atom_ids),
                }
            )
            continue
        counters["mixed_parent_count"] += 1
        records.append(
            {
                "parent_atom_id": parent_id,
                "owner_kind": "MIXED_A0_OWNERS",
                "semantic_label": "MIXED_A0_OWNERS",
                "latent_path_id": None,
                "enclosed_legacy_atom_count": len(atom_ids),
            }
        )
    diagnostics = {
        "profile": A0_PARENT_PROJECTION_PROFILE,
        "containment_rule": "HALF_OPEN_V2_VOXEL",
        "voxel_size_mm": parent_size,
        "nearest_neighbor_used": False,
        "source_points_consumed": False,
        "truth_inputs_consumed": False,
        "counters": counters,
    }
    return records, diagnostics


def build_transition_probe(
    *,
    scan_path: Path,
    candidate_pool_path: Path,
    graph_path: Path,
    a0_solution_path: Path,
) -> dict[str, Any]:
    candidate_pool = _read_json(candidate_pool_path)
    prediction_graph = _read_json(graph_path)
    a0_solution = _read_json(a0_solution_path)
    transition_payload = _transition_payload(candidate_pool)
    transition_parent_owners, projection = project_a0_v1_owners_to_transition_parents(
        candidate_pool,
        transition_payload,
        a0_solution,
    )
    diagnostic_solution = deepcopy(a0_solution)
    diagnostic_solution["transition_parent_owners"] = transition_parent_owners
    diagnostic_solution["transition_parent_ownership_projection"] = projection
    return diagnose_transition_mode_probe(
        transition_payload,
        prediction_graph,
        diagnostic_solution,
        scan_sha256=_sha256(scan_path),
        candidate_pool_sha256=_sha256(candidate_pool_path),
        graph_sha256=_sha256(graph_path),
        a0_solution_sha256=_sha256(a0_solution_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--a0-solution", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_transition_probe(
        scan_path=args.scan.resolve(),
        candidate_pool_path=args.candidate_pool.resolve(),
        graph_path=args.graph.resolve(),
        a0_solution_path=args.a0_solution.resolve(),
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
