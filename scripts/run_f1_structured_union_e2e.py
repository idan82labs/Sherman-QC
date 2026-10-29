#!/usr/bin/env python3
"""Run the professor-guided F1 typed union lane end to end.

This command is research-only.  It sanitizes a strict scan-only candidate pool,
builds the typed support graph, proves the exact CP-SAT solution in the isolated
solver environment, optionally performs bounded BIRTH_V1 column generation,
and emits one finite line per counted continuation component.  It does not
touch the runtime/API detector or authorize promotion.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_structured_residual_birth import (  # noqa: E402
    TOTAL_BIRTH_COLUMN_CAP,
    add_birth_columns,
    generate_residual_birth_columns,
)
from backend.f1_structured_union_graph import (  # noqa: E402
    BEND_STRAIGHT_COUNTED,
    build_structured_union_graph,
)


GLOBAL_POINT_BUDGET = 200_000
ALGORITHM = "f1_typed_union_exact_birth_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pool_contract(pool: Mapping[str, Any]) -> None:
    if pool.get("source_point_serialization") is not False:
        raise ValueError("candidate pool must prohibit source-point serialization")
    contract = dict(pool.get("inference_contract") or {})
    if any(
        bool(contract.get(field))
        for field in (
            "part_identifier_exposed",
            "expected_count_exposed",
            "manual_truth_exposed",
            "cad_exposed",
        )
    ):
        raise ValueError("candidate pool violates strict scan-only inference")
    if pool.get("working_point_contract_version") != "GLOBAL_POINT_BUDGET_V1":
        raise ValueError("candidate pool must declare GLOBAL_POINT_BUDGET_V1")
    if pool.get("working_point_budget") != GLOBAL_POINT_BUDGET:
        raise ValueError("candidate pool declares an incompatible working point budget")
    for field in ("transient_sampled_point_count", "transient_plane_point_count"):
        if field not in pool:
            raise ValueError(f"candidate pool must report {field}")
        value = int(pool[field])
        if value < 0:
            raise ValueError(f"candidate pool {field} must be nonnegative")
        if value > GLOBAL_POINT_BUDGET:
            raise ValueError(
                f"candidate pool {field} exceeds global point budget: "
                f"{value} > {GLOBAL_POINT_BUDGET}"
            )


def _ru_maxrss_bytes(value: float, *, platform: str | None = None) -> int:
    """Normalize getrusage(2) ru_maxrss units to bytes.

    Darwin reports bytes; Linux and the BSDs supported by Python report KiB.
    Keeping the conversion explicit prevents a 1024x understatement when a
    research run is replayed outside the current Mac workstation.
    """

    platform = sys.platform if platform is None else str(platform)
    multiplier = 1 if platform == "darwin" else 1024
    return max(0, int(round(float(value) * multiplier)))


def _atom_owner_paths(solution: Mapping[str, Any]) -> dict[str, str | None]:
    """Normalize legacy and exact-solver atom ownership representations."""

    owners = solution.get("atom_owners")
    if isinstance(owners, Mapping):
        normalized: dict[str, str | None] = {}
        for atom_id, owner in owners.items():
            if isinstance(owner, Mapping):
                owner_kind = str(owner.get("owner_kind") or "").upper()
                path_id = owner.get("latent_path_id")
                normalized[str(atom_id)] = (
                    str(path_id) if owner_kind == "PATH" and path_id is not None else None
                )
            else:
                token = str(owner)
                normalized[str(atom_id)] = (
                    None
                    if token.upper() in {"RESIDUAL", "RESIDUAL_NONCOUNTED"}
                    else token
                )
        return normalized
    if isinstance(owners, Sequence) and not isinstance(owners, (str, bytes)):
        normalized = {}
        for record in owners:
            if not isinstance(record, Mapping):
                raise ValueError("atom_owners records must be objects")
            atom_id = str(record["atom_id"])
            owner_kind = str(record.get("owner_kind") or "").upper()
            path_id = record.get("latent_path_id")
            if owner_kind == "PATH":
                if path_id is None:
                    raise ValueError("PATH atom owner must provide latent_path_id")
                normalized[atom_id] = str(path_id)
            elif owner_kind == "RESIDUAL" or str(
                record.get("owner_id") or record.get("owner") or ""
            ).upper() in {"RESIDUAL", "RESIDUAL_NONCOUNTED"}:
                normalized[atom_id] = None
            else:
                raise ValueError("atom owner record has no recognized owner kind")
        return normalized
    raise ValueError("exact solution must expose per-atom owners")


def _run_exact_solver(
    *,
    solver_python: Path,
    graph_path: Path,
    solution_path: Path,
) -> dict[str, Any]:
    command = [
        str(solver_python),
        str(ROOT / "scripts/solve_f1_structured_union_cpsat.py"),
        "--graph",
        str(graph_path),
        "--output",
        str(solution_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    (solution_path.parent / f"{solution_path.stem}.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (solution_path.parent / f"{solution_path.stem}.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"exact structured solver failed with exit {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    result = _read(solution_path)
    status = str((result.get("solver") or {}).get("status") or "")
    if status != "OPTIMAL":
        raise RuntimeError(f"structured result must be OPTIMAL, got {status or 'UNKNOWN'}")
    if result.get("unsupported_selected_path_count") not in (None, 0):
        raise RuntimeError("exact result contains unsupported non-residual paths")
    return result


def _canonical_axis(value: Sequence[float]) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    for component in axis:
        if abs(float(component)) > 1e-12:
            return -axis if component < 0.0 else axis
    return axis


def _component_groups(
    counted_ids: set[str], continuations: Sequence[Mapping[str, Any]]
) -> list[list[str]]:
    parent = {path_id: path_id for path_id in counted_ids}

    def find(path_id: str) -> str:
        while parent[path_id] != path_id:
            parent[path_id] = parent[parent[path_id]]
            path_id = parent[path_id]
        return path_id

    for record in continuations:
        left, right = (str(value) for value in record["path_ids"])
        if left not in counted_ids or right not in counted_ids:
            raise ValueError("selected continuation references a non-counted path")
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            raise ValueError("selected continuations contain a cycle")
        parent[right_root] = left_root
    groups: dict[str, list[str]] = defaultdict(list)
    for path_id in sorted(counted_ids):
        groups[find(path_id)].append(path_id)
    return sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])


def emit_counted_components(
    graph: Mapping[str, Any], solution: Mapping[str, Any]
) -> list[dict[str, Any]]:
    path_by_id = {
        str(path["latent_path_id"]): dict(path)
        for path in graph.get("latent_paths") or ()
    }
    labels = {
        str(record["latent_path_id"]): str(record["semantic_label"])
        for record in solution.get("assignments") or ()
    }
    counted_ids = {path_id for path_id, label in labels.items() if label == BEND_STRAIGHT_COUNTED}
    atom_by_id = {
        str(atom["atom_id"]): dict(atom)
        for atom in graph.get("support_atom_table") or ()
    }
    atom_owners = _atom_owner_paths(solution)
    components = []
    for path_ids in _component_groups(
        counted_ids,
        [dict(record) for record in solution.get("selected_continuations") or ()],
    ):
        paths = [path_by_id[path_id] for path_id in path_ids]
        all_endpoints = np.vstack(
            [np.asarray(path["span_endpoints"], dtype=np.float64) for path in paths]
        )
        owned_atom_ids = sorted(
            atom_id
            for atom_id, owner in atom_owners.items()
            if owner in set(path_ids)
        )
        support = np.asarray(
            [atom_by_id[atom_id]["centroid"] for atom_id in owned_atom_ids if atom_id in atom_by_id],
            dtype=np.float64,
        )
        if support.ndim == 2 and len(support) >= 2:
            weights = np.asarray(
                [
                    max(float(atom_by_id[atom_id].get("normalized_support_mass") or 1.0), 1.0)
                    for atom_id in owned_atom_ids
                    if atom_id in atom_by_id
                ],
                dtype=np.float64,
            )
            anchor = np.average(support, axis=0, weights=weights)
            centered = support - anchor
            covariance = (centered * weights[:, None]).T @ centered / max(float(np.sum(weights)), 1e-12)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis = _canonical_axis(eigenvectors[:, int(np.argmax(eigenvalues))])
        else:
            longest = max(paths, key=lambda path: float(path.get("span_mm") or 0.0))
            axis = _canonical_axis(longest["axis_direction"])
            anchor = np.mean(all_endpoints, axis=0)
        projections = (all_endpoints - anchor) @ axis
        endpoints = np.asarray(
            (anchor + float(np.min(projections)) * axis, anchor + float(np.max(projections)) * axis),
            dtype=np.float64,
        )
        signature = json.dumps(path_ids, separators=(",", ":")).encode("utf-8")
        component_id = f"F1C_{hashlib.sha256(signature).hexdigest()[:16]}"
        panel_pairs = sorted(
            {
                tuple(path.get("primary_panel_pair") or ())
                for path in paths
                if len(path.get("primary_panel_pair") or ()) == 2
            }
        )
        components.append(
            {
                "bend_id": component_id,
                "semantic_label": BEND_STRAIGHT_COUNTED,
                "counted": True,
                "member_path_ids": path_ids,
                "span_endpoints": endpoints.tolist(),
                "axis_direction": axis.tolist(),
                "span_mm": float(np.linalg.norm(endpoints[1] - endpoints[0])),
                "incident_panel_pairs": [list(pair) for pair in panel_pairs],
                "owned_atom_ids": owned_atom_ids,
                "coordinate_frame": "scan_raw",
            }
        )
    return components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--solver-python",
        type=Path,
        default=ROOT / ".venv-f1-structured/bin/python",
    )
    parser.add_argument("--disable-birth", action="store_true")
    parser.add_argument("--max-birth-iterations", type=int, default=4)
    args = parser.parse_args()

    pool_path = args.candidate_pool.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = _read(pool_path)
    _validate_pool_contract(pool)
    graph = build_structured_union_graph(pool)
    if any(field in graph for field in ("case_id", "scan_path", "filename", "part_id")):
        raise ValueError("sanitized graph leaked storage metadata")

    birth_iterations = []
    total_added = 0
    solution: dict[str, Any] | None = None
    start_child_rss_bytes = _ru_maxrss_bytes(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    for iteration in range(max(1, int(args.max_birth_iterations) + 1)):
        graph_path = output_dir / f"typed_union_graph_{iteration:02d}.json"
        solution_path = output_dir / f"cpsat_solution_{iteration:02d}.json"
        _write(graph_path, graph)
        solution = _run_exact_solver(
            # Do not resolve this symlink: Python virtual environments rely on
            # argv[0] pointing inside the environment to discover site-packages.
            solver_python=args.solver_python.absolute(),
            graph_path=graph_path,
            solution_path=solution_path,
        )
        if args.disable_birth or iteration >= int(args.max_birth_iterations):
            break
        available = TOTAL_BIRTH_COLUMN_CAP - total_added
        proposals = generate_residual_birth_columns(
            graph,
            solution,
            total_column_cap=max(0, available),
        )
        birth_iterations.append(
            {
                "iteration": iteration,
                "negative_lower_bound_proposal_count": len(proposals),
                "proposal_ids": [str(item["birth_id"]) for item in proposals],
            }
        )
        if not proposals:
            break
        graph = add_birth_columns(graph, proposals)
        total_added += len(proposals)
        if total_added >= TOTAL_BIRTH_COLUMN_CAP:
            continue

    if solution is None:
        raise RuntimeError("structured solver did not run")
    counted = emit_counted_components(graph, solution)
    final_graph_path = output_dir / "typed_union_graph.json"
    final_solution_path = output_dir / "cpsat_solution.json"
    _write(final_graph_path, graph)
    _write(final_solution_path, solution)
    _write(
        output_dir / "prediction_owned_regions.json",
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_typed_union_counted_components",
            "algorithm": ALGORITHM,
            "coordinate_frame": "scan_raw",
            "owned_bend_regions": counted,
        },
    )
    child_peak_rss_bytes = _ru_maxrss_bytes(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    summary = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_typed_union_research_summary",
        "algorithm": ALGORITHM,
        "research_only": True,
        "runtime_promotion_authorized": False,
        "strict_scan_only": True,
        "candidate_pool_sha256": _sha256(pool_path),
        "graph_sha256": str(graph.get("graph_sha256") or _sha256(final_graph_path)),
        "solver_status": str((solution.get("solver") or {}).get("status") or ""),
        "exact_restricted_model_optimality_proven": bool(
            (solution.get("solver") or {}).get("exact_primary_optimality_proven")
            or (solution.get("solver") or {}).get("exact_restricted_model_optimality_proven")
        ),
        "complete_structured_model_implemented": bool(
            (solution.get("model_scope") or {}).get("complete_structured_model_implemented", False)
        ),
        "counted_bend_component_count": len(counted),
        "typed_assignment_count": len(solution.get("assignments") or ()),
        "unsupported_selected_path_count": int(
            solution.get("unsupported_selected_path_count") or 0
        ),
        "birth_column_count": total_added,
        "birth_iterations": birth_iterations,
        "global_point_budget": GLOBAL_POINT_BUDGET,
        "pool_working_point_count": int(pool.get("transient_sampled_point_count") or 0),
        "solver_child_peak_rss_bytes": child_peak_rss_bytes,
        "solver_child_peak_rss_source": "getrusage_RUSAGE_CHILDREN_ru_maxrss",
        "solver_child_peak_rss_platform_units_normalized": True,
        # This is an increment in the process-wide child high-water mark, not
        # an independently sampled peak for one solver invocation.
        "solver_child_ru_maxrss_increment_since_measurement_bytes": max(
            0, child_peak_rss_bytes - start_child_rss_bytes
        ),
        "artifacts": {
            "typed_union_graph": str(final_graph_path),
            "cpsat_solution": str(final_solution_path),
            "prediction": str(output_dir / "prediction_owned_regions.json"),
        },
    }
    _write(output_dir / "inference_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
