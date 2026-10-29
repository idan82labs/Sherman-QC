#!/usr/bin/env python3
"""Run the truth-free transition-mode census over a frozen case manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_f1_transition_mode_probe import build_transition_probe  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_case(task: tuple[str, str, str, str, str, str]) -> dict[str, Any]:
    case_token, scan_raw, pool_raw, graph_raw, solution_raw, output_raw = task
    scan = Path(scan_raw)
    pool = Path(pool_raw)
    graph = Path(graph_raw)
    solution = Path(solution_raw)
    output = Path(output_raw)
    try:
        payload = build_transition_probe(
            scan_path=scan,
            candidate_pool_path=pool,
            graph_path=graph,
            a0_solution_path=solution,
        )
    except Exception as exc:
        raise RuntimeError(f"{case_token}: transition-mode probe failed: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "case_token": case_token,
        "path": str(output.resolve()),
        "output_sha256": _sha256(output),
        "scan_sha256": payload["scan_sha256"],
        "candidate_pool_sha256": payload["candidate_pool_sha256"],
        "graph_sha256": payload["graph_sha256"],
        "a0_solution_sha256": payload["a0_solution_sha256"],
        "plane_pair_record_count": len(payload.get("plane_pair_records") or ()),
        "multimodal_parent_record_count": len(
            payload.get("multimodal_parent_records") or ()
        ),
        "transition_component_record_count": len(
            payload.get("transition_component_records") or ()
        ),
        "birth_eligibility_record_count": len(
            payload.get("birth_eligibility_records") or ()
        ),
        "eligible_certificate_count": sum(
            bool(record.get("geometry_certificate_valid"))
            for record in payload.get("birth_eligibility_records") or ()
        ),
    }


def run_batch(
    *,
    inference_manifest_path: Path,
    candidate_pool_manifest_path: Path,
    structured_root: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > 18:
        raise ValueError("workers must be in [1,18]")
    inference_manifest = _read(inference_manifest_path)
    pool_manifest = _read(candidate_pool_manifest_path)
    pools = {
        str(row["case_token"]): Path(str(row["path"])).resolve()
        for row in pool_manifest.get("cases") or ()
    }
    tasks: list[tuple[str, str, str, str, str, str]] = []
    for row in inference_manifest.get("cases") or ():
        case_token = str(row["case_token"])
        pool_path = pools.get(case_token)
        if pool_path is None:
            raise ValueError(f"{case_token}: missing V2 pool")
        case_root = structured_root / "cases" / case_token
        tasks.append(
            (
                case_token,
                str(Path(str(row["source_scan_path"])).resolve()),
                str(pool_path),
                str((case_root / "typed_union_graph.json").resolve()),
                str((case_root / "cpsat_solution.json").resolve()),
                str((output_dir / f"{case_token}.json").resolve()),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        rows = [_run_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_case, tasks))
    rows.sort(key=lambda row: row["case_token"])
    manifest = {
        "schema_version": 1,
        "kind": "f1_transition_mode_probe_batch",
        "inference_safe": True,
        "birth_columns_enabled": False,
        "prediction_changed": False,
        "case_count": len(rows),
        "workers": int(workers),
        "inference_manifest_sha256": _sha256(inference_manifest_path),
        "candidate_pool_manifest_sha256": _sha256(candidate_pool_manifest_path),
        "structured_root": str(structured_root.resolve()),
        "cases": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--structured-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = run_batch(
        inference_manifest_path=args.inference_manifest.resolve(),
        candidate_pool_manifest_path=args.candidate_pool_manifest.resolve(),
        structured_root=args.structured_root.resolve(),
        output_dir=args.output_dir.resolve(),
        workers=args.workers,
    )
    print(json.dumps({"case_count": manifest["case_count"], "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
