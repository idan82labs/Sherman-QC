#!/usr/bin/env python3
"""Bundle frozen transition-probe and A0 inference artifacts for post-lock use.

This command reads no labels or truth.  It verifies every census artifact and
the corresponding structured A0 graph/solution, then emits two deterministic
inference-only bundles consumable by ``attribute_f1_transition_probe_gold.py``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CENSUS_BUNDLE_KIND = "f1_transition_diagnostic_census_bundle_v1"
PREDICTION_BUNDLE_KIND = "f1_transition_probe_prediction_bundle_v1"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _case_rows(payload: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    raw_rows = payload.get("cases") or ()
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in rows:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        rows[token] = dict(raw)
    return rows


def _assert_hash(actual: str, expected: Any, *, label: str) -> None:
    if actual != str(expected or "").lower():
        raise ValueError(f"{label} SHA-256 mismatch")


def _bend_eligible(path: Mapping[str, Any]) -> bool:
    if not bool(path.get("incidence_complete")):
        return False
    eligibility = path.get("label_eligibility")
    return not isinstance(eligibility, Mapping) or bool(
        eligibility.get("BEND_STRAIGHT_COUNTED")
    )


def _stable_records(
    values: Sequence[Mapping[str, Any]], *, id_fields: Sequence[str], label: str
) -> list[dict[str, Any]]:
    rows = [deepcopy(dict(value)) for value in values]

    def identifier(row: Mapping[str, Any]) -> str:
        for field in id_fields:
            value = str(row.get(field) or "")
            if value:
                return value
        raise ValueError(f"{label} record lacks a stable identifier")

    rows.sort(key=identifier)
    ids = [identifier(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} identifiers must be unique")
    return rows


def build_postlock_bundles(
    *,
    probe_manifest_path: Path,
    structured_root: Path,
    grouped_split_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return verified census and prediction bundles without reading truth."""

    manifest_path = probe_manifest_path.resolve()
    structured_root = structured_root.resolve()
    split_path = grouped_split_path.resolve()
    manifest = _read(manifest_path)
    split = _read(split_path)
    if manifest.get("inference_safe") is not True:
        raise ValueError("transition probe manifest must declare inference_safe=true")
    if manifest.get("birth_columns_enabled") is not False:
        raise ValueError("transition probe bundling requires births disabled")
    if manifest.get("prediction_changed") is not False:
        raise ValueError("transition probe manifest must declare prediction_changed=false")

    manifest_cases = _case_rows(manifest, label="probe manifest")
    split_cases = _case_rows(split, label="grouped split")
    if set(manifest_cases) != set(split_cases):
        raise ValueError("probe manifest and grouped split case-token sets must match")
    if int(manifest.get("case_count") or 0) != len(manifest_cases):
        raise ValueError("probe manifest case_count disagrees with case rows")

    census_cases: list[dict[str, Any]] = []
    prediction_cases: list[dict[str, Any]] = []
    for token in sorted(manifest_cases):
        manifest_row = manifest_cases[token]
        census_path = Path(str(manifest_row.get("path") or "")).expanduser().resolve()
        if not census_path.is_file():
            raise FileNotFoundError(f"{token}: missing census artifact: {census_path}")
        census_sha = _sha256(census_path)
        _assert_hash(
            census_sha,
            manifest_row.get("output_sha256"),
            label=f"{token} census artifact",
        )
        census = _read(census_path)
        if census.get("inference_safe") is not True:
            raise ValueError(f"{token}: census must declare inference_safe=true")
        if census.get("truth_inputs_consumed") is not False:
            raise ValueError(f"{token}: census must declare truth_inputs_consumed=false")
        if census.get("prediction_mutated") not in (None, False):
            raise ValueError(f"{token}: transition probe mutated prediction state")
        existing_token = census.get("case_token")
        if existing_token not in (None, token):
            raise ValueError(f"{token}: census case_token disagrees with manifest")

        for field in (
            "scan_sha256",
            "candidate_pool_sha256",
            "graph_sha256",
            "a0_solution_sha256",
        ):
            _assert_hash(
                str(census.get(field) or "").lower(),
                manifest_row.get(field),
                label=f"{token} manifest {field}",
            )

        case_root = structured_root / "cases" / token
        graph_path = case_root / "typed_union_graph.json"
        solution_path = case_root / "cpsat_solution.json"
        prediction_path = case_root / "prediction_owned_regions.json"
        for label, path in (
            ("typed union graph", graph_path),
            ("A0 solution", solution_path),
            ("A0 prediction", prediction_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        _assert_hash(
            _sha256(graph_path), census.get("graph_sha256"), label=f"{token} graph"
        )
        _assert_hash(
            _sha256(solution_path),
            census.get("a0_solution_sha256"),
            label=f"{token} A0 solution",
        )

        graph = _read(graph_path)
        prediction = _read(prediction_path)
        paths = _stable_records(
            [
                dict(path)
                for path in graph.get("latent_paths") or ()
                if isinstance(path, Mapping) and _bend_eligible(path)
            ],
            id_fields=("latent_path_id",),
            label=f"{token} existing path",
        )
        a0_predictions = _stable_records(
            [
                dict(record)
                for record in prediction.get("owned_bend_regions") or ()
                if isinstance(record, Mapping)
            ],
            id_fields=("bend_id", "candidate_id"),
            label=f"{token} A0 prediction",
        )

        bundled_census = deepcopy(census)
        bundled_census["case_token"] = token
        census_cases.append(bundled_census)
        prediction_cases.append(
            {
                "case_token": token,
                "scan_sha256": str(census["scan_sha256"]),
                "candidate_pool_sha256": str(census["candidate_pool_sha256"]),
                "graph_sha256": str(census["graph_sha256"]),
                "a0_solution_sha256": str(census["a0_solution_sha256"]),
                "prediction_owned_regions_sha256": _sha256(prediction_path),
                "a0_predictions": a0_predictions,
                "existing_paths": paths,
            }
        )

    common = {
        "schema_version": 1,
        "inference_safe": True,
        "research_only": True,
        "truth_inputs_consumed": False,
        "birth_columns_enabled": False,
        "prediction_changed": False,
        "case_count": len(census_cases),
        "probe_manifest_sha256": _sha256(manifest_path),
        "grouped_split_sha256": _sha256(split_path),
    }
    census_bundle = {
        **common,
        "kind": CENSUS_BUNDLE_KIND,
        "cases": census_cases,
    }
    prediction_bundle = {
        **common,
        "kind": PREDICTION_BUNDLE_KIND,
        "cases": prediction_cases,
    }
    return census_bundle, prediction_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--structured-root", required=True, type=Path)
    parser.add_argument("--grouped-split", required=True, type=Path)
    parser.add_argument("--census-output", required=True, type=Path)
    parser.add_argument("--prediction-output", required=True, type=Path)
    args = parser.parse_args()
    census, predictions = build_postlock_bundles(
        probe_manifest_path=args.probe_manifest,
        structured_root=args.structured_root,
        grouped_split_path=args.grouped_split,
    )
    _write(args.census_output, census)
    _write(args.prediction_output, predictions)
    print(
        json.dumps(
            {
                "case_count": census["case_count"],
                "census_output": str(args.census_output.resolve()),
                "census_sha256": _sha256(args.census_output),
                "prediction_output": str(args.prediction_output.resolve()),
                "prediction_sha256": _sha256(args.prediction_output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
