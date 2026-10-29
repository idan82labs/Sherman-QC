#!/usr/bin/env python3
"""Build the <=10-file fresh-chat packet for the amended V2 mapping preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_f1_v2_cross_parent_rung_mapping_preflight import (  # noqa: E402
    MAPPING_PREFLIGHT_CRITERIA,
)

RESEARCH = ROOT / "output/research/f1-gold27-20260803"
MAX_FILES = 10
EXPECTED_ROWS = 4_788
PREFLIGHT_KIND = "f1_v2_cross_parent_rung_target_baseline_relation_preflight_v1"
EVIDENCE_KIND = "f1_v2_cross_parent_rung_mapping_preflight_compact_evidence_v1"
PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V2"
DECISION = "GO_NEXT_PROFESSOR_PACKET"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _require_safe_go(payload: Mapping[str, Any], *, kind: str, label: str) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != kind
        or payload.get("profile") != PROFILE
    ):
        raise ValueError(f"{label} is not the amended V2 mapping-preflight identity")
    if payload.get("decision") != DECISION:
        raise ValueError(f"{label} is not an authoritative mapping-preflight GO")
    for field in (
        "ordinary_batch_authorized", "designated_replays_authorized",
        "full_synthetic_resource_run_authorized", "compact_bundle_authorized",
        "gold_open_authorized",
    ):
        if field in payload and payload.get(field) is not False:
            raise ValueError(f"{label} illegally authorizes {field}")
    if payload.get("next_professor_packet_authorized") is not True:
        raise ValueError(f"{label} does not authorize the next professor packet")


def _prompt(
    *, hashes: Mapping[str, str], present_count: int, absent_count: int,
    preflight_sha256: str,
) -> str:
    integrity = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in sorted(hashes.items())
    )
    return f"""# Fresh GPT-5.6 Pro extended-thinking review — amended V2 mapping preflight

Treat this as a fresh conversation. Read every attached file completely before
answering. Verify every attachment against the SHA-256 integrity table below.

## What was executed

Only the mapping-only preflight authorized by the amended ruling was executed.
No cross-parent rung was built. No ordinary batch, designated replay, full
synthetic-resource run, compact result aggregation, gold/truth access, ownership,
birth column, solver, fitting, runtime integration, or production inspection was
performed or authorized.

The authoritative artifact contains the full untruncated 4,788-row relation
table and full untruncated 4,788-row outcome table. Its external SHA-256 is:
`{preflight_sha256}`.

Observed mapping census:

- exact frozen targets: 4,788
- component image present: {present_count}
- structurally unreachable because parent is outside the frozen pair tube: {absent_count}
- total: {present_count + absent_count}

The compact evidence is independently derived from the authoritative artifact;
it is not a substitute for the full tables. The C001 locked regression and all
20 mapping-preflight criteria must be audited against the amended ruling.

## Decision request

Choose exactly one:

1. `AUTHORIZE_ORDINARY_V2_BATCH_SPECIFICATION`: confirm the amended mapping
   preflight is valid and provide the precise, bounded, gold-free ordinary V2
   execution specification, including full target-outcome accounting for both
   relation states; or
2. `INVALID_MAPPING_PREFLIGHT_STOP`: identify the exact violated invariant,
   field, row, hash, count, or implementation defect and prescribe the smallest
   mapping-only correction. Do not authorize a batch when choosing this option.

Do not reinterpret a structurally unreachable target as dropped, negative,
successful, or zero-valued geometry. Do not change the frozen 3 mm tube, fit a
threshold, remove any frozen target, or introduce a C001-specific branch.

## Research and novelty discipline

- Use primary sources only for research-dependent claims and give direct DOI,
  arXiv, publisher, standards-body, or official-project citations.
- Treat `FeatureExtractionFromPointClouds.pdf` only as a historical low-density
  anchor, not evidence of present applicability, publication priority, legal
  novelty, or patentability.
- Separate OBSERVED, INFERENCE, and UNKNOWN.

## Safety boundary

This packet itself authorizes only professor review. Even a favorable answer
does not open gold or authorize runtime/main promotion. Those remain separately
gated. Do not request or infer access to manual labels, expected bend counts, or
sealed evaluation data.

## Attachment integrity

{integrity}
"""


def build_package(
    *, output_dir: Path, mapping_preflight_path: Path,
    compact_evidence_path: Path, ruling_path: Path, v2_core_path: Path,
    runner_path: Path, validator_path: Path, validator_test_path: Path,
    complete_edge_baseline_manifest_path: Path, historical_anchor_pdf: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("package output directory must be new or empty")

    mapping_preflight_path = mapping_preflight_path.resolve()
    compact_evidence_path = compact_evidence_path.resolve()
    preflight = _read(mapping_preflight_path)
    evidence = _read(compact_evidence_path)
    _require_safe_go(preflight, kind=PREFLIGHT_KIND, label="mapping preflight")
    _require_safe_go(evidence, kind=EVIDENCE_KIND, label="compact evidence")
    preflight_sha = _sha(mapping_preflight_path)
    if evidence.get("mapping_preflight_sha256") != preflight_sha:
        raise ValueError("compact evidence does not bind the authoritative external SHA")
    for payload, label in ((preflight, "mapping preflight"), (evidence, "compact evidence")):
        if int(payload.get("target_row_count") or -1) != EXPECTED_ROWS:
            raise ValueError(f"{label} does not preserve exact 4,788 targets")
        present = int(payload.get("target_component_image_present_count") or -1)
        absent = int(payload.get("target_component_image_absent_outside_tube_count") or -1)
        if present < 0 or absent < 0 or present + absent != EXPECTED_ROWS:
            raise ValueError(f"{label} relation census is invalid")
    relations = preflight.get("frozen_target_baseline_relation_table")
    outcomes = preflight.get("frozen_target_outcome_table")
    if not isinstance(relations, list) or len(relations) != EXPECTED_ROWS:
        raise ValueError("authoritative relation table is not exact/untruncated")
    if not isinstance(outcomes, list) or len(outcomes) != EXPECTED_ROWS:
        raise ValueError("authoritative outcome table is not exact/untruncated")
    for payload, label in ((preflight, "mapping preflight"), (evidence, "compact evidence")):
        criteria = payload.get("mapping_preflight_criteria")
        if (
            not isinstance(criteria, Mapping)
            or set(criteria) != set(MAPPING_PREFLIGHT_CRITERIA)
            or any(criteria[key] is not True for key in MAPPING_PREFLIGHT_CRITERIA)
        ):
            raise ValueError(f"{label} criteria are not the exact all-true set")
    if (
        preflight.get("relation_table_sha256") != evidence.get("relation_table_sha256")
        or preflight.get("outcome_table_sha256") != evidence.get("outcome_table_sha256")
    ):
        raise ValueError("compact evidence table hashes do not match authoritative tables")

    plan = [
        ("01_AMENDED_PROFESSOR_RULING.md", ruling_path),
        ("02_AUTHORITATIVE_MAPPING_PREFLIGHT.json", mapping_preflight_path),
        ("03_COMPACT_VALIDATOR_EVIDENCE.json", compact_evidence_path),
        ("04_V2_MAPPING_CORE.py", v2_core_path),
        ("05_MAPPING_PREFLIGHT_RUNNER.py", runner_path),
        ("06_MAPPING_PREFLIGHT_VALIDATOR.py", validator_path),
        ("07_MAPPING_PREFLIGHT_VALIDATOR_TEST.py", validator_test_path),
        ("08_COMPLETE_EDGE_BASELINE_MANIFEST.json", complete_edge_baseline_manifest_path),
        ("09_FeatureExtractionFromPointClouds.pdf", historical_anchor_pdf),
    ]
    if len(plan) + 1 > MAX_FILES:
        raise AssertionError("professor package exceeds ten individual files")
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations: dict[str, Path] = {}
    for name, source in plan:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / name
        shutil.copy2(source, destination)
        destinations[name] = destination
    hashes = {name: _sha(path) for name, path in destinations.items()}
    prompt_path = output_dir / "00_GPT_PRO_PROMPT.md"
    prompt_path.write_text(
        _prompt(
            hashes=hashes,
            present_count=int(preflight["target_component_image_present_count"]),
            absent_count=int(preflight["target_component_image_absent_outside_tube_count"]),
            preflight_sha256=preflight_sha,
        ),
        encoding="utf-8",
    )
    destinations[prompt_path.name] = prompt_path
    return {
        "output_dir": str(output_dir),
        "fresh_conversation_required": True,
        "individual_file_count": len(destinations),
        "mapping_preflight_sha256": preflight_sha,
        "ordinary_batch_authorized": False,
        "gold_open_authorized": False,
        "files": [
            {"name": name, "path": str(path), "sha256": _sha(path)}
            for name, path in sorted(destinations.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-preflight", required=True, type=Path)
    parser.add_argument("--compact-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ruling", type=Path, default=ROOT / "docs/research/PROFESSOR_F1_V2_CROSS_PARENT_MAPPING_AMENDMENT_RESPONSE_20260805.md")
    parser.add_argument("--v2-core", type=Path, default=ROOT / "backend/f1_v2_cross_parent_opposite_fragment_rung_probe_v2.py")
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts/run_f1_v2_cross_parent_rung_mapping_preflight.py")
    parser.add_argument("--validator", type=Path, default=ROOT / "scripts/validate_f1_v2_cross_parent_rung_mapping_preflight.py")
    parser.add_argument("--validator-test", type=Path, default=ROOT / "tests/test_validate_f1_v2_cross_parent_rung_mapping_preflight.py")
    parser.add_argument("--complete-edge-baseline-manifest", type=Path, default=RESEARCH / "v2_complete_same_surface_edge_probe_v1/manifest.json")
    parser.add_argument("--historical-anchor-pdf", type=Path, default=Path("/Users/idantal/Downloads/FeatureExtractionFromPointClouds.pdf"))
    args = parser.parse_args()
    result = build_package(
        output_dir=args.output_dir,
        mapping_preflight_path=args.mapping_preflight,
        compact_evidence_path=args.compact_evidence,
        ruling_path=args.ruling,
        v2_core_path=args.v2_core,
        runner_path=args.runner,
        validator_path=args.validator,
        validator_test_path=args.validator_test,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
        historical_anchor_pdf=args.historical_anchor_pdf,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
