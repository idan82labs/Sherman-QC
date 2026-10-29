#!/usr/bin/env python3
"""Build the corrected <=10-file V2 mapping-preflight professor packet."""

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


MAX_FILES = 10
EXPECTED_TARGET_ROWS = 4_788
EXPECTED_PAIR_ROWS = 16_988
PREFLIGHT_KIND = "f1_v2_cross_parent_rung_target_baseline_relation_preflight_v1"
EVIDENCE_KIND = "f1_v2_cross_parent_rung_mapping_preflight_compact_evidence_v2"
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


def _validate_payloads(
    preflight_path: Path, evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    preflight = _read(preflight_path)
    evidence = _read(evidence_path)
    if (
        preflight.get("schema_version") != 1
        or preflight.get("kind") != PREFLIGHT_KIND
        or preflight.get("profile") != PROFILE
    ):
        raise ValueError("corrected authoritative preflight identity mismatch")
    if (
        evidence.get("schema_version") != 2
        or evidence.get("kind") != EVIDENCE_KIND
        or evidence.get("profile") != PROFILE
    ):
        raise ValueError("independently derived compact-v2 evidence identity mismatch")
    preflight_sha = _sha(preflight_path)
    if evidence.get("mapping_preflight_sha256") != preflight_sha:
        raise ValueError("compact evidence does not bind authoritative external SHA")
    for payload, label in ((preflight, "preflight"), (evidence, "evidence")):
        if payload.get("decision") != DECISION:
            raise ValueError(f"{label} does not authorize only the next professor packet")
        if payload.get("ordinary_batch_authorized") is not False:
            raise ValueError(f"{label} illegally authorizes ordinary batch")
        if payload.get("gold_open_authorized") is not False:
            raise ValueError(f"{label} illegally authorizes gold")
        if payload.get("next_professor_packet_authorized") is not True:
            raise ValueError(f"{label} does not authorize professor review")
        criteria = payload.get("mapping_preflight_criteria")
        if (
            not isinstance(criteria, Mapping)
            or set(criteria) != set(MAPPING_PREFLIGHT_CRITERIA)
            or any(criteria[key] is not True for key in MAPPING_PREFLIGHT_CRITERIA)
        ):
            raise ValueError(f"{label} does not carry exact source-derived 20/20 criteria")
    if (
        int(preflight.get("target_row_count") or -1) != EXPECTED_TARGET_ROWS
        or len(preflight.get("frozen_target_projection") or ()) != EXPECTED_TARGET_ROWS
        or len(preflight.get("frozen_target_baseline_relation_table") or ())
        != EXPECTED_TARGET_ROWS
        or len(preflight.get("frozen_target_outcome_table") or ())
        != EXPECTED_TARGET_ROWS
    ):
        raise ValueError("corrected preflight does not preserve exact 4,788 target tables")
    if (
        int(preflight.get("all_pair_membership_row_count") or -1)
        != EXPECTED_PAIR_ROWS
        or len(preflight.get("all_pair_membership_table") or ())
        != EXPECTED_PAIR_ROWS
    ):
        raise ValueError("corrected preflight lacks complete 16,988-pair evidence")
    if len(preflight.get("threshold_hash_table") or ()) != 37:
        raise ValueError("corrected preflight lacks exact 37-case threshold evidence")
    for field in (
        "relation_table_sha256", "outcome_table_sha256",
        "frozen_target_projection_sha256", "all_pair_membership_table_sha256",
        "threshold_hash_table_sha256",
    ):
        if preflight.get(field) != evidence.get(field):
            raise ValueError(f"compact-v2 evidence hash mismatch: {field}")
    return preflight, evidence, preflight_sha


def _prompt(
    *, hashes: Mapping[str, str], preflight_sha: str,
    present_count: int, absent_count: int,
) -> str:
    integrity = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in sorted(hashes.items())
    )
    return f"""# Fresh GPT-5.6 Pro extended-thinking review — corrected V2 mapping preflight

Treat this as a fresh conversation and read every attachment completely. The
prior mapping packet was correctly stopped by `01_INVALID_PREFLIGHT_RESPONSE.md`;
do not rely on or revive its compact evidence.

## Correction implemented

The corrected mapping-only preflight independently derives and validates:

- the exact 4,788-row frozen-target projection, target-ID set, and mode-hash set;
- the complete 16,988-row all-pair fixed-tube membership table, including the
  12,205 pairs not referenced by a frozen target;
- the exact 37-case threshold-hash table;
- source hashes, frozen control and complete-edge reproduction evidence;
- mapping-only safety predicates; and
- every one of the 20 amended criteria from underlying rows rather than trusted
  Boolean attestations.

The independent validator includes the five required tamper regressions. It
rejects a non-target pair membership change while target relations remain
untouched, equal-count target-ID exchange, equal-count mode-hash exchange, a
valid-looking wrong threshold hash, and asserted True over mismatching all-pair
evidence.

Only this mapping preflight was executed: zero rungs, replays, synthetic-resource
runs, gold/truth parsing, ownership, births, solver use, or runtime integration.

Authoritative external SHA-256: `{preflight_sha}`

Observed census:

- frozen targets: 4,788
- component image present: {present_count}
- structurally unreachable outside the frozen tube: {absent_count}
- complete baseline pair memberships: 16,988

## Decision request

Choose exactly one:

1. `AUTHORIZE_ORDINARY_V2_BATCH_SPECIFICATION`: confirm the corrected mapping
   preflight now establishes every amended predicate and provide the exact next
   bounded, gold-free ordinary V2 execution specification; or
2. `INVALID_MAPPING_PREFLIGHT_STOP`: cite the exact remaining violated row,
   hash, count, predicate, or implementation defect. No batch is authorized.

Do not drop any target, impute zero for structurally unreachable geometry, alter
the frozen 3 mm tube, fit thresholds, introduce a C001 branch, open gold, or
infer runtime/main promotion. Separate OBSERVED, INFERENCE, and UNKNOWN. Use
primary citations only for research-dependent claims.

## Attachment integrity

{integrity}
"""


def build_package(
    *, output_dir: Path, mapping_preflight_path: Path,
    compact_evidence_path: Path, invalid_response_path: Path,
    amended_ruling_path: Path, v2_core_path: Path, runner_path: Path,
    validator_path: Path, validator_test_path: Path,
    complete_edge_baseline_manifest_path: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("package output directory must be new or empty")
    preflight, _evidence, preflight_sha = _validate_payloads(
        mapping_preflight_path.resolve(), compact_evidence_path.resolve()
    )
    plan = [
        ("01_INVALID_PREFLIGHT_RESPONSE.md", invalid_response_path),
        ("02_AMENDED_MAPPING_RULING.md", amended_ruling_path),
        ("03_CORRECTED_AUTHORITATIVE_MAPPING_PREFLIGHT.json", mapping_preflight_path),
        ("04_INDEPENDENT_COMPACT_V2_EVIDENCE.json", compact_evidence_path),
        ("05_CORRECTED_V2_MAPPING_CORE.py", v2_core_path),
        ("06_CORRECTED_MAPPING_RUNNER.py", runner_path),
        ("07_INDEPENDENT_MAPPING_VALIDATOR.py", validator_path),
        ("08_REQUIRED_TAMPER_TESTS.py", validator_test_path),
        ("09_COMPLETE_EDGE_BASELINE_MANIFEST.json", complete_edge_baseline_manifest_path),
    ]
    if len(plan) + 1 != MAX_FILES:
        raise AssertionError("corrected professor packet must contain exactly ten files")
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
            preflight_sha=preflight_sha,
            present_count=int(preflight["target_component_image_present_count"]),
            absent_count=int(preflight["target_component_image_absent_outside_tube_count"]),
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
    research = ROOT / "output/research/f1-gold27-20260803"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-preflight", required=True, type=Path)
    parser.add_argument("--compact-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--invalid-response", type=Path, default=ROOT / "docs/research/PROFESSOR_F1_V2_MAPPING_PREFLIGHT_INVALID_RESPONSE_20260805.md")
    parser.add_argument("--amended-ruling", type=Path, default=ROOT / "docs/research/PROFESSOR_F1_V2_CROSS_PARENT_MAPPING_AMENDMENT_RESPONSE_20260805.md")
    parser.add_argument("--v2-core", type=Path, default=ROOT / "backend/f1_v2_cross_parent_opposite_fragment_rung_probe_v2.py")
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts/run_f1_v2_cross_parent_rung_mapping_preflight.py")
    parser.add_argument("--validator", type=Path, default=ROOT / "scripts/validate_f1_v2_cross_parent_rung_mapping_preflight.py")
    parser.add_argument("--validator-test", type=Path, default=ROOT / "tests/test_validate_f1_v2_cross_parent_rung_mapping_preflight.py")
    parser.add_argument("--complete-edge-baseline-manifest", type=Path, default=research / "v2_complete_same_surface_edge_probe_v1/manifest.json")
    args = parser.parse_args()
    result = build_package(
        output_dir=args.output_dir,
        mapping_preflight_path=args.mapping_preflight,
        compact_evidence_path=args.compact_evidence,
        invalid_response_path=args.invalid_response,
        amended_ruling_path=args.amended_ruling,
        v2_core_path=args.v2_core,
        runner_path=args.runner,
        validator_path=args.validator,
        validator_test_path=args.validator_test,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
