#!/usr/bin/env python3
"""Build the ten-file fresh-chat packet for the validated ordinary V2 funnel."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MAX_FILES = 10
EXPECTED_ARTIFACT_SHA256 = (
    "aa2fd3b083bb18d1a6992a959b42bd35becce2f77bd25c4999d84c463aa7118e"
)
EXPECTED_COMPACT_SHA256 = (
    "df2911e6b2b31dc65224c678ce8214625e5d6f3e25fe70d9d9b3764b1679266f"
)
EXPECTED_FUNNEL = {
    "aggregate_added_cross_parent_rung_count": 161,
    "target_reached_by_added_rung_count": 122,
    "target_side_gate_advanced_count": 0,
    "target_novel_rung_attributable_certificate_count": 0,
    "target_component_image_present_count": 4_772,
    "target_component_image_absent_outside_tube_count": 16,
    "frozen_target_count": 4_788,
    "target_eligible_pair_count": 4_783,
    "all_pair_membership_row_count": 16_988,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
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


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty values")
    # Frozen reporting convention: lower empirical order statistic.  For the
    # 37-case ordinary batch, P95 is the third-slowest case at 294.02591 s.
    position = math.floor((len(ordered) - 1) * percentile)
    return ordered[position]


def derive_manifest_digest(
    artifact: Mapping[str, Any], *, artifact_path: Path, artifact_sha256: str,
    compact_sha256: str,
) -> dict[str, Any]:
    cases = list(artifact.get("cases") or ())
    case_payloads = list(artifact.get("case_payloads") or ())
    if len(cases) != 37 or len(case_payloads) != 37:
        raise ValueError("ordinary artifact must contain exact 37 case rows/payloads")
    payloads_by_token = {
        str(payload["case_token"]): dict(payload) for payload in case_payloads
    }
    if len(payloads_by_token) != 37:
        raise ValueError("ordinary embedded case tokens are not exact/unique")
    counters = dict(artifact.get("aggregate_counters") or {})
    if any(int(counters.get(key, -1)) != value for key, value in EXPECTED_FUNNEL.items()):
        raise ValueError("ordinary empirical funnel differs from validated result")
    wall_seconds = [float(row["wall_seconds"]) for row in cases]
    case_census = []
    for row in sorted(cases, key=lambda item: str(item["case_token"])):
        token = str(row["case_token"])
        payload = payloads_by_token[token]
        payload_counters = dict(payload["case_counters"])
        case_census.append({
            "case_token": token,
            "status": row["status"],
            "case_artifact_sha256": row["case_artifact_sha256"],
            "embedded_case_payload_sha256": _sha_json(payload),
            "pre_rung_gate_sha256": row["pre_rung_gate_sha256"],
            "deletion_evidence_sha256": row["deletion_evidence_sha256"],
            "wall_seconds": float(row["wall_seconds"]),
            "peak_rss_bytes": int(row["peak_rss_bytes"]),
            "serialized_output_bytes": int(row["serialized_output_bytes"]),
            "frozen_target_count": int(payload_counters["frozen_target_count"]),
            "target_eligible_pair_count": int(
                payload_counters["target_eligible_pair_count"]
            ),
            "added_cross_parent_rung_count": int(
                payload_counters["added_cross_parent_rung_count"]
            ),
            "target_reached_by_added_rung_count": int(
                payload_counters["target_reached_by_added_rung_count"]
            ),
            "target_side_gate_advanced_count": int(
                payload_counters["target_side_gate_advanced_count"]
            ),
            "target_novel_rung_attributable_certificate_count": int(
                payload_counters[
                    "target_novel_rung_attributable_certificate_count"
                ]
            ),
        })
    aggregate_tables = dict(artifact["aggregate_tables"])
    table_counts = {name: len(rows) for name, rows in aggregate_tables.items()}
    table_hashes = dict(artifact["aggregate_table_hashes"])
    if any(
        table_hashes.get(f"{name}_sha256") != _sha_json(rows)
        for name, rows in aggregate_tables.items()
    ):
        raise ValueError("ordinary aggregate table hash mismatch during digest")
    longest = sorted(
        (
            {"case_token": row["case_token"], "wall_seconds": row["wall_seconds"]}
            for row in case_census
        ),
        key=lambda row: (-float(row["wall_seconds"]), str(row["case_token"])),
    )[:8]
    digest = {
        "schema_version": 1,
        "kind": "f1_v2_ordinary_authoritative_manifest_digest_v1",
        "profile": artifact["profile"],
        "packet_status": artifact["packet_status"],
        "research_only": True,
        "inference_safe": True,
        "truth_inputs_consumed": False,
        "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "gold_open_authorized": False,
        "runtime_integration_authorized": False,
        "next_professor_packet_authorized": True,
        "authoritative_artifact_local_path": str(artifact_path.resolve()),
        "authoritative_artifact_bytes": artifact_path.stat().st_size,
        "authoritative_artifact_sha256": artifact_sha256,
        "independent_compact_evidence_sha256": compact_sha256,
        "authoritative_artifact_attached": False,
        "authoritative_artifact_retained_untruncated": True,
        "case_count": 37,
        "successful_case_count": 37,
        "aggregate_counters": deepcopy(counters),
        "maximum_peak_rss_bytes": int(artifact["maximum_peak_rss_bytes"]),
        "timing": {
            "case_wall_seconds_sum": round(sum(wall_seconds), 6),
            "case_wall_seconds_p95_lower": round(
                _percentile(wall_seconds, 0.95), 6
            ),
            "case_wall_seconds_max": max(wall_seconds),
            "slowest_case_token": max(
                cases, key=lambda row: float(row["wall_seconds"])
            )["case_token"],
            "longest_cases": longest,
        },
        "aggregate_table_row_counts": table_counts,
        "aggregate_table_hashes": table_hashes,
        "ordinary_v2_criteria": deepcopy(artifact["ordinary_v2_criteria"]),
        "all_ordinary_v2_criteria_valid": artifact[
            "all_ordinary_v2_criteria_valid"
        ],
        "case_census_sha256": _sha_json(case_census),
        "case_census": case_census,
    }
    digest["manifest_digest_sha256"] = hashlib.sha256(
        _canonical_bytes(digest)
    ).hexdigest()
    return digest


def _artifact_statement(
    *, artifact_path: Path, artifact_sha: str, compact_sha: str,
    digest_sha: str,
) -> str:
    return f"""# Full ordinary V2 authoritative artifact statement

The complete authoritative artifact is retained locally and untruncated at:

`{artifact_path.resolve()}`

- Size: `{artifact_path.stat().st_size}` bytes
- External SHA-256: `{artifact_sha}`
- Independent compact-evidence SHA-256: `{compact_sha}`
- Derived manifest/case-census SHA-256: `{digest_sha}`

The authoritative artifact is intentionally not attached because it is roughly
1.1 GB. It contains the complete embedded case payloads and complete aggregate
tables. The independent validator source-rebuilt all 37 cases, compared every
persisted case to its embedded payload, validated raw deletion restoration, and
recomputed the aggregate funnel before emitting compact evidence.

This omission is transport-only. It is not truncation of the local evidence and
does not authorize accepting a count or digest in place of the locally retained
full artifact during reproducibility review.
"""


def _prompt(*, hashes: Mapping[str, str], digest: Mapping[str, Any]) -> str:
    integrity = "\n".join(
        f"- `{name}`: `{digest_value}`"
        for name, digest_value in sorted(hashes.items())
    )
    counters = digest["aggregate_counters"]
    timing = digest["timing"]
    return f"""# Fresh GPT-5.6 Pro extended-thinking review — ordinary V2 funnel

Treat this as a fresh conversation. Read all nine attachments completely and
verify their SHA-256 values below before deciding.

## Validated observation

The one authorized ordinary V2 batch completed and the independent validator
source-rebuilt all 37 cases. The complete 1.1 GB authoritative artifact remains
local and untruncated; it is not attached. Its external SHA and derived complete
case/table census are attached separately.

- Added cross-parent rungs: {counters['aggregate_added_cross_parent_rung_count']}
- Frozen targets reached by an added rung: {counters['target_reached_by_added_rung_count']} / 4,788
- Target side gates advanced: {counters['target_side_gate_advanced_count']} / 4,788
- Novel attributable certificates: {counters['target_novel_rung_attributable_certificate_count']} / 4,788
- Component images present / structurally unreachable: {counters['target_component_image_present_count']} / {counters['target_component_image_absent_outside_tube_count']}
- Maximum RSS: {digest['maximum_peak_rss_bytes']} bytes
- P95 case wall time (lower empirical percentile): {timing['case_wall_seconds_p95_lower']} seconds
- Slowest case: {timing['slowest_case_token']} at {timing['case_wall_seconds_max']} seconds

The ordinary result is integrity-valid, but it produced zero side-gate advances
and zero novel attributable certificates. No gold, replay, synthetic-resource,
or runtime lane was opened.

## Choose exactly one

1. `AUTHORIZE_ONE_BOUNDED_GOLD_FREE_NEXT_STEP` — specify exactly one narrowly
   bounded next execution, its immutable inputs, success/STOP criteria, memory
   envelope, required deletion/counterfactual evidence, and what remains
   forbidden. Do not silently authorize multiple sequential lanes; or
2. `FORMAL_EMPIRICAL_LOCAL_PLANE_FRAGMENT_FAMILY_STOP` — conclude that this
   frozen LOCAL_PLANE_FRAGMENT representation/intervention family has failed
   its ordinary empirical objective, while separating that result from claims
   about all possible bend-detection approaches.

Do not open gold/truth, designated replays, the full synthetic-resource envelope,
learning, threshold fitting, ownership, births, solver use, runtime integration,
main promotion, or production inspection unless your single explicit bounded
authorization names that exact gold-free lane. Gold must remain closed for every
decision.

## Research grounding request

Use primary sources only for research-dependent claims: original papers,
standards, official project documentation, DOI/publisher/arXiv records. Separate
OBSERVED, INFERENCE, and UNKNOWN. Explain how the observed failure relates to
established plane fitting, region growing, graph connectivity, crease extraction,
and counterfactual attribution. Make no publication-priority, legal-novelty,
patentability, or broad first-of-kind claim from this bounded experiment.

## Attachment integrity

{integrity}
"""


def build_package(
    *, output_dir: Path, authoritative_artifact_path: Path,
    compact_evidence_path: Path, authorization_response_path: Path,
    runner_path: Path, core_path: Path, validator_path: Path,
    runner_test_path: Path, validator_test_path: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("package output directory must be new or empty")
    authoritative_artifact_path = authoritative_artifact_path.resolve()
    compact_evidence_path = compact_evidence_path.resolve()
    artifact_sha = _sha_file(authoritative_artifact_path)
    compact_sha = _sha_file(compact_evidence_path)
    if artifact_sha != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("ordinary authoritative artifact external SHA mismatch")
    if compact_sha != EXPECTED_COMPACT_SHA256:
        raise ValueError("ordinary compact evidence external SHA mismatch")
    artifact = _read(authoritative_artifact_path)
    compact = _read(compact_evidence_path)
    if compact.get("ordinary_authoritative_sha256") != artifact_sha:
        raise ValueError("compact evidence does not bind authoritative artifact")
    digest = derive_manifest_digest(
        artifact, artifact_path=authoritative_artifact_path,
        artifact_sha256=artifact_sha, compact_sha256=compact_sha,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = [
        ("01_PRIOR_PROFESSOR_AUTHORIZATION.md", authorization_response_path),
        ("02_INDEPENDENT_COMPACT_EVIDENCE.json", compact_evidence_path),
        ("03_ORDINARY_V2_RUNNER.py", runner_path),
        ("04_ORDINARY_V2_CORE.py", core_path),
        ("05_INDEPENDENT_VALIDATOR.py", validator_path),
        ("06_RUNNER_TAMPER_TESTS.py", runner_test_path),
        ("07_VALIDATOR_TAMPER_TESTS.py", validator_test_path),
    ]
    destinations: dict[str, Path] = {}
    for name, source in plan:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / name
        shutil.copy2(source, destination)
        destinations[name] = destination
    digest_path = output_dir / "08_AUTHORITATIVE_MANIFEST_CASE_CENSUS.json"
    digest_path.write_bytes(_canonical_bytes(digest) + b"\n")
    destinations[digest_path.name] = digest_path
    statement_path = output_dir / "09_FULL_ARTIFACT_SHA_STATEMENT.md"
    statement_path.write_text(
        _artifact_statement(
            artifact_path=authoritative_artifact_path,
            artifact_sha=artifact_sha, compact_sha=compact_sha,
            digest_sha=_sha_file(digest_path),
        ),
        encoding="utf-8",
    )
    destinations[statement_path.name] = statement_path
    hashes = {name: _sha_file(path) for name, path in destinations.items()}
    prompt_path = output_dir / "00_GPT_PRO_PROMPT.md"
    prompt_path.write_text(_prompt(hashes=hashes, digest=digest), encoding="utf-8")
    destinations[prompt_path.name] = prompt_path
    if len(destinations) != MAX_FILES:
        raise AssertionError("ordinary professor packet must contain exactly ten files")
    return {
        "output_dir": str(output_dir),
        "fresh_conversation_required": True,
        "individual_file_count": len(destinations),
        "authoritative_artifact_attached": False,
        "authoritative_artifact_sha256": artifact_sha,
        "compact_evidence_sha256": compact_sha,
        "gold_open_authorized": False,
        "files": [
            {"name": name, "path": str(path), "sha256": _sha_file(path)}
            for name, path in sorted(destinations.items())
        ],
    }


def main() -> int:
    research = ROOT / "output/research/f1-gold27-20260803"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--authoritative-artifact", type=Path, default=research / "v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2.json")
    parser.add_argument("--compact-evidence", type=Path, default=research / "v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2_compact_evidence.json")
    parser.add_argument("--authorization-response", type=Path, default=ROOT / "docs/research/PROFESSOR_F1_V2_CORRECTED_MAPPING_PREFLIGHT_AUTHORIZATION_RESPONSE_20260805.md")
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts/run_f1_v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2.py")
    parser.add_argument("--core", type=Path, default=ROOT / "backend/f1_v2_cross_parent_opposite_fragment_rung_probe_v2.py")
    parser.add_argument("--validator", type=Path, default=ROOT / "scripts/validate_f1_v2_cross_parent_ordinary_batch.py")
    parser.add_argument("--runner-tests", type=Path, default=ROOT / "tests/test_run_f1_v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2.py")
    parser.add_argument("--validator-tests", type=Path, default=ROOT / "tests/test_validate_f1_v2_cross_parent_ordinary_batch.py")
    args = parser.parse_args()
    result = build_package(
        output_dir=args.output_dir,
        authoritative_artifact_path=args.authoritative_artifact,
        compact_evidence_path=args.compact_evidence,
        authorization_response_path=args.authorization_response,
        runner_path=args.runner, core_path=args.core,
        validator_path=args.validator,
        runner_test_path=args.runner_tests,
        validator_test_path=args.validator_tests,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
