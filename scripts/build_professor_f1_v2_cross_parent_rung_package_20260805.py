#!/usr/bin/env python3
"""Build an individual-file professor packet for the cross-parent-rung lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "output/research/f1-gold27-20260803"
MAX_FILES = 10


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def _prompt(*, mode: str, hashes: Mapping[str, str]) -> str:
    integrity = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in sorted(hashes.items())
    )
    if mode == "PREFLIGHT_BLOCKER":
        status = """The authorized experiment cannot enter its ordinary batch. Exact
frozen target `C001/LTC_32d3609a79a9c314` maps to zero complete-edge baseline
components although its modes, fragments, and pair exist. Its parent is
7.421545 mm from the fixed pair line, outside the frozen 3 mm tube. Sections 7
and 9 require every one of the exact 4,788 targets to map to exactly one
baseline pair and one baseline component; zero mappings fail closed.

No authoritative batch, replay, synthetic result, compact bundle, or empirical
STOP is claimed in this packet."""
        question = """Choose exactly one:

1. Confirm that the preregistered experiment is invalid before batch and STOP
   this LOCAL_PLANE_FRAGMENT research family; or
2. Issue a precise amended gold-free specification explaining how the immutable
   4,788-target population is related to the complete-edge baseline without
   dropping/redefining targets, fitting thresholds, or opening gold.

Do not ask us to silently remove the target or relax the line tube."""
    else:
        status = """The complete 37-case pre-gold bundle and all required replay and
synthetic attestations are attached. Read the bundle's exact ordered decision;
do not infer a different decision from aggregate counts."""
        question = """Audit the integrity criteria and exact ordered decision. If and
only if the bundle says `GO_NEXT_PROFESSOR_PACKET`, prescribe the next bounded
professor consultation. Any STOP ends this representation family at the frozen
parent/mode resolution."""
    return f"""# Fresh GPT-5.6 Pro extended-thinking review

Treat this as a fresh conversation. Read every attached file and verify its SHA.

## Status

{status}

## Decision request

{question}

## Research and novelty discipline

- Use primary sources only for research-dependent claims and provide direct DOI,
  arXiv, publisher, standards-body, or official-project citations.
- Treat `FeatureExtractionFromPointClouds.pdf` as a historical low-density
  research anchor, not proof of current novelty or applicability.
- Separate OBSERVED, INFERENCE, and UNKNOWN.
- Qualify novelty narrowly: plane fitting, graph linkage, components, region
  growing, and plane intersections are established. Do not claim publication
  priority, legal novelty, or patentability from a bounded search.

## Immutable safety boundary

`gold_open_authorized` is always false for every possible decision. Do not
authorize gold/truth parsing, ownership, births, learning, expected-count logic,
runtime integration, main-branch promotion, or production inspection. A GO can
authorize only another professor packet.

## Integrity table

{integrity}
"""


def build_package(
    *,
    output_dir: Path,
    preflight_blocker_path: Path | None,
    compact_bundle_path: Path | None,
    batch_manifest_path: Path | None,
    replay_attestation_path: Path | None,
    synthetic_attestation_path: Path | None,
    ruling_path: Path,
    core_path: Path,
    core_test_path: Path,
    memory_test_path: Path,
    bundle_builder_path: Path,
    corrected_baseline_bundle_path: Path,
    prior_causal_report_path: Path,
    historical_anchor_pdf: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("package output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    authoritative = (
        compact_bundle_path,
        batch_manifest_path,
        replay_attestation_path,
        synthetic_attestation_path,
    )
    if preflight_blocker_path is not None and any(value is not None for value in authoritative):
        raise ValueError("preflight blocker and authoritative result files are mutually exclusive")
    if preflight_blocker_path is None and not all(value is not None for value in authoritative):
        raise ValueError("provide either one preflight blocker or all four authoritative artifacts")
    mode = "PREFLIGHT_BLOCKER" if preflight_blocker_path is not None else "AUTHORITATIVE_RESULT"

    plan: list[tuple[str, Path]] = [
        ("01_PROFESSOR_RULING.md", ruling_path),
        ("06_CROSS_PARENT_RUNG_CORE.py", core_path),
        ("07_COMPACT_BUNDLE_BUILDER.py", bundle_builder_path),
        ("08_CORRECTED_COMPLETE_EDGE_BASELINE.json", corrected_baseline_bundle_path),
        ("09_FeatureExtractionFromPointClouds.pdf", historical_anchor_pdf),
    ]
    if mode == "PREFLIGHT_BLOCKER":
        plan.extend(
            [
                ("02_PREFLIGHT_INVALID_EXPERIMENT.json", preflight_blocker_path),
                ("03_PRIOR_GOLD_FREE_CAUSAL_FUNNEL.json", prior_causal_report_path),
                ("04_CORE_FAIL_CLOSED_TEST.py", core_test_path),
                ("05_MEMORY_CAP_TEST.py", memory_test_path),
            ]
        )
    else:
        plan.extend(
            [
                ("02_COMPACT_PRE_GOLD_BUNDLE.json", compact_bundle_path),
                ("03_BATCH_MANIFEST.json", batch_manifest_path),
                ("04_REPLAY_ATTESTATION.json", replay_attestation_path),
                ("05_SYNTHETIC_RESOURCE_ATTESTATION.json", synthetic_attestation_path),
            ]
        )
        bundle = _read(compact_bundle_path.resolve())
        if bundle.get("gold_open_authorized") is not False:
            raise ValueError("cross-parent-rung package cannot contain a gold-authorizing bundle")
        if bundle.get("pre_gold_decision") not in (
            "INVALID_EXPERIMENT_STOP",
            "STOP_NO_ELIGIBLE_CROSS_PARENT_RUNGS",
            "STOP_RUNGS_DO_NOT_REACH_FROZEN_TARGETS",
            "STOP_OPPOSITE_SIDE_SUPPORT_NOT_ADVANCED",
            "STOP_NO_NOVEL_TARGET_RUNG_CERTIFICATE",
            "GO_NEXT_PROFESSOR_PACKET",
        ):
            raise ValueError("compact bundle pre-gold decision is invalid")
    if len(plan) + 1 > MAX_FILES:
        raise AssertionError("professor package exceeds ten individual files")
    destinations: dict[str, Path] = {}
    for name, source in plan:
        destination = output_dir / name
        _copy(source, destination)
        destinations[name] = destination
    hashes = {name: _sha(path) for name, path in destinations.items()}
    prompt = output_dir / "00_GPT_PRO_PROMPT.md"
    prompt.write_text(_prompt(mode=mode, hashes=hashes), encoding="utf-8")
    destinations[prompt.name] = prompt
    return {
        "output_dir": str(output_dir),
        "mode": mode,
        "fresh_conversation_required": True,
        "individual_file_count": len(destinations),
        "gold_open_authorized": False,
        "files": [
            {"name": name, "path": str(path), "sha256": _sha(path)}
            for name, path in sorted(destinations.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-blocker", type=Path)
    parser.add_argument("--compact-bundle", type=Path)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--replay-attestation", type=Path)
    parser.add_argument("--synthetic-attestation", type=Path)
    parser.add_argument(
        "--ruling", type=Path,
        default=ROOT / "docs/research/PROFESSOR_F1_V2_CROSS_PARENT_RUNG_RESPONSE_20260805.md",
    )
    parser.add_argument(
        "--core", type=Path,
        default=ROOT / "backend/f1_v2_cross_parent_opposite_fragment_rung_probe.py",
    )
    parser.add_argument(
        "--bundle-builder", type=Path,
        default=ROOT / "scripts/build_f1_v2_cross_parent_opposite_fragment_rung_probe_bundle.py",
    )
    parser.add_argument(
        "--core-test", type=Path,
        default=ROOT / "tests/test_f1_v2_cross_parent_opposite_fragment_rung_probe.py",
    )
    parser.add_argument(
        "--memory-test", type=Path,
        default=ROOT / "tests/test_f1_v2_cross_parent_opposite_fragment_rung_probe_memory.py",
    )
    parser.add_argument(
        "--corrected-baseline-bundle", type=Path,
        default=RESEARCH / "v2_complete_same_surface_edge_probe_v1_bundle.json",
    )
    parser.add_argument(
        "--prior-causal-report", type=Path,
        default=RESEARCH / "v2_complete_same_surface_edge_probe_v1_causal_funnel.json",
    )
    parser.add_argument(
        "--historical-anchor-pdf", type=Path,
        default=Path("/Users/idantal/Downloads/FeatureExtractionFromPointClouds.pdf"),
    )
    args = parser.parse_args()
    result = build_package(
        output_dir=args.output_dir,
        preflight_blocker_path=args.preflight_blocker,
        compact_bundle_path=args.compact_bundle,
        batch_manifest_path=args.batch_manifest,
        replay_attestation_path=args.replay_attestation,
        synthetic_attestation_path=args.synthetic_attestation,
        ruling_path=args.ruling,
        core_path=args.core,
        core_test_path=args.core_test,
        memory_test_path=args.memory_test,
        bundle_builder_path=args.bundle_builder,
        corrected_baseline_bundle_path=args.corrected_baseline_bundle,
        prior_causal_report_path=args.prior_causal_report,
        historical_anchor_pdf=args.historical_anchor_pdf,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
