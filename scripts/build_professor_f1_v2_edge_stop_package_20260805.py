#!/usr/bin/env python3
"""Assemble a compact fresh-chat GPT-5.6 Pro V2 edge-STOP package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_ROOT = ROOT / "output/research/f1-gold27-20260803"
DEFAULT_HISTORICAL_ANCHOR = Path(
    "/Users/idantal/Downloads/FeatureExtractionFromPointClouds.pdf"
)
MAX_UPLOAD_FILES = 10


def _sha256(path: Path) -> str:
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


def _prompt(
    *,
    causal_report: Mapping[str, Any],
    upload_hashes: Mapping[str, str],
    corrected_bundle_known: bool,
) -> str:
    aggregate = dict(causal_report.get("aggregate") or {})
    bundle_status = (
        "The corrected compact bundle and its externally computed SHA are attached."
        if corrected_bundle_known
        else (
            "02_CORRECTED_BUNDLE_HASH_PENDING.txt is an explicit placeholder. "
            "Do not issue an implementation authorization until it is replaced by the "
            "corrected compact bundle and frozen external SHA."
        )
    )
    hashes = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in sorted(upload_hashes.items())
    )
    return f"""# Fresh GPT-5.6 Pro extended-thinking consultation

You are the independent research professor for Sherman QC F1, a deterministic,
zero-shot 3D scan bend-inspection system. Treat this as a fresh conversation.
Do not rely on prior chat memory. Read every attached file before deciding.

## Evidence boundary

- This packet is gold-free. No sealed bend truth, expected counts, ownership,
  learning, family routing, or runtime mutations were opened or consumed.
- The tested intervention removes mutual-top-eight pruning only for eligible
  SAME_SURFACE mode edges. Frozen CROSS_MODE_LOCAL transition seeds remain
  byte-identical.
- {bundle_status}
- The historical PDF predates modern dense point clouds. Use it as a research
  anchor, not as proof of novelty or as evidence that its density assumptions
  still apply.

## Locked observations to explain

- Added SAME_SURFACE edges: {aggregate.get('added_same_surface_edge_count')}
- Accepted added union edges: {aggregate.get('accepted_added_union_edge_count')}
- Edge-dependent fragments/components: {aggregate.get('edge_dependent_fragment_count')} / {aggregate.get('edge_dependent_component_count')}
- Edge-dependent valid certificates: {aggregate.get('edge_dependent_valid_certificate_count')}
- Edge-dependent valid certificates line-equivalent to frozen controls: {aggregate.get('control_equivalent_edge_dependent_valid_certificate_count')}
- Novel edge-dependent valid certificates: {aggregate.get('novel_edge_dependent_valid_certificate_count')}
- Frozen two-sided one-parent target components: {aggregate.get('frozen_two_sided_one_parent_target_count')}
- Exact causally extended replacement components: {aggregate.get('extended_component_count')}
- Extended valid certificates: {aggregate.get('extended_valid_certificate_count')}
- Extended valid certificates line-equivalent to frozen controls: {aggregate.get('extended_control_equivalent_valid_certificate_count')}
- Extended novel valid certificates: {aggregate.get('extended_novel_valid_certificate_count')}
- The complete replacement lane produces no novel attributable certificate and
  therefore stops before gold.

## Required research method

1. Search the current literature. Use primary sources only: original papers,
   publisher proceedings, standards, or official technical documentation.
2. Cite every research-dependent claim with a direct DOI, arXiv, publisher, or
   official-project link. Do not use search-result pages, vendor blogs, or
   secondary summaries as authority.
3. Compare this method explicitly with prior point-cloud crease/edge extraction,
   region growing, local surface fitting, graph segmentation, CAD-free bend
   inference, and industrial metrology work.
4. Qualify novelty narrowly. Distinguish a novel end-to-end evidence contract or
   causal validation protocol from novelty of plane fitting, graph edges,
   connected components, or line extraction individually. State OBSERVED,
   INFERENCE, and UNKNOWN separately.

## Exact decision question

Does this evidence falsify only the operational hypothesis “removing
mutual-top-eight SAME_SURFACE pruning will create novel bend certificates,” and
localize the next blocker to frozen CROSS_MODE_LOCAL transition seeding or
opposite-side support? Or does it falsify the LOCAL_PLANE_FRAGMENT representation
strongly enough that this research family should stop?

Choose exactly one:

- **ONE NEXT GOLD-FREE INTERVENTION**: prescribe one bounded intervention only,
  with exact input/output schemas, causal-attribution fields, deletion
  counterfactual, hard caps, deterministic replays, memory gate, and exact
  pre-gold GO/STOP criteria; or
- **STOP THIS RESEARCH FAMILY**: explain the falsification boundary and identify
  the next representation class worth consulting on without implementing it.

Do not authorize opening gold, A1 ownership, emitted births, fitted thresholds,
expected-count logic, learning, runtime integration, or main-branch promotion.
Even a GO authorizes only another bounded professor packet.

## File-integrity table

{hashes}
"""


def build_package(
    *,
    output_dir: Path,
    causal_funnel_report_path: Path,
    corrected_bundle_path: Path | None,
    expected_corrected_bundle_sha256: str | None,
    historical_anchor_pdf: Path,
    prior_ruling_path: Path,
    batch_manifest_path: Path,
    replay_attestation_path: Path,
    synthetic_resource_attestation_path: Path,
    edge_core_path: Path,
    frozen_lpf_core_path: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("professor package output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _read(causal_funnel_report_path.resolve())
    if report.get("kind") != "f1_v2_complete_edge_gold_free_causal_funnel_v1":
        raise ValueError("causal funnel report identity mismatch")

    destinations: dict[str, Path] = {}
    copy_plan = {
        "01_PRIOR_PROFESSOR_RULING.md": prior_ruling_path,
        "03_GOLD_FREE_CAUSAL_FUNNEL.json": causal_funnel_report_path,
        "04_BATCH_MANIFEST.json": batch_manifest_path,
        "05_REPLAY_ATTESTATION.json": replay_attestation_path,
        "06_SYNTHETIC_RESOURCE_ATTESTATION.json": synthetic_resource_attestation_path,
        "07_COMPLETE_EDGE_CORE.py": edge_core_path,
        "08_FROZEN_LPF_CONTROL_CORE.py": frozen_lpf_core_path,
        "09_FeatureExtractionFromPointClouds.pdf": historical_anchor_pdf,
    }
    for name, source in copy_plan.items():
        destination = output_dir / name
        _copy(source, destination)
        destinations[name] = destination

    corrected_known = corrected_bundle_path is not None
    if corrected_known:
        source = corrected_bundle_path.resolve()
        actual = _sha256(source)
        if expected_corrected_bundle_sha256 is not None and actual != expected_corrected_bundle_sha256.lower():
            raise ValueError("corrected bundle SHA differs from expected SHA")
        destination = output_dir / "02_CORRECTED_COMPACT_BUNDLE.json"
        _copy(source, destination)
        destinations[destination.name] = destination
    else:
        if expected_corrected_bundle_sha256 is not None:
            raise ValueError("expected corrected bundle SHA supplied without a bundle")
        destination = output_dir / "02_CORRECTED_BUNDLE_HASH_PENDING.txt"
        destination.write_text(
            "PLACEHOLDER ONLY\nReplace this file with the corrected compact bundle "
            "and externally freeze its SHA-256 before submission.\n",
            encoding="utf-8",
        )
        destinations[destination.name] = destination

    if len(destinations) + 1 > MAX_UPLOAD_FILES:
        raise AssertionError("professor upload set exceeds ten individual files")
    upload_hashes = {
        name: _sha256(path) for name, path in sorted(destinations.items())
    }
    prompt_path = output_dir / "00_GPT_PRO_PROMPT.md"
    prompt_path.write_text(
        _prompt(
            causal_report=report,
            upload_hashes=upload_hashes,
            corrected_bundle_known=corrected_known,
        ),
        encoding="utf-8",
    )
    destinations[prompt_path.name] = prompt_path
    final_hashes = {name: _sha256(path) for name, path in sorted(destinations.items())}
    return {
        "output_dir": str(output_dir),
        "fresh_conversation_required": True,
        "individual_file_count": len(destinations),
        "corrected_bundle_known": corrected_known,
        "files": [
            {"name": name, "path": str(path), "sha256": final_hashes[name]}
            for name, path in sorted(destinations.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--causal-funnel-report", required=True, type=Path)
    parser.add_argument("--corrected-bundle", type=Path)
    parser.add_argument("--expected-corrected-bundle-sha256")
    parser.add_argument(
        "--historical-anchor-pdf",
        type=Path,
        default=DEFAULT_HISTORICAL_ANCHOR,
    )
    parser.add_argument(
        "--prior-ruling",
        type=Path,
        default=ROOT / "docs/research/PROFESSOR_F1_V2_PRIMITIVE_EDGE_SEMANTICS_RESPONSE_20260805.md",
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT / "v2_complete_same_surface_edge_probe_v1/manifest.json",
    )
    parser.add_argument(
        "--replay-attestation",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT / "v2_complete_same_surface_edge_probe_v1_replays/replay_attestation.json",
    )
    parser.add_argument(
        "--synthetic-resource-attestation",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT / "v2_complete_same_surface_edge_probe_v1_replays/synthetic_worst_cap_resource.json",
    )
    parser.add_argument(
        "--edge-core",
        type=Path,
        default=ROOT / "backend/f1_v2_complete_same_surface_edge_probe.py",
    )
    parser.add_argument(
        "--frozen-lpf-core",
        type=Path,
        default=ROOT / "backend/f1_local_plane_fragment_probe.py",
    )
    args = parser.parse_args()
    result = build_package(
        output_dir=args.output_dir,
        causal_funnel_report_path=args.causal_funnel_report,
        corrected_bundle_path=args.corrected_bundle,
        expected_corrected_bundle_sha256=args.expected_corrected_bundle_sha256,
        historical_anchor_pdf=args.historical_anchor_pdf,
        prior_ruling_path=args.prior_ruling,
        batch_manifest_path=args.batch_manifest,
        replay_attestation_path=args.replay_attestation,
        synthetic_resource_attestation_path=args.synthetic_resource_attestation,
        edge_core_path=args.edge_core,
        frozen_lpf_core_path=args.frozen_lpf_core,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
