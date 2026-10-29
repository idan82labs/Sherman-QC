#!/usr/bin/env python3
"""Regenerate only the locked GCDF sampler/source identity before claim creation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_gcdf_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json  # noqa: E402
from research.f1_gcdf_v1.core import load_bound_sample  # noqa: E402

RAW_SOURCE = ROOT / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/35705010_A/scans/35705010_A_X.ply"


def build_binding() -> dict[str, object]:
    sample = load_bound_sample(RAW_SOURCE)
    value: dict[str, object] = {
        "schema_version": 1,
        "attempt_id": CONFIG.attempt_id,
        "configuration_sha256": CONFIG_SHA256,
        "raw_source_relative": str(RAW_SOURCE.relative_to(ROOT)),
        "raw_source_sha256": CONFIG.raw_source_sha256,
        "canonical_sample_count": len(sample.xyz),
        "ordered_source_ordinal_sha256": sample.ordered_source_ordinal_sha256,
        "sampled_coordinate_sha256": sample.sampled_coordinate_sha256,
        "sampler_executable_sha256": sample.sampler_executable_sha256,
        "sampler_configuration_sha256": sample.sampler_configuration_sha256,
        "scientific_geometry_evaluated": False,
        "target_capability_granted": False,
        "target_capability_opened": False,
        "target_process_launched": False,
    }
    value["binding_sha256"] = sha_json(value)
    return value


if __name__ == "__main__":
    print(canonical_json_bytes(build_binding()).decode("ascii"))
