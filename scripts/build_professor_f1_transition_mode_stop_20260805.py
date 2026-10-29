#!/usr/bin/env python3
"""Build and verify the GPT-5.6 Pro transition-mode STOP follow-up package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "output/research/f1-gold27-20260803"
POOL_ROOT = RESEARCH / "candidate_pool_transition_v2_probe"
REPLAY_ROOT = RESEARCH / "candidate_pool_transition_v2_replay"
PROBE_ROOT = RESEARCH / "transition_mode_probe_v1"
PACKAGE = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Transition_Mode_STOP_20260805"
)

FILES = {
    "00_GPT_PRO_PROMPT.md": ROOT
    / "docs/research/PROFESSOR_F1_TRANSITION_MODE_STOP_DROPOFF_20260805.md",
    "01_DIAGNOSTIC_STOP_REPORT.md": ROOT
    / "docs/research/F1_TRANSITION_MODE_PROBE_STOP_20260805.md",
    "02_PRIOR_PROFESSOR_RESPONSE.md": ROOT
    / "docs/research/PROFESSOR_F1_TRANSITION_ATOMIZATION_RESPONSE_20260805.md",
    "03_patch_graph_latent_decomposition.py": ROOT
    / "backend/patch_graph_latent_decomposition.py",
    "04_build_f1_scan_only_candidate_pool.py": ROOT
    / "scripts/build_f1_scan_only_candidate_pool.py",
    "05_f1_structured_birth_diagnostics.py": ROOT
    / "backend/f1_structured_birth_diagnostics.py",
    "06_diagnose_f1_transition_mode_probe.py": ROOT
    / "scripts/diagnose_f1_transition_mode_probe.py",
    "07_TRANSITION_V2_POOL_MANIFEST.json": POOL_ROOT / "manifest.json",
    "08_PROBE_MANIFEST.json": PROBE_ROOT / "manifest.json",
    "09_PROBE_CENSUS_BUNDLE.json": PROBE_ROOT / "census_bundle.json",
    "10_POSTLOCK_GOLD_ATTRIBUTION.json": PROBE_ROOT
    / "postlock_gold_attribution.json",
    "11_C001_MEMORY_PROFILE.json": PROBE_ROOT / "C001_memory_profile.json",
    "12_C010_CERTIFICATE_CENSUS.json": PROBE_ROOT / "C010.json",
    "13_C033_CERTIFICATE_CENSUS.json": PROBE_ROOT / "C033.json",
}

EXPECTED_REPLAY_SHA256 = (
    "36a10af46fc295f437cb95522117605675b671066632af9ab4a97c7f172d8b28"
)
EXPECTED_CENSUS_SHA256 = (
    "dd0007286a2182cdfe1bf0c18219b54fb75226db5543b67caf20a3747ce52ac6"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_frozen_evidence() -> dict[str, Any]:
    pool_manifest = _read_json(POOL_ROOT / "manifest.json")
    probe_manifest = _read_json(PROBE_ROOT / "manifest.json")
    postlock = _read_json(PROBE_ROOT / "postlock_gold_attribution.json")
    memory = _read_json(PROBE_ROOT / "C001_memory_profile.json")

    cases = probe_manifest["cases"]
    totals = {
        "plane_pairs": sum(int(row["plane_pair_record_count"]) for row in cases),
        "multimodal_parents": sum(
            int(row["multimodal_parent_record_count"]) for row in cases
        ),
        "components": sum(
            int(row["transition_component_record_count"]) for row in cases
        ),
        "hard_certificates": sum(
            int(row["eligible_certificate_count"]) for row in cases
        ),
    }
    expected_totals = {
        "plane_pairs": 938,
        "multimodal_parents": 10_457,
        "components": 1_994,
        "hard_certificates": 2,
    }
    if totals != expected_totals:
        raise ValueError(f"probe totals changed: {totals!r}")

    if probe_manifest.get("case_count") != 37:
        raise ValueError("probe manifest must contain 37 frozen cases")
    if probe_manifest.get("inference_safe") is not True:
        raise ValueError("probe is not marked inference-safe")
    if probe_manifest.get("birth_columns_enabled") is not False:
        raise ValueError("birth columns unexpectedly enabled")
    if probe_manifest.get("prediction_changed") is not False:
        raise ValueError("probe unexpectedly changed predictions")

    summary = postlock["summary"]
    if summary["stage_counts"] != {
        "1": 67,
        "2": 0,
        "3": 0,
        "4": 3,
        "5": 0,
        "6": 0,
        "7": 70,
    }:
        raise ValueError(f"post-lock stage counts changed: {summary['stage_counts']!r}")
    gate = postlock["professor_probe_gate"]
    metrics = gate["metrics"]
    if gate.get("passed") is not False:
        raise ValueError("formal STOP unexpectedly changed to GO")
    if metrics["recovered_unrepresented_objects"] != 0:
        raise ValueError("gold recovery is no longer zero")
    if metrics["independent_family_group_count"] != 0:
        raise ValueError("independent-family recovery is no longer zero")
    if metrics["a0_false_predictions_passing_certificate"] != 0:
        raise ValueError("false-certificate count is no longer zero")
    if metrics["observed_a0_false_prediction_count"] != 33:
        raise ValueError("frozen A0 false-prediction denominator changed")
    if postlock["diagnostic_census_sha256"] != EXPECTED_CENSUS_SHA256:
        raise ValueError("frozen diagnostic census hash changed")

    emitted_births = 0
    for row in cases:
        census = _read_json(Path(row["path"]))
        if census.get("birth_columns_enabled") is not False:
            raise ValueError(f"birth columns enabled in {row['case_token']}")
        emitted_births += sum(
            1
            for record in census["birth_eligibility_records"]
            if record.get("column_emitted") is True
        )
    if emitted_births != 0:
        raise ValueError(f"expected zero emitted births, got {emitted_births}")

    certificate_cases = {
        row["case_token"]: int(row["eligible_certificate_count"])
        for row in cases
        if int(row["eligible_certificate_count"]) > 0
    }
    if certificate_cases != {"C010": 1, "C033": 1}:
        raise ValueError(f"representative certificate cases changed: {certificate_cases}")

    if memory["peak_rss_bytes"] != 493_977_600:
        raise ValueError("C001 peak RSS changed")
    if memory["peak_rss_gib"] != 0.460052:
        raise ValueError("C001 GiB result changed")
    if memory["peak_rss_below_16_gib"] is not True:
        raise ValueError("C001 exceeded the 16 GiB gate")
    if memory["working_point_count"] != 200_000:
        raise ValueError("C001 did not exercise the 200k-point bound")
    if memory["source_point_serialization"] is not False:
        raise ValueError("C001 memory artifact reports source-point serialization")

    first_hash = _sha256(POOL_ROOT / "C001.json")
    replay_hash = _sha256(REPLAY_ROOT / "C001.json")
    if first_hash != EXPECTED_REPLAY_SHA256 or replay_hash != EXPECTED_REPLAY_SHA256:
        raise ValueError(
            f"C001 replay mismatch: first={first_hash}, replay={replay_hash}"
        )
    if pool_manifest.get("source_point_serialization") is not False:
        raise ValueError("V2 pool manifest permits source-point serialization")

    return {
        **totals,
        "emitted_births": emitted_births,
        "recovered_unrepresented_objects": 0,
        "independent_family_groups": 0,
        "a0_false_certificates": 0,
        "stage_counts": summary["stage_counts"],
        "c001_peak_rss_gib": memory["peak_rss_gib"],
        "c001_replay_sha256": replay_hash,
        "diagnostic_census_sha256": postlock["diagnostic_census_sha256"],
    }


def main() -> int:
    missing = [str(source) for source in FILES.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing professor evidence:\n" + "\n".join(missing))
    if len(FILES) + 1 > 15:
        raise ValueError("professor package exceeds the 15-file upload limit")

    verified = _verify_frozen_evidence()

    PACKAGE.mkdir(parents=True, exist_ok=True)
    expected_names = set(FILES) | {"manifest.json"}
    for stale in PACKAGE.iterdir():
        if stale.is_file() and stale.name not in expected_names:
            stale.unlink()

    for destination_name, source in FILES.items():
        shutil.copy2(source, PACKAGE / destination_name)

    rows: list[dict[str, Any]] = []
    for name in sorted(FILES):
        path = PACKAGE / name
        rows.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "source": str(FILES[name].relative_to(ROOT)),
            }
        )

    manifest = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-transition-mode-stop-20260805-v1",
        "submission_style": "gpt_5_6_pro_file_upload_not_email",
        "decision": "FORMAL_DIAGNOSTIC_STOP",
        "runtime_promotion_authorized": False,
        "a1_ownership_authorized": False,
        "births_authorized": False,
        "learning_authorized": False,
        "threshold_tuning_authorized": False,
        "verified_results": verified,
        "file_count_excluding_manifest": len(rows),
        "file_count_including_manifest": len(rows) + 1,
        "files": rows,
    }
    manifest_path = PACKAGE / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for row in rows:
        actual = _sha256(PACKAGE / row["name"])
        if actual != row["sha256"]:
            raise RuntimeError(f"post-copy hash mismatch: {row['name']}")

    print(
        json.dumps(
            {
                "package": str(PACKAGE),
                "files_including_manifest": len(rows) + 1,
                "prompt": str(PACKAGE / "00_GPT_PRO_PROMPT.md"),
                "manifest_sha256": _sha256(manifest_path),
                "verified_results": verified,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
