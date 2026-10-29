#!/usr/bin/env python3
"""Build the GPT-5.6 Pro LPF-M1 pre-lock blocker package as individual files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "output/research/f1-gold27-20260803"
LPF = RESEARCH / "local_plane_fragment_probe_v1"
REPLAYS = LPF / "replay_profiles"
POOL_ROOT = RESEARCH / "candidate_pool_transition_v2_probe"
PACKAGE = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_LPF_M1_Prelock_Blocker_20260805"
)

STATIC_FILES = {
    "00_GPT_PRO_PROMPT.md": ROOT
    / "docs/research/PROFESSOR_F1_LPF_M1_PRELOCK_BLOCKER_DROPOFF_20260805.md",
    "01_PRELOCK_BLOCKER_REPORT.md": ROOT
    / "docs/research/LPF_M1_PRELOCK_CONTRADICTION_20260805.md",
    "02_PRIOR_PROFESSOR_RESPONSE.md": ROOT
    / "docs/research/PROFESSOR_F1_LOCAL_PLANE_FRAGMENT_RESPONSE_20260805.md",
    "03_FOCUSED_TEST_EVIDENCE.txt": ROOT
    / "docs/research/LPF_M1_PRELOCK_TEST_EVIDENCE_20260805.txt",
    "04_f1_local_plane_fragment_probe.py": ROOT
    / "backend/f1_local_plane_fragment_probe.py",
    "05_diagnose_f1_local_plane_fragment_probe.py": ROOT
    / "scripts/diagnose_f1_local_plane_fragment_probe.py",
    "06_run_f1_local_plane_fragment_probe_batch.py": ROOT
    / "scripts/run_f1_local_plane_fragment_probe_batch.py",
    "07_build_f1_local_plane_fragment_probe_bundle.py": ROOT
    / "scripts/build_f1_local_plane_fragment_probe_bundle.py",
    "08_attribute_f1_local_plane_fragment_probe_gold.py": ROOT
    / "scripts/attribute_f1_local_plane_fragment_probe_gold.py",
    "09_test_f1_local_plane_fragment_probe.py": ROOT
    / "tests/test_f1_local_plane_fragment_probe.py",
    "10_test_f1_local_plane_fragment_probe_memory.py": ROOT
    / "tests/test_f1_local_plane_fragment_probe_memory.py",
    "11_test_build_f1_local_plane_fragment_probe_bundle.py": ROOT
    / "tests/test_build_f1_local_plane_fragment_probe_bundle.py",
    "12_test_attribute_f1_local_plane_fragment_probe_gold.py": ROOT
    / "tests/test_attribute_f1_local_plane_fragment_probe_gold.py",
}

EXPECTED_OUTPUT_HASHES = {
    "C001": "3b73b1dff78777bef60683595ad520d3ec3fec77b342fc149056a19351a88ad7",
    "C030": "b56e707982413aac2065cbf23eba1b2d3f81833d815257b73b75efca36407d8b",
    "C003": "25dd2523bd5838c9762557f2bf58c480d5947813f620b30a7260926bb4b7eb99",
}
EXPECTED_ENGINEERING_PROFILE_C001 = (
    "2795aa85c2adcfe9a10d7cc24dafc25a52720e3541e637bfb901464052df2a06"
)
EXPECTED_MAX_RAW_SINGLETON_FIT = 3.5759435084334337
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
PRELOCK_RSS_OBSERVED_BOUND_BYTES = 135_000_000


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _singleton_contradiction_evidence() -> dict[str, Any]:
    pool_path = POOL_ROOT / "C001.json"
    probe_path = REPLAYS / "C001_run1.json"
    pool = _read(pool_path)
    probe = _read(probe_path)
    tolerance = float(probe["thresholds"]["plane_fit_tolerance_mm"])
    if tolerance != 3.0:
        raise ValueError(f"C001 plane-fit tolerance changed: {tolerance}")
    modes = {
        str(row["mode_atom_id"]): dict(row)
        for row in pool["support_mode_table"]
    }
    high_fragments = [
        dict(row)
        for row in probe["local_plane_fragment_table"]
        if int(row["mode_count"]) == 1
        and float(row["surface_fit_p90_mm"]) > tolerance
    ]
    high_fragments.sort(key=lambda row: str(row["local_plane_fragment_id"]))
    if len(high_fragments) != 6:
        raise ValueError(f"expected six C001 high-fit singletons, got {len(high_fragments)}")
    rows = []
    for fragment in high_fragments:
        witnesses = list(fragment.get("witness_mode_ids") or ())
        if len(witnesses) != 1 or witnesses[0] not in modes:
            raise ValueError("high-fit singleton lacks one resolved V2 witness mode")
        mode = modes[witnesses[0]]
        raw_fit = float(mode["surface_fit_p90_mm"])
        if raw_fit <= tolerance:
            raise ValueError("serialized high-fit singleton is not high-fit in frozen V2")
        if fragment.get("transition_certificate_allowed") is not False:
            raise ValueError("provisional high-fit singleton unexpectedly permits a certificate")
        rows.append(
            {
                "local_plane_fragment_id": fragment["local_plane_fragment_id"],
                "mode_atom_id": witnesses[0],
                "parent_atom_id": mode["parent_atom_id"],
                "mode_point_count": int(mode["point_count"]),
                "raw_v2_surface_fit_p90_mm": raw_fit,
                "serialized_fragment_surface_fit_p90_mm": float(
                    fragment["surface_fit_p90_mm"]
                ),
                "excess_over_tolerance_mm": raw_fit - tolerance,
                "transition_certificate_allowed": False,
            }
        )
    maximum = max(row["raw_v2_surface_fit_p90_mm"] for row in rows)
    if maximum != EXPECTED_MAX_RAW_SINGLETON_FIT:
        raise ValueError(f"C001 maximum singleton fit changed: {maximum!r}")
    if probe["engineering_profile_sha256"] != EXPECTED_ENGINEERING_PROFILE_C001:
        raise ValueError("C001 provisional engineering-profile hash changed")
    counters = probe["case_counters"]
    if int(counters["singleton_fit_ineligible_fragment_count"]) != 6:
        raise ValueError("C001 singleton-fit-ineligible counter changed")

    core_text = (ROOT / "backend/f1_local_plane_fragment_probe.py").read_text(
        encoding="utf-8"
    )
    if "len(mode_ids) > 1 and fit[\"surface_fit_p90_mm\"]" not in core_text:
        raise ValueError("provisional singleton exception is no longer present")
    profile_region = core_text.split("def _engineering_profile_sha256", 1)[1].split(
        "def build_local_plane_fragment_probe", 1
    )[0]
    if "singleton" in profile_region.lower() or "orphan" in profile_region.lower():
        raise ValueError("engineering hash now appears to encode an orphan/singleton rule")

    forbidden_lock_artifacts = (
        LPF / "manifest.json",
        LPF / "bundle.json",
        LPF / "postlock_gold_attribution.json",
    )
    existing = [str(path) for path in forbidden_lock_artifacts if path.exists()]
    if existing:
        raise ValueError("pre-lock blocker package found forbidden lock/gold artifacts: " + ", ".join(existing))

    return {
        "schema_version": 1,
        "kind": "lpf_m1_prelock_singleton_contradiction_evidence",
        "research_only": True,
        "gold_opened": False,
        "batch_37_frozen": False,
        "bundle_locked": False,
        "profile": probe["profile"],
        "plane_fit_tolerance_mm": tolerance,
        "high_fit_singleton_count": len(rows),
        "maximum_raw_v2_surface_fit_p90_mm": maximum,
        "c001_engineering_profile_sha256": probe["engineering_profile_sha256"],
        "c001_candidate_pool_sha256": _sha256(pool_path),
        "c001_probe_sha256": _sha256(probe_path),
        "provisional_exception": {
            "retains_singleton": True,
            "certificate_allowed": False,
            "approved_by_professor": False,
            "encoded_in_engineering_profile_sha256": False,
        },
        "c001_case_counters": {
            "input_mode_count": int(counters["input_mode_count"]),
            "local_plane_fragment_count": int(counters["local_plane_fragment_count"]),
            "singleton_fragment_count": int(counters["singleton_fragment_count"]),
            "singleton_fit_ineligible_fragment_count": int(
                counters["singleton_fit_ineligible_fragment_count"]
            ),
            "local_plane_pair_count": int(counters["local_plane_pair_count"]),
            "local_transition_component_count": int(
                counters["local_transition_component_count"]
            ),
            "geometry_certificate_count": int(counters["geometry_certificate_count"]),
        },
        "high_fit_singletons": rows,
    }


def _replay_evidence() -> dict[str, Any]:
    pool_manifest = _read(POOL_ROOT / "manifest.json")
    pool_rows = {
        str(row["case_token"]): dict(row) for row in pool_manifest["cases"]
    }
    maximum_mode = max(
        pool_rows.values(), key=lambda row: int(row["transition_mode_count"])
    )
    maximum_edge = max(
        pool_rows.values(), key=lambda row: int(row["transition_mode_edge_count"])
    )
    if maximum_mode["case_token"] != "C030" or maximum_edge["case_token"] != "C003":
        raise ValueError("designated maximum-mode/maximum-edge cases changed")

    cases = []
    maximum_peak = 0
    for token, reason in (
        ("C001", "inherited_200000_point_reference"),
        ("C030", "maximum_transition_mode_count"),
        ("C003", "maximum_transition_mode_edge_count"),
    ):
        runs = []
        output_hashes = []
        for run in range(1, 4):
            output_path = REPLAYS / f"{token}_run{run}.json"
            profile_path = REPLAYS / f"{token}_run{run}_profile.json"
            profile = _read(profile_path)
            output_hash = _sha256(output_path)
            if profile.get("probe_sha256") != output_hash:
                raise ValueError(f"{token} run {run}: profile/output hash mismatch")
            peak = int(profile["peak_rss_bytes"])
            if peak >= PEAK_RSS_LIMIT_BYTES:
                raise ValueError(f"{token} run {run}: peak RSS exceeded 16 GiB")
            maximum_peak = max(maximum_peak, peak)
            output_hashes.append(output_hash)
            runs.append(
                {
                    "run": run,
                    "probe_path": str(output_path.relative_to(ROOT)),
                    "probe_sha256": output_hash,
                    "profile_path": str(profile_path.relative_to(ROOT)),
                    "profile_sha256": _sha256(profile_path),
                    "wall_seconds": float(profile["wall_seconds"]),
                    "peak_rss_bytes": peak,
                    "input_json_bytes": int(profile["input_json_bytes"]),
                    "serialized_output_bytes": int(
                        profile["serialized_output_bytes"]
                    ),
                }
            )
        if len(set(output_hashes)) != 1:
            raise ValueError(f"{token}: three probe replays are not byte-identical")
        if output_hashes[0] != EXPECTED_OUTPUT_HASHES[token]:
            raise ValueError(f"{token}: frozen replay hash changed")
        cases.append(
            {
                "case_token": token,
                "selection_reason": reason,
                "transition_mode_count": int(pool_rows[token]["transition_mode_count"]),
                "transition_mode_edge_count": int(
                    pool_rows[token]["transition_mode_edge_count"]
                ),
                "replay_count": 3,
                "byte_identical": True,
                "canonical_probe_sha256": output_hashes[0],
                "maximum_peak_rss_bytes": max(row["peak_rss_bytes"] for row in runs),
                "runs": runs,
            }
        )
    if maximum_peak >= PRELOCK_RSS_OBSERVED_BOUND_BYTES:
        raise ValueError(
            f"observed replay maximum no longer below 135 MB: {maximum_peak}"
        )
    return {
        "schema_version": 1,
        "kind": "lpf_m1_prelock_three_case_replay_evidence",
        "research_only": True,
        "profile_approved_for_lock": False,
        "designated_cases": ["C001", "C030", "C003"],
        "total_isolated_runs": 9,
        "all_probe_payloads_byte_identical_within_case": True,
        "maximum_peak_rss_bytes": maximum_peak,
        "maximum_peak_rss_below_135000000_bytes": True,
        "maximum_peak_rss_below_16_gib": True,
        "candidate_pool_manifest_sha256": _sha256(POOL_ROOT / "manifest.json"),
        "cases": cases,
    }


def main() -> int:
    missing = [str(path) for path in STATIC_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing blocker package input:\n" + "\n".join(missing))

    contradiction = _singleton_contradiction_evidence()
    replay = _replay_evidence()
    PACKAGE.mkdir(parents=True, exist_ok=True)
    expected_names = set(STATIC_FILES) | {
        "13_SINGLETON_CONTRADICTION_EVIDENCE.json",
        "14_THREE_CASE_REPLAY_EVIDENCE.json",
        "manifest.json",
    }
    for stale in PACKAGE.iterdir():
        if stale.is_file() and stale.name not in expected_names:
            stale.unlink()
    for destination, source in STATIC_FILES.items():
        shutil.copy2(source, PACKAGE / destination)
    _write_json(PACKAGE / "13_SINGLETON_CONTRADICTION_EVIDENCE.json", contradiction)
    _write_json(PACKAGE / "14_THREE_CASE_REPLAY_EVIDENCE.json", replay)

    source_by_name = {
        name: str(path.relative_to(ROOT)) for name, path in STATIC_FILES.items()
    }
    source_by_name["13_SINGLETON_CONTRADICTION_EVIDENCE.json"] = "generated_by_builder"
    source_by_name["14_THREE_CASE_REPLAY_EVIDENCE.json"] = "generated_by_builder"
    rows = []
    for name in sorted(expected_names - {"manifest.json"}):
        path = PACKAGE / name
        rows.append(
            {
                "name": name,
                "source": source_by_name[name],
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-lpf-m1-prelock-blocker-20260805-v1",
        "submission_style": "gpt_5_6_pro_individual_file_upload_not_email_not_zip",
        "decision_state": "PRELOCK_SPECIFICATION_BLOCKER",
        "gold_opened": False,
        "batch_37_frozen": False,
        "bundle_locked": False,
        "runtime_promotion_authorized": False,
        "requested_primary_ruling": "FORMAL_PROFILE_FAILURE_OR_VERSIONED_BOUNDED_ORPHAN_RULE",
        "file_count_excluding_manifest": len(rows),
        "file_count_including_manifest": len(rows) + 1,
        "files": rows,
    }
    manifest_path = PACKAGE / "manifest.json"
    _write_json(manifest_path, manifest)

    for row in rows:
        if _sha256(PACKAGE / row["name"]) != row["sha256"]:
            raise RuntimeError(f"post-copy hash mismatch: {row['name']}")
    print(
        json.dumps(
            {
                "package": str(PACKAGE),
                "prompt": str(PACKAGE / "00_GPT_PRO_PROMPT.md"),
                "files_including_manifest": len(rows) + 1,
                "manifest_sha256": _sha256(manifest_path),
                "maximum_raw_singleton_fit_mm": contradiction[
                    "maximum_raw_v2_surface_fit_p90_mm"
                ],
                "maximum_replay_peak_rss_bytes": replay[
                    "maximum_peak_rss_bytes"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
