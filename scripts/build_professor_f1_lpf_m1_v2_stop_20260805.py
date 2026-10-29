#!/usr/bin/env python3
"""Build the GPT-5.6 Pro LPF-M1-V2 locked STOP package (no archive)."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "output/research/f1-gold27-20260803"
PROBE_ROOT = RESEARCH / "local_plane_fragment_probe_v2"
BATCH_MANIFEST = PROBE_ROOT / "manifest.json"
BUNDLE = RESEARCH / "local_plane_fragment_probe_v2_bundle.json"
REPLAY = RESEARCH / "local_plane_fragment_probe_v2_replays/replay_attestation.json"
POSTLOCK = RESEARCH / "local_plane_fragment_probe_v2_postlock_gold_attribution.json"
PACKAGE = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_LPF_M1_V2_Locked_STOP_20260805"
)

EXPECTED_HASHES = {
    "batch_manifest": "3b9dfa156746a01cb38195f7d75c92e23bff839228de96e425b8902edf262510",
    "bundle": "aeed2c3cde05f8532d34e78ff046fa0880ac9170440a04cc39916e8ee800e026",
    "lock_attestation": "6085571d7ab53a4c107ddac0368b63941b0445e9328aae07662c9bf050c93bca",
    "replay": "d36279f3d52222e80ebe5493a8b40033d7f40b4be7357b5de48b00825be9e1c4",
    "postlock": "06c730e74c640dcbedac73d1ce099a6d7256354ae1837eb561d937cff5111cb4",
}

STATIC_FILES = {
    "00_GPT_PRO_PROMPT.md": ROOT
    / "docs/research/PROFESSOR_F1_LPF_M1_V2_LOCKED_STOP_DROPOFF_20260805.md",
    "01_LOCKED_STOP_REPORT.md": ROOT
    / "docs/research/LPF_M1_V2_LOCKED_STOP_20260805.md",
    "02_PRIOR_PROFESSOR_RULING_B.md": ROOT
    / "docs/research/PROFESSOR_F1_LPF_M1_PRELOCK_BLOCKER_RESPONSE_20260805.md",
    "03_BATCH_MANIFEST.json": BATCH_MANIFEST,
    "04_REPLAY_ATTESTATION.json": REPLAY,
    "05_POSTLOCK_ATTRIBUTION.json": POSTLOCK,
    "06_patch_graph_latent_decomposition.py": ROOT
    / "backend/patch_graph_latent_decomposition.py",
    "07_build_f1_scan_only_candidate_pool.py": ROOT
    / "scripts/build_f1_scan_only_candidate_pool.py",
    "08_f1_local_plane_fragment_probe.py": ROOT
    / "backend/f1_local_plane_fragment_probe.py",
    "09_diagnose_f1_local_plane_fragment_probe.py": ROOT
    / "scripts/diagnose_f1_local_plane_fragment_probe.py",
    "10_run_f1_local_plane_fragment_probe_batch.py": ROOT
    / "scripts/run_f1_local_plane_fragment_probe_batch.py",
    "11_run_f1_local_plane_fragment_probe_replays.py": ROOT
    / "scripts/run_f1_local_plane_fragment_probe_replays.py",
    "12_build_f1_local_plane_fragment_probe_bundle.py": ROOT
    / "scripts/build_f1_local_plane_fragment_probe_bundle.py",
    "13_attribute_f1_local_plane_fragment_probe_gold.py": ROOT
    / "scripts/attribute_f1_local_plane_fragment_probe_gold.py",
}

TEST_FILES = (
    ROOT / "tests/test_f1_local_plane_fragment_probe.py",
    ROOT / "tests/test_f1_local_plane_fragment_probe_memory.py",
    ROOT / "tests/test_f1_local_plane_fragment_probe_replays.py",
    ROOT / "tests/test_build_f1_local_plane_fragment_probe_bundle.py",
    ROOT / "tests/test_attribute_f1_local_plane_fragment_probe_gold.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_source_hashes() -> None:
    actual = {
        "batch_manifest": _sha256(BATCH_MANIFEST),
        "bundle": _sha256(BUNDLE),
        "replay": _sha256(REPLAY),
        "postlock": _sha256(POSTLOCK),
    }
    for name, digest in actual.items():
        if digest != EXPECTED_HASHES[name]:
            raise ValueError(f"locked {name} SHA-256 changed: {digest}")


def _locked_result_summary(
    *,
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    replay: Mapping[str, Any],
    postlock: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        bundle.get("kind") != "f1_local_plane_fragment_probe_bundle_v2"
        or bundle.get("profile") != "LOCAL_PLANE_FRAGMENT_PROBE_V2"
        or bundle.get("locked") is not True
        or bundle.get("ready_for_gold") is not True
    ):
        raise ValueError("V2 bundle is not a ready locked research bundle")
    lock = bundle.get("lock_attestation")
    if not isinstance(lock, Mapping):
        raise ValueError("bundle lock attestation is absent")
    lock_hash = str(lock.get("lock_attestation_sha256") or "")
    recomputed_lock = _sha256_json(
        {key: value for key, value in lock.items() if key != "lock_attestation_sha256"}
    )
    if lock_hash != EXPECTED_HASHES["lock_attestation"] or lock_hash != recomputed_lock:
        raise ValueError("lock attestation SHA-256 mismatch")
    criteria = dict(lock.get("criteria") or {})
    if len(criteria) != 25 or not all(value is True for value in criteria.values()):
        raise ValueError("the complete 25-criterion lock is not true")
    if bundle.get("replay_attestation_sha256") != EXPECTED_HASHES["replay"]:
        raise ValueError("bundle replay-attestation hash mismatch")

    rows = list(manifest.get("cases") or ())
    if (
        manifest.get("kind") != "f1_local_plane_fragment_probe_batch_v2"
        or int(manifest.get("case_count") or 0) != 37
        or int(manifest.get("successful_case_count") or 0) != 37
        or len(rows) != 37
        or any(row.get("status") != "SUCCESS" for row in rows)
    ):
        raise ValueError("batch is not the complete 37/37 V2 result")
    aggregate = {
        "local_plane_fragment_count": sum(
            int(row["case_counters"]["local_plane_fragment_count"]) for row in rows
        ),
        "local_plane_orphan_count": sum(
            int(row["case_counters"]["local_plane_orphan_count"]) for row in rows
        ),
        "local_plane_pair_count": sum(
            int(row["case_counters"]["local_plane_pair_count"]) for row in rows
        ),
        "local_transition_component_count": sum(
            int(row["case_counters"]["local_transition_component_count"])
            for row in rows
        ),
        "geometry_certificate_count": sum(
            int(row["case_counters"]["geometry_certificate_count"]) for row in rows
        ),
    }
    expected_aggregate = {
        "local_plane_fragment_count": 13_413,
        "local_plane_orphan_count": 39,
        "local_plane_pair_count": 17_034,
        "local_transition_component_count": 35_813,
        "geometry_certificate_count": 31,
    }
    if aggregate != expected_aggregate:
        raise ValueError(f"locked aggregate changed: {aggregate}")
    maximum_batch_rss = max(int(row["peak_rss_bytes"]) for row in rows)
    if maximum_batch_rss != 138_723_328:
        raise ValueError("maximum batch RSS changed")

    replay_rows = list(replay.get("cases") or ())
    if (
        replay.get("kind")
        != "f1_local_plane_fragment_probe_replay_attestation_v2"
        or {str(row.get("case_token")) for row in replay_rows}
        != {"C001", "C003", "C030"}
        or int(replay.get("total_isolated_run_count") or 0) != 9
    ):
        raise ValueError("designated V2 replay attestation is incomplete")
    for row in replay_rows:
        hashes = list(row.get("replay_probe_sha256s") or ())
        if (
            int(row.get("replay_count") or 0) != 3
            or len(hashes) != 3
            or len(set(hashes)) != 1
            or hashes[0] != row.get("batch_probe_sha256")
            or int(row.get("maximum_peak_rss_bytes") or 0) >= 16 * 1024**3
        ):
            raise ValueError(f"{row.get('case_token')}: replay evidence changed")

    summary = dict(postlock.get("summary") or {})
    gate = dict(postlock.get("professor_probe_gate") or {})
    metrics = dict(gate.get("metrics") or {})
    reason_counts = dict(summary.get("new_failure_reason_counts") or {})
    expected_reasons = {
        "NO_MATCHING_LOCAL_COMPONENT": 61,
        "EXCESSIVE_EMPTY_BIN_RUN": 1,
        "INSUFFICIENT_LEFT_SIDE_BINS": 1,
        "INSUFFICIENT_RIGHT_SIDE_BINS": 2,
        "INTERNAL_BRANCHING": 2,
    }
    if (
        postlock.get("kind") != "postlock_gold_local_plane_fragment_attribution_v2"
        or int(summary.get("primary_stage1_count") or 0) != 67
        or int(summary.get("recovered_stage1_count", -1)) != 0
        or int(summary.get("unmatchable_zero_length_component_count") or 0)
        != 15_609
        or reason_counts != expected_reasons
        or int(metrics.get("recovered_stage1_objects", -1)) != 0
        or int(metrics.get("independent_family_group_count", -1)) != 0
        or int(metrics.get("a0_false_predictions_certified", -1)) != 0
        or int(metrics.get("observed_frozen_a0_false_prediction_count") or 0) != 33
        or gate.get("passed") is not False
    ):
        raise ValueError("post-lock negative result changed")

    return {
        "schema_version": 1,
        "kind": "f1_lpf_m1_v2_locked_stop_compact_evidence",
        "decision": "FORMAL_STOP",
        "research_only": True,
        "runtime_promotion_authorized": False,
        "source_hashes": dict(EXPECTED_HASHES),
        "lock": {
            "bundle_kind": bundle["kind"],
            "profile": bundle["profile"],
            "locked": bundle["locked"],
            "ready_for_gold": bundle["ready_for_gold"],
            "case_count": bundle["case_count"],
            "successful_case_count": bundle["successful_case_count"],
            "maximum_peak_rss_bytes": bundle["maximum_peak_rss_bytes"],
            "probe_manifest_sha256": bundle["probe_manifest_sha256"],
            "replay_attestation_sha256": bundle["replay_attestation_sha256"],
            "probe_hash_table_sha256": bundle["probe_hash_table_sha256"],
            "inference_semantic_hash_table_sha256": bundle[
                "inference_semantic_hash_table_sha256"
            ],
            "engineering_profile_hash_table_sha256": bundle[
                "engineering_profile_hash_table_sha256"
            ],
            "input_contract_attestation_hash_table_sha256": bundle[
                "input_contract_attestation_hash_table_sha256"
            ],
            "a0_prediction_hash_table_sha256": bundle[
                "a0_prediction_hash_table_sha256"
            ],
            "lock_attestation": dict(lock),
        },
        "batch": {
            "workers": int(manifest["workers"]),
            "case_count": 37,
            "successful_case_count": 37,
            "maximum_peak_rss_bytes": maximum_batch_rss,
            "aggregate": aggregate,
        },
        "replay": {
            "total_isolated_run_count": 9,
            "cases": [
                {
                    "case_token": row["case_token"],
                    "selection_reason": row["selection_reason"],
                    "replay_count": row["replay_count"],
                    "batch_probe_sha256": row["batch_probe_sha256"],
                    "replay_probe_sha256s": list(row["replay_probe_sha256s"]),
                    "maximum_peak_rss_bytes": row["maximum_peak_rss_bytes"],
                }
                for row in sorted(replay_rows, key=lambda value: value["case_token"])
            ],
        },
        "postlock": {
            "primary_stage1_count": 67,
            "recovered_stage1_count": 0,
            "independent_family_group_count": 0,
            "a0_false_predictions_certified": 0,
            "frozen_a0_false_prediction_count": 33,
            "unmatchable_zero_length_component_count": 15_609,
            "zero_length_component_structure": {
                "all_support_parent_count_is_one": True,
                "single_mode_no_rung_one_side_count": 10_821,
                "multi_mode_same_parent_cross_rung_both_side_count": 4_788,
                "other_zero_length_structure_count": 0,
                "endpoint_only_finite_extent_supported": False,
            },
            "new_failure_reason_counts": expected_reasons,
            "gate_passed": False,
        },
        "next_tranche": {
            "kind": "V2_PRIMITIVE_REPLACEMENT_PROBE",
            "professor_must_select_exactly_one_upstream_hypothesis": [
                "VOXEL_SCALE_ALIASING",
                "PROJECTIVE_NORMAL_MODES",
                "EDGE_SEMANTICS",
                "EXPLICIT_TRANSITION_STRIP",
            ],
            "degenerate_span_counterfactual_authorized": False,
        },
        "focused_test_evidence": {
            "runtime": ".venv-f1-structured",
            "passed": 103,
            "wall_seconds": 3.04,
        },
    }


def _per_case_census(
    *, manifest: Mapping[str, Any], postlock: Mapping[str, Any]
) -> dict[str, Any]:
    reasons_by_case: dict[str, Counter[str]] = {}
    primary_by_case: Counter[str] = Counter()
    for row in postlock.get("rows") or ():
        token = str(row["case_token"])
        primary_by_case[token] += 1
        reasons_by_case.setdefault(token, Counter())[str(row["reason_code"])] += 1

    output_rows = []
    total_zero = 0
    total_zero_single_mode = 0
    total_zero_multi_mode = 0
    total_certificates = 0
    for manifest_row in sorted(manifest["cases"], key=lambda value: value["case_token"]):
        token = str(manifest_row["case_token"])
        probe_path = Path(str(manifest_row["probe_path"]))
        if _sha256(probe_path) != manifest_row["probe_sha256"]:
            raise ValueError(f"{token}: probe SHA-256 differs from locked manifest")
        probe = _read(probe_path)
        fragments = {
            str(row["local_plane_fragment_id"]): dict(row)
            for row in probe["local_plane_fragment_table"]
        }
        zero_count = 0
        zero_single_mode_count = 0
        zero_multi_mode_count = 0
        certificate_rows = []
        component_count = 0
        for pair in probe["local_plane_pair_records"]:
            fragment_ids = [str(value) for value in pair["fragment_ids"]]
            coverage_states = [
                str(fragments[value]["global_plane_coverage"]["coverage_state"])
                for value in fragment_ids
            ]
            for component in pair["components"]:
                component_count += 1
                endpoints = component["intersection_span_endpoints"]
                span = math.dist(endpoints[0], endpoints[1])
                if span <= 1e-12:
                    zero_count += 1
                    if int(component["support_parent_count"]) != 1:
                        raise ValueError(
                            f"{token}: zero span has more than one physical parent"
                        )
                    one_side = (
                        int(component["left_side_bin_count"]) == 1
                        and int(component["right_side_bin_count"]) == 0
                    ) or (
                        int(component["left_side_bin_count"]) == 0
                        and int(component["right_side_bin_count"]) == 1
                    )
                    if (
                        int(component["support_mode_count"]) == 1
                        and int(component["transition_rung_bin_count"]) == 0
                        and one_side
                    ):
                        zero_single_mode_count += 1
                    elif (
                        int(component["support_mode_count"]) > 1
                        and int(component["transition_rung_bin_count"]) >= 1
                        and int(component["left_side_bin_count"]) >= 1
                        and int(component["right_side_bin_count"]) >= 1
                    ):
                        zero_multi_mode_count += 1
                    else:
                        raise ValueError(
                            f"{token}: zero span has an unregistered support structure"
                        )
                if component.get("geometry_certificate_valid") is True:
                    if span <= 1e-12:
                        raise ValueError(f"{token}: valid certificate has zero span")
                    certificate_rows.append(
                        {
                            "local_transition_component_id": component[
                                "local_transition_component_id"
                            ],
                            "local_plane_pair_id": pair["local_plane_pair_id"],
                            "fragment_ids": fragment_ids,
                            "fragment_global_coverage_states": coverage_states,
                            "global_pair_state": pair["global_pair_comparison"]["state"],
                            "global_pair_line_state": component["global_pair_line_state"],
                            "intersection_span_endpoints": endpoints,
                            "span_length_mm": round(span, 9),
                            "support_parent_count": component["support_parent_count"],
                            "support_mode_count": component["support_mode_count"],
                            "occupied_bin_count": component["occupied_bin_count"],
                            "left_side_bin_count": component["left_side_bin_count"],
                            "right_side_bin_count": component["right_side_bin_count"],
                            "transition_rung_bin_count": component[
                                "transition_rung_bin_count"
                            ],
                            "component_sha256": component["component_sha256"],
                        }
                    )
        if component_count != int(
            manifest_row["case_counters"]["local_transition_component_count"]
        ):
            raise ValueError(f"{token}: component census mismatch")
        if len(certificate_rows) != int(
            manifest_row["case_counters"]["geometry_certificate_count"]
        ):
            raise ValueError(f"{token}: certificate census mismatch")
        total_zero += zero_count
        total_zero_single_mode += zero_single_mode_count
        total_zero_multi_mode += zero_multi_mode_count
        total_certificates += len(certificate_rows)
        output_rows.append(
            {
                "case_token": token,
                "probe_sha256": manifest_row["probe_sha256"],
                "engineering_profile_sha256": manifest_row[
                    "engineering_profile_sha256"
                ],
                "inference_semantic_sha256": manifest_row[
                    "inference_semantic_sha256"
                ],
                "peak_rss_bytes": manifest_row["peak_rss_bytes"],
                "fragment_count": manifest_row["case_counters"][
                    "local_plane_fragment_count"
                ],
                "orphan_count": manifest_row["case_counters"][
                    "local_plane_orphan_count"
                ],
                "pair_count": manifest_row["case_counters"][
                    "local_plane_pair_count"
                ],
                "component_count": component_count,
                "zero_length_component_count": zero_count,
                "zero_length_single_mode_no_rung_one_side_count": (
                    zero_single_mode_count
                ),
                "zero_length_multi_mode_same_parent_cross_rung_both_side_count": (
                    zero_multi_mode_count
                ),
                "certificate_count": len(certificate_rows),
                "primary_stage1_gold_count": int(primary_by_case.get(token, 0)),
                "postlock_reason_counts": dict(
                    sorted(reasons_by_case.get(token, Counter()).items())
                ),
                "certificates": sorted(
                    certificate_rows,
                    key=lambda value: value["local_transition_component_id"],
                ),
            }
        )
    if (
        total_zero != 15_609
        or total_zero_single_mode != 10_821
        or total_zero_multi_mode != 4_788
        or total_certificates != 31
    ):
        raise ValueError(
            "compact component census changed: "
            f"zero={total_zero}, single={total_zero_single_mode}, "
            f"multi={total_zero_multi_mode}, certs={total_certificates}"
        )
    if any(
        certificate["global_pair_line_state"] != "UNAVAILABLE"
        for row in output_rows
        for certificate in row["certificates"]
    ):
        raise ValueError("a locked certificate now has an available global-pair line")
    certificate_cases = [
        {"case_token": row["case_token"], "certificate_count": row["certificate_count"]}
        for row in output_rows
        if row["certificate_count"] > 0
    ]
    if certificate_cases != [
        {"case_token": "C017", "certificate_count": 2},
        {"case_token": "C018", "certificate_count": 3},
        {"case_token": "C019", "certificate_count": 1},
        {"case_token": "C022", "certificate_count": 11},
        {"case_token": "C032", "certificate_count": 1},
        {"case_token": "C035", "certificate_count": 8},
        {"case_token": "C036", "certificate_count": 5},
    ]:
        raise ValueError(f"certificate-case census changed: {certificate_cases}")
    return {
        "schema_version": 1,
        "kind": "f1_lpf_m1_v2_compact_per_case_certificate_census",
        "case_count": 37,
        "total_zero_length_component_count": total_zero,
        "zero_length_component_structure": {
            "all_support_parent_count_is_one": True,
            "single_mode_no_rung_one_side_count": total_zero_single_mode,
            "multi_mode_same_parent_cross_rung_both_side_count": (
                total_zero_multi_mode
            ),
            "other_zero_length_structure_count": 0,
            "lower_equals_upper_before_serialization": True,
            "endpoint_only_finite_extent_supported": False,
        },
        "total_geometry_certificate_count": total_certificates,
        "certificate_case_count": len(certificate_cases),
        "certificate_cases": certificate_cases,
        "cases": output_rows,
    }


def _combined_tests() -> str:
    sections = []
    for path in TEST_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT)
        sections.append(
            f"===== BEGIN {relative} =====\n"
            + path.read_text(encoding="utf-8").rstrip()
            + f"\n===== END {relative} =====\n"
        )
    return "\n".join(sections)


def main() -> int:
    missing = [str(path) for path in STATIC_FILES.values() if not path.is_file()]
    missing.extend(str(path) for path in TEST_FILES if not path.is_file())
    if missing:
        raise FileNotFoundError("missing professor STOP input:\n" + "\n".join(missing))
    _verify_source_hashes()
    manifest = _read(BATCH_MANIFEST)
    bundle = _read(BUNDLE)
    replay = _read(REPLAY)
    postlock = _read(POSTLOCK)
    summary = _locked_result_summary(
        manifest=manifest,
        bundle=bundle,
        replay=replay,
        postlock=postlock,
    )
    census = _per_case_census(manifest=manifest, postlock=postlock)

    PACKAGE.mkdir(parents=True, exist_ok=True)
    generated_names = {
        "14_LOCKED_RESULT_SUMMARY.json",
        "15_PER_CASE_CERTIFICATE_CENSUS.json",
        "16_COMBINED_LPF_V2_TESTS.txt",
    }
    expected_names = set(STATIC_FILES) | generated_names | {"manifest.json"}
    for stale in PACKAGE.iterdir():
        if stale.is_file() and stale.name not in expected_names:
            stale.unlink()
    for destination, source in STATIC_FILES.items():
        shutil.copy2(source, PACKAGE / destination)
    _write_json(PACKAGE / "14_LOCKED_RESULT_SUMMARY.json", summary)
    _write_json(PACKAGE / "15_PER_CASE_CERTIFICATE_CENSUS.json", census)
    (PACKAGE / "16_COMBINED_LPF_V2_TESTS.txt").write_text(
        _combined_tests(), encoding="utf-8"
    )

    source_by_name = {
        name: str(path.relative_to(ROOT)) for name, path in STATIC_FILES.items()
    }
    for name in generated_names:
        source_by_name[name] = "generated_by_builder"
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
    package_manifest = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-lpf-m1-v2-locked-stop-20260805-v1",
        "submission_style": "gpt_5_6_pro_individual_files_not_email_not_zip",
        "decision": "FORMAL_STOP",
        "next_decision": "V2_PRIMITIVE_REPLACEMENT_PROBE_SELECT_ONE_UPSTREAM_HYPOTHESIS",
        "runtime_promotion_authorized": False,
        "ownership_authorized": False,
        "births_authorized": False,
        "learning_authorized": False,
        "threshold_tuning_authorized": False,
        "locked_source_hashes": dict(EXPECTED_HASHES),
        "file_count_excluding_manifest": len(rows),
        "file_count_including_manifest": len(rows) + 1,
        "files": rows,
    }
    manifest_path = PACKAGE / "manifest.json"
    _write_json(manifest_path, package_manifest)
    for row in rows:
        if _sha256(PACKAGE / row["name"]) != row["sha256"]:
            raise RuntimeError(f"post-copy package hash mismatch: {row['name']}")
    print(
        json.dumps(
            {
                "package": str(PACKAGE),
                "prompt": str(PACKAGE / "00_GPT_PRO_PROMPT.md"),
                "files_including_manifest": len(rows) + 1,
                "manifest_sha256": _sha256(manifest_path),
                "locked_bundle_sha256": EXPECTED_HASHES["bundle"],
                "lock_attestation_sha256": EXPECTED_HASHES["lock_attestation"],
                "postlock_sha256": EXPECTED_HASHES["postlock"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
