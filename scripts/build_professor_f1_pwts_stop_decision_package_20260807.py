#!/usr/bin/env python3
"""Build the exact ten-file PWTS-V1 STOP classification packet.

This packet is deliberately non-executable.  It asks GPT-5.6 Pro to classify
what the frozen structural-cap failure establishes and to authorize, at most,
preparation of a fresh post-PWTS architecture-selection packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ROOT = (
    ROOT
    / "output/research/f1-gold27-20260803"
    / "pwts-representation-preflight-v1"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_PWTS_Stop_Classification_20260807_v1"
)
DEFAULT_AUDIT_PATH = (
    PREFLIGHT_ROOT
    / "professor-stop-classification-package-v1"
    / "package_integrity_audit.json"
)
MAX_FILES = 10

CLASSIFICATION_TOKENS = (
    "PWTS_V1_PROTOCOL_INVALID_REPRESENTATION_UNRESOLVED",
    "PWTS_V1_REPRESENTATION_SELECTIVITY_INVALID",
    "PWTS_V1_SCIENTIFIC_CLASSIFICATION_INDETERMINATE",
)
NEXT_STEP_TOKENS = (
    "BUILD_FRESH_POST_PWTS_ARCHITECTURE_SELECTION_PACKET_ONLY",
    "FORMAL_PWTS_FAMILY_STOP_NO_NEW_EXECUTION",
    "PROVIDE_ONE_NONEXECUTION_MISSING_EVIDENCE_ITEM_ONLY",
)

SOURCE_BINDINGS = {
    "01_PRIOR_PWTS_PRO_RULING_VERBATIM.txt": {
        "path": ROOT
        / "docs/research/PROFESSOR_F1_PWTS_REPRESENTATION_PREFLIGHT_RESPONSE_20260806.txt",
        "sha256": "299e8d7f0a85c522ca4454587617d1f5df321c21d458bbda7cf7e2160d41f174",
    },
    "02_SOURCE_LOCK_INPUT_MANIFEST.json": {
        "path": PREFLIGHT_ROOT / "source_lock/input_manifest.json",
        "sha256": "0d7d02698fdfc873ca56ff2248d7df9876a3e00ef1662bb73caab2d68f5b0bc4",
    },
    "03_LOCKED_CONFIGURATION.json": {
        "path": PREFLIGHT_ROOT / "locks/configuration.json",
        "sha256": "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49",
    },
    "04_LOCKED_SCHEMA.json": {
        "path": PREFLIGHT_ROOT / "locks/schema.json",
        "sha256": "f3a83fee7bc5c516041b585361e96376dec846da557e5f8530943be00dc87426",
    },
    "05_RUN_V2_EXECUTION_LOCK.json": {
        "path": PREFLIGHT_ROOT / "run-v2/execution_lock.json",
        "sha256": "6d0305daf64d5cf12268e20e68624f3de1287495e99ede6d900cb6e5ff19023a",
    },
    "06_RUN_V2_ORCHESTRATION_STOP.json": {
        "path": PREFLIGHT_ROOT / "run-v2/orchestration_result.json",
        "sha256": "5493c57110d4158290cb707286fd283fb6e3568d466cdbab580ae093bd7183cf",
    },
    "07_RUN_V2_TERMINAL_CHECKPOINT.json": {
        "path": PREFLIGHT_ROOT / "run-v2/progress_checkpoint.json",
        "sha256": "317291b705a9dc34e0c6c023900742e5a0b5d515e9f0878287b289349aa5027f",
    },
    "08_C001_BASE_A_STDERR.txt": {
        "path": PREFLIGHT_ROOT / "run-v2/processes/C001/BASE_A/stderr.log",
        "sha256": "7e874801f083d6dbc681abb02eec73661d86de0314efa3e4bbc71700413baf8d",
    },
    "09_RUN_V1_ZERO_PROGRESS_PLATFORM_FAILURE.json": {
        "path": PREFLIGHT_ROOT / "run/launch_failure.json",
        "sha256": "213265c6bd42c383a8c777748fc5736b5d224e8f78d3984c18d96efe8ed20743",
    },
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _validate_content_binding(
    value: Mapping[str, Any], *, hash_key: str, label: str
) -> str:
    content = dict(value)
    asserted = content.pop(hash_key, None)
    observed = _sha_bytes(_canonical_json_bytes(content))
    if asserted != observed:
        raise ValueError(f"{label} internal SHA mismatch")
    return observed


def _validate_bound_sources() -> None:
    for name, binding in SOURCE_BINDINGS.items():
        path = Path(binding["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing bound source {name}: {path}")
        observed = _sha_file(path)
        if observed != binding["sha256"]:
            raise ValueError(
                f"bound source {name} differs from expected SHA-256: {observed}"
            )


def _validate_source_lock() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(SOURCE_BINDINGS["02_SOURCE_LOCK_INPUT_MANIFEST.json"]["path"])
    payload = _read_json(path)
    if (
        payload.get("kind") != "pwts_source_lock_v1"
        or payload.get("preflight_id") != "PWTS_REPRESENTATION_PREFLIGHT_V1"
        or payload.get("semantic_work_performed") is not False
        or payload.get("sampling_performed") is not False
        or payload.get("source_points_serialized") is not False
    ):
        raise ValueError("invalid PWTS source-lock safety fields")
    _validate_content_binding(
        payload, hash_key="input_manifest_sha256", label="source lock"
    )
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError("source-lock cases must be a list")
    by_token: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("source-lock row must be an object")
        row = dict(raw)
        token = str(row.get("case_token") or "")
        if token in by_token:
            raise ValueError(f"duplicate source-lock case token: {token}")
        by_token[token] = row
    expected = tuple(f"C{index:03d}" for index in range(1, 38))
    if tuple(sorted(by_token)) != expected:
        raise ValueError("source lock must bind exactly C001-C037")
    return payload, by_token


def _validate_raw_sources(
    source_rows: Mapping[str, Mapping[str, Any]],
    execution_lock: Mapping[str, Any],
) -> None:
    asserted = execution_lock.get("raw_source_sha256")
    if not isinstance(asserted, Mapping):
        raise ValueError("execution lock lacks raw-source hashes")
    for token, row in sorted(source_rows.items()):
        raw_path = Path(str(row.get("raw_path") or ""))
        expected = str(row.get("raw_ply_sha256") or "")
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing raw source for {token}: {raw_path}")
        observed = _sha_file(raw_path)
        if observed != expected or asserted.get(token) != observed:
            raise ValueError(f"{token}: raw-source SHA mismatch")


def _validate_stop_evidence(*, verify_raw_sources: bool) -> dict[str, Any]:
    source_manifest, source_rows = _validate_source_lock()
    config = _read_json(
        Path(SOURCE_BINDINGS["03_LOCKED_CONFIGURATION.json"]["path"])
    )
    schema = _read_json(Path(SOURCE_BINDINGS["04_LOCKED_SCHEMA.json"]["path"]))
    execution = _read_json(
        Path(SOURCE_BINDINGS["05_RUN_V2_EXECUTION_LOCK.json"]["path"])
    )
    orchestration_path = Path(
        SOURCE_BINDINGS["06_RUN_V2_ORCHESTRATION_STOP.json"]["path"]
    )
    orchestration = _read_json(orchestration_path)
    checkpoint = _read_json(
        Path(SOURCE_BINDINGS["07_RUN_V2_TERMINAL_CHECKPOINT.json"]["path"])
    )
    stderr_path = Path(SOURCE_BINDINGS["08_C001_BASE_A_STDERR.txt"]["path"])
    launch_failure = _read_json(
        Path(
            SOURCE_BINDINGS["09_RUN_V1_ZERO_PROGRESS_PLATFORM_FAILURE.json"][
                "path"
            ]
        )
    )

    config_sha = _sha_file(
        Path(SOURCE_BINDINGS["03_LOCKED_CONFIGURATION.json"]["path"])
    )
    schema_sha = _sha_file(Path(SOURCE_BINDINGS["04_LOCKED_SCHEMA.json"]["path"]))
    if config_sha != "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49":
        raise ValueError("locked configuration SHA mismatch")
    if schema_sha != "f3a83fee7bc5c516041b585361e96376dec846da557e5f8530943be00dc87426":
        raise ValueError("locked schema SHA mismatch")
    structural_caps = config.get("structural_caps")
    input_config = config.get("input")
    if not isinstance(structural_caps, Mapping) or not isinstance(input_config, Mapping):
        raise ValueError("locked configuration lacks input or structural caps")
    if structural_caps.get("maximum_seed_rows_per_case") != 1024:
        raise ValueError("unexpected maximum_seed_rows_per_case")
    if input_config.get("maximum_sample_points") != 200000:
        raise ValueError("unexpected maximum_sampled_points")
    if schema.get("source_points_serialized") is not False:
        raise ValueError("schema permits source-point serialization")

    execution_content_sha = _validate_content_binding(
        execution, hash_key="execution_lock_sha256", label="execution lock"
    )
    if execution.get("expected_configuration_sha256") != config_sha:
        raise ValueError("execution lock configuration mismatch")
    if execution.get("input_manifest_sha256") != source_manifest.get(
        "input_manifest_sha256"
    ):
        raise ValueError("execution lock source-manifest mismatch")
    if execution.get("limiter_mode") != "DARWIN_PROCESS_TREE_RSS_WATCHDOG":
        raise ValueError("unexpected run-v2 limiter mode")
    if execution.get("process_tree_monitor_verified") is not True:
        raise ValueError("run-v2 process-tree monitor was not verified")
    if execution.get("rss_sampling_interval_seconds") != 0.01:
        raise ValueError("unexpected RSS sampling interval")
    if execution.get("process_tree_peak_rss_cap_bytes") != 12_884_901_888:
        raise ValueError("unexpected process-tree RSS cap")
    source_hashes = execution.get("source_sha256")
    source_paths = execution.get("source_path")
    if not isinstance(source_hashes, Mapping) or not isinstance(source_paths, Mapping):
        raise ValueError("execution lock lacks source-code bindings")
    for label, expected in source_hashes.items():
        path = Path(str(source_paths.get(label) or ""))
        if not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"run-v2 bound source mismatch: {label}")
    if verify_raw_sources:
        _validate_raw_sources(source_rows, execution)

    if orchestration.get("terminal_status") != "INVALID_NEW_REPRESENTATION_PREFLIGHT_STOP":
        raise ValueError("run-v2 terminal status is not the locked STOP")
    required_orchestration = {
        "failure_code": "WORKER_NONZERO_EXIT",
        "completed_process_count": 1,
        "expected_process_count": 148,
        "semantic_process_started": True,
        "gold_or_truth_consumed": False,
        "source_points_serialized": False,
        "next_professor_packet_authorized": False,
        "runtime_or_production_authorized": False,
        "aggregate_serialized_bytes": 0,
        "workers": 1,
        "cpu_only": True,
    }
    for key, expected in required_orchestration.items():
        if orchestration.get(key) != expected:
            raise ValueError(f"run-v2 orchestration mismatch: {key}")
    resource_rows = orchestration.get("resource_rows")
    if not isinstance(resource_rows, list) or len(resource_rows) != 1:
        raise ValueError("run-v2 must contain exactly one resource row")
    row = resource_rows[0]
    if not isinstance(row, Mapping):
        raise ValueError("run-v2 resource row must be an object")
    required_row = {
        "case_token": "C001",
        "process_role": "BASE_A",
        "return_code": 1,
        "success": False,
        "reported_passing": False,
        "serialized_bytes": 0,
        "scratch_bytes": 0,
        "rss_cap_exceeded": False,
        "scratch_cap_exceeded": False,
        "wall_cap_exceeded": False,
        "full_lifetime_accounting": True,
    }
    for key, expected in required_row.items():
        if row.get(key) != expected:
            raise ValueError(f"run-v2 resource-row mismatch: {key}")
    if row.get("process_tree_peak_rss_bytes") != 844_136_448:
        raise ValueError("run-v2 peak RSS mismatch")
    if row.get("stderr_sha256") != _sha_file(stderr_path):
        raise ValueError("run-v2 stderr binding mismatch")
    stderr = stderr_path.read_text(encoding="utf-8")
    if (
        "maximum_seed_rows_per_case exceeded before serialization" not in stderr
        or "StructuralCapError" not in stderr
    ):
        raise ValueError("run-v2 stderr lacks the structural-cap failure")

    checkpoint_content_sha = _validate_content_binding(
        checkpoint, hash_key="checkpoint_sha256", label="terminal checkpoint"
    )
    if checkpoint.get("execution_lock_sha256") != execution_content_sha:
        raise ValueError("checkpoint execution-lock binding mismatch")
    if checkpoint.get("orchestration_result_sha256") != _sha_file(
        orchestration_path
    ):
        raise ValueError("checkpoint orchestration binding mismatch")
    if checkpoint.get("terminal_failure_code") != "WORKER_NONZERO_EXIT":
        raise ValueError("checkpoint failure code mismatch")
    if checkpoint.get("next_spec_index") != 0:
        raise ValueError("checkpoint must remain at process index zero")
    if checkpoint.get("completed_process_results") != []:
        raise ValueError("checkpoint must contain no successful process results")

    if (
        launch_failure.get("terminal_status")
        != "INVALID_LAUNCH_ATTEMPT_PRESERVED_NOT_PREFLIGHT_RESULT"
        or launch_failure.get("semantic_process_started") is not False
        or launch_failure.get("source_points_serialized") is not False
        or launch_failure.get("next_professor_packet_authorized") is not False
    ):
        raise ValueError("run-v1 is not the preserved zero-progress launch failure")

    return {
        "source_case_count": len(source_rows),
        "raw_sources_verified": verify_raw_sources,
        "configuration_sha256": config_sha,
        "schema_sha256": schema_sha,
        "execution_lock_content_sha256": execution_content_sha,
        "checkpoint_content_sha256": checkpoint_content_sha,
        "orchestration_sha256": _sha_file(orchestration_path),
        "known_seed_count_lower_bound": 1025,
        "exact_seed_count_known": False,
        "peak_rss_bytes": row["process_tree_peak_rss_bytes"],
        "serialized_bytes": row["serialized_bytes"],
    }


def _prompt(evidence_hashes: Mapping[str, str]) -> str:
    attachments = "\n".join(
        f"- `{name}` — `{digest}`" for name, digest in sorted(evidence_hashes.items())
    )
    classifications = "\n".join(f"- `{token}`" for token in CLASSIFICATION_TOKENS)
    next_steps = "\n".join(f"- `{token}`" for token in NEXT_STEP_TOKENS)
    return f"""# GPT-5.6 Pro extended-thinking — PWTS-V1 STOP classification

Treat this as a fresh, evidence-bound failure-classification consultation. Read
all nine attachments completely and independently verify their SHA-256 values.
Prior-chat authorization does not carry into this packet.

## What happened

The exact frozen PWTS-V1 preflight attempted C001/BASE_A only. After computing
the multiscale point field, it found more deterministic local-max seed rows than
the locked `maximum_seed_rows_per_case=1024`. It raised `StructuralCapError`
before serialization. Therefore the only supported seed-count fact is
`seed_count >= 1025`; the exact count and spatial distribution are unknown.

No trace, paired strip, certificate, bilateral-deletion negative, BASE_B,
order-permuted construction, or independent-validator construction completed.
No semantic payload was serialized. No gold/truth, expected bend count, manual
label, CAD/reference, render, prior F1 candidate, benchmark, or runtime path was
consumed. Run-v2 used one CPU worker, 55.21574 CPU seconds, 35.56373308401089
wall seconds, 844,136,448 bytes process-tree peak RSS, zero scratch bytes, and
zero serialized bytes. It did not hit a CPU, wall, RSS, scratch, or serialization
resource cap.

Run-v1 is supplied only to prove it was a separate zero-progress macOS
`RLIMIT_AS` launch failure. It started no semantic process and is not a
preflight result. Run-v2 used a verified 10 ms process-group RSS watchdog.
Disclose the residual limitation: on Darwin this is fail-closed measured
supervision, not an in-kernel memory cgroup, so a sub-sampling-interval transient
overshoot is theoretically possible.

## Scientific boundary

The locked protocol status is unambiguously
`INVALID_NEW_REPRESENTATION_PREFLIGHT_STOP`. Do not reinterpret it as a valid
preflight and do not authorize a rerun. The unresolved scientific question is
whether the result falsifies PWTS seed selectivity or merely proves that the
frozen seed definition and frozen evidence envelope are incompatible.

Primary literature establishes only relevant mechanisms, not the frozen
Sherman thresholds or the 1,024 cap:

- Mitra and Nguyen, *Estimating Surface Normals in Noisy Point Cloud Data*,
  DOI `10.1145/777792.777840`: normal estimation depends on neighborhood size,
  sampling density, curvature, and noise.
- Pauly, Keiser, and Gross, *Multi-scale Feature Extraction on Point-Sampled
  Surfaces*, DOI `10.1111/1467-8659.00675`: line-feature evidence changes with
  neighborhood scale.
- Ioannou et al., *Difference of Normals as a Multi-scale Operator in
  Unorganized Point Clouds*, DOI `10.1109/3DIMPVT.2012.12`: multiscale normal
  differences can expose scale-salient point-cloud structure.
- Demarsin et al., *Detection of Closed Sharp Edges in Point Clouds Using
  Normal Estimation and Graph Theory*, DOI `10.1016/j.cad.2006.12.005`: normal
  discontinuities and graphs can construct feature lines, but graph closure is
  not proof of a physical bend.

These papers make density/scale sensitivity plausible. They do not establish
the exact overflow cause, the correct cap, or bend accuracy on C001.

## Required answer

Return exactly two decision lines first:

`CLASSIFICATION: <exactly one token>`

{classifications}

`NEXT_STEP: <exactly one token>`

{next_steps}

Then provide the shortest rigorous rationale needed to support those lines.

If `BUILD_FRESH_POST_PWTS_ARCHITECTURE_SELECTION_PACKET_ONLY` is selected,
specify the exact non-execution evidence and research constraints that must be
included in that fresh packet. This token authorizes packet preparation and a
new Pro consultation only—not geometry execution.

If `PROVIDE_ONE_NONEXECUTION_MISSING_EVIDENCE_ITEM_ONLY` is selected, name
exactly one item obtainable from existing files without running geometry,
opening gold, or deriving the withheld exact seed count.

## Explicit prohibitions

Do not authorize or suggest a PWTS threshold change, cap increase, top-k or
reservoir truncation, PWTS-V2 rescue, count-only diagnostic, modified rerun,
subset rerun, remaining-case run, synthetic run, gold/truth/count access,
manual-label access, CAD/reference use, render review, accuracy claim, benchmark,
runtime/main integration, production use, or release promotion. Do not infer an
exact seed count or downstream bend behavior from the exception.

## Evidence SHA-256

{attachments}
"""


def build_package(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    audit_path: Path | None = DEFAULT_AUDIT_PATH,
    verify_raw_sources: bool = True,
) -> dict[str, Any]:
    _validate_bound_sources()
    validation = _validate_stop_evidence(verify_raw_sources=verify_raw_sources)

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    evidence_hashes: dict[str, str] = {}
    for name, binding in SOURCE_BINDINGS.items():
        destination = output_dir / name
        shutil.copyfile(Path(binding["path"]), destination)
        observed = _sha_file(destination)
        if observed != binding["sha256"]:
            raise ValueError(f"copied evidence SHA mismatch: {name}")
        evidence_hashes[name] = observed

    prompt_path = output_dir / "00_GPT_PRO_FAILURE_CLASSIFICATION_PROMPT.md"
    prompt_path.write_text(_prompt(evidence_hashes).rstrip() + "\n", encoding="utf-8")

    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    if len(files) != MAX_FILES:
        raise ValueError(f"expected exactly {MAX_FILES} individual files")
    if any(path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"} for path in files):
        raise ValueError("archives are forbidden in the GPT-style packet")

    packet_hashes = {path.name: _sha_file(path) for path in files}
    result = {
        "schema_version": 1,
        "package_id": "SHERMAN_QC_F1_PWTS_STOP_CLASSIFICATION_20260807_V1",
        "output_dir": str(output_dir.resolve()),
        "individual_file_count": len(files),
        "no_archive": True,
        "decision_only": True,
        "geometry_executed_by_builder": False,
        "gold_or_truth_consumed": False,
        "runtime_or_production_authorized": False,
        "pwts_rerun_authorized": False,
        "classification_tokens": list(CLASSIFICATION_TOKENS),
        "next_step_tokens": list(NEXT_STEP_TOKENS),
        "validation": validation,
        "packet_sha256": packet_hashes,
    }
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_bytes(_canonical_json_bytes(result) + b"\n")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument(
        "--skip-raw-source-verification",
        action="store_true",
        help="For isolated unit tests only; production packet builds verify all 37 raw files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_package(
        output_dir=args.output_dir,
        audit_path=args.audit_path,
        verify_raw_sources=not args.skip_raw_source_verification,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
