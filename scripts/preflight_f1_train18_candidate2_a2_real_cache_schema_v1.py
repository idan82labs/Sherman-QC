#!/usr/bin/env python3
"""Label-incapable real-cache schema preflight for candidate2 amendment A2.

The process receives only the sealed TRAIN cache attestation.  Its exact-path
reader never authorizes a label attestation, projection, full-28 manifest,
holdout artifact, or score.  It validates all 18 TRAIN cases / 36
primary-replay records against every locked deletion arm and incumbent feature
representation, then emits aggregate PASS evidence only.  It emits no case,
row, route-distribution, score-distribution, prediction, or metric detail.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]


RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate2_conservative_deletion_a2_v1.py"
PREFLIGHT_SOURCE = Path(__file__).resolve()
BASE_RUNTIME_SOURCE = ROOT / "backend/f1_train18_frozen_source_only_candidate_v1.py"
BASE_RUNTIME_SHA256 = "85aba64cb7c088ee281344c675e3dd1578ac643a744cdc2b84da1967b6a11288"
CACHE_ATTESTATION_SHA256 = "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71"
CACHE_ATTESTATION = ROOT / "output/research/f1-aug30-train-only-custody-v2/cache/train_v51_cache_manifest_attestation.json"
PREFLIGHT_PASS_OUTPUT = ROOT / "research/f1_aug30_train18_candidate2_a2_real_cache_schema_preflight_pass_v1.json"
PASS_KIND = "sherman_qc_f1_aug30_train18_candidate2_a2_real_cache_schema_preflight_pass_v1"
REPRESENTATIONS = ("absolute_quality", "case_relative_quality", "intrinsic_quality")
HASH_MANIFEST_SHA256 = "df8d821ad3816f8fa72f7f9c34e3e0a9f8b07c88a503c9cd5b239a2a195928c8"
HASH_MANIFEST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_hashes_v1.json"
HASH_KIND = "sherman_qc_f1_aug30_retrospective_pseudo_unseen_hash_manifest_v1"
SPLIT_KIND = "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1"
ALLOWLIST_KIND = "sherman_qc_f1_aug30_train_only_scan_allowlist_v1"
ORIGINAL_PREREG_KIND = "sherman_qc_f1_aug30_retrospective_pseudo_unseen_prereg_v1"
TRAIN_CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
TRAIN_CACHE_ATTESTATION_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
BASE_RUNTIME_MODULE = "backend.f1_train18_frozen_source_only_candidate_v1"
A2_RUNTIME_MODULE = "backend.f1_train18_candidate2_conservative_deletion_a2_v1"
FORBIDDEN_PRELOADED_MODULES = (
    "backend",
    BASE_RUNTIME_MODULE,
    A2_RUNTIME_MODULE,
    "scripts.optimize_f1_train18_source_only_v1",
    "scripts.evaluate_f1_against_manual_gold",
)


def _lower_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class LockContext:
    split_sha256: str
    allowlist_sha256: str
    preregistration_sha256: str
    train_case_ids: tuple[str, ...]
    scan_sha_by_case: Mapping[str, str]
    train_commitment_sha256: str
    bootstrap_entries: tuple[dict[str, Any], ...]


@dataclass
class GuardedReader:
    allowed: dict[Path, str]
    expected_sha256: dict[Path, str] = field(default_factory=dict)
    expected_size_bytes: dict[Path, int] = field(default_factory=dict)
    entries: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[dict[str, Any]] = field(default_factory=list)

    def authorize(
        self,
        path: Path,
        *,
        purpose: str,
        expected_sha256: str,
        expected_size_bytes: int | None = None,
    ) -> bool:
        resolved = path.resolve()
        if resolved in self.allowed:
            if (
                self.allowed[resolved] != purpose
                or self.expected_sha256.get(resolved) != expected_sha256
                or self.expected_size_bytes.get(resolved) != expected_size_bytes
            ):
                raise ValueError(f"conflicting preflight authorization: {resolved}")
            return False
        _lower_sha(expected_sha256, name="authorized source hash")
        if expected_size_bytes is not None and (
            isinstance(expected_size_bytes, bool)
            or not isinstance(expected_size_bytes, int)
            or expected_size_bytes <= 0
        ):
            raise ValueError("authorized source size must be a positive integer")
        self.allowed[resolved] = purpose
        self.expected_sha256[resolved] = expected_sha256
        if expected_size_bytes is not None:
            self.expected_size_bytes[resolved] = expected_size_bytes
        return True

    def read_bytes(
        self, path: Path, *, purpose: str, case_id: str | None = None
    ) -> bytes:
        resolved = path.resolve()
        if self.allowed.get(resolved) != purpose:
            self.denied_attempts.append(
                {"path": str(resolved), "purpose": purpose, "case_id": case_id}
            )
            raise PermissionError(f"schema preflight read denied: {resolved}")
        data = resolved.read_bytes()
        if sha256_bytes(data) != self.expected_sha256.get(resolved):
            raise ValueError(f"schema preflight hash mismatch: {resolved}")
        size = self.expected_size_bytes.get(resolved)
        if size is not None and len(data) != size:
            raise ValueError(f"schema preflight size mismatch: {resolved}")
        self.entries.append(
            {
                "path": str(resolved),
                "purpose": purpose,
                "case_id": case_id,
                "sha256": sha256_bytes(data),
                "size_bytes": len(data),
            }
        )
        return data


def _read_bootstrap(
    path: Path, *, expected_sha256: str, purpose: str, entries: list[dict[str, Any]]
) -> tuple[dict[str, Any], bytes]:
    data = path.resolve().read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise ValueError(f"sealed bootstrap hash mismatch: {path}")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("sealed bootstrap must be a JSON object")
    entries.append(
        {
            "path": str(path.resolve()),
            "purpose": purpose,
            "case_id": None,
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
        }
    )
    return payload, data


def load_lock_context() -> LockContext:
    entries: list[dict[str, Any]] = []
    hashes, hash_bytes = _read_bootstrap(
        HASH_MANIFEST,
        expected_sha256=HASH_MANIFEST_SHA256,
        purpose="SEALED_HASH_MANIFEST",
        entries=entries,
    )
    if hashes.get("kind") != HASH_KIND or hashes.get("model_score_performed") is not False:
        raise ValueError("sealed hash manifest status mismatch")
    rows = hashes.get("artifact_bindings")
    if not isinstance(rows, list):
        raise ValueError("sealed hash manifest bindings missing")
    bindings = {str(row.get("path") or ""): row for row in rows if isinstance(row, dict)}

    def read_bound(relative: str, purpose: str) -> tuple[dict[str, Any], str]:
        row = bindings.get(relative)
        if not isinstance(row, dict):
            raise ValueError(f"sealed binding missing: {relative}")
        sha = _lower_sha(row.get("sha256"), name=f"{relative} hash")
        payload, _ = _read_bootstrap(
            ROOT / relative,
            expected_sha256=sha,
            purpose=purpose,
            entries=entries,
        )
        return payload, sha

    split, split_sha = read_bound(
        "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json",
        "SEALED_SPLIT_LOCK",
    )
    prereg, prereg_sha = read_bound(
        "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json",
        "SEALED_PREREGISTRATION",
    )
    allowlist, allowlist_sha = read_bound(
        "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json",
        "SEALED_TRAIN_ALLOWLIST",
    )
    if split.get("kind") != SPLIT_KIND or split.get("label_or_outcome_fields_present") is not False:
        raise ValueError("sealed split is invalid")
    if prereg.get("kind") != ORIGINAL_PREREG_KIND or prereg.get("evaluation_deferred") is not True:
        raise ValueError("sealed original preregistration is invalid")
    if allowlist.get("kind") != ALLOWLIST_KIND or allowlist.get("reject_any_case_not_listed") is not True:
        raise ValueError("sealed TRAIN allowlist is invalid")
    cases = allowlist.get("cases")
    if not isinstance(cases, list) or len(cases) != 18:
        raise ValueError("sealed TRAIN allowlist must contain 18 rows")
    train_ids = tuple(str(row.get("case_id") or "") for row in cases)
    scan_sha = {
        str(row.get("case_id") or ""): _lower_sha(
            row.get("scan_sha256"), name="TRAIN scan hash"
        )
        for row in cases
    }
    if (
        len(set(train_ids)) != 18
        or train_ids != tuple(str(value) for value in split.get("train_case_ids") or [])
        or allowlist.get("split_binding", {}).get("sha256") != split_sha
        or allowlist.get("prereg_binding", {}).get("sha256") != prereg_sha
    ):
        raise ValueError("sealed TRAIN membership bindings disagree")
    if sha256_bytes(hash_bytes) != HASH_MANIFEST_SHA256:
        raise RuntimeError("sealed hash manifest changed during bootstrap")
    commitment = sha256_bytes(
        canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_sha[case_id]}
                for case_id in train_ids
            ]
        )
    )
    return LockContext(
        split_sha256=split_sha,
        allowlist_sha256=allowlist_sha,
        preregistration_sha256=prereg_sha,
        train_case_ids=train_ids,
        scan_sha_by_case=scan_sha,
        train_commitment_sha256=commitment,
        bootstrap_entries=tuple(entries),
    )


def _bound_repository_path(binding: Mapping[str, Any], *, name: str) -> tuple[Path, str, int]:
    relative = str(binding.get("path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"{name} path must be repository-relative")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError(f"{name} path escapes repository root")
    size = binding.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{name} size must be positive")
    return path, _lower_sha(binding.get("sha256"), name=f"{name} hash"), size


def _validate_cache_attestation_before_manifest(
    attestation: Mapping[str, Any], *, lock: LockContext
) -> tuple[Path, str, int, list[tuple[Path, str, int, str]]]:
    if attestation.get("schema_version") != 1 or attestation.get("kind") != TRAIN_CACHE_ATTESTATION_KIND:
        raise ValueError("TRAIN cache attestation schema mismatch")
    checks = {
        "split": attestation.get("split_sha256") == lock.split_sha256,
        "allowlist": attestation.get("train_allowlist_sha256") == lock.allowlist_sha256,
        "commitment": attestation.get("train_set_commitment_sha256") == lock.train_commitment_sha256,
        "case_count": attestation.get("case_count") == 18,
        "record_count": attestation.get("record_count") == 36,
        "train_exact": attestation.get("train_exact_match") is True,
        "intersection": attestation.get("train_holdout_intersection_count") == 0,
        "holdout_entries": attestation.get("holdout_entry_count") == 0,
        "holdout_status": attestation.get("holdout_status_count") == 0,
        "holdout_hash": attestation.get("holdout_hash_count") == 0,
        "optimizer_holdout": attestation.get("optimizer_holdout_access_count") == 0,
        "independent_pass": attestation.get("independent_validation_passed") is True,
    }
    if not all(checks.values()):
        raise ValueError("TRAIN cache custody checks failed")
    evidence: list[tuple[Path, str, int, str]] = []
    for prefix in ("builder", "validator", "staged_validator_pass", "access_evidence"):
        binding = attestation.get(f"{prefix}_binding")
        if not isinstance(binding, dict):
            raise ValueError(f"TRAIN cache {prefix} binding missing")
        path, sha, size = _bound_repository_path(binding, name=f"TRAIN cache {prefix}")
        evidence.append((path, sha, size, f"CUSTODY_{prefix.upper()}_BINDING"))
    binding = attestation.get("manifest_binding")
    if not isinstance(binding, dict) or binding.get("kind") != TRAIN_CACHE_MANIFEST_KIND:
        raise ValueError("TRAIN cache manifest binding invalid")
    path, sha, size = _bound_repository_path(binding, name="TRAIN cache manifest")
    if path.name == "development_execution_manifest.json":
        raise PermissionError("full-28 execution manifest is forbidden")
    return path, sha, size, evidence


def _validate_train_cache_manifest(
    manifest: Mapping[str, Any], *, lock: LockContext
) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != TRAIN_CACHE_MANIFEST_KIND
        or manifest.get("source_only") is not True
        or manifest.get("holdout_entry_count") != 0
        or manifest.get("split_sha256") != lock.split_sha256
        or manifest.get("train_allowlist_sha256") != lock.allowlist_sha256
        or manifest.get("train_set_commitment_sha256") != lock.train_commitment_sha256
        or manifest.get("case_count") != 18
        or manifest.get("record_count") != 36
    ):
        raise ValueError("TRAIN cache manifest header invalid")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 36:
        raise ValueError("TRAIN cache manifest must contain 36 records")
    expected = [
        (case_id, pass_name)
        for case_id in lock.train_case_ids
        for pass_name in ("primary", "replay")
    ]
    allowed_keys = {
        "case_id", "runtime_case_id", "pass", "scan_sha256", "semantic_path",
        "semantic_sha256", "pool_path", "pool_sha256",
    }
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    runtime_by_case: dict[str, str] = {}
    case_by_runtime: dict[str, str] = {}
    binding_rows: list[dict[str, Any]] = []
    for raw, pair in zip(records, expected):
        if not isinstance(raw, dict) or set(raw) != allowed_keys:
            raise ValueError("TRAIN cache record schema invalid")
        case_id, pass_name = pair
        if (str(raw["case_id"]), str(raw["pass"])) != pair:
            raise PermissionError("TRAIN cache order/membership differs")
        runtime_id = str(raw["runtime_case_id"])
        if len(runtime_id) != 4 or not runtime_id.startswith("T") or not runtime_id[1:].isdigit():
            raise ValueError("TRAIN runtime identity invalid")
        if runtime_by_case.setdefault(case_id, runtime_id) != runtime_id:
            raise ValueError("TRAIN primary/replay runtime mapping differs")
        if case_by_runtime.setdefault(runtime_id, case_id) != case_id:
            raise ValueError("TRAIN runtime mapping is not bijective")
        scan_sha = _lower_sha(raw["scan_sha256"], name="TRAIN record scan hash")
        if scan_sha != lock.scan_sha_by_case[case_id]:
            raise ValueError("TRAIN record scan hash mismatch")
        semantic_path = (ROOT / str(raw["semantic_path"])).resolve()
        pool_path = (ROOT / str(raw["pool_path"])).resolve()
        if (
            Path(str(raw["semantic_path"])).is_absolute()
            or Path(str(raw["pool_path"])).is_absolute()
            or ROOT.resolve() not in semantic_path.parents
            or ROOT.resolve() not in pool_path.parents
            or semantic_path.name != "semantic_result.json"
            or pool_path.name != f"{runtime_id}.json"
        ):
            raise ValueError("TRAIN cache source path invalid")
        row = {
            "case_id": case_id,
            "runtime_case_id": runtime_id,
            "pass": pass_name,
            "scan_sha256": scan_sha,
            "semantic_path": semantic_path,
            "semantic_sha256": _lower_sha(raw["semantic_sha256"], name="semantic hash"),
            "pool_path": pool_path,
            "pool_sha256": _lower_sha(raw["pool_sha256"], name="pool hash"),
        }
        mapping[pair] = row
        binding_rows.append(
            {key: str(value) if isinstance(value, Path) else value for key, value in row.items()}
        )
    for case_id in lock.train_case_ids:
        primary = mapping[(case_id, "primary")]
        replay = mapping[(case_id, "replay")]
        if (
            primary["semantic_sha256"] != replay["semantic_sha256"]
            or primary["pool_sha256"] != replay["pool_sha256"]
        ):
            raise ValueError("TRAIN primary/replay cache hashes differ")
    return mapping, sha256_bytes(canonical_bytes(binding_rows))


def _verified_runtime_imports(
    *, runtime_code_sha256: str, preflight_code_sha256: str
) -> tuple[Any, Any, bytes, bytes]:
    for name in FORBIDDEN_PRELOADED_MODULES:
        if name in sys.modules:
            raise RuntimeError(f"preflight dependency was imported before verification: {name}")
    expected = {
        PREFLIGHT_SOURCE.resolve(): _lower_sha(preflight_code_sha256, name="preflight code SHA"),
        BASE_RUNTIME_SOURCE.resolve(): BASE_RUNTIME_SHA256,
        RUNTIME_SOURCE.resolve(): _lower_sha(runtime_code_sha256, name="A2 runtime SHA"),
    }
    source_bytes: dict[Path, bytes] = {}
    for path, sha in expected.items():
        data = path.read_bytes()
        if sha256_bytes(data) != sha:
            raise ValueError(f"pre-import source hash mismatch: {path}")
        source_bytes[path] = data

    package = types.ModuleType("backend")
    package.__file__ = str((ROOT / "backend/__init__.py").resolve())
    package.__package__ = "backend"
    package.__path__ = [str((ROOT / "backend").resolve())]
    sys.modules["backend"] = package

    def load(name: str, path: Path) -> Any:
        module = types.ModuleType(name)
        module.__file__ = str(path.resolve())
        module.__package__ = "backend"
        module.__loader__ = None
        sys.modules[name] = module
        try:
            code = compile(source_bytes[path.resolve()], str(path.resolve()), "exec")
            exec(code, module.__dict__)
        except Exception:
            sys.modules.pop(name, None)
            raise
        if Path(str(module.__file__)).resolve() != path.resolve():
            raise ImportError(f"verified module resolved from wrong file: {name}")
        return module

    try:
        base = load(BASE_RUNTIME_MODULE, BASE_RUNTIME_SOURCE)
        runtime = load(A2_RUNTIME_MODULE, RUNTIME_SOURCE)
    except Exception:
        sys.modules.pop(A2_RUNTIME_MODULE, None)
        sys.modules.pop(BASE_RUNTIME_MODULE, None)
        sys.modules.pop("backend", None)
        raise
    return (
        base,
        runtime,
        source_bytes[RUNTIME_SOURCE.resolve()],
        source_bytes[PREFLIGHT_SOURCE.resolve()],
    )


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite schema preflight: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))


def _synthetic_gate_model(
    *,
    runtime: Any,
    base: Any,
    mode: str,
    representation: str | None = None,
    explosion_rule: str | None = None,
) -> dict[str, Any]:
    """Build a truth-free portable model solely to exercise runtime arms."""

    hashes = {
        field: hashlib.sha256(f"schema-preflight|{field}".encode("ascii")).hexdigest()
        for field in runtime.HASH_BINDINGS
    }
    profile: dict[str, Any] = {"policy_mode": mode}
    classifier = None
    names = None
    if explosion_rule is not None:
        profile["explosion_rule"] = explosion_rule
    if mode in ("VETO_ONLY", "EXPLOSION_THEN_VETO"):
        if representation not in REPRESENTATIONS:
            raise ValueError("schema preflight veto representation is invalid")
        profile.update(
            {
                "learner": "LABEL_FREE_SCHEMA_PREFLIGHT_CONSTANT",
                "representation": representation,
                "veto_keep_threshold": 0.5,
            }
        )
        names = list(base.feature_names(representation))
        classifier = {
            "imputer_medians": [0.0] * len(names),
            "kind": "constant",
            "positive_probability": 1.0,
        }
    return {
        "schema_version": 1,
        "kind": runtime.MODEL_KIND,
        "attempt_ordinal": 2,
        "engineering_amendment": 2,
        "training_gate_passed": True,
        "train_gate_result_kind": (
            "sherman_qc_f1_aug30_train18_candidate2_a2_train_gate_pass_v1"
        ),
        "train_gate_status": "PASS",
        "holdout_score_opened": False,
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "source_only": True,
        "v51_fallback_default": True,
        "deletion_only": True,
        "case_id_feature_used": False,
        "runtime_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "observed_source_multiplicity_used": mode
        in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO"),
        "classifier_count_feature_used": False,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
        "profile_name": "LABEL_FREE_REAL_CACHE_SCHEMA_PREFLIGHT",
        "profile": profile,
        "feature_names": names,
        "classifier": classifier,
        **hashes,
    }


def run_preflight(
    *,
    cache_attestation: Path,
    cache_attestation_sha256: str,
    runtime_code_sha256: str,
    preflight_code_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    cache_attestation = cache_attestation.resolve()
    if cache_attestation != CACHE_ATTESTATION.resolve():
        raise PermissionError("schema preflight requires the immutable custody-v2 cache attestation path")
    if cache_attestation_sha256 != CACHE_ATTESTATION_SHA256:
        raise ValueError("schema preflight requires the immutable custody-v2 cache attestation SHA")
    output = output_path.resolve()
    if output != PREFLIGHT_PASS_OUTPUT.resolve() or output.exists():
        raise FileExistsError("schema preflight requires the exact new A2 PASS output path")
    base, runtime, runtime_bytes, preflight_bytes = _verified_runtime_imports(
        runtime_code_sha256=runtime_code_sha256,
        preflight_code_sha256=preflight_code_sha256,
    )
    lock = load_lock_context()
    reader = GuardedReader(
        allowed={
            cache_attestation: "TRAIN_CACHE_MANIFEST_ATTESTATION",
            RUNTIME_SOURCE.resolve(): "CANDIDATE2_A2_RUNTIME_SOURCE",
            PREFLIGHT_SOURCE.resolve(): "CANDIDATE2_A2_SCHEMA_PREFLIGHT_SOURCE",
            BASE_RUNTIME_SOURCE.resolve(): "BASE_RUNTIME_SOURCE",
        },
        expected_sha256={
            cache_attestation: _lower_sha(
                cache_attestation_sha256, name="TRAIN cache attestation SHA"
            ),
            RUNTIME_SOURCE.resolve(): runtime_code_sha256,
            PREFLIGHT_SOURCE.resolve(): preflight_code_sha256,
            BASE_RUNTIME_SOURCE.resolve(): BASE_RUNTIME_SHA256,
        },
        entries=[dict(row) for row in lock.bootstrap_entries],
    )
    runtime_bytes = reader.read_bytes(
        RUNTIME_SOURCE, purpose="CANDIDATE2_A2_RUNTIME_SOURCE"
    )
    preflight_bytes = reader.read_bytes(
        PREFLIGHT_SOURCE, purpose="CANDIDATE2_A2_SCHEMA_PREFLIGHT_SOURCE"
    )
    reader.read_bytes(BASE_RUNTIME_SOURCE, purpose="BASE_RUNTIME_SOURCE")
    attestation_bytes = reader.read_bytes(
        cache_attestation, purpose="TRAIN_CACHE_MANIFEST_ATTESTATION"
    )
    attestation = json.loads(attestation_bytes)
    if not isinstance(attestation, dict) or canonical_bytes(attestation) != attestation_bytes:
        raise ValueError("TRAIN cache attestation must be canonical JSON")
    manifest_path, manifest_sha, manifest_size, evidence = (
        _validate_cache_attestation_before_manifest(attestation, lock=lock)
    )
    for path, sha, size, purpose in evidence:
        if reader.authorize(
            path,
            purpose=purpose,
            expected_sha256=sha,
            expected_size_bytes=size,
        ):
            reader.read_bytes(path, purpose=purpose)
    reader.authorize(
        manifest_path,
        purpose="TRAIN_CACHE_MANIFEST",
        expected_sha256=manifest_sha,
        expected_size_bytes=manifest_size,
    )
    manifest_bytes = reader.read_bytes(
        manifest_path, purpose="TRAIN_CACHE_MANIFEST"
    )
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or canonical_bytes(manifest) != manifest_bytes:
        raise ValueError("TRAIN cache manifest must be canonical JSON")
    records, cache_binding_sha = _validate_train_cache_manifest(manifest, lock=lock)
    for record in records.values():
        reader.authorize(
            record["semantic_path"],
            purpose="TRAIN_V51_SEMANTIC",
            expected_sha256=record["semantic_sha256"],
        )
        reader.authorize(
            record["pool_path"],
            purpose="TRAIN_SCAN_ONLY_POOL",
            expected_sha256=record["pool_sha256"],
        )

    source_bytes: dict[tuple[str, str], tuple[bytes, bytes]] = {}
    for opaque_id in lock.train_case_ids:
        for pass_name in ("primary", "replay"):
            record = records[(opaque_id, pass_name)]
            semantic_bytes = reader.read_bytes(
                record["semantic_path"],
                purpose="TRAIN_V51_SEMANTIC",
                case_id=opaque_id,
            )
            pool_bytes = reader.read_bytes(
                record["pool_path"],
                purpose="TRAIN_SCAN_ONLY_POOL",
                case_id=opaque_id,
            )
            semantic = json.loads(semantic_bytes)
            pool = json.loads(pool_bytes)
            if not isinstance(semantic, dict) or not isinstance(pool, dict):
                raise ValueError("TRAIN source records must be JSON objects")
            base.validate_source_input(semantic)
            runtime_id = str(record["runtime_case_id"])
            if str(pool.get("case_id") or "") != runtime_id:
                raise ValueError("TRAIN pool runtime identity mismatch")
            inference = pool.get("inference_contract")
            if not isinstance(inference, dict) or any(
                inference.get(field) is not False
                for field in (
                    "part_identifier_exposed",
                    "expected_count_exposed",
                    "manual_truth_exposed",
                    "cad_exposed",
                )
            ):
                raise ValueError("TRAIN pool source-only contract mismatch")
            predictions = semantic.get("predictions")
            validated = runtime.validate_incumbent_predictions(
                predictions, require_sidecar_selector_score=False
            )
            for rule in runtime.EXPLOSION_RULES:
                runtime.apply_explosion_guard(validated, rule)
            observations, raw = base.build_observations(
                semantic, pool, universe="incumbent_only"
            )
            if len(observations) != len(validated):
                raise RuntimeError("incumbent feature alignment changed")
            for representation in REPRESENTATIONS:
                matrix = base.feature_matrix(
                    observations, raw, representation=representation
                )
                if matrix.shape != (
                    len(validated),
                    len(base.feature_names(representation)),
                ):
                    raise RuntimeError("candidate2 feature representation shape mismatch")
                if not base.np.isfinite(matrix).all():
                    raise ValueError("candidate2 veto representation contains non-finite values")
                veto_model = _synthetic_gate_model(
                    runtime=runtime,
                    base=base,
                    mode="VETO_ONLY",
                    representation=representation,
                )
                veto_result = runtime.compose_candidate2(semantic, pool, veto_model)
                if veto_result.get("kind") != runtime.RESULT_KIND:
                    raise RuntimeError("candidate2 VETO_ONLY runtime composition failed")
                for rule in runtime.EXPLOSION_RULES:
                    union_model = _synthetic_gate_model(
                        runtime=runtime,
                        base=base,
                        mode="EXPLOSION_THEN_VETO",
                        representation=representation,
                        explosion_rule=rule,
                    )
                    union_result = runtime.compose_candidate2(
                        semantic, pool, union_model
                    )
                    if union_result.get("kind") != runtime.RESULT_KIND:
                        raise RuntimeError(
                            "candidate2 EXPLOSION_THEN_VETO runtime composition failed"
                        )
            for rule in runtime.EXPLOSION_RULES:
                explosion_model = _synthetic_gate_model(
                    runtime=runtime,
                    base=base,
                    mode="EXPLOSION_ONLY",
                    explosion_rule=rule,
                )
                explosion_result = runtime.compose_candidate2(
                    semantic, pool, explosion_model
                )
                if explosion_result.get("kind") != runtime.RESULT_KIND:
                    raise RuntimeError(
                        "candidate2 EXPLOSION_ONLY runtime composition failed"
                    )
            source_bytes[(opaque_id, pass_name)] = (semantic_bytes, pool_bytes)

    for opaque_id in lock.train_case_ids:
        if source_bytes[(opaque_id, "primary")] != source_bytes[(opaque_id, "replay")]:
            raise ValueError("TRAIN primary/replay source bytes differ")
    case_scoped = [row for row in reader.entries if row.get("case_id") is not None]
    if len(case_scoped) != 72 or reader.denied_attempts:
        raise RuntimeError("schema preflight exact read accounting mismatch")
    authorized_rows = [
        {"path": str(path), "purpose": purpose}
        for path, purpose in reader.allowed.items()
    ]
    label_authorized_count = sum(
        "LABEL" in row["purpose"].upper() for row in authorized_rows
    )
    holdout_authorized_count = sum(
        "HOLDOUT" in row["purpose"].upper() for row in authorized_rows
    )
    forbidden_full28_names = {
        "development_execution_manifest.json",
        "bend_line_development_catalog_v2_20260830.json",
        "complete_label_manifest.json",
    }
    full28_authorized_count = sum(
        Path(row["path"]).name in forbidden_full28_names for row in authorized_rows
    )
    opened_purposes = [str(row["purpose"]).upper() for row in reader.entries]
    label_opened_count = sum("LABEL" in purpose for purpose in opened_purposes)
    holdout_opened_count = sum("HOLDOUT" in purpose for purpose in opened_purposes)
    full28_opened_count = sum(
        Path(str(row["path"])).name in forbidden_full28_names
        for row in reader.entries
    )
    case_ids_read = {str(row["case_id"]) for row in case_scoped}
    manifest_case_ids = {key[0] for key in records}
    if (
        label_authorized_count
        or holdout_authorized_count
        or full28_authorized_count
        or label_opened_count
        or holdout_opened_count
        or full28_opened_count
        or case_ids_read != set(lock.train_case_ids)
        or manifest_case_ids != set(lock.train_case_ids)
    ):
        raise PermissionError("schema preflight capability inventory is not TRAIN-only")
    committed_reads = [
        {
            "purpose": str(row["purpose"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in reader.entries
    ]
    committed_reads.sort(
        key=lambda row: (row["purpose"], row["sha256"], row["size_bytes"])
    )
    read_commitment = sha256_bytes(canonical_bytes(committed_reads))
    result = {
        "schema_version": 1,
        "kind": PASS_KIND,
        "status": "PASS",
        "attempt_ordinal": 2,
        "engineering_amendment": 2,
        "split_sha256": lock.split_sha256,
        "train_allowlist_sha256": lock.allowlist_sha256,
        "cache_attestation_sha256": sha256_bytes(attestation_bytes),
        "cache_manifest_sha256": manifest_sha,
        "train_source_cache_binding_sha256": cache_binding_sha,
        "runtime_code_sha256": sha256_bytes(runtime_bytes),
        "preflight_code_sha256": sha256_bytes(preflight_bytes),
        "base_runtime_code_sha256": BASE_RUNTIME_SHA256,
        "case_count": 18,
        "cache_record_count": 36,
        "semantic_pool_pair_count": 36,
        "case_scoped_file_read_count": 72,
        "primary_replay_source_bytes_exact": True,
        "policy_arms_qualified": list(runtime.POLICY_MODES),
        "explosion_rules_qualified": list(runtime.EXPLOSION_RULES),
        "representations_qualified": list(REPRESENTATIONS),
        "selector_score_contract": {
            "missing_or_null_is_unavailable": True,
            "non_null_must_be_finite_numeric_non_bool": True,
            "keep_top1_operational_sidecars_require_availability": True,
        },
        "aggregate_only_no_case_or_row_details": True,
        "read_set_commitment_sha256": read_commitment,
        "label_path_authorized_count": label_authorized_count,
        "label_path_opened_count": label_opened_count,
        "label_projection_resolved_or_opened": bool(label_authorized_count or label_opened_count),
        "full_28_manifest_authorized_or_opened": bool(
            full28_authorized_count or full28_opened_count
        ),
        "holdout_path_authorized_count": holdout_authorized_count,
        "holdout_artifact_opened": bool(holdout_opened_count),
        "score_or_metric_computed": False,
        "denied_read_count": 0,
    }
    _write_new(output, result)
    return {
        "status": "PASS",
        "output_path": str(output),
        "output_sha256": hashlib.sha256(canonical_bytes(result)).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-attestation", type=Path, required=True)
    parser.add_argument("--cache-attestation-sha256", required=True)
    parser.add_argument("--runtime-code-sha256", required=True)
    parser.add_argument("--preflight-code-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(
        cache_attestation=args.cache_attestation,
        cache_attestation_sha256=args.cache_attestation_sha256,
        runtime_code_sha256=args.runtime_code_sha256,
        preflight_code_sha256=args.preflight_code_sha256,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
