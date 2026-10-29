"""Load and verify the immutable V7 Cubical-IIP paper lock."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_DIR = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Cubical_IIP_V1_Preflight_Lock_Review_20260809_v7"
)
CONFIG_FILENAME = "05_V7_LOCKED_CUBICAL_IIP_V1_CONFIGURATION.json"
SCHEMA_FILENAME = "06_V7_LOCKED_CUBICAL_IIP_V1_SCHEMA.json"
PROOF_FILENAME = "07_V7_STATIC_CARDINALITY_EVIDENCE_RESOURCE_PROOF.json"
SOURCE_FILENAME = "01_V7_SOURCE_LOCK_INPUT_MANIFEST.json"

EXPECTED_CONFIG_FILE_SHA256 = (
    "813acdd96f5a8108caf2c87dfdd733b5c856248c4522d2789d733e84016dce04"
)
EXPECTED_CONFIG_SHA256 = (
    "876cf418b6a6411a2ea34be969d4d9bd9e66bcc167cac79f3a83ac3a06c40e95"
)
EXPECTED_SCHEMA_FILE_SHA256 = (
    "e996566e2b7da94dfc99b97e8bb59e5b6f1fdb25692a16890b63731b5c16ea4f"
)
EXPECTED_SCHEMA_SHA256 = (
    "b101e8177a96bf0a623db77c9b867d913371894cd9128ca7f25a1c8930c53376"
)
EXPECTED_PROOF_FILE_SHA256 = (
    "9726a1624afa115134135c0451ebf89e3f82241515000a18557988ab721da160"
)
EXPECTED_PROOF_SHA256 = (
    "a0d8b6b440ed6611db35f1f5ae7904eff1820a6b847be08dda8381949a4ec279"
)
EXPECTED_SOURCE_FILE_SHA256 = (
    "0d7d02698fdfc873ca56ff2248d7df9876a3e00ef1662bb73caab2d68f5b0bc4"
)
EXPECTED_SOURCE_INTERNAL_SHA256 = (
    "a87b10e9b14119a543d5d114d84579007a1272616b29c317f505812c4f4667fd"
)
EXPECTED_PACKAGE_ID = (
    "SHERMAN_QC_F1_CUBICAL_IIP_V1_PREFLIGHT_LOCK_REVIEW_20260809_V7"
)
_LOWER_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_SOURCE_CASE_COUNT = 37


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(
    path: Path, expected_file_hash: str, *, require_canonical_file: bool = True
) -> dict[str, Any]:
    raw = path.read_bytes()
    observed = sha256_bytes(raw)
    if observed != expected_file_hash:
        raise ValueError(
            f"frozen V7 file hash mismatch for {path.name}: "
            f"expected {expected_file_hash}, observed {observed}"
        )
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"frozen V7 JSON must be one object: {path.name}")
    if require_canonical_file and canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"frozen V7 JSON is not canonical: {path.name}")
    return dict(value)


def _verify_self_hash(value: Mapping[str, Any], key: str, expected: str) -> None:
    body = dict(value)
    asserted = body.pop(key, None)
    observed = sha256_bytes(canonical_json_bytes(body))
    if asserted != expected or observed != expected:
        raise ValueError(f"frozen V7 self-hash mismatch: {key}")


@dataclass(frozen=True)
class FrozenV7:
    packet_dir: Path
    configuration: Mapping[str, Any]
    schema: Mapping[str, Any]
    resource_proof: Mapping[str, Any]
    source_manifest: Mapping[str, Any]

    @property
    def sample_cap(self) -> int:
        return int(self.configuration["input"]["sample_cap"]["value"])

    @property
    def rss_watchdog_bytes(self) -> int:
        return int(
            self.configuration["resources"][
                "measured_process_tree_watchdog_bytes"
            ]["value"]
        )

    @property
    def wall_seconds(self) -> int:
        return int(
            self.configuration["resources"]["wall_seconds_per_role_maximum"][
                "value"
            ]
        )


def _frozen_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"frozen allocation integer invalid: {label}")
    return value


def _engineering_value(parent: Mapping[str, Any], key: str) -> int:
    row = parent.get(key)
    if not isinstance(row, Mapping) or set(row) != {"provenance", "value"}:
        raise ValueError(f"frozen engineering constant shape mismatch: {key}")
    if row["provenance"] != "PROPOSED_ENGINEERING_CONSTANT_NOT_LITERATURE_ESTABLISHED":
        raise ValueError(f"frozen engineering provenance mismatch: {key}")
    return _frozen_integer(row["value"], key)


def validate_frozen_allocation_plan(frozen: FrozenV7) -> Mapping[str, int]:
    """Recompute the complete V7 static allocation proof from frozen constants.

    This gate is source-free and must run before a worker reads any raw geometry.
    It deliberately derives every ledger entry instead of trusting the totals in
    file 07.
    """

    config = frozen.configuration
    proof = frozen.resource_proof
    allocation = config.get("allocation_model")
    resources = config.get("resources")
    reserves = config.get("static_allocation_reserves")
    if not all(isinstance(value, Mapping) for value in (allocation, resources, reserves)):
        raise ValueError("frozen allocation plan objects are missing")
    assert isinstance(allocation, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(reserves, Mapping)

    sample_cap = _engineering_value(config["input"], "sample_cap")
    occupied_max = _engineering_value(config["partition"], "logical_cell_cap")
    scale_count = _engineering_value(config["scales"], "scale_count")
    arm_count = _engineering_value(allocation, "baseline_plus_isotropic_arm_count")
    observed_records = _frozen_integer(
        config["source_envelope_observed"]["maximum_finite_coordinate_records_per_scan"],
        "maximum_finite_coordinate_records_per_scan",
    )
    vertex_max = scale_count * occupied_max
    edge_max = (4 * scale_count - 1) * occupied_max

    hard_payload_cap = _engineering_value(resources, "serialized_bytes_per_role_maximum")
    metadata_bytes = _frozen_integer(proof.get("metadata_reserve_bytes"), "metadata_reserve_bytes")
    ledger = {
        "raw_parser_and_chunk_buffers": _engineering_value(allocation, "raw_parser_and_chunk_buffers_bytes"),
        "all_canonical_xyz": observed_records * _engineering_value(allocation, "canonical_xyz_bytes_per_record"),
        "coordinate_radix_primary_and_buffer": observed_records
        * _engineering_value(allocation, "canonical_xyz_bytes_per_record")
        * _engineering_value(allocation, "coordinate_radix_buffer_copies"),
        "coordinate_radix_indices_primary_and_buffer": observed_records
        * _engineering_value(allocation, "coordinate_radix_index_bytes_per_record_per_buffer")
        * _engineering_value(allocation, "coordinate_radix_index_buffer_copies"),
        "hash_radix_primary_and_buffer": observed_records
        * _engineering_value(allocation, "hash_radix_bytes_per_record_per_buffer")
        * _engineering_value(allocation, "hash_radix_buffer_copies"),
        "point_id_and_sample_rank": observed_records
        * _engineering_value(allocation, "point_id_and_sample_rank_bytes_per_record"),
        "sampled_xyz_and_ids": sample_cap
        * _engineering_value(allocation, "sampled_xyz_and_ids_bytes_per_sample"),
        "partition_work": occupied_max
        * _engineering_value(allocation, "partition_work_bytes_per_occupied_cell"),
        "field_and_moment_work": vertex_max
        * _engineering_value(allocation, "field_and_moment_work_bytes_per_vertex"),
        "adjacency_work": edge_max
        * _engineering_value(allocation, "adjacency_work_bytes_per_edge"),
        "baseline_plus_isotropic_union_find_work": vertex_max
        * arm_count
        * _engineering_value(allocation, "union_find_work_bytes_per_vertex_per_arm"),
        "serializer_double_buffer": hard_payload_cap
        * _engineering_value(allocation, "serializer_buffer_count"),
        "fixed_logs_and_metadata": metadata_bytes,
    }
    observed_ledger = proof.get("semantic_allocation_ledger_bytes")
    if observed_ledger != ledger:
        raise ValueError("frozen semantic allocation ledger mismatch")
    ledger_total = sum(ledger.values())

    fixed = {
        "allocator_fragmentation": _engineering_value(reserves, "allocator_fragmentation_bytes"),
        "native_library_workspace": _engineering_value(reserves, "native_library_workspace_bytes"),
        "python_or_native_runtime": _engineering_value(reserves, "python_or_native_runtime_bytes"),
        "semantic_process_stack_and_misc": _engineering_value(reserves, "semantic_process_stack_and_misc_bytes"),
    }
    if reserves.get("preallocate_all_semantic_workspaces") is not True or (
        reserves.get("python_object_per_point_cell_edge_or_row_allowed") is not False
    ):
        raise ValueError("frozen allocation discipline mismatch")
    if proof.get("fixed_reserve_ledger_bytes") != fixed:
        raise ValueError("frozen fixed-reserve ledger mismatch")
    fixed_total = sum(fixed.values())
    static_total = ledger_total + fixed_total
    static_cap = _engineering_value(resources, "static_process_tree_bytes_maximum")
    array_cap = _engineering_value(resources, "application_array_bytes_maximum")
    scratch_cap = _engineering_value(resources, "scratch_bytes_per_role_maximum")

    tables = frozen.schema.get("tables")
    if not isinstance(tables, list) or len(tables) != 6:
        raise ValueError("frozen allocation schema table domain mismatch")
    row_sizes = {row["name"]: _frozen_integer(row["row_size_bytes"], f"{row['name']}.row_size") for row in tables}
    maximum_rows = {
        "partition_node": 2 * occupied_max - 1,
        "leaf_aggregate": occupied_max,
        "cell_scale_field": arm_count * scale_count * occupied_max,
        "space_scale_adjacency": edge_max,
        "persistence_disposition": arm_count * scale_count * occupied_max,
        "rejection": arm_count * scale_count * occupied_max,
    }
    maximum_bytes = {name: maximum_rows[name] * row_sizes[name] for name in maximum_rows}
    calculated_payload = metadata_bytes + sum(maximum_bytes.values())
    calculated_rows = 1 + sum(maximum_rows.values())
    role_count = _frozen_integer(proof.get("constants", {}).get("roles"), "roles")
    expected_proof = {
        "semantic_allocation_ledger_total_bytes": ledger_total,
        "semantic_allocation_headroom_bytes": array_cap - ledger_total,
        "fixed_reserve_total_bytes": fixed_total,
        "static_process_tree_calculated_bytes": static_total,
        "static_process_tree_configured_cap_bytes": static_cap,
        "static_process_tree_headroom_bytes": static_cap - static_total,
        "scratch_required_bytes": ledger["serializer_double_buffer"] + metadata_bytes,
        "scratch_cap_bytes": scratch_cap,
        "hard_payload_cap_bytes_per_complete_role": hard_payload_cap,
        "calculated_payload_bytes_per_complete_role": calculated_payload,
        "payload_headroom_bytes": hard_payload_cap - calculated_payload,
        "calculated_rows_including_metadata_record": calculated_rows,
        "maximum_hard_cap_bytes_per_case_all_roles": role_count * hard_payload_cap,
        "maximum_hard_cap_bytes_source_manifest_all_roles": role_count
        * hard_payload_cap
        * EXPECTED_SOURCE_CASE_COUNT,
        "application_array_cap_bytes": array_cap,
        "maximum_rows_per_complete_role": maximum_rows,
        "maximum_bytes_per_complete_role": maximum_bytes,
    }
    for key, expected in expected_proof.items():
        if proof.get(key) != expected:
            raise ValueError(f"frozen recomputed allocation mismatch: {key}")
    if any(value < 0 for value in (
        expected_proof["semantic_allocation_headroom_bytes"],
        expected_proof["static_process_tree_headroom_bytes"],
        expected_proof["payload_headroom_bytes"],
        scratch_cap - expected_proof["scratch_required_bytes"],
    )):
        raise ValueError("frozen allocation plan exceeds a configured cap")
    constants = proof.get("constants")
    expected_constants = {
        "N": sample_cap,
        "M_observed_max_finite_records": observed_records,
        "O_max": occupied_max,
        "S": scale_count,
        "V_max": vertex_max,
        "E_max": edge_max,
        "arms": arm_count,
        "roles": role_count,
        "raw_bytes_observed_max": _frozen_integer(
            config["source_envelope_observed"]["maximum_raw_bytes_per_scan"],
            "maximum_raw_bytes_per_scan",
        ),
        "stencil_widths": [(2 * radius + 1) ** 3 for radius in (2, 3, 4)],
        "union_find_parent_depth_bound": (vertex_max - 1).bit_length(),
    }
    if constants != expected_constants:
        raise ValueError("frozen allocation constants mismatch")
    return {
        "semantic_allocation_bytes": ledger_total,
        "fixed_reserve_bytes": fixed_total,
        "static_process_tree_bytes": static_total,
        "scratch_required_bytes": expected_proof["scratch_required_bytes"],
        "serialized_payload_bytes": calculated_payload,
    }


def load_frozen_v7(packet_dir: Path = DEFAULT_PACKET_DIR) -> FrozenV7:
    packet_dir = packet_dir.resolve()
    config = _load_json(packet_dir / CONFIG_FILENAME, EXPECTED_CONFIG_FILE_SHA256)
    schema = _load_json(packet_dir / SCHEMA_FILENAME, EXPECTED_SCHEMA_FILE_SHA256)
    proof = _load_json(packet_dir / PROOF_FILENAME, EXPECTED_PROOF_FILE_SHA256)
    source = _load_json(
        packet_dir / SOURCE_FILENAME,
        EXPECTED_SOURCE_FILE_SHA256,
        require_canonical_file=False,
    )

    _verify_self_hash(config, "configuration_sha256", EXPECTED_CONFIG_SHA256)
    _verify_self_hash(schema, "schema_sha256", EXPECTED_SCHEMA_SHA256)
    _verify_self_hash(proof, "resource_proof_sha256", EXPECTED_PROOF_SHA256)
    _verify_self_hash(source, "input_manifest_sha256", EXPECTED_SOURCE_INTERNAL_SHA256)

    if config.get("package_id") != EXPECTED_PACKAGE_ID:
        raise ValueError("frozen V7 package id mismatch")
    if config.get("authority", {}).get("execution_authority") != "NONE":
        raise ValueError("paper packet unexpectedly grants execution authority")
    if proof.get("configuration_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("resource proof does not bind frozen configuration")
    if proof.get("schema_sha256") != EXPECTED_SCHEMA_SHA256:
        raise ValueError("resource proof does not bind frozen schema")
    if source.get("semantic_work_performed") is not False:
        raise ValueError("source lock was semantically contaminated")
    if source.get("sampling_performed") is not False:
        raise ValueError("source lock unexpectedly sampled input")
    if source.get("source_points_serialized") is not False:
        raise ValueError("source lock contains source points")
    frozen = FrozenV7(packet_dir, config, schema, proof, source)
    validate_frozen_allocation_plan(frozen)
    return frozen


def select_smoke_source_by_content(frozen: FrozenV7) -> Mapping[str, Any]:
    """Choose exactly one source solely by its frozen raw-content digest."""

    rows = frozen.source_manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_CASE_COUNT:
        raise ValueError(
            f"frozen source manifest must contain exactly {EXPECTED_SOURCE_CASE_COUNT} cases"
        )
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("frozen source manifest contains a non-object row")
    hashes = [row.get("raw_ply_sha256") for row in rows]
    if (
        any(not isinstance(value, str) or _LOWER_HEX_SHA256.fullmatch(value) is None for value in hashes)
        or len(set(hashes)) != len(hashes)
    ):
        raise ValueError("source manifest raw hashes are invalid or duplicated")
    selected_hash = min(hashes)
    return dict(rows[hashes.index(selected_hash)])
