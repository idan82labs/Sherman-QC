from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from backend import f1_line_object_clustering as clusterer
from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as hypothesis
from backend import f1_train18_candidate9_lossless_pool_alias_adapter_v1 as adapter
from scripts import build_f1_scan_only_candidate_pool as builder


ROOT = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _context() -> adapter.AdapterContext:
    return adapter.make_verified_context(
        hypothesis=hypothesis,
        builder=builder,
        clusterer=clusterer,
        hypothesis_source_sha256=adapter.HYPOTHESIS_SOURCE_SHA256,
        builder_source_sha256=adapter.BUILDER_SOURCE_SHA256,
        cluster_source_sha256=adapter.CLUSTER_SOURCE_SHA256,
    )


def _row(identifier: str = "LO1", **updates) -> dict[str, object]:  # type: ignore[no-untyped-def]
    value: dict[str, object] = {
        "bend_id": identifier,
        "candidate_id": "plane_lattice::RAW1",
        "candidate_source": "cross_source_line_object",
        "cluster_member_count": 1,
        "cluster_member_ids": ["plane_lattice::RAW1"],
        "cluster_source_count": 1,
        "cluster_sources": ["plane_lattice"],
        "representative_source": "plane_lattice",
        "source_kind": "scan_supported_plane_intersection",
        "source_candidate_id": "RAW1",
        "span_endpoints": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "score": 0.75,
        "finite_support_fraction": 0.8,
        "endpoint_support_fraction": 0.7,
        "two_sided_transition_score": 0.6,
        "normal_second_eigenvalue": 0.02,
    }
    value.update(updates)
    return value


def _pool(rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "algorithm": adapter.POOL_ALGORITHM,
        "inference_contract": dict(hypothesis.POOL_INFERENCE_CONTRACT),
        "line_objects": list(rows if rows is not None else [_row()]),
        "schema_version": adapter.POOL_SCHEMA_VERSION,
        "source_point_serialization": False,
        "working_point_budget": adapter.WORKING_POINT_BUDGET,
    }


def _pool_bytes(rows: list[dict[str, object]] | None = None) -> bytes:
    return adapter.canonical_bytes(_pool(rows))


def _normalize(data: bytes) -> bytes:
    return adapter.normalize_pool_bytes(
        _context(), pool_bytes=data, expected_pool_sha256=_sha(data)
    )


def test_runtime_top_level_imports_are_stdlib_only() -> None:
    source = ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    allowed_roots = {
        "__future__",
        "dataclasses",
        "hashlib",
        "json",
        "re",
        "types",
        "typing",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed_roots for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert str(node.module or "").split(".")[0] in allowed_roots


def test_bound_builder_cluster_and_hypothesis_hashes_match_exact_sources() -> None:
    bindings = {
        ROOT / "scripts/build_f1_scan_only_candidate_pool.py": adapter.BUILDER_SOURCE_SHA256,
        ROOT / "backend/f1_line_object_clustering.py": adapter.CLUSTER_SOURCE_SHA256,
        ROOT
        / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py": adapter.HYPOTHESIS_SOURCE_SHA256,
    }
    for path, expected in bindings.items():
        assert _sha(path.read_bytes()) == expected


@pytest.mark.parametrize(
    "binding_name",
    ("hypothesis_source_sha256", "builder_source_sha256", "cluster_source_sha256"),
)
def test_verified_context_rejects_any_source_binding_drift(binding_name: str) -> None:
    values = {
        "hypothesis": hypothesis,
        "builder": builder,
        "clusterer": clusterer,
        "hypothesis_source_sha256": adapter.HYPOTHESIS_SOURCE_SHA256,
        "builder_source_sha256": adapter.BUILDER_SOURCE_SHA256,
        "cluster_source_sha256": adapter.CLUSTER_SOURCE_SHA256,
    }
    values[binding_name] = "0" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        adapter.make_verified_context(**values)  # type: ignore[arg-type]


def test_forged_context_is_rejected() -> None:
    context = adapter.AdapterContext(
        hypothesis=hypothesis,
        builder=builder,
        clusterer=clusterer,
        allowed_source_kinds=hypothesis.ALLOWED_SOURCE_KINDS,
        _sentinel=object(),
    )
    data = _pool_bytes()
    with pytest.raises(PermissionError, match="verified source context"):
        adapter.normalize_pool_bytes(
            context, pool_bytes=data, expected_pool_sha256=_sha(data)
        )


def test_direct_builder_cluster_fixture_removes_only_inherited_source_alias() -> None:
    raw = {
        "bend_id": "RAW1",
        "source_kind": "scan_supported_plane_intersection",
        "span_endpoints": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "score": 0.75,
        "finite_support_fraction": 0.8,
        "endpoint_support_fraction": 0.7,
        "two_sided_transition_score": 0.6,
        "normal_second_eigenvalue": 0.02,
    }
    tagged = builder._tag([raw], "plane_lattice")
    clustered = clusterer.cluster_line_candidates(tagged)
    assert len(clustered) == 1
    assert clustered[0]["candidate_source"] == "cross_source_line_object"
    assert clustered[0]["representative_source"] == "plane_lattice"
    assert clustered[0]["source_kind"] == "scan_supported_plane_intersection"
    data = adapter.canonical_bytes(_pool(clustered))
    normalized = _normalize(data)
    output = json.loads(normalized)
    expected_row = deepcopy(clustered[0])
    expected_row.pop("source_kind")
    assert output["line_objects"] == [expected_row]
    _pool_value, candidates = hypothesis._validate_pool_bytes(
        normalized, expected_sha256=_sha(normalized)
    )
    assert len(candidates) == 1
    assert candidates[0]["features"]["source_kind"] == "plane_lattice"


def test_direct_builder_cluster_partial_geometry_fixture_is_projection_equivalent() -> None:
    raw = {
        "bend_id": "RAW_PARTIAL",
        "source_kind": "scan_supported_plane_intersection",
        "span_endpoints": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "span_mm": 10.0,
        "score": 0.5,
    }
    clustered = clusterer.cluster_line_candidates(builder._tag([raw], "plane_lattice"))
    assert len(clustered) == 1
    assert clustered[0]["span_mm"] == 10.0
    assert "anchor" not in clustered[0] and "axis_direction" not in clustered[0]
    original = adapter.canonical_bytes(_pool(clustered))
    normalized = adapter.normalize_pool_bytes(
        _context(), pool_bytes=original, expected_pool_sha256=_sha(original)
    )
    output_row = json.loads(normalized)["line_objects"][0]
    expected = deepcopy(clustered[0])
    expected.pop("source_kind")
    expected.pop("span_mm")
    assert output_row == expected
    adapter.verify_projection_byte_equivalence(
        _context(),
        original_pool_bytes=original,
        normalized_pool_bytes=normalized,
    )


def test_projection_preserves_every_other_value_type_missingness_and_order() -> None:
    rows = [
        _row("LO2", score=2, optional_null=None),
        _row(
            "LO1",
            candidate_source="scan_mesh_adjacency",
            representative_source="scan_mesh_adjacency",
            source_kind="scan_mesh_adjacency",
            score=-0.0,
        ),
    ]
    original = _pool(rows)
    data = adapter.canonical_bytes(original)
    output = json.loads(_normalize(data))
    expected = deepcopy(original)
    del expected["line_objects"][0]["source_kind"]  # type: ignore[index]
    assert output == expected
    assert [row["bend_id"] for row in output["line_objects"]] == ["LO2", "LO1"]
    assert output["line_objects"][0]["score"] == 2
    assert output["line_objects"][0]["optional_null"] is None


@pytest.mark.parametrize(
    "partial",
    (
        {"anchor": [5.0, 0.0, 0.0]},
        {"axis_direction": [1.0, 0.0, 0.0]},
        {"span_mm": 10.0},
        {"anchor": [5.0, 0.0, 0.0], "axis_direction": [1.0, 0.0, 0.0]},
        {"anchor": [5.0, 0.0, 0.0], "span_mm": 10.0},
        {"axis_direction": [1.0, 0.0, 0.0], "span_mm": 10.0},
    ),
)
def test_authoritative_span_drops_only_incomplete_auxiliary_geometry_aliases(
    partial: dict[str, object],
) -> None:
    row = _row(**partial)
    normalized = json.loads(_normalize(_pool_bytes([row])))
    output = normalized["line_objects"][0]
    for field in adapter.AUXILIARY_GEOMETRY_FIELDS:
        assert field not in output
    assert output["span_endpoints"] == row["span_endpoints"]


@pytest.mark.parametrize(
    "conflict",
    (
        {"anchor": [999.0, 0.0, 0.0]},
        {"axis_direction": [0.0, 1.0, 0.0]},
        {"span_mm": 11.0},
        {
            "anchor": [999.0, 0.0, 0.0],
            "axis_direction": [0.0, 1.0, 0.0],
        },
    ),
)
def test_conflicting_partial_geometry_is_rejected_before_alias_removal(
    conflict: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="conflicts with authoritative span"):
        _normalize(_pool_bytes([_row(**conflict)]))


def test_complete_equivalent_redundant_geometry_is_preserved() -> None:
    row = _row(
        anchor=[5.0, 0.0, 0.0],
        axis_direction=[1.0, 0.0, 0.0],
        span_mm=10.0,
    )
    output = json.loads(_normalize(_pool_bytes([row])))["line_objects"][0]
    for field in adapter.AUXILIARY_GEOMETRY_FIELDS:
        assert output[field] == row[field]


def test_complete_conflicting_redundant_geometry_is_rejected() -> None:
    row = _row(
        anchor=[50.0, 0.0, 0.0],
        axis_direction=[1.0, 0.0, 0.0],
        span_mm=10.0,
    )
    with pytest.raises(ValueError, match="conflicting line representations"):
        _normalize(_pool_bytes([row]))


def test_incomplete_auxiliary_geometry_without_authoritative_span_is_rejected() -> None:
    row = _row(anchor=[5.0, 0.0, 0.0], axis_direction=[1.0, 0.0, 0.0])
    row.pop("span_endpoints")
    with pytest.raises(ValueError, match="incomplete nonredundant geometry"):
        _normalize(_pool_bytes([row]))


def test_complete_auxiliary_geometry_without_span_uses_unchanged_c6_reconstruction() -> None:
    row = _row(
        anchor=[5.0, 0.0, 0.0],
        axis_direction=[1.0, 0.0, 0.0],
        span_mm=10.0,
    )
    row.pop("span_endpoints")
    output = json.loads(_normalize(_pool_bytes([row])))["line_objects"][0]
    assert output["anchor"] == row["anchor"]
    assert output["axis_direction"] == row["axis_direction"]
    assert output["span_mm"] == row["span_mm"]
    assert "span_endpoints" not in output


@pytest.mark.parametrize(
    "mutation",
    (
        {"representative_source": None},
        {"representative_source": "unknown_generator"},
        {"sourceKind": "plane_lattice"},
    ),
)
def test_unknown_or_ambiguous_clustered_source_representation_is_rejected(
    mutation: dict[str, object],
) -> None:
    row = _row()
    row.update(mutation)
    with pytest.raises(ValueError):
        _normalize(_pool_bytes([row]))


def test_nonclustered_disagreeing_source_aliases_are_rejected() -> None:
    row = _row(
        candidate_source="plane_lattice",
        representative_source="plane_lattice",
        source_kind="scan_mesh_adjacency",
    )
    with pytest.raises(ValueError, match="ambiguous source aliases"):
        _normalize(_pool_bytes([row]))


@pytest.mark.parametrize(
    "mutation",
    (
        {"cluster_sources": ["cylindrical_standard"]},
        {"cluster_source_count": 2},
        {"cluster_member_count": 2},
        {"cluster_sources": ["plane_lattice", "plane_lattice"]},
        {"cluster_member_ids": ["B", "A"], "cluster_member_count": 2},
        {"cluster_member_ids": ["cylindrical_wide::FOREIGN"]},
        {
            "cluster_member_ids": ["BAD", "plane_lattice::RAW1"],
            "cluster_member_count": 2,
        },
        {"cluster_member_ids": ["plane_lattice::"]},
        {"candidate_id": "cylindrical_wide::FOREIGN"},
        {"source_candidate_id": "FOREIGN"},
    ),
)
def test_ambiguous_or_noncanonical_cluster_provenance_is_rejected(
    mutation: dict[str, object],
) -> None:
    row = _row()
    row.update(mutation)
    with pytest.raises(ValueError, match="cluster"):
        _normalize(_pool_bytes([row]))


@pytest.mark.parametrize(
    "partial",
    (
        {"anchor": [1.0e9, 0.0, 0.0]},
        {"axis_direction": [0.0, 0.0, 0.0]},
        {"span_mm": -1.0},
    ),
)
def test_malformed_or_out_of_domain_partial_geometry_is_rejected(
    partial: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _normalize(_pool_bytes([_row(**partial)]))


def test_duplicate_line_object_identifiers_are_rejected_without_deduplication() -> None:
    with pytest.raises(ValueError, match="duplicate line-object identifiers"):
        _normalize(_pool_bytes([_row("LO1"), _row("LO1")]))


def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected() -> None:
    duplicate = b'{"algorithm":"x","algorithm":"y"}\n'
    with pytest.raises(ValueError, match="invalid JSON"):
        adapter.normalize_pool_bytes(
            _context(), pool_bytes=duplicate, expected_pool_sha256=_sha(duplicate)
        )
    nonfinite = b'{"value":NaN}\n'
    with pytest.raises(ValueError, match="invalid JSON"):
        adapter.normalize_pool_bytes(
            _context(), pool_bytes=nonfinite, expected_pool_sha256=_sha(nonfinite)
        )


def test_recursive_forbidden_truth_key_is_not_removed_or_hidden() -> None:
    pool = _pool()
    pool["line_objects"][0]["gold_label"] = True  # type: ignore[index]
    data = adapter.canonical_bytes(pool)
    with pytest.raises(ValueError, match="forbidden key"):
        _normalize(data)


def test_external_input_hash_is_mandatory() -> None:
    data = _pool_bytes()
    with pytest.raises(ValueError, match="external binding"):
        adapter.normalize_pool_bytes(
            _context(), pool_bytes=data, expected_pool_sha256="0" * 64
        )


def test_adapted_result_binds_original_and_adapted_hashes() -> None:
    data = _pool_bytes()
    result = adapter.adapt_pool_bytes(
        _context(), pool_bytes=data, expected_pool_sha256=_sha(data)
    )
    assert result.original_pool_sha256 == _sha(data)
    assert result.adapted_pool_sha256 == _sha(result.adapted_pool_bytes)
    assert result.projection_byte_equivalent is True
    assert result.original_pool_sha256 != result.adapted_pool_sha256


def test_projection_equivalence_rejects_any_nonadapter_mutation() -> None:
    context = _context()
    data = _pool_bytes()
    normalized = adapter.normalize_pool_bytes(
        context, pool_bytes=data, expected_pool_sha256=_sha(data)
    )
    mutated = json.loads(normalized)
    mutated["line_objects"][0]["score"] = 0.76
    with pytest.raises(ValueError, match="projection-byte equivalent"):
        adapter.verify_projection_byte_equivalence(
            context,
            original_pool_bytes=data,
            normalized_pool_bytes=adapter.canonical_bytes(mutated),
        )


def test_contract_declares_no_scientific_or_selection_change() -> None:
    value = adapter.contract_declaration()
    assert value["attempt_ordinal"] == 9
    assert value["candidate8_disposition_sha256"] == adapter.CANDIDATE8_DISPOSITION_SHA256
    assert value["external_verified_source_bytes_required"] is True
    assert value["bound_source_sha256"] == {
        "builder": adapter.BUILDER_SOURCE_SHA256,
        "candidate6_hypothesis": adapter.HYPOTHESIS_SOURCE_SHA256,
        "clusterer": adapter.CLUSTER_SOURCE_SHA256,
    }
    assert value["scientific_hypothesis_id"] == adapter.SCIENTIFIC_HYPOTHESIS_ID
    assert value["scientific_hypothesis_performance_tested"] is False
    contract = value["representation_contract"]
    for key in (
        "added_rows",
        "authoritative_candidate_geometry_changed",
        "candidate_features_changed",
        "candidate_identifiers_changed",
        "candidate_scores_changed",
        "deduplication_performed",
        "dropped_rows",
        "feature_imputation_performed",
        "filtering_performed",
        "ownership_changed",
        "rule_or_threshold_changed",
        "selection_performed",
    ):
        assert contract[key] is False
    assert contract["one_to_one_row_mapping_required"] is True
    assert contract["order_preserved"] is True
