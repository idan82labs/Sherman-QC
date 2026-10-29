from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "render_exp_cad_0030_cfpdg_native.py"
SPEC = importlib.util.spec_from_file_location("exp30_renderer", PATH)
RENDERER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RENDERER)

SYNTHETIC_E1 = np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0)
SYNTHETIC_NORMAL = np.asarray([1.0, -1.0, 2.0]) / np.sqrt(6.0)
SYNTHETIC_E2 = np.cross(SYNTHETIC_NORMAL, SYNTHETIC_E1)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic(slot: str, origin: np.ndarray, bias: float) -> dict:
    local = []
    sheets = []
    selected = []
    nearest_valid = []
    for radial_band in range(2):
        radius = 1.0 + radial_band * 1.2
        for sector in range(16):
            angle = 2.0 * np.pi * (sector + 0.5) / 16.0
            u = radius * np.cos(angle)
            v = radius * np.sin(angle)
            height = bias + 0.015 * u - 0.01 * v + 0.008 * np.sin(3.0 * angle)
            local.append(
                origin
                + u * SYNTHETIC_E1
                + v * SYNTHETIC_E2
                + height * SYNTHETIC_NORMAL
            )
            valid = not (radial_band == 1 and sector == 15)
            nearest_valid.append(valid)
            sheets.append((-1 if radial_band == 0 else 1) if valid else None)
            selected.append(valid)
    return {
        "slot": slot,
        "valid": True,
        "invalid_reason": None,
        "local_support_xyz_mm": np.asarray(local).tolist(),
        "local_support_nearest_cad_valid_mask": nearest_valid,
        "local_support_selected_mask": selected,
        "local_sheet_ids": sheets,
        "height_clusters_mm": [
            [-0.52 + bias, -0.50 + bias, -0.48 + bias],
            [0.02 + bias, 0.03 + bias, 0.04 + bias, 0.05 + bias],
            [0.91 + bias, 0.93 + bias],
        ],
        "winner_cluster_index": 1,
        "runner_cluster_index": 2,
        "footprint_occupancy": [
            [True] * 16,
            [True] * 15 + [False],
        ],
        "plane_coefficients": [bias, 0.015, -0.01],
        "plane_p90_mm": 0.008,
        "feature_pre_nuisance": [0.22 + bias, -0.12, 0.07],
        "feature_post_nuisance": [0.08 + bias, -0.04, 0.03],
    }


def _invalid_diagnostic(slot: str, origin: np.ndarray) -> dict:
    diagnostic = _diagnostic(slot, origin, 0.0)
    diagnostic.update(
        {
            "valid": False,
            "invalid_reason": "SYNTHETIC_NO_VALID_LOCAL_CLUSTER",
            "local_support_selected_mask": [False] * len(diagnostic["local_support_xyz_mm"]),
            "winner_cluster_index": None,
            "runner_cluster_index": None,
            "footprint_occupancy": [[False] * 16, [False] * 16],
            "plane_coefficients": None,
            "plane_p90_mm": None,
            "feature_pre_nuisance": None,
            "feature_post_nuisance": None,
        }
    )
    return diagnostic


def synthetic_payload(*, node_count: int = 2, components: bool = True) -> dict:
    # Nondegenerate, deterministic CAD surface atlas with one diagonal edge.
    cad_xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [8.0, 0.0, 0.2],
            [0.0, 4.0, 0.1],
            [4.0, 4.0, 0.3],
            [8.0, 4.0, 0.5],
            [0.0, 8.0, 0.2],
            [4.0, 8.0, 0.5],
            [8.0, 8.0, 0.8],
            [12.0, 8.0, 1.0],
            [12.0, 4.0, 0.7],
            [12.0, 0.0, 0.4],
            [16.0, 0.0, 0.6],
            [16.0, 4.0, 0.9],
        ],
        dtype=float,
    )
    tangent_e1_row = SYNTHETIC_E1
    normal_row = SYNTHETIC_NORMAL
    tangent_e2_row = SYNTHETIC_E2
    normals = np.tile(normal_row, (len(cad_xyz), 1))
    tangent_e1 = np.tile(tangent_e1_row, (len(cad_xyz), 1))
    tangent_e2 = np.tile(tangent_e2_row, (len(cad_xyz), 1))
    edges = [[i, i + 1] for i in range(len(cad_xyz) - 1)]
    if node_count < 0 or node_count > len(cad_xyz):
        raise ValueError("node_count is the finite-node prefix length and must fit the CAD atlas")

    raw_base = []
    for x in np.linspace(-1.0, 17.0, 38):
        for y in np.linspace(-1.0, 9.0, 22):
            raw_base.append([x, y, 0.035 * x + 0.015 * y])
    raw_base = np.asarray(raw_base)

    members = []
    for ordinal, slot in enumerate(("FIRST", "SECOND")):
        raw = raw_base + [0.0, 0.0, ordinal * 0.06]
        voxel = raw[::5]
        voxel_valid = np.arange(len(voxel)) % 13 != 0
        voxel_selected = np.ones(len(voxel), dtype=bool)
        voxel_selected[::11] = False
        voxel_selected &= voxel_valid
        voxel_sheets = [
            (-1 if index % 2 == 0 else 1) if valid else None
            for index, valid in enumerate(voxel_valid)
        ]
        members.append(
            {
                "slot": slot,
                "opaque_token": f"OPAQUE_{ordinal}",
                "raw_xyz_mm": raw.tolist(),
                "raw_selected_mask": (np.arange(len(raw)) % 7 != 0).tolist(),
                "raw_nearest_cad_valid_mask": (np.arange(len(raw)) % 7 != 0).tolist(),
                "voxel_xyz_mm": voxel.tolist(),
                "voxel_selected_mask": voxel_selected.tolist(),
                "voxel_nearest_cad_valid_mask": voxel_valid.tolist(),
                "voxel_selected_by_any_winning_cluster_mask": voxel_selected.tolist(),
                "voxel_region_ids": (np.arange(len(voxel)) % 3).tolist(),
                "voxel_sheet_ids": voxel_sheets,
            }
        )

    nodes = []
    for node_id in range(len(cad_xyz)):
        valid = node_id < node_count
        component_id = "C0000" if components and valid else None
        nodes.append(
            {
                "node_id": node_id,
                "cad_index": node_id,
                "valid": valid,
                "invalid_reason": None if valid else "SYNTHETIC_NODE_UNAVAILABLE",
                "amplitude_mm": 0.45 - 0.05 * node_id if valid else None,
                "split_noise_mm": 0.025 + 0.005 * node_id if valid else None,
                "threshold_mm": 0.15 if valid else None,
                "active": valid,
                "component_id": component_id,
                "member_diagnostics": [
                    _diagnostic("FIRST", cad_xyz[node_id], 0.00)
                    if valid else _invalid_diagnostic("FIRST", cad_xyz[node_id]),
                    _diagnostic("SECOND", cad_xyz[node_id], 0.04)
                    if valid else _invalid_diagnostic("SECOND", cad_xyz[node_id]),
                ],
            }
        )

    has_passed_changed = bool(components and node_count >= 3)
    component_rows = []
    if components:
        component_rows.append(
            {
                "component_id": "C0000",
                "node_ids": list(range(node_count)),
                "retained": True,
                "robustness_changed": True,
                "area_mm2": 12.5,
                "geodesic_span_mm": 5.2,
                "member_energy_mm2": [0.021, 0.092],
                "energy_ratio": 4.380952380952381,
                "attributed_slot": "SECOND",
                "pairwise_energy_fraction": 0.72,
                "variant_rows": (
                    ([{
                        "row_id": "R01",
                        "matched_component_id": "C0000",
                        "jaccard": (node_count - 1) / node_count,
                        "medoid_displacement_mm": 0.4,
                        "energy_ratio": 3.6,
                        "attributed_slot": "SECOND",
                        "passed": True,
                    }] if has_passed_changed else [])
                    + [{
                        "row_id": "R07",
                        "matched_component_id": "C0000",
                        "jaccard": 0.52,
                        "medoid_displacement_mm": 3.4,
                        "energy_ratio": 1.8,
                        "attributed_slot": None,
                        "passed": False,
                    }]
                ),
            }
        )

    active = [False] * len(cad_xyz)
    for index in range(node_count):
        active[index] = True
    robustness_rows = []
    for row_index in range(32):
        row_id = f"R{row_index:02d}"
        passed_changed = bool(has_passed_changed and row_id == "R01")
        failed_variant = bool(components and row_id == "R07")
        vanished_variant = bool(components and row_id == "R08")
        central_empty = not components
        row = {
            "row_id": row_id,
            "status": "ABSTAIN" if (failed_variant or vanished_variant or central_empty) else "ACTIONABLE",
            "passed": not (failed_variant or vanished_variant or central_empty),
            "reason": (
                "COMPONENT_OR_ATTRIBUTION_ROBUSTNESS_GATE"
                if failed_variant else "COMPONENT_CARDINALITY_MISMATCH"
                if vanished_variant else "CENTRAL_EMPTY_TERMINAL_VISUAL_ABSTAIN"
                if central_empty else None
            ),
            "matches": [],
            "row_changed": bool(passed_changed or failed_variant or vanished_variant),
            "row_change_reasons": (
                ["COMPONENT_JACCARD_CHANGED", "COMPONENT_MEDOID_CHANGED"]
                if passed_changed or failed_variant else
                ["COMPONENT_CARDINALITY_CHANGED", "COMPONENT_MATCH_INCOMPLETE", "COMPONENT_UNMATCHED"]
                if vanished_variant else []
            ),
            "member_support": [],
            "changed_component_local_diagnostics": [],
            "changed_visual_components": [],
        }
        if passed_changed or failed_variant or vanished_variant:
            row["member_support"] = [
                {
                    "slot": member["slot"],
                    "opaque_token": member["opaque_token"],
                    "voxel_xyz_mm": member["voxel_xyz_mm"],
                    "voxel_selected_mask": member["voxel_selected_mask"],
                    "voxel_nearest_cad_valid_mask": member["voxel_nearest_cad_valid_mask"],
                    "voxel_selected_by_any_winning_cluster_mask": member["voxel_selected_by_any_winning_cluster_mask"],
                    "voxel_region_ids": member["voxel_region_ids"],
                    "voxel_sheet_ids": member["voxel_sheet_ids"],
                }
                for member in members
            ]
            row["changed_component_local_diagnostics"] = [
                {
                    "node_id": node_id,
                    "first_token": members[0]["opaque_token"],
                    "second_token": members[1]["opaque_token"],
                    "member_diagnostics": [
                        _diagnostic("FIRST", cad_xyz[node_id], 0.00),
                        _diagnostic(
                            "SECOND", cad_xyz[node_id], 0.40 if failed_variant else 0.06
                        ),
                    ],
                }
                for node_id in range(node_count)
            ]
        if passed_changed:
            variant_node_ids = list(range(node_count - 1))
            row["matches"] = [
                {
                    "central_component_id": 0,
                    "variant_component_id": 0,
                    "area_weighted_jaccard": (node_count - 1) / node_count,
                    "medoid_graph_geodesic_displacement_mm": 0.4,
                    "passed": True,
                }
            ]
            row["changed_visual_components"] = [
                {
                    "component_id": "V_R01_C0000",
                    "node_ids": variant_node_ids,
                    "area_mm2": 11.375,
                    "geodesic_span_mm": 4.8,
                    "member_energy_mm2": [0.022, 0.0792],
                    "energy_ratio": 3.6,
                    "attributed_opaque_member": members[1]["opaque_token"],
                    "pairwise_energy_fraction": 0.70,
                }
            ]
        elif failed_variant:
            row["matches"] = [
                {
                    "central_component_id": 0,
                    "variant_component_id": 0,
                    "area_weighted_jaccard": 0.52,
                    "medoid_graph_geodesic_displacement_mm": 3.4,
                    "passed": False,
                }
            ]
            row["changed_visual_components"] = [
                {
                    "component_id": "V_R07_C0000",
                    "node_ids": list(range(node_count)),
                    "area_mm2": 12.5,
                    "geodesic_span_mm": 5.2,
                    "member_energy_mm2": [0.021, 0.092],
                    "energy_ratio": 4.380952380952381,
                    "attributed_opaque_member": members[1]["opaque_token"],
                    "pairwise_energy_fraction": 0.72,
                }
            ]
        robustness_rows.append(row)

    return {
        "schema": RENDERER.SCHEMA,
        "pair_id": "SYNTHETIC_OPAQUE_PAIR",
        "members": members,
        "cad": {
            "nodes_xyz_mm": cad_xyz.tolist(),
            "normals": normals.tolist(),
            "tangent_e1": tangent_e1.tolist(),
            "tangent_e2": tangent_e2.tolist(),
            "smooth_region_ids": (np.arange(len(cad_xyz)) // 5).tolist(),
            "interior_mask": [True] * len(cad_xyz),
            "common_support_mask": ([True] * (len(cad_xyz) - 2)) + [False, False],
            "excluded_mask": ([False] * (len(cad_xyz) - 2)) + [True, True],
            "active_mask": active,
            "edges": edges,
            "fold_ids": (np.arange(len(cad_xyz)) % 4).tolist(),
            "fold_halo_mask": (np.arange(len(cad_xyz)) % 5 == 0).tolist(),
            "area_mm2": [1.125] * len(cad_xyz),
        },
        "global": {
            "status": "SYNTHETIC_RENDER_TEST_ONLY",
            "evaluator_manifest_accessed": False,
            "evaluator_roles_opened": False,
            "labels_opened": False,
            "gold_opened": False,
            "expected_counts_opened": False,
            "quality_identity_opened": False,
        },
        "nodes": nodes,
        "components": component_rows,
        "robustness": {"semantic_row_count": 32, "rows": robustness_rows},
    }


def test_inventory_is_union_of_retained_changed_and_fixed_top12() -> None:
    payload = synthetic_payload(node_count=14, components=True)
    # Prove exact top-12 tie ordering without rendering fourteen native packs.
    for node_id, row in enumerate(payload["nodes"]):
        row["amplitude_mm"] = 0.2 if node_id in (12, 13) else 1.0 - 0.02 * node_id
    data = RENDERER.validate_payload(payload)
    component_rows, node_rows = RENDERER.select_visual_inventory(data)
    assert [row["component_id"] for row in component_rows] == [
        "C0000", "V_R01_C0000", "V_R07_C0000", "V_R08_VANISHED_C0000"
    ]
    assert len(node_rows) == 12
    expected = sorted(data["nodes"], key=lambda row: (-row["amplitude"], row["node_id"]))[:12]
    assert [row["node_id"] for row in node_rows] == [row["node_id"] for row in expected]


def test_renderer_is_byte_deterministic_and_covers_local_obligations(tmp_path: Path) -> None:
    payload = synthetic_payload(node_count=1, components=True)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = RENDERER.render_native_visuals(payload, first)
    second_manifest = RENDERER.render_native_visuals(payload, second)

    expected = {
        "global_paired_diagnostic.png",
        "component_C0000.png",
        "component_V_R07_C0000.png",
        "component_V_R08_VANISHED_C0000.png",
        "node_000000.png",
    }
    assert {path.name for path in first.glob("*.png")} == expected
    assert {path.name for path in second.glob("*.png")} == expected
    for filename in expected:
        assert _hash(first / filename) == _hash(second / filename)
        with Image.open(first / filename) as image:
            assert image.size == (3000, 2400)

    assert first_manifest["global_asset"]["sha256"] == second_manifest["global_asset"]["sha256"]
    assert first_manifest["experiment_id"] == "EXP_CAD_0030"
    assert first_manifest["architecture"] == "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
    assert first_manifest["component_assets"][0]["selection_reasons"] == [
        "RETAINED_COMPONENT", "ROBUSTNESS_CHANGED_COMPONENT"
    ]
    assert first_manifest["global_fixed_same_scale_views"] == [
        "PCA1_PCA2", "PCA1_PCA3_DEPTH"
    ]
    assert first_manifest["component_assets"][0]["fixed_same_scale_views"] == [
        "PCA1_PCA2", "PCA1_PCA3_DEPTH"
    ]
    assert first_manifest["node_assets"][0]["central_amplitude_rank"] == 1
    assert first_manifest["row_specific_changed_component_asset_count"] == 2
    assert first_manifest["vanished_central_component_asset_count"] == 1
    assert any(
        row["component_id"] == "V_R08_VANISHED_C0000"
        and row["vanished_in_variant"] is True
        for row in first_manifest["component_assets"]
    )
    assert first_manifest["manual_review_status"] == "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW"


def test_zero_component_abstention_still_renders_global_native_diagnostic(tmp_path: Path) -> None:
    payload = synthetic_payload(node_count=0, components=False)
    manifest = RENDERER.render_native_visuals(payload, tmp_path)
    assert (tmp_path / "global_paired_diagnostic.png").is_file()
    assert list(tmp_path.glob("component_*.png")) == []
    assert list(tmp_path.glob("node_*.png")) == []
    assert manifest["global_diagnostic_emitted_even_zero_components"] is True
    assert manifest["fixed_top_amplitude_node_count"] == 0
    with Image.open(tmp_path / "global_paired_diagnostic.png") as image:
        assert image.size == (3000, 2400)


def test_renderer_rejects_forbidden_provenance_and_malformed_footprint() -> None:
    payload = synthetic_payload(node_count=1, components=True)
    payload["global"]["gold_opened"] = True
    with pytest.raises(RENDERER.RenderPayloadError, match="global.gold_opened"):
        RENDERER.validate_payload(payload)


def test_passed_but_changed_row_requires_and_accepts_full_variant_evidence() -> None:
    payload = synthetic_payload(node_count=3, components=True)
    normalized = RENDERER.validate_payload(payload)
    changed = [
        row for row in normalized["changed_components"]
        if row["component_id"] == "V_R01_C0000"
    ]
    assert len(changed) == 1
    assert changed[0]["source_row_id"] == "R01"
    assert changed[0]["variants"][0]["passed"] is True

    payload = synthetic_payload(node_count=3, components=True)
    row = payload["robustness"]["rows"][1]
    row["member_support"] = []
    with pytest.raises(RENDERER.RenderPayloadError, match="requires full local evidence"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=3, components=True)
    row = payload["robustness"]["rows"][1]
    row["row_changed"] = False
    row["row_change_reasons"] = []
    with pytest.raises(RENDERER.RenderPayloadError, match="passed unchanged row carries"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=3, components=True)
    row = payload["robustness"]["rows"][1]
    row["row_change_reasons"] = []
    with pytest.raises(RENDERER.RenderPayloadError, match="must equal bool"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=1, components=True)
    payload["nodes"][0]["member_diagnostics"][0]["footprint_occupancy"] = [[True] * 16]
    with pytest.raises(RENDERER.RenderPayloadError, match="footprint must be 2x16"):
        RENDERER.validate_payload(payload)


def test_renderer_rejects_sheet_zero_and_null_validity_mismatch() -> None:
    payload = synthetic_payload(node_count=1, components=True)
    valid_index = payload["members"][0]["voxel_nearest_cad_valid_mask"].index(True)
    payload["members"][0]["voxel_sheet_ids"][valid_index] = 0
    with pytest.raises(RENDERER.RenderPayloadError, match="must be -1, \\+1, or null"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=1, components=True)
    invalid_index = payload["members"][0]["voxel_nearest_cad_valid_mask"].index(False)
    payload["members"][0]["voxel_sheet_ids"][invalid_index] = 1
    with pytest.raises(RENDERER.RenderPayloadError, match="null iff nearest-CAD-valid"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=1, components=True)
    diagnostic = payload["nodes"][0]["member_diagnostics"][0]
    invalid_index = diagnostic["local_support_nearest_cad_valid_mask"].index(False)
    diagnostic["local_support_selected_mask"][invalid_index] = True
    with pytest.raises(RENDERER.RenderPayloadError, match="selected local support"):
        RENDERER.validate_payload(payload)


def test_renderer_requires_exact_all_cad_node_identity_and_valid_field_coupling() -> None:
    payload = synthetic_payload(node_count=1, components=True)
    payload["nodes"].pop()
    with pytest.raises(RENDERER.RenderPayloadError, match="cover every CAD node exactly once"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=1, components=True)
    payload["nodes"][0]["cad_index"] = 1
    with pytest.raises(RENDERER.RenderPayloadError, match="cad_index == node_id"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=1, components=True)
    payload["nodes"][0]["amplitude_mm"] = None
    with pytest.raises(RENDERER.RenderPayloadError, match="requires finite amplitude"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=1, components=True)
    payload["nodes"][0]["member_diagnostics"][0]["plane_coefficients"] = None
    with pytest.raises(RENDERER.RenderPayloadError, match="valid diagnostic requires finite plane"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=0, components=False)
    payload["nodes"][0]["active"] = True
    with pytest.raises(RENDERER.RenderPayloadError, match="active row must be valid"):
        RENDERER.validate_payload(payload)


def test_invalid_rows_preserve_partial_evidence_and_valid_local_fits() -> None:
    payload = synthetic_payload(node_count=0, components=False)
    node = payload["nodes"][0]
    node["amplitude_mm"] = 0.11
    node["member_diagnostics"] = [
        _diagnostic("FIRST", np.asarray(payload["cad"]["nodes_xyz_mm"][0]), 0.0),
        _diagnostic("SECOND", np.asarray(payload["cad"]["nodes_xyz_mm"][0]), 0.04),
    ]
    data = RENDERER.validate_payload(payload)
    assert data["nodes"][0]["valid"] is False
    assert data["nodes"][0]["diagnostics"][0]["valid"] is True
    assert data["nodes"][0]["amplitude"] == pytest.approx(0.11)

    payload = synthetic_payload(node_count=0, components=False)
    diagnostic = payload["nodes"][0]["member_diagnostics"][0]
    diagnostic["feature_pre_nuisance"] = [0.12, None, None]
    data = RENDERER.validate_payload(payload)
    assert data["nodes"][0]["diagnostics"][0]["feature_pre"][0] == pytest.approx(0.12)
    assert np.isnan(data["nodes"][0]["diagnostics"][0]["feature_pre"][1])


def test_renderer_uses_exact_non_axis_aligned_producer_tangent_basis() -> None:
    payload = synthetic_payload(node_count=1, components=True)
    origin = np.asarray(payload["cad"]["nodes_xyz_mm"][0], dtype=float)
    e1 = np.asarray(payload["cad"]["tangent_e1"][0], dtype=float)
    e2 = np.asarray(payload["cad"]["tangent_e2"][0], dtype=float)
    normal = np.asarray(payload["cad"]["normals"][0], dtype=float)
    coefficients = np.asarray([0.07, 0.12, -0.09])
    uv = np.asarray([[-1.2, -0.5], [-0.4, 0.8], [0.0, 0.0], [0.7, -0.3], [1.1, 0.9]])
    height = coefficients[0] + coefficients[1] * uv[:, 0] + coefficients[2] * uv[:, 1]
    xyz = origin + uv[:, :1] * e1 + uv[:, 1:] * e2 + height[:, None] * normal
    diagnostic = payload["nodes"][0]["member_diagnostics"][0]
    diagnostic.update(
        {
            "local_support_xyz_mm": xyz.tolist(),
            "local_support_nearest_cad_valid_mask": [True] * len(xyz),
            "local_support_selected_mask": [True] * len(xyz),
            "local_sheet_ids": [-1, -1, 1, 1, 1],
            "plane_coefficients": coefficients.tolist(),
            "plane_p90_mm": 0.0,
        }
    )
    data = RENDERER.validate_payload(payload)
    tangent1, tangent2, rendered_normal = RENDERER._node_axes(data["nodes"][0], data["cad"])
    assert np.allclose(tangent1, e1)
    assert np.allclose(tangent2, e2)
    assert np.allclose(rendered_normal, normal)
    local = RENDERER._project(xyz, origin, np.stack([tangent1, tangent2, rendered_normal]))
    fitted = coefficients[0] + coefficients[1] * local[:, 0] + coefficients[2] * local[:, 1]
    assert np.allclose(local[:, 2], fitted, atol=1e-12)

    payload = synthetic_payload(node_count=1, components=True)
    payload["cad"]["tangent_e2"][0] = (-np.asarray(payload["cad"]["tangent_e2"][0])).tolist()
    with pytest.raises(RENDERER.RenderPayloadError, match="right-handed"):
        RENDERER.validate_payload(payload)


def test_exp29_legacy_schema_is_explicitly_rejected() -> None:
    payload = synthetic_payload(node_count=0, components=False)
    payload["schema"] = "EXP_CAD_0029_CFPDG_VISUAL_PAYLOAD_V1"
    with pytest.raises(RENDERER.RenderPayloadError, match="EXP_CAD_0030"):
        RENDERER.validate_payload(payload)

    payload = synthetic_payload(node_count=0, components=False)
    payload["members"][0]["slot"] = "A"
    payload["members"][1]["slot"] = "B"
    with pytest.raises(RENDERER.RenderPayloadError, match="slot must be FIRST"):
        RENDERER.validate_payload(payload)


def test_live_source_producer_payload_is_canonical_and_renders_without_source_files(
    tmp_path: Path,
) -> None:
    """Exercise the real numeric producer, not the hand-built render fixture."""

    import open3d as o3d

    runner_path = ROOT / "scripts" / "run_exp_cad_0030_cfpdg_source.py"
    if not runner_path.exists():
        pytest.skip("EXP30 source producer is still awaiting its independent pre-source freeze")
    runner_spec = importlib.util.spec_from_file_location("exp30_source_runner", runner_path)
    runner = importlib.util.module_from_spec(runner_spec)
    assert runner_spec and runner_spec.loader
    runner_spec.loader.exec_module(runner)

    dtype = np.dtype(
        [
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
            ("e1x", "<f8"), ("e1y", "<f8"), ("e1z", "<f8"),
            ("e2x", "<f8"), ("e2y", "<f8"), ("e2z", "<f8"),
            ("parent_face", "<i8"), ("candidate_ordinal", "<i8"),
            ("smooth_region", "<i8"), ("area_mm2", "<f8"),
            ("fold_id", "u1"), ("graph_component", "<i8"),
        ]
    )
    grid = np.asarray([(x, y, 0.0) for y in (-4.0, -1.25, 1.25, 4.0) for x in (-4.0, -1.25, 1.25, 4.0)])
    nodes = np.zeros(len(grid), dtype=dtype)
    nodes["x"], nodes["y"], nodes["z"] = grid.T
    nodes["nz"] = 1.0
    nodes["e1x"] = 1.0
    nodes["e2y"] = 1.0
    nodes["parent_face"] = 0
    nodes["candidate_ordinal"] = np.arange(len(nodes))
    nodes["smooth_region"] = 0
    nodes["area_mm2"] = 1.125
    nodes["fold_id"] = np.arange(len(nodes)) % 4
    nodes["graph_component"] = 0
    edges = np.column_stack((np.arange(len(nodes) - 1), np.arange(1, len(nodes)))).astype(np.int64)
    node_xyz = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    weights = np.linalg.norm(node_xyz[edges[:, 0]] - node_xyz[edges[:, 1]], axis=1)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.asarray([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]])
        ),
        o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
    )
    xy = np.asarray([(x, y) for y in np.linspace(-7.0, 7.0, 57) for x in np.linspace(-7.0, 7.0, 57)])
    raw_first = np.column_stack((xy, np.full(len(xy), 0.02)))
    raw_second = np.column_stack((xy, np.full(len(xy), 0.035)))
    scan_normals = np.tile([0.0, 0.0, 1.0], (len(xy), 1))
    prereg = json.loads(
        (ROOT / "docs" / "research" / "2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json").read_text(encoding="utf-8")
    )
    execute_source = getattr(runner, "execute_source", runner.BASE.execute_source)
    semantic, producer_payload = execute_source(
        prereg=prereg,
        cad_preflight={"cad_metrics": {"cad_interior_area_weighted_centroid_mm": [0.0, 0.0, 0.0]}},
        nodes=nodes,
        edges=edges,
        edge_weights=weights,
        mesh=mesh,
        raw_first=raw_first,
        normals_first=scan_normals,
        raw_second=raw_second,
        normals_second=scan_normals,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="OPAQUE_A",
        second_token="OPAQUE_B",
    )
    assert semantic["evaluator_roles_labels_gold_expected_counts_quality_used"] is False
    assert [row["slot"] for row in producer_payload["members"]] == ["FIRST", "SECOND"]
    perturbation = next(
        row for row in producer_payload["robustness"]["rows"] if row["row_id"] == "R07"
    )
    assert perturbation["passed"] is False
    assert perturbation["row_descriptor"] == "FIRST_TRANSLATION_X_NEG"
    manifest = RENDERER.render_native_visuals(producer_payload, tmp_path)
    assert manifest["global_diagnostic_emitted_even_zero_components"] is True
    assert manifest["provenance"]["external_source_opened_by_renderer"] is False
    assert manifest["provenance"]["canonical_exp30_schema_only"] is True
    with Image.open(tmp_path / "global_paired_diagnostic.png") as image:
        assert image.size == (3000, 2400)
