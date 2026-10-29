#!/usr/bin/env python3
"""Direct, hand-solvable contracts for the isolated structured CP-SAT worker."""

from __future__ import annotations

import copy
import json
import unittest

from solve_f1_structured_union_cpsat import (
    BEND,
    CORNER,
    CURVE,
    HOLE,
    JUNCTION,
    LABELS,
    PERIMETER,
    RESIDUAL,
    solve,
)


def _contract() -> dict:
    return {
        "uses_part_identifier": False,
        "uses_filename_features": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
        "uses_cad": False,
        "source_point_serialization": False,
    }


def _atom(atom_id: str, *, area: float = 1.0) -> dict:
    return {
        "atom_id": atom_id,
        "centroid": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "local_spacing_mm": 1.0,
        "area_proxy_mm2": area,
    }


def _path(
    path_id: str,
    atom_ids: list[str],
    *,
    eligible: tuple[str, ...] = (BEND,),
    preferred: str | None = BEND,
    geometry_kind: str = "LINE",
    panel_pair: tuple[str, str] = ("PLANE:P1", "PLANE:P2"),
) -> dict:
    bin_map = {atom_id: f"B{index}" for index, atom_id in enumerate(atom_ids)}
    side_map = {
        atom_id: panel_pair[index % 2]
        for index, atom_id in enumerate(atom_ids)
    }
    energies = {label: 0 for label in LABELS}
    if preferred is not None:
        energies[preferred] = -10_000_000
    return {
        "latent_path_id": path_id,
        "geometry_kind": geometry_kind,
        "span_endpoints": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "axis_direction": [1.0, 0.0, 0.0],
        "midpoint": [5.0, 0.0, 0.0],
        "span_mm": 10.0,
        "primary_panel_pair": list(panel_pair),
        "support_atom_ids": list(atom_ids),
        "support_bin_ids": sorted(set(bin_map.values())),
        "support_atom_bins": bin_map,
        "support_atom_sides": side_map,
        "support_atom_ids_truncated": False,
        "incidence_complete": True,
        "incidence_certificate": {
            "tolerance_profile": "INCIDENCE_V1",
            "atom_table_sha256": "fixture-atoms",
            "max_dimensionless_residuals": {
                "perpendicular": 0.1,
                "endpoint": 0.1,
                "normal": 0.1,
                "axis": 0.1,
            },
        },
        "straight_line_eligible": geometry_kind == "LINE",
        "label_eligibility": {
            label: label in set(eligible) | {RESIDUAL}
            for label in LABELS
        },
        "label_energy_int": energies,
    }


def _graph(paths: list[dict], atoms: list[dict], **extra: object) -> dict:
    return {
        "graph_sha256": "fixture",
        "inference_contract": _contract(),
        "semantic_labels": list(LABELS),
        "latent_paths": paths,
        "support_atom_table": atoms,
        "support_atom_edges": [],
        "topology_factors": [],
        "continuations": [],
        "duplicate_pairs": [],
        "panel_boundary_arcs": [],
        "energy_parameters": {
            "object_cost_int": 0,
            "residual_cost_per_mass_int": 0,
        },
        **extra,
    }


def _continuation(left: str, right: str, *, continuation_id: str = "CONT_AB") -> dict:
    return {
        "continuation_id": continuation_id,
        "path_ids": [left, right],
        "endpoint_ids": [1, 0],
        "dimensionless_residuals": {
            "axis": 0.1,
            "perpendicular": 0.1,
            "gap": 0.1,
            "chord": 0.1,
            "refit": 0.1,
            "overlap": 0.1,
        },
        "engineering_tolerance_profile": "CONTINUATION_V1",
        "bridge_atom_ids": ["A0"],
        "bridge_certificate_sha256": "bridge-fixture",
    }


def _labels(result: dict) -> dict[str, str]:
    return {
        item["latent_path_id"]: item["semantic_label"]
        for item in result["assignments"]
    }


class StructuredUnionCpSatProfessorContractTest(unittest.TestCase):
    def test_all_seven_labels_are_operational_with_class_conditioned_geometry(self) -> None:
        specifications = [
            ("P_BEND", BEND, "LINE"),
            ("P_CORNER", CORNER, "LINE"),
            ("P_PERIMETER", PERIMETER, "LINE"),
            ("P_HOLE", HOLE, "LOOP"),
            ("P_JUNCTION", JUNCTION, "NODE"),
            ("P_CURVE", CURVE, "POLYLINE"),
        ]
        atoms = []
        paths = []
        for index, (path_id, label, geometry_kind) in enumerate(specifications):
            atom_ids = [f"A{index}_0", f"A{index}_1"]
            atoms.extend(_atom(atom_id) for atom_id in atom_ids)
            paths.append(
                _path(
                    path_id,
                    atom_ids,
                    eligible=(label,),
                    preferred=label,
                    geometry_kind=geometry_kind,
                )
            )

        result = solve(_graph(paths, atoms))

        self.assertEqual(
            _labels(result),
            {path_id: label for path_id, label, _geometry_kind in specifications},
        )
        self.assertTrue(result["model_scope"]["all_seven_semantic_labels_operational"])
        self.assertEqual(result["solver"]["status"], "OPTIMAL")

    def test_zero_or_uncertified_incidence_hard_prohibits_nonresidual(self) -> None:
        atoms = [_atom("A0"), _atom("A1")]
        incomplete = _path("P_INCOMPLETE", ["A0", "A1"])
        incomplete["incidence_complete"] = False
        uncertified = _path("P_UNCERTIFIED", ["A0", "A1"])
        uncertified["incidence_certificate"] = {}

        result = solve(_graph([incomplete, uncertified], atoms))

        self.assertEqual(set(_labels(result).values()), {RESIDUAL})
        self.assertEqual(result["counted_fragment_count"], 0)

    def test_truncated_incidence_rejects_exact_solve(self) -> None:
        path = _path("P", ["A0", "A1"])
        path["support_atom_ids_truncated"] = True
        with self.assertRaisesRegex(ValueError, "truncated support incidence"):
            solve(_graph([path], [_atom("A0"), _atom("A1")]))

    def test_class_specific_axial_minimum_cannot_be_bypassed_by_atom_count(self) -> None:
        path = _path("P", ["A0", "A1"])
        path["support_atom_bins"] = {"A0": "B0", "A1": "B0"}
        path["support_bin_ids"] = ["B0"]

        result = solve(_graph([path], [_atom("A0"), _atom("A1")]))

        self.assertEqual(_labels(result)["P"], RESIDUAL)
        self.assertEqual(result["owned_axial_bin_count_by_path"]["P"], 0)

    def test_atom_mass_ownership_is_explicit_and_weighted(self) -> None:
        atoms = [_atom("A0", area=1.0), _atom("A1", area=4.0), _atom("A2", area=1.0)]
        path = _path("P", ["A0", "A1"])

        result = solve(_graph([path], atoms))
        owners = {item["atom_id"]: item for item in result["atom_owners"]}

        self.assertEqual(owners["A0"]["owner_kind"], "PATH")
        self.assertEqual(owners["A0"]["latent_path_id"], "P")
        self.assertEqual(owners["A1"]["latent_path_id"], "P")
        self.assertEqual(owners["A2"]["owner_kind"], "RESIDUAL")
        self.assertIsNone(owners["A2"]["latent_path_id"])
        self.assertEqual(owners["A0"]["normalized_support_mass"], 1)
        self.assertEqual(owners["A1"]["normalized_support_mass"], 4)
        self.assertEqual(result["owned_support_mass_by_path"]["P"], 5)

    def test_hard_duplicate_pair_blocks_two_counted_objects(self) -> None:
        atoms = [_atom(f"A{i}") for i in range(4)]
        paths = [_path("P_A", ["A0", "A1"]), _path("P_B", ["A2", "A3"])]
        graph = _graph(
            paths,
            atoms,
            duplicate_pairs=[{"path_ids": ["P_A", "P_B"], "hard": True}],
        )

        result = solve(graph)

        self.assertEqual(_labels(result)["P_A"], BEND)
        self.assertEqual(_labels(result)["P_B"], RESIDUAL)
        self.assertEqual(result["counted_fragment_count"], 1)

    def test_disjoint_factors_can_activate_and_contradictory_factors_conflict(self) -> None:
        atoms = [_atom(f"A{i}") for i in range(6)]
        paths = [
            _path("P_A", ["A0", "A1"], eligible=(BEND,), preferred=None),
            _path("P_B", ["A2", "A3"], eligible=(CORNER,), preferred=None),
            _path("P_C", ["A4", "A5"], eligible=(BEND, CORNER), preferred=None),
        ]
        factors = [
            {
                "factor_id": "F_A",
                "energy_int": -2_000_000,
                "weight_status": "ENABLED_ENGINEERING_FIXED",
                "label_implications": [{"path_id": "P_A", "label": BEND}],
            },
            {
                "factor_id": "F_B",
                "energy_int": -2_000_000,
                "weight_status": "ENABLED_ENGINEERING_FIXED",
                "label_implications": [{"path_id": "P_B", "label": CORNER}],
            },
            {
                "factor_id": "F_C_BEND",
                "energy_int": -1_000_000,
                "weight_status": "ENABLED_ENGINEERING_FIXED",
                "label_implications": [{"path_id": "P_C", "label": BEND}],
            },
            {
                "factor_id": "F_C_CORNER",
                "energy_int": -1_000_000,
                "weight_status": "ENABLED_ENGINEERING_FIXED",
                "label_implications": [{"path_id": "P_C", "label": CORNER}],
            },
        ]

        result = solve(_graph(paths, atoms, topology_factors=factors))

        self.assertIn("F_A", result["selected_factor_ids"])
        self.assertIn("F_B", result["selected_factor_ids"])
        self.assertEqual(
            len(set(result["selected_factor_ids"]) & {"F_C_BEND", "F_C_CORNER"}),
            1,
        )
        self.assertFalse(result["model_scope"]["global_topology_factor_at_most_one"])

    def test_empty_disabled_factor_is_audit_only_but_empty_enabled_factor_fails(self) -> None:
        atoms = [_atom("A0"), _atom("A1")]
        path = _path("P", ["A0", "A1"])
        disabled = {
            "factor_id": "F_AUDIT_ONLY",
            "factor_kind": "BICLIQUE_CHORD",
            "weight_status": "DISABLED_PENDING_FAMILY_EVIDENCE",
            "label_implications": [],
            "feature_vector": {"topology_coverage": 0.5},
        }

        result = solve(_graph([path], atoms, topology_factors=[disabled]))

        self.assertEqual(result["selected_factor_ids"], [])
        self.assertEqual(result["disabled_audit_factor_ids"], ["F_AUDIT_ONLY"])
        self.assertEqual(result["input_topology_factor_count"], 1)
        self.assertEqual(result["modeled_topology_factor_count"], 0)
        self.assertTrue(
            result["model_scope"]["disabled_factors_retained_as_audit_only"]
        )

        enabled = dict(disabled)
        enabled["factor_id"] = "F_ENABLED_EMPTY"
        enabled["weight_status"] = "ENABLED_ENGINEERING_FIXED"
        with self.assertRaisesRegex(ValueError, "at least one typed path label"):
            solve(_graph([path], atoms, topology_factors=[enabled]))

    def test_shared_boundary_arc_has_exactly_one_owner(self) -> None:
        atoms = [_atom(f"A{i}") for i in range(4)]
        paths = [_path("P_A", ["A0", "A1"]), _path("P_B", ["A2", "A3"])]
        arcs = [
            {
                "arc_id": "ARC_1",
                "compatible_path_ids": ["P_A", "P_B"],
                "residual_owner_allowed": True,
            }
        ]

        result = solve(_graph(paths, atoms, panel_boundary_arcs=arcs))

        self.assertEqual(result["panel_boundary_arc_owners"], [{"arc_id": "ARC_1", "owner": "P_A"}])
        self.assertTrue(result["model_scope"]["panel_boundary_arc_ownership"])

    def test_certified_continuation_is_endpoint_aware_and_reduces_component_count(self) -> None:
        atoms = [_atom(f"A{i}") for i in range(4)]
        paths = [_path("P_A", ["A0", "A1"]), _path("P_B", ["A2", "A3"])]

        result = solve(_graph(paths, atoms, continuations=[_continuation("P_A", "P_B")]))

        self.assertEqual(result["counted_fragment_count"], 2)
        self.assertEqual(result["counted_component_count"], 1)
        self.assertEqual(result["counted_components"], [["P_A", "P_B"]])
        self.assertEqual(result["endpoint_types_by_path"]["P_A"][1], "CONTINUATION")
        self.assertEqual(result["endpoint_types_by_path"]["P_B"][0], "CONTINUATION")

    def test_invalid_or_cyclic_continuation_certificate_is_rejected(self) -> None:
        atoms = [_atom(f"A{i}") for i in range(6)]
        paths = [
            _path("P_A", ["A0", "A1"]),
            _path("P_B", ["A2", "A3"]),
            _path("P_C", ["A4", "A5"]),
        ]
        invalid = _continuation("P_A", "P_B")
        invalid["dimensionless_residuals"].pop("bridge", None)
        invalid["dimensionless_residuals"].pop("refit")
        with self.assertRaisesRegex(ValueError, "incomplete continuation residual"):
            solve(_graph(paths, atoms, continuations=[invalid]))

        overlapping = _continuation("P_A", "P_B")
        overlapping["dimensionless_residuals"]["overlap"] = 1.01
        with self.assertRaisesRegex(ValueError, "outside frozen engineering tolerance"):
            solve(_graph(paths, atoms, continuations=[overlapping]))

        cycle = [
            _continuation("P_A", "P_B", continuation_id="AB"),
            _continuation("P_B", "P_C", continuation_id="BC"),
            _continuation("P_C", "P_A", continuation_id="CA"),
        ]
        with self.assertRaisesRegex(ValueError, "acyclic"):
            solve(_graph(paths, atoms, continuations=cycle))

    def test_birth_activation_gates_labels_ownership_and_continuation(self) -> None:
        atoms = [_atom(f"A{i}") for i in range(4)]
        existing = _path("P_A", ["A0", "A1"])
        birth = _path("P_BIRTH", ["A2", "A3"])
        birth.update({"is_birth": True, "birth_id": "BIRTH_1", "birth_cost_int": 1})

        result = solve(
            _graph(
                [existing, birth],
                atoms,
                continuations=[_continuation("P_A", "P_BIRTH")],
            )
        )

        self.assertEqual(result["selected_birth_ids"], ["BIRTH_1"])
        self.assertEqual(result["counted_component_count"], 1)
        self.assertEqual(
            {
                item["latent_path_id"]
                for item in result["atom_owners"]
                if item["atom_id"] in {"A2", "A3"}
            },
            {"P_BIRTH"},
        )

    def test_second_best_assignment_and_lex_order_are_exact_and_order_invariant(self) -> None:
        atoms = [_atom("A0"), _atom("A1")]
        path = _path(
            "P",
            ["A0", "A1"],
            eligible=(BEND, PERIMETER),
            preferred=None,
        )
        graph = _graph([path], atoms)

        first = solve(graph)
        permuted = copy.deepcopy(graph)
        permuted["support_atom_table"].reverse()
        permuted["latent_paths"].reverse()
        second = solve(permuted)

        self.assertEqual(_labels(first)["P"], BEND)
        self.assertEqual(
            _labels({"assignments": first["second_best_assignment"]["assignments"]})["P"],
            PERIMETER,
        )
        self.assertEqual(first["solver"]["second_best_energy_margin_int"], 0)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )

    def test_replay_is_byte_identical(self) -> None:
        graph = _graph([_path("P", ["A0", "A1"])], [_atom("A0"), _atom("A1")])
        payloads = [
            json.dumps(solve(graph), sort_keys=True, separators=(",", ":"))
            for _ in range(3)
        ]
        self.assertEqual(len(set(payloads)), 1)


if __name__ == "__main__":
    unittest.main()
