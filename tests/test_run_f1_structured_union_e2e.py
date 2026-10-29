from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_f1_structured_union_e2e import (
    GLOBAL_POINT_BUDGET,
    _atom_owner_paths,
    _component_groups,
    _run_exact_solver,
    _ru_maxrss_bytes,
    _validate_pool_contract,
)


def _bounded_pool(**updates):
    payload = {
        "source_point_serialization": False,
        "inference_contract": {},
        "working_point_contract_version": "GLOBAL_POINT_BUDGET_V1",
        "working_point_budget": GLOBAL_POINT_BUDGET,
        "transient_sampled_point_count": 10,
        "transient_plane_point_count": 10,
    }
    payload.update(updates)
    return payload


def test_e2e_rejects_pool_above_global_point_budget():
    with pytest.raises(ValueError, match="exceeds global point budget"):
        _validate_pool_contract(
            _bounded_pool(transient_sampled_point_count=GLOBAL_POINT_BUDGET + 1)
        )


def test_e2e_rejects_truth_exposed_pool():
    with pytest.raises(ValueError, match="strict scan-only"):
        _validate_pool_contract(
            _bounded_pool(inference_contract={"expected_count_exposed": True})
        )


def test_e2e_requires_explicit_global_working_set_contract():
    payload = _bounded_pool()
    payload.pop("transient_plane_point_count")
    with pytest.raises(ValueError, match="must report transient_plane_point_count"):
        _validate_pool_contract(payload)


def test_exact_solver_owner_records_are_normalized_for_component_refit():
    assert _atom_owner_paths(
        {
            "atom_owners": [
                {
                    "atom_id": "A0",
                    "owner_kind": "PATH",
                    "latent_path_id": "P0",
                },
                {
                    "atom_id": "A1",
                    "owner_kind": "RESIDUAL",
                    "latent_path_id": None,
                },
            ]
        }
    ) == {"A0": "P0", "A1": None}


def test_ru_maxrss_units_are_correct_on_macos_and_linux():
    assert _ru_maxrss_bytes(4096, platform="darwin") == 4096
    assert _ru_maxrss_bytes(4096, platform="linux") == 4096 * 1024


def test_exact_solver_runner_preserves_virtualenv_launcher_symlink(tmp_path: Path):
    solver_python = Path(".venv-f1-structured/bin/python").absolute()
    assert solver_python.is_file()
    graph = Path("tests/fixtures/f1_structured_union/exact_seven_label_graph.json").absolute()

    result = _run_exact_solver(
        solver_python=solver_python,
        graph_path=graph,
        solution_path=tmp_path / "solution.json",
    )

    assert result["solver"]["status"] == "OPTIMAL"


def test_component_grouping_counts_maximal_continuation_paths():
    groups = _component_groups(
        {"A", "B", "C", "D"},
        [
            {"path_ids": ["A", "B"], "endpoint_ids": [1, 0]},
            {"path_ids": ["B", "C"], "endpoint_ids": [1, 0]},
        ],
    )

    assert groups == [["A", "B", "C"], ["D"]]
