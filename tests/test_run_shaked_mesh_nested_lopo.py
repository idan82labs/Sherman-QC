from scripts.run_shaked_mesh_nested_lopo import outer_training_case_ids


def test_outer_training_cases_exclude_held_out_case():
    case_ids = ["01_a", "02_b", "03_c"]

    assert outer_training_case_ids(case_ids, "02_b") == ["01_a", "03_c"]
