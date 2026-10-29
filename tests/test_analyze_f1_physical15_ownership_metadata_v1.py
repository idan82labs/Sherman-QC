from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_f1_physical15_ownership_metadata_v1.py"
SPEC = importlib.util.spec_from_file_location("ownership_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_extracts_surface_plane_flange_and_seed_ownership():
    assert MODULE.ownership_keys({"surface1_id": "S2", "surface2_id": "S1"}) == {"SURFACE:S1|S2"}
    assert MODULE.ownership_keys({"incident_plane_ids": ["P9", "P2"]}) == {"PLANE:P2|P9"}
    assert MODULE.ownership_keys({"seed_ids": ["BIRTH_F7_F75_9"]}) == {"FLANGE:F7|F75"}
    assert MODULE.ownership_keys({"canonical_flange_pair": ["B", "A"]}) == {"FLANGE:A|B"}


def test_invalid_or_singleton_pairs_do_not_create_ownership():
    assert MODULE.ownership_keys({"surface1_id": "S1"}) == set()
    assert MODULE.ownership_keys({"incident_plane_ids": ["P1"]}) == set()
    assert MODULE.ownership_keys({"seed_ids": ["NO_PAIR"]}) == set()

