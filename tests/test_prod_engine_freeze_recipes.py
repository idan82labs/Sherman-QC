"""E4b DEV_AUTO_CONFIRMED recipes (amendment 4d): E1 policy, blindness, compile outcomes, denominators.

A synthetic E2 STEP goes through E1's own DEV policy and compile command line;
no authorized STEP is read here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from domains.production.engine.eval.freeze import recipes as rc
from domains.production.engine.eval.runner import RunnerConfig, engine_env
from domains.production.engine.synthetic.cases import generate_tier

CASE = "E2-DEV-SMTD-000"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("e4b_recipes")
    generate_tier("smoke", "DEV", root / "tier", case_ids=[CASE])
    step = root / "tier" / CASE / "part.step"
    mp = pytest.MonkeyPatch()
    mp.setattr(rc, "custody_check", lambda p, purpose: Path(p).resolve())
    try:
        summary = rc.build_recipes_for_step(step, root / "work")
    finally:
        mp.undo()
    return root, step, summary


def test_dev_recipe_is_blind_confirmed_and_requires_both_sides(built):
    root, step, summary = built
    dev = summary["variants"]["dev"]
    assert dev["status"] == "RECIPE_BUILT" and dev["bends"] >= 1
    assert dev["sku"] == "DEV-" + summary["cad_sha256"][:10].upper()
    assert dev["cad_original_filename"] == f"cad_{summary['cad_sha256'][:12]}.step"  # no file name reaches E1/E3
    assert dev["confirmation"] == "DEV_AUTO_CONFIRMED" and dev["inspection_stage"] == "FINAL"
    assert dev["both_sides_required"]
    raw = Path(dev["recipe_path"]).read_text(encoding="utf-8")
    assert CASE not in raw and "SYN-" not in raw and "part.step" not in raw
    policy = summary["dev_policy_report"]["policy"]
    assert policy["version"] == "E1-DEV-POLICY-2" and len(policy["sha256"]) == 64  # 2026-10-19: + E1-graph marks
    # a variant without candidates is recorded, never dropped
    assert summary["variants"]["v10_strict"]["status"] in ("RECIPE_BUILT", "NO_RECIPE")


def test_compile_outcomes_are_recorded(built, tmp_path):
    root, step, summary = built
    env = engine_env(RunnerConfig(tier_dir=tmp_path, out_dir=tmp_path))
    staged = Path(summary["staged_cad"])
    ok = rc.compile_recipe(Path(summary["variants"]["dev"]["recipe_path"]), staged, tmp_path / "c1", env=env)
    assert ok["status"] == "COMPILED" and ok["compiled_status"] == "COMPILED"
    assert ok["recipe_consistency_problems"] == [] and len(ok["compiled_part_sha256"]) == 64
    again = rc.compile_recipe(Path(summary["variants"]["dev"]["recipe_path"]), staged, tmp_path / "c2", env=env)
    assert again["compiled_part_sha256"] == ok["compiled_part_sha256"]  # reproducible (compiled_at is not hashed)
    bad = tmp_path / "bad.recipe.json"
    bad.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    fail = rc.compile_recipe(bad, staged, tmp_path / "c3", env=env)
    assert fail["status"] == "FAILED" and fail["reason"] == "EXIT_2:RECIPE_INVALID"
    other = tmp_path / "other.step"
    other.write_bytes(staged.read_bytes() + b"\n")
    mism = rc.compile_recipe(Path(summary["variants"]["dev"]["recipe_path"]), other, tmp_path / "c4", env=env)
    assert mism["status"] == "FAILED" and mism["reason"] == "EXIT_2:CAD_HASH_MISMATCH"


def test_compile_rates_keep_failures_in_the_denominator():
    rows = [{"variants": {"dev": {"status": s}}} for s in ("COMPILED", "COMPILED", "BLOCKED", "FAILED", "NO_RECIPE")]
    rows.append({"variants": {}})
    r = rc.compile_rates(rows, "dev")
    assert r["authorized_unique_steps"] == 6 and r["compiled"] == 2 and r["rate"] == pytest.approx(2 / 6)
    assert (r["blocked"], r["failed"], r["no_recipe"]) == (1, 1, 2)
    assert rc.compile_rates([], "dev")["rate"] is None


def test_public_rows_drop_work_paths():
    row = {"sku": "DEV-X", "cad_rel": "a.stp", "variants": {"dev": {"status": "COMPILED", "recipe_path": "/w/r.json",
                                                                   "compiled_dir": "/w/c"}}}
    pub = rc.public_row(row)
    assert "recipe_path" not in pub["variants"]["dev"] and "compiled_dir" not in pub["variants"]["dev"]
    assert pub["cad_rel"] == "../source-assets/02_CAD_QC_3D/a.stp"
