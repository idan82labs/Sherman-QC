"""E1 compile CLI (ENGINE_IO §3): stdout protocol, exit codes, output files."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from contracts.production.canonical import canonical_json_bytes
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.dev_recipe import dev_recipe

REPO = Path(__file__).resolve().parents[1]


def run_cli(*args: str | Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{REPO / 'backend'}", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "domains.production.engine.compile", *map(str, args)],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=300)


def records(stdout: str) -> list[tuple[str, dict]]:
    out = []
    for line in stdout.splitlines():
        kind, _, payload = line.partition("\t")
        assert kind in ("PROGRESS", "RESULT", "ERROR"), f"non-protocol stdout line: {line!r}"
        out.append((kind, json.loads(payload)))
    return out


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli")
    shape, info = tp.l_part(width=30.0)
    step = tp.write_step(shape, tmp / "l.step")
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-CLI")
    (tmp / "recipe.json").write_bytes(canonical_json_bytes(recipe))
    blocked = dev_recipe(step, tp.candidates_from_info(info) + [
        {"line_start": [10.0, 0.0, 1.0], "line_end": [10.0, 30.0, 1.0], "target_angle": 90.0, "target_radius": 2.0}],
        sku="SYN-CLI")
    (tmp / "blocked.json").write_bytes(canonical_json_bytes(blocked))
    other_shape, _ = tp.z_part(width=30.0)
    other = tp.write_step(other_shape, tmp / "other.step")
    return tmp, step, other, recipe


def test_compiled_run_prints_only_protocol_records_and_exits_0(inputs):
    tmp, step, _, recipe = inputs
    out = tmp / "ok"
    r = run_cli("--recipe", tmp / "recipe.json", "--cad", step, "--out", out)
    assert r.returncode == 0, r.stderr[-2000:]
    recs = records(r.stdout)
    assert recs[-1][0] == "RESULT" and all(k == "PROGRESS" for k, _ in recs[:-1])
    assert all(0 <= p["progress"] <= 100 and p["stage"] for _, p in recs[:-1])
    result = recs[-1][1]
    assert result["status"] == "COMPILED" and result["blocker_codes"] == [] and result["out"] == str(out)
    compiled = CompiledPartV1.model_validate_json((out / "compiled_part.json").read_bytes())
    assert compiled.compiled_part_sha256 == result["compiled_part_sha256"]
    assert compiled.recipe.recipe_sha256 == recipe.recipe_sha256
    assert (out / "compile.log").is_file() and sorted(p.name for p in (out / "stages").iterdir()) == [
        "S0.npz", "S0.ply", "S1.npz", "S1.ply"]
    assert not (out / "compile_error.json").exists()


def test_blocked_compile_exits_2_and_still_writes_the_part(inputs):
    tmp, step, _, _ = inputs
    out = tmp / "blocked"
    r = run_cli("--recipe", tmp / "blocked.json", "--cad", step, "--out", out)
    assert r.returncode == 2
    kind, result = records(r.stdout)[-1]
    assert kind == "RESULT" and result["status"] == "BLOCKED"
    assert result["blocker_codes"] == ["MARK_NOT_IN_FLANGE_GRAPH"]
    compiled = CompiledPartV1.model_validate_json((out / "compiled_part.json").read_bytes())
    assert compiled.status == "BLOCKED" and compiled.blocker_codes == ("MARK_NOT_IN_FLANGE_GRAPH",)


@pytest.mark.parametrize("case, code", [("cad_hash", "CAD_HASH_MISMATCH"), ("recipe", "RECIPE_INVALID"),
                                        ("flat_unexpected", "FLAT_UNEXPECTED"), ("cad_missing", "CAD_MISSING"),
                                        ("shop_k", "SHOP_K_INVALID"), ("config", "CONFIG_INVALID")])
def test_invalid_input_exits_2_with_compile_error_json(inputs, case, code):
    tmp, step, other, _ = inputs
    out = tmp / f"err_{case}"
    args = ["--recipe", tmp / "recipe.json", "--cad", step, "--out", out]
    if case == "cad_hash":
        args[3] = other
    elif case == "recipe":
        broken = tmp / "broken.json"
        data = json.loads((tmp / "recipe.json").read_text())
        data["bends"][0]["label"] = "Z9"  # content no longer matches its hashes
        broken.write_text(json.dumps(data))
        args[1] = broken
    elif case == "flat_unexpected":
        flat = tmp / "f.dxf"
        flat.write_text("0\nEOF\n")
        args += ["--flat", flat]
    elif case == "cad_missing":
        args[3] = tmp / "nope.step"
    elif case == "shop_k":
        args += ["--shop-k", "1.5"]
    elif case == "config":
        cfg = tmp / "cfg.json"
        cfg.write_text(json.dumps({"no_such_key": 1}))
        args += ["--config", cfg]
    r = run_cli(*args)
    assert r.returncode == 2, r.stderr[-1500:]
    kind, payload = records(r.stdout)[-1]
    assert kind == "ERROR" and payload["code"] == code
    err = json.loads((out / "compile_error.json").read_text())
    assert err["code"] == code and err["message"]
    assert not (out / "compiled_part.json").exists()


def test_recipe_with_a_flat_needs_the_flat(tmp_path, inputs):
    _, step, _, _ = inputs
    shape, info = tp.l_part(width=30.0)
    outer, lines, _ = tp.profile_flat(info, 0.33)
    dxf = tp.write_flat_dxf(tmp_path / "f.dxf", outer, lines)
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-CLIF", flat_path=dxf)
    (tmp_path / "r.json").write_bytes(canonical_json_bytes(recipe))
    r = run_cli("--recipe", tmp_path / "r.json", "--cad", step, "--out", tmp_path / "o")
    assert r.returncode == 2 and records(r.stdout)[-1][1]["code"] == "FLAT_MISSING"
    other = tmp_path / "other.dxf"
    other.write_bytes(dxf.read_bytes() + b"\n")
    r = run_cli("--recipe", tmp_path / "r.json", "--cad", step, "--flat", other, "--out", tmp_path / "o2")
    assert r.returncode == 2 and records(r.stdout)[-1][1]["code"] == "FLAT_HASH_MISMATCH"
    r = run_cli("--recipe", tmp_path / "r.json", "--cad", step, "--flat", dxf, "--out", tmp_path / "o3")
    assert r.returncode == 0, r.stderr[-1500:]


def test_non_empty_out_dir_is_refused_and_left_untouched(inputs):
    tmp, step, _, _ = inputs
    out = tmp / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("user data")
    r = run_cli("--recipe", tmp / "recipe.json", "--cad", step, "--out", out)
    assert r.returncode == 2
    kind, payload = records(r.stdout)[-1]
    assert kind == "ERROR" and payload["code"] == "OUT_DIR_NOT_EMPTY"
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_an_out_path_that_is_a_file_is_refused_by_the_library_too(inputs):
    # the library is called directly by C05/E4; a file must be an input error, not a crash
    from domains.production.engine.compile import CompileInputError, compile_part

    tmp, step, _, recipe = inputs
    target = tmp / "a_file"
    target.write_text("user data")
    with pytest.raises(CompileInputError) as err:
        compile_part(recipe, step, target)
    assert err.value.code == "OUT_DIR_NOT_EMPTY"
    assert target.read_text() == "user data"
    r = run_cli("--recipe", tmp / "recipe.json", "--cad", step, "--out", target)
    assert r.returncode == 2 and records(r.stdout)[-1][1]["code"] == "OUT_DIR_NOT_EMPTY"
