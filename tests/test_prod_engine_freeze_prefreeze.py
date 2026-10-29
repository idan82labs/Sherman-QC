"""E4b freeze command line: the DEV-CHECK seal at the commands A1 runs, and the amendment-7 gate.

A temporary freeze directory stands in for the real one; E1 compiles, E3 runs and the G3 renderer
are stubbed (they are tested elsewhere), so these tests exercise only what the commands read, where
they write and what they refuse:

- ``compile-frozen`` compiles the A1 audience only (a sealed document that would fail to parse is
  never opened); ``--include-sealed`` adds DEV-CHECK, into ``W/devcheck_SEALED``;
- every A1-facing command refuses the builder's work directory and, without ``--include-sealed``,
  a DEV-CHECK work directory or case document (refused before it is opened);
- ``run-physical`` refuses E3 on truth-relevant cases until a frozen truth table covering their
  template rows is recorded (``record-truth-table``), and runs the others;
- ``g3-material`` writes the reviewer-blind placements to the builder's work area only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from domains.production.engine.eval import freeze as e4a
from domains.production.engine.eval.freeze import cases as cs
from domains.production.engine.eval.freeze import g3_material, physical, prefreeze, recipes
from domains.production.engine.eval.freeze.common import dev_sku_for, sha256_file


def _write_json(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture()
def freeze(tmp_path, monkeypatch):
    """A freeze directory with one A1 and one DEV-CHECK STEP / recipe / case, a G3 template, a ledger."""
    fd, g3 = tmp_path / "repo_runlog" / "freeze", tmp_path / "repo_runlog" / "g3_truth_material"
    steps = {}
    for aud, data in (("A1", b"ISO-10303-21;a1"), ("DEVCHECK", b"ISO-10303-21;sealed")):
        p = tmp_path / "cad" / f"{aud}.stp"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        steps[aud] = {"sha256": sha256_file(p), "rel_path": f"cad/{aud}.stp", "aliases": [], "kind": "STEP",
                      "suffix": ".stp", "origin": "EXTERNAL_CONSUMED", "abs_path": str(p), "sha_matches": True}
    sku = {aud: dev_sku_for(s["sha256"]) for aud, s in steps.items()}
    _write_json(fd / "inventory.json", {"entries": [], "external_entries": list(steps.values()),
                                        "inventory_sha256": "0" * 64})
    _write_json(fd / "recipes" / f"{sku['A1']}.recipe.json", {"r": "a1"})
    _write_json(fd / "recipes_devcheck_SEALED" / f"{sku['DEVCHECK']}.recipe.json", {"r": "sealed"})
    _write_json(fd / "compile_report.json", {"rows": [{"sku": sku["A1"], "variants": {
        "dev": {"compiled_part_sha256": "a" * 64}}}]})

    def case(ref, lists, aud="A1", **kw):
        return {"case_ref": ref, "family": "11111111" if aud == "A1" else "22222222", "lists": lists,
                "scan_sha256": ref[3:] * 5, "scan_path": str(tmp_path / f"{ref}.ply"), "recipe_sku": sku[aud],
                "compiled_part_sha256": "a" * 64 if aud == "A1" else "d" * 64, **kw}

    _write_json(fd / "cases_A1.json", {"schema": cs.CASES_SCHEMA, "recipes_dir": "unused", "cases": [
        case("pc_state0000001", ["G2_COMPLETE", "BROAD"]), case("pc_staged000002", ["G3_STAGED"]),
        case("pc_provis000003", ["G2_COMPLETE"], provisional_complete=True)]})
    (fd / "devcheck_SEALED.json").write_text("{ this is sealed and must never be parsed by A1 commands",
                                             encoding="utf-8")
    template = {"schema": physical.TRUTH_TABLE_SCHEMA, "frozen_at": None, "rows": [
        {"case_ref": ref, "bend_uid": f"bu_{ref[3:9]}{i}", "reviewer_1": None, "reviewer_2": None, "tiebreak": None,
         "final": None} for ref in ("pc_state0000001", "pc_staged000002", "pc_provis000003") for i in range(2)]}
    _write_json(g3 / "truth_table_TEMPLATE.json", template)
    ledger = fd / "LEDGER.json"
    e4a.append_entry(ledger, label="pre-A1", items=e4a.freeze_items())
    calls = {"compile": [], "e3": []}

    def fake_compile(recipe_json, cad_path, out_dir, *, env, **kw):
        calls["compile"].append((Path(recipe_json).name, Path(out_dir)))
        out_dir.mkdir(parents=True, exist_ok=True)
        sealed = "recipes_devcheck_SEALED" in str(recipe_json)
        (out_dir / "compiled_part.json").write_text("{}", encoding="utf-8")
        return {"status": "COMPILED", "compiled_part_sha256": ("d" if sealed else "a") * 64}

    def fake_e3(c, scan_path, recipe_path, compiled_dir, work_root, **kw):
        calls["e3"].append((c["case_ref"], Path(work_root)))
        return {"status": "OK", "wall_s": 1.0}

    monkeypatch.setattr(recipes, "compile_recipe", fake_compile)
    monkeypatch.setattr(physical, "run_e3_case", fake_e3)
    monkeypatch.setattr(prefreeze, "custody_check", lambda p, purpose: Path(p).resolve())
    monkeypatch.setattr(prefreeze, "G3_DIR", g3)
    return {"fd": fd, "g3": g3, "sku": sku, "ledger": ledger, "calls": calls, "template": template,
            "tmp": tmp_path}


def _run(f, *args) -> int:
    return prefreeze.main(["--freeze-dir", str(f["fd"]), *map(str, args)])


def _out(capsys) -> dict:
    """The command's JSON report (PROGRESS lines dropped)."""
    text = capsys.readouterr().out
    return json.loads("\n".join(line for line in text.splitlines() if not line.startswith("PROGRESS\t")))


def test_compile_frozen_reads_and_compiles_the_a1_audience_only(freeze, capsys):
    w = freeze["tmp"] / "a1_work"
    assert _run(freeze, "compile-frozen", "--work", w) == 0  # the unparsable sealed document was never opened
    out = _out(capsys)
    assert out["audiences"] == ["A1"] and out["recompiled"] == {"A1": 1} and out["mismatches"] == []
    assert [c[0] for c in freeze["calls"]["compile"]] == [f"{freeze['sku']['A1']}.recipe.json"]
    assert (w / "compiled" / "dev" / freeze["sku"]["A1"] / "compiled_part.json").is_file()
    assert not (w / "devcheck_SEALED").exists() and not (w / "compiled" / "dev" / freeze["sku"]["DEVCHECK"]).exists()
    assert (w / "inputs").is_dir() and len(list((w / "inputs").iterdir())) == 1


def test_include_sealed_compiles_devcheck_into_its_own_sealed_area(freeze, capsys):
    _write_json(freeze["fd"] / "devcheck_SEALED.json", {"schema": cs.DEVCHECK_SCHEMA, "cases": [], "compile_rows": [
        {"sku": freeze["sku"]["DEVCHECK"], "variants": {"dev": {"compiled_part_sha256": "d" * 64}}}]})
    w = freeze["tmp"] / "final_devcheck_work"
    assert _run(freeze, "compile-frozen", "--work", w, "--include-sealed") == 0
    out = _out(capsys)
    assert out["recompiled"] == {"A1": 1, "DEVCHECK": 1} and out["matches_freeze"] == 2
    assert (w / "devcheck_SEALED" / "compiled" / "dev" / freeze["sku"]["DEVCHECK"] / "compiled_part.json").is_file()
    assert not (w / "compiled" / "dev" / freeze["sku"]["DEVCHECK"]).exists()
    assert _run(freeze, "compile-frozen", "--work", w) == 2  # it now holds DEV-CHECK material: never A1's again
    assert "holds DEV-CHECK material" in capsys.readouterr().err


def test_a1_facing_commands_refuse_builder_and_sealed_work_dirs_and_documents(freeze, capsys):
    builder = freeze["tmp"] / "e4b_work"
    (builder / "builder_SEALED").mkdir(parents=True)
    cases = freeze["fd"] / "cases_A1.json"
    for args in (("compile-frozen", "--work", builder),
                 ("run-physical", "--work", builder, "--cases", cases, "--ledger", freeze["ledger"]),
                 ("score-physical", "--work", builder, "--cases", cases, "--out", freeze["tmp"] / "x.json")):
        assert _run(freeze, *args) == 2
        assert "builder's work directory" in capsys.readouterr().err
    assert _run(freeze, "compile-frozen", "--work", freeze["tmp"] / "work_SEALED") == 2
    assert "holds DEV-CHECK material" in capsys.readouterr().err
    sealed_doc = freeze["fd"] / "devcheck_SEALED.json"
    for cmd in ("run-physical", "score-physical"):
        extra = ("--out", freeze["tmp"] / "y.json") if cmd == "score-physical" else ()
        assert _run(freeze, cmd, "--work", freeze["tmp"] / "a1_work", "--cases", sealed_doc, *extra) == 2
        assert "is DEV-CHECK material (a *SEALED* path)" in capsys.readouterr().err  # refused before parsing
    renamed = _write_json(freeze["tmp"] / "copy.json", {"schema": cs.DEVCHECK_SCHEMA, "cases": []})
    assert _run(freeze, "run-physical", "--work", freeze["tmp"] / "a1_work", "--cases", renamed) == 2
    assert "is the DEV-CHECK case document" in capsys.readouterr().err
    assert freeze["calls"]["compile"] == [] and freeze["calls"]["e3"] == []


def _fill(template: dict, *, drop: int = 0) -> dict:
    rows = [{**r, "reviewer_1": "FORMED", "reviewer_2": "FORMED", "final": "FORMED"} for r in template["rows"]]
    return {**template, "frozen_at": "2026-10-17T12:00:00Z", "reviewers": ["r1", "r2", "t"],
            "rows": rows[:len(rows) - drop]}


def test_run_physical_refuses_truth_relevant_cases_until_the_frozen_table_is_recorded(freeze, capsys):
    w = freeze["tmp"] / "a1_work"
    cases = freeze["fd"] / "cases_A1.json"
    assert _run(freeze, "run-physical", "--work", w, "--cases", cases, "--ledger", freeze["ledger"]) == 2
    out = _out(capsys)
    assert [r["case_ref"] for r in out["runs"]] == ["pc_state0000001"]  # the STATE_CHECK case runs
    refused = out["refused_by_amendment7_gate"]
    assert refused["cases"] == ["pc_provis000003", "pc_staged000002"]
    assert refused["reason"].startswith("NO_FROZEN_TRUTH_TABLE")
    table = _write_json(freeze["g3"] / "truth_table_FROZEN.json", _fill(freeze["template"]))
    assert _run(freeze, "run-physical", "--work", w, "--cases", cases, "--ledger", freeze["ledger"]) == 2
    assert "TRUTH_TABLE_NOT_RECORDED_IN_LEDGER" in _out(capsys)[
        "refused_by_amendment7_gate"]["reason"]
    short = _write_json(freeze["tmp"] / "short.json", _fill(freeze["template"], drop=1))
    assert _run(freeze, "record-truth-table", "--truth-table", short, "--ledger", freeze["ledger"]) == 2
    assert "1 template row(s) missing" in capsys.readouterr().err
    assert _run(freeze, "record-truth-table", "--ledger", freeze["ledger"]) == 0
    rec = _out(capsys)
    assert rec["a1_rows"] == 6 and rec["devcheck_recorded"] is False
    entry = e4a.latest_freeze(freeze["ledger"])
    assert entry["items"]["files"]["g3_truth_table_A1"]["sha256"] == sha256_file(table)
    assert "devcheck_sealed" not in entry["items"] and e4a.verify_ledger(freeze["ledger"])["ok"]
    freeze["calls"]["e3"].clear()
    assert _run(freeze, "run-physical", "--work", w, "--cases", cases, "--ledger", freeze["ledger"]) == 0
    out = _out(capsys)
    assert sorted(r["case_ref"] for r in out["runs"]) == ["pc_provis000003", "pc_staged000002", "pc_state0000001"]
    assert "refused_by_amendment7_gate" not in out and all(root == w.resolve() for _, root in freeze["calls"]["e3"])


def test_reviewer_blind_placements_go_to_the_builder_area_only(freeze, monkeypatch):
    _write_json(freeze["fd"] / "devcheck_SEALED.json", {"schema": cs.DEVCHECK_SCHEMA, "cases": [
        {"case_ref": "pc_sealed000009", "family": "22222222", "lists": ["G3_STAGED"], "scan_sha256": "9" * 64,
         "scan_path": "/nowhere.ply", "recipe_sku": freeze["sku"]["DEVCHECK"], "compiled_part_sha256": "d" * 64}]})

    def fake_render(case, scan_path, compiled_dir, out_dir, blind_dir, *, sections):
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(blind_dir / f"{case['case_ref']}.registration.json", {"placement_formed_set": ["x"]})
        return {"case_ref": case["case_ref"], "scan_sha256": case["scan_sha256"], "images": {}, "sections": sections,
                "bends": [{"bend_uid": "bu_1", "sequence_index": 1, "label": "B1"}]}

    monkeypatch.setattr(g3_material, "render_case_material", fake_render)
    work = freeze["tmp"] / "e4b_builder_work"
    assert _run(freeze, "g3-material", "--work", work) == 0
    blind = work / "builder_SEALED" / "g3_reviewer_blind"
    assert sorted(p.name for p in (blind / "A1").iterdir()) == [
        "pc_provis000003.registration.json", "pc_staged000002.registration.json", "pc_state0000001.registration.json"]
    assert [p.name for p in (blind / "DEVCHECK").iterdir()] == ["pc_sealed000009.registration.json"]
    assert not any("blind" in p.name for p in freeze["g3"].rglob("*"))  # nothing of it in the reviewer tree
    instructions = (freeze["g3"] / "REVIEWER_INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "_reviewer_blind" not in instructions and "record-truth-table" in instructions
    assert (freeze["g3"] / "devcheck_SEALED" / "MANIFEST.json").is_file()
