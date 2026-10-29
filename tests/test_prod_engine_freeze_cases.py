"""E4b frozen case lists and the DEV-CHECK draw (amendments 4e, 4f, 6f, 7).

A synthetic inventory with the real file names of the plan's lists (no real file
is read; corpus metadata comes from a temporary tree) exercises every rule:
staged / complete / blank classes, format alternates, byte-identical repeat
copies, excluded scans, provisional-complete stage keys, known-good membership,
the stratified input-seeded draw, and the public/sealed split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from domains.production.engine.eval.freeze import cases as cs
from domains.production.engine.eval.freeze.common import case_ref_for, dev_sku_for


def _sha(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _scan(rel: str, fam: str | None, aliases=(), fmt="PLY_BINARY_LE"):
    suffix = Path(rel).suffix.lower()
    return {"sha256": _sha(rel), "rel_path": rel, "aliases": list(aliases), "kind": "SCAN", "suffix": suffix,
            "bytes": 10, "family": fam, "header": {"format": fmt if suffix == ".ply" else "STL_BINARY",
                                                    "vertex_count": 100, "face_count": 50}}


def _step(rel: str, fam: str | None):
    return {"sha256": _sha(rel), "rel_path": rel, "aliases": [], "kind": "STEP", "suffix": ".stp", "bytes": 10,
            "family": fam}


SCANS = [
    ("10839000_01_full_scan.ply", "10839000", ["10839000/10839000_01_full_scan.ply"]),
    ("10839000/10839000_01_scan_1.ply", "10839000", []), ("10839000/10839000_01_scan_2.ply", "10839000", []),
    ("BEFORE BENDING  11979003_C.ply", "11979003", []), ("11979003_C.stl", "11979003", ["11979003_C (1).stl"]),
    ("23553000_B_FULL_SCAN 1.ply", "23553000", []), ("23553000_B_SCAN_1.ply", "23553000", []),
    ("23553000_B_SCAN_2.ply", "23553000", []), ("28870000_A_FULL_SCAN.ply", "28870000", []),
    ("28870000_A_SCAN_1.ply", "28870000", []), ("38172000_01_SCAN_1.ply", "38172000", []),
    ("38172000_01_SCAN_2 1.ply", "38172000", []), ("40500000_0_SCAN1 1.ply", "40500000", []),
    ("40500000_0_SCAN2 1.ply", "40500000", []),
    ("15651004_01.ply", "15651004", []), ("15651004_01_SECOND_SCAN.stl", "15651004", []),
    ("15651004_01_3RD_SCAN.stl", "15651004", []),
    ("42288005_01_FULL_SCAN.ply", "42288005", ["42288005_01_FULL_SCAN (1).ply"]),
    ("47959001_01PREL_scan1.ply", "47959001", []), ("47959001_01PREL_scan1.stl", "47959001", []),
    ("37361005_01.ply", "37361005", []), ("48963008_B.ply", "48963008", []), ("48991006_A 4.ply", "48991006", []),
    ("49024000_A.ply", "49024000", []), ("49024000_A.stl", "49024000", []), ("44211000_A.ply", "44211000", []),
    ("47266000_F.stl", "47266000", []), ("47479000_C.ply", "47479000", []), ("49001000_A-PRELIM.ply", "49001000", []),
    ("49204020_04.ply", "49204020", []),
    ("17139000_01.ply", "17139000", []), ("part 1.ply", None, []),
]
STEP_FAMS = ["10839000", "11979003", "23553000", "28870000", "38172000", "40500000", "15651004", "42288005",
             "47959001", "37361005", "48963008", "48991006", "49024000", "44211000", "47266000", "47479000",
             "49001000", "49204020"]


@pytest.fixture()
def inputs(tmp_path, monkeypatch):
    entries = [_scan(r, f, a) for r, f, a in SCANS] + [_step(f"{f}.stp", f) for f in STEP_FAMS]
    inv = {"entries": entries, "inventory_sha256": "a" * 64}
    rows = [{"cad_sha256": _sha(f"{f}.stp"), "family": f, "sku": dev_sku_for(_sha(f"{f}.stp")),
             "variants": {"dev": {"status": "BLOCKED" if f == "47479000" else "COMPILED", "bends": 3,
                                  "compiled_part_sha256": "c" * 64, "recipe_sha256": "r" * 64}}} for f in STEP_FAMS]
    gold = [{"case_id": case_ref_for(_sha("10839000_01_full_scan.ply")), "scan_sha256": _sha("10839000_01_full_scan.ply"),
             "family": "10839000", "frame": "reference", "gold_lines": []},
            {"case_id": case_ref_for(_sha("49024000_A.ply")), "scan_sha256": _sha("49024000_A.ply"),
             "family": "49024000", "frame": "reference", "gold_lines": []}]
    repo = tmp_path / "repo"
    for fam, files in {"15651004": [("15651004_01_SECOND_SCAN.stl", "partial"), ("15651004_01.ply", "full")],
                       "23553000": [("23553000_B_SCAN_1.ply", "partial")]}.items():
        d = repo / "data" / "bend_corpus" / "parts" / fam
        d.mkdir(parents=True)
        meta = {"scan_files": [{"name": n, "state": s, "kind": s, "sequence": 1} for n, s in files]}
        if fam == "23553000":
            meta["user_truth"] = {"scan_truth": {"23553000_B_SCAN_1.ply": {"stage": "partial_stage_1",
                                                                           "completed_bends": 4}},
                                  "expected_total_bends": 16}
        (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(cs, "custody_check", lambda p, purpose: Path(p))
    return inv, {"rows": rows}, gold, repo


def _by_name(doc, rel):
    return next(c for c in doc["cases"] if c["scan_rel"].endswith(rel))


def test_lists_classes_alternates_repeats_and_exclusions(inputs):
    inv, report, gold, repo = inputs
    doc = cs.build_cases(inv, report, gold, repo_root=repo)
    full = _by_name(doc, "10839000_01_full_scan.ply")
    assert full["stage_class"] == "STAGED_FULL" and {"G3_STAGED", "G2_COMPLETE", "KNOWN_GOOD:stable",
                                                       "GOLD_G2A"} <= set(full["lists"])
    part = _by_name(doc, "10839000/10839000_01_scan_1.ply")
    assert part["stage_class"] == "STAGED_PARTIAL" and "G2_COMPLETE" not in part["lists"]
    blank = _by_name(doc, "BEFORE BENDING  11979003_C.ply")
    assert blank["stage_class"] == "BLANK_STAGE0" and blank["stage_keys"]["expected_completed"] == 0
    stl = _by_name(doc, "11979003_C.stl")
    assert "G2_COMPLETE" in stl["lists"] and "KNOWN_GOOD:secondary" in stl["lists"]
    s1 = _by_name(doc, "23553000_B_SCAN_1.ply")
    assert s1["stage_keys"]["expected_completed"] == 4 and s1["stage_keys"]["expected_total_bends"] == 16
    assert not any(c["scan_rel"].endswith("49024000_A.stl") for c in doc["cases"])  # a format alternate in no list
    fmt_pair = _by_name(doc, "47959001_01PREL_scan1.stl")
    assert "REPEATABILITY:47959001" in fmt_pair["lists"] and "G2_COMPLETE" not in fmt_pair["lists"]
    second = _by_name(doc, "15651004_01_SECOND_SCAN.stl")
    assert second["provisional_complete"] and "STAGE_KEYS_SAY_PARTIAL" in second["notes"][0]
    assert not _by_name(doc, "15651004_01.ply")["provisional_complete"]
    sets = {s["family"]: s for s in doc["repeatability_sets"]}
    assert sets["15651004"]["status"] == "EVALUABLE" and sets["15651004"]["kind"] == "PHYSICAL_REPEATS"
    assert len(sets["15651004"]["case_refs"]) == 3
    assert sets["42288005"]["status"] == "NOT_EVALUABLE" and "BYTE_IDENTICAL" in sets["42288005"]["reason"]
    assert sets["47959001"]["kind"] == "FORMAT_PAIR"
    reasons = {e["scan_rel"].split("/")[-1]: e["reason"] for e in doc["excluded_scans"]}
    assert reasons["17139000_01.ply"] == "NO_AUTHORIZED_STEP" and reasons["part 1.ply"] == "NO_PART_NUMBER"
    assert reasons["49024000_A.stl"].startswith("FORMAT_ALTERNATE_IN_NO_LIST") and len(reasons) == 3
    blocked = _by_name(doc, "47479000_C.ply")
    assert blocked["compile_status"] == "BLOCKED" and "G2_COMPLETE" in blocked["lists"]  # stays in the denominators


def test_devcheck_draw_is_stratified_seeded_and_reproducible(inputs):
    inv, report, gold, repo = inputs
    doc = cs.build_cases(inv, report, gold, repo_root=repo)
    sets = doc["repeatability_sets"]
    prev = {"families": ["15651004"], "seed_sha256": "e" * 64, "rule_version": "E4B-DEVCHECK-V2"}
    d1 = cs.draw_devcheck(doc["cases"], inv["inventory_sha256"], sets, superseded=prev)
    d2 = cs.draw_devcheck(doc["cases"], inv["inventory_sha256"], sets, superseded=prev)
    assert d1 == d2 and d1["rule_version"] == "E4B-DEVCHECK-V4"
    assert d1["families_total"] == 18 and len(d1["families"]) == 7
    assert d1["strata"]["GOLD"] == ["49024000"] and d1["selected_by_stratum"]["GOLD"] == ["49024000"]
    assert d1["gold11_families_selected"] == ["49024000"]  # G2(a) stays evaluable on DEV-CHECK (amendment 4f)
    assert d1["superseded_rule_v3"]["families"] and "GOLD" in d1["superseded_rule_v3"]["why"]
    assert d1["strata"]["REPEATABILITY"] == ["15651004", "47959001"]  # 42288005's "pair" is one file
    assert "42288005" in d1["strata"]["OTHER"]
    assert len(d1["selected_by_stratum"]["STAGED"]) == 2 and len(d1["selected_by_stratum"]["REPEATABILITY"]) == 1
    assert len(d1["selected_by_stratum"]["OTHER"]) == 3
    assert set(d1["selected_by_stratum"]["STAGED"]) <= set(cs.G3_PLAN_STAGED)
    draws = [cs.draw_devcheck(doc["cases"], h * 64, sets) for h in "0123456789abcdef"]
    assert len({tuple(d["families"]) for d in draws}) > 1  # the seed really depends on the inventory hash
    assert all("10839000" not in d["families"] for d in [d1, *draws])  # the app E2E part is never held out
    assert all(d["selected_by_stratum"]["STAGED"] for d in draws)  # always >= 1 staged family
    assert all(d["selected_by_stratum"]["GOLD"] for d in draws)  # always >= 1 Gold11 family
    assert d1["superseded_rule_v2"]["families"] == ["15651004"] and "BROAD" in d1["superseded_rule_v2"]["why"]
    assert "superseded_rule_v2" not in cs.draw_devcheck(doc["cases"], inv["inventory_sha256"], sets)


def test_public_lists_contain_no_devcheck_family(inputs):
    inv, report, gold, repo = inputs
    doc = cs.build_cases(inv, report, gold, repo_root=repo)
    dev = cs.draw_devcheck(doc["cases"], inv["inventory_sha256"], doc["repeatability_sets"])
    # force a Gold family into DEV-CHECK to check the gold split too
    dev = {**dev, "families": sorted(set(dev["families"]) | {"49024000"})}
    public, sealed = cs.split_public_sealed(doc, dev, gold)
    sealed_fams = set(dev["families"])
    text = json.dumps(public)
    for c in sealed["cases"]:
        assert c["family"] in sealed_fams and c["case_ref"] not in text and c["scan_sha256"] not in text
    assert all(c["family"] not in sealed_fams for c in public["cases"])
    assert {g["family"] for g in sealed["gold_cases"]} == {"49024000"}
    assert all(g["family"] not in sealed_fams for g in public["gold_cases"])
    assert all(s["family"] not in sealed_fams for s in public["repeatability_sets"])
    for fams in public["known_good_families"].values():
        assert not set(fams) & sealed_fams
    assert len(public["cases"]) + len(sealed["cases"]) == len(doc["cases"])
    assert all(e.get("family") not in sealed_fams for e in public["excluded_scans"])
    assert any(e["scan_rel"].endswith("49024000_A.stl") for e in sealed["excluded_scans"])
    assert public["lists"]["G2_COMPLETE"] and sealed["draw"]["seed_sha256"] == dev["seed_sha256"]


def _broad_row(name: str, fam: str, *, status="BROAD", hints=(), gold11=False, cad=None, reason=None):
    sha = _sha(name)
    return {"broad_case_id": "sc_" + sha[:12], "case_ref": case_ref_for(sha), "family": fam, "tier": "AI_GOLD_ACCEPTED",
            "scan_sha256": sha, "scan_path": f"/data/{name}", "scan_rel": f"data/{name}", "status": status,
            "reason": reason, "cad_sha256": cad or _sha(f"{fam}.stp"), "stage_hints": list(hints),
            "staged_or_incomplete": bool(hints), "flags": [], "label_complete": True, "zero_bend_control": False,
            "in_gold11": gold11, "gold11_id": "CGX" if gold11 else None}


def test_broad_rows_external_cases_and_manifest_exclusions(inputs):
    inv, report, gold, repo = inputs
    inv = dict(inv)
    ext = [{"sha256": _sha(n), "rel_path": f"data/{n}", "aliases": [], "kind": "SCAN", "suffix": ".ply",
            "origin": "EXTERNAL_CONSUMED", "abs_path": f"/data/{n}", "sha_matches": True, "family": f,
            "header": {"format": "PLY_BINARY_LE", "vertex_count": 9, "face_count": 9}}
           for n, f in (("38172000_01_X.ply", "38172000"), ("45796000_A_SCAN_1.ply", "45796000"),
                        ("45796000_A_FULL_SCAN.ply", "45796000"))]
    ext.append({"sha256": _sha("45796000.stp"), "rel_path": "~/Downloads/45796000_A.stp", "aliases": [],
                "kind": "STEP", "suffix": ".stp", "origin": "EXTERNAL_CONSUMED", "abs_path": "/dl/45796000_A.stp",
                "sha_matches": True, "family": "45796000"})
    inv["external_entries"] = ext
    report = {"rows": report["rows"] + [{"cad_sha256": _sha("45796000.stp"), "family": "45796000", "variants": {
        "dev": {"status": "COMPILED", "bends": 8, "compiled_part_sha256": "c" * 64, "recipe_sha256": "r" * 64}}}]}
    xv_sha = _sha("38172000_01_X.ply")
    gold = gold + [{"case_id": case_ref_for(xv_sha), "scan_sha256": xv_sha, "family": "38172000", "tier": "GOLD11",
                    "frame": "raw_scan", "gold_lines": []}]
    rows = [_broad_row("38172000_01_X.ply", "38172000", gold11=True),
            _broad_row("45796000_A_SCAN_1.ply", "45796000", hints=["PARTIAL_SCAN_1"]),
            _broad_row("45796000_A_FULL_SCAN.ply", "45796000", hints=["FULL"]),
            _broad_row("49001000_A-PRELIM.obj", "49001000", status="EXCLUDED",
                       reason="SCAN_FORMAT_NOT_ACCEPTED_BY_LINE (obj)"),
            {**_broad_row("23553000_B_SCAN_1.ply", "23553000"), "scan_path": None}]
    bm = {_sha("17139000_01.ply"): {"eval_tier": "FRESH_SCAN_OF_CONSUMED_PART", "role": "benchmark_scan"},
          _sha("49024000_A.stl"): {"eval_tier": None, "role": "FORMAT_ALTERNATE", "primary_truth": "CAD_RECIPE"}}
    doc = cs.build_cases(inv, report, gold, broad_rows=rows, benchmark_scans=bm, repo_root=repo)
    x = next(c for c in doc["cases"] if c["scan_sha256"] == xv_sha)
    assert x["lists"][:1] == ["G2_COMPLETE"] and {"BROAD", "GOLD_G2A"} <= set(x["lists"])
    assert x["scan_origin"] == "EXTERNAL_CONSUMED" and x["scan_path"] == "/data/38172000_01_X.ply"
    s1 = next(c for c in doc["cases"] if c["scan_sha256"] == _sha("45796000_A_SCAN_1.ply"))
    assert s1["lists"] == ["BROAD", "BROAD_STAGED"] and s1["stage_class"] == "STAGED_PARTIAL"
    assert s1["cad_path"] == "/dl/45796000_A.stp" and s1["compile_status"] == "COMPILED"
    full = next(c for c in doc["cases"] if c["scan_sha256"] == _sha("45796000_A_FULL_SCAN.ply"))
    assert full["lists"] == ["BROAD"] and full["stage_class"] == "COMPLETE"
    sa = next(c for c in doc["cases"] if c["scan_rel"].endswith("23553000_B_SCAN_1.ply"))
    assert "BROAD" in sa["lists"] and "G3_STAGED" in sa["lists"]  # a source-tree scan joins BROAD in place
    reasons = {str(e["scan_rel"]).split("/")[-1]: e["reason"] for e in doc["excluded_scans"]}
    assert reasons["17139000_01.ply"].startswith("RESERVED_FRESH_SCAN_OF_CONSUMED_PART")
    assert reasons["49001000_A-PRELIM.obj"].startswith("BROAD_EXCLUDED: SCAN_FORMAT_NOT_ACCEPTED_BY_LINE")
    assert reasons["49024000_A.stl"].startswith("FORMAT_ALTERNATE_IN_NO_LIST")  # stem pair rule first
    dev = cs.draw_devcheck(doc["cases"], "b" * 64, doc["repeatability_sets"])
    assert "45796000" in dev["strata"]["STAGED"] and dev["families_total"] == 19
    assert "10839000" not in dev["families"] and dev["selected_by_stratum"]["STAGED"]


def test_split_seals_broad_gold_and_relabel_of_devcheck_families(inputs):
    inv, report, gold, repo = inputs
    doc = cs.build_cases(inv, report, gold, repo_root=repo)
    dev = {**cs.draw_devcheck(doc["cases"], inv["inventory_sha256"], doc["repeatability_sets"]), "families": ["49024000"]}
    bgold = [{"case_id": "pc_1", "family": "49024000", "gold_lines": []}, {"case_id": "pc_2", "family": "37361005"}]
    relabel = [{"case_id": "pc_3", "family": "49024000"}, {"case_id": "pc_4", "family": "10839000"}]
    excl = [{"case_id": "pc_5", "family": "49024000", "reason": "NO_RECOVERABLE_TRANSFORM"}]
    public, sealed = cs.split_public_sealed(doc, dev, gold, broad_gold=bgold, broad_excluded=excl,
                                            relabel_cases=relabel)
    assert [g["case_id"] for g in public["broad_gold_cases"]] == ["pc_2"]
    assert [g["case_id"] for g in sealed["broad_gold_cases"]] == ["pc_1"]
    assert [g["case_id"] for g in public["relabel_gold_cases"]] == ["pc_4"]
    assert [g["case_id"] for g in sealed["relabel_gold_cases"]] == ["pc_3"]
    assert sealed["broad_gold_excluded_before_scoring"] and not public["broad_gold_excluded_before_scoring"]
    assert all(c["family"] != "49024000" for c in public["cases"]) and public["schema"] == cs.CASES_SCHEMA


def test_split_lists_each_audience_only_its_own_relabel_files(inputs):
    inv, report, gold, repo = inputs
    doc = cs.build_cases(inv, report, gold, repo_root=repo)
    dev = {**cs.draw_devcheck(doc["cases"], inv["inventory_sha256"], doc["repeatability_sets"]), "families": ["49024000"]}
    a1_ref, sealed_ref = gold[0]["case_id"], gold[1]["case_id"]  # 10839000 (A1) and 49024000 (sealed)
    snap = {"status": "COMPLETE_PENDING_OWNER_SIGNOFF", "files": [
        {"file": "data/rel/a.json", "file_sha256": "1" * 64, "provenance_sha256": "2" * 64, "case_ref": a1_ref},
        {"file": "data/rel/s.json", "file_sha256": "3" * 64, "provenance_sha256": "4" * 64, "case_ref": sealed_ref},
        {"file": "data/rel/u.json", "file_sha256": "5" * 64, "provenance_sha256": None, "case_ref": None}]}
    public, sealed = cs.split_public_sealed(doc, dev, gold, relabel_snapshot=snap)
    assert [f["file"] for f in public["relabel_files"]] == ["data/rel/a.json"] and public["relabel_other_files"] == 2
    assert [f["file"] for f in sealed["relabel_files"]] == ["data/rel/s.json", "data/rel/u.json"]
    assert sealed["relabel_other_files"] == 1
    assert "data/rel/s.json" not in json.dumps(public) and "3" * 64 not in json.dumps(public)
    from domains.production.engine.eval.freeze.common import doc_sha256

    assert public["relabel_snapshot_sha256"] == doc_sha256([["data/rel/a.json", "1" * 64]])
    assert public["relabel_snapshot_status"] == "COMPLETE_PENDING_OWNER_SIGNOFF"
