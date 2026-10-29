"""E4b BROAD list and gold (owner directions 2026-10-17; amendment 6a) and the external inventory.

A temporary tree stands in for the broad case list, the CAD/scan files and the
AI gold; no real scan or label is read. Every BROAD exclusion reason fires on
the defect it guards against, every gold kind lands in the raw-scan frame, and
the sha256 bindings are enforced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from domains.production.engine.eval.freeze import broad as br
from domains.production.engine.eval.freeze import inventory as iv

LINE = {"line_start": [0.0, 0.0, 0.0], "line_end": [80.0, 0.0, 0.0]}


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "custody_check", lambda p, purpose: Path(p).resolve())
    monkeypatch.setattr(br, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(br, "rel_to_repo", lambda p: Path(p).resolve().relative_to(tmp_path.resolve()).as_posix()
                        if Path(p).resolve().is_relative_to(tmp_path.resolve()) else str(p))
    cad = _write(tmp_path / "cad" / "a.stp", b"ISO-10303-21;a")
    bad_cad = _write(tmp_path / "cad" / "b.stp", b"ISO-10303-21;b")
    scans = {n: _write(tmp_path / "scans" / n, n.encode()) for n in ("s1.ply", "s2.stl", "s3.obj", "s4.ply", "s5.ply",
                                                                       "s6.ply", "s7.ply")}
    ai = tmp_path / "ai" / "0123456789abcdef01234567.json"
    ai.parent.mkdir(parents=True)
    ai.write_text(json.dumps({"coordinate_frame": "source_ply_raw", "bends": [{"bend_id": "B1", **LINE}],
                              "no_bends_present": False}), encoding="utf-8")
    (ai.parent / "provenance").mkdir()
    (ai.parent / "provenance" / ai.name).write_text(json.dumps({"asset_sha256": scans["s1.ply"]}), encoding="utf-8")
    zero = tmp_path / "ai" / "fedcba9876543210fedcba98.json"
    zero.write_text(json.dumps({"coordinate_frame": "source_ply_raw", "bends": [], "no_bends_present": True}),
                    encoding="utf-8")
    (ai.parent / "provenance" / zero.name).write_text(json.dumps({"asset_sha256": scans["s6.ply"]}), encoding="utf-8")
    lines = tmp_path / "bench" / "lines.json"
    lines.parent.mkdir(parents=True)
    lines.write_text(json.dumps({"coordinate_frame": "source_ply_raw", "scan_sha256": scans["s2.stl"], "bends": [
        {"bend_id": "B1", "status": "ok", "label_line": {"start": [0, 0, 0], "end": [0, 50, 0]}},
        {"bend_id": "B2", "status": "rejected", "label_line": {"start": [0, 0, 0], "end": [0, 50, 0]}}]}),
        encoding="utf-8")

    def case(cid, scan, fmt, gold, *, cad_path="cad/a.stp", cad_sha=cad, n=1, zero_ctrl=False, sha=None):
        return {"case_id": cid, "family_id": "12345678", "tier": "AI_GOLD_ACCEPTED", "format": fmt,
                "scan_path": str(tmp_path / "scans" / scan), "scan_sha256": sha or scans[scan],
                "cad_path": None if cad_path is None else str(tmp_path / cad_path), "cad_sha256": cad_sha,
                "gold": gold, "gold_line_count": n, "label_complete": True, "zero_bend_control": zero_ctrl,
                "stage_hints": [], "flags": []}

    cl = {"cases": [
        case("sc_1", "s1.ply", "ply", {"kind": "ai_studio", "path": "ai/0123456789abcdef01234567.json"}),
        case("sc_2", "s2.stl", "stl", {"kind": "bench_lines", "path": "bench/lines.json"}),
        case("sc_3", "s3.obj", "obj", {"kind": "ai_studio", "path": "ai/x.json"}),
        case("sc_4", "s4.ply", "ply", {"kind": "ai_studio", "path": "ai/x.json"}, cad_path=None, cad_sha=None),
        case("sc_5", "s5.ply", "ply", {"kind": "ai_studio", "path": "ai/x.json"}, cad_path="cad/b.stp", cad_sha=bad_cad),
        case("sc_6", "s6.ply", "ply", {"kind": "ai_studio", "path": "ai/fedcba9876543210fedcba98.json"}, n=0,
             zero_ctrl=True),
        case("sc_7", "s7.ply", "ply", {"kind": "gold11_catalog", "path": "catalog.json"}, sha="9" * 64),
    ]}
    compile_by_cad = {cad: {"variants": {"dev": {"status": "COMPILED"}}},
                      bad_cad: {"variants": {"dev": {"status": "BLOCKED"}}}}
    return tmp_path, cl, compile_by_cad, scans


def test_broad_rows_exclusion_reasons(tree):
    tmp, cl, comp, _ = tree
    rows = {r["broad_case_id"]: r for r in br.broad_rows(cl, comp)}
    assert rows["sc_1"]["status"] == "BROAD" and rows["sc_1"]["compile_status"] == "COMPILED"
    assert rows["sc_3"]["reason"].startswith("SCAN_FORMAT_NOT_ACCEPTED_BY_LINE")
    assert rows["sc_4"]["reason"] == "NO_STEP_CAD"
    assert rows["sc_5"]["reason"].startswith("CAD_DOES_NOT_COMPILE (BLOCKED)")
    assert rows["sc_7"]["reason"] == "SCAN_SHA_MISMATCH"  # the bytes differ from the list's sha256
    assert rows["sc_1"]["case_ref"].startswith("pc_") and rows["sc_1"]["scan_path"].endswith("s1.ply")


def test_broad_gold_kinds_and_bindings(tree):
    tmp, cl, comp, scans = tree
    rows = br.broad_rows(cl, comp)
    # a recovered human case for a gold11_catalog row (sha-matched)
    human = [{"scan_sha256": scans["s7.ply"], "frame": "reference", "gold_lines": [{"bend_id": "H1", **LINE}],
              "raw_to_reference": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "residual_mm": 0.0,
              "transform_method": "DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY", "labels_sha256": "l" * 64}]
    gold, excl = br.broad_gold(rows, human)
    by = {g["broad_case_id"]: g for g in gold}
    assert by["sc_1"]["frame"] == "raw_scan" and len(by["sc_1"]["gold_lines"]) == 1
    assert [l["bend_id"] for l in by["sc_2"]["gold_lines"]] == ["B1"]  # only status ok, apex label_line
    assert by["sc_6"]["zero_bend_control"] and by["sc_6"]["gold_lines"] == []
    assert "sc_7" not in by  # excluded before gold (sha mismatch row)
    # a wrong provenance sha excludes the case before scoring
    prov = tmp / "ai" / "provenance" / "0123456789abcdef01234567.json"
    prov.write_text(json.dumps({"asset_sha256": "0" * 64}), encoding="utf-8")
    gold, excl = br.broad_gold(rows, human)
    assert any(e["broad_case_id"] == "sc_1" and e["reason"] == "SCAN_SHA_MISMATCH" for e in excl)
    # a gold11_catalog row without a recovered transform is excluded, never scored blind
    row7 = {**next(r for r in rows if r["broad_case_id"] == "sc_7"), "status": "BROAD", "scan_sha256": scans["s7.ply"]}
    gold, excl = br.broad_gold([row7], [])
    assert not gold and excl[0]["reason"] == "NO_RECOVERABLE_TRANSFORM"
    gold, _ = br.broad_gold([row7], human)
    assert gold[0]["frame"] == "reference" and gold[0]["transform_method"].startswith("DISPLAY_CLOUD")


def test_external_inventory_entries_are_hashed_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(iv, "custody_check", lambda p, purpose: Path(p).resolve())
    src = tmp_path / "src"
    src.mkdir()
    same = _write(src / "dup.stp", b"same")
    ext_ok = _write(tmp_path / "ext" / "e.stp", b"ext")
    scan_ok = _write(tmp_path / "ext" / "s.ply", b"scan")
    allow = {str(tmp_path / "ext" / "e.stp"): {"kind": "STEP", "part_id": "12345678", "sha256": ext_ok},
             str(tmp_path / "ext" / "bad.stp"): {"kind": "STEP", "part_id": "12345678", "sha256": "0" * 64},
             str(tmp_path / "ext" / "copy.stp"): {"kind": "STEP", "part_id": "12345678", "sha256": same}}
    _write(tmp_path / "ext" / "bad.stp", b"tampered")
    _write(tmp_path / "ext" / "copy.stp", b"same")
    monkeypatch.setattr(iv, "consumed_allowlist", lambda manifest=None: allow)
    cl = tmp_path / "cl.json"
    cl.write_text(json.dumps({"cases": [{"case_id": "sc_x", "family_id": "12345678", "scan_path": str(tmp_path / "ext/s.ply"),
                                         "scan_sha256": scan_ok, "cad_path": str(tmp_path / "ext/e.stp")}]}),
                  encoding="utf-8")
    src_entries = [{"sha256": same, "rel_path": "dup.stp", "aliases": [], "kind": "STEP"}]
    ext = iv.external_entries(src_entries, manifest=tmp_path / "m.json", case_list=cl, source_root=src)
    by = {Path(e["rel_path"]).name: e for e in ext}
    assert by["e.stp"]["sha_matches"] and by["s.ply"]["kind"] == "SCAN" and by["s.ply"]["sha_matches"]
    assert not by["bad.stp"]["sha_matches"]  # listed, never used
    assert "copy.stp" not in by and src_entries[0]["external_aliases"]  # bytes equal a source entry: an alias
    inv = {"entries": src_entries, "external_entries": ext}
    assert {Path(e["rel_path"]).name for e in iv.all_entries(inv, "STEP")} == {"dup.stp", "e.stp"}


def test_recipe_count_check_flags_gold_a_recipe_guided_engine_cannot_match():
    def row(bends, *, excluded=(), merges=(), status="COMPILED"):
        return {"variants": {"dev": {"status": status, "bends": bends}},
                "dev_policy_report": {"excluded_by_unsupported_rule": [{"rule": r} for r in excluded],
                                      "merge_notes": list(merges)}}

    flags, check = br.recipe_count_check(9, row(7, merges=("merged 2 pieces of graph bend G1",)), staged=False)
    assert flags == ["GOLD_LINES_EXCEED_RECIPE_BENDS", "GOLD_CONVENTION_CONFLICT"]
    assert check["true_positive_max"] == 7 and check["recall_ceiling"] == pytest.approx(7 / 9)
    assert check["f1_ceiling"] == pytest.approx(14 / 16)
    flags, check = br.recipe_count_check(2, row(1, excluded=("ROLLED", "OTHER")), staged=True)
    assert flags == ["GOLD_LINES_EXCEED_RECIPE_BENDS", "RECIPE_EXCLUDED_FEATURE"] and "ROLLED" in check["causes"][0]
    assert br.recipe_count_check(3, row(3), staged=False) == ([], {"gold_lines": 3, "recipe_bends": 3, "rule": (
        "a recipe-guided engine publishes at most one line per recipe bend")})
    flags, check = br.recipe_count_check(10, row(12), staged=False)
    assert flags == ["GOLD_LINES_FEWER_THAN_RECIPE_BENDS"] and "no hard ceiling" in check["note"]
    assert br.recipe_count_check(4, row(12), staged=True)[0] == []  # a staged scan's unformed bends emit no line
    assert br.recipe_count_check(0, row(2), staged=False, zero_bend_control=True) == ([], None)
    assert br.recipe_count_check(5, row(2, status="BLOCKED"), staged=False) == ([], None)
    gold = [{"case_id": "pc_1", "gold_lines": [LINE] * 9, "flags": ["KEEP"]},
            {"case_id": "pc_2", "gold_lines": [LINE] * 2, "flags": []}, {"case_id": "pc_x", "gold_lines": [LINE]}]
    cases = [{"case_ref": "pc_1", "cad_sha256": "c1", "stage_class": "COMPLETE"},
             {"case_ref": "pc_2", "cad_sha256": "c2", "stage_class": "COMPLETE"}]
    n = br.annotate_recipe_counts(gold, cases, {"c1": row(7, merges=("m",)), "c2": row(2)})
    assert n == 1 and gold[0]["flags"] == ["KEEP", "GOLD_LINES_EXCEED_RECIPE_BENDS", "GOLD_CONVENTION_CONFLICT"]
    assert gold[1]["flags"] == [] and gold[1]["recipe_count_check"]["recipe_bends"] == 2
    assert "recipe_count_check" not in gold[2]  # no physical case: left as it is
