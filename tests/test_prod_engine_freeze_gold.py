"""E4b Gold11 transform recovery (amendment 6a) and the custody guard.

Synthetic label jobs in a temporary tree stand in for ``output/<job>/``; the real
Gold files are never read here (only the Gold script's source is parsed, by AST).
Every recovery method is shown to recover a known transform and every exclusion
reason to fire on the defect it guards against.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from domains.production.engine.eval.freeze import common, gold
from domains.production.engine.eval.freeze.common import CustodyError, custody_check
from domains.production.engine.eval.line_metric import GoldCase, prepare_gold

REPO = Path(__file__).resolve().parents[1]


def _ply_double(path: Path, pts: np.ndarray) -> str:
    """Binary LE double-precision PLY (the label jobs' display-cloud format)."""
    head = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty double x\nproperty double y\n" \
           "property double z\nend_header\n"
    data = head.encode() + np.ascontiguousarray(pts, dtype="<f8").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _rigid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.eye(4)
    t[:3, :3] = q
    t[:3, 3] = rng.normal(scale=50.0, size=3)
    return t


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A fake repo (script + label jobs) and source tree; custody admits the temp tree."""
    repo, src = tmp_path / "repo", tmp_path / "src"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "evaluate_f1_against_manual_gold.py").write_text(
        'CASE_JOBS = {"01_X": "bend_label_01_x", "02_Y": "bend_label_02_y"}\n', encoding="utf-8")
    monkeypatch.setattr(gold, "custody_check", lambda p, purpose: Path(p).resolve())
    monkeypatch.setattr(common, "custody_check", lambda p, purpose: Path(p).resolve())
    rng = np.random.default_rng(3)
    raw = rng.uniform(-100, 100, size=(4000, 3))
    raw_sha = _ply_double(src / "scan_x.ply", raw)
    return repo, src, raw, raw_sha


def _job(repo: Path, job: str, *, frame: str, raw_sha: str, display: np.ndarray | None,
         lines: list[dict], stored: np.ndarray | None = None) -> None:
    d = repo / "output" / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "bend_instance_labels.json").write_text(json.dumps(
        {"coordinate_frame": frame, "manual_bends": lines, "review_status": "complete"}), encoding="utf-8")
    (d / "bend_inspection_meta.json").write_text(json.dumps({"seed_case": {
        "coordinate_frame": frame, "coordinate_frame_provenance": "unaligned_source_scan_raw",
        "source_sha256": raw_sha, "labeling_mode": "scan_only"}}), encoding="utf-8")
    report = {"matches": []}
    if stored is not None:
        report["alignment"] = {"raw_scan_to_reference_transform": stored.tolist()}
    (d / "bend_inspection_report.json").write_text(json.dumps(report), encoding="utf-8")
    if display is not None:
        _ply_double(d / "aligned_scan.ply", display)


LINES = [{"bend_id": "M01", "line_start": [0.0, 0.0, 0.0], "line_end": [100.0, 0.0, 0.0]},
         {"bend_id": "M02", "line_start": [0.0, 50.0, 0.0], "line_end": [0.0, 50.0, 90.0]}]


def _rec(raw_sha: str, job: str = "bend_label_01_x", **kw) -> gold.GoldRecord:
    base = dict(gold11_id="CGTEST", catalog_case_id="corpus/01_X", family="12345678", label_job=job,
                scan_rel="scan_x.ply", scan_sha256=raw_sha, declared_frame="reference_mesh_raw",
                gold_line_count=len(LINES), custody="AUTHORIZED_SCRIPT_LOCATED")
    base.update(kw)
    return gold.GoldRecord(**base)


def test_identity_subset_is_recovered_and_scores_in_the_raw_frame(tree):
    repo, src, raw, sha = tree
    _job(repo, "bend_label_01_x", frame="reference_mesh_raw", raw_sha=sha, display=raw[::3], lines=LINES)
    row, case = gold.recover_case(_rec(sha), repo_root=repo, source_root=src)
    assert row["status"] == "RECOVERED" and row["method"] == "DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY"
    assert row["residual_max_mm"] == 0.0 and row["provenance_consistent"]
    assert case["case_id"] == "pc_" + sha[:12] and case["frame"] == "reference"
    usable, excluded = prepare_gold([GoldCase(case["case_id"], tuple(case["gold_lines"]), case["frame"],
                                              np.asarray(case["raw_to_reference"]), case["residual_mm"])])
    assert not excluded and np.allclose(usable[case["case_id"]][0]["line_end"], [100.0, 0.0, 0.0])


def test_a_display_cloud_off_by_0p05_mm_is_excluded(tree):
    repo, src, raw, sha = tree
    _job(repo, "bend_label_01_x", frame="reference_mesh_raw", raw_sha=sha, display=raw[::3] + [0.05, 0.0, 0.0],
         lines=LINES)
    row, case = gold.recover_case(_rec(sha), repo_root=repo, source_root=src)
    assert case is None and row["status"] == "EXCLUDED" and row["reason"] == "TRANSFORM_RESIDUAL_TOO_LARGE"
    assert row["attempts"][0]["residual_max_mm"] > 0.01


def test_stored_alignment_is_recovered_and_gold_maps_back_to_raw(tree):
    repo, src, raw, sha = tree
    t = _rigid(7)
    display = raw[::2] @ t[:3, :3].T + t[:3, 3]
    ref_lines = [{"bend_id": "M01", "line_start": (t[:3, :3] @ [1.0, 2, 3] + t[:3, 3]).tolist(),
                  "line_end": (t[:3, :3] @ [40.0, 2, 3] + t[:3, 3]).tolist()},
                 LINES[1]]
    _job(repo, "bend_label_01_x", frame="reference_mesh_raw", raw_sha=sha, display=display, lines=ref_lines, stored=t)
    row, case = gold.recover_case(_rec(sha), repo_root=repo, source_root=src)
    assert row["status"] == "RECOVERED" and row["method"] == "STORED_ALIGNMENT" and row["residual_max_mm"] < 1e-6
    usable, _ = prepare_gold([GoldCase(case["case_id"], tuple(case["gold_lines"]), "reference",
                                       np.asarray(case["raw_to_reference"]), case["residual_mm"])])
    assert np.allclose(usable[case["case_id"]][0]["line_start"], [1.0, 2.0, 3.0], atol=1e-6)


def test_same_order_kabsch_recovers_a_transformed_full_cloud(tree):
    repo, src, raw, sha = tree
    t = _rigid(11)
    _job(repo, "bend_label_01_x", frame="reference_mesh_raw", raw_sha=sha, display=raw @ t[:3, :3].T + t[:3, 3],
         lines=LINES)
    row, case = gold.recover_case(_rec(sha), repo_root=repo, source_root=src)
    assert row["method"] == "SAME_ORDER_KABSCH" and np.allclose(case["raw_to_reference"], t, atol=1e-8)
    methods = [a["method"] for a in row["attempts"]]
    assert methods == ["DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY", "SAME_ORDER_KABSCH"]


def test_sha_mismatch_custody_exclusion_and_unlocated_jobs(tree, tmp_path):
    repo, src, raw, sha = tree
    _job(repo, "bend_label_01_x", frame="reference_mesh_raw", raw_sha=sha, display=raw[::3], lines=LINES)
    row, case = gold.recover_case(_rec("0" * 64), repo_root=repo, source_root=src)
    assert case is None and row["reason"] == "SCAN_SHA_MISMATCH"
    email = _rec(sha, label_job=None, scan_rel=None, custody="CUSTODY_EMAIL_ATTACHMENT_NOT_AUTHORIZED")
    row, case = gold.recover_case(email, repo_root=repo, source_root=tmp_path / "does-not-exist")
    assert case is None and row["reason"] == "CUSTODY_EMAIL_ATTACHMENT_NOT_AUTHORIZED" and "scan_sha256" not in row
    row, case = gold.recover_case(_rec(sha, job="bend_label_99_z"), repo_root=repo, source_root=src)
    assert row["reason"] == "NOT_SCRIPT_LOCATED"
    _job(repo, "bend_label_02_y", frame="reference_mesh_raw", raw_sha=sha, display=raw[::3], lines=LINES[:1])
    row, _ = gold.recover_case(_rec(sha, job="bend_label_02_y"), repo_root=repo, source_root=src)
    assert row["reason"] == "GOLD_LINE_COUNT_DIFFERS_FROM_PINNED"


def test_scan_raw_labels_need_the_bound_sha(tree):
    repo, src, raw, sha = tree
    _job(repo, "bend_label_01_x", frame="scan_raw", raw_sha=sha, display=None, lines=LINES)
    row, case = gold.recover_case(_rec(sha, declared_frame="scan_raw"), repo_root=repo, source_root=src)
    assert row["method"] == "SHA256_BOUND_SCAN_RAW" and case["frame"] == "raw_scan"
    _job(repo, "bend_label_01_x", frame="reference_mesh_raw", raw_sha=sha, display=None, lines=LINES)
    row, case = gold.recover_case(_rec(sha), repo_root=repo, source_root=src)
    assert row["reason"] == "NO_RECOVERABLE_TRANSFORM"


def test_pinned_gold11_table_matches_the_script_and_the_review_receipt():
    jobs = gold.script_case_jobs(REPO)
    located = [r for r in gold.GOLD11 if r.custody == "AUTHORIZED_SCRIPT_LOCATED"]
    assert len(gold.GOLD11) == 11 and sum(r.gold_line_count for r in gold.GOLD11) == 71
    assert {r.label_job for r in located} == set(common.GOLD_LABEL_JOBS) <= set(jobs.values())
    xv = [r for r in gold.GOLD11 if r.custody == "AUTHORIZED_GOLD11_LINE_REF"]
    assert len(xv) == 2 and {r.labels_rel for r in xv} == set(common.GOLD11_XV_LABELS)
    assert all(r.declared_frame == "scan_raw" and "GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN" in r.flags for r in xv)
    assert {r.label_job for r in gold.HUMAN_NON_GOLD11} == set(common.HUMAN_NON_GOLD11_JOBS) <= set(jobs.values())
    assert all(r.tier == "HUMAN_NON_GOLD11" for r in gold.HUMAN_NON_GOLD11)
    receipt = json.loads((REPO / "docs/research/2026-09-09_gold11_input_review_summary.json").read_text())
    by_id = {c["case_id"]: c for c in receipt["cases"]}
    for r in gold.GOLD11:
        assert by_id[r.gold11_id]["family_id"] == r.family
        assert by_id[r.gold11_id]["reference_count"] == r.gold_line_count
        assert by_id[r.gold11_id]["reference_case_declarations"]["coordinate_frame"] == r.declared_frame


def test_custody_guard_admits_only_the_authorized_inputs():
    assert custody_check(common.SOURCE_ASSETS / "15651004_01.ply", "t")
    assert custody_check(REPO / "output/bend_label_03_10839000/bend_instance_labels.json", "t")
    assert custody_check(REPO / "output/bend_label_04_15651004/bend_instance_labels.json", "t")
    assert custody_check(REPO / common.GOLD11_XV_LABELS[0], "t")
    assert custody_check(REPO / "data/bend_corpus/parts/10839000/metadata.json", "t")
    assert custody_check(REPO / "docs/production/PRODUCTION_LINE_PLAN.md", "t")
    assert custody_check(REPO / "docs/production/runlog/freeze/devcheck_SEALED.json", "t")  # E4b's own output
    # a CONSUMED benchmark scan listed in BENCHMARK_V2 (Gold11 38172000 X) and an AI gold label file
    assert custody_check(REPO / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/"
                                "38172000_01_X.ply", "t")
    assert custody_check(REPO / "data/bend_corpus/annotations/ai_bend_lines_v1/1c62d21a1fceedefd236259a.json", "t")
    assert custody_check(REPO / "data/bend_corpus/annotations/ai_bend_lines_v1/provenance/"
                                "1c62d21a1fceedefd236259a.json", "t")
    for bad in ("data/bend_corpus/annotations/bend_line_gold_catalog_v1.json",
                "data/bend_corpus/annotations/ai_bend_lines_fresh_v1_SEALED/0123456789abcdef01234567.json",
                "docs/production/runlog/benchmark_v3/fresh_SEALED/x.json",
                "data/bend_corpus/incoming_codex_fetch_2026-10-17/x/48389000.stl",
                "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/other.ply",
                "output/bend_label_12_49125000/bend_instance_labels.json",
                "docs/research/HOLDOUT10_manifest.json", "output/improvement_loop/best_config.latest.json",
                "docs/notes/N016_case.json", "docs/notes/38168001_02.ply", "uploads/x/scan.ply"):
        with pytest.raises(CustodyError):
            custody_check(REPO / bad, "t")
    with pytest.raises(CustodyError):
        custody_check(REPO.parent / "elsewhere" / "x.ply", "t")
    with pytest.raises(CustodyError):
        custody_check(Path.home() / "rhino-preview" / "face.ply", "t")
    # the BENCHMARK_V2 eval tier FRESH_SCAN_OF_CONSUMED_PART keeps a scan out of the allowlist
    allow = common.consumed_allowlist()
    assert not any(v.get("scan_id") == "sc_90257b128d80" for v in allow.values())


def test_gold11_line_ref_record_recovers_by_the_bound_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(gold, "custody_check", lambda p, purpose: Path(p).resolve())
    repo = tmp_path / "repo"
    scan = repo / "data" / "x" / "scan_v.ply"
    sha = _ply_double(scan, np.random.default_rng(1).uniform(-50, 50, size=(500, 3)))
    lab = repo / "output" / "research" / "labels" / "C06" / "bend_instance_labels.json"
    lab.parent.mkdir(parents=True)
    lab.write_text(json.dumps({"coordinate_frame": "scan_raw", "scan_sha256": sha, "manual_bends": LINES,
                               "review_status": "codex_manual_multiview_complete"}), encoding="utf-8")
    rec = gold.GoldRecord("CGXV", "shaked_xv/C06", "12345678", None, None, sha, "scan_raw", 2,
                          "AUTHORIZED_GOLD11_LINE_REF", labels_rel="output/research/labels/C06/bend_instance_labels.json",
                          scan_path="data/x/scan_v.ply", flags=("GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN",))
    row, case = gold.recover_case(rec, repo_root=repo, source_root=tmp_path / "src", case_jobs={})
    assert row["status"] == "RECOVERED" and row["method"] == "SHA256_BOUND_SCAN_RAW" and case["frame"] == "raw_scan"
    assert case["flags"] == ["GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN"] and case["tier"] == "GOLD11"
    lab.write_text(json.dumps({"coordinate_frame": "scan_raw", "scan_sha256": "0" * 64, "manual_bends": LINES}),
                   encoding="utf-8")
    row, case = gold.recover_case(rec, repo_root=repo, source_root=tmp_path / "src", case_jobs={})
    assert case is None and row["reason"] == "SCAN_SHA_MISMATCH"


def test_a_prior_is_reused_only_while_it_is_valid(tree, monkeypatch):
    repo, src, raw, sha = tree
    _job(repo, "bend_label_01_x", frame="reference_mesh_raw", raw_sha=sha, display=raw[::3], lines=LINES)
    row, _ = gold.recover_case(_rec(sha), repo_root=repo, source_root=src)
    prior = {"status": "RECOVERED", "scan_sha256": sha, "labels_sha256": row["labels_sha256"],
             "display_sha256": row["display_cloud"]["sha256"], "method": row["method"],
             "residual_max_mm": row["residual_max_mm"], "raw_to_reference": row["raw_to_reference"],
             "source": {"path": "prior.json", "sha256": "p" * 64}}
    calls = []
    orig = gold.read_scan
    monkeypatch.setattr(gold, "read_scan", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError))
    reused, case = gold.recover_case(_rec(sha), repo_root=repo, source_root=src, prior=prior)
    assert reused["reused_from"]["path"] == "prior.json" and case["transform_method"] == row["method"] and not calls
    monkeypatch.setattr(gold, "read_scan", orig)
    for bad in ({"display_sha256": "d" * 64}, {"labels_sha256": "l" * 64}, {"scan_sha256": "s" * 64},
                {"residual_max_mm": 0.5}, {"method": "GUESS"}):
        again, case = gold.recover_case(_rec(sha), repo_root=repo, source_root=src, prior={**prior, **bad})
        assert "reused_from" not in again and again["status"] == "RECOVERED"  # recovered again from the files


def test_relabel_snapshot_maps_by_the_bound_scan_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(gold, "custody_check", lambda p, purpose: Path(p).resolve())
    d = tmp_path / "relabel"
    (d / "provenance").mkdir(parents=True)
    gcases = [{"case_id": "pc_" + "a" * 12, "scan_sha256": "a" * 64, "tier": "GOLD11", "gold11_id": "CGA"},
              {"case_id": "pc_" + "b" * 12, "scan_sha256": "b" * 64, "tier": "GOLD11", "gold11_id": "CGB"}]
    name = "0123456789abcdef01234567.json"
    (d / name).write_text(json.dumps({"coordinate_frame": "source_ply_raw", "bends": LINES, "asset_id": "x"}),
                          encoding="utf-8")
    (d / "provenance" / name).write_text(json.dumps({"asset_sha256": "a" * 64}), encoding="utf-8")
    monkeypatch.setattr(gold, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gold, "rel_to_repo", lambda p: Path(p).relative_to(tmp_path).as_posix())
    cases, snap = gold.relabel_gold_cases(gcases, d)
    assert snap["status"] == "IN_PROGRESS" and snap["gold11_cases_covered"] == ["pc_" + "a" * 12]
    assert len(cases) == 1 and cases[0]["frame"] == "raw_scan" and len(cases[0]["gold_lines"]) == 2
    assert gold.relabel_snapshot(gcases, tmp_path / "absent")["status"] == "ABSENT"


def _full_list() -> dict:
    """A full transform list with one A1 and one DEV-CHECK Gold11 case, an excluded row of each, and the relabel."""
    rows = [{"gold11_id": "CGA", "catalog_case_id": "corpus/01_11111111", "family": "11111111", "status": "RECOVERED",
             "case_ref": "pc_" + "a" * 12, "scan_sha256": "a" * 64, "labels_rel": "output/job_a/labels.json"},
            {"gold11_id": "CGS", "catalog_case_id": "corpus/02_22222222", "family": "22222222", "status": "RECOVERED",
             "case_ref": "pc_" + "s" * 12, "scan_sha256": "s" * 64, "labels_rel": "output/job_s/labels.json"},
            {"gold11_id": "CGX", "catalog_case_id": "corpus/03_11111111", "family": "11111111", "status": "EXCLUDED",
             "reason": "SCAN_SHA_MISMATCH"},
            {"gold11_id": "CGY", "catalog_case_id": "corpus/04_22222222", "family": "22222222", "status": "EXCLUDED",
             "reason": "NO_RECOVERABLE_TRANSFORM"}]
    rel_files = [{"file": "data/rel/a.json", "file_sha256": "1" * 64, "provenance_sha256": "7" * 64,
                  "case_ref": "pc_" + "a" * 12, "lines": 5, "usable": True},
                 {"file": "data/rel/s.json", "file_sha256": "3" * 64, "provenance_sha256": "4" * 64,
                  "case_ref": "pc_" + "s" * 12, "lines": 7, "usable": True},
                 {"file": "data/rel/z.json", "file_sha256": "5" * 64, "provenance_sha256": None, "case_ref": None,
                  "lines": 1, "usable": False}]
    return {"schema": gold.GOLD_TRANSFORMS_SCHEMA, "created_at": "2026-10-17T19:14:37Z", "recovered": 2,
            "gold11_records": 4, "gold_lines_recovered": 9, "scored_before_this_list": False,
            "excluded": [{"gold11_id": r["gold11_id"], "catalog_case_id": r["catalog_case_id"], "reason": r["reason"]}
                         for r in rows if r["status"] == "EXCLUDED"],
            "cases": rows, "gold11_relabel_v1": {"dir": "data/rel", "status": "IN_PROGRESS", "files": rel_files,
                                                 "gold11_cases_covered": ["pc_" + "a" * 12, "pc_" + "s" * 12],
                                                 "gold11_cases_total": 2, "snapshot_sha256": "f" * 64}}


def test_the_a1_part_of_the_transform_list_names_no_devcheck_gold():
    full = _full_list()
    pub = gold.public_gold_transforms(full, {"22222222"}, full_sha256="e" * 64,
                                      full_path="docs/production/runlog/freeze/gold_transforms_devcheck_SEALED.json")
    text = json.dumps(pub)
    for token in ("s" * 12, "s" * 64, "CGS", "CGY", "22222222", "job_s", "data/rel/s.json", "3" * 64, "data/rel/z.json"):
        assert token not in text, token
    assert [r["gold11_id"] for r in pub["cases"]] == ["CGA", "CGX"] and pub["cases_sealed"] == 2
    assert [e["gold11_id"] for e in pub["excluded"]] == ["CGX"] and pub["excluded_sealed_count"] == 1
    assert pub["recovered"] == 2 and pub["gold_lines_recovered"] == 9  # counts stay public
    assert pub["full_list"]["sha256"] == "e" * 64 and pub["scored_before_this_list"] is False
    rel = pub["gold11_relabel_v1"]
    assert [f["file"] for f in rel["files"]] == ["data/rel/a.json"] and rel["files_sealed"] == 2
    assert "lines" not in rel["files"][0] and rel["gold11_cases_covered_A1"] == ["pc_" + "a" * 12]
    assert rel["snapshot_sha256_A1"] == common.sha256_hex([["data/rel/a.json", "1" * 64]])
    a1, other = gold.relabel_file_split(full["gold11_relabel_v1"], {"pc_" + "a" * 12})
    assert [f["file"] for f in a1] == ["data/rel/a.json"] and [f["file"] for f in other] == ["data/rel/s.json",
                                                                                          "data/rel/z.json"]


def test_relabel_reading_restricted_to_listed_files_never_opens_the_others(tmp_path, monkeypatch):
    d = tmp_path / "relabel"
    (d / "provenance").mkdir(parents=True)
    gcases = [{"case_id": "pc_" + "a" * 12, "scan_sha256": "a" * 64, "tier": "GOLD11", "gold11_id": "CGA",
               "family": "11111111"}]
    names = {"0123456789abcdef01234567.json": "a" * 64, "fedcba9876543210fedcba98.json": "b" * 64}
    for name, sha in names.items():
        (d / name).write_text(json.dumps({"coordinate_frame": "source_ply_raw", "bends": LINES}), encoding="utf-8")
        (d / "provenance" / name).write_text(json.dumps({"asset_sha256": sha}), encoding="utf-8")
    opened = []
    monkeypatch.setattr(gold, "custody_check", lambda p, purpose: opened.append(Path(p).name) or Path(p).resolve())
    monkeypatch.setattr(gold, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gold, "rel_to_repo", lambda p: Path(p).relative_to(tmp_path).as_posix())
    cases, snap = gold.relabel_gold_cases(gcases, d, only_files=["relabel/0123456789abcdef01234567.json"])
    assert len(cases) == 1 and snap["read_scope"] == "LISTED_FILES_ONLY" and snap["other_files"] == 1
    assert "fedcba9876543210fedcba98.json" not in opened  # the other audience's file is counted, never opened
    assert snap["snapshot_sha256"] == common.sha256_hex(
        [["relabel/0123456789abcdef01234567.json", common.sha256_file(d / "0123456789abcdef01234567.json")]])
    gone = gold.relabel_snapshot(gcases, d, only_files=["missing00000000000000000.json"])
    assert gone["missing_files"] == ["missing00000000000000000.json"] and gone["files"] == []
