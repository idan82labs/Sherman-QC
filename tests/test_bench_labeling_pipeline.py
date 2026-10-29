"""AI-vision bend labeling tool: synthetic L / U / hat-with-holes accuracy and contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

from research.bench_labeling import AI_ANNOTATOR, AI_REVIEW_STATUS  # noqa: E402
from research.bench_labeling.cli import main as cli_main  # noqa: E402
from research.bench_labeling.custody import CustodyError, check_path  # noqa: E402
from research.bench_labeling.label import LabelWriteError, validate_ai_label, write_label  # noqa: E402
from research.bench_labeling.lines import parse_pairs, run_lines  # noqa: E402
from research.bench_labeling.prep import run_prep  # noqa: E402
from research.bench_labeling.render import REQUIRED_VIEWS  # noqa: E402
from research.bench_labeling.synthetic import make_part, write_part  # noqa: E402

POS_TOL_MM = 0.5
ANG_TOL_DEG = 0.5


def _patch_at(face, ds):
    d = np.linalg.norm(ds["points"] - face["probe"], axis=1)
    near = ds["labels"][np.argsort(d)[:7]]
    near = near[near > 0]
    assert len(near), "probe point fell on unsegmented points"
    return int(np.bincount(near).argmax())


def _outer_pairs(part, prep_dir):
    ds = np.load(Path(prep_dir) / "points_ds.npz")
    pairs = []
    for bend in part.bends:
        sign = bend["outer_sign"]
        ka, kb = bend["segments"]
        fa = next(f for f in part.faces if f["segment"] == ka and f["skin"] == sign)
        fb = next(f for f in part.faces if f["segment"] == kb and f["skin"] == sign)
        pairs.append([_patch_at(fa, ds), _patch_at(fb, ds)])
    return pairs


def _line_errors(start, end, truth):
    g0, g1 = (np.asarray(p, float) for p in truth)
    gd = (g1 - g0) / np.linalg.norm(g1 - g0)

    def perp(p):
        r = np.asarray(p, float) - g0
        return float(np.linalg.norm(r - (r @ gd) * gd))

    d = np.asarray(end, float) - np.asarray(start, float)
    d /= np.linalg.norm(d)
    angle = float(np.degrees(np.arccos(min(1.0, abs(d @ gd)))))
    span = sorted([float((np.asarray(start) - g0) @ gd), float((np.asarray(end) - g0) @ gd)])
    return max(perp(start), perp(end)), angle, span


CASES = [("L", "mesh", ".ply"), ("U", "stl", ".stl"), ("U", "cloud", ".ply"), ("hat", "mesh", ".ply")]


@pytest.mark.parametrize("kind,fmt,ext", CASES)
def test_synthetic_bend_lines_within_half_mm_and_half_degree(tmp_path, kind, fmt, ext):
    part = make_part(kind, seed=11)
    scan = write_part(part, tmp_path / f"{kind}{ext}", as_point_cloud=(fmt == "cloud"),
                      with_normals=(fmt != "cloud"))
    manifest = run_prep(scan, tmp_path / "prep")
    assert manifest["coordinate_frame"] == "source_ply_raw"
    assert manifest["thickness_mm_estimated"] == pytest.approx(part.thickness, abs=0.05)
    for view in REQUIRED_VIEWS:
        assert (tmp_path / "prep" / "render" / f"patches_{view}.png").stat().st_size > 20_000
    assert (tmp_path / "prep" / "render" / "patches_contact_sheet.png").is_file()

    pairs = _outer_pairs(part, tmp_path / "prep")
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps({"bends": [{"patches": p, "note": "test"} for p in pairs]}))
    doc = run_lines(tmp_path / "prep", pairs_path, tmp_path / "lines",
                    thickness_mm=part.thickness, ri_mm=part.ri)
    assert doc["n_failed"] == 0 and doc["n_ok"] == len(part.bends)
    for got, truth in zip(doc["bends"], part.bends):
        skin_truth = truth["skins"][truth["outer_sign"]]
        for locus, line in (("apex", got["apex"]), ("sharp", got["sharp"])):
            perp, angle, span = _line_errors(line["start"], line["end"], skin_truth[locus])
            assert perp < POS_TOL_MM, (kind, got["bend_id"], locus, perp)
            assert angle < ANG_TOL_DEG, (kind, got["bend_id"], locus, angle)
            assert span[0] == pytest.approx(0.0, abs=1.5) and span[1] == pytest.approx(part.width, abs=1.5)
        assert got["label_line"] == got["apex"]
        assert got["included_angle_deg"] == pytest.approx(truth["included_deg"], abs=ANG_TOL_DEG)
        assert got["skin"] == "outer"
        assert got["confidence"] == "certain", got["confidence_reasons"]
        assert (tmp_path / "lines" / "render" / f"closeup_{got['bend_id']}.png").stat().st_size > 10_000
    for view in REQUIRED_VIEWS:
        assert (tmp_path / "lines" / "render" / f"verify_{view}.png").stat().st_size > 20_000


def test_sharp_locus_and_mixed_skin_flag(tmp_path):
    part = make_part("L", seed=5)
    scan = write_part(part, tmp_path / "L.ply")
    run_prep(scan, tmp_path / "prep")
    (pa, pb), = _outer_pairs(part, tmp_path / "prep")
    patches = {p["id"]: p for p in json.loads((tmp_path / "prep" / "patches.json").read_text())["patches"]}
    mixed = [pa, patches[pb]["twin"]]
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([[pa, pb], mixed]))
    doc = run_lines(tmp_path / "prep", pairs_path, tmp_path / "lines", locus="sharp",
                    thickness_mm=part.thickness, ri_mm=part.ri)
    good, bad = doc["bends"]
    assert good["label_line"] == good["sharp"]
    perp, _, _ = _line_errors(good["sharp"]["start"], good["sharp"]["end"],
                              part.bends[0]["skins"][part.bends[0]["outer_sign"]]["sharp"])
    assert perp < POS_TOL_MM
    assert bad["skin"] == "mixed" and bad["confidence"] == "uncertain"


def _fake_project(tmp_path):
    root = tmp_path / "FullProject"
    (root / "data" / "bend_corpus" / "incoming" / "batch").mkdir(parents=True)
    return root


def test_write_label_contract_and_revision(tmp_path):
    part = make_part("U", seed=2)
    root = _fake_project(tmp_path)
    asset = Path(write_part(part, root / "data" / "bend_corpus" / "incoming" / "batch" / "u_part.ply"))
    run_prep(asset, tmp_path / "prep")
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps({"bends": [{"patches": p} for p in _outer_pairs(part, tmp_path / "prep")],
                                      "notes": "synthetic"}))
    run_lines(tmp_path / "prep", pairs_path, tmp_path / "lines", thickness_mm=part.thickness, ri_mm=part.ri)
    labels_root = tmp_path / "annotations" / "ai_bend_lines_v1"
    rc = cli_main(["write-label", "--lines", str(tmp_path / "lines"), "--asset-path", str(asset),
                   "--labels-root", str(labels_root), "--project-root", str(root)])
    assert rc == 0
    from backend.bend_training_labels import asset_id_for

    asset_id = asset_id_for("data", "bend_corpus/incoming/batch/u_part.ply")
    label = json.loads((labels_root / f"{asset_id}.json").read_text())
    assert label["annotator"] == AI_ANNOTATOR and label["review_status"] == AI_REVIEW_STATUS
    assert label["kind"] == "sherman_qc_human_bend_labels" and label["schema_version"] == 1
    assert label["coordinate_frame"] == "source_ply_raw" and label["revision"] == 1
    assert label["source"] == "data" and label["source_relative_path"] == "bend_corpus/incoming/batch/u_part.ply"
    assert [b["bend_id"] for b in label["bends"]] == ["B1", "B2"]
    asset_meta = {"asset_id": asset_id, "source": "data", "relative_path": label["source_relative_path"]}
    validate_ai_label(label, asset_meta)
    assert (labels_root / "provenance" / f"{asset_id}.json").is_file()
    write_label(tmp_path / "lines", asset, labels_root=labels_root, project_root=root)
    assert json.loads((labels_root / f"{asset_id}.json").read_text())["revision"] == 2
    with pytest.raises(LabelWriteError):
        write_label(tmp_path / "lines", asset, labels_root=tmp_path / "human_bend_lines_v1", project_root=root)
    outside = Path(write_part(part, tmp_path / "outside.ply"))
    with pytest.raises(LabelWriteError):
        write_label(tmp_path / "lines", outside, labels_root=labels_root, project_root=root)


def test_pairs_parsing_and_custody():
    parsed = parse_pairs({"bends": [{"patches": [[3, 12], 7], "note": "n"}]})
    assert parsed["bends"][0]["sides"] == [[3, 12], [7]]
    with pytest.raises(ValueError):
        parse_pairs([[3, 3]])
    with pytest.raises(ValueError):
        parse_pairs({"bends": [[1, 2]], "no_bends_present": True})
    for bad in ("/x/HOLDOUT10/a.ply", "/x/sealed_truth/a.ply", "/x/N007.ply", "/x/EHO4_part.ply",
                str(Path.home() / "rhino-preview" / "face.ply"), "/x/node_modules/a.ply"):
        with pytest.raises(CustodyError):
            check_path(bad)
    assert check_path("/x/48333006_01.ply").name == "48333006_01.ply"
