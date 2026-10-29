import json
from pathlib import Path

from scripts.record_f1_v57_capability_not_opened_stop import (
    EXPECTED_CASE_IDS,
    STOP_STATUS,
    build_stop,
)


def test_capability_stop_is_total_and_never_invokes_detector(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cases = []
    scans = corpus / "scans"
    scans.mkdir()
    for case_id in EXPECTED_CASE_IDS:
        scan = scans / f"{case_id}.ply"
        scan.write_bytes(b"ply\n")
        cases.append({"case_id": case_id, "scan_path": str(scan)})
    (corpus / "manifest.json").write_text(
        json.dumps({"case_count": 30, "cases": cases}), encoding="utf-8"
    )
    generator = tmp_path / "generator.py"
    generator.write_text(
        "def build_case():\n    o3d.geometry.TriangleMesh()\n    o3d.io.write_triangle_mesh()\n",
        encoding="utf-8",
    )
    ruling = tmp_path / "ruling.md"
    ruling.write_text("OPEN_CAPABILITY\n", encoding="utf-8")

    payload = build_stop(
        corpus_root=corpus,
        generator_path=generator,
        ruling_path=ruling,
        metadata_read_disclosure=True,
    )
    assert payload["status"] == STOP_STATUS
    assert payload["missing_case_ids"] == list(EXPECTED_CASE_IDS)
    assert payload["detector_invocation_count"] == 0
    assert payload["source_free_fixture_invocation_count"] == 0
    assert payload["truth_opened_for_v57"] is False
    assert payload["v56_outputs_opened"] is False
    assert payload["operator_disclosure"]["generation_metadata_read_after_ruling"] is True


def test_candidate_raw_bundle_requires_manual_adjudication(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cases = []
    for case_id in EXPECTED_CASE_IDS:
        cases.append({"case_id": case_id})
    (corpus / "manifest.json").write_text(
        json.dumps({"case_count": 30, "cases": cases}), encoding="utf-8"
    )
    (corpus / "S001_depth.npy").write_bytes(b"not-a-real-array")
    generator = tmp_path / "generator.py"
    generator.write_text(
        "def build_case():\n    o3d.geometry.TriangleMesh()\n    o3d.io.write_triangle_mesh()\n",
        encoding="utf-8",
    )
    ruling = tmp_path / "ruling.md"
    ruling.write_text("OPEN_CAPABILITY\n", encoding="utf-8")

    try:
        build_stop(
            corpus_root=corpus,
            generator_path=generator,
            ruling_path=ruling,
            metadata_read_disclosure=False,
        )
    except ValueError as error:
        assert "manual custody adjudication" in str(error)
    else:
        raise AssertionError("candidate raw bundle must prevent an automatic missing-capability STOP")
