import json
from pathlib import Path

from scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v1 import TOKEN, generate
from scripts.validate_f1_calibrated_raw_range_bundle_v1 import validate_bundle


def test_small_prospective_corpus_has_valid_opaque_bundles(tmp_path: Path):
    root = tmp_path / "corpus"
    report = generate(root, width=80, height=60, view_count=3, family_limit=1)
    assert report["case_count"] == 2
    assert report["family_count"] == 1
    assert report["confirmatory_eligible"] is False
    assert report["token"] == TOKEN
    for opaque_id in ("O001", "O002"):
        preflight = validate_bundle(root / "public" / opaque_id)
        assert preflight["status"] == "PASS"
        assert preflight["frame_count"] == 3
        manifest = json.loads((root / "public" / opaque_id / "manifest.json").read_text())
        serialized = json.dumps(manifest).lower()
        assert "family" not in serialized
        assert "truth" not in serialized
        assert "bend_count" not in serialized


def test_generation_is_byte_deterministic(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(first, width=64, height=48, view_count=3, family_limit=1)
    generate(second, width=64, height=48, view_count=3, family_limit=1)
    for opaque_id in ("O001", "O002"):
        a = json.loads((first / "public" / opaque_id / "manifest.json").read_text())
        b = json.loads((second / "public" / opaque_id / "manifest.json").read_text())
        # Bundle paths differ only in the corpus-level orchestration manifest;
        # semantic capture bytes and per-case manifests remain identical.
        assert a == b
        assert (first / "public" / opaque_id / "source.ply").read_bytes() == (
            second / "public" / opaque_id / "source.ply"
        ).read_bytes()
