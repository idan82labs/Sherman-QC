import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_prediction_hidden_multiview_render_pack import (
    SCHEMA,
    build_render_pack,
    canonical_json_bytes,
    sha256_file,
    validate_opaque_manifest,
)


def _scan(path: Path, payload: bytes) -> dict[str, str]:
    path.write_bytes(payload)
    return {"scan_path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _manifest(path: Path, cases: list[dict[str, str]]) -> Path:
    path.write_bytes(
        canonical_json_bytes(
            {"schema": SCHEMA, "cohort_date": "2026-08-30", "cases": cases}
        )
    )
    return path


def _deterministic_stub(
    scan_path: Path,
    output_path: Path,
    *,
    max_points: int,
    display_name: str,
    write_frame: bool,
) -> Path:
    assert display_name.startswith("O")
    assert write_frame is False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        b"PNG-STUB\n" + display_name.encode() + b"\n" + scan_path.read_bytes()
    )
    return output_path


def test_pack_is_prediction_hidden_hash_locked_and_deterministic(tmp_path: Path) -> None:
    cases = [_scan(tmp_path / "native-secret-a.ply", b"scan-a"), _scan(tmp_path / "native-secret-b.ply", b"scan-b")]
    source = _manifest(tmp_path / "opaque.json", cases)

    first_manifest, first_sha = build_render_pack(
        source, tmp_path / "pack-a", expected_count=2, renderer=_deterministic_stub
    )
    second_manifest, second_sha = build_render_pack(
        source, tmp_path / "pack-b", expected_count=2, renderer=_deterministic_stub
    )

    assert first_sha == second_sha
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    payload = json.loads(first_manifest.read_text())
    assert [row["opaque_case_id"] for row in payload["cases"]] == ["O001", "O002"]
    assert payload["review_boundary"] == {
        "predictions_included": False,
        "sender_counts_included": False,
        "labels_included": False,
        "cad_included": False,
        "drawings_included": False,
        "source_filenames_rendered": False,
    }
    serialized = first_manifest.read_text()
    assert "native-secret" not in serialized
    assert sha256_file(first_manifest) == first_sha
    assert (tmp_path / "pack-a" / "render_pack_manifest.sha256").read_text() == (
        f"{first_sha}  render_pack_manifest.json\n"
    )
    for row in payload["cases"]:
        png = first_manifest.parent / row["png_path"]
        assert sha256_file(png) == row["png_sha256"]


@pytest.mark.parametrize("extra", ["prediction", "sender_count", "cad_path", "drawing_path"])
def test_manifest_rejects_nonopaque_case_metadata(tmp_path: Path, extra: str) -> None:
    case = _scan(tmp_path / "one.ply", b"one")
    case[extra] = "withheld"
    source = _manifest(tmp_path / "opaque.json", [case])
    with pytest.raises(ValueError, match="scan_path and sha256 only"):
        validate_opaque_manifest(source, expected_count=1)


def test_all_hashes_are_checked_before_first_render(tmp_path: Path) -> None:
    first = _scan(tmp_path / "one.ply", b"one")
    second = _scan(tmp_path / "two.ply", b"two")
    source = _manifest(tmp_path / "opaque.json", [first, second])
    (tmp_path / "two.ply").write_bytes(b"tampered")
    calls: list[Path] = []

    def renderer(*args, **kwargs):
        calls.append(args[0])
        return _deterministic_stub(*args, **kwargs)

    with pytest.raises(ValueError, match="scan hash drift"):
        build_render_pack(source, tmp_path / "pack", expected_count=2, renderer=renderer)
    assert calls == []
    assert not (tmp_path / "pack").exists()


def test_refuses_existing_output_namespace(tmp_path: Path) -> None:
    source = _manifest(tmp_path / "opaque.json", [_scan(tmp_path / "one.ply", b"one")])
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to mix or overwrite"):
        build_render_pack(source, output, expected_count=1, renderer=_deterministic_stub)
