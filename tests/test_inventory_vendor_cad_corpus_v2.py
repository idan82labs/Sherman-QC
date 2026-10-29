"""Source-only fixtures; never access the established corpus."""
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "scripts/inventory_vendor_cad_corpus_v2.py"
spec = importlib.util.spec_from_file_location("inventory_v2", SOURCE)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def run(tmp_path, config=None):
    root = tmp_path / "synthetic"
    root.mkdir()
    attempt = tmp_path / "attempt"
    inventory.freeze(attempt, root, config)
    return root, attempt


def execute(attempt):
    return inventory.execute(attempt, inventory.digest(attempt / "FROZEN.json"))


def test_freeze_does_not_access_root(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "open_root", lambda *_: pytest.fail("raw access"))
    root = tmp_path / "nonexistent"
    inventory.freeze(tmp_path / "attempt", root)
    assert not root.exists()


def test_duplicates_unsupported_and_reserved(tmp_path):
    root, attempt = run(tmp_path)
    (root / "a.STEP").write_bytes(b"synthetic")
    (root / "b.stp").write_bytes(b"synthetic")
    (root / "unsupported.xyz").write_bytes(b"opaque")
    (root / "unsupported.dwg").write_bytes(b"synthetic native")
    (root / "link.step").symlink_to(root / "a.STEP")
    (root / "HOLDOUT10").mkdir()
    (root / "HOLDOUT10" / "secret.step").write_bytes(b"do not read")
    assert execute(attempt) == 0
    rows = [json.loads(line) for line in (attempt / "FILES.jsonl").read_text().splitlines()]
    assert {r["disposition"] for r in rows} == {"supported_format", "unsupported_format", "non_cad_metadata_only", "symlink_refused", "reserved_scope_excluded"}
    metadata = next(r for r in rows if r["relative_path"] == "unsupported.xyz")
    assert "sha256" not in metadata
    assert not any("secret" in r["relative_path"] for r in rows)
    groups = json.loads((attempt / "DUPLICATE_GROUPS.json").read_text())
    assert sorted(next(iter(groups.values()))) == ["a.STEP", "b.stp"]
    with pytest.raises(FileExistsError):
        execute(attempt)


@pytest.mark.parametrize("key,value", [("entries", 0), ("cad_files", 0), ("file_bytes", 1), ("total_bytes", 1), ("seconds", -1), ("rss_bytes", 1)])
def test_each_budget_preserves_terminal(tmp_path, key, value):
    config = {**inventory.CONFIG, key: value}
    root, attempt = run(tmp_path, config)
    (root / "a.step").write_bytes(b"synthetic")
    assert execute(attempt) == 1
    assert json.loads((attempt / "TERMINAL.json").read_text())["status"] == "INCOMPLETE_FAIL_CLOSED"
    assert (attempt / "FILES.jsonl").exists()


def test_wrong_pin_never_opens_root(tmp_path, monkeypatch):
    _, attempt = run(tmp_path)
    monkeypatch.setattr(inventory, "open_root", lambda *_: pytest.fail("raw access"))
    with pytest.raises(ValueError, match="freeze digest"):
        inventory.execute(attempt, "0" * 64)
    assert not (attempt / "EXECUTION.json").exists()


def test_symlink_root_refused(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    attempt = tmp_path / "attempt"
    inventory.freeze(attempt, link)
    assert execute(attempt) == 1


def test_attempt_inside_corpus_refused(tmp_path):
    with pytest.raises(ValueError):
        inventory.freeze(tmp_path / "root" / "attempt", tmp_path / "root")


def test_non_cad_files_never_opened_or_read(tmp_path, monkeypatch):
    root, attempt = run(tmp_path)
    for extension in ("csv", "json", "pdf", "xyz"):
        (root / f"support.{extension}").write_bytes(b"not authorized content")
    original_open = inventory.os.open

    def guarded_open(path, *args, **kwargs):
        assert not str(path).startswith("support."), "non-CAD content opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(inventory.os, "open", guarded_open)
    monkeypatch.setattr(inventory.os, "read", lambda *_: pytest.fail("non-CAD bytes read"))
    assert execute(attempt) == 0
    rows = [json.loads(line) for line in (attempt / "FILES.jsonl").read_text().splitlines()]
    assert all(r["disposition"] == "non_cad_metadata_only" and "sha256" not in r for r in rows)
    assert json.loads((attempt / "TERMINAL.json").read_text())["counters"]["total_bytes"] == 0


def test_ambiguous_mesh_and_scan_names_never_opened(tmp_path, monkeypatch):
    root, attempt = run(tmp_path)
    names = ["mesh.ply", "mesh.STL", "mesh.obj", "part_SCAN.step", "scan.iges", "SCAN.dwg", "scan_notes.pdf"]
    for name in names:
        (root / name).write_bytes(b"unauthorized bytes")
    (root / "SCAN_exports").mkdir()
    (root / "SCAN_exports" / "apparently_cad.step").write_bytes(b"no read")
    original_open = inventory.os.open

    def guarded_open(path, *args, **kwargs):
        assert str(path) not in names + ["SCAN_exports", "apparently_cad.step"]
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(inventory.os, "open", guarded_open)
    monkeypatch.setattr(inventory.os, "read", lambda *_: pytest.fail("mesh/scan bytes read"))
    assert execute(attempt) == 0
    rows = [json.loads(line) for line in (attempt / "FILES.jsonl").read_text().splitlines()]
    assert not any("sha256" in r for r in rows)
    assert {r["disposition"] for r in rows} == {"ambiguous_mesh_metadata_only", "scan_named_metadata_only", "scan_named_directory_metadata_only"}
    assert not any("apparently_cad" in r["relative_path"] for r in rows)
    assert json.loads((attempt / "DUPLICATE_GROUPS.json").read_text()) == {}


@pytest.mark.parametrize("extension", sorted(inventory.CAD_EXTENSIONS))
def test_explicit_cad_formats_are_hashed(tmp_path, extension):
    root, attempt = run(tmp_path)
    (root / ("part" + extension)).write_bytes(b"synthetic CAD bytes")
    assert execute(attempt) == 0
    row = json.loads((attempt / "FILES.jsonl").read_text())
    assert row["sha256"] and row["file_bytes"] == 19
