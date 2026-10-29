"""C04 engine bundle and app-copy builder, on a synthetic mini repo in tmp_path."""
from __future__ import annotations

import builtins
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from apps.production_line import engine_bundle as eb

DEPS = {"numpy": "1.0", "scipy": "2.0"}


def _w(root: Path, rel: str, text: str = "") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    # Engine seed tree with relative imports and a lazy (function-level) backend import.
    _w(r, "domains/__init__.py")
    _w(r, "domains/production/__init__.py")
    _w(r, "domains/production/engine/__init__.py")
    _w(r, "domains/production/engine/inspect/__init__.py")
    _w(
        r,
        "domains/production/engine/inspect/__main__.py",
        """
        from ..geometry import fit_plane

        def main():
            import bend_helper  # lazy, resolved against R/backend
            import output.y  # decoy: an import into output/ must never be followed
            return bend_helper.VALUE, fit_plane
        """,
    )
    _w(r, "domains/production/engine/geometry.py", "from . import util\n\ndef fit_plane():\n    return util.X\n")
    _w(r, "domains/production/engine/util.py", "X = 1\n")
    _w(r, "domains/production/engine/tests/test_engine.py", "assert False\n")
    _w(r, "domains/production/engine/test_inline.py", "assert False\n")
    # C06 entry with literal dynamic imports.
    _w(r, "domains/production/inspection/__init__.py")
    _w(
        r,
        "domains/production/inspection/postprocess_cli.py",
        """
        import importlib, subprocess, sys

        def run():
            importlib.import_module("domains.shared.lazy_mod")
            subprocess.run([sys.executable, "-m", "backend.spawned_worker"])
        """,
    )
    _w(r, "domains/shared/__init__.py")
    _w(r, "domains/shared/lazy_mod.py", "Y = 2\n")
    _w(r, "domains/production/reporting/__init__.py")
    _w(r, "domains/production/reporting/render_cli.py", "import contracts.production.models\n")
    (r / "domains/production/reporting/fonts").mkdir(parents=True)
    (r / "domains/production/reporting/fonts/Heebo-Regular.ttf").write_bytes(b"\x00\x01font")
    _w(r, "domains/production/reporting/fonts/OFL.txt", "SIL Open Font License\n")
    _w(r, "domains/production/reporting/fonts/notes.json", "{}\n")
    _w(r, "contracts/__init__.py")
    _w(r, "contracts/production/__init__.py")
    _w(r, "contracts/production/models.py", "from contracts.production import common\n")
    _w(r, "contracts/production/common.py", "Z = 3\n")
    # backend (on the runtime path as a top-level root, no __init__).
    _w(r, "backend/bend_helper.py", "VALUE = 42\n")
    _w(r, "backend/spawned_worker.py", "print('worker')\n")
    _w(r, "backend/unused_backend.py", "UNUSED = True\n")
    _w(r, "backend/legacy_stub.py", "import bend_helper\n")
    _w(r, "backend/tests/test_backend.py", "assert False\n")
    # V10 extractor chain in a namespace package.
    _w(r, "scripts/extract_step_topology_bend_axes_v10.py", "from scripts import extract_step_topology_bend_axes_v7 as legacy\n")
    _w(r, "scripts/extract_step_topology_bend_axes_v7.py", "import numpy as np\n")
    _w(r, "scripts/unrelated.py", "print('x')\n")
    # App entries.
    _w(r, "apps/__init__.py")
    _w(r, "apps/api/__init__.py")
    _w(r, "apps/api/main.py", "import legacy_stub\n")
    # Frontend dist.
    _w(r, "frontend/dist/index.html", "<!doctype html><title>x</title>\n")
    _w(r, "frontend/dist/assets/app.js", "console.log(1)\n")
    # Custody decoys.
    _w(r, "data/x.json", '{"secret": true}\n')
    _w(r, "output/y.py", "raise SystemExit('decoy executed')\n")
    _w(r, "output/improvement_loop/best_config.latest.json", "{}\n")
    _w(r, "research/jev/labels.json", "[]\n")
    _w(r, "uploads/parts_cad/a.py", "X = 1\n")
    return r


ENGINE_EXPECTED = {
    "domains/__init__.py",
    "domains/production/__init__.py",
    "domains/production/engine/__init__.py",
    "domains/production/engine/inspect/__init__.py",
    "domains/production/engine/inspect/__main__.py",
    "domains/production/engine/geometry.py",
    "domains/production/engine/util.py",
    "domains/production/inspection/__init__.py",
    "domains/production/inspection/postprocess_cli.py",
    "domains/shared/__init__.py",
    "domains/shared/lazy_mod.py",
    "domains/production/reporting/__init__.py",
    "domains/production/reporting/render_cli.py",
    "domains/production/reporting/fonts/Heebo-Regular.ttf",
    "domains/production/reporting/fonts/OFL.txt",
    "contracts/__init__.py",
    "contracts/production/__init__.py",
    "contracts/production/models.py",
    "contracts/production/common.py",
    "backend/bend_helper.py",
    "backend/spawned_worker.py",
    "scripts/extract_step_topology_bend_axes_v10.py",
    "scripts/extract_step_topology_bend_axes_v7.py",
}


def _build(repo: Path, dest: Path, **kw) -> eb.BuildResult:
    kw.setdefault("deps", DEPS)
    return eb.build_bundle(repo, dest, **kw)


def _make_writable(p: Path) -> None:
    os.chmod(p, 0o644)


def test_engine_bundle_closure_and_exclusions(repo: Path, tmp_path: Path) -> None:
    res = _build(repo, tmp_path / "bundle")
    assert set(res.files) == ENGINE_EXPECTED
    for rel in ENGINE_EXPECTED:
        assert (res.dest / rel).is_file(), rel
    for absent in (
        "data/x.json",
        "output/y.py",
        "output/improvement_loop/best_config.latest.json",
        "research/jev/labels.json",
        "uploads/parts_cad/a.py",
        "backend/unused_backend.py",
        "backend/tests/test_backend.py",
        "domains/production/engine/tests/test_engine.py",
        "domains/production/engine/test_inline.py",
        "domains/production/reporting/fonts/notes.json",
        "scripts/unrelated.py",
        "apps/api/main.py",
    ):
        assert not (res.dest / absent).exists(), absent
    assert not (res.dest / "data").exists() and not (res.dest / "output").exists()
    m = eb.read_manifest(res.dest)
    assert m["schema"] == eb.MANIFEST_SCHEMA and m["profile"] == "engine"
    assert m["bundle_id"] == res.bundle_id and len(res.bundle_id) == 16
    assert m["engine_config"] is None and m["dependency_versions"] == DEPS
    assert m["git_sha"] is None and m["git_dirty_count"] is None  # not a git repo
    assert "scripts.extract_step_topology_bend_axes_v10" in m["extra_modules"]
    assert any(w.startswith("ENTRY_MISSING") for w in m["warnings"]) is False
    assert eb.verify(res.dest) == []
    # Copies are read-only.
    assert ((res.dest / "backend/bend_helper.py").stat().st_mode & 0o222) == 0


def test_builder_never_touches_research_dirs(repo: Path, tmp_path: Path, monkeypatch) -> None:
    forbidden = [str(repo / d) for d in ("data", "output", "outputs", "uploads", "artifacts", "live_scans")]
    touched: list[str] = []

    def guard(path) -> None:
        try:
            p = os.fspath(path)
        except TypeError:
            return
        if isinstance(p, bytes):
            p = p.decode()
        real = os.path.abspath(p)
        if any(real == f or real.startswith(f + os.sep) for f in forbidden):
            touched.append(real)

    for mod, name in ((os, "stat"), (os, "lstat"), (os, "scandir"), (os, "listdir"), (builtins, "open"), (io, "open")):
        orig = getattr(mod, name)

        def wrapper(path=None, *a, __orig=orig, **k):
            if path is not None:
                guard(path)
            return __orig(path, *a, **k) if path is not None else __orig(*a, **k)

        monkeypatch.setattr(mod, name, wrapper)
    _build(repo, tmp_path / "b-engine")
    _build(repo, tmp_path / "b-app", profile="app", frontend="copy")
    monkeypatch.undo()
    # realpath() of the repo's research directory entries (symlink check) is allowed;
    # nothing *inside* them may be stat-ed, listed or opened.
    inside = [t for t in touched if any(t.startswith(f + os.sep) for f in forbidden)]
    assert inside == []


def test_verify_detects_modified_missing_and_unexpected(repo: Path, tmp_path: Path) -> None:
    res = _build(repo, tmp_path / "bundle")
    target = res.dest / "backend/bend_helper.py"
    _make_writable(target)
    target.write_text("VALUE = 43\n")
    assert eb.verify(res.dest) == ["MODIFIED backend/bend_helper.py"]
    (res.dest / "domains/shared/lazy_mod.py").unlink()
    (res.dest / "domains/production/engine/extra.py").write_text("x = 1\n")
    problems = eb.verify(res.dest)
    assert "MISSING domains/shared/lazy_mod.py" in problems
    assert "UNEXPECTED domains/production/engine/extra.py" in problems
    # Engines run with PYTHONDONTWRITEBYTECODE, so a bytecode cache is planted code.
    cache = res.dest / "backend" / "__pycache__"
    cache.mkdir()
    (cache / "bend_helper.cpython-311.pyc").write_bytes(b"x")
    assert "UNEXPECTED backend/__pycache__/" in eb.verify(res.dest)


def test_verify_detects_manifest_tampering_and_config(repo: Path, tmp_path: Path) -> None:
    res = _build(repo, tmp_path / "bundle")
    mpath = res.dest / eb.MANIFEST_NAME
    m = json.loads(mpath.read_text())
    m["files"]["backend/bend_helper.py"] = eb._sha256_file(res.dest / "backend/bend_helper.py")  # unchanged
    m["files"].pop("domains/shared/lazy_mod.py")
    _make_writable(mpath)
    mpath.write_text(json.dumps(m))
    problems = eb.verify(res.dest)
    assert "BUNDLE_ID_MISMATCH" in problems and "UNEXPECTED domains/shared/lazy_mod.py" in problems
    (res.dest / eb.NO_CONFIG_REL).write_text("")
    assert f"UNEXPECTED {eb.NO_CONFIG_REL}" in eb.verify(res.dest)
    mpath.unlink()
    assert eb.verify(res.dest) == ["MANIFEST_MISSING"]


def test_bundle_id_is_deterministic(repo: Path, tmp_path: Path) -> None:
    a = _build(repo, tmp_path / "a")
    b = _build(repo, tmp_path / "b")
    assert a.bundle_id == b.bundle_id and a.files == b.files
    assert _build(repo, tmp_path / "c", deps={"numpy": "9.9"}).bundle_id != a.bundle_id
    (repo / "domains/production/engine/util.py").write_text("X = 2\n")
    assert _build(repo, tmp_path / "d").bundle_id != a.bundle_id


def test_engine_config_pinning(repo: Path, tmp_path: Path) -> None:
    cfg = _w(tmp_path, "cfg/runtime.json", '{"runtime_config": {"min_fitness": 0.6}}\n')
    res = _build(repo, tmp_path / "bundle", engine_config=cfg)
    m = eb.read_manifest(res.dest)
    assert m["engine_config"]["relpath"] == eb.ENGINE_CONFIG_REL
    assert (res.dest / eb.ENGINE_CONFIG_REL).read_bytes() == cfg.read_bytes()
    assert res.bundle_id != _build(repo, tmp_path / "noconf").bundle_id
    assert eb.verify(res.dest) == []
    env = eb.engine_runtime_env(res.dest, {"PATH": "/usr/bin"}, program_status_path=tmp_path / "status.json")
    assert env["BEND_INSPECTION_CONFIG"] == str(res.dest / eb.ENGINE_CONFIG_REL)
    assert env["PYTHONPATH"] == f"{res.dest}{os.pathsep}{res.dest / 'backend'}"
    assert env["SHERMAN_ENGINE_BUNDLE_ID"] == res.bundle_id
    assert env["SHERMAN_QC_F1_PROGRAM_STATUS_PATH"] == str(tmp_path / "status.json")
    noconf_env = eb.engine_runtime_env(tmp_path / "noconf", {})
    assert noconf_env["BEND_INSPECTION_CONFIG"] == str(tmp_path / "noconf" / eb.NO_CONFIG_REL)
    assert not Path(noconf_env["BEND_INSPECTION_CONFIG"]).exists()
    target = res.dest / eb.ENGINE_CONFIG_REL
    _make_writable(target)
    target.write_text("{}")
    assert eb.verify(res.dest) == ["CONFIG_MODIFIED"]


def test_engine_config_refusals(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "b1", engine_config=repo / "output/improvement_loop/best_config.latest.json")
    assert exc.value.code == "ENGINE_CONFIG_FORBIDDEN"
    bad = _w(tmp_path, "cfg/list.json", "[1, 2]\n")
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "b2", engine_config=bad)
    assert exc.value.code == "ENGINE_CONFIG_INVALID"
    assert not (tmp_path / "b1").exists() and not (tmp_path / "b2").exists()


def test_dest_rules_and_force(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, repo / "build" / "bundle")
    assert exc.value.code == "DEST_INSIDE_REPO"
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path)  # contains the repo
    assert exc.value.code == "DEST_INSIDE_REPO"
    dest = tmp_path / "bundle"
    first = _build(repo, dest)
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, dest)
    assert exc.value.code == "DEST_NOT_EMPTY"
    (repo / "domains/shared/lazy_mod.py").write_text("Y = 3\n")
    second = _build(repo, dest, force=True)
    assert second.bundle_id != first.bundle_id and eb.verify(dest) == []
    assert not [p for p in tmp_path.iterdir() if ".staging-" in p.name or ".old-" in p.name]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _build(repo, empty).dest == empty.resolve()


def test_missing_entries_warn_and_extra_modules(repo: Path, tmp_path: Path) -> None:
    res = _build(
        repo,
        tmp_path / "bundle",
        entries=["domains/production/inspection/postprocess_cli.py", "domains/production/nope.py"],
        extra_modules=["backend.unused_backend", "no.such.module"],
        include_legacy_worker=True,
    )
    assert "ENTRY_MISSING domains/production/nope.py" in res.warnings
    assert "ENTRY_MISSING backend/bend_analysis_worker.py" in res.warnings
    assert "EXTRA_MODULE_MISSING no.such.module" in res.warnings
    assert "backend/unused_backend.py" in res.files
    assert "domains/production/reporting/render_cli.py" not in res.files  # default entry replaced


def test_resolver_prefers_a_regular_module_over_an_earlier_namespace_dir(repo: Path) -> None:
    """Like importlib's PathFinder: ``R/config/`` (no ``__init__``) on the first root
    must not shadow ``R/backend/config.py`` on the second (as in a real app copy)."""
    _w(repo, "config/bend_runtime_config.json", "{}\n")
    _w(repo, "backend/config.py", "import cfg_dep\n")
    _w(repo, "backend/cfg_dep.py", "D = 1\n")
    _w(repo, "apps/entry.py", "import config\nimport nsonly.leaf\n")
    _w(repo, "nsonly/leaf.py", "L = 1\n")
    resolver = eb._Resolver(repo)
    assert resolver.resolve("config") == ("backend/config.py", ("backend", "config"))
    # A bare directory with no regular module anywhere is still a namespace package.
    assert resolver.resolve("nsonly") == ("", ("nsonly",))
    files, warnings = eb.compute_closure(repo, ["apps/entry.py"])
    assert {"backend/config.py", "backend/cfg_dep.py", "nsonly/leaf.py"} <= set(files)
    assert not any("config" in w for w in warnings), warnings
    # A regular package on the first root still wins over a module on the second.
    _w(repo, "config/__init__.py", "")
    assert eb._Resolver(repo).resolve("config") == ("config/__init__.py", ("config",))


def test_unparseable_source_in_closure_refused(repo: Path, tmp_path: Path) -> None:
    _w(repo, "domains/production/engine/broken.py", "def (:\n")
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "bundle")
    assert exc.value.code == "SOURCE_UNPARSEABLE" and "broken.py" in exc.value.message
    assert not (tmp_path / "bundle").exists()


def test_symlink_out_of_repo_not_copied(repo: Path, tmp_path: Path) -> None:
    outside = _w(tmp_path, "outside/evil.py", "EVIL = 1\n")
    (repo / "domains/production/engine/evil.py").symlink_to(outside)
    res = _build(repo, tmp_path / "bundle")
    assert "domains/production/engine/evil.py" not in res.files
    assert "SYMLINK_OUTSIDE_REPO domains/production/engine/evil.py" in res.warnings


def test_app_profile_is_a_runnable_copy(repo: Path, tmp_path: Path) -> None:
    res = _build(repo, tmp_path / "app", profile="app", frontend="copy")
    files = set(res.files)
    assert ENGINE_EXPECTED <= files
    for rel in (
        "apps/api/main.py",
        "backend/legacy_stub.py",
        "backend/unused_backend.py",  # whole backend tree: workers launched with -m
        "scripts/unrelated.py",
        "frontend/dist/index.html",
        "frontend/dist/assets/app.js",
    ):
        assert rel in files, rel
    for absent in ("data/x.json", "output/y.py", "research/jev/labels.json", "backend/tests/test_backend.py"):
        assert absent not in files and not (res.dest / absent).exists()
    assert "ENTRY_MISSING apps/production_line/server.py" in res.warnings
    assert eb.read_manifest(res.dest)["profile"] == "app"
    assert eb.verify(res.dest) == []
    # Legacy state the running copy creates for itself does not break verification...
    _w(res.dest, "data/users.db", "x")
    _w(res.dest, "uploads/parts_cad/p.step", "x")
    _w(res.dest, "backend/server.log", "x")
    assert eb.verify(res.dest) == []
    # ...but injected code and frontend tampering do.
    _w(res.dest, "apps/injected.py", "x = 1\n")
    js = res.dest / "frontend/dist/assets/app.js"
    _make_writable(js)
    js.write_text("alert(1)\n")
    assert set(eb.verify(res.dest)) == {"UNEXPECTED apps/injected.py", "MODIFIED frontend/dist/assets/app.js"}
    env = eb.app_runtime_env(res.dest, {})
    assert env["PYTHONPATH"].split(os.pathsep) == [str(res.dest), str(res.dest / "backend")]
    # A copy must never write bytecode: verify treats any __pycache__ as planted code.
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_app_profile_frontend_modes(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "a1", profile="app", frontend="copy", frontend_dist=tmp_path / "nodist")
    assert exc.value.code == "FRONTEND_DIST_MISSING"
    res = _build(repo, tmp_path / "a2", profile="app", frontend="auto", frontend_dist=tmp_path / "nodist")
    assert not any(r.startswith("frontend/") for r in res.files)
    assert any(w.startswith("FRONTEND_DIST_MISSING") for w in res.warnings)
    alt = tmp_path / "alt_dist"
    _w(alt, "index.html", "<p>alt</p>")
    res = _build(repo, tmp_path / "a3", profile="app", frontend="copy", frontend_dist=alt)
    assert (res.dest / "frontend/dist/index.html").read_text() == "<p>alt</p>"
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "a6", profile="app", frontend="copy", frontend_dist=repo / "output")
    assert exc.value.code == "FRONTEND_DIST_FORBIDDEN"
    res = _build(repo, tmp_path / "a4", profile="app", frontend="none")
    assert not any(r.startswith("frontend/") for r in res.files)
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "a5", profile="app", frontend="build")  # no vite in the mini repo
    assert exc.value.code == "FRONTEND_BUILD_FAILED"
    assert not (tmp_path / "a5").exists()


def test_cli_build_shorthand_and_verify(repo: Path, tmp_path: Path, capsys) -> None:
    dest = tmp_path / "cli-app"
    assert eb.main(["--profile", "app", "--out", str(dest), "--repo-root", str(repo), "--frontend", "copy"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["profile"] == "app" and Path(out["dest"]) == dest.resolve()
    assert eb.main(["verify", "--dest", str(dest)]) == 0
    assert json.loads(capsys.readouterr().out)["mismatches"] == []
    edit = dest / "apps/api/main.py"
    _make_writable(edit)
    edit.write_text("import os\n")
    assert eb.main(["verify", "--dest", str(dest)]) == 1
    assert json.loads(capsys.readouterr().out)["mismatches"] == ["MODIFIED apps/api/main.py"]
    assert eb.main(["build", "--repo-root", str(repo), "--dest", str(repo / "inside")]) == 2
    assert "DEST_INSIDE_REPO" in capsys.readouterr().err
    engine = tmp_path / "cli-engine"
    assert eb.main(["build", "--repo-root", str(repo), "--dest", str(engine), "--extra-module", "backend.unused_backend"]) == 0
    assert "backend/unused_backend.py" in eb.read_manifest(engine)["files"]


# --------------------------------------------------------------------------- review fixes


def test_force_never_replaces_a_non_bundle_directory(repo: Path, tmp_path: Path) -> None:
    data_root = tmp_path / "line-root"
    _w(data_root, "db/line.sqlite3", "precious")
    _w(data_root, "secrets/stream_token.key", "k" * 32)
    for kw in ({}, {"profile": "app", "frontend": "copy"}):
        with pytest.raises(eb.BundleError) as exc:
            _build(repo, data_root, force=True, **kw)
        assert exc.value.code == "DEST_NOT_A_BUNDLE"
    assert (data_root / "db/line.sqlite3").read_text() == "precious"
    assert (data_root / "secrets/stream_token.key").exists()
    # A manifest of another schema (or an unreadable one) does not count as a bundle.
    _w(data_root, eb.MANIFEST_NAME, json.dumps({"schema": "other_v1"}))
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, data_root, force=True)
    assert exc.value.code == "DEST_NOT_A_BUNDLE"
    assert eb.main(["build", "--repo-root", str(repo), "--out", str(data_root), "--force"]) == 2
    assert (data_root / "db/line.sqlite3").read_text() == "precious"
    assert not [p for p in tmp_path.iterdir() if ".staging-" in p.name or ".old-" in p.name]


def test_manifest_records_source_repo_and_copies_inherit_it(repo: Path, tmp_path: Path) -> None:
    app = _build(repo, tmp_path / "app", profile="app", frontend="copy")
    assert app.manifest["source_repo_root"] == str(repo.resolve())
    assert eb.recorded_source_repo(app.dest) == repo.resolve()
    assert _build(repo, tmp_path / "eng").manifest["source_repo_root"] == str(repo.resolve())
    # A bundle built *from* the copy still names the original dev repo, and refuses
    # destinations inside it.
    nested = _build(app.dest, tmp_path / "nested", profile="app", frontend="copy")
    assert nested.manifest["source_repo_root"] == str(repo.resolve())
    with pytest.raises(eb.BundleError) as exc:
        _build(app.dest, repo / "inside-dev-repo")
    assert exc.value.code == "DEST_INSIDE_REPO"
    with pytest.raises(eb.BundleError) as exc:
        _build(app.dest, tmp_path / "x", engine_config=repo / "output/improvement_loop/best_config.latest.json")
    assert exc.value.code == "ENGINE_CONFIG_FORBIDDEN"
    assert eb.verify(app.dest) == []


def test_app_profile_defaults_to_building_the_frontend(repo: Path, tmp_path: Path, capsys) -> None:
    assert eb.DEFAULT_FRONTEND_MODE["app"] == "build"
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "a", profile="app")  # no vite in the mini repo: refused, not copied
    assert exc.value.code == "FRONTEND_BUILD_FAILED"
    assert eb.main(["--profile", "app", "--out", str(tmp_path / "b"), "--repo-root", str(repo)]) == 2
    assert "FRONTEND_BUILD_FAILED" in capsys.readouterr().err
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    # The engine profile never touches the frontend.
    assert not any(r.startswith("frontend/") for r in _build(repo, tmp_path / "e").files)


def test_copied_frontend_dist_staleness(repo: Path, tmp_path: Path) -> None:
    src = _w(repo, "frontend/react/src/App.tsx", "export const A = 1\n")
    _w(repo, "frontend/react/package.json", "{}\n")
    old = time.time() - 3600
    os.utime(repo / "frontend/dist/index.html", (old, old))
    res = _build(repo, tmp_path / "a", profile="app", frontend="copy")
    assert any(w.startswith("FRONTEND_DIST_STALE") for w in res.warnings)
    fe = res.manifest["frontend"]
    assert fe["mode"] == "copy" and fe["stale"] is True and fe["included"] is True
    assert isinstance(fe["source_sha256"], str) and len(fe["source_sha256"]) == 64
    with pytest.raises(eb.BundleError) as exc:
        _build(repo, tmp_path / "b", profile="app", frontend="copy", strict_frontend=True)
    assert exc.value.code == "FRONTEND_DIST_STALE"
    assert eb.main(["--profile", "app", "--out", str(tmp_path / "c"), "--repo-root", str(repo),
                    "--frontend", "auto", "--strict-frontend"]) == 2
    # A dist newer than every source file is not stale; the source hash tracks edits.
    os.utime(repo / "frontend/dist/index.html", None)
    os.utime(src, (old, old))
    os.utime(repo / "frontend/react/package.json", (old, old))
    fresh = _build(repo, tmp_path / "d", profile="app", frontend="copy", strict_frontend=True)
    assert fresh.manifest["frontend"]["stale"] is False
    assert fresh.manifest["frontend"]["source_sha256"] == fe["source_sha256"]
    src.write_text("export const A = 2\n")
    os.utime(src, (old, old))
    assert _build(repo, tmp_path / "e", profile="app", frontend="copy").manifest["frontend"]["source_sha256"] != fe["source_sha256"]


def test_verify_reports_native_extensions_and_sourceless_bytecode(repo: Path, tmp_path: Path) -> None:
    app = _build(repo, tmp_path / "app", profile="app", frontend="copy")
    assert eb.verify(app.dest) == []
    # A __pycache__ in an app copy is reported: a timestamp .pyc whose header matches the
    # verified source's mtime/size would run instead of it (copies run without bytecode).
    cache = app.dest / "backend" / "__pycache__"
    cache.mkdir()
    (cache / "legacy_stub.cpython-311.pyc").write_bytes(b"x")
    nested = app.dest / "apps" / "api" / "__pycache__"
    nested.mkdir()
    assert set(eb.verify(app.dest)) == {"UNEXPECTED backend/__pycache__/", "UNEXPECTED apps/api/__pycache__/"}
    shutil.rmtree(cache)
    shutil.rmtree(nested)
    # Legacy runtime state the copy creates for itself is still ignored.
    _w(app.dest, "data/users.db", "x")
    _w(app.dest, "output/run/x.json", "{}")
    assert eb.verify(app.dest) == []
    # An extension module next to a source, or a sourceless .pyc, is reported.
    (app.dest / "backend" / "bend_helper.cpython-311-darwin.so").write_bytes(b"\x7fELF")
    (app.dest / "apps" / "api" / "main.pyc").write_bytes(b"x")
    (app.dest / "domains" / "libx.dylib").write_bytes(b"x")
    (app.dest / "domains" / "liby.so.1").write_bytes(b"x")
    assert set(eb.verify(app.dest)) == {
        "UNEXPECTED backend/bend_helper.cpython-311-darwin.so",
        "UNEXPECTED apps/api/main.pyc",
        "UNEXPECTED domains/libx.dylib",
        "UNEXPECTED domains/liby.so.1",
    }
    eng = _build(repo, tmp_path / "eng")
    (eng.dest / "backend" / "bend_helper.pyc").write_bytes(b"x")
    (eng.dest / "backend" / "x.pyd").write_bytes(b"x")
    assert set(eb.verify(eng.dest)) == {"UNEXPECTED backend/bend_helper.pyc", "UNEXPECTED backend/x.pyd"}


# ---- integration: the real C04 settings/bootstrap modules running *from* an app copy

_REAL_REPO = Path(__file__).resolve().parents[1]
_C04_RUNTIME_FILES = (
    "apps/__init__.py",
    "apps/production_line/__init__.py",
    "apps/production_line/settings.py",
    "apps/production_line/health.py",
    "apps/production_line/security/__init__.py",
    "apps/production_line/security/principals.py",
    "apps/production_line/security/hardening.py",
    "apps/production_line/security/bootstrap_admin.py",
    "backend/auth.py",
)


@pytest.fixture()
def real_app_copy(repo: Path, tmp_path: Path) -> Path:
    for rel in _C04_RUNTIME_FILES:
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REAL_REPO / rel, dst)
    return _build(repo, tmp_path / "app-copy", profile="app", frontend="copy").dest


def _run_in_copy(copy: Path, code: str, *, stdin: str = "", extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = eb.app_runtime_env(copy, {"PATH": os.environ.get("PATH", ""), "HOME": str(copy.parent)})
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONSAFEPATH": "1"})
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(copy), env=env, input=stdin, capture_output=True, text=True, timeout=120
    )


def test_settings_from_an_app_copy_refuse_the_dev_repo(real_app_copy: Path, repo: Path, tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import json
        import apps.production_line.settings as st
        out = {"module": st.__file__, "repo_root": str(st.default_repo_root())}
        for label, root in json.loads(%r):
            env = {"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": root}
            try:
                s = st.load_settings(env)
                out[label] = "ACCEPTED"
            except st.SettingsError as exc:
                out[label] = exc.code
        inbox_env = {"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": %r,
                     "SHERMAN_LINE_STATION_ID": "c1", "SHERMAN_LINE_INBOX": %r}
        try:
            st.load_settings(inbox_env); out["inbox"] = "ACCEPTED"
        except st.SettingsError as exc:
            out["inbox"] = exc.code
        print(json.dumps(out))
        """
    ) % (
        json.dumps([
            ["dev_tmp", str(repo / "tmp" / "root")],
            ["dev_output", str(repo / "output" / "root")],
            ["outside", str(tmp_path / "srv" / "root")],
        ]),
        str(tmp_path / "srv" / "root"),
        str(repo / "tmp" / "inbox"),
    )
    proc = _run_in_copy(real_app_copy, code)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert Path(out["module"]).resolve().is_relative_to(real_app_copy.resolve())  # really running from the copy
    assert Path(out["repo_root"]).resolve() == real_app_copy.resolve()
    assert out["dev_tmp"] == "DATA_ROOT_INSIDE_REPO"
    assert out["dev_output"] == "DATA_ROOT_INSIDE_REPO"
    assert out["outside"] == "ACCEPTED"
    assert out["inbox"] == "STATION_INVALID"


def test_bootstrap_admin_from_an_app_copy_accepts_the_copys_users_db(real_app_copy: Path, repo: Path) -> None:
    password = "a long enough admin password"
    copy_db = real_app_copy / "data" / "users.db"
    run = (
        "import sys\n"
        "from apps.production_line.security import bootstrap_admin as b\n"
        "raise SystemExit(b.main(sys.argv[1:]))\n"
    )

    def bootstrap(db: Path) -> subprocess.CompletedProcess:
        code = run.replace("sys.argv[1:]", repr(["--users-db", str(db), "--username", "alice", "--password-stdin"]))
        return _run_in_copy(real_app_copy, code, stdin=password + "\n", extra_env={"PRODUCTION": ""})

    proc = bootstrap(copy_db)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "created" and password not in proc.stdout + proc.stderr
    assert copy_db.is_file()
    # The dev repo's data/users.db stays refused (exit 2) and is never created.
    dev_db = repo / "data" / "users.db"
    proc = bootstrap(dev_db)
    assert proc.returncode == 2 and "dev repo" in proc.stderr
    assert not dev_db.exists()
    # The default admin of the copy's DB is neutralised: hardening's probe agrees.
    probe = (
        "from apps.production_line.security.hardening import legacy_user_probe\n"
        f"print(legacy_user_probe({str(copy_db)!r})('admin', 'admin123'))\n"
    )
    proc = _run_in_copy(real_app_copy, probe)
    assert proc.returncode == 0 and proc.stdout.strip() == "False", proc.stderr
