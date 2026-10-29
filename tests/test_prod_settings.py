"""C04 settings: env parsing, data-root custody, stations, layout and secrets."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from apps.production_line.settings import (
    DATA_ROOT_DIRS,
    LinePaths,
    SettingsError,
    StationConfig,
    ensure_layout,
    get_secret,
    load_settings,
)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    for d in ("data", "output", "backend"):
        (r / d).mkdir(parents=True)
    return r


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "line-root")}
    env.update(extra)
    return env


def _code(env: dict[str, str], repo: Path) -> str:
    with pytest.raises(SettingsError) as exc:
        load_settings(env, repo_root=repo)
    return exc.value.code


def test_valid_minimal_settings_and_defaults(tmp_path: Path, repo: Path) -> None:
    s = load_settings(_env(tmp_path), repo_root=repo)
    root = (tmp_path / "line-root").resolve()
    assert s.enabled and s.paths.data_root == root
    assert s.paths.db_line == root / "db" / "line.sqlite3"
    assert s.paths.db_security == root / "db" / "security.sqlite3"
    assert s.engine_bundle_dir == root / "engine" / "current"
    assert s.program_status_path == root / "config" / "f1_program_status.json"
    assert s.job_timeout_s == 180 and s.postprocess_timeout_s == 300
    assert s.rss_limit_mb == 32768 and s.rss_limit_bytes == 32768 * 1024 * 1024  # ENGINE_IO-A5
    assert s.release_roles == frozenset({"admin"})
    assert s.allow_self_register is False and s.production_hardening is False
    assert s.legacy_live_scan_enabled is False
    assert s.log_retention_days == 30 and s.filename_regex is None and s.stations == ()


@pytest.mark.parametrize("flag", ["", "0", "false", "off"])
def test_line_disabled(tmp_path: Path, repo: Path, flag: str) -> None:
    env = _env(tmp_path, SHERMAN_LINE_ENABLED=flag)
    assert _code(env, repo) == "LINE_DISABLED"
    assert _code({}, repo) == "LINE_DISABLED"


def test_data_root_missing_and_relative(tmp_path: Path, repo: Path) -> None:
    assert _code({"SHERMAN_LINE_ENABLED": "1"}, repo) == "DATA_ROOT_MISSING"
    assert _code(_env(tmp_path, SHERMAN_LINE_DATA_ROOT="relative/root"), repo) == "DATA_ROOT_NOT_ABSOLUTE"


@pytest.mark.parametrize("sub", ["", "line", "data/line", "output", "output/x/y", "backend/run"])
def test_data_root_inside_repo_refused(tmp_path: Path, repo: Path, sub: str) -> None:
    root = repo / sub if sub else repo
    assert _code(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(root)), repo) == "DATA_ROOT_INSIDE_REPO"


def test_data_root_symlink_into_repo_refused(tmp_path: Path, repo: Path) -> None:
    link = tmp_path / "innocent-looking"
    link.symlink_to(repo / "data")
    assert _code(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(link / "line")), repo) == "DATA_ROOT_INSIDE_REPO"


def test_data_root_under_symlinked_research_dir_target_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo2"
    repo.mkdir()
    external = tmp_path / "external-volume"
    external.mkdir()
    (repo / "outputs").symlink_to(external)  # repo/outputs -> outside the repo
    env = _env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(external / "line"))
    assert _code(env, repo) == "DATA_ROOT_INSIDE_REPO"
    # A sibling of the external target is fine.
    ok = load_settings(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(tmp_path / "elsewhere")), repo_root=repo)
    assert ok.data_root.name == "elsewhere"


def test_data_root_containing_repo_refused(tmp_path: Path, repo: Path) -> None:
    assert _code(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(tmp_path)), repo) == "DATA_ROOT_INSIDE_REPO"


@pytest.mark.parametrize("value", ["40000", "32769", "abc", "0", "-5"])
def test_rss_limit_invalid(tmp_path: Path, repo: Path, value: str) -> None:
    assert _code(_env(tmp_path, SHERMAN_LINE_RSS_LIMIT_MB=value), repo) == "RSS_LIMIT_INVALID"


def test_rss_limit_upper_bound_accepted(tmp_path: Path, repo: Path) -> None:
    assert load_settings(_env(tmp_path, SHERMAN_LINE_RSS_LIMIT_MB="32768"), repo_root=repo).rss_limit_mb == 32768


def test_stations_json_parses_into_station_configs(tmp_path: Path, repo: Path) -> None:
    stations = [
        {"station_id": "cell-1", "inbox": str(tmp_path / "inbox1")},
        {
            "station_id": "cell-2",
            "inbox": str(tmp_path / "inbox2"),
            "sidecar_mode": "required",
            "source_policy": "copy_leave",
            "recursive": True,
        },
    ]
    s = load_settings(_env(tmp_path, SHERMAN_LINE_STATIONS=json.dumps(stations)), repo_root=repo)
    assert s.stations == (
        StationConfig("cell-1", tmp_path / "inbox1", "off", "move", False),
        StationConfig("cell-2", tmp_path / "inbox2", "required", "copy_leave", True),
    )
    assert all(isinstance(st, StationConfig) for st in s.stations)
    assert s.station("cell-2") is s.stations[1] and s.station("nope") is None
    assert s.station_ids == ("cell-1", "cell-2")


def test_single_station_shorthand(tmp_path: Path, repo: Path) -> None:
    env = _env(tmp_path, SHERMAN_LINE_STATION_ID="press-7", SHERMAN_LINE_INBOX=str(tmp_path / "in"))
    s = load_settings(env, repo_root=repo)
    assert s.stations == (StationConfig("press-7", tmp_path / "in"),)
    assert _code(_env(tmp_path, SHERMAN_LINE_STATION_ID="press-7"), repo) == "STATION_INVALID"


@pytest.mark.parametrize(
    "stations",
    [
        "not json",
        json.dumps({"station_id": "cell-1"}),
        json.dumps([{"station_id": "Cell_1", "inbox": "/tmp/x"}]),
        json.dumps([{"station_id": "x" * 33, "inbox": "/tmp/x"}]),
        json.dumps([{"station_id": "cell-1", "inbox": "relative"}]),
        json.dumps([{"station_id": "cell-1"}]),
        json.dumps([{"station_id": "cell-1", "inbox": "/tmp/x", "colour": "red"}]),
        json.dumps([{"station_id": "cell-1", "inbox": "/tmp/x", "sidecar_mode": "maybe"}]),
        json.dumps([{"station_id": "cell-1", "inbox": "/tmp/x", "source_policy": "delete"}]),
        json.dumps([{"station_id": "cell-1", "inbox": "/tmp/x", "recursive": "yes"}]),
        json.dumps([{"station_id": "cell-1", "inbox": "/tmp/a"}, {"station_id": "cell-1", "inbox": "/tmp/b"}]),
        json.dumps([{"station_id": "cell-1", "inbox": "/tmp/a"}, {"station_id": "cell-2", "inbox": "/tmp/a/b"}]),
    ],
)
def test_invalid_stations(tmp_path: Path, repo: Path, stations: str) -> None:
    assert _code(_env(tmp_path, SHERMAN_LINE_STATIONS=stations), repo) == "STATION_INVALID"


def test_station_inbox_in_repo_or_data_root_refused(tmp_path: Path, repo: Path) -> None:
    in_repo = json.dumps([{"station_id": "cell-1", "inbox": str(repo / "data" / "inbox")}])
    assert _code(_env(tmp_path, SHERMAN_LINE_STATIONS=in_repo), repo) == "STATION_INVALID"
    in_root = json.dumps([{"station_id": "cell-1", "inbox": str(tmp_path / "line-root" / "jobs")}])
    assert _code(_env(tmp_path, SHERMAN_LINE_STATIONS=in_root), repo) == "STATION_INVALID"


@pytest.mark.parametrize("regex", [r"^(?P<part>\d+)_", r"^(?P<sku", r"^\d+"])
def test_filename_regex_invalid(tmp_path: Path, repo: Path, regex: str) -> None:
    assert _code(_env(tmp_path, SHERMAN_LINE_FILENAME_REGEX=regex), repo) == "FILENAME_REGEX_INVALID"


def test_filename_regex_with_sku_group_accepted(tmp_path: Path, repo: Path) -> None:
    rx = r"^(?P<sku>[A-Z0-9]+)_(?P<serial>\d+)"
    assert load_settings(_env(tmp_path, SHERMAN_LINE_FILENAME_REGEX=rx), repo_root=repo).filename_regex == rx


@pytest.mark.parametrize(
    "key,value",
    [
        ("SHERMAN_RELEASE_ROLES", "viewer"),
        ("SHERMAN_RELEASE_ROLES", "admin,root"),
        ("SHERMAN_ALLOW_SELF_REGISTER", "maybe"),
        ("SHERMAN_LINE_JOB_TIMEOUT_S", "fast"),
        ("SHERMAN_LINE_JOB_TIMEOUT_S", "1"),
        ("SHERMAN_LINE_LOG_RETENTION_DAYS", "0"),
        ("SHERMAN_LINE_PYTHON", "python3"),
        ("SHERMAN_REPORT_FONT_PATH", "fonts/Heebo.ttf"),
    ],
)
def test_other_invalid_values(tmp_path: Path, repo: Path, key: str, value: str) -> None:
    assert _code(_env(tmp_path, **{key: value}), repo) == "VALUE_INVALID"


def test_bundle_dir_and_status_path_custody(tmp_path: Path, repo: Path) -> None:
    assert _code(_env(tmp_path, SHERMAN_LINE_ENGINE_BUNDLE_DIR=str(repo / "bundle")), repo) == "VALUE_INVALID"
    status_in_output = str(repo / "output" / "status.json")
    assert _code(_env(tmp_path, SHERMAN_LINE_PROGRAM_STATUS_PATH=status_in_output), repo) == "VALUE_INVALID"


def test_flags_and_roles_parse(tmp_path: Path, repo: Path) -> None:
    env = _env(
        tmp_path,
        SHERMAN_RELEASE_ROLES="admin, operator",
        SHERMAN_ALLOW_SELF_REGISTER="true",
        SHERMAN_PRODUCTION_HARDENING="1",
        SHERMAN_LEGACY_LIVE_SCAN="yes",
        SHERMAN_LINE_JOB_TIMEOUT_S="240",
    )
    s = load_settings(env, repo_root=repo)
    assert s.release_roles == frozenset({"admin", "operator"})
    assert s.allow_self_register and s.production_hardening and s.legacy_live_scan_enabled
    assert s.job_timeout_s == 240


def test_load_settings_creates_nothing(tmp_path: Path, repo: Path) -> None:
    load_settings(_env(tmp_path), repo_root=repo)
    assert not (tmp_path / "line-root").exists()


def test_ensure_layout_permissions(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    os.chmod(root, 0o755)
    paths = ensure_layout(LinePaths.for_root(root))
    for rel in ("",) + DATA_ROOT_DIRS:
        d = root / rel if rel else root
        assert d.is_dir(), d
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d
    assert paths.recipes_compiled == root / "recipes" / "compiled"


def test_ensure_layout_refuses_symlinked_dir(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (root / "jobs").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(SettingsError) as exc:
        ensure_layout(LinePaths.for_root(root))
    assert exc.value.code == "VALUE_INVALID"


def test_get_secret_created_once_and_private(tmp_path: Path) -> None:
    paths = ensure_layout(LinePaths.for_root(tmp_path / "root"))
    key = get_secret(paths, "stream_token")
    assert isinstance(key, bytes) and len(key) == 32
    assert get_secret(paths, "stream_token") == key
    assert get_secret(paths, "other") != key
    f = paths.secrets / "stream_token.key"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert not [p for p in paths.secrets.iterdir() if p.name.endswith(".tmp")]


def test_get_secret_rejects_corrupt_and_bad_names(tmp_path: Path) -> None:
    paths = ensure_layout(LinePaths.for_root(tmp_path / "root"))
    (paths.secrets / "short.key").write_bytes(b"x" * 5)
    with pytest.raises(SettingsError) as exc:
        get_secret(paths, "short")
    assert exc.value.code == "SECRET_INVALID"
    for bad in ("../escape", "Upper", ""):
        with pytest.raises(SettingsError):
            get_secret(paths, bad)


# --------------------------------------------------------------------------- app copies (amendment 10)


def _fake_app_copy(tmp_path: Path, source: Path, **manifest: object) -> Path:
    """An app-copy root whose manifest records ``source`` as its dev repo."""
    from apps.production_line import settings as st

    copy = tmp_path / "app-copy"
    copy.mkdir()
    body = {"schema": st.APP_MANIFEST_SCHEMA, "profile": "app", "source_repo_root": str(source)}
    body.update(manifest)
    (copy / st.APP_MANIFEST_NAME).write_text(json.dumps(body))
    return copy


def test_app_manifest_constants_match_engine_bundle() -> None:
    from apps.production_line import engine_bundle as eb
    from apps.production_line import settings as st

    assert (st.APP_MANIFEST_NAME, st.APP_MANIFEST_SCHEMA) == (eb.MANIFEST_NAME, eb.MANIFEST_SCHEMA)


@pytest.mark.parametrize("where", ["tmp/line-root", "output/line-root", "data/x", ""])
def test_app_copy_refuses_data_root_inside_the_dev_repo(tmp_path: Path, repo: Path, where: str) -> None:
    copy = _fake_app_copy(tmp_path, repo)
    root = repo / where if where else repo
    assert _code(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(root)), copy) == "DATA_ROOT_INSIDE_REPO"
    # Without the manifest (a plain checkout at that path) the same root is accepted,
    # proving the refusal comes from the recorded dev repo.
    (copy / "bundle_manifest.json").unlink()
    if where.startswith("tmp"):
        assert load_settings(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(root)), repo_root=copy).data_root == root


def test_app_copy_guards_dev_repo_inbox_bundle_and_symlinked_research_dir(tmp_path: Path, repo: Path) -> None:
    copy = _fake_app_copy(tmp_path, repo)
    inbox = json.dumps([{"station_id": "c1", "inbox": str(repo / "tmp" / "inbox")}])
    assert _code(_env(tmp_path, SHERMAN_LINE_STATIONS=inbox), copy) == "STATION_INVALID"
    assert _code(_env(tmp_path, SHERMAN_LINE_ENGINE_BUNDLE_DIR=str(repo / "eng")), copy) == "VALUE_INVALID"
    status = str(repo / "output" / "status.json")
    assert _code(_env(tmp_path, SHERMAN_LINE_PROGRAM_STATUS_PATH=status), copy) == "VALUE_INVALID"
    # The dev repo's research dir is a symlink: its target is protected as well.
    external = tmp_path / "external-scans"
    external.mkdir()
    (repo / "uploads").symlink_to(external)
    assert _code(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(external / "root")), copy) == "DATA_ROOT_INSIDE_REPO"
    # The data root containing the dev repo is refused too.
    assert _code(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(tmp_path)), copy) == "DATA_ROOT_INSIDE_REPO"
    ok = load_settings(_env(tmp_path, SHERMAN_LINE_DATA_ROOT=str(tmp_path / "elsewhere" / "root")), repo_root=copy)
    assert ok.repo_root == copy.resolve()


@pytest.mark.parametrize(
    "manifest",
    [
        {"source_repo_root": None},
        {"source_repo_root": "relative/repo"},
        {"schema": "something_else_v1"},
    ],
)
def test_app_copy_with_unverifiable_manifest_fails_closed(tmp_path: Path, repo: Path, manifest: dict) -> None:
    from apps.production_line.settings import forbidden_location_reason

    copy = _fake_app_copy(tmp_path, repo, **manifest)
    assert _code(_env(tmp_path), copy) == "VALUE_INVALID"
    assert forbidden_location_reason(tmp_path / "anywhere", copy) is not None


def test_app_copy_naming_itself_or_corrupt_manifest_fails_closed(tmp_path: Path, repo: Path) -> None:
    copy = _fake_app_copy(tmp_path, repo)
    (copy / "bundle_manifest.json").write_text(json.dumps({"schema": "sherman_engine_bundle_manifest_v1", "source_repo_root": str(copy)}))
    assert _code(_env(tmp_path), copy) == "VALUE_INVALID"
    (copy / "bundle_manifest.json").write_text("{not json")
    assert _code(_env(tmp_path), copy) == "VALUE_INVALID"


def test_get_secret_without_hard_link_support(tmp_path: Path, monkeypatch) -> None:
    import errno

    paths = ensure_layout(LinePaths.for_root(tmp_path / "root"))

    def no_link(*_a, **_k):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_link)
    key = get_secret(paths, "stream_token")
    assert len(key) == 32 and get_secret(paths, "stream_token") == key
    f = paths.secrets / "stream_token.key"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert not [p for p in paths.secrets.iterdir() if p.name.endswith(".tmp")]

    real_open = os.open

    def failing_open(path, flags, *a, **k):
        if os.fspath(path).endswith("broken.key"):
            raise OSError(errno.ENOTSUP, "Operation not supported")
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(os, "open", failing_open)
    with pytest.raises(SettingsError) as exc:
        get_secret(paths, "broken")
    assert exc.value.code == "SECRET_INVALID"
    assert not (paths.secrets / "broken.key").exists()
    assert not [p for p in paths.secrets.iterdir() if p.name.endswith(".tmp")]
