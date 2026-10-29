"""Fake-child tests for the V9 execute waiter: outer-exit evidence only, never the benchmark.

Every test uses a temporary tree, a fake child command and an injected empty process table. The real
attempt directory, runner, raw data, Gold and caches are never touched.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE = [sys.executable, "-c", "import sys; print('fake child'); sys.exit(int(sys.argv[1]))"]


@pytest.fixture(scope="module")
def waiter():
    spec = importlib.util.spec_from_file_location("_invented_v9_waiter", ROOT/"scripts"/"gold11_v9_execute_waiter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path):
    attempt = tmp_path/"docs"/"research"/"gold11_mesh_development_v9_attempt"
    attempt.mkdir(parents=True)
    for name in ("FROZEN.json", "PRE_RUN_AUDIT.json"):
        (attempt/name).write_text("{}")
    (attempt/"source").mkdir()
    return attempt


def _config(waiter, tmp_path, code=0):
    return waiter.configuration(root=str(tmp_path), attempt=str(_tree(tmp_path)), command=FAKE+[str(code)])


def _launch_then_wait(waiter, config, code):
    captured = {}
    def spawn(env):
        captured["env"] = env
        return 4242
    assert waiter.launch(config, process_table=lambda: [], spawn=spawn) == 0
    paths = config["records"]
    pid_line = open(paths["pid"]).read().split()
    assert pid_line[0] == "RESERVED" and captured["env"][waiter.TOKEN_ENV] == pid_line[1]
    pending = json.load(open(paths["exit"]))
    assert pending["observation"] == "UNOBSERVED_PENDING" and pending["returncode"] is None and pending["nonce"] == pid_line[1]
    assert os.path.getsize(paths["stdout"]) == 0
    assert waiter.waiter(captured["env"]) == 0
    return captured["env"], paths


@pytest.mark.parametrize("code", [0, 3])
def test_launch_reserves_records_exclusively_and_the_waiter_records_the_observed_return_code(waiter, tmp_path, code):
    config = _config(waiter, tmp_path, code)
    env, paths = _launch_then_wait(waiter, config, code)
    record = json.load(open(paths["exit"]))
    assert record["schema"] == waiter.SCHEMA and record["returncode"] == code
    assert record["observation"] == "OBSERVED_BY_WAITING_PARENT" and record["command"] == config["command"]
    assert record["observed_by"].startswith("WAITING_PARENT_PROCESS_") and record["runner_pid"] > 0
    assert open(paths["stdout"]).read() == "fake child\n"
    assert open(paths["pid"]).read().startswith("FINISHED "+env[waiter.TOKEN_ENV]+" ")
    assert "returncode="+str(code) in open(paths["pid"]).read()
    assert not [name for name in os.listdir(os.path.dirname(paths["exit"])) if name.endswith(".tmp."+str(os.getpid()))]


def test_second_launch_and_duplicate_or_direct_waiter_refuse_without_touching_existing_records(waiter, tmp_path):
    config = _config(waiter, tmp_path, 0)
    env, paths = _launch_then_wait(waiter, config, 0)
    before = {name: open(path, "rb").read() for name, path in paths.items()}
    assert waiter.launch(config, process_table=lambda: [], spawn=lambda env: 1) == 97      # records already present
    assert waiter.waiter(env) == 91                                                         # spent reservation
    assert waiter.waiter({waiter.CONFIG_ENV: env[waiter.CONFIG_ENV]}) == 94                 # no token: direct bypass
    assert waiter.waiter({}) == 94
    assert waiter.waiter(dict(env, **{waiter.TOKEN_ENV: "0"*32})) == 91                    # wrong token
    assert {name: open(path, "rb").read() for name, path in paths.items()} == before


def test_waiter_refuses_when_no_reservation_exists_and_leaves_no_records(waiter, tmp_path):
    config = _config(waiter, tmp_path, 0)
    env = {waiter.TOKEN_ENV: "a"*32, waiter.CONFIG_ENV: json.dumps(config, sort_keys=True)}
    assert waiter.waiter(env) == 93
    assert not any(os.path.exists(path) for path in config["records"].values())


def test_concurrent_waiters_on_one_reservation_are_serialized_by_the_lock(waiter, tmp_path):
    import fcntl
    config = _config(waiter, tmp_path, 0)
    captured = {}
    assert waiter.launch(config, process_table=lambda: [], spawn=lambda env: captured.setdefault("env", env) and 1) == 0
    held = os.open(config["records"]["pid"], os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert waiter.waiter(captured["env"]) == 92                                     # locked by the other waiter
        assert json.load(open(config["records"]["exit"]))["observation"] == "UNOBSERVED_PENDING"
    finally:
        os.close(held)
    assert waiter.waiter(captured["env"]) == 0
    assert json.load(open(config["records"]["exit"]))["returncode"] == 0


def test_launch_admission_refuses_overlap_non_pristine_tree_and_existing_records(waiter, tmp_path):
    config = _config(waiter, tmp_path, 0)
    assert waiter.launch(config, process_table=lambda: ["1 00:01 100 python pytest something"], spawn=lambda env: 1) == 99
    assert not any(os.path.exists(path) for path in config["records"].values())
    Path(config["attempt"], "RESULT.json").write_text("{}")
    assert waiter.launch(config, process_table=lambda: [], spawn=lambda env: 1) == 98
    os.remove(Path(config["attempt"], "RESULT.json"))
    Path(config["records"]["stdout"]).write_text("stale")
    assert waiter.launch(config, process_table=lambda: [], spawn=lambda env: 1) == 97
    assert open(config["records"]["stdout"]).read() == "stale"                          # never truncated
    assert not os.path.exists(config["records"]["exit"]) and not os.path.exists(config["records"]["pid"])


def test_waiter_launch_failure_leaves_the_exit_record_unobserved(waiter, tmp_path):
    config = _config(waiter, tmp_path, 0)
    def failing(env):
        raise OSError("spawn refused")
    assert waiter.launch(config, process_table=lambda: [], spawn=failing) == 95
    record = json.load(open(config["records"]["exit"]))
    assert record["returncode"] is None and record["observation"] == "UNOBSERVED_WAITER_LAUNCH_FAILED"
    assert record["launch_error"] == "OSError"


def test_waiting_parent_failure_records_unobserved_and_reraises(waiter, tmp_path):
    config = _config(waiter, tmp_path, 0)
    captured = {}
    assert waiter.launch(config, process_table=lambda: [], spawn=lambda env: captured.setdefault("env", env) and 1) == 0
    def broken_popen(*args, **kwargs):
        raise RuntimeError("waiting parent died before observing")
    with pytest.raises(RuntimeError):
        waiter.waiter(captured["env"], popen=broken_popen)
    record = json.load(open(config["records"]["exit"]))
    assert record["returncode"] is None and record["observation"] == "UNOBSERVED_WAITING_PARENT_FAILED:RuntimeError"


def test_real_entry_point_runs_the_fake_child_as_a_detached_waiting_parent(waiter, tmp_path):
    """The actual `--waiter` process path with the fake child (no benchmark): the launcher spawns the
    script itself; the observed return code lands in the exit record."""
    config = _config(waiter, tmp_path, 5)
    env = dict(os.environ)
    env[waiter.TOKEN_ENV] = "b"*32
    env[waiter.CONFIG_ENV] = json.dumps(config, sort_keys=True)
    paths = config["records"]
    waiter._write_exclusive(paths["pid"], "RESERVED "+env[waiter.TOKEN_ENV]+" now\n")
    waiter._write_exclusive(paths["exit"], waiter._exit_record(config, env[waiter.TOKEN_ENV]))
    waiter._write_exclusive(paths["stdout"], "")
    done = subprocess.run([sys.executable, str(ROOT/"scripts"/"gold11_v9_execute_waiter.py"), "--waiter"],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout+done.stderr
    record = json.load(open(paths["exit"]))
    assert record["returncode"] == 5 and record["observation"] == "OBSERVED_BY_WAITING_PARENT"
    assert open(paths["stdout"]).read() == "fake child\n"
    direct = subprocess.run([sys.executable, str(ROOT/"scripts"/"gold11_v9_execute_waiter.py"), "--waiter"],
                            cwd=str(tmp_path), env={k: v for k, v in os.environ.items() if not k.startswith("GOLD11_V9_")},
                            capture_output=True, text=True, timeout=60)
    assert direct.returncode == 94 and json.load(open(paths["exit"]))["returncode"] == 5
