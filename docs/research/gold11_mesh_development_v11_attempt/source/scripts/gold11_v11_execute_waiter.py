"""Fresh no-overlap check, then the sole V11 --execute under a detached WAITING parent that durably records
the observed outer return code (the V9 launch observed its outer exit through this same contract; V8 had not).

Evidence contract:
  * The launcher reserves the three outer records EXCLUSIVELY (O_CREAT|O_EXCL) before anything starts:
      docs/research/gold11_mesh_development_v11_attempt_outer_exit.json    (pending until observed)
      docs/research/gold11_mesh_development_v11_attempt_outer_pid.txt      (reservation + nonce)
      docs/research/gold11_mesh_development_v11_attempt_outer_stdout.txt   (empty)
    beside, never inside, the attempt directory (the runner owns that directory). Existing records are
    never appended to or truncated: any pre-existing record refuses the launch.
  * The waiting parent can only run against a reservation it was handed: it needs the launcher's
    one-time token (environment), the reservation line carrying that nonce, an exclusive advisory lock
    on the reservation for its whole life, a still-pending exit record and an empty stdout log. A
    direct `--waiter` invocation without those refuses without touching any record.
  * The exit record is replaced atomically once (temp file + rename). If the waiting parent itself
    fails before observing the child, the record says so and the return code stays UNOBSERVED.
Scope: the launcher lists the attempt directory's entry NAMES to check that the tree is pristine
(FROZEN.json, PRE_RUN_AUDIT.json, source); it never reads attempt, cache, raw or Gold contents.
"""
import datetime
import fcntl
import json
import os
import re
import secrets
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTEMPT = os.path.join(ROOT, "docs/research/gold11_mesh_development_v11_attempt")
COMMAND = [os.path.join(ROOT, ".venv/bin/python"), "-I", "-S", "-B", "-u",
           "scripts/run_gold11_mesh_development_v11.py", "--execute"]
PRISTINE_TREE = ["FROZEN.json", "PRE_RUN_AUDIT.json", "source"]
TOKEN_ENV = "GOLD11_V11_WAITER_TOKEN"
CONFIG_ENV = "GOLD11_V11_WAITER_CONFIG"
OVERLAP_PATTERN = re.compile(r"run_gold11_|check_|pytest|cad_guided|_gold11_v\d+_exact_scoring")
SCHEMA = "GOLD11_V11_OUTER_EXIT_RECORD_V1"
SCOPE = "OUTER_PROCESS_RETURN_CODE_ONLY_NOT_A_SCORE_OR_AUDIT"


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def records(attempt):
    return dict(exit=attempt+"_outer_exit.json", pid=attempt+"_outer_pid.txt", stdout=attempt+"_outer_stdout.txt")


def configuration(root=None, attempt=None, command=None):
    root = ROOT if root is None else root
    attempt = os.path.join(root, "docs/research/gold11_mesh_development_v11_attempt") if attempt is None else attempt
    return dict(root=root, attempt=attempt, command=list(COMMAND if command is None else command),
                records=records(attempt))


def _process_table():
    out = subprocess.run(["ps", "-axo", "pid,etime,rss,command"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[1:]


def _write_exclusive(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def _replace_atomically(path, text):
    temp = path+".tmp."+str(os.getpid())
    with open(temp, "w") as handle:
        handle.write(text)
    os.replace(temp, path)


def _exit_record(config, nonce, **fields):
    record = dict(schema=SCHEMA, command=config["command"], nonce=nonce, scope=SCOPE,
                  returncode=None, observation="UNOBSERVED_PENDING", started=None, finished=None,
                  runner_pid=None, observed_by=None)
    record.update(fields)
    return json.dumps(record, indent=1, sort_keys=True)+"\n"


def launch(config=None, *, process_table=_process_table, spawn=None):
    """Admission (overlap, pristine tree, no existing records), exclusive reservation, then the waiter."""
    config = configuration() if config is None else config
    paths = config["records"]
    hits = [line for line in process_table() if OVERLAP_PATTERN.search(line)
            and "gold11_v11_execute_waiter" not in line]
    if hits:
        print("OVERLAP_REFUSED", stamp())
        print("\n".join(hits))
        return 99
    if not os.path.isdir(config["attempt"]) or sorted(os.listdir(config["attempt"])) != PRISTINE_TREE:
        print("TREE_NOT_PRISTINE", sorted(os.listdir(config["attempt"])) if os.path.isdir(config["attempt"]) else None)
        return 98
    if any(os.path.lexists(path) for path in paths.values()):
        print("OUTER_RECORDS_ALREADY_PRESENT")
        return 97
    nonce = secrets.token_hex(16)
    try:
        _write_exclusive(paths["pid"], "RESERVED "+nonce+" "+stamp()+"\n")
        _write_exclusive(paths["exit"], _exit_record(config, nonce))
        _write_exclusive(paths["stdout"], "")
    except FileExistsError:
        print("OUTER_RECORD_RESERVATION_RACE_REFUSED")
        return 96
    env = dict(os.environ)
    env[TOKEN_ENV] = nonce
    env[CONFIG_ENV] = json.dumps(config, sort_keys=True)
    try:
        if spawn is None:
            child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--waiter"], cwd=config["root"],
                                     env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
            waiter_pid = child.pid
        else:
            waiter_pid = spawn(env)
    except Exception as error:
        _replace_atomically(paths["exit"], _exit_record(config, nonce, observation="UNOBSERVED_WAITER_LAUNCH_FAILED",
                                                        launch_error=type(error).__name__, finished=stamp()))
        print("WAITER_LAUNCH_FAILED", type(error).__name__)
        return 95
    time.sleep(3 if spawn is None else 0)
    print("LAUNCHED", stamp(), "waiter", waiter_pid, "reservation", paths["pid"], "no overlapping process; tree pristine")
    return 0


def waiter(environ=None, *, popen=subprocess.Popen):
    """Run the reserved command once as a direct child and record its observed return code."""
    environ = os.environ if environ is None else environ
    token, encoded = environ.get(TOKEN_ENV), environ.get(CONFIG_ENV)
    if not token or not encoded:
        print("WAITER_REFUSED_NO_RESERVATION_TOKEN")
        return 94
    config = json.loads(encoded)
    paths = config["records"]
    try:
        reservation = os.open(paths["pid"], os.O_RDWR)
    except FileNotFoundError:
        print("WAITER_REFUSED_NO_RESERVATION")
        return 93
    try:
        fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(reservation)
        print("WAITER_REFUSED_RESERVATION_LOCKED")
        return 92
    with os.fdopen(reservation, "r+") as handle:
        line = handle.readline().split()
        try:
            with open(paths["exit"]) as pending:
                current = json.load(pending)
        except (OSError, ValueError):
            current = None
        if (len(line) < 2 or line[0] != "RESERVED" or line[1] != token or type(current) is not dict
                or current.get("nonce") != token or current.get("observation") != "UNOBSERVED_PENDING"
                or not os.path.isfile(paths["stdout"]) or os.path.getsize(paths["stdout"]) != 0):
            print("WAITER_REFUSED_RESERVATION_NOT_PENDING_FOR_THIS_TOKEN")
            return 91
        handle.seek(0)
        handle.write("CLAIMED "+token+" waiter_pid="+str(os.getpid())+" "+stamp()+"\n")
        handle.truncate()
        handle.flush()
        started = stamp()
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        try:
            with open(paths["stdout"], "ab") as log:
                proc = popen(config["command"], cwd=config["root"], env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
                handle.seek(0)
                handle.write("RUNNING "+token+" waiter_pid="+str(os.getpid())+" runner_pid="+str(proc.pid)+" "+started+"\n")
                handle.truncate()
                handle.flush()
                code = proc.wait()
        except BaseException as error:
            _replace_atomically(paths["exit"], _exit_record(config, token, started=started, finished=stamp(),
                observation="UNOBSERVED_WAITING_PARENT_FAILED:"+type(error).__name__,
                observed_by="WAITING_PARENT_PROCESS_"+str(os.getpid())))
            raise
        _replace_atomically(paths["exit"], _exit_record(config, token, started=started, finished=stamp(),
            returncode=code, observation="OBSERVED_BY_WAITING_PARENT", runner_pid=proc.pid,
            observed_by="WAITING_PARENT_PROCESS_"+str(os.getpid())))
        handle.seek(0)
        handle.write("FINISHED "+token+" waiter_pid="+str(os.getpid())+" runner_pid="+str(proc.pid)
                     +" returncode="+str(code)+" "+stamp()+"\n")
        handle.truncate()
    return 0


if __name__ == "__main__":
    sys.exit(waiter() if len(sys.argv) > 1 and sys.argv[1] == "--waiter" else launch())
