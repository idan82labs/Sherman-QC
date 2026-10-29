"""rsscap.py CMD...: run CMD, sample the process-tree RSS every second, kill the tree above 16 GiB (exit 137)."""
import os, signal, subprocess, sys, time
LIMIT = 16 * 1024**3
p = subprocess.Popen(sys.argv[1:], start_new_session=True)
peak = 0
while p.poll() is None:
    try:
        out = subprocess.run(["ps", "-o", "rss=,pgid=", "-A"], capture_output=True, text=True).stdout
        rss = sum(int(a) * 1024 for a, g in (l.split() for l in out.splitlines() if l.strip()) if int(g) == p.pid)
    except Exception:
        rss = 0
    peak = max(peak, rss)
    if rss > LIMIT:
        os.killpg(p.pid, signal.SIGKILL); sys.stderr.write(f"rsscap: killed at {rss/2**30:.2f} GiB\n"); sys.exit(137)
    time.sleep(1)
sys.exit(p.returncode)
