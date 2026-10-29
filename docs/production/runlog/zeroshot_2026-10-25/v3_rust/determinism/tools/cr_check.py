"""For libm calls where Apple libm and the libm crate (musl) disagree (probe samples), decide which one is the
correctly rounded result, with 60-digit decimal arithmetic (exp/ln built in; sin/cos/atan by series).
usage: .venv/bin/python -B cr_check.py PROBE_RUN_DIR [--per-fn N] [--out JSON]"""
import argparse, glob, json, random, struct
from decimal import Decimal as D, getcontext
getcontext().prec = 70
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--per-fn", type=int, default=3000); ap.add_argument("--out", default="")
a = ap.parse_args()
f = lambda b: struct.unpack("<d", struct.pack("<Q", b))[0]
PI = D("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798")

def dsin(x):
    x = x % (2 * PI)
    s, t, n = D(0), x, 1
    while abs(t) > D(10) ** -68:
        s += t; t = -t * x * x / ((n + 1) * (n + 2)); n += 2
    return s
def dcos(x): return dsin(x + PI / 2)
def datan(x):
    if x < 0: return -datan(-x)
    if x > 1: return PI / 2 - datan(1 / x)
    k = 0
    while x > D("0.1"):
        x = x / (1 + (1 + x * x).sqrt()); k += 1
    s, t, n = D(0), x, 1
    while abs(t) > D(10) ** -68:
        s += t / n; t = -t * x * x; n += 2
    return s * (2 ** k)
def datan2(y, x):
    if x > 0: return datan(y / x)
    if x < 0: return datan(y / x) + (PI if y >= 0 else -PI)
    return PI / 2 if y > 0 else -PI / 2
def dacos(x): return datan2((1 - x * x).sqrt(), x)
F = {"sin": lambda a: dsin(a[0]), "cos": lambda a: dcos(a[0]), "tan": lambda a: dsin(a[0]) / dcos(a[0]),
     "acos": lambda a: dacos(a[0]), "atan2": lambda a: datan2(a[0], a[1]), "exp": lambda a: a[0].exp(),
     "ln": lambda a: a[0].ln(), "hypot": lambda a: (a[0] * a[0] + a[1] * a[1]).sqrt(), "pow": lambda a: (a[1] * a[0].ln()).exp()}
by = {}
for p in sorted(glob.glob(a.run + "/*/probe.json")):
    for fn, args, port, host in json.load(open(p))["math_samples"]:
        by.setdefault(fn, []).append((args, port, host))
random.seed(0)
res = {}
for fn, L in sorted(by.items()):
    L = random.sample(L, min(a.per_fn, len(L)))
    c = {"n": len(L), "host_cr": 0, "port_cr": 0, "neither": 0}
    for args, port, host in L:
        r = float(F[fn]([D(f(x)) for x in args]))
        if r == f(host): c["host_cr"] += 1
        elif r == f(port): c["port_cr"] += 1
        else: c["neither"] += 1
    res[fn] = c; print(fn, c, flush=True)
if a.out: open(a.out, "w").write(json.dumps(res, indent=1))
