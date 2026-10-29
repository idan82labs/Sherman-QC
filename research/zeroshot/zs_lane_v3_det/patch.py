"""Deterministic-math research reference for zs_lane_v3 (in memory only; no lane, pin or production file is
edited). Applies, before the lane runs:

1. every numpy / math transcendental the lane uses (np.sin cos tan arccos arctan2 exp log hypot power; math.sin
   cos tan acos atan2 exp log hypot) and sklearn's expit -> the libm crate (dm.py);
2. np.linalg.eigh / solve / det / lstsq -> zs_rust's pure-Rust linear algebra (dm.py);
3. in-memory rewritten copies of the lane functions that contain `X.T @ X` / `q.T @ q` (Accelerate dsyrk),
   `n @ pk.T` (dgemm) and `LADDER ** k` (C pow): the expression is replaced by the dm.py call, the function is
   recompiled in its own module namespace and every module-level reference to the old function is rebound.
The rewritten functions are listed by `apply()` and must all be found (else it raises)."""
from __future__ import annotations

import inspect
import math
import re
import sys
import textwrap
import types

import numpy as np

from research.zeroshot.zs_lane_v3_det import dm

REWRITES = [  # (module, pattern, replacement)
    ("research.zeroshot.zs_lane_v2.z3s", r"\bX\.T @ X\b", "_DM.xtx(X)"),
    ("research.zeroshot.zs_lane_v2.z3s", r"\bn @ pk\.T\b", "_DM.gemm_nt(n, pk)"),
    ("research.zeroshot.z3_slicing.slicer", r"\bX\.T @ X\b", "_DM.xtx(X)"),
    ("research.zeroshot.z3_slicing.slicer", r"\bn @ pk\.T\b", "_DM.gemm_nt(n, pk)"),
    ("research.zeroshot.z2_self_recipe.segment", r"\bq\.T @ q\b", "_DM.xtx(q)"),
    ("research.zeroshot.z2_self_recipe.graph", r"\bq\.T @ q\b", "_DM.xtx(q)"),
    ("domains.production.engine.inspect.geom", r"\bq\.T @ q\b", "_DM.xtx(q)"),
    ("research.zeroshot.zs_lane_v3.budget", r"\bLADDER \*\* ([jk])\b", r"_DM.pow_(LADDER, \1)"),
]
_APPLIED: list[str] = []


def _rebind(old, new):
    for m in list(sys.modules.values()):
        d = getattr(m, "__dict__", None)
        if not d:
            continue
        for k, v in list(d.items()):
            if v is old:
                setattr(m, k, new)


def _rewrite_module(modname, pat, rep):
    mod = __import__(modname, fromlist=["_"])
    mod.__dict__["_DM"] = dm
    rx = re.compile(pat)
    hits = 0
    for name, obj in list(vars(mod).items()):
        targets = []
        if isinstance(obj, types.FunctionType) and obj.__module__ == modname:
            targets.append((None, name, obj))
        elif isinstance(obj, type) and obj.__module__ == modname:
            targets += [(obj, n, f) for n, f in vars(obj).items() if isinstance(f, types.FunctionType)]
        for owner, n, fn in targets:
            try:
                src = textwrap.dedent(inspect.getsource(fn))
            except (OSError, TypeError):
                continue  # generated (dataclass) methods have no source
            if not rx.search(src):
                continue
            if fn.__closure__:
                raise RuntimeError(f"cannot rewrite closure {modname}.{n}")
            new_src, k = rx.subn(rep, src)
            ns: dict = {}
            code = compile(new_src, f"<det-rewrite {inspect.getsourcefile(fn)}>", "exec", dont_inherit=True)
            exec(code, mod.__dict__, ns)
            new = ns[n]
            new.__defaults__, new.__kwdefaults__ = fn.__defaults__, fn.__kwdefaults__
            if owner is None:
                _rebind(fn, new)
            else:
                setattr(owner, n, new)
            hits += k
            _APPLIED.append(f"{modname}.{n}: {k} x {pat} -> {rep}")
    return hits


def apply() -> list[str]:
    if _APPLIED:
        return _APPLIED
    # import the lane and its libraries first, with numpy untouched (scipy evaluates ufuncs at import time)
    import scipy.spatial, scipy.stats, sklearn, sklearn.ensemble, sklearn.linear_model, sklearn.preprocessing  # noqa: F401,E401
    from research.zeroshot.zs_lane_v3 import lane  # noqa: F401
    for mn, *_ in REWRITES:
        __import__(mn)
    for mn, pat, rep in REWRITES:
        if not _rewrite_module(mn, pat, rep):
            raise RuntimeError(f"rewrite target not found: {mn} {pat}")
    for k in ("sin", "cos", "tan", "arccos", "arctan2", "exp", "log", "hypot", "power"):
        setattr(np, k, getattr(dm, "np_" + k))
    for k in ("sin", "cos", "tan", "acos", "atan2", "exp", "log", "hypot"):
        setattr(math, k, getattr(dm, "math_" + k))
    np.linalg.eigh, np.linalg.solve, np.linalg.det, np.linalg.lstsq = dm.eigh, dm.solve, dm.det, dm.lstsq
    import sklearn._loss.link as SL
    SL._expit = dm.expit
    try:
        import sklearn.linear_model._base as LB
        LB.expit = dm.expit
    except Exception:
        pass
    _APPLIED.append("numpy/math transcendentals, np.linalg.{eigh,solve,det,lstsq}, sklearn expit -> detmath")
    return _APPLIED
