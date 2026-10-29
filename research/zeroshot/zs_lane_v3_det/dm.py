"""numpy / math facing wrappers over detmath (zs_rust's portable kernels: libm-crate transcendentals, pure-Rust
linear algebra). Research reference only: the production lane keeps numpy's platform math (Accelerate, Apple libm)."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB = Path(__file__).resolve().parent / "detmath/target/release/libdetmath.dylib"
L = ctypes.CDLL(str(_LIB))
_P = ctypes.POINTER(ctypes.c_double)
_SZ = ctypes.c_size_t
L.dm_unary.argtypes = [ctypes.c_int, _P, _P, _SZ]
L.dm_binary.argtypes = [ctypes.c_int, _P, _P, _P, _SZ]
L.dm_xtx.argtypes = [_P, _SZ, _SZ, _P]
L.dm_gemm_nt.argtypes = [_P, _SZ, _P, _SZ, _SZ, _P]
L.dm_eigh.argtypes = [_P, _SZ, _SZ, _P, _P]
L.dm_solve.argtypes = [_P, _P, _SZ, _SZ, _P]
L.dm_solve.restype = _SZ
L.dm_det.argtypes = [_P, _SZ]
L.dm_det.restype = ctypes.c_double
L.dm_lstsq.argtypes = [_P, _SZ, _SZ, _P, ctypes.c_double, _P]
CALLS: dict[str, int] = {}


def _p(a):
    return a.ctypes.data_as(_P)


def _arr(x, name):
    a = np.asarray(x)
    if a.dtype != np.float64:
        if a.dtype.kind in "iub" or (a.dtype == object):
            a = a.astype(np.float64)
        else:
            raise TypeError(f"detmath {name}: dtype {a.dtype} (only float64 is reproduced)")
    return a if a.flags.c_contiguous else np.array(a, order="C")


_ORIG = {k: getattr(np, k) for k in ("sin", "cos", "tan", "arccos", "arctan2", "exp", "log", "hypot", "power")}


def _plain(*xs):
    """float64 / int / bool data without subclasses or out-of-band dtypes (else the original ufunc is used)"""
    for x in xs:
        if isinstance(x, (float, int, np.floating, np.integer)) and not isinstance(x, np.float32) and not isinstance(x, np.float16):
            continue
        if type(x) is np.ndarray and (x.dtype == np.float64 or x.dtype.kind in "iub"):
            continue
        return False
    return True


def _unary(fid, name):
    def f(x, out=None, **kw):
        if kw or not _plain(x):
            CALLS["fallback:" + name] = CALLS.get("fallback:" + name, 0) + 1
            return _ORIG[name](x, out=out, **kw) if out is not None else _ORIG[name](x, **kw)
        CALLS[name] = CALLS.get(name, 0) + 1
        a = _arr(x, name)
        y = np.empty(a.shape)
        L.dm_unary(fid, _p(a), _p(y), a.size)
        if out is not None:
            out[...] = y
            return out
        return y[()] if y.ndim == 0 else y
    f.__name__ = name
    return f


def _binary(fid, name):
    def f(x1, x2, out=None, **kw):
        if kw or not _plain(x1, x2):
            CALLS["fallback:" + name] = CALLS.get("fallback:" + name, 0) + 1
            return _ORIG[name](x1, x2, out=out, **kw) if out is not None else _ORIG[name](x1, x2, **kw)
        CALLS[name] = CALLS.get(name, 0) + 1
        a, b = np.broadcast_arrays(_arr(x1, name), _arr(x2, name))
        a, b = np.array(a, order="C"), np.array(b, order="C")
        y = np.empty(a.shape)
        L.dm_binary(fid, _p(a), _p(b), _p(y), a.size)
        if out is not None:
            out[...] = y
            return out
        return y[()] if y.ndim == 0 else y
    f.__name__ = name
    return f


np_sin, np_cos, np_tan, np_arccos, np_exp, np_log = (_unary(i, n) for i, n in enumerate(["sin", "cos", "tan", "arccos", "exp", "log"]))
np_arctan2, np_hypot, np_power = (_binary(i, n) for i, n in enumerate(["arctan2", "hypot", "power"]))


def _scalar1(f, name):
    def g(x):
        if not isinstance(x, (int, float, np.floating, np.integer)):
            x = float(x)
        x = float(x)
        r = float(f(x))
        if r != r and x == x and name in ("acos", "log"):
            raise ValueError("math domain error")
        return r
    g.__name__ = name
    return g


def _math_log(x, base=None):
    if base is not None:
        raise NotImplementedError("math.log with base")
    x = float(x)
    if x <= 0.0:
        raise ValueError("math domain error")
    return float(np_log(x))


math_sin, math_cos, math_tan, math_acos, math_exp = (_scalar1(f, n) for f, n in ((np_sin, "sin"), (np_cos, "cos"), (np_tan, "tan"), (np_arccos, "acos"), (np_exp, "exp")))
math_log = _math_log


def math_atan2(y, x):
    return float(np_arctan2(float(y), float(x)))


def math_hypot(*c):
    if len(c) != 2:
        raise NotImplementedError("math.hypot with != 2 arguments")
    return float(np_hypot(float(c[0]), float(c[1])))


def pow_(a, b):
    """Python float ** (C pow) through the libm crate"""
    return float(np_power(float(a), float(b)))


def expit(x, out=None, xp=None):
    """scipy.special.expit as zs_rust gbm.rs: 1 / (1 + exp(-x))"""
    y = 1.0 / (1.0 + np_exp(-_arr(x, "expit")))
    if out is not None:
        out[...] = y
        return out
    return y


# ---------------------------------------------------------------------------------------------- linear algebra

def xtx(X):
    """X.T @ X (2-D X)"""
    CALLS["xtx"] = CALLS.get("xtx", 0) + 1
    a = _arr(X, "xtx")
    r, k = a.shape
    out = np.empty((k, k))
    L.dm_xtx(_p(a), r, k, _p(out))
    return out


def gemm_nt(A, B):
    """A @ B.T (2-D A, B)"""
    CALLS["gemm_nt"] = CALLS.get("gemm_nt", 0) + 1
    a, b = _arr(A, "gemm"), _arr(B, "gemm")
    m, k = a.shape
    p = b.shape[0]
    out = np.empty((m, p))
    L.dm_gemm_nt(_p(a), m, _p(b), p, k, _p(out))
    return out


def eigh(a, UPLO="L"):
    CALLS["eigh"] = CALLS.get("eigh", 0) + 1
    if UPLO != "L":
        raise NotImplementedError("eigh UPLO")
    m = _arr(a, "eigh")
    n = m.shape[-1]
    lead = m.shape[:-2]
    cnt = int(np.prod(lead)) if lead else 1
    w = np.empty(lead + (n,))
    v = np.empty(lead + (n, n))
    if n > 0 and cnt > 0:
        L.dm_eigh(_p(m), n, cnt, _p(w), _p(v))
    return np.linalg._linalg.EighResult(w, v)


def solve(a, b):
    CALLS["solve"] = CALLS.get("solve", 0) + 1
    A = _arr(a, "solve")
    B = _arr(b, "solve")
    n = A.shape[-1]
    if B.ndim == 1:  # vector right-hand side (numpy solve1)
        if A.ndim != 2:
            raise NotImplementedError("batched A with 1-D b")
        x = np.empty(n)
        if L.dm_solve(_p(A), _p(B), n, 1, _p(x)):
            raise np.linalg.LinAlgError("Singular matrix")
        return x
    if B.shape[-1] != 1 or B.shape[:-1] != A.shape[:-1]:
        raise NotImplementedError(f"solve shapes {A.shape} {B.shape}")
    cnt = int(np.prod(A.shape[:-2])) if A.ndim > 2 else 1
    Bv = np.ascontiguousarray(B[..., 0])
    x = np.empty(Bv.shape)
    if L.dm_solve(_p(A), _p(Bv), n, cnt, _p(x)):
        raise np.linalg.LinAlgError("Singular matrix")
    return x[..., None]


def det(a):
    CALLS["det"] = CALLS.get("det", 0) + 1
    A = _arr(a, "det")
    if A.ndim != 2:
        raise NotImplementedError("batched det")
    return np.float64(L.dm_det(_p(A), A.shape[0]))


def lstsq(a, b, rcond=None):
    CALLS["lstsq"] = CALLS.get("lstsq", 0) + 1
    A, B = _arr(a, "lstsq"), _arr(b, "lstsq")
    if B.ndim != 1:
        raise NotImplementedError("lstsq with 2-D b")
    m, n = A.shape
    rc = np.finfo(np.float64).eps * max(m, n) if rcond is None else float(rcond)
    x = np.empty(n)
    L.dm_lstsq(_p(A), m, n, _p(B), rc, _p(x))
    return x, None, None, None
