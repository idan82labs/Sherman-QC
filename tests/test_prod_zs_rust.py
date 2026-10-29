"""zs_rust (zs_lane_v3 as one portable Rust binary): unit checks on synthetic meshes (no scan data, no gold, no
custody-relevant input).

Since 2026-10-26 the binary is portable (pure-Rust libm and linear algebra, no Accelerate, no CommonCrypto) and
it is the reference; the frozen Python lane is a comparison, no longer an oracle
(docs/production/runlog/zeroshot_2026-10-25/v3_rust/PORTABLE.md).

* bit for bit against numpy / scipy / open3d where no transcendental or LAPACK call is involved: numpy
  default_rng (random, choice with / without replacement, with / without p), numpy argsort, scipy cKDTree kNN
  order (ties included, FMA distances), Embree ray casts (open3d RaycastingScene, f32 FMA), mesh reading (PLY
  binary / ascii, STL) against open3d;
* against the frozen Python within the measured drift of the portable math (PORTABLE.md): face-budget
  decimation, revision 2 and revision 3 (same faces, vertices within 1e-9 mm, same budget record); the exported
  GBM trees against the pickles (within 4 ulp: exp); Z3, Z2 and the whole lane (same lines and discrete fields,
  geometry within 1e-6);
* determinism: byte-identical outputs at 1 and 4 threads (budget, whole lane, stage dump);
* the station self-check: `selftest --golden` passes for both pinned configs, and refuses (exit 3) a tampered
  golden mesh and a config whose lane parameters it has no expectations for; the x86-64 generic and x86-64-v2
  builds (Rosetta) pass it too when present;
* no host math in the sources (every transcendental through fmath / libm, no mul_add outside fmath, no
  Accelerate / CommonCrypto), the point-cloud refusal (exit 10), the fingerprint pins.
Skipped when the release binary has not been built (cargo build --release --locked in research/zeroshot/zs_rust).
"""
from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.zeroshot.zs_lane_v3_rs import rs  # noqa: E402
from research.zeroshot.zs_lane_v3_rs import rev2ref  # noqa: E402

# the reference is revision 2 as frozen (byte copies, hash-checked), not the live zs_lane_v3 (now revision 3)
rev2ref.install()

pytestmark = pytest.mark.skipif(not rs.available(), reason="zs_rust release binary not built")
CFG = REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json"


def _call(cmd: str, job: dict, d: Path, threads: int = 1, check: bool = True):
    (d / "job.json").write_text(json.dumps(job))
    r = subprocess.run([str(rs.binary()), cmd, "--job", str(d / "job.json"), "--out", str(d / "out.json"),
                        "--threads", str(threads)], capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stderr
        return json.loads((d / "out.json").read_text())
    return r


# ----------------------------------------------------------------------------------------- synthetic meshes

def _profile(segments, t, ri, step=0.25):
    p = np.zeros(2)
    ang = 0.0
    mids, nrms = [], []
    rm = ri + t / 2
    for kind, val in segments:
        if kind == "L":
            n = max(2, int(val / step))
            for s in np.linspace(0, val, n, endpoint=False):
                d = np.array([math.cos(ang), math.sin(ang)])
                mids.append(p + s * d)
                nrms.append(np.array([-d[1], d[0]]))
            p = p + val * np.array([math.cos(ang), math.sin(ang)])
        else:
            b = math.radians(val)
            sgn = 1 if b > 0 else -1
            d = np.array([math.cos(ang), math.sin(ang)])
            c = p + sgn * rm * np.array([-d[1], d[0]])
            n = max(4, int(abs(b) * rm / step))
            for k in range(n):
                a = ang + b * k / n
                mids.append(c - sgn * rm * np.array([-math.sin(a), math.cos(a)]))
                nrms.append(np.array([-math.sin(a), math.cos(a)]))
            ang += b
            p = c - sgn * rm * np.array([-math.sin(ang), math.cos(ang)])
    mids.append(p)
    d = np.array([math.cos(ang), math.sin(ang)])
    nrms.append(np.array([-d[1], d[0]]))
    M, N = np.array(mids), np.array(nrms)
    return M + N * t / 2, M - N * t / 2


def _extrude(segments, t, ri, z1, step=0.5, noise=0.0, seed=0):
    up, lo = _profile(segments, t, ri)
    loop = np.vstack([up, lo[::-1]])
    area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    loop = loop[::-1] if area < 0 else loop
    zs = np.arange(0, z1 + 1e-9, step)
    n = len(loop)
    V = np.array([[x, y, z] for z in zs for (x, y) in loop], float)
    F = []
    for k in range(len(zs) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]
    F = np.array(F, np.int64)
    if noise:
        V = V + np.random.default_rng(seed).normal(scale=noise, size=V.shape)
    fc = V[F].mean(1)
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.mean(np.sum(cr * (fc - V.mean(0)), axis=1)) < 0:
        F = F[:, ::-1].copy()
    return V, F


@pytest.fixture(scope="module")
def channel(tmp_path_factory):
    import open3d as o3d
    V, F = _extrude([("L", 40), ("B", 90), ("L", 70), ("B", 90), ("L", 40)], 2.0, 2.0, 90, noise=0.01)
    d = tmp_path_factory.mktemp("zs_rust_channel")
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    path = d / "channel.ply"
    o3d.io.write_triangle_mesh(str(path), m)
    return path, V, F


def _rt(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ----------------------------------------------------------------------------------------------- primitives

def test_numpy_rng_and_argsort(tmp_path):
    r0 = np.random.default_rng(5)
    p = r0.random(30000) ** 3
    p /= p.sum()
    np.save(tmp_path / "prob.npy", p)
    keys = np.round(r0.random(5000) * 40) / 4.0          # many exact ties
    np.save(tmp_path / "keys.npy", keys)
    ops = [{"seed": 0, "op": "random", "n": 50}, {"seed": 1, "op": "choice_noreplace", "pop": 200000, "size": 150000},
           {"seed": 0, "op": "choice_noreplace", "pop": 3000, "size": 2000},
           {"seed": 0, "op": "choice_p", "p": "prob.npy", "size": 5000},
           {"seed": 0, "op": "choice_p_noreplace", "p": "prob.npy", "size": 5000}]
    out = _call("selftest", {"rng": ops, "argsort": "keys.npy"}, tmp_path)
    ref = [np.random.default_rng(0).random(50).tolist(),
           np.random.default_rng(1).choice(200000, 150000, replace=False).tolist(),
           np.random.default_rng(0).choice(3000, 2000, replace=False).tolist(),
           np.random.default_rng(0).choice(len(p), size=5000, p=p).tolist(),
           np.random.default_rng(0).choice(len(p), size=5000, replace=False, p=p).tolist()]
    assert out["rng"] == ref
    assert np.array_equal(np.load(tmp_path / "argsort_out.npy"), np.argsort(keys))
    assert np.array_equal(np.load(tmp_path / "argsort_out.npy"), np.argsort(keys)) and len(set(keys)) < len(keys)


def test_ckdtree_knn_order_with_ties(tmp_path):
    from scipy.spatial import cKDTree
    g = np.stack(np.meshgrid(np.arange(30), np.arange(30), np.arange(6), indexing="ij"), -1).reshape(-1, 3) * 0.5
    pts = np.vstack([g, np.random.default_rng(2).normal(size=(5000, 3)) * 20])
    np.save(tmp_path / "pts.npy", pts)
    _call("selftest", {"knn": {"P": "pts.npy", "k": 17}}, tmp_path, threads=4)
    dd, ii = cKDTree(pts).query(pts, k=17)
    assert np.array_equal(np.load(tmp_path / "knn_idx.npy"), ii)
    assert np.array_equal(np.load(tmp_path / "knn_d.npy"), dd)
    assert (dd[:, 1:] == dd[:, :-1]).any()


def test_raycast_equals_open3d(tmp_path, channel):
    import open3d as o3d
    _, V, F = channel
    rng = np.random.default_rng(3)
    fi = rng.choice(len(F), 4000, replace=False)
    c = V[F[fi]].mean(1)
    n = np.cross(V[F[fi, 1]] - V[F[fi, 0]], V[F[fi, 2]] - V[F[fi, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    R = np.concatenate([c - n * 1e-3, -n], 1).astype(np.float32)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    ans = sc.cast_rays(o3d.core.Tensor(R))
    np.save(tmp_path / "V.npy", V)
    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "R.npy", R.astype(np.float64))
    _call("selftest", {"rays": {"V": "V.npy", "F": "F.npy", "R": "R.npy"}}, tmp_path)
    th = ans["t_hit"].numpy().astype(np.float64)
    rt = np.load(tmp_path / "ray_t.npy")
    assert np.array_equal(np.isfinite(th), np.isfinite(rt))
    assert np.array_equal(th[np.isfinite(th)], rt[np.isfinite(rt)])


# ------------------------------------------------------------------------------------------- mesh, budget

def test_mesh_readers_equal_open3d(tmp_path, channel):
    import open3d as o3d
    _, V, F = channel
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    m.compute_triangle_normals()
    for name, kw in (("b.ply", {}), ("a.ply", {"write_ascii": True}), ("s.stl", {})):
        p = tmp_path / name
        o3d.io.write_triangle_mesh(str(p), m, **kw)
        ref = o3d.io.read_triangle_mesh(str(p))
        _call("selftest", {"mesh": str(p)}, tmp_path)
        assert np.array_equal(np.load(tmp_path / "mesh_V.npy"), np.asarray(ref.vertices)), name
        assert np.array_equal(np.load(tmp_path / "mesh_F.npy"), np.asarray(ref.triangles)), name


def test_budget_decimation_close_to_python(tmp_path, channel):
    from research.zeroshot.zs_lane_v3 import budget
    _, V, F = channel
    mf = len(F) // 3
    X, Fo, _, info = budget.decimate(V, F, mf)
    np.save(tmp_path / "V.npy", V)
    np.save(tmp_path / "F.npy", F)
    out = _call("selftest", {"decimate": {"V": "V.npy", "F": "F.npy", "max_faces": mf}}, tmp_path)
    assert np.array_equal(np.load(tmp_path / "dec_F.npy"), Fo.astype(np.int64))  # same topology
    dv = np.load(tmp_path / "dec_V.npy")
    assert dv.shape == X.shape and np.abs(dv - X).max() <= 1e-9  # QEF solve: LAPACK -> pure Rust (~1e-12 mm)
    assert out["decimate"]["faces_out"] == info["faces_out"] <= mf


def _rev3_budget_module():
    """the frozen revision-3 budget.py (live zs_lane_v3, checked against config_v3.json's fingerprint dc23a905...),
    loaded under its own name (this module's zs_lane_v3 names resolve to revision 2)"""
    import hashlib
    import importlib.util
    cfg = json.loads((REPO / "research/zeroshot/zs_lane_v3/config_v3.json").read_text())
    rel = "research/zeroshot/zs_lane_v3/budget.py"
    if cfg.get("revision") != 3 or hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != cfg["code_fingerprint"]["files"][rel]:
        pytest.skip("live zs_lane_v3 is not the frozen revision 3")
    spec = importlib.util.spec_from_file_location("zs_lane_v3_budget_rev3", REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, float(cfg["budget"]["cell_mm"])


def test_rev3_budget_close_to_python_and_threads(tmp_path, channel):
    b3, cell = _rev3_budget_module()
    _, V, F = channel
    mf = len(F) // 3
    X, Fo, _, info = b3.decimate(V, F, mf, None, cell)
    np.save(tmp_path / "V.npy", V)
    np.save(tmp_path / "F.npy", F)
    got = []
    for th in (1, 4):
        out = _call("selftest", {"decimate": {"V": "V.npy", "F": "F.npy", "max_faces": mf, "cell_mm": cell}}, tmp_path,
                    threads=th)
        got.append((np.load(tmp_path / "dec_V.npy"), np.load(tmp_path / "dec_F.npy"), json.dumps(out["decimate"])))
    assert np.array_equal(got[0][1], Fo.astype(np.int64))
    assert np.abs(got[0][0] - X).max() <= 1e-9
    assert got[0][2] == json.dumps(_rt(info))
    assert info["faces_out"] <= mf and len(info["ladder"]) >= 1
    assert np.array_equal(got[0][0], got[1][0]) and np.array_equal(got[0][1], got[1][1]) and got[0][2] == got[1][2]


def test_exported_gbm_close_to_pickle(tmp_path):
    cfg = json.loads(CFG.read_text())
    rng = np.random.default_rng(4)
    for key, src in (("zs", "model"), ("z3", "z3_model")):
        mj = REPO / cfg["rust"]["models"][key]["path"]
        mdl = pickle.loads((REPO / cfg["variants"]["zs"][src]).read_bytes())["model"]
        X = rng.normal(size=(3000, mdl.m.n_features_in_)) * 3.0
        X[rng.random(X.shape) < 0.1] = np.nan
        np.save(tmp_path / "X.npy", X)
        _call("selftest", {"gbm": {"model": str(mj), "X": "X.npy"}}, tmp_path)
        p, q = np.load(tmp_path / "gbm_p.npy"), mdl.predict(X)
        assert np.all(np.abs(p.view(np.int64) - q.view(np.int64)) <= 4), key  # the sigmoid's exp: libm vs host


# --------------------------------------------------------------------------------------- stages, whole lane

def _close(a, b, tol, path="$"):
    """same structure, same non-float values, floats within tol"""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _close(a[k], b[k], tol, f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= tol, (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_z3_close_to_python(tmp_path, channel):
    from research.zeroshot.zs_lane_v2 import z3s
    from research.zeroshot.zs_lane_v3 import z3_merge
    path, _, _ = channel
    py = z3_merge.detect(z3s.load_mesh(str(path)), z3s.Params(fallback_frac_only=True))
    rs_ = _call("z3full", {"scan": str(path), "fallback_frac_only": True}, tmp_path)
    assert py["candidates"]
    for k in ("t_mm", "families", "raw_count", "candidates", "family_fallback"):
        _close(_rt(py[k]), _rt(rs_[k]), 1e-6, k)


def test_z2_close_to_python(tmp_path, channel):
    from research.zeroshot.zs_lane_v2 import core as V2
    path, _, _ = channel
    V2.Z2_FALLBACK = True
    lines, scan, _ = V2.run_z2(str(path))
    rs_ = _call("z2full", {"scan": str(path), "fallback": True}, tmp_path)
    assert lines
    _close(_rt(lines), rs_["lines"], 1e-6)
    assert abs(float(scan.t) - rs_["t"]) <= 1e-9


def test_whole_lane_close_to_python_rev2_and_threads(tmp_path, channel):
    from research.zeroshot.zs_lane_v3 import core as C3
    path, _, _ = channel
    v3cfg = json.loads(rev2ref.REV2_CONFIG.read_text())
    C3.configure(v3cfg, "zs")
    rec = C3.process(str(path))
    v = v3cfg["variants"]["zs"]
    zs = pickle.loads((REPO / v["model"]).read_bytes())
    z3 = pickle.loads((REPO / v["z3_model"]).read_bytes())
    ref = _rt(C3.payload(str(path), None, C3.select_lines(rec, v, zs, z3), status="OK", conversion={})["owned_bend_regions"])
    got = []
    for th in (1, 4):
        d = tmp_path / f"t{th}"
        d.mkdir()
        _call("lane", {"scan": str(path), "config": str(CFG), "repo": str(REPO), "out": str(d / "eval_lines.json"),
                       "dump": str(d / "dump.json")}, d, threads=th)
        got.append((json.loads((d / "eval_lines.json").read_text())["owned_bend_regions"],
                    json.loads((d / "dump.json").read_text())))
    assert ref
    # same lines (count, order, src / op / parts / fix), geometry within 1e-6 mm / deg; the score is a GBM
    # probability, which a last-bit feature change can move across a tree split (PORTABLE.md: up to 0.08 on DEV)
    assert len(got[0][0]) == len(ref)
    for a, b in zip(ref, got[0][0]):
        assert {k: a[k] for k in ("bend_id", "src", "op", "parts")} == {k: b[k] for k in ("bend_id", "src", "op", "parts")}
        assert a.get("fix") == b.get("fix")
        _close(a["line_start"] + a["line_end"] + [a["incl"]], b["line_start"] + b["line_end"] + [b["incl"]], 1e-6)
        assert abs(a["score"] - b["score"]) <= 0.1
    assert got[0][0] == got[1][0]
    for k in ("candidates", "X", "z2_lines", "evidence", "prob", "prob_z3"):
        assert got[0][1][k] == got[1][1][k], k


def test_point_cloud_refused(tmp_path):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.default_rng(0).random((500, 3))))
    p = tmp_path / "cloud.ply"
    o3d.io.write_point_cloud(str(p), pc)
    r = _call("lane", {"scan": str(p), "config": str(CFG), "repo": str(REPO)}, tmp_path, check=False)
    assert r.returncode == 10 and "POINT_CLOUD" in r.stderr


def test_fingerprint_pins_binary_models_and_rev2():
    from research.zeroshot.zs_lane_v3_rs import fingerprint as FP
    fp = FP.lane_fingerprint()
    assert fp["files"]["BINARY:zs_rust"] == FP.sha256_file(rs.binary())
    assert "research/zeroshot/zs_rust/Cargo.lock" in fp["files"]
    assert "research/zeroshot/zs_rust/golden/golden.json" in fp["files"]
    assert "research/zeroshot/zs_lane_v3_rs/models/zs_lane_v3_zs_gbm.gbm.json" in fp["files"]
    assert "research/zeroshot/zs_lane_v3/budget.py" in fp["files"]
    rev2 = rev2ref.pinned()
    assert all(fp["files"][f] == h for f, h in rev2.items() if f in fp["files"] and f.startswith("research/zeroshot/zs_lane_v3/"))
    cfg = json.loads(CFG.read_text())
    assert cfg["base_v3_code_fingerprint"].startswith("ab31d061") and cfg["budget"]["max_faces"] == 3_000_000
    assert "cell_mm" not in cfg["budget"]  # revision 2's budget in the binary
    assert cfg["rust"]["portable"] is True and cfg["rust"]["binary_sha256"] == fp["binary_sha256"]
    c3 = REPO / "research/zeroshot/zs_lane_v3_rs/config_v3r3_rs.json"
    if c3.is_file():
        c3 = json.loads(c3.read_text())
        # dc23a905...: frozen revision 3; 49a16399...: the same revision after the 2026-10-26 production re-pin
        assert c3["base_v3_revision"] == 3 and c3["base_v3_code_fingerprint"][:8] in ("dc23a905", "49a16399")
        assert c3["budget"]["cell_mm"] == 0.3 and c3["rust"]["binary_sha256"] == fp["binary_sha256"]


# ------------------------------------------------------------------------------------- portability, station

GOLDEN = REPO / "research/zeroshot/zs_rust/golden"
CFG3 = REPO / "research/zeroshot/zs_lane_v3_rs/config_v3r3_rs.json"


def _golden(binary, cfg, golden=GOLDEN, threads=4, prefix=()):
    r = subprocess.run([*prefix, str(binary), "selftest", "--golden", str(golden), "--config", str(cfg), "--repo",
                        str(REPO), "--threads", str(threads)], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}


@pytest.mark.parametrize("cfg", [CFG, CFG3])
def test_golden_selftest_passes(cfg):
    rc, rep = _golden(rs.binary(), cfg)
    assert rc == 0 and rep["ok"] and not rep["failures"], rep.get("failures")
    assert rep["got"]["cases"]["g1_channel"]["n_lines"] == 2 and rep["got"]["cases"]["g3_hat"]["n_lines"] == 4


def test_golden_selftest_refuses_tampering(tmp_path):
    import shutil
    g = tmp_path / "golden"
    shutil.copytree(GOLDEN, g)
    p = g / "g3_hat.ply"
    txt = p.read_text().splitlines()
    i = next(k for k, ln in enumerate(txt) if ln == "end_header") + 7
    x, y, z = txt[i].split()
    txt[i] = f"{x} {y} {float(z) + 1e-3!r}"  # one vertex moved by a micrometre
    p.write_text("\n".join(txt) + "\n")
    rc, rep = _golden(rs.binary(), CFG3, golden=g, threads=1)
    assert rc == 3 and "MESH_FILE:g3_hat.ply" in rep["failures"]
    cfg = json.loads(CFG3.read_text())
    cfg["variants"]["zs"]["threshold"] = cfg["variants"]["zs"].get("threshold", 0.5) + 0.01
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    rc, rep = _golden(rs.binary(), tmp_path / "cfg.json", threads=1)
    assert rc == 3 and any(f.startswith("NO_GOLDEN_FOR_CONFIG_PARAMS") for f in rep["failures"])


# x86-64-v3 (AVX) is deliberately absent: under Rosetta it passes the golden set but differed on one DEV scan
# (PORTABLE.md), so a pass here would not clear it
@pytest.mark.parametrize("sub", ["target/x86_64-apple-darwin/release/zs_rust",
                                 "target/x86_64-v2/x86_64-apple-darwin/release/zs_rust"])
def test_x86_64_build_passes_golden_under_rosetta(sub):
    b = REPO / "research/zeroshot/zs_rust" / sub
    if sys.platform != "darwin" or not b.is_file() or subprocess.run(["/usr/bin/arch", "-x86_64", "/usr/bin/true"]).returncode:
        pytest.skip("x86-64 build or Rosetta not available")
    rc, rep = _golden(b, CFG3, prefix=("/usr/bin/arch", "-x86_64"))
    assert rc == 0 and rep["ok"] and rep["build"]["target_arch"] == "x86_64", rep.get("failures")


def test_no_host_math_in_sources():
    import re
    src = REPO / "research/zeroshot/zs_rust/src"
    std_math = re.compile(r"\.(sin|cos|tan|asin|acos|atan|atan2|exp|exp2|exp_m1|ln|ln_1p|log|log2|log10|powf|powi|"
                          r"hypot|cbrt|sinh|cosh|tanh|sin_cos|mul_add)\(")
    bad = []
    for f in sorted(src.glob("*.rs")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("//")[0]
            if f.name != "fmath.rs" and std_math.search(code):
                bad.append(f"{f.name}:{n}: {line.strip()}")
            if re.search(r"Accelerate|CommonCrypto|CC_SHA256|\$NEWLAPACK", code):
                bad.append(f"{f.name}:{n}: {line.strip()}")
    assert not bad, bad
    toml = (REPO / "research/zeroshot/zs_rust/Cargo.toml").read_text()
    assert "accelerate" not in toml and 'libm = "0.2"' in toml
