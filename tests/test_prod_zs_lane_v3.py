"""zs_lane_v3: synthetic contract checks for the two v3 line fixes (no scan data, no gold).

EXT: a selected line whose candidate absorbed a longer Z2 line at pooling takes the union of the two extents along
its own axis (ratio and containment gates), drops the lines it now duplicates, never adds a line and only uses Z2
lines attached to its own candidates. Z3R: a rejected Z3-only candidate is kept only with a strong Z3-alone
probability and a station sweep, never when Z2 agreed, when it is already used or when it duplicates a line.
Also: the frozen parameters, the fingerprint coverage, the config, and v1 / v2 left unchanged.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.zeroshot.zs_lane_v3 import fixes as FX  # noqa: E402

PRM = FX.FixParams()


def _cand(s, e, src=("z3",), has_z3=True, snap_dy=0.0):
    c = {"start": [float(v) for v in s], "end": [float(v) for v in e], "src": list(src), "has_z3": has_z3,
         "incl": 90.0}
    c["start_s"] = [c["start"][0], c["start"][1] + snap_dy, c["start"][2]]
    c["end_s"] = [c["end"][0], c["end"][1] + snap_dy, c["end"][2]]
    return c


def _line(c, parts, score=0.9, op="keep"):
    return {"start_s": list(c["start_s"]), "end_s": list(c["end_s"]), "score": score, "src": c["src"],
            "incl": 90.0, "parts": list(parts), "op": op}


def _z2(s, e):
    return {"start": [float(v) for v in s], "end": [float(v) for v in e], "incl": 90.0}


def _len(l):
    return float(np.linalg.norm(np.subtract(l["end_s"], l["start_s"])))


# ----------------------------------------------------------------------------------------------------- EXT

def test_frozen_parameters():
    assert (PRM.ext_ratio, PRM.ext_min_in, PRM.z3r_tau, PRM.z3r_sweep_min) == (1.3, 0.6, 0.9, 0.25)
    assert PRM.ext and PRM.z3r and PRM.ext_dedupe


def test_ext_adopts_the_longer_z2_extent_along_the_line_axis():
    # B1-like: Z3 saw the last 36 mm of an 80 mm fold, Z2 the whole fold (0.3 mm off, slightly tilted)
    c = _cand([44, 0, 0], [80, 0, 0], src=("z2", "z3"), snap_dy=0.2)
    z2 = [_z2([0.5, 0.3, 0], [79.9, 0.3, 0.1])]
    out = FX.extend([_line(c, [0])], [c], z2, PRM)
    assert len(out) == 1 and out[0]["fix"] == "EXT"
    assert _len(out[0]) == pytest.approx(80.0 - 0.5, abs=0.05)
    # axis and transverse placement are the line's own (snapped y = 0.2, not Z2's 0.3)
    assert out[0]["start_s"][1] == pytest.approx(0.2) and out[0]["end_s"][1] == pytest.approx(0.2)
    assert out[0]["start_s"][2] == pytest.approx(0.0) and out[0]["end_s"][2] == pytest.approx(0.0)


def test_ext_ratio_and_containment_gates():
    c = _cand([20, 0, 0], [60, 0, 0], src=("z2", "z3"))
    # 1.2x longer only: no extension
    out = FX.extend([_line(c, [0])], [c], [_z2([16, 0, 0], [64, 0, 0])], PRM)
    assert "fix" not in out[0] and _len(out[0]) == pytest.approx(40.0)
    # 1.5x longer but contains only 50 % of the line (< 0.6): no extension
    out = FX.extend([_line(c, [0])], [c], [_z2([40, 0, 0], [100, 0, 0])], PRM)
    assert "fix" not in out[0]
    # contains 62.5 % (>= 0.6) and >= 1.3x: extended (the 0.75 setting would refuse it)
    z = [_z2([35, 0, 0], [87, 0, 0])]
    out = FX.extend([_line(c, [0])], [c], z, PRM)
    assert out[0]["fix"] == "EXT" and out[0]["end_s"][0] == pytest.approx(87.0) and out[0]["start_s"][0] == pytest.approx(20.0)
    assert "fix" not in FX.extend([_line(c, [0])], [c], z, replace(PRM, ext_min_in=0.75))[0]


def test_ext_only_uses_z2_lines_absorbed_by_the_lines_own_candidates():
    c0 = _cand([0, 0, 0], [30, 0, 0], src=("z3",))
    c1 = _cand([0, 1.0, 0], [30, 1.0, 0], src=("z2", "z3"))     # the Z2 line is closer to c1: pooled onto c1
    z2 = [_z2([0, 1.0, 0], [80, 1.0, 0])]
    assert FX.z2_attachment([c0, c1], z2) == {1: [0]}
    out = FX.extend([_line(c0, [0])], [c0, c1], z2, PRM)
    assert "fix" not in out[0]
    # a Z2 line that duplicates no Z3 candidate is not attached at all
    assert FX.z2_attachment([c0], [_z2([0, 20, 0], [80, 20, 0])]) == {}


def test_ext_drops_the_lines_it_now_duplicates_and_never_adds_a_line():
    # B4-like: two 30 mm halves of a 70 mm fold, Z2 absorbed by the first half only
    a = _cand([0, 0, 0], [30, 0, 0], src=("z2", "z3"))
    b = _cand([40, 0, 0], [70, 0, 0], src=("z3",))
    other = _cand([0, 50, 0], [70, 50, 0], src=("z2", "z3"))      # an unrelated parallel line, kept
    z2 = [_z2([0, 0.2, 0], [70, 0.2, 0]), _z2([0, 50, 0], [70, 50, 0])]
    lines = [_line(a, [0]), _line(b, [1], score=0.6), _line(other, [2])]
    out = FX.extend(lines, [a, b, other], z2, PRM)
    assert len(out) == 2 and len(out) <= len(lines)
    assert out[0]["fix"] == "EXT" and _len(out[0]) == pytest.approx(70.0)
    assert out[1]["parts"] == [2] and "fix" not in out[1]
    # without the dedupe, the half stays
    assert len(FX.extend(lines, [a, b, other], z2, replace(PRM, ext_dedupe=False))) == 3


def test_ext_never_shortens_and_is_inert_without_z2():
    c = _cand([0, 0, 0], [50, 0, 0])
    out = FX.extend([_line(c, [0])], [c], [], PRM)
    assert out == [_line(c, [0])]


# ----------------------------------------------------------------------------------------------------- Z3R

def _z3r_case():
    sel = _cand([0, 0, 0], [40, 0, 0], src=("z2", "z3"))
    rej = _cand([0, 30, 0], [25, 30, 0], src=("z3",))
    cands = [sel, rej]
    lines = [_line(sel, [0])]
    prob = np.array([0.95, 0.2])
    p3 = np.array([0.97, 0.904])
    sweep = [[1.0] + [0] * 5, [0.8] + [0] * 5]
    return cands, lines, prob, p3, sweep


def test_z3r_keeps_a_strong_z3_only_line_with_sweep():
    cands, lines, prob, p3, sweep = _z3r_case()
    out = FX.rescue(lines, cands, prob, p3, sweep, PRM)
    assert len(out) == 2 and out[1]["fix"] == "Z3R" and out[1]["parts"] == [1]
    assert out[1]["score"] == pytest.approx(0.2) and out[1]["p_z3"] == pytest.approx(0.904)
    assert out[1]["start_s"] == cands[1]["start_s"]


@pytest.mark.parametrize("change", ["p3_low", "sweep_low", "sweep_none", "no_evidence", "z2_agreed", "used", "dup"])
def test_z3r_refuses(change):
    cands, lines, prob, p3, sweep = _z3r_case()
    if change == "p3_low":
        p3[1] = 0.89
    elif change == "sweep_low":
        sweep[1][0] = 0.2
    elif change == "sweep_none":
        sweep[1] = None
    elif change == "no_evidence":
        sweep = None
    elif change == "z2_agreed":
        cands[1]["src"] = ["z2", "z3"]
    elif change == "used":
        lines.append(_line(cands[1], [1]))
    elif change == "dup":
        lines.append(_line(_cand([0, 30.5, 0], [24, 30.5, 0]), [0]))
    out = FX.rescue(lines, cands, prob, p3, sweep, PRM)
    assert not any(l.get("fix") == "Z3R" for l in out) and len(out) == len(lines)


def test_z3r_ignores_candidates_without_z3_features():
    cands, lines, prob, p3, sweep = _z3r_case()
    cands[1]["has_z3"] = False
    p3[1] = np.nan
    assert len(FX.rescue(lines, cands, prob, p3, sweep, PRM)) == 1


def test_apply_order_and_switches():
    a = _cand([0, 0, 0], [30, 0, 0], src=("z2", "z3"))
    r = _cand([0, 40, 0], [25, 40, 0], src=("z3",))
    z2 = [_z2([0, 0, 0], [70, 0, 0])]
    lines = [_line(a, [0])]
    args = ([a, r], z2, np.array([0.9, 0.1]), np.array([0.9, 0.95]), [[1] * 6, [1] * 6])
    out = FX.apply(lines, *args, PRM)
    assert [l.get("fix") for l in out] == ["EXT", "Z3R"]
    assert FX.apply(lines, *args, replace(PRM, ext=False, z3r=False)) == lines


# --------------------------------------------------------------------------------------- packaging / pins

def test_fingerprint_covers_v3_and_the_reused_v2_modules():
    from research.zeroshot.zs_lane_v3 import fingerprint as FP
    files = set(FP.LANE_FILES)
    for f in ("core.py", "lane.py", "fixes.py", "fingerprint.py", "__init__.py", "z3_merge.py", "budget.py"):
        assert f"research/zeroshot/zs_lane_v3/{f}" in files
    for f in ("core.py", "z3s.py", "evidence.py", "postproc.py"):
        assert f"research/zeroshot/zs_lane_v2/{f}" in files
    assert not any("n1_diag" in f for f in files)
    src = "".join((REPO / f).read_text() for f in files if f.startswith("research/zeroshot/zs_lane_v3/"))
    assert "n1_diag" not in src


#: Shared E3 / contracts files the lanes import read-only, changed by the re-pins of 2026-10-26 (e3-1.13.0 +
#: ENGINE_IO-A5) and 2026-10-28 (e3-1.14.0: config.py / engine.py). No lane-own file changed.
REPIN_SHARED = {"contracts/production/line_job.py", "domains/production/engine/inspect/config.py",
                "domains/production/engine/inspect/engine.py", "domains/production/engine/inspect/metrology.py"}
REPIN_ARCHIVE = REPO / "docs/production/runlog/zeroshot_2026-10-25/repin_2026-10-26/archive"


def test_v1_and_v2_fingerprints_unchanged():
    """v1: re-pinned 2026-10-26 (af87df2f... -> 52fd21d6...) and 2026-10-28 with e3-1.14.0 (-> c712c15c...); against
    the frozen TEST-B config only the shared E3 / contracts files moved. v2 (research
    only, not pinned in production) is not re-frozen: its lane-own files still equal config_v2.json, and the only
    files that differ are the same shared files."""
    from research.zeroshot.zs_lane_v1 import fingerprint as F1
    from research.zeroshot.zs_lane_v2 import fingerprint as F2
    c1 = json.loads((REPO / "research/zeroshot/zs_lane_v1/config_v1.json").read_text())
    c2 = json.loads((REPO / "research/zeroshot/zs_lane_v2/config_v2.json").read_text())
    assert F1.lane_fingerprint()["sha256"] == c1["code_fingerprint"]["sha256"] == \
        "c712c15c0843d89db98eca5d8b757e75976f884520e5543933acd4fabb64555b"
    c1_pre = json.loads((REPIN_ARCHIVE / "config_v1_pre_repin.json").read_text())
    assert c1_pre["code_fingerprint"]["sha256"] == "af87df2f96426a9e69a46ed942ced4f19f0eb770e71e50aada43ddd27ec1441c"
    moved = {k for k, v in c1["code_fingerprint"]["files"].items() if c1_pre["code_fingerprint"]["files"][k] != v}
    assert moved == REPIN_SHARED
    assert c2["code_fingerprint"]["sha256"] == "39c0efb92c3c3962605168fb2cda977edff5840a2da1cd4518659e7544209097"
    now2 = F2.lane_fingerprint()["files"]
    assert set(now2) == set(c2["code_fingerprint"]["files"])
    assert {k for k, v in now2.items() if c2["code_fingerprint"]["files"][k] != v} <= REPIN_SHARED


def test_config_v3_is_v2_plus_the_frozen_fixes():
    p = REPO / "research/zeroshot/zs_lane_v3/config_v3.json"
    if not p.is_file():
        pytest.skip("config_v3.json not frozen yet")
    from research.zeroshot.zs_lane_v3 import fingerprint as FP
    c3 = json.loads(p.read_text())
    c2 = json.loads((REPO / "research/zeroshot/zs_lane_v2/config_v2.json").read_text())
    v3, v2 = c3["variants"]["zs"], c2["variants"]["zs"]
    assert v3["post"] == v2["post"] and v3["threshold"] == v2["threshold"] == 0.5
    assert v3["fix"] == asdict(FX.FixParams())
    assert c3["z3_params"] == c2["z3_params"] and c3["z2_fallback"] == c2["z2_fallback"]
    assert c3["code_fingerprint"]["sha256"] == FP.lane_fingerprint()["sha256"]
    from research.zeroshot.zs_lane_v3 import budget as B
    assert c3["revision"] == 3 and c3["budget"]["max_faces"] == B.MAX_FACES == 3_000_000
    assert c3["budget"]["cell_mm"] == B.CELL_MM == 0.30
    for key in ("model", "z3_model"):
        assert FP.sha256_file(REPO / v3[key]) == v3[f"{key}_sha256"]


# ------------------------------------------------------------------------------- production pin (prepared)

EXAMPLES = REPO / "contracts" / "production" / "examples"


REV2_ARCHIVE = REPO / "docs/production/runlog/zeroshot_2026-10-25/v3/rev3/rev2_archive"
REV3_CHANGED = {"research/zeroshot/zs_lane_v3/budget.py": "budget_rev2.py", "research/zeroshot/zs_lane_v3/core.py": "core_rev2.py"}


def test_v3_pin_matches_config_and_verifies():
    """Re-pinned 2026-10-26 (v3/REPIN_PLAN.md "The re-pin event"): the production pin is revision 3, equals the live
    config_v3.json and verifies against today's tree. Against revision 2 only the two revision-3 files and the shared
    E3 / contracts files moved; against the pre-re-pin revision 3 (dc23a905...) only the shared files."""
    from domains.production.zeroshot import pin as PIN
    pin = PIN.load_pin(REPO, lane="zs_lane_v3")
    cfg = json.loads((REPO / "research/zeroshot/zs_lane_v3/config_v3.json").read_text())
    v = cfg["variants"]["zs"]
    assert pin["code_fingerprint"] == cfg["code_fingerprint"]
    assert pin["model"]["sha256"] == v["model_sha256"] and pin["extra_models"]["z3"]["sha256"] == v["z3_model_sha256"]
    assert pin["lane_params"]["variant"]["fix"] == v["fix"] == asdict(FX.FixParams())
    assert pin["lane_params"]["z3_params"] == cfg["z3_params"] and pin["lane_params"]["z2_fallback"] == cfg["z2_fallback"]
    assert pin["lane_params"]["budget"] == cfg["budget"] and pin["revision"] == cfg["revision"] == 3
    assert pin["lane_params"]["budget"]["cell_mm"] == 0.30 and pin["lane_params"]["budget"]["max_faces"] == 3_000_000
    assert PIN.verify(REPO, pin) == []
    files = PIN.pinned_files(pin)
    assert PIN.PIN_RELS["zs_lane_v3"] in files and pin["extra_models"]["z3"]["rel_path"] in files
    assert "research/zeroshot/zs_lane_v3/fixes.py" in files
    assert {"research/zeroshot/zs_lane_v3/z3_merge.py", "research/zeroshot/zs_lane_v3/budget.py"} <= set(files)
    rev2 = json.loads((REV2_ARCHIVE / "config_v3_rev2.json").read_text())
    moved = {k for k, h in cfg["code_fingerprint"]["files"].items() if rev2["code_fingerprint"]["files"][k] != h}
    assert moved == set(REV3_CHANGED) | REPIN_SHARED
    pre = json.loads((REPIN_ARCHIVE / "config_v3_rev3_pre_repin.json").read_text())
    assert pre["code_fingerprint"]["sha256"] == "dc23a905bc2cf83a2fff2a4e1270ca654e95b8d3d6f2077db97c50559c85476f"
    moved = {k for k, h in cfg["code_fingerprint"]["files"].items() if pre["code_fingerprint"]["files"][k] != h}
    assert moved == REPIN_SHARED
    for rel, arch in REV3_CHANGED.items():   # the archived revision-2 sources still match the revision-2 config
        assert PIN.sha256_file(REV2_ARCHIVE / arch) == rev2["code_fingerprint"]["files"][rel]


def test_v3_pin_fails_closed_on_the_second_model(tmp_path):
    import shutil
    from domains.production.zeroshot import pin as PIN
    pin = PIN.load_pin(REPO, lane="zs_lane_v3")
    for rel in PIN.pinned_files(pin):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp_path / rel)
    assert PIN.verify(tmp_path, pin) == []
    z3 = tmp_path / pin["extra_models"]["z3"]["rel_path"]
    z3.chmod(0o644)
    z3.write_bytes(z3.read_bytes() + b"x")
    assert f"MODEL:{pin['extra_models']['z3']['rel_path']}" in PIN.verify(tmp_path, pin)
    victim = "research/zeroshot/zs_lane_v3/fixes.py"
    (tmp_path / victim).write_text((tmp_path / victim).read_text() + "\n# edited\n")
    assert f"CODE:{victim}" in PIN.verify(tmp_path, pin)
    shutil.copyfile(REV2_ARCHIVE / "budget_rev2.py", tmp_path / "research/zeroshot/zs_lane_v3/budget.py")
    assert "CODE:research/zeroshot/zs_lane_v3/budget.py" in PIN.verify(tmp_path, pin)   # revision 2 is refused


def test_pin_selection_default_env_and_unknown(monkeypatch):
    from domains.production.zeroshot import pin as PIN
    monkeypatch.delenv(PIN.PIN_ENV, raising=False)
    assert PIN.DEFAULT_LANE == "zs_lane_v1" and PIN.selected_lane() == "zs_lane_v1"
    assert PIN.load_pin(REPO)["lane"] == "zs_lane_v1"
    monkeypatch.setenv(PIN.PIN_ENV, "zs_lane_v3")
    assert PIN.selected_lane() == "zs_lane_v3" and PIN.load_pin(REPO)["lane"] == "zs_lane_v3"
    monkeypatch.setenv(PIN.PIN_ENV, "zs_lane_v9")
    with pytest.raises(ValueError):
        PIN.selected_lane()
    assert PIN.verify(REPO) and PIN.verify(REPO)[0].startswith("PIN_UNREADABLE")


def _staged(tmp_path):
    (tmp_path / "input").mkdir()
    scan = tmp_path / "input" / "scan.ply"
    scan.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    return scan, tmp_path / "zeroshot" / "zs_lines.json"


def test_lane_cli_unknown_pin_writes_no_lines(tmp_path, monkeypatch):
    from domains.production.zeroshot import lane_cli, pin as PIN
    monkeypatch.setenv(PIN.PIN_ENV, "zs_lane_v9")
    scan, out = _staged(tmp_path)
    rc = lane_cli.main(["--scan", str(scan), "--compiled", str(EXAMPLES / "compiled_part_v1.example.json"), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["status"] == "FAILED" and doc["lines"] == []


def test_lane_cli_v3_pin_mismatch_and_refusal(tmp_path, monkeypatch):
    from domains.production.zeroshot import lane_cli, pin as PIN
    monkeypatch.setenv(PIN.PIN_ENV, "zs_lane_v3")
    bad = tmp_path / "bad"
    bad.mkdir()
    scan = bad / "SN0042.ply"
    scan.write_bytes(b"ply\n")
    out = bad / "zs.json"
    assert lane_cli.main(["--scan", str(scan), "--compiled", str(EXAMPLES / "compiled_part_v1.example.json"),
                          "--out", str(out)]) == 1
    assert json.loads(out.read_text())["status"] == "REFUSED_INPUT"
    monkeypatch.setattr(PIN, "verify", lambda root=None, pin=None: ["MODEL:domains/production/zeroshot/assets/zs_lane_v3_z3_gbm.pkl"])
    scan, out = _staged(tmp_path)
    rc = lane_cli.main(["--scan", str(scan), "--compiled", str(EXAMPLES / "compiled_part_v1.example.json"), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["status"] == "PIN_MISMATCH" and doc["lines"] == [] and doc["lane"] == "zs_lane_v3"


def test_run_lane_v3_count_cap_drops_rescued_lines_first(monkeypatch):
    from domains.production.zeroshot import lane_cli
    from research.zeroshot.zs_lane_v3 import core
    seen = {}

    def fake_process(path, *, thickness=None, max_families=None):
        seen.update(thickness=thickness, max_families=max_families)
        return {"status": "OK", "candidates": [{}] * 5, "stages": {}, "errors": {}}

    lines = [{"start_s": [0, 0, 0], "end_s": [10, 0, 0], "score": s, "op": "keep", "src": ["z3"], "incl": 90.0,
              **({"fix": f} if f else {})} for s, f in ((0.95, None), (0.30, "Z3R"), (0.80, "EXT"), (0.60, None))]
    monkeypatch.setattr(core, "process", fake_process)
    monkeypatch.setattr(core, "select_lines", lambda rec, v, zs, z3: [dict(l) for l in lines])
    pin = json.loads((REPO / "domains/production/zeroshot/assets/zs_lane_v3.pin.json").read_text())
    facts = {"n_bends": 3, "t_mm": 2.0, "K": 1, "max_families": 3}
    res = lane_cli.run_lane_v3(Path("scan.ply"), facts, {"zs": None, "z3": None}, pin["lane_params"])
    assert seen == {"thickness": {"t_mm": 2.0, "source": "CAD"}, "max_families": 3}
    assert res["status"] == "OK" and res["n_selected"] == 4
    assert [l["score"] for l in res["lines"]] == [0.95, 0.8, 0.6]
    assert [l["fix"] for l in res["lines"]] == [None, "EXT", None]


def test_overlay_accepts_a_v3_stamp_and_still_never_upgrades():
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.inspection_result import ShopfloorInspectionResultV1
    from domains.production.zeroshot.overlay import PASS_LIKE_STATES, build_overlay
    compiled = CompiledPartV1.model_validate_json((EXAMPLES / "compiled_part_v1.example.json").read_bytes())
    result = ShopfloorInspectionResultV1.model_validate_json((EXAMPLES / "shopfloor_result_v1.example.json").read_bytes())
    doc = {"schema": "sherman_zeroshot_lines_v1", "status": "OK", "lane": "zs_lane_v3", "variant": "zs",
           "runtime_version": "zs-prod-1.0.0", "code_fingerprint": "a" * 64, "model_sha256": "b" * 64,
           "scan_sha256": result.scan.sha256, "priors": {"n_bends": 4, "t_mm": 2.0, "K": 1, "max_families": 3},
           "lines": []}
    ov = build_overlay(result, compiled, doc, engine_result_sha256="c" * 64)
    assert ov.lane.lane == "zs_lane_v3"
    assert ov.effective_summary_state == result.summary.state or ov.effective_summary_state not in PASS_LIKE_STATES


# --------------------------------------------------------------------- revision 2: runtime / memory (synthetic)

def _l_bend_mesh(L=200.0, W=40.0, step=1.0):
    """An L-bent sheet along x (flange A in z=0, y<0; flange B in y=0, z>0), one skin."""
    from research.zeroshot.zs_lane_v2 import z3s as Z
    xs, ws = np.arange(0.0, L + step / 2, step), np.arange(0.0, W + step / 2, step)
    pts, faces = [], []
    for k, f in enumerate((lambda x, w: (x, -w, 0.0), lambda x, w: (x, 0.0, w))):
        base = len(pts)
        pts += [f(x, w) for x in xs for w in ws]
        nw = len(ws)
        for i in range(len(xs) - 1):
            for j in range(nw - 1):
                a, b, c, d = base + i * nw + j, base + (i + 1) * nw + j, base + (i + 1) * nw + j + 1, base + i * nw + j + 1
                faces += [(a, b, c), (a, c, d)] if k == 0 else [(a, c, b), (a, d, c)]
    return Z.mesh_from_arrays(np.asarray(pts, float), np.asarray(faces))


def _segments(rng, n=260):
    """Many parallel segments of one family: collinear runs with gaps (some bridged by the plates), parallel
    offset rows, outliers in angle / bisector, short and long pieces."""
    segs = []
    bis = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
    for _ in range(n):
        row = rng.choice([0.0, 0.0, 0.0, 0.6, 3.0, 8.0])
        x0 = rng.uniform(0, 190)
        k = int(rng.integers(2, 9))
        lev = x0 + 2.0 * np.arange(k)
        P = np.c_[lev, np.full(k, -row) + rng.normal(0, 0.05, k), rng.normal(0, 0.05, k)]
        incl = np.full(k, 90.0 + rng.choice([0.0, 0.0, 3.0, 12.0]))
        b = bis if rng.random() > 0.1 else np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
        segs.append({"family": 0, "axis": np.array([1.0, 0.0, 0.0]), "P": P, "lev": lev, "incl": incl,
                     "fmin": np.full(k, 30.0), "fmax": np.full(k, 40.0), "fla": np.full(k, 40.0), "flb": np.full(k, 30.0),
                     "outer": np.ones(k, bool), "inner": np.zeros(k, bool), "both": np.zeros(k, bool),
                     "rms": np.full(k, 0.05), "gap": np.zeros(k), "bis": np.tile(b, (k, 1)),
                     "da": np.tile([0.0, -1.0, 0.0], (k, 1)), "db": np.tile([0.0, 0.0, 1.0], (k, 1))})
    return segs


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rev2_bounded_merge_equals_the_frozen_merge(seed):
    from scipy.spatial import cKDTree
    from research.zeroshot.zs_lane_v2 import z3s as Z
    from research.zeroshot.zs_lane_v3 import z3_merge as M
    mesh = _l_bend_mesh()
    tree = cKDTree(mesh.V)
    segs = _segments(np.random.default_rng(seed))
    a = Z.merge_segments(segs, Z.Params(), mesh, tree, 1.0)
    b = M.merge_segments(segs, Z.Params(), mesh, tree, 1.0)
    dump = lambda x: json.dumps(x, sort_keys=True, default=lambda o: o.tolist())  # noqa: E731
    assert len(a) < len(segs)                       # merges happened
    assert dump(a) == dump(b)


def test_rev2_detect_is_the_frozen_detect_with_the_bounded_merge():
    import ast
    import inspect
    import textwrap
    from research.zeroshot.zs_lane_v2 import z3s as Z
    from research.zeroshot.zs_lane_v3 import z3_merge as M

    def body(f):
        stmts = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0].body
        if isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant):
            stmts = stmts[1:]                                   # the docstring
        return "\n".join(ast.unparse(x) for x in stmts).replace("Z.", "")
    # verbatim: the only difference is which module's merge_segments the name binds to
    assert body(M.detect) == body(Z.detect)
    assert "merge_segments(segs, prm, mesh, tree, t)" in body(M.detect)


def _write_ply(path, V, F):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    assert o3d.io.write_triangle_mesh(str(path), m, write_ascii=False)


def _two_skin(t=1.0, n=240):
    """A 90 deg bent sheet with both skins (t apart, opposite winding), n x n/2 grid per skin."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n // 2), indexing="ij")
    s, w = u.ravel() * 120.0 - 60.0, v.ravel() * 60.0

    def skin(off):
        return np.c_[np.where(s < 0, s, off), w, np.where(s < 0, -off, s)]
    idx = np.arange(u.size).reshape(u.shape)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F1 = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    V = np.r_[skin(0.0), skin(t)]
    return V, np.r_[F1, F1[:, ::-1] + u.size]


def test_rev3_budget_keeps_scans_within_it_unchanged(tmp_path):
    from research.zeroshot.zs_lane_v3 import budget as B
    V, F = _two_skin()
    p = tmp_path / "scan.ply"
    _write_ply(p, V, F)
    assert B.face_count(str(p)) == len(F)
    assert B.budget_scan(str(p), len(F), str(tmp_path)) == (str(p), {"applied": False, "faces": len(F), "max_faces": len(F)})
    assert B.budget_scan(str(p), None, str(tmp_path))[0] == str(p)


def test_rev3_budget_resamples_above_it_and_never_merges_the_two_skins(tmp_path):
    from research.zeroshot.zs_lane_v3 import budget as B
    V, F = _two_skin(t=0.6)
    p = tmp_path / "scan.ply"
    _write_ply(p, V, F)
    work = tmp_path / "w"
    work.mkdir()
    dst, info = B.budget_scan(str(p), len(F) // 4, str(work))
    assert info["applied"] and dst != str(p) and info["faces_out"] <= len(F) // 4
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(dst)
    X = np.asarray(m.vertices)
    web = X[(X[:, 2] > 10) & (X[:, 2] < 50)]               # the vertical flange: two sheets at x = 0 and x = t
    assert set(np.unique(np.round(web[:, 0], 3))) == {0.0, 0.6}
    flat = X[(X[:, 0] < -10) & (X[:, 0] > -50)]             # the horizontal flange: z = 0 and z = -t
    assert set(np.unique(np.round(flat[:, 2], 3))) == {0.0, -0.6}


def test_rev3_cell_is_fixed_and_laddered_only_to_meet_the_ceiling():
    from research.zeroshot.zs_lane_v3 import budget as B
    V, F = _two_skin(t=1.0)
    X, T, _N, info = B.decimate(V, F, 10 ** 9)
    assert info["cell_mm"] == B.CELL_MM == 0.30 and len(info["ladder"]) == 1     # no density term: the fixed cell
    assert np.all(np.abs(np.round(X / 1.0, 6)) < 1e6)
    cap = info["faces_out"] // 3
    X2, T2, _N2, info2 = B.decimate(V, F, cap)
    k = round(np.log(info2["cell_mm"] / B.CELL_MM) / np.log(B.LADDER))
    assert k >= 1 and abs(info2["cell_mm"] - round(B.CELL_MM * B.LADDER ** k, 5)) < 1e-9
    assert len(T2) <= cap
    assert 2 * B.crossings(V, F, B.CELL_MM * B.LADDER ** (k - 1), count_only=True) > cap    # the smallest k


def test_rev3_resampling_is_density_invariant_and_order_free():
    """The same surface sampled 1x and 2x as densely (and with shuffled vertex / face order) resamples to the same
    mesh: the output depends on the surface, not on its sampling."""
    from research.zeroshot.zs_lane_v3 import budget as B
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [17.0, -23.0, 31.0], degrees=True).as_matrix()     # a generic pose
    def one_skin(n):                            # the sharp skin; its corner (s = 0) is a grid vertex for n = 121, 241
        V, F = _two_skin(t=1.0, n=n)
        return V[: len(V) // 2], F[: len(F) // 2]
    V1, F1 = one_skin(121)
    V2, F2 = one_skin(241)
    V1, V2 = V1 @ R.T + [3.1, -7.7, 11.3], V2 @ R.T + [3.1, -7.7, 11.3]
    a = B.decimate(V1, F1, 10 ** 9)
    b = B.decimate(V2, F2, 10 ** 9)
    assert abs(a[3]["faces_out"] - b[3]["faces_out"]) <= 1e-3 * a[3]["faces_out"]
    assert abs(a[3]["vertices_out"] - b[3]["vertices_out"]) <= 1e-3 * a[3]["vertices_out"]
    d, _ = cKDTree(a[0]).query(b[0])
    assert np.quantile(d, 0.999) < 1e-6 and d.max() < 0.3            # the same vertices (a few at the sheet rims)
    V2, F2 = _two_skin(t=1.0, n=241)
    V2 = V2 @ R.T + [3.1, -7.7, 11.3]
    b = B.decimate(V2, F2, 10 ** 9)
    rng = np.random.default_rng(7)
    pv = rng.permutation(len(V2))
    inv = np.empty_like(pv)
    inv[pv] = np.arange(len(pv))
    F3 = inv[F2][rng.permutation(len(F2))]
    c = B.decimate(V2[pv], F3, 10 ** 9)
    assert np.array_equal(b[0], c[0]) and np.array_equal(b[1], c[1])
    d = B.decimate(V2, F2, 10 ** 9)
    assert np.array_equal(b[0], d[0]) and np.array_equal(b[1], d[1])                     # deterministic


def test_rev2_run_lane_v3_passes_the_pinned_budget(monkeypatch):
    """lane_cli is unchanged: it hands the pin's whole ``budget`` to core.configure, so the revision-3 pin's cell_mm
    reaches the budget step (re-pinned 2026-10-26; the revision-2 pin had none: the frozen budget.CELL_MM default)."""
    from domains.production.zeroshot import lane_cli
    from research.zeroshot.zs_lane_v3 import core
    pin = json.loads((REPO / "domains/production/zeroshot/assets/zs_lane_v3.pin.json").read_text())
    seen = {}

    def fake_process(path, *, thickness=None, max_families=None):
        seen["budget"] = core.BUDGET_MAX_FACES
        seen["cell"] = core.BUDGET_CELL_MM
        return {"status": "OK", "candidates": [], "stages": {}, "errors": {}}
    monkeypatch.setattr(core, "process", fake_process)
    monkeypatch.setattr(core, "select_lines", lambda rec, v, zs, z3: [])
    lane_cli.run_lane_v3(Path("scan.ply"), {"n_bends": 3, "t_mm": 2.0, "K": 1, "max_families": 3},
                         {"zs": None, "z3": None}, pin["lane_params"])
    assert seen["budget"] == pin["lane_params"]["budget"]["max_faces"] == 3_000_000 and seen["cell"] == 0.30
    cfg = json.loads((REPO / "research/zeroshot/zs_lane_v3/config_v3.json").read_text())
    assert pin["lane_params"]["budget"] == cfg["budget"]                     # the revision-3 pin carries the config's
    params = dict(pin["lane_params"], budget=cfg["budget"])
    lane_cli.run_lane_v3(Path("scan.ply"), {"n_bends": 3, "t_mm": 2.0, "K": 1, "max_families": 3},
                         {"zs": None, "z3": None}, params)
    assert seen["budget"] == 3_000_000 and seen["cell"] == 0.30
