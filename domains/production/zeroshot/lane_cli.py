"""Runner zs step: the pinned zs_lane_v1 with CAD priors on one staged scan -> ``zeroshot/zs_lines.json``.

Usage (runner, cwd = engine bundle, alongside the E3 step)::

    python -m domains.production.zeroshot.lane_cli --scan <ws>/input/scan.<ext>
        --compiled <ws>/compiled/compiled_part.json --out <ws>/zeroshot/zs_lines.json

Blindness: the only inputs are the staged scan (its file name must be ``scan.ply`` / ``scan.stl``, the
workspace convention, so no original file name, serial or job token can reach the lane; anything else is
refused) and rotation-invariant CAD facts read from the CompiledPart: the full-stage bend count, thickness and
the number K of distinct bend-axis directions. No registration, no CAD pose, no CAD lines, no recipe, no
context.

Pipeline (the frozen lane, ``research/zeroshot/zs_lane_v1/core.py``, imported read-only after the pin check):
Z3 axis-family slicing with the CAD thickness and the K + 2 family cap -> Z2 scan-derived flange graph -> pooling
-> the pinned GBM selector at the frozen threshold 0.5 -> count cap (at most n_bends lines, highest probability
first) -> D1 gated outer-apex snap with the CAD thickness. No angle window (CAD_PATH.md option 3).

Exit codes: 0 = ``status OK`` written; 1 = a non-OK status written (PIN_MISMATCH, REFUSED_INPUT, UNPINNED_CODE,
FAILED); 3 = internal error before anything could be written. Every outcome other than exit 0 means "no
overlay" to the runner; it is never a job failure.

Pin selection (``pin.selected_lane``): the default pin is zs_lane_v1 (the pipeline above). With the zs_lane_v3 pin
selected explicitly, :func:`run_lane_v3` runs instead: zs_lane_v2's pipeline (``research/zeroshot/zs_lane_v3/core.py``
reusing the frozen v2 modules: Z3 with the CAD thickness and the K + 2 family cap, Z2 with its fallback, pooling,
D1 snap of every candidate with the CAD thickness, v2 evidence, the pooled GBM at 0.5, v2's span post-processing),
then EXT and Z3R with the pinned parameters, then the same count cap (highest line score first; a Z3R line keeps
its own pooled probability, so it is the first to go under the cap). zs_lane_v3 revision 2 adds the pinned face
budget (``lane_params.budget``: a scan with more triangles runs on a decimated copy of itself, so the step stays
within the runner's 90 s / 4 GiB) and the exact bounded Z3 segment merge; Z2 runs before Z3.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from domains.production.zeroshot import LANE_ID, LINES_SCHEMA, ZS_RUNTIME_VERSION
from domains.production.zeroshot import pin as PIN

SCAN_NAME_RE = re.compile(r"^scan\.(ply|stl)$")
AXIS_SAME_DEG = 3.0          # CAD bend axes closer than this are one direction (cad_prior.cad_facts)
MAX_FAMILIES = 20            # Z3 default family cap; the prior lowers it to K + 2

__all__ = ["cad_facts", "run_lane", "run_lane_v3", "main"]


def cad_facts(compiled_json: dict[str, Any]) -> dict[str, Any]:
    """Rotation-invariant CAD priors from a CompiledPart document (no pose, no lines)."""
    bends = [b for b in compiled_json.get("bends") or [] if b.get("supported", True)]
    if not bends:
        raise ValueError("the CompiledPart has no supported bends")
    axes: list[tuple[float, float, float]] = []
    for b in bends:
        d = [float(x) for x in b["axis"]["direction"]]
        n = math.sqrt(sum(x * x for x in d))
        u = tuple(x / n for x in d)
        if not any(abs(sum(p * q for p, q in zip(u, a))) > math.cos(math.radians(AXIS_SAME_DEG)) for a in axes):
            axes.append(u)
    t = float(compiled_json["thickness"]["t_mm"])
    return {"n_bends": len(bends), "t_mm": t, "K": len(axes), "max_families": min(MAX_FAMILIES, len(axes) + 2)}


def _rss_gib() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**30 if sys.platform == "darwin" else rss / 2**20


def run_lane(scan: Path, facts: dict[str, Any], model: Any) -> dict[str, Any]:
    """The frozen lane with CAD priors (module docstring). Returns lines in the raw scan frame."""
    import numpy as np

    from research.zeroshot.z3_slicing import slicer
    from research.zeroshot.zs_lane_v1 import core

    stages: dict[str, Any] = {}
    errors: dict[str, str] = {}
    t0 = time.time()
    z3 = None
    mesh = None
    try:
        mesh = slicer.load_mesh(str(scan))
        prm = slicer.Params(max_families=int(facts["max_families"]))
        z3 = slicer.detect(mesh, prm, thickness={"t_mm": float(facts["t_mm"]), "source": "CAD"})
        z3["_mesh"] = mesh
        stages["z3"] = {"wall": round(time.time() - t0, 3), "n_candidates": len(z3["candidates"]),
                        "families": len(z3["families"])}
    except Exception as exc:  # noqa: BLE001 - a failed stage is recorded, as in the frozen lane
        errors["z3"] = f"{type(exc).__name__}: {exc}"[:500]
    z2_lines = None
    t1 = time.time()
    try:
        z2_lines, _scan, info = core.run_z2(str(scan))
        stages["z2"] = {"wall": round(time.time() - t1, 3), "n_lines": len(z2_lines), "t_mm_scan": round(info["t_mm"], 3)}
    except Exception as exc:  # noqa: BLE001
        errors["z2"] = f"{type(exc).__name__}: {exc}"[:500]
    if z3 is None and z2_lines is None:
        return {"status": "FAILED", "lines": [], "stages": stages, "errors": errors, "n_candidates": 0,
                "n_selected": 0}
    cands, X = core.pool(z3, z2_lines, True)
    stages["pool"] = {"n": len(cands)}
    prob = model["model"].predict(X) if len(cands) else np.zeros(0)
    thr = float(model.get("threshold", 0.5))
    keep = sorted(((float(q), c) for c, q in zip(cands, prob) if q >= thr), key=lambda x: -x[0])
    n_selected = len(keep)
    keep = keep[: int(facts["n_bends"])]                       # count cap at full stage
    lines = []
    for q, c in keep:
        c = dict(c)
        c["score"] = q
        lines.append(c)
    pts = None
    if mesh is not None:
        pts = (np.asarray(mesh.V, float), np.asarray(mesh.vn, float), float(facts["t_mm"]))
    t2 = time.time()
    core.snap(lines, pts)
    stages["snap"] = {"wall": round(time.time() - t2, 3), "n": len(lines),
                      "moved": sum(1 for c in lines if c.get("snap", {}).get("moved"))}
    out = []
    for i, c in enumerate(lines):
        out.append({
            "line_id": f"ZS{i + 1:02d}",
            "start_mm": [float(v) for v in c.get("start_s", c["start"])],
            "end_mm": [float(v) for v in c.get("end_s", c["end"])],
            "incl_deg": None if c.get("incl") is None else float(c["incl"]),
            "score": round(float(c["score"]), 6),
            "src": list(c.get("src") or []),
            "snap_moved": bool(c.get("snap", {}).get("moved")),
        })
    return {"status": "OK", "lines": out, "stages": stages, "errors": errors, "n_candidates": len(cands),
            "n_selected": n_selected, "threshold": thr}


def run_lane_v3(scan: Path, facts: dict[str, Any], models: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """The pinned zs_lane_v3 with CAD priors (module docstring). Returns lines in the raw scan frame."""
    from research.zeroshot.zs_lane_v3 import core

    t0 = time.time()
    cfg = {"variants": {"zs": params["variant"]}, "z3_params": params.get("z3_params"),
           "z2_fallback": bool(params.get("z2_fallback")), "budget": params.get("budget")}
    core.configure(cfg, "zs")
    rec = core.process(str(scan), thickness={"t_mm": float(facts["t_mm"]), "source": "CAD"},
                       max_families=int(facts["max_families"]))
    stages = {k: v for k, v in rec.get("stages", {}).items()}
    errors = {k: str(v)[:500] for k, v in rec.get("errors", {}).items() if not k.endswith("_trace")}
    if rec.get("status") != "OK":
        return {"status": "FAILED", "lines": [], "stages": stages, "errors": errors, "n_candidates": 0,
                "n_selected": 0}
    t1 = time.time()
    sel = core.select_lines(rec, params["variant"], models["zs"], models.get("z3"))
    n_selected = len(sel)
    sel = sorted(sel, key=lambda c: -float(c["score"]))[: int(facts["n_bends"])]      # count cap at full stage
    stages["select_post_fix"] = {"wall": round(time.time() - t1, 3), "n": len(sel),
                                 "fixes": {k: sum(1 for c in sel if c.get("fix") == k) for k in ("EXT", "Z3R")}}
    out = []
    for i, c in enumerate(sel):
        out.append({
            "line_id": f"ZS{i + 1:02d}",
            "start_mm": [float(v) for v in c["start_s"]],
            "end_mm": [float(v) for v in c["end_s"]],
            "incl_deg": None if c.get("incl") is None else float(c["incl"]),
            "score": round(float(c["score"]), 6),
            "src": list(c.get("src") or []),
            "op": str(c.get("op") or "keep"),
            "fix": c.get("fix"),
        })
    stages["lane_wall"] = round(time.time() - t0, 3)
    return {"status": "OK", "lines": out, "stages": stages, "errors": errors, "n_candidates": len(rec["candidates"]),
            "n_selected": n_selected, "threshold": float(params["variant"]["threshold"])}


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m domains.production.zeroshot.lane_cli")
    ap.add_argument("--scan", required=True, type=Path)
    ap.add_argument("--compiled", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    t0 = time.time()
    root = PIN.root_of_package()
    doc: dict[str, Any] = {"schema": LINES_SCHEMA, "lane": LANE_ID, "runtime_version": ZS_RUNTIME_VERSION,
                           "status": "FAILED", "lines": [], "errors": {}}
    try:
        pin = PIN.load_pin(root)
        doc.update(lane=pin["lane"], code_fingerprint=pin["code_fingerprint"]["sha256"],
                   model_sha256=pin["model"]["sha256"], variant=pin["variant"])
        if pin.get("extra_models"):
            doc["extra_model_sha256"] = {k: m["sha256"] for k, m in pin["extra_models"].items()}
        if not SCAN_NAME_RE.match(a.scan.name):
            doc.update(status="REFUSED_INPUT", errors={"input": "the scan must be the staged input/scan.<ply|stl>"})
            _write(a.out, doc)
            return 1
        bad = PIN.verify(root, pin)
        if bad:
            doc.update(status="PIN_MISMATCH", pin_mismatch=bad[:50])
            _write(a.out, doc)
            print(f"zs: PIN_MISMATCH {len(bad)} ({', '.join(bad[:3])})", flush=True)
            return 1
        doc["scan_sha256"] = PIN.sha256_file(a.scan)
        facts = cad_facts(json.loads(a.compiled.read_text(encoding="utf-8")))
        doc["priors"] = facts
        import pickle

        models = {k: pickle.loads((root / m["rel_path"]).read_bytes())  # sha256 verified above
                  for k, m in PIN.pinned_models(pin).items()}
        if pin["lane"] == "zs_lane_v3":
            res = run_lane_v3(a.scan, facts, models, pin["lane_params"])
        else:
            res = run_lane(a.scan, facts, models["zs"])
        doc.update(res)
        unpinned = PIN.unpinned_loaded(root, pin)
        if unpinned:
            doc.update(status="UNPINNED_CODE", unpinned_loaded=unpinned[:50], lines=[])
    except Exception as exc:  # noqa: BLE001 - contained: the runner treats it as "no overlay"
        doc.update(status="FAILED", lines=[])
        doc["errors"] = {**doc.get("errors", {}), "lane": f"{type(exc).__name__}: {exc}"[:500],
                         "trace": traceback.format_exc()[-1500:]}
    doc["wall_s"] = round(time.time() - t0, 3)
    doc["peak_rss_gib"] = round(_rss_gib(), 3)
    try:
        _write(a.out, doc)
    except OSError:
        traceback.print_exc()
        return 3
    print(f"zs: {doc['status']} lines={len(doc['lines'])} wall={doc['wall_s']:.1f}s rss={doc['peak_rss_gib']:.2f}G",
          flush=True)
    return 0 if doc["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
