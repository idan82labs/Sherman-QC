"""Scan-side left/right (chirality) check against a ``chirality_map_v0`` (E3 side, research candidate).

Self-contained (numpy + scipy only, no import from research/): the e3-1.15.0-rc candidate ships this file
verbatim as ``domains/production/engine/inspect/chirality.py``.

After E3 has registered the scan and fitted every panel (``PanelFit.A``: final pose -> registered frame), each
probe of the decoded stage's map is carried onto the scan by its owning panel's pose and tested for material:
a scan point within ``hit_mm``.

Per feature, per skin side (+1 / -1 of the owning panel's skin-0 normal):

- *proper-material* probes (the CAD has material, the mirror image has none) and *mirror-material* probes (the
  reverse) are PRESENT when at least ``present_fraction`` of them are hit, ABSENT when at most
  ``absent_fraction`` are hit **and** the ring around the feature (material both images share) is observed on the
  same side (``ring_coverage_min``); anything else is not evidence. Not scanned is never absent.
- proper votes: proper-material PRESENT, mirror-material ABSENT; mirror votes: the opposite.
- feature: PROPER (only proper votes), MIRROR (only mirror votes), CONFLICT (both) or UNOBSERVED (none).

Per mirror hypothesis (a place where a mirrored part would sit): PROPER when at least one feature is PROPER and
none is MIRROR or CONFLICT; MIRROR when at least one is MIRROR and none is PROPER or CONFLICT.

Decision:

- ``ACHIRAL``: the stage is its own mirror image (no check needed);
- ``PROPER_CONFIRMED``: every hypothesis is PROPER;
- ``MIRROR_SUSPECTED``: some hypothesis is MIRROR (its observed features all match that mirror image; another
  hypothesis reading PROPER only says the scan is not *that* mirror placement);
- ``UNCONFIRMED``: anything else (features unobserved, inconclusive or contradictory; stage not mapped; map
  missing or not matching the compiled part). The caller withholds every PASS.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = "chirality_map_v0"
CHECK_VERSION = "chirality_check_v0"

#: Frozen 2026-10-28 before any synthetic evaluation.
CHECK_THRESHOLDS: dict[str, float] = {
    "hit_mm": 0.8,               # a probe is covered by material when a registered scan point is this close
    "present_fraction": 0.5,
    "absent_fraction": 0.1,
    "ring_coverage_min": 0.5,    # absence needs the shared surroundings observed on the same side
    "min_side_probes": 4,        # a side's role/ring needs this many probes to count
}
CHECK_THRESHOLDS_SHA256 = hashlib.sha256(json.dumps(CHECK_THRESHOLDS, sort_keys=True).encode()).hexdigest()

ROLE_A, ROLE_B, ROLE_RING = 0, 1, 2


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_map(path: Path | str | None, compiled_part_sha256: str) -> tuple[dict | None, dict | None, str | None]:
    """(map, probes, problem). ``problem`` is a reason code when the map cannot be used."""
    if path is None:
        return None, None, "CHIRALITY_MAP_MISSING"
    path = Path(path)
    if not path.is_file():
        return None, None, "CHIRALITY_MAP_MISSING"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, None, "CHIRALITY_MAP_UNREADABLE"
    if doc.get("schema") != SCHEMA:
        return None, None, "CHIRALITY_MAP_UNREADABLE"
    if doc.get("compiled_part_sha256") != compiled_part_sha256:
        return None, None, "CHIRALITY_MAP_OTHER_PART"
    pz = path.parent / str(doc.get("probes_file") or "")
    if not pz.is_file() or _sha_file(pz) != doc.get("probes_sha256"):
        return None, None, "CHIRALITY_MAP_PROBES_MISMATCH"
    z = np.load(pz, allow_pickle=False)
    return doc, {k: z[k] for k in z.files}, None


def stage_record(doc: dict, stage_id: str | None) -> dict | None:
    for i, s in enumerate(doc["stages"]):
        if s["stage_model_id"] == stage_id:
            s = dict(s)
            s["_ordinal"] = i
            return s
    return None


def stage_class(doc: dict | None, stage_id: str | None) -> str | None:
    """The map's class of a stage; for an unmapped stage, ACHIRAL only when every mapped stage is ACHIRAL."""
    if doc is None:
        return None
    rec = stage_record(doc, stage_id)
    if rec is not None:
        return rec["class"]
    if doc["stages"] and all(s["class"] == "ACHIRAL" for s in doc["stages"]):
        return "ACHIRAL"
    return None


def _side_state(hit: np.ndarray, side: np.ndarray, ring_cov: dict[int, float | None], th: dict) -> str | None:
    states = set()
    for s in (1, -1):
        m = side == s
        if int(m.sum()) < th["min_side_probes"]:
            continue
        f = float(hit[m].mean())
        if f >= th["present_fraction"]:
            states.add("PRESENT")
        elif f <= th["absent_fraction"] and (ring_cov.get(s) or 0.0) >= th["ring_coverage_min"]:
            states.add("ABSENT")
    if states == {"PRESENT"}:
        return "PRESENT"
    if states == {"ABSENT"}:
        return "ABSENT"
    if states:
        return "CONFLICT"
    return None


def evaluate(doc: dict | None, probes: dict | None, stage_id: str | None, panel_pose: dict[int, np.ndarray | None],
             reg_pts: np.ndarray, th: dict = CHECK_THRESHOLDS, problem: str | None = None) -> dict:
    """Left/right decision of a registered scan (module docstring). ``panel_pose[i]``: final -> registered 4x4 of
    compiled panel ``i`` (None: the panel was not fitted)."""
    out = {"version": CHECK_VERSION, "thresholds_sha256": hashlib.sha256(json.dumps(th, sort_keys=True).encode()).hexdigest(),
           "stage_model_id": stage_id, "decision": "UNCONFIRMED", "reason": problem, "class": None, "hypotheses": []}
    if doc is None or probes is None:
        out["reason"] = problem or "CHIRALITY_MAP_MISSING"
        return out
    out["map_sha256"] = doc.get("map_sha256")
    rec = stage_record(doc, stage_id)
    if rec is None:
        if stage_class(doc, stage_id) == "ACHIRAL":
            out.update(decision="ACHIRAL", reason="ALL_STAGES_ACHIRAL", **{"class": "ACHIRAL"})
        else:
            out["reason"] = "CHIRALITY_STAGE_UNMAPPED"
        return out
    out["class"] = rec["class"]
    if rec["class"] == "ACHIRAL":
        out.update(decision="ACHIRAL", reason="STAGE_ACHIRAL")
        return out
    sel = probes["stage"] == rec["_ordinal"]
    pos, own, side = probes["pos"][sel].astype(np.float64), probes["owner"][sel], probes["side"][sel]
    hyp, feat, role = probes["hyp"][sel], probes["feature"][sel], probes["role"][sel]
    # carry every probe onto the scan with its panel's fitted pose
    reg = np.full_like(pos, np.nan)
    for i in np.unique(own):
        A = panel_pose.get(int(i))
        if A is None:
            continue
        m = own == i
        reg[m] = pos[m] @ np.asarray(A)[:3, :3].T + np.asarray(A)[:3, 3]
    ok = np.isfinite(reg).all(axis=1)
    hit = np.zeros(len(pos), bool)
    if ok.any() and len(reg_pts):
        lo, hi = reg[ok].min(axis=0) - 2.0, reg[ok].max(axis=0) + 2.0
        near = np.all((reg_pts >= lo) & (reg_pts <= hi), axis=1)
        if near.any():
            d, _ = cKDTree(reg_pts[near]).query(reg[ok], distance_upper_bound=th["hit_mm"])
            hit[np.where(ok)[0]] = np.isfinite(d)
    verdicts = []
    for h in rec["hypotheses"]:
        hi_ = h["hypothesis"]
        frec = []
        for f in h["features"]:
            fi = f["feature"]
            base = (hyp == hi_) & (feat == fi) & ok
            ring = base & (role == ROLE_RING)
            ring_cov = {}
            for s in (1, -1):
                m = ring & (side == s)
                ring_cov[s] = float(hit[m].mean()) if int(m.sum()) >= th["min_side_probes"] else None
            st = {}
            for r, name in ((ROLE_A, "proper_material"), (ROLE_B, "mirror_material")):
                m = base & (role == r)
                st[name] = _side_state(hit[m], side[m], ring_cov, th) if m.any() else None
            pv = int(st["proper_material"] == "PRESENT") + int(st["mirror_material"] == "ABSENT")
            mv = int(st["proper_material"] == "ABSENT") + int(st["mirror_material"] == "PRESENT")
            confl = "CONFLICT" in (st["proper_material"], st["mirror_material"])
            if confl or (pv and mv):
                fs = "CONFLICT"
            elif pv:
                fs = "PROPER"
            elif mv:
                fs = "MIRROR"
            else:
                fs = "UNOBSERVED"
            frec.append({"feature": fi, "type": f.get("type"), "panel_id": f.get("panel_id"), "state": fs,
                         "proper_material": st["proper_material"], "mirror_material": st["mirror_material"],
                         "ring_coverage": {str(k): (None if v is None else round(v, 3)) for k, v in ring_cov.items()}})
        n = {k: sum(1 for x in frec if x["state"] == k) for k in ("PROPER", "MIRROR", "CONFLICT", "UNOBSERVED")}
        if n["PROPER"] and not n["MIRROR"] and not n["CONFLICT"]:
            v = "PROPER"
        elif n["MIRROR"] and not n["PROPER"] and not n["CONFLICT"]:
            v = "MIRROR"
        elif n["CONFLICT"] or (n["PROPER"] and n["MIRROR"]):
            v = "CONFLICT"
        else:
            v = "UNOBSERVED"
        verdicts.append(v)
        out["hypotheses"].append({"hypothesis": hi_, "verdict": v, "counts": n, "features": frec})
    if verdicts and all(v == "PROPER" for v in verdicts):
        out.update(decision="PROPER_CONFIRMED", reason="FEATURES_CONFIRM_PROPER")
    elif any(v == "MIRROR" for v in verdicts):
        out.update(decision="MIRROR_SUSPECTED", reason="FEATURES_MATCH_MIRROR")
    else:
        bad = sorted(set(verdicts) - {"PROPER"})
        out.update(decision="UNCONFIRMED", reason="FEATURES_" + ("_".join(bad) if bad else "NONE"))
    return out
