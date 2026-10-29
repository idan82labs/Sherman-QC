"""Jev input optimisation on the broad benchmark (v2, research only).

Follow-up to ``broad_ab_v1`` (docs/production/runlog/jev_ab/JEV_AB.md). It changes
what Jev is given, and compares against plain-feature baselines on the same input:

- **Question** ``is_real_bend`` v2 (``questions_v2``): the owner's conventions
  (formed press-brake fold at the outer-skin apex; a joggle is two bends; small
  15-30 deg formed bends count; hole walls, thickness/perimeter edges, embosses,
  gentle rolled kinks and artifacts do not).
- **Evidence** (``states_v2``): typed measurements with units and one-line
  meanings, in three input designs.
- **Candidates**: the old lane's final lines (filter arms) and the lane's whole
  scan-only candidate pool, clustered (liberal arms; recall can rise).

Subcommands (all offline except ``jev --live``)::

  build   heavy (scan reads; run under the geometry lock): candidates, features,
          labels -> OUT/candidates.jsonl (+ per-scan feature cache)
  states  -> OUT/states_<design>.jsonl (no coordinates, no ids)
  jev     dry-run (default) or --live; cached by body sha
  score   classifiers, thresholds (calib only), arms, CIs -> results.json

Truth: case-list gold (accepted AI gold ``ai_bend_lines_v1`` and the human
non-Gold11 lines), with the 11 Gold11 scans overridden by the vision relabel
``ai_bend_lines_gold11_relabel_v1`` (matched by scan sha256). ``--gold-set``
optionally adds a second truth (``ai_bend_lines_gold_v1``) when it exists.
Never wired into a runtime decision.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research.jev.export_onboarding_v1 import REPO_ROOT
from research.jev.pool_v2 import (
    DEFAULT_WORK_ROOT,
    cluster_lines,
    load_pool_lines,
    pair_metrics_matrix,
    pool_file,
)

BROAD = REPO_ROOT / "docs" / "production" / "runlog" / "broad_baseline"
CASE_LIST = BROAD / "case_list_broad_v3.json"
FRAME_RECORDS = BROAD / "broad_frame_records.json"
RUNS = BROAD / "runs"
RELABEL_DIR = REPO_ROOT / "data" / "bend_corpus" / "annotations" / "ai_bend_lines_gold11_relabel_v1"
GOLD_SET_DIR = REPO_ROOT / "data" / "bend_corpus" / "annotations" / "ai_bend_lines_gold_v1"
OLD_SPLITS = REPO_ROOT / "docs" / "production" / "runlog" / "jev_ab" / "export" / "splits.json"
LANE = "best_f1"
LANE_FILE = "predictions_best_f1.json"


def _score_module():
    p = str(REPO_ROOT / "research" / "broad_baseline")
    if p not in sys.path:
        sys.path.insert(0, p)
    import score as sc  # noqa: WPS433

    return sc


def _read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Truth
# --------------------------------------------------------------------------


def label_dir_by_sha(directory: Path) -> dict[str, Path]:
    """``{asset_sha256: label file}`` for an ai_studio-style label directory (via provenance/)."""
    out: dict[str, Path] = {}
    for prov in sorted(glob.glob(str(Path(directory) / "provenance" / "*.json"))):
        sha = _read(prov).get("asset_sha256")
        label = Path(directory) / Path(prov).name
        if sha and label.is_file():
            out[sha] = label
    return out


def truth_specs(case_list: Mapping[str, Any], truth: str) -> dict[str, dict[str, Any]]:
    """``{case_id: gold spec}`` for a truth name.

    ``ai_relabel``: case-list gold with Gold11 cases replaced by the vision relabel.
    ``gold_set_v1``: ``ai_bend_lines_gold_v1`` where it has the scan, else ``ai_relabel``.
    ``human_gold11``: the case-list gold as is (Gold11 = human catalog).
    """
    relabel = label_dir_by_sha(RELABEL_DIR)
    out = {}
    for c in case_list["cases"]:
        spec = c["gold"]
        if truth in ("ai_relabel", "gold_set_v1") and c.get("tier") == "GOLD11":
            if c["scan_sha256"] not in relabel:
                raise SystemExit(f"Gold11 relabel missing for {c['case_id']}")
            spec = {"kind": "ai_studio", "path": str(relabel[c["scan_sha256"]].relative_to(REPO_ROOT))}
        out[c["case_id"]] = spec
    if truth == "gold_set_v1":
        gs = label_dir_by_sha(GOLD_SET_DIR) if GOLD_SET_DIR.is_dir() else {}
        for c in case_list["cases"]:
            if c["scan_sha256"] in gs:
                out[c["case_id"]] = {"kind": "ai_studio", "path": str(gs[c["scan_sha256"]].relative_to(REPO_ROOT))}
    return out


def load_truth(case_list, truth: str) -> dict[str, dict[str, Any]]:
    sc = _score_module()
    fr = _read(FRAME_RECORDS)
    out = {}
    for cid, spec in truth_specs(case_list, truth).items():
        g = sc.load_gold(spec, frame_records=fr)
        out[cid] = {"lines": g["lines"], "zero_bend_control": bool(g["zero_bend_control"]),
                    "source": g.get("gold_source")}
    return out


# --------------------------------------------------------------------------
# Matching / neighbour context
# --------------------------------------------------------------------------


def match_mask(m: Mapping[str, np.ndarray]) -> np.ndarray:
    """Frozen matcher criteria (5 deg, 4 mm, 0.6 / 0.6 overlap), all pairs."""
    return ((m["axis_angle_deg"] <= 5.0) & (m["transverse_offset_mm"] <= 4.0)
            & (m["gold_span_overlap"] >= 0.6) & (m["prediction_span_overlap"] >= 0.6))


def neighbour_context(ends: np.ndarray, ref_ends: np.ndarray, self_in_ref: Sequence[int | None]) -> list[dict]:
    """Coordinate-free context of each line against the scan's pool lines."""
    if len(ref_ends) == 0 or len(ends) == 0:
        return [{} for _ in range(len(ends))]
    m = pair_metrics_matrix(ends, ref_ends)
    ang, off = m["axis_angle_deg"], m["transverse_offset_mm"]
    ov = m["shorter_span_overlap"]
    a = ends[:, 0, :]
    out = []
    for i in range(len(ends)):
        not_self = np.ones(len(ref_ends), bool)
        if self_in_ref[i] is not None:
            not_self[self_in_ref[i]] = False
        par = not_self & (ang[i] <= 5.0) & (ov[i] >= 0.5)
        near_dup = par & (off[i] <= 4.0)
        skin = par & (off[i] > 0.4) & (off[i] <= 6.0)
        fam = par & (off[i] > 6.0) & (off[i] <= 40.0)
        # collinear partners: same carrier line, not overlapping, gap <= 80 mm (window / relief)
        col = not_self & (ang[i] <= 3.0) & (off[i] <= 3.0) & (m["gold_span_overlap"][i] <= 0.05)
        gaps = []
        if col.any():
            d = ends[i, 1] - ends[i, 0]
            L = np.linalg.norm(d)
            ax = d / L
            for j in np.flatnonzero(col):
                s = sorted([float((ref_ends[j, 0] - a[i]) @ ax), float((ref_ends[j, 1] - a[i]) @ ax)])
                gap = s[0] - L if s[0] >= L else (-s[1] if s[1] <= 0 else 0.0)
                gaps.append(gap)
        gaps = [g for g in gaps if 0.0 <= g <= 80.0]
        skin_off = off[i][skin]
        out.append({
            "near_duplicate_count": int(near_dup.sum()),
            "parallel_within_6mm_count": int(skin.sum()),
            "nearest_parallel_offset_mm": float(skin_off.min()) if len(skin_off) else None,
            "parallel_6_to_40mm_count": int(fam.sum()),
            "collinear_partner_count": int(len(gaps)),
            "nearest_collinear_gap_mm": float(min(gaps)) if gaps else None,
        })
    return out


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _lane_rows(cid: str) -> list[dict[str, Any]]:
    p = RUNS / LANE / cid / LANE_FILE
    if not p.is_file():
        return []
    return [r for r in _read(p).get("owned_bend_regions") or [] if r.get("line_start") and r.get("line_end")]


def _fkey(a, b) -> str:
    return "|".join(f"{v:.3f}" for v in [*a, *b])


def cmd_build(args: argparse.Namespace) -> int:
    from research.jev.features_v2 import FEATURE_VERSION, line_features_v2, load_clouds

    case_list = _read(args.case_list)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "features_cache"
    cache_dir.mkdir(exist_ok=True)
    splits = _read(args.splits)["families"]
    rows = []
    only = set(args.only or [])
    t0 = time.time()
    for c in case_list["cases"]:
        cid = c["case_id"]
        if only and cid not in only:
            continue
        pf = pool_file(cid, Path(args.work_root))
        lines, meta = load_pool_lines(pf) if pf else ([], {})
        converted = False
        if meta and meta.get("scan_sha256") not in (None, c["scan_sha256"]):
            # the pool stage converts STL/OBJ scans to PLY first (same coordinates, new bytes)
            if c.get("format") == "ply":
                raise SystemExit(f"{cid}: pool scan sha mismatch on a PLY scan")
            converted = True
        clusters = cluster_lines(lines)
        pool_ends = np.array([[lines[x["rep"]].start, lines[x["rep"]].end] for x in clusters]).reshape(-1, 2, 3)
        lane = _lane_rows(cid)
        lane_ends = np.array([[r["line_start"], r["line_end"]] for r in lane], float).reshape(-1, 2, 3)
        cache_file = cache_dir / f"{cid}.json"
        cache = _read(cache_file) if cache_file.is_file() else {}
        if cache.get("__version__") not in (None, FEATURE_VERSION):
            cache = {}
        cache["__version__"] = FEATURE_VERSION
        clouds = None
        need = [e for e in list(pool_ends) + list(lane_ends) if _fkey(e[0], e[1]) not in cache]
        if need:
            clouds = load_clouds(Path(c["scan_path"]))
            cache["__scan__"] = {"fine_spacing_mm": clouds.fine_spacing, "coarse_spacing_mm": clouds.coarse_spacing,
                                 "extent_mm": clouds.extent, "points": int(len(clouds.fine))}
            for e in need:
                cache[_fkey(e[0], e[1])] = line_features_v2(clouds, e[0], e[1])
            cache_file.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
        scan_meta = cache.get("__scan__", {})
        pool_ctx = neighbour_context(pool_ends, pool_ends, list(range(len(pool_ends))))
        lane_ctx = neighbour_context(lane_ends, pool_ends, [None] * len(lane_ends))
        # lane lines inherit pool-generator descriptors from the pool cluster they duplicate (if any)
        lane_to_pool = [None] * len(lane_ends)
        if len(lane_ends) and len(pool_ends):
            mm = pair_metrics_matrix(lane_ends, pool_ends)
            ok = match_mask(mm)
            cost = mm["axis_angle_deg"] / 5 + mm["transverse_offset_mm"] / 4
            for i in range(len(lane_ends)):
                if ok[i].any():
                    lane_to_pool[i] = int(np.argmin(np.where(ok[i], cost[i], 1e9)))

        def pool_block(k: int | None) -> dict[str, Any]:
            if k is None:
                return {}
            x = clusters[k]
            rep = lines[x["rep"]]
            radii = [lines[j].radius_mm for j in x["members"] if lines[j].radius_mm is not None]
            return {"pool_member_count": x["member_count"], "pool_generator_count": len(x["generators"]),
                    "pool_generator_kind": "+".join(x["generators"]),
                    "pool_radius_mm": float(np.median(radii)) if radii else None,
                    "pool_radius_votes_count": len(radii),
                    **{f"pool_{key}": val for key, val in rep.descriptors.items()}}

        common = {"case_id": cid, "family_id": c["family_id"], "tier": c["tier"],
                  "split": splits[c["family_id"]]["split"], "scan_sha256": c["scan_sha256"],
                  "scan_extent_mm": scan_meta.get("extent_mm"), "pool_size_count": len(pool_ends),
                  "lane_size_count": len(lane_ends),
                  "pool_scan_converted_from_mesh": converted}
        for k, e in enumerate(pool_ends):
            rows.append({**common, "set": "pool", "cand": f"P{k}", "line_start": e[0].tolist(),
                         "line_end": e[1].tolist(), "feat": cache[_fkey(e[0], e[1])], "ctx": pool_ctx[k],
                         "pool": pool_block(k), "in_lane": False})
        for k, (e, r) in enumerate(zip(lane_ends, lane)):
            rows.append({**common, "set": "lane", "cand": str(r["bend_id"]), "line_start": e[0].tolist(),
                         "line_end": e[1].tolist(), "feat": cache[_fkey(e[0], e[1])], "ctx": lane_ctx[k],
                         "pool": pool_block(lane_to_pool[k]), "in_lane": True,
                         "lane_proposal_kind": "SCAN_ONLY" if r.get("fusion_source") == "scan_only_a0" else "CAD_GUIDED"})
        print(f"{cid}: pool {len(pool_ends)} lane {len(lane_ends)} computed {len(need)} "
              f"t={time.time() - t0:.0f}s", flush=True)
        del clouds
    path = out / ("candidates.jsonl" if not only else "candidates_partial.jsonl")
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({"out": str(path), "rows": len(rows), "seconds": round(time.time() - t0, 1)}))
    return 0


def read_candidates(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _key_from_env_file(path: Path) -> dict[str, str]:
    """Read TYPESAFE_API_KEY from a dotenv file into a private env mapping (never printed or written)."""
    import os

    env = dict(os.environ)
    if path and Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                env["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                break
    return env


def select_rows(rows, *, split: str, cand_set: str) -> list[dict[str, Any]]:
    out = [r for r in rows if (split == "all" or r["split"] == split)
           and (cand_set == "all" or r["set"] == cand_set)]
    return out


def cmd_jev(args: argparse.Namespace) -> int:
    from research.jev.runner_v2 import dry_run, live_run
    from research.jev.states_v2 import build_state_v2, record_id_v2

    if args.split in ("eval", "all") and not Path(args.design_choice).is_file():
        raise SystemExit("test-family calls need the frozen design choice file first (--design-choice)")
    rows = select_rows(read_candidates(Path(args.build) / "candidates.jsonl"), split=args.split,
                       cand_set=args.set)
    variant = "composite_v2" if args.design == "composite" else "v2"
    states, seen = [], set()
    for r in rows:
        st = build_state_v2(r, args.design)
        rid = record_id_v2(r)
        if rid not in seen:
            seen.add(rid)
            states.append((rid, st))
    if args.limit:
        states = states[: args.limit]
    out = Path(args.answers) / args.design
    summary = dry_run(states, out / f"dry_run_{args.split}_{args.set}", model=args.model, variant=variant)
    print(json.dumps(summary))
    if not args.live:
        return 0
    env = _key_from_env_file(Path(args.env_file)) if args.env_file else None
    res = live_run(states, out / "cache", live=True, model=args.model, variant=variant,
                   budget_usd=args.budget, workers=args.workers, env=env)
    log = out / "live_log.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"split": args.split, "set": args.set, "time": time.time(), **res}) + "\n")
    print(json.dumps(res))
    return 0 if res["error"] is None else 3


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m research.jev.input_opt_v2")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--case-list", default=str(CASE_LIST))
    b.add_argument("--splits", default=str(OLD_SPLITS))
    b.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    b.add_argument("--out", required=True)
    b.add_argument("--only", action="append")
    j = sub.add_parser("jev")
    j.add_argument("--build", required=True)
    j.add_argument("--answers", required=True)
    j.add_argument("--design", choices=("typed", "typed_reading", "composite"), required=True)
    j.add_argument("--split", choices=("calib", "eval", "all"), default="calib")
    j.add_argument("--set", choices=("pool", "lane", "all"), default="all")
    j.add_argument("--model", default="jev-1.13.0")
    j.add_argument("--design-choice", default="")
    j.add_argument("--live", action="store_true")
    j.add_argument("--env-file", default="")
    j.add_argument("--budget", type=float, default=5.0)
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("--limit", type=int, default=0, help="send only the first N records (smoke test)")
    args = p.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    if args.command == "jev":
        return cmd_jev(args)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
