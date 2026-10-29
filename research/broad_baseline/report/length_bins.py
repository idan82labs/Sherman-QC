"""Recall of each lane by gold-line length (read-only).

Loads the gold lines of every scored case in case_list_broad_v3.json through the
same loader the scorer used (score.load_gold, raw-scan frame) and splits each
lane's matched / unmatched gold ids by gold length. Gold-length bins are a
property of the label only, so both lanes are binned identically.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score as sc  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "docs/production/runlog/broad_baseline"
BINS = [(0, 30), (30, 60), (60, 120), (120, 250), (250, 1e9)]


def bin_name(lo, hi):
    return f"<{hi:g} mm" if lo == 0 else (f">={lo:g} mm" if hi >= 1e8 else f"{lo:g}-{hi:g} mm")


def gold_lengths() -> dict[str, dict[str, float]]:
    case_list = json.loads((OUT / "case_list_broad_v3.json").read_text())
    frames = json.loads((OUT / "broad_frame_records.json").read_text())
    out = {}
    for entry in case_list["cases"]:
        try:
            gold = sc.load_gold(entry["gold"], frame_records=frames)
        except Exception as exc:  # excluded cases are not scored anyway
            out[entry["case_id"]] = {"__error__": str(exc)}
            continue
        lens = {}
        for line in gold["lines"]:
            a, b = np.asarray(line["line_start"], float), np.asarray(line["line_end"], float)
            lens[line["bend_id"]] = float(np.linalg.norm(b - a))
        out[entry["case_id"]] = lens
    return out


def recall_by_length(lengths: dict[str, dict[str, float]]) -> dict:
    res = {}
    for lane in ("best_f1", "runtime"):
        cases = json.loads((OUT / f"scores_{lane}.json").read_text())["cases"]
        agg = defaultdict(lambda: {"gold": 0, "tp": 0})
        for c in cases:
            if c["disposition"] != "SCORED_COMPLETE_LABEL":
                continue
            lens = lengths.get(c["case_id"], {})
            matched = {m["gold_id"] for m in c.get("matches") or []}
            for gid, L in lens.items():
                if gid == "__error__":
                    continue
                for lo, hi in BINS:
                    if lo <= L < hi:
                        k = bin_name(lo, hi)
                        agg[k]["gold"] += 1
                        agg[k]["tp"] += int(gid in matched)
                        break
        res[lane] = {k: {**v, "recall": round(v["tp"] / v["gold"], 3) if v["gold"] else None}
                     for k, v in sorted(agg.items(), key=lambda kv: [bin_name(*b) for b in BINS].index(kv[0]))}
    return res


def fn_causes_by_length(lengths: dict[str, dict[str, float]], lane: str = "best_f1") -> dict:
    """Scorer's nearest-counterpart diagnostic for each missed gold line, by gold length (not the metric)."""
    cases = json.loads((OUT / f"scores_{lane}.json").read_text())["cases"]
    groups = [("<60 mm", 0, 60), ("60-250 mm", 60, 250), (">=250 mm", 250, 1e9)]
    agg = {g[0]: defaultdict(int) for g in groups}
    for c in cases:
        if c["disposition"] != "SCORED_COMPLETE_LABEL":
            continue
        for fn in (c.get("diagnostic") or {}).get("false_negatives", []):
            L = lengths[c["case_id"]].get(fn["gold_id"])
            for name, lo, hi in groups:
                if L is not None and lo <= L < hi:
                    agg[name][fn["cause"]] += 1
    return {k: dict(v) for k, v in agg.items()}


if __name__ == "__main__":
    L = gold_lengths()
    errs = {k: v["__error__"] for k, v in L.items() if "__error__" in v}
    print("cases", len(L), "errors", errs)
    print(json.dumps(recall_by_length(L), indent=1))
