"""Reproduction check of the v4 candidate cache against the stored pinned outputs (no geometry).

For every cached pool scan: the pinned zs_lane_v1 selection replayed on cache["v1"] must equal the stored v1 lines;
the pinned zs_lane_v3 selection replayed on cache["v4"] (scans within the face budget) or cache["v3x"] (budgeted
scans, v1-job thickness) must equal the stored v3 lines. Writes results/CACHE_VALIDATION.json.
usage (repo root): .venv/bin/python -B docs/production/runlog/zs_v4/tools/validate_cache.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v4 import cache_io as CI, evalkit as EK  # noqa: E402


def main() -> None:
    pool = EK.load_pool()
    rows, ok1, ok3, n = [], 0, 0, 0
    for s in pool:
        d = CI.load(s["case_id"])
        if d is None:
            continue
        n += 1
        st1 = EK.pinned_lines(s, "zs_lane_v1") or []
        st3 = EK.pinned_lines(s, "zs_lane_v3") or []
        r1 = CI.v1_pinned_lines(d)
        rv = d.get("v3x") if d.get("v3x") else d["v4"]
        r3 = CI.v3_lines(rv) if rv.get("status") == "OK" else []
        e1, e3 = CI.same_lines(r1, st1), CI.same_lines(r3, st3)
        ok1 += e1; ok3 += e3
        if not (e1 and e3):
            rows.append({"case_id": s["case_id"], "v1_equal": e1, "v3_equal": e3, "n_v1": [len(r1), len(st1)],
                         "n_v3": [len(r3), len(st3)], "budgeted": bool(d.get("v3x"))})
    out = {"cached": n, "v1_equal": ok1, "v3_equal": ok3, "mismatches": rows}
    (EK.V4RUN / "results/CACHE_VALIDATION.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("cached", "v1_equal", "v3_equal")}), len(rows))
    for r in rows[:40]:
        print(r)


if __name__ == "__main__":
    main()
