from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as rung


def test_cross_parent_rung_caps_are_frozen() -> None:
    assert rung.HARD_CAPS["input_working_point_cap"] == 200_000
    assert rung.HARD_CAPS["transition_parent_cap"] == 4_000
    assert rung.HARD_CAPS["transition_mode_cap"] == 16_000
    assert rung.HARD_CAPS["modes_per_parent_cap"] == 4
    assert rung.HARD_CAPS["adjacent_parent_pair_cap"] == 52_000
    assert rung.HARD_CAPS["cross_parent_rung_evaluation_cap"] == 832_000
    assert rung.HARD_CAPS["added_cross_parent_rung_cap"] == 832_000
    assert rung.HARD_CAPS["effective_mode_edge_cap"] == 1_688_000
    assert rung.HARD_CAPS["frozen_target_row_cap"] == 4_788
    assert rung.HARD_CAPS["batch_worker_cap"] == 8


def test_three_fresh_synthetic_replays_are_identical_and_below_16_gib() -> None:
    code = r'''
import hashlib, importlib.util, json, resource, sys
from pathlib import Path
root=Path.cwd()
spec=importlib.util.spec_from_file_location("rung_fixture", root/"tests/test_f1_v2_cross_parent_opposite_fragment_rung_probe.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as rung
projected, _control, _baseline, kwargs=module._fixture()
payload=rung.build_v2_cross_parent_opposite_fragment_rung_probe(projected, **kwargs)
encoded=rung.canonical_json_bytes(payload)
peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
if sys.platform != "darwin": peak *= 1024
print(json.dumps({"sha256":hashlib.sha256(encoded).hexdigest(),"bytes":len(encoded),"peak_rss_bytes":peak,"rungs":payload["case_counters"]["added_cross_parent_rung_count"],"targets":payload["case_counters"]["frozen_target_count"],"one_parent_finite":payload["case_counters"]["one_parent_finite_span_count"]}))
'''
    rows = []
    for _run in range(3):
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=Path.cwd(), text=True,
            capture_output=True, check=True, timeout=120,
        )
        rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert len({row["sha256"] for row in rows}) == 1
    assert len({row["bytes"] for row in rows}) == 1
    assert all(row["peak_rss_bytes"] < rung.PEAK_RSS_LIMIT_BYTES for row in rows)
    assert all(row["rungs"] == 2 and row["targets"] == 2 for row in rows)
    assert all(row["one_parent_finite"] == 0 for row in rows)

