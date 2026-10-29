"""Z4 selector unit tests (no scans, no labels read)."""
import numpy as np
import pytest

from research.zeroshot.z4_selector import custody
from research.zeroshot.z4_selector.selector import nms, fast_counts
from research.zeroshot.z4_selector.data import match_matrix, prf


def _line(a, b):
    return {"line_start": list(map(float, a)), "line_end": list(map(float, b))}


def test_match_matrix_thresholds():
    gold = [_line([0, 0, 0], [100, 0, 0])]
    ends = np.array([
        [[0, 3.9, 0], [100, 3.9, 0]],     # 3.9 mm offset: match
        [[0, 4.1, 0], [100, 4.1, 0]],     # 4.1 mm: no
        [[0, 0, 0], [50, 0, 0]],          # 50 % of gold span: no
        [[0, 0, 0], [100, 100 * np.tan(np.radians(4.5)), 0]],  # 4.5 deg, offset at mid ~3.9: match
    ], float)
    mm = match_matrix(ends, gold)
    assert mm[:, 0].tolist() == [True, False, False, True]


def test_nms_drops_near_duplicates_keeps_parallel_far():
    ends = np.array([[[0, 0, 0], [100, 0, 0]], [[0, 1, 0], [100, 1, 0]], [[0, 20, 0], [100, 20, 0]]], float)
    keep = nms(ends, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep == [0, 2]
    assert nms(ends, np.array([0.9, 0.8, 0.7]), 0.95) == []


def test_fast_counts_one_to_one():
    gold = [_line([0, 0, 0], [100, 0, 0]), _line([0, 50, 0], [100, 50, 0])]
    ends = np.array([[[0, 0, 0], [100, 0, 0]], [[0, 30, 0], [100, 30, 0]]], float)
    tp, fp, fn = fast_counts(ends, np.array([0.9, 0.9]), gold, 0.5)
    assert (tp, fp, fn) == (1, 1, 1)
    assert prf(1, 1, 1)[2] == pytest.approx(0.5)


def test_custody_refuses_sealed_and_testb_paths():
    with pytest.raises(custody.CustodyError):
        custody.check_source_path("/tmp/x/testb_SEALED/a/eval_lines.json")
    with pytest.raises(custody.CustodyError):
        custody.check_source_path("/tmp/x/test_b/a/eval_lines.json")


def test_dev_inventory_excludes_testb():
    import json

    sp = json.loads((custody.RUNLOG / "SPLIT_FROZEN.json").read_text())
    inv = custody.dev_inventory()
    assert not (set(inv) & set(sp["test_b"]["scans"]))
    assert set(inv) == set(sp["dev"]["scans"])
    sid = sp["test_b"]["scans"][0]
    with pytest.raises(custody.CustodyError):
        custody.check_dev_scan(sid, inv)


def test_source_lines_load_and_meta(tmp_path):
    import json
    from research.zeroshot.z4_selector.candidates import load_source_lines

    d = tmp_path / "z2src" / "sc_x"
    d.mkdir(parents=True)
    f = d / "eval_lines.json"
    f.write_text(json.dumps({"frame": "scan_raw", "owned_bend_regions": [
        {"line_start": [0, 0, 0], "line_end": [50, 0, 0], "score": 0.9, "stations": 7, "cover": 1.0},
        {"line_start": [0, 0, 0], "line_end": [2, 0, 0]}]}))
    lines = load_source_lines(f, "z2")
    assert len(lines) == 1 and lines[0].generator == "src:z2"
    assert lines[0].meta == {"score": 0.9, "stations": 7.0, "cover": 1.0}


def test_z5_solver_conflicts_and_tree():
    import json
    import subprocess
    from pathlib import Path

    py = custody.REPO / ".venv-f1-structured/bin/python"
    if not py.is_file():
        pytest.skip("no structured venv")
    probs = [{"id": "a", "w": [500, 300, -100, 200], "conflicts": [[0, 1]], "edges": [], "n_nodes": 0,
              "tree_slack": None, "nopair_penalty": 0},
             {"id": "b", "w": [500, 500, 500], "edges": [[0, 0, 1], [1, 1, 2], [2, 2, 0]], "n_nodes": 3,
              "tree_slack": 0, "nopair_penalty": 0},
             {"id": "c", "w": [-5], "edges": [[0, 0, 1]], "n_nodes": 2, "tree_slack": 0, "nopair_penalty": 0}]
    solver = Path(custody.REPO / "research/zeroshot/z4_selector/z5_solve.py")
    out = subprocess.run([str(py), "-B", str(solver)], input="\n".join(map(json.dumps, probs)),
                         capture_output=True, text=True, check=True).stdout
    res = {r["id"]: r for r in map(json.loads, out.splitlines())}
    assert res["a"]["keep"] == [0, 3]
    assert len(res["b"]["keep"]) == 2          # a 3-cycle is cut to a tree
    assert res["c"]["keep"] == [] and res["c"]["status"] == "OPTIMAL"  # empty selection stays feasible
