import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_bend_corpus_seed_matrix.py"
spec = importlib.util.spec_from_file_location("run_bend_corpus_seed_matrix", MODULE_PATH)
run_bend_corpus_seed_matrix = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(run_bend_corpus_seed_matrix)


def _write_summary(path: Path, confident_bends: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": {
            "scoreboards": {
                "correspondence": {
                    "confident_bends": confident_bends,
                    "ambiguous_bends": 20 - confident_bends,
                }
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_cmd_includes_seed_controls():
    cmd = run_bend_corpus_seed_matrix._run_cmd(
        corpus=Path("/tmp/corpus"),
        output_dir=Path("/tmp/out"),
        parts=["38172000"],
        timeout_sec=123,
        unary_model=None,
        skip_heatmap_consistency=True,
        heatmap_cache_dir=None,
        deterministic_seed=101,
        retry_seeds=[211, 307],
    )

    assert "--deterministic-seed" in cmd
    assert "101" in cmd
    assert "--retry-seeds-json" in cmd
    assert "--skip-heatmap-consistency" in cmd


def test_main_writes_manifest_and_repeat_comparisons(monkeypatch, tmp_path):
    run_counter = {"count": 0}

    def fake_run(cmd, cwd, check, capture_output, text):
        output_dir = Path(cmd[cmd.index("--output") + 1])
        run_counter["count"] += 1
        _write_summary(output_dir / "summary.json", confident_bends=10 + run_counter["count"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_bend_corpus_seed_matrix.subprocess, "run", fake_run)

    rc = run_bend_corpus_seed_matrix.main(
        [
            "--corpus", str(tmp_path / "corpus"),
            "--output-root", str(tmp_path / "seed-matrix"),
            "--part", "38172000",
            "--repeat-count", "2",
            "--deterministic-seed", "101",
            "--retry-seeds-json", "[211,307]",
            "--profile-label", "baseline",
        ]
    )

    assert rc == 0
    manifest_path = tmp_path / "seed-matrix" / "seed_matrix_manifest.json"
    report_path = tmp_path / "seed-matrix" / "seed_matrix_report.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["repeat_count"] == 2
    assert manifest["profiles"][0]["label"] == "baseline"
    assert manifest["profiles"][0]["repeat_comparisons"][0]["changed_metric_count"] == 2
    assert "run_01 vs run_02" in report_path.read_text(encoding="utf-8")


def test_seed_profiles_accept_matrix_json():
    args = SimpleNamespace(
        seed_matrix_json='[{"label":"a","deterministic_seed":101,"retry_seeds":[211]},{"label":"b","deterministic_seed":17,"retry_seeds":[]}]',
        retry_seeds_json=None,
        deterministic_seed=None,
        profile_label=None,
    )

    profiles = run_bend_corpus_seed_matrix._seed_profiles(args)

    assert profiles[0] == {"label": "a", "deterministic_seed": 101, "retry_seeds": [211]}
    assert profiles[1] == {"label": "b", "deterministic_seed": 17, "retry_seeds": []}
