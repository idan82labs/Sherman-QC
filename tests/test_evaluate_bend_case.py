import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bend_case.py"
spec = importlib.util.spec_from_file_location("evaluate_bend_case", MODULE_PATH)
evaluate_bend_case = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(evaluate_bend_case)


class _FakeReport:
    def to_dict(self):
        return {"summary": {"release_decision": "HOLD"}, "matches": []}


def test_main_accepts_heatmap_flags_and_seed_overrides_and_writes_payload(monkeypatch, tmp_path):
    output_json = tmp_path / "case.json"
    captured = {}

    monkeypatch.setattr(
        evaluate_bend_case,
        "load_bend_runtime_config",
        lambda _: SimpleNamespace(local_refinement_kwargs={"deterministic_seed": 17, "retry_seeds": [29, 43, 59]}),
    )
    monkeypatch.setattr(
        evaluate_bend_case,
        "run_progressive_bend_inspection",
        lambda **kwargs: (
            captured.setdefault("runtime_config", kwargs["runtime_config"]),
            (_FakeReport(), SimpleNamespace(to_dict=lambda: {"cad_strategy": "detector_mesh"}, warnings=[])),
        )[1],
    )
    monkeypatch.setattr(
        evaluate_bend_case.sys,
        "argv",
        [
            "evaluate_bend_case.py",
            "--part-key", "P1",
            "--cad-path", "/tmp/cad.step",
            "--scan-path", "/tmp/scan.ply",
            "--output-json", str(output_json),
            "--state", "full",
            "--expected-total", "3",
            "--expected-completed", "2",
            "--spec-bend-angles-json", "[90.0]",
            "--spec-schedule-type", "complete",
            "--skip-heatmap-consistency",
            "--heatmap-cache-dir", "/tmp/heatmap-cache",
            "--deterministic-seed", "101",
            "--retry-seeds-json", "[211,307]",
        ],
    )

    assert evaluate_bend_case.main() == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["skip_heatmap_consistency"] is True
    assert payload["heatmap_cache_dir"] == "/tmp/heatmap-cache"
    assert payload["runtime_seed_profile"] == {"deterministic_seed": 101, "retry_seeds": [211, 307]}
    assert payload["expectation"]["expected_total_bends"] == 3
    assert payload["expectation"]["expected_completed_bends"] == 2
    assert captured["runtime_config"].local_refinement_kwargs["deterministic_seed"] == 101
    assert captured["runtime_config"].local_refinement_kwargs["retry_seeds"] == [211, 307]
