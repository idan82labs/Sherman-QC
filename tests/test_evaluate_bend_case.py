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


def test_main_accepts_heatmap_flags_and_writes_payload(monkeypatch, tmp_path):
    output_json = tmp_path / "case.json"

    monkeypatch.setattr(evaluate_bend_case, "load_bend_runtime_config", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        evaluate_bend_case,
        "run_progressive_bend_inspection",
        lambda **kwargs: (_FakeReport(), SimpleNamespace(to_dict=lambda: {"cad_strategy": "detector_mesh"}, warnings=[])),
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
        ],
    )

    assert evaluate_bend_case.main() == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["skip_heatmap_consistency"] is True
    assert payload["heatmap_cache_dir"] == "/tmp/heatmap-cache"
    assert payload["expectation"]["expected_total_bends"] == 3
    assert payload["expectation"]["expected_completed_bends"] == 2
