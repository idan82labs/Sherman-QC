import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_zero_shot_theory_experiments.py"
    spec = importlib.util.spec_from_file_location("summarize_zero_shot_theory_experiments", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summary = _load_module()


def test_score_prefers_keep_then_more_improvements():
    keep_row = {"decision": "keep", "improvements": ["ysherman tightened"], "regressions": [], "method_family": "a"}
    reject_row = {"decision": "reject", "improvements": [], "regressions": [], "method_family": "b"}

    assert summary._score(keep_row) < summary._score(reject_row)


def test_render_markdown_reports_no_winning_tranche_when_all_rejected():
    markdown = summary._render_markdown(
        [
            {
                "method_family": "fold_contour_diagnostics",
                "decision": "reject",
                "improvements": [],
                "regressions": [],
                "rejection_reason": "no hard-case improvement",
                "professor_brief": "/tmp/a.md",
            },
            {
                "method_family": "trace_informed_evidence_atoms",
                "decision": "reject",
                "improvements": [],
                "regressions": [],
                "rejection_reason": "no hard-case improvement",
                "professor_brief": "/tmp/b.md",
            },
        ]
    )

    assert "No winning tranche in this first-pass theory-to-reality run." in markdown
