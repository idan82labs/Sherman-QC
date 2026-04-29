import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_patch_graph_f1_prototype.py"
    spec = importlib.util.spec_from_file_location("run_patch_graph_f1_prototype", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_module()


def test_attach_candidate_render_semantics_rejects_suppressed_marker_count():
    payload = {
        "candidate": {
            "candidate_source": "raw_f1",
            "exact_bend_count": 12,
            "bend_count_range": [12, 12],
        }
    }

    runner._attach_candidate_render_semantics(
        payload,
        render_info={"overview_path": "/tmp/raw.png", "scan_context_path": "/tmp/context.png"},
        rendered_owned_region_count=9,
    )

    assert payload["candidate"]["accepted_render_semantics"] == "scan_context_only_raw_f1_debug_not_final"
    assert payload["candidate"]["accepted_render_marker_count"] == 0
    assert payload["candidate"]["accepted_render_overview_path"] == "/tmp/context.png"


def test_attach_candidate_render_semantics_accepts_exact_render_count():
    payload = {
        "candidate": {
            "candidate_source": "raw_f1",
            "exact_bend_count": 12,
            "bend_count_range": [12, 12],
        }
    }

    runner._attach_candidate_render_semantics(
        payload,
        render_info={"overview_path": "/tmp/raw.png", "scan_context_path": "/tmp/context.png"},
        rendered_owned_region_count=12,
    )

    assert payload["candidate"]["accepted_render_semantics"] == "owned_regions_match_accepted_exact"
    assert payload["candidate"]["accepted_render_marker_count"] == 12
    assert payload["candidate"]["accepted_render_overview_path"] == "/tmp/raw.png"
