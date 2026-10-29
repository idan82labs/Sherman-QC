import importlib.util
import json
from pathlib import Path

from PIL import Image


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_f1_visual_review_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_f1_visual_review_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _write_png(path: Path, size=(1920, 1080)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _write_manifest(path: Path, *, bend_count=1, anchor_source="owned_atom_medoid"):
    path.parent.mkdir(parents=True, exist_ok=True)
    bends = []
    for index in range(bend_count):
        point = [float(index), 1.0, 2.0]
        bends.append(
            {
                "bend_id": f"OB{index + 1}",
                "anchor": point,
                "metadata": {
                    "render_anchor_source": anchor_source,
                    "owned_support_points": [point, [float(index), 2.0, 3.0]],
                },
            }
        )
    path.write_text(json.dumps({"bends": bends}), encoding="utf-8")


def test_audit_passes_final_exact_and_range_context(tmp_path):
    exact_sheet = tmp_path / "exact.png"
    range_sheet = tmp_path / "range.png"
    context = tmp_path / "context.png"
    manifest = tmp_path / "manifest.json"
    _write_png(exact_sheet)
    _write_png(range_sheet)
    _write_png(context)
    _write_manifest(manifest, bend_count=2)

    summary = {
        "results": [
            {
                "track": "known",
                "part_key": "exact",
                "scan_path": "/tmp/exact.ply",
                "algorithm": {
                    "accepted_exact_bend_count": 2,
                    "accepted_bend_count_range": [2, 2],
                },
                "render_artifacts": {
                    "contact_sheet_1080p_path": str(exact_sheet),
                    "manifest_path": str(manifest),
                },
            },
            {
                "track": "known",
                "part_key": "range",
                "scan_path": "/tmp/range.ply",
                "algorithm": {
                    "accepted_exact_bend_count": 2,
                    "accepted_bend_count_range": [1, 2],
                },
                "render_artifacts": {
                    "contact_sheet_1080p_path": str(range_sheet),
                    "manifest_path": str(manifest),
                    "scan_context_path": str(context),
                },
            },
        ]
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = audit.audit_visual_review_artifacts(summary_paths=[summary_path])

    assert report["issue_count"] == 0
    assert report["final_exact_case_count"] == 1
    assert report["range_or_review_case_count"] == 1


def test_non_singleton_exact_requires_context_not_final_manifest_count(tmp_path):
    sheet = tmp_path / "sheet.png"
    context = tmp_path / "context.png"
    manifest = tmp_path / "manifest.json"
    _write_png(sheet)
    _write_png(context)
    _write_manifest(manifest, bend_count=9)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "track": "known",
                        "part_key": "range_exact_field",
                        "scan_path": "/tmp/range.ply",
                        "algorithm": {
                            "accepted_exact_bend_count": 10,
                            "accepted_bend_count_range": [9, 10],
                        },
                        "render_artifacts": {
                            "contact_sheet_1080p_path": str(sheet),
                            "manifest_path": str(manifest),
                            "scan_context_path": str(context),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit.audit_visual_review_artifacts(summary_paths=[summary_path])

    assert report["issue_count"] == 0
    assert report["final_exact_case_count"] == 0


def test_audit_fails_floating_marker_metadata(tmp_path):
    sheet = tmp_path / "sheet.png"
    manifest = tmp_path / "manifest.json"
    _write_png(sheet)
    _write_manifest(manifest, bend_count=1, anchor_source="centroid")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "track": "known",
                        "part_key": "bad",
                        "scan_path": "/tmp/bad.ply",
                        "algorithm": {
                            "accepted_exact_bend_count": 1,
                            "accepted_bend_count_range": [1, 1],
                        },
                        "render_artifacts": {
                            "contact_sheet_1080p_path": str(sheet),
                            "manifest_path": str(manifest),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit.audit_visual_review_artifacts(summary_paths=[summary_path])

    assert report["issue_count"] == 1
    assert report["issues"][0]["issue"] == "anchor_source_not_owned_atom_medoid"
