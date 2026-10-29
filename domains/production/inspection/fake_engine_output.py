"""Synthetic E3 output directory for tests and offline development (never on the line path).

E3 (the per-scan engine) is built concurrently, so C06 is developed and tested
against a *fake* engine output that follows ENGINE_IO §4.1 exactly. The fake is
built from the C01 contract examples for the synthetic asymmetric hat
``SYN-HAT-1`` (``contracts.production.examples.generate_examples``):

- ``<ws>/recipe.json`` and ``<ws>/compiled/compiled_part.json``: the example
  recipe and CompiledPart (canonical bytes);
- ``<ws>/engine/result.json``: one of the example results (``final``,
  ``in_process``, ``incomplete`` or ``invalid``) with ``viewer_assets``
  pointing at the files below;
- ``registered_scan.ply`` (binary LE float32): both skins of every panel of the
  decoded stage model, sampled on a grid with deterministic noise;
- ``deviations.f32``: a smooth signed field with every 50th value NaN;
- ``segment_labels.u16``: ``1 + panel index``;
- ``display/stage_mesh.ply`` / ``display/final_mesh.ply``: two triangles per
  panel skin, posed by the stage model's ``panel_transforms``;
- ``bend_geometry.json``: ``BendGeometryFileV1.from_result``;
- ``engine.log``.

No real scan or CAD is read. Everything is deterministic for a given input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = ["SCENARIOS", "FakeEngineOutput", "build_fake_engine_output", "panel_quads"]

#: scenario -> example file name
SCENARIOS: dict[str, str] = {
    "final": "shopfloor_result_v1.example.json",
    "in_process": "shopfloor_result_v1.in_process.example.json",
    "incomplete": "shopfloor_result_v1.incomplete.example.json",
    "invalid": "shopfloor_result_v1.invalid.example.json",
}


@dataclass(frozen=True)
class FakeEngineOutput:
    workspace: Path
    recipe_path: Path
    compiled_path: Path
    engine_dir: Path
    result_path: Path
    point_count: int


def panel_quads(compiled: Any, stage_model_id: str | None) -> list[tuple[int, Any, Any]]:
    """``[(panel_index, skin0 quad (4,3), skin1 quad (4,3))]`` posed at the stage model
    (final pose when ``stage_model_id`` is None)."""
    import numpy as np

    transforms = {} if stage_model_id is None else compiled.stage(stage_model_id).panel_transforms
    out = []
    for i, panel in enumerate(compiled.panels):
        q0 = np.asarray(panel.outer_boundary_mm, dtype=np.float64)
        n0 = np.asarray(panel.skin0_plane.normal, dtype=np.float64)
        t = float(panel.thickness_mm or compiled.thickness.t_mm)
        q1 = q0 - t * n0[None, :]
        m = np.asarray(transforms.get(panel.panel_id, np.eye(4)), dtype=np.float64)
        pose = lambda q: q @ m[:3, :3].T + m[:3, 3]  # noqa: E731
        out.append((i, pose(q0), pose(q1)))
    return out


def _sample_quad(q: Any, spacing: float) -> Any:
    import numpy as np

    u, v = q[1] - q[0], q[3] - q[0]
    nu = max(2, int(round(np.linalg.norm(u) / spacing)) + 1)
    nv = max(2, int(round(np.linalg.norm(v) / spacing)) + 1)
    s, t = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    return q[0] + s.reshape(-1, 1) * u + t.reshape(-1, 1) * v


def build_fake_engine_output(
    workspace: str | Path,
    *,
    scenario: str = "final",
    spacing_mm: float = 1.0,
    noise_mm: float = 0.02,
    mutate_result: Callable[[dict[str, Any]], None] | None = None,
) -> FakeEngineOutput:
    """Write a complete fake job workspace (module docstring) and return its paths.

    ``mutate_result`` may edit the result JSON dict before it is written (tests
    use it to build inconsistent or dishonest engine outputs).
    """
    import json

    import numpy as np

    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.examples.generate_examples import build_examples
    from contracts.production.inspection_result import BendGeometryFileV1, ShopfloorInspectionResultV1
    from domains.production.inspection.ply_io import write_mesh_ply, write_points_ply

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; one of {sorted(SCENARIOS)}")
    ws = Path(workspace)
    engine = ws / "engine"
    (engine / "display").mkdir(parents=True, exist_ok=True)
    (ws / "compiled").mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    recipe = examples["part_recipe_v1.example.json"]
    compiled = examples["compiled_part_v1.example.json"]
    template: ShopfloorInspectionResultV1 = examples[SCENARIOS[scenario]]
    (ws / "recipe.json").write_bytes(canonical_json_bytes(recipe))
    (ws / "compiled" / "compiled_part.json").write_bytes(canonical_json_bytes(compiled))

    stage_id = template.stage_decision.stage_model_id
    rng = np.random.default_rng(20261017)
    pts_parts, labels_parts = [], []
    verts, faces = [], []
    for idx, q0, q1 in panel_quads(compiled, stage_id):
        normal = np.cross(q0[1] - q0[0], q0[3] - q0[0])
        normal = normal / np.linalg.norm(normal)
        for q in (q0, q1):
            p = _sample_quad(q, spacing_mm)
            p = p + rng.normal(0.0, noise_mm, size=(len(p), 1)) * normal[None, :]
            pts_parts.append(p)
            labels_parts.append(np.full(len(p), 1 + idx, dtype="<u2"))
            base = sum(len(v) for v in verts)
            verts.append(q)
            faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + base)
    points = np.concatenate(pts_parts)
    labels = np.concatenate(labels_parts)
    deviations = (0.18 * np.sin(points[:, 0] / 7.0) * np.cos(points[:, 1] / 11.0) + 0.02 * np.sign(points[:, 2] - 15.0)).astype("<f4")
    deviations[::50] = np.nan

    final_v, final_f = [], []
    for idx, q0, q1 in panel_quads(compiled, None):
        for q in (q0, q1):
            base = sum(len(v) for v in final_v)
            final_v.append(q)
            final_f.append(np.array([[0, 1, 2], [0, 2, 3]]) + base)
    write_mesh_ply(engine / "display" / "final_mesh.ply", np.concatenate(final_v), np.concatenate(final_f))

    data = template.model_dump(mode="json")
    invalid = template.inspection_validity.state != "VALID"
    if invalid:
        data["viewer_assets"] = {
            "frame": "cad_model_mm", "heatmap_band_mm": recipe.default_tolerances.surface_profile_mm,
            "heatmap_unavailable_reason": "REGISTRATION_UNTRUSTED", "final_mesh": "display/final_mesh.ply",
            "bend_geometry": "bend_geometry.json",
        }
        n_points = 0
    else:
        write_points_ply(engine / "registered_scan.ply", points)
        (engine / "deviations.f32").write_bytes(np.ascontiguousarray(deviations).tobytes())
        (engine / "segment_labels.u16").write_bytes(labels.tobytes())
        write_mesh_ply(engine / "display" / "stage_mesh.ply", np.concatenate(verts), np.concatenate(faces))
        data["viewer_assets"] = {
            "frame": "cad_model_mm", "registered_scan": "registered_scan.ply", "point_count": int(len(points)),
            "deviations": "deviations.f32", "deviation_reference_stage": stage_id,
            "heatmap_band_mm": recipe.default_tolerances.surface_profile_mm,
            "segment_labels": "segment_labels.u16", "stage_mesh": "display/stage_mesh.ply",
            "final_mesh": "display/final_mesh.ply", "bend_geometry": "bend_geometry.json",
        }
        n_points = int(len(points))
    data["warnings"] = ["SYNTHETIC FAKE ENGINE OUTPUT"]
    if mutate_result is not None:
        mutate_result(data)
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    try:
        result = ShopfloorInspectionResultV1.model_validate_json(raw)
    except ValueError:
        result = None  # a deliberately invalid mutation: write the raw bytes as-is
    if result is not None:
        (engine / "result.json").write_bytes(canonical_json_bytes(result))
        (engine / "bend_geometry.json").write_bytes(canonical_json_bytes(BendGeometryFileV1.from_result(result)))
    else:
        (engine / "result.json").write_bytes(raw)
    (engine / "engine.log").write_text("fake engine output (C06 tests)\n", encoding="utf-8")
    return FakeEngineOutput(
        workspace=ws,
        recipe_path=ws / "recipe.json",
        compiled_path=ws / "compiled" / "compiled_part.json",
        engine_dir=engine,
        result_path=engine / "result.json",
        point_count=n_points,
    )
