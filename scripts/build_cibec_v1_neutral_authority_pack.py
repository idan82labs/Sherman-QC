#!/usr/bin/env python3
"""Build neutral S01-S14 source-free geometry and six-view authority renders.

No expected event count, canonical marker, prediction, or physical asset is
included.  The pack exists only so manufacturing authorities can freeze the
CIBEC charter before the signed semantic fixture run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_physical15_development_lane import _contact_sheet
from scripts.render_f1_source_only_prediction_audit import render


TARGET = ROOT / "output/research/cibec-v1-neutral-authority-pack-v4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(("ply", "format ascii 1.0", f"element vertex {len(points)}", "property float x", "property float y", "property float z", "end_header")) + "\n"
    rows = "\n".join("{:.6f} {:.6f} {:.6f}".format(*row) for row in points) + "\n"
    path.write_text(header + rows, encoding="ascii")


def _render_profile(points: np.ndarray, fixture_id: str, condition: str, output: Path) -> None:
    yz = np.unique(np.round(points[:, 1:3], 6), axis=0)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=220, facecolor="#f8fafc")
    ax.scatter(yz[:, 0], yz[:, 1], s=12, c="#0f172a", linewidths=0)
    fig.suptitle(f"{fixture_id} — {condition.replace('_', ' ')}", fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925, "NEUTRAL MID-SPAN PROFILE · NO EVENT COUNT OR MARKER PROPOSED", ha="center", va="center", fontsize=8, color="#475569")
    ax.set_xlabel("profile Y (mm)"); ax.set_ylabel("profile Z (mm)")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, color="#cbd5e1", linewidth=0.45)
    fig.subplots_adjust(top=0.84)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12, facecolor=fig.get_facecolor())
    plt.close(fig)


def _sample_profile(profile: np.ndarray, samples_per_segment: int = 36) -> np.ndarray:
    rows = []
    for index in range(len(profile) - 1):
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=index == len(profile) - 2)
        rows.append(profile[index][None, :] * (1.0 - t[:, None]) + profile[index + 1][None, :] * t[:, None])
    return np.vstack(rows)


def _extrude(profile: np.ndarray, *, width: float = 80.0, x0: float = 0.0, x1: float | None = None) -> np.ndarray:
    yz = _sample_profile(profile)
    end = width if x1 is None else x1
    xs = np.linspace(x0, end, 91)
    return np.asarray([[x, y, z] for x in xs for y, z in yz], dtype=np.float64)


def _arc(center: tuple[float, float], radius: float, start: float, stop: float, count: int = 24) -> np.ndarray:
    theta = np.linspace(math.radians(start), math.radians(stop), count)
    return np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))


def _fixtures() -> dict[str, np.ndarray]:
    l_profile = np.vstack((np.asarray(((0, 0), (28, 0))), _arc((28, 5), 5, -90, 0)[1:], np.asarray(((33, 5), (33, 30)))))
    paired = np.vstack((l_profile, _arc((28, 5), 7, -90, 0)))
    compound = np.vstack((_arc((18, 5), 5, -90, 0), _arc((28, 5), 5, 180, 90)[1:], _arc((38, 5), 5, -90, 0)[1:]))
    zigzag = np.asarray(((0, 0), (24, 0), (30, 8), (36, 8), (42, 0), (66, 0)), dtype=float)
    return_lip = np.vstack((np.asarray(((0, 0), (30, 0))), _arc((30, 5), 5, -90, 0)[1:], np.asarray(((35, 5), (35, 26), (29, 26), (29, 12)))))
    open_hem = np.vstack((np.asarray(((0, 0), (42, 0))), _arc((42, 5), 5, -90, 180)[1:], np.asarray(((37, 5), (10, 5)))))
    closed_hem = np.vstack((np.asarray(((0, 0), (50, 0))), _arc((50, 0.75), 0.75, -90, 180)[1:], np.asarray(((49.25, 0.75), (12, 0.75)))))
    bead = np.column_stack((np.linspace(0, 60, 61), 8 * np.exp(-((np.linspace(0, 60, 61) - 30) / 8) ** 2)))
    rolled = np.vstack((np.asarray(((0, 0), (50, 0))), _arc((50, 8), 8, -90, 275, 42)[1:]))
    s_profile = np.vstack((np.asarray(((0, 0), (18, 0))), _arc((18, 6), 6, -90, 30)[1:], _arc((30, 12), 6, 210, 90)[1:], np.asarray(((30, 18), (48, 18)))))
    discontinuous = np.vstack((_extrude(l_profile, x0=0, x1=32), _extrude(l_profile, x0=48, x1=80)))
    fixtures = {
        "S01": _extrude(l_profile),
        "S02": _extrude(l_profile)[:, (0, 1, 2)] * np.asarray((1, -1, 1)),
        "S03": _extrude(paired),
        "S04": _extrude(compound),
        "S05": _extrude(zigzag),
        "S06": discontinuous,
        "S07": _extrude(return_lip),
        "S08": _extrude(open_hem),
        "S09": _extrude(closed_hem),
        "S10": _extrude(bead),
        "S11": _extrude(rolled),
        "S12": _extrude(s_profile),
        "S13": _extrude(l_profile, x0=17, x1=58),
        "S14": _extrude(l_profile),
    }
    return {key: np.unique(np.round(value, 6), axis=0) for key, value in fixtures.items()}


def main() -> int:
    if TARGET.exists() and any(TARGET.iterdir()):
        raise FileExistsError(f"neutral authority pack already exists: {TARGET}")
    empty = TARGET / "empty_predictions.json"
    _write_json(empty, {"route": "CIBEC_NEUTRAL_AUTHORITY_REVIEW", "source_only": True, "predictions": []})
    condition_by_id = {row[0]: row[1] for row in json.loads((ROOT / "research/cibec_v1/source_free_fixture_manifest.json").read_text())["fixtures"]}
    records = []
    renders = []
    profiles = []
    for fixture_id, points in _fixtures().items():
        geometry = TARGET / "geometry" / f"{fixture_id}.ply"
        image = TARGET / "six_view" / f"{fixture_id}.png"
        profile_image = TARGET / "profiles" / f"{fixture_id}.png"
        _write_ply(geometry, points)
        metadata = render(geometry=geometry, prediction_path=empty, output=image, max_points=250_000)
        _render_profile(points, fixture_id, condition_by_id[fixture_id], profile_image)
        records.append({"fixture_id": fixture_id, "condition": condition_by_id[fixture_id], "geometry_path": str(geometry.relative_to(TARGET)), "geometry_sha256": _sha(geometry), "render_path": str(image.relative_to(TARGET)), "render_sha256": _sha(image), "profile_path": str(profile_image.relative_to(TARGET)), "profile_sha256": _sha(profile_image), "point_count": len(points), "proposed_event_count": None, "proposed_markers": []})
        renders.append(image)
        profiles.append(profile_image)
    sheets = []
    for first in range(0, len(renders), 3):
        output = TARGET / "contact_sheets" / f"contact_sheet_{first // 3 + 1:02d}.png"
        _contact_sheet(renders[first:first + 3], output)
        sheets.append({"path": str(output.relative_to(TARGET)), "sha256": _sha(output)})
    profile_sheets = []
    for first in range(0, len(profiles), 3):
        output = TARGET / "profile_contact_sheets" / f"profile_sheet_{first // 3 + 1:02d}.png"
        _contact_sheet(profiles[first:first + 3], output)
        profile_sheets.append({"path": str(output.relative_to(TARGET)), "sha256": _sha(output)})
    manifest = {"schema": "CIBEC_V1_NEUTRAL_AUTHORITY_PACK", "source_free": True, "physical_assets_opened": 0, "prediction_assets_opened": 0, "answer_lines_present": False, "renderer_sha256": _sha(ROOT / "scripts/render_f1_source_only_prediction_audit.py"), "records": records, "contact_sheets": sheets, "profile_contact_sheets": profile_sheets, "authority_status": "UNSIGNED"}
    _write_json(TARGET / "manifest.json", manifest)
    worksheet = ["# CIBEC_V1 authority decision worksheet", "", "All renders are neutral source-free geometry. No model output, prior truth, expected count, or proposed marker is shown.", "", "## Charter decisions", "", "| Field | Manufacturing-authority decision | Countersign |", "|---|---|---|"]
    for field in ("COUNTING_UNIT", "CANONICAL_LINE", "INNER_OUTER_RULE", "COMPOUND_RADIUS_RULE", "RETURN_LIP_RULE", "OPEN_HEM_RULE", "CLOSED_HEM_RULE", "BEAD_RULE", "ROLLED_EDGE_RULE", "INTERRUPTION_RULE", "PARTIAL_VISIBILITY_RULE", "NONLINE_EVENT_RULE", "ENDPOINT_RULE"):
        worksheet.append(f"| `{field}` |  |  |")
    worksheet += ["", "## Fixture decisions", "", "For every S01–S14 case, record the exact event count, one canonical straight segment per event, class code, visibility, and applicable charter rule. If this cannot be done generically, record `ONTOLOGY_UNRESOLVED` or `NONLINE_EVENT_CONFLICT`.", "", "| Fixture | Condition | Exact count | Marker coordinates | Rule IDs | Countersign |", "|---|---|---:|---|---|---|"]
    for record in records:
        worksheet.append(f"| {record['fixture_id']} | `{record['condition']}` |  |  |  |  |")
    worksheet += ["", "## Signatures", "", "Manufacturing authority: ____________________  Date: __________", "", "Countersigning authority: ___________________  Date: __________", ""]
    (TARGET / "AUTHORITY_DECISION_WORKSHEET.md").write_text("\n".join(worksheet), encoding="utf-8")
    print(json.dumps({"target": str(TARGET), "fixtures": len(records), "contact_sheets": len(sheets), "profile_contact_sheets": len(profile_sheets)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
