#!/usr/bin/env python3
"""Build a non-authoritative CIBEC_V1 decision candidate and marker review pack.

This source-free helper does not sign, freeze, or qualify the charter.  It reads
only the neutral synthetic S01-S14 geometry and gives manufacturing authorities
a concrete proposal to accept or edit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_physical15_development_lane import _contact_sheet


SOURCE = ROOT / "output/research/cibec-v1-neutral-authority-pack-v4"
TARGET = ROOT / "output/research/cibec-v1-authority-candidate-v2"
CHARTER_FIELDS = (
    "COUNTING_UNIT", "CANONICAL_LINE", "INNER_OUTER_RULE", "COMPOUND_RADIUS_RULE",
    "RETURN_LIP_RULE", "OPEN_HEM_RULE", "CLOSED_HEM_RULE", "BEAD_RULE",
    "ROLLED_EDGE_RULE", "INTERRUPTION_RULE", "PARTIAL_VISIBILITY_RULE",
    "NONLINE_EVENT_RULE", "ENDPOINT_RULE",
)


DECISIONS = {
    "COUNTING_UNIT": "One observable contiguous forming-axis inspection feature. Paired material skins are not separate events. Distinct parallel axes, curvature-sign reversals separated by a tangent boundary, or disconnected observed axial spans are separate events.",
    "CANONICAL_LINE": "A straight segment parallel to the forming axis through the observed neutral-surface turn midpoint. Rounded bends use half accumulated absolute normal turn; faceted bends use the observed vertex; named hem, bead, and rolled-edge features use their corresponding rule. Endpoints are cropped to contiguous observed support.",
    "INNER_OUTER_RULE": "Paired inner and outer cylindrical skins with the same axis and axial support are one event. Place the line halfway between the two skins at half accumulated normal turn.",
    "COMPOUND_RADIUS_RULE": "Merge adjacent tangent rounded patches only when curvature sign, forming axis, and observed axial support are continuous with no straight land. Split at curvature-sign reversal, axis change, straight land, or support discontinuity.",
    "RETURN_LIP_RULE": "Count each observable tangent-changing bend axis in the return-lip profile as a distinct BEND event; use the rounded-turn midpoint or observed faceted vertex.",
    "OPEN_HEM_RULE": "Count one OPEN_HEM inspection event for one continuous open return arc; place the line at half accumulated return angle and crop to observed axial support.",
    "CLOSED_HEM_RULE": "Count one CLOSED_HEM inspection event for one continuous closed return arc; place the line at half accumulated return angle and crop to observed axial support.",
    "BEAD_RULE": "Count one BEAD inspection event per straight extruded bead; place the line on the observed bead crown axis and crop to observed axial support.",
    "ROLLED_EDGE_RULE": "Count one ROLLED_EDGE inspection event per straight rolled feature; place the line on the observed maximum-outward crown axis and crop to observed axial support.",
    "INTERRUPTION_RULE": "Disconnected observed axial spans are separate events even when coaxial. Never bridge an unsupported gap.",
    "PARTIAL_VISIBILITY_RULE": "Retain an otherwise unambiguous event as PARTIAL and crop both endpoints to its single observed contiguous span. Do not infer missing span from symmetry or CAD.",
    "NONLINE_EVENT_RULE": "Emit NONLINE_EVENT_CONFLICT when an inspection event's observed axis cannot be represented by one straight segment within the frozen marker tolerance; do not approximate it with a convenient chord.",
    "ENDPOINT_RULE": "Project the first and last supported samples of each contiguous event span onto the canonical axis. Do not extrapolate beyond observed evidence and do not bridge a gap larger than twice the local median axial sampling interval.",
}


def _event(y: float, z: float, *, x0: float = 0.0, x1: float = 80.0,
           class_code: str = "BEND", rule: str = "CANONICAL_LINE",
           visibility: str = "FULL") -> dict[str, Any]:
    return {
        "class_code": class_code,
        "line_start_mm": [round(x0, 6), round(y, 6), round(z, 6)],
        "line_end_mm": [round(x1, 6), round(y, 6), round(z, 6)],
        "visibility": visibility,
        "rule_id": rule,
    }


def _arc_mid(center_y: float, center_z: float, radius: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    return center_y + radius * math.cos(angle), center_z + radius * math.sin(angle)


def candidate_outputs() -> dict[str, dict[str, Any]]:
    s01 = _arc_mid(28, 5, 5, -45)
    s03 = _arc_mid(28, 5, 6, -45)
    s04a = _arc_mid(18, 5, 5, -45)
    s04b = _arc_mid(28, 5, 5, 135)
    s04c = _arc_mid(38, 5, 5, -45)
    s08 = _arc_mid(42, 5, 5, 45)
    s09 = _arc_mid(50, 0.75, 0.75, 45)
    s12a = _arc_mid(18, 6, 6, -30)
    s12b = _arc_mid(30, 12, 6, 150)
    return {
        "S01": {"case_status": "RESOLVED", "events": [_event(*s01)]},
        "S02": {"case_status": "RESOLVED", "events": [_event(-s01[0], s01[1])]},
        "S03": {"case_status": "RESOLVED", "events": [_event(*s03, rule="INNER_OUTER_RULE")]},
        "S04": {"case_status": "RESOLVED", "events": [_event(*s04a, rule="COMPOUND_RADIUS_RULE"), _event(*s04b, rule="COMPOUND_RADIUS_RULE"), _event(*s04c, rule="COMPOUND_RADIUS_RULE")]},
        "S05": {"case_status": "RESOLVED", "events": [_event(24, 0), _event(30, 8), _event(36, 8), _event(42, 0)]},
        "S06": {"case_status": "RESOLVED", "events": [_event(*s01, x0=0, x1=32, rule="INTERRUPTION_RULE"), _event(*s01, x0=48, x1=80, rule="INTERRUPTION_RULE")]},
        "S07": {"case_status": "RESOLVED", "events": [_event(*_arc_mid(30, 5, 5, -45), rule="RETURN_LIP_RULE"), _event(35, 26, rule="RETURN_LIP_RULE"), _event(29, 26, rule="RETURN_LIP_RULE")]},
        "S08": {"case_status": "RESOLVED", "events": [_event(*s08, class_code="OPEN_HEM", rule="OPEN_HEM_RULE")]},
        "S09": {"case_status": "RESOLVED", "events": [_event(*s09, class_code="CLOSED_HEM", rule="CLOSED_HEM_RULE")]},
        "S10": {"case_status": "RESOLVED", "events": [_event(30, 8, class_code="BEAD", rule="BEAD_RULE")]},
        "S11": {"case_status": "RESOLVED", "events": [_event(50, 16, class_code="ROLLED_EDGE", rule="ROLLED_EDGE_RULE")]},
        "S12": {"case_status": "RESOLVED", "events": [_event(*s12a, rule="COMPOUND_RADIUS_RULE"), _event(*s12b, rule="COMPOUND_RADIUS_RULE")]},
        "S13": {"case_status": "RESOLVED", "events": [_event(*s01, x0=17, x1=58, visibility="PARTIAL", rule="PARTIAL_VISIBILITY_RULE")]},
        "S14": {"case_status": "RESOLVED", "events": [_event(*s01)], "cad_overlay_status": "ABSTAIN", "cad_overlay_reason": "OVERLAY_UNAVAILABLE_OR_UNTRUSTED"},
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))


def _read_ply(path: Path) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    first = rows.index("end_header") + 1
    return np.asarray([[float(item) for item in row.split()] for row in rows[first:] if row], dtype=np.float64)


def _render_case(points: np.ndarray, fixture_id: str, condition: str,
                 events: list[dict[str, Any]], output: Path) -> None:
    yz = np.unique(np.round(points[:, 1:3], 6), axis=0)
    colors = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, max(len(events), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200, facecolor="#f8fafc")
    fig.suptitle(f"{fixture_id} — {condition.replace('_', ' ')} — PROVISIONAL AUTHORITY CANDIDATE", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.915, "NOT SIGNED · NOT QUALIFICATION TRUTH · ACCEPT, EDIT, OR REJECT EACH MARKER", ha="center", color="#b91c1c", fontsize=9, fontweight="bold")
    axes[0].scatter(yz[:, 0], yz[:, 1], s=10, c="#334155", linewidths=0, alpha=0.78)
    axes[1].scatter(points[:, 0], points[:, 1], s=0.25, c="#64748b", linewidths=0, alpha=0.18, rasterized=True)
    for index, (event, color) in enumerate(zip(events, colors), 1):
        start = np.asarray(event["line_start_mm"], dtype=float)
        end = np.asarray(event["line_end_mm"], dtype=float)
        axes[0].scatter([start[1]], [start[2]], s=150, facecolor="white", edgecolor=color, linewidth=3, zorder=5)
        axes[0].text(start[1], start[2], str(index), ha="center", va="center", fontweight="bold", fontsize=8, zorder=6)
        axes[1].plot([start[0], end[0]], [start[1], end[1]], color="white", linewidth=7, solid_capstyle="round", zorder=4)
        axes[1].plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=3.2, solid_capstyle="round", zorder=5)
        axes[1].text((start[0] + end[0]) / 2, start[1], str(index), ha="center", va="center", fontweight="bold", fontsize=8, bbox={"boxstyle": "circle,pad=0.15", "facecolor": "white", "edgecolor": color}, zorder=6)
    axes[0].set_title("Cross-section: Y / Z")
    axes[1].set_title("Axial support: X / Y")
    for ax, xlabel, ylabel in ((axes[0], "Y (mm)", "Z (mm)"), (axes[1], "X (mm)", "Y (mm)")):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, color="#cbd5e1", linewidth=0.4); ax.set_aspect("equal", adjustable="datalim")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.12, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    if TARGET.exists() and any(TARGET.iterdir()):
        raise FileExistsError(f"authority candidate already exists: {TARGET}")
    neutral = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    conditions = {row["fixture_id"]: row["condition"] for row in neutral["records"]}
    outputs = candidate_outputs()
    charter = {
        "schema": "CIBEC_V1_CHARTER_CANDIDATE",
        "status": "PROPOSED_NON_AUTHORITATIVE",
        "decisions": DECISIONS,
        "signatures": [],
    }
    answer_key = {
        "schema": "CIBEC_V1_SOURCE_FREE_EXPECTATIONS_CANDIDATE",
        "status": "PROPOSED_NON_AUTHORITATIVE",
        "charter_candidate_sha256": hashlib.sha256(_bytes(charter)).hexdigest(),
        "fixtures": outputs,
        "signatures": [],
    }
    _write_json(TARGET / "CIBEC_V1_CHARTER_CANDIDATE.json", charter)
    _write_json(TARGET / "S01_S14_EXPECTATIONS_CANDIDATE.json", answer_key)
    renders: list[Path] = []
    records = []
    for fixture_id in sorted(outputs):
        geometry = SOURCE / "geometry" / f"{fixture_id}.ply"
        image = TARGET / "marker_reviews" / f"{fixture_id}.png"
        _render_case(_read_ply(geometry), fixture_id, conditions[fixture_id], outputs[fixture_id]["events"], image)
        records.append({"fixture_id": fixture_id, "geometry_sha256": _sha(geometry), "marker_review_path": str(image.relative_to(TARGET)), "marker_review_sha256": _sha(image), "proposed_event_count": len(outputs[fixture_id]["events"])})
        renders.append(image)
    sheets = []
    for first in range(0, len(renders), 2):
        output = TARGET / "contact_sheets" / f"candidate_sheet_{first // 2 + 1:02d}.png"
        _contact_sheet(renders[first:first + 2], output)
        sheets.append({"path": str(output.relative_to(TARGET)), "sha256": _sha(output)})
    combined = hashlib.sha256(_bytes({"charter": charter, "answer_key": answer_key})).hexdigest()
    attestation = {
        "schema": "CIBEC_V1_AUTHORITY_REVIEW_ATTESTATION_TEMPLATE",
        "status": "AWAITING_TWO_AUTHORITIES",
        "review_payload_sha256": combined,
        "manufacturing_authority": {"name": "", "decision": "ACCEPT|EDIT|REJECT", "signed_payload_sha256": ""},
        "countersigning_authority": {"name": "", "decision": "ACCEPT|EDIT|REJECT", "signed_payload_sha256": ""},
    }
    _write_json(TARGET / "AUTHORITY_ATTESTATION_TEMPLATE.json", attestation)
    readme = """# CIBEC_V1 provisional authority review

This packet is **not signed and is not qualification truth**. It is a concrete production-floor proposal intended to make the required authority review fast and auditable.

Review the candidate charter, the S01-S14 expectation key, and every native-resolution contact sheet. Edit any disputed rule or marker before signing. Both authorities must then enter their real names, `ACCEPT`, and the exact `review_payload_sha256` in `AUTHORITY_ATTESTATION_TEMPLATE.json`.

Validate from the project root with:

```bash
.venv/bin/python3.11 scripts/validate_cibec_v1_authority_review.py
```
"""
    (TARGET / "README_AUTHORITY_REVIEW.md").write_text(readme, encoding="utf-8")
    manifest = {
        "schema": "CIBEC_V1_AUTHORITY_CANDIDATE_PACK",
        "source_free": True,
        "physical_assets_opened": 0,
        "prediction_assets_opened": 0,
        "qualification_truth": False,
        "authority_status": "UNSIGNED",
        "review_payload_sha256": combined,
        "records": records,
        "contact_sheets": sheets,
    }
    _write_json(TARGET / "manifest.json", manifest)
    print(json.dumps({"target": str(TARGET), "fixtures": len(records), "review_payload_sha256": combined}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
