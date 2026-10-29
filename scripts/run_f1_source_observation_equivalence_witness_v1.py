#!/usr/bin/env python3
"""Execute the one-shot source-observation equivalence witness.

This is the sole source-only action authorized by the post-PNCT Pro ruling.
It is a terminal proof artifact, not a V21 detector, and never opens a real
scan, gold, labels, CAD, expected counts, or filename/identity information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_projective_normal_chord_transport_v20 import analyze_candidate  # noqa: E402
from research.f1_pnct_v20_fixtures import circular_fold  # noqa: E402
from research.f1_source_observation_equivalence_witness_v1 import (  # noqa: E402
    Observation,
    PROFILE,
    ROLES,
    canonical_json_bytes,
    public_pair_record,
    role_pair,
    serialize_observation,
)


FIXTURE_NAME = "PNCT_SF01_CLEAN_STANDARD_FOLD"
FROZEN_VARIANTS = (
    "triangle_permutation",
    "cyclic_vertices",
    "reversed_vertices",
    "axis_seed_reversal",
    "rigid_17",
    "rigid_53",
    "rigid_101",
    "scale_0.5",
    "scale_2.0",
)


def _rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(length: float) -> dict:
    return {
        "candidate_id": "SOURCE_OBSERVATION_X",
        "candidate_key": hashlib.sha256(b"SOURCE_OBSERVATION_X").hexdigest()[:32],
        "span_endpoints": [[-length / 2.0, 0.0, 0.0], [length / 2.0, 0.0, 0.0]],
    }


def _seed_normals(angle_deg: float) -> tuple[tuple[float, float, float], ...]:
    angle = math.radians(angle_deg)
    return ((0.0, 1.0, 0.0), (0.0, math.cos(angle), math.sin(angle)))


def _observation(name, vertices, triangles, endpoints, scale, seeds) -> Observation:
    return Observation(
        name,
        serialize_observation(
            vertices=vertices,
            triangles=triangles,
            span_endpoints=endpoints,
            scale=scale,
            seed_normals=seeds,
        ),
    )


def _variant_observations(vertices, triangles, *, angle_deg: float, length: float) -> list[Observation]:
    base_row = _row(length)
    base_seeds = _seed_normals(angle_deg)
    observations = [
        _observation("base", vertices, triangles, base_row["span_endpoints"], 1.0, base_seeds),
        _observation("triangle_permutation", vertices, triangles[::-1], base_row["span_endpoints"], 1.0, base_seeds),
        _observation("cyclic_vertices", vertices, np.roll(triangles, 1, axis=1), base_row["span_endpoints"], 1.0, base_seeds),
        _observation("reversed_vertices", vertices, triangles[:, ::-1], base_row["span_endpoints"], 1.0, base_seeds),
        _observation(
            "axis_seed_reversal",
            vertices,
            triangles,
            list(reversed(base_row["span_endpoints"])),
            1.0,
            tuple(reversed(base_seeds)),
        ),
    ]
    for degrees in (17, 53, 101):
        radians = math.radians(degrees)
        rotation = np.asarray(
            (
                (math.cos(radians), -math.sin(radians), 0.0),
                (math.sin(radians), math.cos(radians), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        translation = np.asarray((11.0 + degrees, -7.0, 23.0), dtype=np.float64)
        points = vertices @ rotation.T + translation
        endpoints = np.asarray(base_row["span_endpoints"]) @ rotation.T + translation
        seeds = np.asarray(base_seeds) @ rotation.T
        observations.append(_observation(f"rigid_{degrees}", points, triangles, endpoints, 1.0, seeds))
    for factor in (0.5, 2.0):
        observations.append(
            _observation(
                f"scale_{factor}",
                factor * vertices,
                triangles,
                factor * np.asarray(base_row["span_endpoints"]),
                factor,
                base_seeds,
            )
        )
    assert tuple(item.name for item in observations[1:]) == FROZEN_VARIANTS
    return observations


def _build_evidence(observations, *, axis_count: int, detector_digest: str) -> dict:
    pairs = [
        role_pair(
            observation,
            frozen_axis_count=axis_count,
            frozen_detector_output_sha256=detector_digest,
        )
        for observation in observations
    ]
    return {
        "profile": PROFILE,
        "fixture_ref": observations[0].sha256,
        "roles": [
            {
                "fixture_id": role.fixture_id,
                "physical_interpretation": role.physical_interpretation,
                "required_axis_count": role.required_axis_count,
                "fixture_ref": observations[0].sha256,
            }
            for role in ROLES
        ],
        "base_pair": public_pair_record(pairs[0]),
        "metamorphic_pairs": [public_pair_record(pair) for pair in pairs[1:]],
    }, pairs


def _surface_panel(vertices: np.ndarray, triangles: np.ndarray, record, output: Path) -> None:
    """Render one role-blind, exact-4096 surface-evidence panel."""

    figure = plt.figure(figsize=(16, 16), dpi=256, facecolor="#050910")
    grid = figure.add_gridspec(2, 2, hspace=0.15, wspace=0.12)
    face_points = vertices[triangles]
    face_normals = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    nonzero = lengths > 0
    face_normals[nonzero] /= lengths[nonzero, None]
    colors = plt.cm.viridis(Normalize(-1.0, 1.0)(face_normals[:, 2]))

    ax3 = figure.add_subplot(grid[0, 0], projection="3d")
    mesh = Poly3DCollection(face_points, facecolors=colors, edgecolors=(0.55, 0.72, 0.86, 0.28), linewidths=0.18)
    ax3.add_collection3d(mesh)
    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    ax3.set_xlim(center[0] - radius, center[0] + radius)
    ax3.set_ylim(center[1] - radius, center[1] + radius)
    ax3.set_zlim(center[2] - radius, center[2] + radius)
    ax3.plot([-20, 20], [0, 0], [0, 0], color="#ff7a18", linewidth=3.0, label="source candidate axis")
    ax3.view_init(elev=24, azim=-58)
    ax3.set_title("Observed source surface · normal-colored faces", color="white", fontsize=13)
    ax3.set_facecolor("#0b1220")
    ax3.tick_params(colors="#7890a8", labelsize=7)
    ax3.legend(loc="upper right", fontsize=8)

    axp = figure.add_subplot(grid[0, 1])
    axp.scatter(vertices[:, 1], vertices[:, 2], s=2.1, c="#8bd8ff", alpha=0.55, rasterized=True)
    arc = record.selected_arc or (record.arcs[0] if record.arcs else None)
    station = arc.stations[len(arc.stations) // 2] if arc else None
    if station:
        segments = [[segment.p0[1:], segment.p1[1:]] for segment in station.segments]
        axp.add_collection(LineCollection(segments, colors="#e2e8f0", linewidths=0.65, alpha=0.7))
        for bin_index, bin_ in enumerate(station.bins):
            angle = (bin_index + 0.5) * math.pi / 12.0
            direction = np.asarray((math.cos(angle), math.sin(angle)))
            for chord in bin_.chords:
                midpoint = chord.midpoint[1:]
                half = 0.5 * chord.distance * direction
                axp.plot(
                    [midpoint[0] - half[0], midpoint[0] + half[0]],
                    [midpoint[1] - half[1], midpoint[1] + half[1]],
                    color="#2dd4bf" if not chord.ambiguous else "#fb7185",
                    linewidth=0.55,
                    alpha=0.7,
                )
        if station.midpoint_path:
            path = np.asarray([point[1:] for point in station.midpoint_path])
            axp.plot(path[:, 0], path[:, 1], color="#fbbf24", linewidth=2.4, marker=".", markersize=3)
    axp.set_aspect("equal", adjustable="datalim")
    axp.set_title("Mid-station YZ evidence · chords and midpoint path", color="white", fontsize=13)
    axp.set_facecolor("#0b1220")
    axp.tick_params(colors="#7890a8", labelsize=7)

    axxy = figure.add_subplot(grid[1, 0])
    axxy.scatter(vertices[:, 0], vertices[:, 1], s=1.8, color="#9bdcff", alpha=0.6, rasterized=True)
    axxy.plot([-20, 20], [0, 0], color="#ff7a18", linewidth=2.6)
    axxy.set_aspect("equal", adjustable="datalim")
    axxy.set_title("XY axial projection · identical line result", color="white", fontsize=13)
    axxy.set_facecolor("#0b1220")
    axxy.tick_params(colors="#7890a8", labelsize=7)

    axh = figure.add_subplot(grid[1, 1])
    if arc:
        owned = np.asarray([[int(bin_.owned) for bin_ in item.bins] for item in arc.stations])
        axh.imshow(owned, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        axh.set_yticks(range(len(arc.stations)), [f"S{i:02d}" for i in range(len(arc.stations))])
        axh.set_xticks(range(12), [f"B{i:02d}" for i in range(12)], rotation=45)
    axh.set_title("Frozen PNCT station/bin support", color="white", fontsize=13)
    axh.set_facecolor("#0b1220")
    axh.tick_params(colors="#d8e5f0", labelsize=8)

    figure.suptitle(
        "SOURCE_OBSERVATION_EQUIVALENCE_WITNESS_V1 · role-blind surface panel",
        color="white",
        fontsize=18,
        y=0.985,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=256, facecolor=figure.get_facecolor(), metadata={"Software": "ShermanQC"})
    plt.close(figure)
    with Image.open(output) as image:
        if image.size != (4096, 4096):
            raise RuntimeError(f"expected 4096x4096 audit, got {image.size}")


def _caption_surface(source: Path, output: Path, *, role: str, interpretation: str, required: int) -> None:
    with Image.open(source).convert("RGB") as image:
        canvas = Image.new("RGB", (4096, 4096), "#050910")
        canvas.paste(image.crop((0, 0, 4096, 3920)), (0, 176))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
            small = ImageFont.truetype("DejaVuSans.ttf", 30)
        except OSError:
            font = ImageFont.load_default()
            small = font
        draw.text((54, 34), f"{role} · {interpretation}", fill="#f8fafc", font=font)
        draw.text(
            (54, 102),
            f"Required semantic output: {required} axis  ·  Caption added after role-blind rendering",
            fill="#b9c9d8",
            font=small,
        )
        canvas.save(output, format="PNG", optimize=False)


def execute(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "execution.lock"
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    os.write(descriptor, b"SOURCE_OBSERVATION_EQUIVALENCE_WITNESS_V1\nEXECUTION_COUNT=1\n")
    os.close(descriptor)

    total_started = time.perf_counter()
    rss_started = _rss_bytes()
    vertices, triangles = circular_fold(
        angle_deg=90.0,
        mid_radius=12.0,
        thickness=8.0,
        length=40.0,
    )
    vertices.setflags(write=False)
    triangles.setflags(write=False)
    row = _row(40.0)
    seeds = _seed_normals(90.0)

    preprocess_started = time.perf_counter()
    frozen_record = analyze_candidate(
        vertices=vertices,
        triangles=triangles,
        row=row,
        scale=1.0,
        seed_normals=seeds,
    )
    preprocess_elapsed = time.perf_counter() - preprocess_started
    frozen_detector_output = frozen_record.encode()
    frozen_detector_digest = hashlib.sha256(frozen_detector_output).hexdigest()
    frozen_axis_count = int(frozen_record.retain)

    witness_started = time.perf_counter()
    observations = _variant_observations(vertices, triangles, angle_deg=90.0, length=40.0)
    evidence_first, pairs_first = _build_evidence(
        observations,
        axis_count=frozen_axis_count,
        detector_digest=frozen_detector_digest,
    )
    evidence_second, _pairs_second = _build_evidence(
        observations,
        axis_count=frozen_axis_count,
        detector_digest=frozen_detector_digest,
    )
    replay_first = canonical_json_bytes(evidence_first)
    replay_second = canonical_json_bytes(evidence_second)
    replay_pass = replay_first == replay_second
    witness_elapsed = time.perf_counter() - witness_started
    witness_added_peak_rss = max(0, _rss_bytes() - rss_started)

    base_pair = pairs_first[0]
    (output / "NI01.worker_output.bin").write_bytes(base_pair["positive_output"])
    (output / "NI02.worker_output.bin").write_bytes(base_pair["negative_output"])
    (output / "serialized_evidence.json").write_bytes(replay_first)
    harness_records = {
        "fixture_storage": "existing generator reference; source mesh is not duplicated",
        "fixture_name": FIXTURE_NAME,
        "fixture_ref": observations[0].sha256,
        "records": evidence_first["roles"],
        "worker_received_fields": [
            "source coordinates",
            "triangle connectivity",
            "span endpoints",
            "scale",
            "seed normals",
        ],
        "worker_prohibited_fields": [
            "fixture role",
            "physical construction",
            "expected count",
            "filename/path identity",
            "gold/labels",
            "CAD/drawing",
            "material/process record",
            "benchmark outcome",
        ],
    }
    (output / "harness_records.json").write_text(json.dumps(harness_records, indent=2, sort_keys=True) + "\n")

    audit_dir = output / "audits"
    ni01_surface = audit_dir / "NI01_surface_4096.png"
    ni02_surface = audit_dir / "NI02_surface_4096.png"
    _surface_panel(vertices, triangles, frozen_record, ni01_surface)
    _surface_panel(vertices, triangles, frozen_record, ni02_surface)
    ni01_surface_sha = _sha256_file(ni01_surface)
    ni02_surface_sha = _sha256_file(ni02_surface)
    pixel_xor_nonzero = 0 if ni01_surface_sha == ni02_surface_sha else -1
    _caption_surface(
        ni01_surface,
        audit_dir / "NI01_MONOLITHIC_BEND_audit_4096.png",
        role="NI01",
        interpretation="MONOLITHIC_BEND",
        required=1,
    )
    _caption_surface(
        ni02_surface,
        audit_dir / "NI02_GROUND_FULL_PENETRATION_JOINT_audit_4096.png",
        role="NI02",
        interpretation="GROUND_FULL_PENETRATION_JOINT",
        required=0,
    )

    metamorphic_pass = all(
        pair["input_byte_identical"]
        and pair["output_byte_identical"]
        and pair["contradiction"]
        for pair in pairs_first[1:]
    )
    proof_valid = (
        base_pair["same_input_object"]
        and base_pair["input_byte_identical"]
        and base_pair["output_byte_identical"]
        and base_pair["contradiction"]
        and base_pair["semantic_success_count"] == 1
        and metamorphic_pass
        and replay_pass
        and pixel_xor_nonzero == 0
    )
    report = {
        "profile": PROFILE,
        "ruling": "FORMAL_STOP_SCAN_SURFACE_CAUSAL_NONIDENTIFIABILITY_STOP_V21",
        "source_only_lane_closed": bool(proof_valid),
        "v21_detector_authorized": False,
        "execution_count": 1,
        "preprocess_invocations": 1,
        "fixture_copy_count": 0,
        "source_only": True,
        "real_scans_opened": False,
        "u02_opened": False,
        "c017_opened": False,
        "locked_35_opened": False,
        "held_out_opened": False,
        "gold_used": False,
        "labels_used": False,
        "cad_used": False,
        "identity_used": False,
        "expected_count_passed_to_worker": False,
        "fixture_ref": observations[0].sha256,
        "fixture_input_bytes": len(observations[0].payload),
        "frozen_detector_output_sha256": frozen_detector_digest,
        "frozen_axis_count": frozen_axis_count,
        "base_pair": public_pair_record(base_pair),
        "metamorphic_pass": metamorphic_pass,
        "metamorphic_pairs": [public_pair_record(pair) for pair in pairs_first[1:]],
        "replay_pass": replay_pass,
        "replay_sha256": hashlib.sha256(replay_first).hexdigest(),
        "surface_panel_sha256": ni01_surface_sha,
        "surface_panel_byte_identical": ni01_surface_sha == ni02_surface_sha,
        "surface_pixel_xor_nonzero": pixel_xor_nonzero,
        "manual_visual_ruling_required": True,
        "preprocess_elapsed_seconds": preprocess_elapsed,
        "witness_elapsed_beyond_preprocess_seconds": witness_elapsed,
        "witness_added_peak_rss_bytes_before_visuals": witness_added_peak_rss,
        "total_elapsed_seconds_including_visuals": time.perf_counter() - total_started,
        "hard_bounds": {
            "witness_beyond_preprocess_lt_1_second": witness_elapsed < 1.0,
            "witness_added_peak_rss_lt_64_mib_before_visuals": witness_added_peak_rss < 64 * 1024 * 1024,
            "nonvisual_artifact_lt_1_mib": None,
        },
        "smallest_capability_opening": {
            "name": "CO_REGISTERED_THROUGH_THICKNESS_ULTRASONIC_BSCAN",
            "candidate_local": True,
            "stations": [0.25, 0.50, 0.75],
            "maximum_candidate_strips_per_scan": 16,
            "explicit_user_authorization_required": True,
        },
        "proof_valid": bool(proof_valid),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    nonvisual_bytes = sum(
        path.stat().st_size
        for path in output.iterdir()
        if path.is_file() and path.suffix.lower() != ".png"
    )
    report["nonvisual_artifact_bytes"] = nonvisual_bytes
    report["hard_bounds"]["nonvisual_artifact_lt_1_mib"] = nonvisual_bytes < 1024 * 1024
    report["proof_valid"] = bool(proof_valid and report["hard_bounds"]["nonvisual_artifact_lt_1_mib"])
    report["source_only_lane_closed"] = report["proof_valid"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.output)
    print(
        json.dumps(
            {
                "proof_valid": report["proof_valid"],
                "source_only_lane_closed": report["source_only_lane_closed"],
                "metamorphic_pass": report["metamorphic_pass"],
                "replay_pass": report["replay_pass"],
                "surface_pixel_xor_nonzero": report["surface_pixel_xor_nonzero"],
                "witness_elapsed_beyond_preprocess_seconds": report["witness_elapsed_beyond_preprocess_seconds"],
                "witness_added_peak_rss_bytes_before_visuals": report["witness_added_peak_rss_bytes_before_visuals"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["proof_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
