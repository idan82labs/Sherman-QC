#!/usr/bin/env python3
"""Execute the frozen PNCT_V1 source-free falsification suite once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_projective_normal_chord_transport_v20 import analyze_candidate, encode_sidecar  # noqa: E402
from research.f1_pnct_v20_fixtures import source_free_cases  # noqa: E402


def _row(name: str, length: float) -> dict:
    return {"candidate_id": name, "candidate_key": hashlib.sha256(name.encode()).hexdigest()[:32], "span_endpoints": [[-length / 2, 0, 0], [length / 2, 0, 0]]}


def _analyze(name, vertices, triangles, angle, length, *, scale=1.0, row=None, seeds=None):
    if seeds is None:
        seeds = ((0, 1, 0), (0, math.cos(math.radians(angle)), math.sin(math.radians(angle))))
    return analyze_candidate(vertices=vertices, triangles=triangles, row=row or _row(name, length), scale=scale, seed_normals=seeds)


def run(output: Path) -> dict:
    positives, negatives = source_free_cases()
    started = time.perf_counter()
    records, rows = {}, []
    for expected, cases in ((True, positives), (False, negatives)):
        for name, (vertices, triangles, angle, length) in cases.items():
            begin = time.perf_counter()
            record = _analyze(name, vertices, triangles, angle, length)
            records[name] = record
            rows.append({"name": name, "expected": expected, "elapsed_seconds": time.perf_counter() - begin, **record.summary()})

    metamorphic = []
    for name, (vertices, triangles, angle, length) in {**positives, **negatives}.items():
        base = records[name].encode()
        variants = []
        variants.append(("triangle_permutation", vertices, triangles[::-1], _row(name, length), None, 1.0))
        variants.append(("cyclic_vertices", vertices, np.roll(triangles, 1, axis=1), _row(name, length), None, 1.0))
        variants.append(("reversed_vertices", vertices, triangles[:, ::-1], _row(name, length), None, 1.0))
        reverse_row = _row(name, length); reverse_row["span_endpoints"].reverse()
        reverse_seeds = ((0, math.cos(math.radians(angle)), math.sin(math.radians(angle))), (0, 1, 0))
        variants.append(("axis_seed_reversal", vertices, triangles, reverse_row, reverse_seeds, 1.0))
        for degrees in (17, 53, 101):
            radians = math.radians(degrees)
            rotation = np.asarray(((math.cos(radians), -math.sin(radians), 0), (math.sin(radians), math.cos(radians), 0), (0, 0, 1)))
            translation = np.asarray((11.0 + degrees, -7.0, 23.0))
            points = vertices @ rotation.T + translation
            transformed_row = _row(name, length)
            transformed_row["span_endpoints"] = (np.asarray(transformed_row["span_endpoints"]) @ rotation.T + translation).tolist()
            seeds = tuple((np.asarray(seed) @ rotation.T).tolist() for seed in ((0, 1, 0), (0, math.cos(math.radians(angle)), math.sin(math.radians(angle)))))
            variants.append((f"rigid_{degrees}", points, triangles, transformed_row, seeds, 1.0))
        for factor in (.5, 2.0):
            scaled_row = _row(name, factor * length)
            variants.append((f"scale_{factor}", factor * vertices, triangles, scaled_row, None, factor))
        for variant_name, points, faces, row, seeds, factor in variants:
            result = _analyze(name, points, faces, angle, factor * length, scale=factor, row=row, seeds=seeds)
            metamorphic.append({"fixture": name, "variant": variant_name, "byte_identical": result.encode() == base, "retain": result.retain})

    replay = all(_analyze(name, *case[:2], case[2], case[3]).encode() == records[name].encode() for name, case in {**positives, **negatives}.items())
    output.mkdir(parents=True, exist_ok=True)
    sidecar = encode_sidecar(tuple(records.values()))
    (output / "source_free.pnct").write_bytes(sidecar)
    report = {
        "profile": "PROJECTIVE_NORMAL_CHORD_TRANSPORT_V1",
        "source_only": True, "gold_used": False, "labels_used": False, "identity_used": False,
        "positive_pass": all(row["retain"] == row["expected"] for row in rows if row["expected"]),
        "negative_pass": all(row["retain"] == row["expected"] for row in rows if not row["expected"]),
        "metamorphic_pass": all(row["byte_identical"] for row in metamorphic),
        "replay_pass": replay,
        "rows": rows, "metamorphic": metamorphic,
        "elapsed_seconds": time.perf_counter() - started,
        "p95_seconds": float(np.quantile([row["elapsed_seconds"] for row in rows], .95)),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sidecar_bytes": len(sidecar), "sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({key: report[key] for key in ("positive_pass", "negative_pass", "metamorphic_pass", "replay_pass", "p95_seconds", "peak_rss_bytes")}, sort_keys=True))
    return 0 if all(report[key] for key in ("positive_pass", "negative_pass", "metamorphic_pass", "replay_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
