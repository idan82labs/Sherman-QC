#!/usr/bin/env python3
"""Freeze hashes for the falsified Gold37 structured-union A0 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/research/f1-gold27-20260803"
DEFAULT_OUTPUT = BASE / "structured_union_exact_v1/a0_evidence_manifest.json"
DEFAULT_FILES = {
    "bounded_pool_manifest": BASE / "candidate_pool_bounded_v2/manifest.json",
    "exact_batch_summary": BASE / "structured_union_exact_v1/batch_summary.json",
    "gold37_score": BASE / "structured_union_exact_v1/gold27_score.json",
    "operational_summary": BASE / "structured_union_exact_v1/operational_summary.json",
    "candidate_recall": BASE / "structured_union_exact_v1/candidate_recall_score.json",
    "birth_v1_precheck": BASE / "structured_union_exact_v1/birth_v1_precheck.json",
    "birth_v1_stage_census": BASE / "structured_union_exact_v1/birth_v1_stage_census.json",
    "side_minimum_screen": BASE / "structured_union_exact_v1/side_minimum_posthoc_screen.json",
    "visual_review": BASE / "structured_union_exact_v1/visual_review/review.json",
    "split_integrity": BASE / "freeze/grouped_five_fold_split_integrity.json",
    "goal_readiness": BASE / "structured_union_exact_v1/generalization_goal_readiness.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(files: Mapping[str, Path]) -> dict[str, Any]:
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing A0 evidence:\n" + "\n".join(missing))
    rows = [
        {
            "artifact_id": artifact_id,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for artifact_id, path in sorted(files.items())
    ]
    score = json.loads(files["gold37_score"].read_text(encoding="utf-8"))
    readiness = json.loads(files["goal_readiness"].read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_frozen_negative_a0_evidence_manifest",
        "baseline_id": "GOLD37_STRUCTURED_UNION_A0_20260805",
        "disposition": "FALSIFIED_REPRESENTATION_NOT_PROMOTABLE",
        "research_only": True,
        "runtime_promotion_authorized": False,
        "corpus_freeze_id": score.get("corpus_freeze_id"),
        "goal_achieved": bool(readiness.get("goal_achieved")),
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def verify_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return hash/size drift without mutating the frozen manifest."""

    drift = []
    for row in manifest.get("artifacts") or ():
        path = Path(str(row["path"]))
        observed_size = path.stat().st_size if path.is_file() else None
        observed_sha = _sha256(path) if path.is_file() else None
        if observed_size != int(row["size_bytes"]) or observed_sha != str(row["sha256"]):
            drift.append(
                {
                    "artifact_id": str(row["artifact_id"]),
                    "path": str(path),
                    "expected_size_bytes": int(row["size_bytes"]),
                    "observed_size_bytes": observed_size,
                    "expected_sha256": str(row["sha256"]),
                    "observed_sha256": observed_sha,
                }
            )
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(DEFAULT_FILES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file():
        frozen = json.loads(args.output.read_text(encoding="utf-8"))
        drift = verify_manifest(frozen)
        if drift:
            raise RuntimeError(
                "frozen A0 evidence drifted; refusing to overwrite:\n"
                + json.dumps(drift, indent=2, sort_keys=True)
            )
        if frozen != manifest:
            raise RuntimeError(
                "rebuilt A0 manifest differs despite matching artifact hashes; "
                "refusing to overwrite"
            )
        print(json.dumps(frozen, indent=2, sort_keys=True))
        return 0
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
