#!/usr/bin/env python3
"""Build the GPT Pro transition-atomization follow-up as individual files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "output/professor_package"
    / "Sherman_QC_F1_Transition_Atomization_Followup_20260805"
)

FILES = {
    "00_GPT_PRO_PROMPT.md": ROOT
    / "docs/research/PROFESSOR_F1_TRANSITION_ATOMIZATION_DROPOFF_20260805.md",
    "01_GOLD37_CHECKPOINT.md": ROOT
    / "docs/research/STRUCTURED_UNION_GOLD37_E2E_20260805.md",
    "02_PRIOR_PROFESSOR_RESPONSE.md": ROOT
    / "docs/research/PROFESSOR_F1_STRUCTURED_FOLLOWUP_RESPONSE_20260804.md",
    "04_CANDIDATE_RECALL.json": ROOT
    / "output/research/f1-gold27-20260803/structured_union_exact_v1/candidate_recall_score.json",
    "05_BIRTH_V1_PRECHECK.json": ROOT
    / "output/research/f1-gold27-20260803/structured_union_exact_v1/birth_v1_precheck.json",
    "06_SIDE_MINIMUM_SCREEN.json": ROOT
    / "output/research/f1-gold27-20260803/structured_union_exact_v1/side_minimum_posthoc_screen.json",
    "07_BIRTH_V1_STAGE_CENSUS.json": ROOT
    / "output/research/f1-gold27-20260803/structured_union_exact_v1/birth_v1_stage_census.json",
    "09_AUTHORITATIVE_CONTACT_1.jpg": ROOT
    / "output/research/f1-gold27-20260803/structured_union_exact_v1/visual_review/authoritative_contact_sheet_1.jpg",
    "10_AUTHORITATIVE_CONTACT_2.jpg": ROOT
    / "output/research/f1-gold27-20260803/structured_union_exact_v1/visual_review/authoritative_contact_sheet_2.jpg",
    "13_patch_graph_latent_decomposition.py": ROOT
    / "backend/patch_graph_latent_decomposition.py",
    "14_build_f1_scan_only_candidate_pool.py": ROOT
    / "scripts/build_f1_scan_only_candidate_pool.py",
    "15_f1_structured_residual_birth.py": ROOT
    / "backend/f1_structured_residual_birth.py",
    "16_f1_structured_union_graph.py": ROOT
    / "backend/f1_structured_union_graph.py",
    "17_solve_f1_structured_union_cpsat.py": ROOT
    / "scripts/solve_f1_structured_union_cpsat.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    missing = [str(source) for source in FILES.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing professor evidence:\n" + "\n".join(missing))

    PACKAGE.mkdir(parents=True, exist_ok=True)
    expected_names = set(FILES) | {"manifest.json"}
    for stale in PACKAGE.iterdir():
        if stale.is_file() and stale.name not in expected_names:
            stale.unlink()

    for destination_name, source in FILES.items():
        shutil.copy2(source, PACKAGE / destination_name)

    rows: list[dict[str, Any]] = []
    for path in sorted(PACKAGE.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-transition-atomization-20260805-v1",
        "submission_style": "gpt_5_6_pro_file_upload_not_email",
        "runtime_promotion_authorized": False,
        "frozen_baseline": "Gold37 structured union A0",
        "primary_decision": "TRANSITION_ATOM_V2 representation before learning",
        "file_count_excluding_manifest": len(rows),
        "files": rows,
    }
    (PACKAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "package": str(PACKAGE),
                "files_including_manifest": len(rows) + 1,
                "prompt": str(PACKAGE / "00_GPT_PRO_PROMPT.md"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
