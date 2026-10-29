#!/usr/bin/env python3
"""Build the provenance-safe CAD-guided physical Gold13 slice.

The slice contains every frozen line-annotated scan whose part family has a
locally available STEP reference.  Labels and expected counts remain scoring
metadata; the blind CAD+PLY runner replaces them with unknown values before
crossing the inference subprocess boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
OUTPUT = ROOT / "data/bend_corpus/evaluation_slices/cad_guided_gold13_physical_20260822.json"
CAD_ROOT = ROOT.parent / "source-assets/02_CAD_QC_3D"

CAD_BY_PART = {
    "10839000": CAD_ROOT / "10839000_01.stp",
    "15651004": CAD_ROOT / "15651004_01.stp",
    "37361005": CAD_ROOT / "37361005_01.stp",
    "38172000": CAD_ROOT / "38172000_01.stp",
    "40500000": CAD_ROOT / "40500000_0 1.stp",
    "47959001": CAD_ROOT / "47959001_01PREL.stp",
    "49001000": CAD_ROOT / "49001000_A-PRELIM.stp",
    "49024000": CAD_ROOT / "49024000_A.stp",
}
ALLOWED_TIERS = frozenset(
    {"authoritative_user_verified", "codex_manual_multiview_drawing_backed"}
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entries = []
    for row in catalog["cases"]:
        part_id = str(row["part_id"])
        cad = CAD_BY_PART.get(part_id)
        if cad is None or str(row.get("truth_tier")) not in ALLOWED_TIERS:
            continue
        scan = Path(str(row["scan_path"])).expanduser().resolve()
        if not scan.is_file() or _sha(scan) != str(row["scan_sha256"]):
            raise ValueError(f"scan provenance mismatch: {row['case_id']}")
        if not cad.is_file():
            raise ValueError(f"missing CAD: {cad}")
        case_token = "CG" + hashlib.sha256(
            (str(row["case_id"]) + "\0" + str(row["scan_sha256"])).encode("utf-8")
        ).hexdigest()[:10].upper()
        entries.append(
            {
                "case_token": case_token,
                "part_key": part_id,
                "cad_path": str(cad.resolve()),
                "cad_sha256": _sha(cad),
                "scan_path": str(scan),
                "scan_sha256": str(row["scan_sha256"]),
                "expectation": {
                    "state": "unknown",
                    "expected_bend_count": int(row["bend_count"]),
                    "expected_conventional_bend_count": int(row["bend_count"]),
                    "truth_record_id": str(row["case_id"]),
                    "truth_tier": str(row["truth_tier"]),
                },
            }
        )
    entries.sort(key=lambda item: item["case_token"])
    if len(entries) != 13 or len({row["part_key"] for row in entries}) != 8:
        raise ValueError("expected exactly 13 scans across 8 CAD-backed families")
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cad_guided_gold13_physical_slice",
        "target": "visible_conventional_bend_lines_in_scan_raw_frame",
        "source_only_zero_shot": False,
        "cad_guided_zero_shot": True,
        "promotion_eligible": False,
        "reason_not_promotion_eligible": "development_consumed_and_mixed_truth_tiers",
        "inference_contract": {
            "cad_and_scan_cross_boundary": True,
            "expected_count_crosses_boundary": False,
            "gold_lines_cross_boundary": False,
            "part_identity_crosses_boundary": False,
        },
        "case_count": len(entries),
        "family_count": len({row["part_key"] for row in entries}),
        "entries": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "cases": len(entries), "families": 8}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
