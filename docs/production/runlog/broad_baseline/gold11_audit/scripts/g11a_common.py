"""Shared helpers for the Gold11 audit roll-up (read-only over every input).

Nothing here re-runs an auditor, relabel, verifier, prep, lines or engine step. It reads:
  - the finished agent outputs of workflow wf_37989420-76c (sherman-gold11-label-audit-continue),
    from its subagent transcripts, plus the 5 auditor results that run re-used from the earlier
    killed run's cache (scratchpad cont/audit_done/*.json);
  - the human Gold11 lines through research/broad_baseline/score.load_gold (raw-scan frame, frozen
    frame records), never editing them;
  - the AI relabel lines (gold11_audit/relabel/<scan>/lines.json) and labels
    (data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
BB = ROOT / "docs" / "production" / "runlog" / "broad_baseline"
AUDIT = BB / "gold11_audit"
RELABEL_ROOT = ROOT / "data" / "bend_corpus" / "annotations" / "ai_bend_lines_gold11_relabel_v1"
HUMAN_MIRROR_ROOT = ROOT / "data" / "bend_corpus" / "annotations" / "human_bend_lines_v1"
CASE_LIST = BB / "case_list_broad_v3.json"
FRAME_RECORDS = BB / "broad_frame_records.json"
TRIAGE = BB / "triage" / "triage_results_24.json"

WORKFLOW_RUN = "wf_37989420-76c"
TRANSCRIPTS = Path.home() / ".claude/projects/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/subagents/workflows" / WORKFLOW_RUN
CACHE_DIR = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/cont/audit_done")

sys.path.insert(0, str(ROOT / "research" / "broad_baseline"))
import bb_common as bb  # noqa: E402  (also puts ROOT/backend/scripts on sys.path, read-only)

# scan_id -> asset id of the AI relabel label (= the human_bend_lines_v1 mirror name)
RELABEL_ASSET = {
    "sc_cec08dfca9ab": "17293ebcf404be5a87e25b1f",
    "sc_7a1773eca9fa": "3aa393ad34b7ace90d84985c",
    "sc_4f8188b4d1a8": "b7155cc16c9bf0290d331831",
    "sc_d1732de18023": "05df0a708054946a1d77f7ee",
    "sc_70b39f530f00": "8778c5e57263f92092cf88b4",
    "sc_152f1b1aacb2": "8370b5e44945adfba2f2fb34",
    "sc_07062eee17aa": "a962166c3019434c10628722",
    "sc_71cf0b216f43": "c615c4cd283605201afac7e8",
    "sc_08686dc1d5d4": "72bd51c4885d493d805f1108",
    "sc_f2475286312b": "a6a0b1c3f8052cb3a45412e8",
    "sc_2714fb7e0c0b": "c57b956b11cab8f4a8d3dc17",
}
SCANS = list(RELABEL_ASSET)

# The single-agent triage verdicts on the 6 triaged Gold11 scans, exactly as the broad baseline
# report counted them (research/broad_baseline/report/build_summary.py LABEL_DEFECT_SCANS; raw
# verdicts in triage/triage_results_24.json).
TRIAGE_GOLD11 = {
    "sc_71cf0b216f43": {"wrong": ["M01", "M02", "M05", "M06", "M07", "M08"], "omitted_real_bends": 0,
                        "omission_note": "two 35 deg joggles noted but not counted (policy question)"},
    "sc_f2475286312b": {"wrong": ["M01", "M03", "M04", "M05", "M06", "M07", "M08"], "omitted_real_bends": 3,
                        "omission_note": "+PC1 lip and two ear bends; M02 noted as off the corner but not counted (it matched)"},
    "sc_cec08dfca9ab": {"wrong": ["HB1", "HB2", "HB3", "HB4", "HB5", "HB6"], "omitted_real_bends": 0,
                        "omission_note": "none"},
    "sc_152f1b1aacb2": {"wrong": ["M03", "M04", "M05", "M06"], "omitted_real_bends": 4,
                        "omission_note": "two 20 mm lips and two 14 mm tabs; M01/M02/M07 over-long extents noted but not counted"},
    "sc_2714fb7e0c0b": {"wrong": ["B06", "B08"], "omitted_real_bends": 2,
                        "omission_note": "the two 44 mm end-wall bends"},
    "sc_4f8188b4d1a8": {"wrong": ["B07", "B08"], "omitted_real_bends": 2,
                        "omission_note": "the two end-flange hinges (STEP C1/C9); slot-end jogs C5/C7 not verified"},
}

# The compact audit record the workflow handed to this step (its task DATA), reduced to the
# fields that are not truncated there: relabel_pass and, per auditor (1, 2, 3), the verdict
# counts, the missing-bend count and the CONVENTION_OFFSET offsets in mm. Used only to check
# that the transcript extraction reproduces the workflow's own record.
EXPECTED_COMPACT = {
    "sc_cec08dfca9ab": (True, [({"REAL_ERROR": 6}, 4, []), ({"REAL_ERROR": 6}, 4, []), ({"REAL_ERROR": 6}, 4, [])]),
    "sc_7a1773eca9fa": (True, [({"CONVENTION_OFFSET": 5}, 0, [2.8, 4.9, 5.5, 3.2, 5.9]),
                               ({"OK": 2, "CONVENTION_OFFSET": 3}, 0, [5.3, 5.45, 5.58]),
                               ({"CONVENTION_OFFSET": 5}, 0, [2.8, 5.1, 5.4, 3.3, 5.7])]),
    "sc_4f8188b4d1a8": (True, [({"OK": 3, "CONVENTION_OFFSET": 3, "NOT_A_BEND": 2, "REAL_ERROR": 2}, 4, [0.99, 1.36, 1.06]),
                               ({"OK": 6, "NOT_A_BEND": 2, "REAL_ERROR": 2}, 4, []),
                               ({"OK": 3, "CONVENTION_OFFSET": 3, "NOT_A_BEND": 2, "REAL_ERROR": 2}, 4, [1.01, 1.38, 1.06])]),
    "sc_d1732de18023": (True, [({"REAL_ERROR": 2, "CONVENTION_OFFSET": 2}, 0, [1.61, 1.08]),
                               ({"REAL_ERROR": 2, "CONVENTION_OFFSET": 2}, 0, [1.61, 1.08]),
                               ({"REAL_ERROR": 2, "OK": 2}, 0, [])]),
    "sc_70b39f530f00": (True, [({"CONVENTION_OFFSET": 3}, 0, [2.6, 2.5, 2.6]),
                               ({"CONVENTION_OFFSET": 3}, 0, [2.59, 2.51, 2.59]),
                               ({"CONVENTION_OFFSET": 3}, 0, [2.6, 2.5, 2.6])]),
    "sc_152f1b1aacb2": (True, [({"REAL_ERROR": 3, "NOT_A_BEND": 4}, 3, []), ({"REAL_ERROR": 3, "NOT_A_BEND": 4}, 4, []),
                               ({"REAL_ERROR": 3, "NOT_A_BEND": 4}, 4, [])]),
    "sc_07062eee17aa": (False, [({"REAL_ERROR": 2, "OK": 2, "NOT_A_BEND": 2}, 0, []),
                                ({"REAL_ERROR": 2, "OK": 2, "NOT_A_BEND": 2}, 0, []),
                                ({"REAL_ERROR": 2, "OK": 2, "NOT_A_BEND": 2}, 2, [])]),
    "sc_71cf0b216f43": (True, [({"REAL_ERROR": 6, "OK": 2}, 4, []),
                               ({"CONVENTION_OFFSET": 2, "OK": 2, "REAL_ERROR": 4}, 4, [4.19, 4.77]),
                               ({"CONVENTION_OFFSET": 2, "OK": 2, "REAL_ERROR": 4}, 4, [4.08, 4.53])]),
    "sc_08686dc1d5d4": (True, [({"REAL_ERROR": 2, "NOT_A_BEND": 2}, 6, []),
                               ({"CONVENTION_OFFSET": 2, "NOT_A_BEND": 2}, 6, [3.26, 2.23]),
                               ({"REAL_ERROR": 2, "NOT_A_BEND": 2}, 6, [])]),
    "sc_f2475286312b": (True, [({"CONVENTION_OFFSET": 3, "REAL_ERROR": 4, "NOT_A_BEND": 1}, 5, [4.68, 4.57, 5.14]),
                               ({"CONVENTION_OFFSET": 3, "REAL_ERROR": 2, "NOT_A_BEND": 3}, 7, [4.68, 4.58, 5.14]),
                               ({"CONVENTION_OFFSET": 3, "REAL_ERROR": 2, "NOT_A_BEND": 3}, 5, [4.67, 4.61, 5.15])]),
    "sc_2714fb7e0c0b": (True, [({"OK": 6, "CONVENTION_OFFSET": 2, "NOT_A_BEND": 2}, 4, [2.47, 3.08]),
                               ({"OK": 6, "CONVENTION_OFFSET": 1, "NOT_A_BEND": 2, "REAL_ERROR": 1}, 4, [2.49]),
                               ({"OK": 6, "CONVENTION_OFFSET": 2, "NOT_A_BEND": 2}, 3, [2.49, 3.1])]),
}


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any) -> Path:
    return bb.write_json(path, payload)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def r(value: float | None, nd: int = 3) -> float | None:
    if value is None:
        return None
    value = float(value)
    return round(value, nd) if math.isfinite(value) else None
