#!/usr/bin/env python3
"""Build the direct-GPT professor package for the 45796000 blind holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.f1_bend_evidence_score import score_bend_evidence  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines  # noqa: E402


BASE = ROOT / "output/research/f1-unseen-challenge-20260804/45796000"
PACKAGE = ROOT / "output/professor_package/Sherman_QC_F1_Structured_Inference_Dropoff_20260804"
ZIP_PATH = PACKAGE.with_suffix(".zip")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, relative: str) -> Path:
    destination = PACKAGE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _context() -> dict:
    truth = _read(BASE / "manual_adjudication/cad_drawing_backed_bend_labels.json")
    evaluation = _read(BASE / "manual_adjudication/line_evaluation.json")
    pool = _read(BASE / "candidate_pool/01_45796000.json")
    inference = _read(BASE / "v23_locked_prediction/inference_summary.json")
    routed = _read(BASE / "v23_locked_prediction/routed_prediction_owned_regions.json")["owned_bend_regions"]
    mesh = _read(BASE / "v23_locked_prediction/mesh_prediction_owned_regions.json")["owned_bend_regions"]
    final = _read(BASE / "v23_locked_prediction/prediction_owned_regions.json")["owned_bend_regions"]
    gold = truth["owned_bend_regions"]
    thresholds = Thresholds()

    source_results = []
    sources = sorted({str(row.get("candidate_source") or "unknown") for row in pool["candidates"]})
    for source in sources:
        candidates = [row for row in pool["candidates"] if str(row.get("candidate_source") or "unknown") == source]
        result = match_lines(gold, candidates, thresholds)
        source_results.append(
            {
                "source": source,
                "candidate_count": len(candidates),
                "true_positive": result["true_positive"],
                "false_positive": result["false_positive"],
                "false_negative": result["false_negative"],
                "matched_gold_ids": [match["gold_id"] for match in result["matches"]],
            }
        )

    routed_result = match_lines(gold, routed, thresholds)
    mesh_result = match_lines(gold, mesh, thresholds)
    final_result = match_lines(gold, final, thresholds)
    matched_routed = {match["prediction_id"]: match["gold_id"] for match in routed_result["matches"]}
    routed_scores = []
    for bend in routed:
        bend_id = str(bend.get("bend_id") or bend.get("candidate_id"))
        score = score_bend_evidence(bend, pool["line_objects"])
        routed_scores.append(
            {
                "bend_id": bend_id,
                "postlock_class": "true_bend" if bend_id in matched_routed else "corner_or_perimeter_alias",
                "gold_id": matched_routed.get(bend_id),
                "evidence_score": score["evidence_score"],
                "evidence_rating": score["evidence_rating"],
                "review_queue": score["review_queue"],
                "parallel_lane_ownership": score["parallel_lane_ownership"],
            }
        )

    return {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-structured-inference-20260804-v1",
        "evidence_boundary": {
            "v23_frozen_before_45796000_inference": True,
            "cad_and_drawing_opened_after_prediction_hash": True,
            "authoritative_independent_human_truth": False,
            "truth_status": "postlock_STEP_cylinder_pair_and_drawing_backed_Codex_adjudication",
            "runtime_promotion_authorized": False,
        },
        "locked_inference": {
            "algorithm": inference["algorithm"],
            "prediction_sha256": _sha256(BASE / "v23_locked_prediction/prediction_owned_regions.json"),
            "replay_sha256": _sha256(BASE / "v23_locked_replay/prediction_owned_regions.json"),
            "replay_byte_identical": (
                (BASE / "v23_locked_prediction/prediction_owned_regions.json").read_bytes()
                == (BASE / "v23_locked_replay/prediction_owned_regions.json").read_bytes()
            ),
            "scan_sha256": inference["inputs"]["scan_sha256"],
            "selected_lane": inference["routing"]["selected_lane"],
            "routed_prediction_count": inference["routing"]["routed_prediction_count"],
            "mesh_prediction_count": inference["routing"]["mesh_prediction_count"],
            "final_prediction_count": inference["prediction_count"],
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
            "uses_cad": False,
        },
        "postlock_truth": {
            "paired_STEP_conventional_bend_count": truth["paired_conventional_bend_count"],
            "STEP_cylinder_surface_count_after_area_gate": truth["step_cylinder_surface_count_after_area_gate"],
            "sheet_thickness_mm": truth["sheet_thickness_mm"],
            "alignment": truth["alignment"],
            "bend_lines": gold,
        },
        "line_contract": {
            "axis_angle_deg": thresholds.angle_deg,
            "transverse_offset_mm": thresholds.transverse_offset_mm,
            "gold_span_overlap": thresholds.gold_span_overlap,
            "prediction_span_overlap": thresholds.prediction_span_overlap,
        },
        "lane_results": {
            "routed_12": {key: routed_result[key] for key in ("true_positive", "false_positive", "false_negative", "matches", "unmatched_prediction_ids")},
            "mesh_6": {key: mesh_result[key] for key in ("true_positive", "false_positive", "false_negative", "matches", "unmatched_gold_ids", "unmatched_prediction_ids")},
            "final_v23_6": {key: final_result[key] for key in ("true_positive", "false_positive", "false_negative", "matches", "unmatched_gold_ids", "unmatched_prediction_ids")},
        },
        "candidate_recall_by_source": source_results,
        "routed_candidate_evidence_scores": routed_scores,
        "observed_failure": {
            "candidate_generation_ceiling": "not_the_blocker_on_this_part",
            "all_eight_true_bends_present_in_routed_lane": True,
            "routed_extra_alias_count": 4,
            "lane_router_selected_incomplete_mesh_lane": True,
            "scalar_score_separates_true_from_false": False,
            "highest_false_score": max(row["evidence_score"] for row in routed_scores if row["postlock_class"] != "true_bend"),
            "lowest_true_score": min(row["evidence_score"] for row in routed_scores if row["postlock_class"] == "true_bend"),
            "diagnosis": "structured semantic selection failure: complete bend set and corner/perimeter aliases coexist",
        },
        "historical_context": {
            "gold27_v23": "182 TP, 0 FP, 0 FN; 37/37 development scans exact",
            "honest_locked_v21_unseen_exact": "0/3",
            "45796000_is_first_untouched_v23_holdout": True,
        },
        "raw_line_evaluation": evaluation,
    }


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=colors.HexColor("#102A43"), alignment=TA_CENTER, spaceAfter=8 * mm))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=colors.HexColor("#102A43"), spaceAfter=4 * mm))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=14, textColor=colors.HexColor("#1F5D7A"), spaceBefore=2 * mm, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.3, textColor=colors.HexColor("#243B53"), spaceAfter=2.5 * mm))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.5, leading=9.4, textColor=colors.HexColor("#486581")))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.white, backColor=colors.HexColor("#C44536"), borderPadding=8, spaceBefore=3 * mm, spaceAfter=4 * mm))
    return styles


def _image(path: Path, max_width_mm: float, max_height_mm: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width_mm * mm / width, max_height_mm * mm / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _table(data, widths=None, header=True, font_size=7.5):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCCDC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F5D7A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#627D98"))
    canvas.drawString(13 * mm, 7 * mm, "Sherman QC F1 - research-only structured-inference review")
    canvas.drawRightString(284 * mm, 7 * mm, f"Page {document.page}")
    canvas.restoreState()


def _build_pdf(context: dict, output: Path) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(str(output), pagesize=landscape(A4), rightMargin=13 * mm, leftMargin=13 * mm, topMargin=12 * mm, bottomMargin=12 * mm)
    story = []
    story.extend([
        Spacer(1, 16 * mm),
        Paragraph("Sherman QC F1: Untouched v23 Holdout", styles["Cover"]),
        Paragraph("45796000 structured-inference architecture dropoff", ParagraphStyle(name="Sub", parent=styles["H1x"], alignment=TA_CENTER, fontSize=16)),
        Spacer(1, 8 * mm),
        Paragraph("LOCKED RESULT: 3 TP, 3 FP, 5 FN against 8 STEP-backed bend axes", styles["Callout"]),
        _table([
            ["Evidence", "Result", "Interpretation"],
            ["Blind lock", "v23 predicted 6", "No identity, count, CAD, drawing, or manual truth exposed"],
            ["Determinism", "SHA-256 replay identical", context["locked_inference"]["prediction_sha256"]],
            ["Post-lock truth", "8 bend axes", "Eight inner/outer STEP cylindrical-face pairs; 2 mm sheet"],
            ["Routed lane", "8 TP + 4 aliases", "Complete recall before semantic selection"],
            ["Chosen mesh lane", "3 TP + 3 FP", "Router discarded five true objects and retained three corner residuals"],
            ["Score overlap", "false max 84.846; true min 70.390", "No scalar threshold separates classes"],
        ], widths=[42 * mm, 64 * mm, 153 * mm], font_size=8.3),
        Spacer(1, 6 * mm),
        Paragraph("Evidence boundary", styles["H2x"]),
        Paragraph("This is the first untouched v23 family. The prediction was hashed and replayed before the STEP or drawing was opened. The post-lock truth is CAD/drawing-backed Codex adjudication, not independent human ground truth. Runtime promotion remains prohibited.", styles["Bodyx"]),
        PageBreak(),
        KeepTogether([
            Paragraph("1. Raw scan geometry", styles["H1x"]),
            Paragraph("The dense scan shows a rectangular plug box with four walls and four outward mounting flanges. The slot and circular holes are non-bend boundaries.", styles["Bodyx"]),
            _image(BASE / "v23_locked_prediction/scan_multiview.png", 258, 145),
        ]),
        PageBreak(),
        Paragraph("2. Frozen v23 output versus post-lock STEP truth", styles["H1x"]),
        Table([
            [[Paragraph("Frozen v23: 6 selected lines", styles["H2x"]), _image(BASE / "v23_locked_prediction/final_line_audit.png", 130, 72)], [Paragraph("STEP-backed truth: 8 bend axes", styles["H2x"]), _image(BASE / "manual_adjudication/cad_truth_on_scan.png", 130, 72)]],
            [Paragraph("Only F1U1, F1U2, and F1U6 satisfy the line contract. F1U3/F1U4/F1U5 are short corner residuals.", styles["Smallx"]), Paragraph("Eight large inner/outer cylindrical-face pairs align to the visible wall-to-base and wall-to-flange transitions.", styles["Smallx"])],
        ], colWidths=[134 * mm, 134 * mm], style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9FB3C8")), ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D9E2EC")), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]),
        Spacer(1, 4 * mm),
        Paragraph(f"Rigid CAD-to-scan registration: fitness {context['postlock_truth']['alignment']['fitness']:.4f}, RMSE {context['postlock_truth']['alignment']['rmse_mm']:.3f} mm.", styles["Bodyx"]),
        PageBreak(),
        Paragraph("3. The complete hypothesis existed before selection", styles["H1x"]),
        Paragraph("The routed lane contains all eight physical bends (S1-S8) plus four corner/perimeter aliases (S9-S12). The architecture problem is therefore joint semantic ownership, not candidate birth on this part.", styles["Bodyx"]),
        _image(BASE / "manual_adjudication/routed_12_on_scan.png", 258, 128),
        PageBreak(),
        Paragraph("3b. Unary evidence cannot separate the aliases", styles["H1x"]),
        Paragraph("The four false routed candidates overlap the true-bend score range. This requires relational semantics and support ownership rather than a new scalar cutoff.", styles["Bodyx"]),
    ])
    score_rows = [["ID", "Post-lock class", "Gold", "Score", "Rating", "Queue"]]
    for row in context["routed_candidate_evidence_scores"]:
        score_rows.append([row["bend_id"], row["postlock_class"], row["gold_id"] or "-", f"{row['evidence_score']:.3f}", row["evidence_rating"], row["review_queue"]])
    story.extend([
        _table(score_rows, widths=[15 * mm, 50 * mm, 20 * mm, 20 * mm, 18 * mm, 62 * mm], font_size=7.2),
        Paragraph("S10 is a false alias with score 84.846, while true S6 scores 70.390. A scalar threshold cannot solve this case.", styles["Smallx"]),
        PageBreak(),
        Paragraph("4. Candidate recall by source", styles["H1x"]),
    ])
    source_rows = [["Source", "Candidates", "TP", "FP", "FN", "Truth recall"]]
    for row in context["candidate_recall_by_source"]:
        source_rows.append([row["source"], row["candidate_count"], row["true_positive"], row["false_positive"], row["false_negative"], f"{row['true_positive']}/8"])
    story.extend([
        _table(source_rows, widths=[62 * mm, 28 * mm, 20 * mm, 25 * mm, 20 * mm, 30 * mm], font_size=8),
        Spacer(1, 6 * mm),
        Paragraph("Architecture question", styles["H2x"]),
        Paragraph("How should an identity-free scan-only system choose eight maximal flange-pair bend paths and label four competing corner/perimeter paths as non-counted, when unary geometric evidence overlaps and an alternate lane contains only a partial set?", styles["Callout"]),
        Paragraph("Requested output: a literal MAP/MDL or equivalent structured objective; node and edge variables; explicit bend, edge, hole, junction, curve, and residual labels; duplicate/chord constraints; lane-fusion logic; candidate-birth rule; deterministic optimization procedure; learnable versus fixed parameters; and falsifiable family-held-out experiments.", styles["Bodyx"]),
        PageBreak(),
        KeepTogether([
            Paragraph("5. Engineering drawing", styles["H1x"]),
            Paragraph("The drawing identifies a 2 mm aluminum plug box and directs the reviewer to the 3D CAD for additional geometry. It is supporting context; the eight-axis count comes from paired large STEP cylindrical faces after the blind lock.", styles["Bodyx"]),
            _image(ROOT / "tmp/pdfs/45796000/drawing-1.png", 258, 145),
        ]),
    ])
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    context = _context()
    _write(PACKAGE / "02_machine_context.json", context)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Sherman QC F1 structured-inference professor response",
        "type": "object",
        "additionalProperties": False,
        "required": ["submission_id", "executive_diagnosis", "formal_model", "decision_procedure", "45796000_adjudication", "experiments", "evidence_limits"],
        "properties": {
            "submission_id": {"const": context["submission_id"]},
            "executive_diagnosis": {"type": "string"},
            "formal_model": {
                "type": "object",
                "required": ["variables", "labels", "energy", "constraints", "count_rule"],
                "properties": {
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "energy": {"type": "string"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "count_rule": {"type": "string"},
                },
            },
            "decision_procedure": {"type": "string"},
            "45796000_adjudication": {
                "type": "object",
                "required": ["why_S1_to_S8_are_bends", "why_S9_to_S12_are_not", "why_mesh_lane_failed", "features_needed"],
                "properties": {
                    "why_S1_to_S8_are_bends": {"type": "string"},
                    "why_S9_to_S12_are_not": {"type": "string"},
                    "why_mesh_lane_failed": {"type": "string"},
                    "features_needed": {"type": "array", "items": {"type": "string"}},
                },
            },
            "experiments": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["name", "success", "refutation"], "properties": {"name": {"type": "string"}, "success": {"type": "string"}, "refutation": {"type": "string"}}}},
            "evidence_limits": {"type": "array", "items": {"type": "string"}},
        },
    }
    _write(PACKAGE / "03_response_schema.json", schema)

    prompt = """# Sherman QC F1 - structured bend graph architecture review

You are reviewing a deterministic, scan-only sheet-metal bend detector. This is a direct GPT submission, not an email. Inspect the uploaded evidence before answering.

## Central result

The first untouched v23 holdout, 45796000, was locked before CAD or drawing inspection. v23 selected 6 lines. Post-lock STEP topology contains eight large paired cylindrical bend bands. Under the frozen line contract, the final result is 3 TP, 3 FP, 5 FN.

The crucial observation is that the independent routed lane already contained all eight true bends plus four corner/perimeter aliases. Existing unary evidence scores overlap: a false alias scores 84.846 while a true bend scores 70.390. The chosen mesh lane contained only 3/8 true bends and three corner residuals.

## Your task

Specify the architecture needed to jointly select maximal countable bend paths while assigning corner, perimeter, hole, junction, curve, and residual support to explicit non-counted classes.

Provide:

1. Literal variables and label ontology.
2. Exact MAP/MDL energy or equivalent structured objective.
3. Unary evidence terms with units and normalization.
4. Pairwise/higher-order terms for continuation, duplicates, shared flange ownership, corner aliases, chord rejection, and lane fusion.
5. A missing-bend birth rule for cases where candidate recall is not complete.
6. A deterministic solver and tie-breaking procedure.
7. Which parameters are physical constants, engineering tolerances, learned weights, or unknown.
8. A data-efficient learning/calibration plan using 27 labeled families with family-held-out validation.
9. Falsifiable experiments and ablations.
10. A direct adjudication of S1-S12 on 45796000, separating observed facts, inference, and unknowns.

## Hard constraints

- No part ID, expected count, CAD, drawing, filename semantics, or manual truth at inference.
- Do not solve 45796000 by adding a part-specific threshold.
- Do not recommend an end-to-end deep network without a sample-complexity argument and a geometry-preserving fallback.
- Preserve deterministic CPU inference and the 200k-point bound.
- Treat Gold-27 as development evidence, not proof of generalization.
- Treat the post-lock STEP labels as strong engineering adjudication but not independent human ground truth.
- Runtime promotion is prohibited.

## Upload order

1. `01_evidence_book.pdf`
2. `02_machine_context.json`
3. `03_response_schema.json`
4. `evidence/45796000_A_FULL_SCAN.ply`
5. `evidence/45796000_A.stp`
6. `evidence/engineering_drawing.pdf`
7. `evidence/candidate_pool.json`
8. `evidence/v23_locked_prediction.json`
9. `code/run_f1_gold27_unseen_profile.py`
10. `code/f1_bend_evidence_score.py`
11. `code/adjudicate_f1_step_holdout.py`

Return exactly one JSON object valid against `03_response_schema.json`. Do not use Markdown fences and do not add prose outside the JSON.
"""
    (PACKAGE / "00_PASTE_THIS_PROMPT.md").write_text(prompt, encoding="utf-8")

    pdf_path = PACKAGE / "01_evidence_book.pdf"
    _build_pdf(context, pdf_path)

    copies = {
        "evidence/45796000_A_FULL_SCAN.ply": BASE / "source/45796000_A_FULL_SCAN.ply",
        "evidence/45796000_A.stp": BASE / "source/45796000_A.stp",
        "evidence/engineering_drawing.pdf": BASE / "source/שרטוט החלק.pdf",
        "evidence/candidate_pool.json": BASE / "candidate_pool/01_45796000.json",
        "evidence/v23_locked_prediction.json": BASE / "v23_locked_prediction/prediction_owned_regions.json",
        "evidence/v23_inference_summary.json": BASE / "v23_locked_prediction/inference_summary.json",
        "evidence/STEP_bend_labels.json": BASE / "manual_adjudication/cad_drawing_backed_bend_labels.json",
        "evidence/line_evaluation.json": BASE / "manual_adjudication/line_evaluation.json",
        "evidence/final_v23_on_scan.png": BASE / "v23_locked_prediction/final_line_audit.png",
        "evidence/STEP_truth_on_scan.png": BASE / "manual_adjudication/cad_truth_on_scan.png",
        "evidence/routed_12_on_scan.png": BASE / "manual_adjudication/routed_12_on_scan.png",
        "code/run_f1_gold27_unseen_profile.py": ROOT / "scripts/run_f1_gold27_unseen_profile.py",
        "code/f1_bend_evidence_score.py": ROOT / "backend/f1_bend_evidence_score.py",
        "code/adjudicate_f1_step_holdout.py": ROOT / "scripts/adjudicate_f1_step_holdout.py",
    }
    for relative, source in copies.items():
        _copy(source, relative)

    submission = {
        "schema_version": 1,
        "submission_id": context["submission_id"],
        "mode": "direct_multimodal_gpt_submission",
        "paste_file": "00_PASTE_THIS_PROMPT.md",
        "upload_order": [
            "01_evidence_book.pdf",
            "02_machine_context.json",
            "03_response_schema.json",
            "evidence/45796000_A_FULL_SCAN.ply",
            "evidence/45796000_A.stp",
            "evidence/engineering_drawing.pdf",
            "evidence/candidate_pool.json",
            "evidence/v23_locked_prediction.json",
            "code/run_f1_gold27_unseen_profile.py",
            "code/f1_bend_evidence_score.py",
            "code/adjudicate_f1_step_holdout.py",
        ],
        "expected_response_mime_type": "application/json",
        "response_schema": "03_response_schema.json",
        "runtime_promotion_authorized": False,
    }
    _write(PACKAGE / "gpt_submission.json", submission)

    readme = """# Sherman QC F1 structured-inference professor dropoff

This is a direct GPT upload package. Paste `00_PASTE_THIS_PROMPT.md`, then upload files in the order listed by `gpt_submission.json`.

The central evidence is the untouched 45796000 v23 holdout: final 3 TP / 3 FP / 5 FN, while the routed lane contained all 8 true bends plus 4 corner/perimeter aliases. The package requests a formal structured solver rather than another part-specific threshold.

No file in this package authorizes runtime promotion.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")

    files = sorted(path for path in PACKAGE.rglob("*") if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS.txt"})
    manifest = {
        "schema_version": 1,
        "submission_id": context["submission_id"],
        "files": [{"path": str(path.relative_to(PACKAGE)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
    }
    _write(PACKAGE / "MANIFEST.json", manifest)
    checksum_files = sorted(path for path in PACKAGE.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (PACKAGE / "SHA256SUMS.txt").write_text("".join(f"{_sha256(path)}  {path.relative_to(PACKAGE)}\n" for path in checksum_files), encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                archive.write(path, Path(PACKAGE.name) / path.relative_to(PACKAGE))
    print(json.dumps({"package": str(PACKAGE), "zip": str(ZIP_PATH), "file_count": len(list(PACKAGE.rglob('*')))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
