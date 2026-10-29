"""Bilingual (English / Hebrew) report vocabulary.

Decision, bend-state and summary labels and colours come from
``contracts.production.vocabulary.DISPLAY_TOKENS`` (the station UI mirrors the
same tokens). This module adds the report-only strings: characteristic kinds,
UNKNOWN / NOT_APPLICABLE reason explanations, stage and validity states,
column headers and fixed notes.

Every entry is ``(english, hebrew)``. Hebrew strings are stored in logical
order; the PDF wraps them in ``dir="rtl"`` and MuPDF's HTML layout applies the
Unicode bidi algorithm and HarfBuzz shaping, so they are never pre-reversed.

The Hebrew wording has not been reviewed by a native speaker (same caveat as
the contract vocabulary).
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from contracts.production.characteristics import NOT_APPLICABLE_REASONS, UNKNOWN_REASONS
from contracts.production.states import STAGE_STATUSES, VALIDITY_STATES
from contracts.production.vocabulary import DISPLAY_TOKENS

__all__ = [
    "Bi",
    "bend_display_label",
    "KIND_LABELS",
    "REASON_LABELS",
    "STAGE_STATUS_LABELS",
    "VALIDITY_LABELS",
    "INSPECTION_STAGE_LABELS",
    "BOUNDARY_LABELS",
    "TEXT",
    "token",
    "kind_label",
    "reason_label",
]

Bi = tuple[str, str]


def _ro(d: dict[str, Bi]) -> Mapping[str, Bi]:
    return MappingProxyType(d)


KIND_LABELS: Mapping[str, Bi] = _ro({
    "FLANGE_LENGTH/A": ("Flange length A", "אורך כנף A"),
    "FLANGE_LENGTH/B": ("Flange length B", "אורך כנף B"),
    "INCLUDED_ANGLE": ("Included angle", "זווית כלואה"),
    "BEND_LINE_POSITION": ("Bend-line position", "מיקום קו הכיפוף"),
    "BACKGAUGE_LENGTH": ("Backgauge length", "אורך מעצור אחורי"),
    "GENERAL_MEASUREMENT": ("General measurement", "מידה כללית"),
})

REASON_LABELS: Mapping[str, Bi] = _ro({
    # UNKNOWN reasons (plan §3.2)
    "OCCLUDED": ("Surface hidden from the scanner (occluded).", "המשטח מוסתר מהסורק."),
    "EDGE_NOT_SEEN": ("The far edge of the flange was not captured.", "הקצה הרחוק של הכנף לא נסרק."),
    "LOW_DENSITY": ("Too few scan points for a reliable fit.", "מעט מדי נקודות סריקה להתאמה אמינה."),
    "AMBIGUOUS_SEGMENTATION": ("The flange surfaces could not be separated reliably.", "לא ניתן להפריד את משטחי הכנפיים באופן אמין."),
    "REGISTRATION_UNTRUSTED": ("Scan-to-CAD alignment is not trusted for this value.", "יישור הסריקה לשרטוט אינו אמין עבור ערך זה."),
    "AMBIGUOUS_POSE": ("Part symmetry: the bend identity cannot be assigned with confidence.", "סימטריה של החלק: לא ניתן לזהות את הכיפוף בוודאות."),
    "TIMEOUT": ("The time budget ran out before this value was measured.", "זמן העיבוד הסתיים לפני שהערך נמדד."),
    "UNSUPPORTED_FEATURE": ("Feature type is not supported by the engine (e.g. hem, rolled).", "סוג המאפיין אינו נתמך (למשל קיפול כפול, גלגול)."),
    # contract additions
    "INVALID_INSPECTION": ("The inspection is invalid; nothing was measured.", "הבדיקה אינה תקפה; לא נמדד דבר."),
    "BEND_STATE_UNKNOWN": ("The bend state could not be decided.", "מצב הכיפוף לא הוכרע."),
    "ADJACENT_BEND_NOT_MEASURED": ("The neighbouring bend was not measured, so the length is undefined.", "הכיפוף השכן לא נמדד ולכן האורך אינו מוגדר."),
    "NOMINAL_UNAVAILABLE": ("No nominal value is available.", "אין ערך נומינלי."),
    "NO_TOLERANCE": ("No tolerance is defined.", "לא הוגדר טולרנס."),
    # NOT_APPLICABLE reasons
    "BEND_NOT_YET_FORMED": ("The bend is not yet formed at this stage; this is not a defect.", "הכיפוף טרם בוצע בשלב זה; אין זה פגם."),
    "NOT_INSPECTED_BY_RECIPE": ("Not inspected according to the recipe.", "לא נבדק לפי המתכון."),
    "NOT_APPLICABLE_AT_STAGE": ("Not applicable at this stage of the sequence.", "לא רלוונטי בשלב זה של רצף הכיפופים."),
})

STAGE_STATUS_LABELS: Mapping[str, Bi] = _ro({
    "CONSISTENT": ("Consistent with the bend order", "תואם לסדר הכיפופים"),
    "STAGE_MISMATCH": ("Declared stage differs from the decoded stage", "השלב המוצהר שונה מהשלב שזוהה"),
    "OUT_OF_ORDER_SUSPECTED": ("Bends formed out of order (suspected)", "חשד לכיפוף שלא לפי הסדר"),
    "PARTIAL_BEND_FOUND": ("Partially formed bend found", "נמצא כיפוף חלקי"),
    "FINAL_INCOMPLETE": ("Final inspection, but some bends are not formed", "בדיקה סופית אך חלק מהכיפופים לא בוצעו"),
    "UNDETERMINED": ("Stage undetermined (some bend state unknown)", "השלב לא הוכרע (מצב כיפוף לא ידוע)"),
    "AMBIGUOUS_POSE": ("Pose ambiguous (part symmetry)", "מנח לא חד-משמעי (סימטריה)"),
    "NOT_EVALUATED": ("Not evaluated (invalid inspection)", "לא הוערך (בדיקה לא תקפה)"),
})

VALIDITY_LABELS: Mapping[str, Bi] = _ro({
    "VALID": ("Valid", "תקפה"),
    "INSUFFICIENT_CAPTURE": ("Insufficient capture", "כיסוי סריקה לא מספיק"),
    "UNTRUSTED_REGISTRATION": ("Untrusted registration", "יישור לא אמין"),
    "UNIT_MISMATCH_SUSPECTED": ("Unit mismatch suspected", "חשד לאי-התאמת יחידות"),
    "UNSUPPORTED_INPUT": ("Unsupported input", "קלט לא נתמך"),
    "WRONG_PART_SUSPECTED": ("Wrong part suspected", "חשד לחלק שגוי"),
    "MIRRORED_PART_SUSPECTED": ("Mirrored part suspected", "חשד לחלק בתמונת ראי"),
    "TIMEOUT": ("Timed out", "חריגת זמן"),
})

INSPECTION_STAGE_LABELS: Mapping[str, Bi] = _ro({
    "FINAL": ("Final (finished part)", "סופית (חלק מוגמר)"),
    "IN_PROCESS": ("In process (mid-sequence)", "בתהליך (באמצע רצף הכיפופים)"),
})

BOUNDARY_LABELS: Mapping[str, Bi] = _ro({
    "FREE_EDGE": ("OML to free edge", "מקו OML לקצה חופשי"),
    "ADJACENT_BEND": ("OML to OML of", "מקו OML לקו OML של"),
})

TEXT: Mapping[str, Bi] = _ro({
    "title": ("Inspection report", "דוח בדיקה"),
    "subtitle": ("Press-brake sheet-metal part, scan vs. released recipe", "חלק פח בכיפוף, סריקה מול מתכון משוחרר"),
    "authority": (
        "Operator assist — not an automatic accept/reject decision. Final disposition is made by the operator.",
        "סיוע למפעיל — אינו החלטת קבלה/דחייה אוטומטית. ההחלטה הסופית בידי המפעיל.",
    ),
    "authority_short": ("Operator assist — not an automatic accept/reject decision", "סיוע למפעיל — אינו החלטת קבלה/דחייה אוטומטית"),
    "identity": ("Part and scan", "חלק וסריקה"),
    "verdict": ("Verdict summary", "סיכום תוצאה"),
    "views": ("3D views", "מבטים תלת-ממדיים"),
    "bends": ("Bends in sequence order (flange first)", "כיפופים לפי סדר הביצוע (כנפיים תחילה)"),
    "stage": ("Stage decision and backgauge", "החלטת שלב ומעצור אחורי"),
    "general": ("General measurements", "מידות כלליות"),
    "creases": ("Unplanned creases", "קפלים לא מתוכננים"),
    "validity": ("Validity and registration", "תקפות ויישור"),
    "warnings": ("Warnings", "אזהרות"),
    "legend": ("Undetermined and not-applicable reasons", "סיבות לערכים שלא הוכרעו או לא רלוונטיים"),
    "decisions": ("Decision colours", "צבעי החלטה"),
    "conventions": ("Conventions", "מוסכמות"),
    "sku": ("SKU", "מק\"ט"),
    "part_name": ("Part name", "שם החלק"),
    "revision": ("Revision", "גרסת שרטוט"),
    "recipe": ("Recipe", "מתכון"),
    "compiled": ("Compiled part", "חלק מהודר"),
    "scan_file": ("Scan file", "קובץ סריקה"),
    "serial": ("Serial", "מספר סידורי"),
    "station": ("Station", "עמדה"),
    "scan_time": ("Scan time", "זמן סריקה"),
    "received": ("Received", "התקבל"),
    "inspected": ("Inspected", "נבדק"),
    "engine": ("Engine", "מנוע"),
    "bundle": ("Engine bundle", "חבילת מנוע"),
    "job": ("Job", "עבודה"),
    "material": ("Material / thickness", "חומר / עובי"),
    "tol_standard": ("Tolerance standard", "תקן טולרנס"),
    "inspection_stage": ("Inspection stage", "סוג בדיקה"),
    "seq": ("Seq", "סדר"),
    "bend": ("Bend", "כיפוף"),
    "move": ("Move", "מהלך"),
    "state": ("State", "מצב"),
    "characteristic": ("Characteristic", "מאפיין"),
    "expected": ("Expected", "נומינלי"),
    "actual": ("Actual", "בפועל"),
    "dev": ("Dev.", "סטייה"),
    "u": ("U (±)", "אי-ודאות"),
    "tol": ("Tolerance", "טולרנס"),
    "decision": ("Decision", "החלטה"),
    "reason": ("Reason / notes", "סיבה / הערות"),
    "required": ("Req.", "חובה"),
    "counts": ("Required characteristics", "מאפייני חובה"),
    "bends_count": ("Bends", "כיפופים"),
    "worst": ("Worst characteristic", "המאפיין החריג ביותר"),
    "stage_decision": ("Stage decision", "החלטת שלב"),
    "validity_state": ("Inspection validity", "תקפות הבדיקה"),
    "no_views": ("3D views unavailable", "מבטים תלת-ממדיים אינם זמינים"),
    "no_heatmap": ("Heatmap unavailable", "מפת סטיות אינה זמינה"),
    "move_group": ("Move", "מהלך"),
    "single": ("single bend", "כיפוף בודד"),
    "backgauge": ("Backgauge lengths", "אורכי מעצור אחורי"),
    "no_backgauge": ("No backgauge characteristic at this stage.", "אין מאפיין מעצור אחורי בשלב זה."),
    "backgauge_none_measured": (
        "Not measured for any bend: the inspection is not valid, so no backgauge length could be checked.",
        "לא נמדד באף כיפוף: הבדיקה אינה תקפה, ולכן לא נבדק אף אורך מעצור אחורי.",
    ),
    "none": ("None", "אין"),
    "page": ("Page", "עמוד"),
    "not_yet_formed_note": (
        "Bends not yet formed are shown greyed. They are never a defect.",
        "כיפופים שטרם בוצעו מוצגים באפור. הם לעולם אינם פגם.",
    ),
    # ---- formal report (v2) ----
    "company": ("I. Sherman Industries", "י. שרמן תעשיות"),
    "doc_title": ("Dimensional inspection report", "דוח בדיקת מידות"),
    "doc_subtitle": ("Press-brake sheet-metal part · 3D scan against the released recipe",
                     "חלק פח בכיפוף · סריקה תלת-ממדית מול המתכון המשוחרר"),
    "doc_no": ("Document no.", "מספר מסמך"),
    "part_no": ("Part number", "מק\"ט"),
    "operator": ("Operator", "מפעיל"),
    "station_scanner": ("Station / scanner", "עמדה / סורק"),
    "scan_date": ("Scan time", "זמן סריקה"),
    "summary_title": ("Inspection summary", "סיכום הבדיקה"),
    "result_label": ("Result", "תוצאה"),
    "stage_progress": ("Stage", "שלב"),
    "bends_formed_of": ("bends formed", "כיפופים בוצעו"),
    "req_chars_of": ("required characteristics", "מאפייני חובה"),
    "count_pass": ("Pass", "תקין"),
    "count_review": ("Review", "לבדיקה"),
    "count_fail": ("Fail", "חריגה"),
    "count_unknown": ("Undetermined", "לא הוכרע"),
    "overview": ("Part overview (bends numbered in shop order)", "מבט על החלק (הכיפופים ממוספרים לפי סדר העבודה)"),
    "overview_none": ("Overview image unavailable", "תמונת מבט אינה זמינה"),
    "signoff": ("Inspector review and disposition", "בדיקת מבקר והחלטה"),
    "disposition": ("Disposition", "החלטה"),
    "accept": ("Accept", "קבלה"),
    "rework": ("Rework", "תיקון"),
    "reject": ("Reject", "דחייה"),
    "name": ("Name", "שם"),
    "signature": ("Signature", "חתימה"),
    "date": ("Date", "תאריך"),
    "sec_chars": ("Characteristic results", "תוצאות מאפיינים"),
    "sec_chars_note": ("Bends in shop order, grouped by machine move (מהלך); per bend the flange lengths first, then the angle.",
                       "כיפופים לפי סדר העבודה, מקובצים לפי מהלך; בכל כיפוף אורכי הכנפיים תחילה ואחריהם הזווית."),
    "sec_nyf": ("Bends not yet formed at this stage", "כיפופים שטרם בוצעו בשלב זה"),
    "sec_nyf_note": ("Not evaluated at this stage. A bend that is not yet formed is never a defect.",
                     "לא נבדקו בשלב זה. כיפוף שטרם בוצע לעולם אינו פגם."),
    "sec_measurements": ("Measurements", "מידות"),
    # zero-shot overlay (sherman_zeroshot_overlay_v1): detections, never measurements
    "sec_zs": ("Detections (unverified)", "זיהויים (לא מאומתים)"),
    "sec_zs_note": ("Bend lines found in the scan by the AI detection lane, without the CAD measurement. They are not "
                    "measurements and never a pass; they only point at places to check.",
                    "קווי כיפוף שזוהו בסריקה על ידי מסלול הזיהוי האוטומטי, ללא מדידה מול השרטוט. אלה אינן מדידות "
                    "ולעולם אינן אישור; הן רק מצביעות על מקומות לבדיקה."),
    "sec_zs_refused": ("Detected bends, unverified: the scan did not match the part, so nothing here is checked against the CAD.",
                       "כיפופים שזוהו, לא מאומתים: הסריקה לא הותאמה לחלק, ולכן דבר כאן לא נבדק מול השרטוט."),
    "zs_lowered": ("Detection review: an unexpected bend was found; the result is lowered to CHECK.",
                   "בדיקת זיהוי: נמצא כיפוף לא צפוי; התוצאה הורדה ל'לבדוק'."),
    "zs_kind_WITHHELD_BEND": ("Position not measured, detected line", "מיקום לא נמדד, קו שזוהה"),
    "zs_kind_UNEXPECTED_BEND": ("Unexpected bend (not in the recipe)", "כיפוף לא צפוי (לא במתכון)"),
    "zs_kind_REFUSED_SCAN_DETECTION": ("Detected bend, unverified", "כיפוף שזוהה, לא מאומת"),
    "zs_decision_REVIEW": ("Check", "לבדוק"),
    "zs_decision_UNVERIFIED": ("Unverified", "לא מאומת"),
    "zs_length": ("Length", "אורך"),
    "zs_score": ("Score", "ציון"),
    "zs_kind": ("Detection", "זיהוי"),
    "zs_flag_UNEXPECTED_FEATURE_REVIEW": ("Unexpected bend: check the part", "כיפוף לא צפוי: יש לבדוק את החלק"),
    "zs_angle_joggle": ("joggle, angle not measured", "קיזוז (ג'וגל), הזווית לא נמדדה"),
    "zs_flag_INDEPENDENT_ANGLE_DISAGREES": ("Independent angle disagrees with the measured FAIL: check the angle",
                                           "הזווית העצמאית אינה תואמת לחריגה שנמדדה: יש לבדוק את הזווית"),
    "sec_appendix": ("Appendix", "נספח"),
    "app_views": ("Deviation heatmap and views", "מפת סטיות ומבטים"),
    "app_method": ("Method, uncertainty and traceability", "שיטה, אי-ודאות ועקיבות"),
    "app_evidence": ("Measurement evidence per characteristic", "ראיות מדידה לכל מאפיין"),
    "notes_title": ("Notes", "הערות"),
    "ref_only": ("Reference characteristic (not required)", "מאפיין לייחוס (לא חובה)"),
    "no": ("No.", "מס'"),
    "nominal": ("Nominal", "נומינלי"),
    "result_col": ("Result", "תוצאה"),
    "notes": ("Notes", "הערות"),
    "measurement": ("Measurement", "מידה"),
    "boundary": ("Boundary", "גבול"),
    "stations_col": ("Stations (n · min / max)", "תחנות (n · מינ' / מקס')"),
    "method": ("Method · points", "שיטה · נקודות"),
    "generated_by": ("Generated by the I. Sherman bend inspection system from the stored inspection result", "הופק במערכת בקרת הכיפופים של שרמן מתוצאת הבדיקה השמורה"),
    "confidential": ("Confidential", "סודי"),
    "of": ("of", "מתוך"),
    "id_hashes": ("Identifiers and hashes", "מזהים וגיבובים"),
    "id_col": ("ID", "מזהה"),
    "angle_col": ("Included angle", "זווית כלואה"),
})


def token(term: str) -> Mapping[str, str]:
    """Display token (``tone``, ``color``, ``en``, ``he``) of a decision / state / summary."""
    return DISPLAY_TOKENS[term]


def kind_label(kind: str, side: str | None = None) -> Bi:
    key = f"{kind}/{side}" if kind == "FLANGE_LENGTH" and side else kind
    return KIND_LABELS.get(key, (kind, kind))


def reason_label(reason: str) -> Bi:
    return REASON_LABELS.get(reason, (reason, reason))


# Every contract reason / state must have a bilingual label (checked at import).
_missing = (
    (set(UNKNOWN_REASONS) | set(NOT_APPLICABLE_REASONS)) - set(REASON_LABELS)
    | set(STAGE_STATUSES) - set(STAGE_STATUS_LABELS)
    | set(VALIDITY_STATES) - set(VALIDITY_LABELS)
)
if _missing:  # pragma: no cover - guards vocabulary drift
    raise RuntimeError(f"report labels missing for: {sorted(_missing)}")


def bend_display_label(result: object, row: object) -> str:
    """The bend's name on screen and in the report: the studio (CAD) label the
    runner put in ``result.context.bend_display_labels``, else ``row.label``."""
    labels = getattr(getattr(result, "context", None), "bend_display_labels", None) or {}
    return labels.get(getattr(row, "bend_uid", ""), None) or getattr(row, "label")
