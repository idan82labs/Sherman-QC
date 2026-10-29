"""Frozen Jev question sets for the bend-candidate dataset (C10, v1).

The primary judgment is one Noul, ``is_real_bend``: "Is this candidate a real
press-brake bend line?". Its ``false`` criteria list the documented confusers
(outline, hole wall, rolled band, step, duplicate skin, tongue silhouette,
artifact). An optional Choice, ``candidate_kind``, names the confuser type, and
an optional Score, ``evidence_quality``, rates how much evidence the state holds.

Question ids are for code only (the API does not send them to the model), so
every instruction is self-contained and refers to state fields by backticked
path, as the TypeSafe docs recommend.

Two variants exist and are hashed separately:

- ``v1``: compact instructions and criteria;
- ``verbalized_v1``: the same judgments with longer, example-bearing criteria.

A question set is identified by :func:`question_set_sha256` over the canonical
JSON of the selected questions. Changing any byte changes the hash; evaluation
and operating points record it, so answers from different sets never mix.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "QUESTION_SET_V1",
    "QUESTION_SET_VERBALIZED_V1",
    "QUESTION_SETS",
    "DEFAULT_QUESTIONS",
    "PRIMARY_QUESTION",
    "CONFUSER_KINDS",
    "select_questions",
    "question_set_sha256",
]

PRIMARY_QUESTION = "is_real_bend"
#: By default only the Noul is asked; the Choice/Score are opt-in.
DEFAULT_QUESTIONS = (PRIMARY_QUESTION,)

_STATE_GUIDE = (
    "The `state` describes one candidate line on a sheet-metal part, without coordinates or "
    "identities. Lengths are in mm and angles in degrees. `state.candidate` holds the candidate's "
    "own descriptors, `state.neighbours` summarises other candidates of the same part, "
    "`state.context` holds part-level facts such as sheet thickness and candidate count, and "
    "`state.cad_relation` (when present) relates the candidate to the nearest planned bend of the "
    "part's CAD recipe. `state.candidate_source` says where the candidate came from: "
    "CAD_ONBOARDING (a bend candidate proposed from the folded CAD), E1_FLANGE_GRAPH (a bend found "
    "by CAD flange-graph analysis), SCAN_E3 / SCAN_LEGACY_REPORT / SCAN_F1_POOL (a candidate found "
    "or measured on a 3D scan). A field ending in `_missing` means that value was not available."
)

#: The confuser classes named in the plan (§3.3), plus the positive class and a no-match.
CONFUSER_KINDS: dict[str, str] = {
    "bend": "A real press-brake bend line: two flat flanges joined by a cylindrical bend zone.",
    "outline": "An outer contour or free edge of the part (a cut edge, not a fold).",
    "hole_wall": "The wall or rim of a hole, slot or cutout.",
    "rolled_band": "A rolled or continuously curved region (a rolled shell or large-radius curve), "
    "not a discrete press-brake line.",
    "step": "A joggle, offset step or thickness transition between parallel faces.",
    "duplicate_skin": "A second copy of a bend already represented by another candidate (the "
    "inner and outer skin, or a doubled scan surface).",
    "tongue_silhouette": "The silhouette or edge of a tab, tongue or cutout that looks like a "
    "crease from some viewpoint.",
    "artifact": "Scan or tessellation noise, a registration seam, a fixture or other non-part "
    "geometry.",
    "other": "Something that fits none of the above.",
}

QUESTION_SET_V1: dict[str, dict[str, Any]] = {
    PRIMARY_QUESTION: {
        "type": "noul",
        "instructions": {
            "how_to_read_the_state": _STATE_GUIDE,
            "question": "Is this candidate a real press-brake bend line?",
        },
        "criteria": {
            "true": "The candidate lies on a deliberate press-brake bend: two flat flanges meet at "
            "a bend zone whose inner radius is comparable to the sheet thickness or tooling radius, "
            "and the line runs along that bend.",
            "false": {
                "summary": "The candidate is not a press-brake bend line. Typical confusers:",
                "confusers": [
                    CONFUSER_KINDS["outline"],
                    CONFUSER_KINDS["hole_wall"],
                    CONFUSER_KINDS["rolled_band"],
                    CONFUSER_KINDS["step"],
                    CONFUSER_KINDS["duplicate_skin"],
                    CONFUSER_KINDS["tongue_silhouette"],
                    CONFUSER_KINDS["artifact"],
                ],
            },
        },
    },
    "candidate_kind": {
        "type": "choice",
        "instructions": {
            "how_to_read_the_state": _STATE_GUIDE,
            "question": "What kind of feature is this candidate line?",
        },
        "criteria": dict(CONFUSER_KINDS),
    },
    "evidence_quality": {
        "type": "score",
        "instructions": {
            "how_to_read_the_state": _STATE_GUIDE,
            "question": "How much evidence does the state give for judging whether this candidate "
            "is a real bend line?",
        },
        "criteria": [
            "Almost no usable evidence: most descriptors are missing or contradictory.",
            "Weak evidence: a few descriptors, low support or coverage.",
            "Adequate evidence: the main descriptors are present and mostly consistent.",
            "Strong evidence: complete, consistent descriptors with good support on both sides.",
        ],
    },
}

_VERBALIZED_TRUE = {
    "definition": "Yes: the candidate follows a deliberate press-brake bend line.",
    "signs": [
        "Two flat flanges on either side, at a clear included angle (for example 90 deg), with "
        "support on both sides.",
        "An inner radius of roughly 0.5 to 4 times the sheet thickness, matching tooling.",
        "A straight line that spans most of the bend's length, usually near a planned bend of the "
        "CAD recipe when `state.cad_relation` is present.",
    ],
}
_VERBALIZED_FALSE = {
    "definition": "No: the candidate is something other than a press-brake bend line.",
    "confusers": {
        "outline": CONFUSER_KINDS["outline"] + " Often one-sided support and an included angle near "
        "90 deg against the sheet edge rather than between two flanges.",
        "hole_wall": CONFUSER_KINDS["hole_wall"] + " Usually short, with a tiny radius or none.",
        "rolled_band": CONFUSER_KINDS["rolled_band"] + " Large radius relative to thickness and a "
        "spread of normals instead of two flat sides.",
        "step": CONFUSER_KINDS["step"] + " Parallel faces offset by about the thickness or less.",
        "duplicate_skin": CONFUSER_KINDS["duplicate_skin"] + " Shows as a near-parallel neighbour "
        "within about one thickness (`state.neighbours`).",
        "tongue_silhouette": CONFUSER_KINDS["tongue_silhouette"],
        "artifact": CONFUSER_KINDS["artifact"] + " Low support, poor fit, or implausible geometry.",
    },
}

QUESTION_SET_VERBALIZED_V1: dict[str, dict[str, Any]] = copy.deepcopy(QUESTION_SET_V1)
QUESTION_SET_VERBALIZED_V1[PRIMARY_QUESTION]["criteria"] = {
    "true": _VERBALIZED_TRUE,
    "false": _VERBALIZED_FALSE,
}
QUESTION_SET_VERBALIZED_V1[PRIMARY_QUESTION]["instructions"]["question"] = (
    "Judge the candidate described in `state`. Is it a real press-brake bend line, that is, a line "
    "along which the sheet was deliberately folded on a press brake?"
)

QUESTION_SETS: dict[str, dict[str, dict[str, Any]]] = {
    "v1": QUESTION_SET_V1,
    "verbalized_v1": QUESTION_SET_VERBALIZED_V1,
}


def select_questions(variant: str = "v1", names: Iterable[str] = DEFAULT_QUESTIONS) -> dict[str, dict[str, Any]]:
    """Deep copy of the named questions of one variant. The primary Noul is mandatory."""
    if variant not in QUESTION_SETS:
        raise KeyError(f"unknown question-set variant {variant!r}; known: {sorted(QUESTION_SETS)}")
    chosen = list(dict.fromkeys(names))
    if PRIMARY_QUESTION not in chosen:
        raise ValueError(f"the question set must include {PRIMARY_QUESTION!r}")
    source = QUESTION_SETS[variant]
    unknown = [n for n in chosen if n not in source]
    if unknown:
        raise KeyError(f"unknown questions {unknown}; known: {sorted(source)}")
    return {name: copy.deepcopy(source[name]) for name in chosen}


def question_set_sha256(questions: Mapping[str, Any], variant: str = "v1") -> str:
    """sha256 over canonical JSON of ``{"variant", "questions"}``."""
    payload = {"variant": variant, "questions": questions}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
