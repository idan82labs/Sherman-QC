"""Jev question set v2 for bend candidates: the owner's labelling conventions.

v1 (``questions_v1``) listed "a joggle, offset step" as a non-bend. The owner's
convention (docs/production/runlog/gold_set_v1/candidates.json ``gold_rules``) is:

- a line counts iff it lies on the outer-skin apex of a real formed fold;
- a joggle is two bends; formed 15-30 deg bends count;
- a window- or relief-interrupted fold of the same two plates is one line;
- hole walls, sheet-thickness / perimeter edges, embosses, gentle rolled kinks
  and artifacts are not bends.

Following the TypeSafe docs (primitives/noul, primitives/advanced, jev-1.13
jaggedness): one narrow yes/no per Noul, instructions and criteria aligned, the
boundary cases written out in the criteria, state fields referenced by
backticked path. ``composite_v2`` adds four atomic Nouls (each one judgment)
whose answers code combines; the primary Noul is the same text in every set.

Sets are hashed with :func:`question_set_sha256` (same rule as v1) so answers
from different sets never mix.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

__all__ = ["PRIMARY_QUESTION", "QUESTION_SETS_V2", "select_questions_v2", "question_set_sha256",
           "OWNER_RULES"]

PRIMARY_QUESTION = "is_real_bend"

OWNER_RULES = (
    "formed press-brake fold at the outer-skin apex; joggle = 2 bends; formed 15-30 deg bends count; "
    "window/relief-interrupted fold of the same two plates = one line; not bends: hole walls, "
    "thickness/perimeter edges, embosses, gentle rolled kinks, artifacts"
)

_HOW_TO_READ = (
    "`state` describes one candidate line found on a 3-D scan of a sheet-metal part. It has no "
    "coordinates. `state.candidate` holds measurements taken on the scan around the line: each one "
    "has a `value`, a `unit` and a `meaning`, and some have a `reading` that puts the value in words. "
    "Cross-section slices are cut across the line at evenly spaced points along it. "
    "`state.surroundings` describes other candidate lines on the same scan."
)

_TRUE = {
    "definition": "Yes: the line runs along a fold that was formed on a press brake, on the outer "
                  "(convex) skin at the apex of the fold.",
    "counts_as_a_bend": [
        "Two flat flanges that meet at a formed fold, at any formed angle.",
        "Small formed bends of about 15 to 30 deg count as bends.",
        "A joggle (offset step) is two bends: each of its two folds is a real bend line.",
        "A fold interrupted by a window or relief cut is still the same bend; a stretch of it counts.",
    ],
}

_FALSE = {
    "definition": "No: the line is not the outer-skin apex of a formed fold.",
    "not_bends": [
        "The wall or rim of a hole, slot or cutout.",
        "A sheet-thickness edge or the outer perimeter of the part (a cut edge, not a fold).",
        "An emboss, bead, rib or dimple pressed into a flat face.",
        "A gentle kink or curvature on a rolled or curved surface, with no distinct fold.",
        "Scan noise, an artifact, a registration seam or fixture geometry.",
        "The inner skin of a fold, or a copy of a fold line offset from it by about one sheet thickness.",
    ],
}

PRIMARY_V2: dict[str, Any] = {
    "type": "noul",
    "instructions": {
        "how_to_read_the_state": _HOW_TO_READ,
        "question": "Is the candidate line in `state.candidate` the outer-skin apex of a real formed "
                    "press-brake bend?",
    },
    "criteria": {"true": _TRUE, "false": _FALSE},
}

_ATOMIC: dict[str, dict[str, Any]] = {
    "has_two_flat_flanges": {
        "type": "noul",
        "instructions": {
            "how_to_read_the_state": _HOW_TO_READ,
            "question": "Do the cross-section measurements in `state.candidate` show two flat flanges "
                        "meeting at this line along most of its length?",
        },
        "criteria": {
            "true": "Most slices show two separate flanges, each one flat (low fit residual), meeting "
                    "at the line at a clear angle.",
            "false": "Most slices show one flange only, no flange, a closed ring of material, or "
                     "flanges that are rough or curved rather than flat.",
        },
    },
    "is_cut_edge_or_hole": {
        "type": "noul",
        "instructions": {
            "how_to_read_the_state": _HOW_TO_READ,
            "question": "Does the evidence in `state` indicate that this line is a cut edge (the part "
                        "perimeter, a sheet-thickness edge, or the wall of a hole or slot) rather than a fold?",
        },
        "criteria": {
            "true": "Material on one side only, a very narrow flange of about one sheet thickness, a "
                    "closed ring of material around the line, or a short line typical of a hole wall.",
            "false": "Material continues on both sides of the line as two flanges of useful width.",
        },
    },
    "is_inner_or_duplicate_skin": {
        "type": "noul",
        "instructions": {
            "how_to_read_the_state": _HOW_TO_READ,
            "question": "Does the evidence in `state` indicate that this line lies on the inner (concave) "
                        "skin of a fold or duplicates a parallel line about one sheet thickness away?",
        },
        "criteria": {
            "true": "The line sits on the concave side of the material at the fold, or a parallel "
                    "candidate lies within a few millimetres and this line looks like its offset copy.",
            "false": "The line sits on the convex outer side of the fold, and no parallel copy of it "
                     "lies about one sheet thickness away.",
        },
    },
    "is_gentle_curvature_or_artifact": {
        "type": "noul",
        "instructions": {
            "how_to_read_the_state": _HOW_TO_READ,
            "question": "Does the evidence in `state` indicate a gentle kink on a curved or rolled "
                        "surface, an emboss, or scan noise, rather than a distinct formed fold?",
        },
        "criteria": {
            "true": "The angle change is very small or unstable along the line, the surfaces are "
                    "rough or curved, or support along the line is sparse.",
            "false": "A distinct fold with a stable angle of about 15 deg or more between two flat "
                     "flanges, supported along the line.",
        },
    },
}

QUESTION_SETS_V2: dict[str, dict[str, dict[str, Any]]] = {
    "v2": {PRIMARY_QUESTION: PRIMARY_V2},
    "composite_v2": {PRIMARY_QUESTION: PRIMARY_V2, **_ATOMIC},
}


def select_questions_v2(variant: str) -> dict[str, dict[str, Any]]:
    if variant not in QUESTION_SETS_V2:
        raise KeyError(f"unknown v2 question set {variant!r}; known {sorted(QUESTION_SETS_V2)}")
    return copy.deepcopy(QUESTION_SETS_V2[variant])


def question_set_sha256(questions: Mapping[str, Any], variant: str) -> str:
    payload = {"variant": variant, "questions": questions}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
