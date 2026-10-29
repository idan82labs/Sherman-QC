"""Label sidecar rules for the Jev bend-candidate dataset (C10, v1).

Labels never enter a state. Each exported candidate gets one row in
``labels.jsonl``; the evaluation truth is derived from it by
:func:`evaluation_truth`, never read from a raw column.

Label sources (``label_source``) and what they mean:

======================================  =====  ===========================================
source                                  label  meaning
======================================  =====  ===========================================
operator_confirmed                      1      active, review-confirmed CAD candidate of an
                                               OPERATOR_CONFIRMED recipe
operator_removed_not_a_bend             0      removed with issue ``not_a_bend``
operator_wrong_location                 null   removed with ``wrong_location``: a real bend
                                               with wrong geometry, excluded from truth
operator_removed_issue_missing          null   removed with ``missing`` (ambiguous), excluded
operator_manual_missing                 1      a manual bend the candidates missed: a
                                               recall-accounting row with **no state**
human_marker_real_bend                  1      explicit marker review ``real_bend``
human_marker_not_bend_outline           0      explicit marker review ``not_bend_outline``
human_marker_uncertain                  null   explicit marker review ``uncertain``
gold_line_matched                       1      matches a manual gold bend line
gold_line_unmatched                     0      matches no line of a *complete* gold set
gold_line_unmatched_incomplete          null   matches no line of an incomplete gold set
synthetic_truth_real                    1      E2 synthetic truth (engineering ledger only)
synthetic_truth_not_bend                0      E2 synthetic truth (engineering ledger only)
unreviewed                              null   nobody adjudicated this candidate
dev_auto_confirmed                      null   DEV_AUTO_CONFIRMED recipe: self-referential,
                                               never truth (see weak column)
e1_unmarked_graph_bend                  null   an E1 flange-graph bend nobody marked
======================================  =====  ===========================================

Weak labels live in a **separate** column (``weak_label`` / ``weak_label_source``):

- ``recipe_binding_bound_formed`` (1): bound to a recipe bend, FORMED and measured;
- ``recipe_binding_unbound_far`` (0): unbound and more than 10 mm from every recipe bend;
- ``dev_auto_confirmed_active`` (1): an active bend of a DEV_AUTO_CONFIRMED recipe.

Weak labels are never evaluation truth. The pipeline's observability or bend
state is a state feature, never a label.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from research.jev.questions_v1 import CONFUSER_KINDS

__all__ = [
    "LABEL_SCHEMA",
    "CANDIDATE_LABEL_DOC",
    "TRUTH_SOURCES",
    "NULL_SOURCES",
    "RECALL_SOURCES",
    "WEAK_SOURCES",
    "LABEL_WEIGHTS",
    "MARKER_FRAME",
    "MARKER_DECISIONS",
    "LabelError",
    "LabelRow",
    "make_label",
    "evaluation_truth",
    "marker_label_source",
    "parse_marker_reviews",
    "parse_candidate_label_doc",
    "write_jsonl",
    "read_jsonl",
]

LABEL_SCHEMA = "jev_candidate_label_v1"

#: Sources that are evaluation truth, with their fixed label.
TRUTH_SOURCES: dict[str, int] = {
    "operator_confirmed": 1,
    "operator_removed_not_a_bend": 0,
    "human_marker_real_bend": 1,
    "human_marker_not_bend_outline": 0,
    "gold_line_matched": 1,
    "gold_line_unmatched": 0,
    "synthetic_truth_real": 1,
    "synthetic_truth_not_bend": 0,
}
#: Sources whose label is always null (excluded from truth).
NULL_SOURCES = frozenset(
    {
        "operator_wrong_location",
        "operator_removed_issue_missing",
        "human_marker_uncertain",
        "gold_line_unmatched_incomplete",
        "unreviewed",
        "dev_auto_confirmed",
        "e1_unmarked_graph_bend",
    }
)
#: Recall-accounting rows: a real bend no candidate represents (no state exists).
RECALL_SOURCES: dict[str, int] = {"operator_manual_missing": 1}
#: Weak-label sources (separate column, never truth).
WEAK_SOURCES: dict[str, int] = {
    "recipe_binding_bound_formed": 1,
    "recipe_binding_unbound_far": 0,
    "dev_auto_confirmed_active": 1,
}
#: Per-source sample weight for training-style use (evaluation is unweighted).
LABEL_WEIGHTS: dict[str, float] = {
    "operator_confirmed": 1.0,
    "operator_removed_not_a_bend": 1.0,
    "human_marker_real_bend": 1.0,
    "human_marker_not_bend_outline": 1.0,
    "gold_line_matched": 1.0,
    "gold_line_unmatched": 0.8,
    "synthetic_truth_real": 0.5,
    "synthetic_truth_not_bend": 0.5,
    "operator_manual_missing": 1.0,
}

MARKER_KIND = "sherman_qc_bend_instance_labels"
MARKER_FRAME = "reference_mesh_raw"
MARKER_DECISIONS = {
    "real_bend": "human_marker_real_bend",
    "not_bend_outline": "human_marker_not_bend_outline",
    "uncertain": "human_marker_uncertain",
}


class LabelError(ValueError):
    """A label row or label document is inconsistent."""


@dataclass(frozen=True)
class LabelRow:
    """One ``labels.jsonl`` row. Identity fields here never reach a state."""

    record_id: str
    has_state: bool
    group_sku: str | None
    label: int | None
    label_source: str
    label_weight: float
    weak_label: int | None = None
    weak_label_source: str | None = None
    confuser_kind: str | None = None
    candidate_key: str = ""
    source_sha256: str = ""
    provenance_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Every row is validated, however it was built.
        _validate(self)

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema"] = LABEL_SCHEMA
        row["provenance_flags"] = list(self.provenance_flags)
        return row

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "LabelRow":
        if row.get("schema") != LABEL_SCHEMA:
            raise LabelError(f"label row schema must be {LABEL_SCHEMA}")
        data = {k: row.get(k) for k in cls.__dataclass_fields__}
        data["provenance_flags"] = tuple(row.get("provenance_flags") or ())
        data["candidate_key"] = row.get("candidate_key") or ""
        data["source_sha256"] = row.get("source_sha256") or ""
        return cls(**data)  # type: ignore[arg-type]


def _validate(row: LabelRow) -> None:
    src = row.label_source
    if src in TRUTH_SOURCES:
        if row.label != TRUTH_SOURCES[src]:
            raise LabelError(f"{src} carries label {TRUTH_SOURCES[src]}, got {row.label!r}")
        if not row.has_state:
            raise LabelError(f"{src} is a candidate label and needs a state")
    elif src in NULL_SOURCES:
        if row.label is not None:
            raise LabelError(f"{src} always carries a null label")
    elif src in RECALL_SOURCES:
        if row.label != RECALL_SOURCES[src] or row.has_state:
            raise LabelError(f"{src} is a recall row: label {RECALL_SOURCES[src]} and no state")
    else:
        raise LabelError(f"unknown label_source {src!r}")
    if row.weak_label_source is None:
        if row.weak_label is not None:
            raise LabelError("a weak label needs its weak_label_source")
    else:
        if row.weak_label_source not in WEAK_SOURCES:
            raise LabelError(f"unknown weak_label_source {row.weak_label_source!r}")
        if row.weak_label != WEAK_SOURCES[row.weak_label_source]:
            raise LabelError(f"{row.weak_label_source} carries weak label {WEAK_SOURCES[row.weak_label_source]}")
    if row.label_weight < 0:
        raise LabelError("label_weight must be >= 0")


def make_label(
    record_id: str,
    *,
    label_source: str,
    group_sku: str | None,
    has_state: bool = True,
    weak_label_source: str | None = None,
    confuser_kind: str | None = None,
    candidate_key: str = "",
    source_sha256: str = "",
    provenance_flags: Iterable[str] = (),
) -> LabelRow:
    """Build a validated row; the label value follows from ``label_source``."""
    if label_source in TRUTH_SOURCES:
        label: int | None = TRUTH_SOURCES[label_source]
    elif label_source in RECALL_SOURCES:
        label = RECALL_SOURCES[label_source]
    else:
        label = None
    row = LabelRow(
        record_id=record_id,
        has_state=has_state,
        group_sku=group_sku,
        label=label,
        label_source=label_source,
        label_weight=LABEL_WEIGHTS.get(label_source, 0.0),
        weak_label=None if weak_label_source is None else WEAK_SOURCES.get(weak_label_source),
        weak_label_source=weak_label_source,
        confuser_kind=confuser_kind,
        candidate_key=candidate_key,
        source_sha256=source_sha256,
        provenance_flags=tuple(sorted(set(provenance_flags))),
    )
    return row


def evaluation_truth(row: LabelRow) -> int | None:
    """The only accessor evaluation may use: truth sources with a state, else ``None``."""
    if row.has_state and row.label_source in TRUTH_SOURCES:
        return TRUTH_SOURCES[row.label_source]
    return None


def marker_label_source(decision: str) -> str:
    if decision not in MARKER_DECISIONS:
        raise LabelError(f"unknown marker decision {decision!r}")
    return MARKER_DECISIONS[decision]


def parse_marker_reviews(doc: Any, *, required_frame: str = MARKER_FRAME) -> dict[str, str]:
    """``{bend_id: label_source}`` from a ``sherman_qc_bend_instance_labels`` v1 document.

    The document must be explicit about its frame: anything other than
    ``reference_mesh_raw`` (or the caller's ``required_frame``) is refused, as is
    an unknown kind, schema version or decision. Manual bends drawn in that
    document are *not* candidate labels and are ignored here.
    """
    if not isinstance(doc, Mapping):
        raise LabelError("marker label document must be an object")
    if doc.get("kind") != MARKER_KIND or doc.get("schema_version") != 1:
        raise LabelError(f"marker labels must be {MARKER_KIND} schema_version 1")
    if doc.get("coordinate_frame") != required_frame:
        raise LabelError(f"marker labels must be in frame {required_frame!r}")
    reviews = doc.get("marker_reviews")
    if not isinstance(reviews, list):
        raise LabelError("marker_reviews must be an array")
    out: dict[str, str] = {}
    for i, review in enumerate(reviews):
        if not isinstance(review, Mapping) or not isinstance(review.get("bend_id"), str):
            raise LabelError(f"marker_reviews[{i}] needs a bend_id")
        bend_id = review["bend_id"]
        if bend_id in out:
            raise LabelError(f"duplicate marker review for {bend_id!r}")
        out[bend_id] = marker_label_source(str(review.get("decision")))
    return out


CANDIDATE_LABEL_DOC = "jev_candidate_labels_v1"
_DOC_SOURCES = {
    "human_marker_review": MARKER_DECISIONS,
    "synthetic_truth": {"real_bend": "synthetic_truth_real", "not_bend_outline": "synthetic_truth_not_bend",
                        "uncertain": "unreviewed"},
}


def parse_candidate_label_doc(doc: Any, *, required_frame: str) -> tuple[dict[str, str], dict[str, str]]:
    """``({candidate_key: label_source}, {candidate_key: confuser_kind})`` from a C10 label doc.

    Format::

        {"schema": "jev_candidate_labels_v1", "source": "human_marker_review"|"synthetic_truth",
         "coordinate_frame": "<frame of the candidates>",
         "labels": {"<candidate key>": "real_bend"|"not_bend_outline"|"uncertain"},
         "confuser_kinds": {"<candidate key>": "<CONFUSER_KINDS key>"}}   # optional
    """
    if not isinstance(doc, Mapping) or doc.get("schema") != CANDIDATE_LABEL_DOC:
        raise LabelError(f"label document schema must be {CANDIDATE_LABEL_DOC}")
    source = doc.get("source")
    if source not in _DOC_SOURCES:
        raise LabelError(f"label document source must be one of {sorted(_DOC_SOURCES)}")
    if doc.get("coordinate_frame") != required_frame:
        raise LabelError(f"label document frame must be {required_frame!r}")
    labels = doc.get("labels")
    if not isinstance(labels, Mapping):
        raise LabelError("labels must be an object")
    mapping = _DOC_SOURCES[source]
    out: dict[str, str] = {}
    for key, decision in labels.items():
        if decision not in mapping:
            raise LabelError(f"unknown decision {decision!r} for {key!r}")
        out[str(key)] = mapping[decision]
    kinds_raw = doc.get("confuser_kinds") or {}
    if not isinstance(kinds_raw, Mapping):
        raise LabelError("confuser_kinds must be an object")
    kinds: dict[str, str] = {}
    for key, kind in kinds_raw.items():
        if kind not in CONFUSER_KINDS:
            raise LabelError(f"unknown confuser kind {kind!r}")
        kinds[str(key)] = str(kind)
    return out, kinds


# --------------------------------------------------------------------------
# JSONL io
# --------------------------------------------------------------------------


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows as canonical JSON lines (tmp + rename). Returns the row count."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(_dumps(row))
            fh.write("\n")
            count += 1
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise LabelError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
