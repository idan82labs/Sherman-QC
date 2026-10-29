"""Offline evaluation of Jev bend-candidate answers (C10, v1).

Inputs: ``answers.jsonl`` (from :func:`.client_v1.live_run`, optional),
``labels.jsonl`` and ``states.jsonl`` from an export, and the split manifest.

Protocol (frozen before any eval answer is read):

1. Truth is :func:`.labels_v1.evaluation_truth` only: weak labels, null labels
   and recall rows never enter it. Recall rows (real bends no candidate
   represents) are added to the recall denominator as misses ("end-to-end
   recall").
2. The operating point is frozen on ``dev`` + ``calib`` and written to
   ``operating_point.json`` with its sha256, or read from a given file and
   never refitted:
   - ``prune_below``: the highest threshold whose dev+calib recall is at least
     :data:`TARGET_RECALL` (0.99). A candidate below it would be pruned;
   - ``accept_at``: the lowest threshold at or above ``prune_below`` whose
     dev+calib precision is at least :data:`TARGET_PRECISION` (0.99), or null;
   - between the two is the abstention band, which maps to ``review``.
3. Metrics per split and scorer: AUC, Brier, log-loss, ECE (10 equal-width
   bins), precision and recall at ``prune_below``, the pruning rate (share of
   candidates below it, and share of negatives below it), the decision mix,
   group-bootstrap 95 % CIs over SKU families, and the worst-family recall.
4. Baselines on the identical records and split: the engine confidence where a
   state carries ``engine_confidence_score``, and a scikit-learn logistic
   regression on the numeric state fields fitted on ``dev`` only.
5. Answers must belong to the current states and question set. Every
   ``answers.jsonl`` row carries its request identity
   (:func:`.client_v1.request_identity`); the evaluator rebuilds each request body
   from the current state and refuses (:class:`EvaluationError`) a row whose
   ``body_sha256`` or ``question_set_sha256`` differs, a row without identity, an
   answer for a record that is not in the states, a duplicate record, and a file
   that mixes request models, variants, question sets or ablations. The identity is
   recorded in ``evaluation.json`` and ``operating_point.json``.
6. A given operating point is refused when its sha256 (over the whole document)
   does not match, or when its split manifest, answer identity or baseline options
   differ from the current run.
7. Flags: ``exploratory_not_unseen`` whenever a consumed or retrospective
   family or record is involved; ``in_sample_only`` when no eval record exists;
   ``synthetic_only`` when every truth label is E2 synthetic truth (engineering
   ledger, never accuracy evidence).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.jev import HARNESS_VERSION
from research.jev.client_v1 import request_identity
from research.jev.labels_v1 import LabelRow, RECALL_SOURCES, evaluation_truth, read_jsonl
from research.jev.questions_v1 import PRIMARY_QUESTION
from research.jev.splits_v1 import EXPLORATORY_FLAGS, SplitManifest, load_split_manifest

__all__ = [
    "TARGET_RECALL",
    "TARGET_PRECISION",
    "ECE_BINS",
    "BOOTSTRAP_ROUNDS",
    "EvaluationError",
    "auc",
    "brier",
    "log_loss",
    "ece",
    "recall_preserving_threshold",
    "precision_threshold",
    "decide",
    "freeze_operating_point",
    "evaluate",
]

TARGET_RECALL = 0.99
TARGET_PRECISION = 0.99
ECE_BINS = 10
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 20261017
LOG_EPS = 1e-6
SPLITS_SCORED = ("dev", "calib", "eval")
OPERATING_POINT_SCHEMA = "jev_operating_point_v1"


class EvaluationError(ValueError):
    pass


# --------------------------------------------------------------------------
# Metrics (pure)
# --------------------------------------------------------------------------


def auc(y: Sequence[int], p: Sequence[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(list(y), list(p)))


def brier(y: Sequence[int], p: Sequence[float]) -> float | None:
    if not y:
        return None
    return sum((pi - yi) ** 2 for yi, pi in zip(y, p)) / len(y)


def log_loss(y: Sequence[int], p: Sequence[float]) -> float | None:
    if not y:
        return None
    total = 0.0
    for yi, pi in zip(y, p):
        q = min(1.0 - LOG_EPS, max(LOG_EPS, pi))
        total -= math.log(q) if yi == 1 else math.log(1.0 - q)
    return total / len(y)


def ece(y: Sequence[int], p: Sequence[float], bins: int = ECE_BINS) -> float | None:
    """Expected calibration error with ``bins`` equal-width bins on [0, 1]."""
    if not y:
        return None
    sums = [[0.0, 0.0, 0] for _ in range(bins)]
    for yi, pi in zip(y, p):
        b = min(bins - 1, max(0, int(pi * bins)))
        sums[b][0] += pi
        sums[b][1] += yi
        sums[b][2] += 1
    n = len(y)
    return sum(abs(s[0] / s[2] - s[1] / s[2]) * s[2] / n for s in sums if s[2])


def recall_preserving_threshold(y: Sequence[int], p: Sequence[float], target: float = TARGET_RECALL) -> float | None:
    """Highest ``t`` with ``recall(p >= t) >= target``; ``None`` without positives."""
    pos = sorted(pi for yi, pi in zip(y, p) if yi == 1)
    if not pos:
        return None
    allowed_misses = math.floor((1.0 - target) * len(pos) + 1e-9)
    return pos[allowed_misses]


def precision_threshold(y: Sequence[int], p: Sequence[float], *, floor: float,
                        target: float = TARGET_PRECISION) -> float | None:
    """Lowest ``t >= floor`` with ``precision(p >= t) >= target`` (and at least one positive)."""
    for t in sorted({pi for pi in p if pi >= floor}):
        chosen = [yi for yi, pi in zip(y, p) if pi >= t]
        if chosen and sum(chosen) >= 1 and sum(chosen) / len(chosen) >= target:
            return t
    return None


def decide(p: float, op: Mapping[str, Any]) -> str:
    """``prune`` | ``review`` | ``accept`` for one score under a frozen operating point."""
    if op["prune_below"] is not None and p < op["prune_below"]:
        return "prune"
    if op["accept_at"] is not None and p >= op["accept_at"]:
        return "accept"
    return "review"


def _pr_at(y: Sequence[int], p: Sequence[float], t: float | None, extra_misses: int = 0) -> dict[str, Any]:
    if t is None:
        return {"precision": None, "recall": None, "recall_end_to_end": None}
    tp = sum(1 for yi, pi in zip(y, p) if yi == 1 and pi >= t)
    pp = sum(1 for pi in p if pi >= t)
    pos = sum(y)
    return {
        "precision": tp / pp if pp else None,
        "recall": tp / pos if pos else None,
        "recall_end_to_end": tp / (pos + extra_misses) if (pos + extra_misses) else None,
    }


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    record_id: str
    truth: int
    split: str
    family: str
    exploratory: bool
    synthetic: bool


def _flatten_numeric(obj: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten_numeric(value, name + "."))
        elif isinstance(value, bool):
            out[name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and math.isfinite(value):
            out[name] = float(value)
    return out


def _engine_confidence(state: Mapping[str, Any]) -> float | None:
    for block in ("cad_relation", "candidate"):
        value = (state.get(block) or {}).get("engine_confidence_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return min(1.0, max(0.0, float(value)))
    return None


def freeze_operating_point(y: Sequence[int], p: Sequence[float], *, scorer: str,
                           target_recall: float = TARGET_RECALL,
                           target_precision: float = TARGET_PRECISION) -> dict[str, Any]:
    prune = recall_preserving_threshold(y, p, target_recall)
    accept = None if prune is None else precision_threshold(y, p, floor=prune, target=target_precision)
    return {"scorer": scorer, "prune_below": prune, "accept_at": accept, "fit_on": ["dev", "calib"],
            "n_fit": len(y), "n_fit_positive": int(sum(y)),
            "target_recall": target_recall, "target_precision": target_precision}


def _bootstrap(rows: Sequence[_Row], scores: Mapping[str, float], op: Mapping[str, Any],
               recall_misses: Mapping[str, int], rounds: int) -> dict[str, Any]:
    families = sorted({r.family for r in rows})
    if len(families) < 2 or op["prune_below"] is None:
        return {"rounds": 0, "note": "needs >= 2 families and a prune threshold"}
    by_family: dict[str, list[_Row]] = {f: [] for f in families}
    for r in rows:
        by_family[r.family].append(r)
    rng = random.Random(BOOTSTRAP_SEED)
    prune_rates, recalls = [], []
    t = op["prune_below"]
    for _ in range(rounds):
        sample = [rng.choice(families) for _ in families]
        rs = [r for f in sample for r in by_family[f]]
        misses = sum(recall_misses.get(f, 0) for f in sample)
        if not rs:
            continue
        prune_rates.append(sum(1 for r in rs if scores[r.record_id] < t) / len(rs))
        pos = sum(r.truth for r in rs) + misses
        if pos:
            recalls.append(sum(1 for r in rs if r.truth == 1 and scores[r.record_id] >= t) / pos)

    def _ci(values: list[float]) -> list[float] | None:
        if not values:
            return None
        v = sorted(values)
        return [v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))]]

    return {"rounds": rounds, "group": "sku_family", "pruning_rate_ci95": _ci(prune_rates),
            "recall_end_to_end_ci95": _ci(recalls)}


def _metrics(rows: Sequence[_Row], scores: Mapping[str, float], op: Mapping[str, Any],
             recall_misses: Mapping[str, int], rounds: int) -> dict[str, Any]:
    rows = [r for r in rows if r.record_id in scores]
    y = [r.truth for r in rows]
    p = [scores[r.record_id] for r in rows]
    misses = sum(recall_misses.get(f, 0) for f in {r.family for r in rows})
    out: dict[str, Any] = {"n": len(rows), "n_positive": int(sum(y)), "recall_rows": misses,
                           "families": len({r.family for r in rows})}
    if not rows:
        return out
    out.update({"auc": auc(y, p), "brier": brier(y, p), "log_loss": log_loss(y, p), "ece_10": ece(y, p)})
    t = op["prune_below"]
    out["at_prune_below"] = _pr_at(y, p, t, misses)
    decisions = {"prune": 0, "review": 0, "accept": 0}
    pruned_pos = pruned_neg = 0
    for yi, pi in zip(y, p):
        d = decide(pi, op)
        decisions[d] += 1
        if d == "prune":
            pruned_pos += yi
            pruned_neg += 1 - yi
    negatives = len(y) - sum(y)
    out["decisions"] = decisions
    out["pruning_rate"] = decisions["prune"] / len(y)
    out["negatives_pruned_fraction"] = pruned_neg / negatives if negatives else None
    out["positives_pruned"] = pruned_pos
    worst = None
    for family in sorted({r.family for r in rows}):
        fy = [r.truth for r in rows if r.family == family]
        fp = [scores[r.record_id] for r in rows if r.family == family]
        rec = _pr_at(fy, fp, t, recall_misses.get(family, 0))["recall_end_to_end"]
        if rec is not None and (worst is None or rec < worst["recall_end_to_end"]):
            worst = {"family": family, "recall_end_to_end": rec}
    out["worst_family_recall"] = worst
    out["bootstrap"] = _bootstrap(rows, scores, op, recall_misses, rounds)
    return out


def _logreg_scores(rows: Sequence[_Row], states: Mapping[str, Mapping[str, Any]], *,
                   cad_free: bool) -> tuple[dict[str, float], dict[str, Any]]:
    dev = [r for r in rows if r.split == "dev"]
    if len({r.truth for r in dev}) < 2:
        return {}, {"status": "NOT_EVALUATED", "reason": "dev split needs both classes"}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _features(rid: str) -> dict[str, float]:
        state = dict(states[rid])
        if cad_free:
            state.pop("cad_relation", None)
        return _flatten_numeric(state)

    keys = sorted({k for r in dev for k in _features(r.record_id)})
    if not keys:
        return {}, {"status": "NOT_EVALUATED", "reason": "no numeric state fields"}

    def _vector(rid: str) -> list[float]:
        f = _features(rid)
        return [f.get(k, 0.0) for k in keys]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    model.fit([_vector(r.record_id) for r in dev], [r.truth for r in dev])
    probs = model.predict_proba([_vector(r.record_id) for r in rows])[:, 1]
    return ({r.record_id: float(pv) for r, pv in zip(rows, probs)},
            {"status": "FITTED", "fit_on": "dev", "n_fit": len(dev), "features": len(keys), "cad_free": cad_free})


_IDENTITY_FIELDS = ("body_sha256", "request_model", "question_variant", "question_ids",
                    "question_set_sha256", "ablation")


def _load_answers(path: Path | None, state_map: Mapping[str, Mapping[str, Any]]
                  ) -> tuple[dict[str, float], list[str], dict[str, Any] | None]:
    """``(scores, response_models, identity)`` after checking every row against the states."""
    if path is None:
        return {}, [], None
    scores: dict[str, float] = {}
    models: set[str] = set()
    identities: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for n, row in enumerate(read_jsonl(Path(path)), 1):
        rid = str(row.get("record_id"))
        if rid in seen:
            raise EvaluationError(f"answers row {n}: duplicate record_id {rid}")
        seen.add(rid)
        missing = [f for f in _IDENTITY_FIELDS if f not in row]
        if missing:
            raise EvaluationError(f"answers row {n}: no request identity {missing} (older harness?); "
                                  "re-run live to regenerate answers")
        if rid not in state_map:
            raise EvaluationError(f"answers row {n}: record {rid} is not in the current states (stale answers)")
        ablation = row["ablation"] if isinstance(row["ablation"], Mapping) else {}
        try:
            ident = request_identity(state_map[rid], model=str(row["request_model"]),
                                     variant=str(row["question_variant"]),
                                     question_names=[str(q) for q in row["question_ids"] or ()],
                                     cad_free=bool(ablation.get("cad_free")),
                                     context_free=bool(ablation.get("context_free")))
        except (KeyError, ValueError, TypeError) as exc:
            raise EvaluationError(f"answers row {n}: request cannot be rebuilt: {exc}") from exc
        if ident["question_set_sha256"] != row["question_set_sha256"]:
            raise EvaluationError(f"answers row {n}: question set differs from the current {row['question_variant']} set")
        if ident["body_sha256"] != row["body_sha256"]:
            raise EvaluationError(f"answers row {n}: body_sha256 differs from the request rebuilt from the "
                                  "current state (stale answers or a changed state)")
        key = {k: ident[k] for k in ("request_model", "question_variant", "question_ids",
                                     "question_set_sha256", "ablation")}
        identities[_canonical_sha(key)] = key
        answer = (row.get("answers") or {}).get(PRIMARY_QUESTION) or {}
        value = answer.get("noul")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            scores[rid] = float(value)
        if isinstance(row.get("response_model"), str):
            models.add(row["response_model"])
    if len(identities) > 1:
        raise EvaluationError("answers mix request models, question sets, variants or ablations; "
                              "evaluate one identity at a time")
    identity = next(iter(identities.values())) if identities else None
    return scores, sorted(models), identity


def _canonical_sha(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def evaluate(
    *,
    labels: Path,
    splits: Path | Mapping[str, Any] | SplitManifest,
    states: Path,
    out_dir: Path,
    answers: Path | None = None,
    operating_point: Path | None = None,
    cad_free_baseline: bool = False,
    bootstrap_rounds: int = BOOTSTRAP_ROUNDS,
) -> dict[str, Any]:
    """Score answers and baselines; write ``evaluation.json`` and ``operating_point.json``."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise EvaluationError(f"{out} exists and is not empty")
    manifest = splits if isinstance(splits, SplitManifest) else load_split_manifest(splits)  # type: ignore[arg-type]
    label_rows = [LabelRow.from_json(r) for r in read_jsonl(Path(labels))]
    state_map = {str(r["record_id"]): r["state"] for r in read_jsonl(Path(states))}
    assignment = manifest.assign(label_rows)
    jev_scores, response_models, answers_identity = _load_answers(answers, state_map)

    rows: list[_Row] = []
    recall_misses: dict[str, dict[str, int]] = {s: {} for s in SPLITS_SCORED}
    exploratory_records = 0
    for lr in label_rows:
        split = assignment[lr.record_id]
        if split not in SPLITS_SCORED:
            continue
        family = manifest.family_of(lr.group_sku) or "UNASSIGNED"
        exploratory = manifest.is_exploratory(lr.group_sku) or bool(set(lr.provenance_flags) & EXPLORATORY_FLAGS)
        if lr.label_source in RECALL_SOURCES and not lr.has_state:
            recall_misses[split][family] = recall_misses[split].get(family, 0) + 1
            exploratory_records += int(exploratory)
            continue
        truth = evaluation_truth(lr)
        if truth is None:
            continue
        if lr.record_id not in state_map:
            raise EvaluationError(f"label {lr.record_id} has no state")
        exploratory_records += int(exploratory)
        rows.append(_Row(lr.record_id, truth, split, family, exploratory, lr.label_source.startswith("synthetic_")))

    if answers is not None:
        evaluated = [r for r in rows if r.record_id in jev_scores]
        missing_answers = len(rows) - len(evaluated)
    else:
        evaluated, missing_answers = rows, None

    scorers: dict[str, dict[str, float]] = {}
    notes: dict[str, Any] = {}
    if answers is not None:
        scorers["jev"] = jev_scores
    engine = {r.record_id: v for r in evaluated if (v := _engine_confidence(state_map[r.record_id])) is not None}
    scorers["engine_confidence"] = engine
    notes["engine_confidence"] = {"coverage": len(engine) / len(evaluated) if evaluated else None,
                                  "note": "engine confidence is not necessarily a calibrated probability"}
    lr_scores, lr_note = _logreg_scores(evaluated, state_map, cad_free=cad_free_baseline)
    scorers["logistic_regression"] = lr_scores
    notes["logistic_regression"] = lr_note

    binding = {"split_manifest_sha256": manifest.sha256, "answers_identity": answers_identity,
               "cad_free_baseline": cad_free_baseline}
    if operating_point is not None:
        frozen_doc = json.loads(Path(operating_point).read_text(encoding="utf-8"))
        if not isinstance(frozen_doc, Mapping) or frozen_doc.get("schema") != OPERATING_POINT_SCHEMA:
            raise EvaluationError(f"operating point schema must be {OPERATING_POINT_SCHEMA}")
        if frozen_doc.get("sha256") != _canonical_sha({k: v for k, v in frozen_doc.items() if k != "sha256"}):
            raise EvaluationError("operating point sha256 does not match its content")
        if frozen_doc.get("split_manifest_sha256") != manifest.sha256:
            raise EvaluationError("operating point was frozen on a different split manifest")
        if frozen_doc.get("cad_free_baseline") != cad_free_baseline:
            raise EvaluationError("operating point was frozen with a different --cad-free-baseline setting")
        frozen_identity = frozen_doc.get("answers_identity")
        if answers_identity is not None and frozen_identity != answers_identity:
            raise EvaluationError("operating point was frozen on answers of a different model, question set, "
                                  "variant or ablation (or without Jev answers)")
        points = frozen_doc["points"]
        op_source = "given"
    else:
        points = {}
        for name, scores in scorers.items():
            fit = [r for r in evaluated if r.split in ("dev", "calib") and r.record_id in scores]
            points[name] = freeze_operating_point([r.truth for r in fit], [scores[r.record_id] for r in fit],
                                                  scorer=name)
        op_source = "frozen_on_dev_calib"
    if operating_point is not None:
        op_doc = dict(frozen_doc)
    else:
        op_doc = {"schema": OPERATING_POINT_SCHEMA, "points": points, **binding}
        op_doc["sha256"] = _canonical_sha({k: v for k, v in op_doc.items() if k != "sha256"})

    results: dict[str, Any] = {}
    for name, scores in scorers.items():
        op = points.get(name) or {"prune_below": None, "accept_at": None}
        per_split = {}
        for split in SPLITS_SCORED:
            per_split[split] = _metrics([r for r in evaluated if r.split == split], scores, op,
                                        recall_misses[split], bootstrap_rounds)
            per_split[split]["in_sample"] = split in ("dev", "calib")
        results[name] = per_split

    flags = []
    if exploratory_records:
        flags.append("exploratory_not_unseen")
    if not any(r.split == "eval" for r in evaluated):
        flags.append("in_sample_only")
    if evaluated and all(r.synthetic for r in evaluated):
        flags.append("synthetic_only")
    report = {
        "schema": "jev_evaluation_v1",
        "harness_version": HARNESS_VERSION,
        "flags": flags,
        "operating_point_source": op_source,
        "operating_point_sha256": op_doc["sha256"],
        "split_manifest_sha256": manifest.sha256,
        "split_overrides": list(manifest.overrides),
        "response_models": response_models,
        "answers_identity": answers_identity,
        "cad_free_baseline": cad_free_baseline,
        "counts": {
            "truth_rows": len(rows),
            "evaluated_rows": len(evaluated),
            "missing_answers": missing_answers,
            "exploratory_records": exploratory_records,
            "by_split": {s: sum(1 for r in evaluated if r.split == s) for s in SPLITS_SCORED},
        },
        "scorers": results,
        "scorer_notes": notes,
        "protocol": {"target_recall": TARGET_RECALL, "target_precision": TARGET_PRECISION, "ece_bins": ECE_BINS,
                     "bootstrap_rounds": bootstrap_rounds, "bootstrap_seed": BOOTSTRAP_SEED,
                     "truth": "labels_v1.evaluation_truth (weak labels never used)"},
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("operating_point.json", op_doc), ("evaluation.json", report)):
        (out / name).write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                encoding="utf-8")
    return report
