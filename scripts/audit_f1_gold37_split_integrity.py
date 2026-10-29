#!/usr/bin/env python3
"""Audit the frozen Gold37 family split for evaluation leakage.

This command is evaluation-only.  It never modifies an inference pool and
never exposes part-family, filename, truth, or fold metadata to F1.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/research/f1-gold27-20260803/freeze"
DEFAULT_GOLD = BASE / "sealed_gold_manifest.json"
DEFAULT_INFERENCE = BASE / "inference_manifest.json"
DEFAULT_SPLIT = BASE / "grouped_five_fold_split.json"
DEFAULT_OUTPUT = BASE / "grouped_five_fold_split_integrity.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_split_integrity(
    gold: Mapping[str, Any],
    inference: Mapping[str, Any],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    gold_cases = {str(row["case_token"]): dict(row) for row in gold.get("cases") or ()}
    inference_cases = {
        str(row["case_token"]): dict(row) for row in inference.get("cases") or ()
    }
    split_cases = {str(row["case_token"]): dict(row) for row in split.get("cases") or ()}
    case_sets_equal = set(gold_cases) == set(inference_cases) == set(split_cases)

    errors: list[str] = []
    if not case_sets_equal:
        errors.append("gold, inference, and split case-token sets differ")
    corpus_ids = {
        str(payload.get("corpus_freeze_id") or "")
        for payload in (gold, inference, split)
    }
    corpus_freeze_consistent = len(corpus_ids) == 1 and "" not in corpus_ids
    if not corpus_freeze_consistent:
        errors.append("corpus freeze IDs are missing or inconsistent")

    family_to_folds: dict[str, set[int]] = defaultdict(set)
    scan_sha_to_folds: dict[str, set[int]] = defaultdict(set)
    scan_sha_to_cases: dict[str, list[str]] = defaultdict(list)
    label_sha_to_folds: dict[str, set[int]] = defaultdict(set)
    label_sha_to_cases: dict[str, list[str]] = defaultdict(list)
    fold_case_counts: Counter[int] = Counter()
    fold_family_sets: dict[int, set[str]] = defaultdict(set)
    fold_gold_bends: Counter[int] = Counter()

    for token in sorted(set(gold_cases) & set(inference_cases) & set(split_cases)):
        gold_row = gold_cases[token]
        inference_row = inference_cases[token]
        split_row = split_cases[token]
        family = str(gold_row.get("part_key") or "")
        split_family = str(split_row.get("part_key") or "")
        if not family or family != split_family:
            errors.append(f"{token}: split part family disagrees with sealed gold")
        fold = int(split_row["fold"])
        family_to_folds[family].add(fold)
        fold_case_counts[fold] += 1
        fold_family_sets[fold].add(family)
        fold_gold_bends[fold] += int(
            gold_row.get("gold_bend_count")
            if gold_row.get("gold_bend_count") is not None
            else len(gold_row.get("manual_bends") or ())
        )
        gold_scan_sha = str(gold_row.get("source_scan_sha256") or "")
        inference_scan_sha = str(inference_row.get("source_scan_sha256") or "")
        if not gold_scan_sha or gold_scan_sha != inference_scan_sha:
            errors.append(f"{token}: inference scan hash disagrees with sealed gold")
        scan_sha_to_folds[gold_scan_sha].add(fold)
        scan_sha_to_cases[gold_scan_sha].append(token)
        label_sha = str(gold_row.get("label_sha256") or "")
        if label_sha:
            label_sha_to_folds[label_sha].add(fold)
            label_sha_to_cases[label_sha].append(token)

    family_cross_fold = {
        family: sorted(folds)
        for family, folds in family_to_folds.items()
        if len(folds) > 1
    }
    exact_scan_cross_fold = {
        digest: {
            "folds": sorted(scan_sha_to_folds[digest]),
            "case_tokens": sorted(scan_sha_to_cases[digest]),
        }
        for digest in sorted(scan_sha_to_folds)
        if digest and len(scan_sha_to_folds[digest]) > 1
    }
    exact_label_cross_fold = {
        digest: {
            "folds": sorted(label_sha_to_folds[digest]),
            "case_tokens": sorted(label_sha_to_cases[digest]),
        }
        for digest in sorted(label_sha_to_folds)
        if len(label_sha_to_folds[digest]) > 1
    }
    if family_cross_fold:
        errors.append("one or more part families cross folds")
    if exact_scan_cross_fold:
        errors.append("one or more byte-identical scans cross folds")
    if exact_label_cross_fold:
        errors.append("one or more byte-identical label files cross folds")

    all_families = set(family_to_folds)
    folds = list(split.get("folds") or ())
    declared_fold_ids = {int(row["fold"]) for row in folds}
    observed_fold_ids = set(fold_case_counts)
    if declared_fold_ids != observed_fold_ids:
        errors.append("declared and observed fold IDs differ")
    declared_membership_consistent = True
    for row in folds:
        fold = int(row["fold"])
        locked = {str(value) for value in row.get("locked_part_families") or ()}
        development = {
            str(value) for value in row.get("development_part_families") or ()
        }
        if locked != fold_family_sets[fold] or development != all_families - locked:
            declared_membership_consistent = False
            errors.append(f"fold {fold}: declared family membership is inconsistent")
        if locked & development:
            declared_membership_consistent = False
            errors.append(f"fold {fold}: locked and development families overlap")

    exact_scan_duplicate_groups = [
        {
            "source_scan_sha256": digest,
            "case_tokens": sorted(tokens),
            "fold": next(iter(scan_sha_to_folds[digest])),
        }
        for digest, tokens in sorted(scan_sha_to_cases.items())
        if digest and len(tokens) > 1 and len(scan_sha_to_folds[digest]) == 1
    ]
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold37_split_integrity_audit",
        "evaluation_only": True,
        "inference_metadata_exposure": False,
        "corpus_freeze_id": str(gold.get("corpus_freeze_id") or ""),
        "split_id": str(split.get("split_id") or ""),
        "case_count": len(gold_cases),
        "part_family_count": len(all_families),
        "fold_count": len(observed_fold_ids),
        "checks": {
            "case_sets_equal": case_sets_equal,
            "corpus_freeze_consistent": corpus_freeze_consistent,
            "part_families_confined_to_one_fold": not family_cross_fold,
            "byte_identical_scans_confined_to_one_fold": not exact_scan_cross_fold,
            "byte_identical_labels_confined_to_one_fold": not exact_label_cross_fold,
            "declared_membership_consistent": declared_membership_consistent,
        },
        "passed": not errors,
        "errors": errors,
        "cross_fold_part_families": family_cross_fold,
        "cross_fold_scan_hashes": exact_scan_cross_fold,
        "cross_fold_label_hashes": exact_label_cross_fold,
        "within_fold_exact_scan_duplicate_groups": exact_scan_duplicate_groups,
        "folds": [
            {
                "fold": fold,
                "case_count": int(fold_case_counts[fold]),
                "part_family_count": len(fold_family_sets[fold]),
                "gold_bend_count": int(fold_gold_bends[fold]),
                "part_families": sorted(fold_family_sets[fold]),
            }
            for fold in sorted(observed_fold_ids)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--inference", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_split_integrity(
        _read(args.gold.resolve()),
        _read(args.inference.resolve()),
        _read(args.split.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
