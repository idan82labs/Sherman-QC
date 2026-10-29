"""Promote repeated interface-birth axes without using a bend-count oracle."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_line_object_clustering import cluster_line_candidates


def _endpoints(record: Mapping[str, Any]) -> np.ndarray:
    if record.get("span_endpoints") is not None:
        endpoints = np.asarray(record["span_endpoints"], dtype=np.float64)
    else:
        anchor = np.asarray(record["anchor"], dtype=np.float64)
        axis = np.asarray(record["axis_direction"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        half = 0.5 * float(record["span_mm"]) * axis
        endpoints = np.asarray((anchor - half, anchor + half))
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("invalid finite line")
    return endpoints


def _duplicates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    first = _endpoints(left)
    second = _endpoints(right)
    first_vector = first[1] - first[0]
    second_vector = second[1] - second[0]
    first_span = float(np.linalg.norm(first_vector))
    second_span = float(np.linalg.norm(second_vector))
    if min(first_span, second_span) <= 1e-9:
        return False
    first_axis = first_vector / first_span
    second_axis = second_vector / second_span
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(first_axis, second_axis)), 0.0, 1.0))))
    delta = np.mean(second, axis=0) - np.mean(first, axis=0)
    offset = max(
        float(np.linalg.norm(delta - first_axis * np.dot(delta, first_axis))),
        float(np.linalg.norm(delta - second_axis * np.dot(delta, second_axis))),
    )
    interval = sorted(float(np.dot(point - np.mean(first, axis=0), first_axis)) for point in second)
    overlap = max(0.0, min(0.5 * first_span, interval[1]) - max(-0.5 * first_span, interval[0]))
    return angle <= 5.0 and offset <= 4.0 and overlap / min(first_span, second_span) >= 0.55


def promote_repeated_interface_births(
    selected: Sequence[Mapping[str, Any]],
    bend_hypotheses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    births = []
    by_id = {}
    for source in bend_hypotheses:
        if str(source.get("source_kind") or "") != "interface_birth":
            continue
        item = dict(source)
        item["candidate_source"] = "interface_birth"
        identifier = str(item.get("bend_id") or "")
        if not identifier:
            continue
        births.append(item)
        by_id[identifier] = item
    objects = cluster_line_candidates(
        births,
        axis_tolerance_deg=4.0,
        offset_tolerance_mm=2.0,
        minimum_shorter_overlap=0.55,
    )
    output = [dict(item) for item in selected]
    promoted = []
    for line_object in objects:
        member_ids = list(line_object.get("cluster_member_ids") or [])
        members = [by_id[identifier] for identifier in member_ids if identifier in by_id]
        flange_pairs = {
            tuple(sorted(str(value) for value in member.get("canonical_flange_pair") or ()))
            for member in members
        }
        flange_pairs.discard(())
        if len(members) < 2 or len(flange_pairs) < 2:
            continue
        representative = max(
            members,
            key=lambda item: (float(item.get("span_mm") or 0.0), str(item.get("bend_id") or "")),
        )
        candidate = dict(representative)
        candidate["bend_id"] = f"IBC{len(promoted) + 1}"
        candidate["candidate_source"] = "interface_birth_consensus"
        candidate["selection_reason"] = "repeated_interface_birth_consensus"
        candidate["consensus_member_ids"] = sorted(member_ids)
        candidate["consensus_flange_pairs"] = [list(pair) for pair in sorted(flange_pairs)]
        try:
            if any(_duplicates(candidate, retained) for retained in output):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        output.append(candidate)
        promoted.append(candidate)
    return {
        "selected": output,
        "promoted": promoted,
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
    }
