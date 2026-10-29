"""Recover short terminal bends that close a mesh-surface quadrilateral."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_return_doublet_completion import parallel_line_metrics


def _surface_pair(item: Mapping[str, Any]) -> tuple[Any, Any] | None:
    left = item.get("surface1_id")
    right = item.get("surface2_id")
    if left is None or right is None or left == right:
        return None
    return left, right


def _is_duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    metrics = parallel_line_metrics(left, right)
    return (
        metrics["axis_angle_deg"] <= 7.5
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.45
    )


def complete_terminal_surface_cycles(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Complete high-confidence short edges opposite a selected long edge.

    The candidate surface graph must contain the complete four-edge cycle.  The
    target is the short edge opposite a long edge already represented in the
    selection; the two remaining raw interfaces are required as topology
    witnesses.  This lets weak normal-transition evidence survive at thin end
    walls without admitting isolated boundary outlines.
    """

    output = [dict(item) for item in selected]
    raw = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "scan_mesh_adjacency"
        and _surface_pair(item) is not None
    ]
    selected_ids = {str(item.get("candidate_id") or "") for item in output}
    represented = [item for item in raw if any(_is_duplicate(item, keep) for keep in output)]
    represented_ids = {str(item.get("candidate_id") or "") for item in represented}
    by_surface_pair = {
        frozenset(_surface_pair(item)): item
        for item in raw
        if _surface_pair(item) is not None
    }
    additions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for candidate in raw:
        candidate_id = str(candidate.get("candidate_id") or "")
        confidence = float(candidate.get("confidence") or 0.0)
        minimum_gap = float(candidate.get("min_surface_gap") or 1.0e9)
        target_span = max(float(candidate.get("span_mm") or 0.0), 1.0e-9)
        strong_raw_cycle_target = confidence >= 0.98 and minimum_gap <= 0.50 and target_span <= 45.0
        selected_cycle_target = confidence >= 0.60 and minimum_gap <= 6.0 and target_span <= 25.0
        if (
            not candidate_id
            or candidate_id in selected_ids
            or any(_is_duplicate(candidate, item) for item in output)
            or not (strong_raw_cycle_target or selected_cycle_target)
        ):
            continue
        target_pair = _surface_pair(candidate)
        assert target_pair is not None
        completion_evidence = None
        for central in represented:
            central_pair = _surface_pair(central)
            assert central_pair is not None
            if (
                len(set(target_pair + central_pair)) != 4
                or float(central.get("span_mm") or 0.0) < 3.0 * target_span
            ):
                continue
            for connectors in (
                ((target_pair[0], central_pair[0]), (target_pair[1], central_pair[1])),
                ((target_pair[0], central_pair[1]), (target_pair[1], central_pair[0])),
            ):
                witnesses = [by_surface_pair.get(frozenset(pair)) for pair in connectors]
                witnesses_are_selected = all(
                    witness is not None
                    and str(witness.get("candidate_id") or "") in represented_ids
                    for witness in witnesses
                )
                if all(witnesses) and (strong_raw_cycle_target or witnesses_are_selected):
                    completion_evidence = {
                        "candidate_id": candidate_id,
                        "opposite_represented_candidate_id": central.get("candidate_id"),
                        "connector_candidate_ids": [item.get("candidate_id") for item in witnesses],
                        "span_ratio": float(central.get("span_mm") or 0.0) / target_span,
                        "evidence_mode": (
                            "strong_raw_cycle"
                            if strong_raw_cycle_target
                            else "fully_selected_cycle"
                        ),
                    }
                    break
            if completion_evidence is not None:
                break
        if completion_evidence is None:
            continue
        candidate["candidate_source"] = "terminal_surface_cycle_completion"
        candidate["completion_evidence"] = completion_evidence
        additions.append(candidate)
        evidence.append(completion_evidence)
        output.append(candidate)
        selected_ids.add(candidate_id)

    return {
        "selected": output,
        "added": additions,
        "activated": bool(additions),
        "evidence": evidence,
    }
