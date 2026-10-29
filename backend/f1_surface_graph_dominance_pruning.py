"""Prune scan-mesh outlines contradicted by stronger selected surface topology."""

from __future__ import annotations

from collections import deque
from typing import Any, Hashable, Mapping, Sequence


def _pair(record: Mapping[str, Any]) -> frozenset[Hashable] | None:
    first = record.get("surface1_id")
    second = record.get("surface2_id")
    if first is None or second is None or first == second:
        return None
    return frozenset((first, second))


def _score(record: Mapping[str, Any]) -> float:
    return float(
        record.get("two_sided_transition_score")
        or record.get("cluster_max_two_sided_transition_score")
        or 0.0
    )


def _span(record: Mapping[str, Any]) -> float:
    return float(record.get("span_mm") or 0.0)


def _alternate_path_minimum_score(
    start: Hashable,
    end: Hashable,
    excluded_index: int,
    adjacency: Mapping[Hashable, list[tuple[Hashable, int]]],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[float, list[int]] | None:
    queue = deque([(start, float("inf"), [])])
    best = {start: float("inf")}
    while queue:
        node, path_minimum, path = queue.popleft()
        for neighbor, edge_index in adjacency.get(node, []):
            if edge_index == excluded_index:
                continue
            candidate_minimum = min(path_minimum, _score(selected[edge_index]))
            if candidate_minimum <= best.get(neighbor, -1.0):
                continue
            if neighbor == end:
                return candidate_minimum, [*path, edge_index]
            best[neighbor] = candidate_minimum
            queue.append((neighbor, candidate_minimum, [*path, edge_index]))
    return None


def prune_surface_graph_dominance(
    selected: Sequence[Mapping[str, Any]],
    *,
    minimum_leaf_transition_advantage: float = 0.25,
    maximum_leaf_joint_degree: int = 3,
    maximum_leaf_selection_probability: float = 0.50,
    maximum_short_branch_span_ratio: float = 0.35,
    maximum_triangle_span_ratio: float = 0.25,
    minimum_triangle_transition_advantage: float = 0.20,
    minimum_cycle_transition_advantage: float = 0.20,
) -> dict[str, Any]:
    """Remove only candidates dominated by an explicit selected graph witness."""

    output = [dict(item) for item in selected]
    adjacency: dict[Hashable, list[tuple[Hashable, int]]] = {}
    edge_by_pair: dict[frozenset[Hashable], int] = {}
    ambiguous_pairs: set[frozenset[Hashable]] = set()
    for index, item in enumerate(output):
        pair = _pair(item)
        if pair is None:
            continue
        if pair in edge_by_pair:
            edge_by_pair.pop(pair, None)
            ambiguous_pairs.add(pair)
            continue
        if pair in ambiguous_pairs:
            continue
        edge_by_pair[pair] = index
    for pair, index in edge_by_pair.items():
        first, second = tuple(pair)
        adjacency.setdefault(first, []).append((second, index))
        adjacency.setdefault(second, []).append((first, index))

    pruned: dict[int, dict[str, Any]] = {}
    for pair, index in edge_by_pair.items():
        first, second = tuple(pair)
        candidate = output[index]
        candidate_score = _score(candidate)
        common = {neighbor for neighbor, _ in adjacency.get(first, [])} & {
            neighbor for neighbor, _ in adjacency.get(second, [])
        }
        for neighbor in common:
            first_index = edge_by_pair.get(frozenset((first, neighbor)))
            second_index = edge_by_pair.get(frozenset((second, neighbor)))
            if first_index is None or second_index is None:
                continue
            witnesses = (output[first_index], output[second_index])
            witness_span = min(_span(item) for item in witnesses)
            witness_score = min(_score(item) for item in witnesses)
            if (
                witness_span > 0.0
                and _span(candidate) <= maximum_triangle_span_ratio * witness_span
                and candidate_score + minimum_triangle_transition_advantage <= witness_score
            ):
                pruned[index] = {
                    "candidate_id": candidate.get("candidate_id") or candidate.get("bend_id"),
                    "reason": "weak_short_triangle_shortcut",
                    "common_surface_id": neighbor,
                }
                break
        if index in pruned:
            continue

        degrees = (len(adjacency.get(first, [])), len(adjacency.get(second, [])))
        if 1 in degrees:
            leaf = first if degrees[0] == 1 else second
            joint = second if leaf == first else first
            joint_degree = len(adjacency.get(joint, []))
            if 2 <= joint_degree <= maximum_leaf_joint_degree:
                witness_scores = [
                    _score(output[edge_index])
                    for _neighbor, edge_index in adjacency[joint]
                    if edge_index != index
                ]
                witness_spans = [
                    _span(output[edge_index])
                    for _neighbor, edge_index in adjacency[joint]
                    if edge_index != index
                ]
                probability_value = candidate.get("selection_probability")
                probability = (
                    float(probability_value) if probability_value is not None else 0.0
                )
                short_branch = bool(
                    witness_spans
                    and max(witness_spans) > 0.0
                    and _span(candidate)
                    <= maximum_short_branch_span_ratio * max(witness_spans)
                )
                if (
                    witness_scores
                    and candidate_score + minimum_leaf_transition_advantage <= max(witness_scores)
                    and probability < maximum_leaf_selection_probability
                    and not short_branch
                ):
                    pruned[index] = {
                        "candidate_id": candidate.get("candidate_id") or candidate.get("bend_id"),
                        "reason": "weak_leaf_extension",
                        "leaf_surface_id": leaf,
                        "joint_surface_id": joint,
                    }
                    continue

        alternate = _alternate_path_minimum_score(
            first, second, index, adjacency, output
        )
        if (
            alternate is not None
            and _span(candidate)
            <= 0.5 * max((_span(item) for item in output), default=0.0)
            and candidate_score + minimum_cycle_transition_advantage <= alternate[0]
        ):
            pruned[index] = {
                "candidate_id": candidate.get("candidate_id") or candidate.get("bend_id"),
                "reason": "weak_cycle_edge",
                "alternate_path_minimum_transition_score": alternate[0],
            }
            hub = max(adjacency, key=lambda node: (len(adjacency[node]), str(node)))
            if len(adjacency[hub]) >= 3:
                for path_index in alternate[1]:
                    path_pair = _pair(output[path_index])
                    if path_pair is None or hub in path_pair:
                        continue
                    pruned[path_index] = {
                        "candidate_id": output[path_index].get("candidate_id")
                        or output[path_index].get("bend_id"),
                        "reason": "weak_cycle_nonhub_alternate_branch",
                        "dominant_hub_surface_id": hub,
                    }

    return {
        "selected": [item for index, item in enumerate(output) if index not in pruned],
        "pruned": [pruned[index] for index in sorted(pruned)],
        "activated": bool(pruned),
    }
