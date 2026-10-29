"""Rewire an isolated short mesh edge into the established surface graph."""

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


def rewire_isolated_short_component(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace a two-node island with a candidate joining the main component."""

    output = [dict(item) for item in selected]
    raw = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "scan_mesh_adjacency"
        and _surface_pair(item) is not None
    ]
    adjacency: dict[Any, set[Any]] = {}
    for item in output:
        pair = _surface_pair(item)
        if pair is None:
            continue
        adjacency.setdefault(pair[0], set()).add(pair[1])
        adjacency.setdefault(pair[1], set()).add(pair[0])
    components: list[set[Any]] = []
    seen: set[Any] = set()
    for node in adjacency:
        if node in seen:
            continue
        component: set[Any] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency.get(current, set()) - component)
        seen.update(component)
        components.append(component)

    removals: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for isolated in list(output):
        pair = _surface_pair(isolated)
        if pair is None:
            continue
        component = next((item for item in components if pair[0] in item), set())
        if (
            len(component) != 2
            or any(len(adjacency.get(surface, set())) != 1 for surface in pair)
            or float(isolated.get("span_mm") or 0.0) > 40.0
        ):
            continue
        for candidate in raw:
            candidate_pair = _surface_pair(candidate)
            assert candidate_pair is not None
            if (
                any(_is_duplicate(candidate, item) for item in output)
                or float(candidate.get("span_mm") or 0.0) > 40.0
                or float(candidate.get("confidence") or 0.0) < 0.60
                or abs(
                    float(candidate.get("bend_angle") or 0.0)
                    - float(isolated.get("bend_angle") or 0.0)
                )
                > 10.0
            ):
                continue
            shared = set(pair) & set(candidate_pair)
            other = set(candidate_pair) - shared
            if len(shared) != 1 or len(other) != 1:
                continue
            main_surface = next(iter(other))
            main_component = next(
                (item for item in components if main_surface in item),
                set(),
            )
            if len(main_component) < 3:
                continue
            output.remove(isolated)
            replacement = dict(candidate)
            replacement["candidate_source"] = "isolated_component_rewire"
            replacement["rewired_from_candidate_id"] = isolated.get("candidate_id")
            output.append(replacement)
            removals.append(isolated)
            additions.append(replacement)
            evidence.append(
                {
                    "removed_candidate_id": isolated.get("candidate_id"),
                    "added_candidate_id": candidate.get("candidate_id"),
                    "shared_surface_id": next(iter(shared)),
                    "main_component_size": len(main_component),
                }
            )
            break

    return {
        "selected": output,
        "removed": removals,
        "added": additions,
        "activated": bool(additions),
        "evidence": evidence,
    }

