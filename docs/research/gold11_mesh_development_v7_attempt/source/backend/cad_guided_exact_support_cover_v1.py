"""Bounded exact raw-role cover proposals, not physical bend identities.

Every ownership component includes all arc-related exact-position claims. A
cover must be an existing member's RAW-ID support, never an assembled union.
This module supplies only a set-level proposal; fresh geometric/finite fitting
and source-context publication checks remain mandatory downstream.
"""

import hashlib
import json
import math

import numpy as np


MAX_RECORDS = 512
MAX_RAW_ROWS = 1_000_000
MAX_GRAPH_ROW_VISITS = 4_000_000
MAX_GRAPH_LINKS = 4_000_000
MAX_COVER_ROW_CHECKS = 32_000_000
MAX_COMPONENT_MEMBER_COMPARISONS = 262_144
MAX_PLAN_BYTES = 8*1024**2
FIT_NUMERIC_FIELDS = ("max_fit_residual_mm", "max_tangency_error_mm", "min_arc_sweep_deg",
    "max_circle_condition", "min_plane_spread_ratio", "max_normal_deviation_deg")


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _policy_bytes(policy):
    _require(type(policy) is dict and len(policy) == len(FIT_NUMERIC_FIELDS)+1
        and all(type(key) is str for key in policy)
        and set(policy) == set(FIT_NUMERIC_FIELDS) | {"fit_model"},
             "INVALID_COVER_FIT_POLICY_FIELDS")
    _require(type(policy["fit_model"]) is str and policy["fit_model"] == "free_circle"
        and all(type(policy[key]) in (int, float) and 0 <= policy[key] <= 1e12
            and math.isfinite(policy[key]) for key in FIT_NUMERIC_FIELDS),
        "INVALID_COVER_FIT_POLICY_SCALARS")
    # Fixed keys and bounded scalars are admitted BEFORE JSON allocation.
    result = _encoded(policy)
    _require(len(result) <= 4096, "COVER_FIT_POLICY_BYTES")
    return result


def plan_exact_support_covers_v1(points, records):
    """Return every component, including unavailable ones; fatal errors clear all.

    RAW-ID subset equality preserves byte lineage. Exact coordinate equality
    only detects ownership aliases/conflicts, and cannot supply a missing ID.
    Record order, nominal identity and serialized fit quality do not rank covers.
    """
    work = dict(graph_row_visits=0, graph_links=0, cover_row_checks=0, member_comparisons=0)
    result = dict(schema="EXACT_RAW_SUPPORT_COVER_PLAN_V1", status="UNAVAILABLE",
        groups=[], reasons=[], work=work, production_use_allowed=False,
        observed_bend_count=None, physical_event_identity="NOT_ESTABLISHED",
        policy="NARY_EXISTING_EXACT_RAW_SUPPORT_COVER",
        cover_row_checks_semantics="CONSERVATIVE_SET_ROW_WORK_NOT_LITERAL_CPU_OR_ALL_ALLOCATION_WORK")
    try:
        _require(type(points) is np.ndarray, "COVER_POINTS_REQUIRE_NDARRAY")
        p = points
        _require(p.ndim == 2 and p.shape[1] == 3 and p.dtype.kind in "fiu"
            and 1 <= len(p) <= MAX_RAW_ROWS and np.isfinite(p).all()
            and np.max(np.abs(p)) <= 1e6, "INVALID_COVER_POINTS")
        p = np.frombuffer(np.asarray(p, dtype="<f8").tobytes(), dtype="<f8").reshape(-1, 3)
        _require(1 <= len(p) <= MAX_RAW_ROWS and np.isfinite(p).all()
            and np.max(np.abs(p)) <= 1e6, "INVALID_OWNED_COVER_POINTS")
        _require(type(records) is list and len(records) <= MAX_RECORDS, "COVER_RECORD_BUDGET")
        owned, seen, planned = [], set(), 0
        for record in records:
            _require(type(record) is dict, "INVALID_COVER_RECORD")
            identity, raw = record["record_id"], record["raw_support"]
            _require(type(identity) is str and len(identity) == 64
                and all(c in "0123456789abcdef" for c in identity) and identity not in seen,
                "INVALID_COVER_RECORD_ID")
            seen.add(identity)
            _require(type(raw) is dict and type(raw["flange_rows"]) is list
                and len(raw["flange_rows"]) == 2, "INVALID_COVER_ROLES")
            roles = [raw["arc_rows"], *raw["flange_rows"]]
            _require(all(type(rows) is list and 1 <= len(rows) <= MAX_RAW_ROWS for rows in roles),
                     "INVALID_COVER_ROLE_SIZE")
            planned += sum(map(len, roles))
            _require(planned <= MAX_GRAPH_ROW_VISITS, "COVER_GRAPH_ROW_PREFLIGHT_BUDGET")
            _require(all(all(type(row) is int and 0 <= row < len(p) for row in rows)
                and rows == sorted(set(rows)) for rows in roles), "INVALID_COVER_RAW_ROWS")
            sets = tuple(frozenset(rows) for rows in roles)
            _require(not sets[0] & (sets[1] | sets[2]) and not sets[1] & sets[2],
                     "COVER_RAW_ROLE_OVERLAP")
            policy_bytes = _policy_bytes(record["fit_policy"])
            owned.append(dict(identity=identity, roles=sets, policy=policy_bytes))
        owned.sort(key=lambda row: row["identity"])
        parent = list(range(len(owned)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def join(first, second):
            work["graph_links"] += 1
            _require(work["graph_links"] <= MAX_GRAPH_LINKS, "COVER_GRAPH_LINK_BUDGET")
            a, b = find(first), find(second)
            if a != b:
                parent[max(a, b)] = min(a, b)

        claims = {}
        for index, record in enumerate(owned):
            for role, rows in enumerate(record["roles"]):
                for row in sorted(rows):
                    work["graph_row_visits"] += 1
                    key = tuple(float(value) for value in p[row])
                    claim = claims.setdefault(key, [set(), set()])
                    claim[0 if role == 0 else 1].add(index)
        bad = set()
        for arcs, flanges in claims.values():
            if not arcs:
                continue  # Shared flange context is not an arc ownership edge.
            members = sorted(arcs | flanges)
            for index in members[1:]:
                join(members[0], index)
            if flanges:
                bad.update(members)  # Also catches own-record position aliases.
        components = {}
        for index in range(len(owned)):
            components.setdefault(find(index), []).append(index)

        def charge_rows(count):
            work["cover_row_checks"] += count
            _require(work["cover_row_checks"] <= MAX_COVER_ROW_CHECKS, "COVER_ROW_CHECK_BUDGET")

        for members in components.values():
            identities = [owned[index]["identity"] for index in members]
            group = dict(group_id=hashlib.sha256(_encoded(identities)).hexdigest(),
                member_record_ids=identities, cover_record_id=None, member_flange_permutations={},
                status="UNAVAILABLE", reasons=[])
            result["groups"].append(group)
            if bad.intersection(members):
                group["reasons"] = ["CROSS_ROLE_POSITION_CONFLICT"]
                continue
            if len({owned[index]["policy"] for index in members}) != 1:
                group["reasons"] = ["INCOMPATIBLE_MEMBER_FIT_POLICIES"]
                continue
            arc_union, flange_union = set(), set()
            for index in members:
                roles = owned[index]["roles"]
                charge_rows(len(roles[0])+2*len(roles[1])+2*len(roles[2]))
                arc_union.update(roles[0])
                flange_union.update(roles[1] | roles[2])
            covers = []
            for candidate in members:
                cover = owned[candidate]["roles"]
                charge_rows(len(cover[0])+2*len(cover[1])+2*len(cover[2]))
                if cover[0] != arc_union or cover[1] | cover[2] != flange_union:
                    continue
                permutations = {}
                for index in members:
                    work["member_comparisons"] += 1
                    _require(work["member_comparisons"] <= MAX_COMPONENT_MEMBER_COMPARISONS,
                             "COVER_MEMBER_COMPARISON_BUDGET")
                    roles = owned[index]["roles"]
                    charge_rows(len(roles[0])+2*len(roles[1])+2*len(roles[2]))
                    arc_subset = roles[0] <= cover[0]
                    choices = [order for order in ((0, 1), (1, 0))
                        if arc_subset and roles[1] <= cover[1+order[0]] and roles[2] <= cover[1+order[1]]]
                    if len(choices) != 1:
                        break
                    permutations[owned[index]["identity"]] = list(choices[0])
                if len(permutations) == len(members):
                    covers.append((owned[candidate]["identity"], permutations))
            if not covers:
                group["reasons"] = ["NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER"]
                continue
            # Every admitted cover must cover every other admitted cover. Thus
            # they have identical arc and unordered flange sets; choosing this
            # stable identity selects serialization only, not a better model.
            identity, permutations = min(covers, key=lambda value: value[0])
            group.update(status="COVERED", cover_record_id=identity,
                         member_flange_permutations=permutations)
        result["groups"].sort(key=lambda row: row["group_id"])
        result.update(status="COMPLETE", raw_points_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
            input_record_count=len(owned), input_role_row_count=planned,
            input_record_ids=sorted(seen), component_count=len(result["groups"]))
        size = 0
        for fragment in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(result):
            size += len(fragment.encode())
            _require(size <= MAX_PLAN_BYTES, "COVER_PLAN_BYTE_BUDGET")
        return result
    except Exception as error:
        result.update(status="UNAVAILABLE", groups=[],
            reasons=["COVER_PLAN_FAILURE:"+type(error).__name__+":"+str(error)[:180]],
            discarded_plan_prefix=True)
        for key in ("raw_points_sha256", "input_record_count", "input_role_row_count", "input_record_ids", "component_count"):
            result.pop(key, None)
        return result
