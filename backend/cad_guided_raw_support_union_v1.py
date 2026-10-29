"""Conditional raw-role union proposals; unqualified, not measurement output.

Every structurally valid input record participates, irrespective of prior fit
or finite-support status. Exact positions establish arc-related ownership and
cross-role conflicts only; RAW IDs establish covers, parity, anchors and unions.
No coordinates are welded, no fit is performed, and no failed member is dropped.

The existing-cover branch precedes the structural union fallback. Its geometry
must not be retried through the fallback. Common raw anchors and consistent
flange parity are conservative sample-association conditions, NOT proof of one
physical bend, adequate fit conditioning, a unique pose, or a manufacturing sign.
In particular, compatible nearby sheets can remain indistinguishable here.

All outputs have measurement_admissible=False. A future wrapper must validate
complete original context publication, every contributing association's role
filter exclusions, byte/row lineage, and a single whole-support fresh fit plus
finite-support checks. None of those validations is performed by this planner.

Working bytes are a coarse prospective payload/object reservation, not observed
RSS or a portable Python allocator theorem. Input arrays/records owned by the
caller, interpreter/library memory and downstream fitting are outside it. A
quiescent caller is required during snapshotting; no atomic concurrency claim
is made. The running V5 benchmark does not import or use this future module.
"""

from dataclasses import dataclass, fields
import hashlib
import json

import numpy as np

from . import cad_guided_exact_support_cover_v1 as _cover


@dataclass(frozen=True)
class RawSupportUnionBudgetsV1:
    max_records: int = 512
    max_raw_rows: int = 1_000_000
    max_input_role_rows: int = 4_000_000
    max_ownership_links: int = 4_000_000
    max_parity_links: int = 4_000_000
    max_set_row_work: int = 32_000_000
    max_member_comparisons: int = 262_144
    max_ordering_work: int = 128_000_000
    max_input_binding_bytes: int = 64 * 1024**2
    max_plan_bytes: int = 8 * 1024**2
    # Funds the declared 4M role memberships at the 1M-row/512-record maxima
    # under this accounting model, not actual RSS or every other work/output cap.
    max_working_reservation_bytes: int = 4 * 1024**3


_MAXIMA = {field.name: field.default for field in fields(RawSupportUnionBudgetsV1)}
_MINIMA = dict(max_raw_rows=1, max_input_binding_bytes=1,
               max_plan_bytes=8192, max_working_reservation_bytes=1)
_REQUIRED_DOWNSTREAM = (
    "COMPLETE_ORIGINAL_CONTEXT_PUBLICATION",
    "ALL_CONTRIBUTING_ASSOCIATION_ROLE_FILTER_EXCLUSIONS",
    "ORIGINAL_BYTE_AND_RAW_ROW_LINEAGE",
    "ONE_FRESH_WHOLE_SUPPORT_FIT_AND_FINITE_SUPPORT_CHECKS",
)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _configuration(budgets):
    _require(type(budgets) is RawSupportUnionBudgetsV1, "INVALID_UNION_BUDGET_TYPE")
    snapshot = {name: getattr(budgets, name) for name in _MAXIMA}
    _require(all(type(value) is int and _MINIMA.get(name, 0) <= value <= _MAXIMA[name]
                 for name, value in snapshot.items()), "INVALID_UNION_BUDGET_VALUE")
    return snapshot


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class _Work:
    def __init__(self, config):
        self.config = config
        self.values = dict(input_role_rows=0, ownership_row_visits=0,
            ownership_links=0, ownership_find_steps=0, parity_row_visits=0,
            parity_links=0, parity_find_steps=0, set_row_work=0,
            member_comparisons=0, ordering_work=0, input_binding_bytes=0,
            plan_bytes_reserved=8192, working_reservation_bytes=0)
        self.point_reservation = 0
        self.record_count = 0

    def charge(self, key, count, limit=None):
        self.values[key] += count  # Attempted work survives a refused preflight.
        cap = self.config[limit or "max_" + key]
        _require(self.values[key] <= cap, "UNION_BUDGET:" + key)

    def memory(self):
        # Reserve owned canonical/source conversion overlap, bounded metadata,
        # all cumulative role memberships and transient sets/position postings,
        # plus three plan envelopes. No nominal row cap promises simultaneous
        # admission at that maximum under this independent reservation cap.
        reservation = (1024**2 + self.point_reservation + 8192*self.record_count
            + 1024*self.values["input_role_rows"] + 3*self.config["max_plan_bytes"])
        self.values["working_reservation_bytes"] = reservation
        _require(reservation <= self.config["max_working_reservation_bytes"],
                 "UNION_BUDGET:working_reservation_bytes")

    def rows(self, count):
        self.charge("set_row_work", count)

    def ordered(self, values, *, key=None):
        count = len(values)
        self.charge("ordering_work", 2*count*max(1, count.bit_length()))
        return sorted(values, key=key)

    def report(self, count):
        self.charge("plan_bytes_reserved", count, "max_plan_bytes")


class _ParityForest:
    """Bounded union-by-size forest; relation is the XOR between node labels.

    A find has at most floor(log2(number of records)) links; it never scans
    unbounded external data. Find telemetry is separate from admitted relation
    count, and the number of calls is bounded by twice admitted links plus at
    most two record scans per component. No all-pairs graph is materialized.
    """
    def __init__(self, count, work, prefix):
        self.parent = list(range(count))
        self.size = [1]*count
        self.parity = [0]*count
        self.work, self.prefix = work, prefix

    def find(self, index):
        parity, steps = 0, 0
        while self.parent[index] != index:
            steps += 1
            self.work.values[self.prefix + "_find_steps"] += 1
            _require(steps <= len(self.parent).bit_length(), "UNION_FOREST_DEPTH")
            parity ^= self.parity[index]
            index = self.parent[index]
        return index, parity

    def join(self, first, second, relation=0):
        self.work.charge(self.prefix + "_links", 1)
        a, pa = self.find(first)
        b, pb = self.find(second)
        if a == b:
            return (pa ^ pb) == relation
        if (self.size[a], -a) < (self.size[b], -b):
            a, b = b, a
        self.parent[b] = a
        self.parity[b] = pa ^ pb ^ relation
        self.size[a] += self.size[b]
        return True


def _own(points, records, work):
    _require(type(points) is np.ndarray, "UNION_POINTS_REQUIRE_NDARRAY")
    shape, dtype, source_bytes = tuple(points.shape), points.dtype, points.nbytes
    _require(len(shape) == 2 and shape[1] == 3 and dtype.kind in "fiu"
        and 1 <= shape[0] <= work.config["max_raw_rows"], "INVALID_UNION_POINTS")
    _require(type(records) is list and len(records) <= work.config["max_records"],
             "UNION_RECORD_BUDGET")
    work.record_count = len(records)
    work.point_reservation = 3*max(source_bytes, shape[0]*24)
    work.memory()
    # Direct comparisons avoid signed-integer abs(INT64_MIN) overflow. The
    # canonical float64 snapshot matches the existing cover's position domain.
    _require(np.isfinite(points).all() and (points >= -1e6).all()
        and (points <= 1e6).all(), "INVALID_UNION_POINTS")
    point_blob = np.asarray(points, dtype="<f8", order="C").tobytes()
    _require(len(point_blob) == shape[0]*24, "UNION_POINT_SNAPSHOT_SIZE_CHANGED")
    p = np.frombuffer(point_blob, dtype="<f8").reshape(shape)
    _require(tuple(p.shape) == shape and np.isfinite(p).all()
        and (p >= -1e6).all() and (p <= 1e6).all(), "INVALID_OWNED_UNION_POINTS")
    record_snapshot = tuple(records)
    _require(len(record_snapshot) == work.record_count, "UNION_RECORD_SNAPSHOT_CHANGED")
    owned, seen = [], set()
    for record in record_snapshot:
        _require(type(record) is dict, "INVALID_UNION_RECORD")
        identity, raw = record["record_id"], record["raw_support"]
        _require(type(identity) is str and len(identity) == 64
            and all(c in "0123456789abcdef" for c in identity) and identity not in seen,
            "INVALID_UNION_RECORD_ID")
        seen.add(identity)
        _require(type(raw) is dict and type(raw["flange_rows"]) is list
            and len(raw["flange_rows"]) == 2, "INVALID_UNION_RAW_ROLES")
        roles = (raw["arc_rows"], *raw["flange_rows"])
        _require(all(type(rows) is list and 1 <= len(rows) <= shape[0] for rows in roles),
                 "INVALID_UNION_ROLE_SIZE")
        sizes = tuple(map(len, roles))
        work.charge("input_role_rows", sum(sizes))
        work.memory()  # Before tuple/set snapshots or position-claim allocation.
        snapshots = tuple(tuple(rows) for rows in roles)
        _require(tuple(map(len, snapshots)) == sizes, "UNION_ROLE_SNAPSHOT_CHANGED")
        _require(all(all(type(row) is int and 0 <= row < shape[0] for row in rows)
            and all(rows[i-1] < rows[i] for i in range(1, len(rows)))
            for rows in snapshots), "INVALID_UNION_RAW_ROWS")
        work.rows(sum(sizes)*3)  # Set construction plus disjoint-role checks.
        sets = tuple(frozenset(rows) for rows in snapshots)
        _require(not (sets[0] & sets[1]) and not (sets[0] & sets[2])
            and not (sets[1] & sets[2]), "UNION_RAW_ROLE_OVERLAP")
        policy = record["fit_policy"]
        _require(type(policy) is dict and len(policy) == 7
            and all(type(key) is str for key in policy), "INVALID_UNION_FIT_POLICY")
        policy = dict(policy)
        policy_blob = _cover._policy_bytes(policy)
        owned.append(dict(identity=identity, rows=snapshots, roles=sets,
                          policy=policy_blob, fit_policy=policy))
    owned = work.ordered(owned, key=lambda row: row["identity"])
    return p, owned


def _structural_digest(owned, work):
    # Views reuse admitted immutable role tuples: no second raw-ID list copy.
    views = [dict(record_id=row["identity"], raw_support=dict(arc_rows=row["rows"][0],
        flange_rows=row["rows"][1:]), fit_policy=row["fit_policy"]) for row in owned]
    digest = hashlib.sha256()
    for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).iterencode(views):
        blob = fragment.encode("utf-8")
        work.charge("input_binding_bytes", len(blob))
        digest.update(blob)
    return digest.hexdigest()


def _components(points, owned, work):
    forest = _ParityForest(len(owned), work, "ownership")
    claims, bad = {}, set()
    for index, record in enumerate(owned):
        for role, rows in enumerate(record["rows"]):
            for row in rows:
                work.values["ownership_row_visits"] += 1
                _require(work.values["ownership_row_visits"] <= work.values["input_role_rows"],
                         "UNION_OWNERSHIP_ROW_ACCOUNTING")
                key = tuple(float(value) for value in points[row])
                claim = claims.setdefault(key, [set(), set()])
                claim[0 if role == 0 else 1].add(index)
    for arcs, flanges in claims.values():
        if not arcs:
            continue
        work.rows(len(arcs)+len(flanges))
        members = work.ordered(arcs | flanges)
        for index in members[1:]:
            _require(forest.join(members[0], index), "UNION_OWNERSHIP_FOREST")
        if flanges:
            work.rows(len(members))
            bad.update(members)
    components = {}
    for index in range(len(owned)):
        root, _ = forest.find(index)
        components.setdefault(root, []).append(index)
    return list(components.values()), bad


def _existing_cover(members, owned, arc_union, flange_union, work):
    first_cover = None
    for candidate in members:
        cover = owned[candidate]["roles"]
        work.rows(len(cover[0])+2*len(cover[1])+2*len(cover[2]))
        if cover[0] != arc_union or cover[1] | cover[2] != flange_union:
            continue
        permutations = {}
        for index in members:
            work.charge("member_comparisons", 1)
            roles = owned[index]["roles"]
            work.rows(len(roles[0])+2*len(roles[1])+2*len(roles[2]))
            arc_subset = roles[0] <= cover[0]
            choices = [order for order in ((0, 1), (1, 0))
                if arc_subset and roles[1] <= cover[1+order[0]]
                and roles[2] <= cover[1+order[1]]]
            if len(choices) != 1:
                break
            permutations[owned[index]["identity"]] = list(choices[0])
        if len(permutations) == len(members) and first_cover is None:
            first_cover = (candidate, permutations)
    # Members are already in identity order. Every valid existing cover covers
    # every other valid cover, so identity picks equivalent serialization only.
    # Continue checking/charging every candidate, but never retain a quadratic
    # collection of equivalent permutation maps.
    return first_cover


def _aligned_union(members, owned, points, work):
    forest = _ParityForest(len(members), work, "parity")
    initial_links = work.values["parity_links"]
    postings, consistent = {}, True
    for local, index in enumerate(members):
        for side, rows in enumerate(owned[index]["rows"][1:]):
            for row in rows:
                work.values["parity_row_visits"] += 1
                _require(work.values["parity_row_visits"] <= work.values["input_role_rows"],
                         "UNION_PARITY_ROW_ACCOUNTING")
                if row not in postings:
                    postings[row] = (local, side)
                else:
                    first, first_side = postings[row]
                    # Check every equality, including cycles; never stop after
                    # constructing a favorable spanning tree or an early error.
                    equal = forest.join(first, local, first_side ^ side)
                    consistent = equal and consistent
    labels = [forest.find(local) for local in range(len(members))]
    evidence = dict(policy="ALL_SHARED_RAW_ID_EQUALITIES_CHECKED_STAR",
        relative_component_count=len({root for root, _ in labels}),
        constraint_consistent=consistent, member_count=len(members),
        checked_star_links=work.values["parity_links"]-initial_links,
        global_flip_rule="LEXICOGRAPHIC_SORTED_RAW_FLANGE_UNIONS",
        global_flip=None, fallback_reason="NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER")
    if not consistent:
        return None, None, None, evidence, "INCONSISTENT_GLOBAL_FLANGE_PARITY"
    if evidence["relative_component_count"] != 1:
        return None, None, None, evidence, "DISCONNECTED_GLOBAL_FLANGE_PARITY"
    unions, anchors = [set(), set(), set()], None
    for local, index in enumerate(members):
        roles, swap = owned[index]["roles"], labels[local][1]
        aligned = (roles[0], roles[1+swap], roles[2-swap])
        if anchors is None:
            work.rows(sum(map(len, aligned)))
            anchors = [set(rows) for rows in aligned]
        for role, rows in enumerate(aligned):
            work.rows(len(rows)+len(anchors[role]))
            anchors[role].intersection_update(rows)
            work.rows(len(rows))
            unions[role].update(rows)
    evidence["common_anchor_counts"] = [len(rows) for rows in anchors]
    if not all(anchors):
        return None, None, None, evidence, "NO_COMMON_RAW_ROLE_ANCHORS"
    work.rows(sum(map(len, unions))*2)
    if unions[0] & (unions[1] | unions[2]) or unions[1] & unions[2]:
        return None, None, None, evidence, "CROSS_ROLE_POSITION_CONFLICT"
    # New union fallback must not gain evidence by counting duplicate positions.
    # Existing covers retain their original structural semantics and undergo
    # their own unchanged duplicate/geometry rejection in the future wrapper.
    positions = set()
    for rows in unions:
        work.rows(len(rows))
        for row in rows:
            key = tuple(float(value) for value in points[row])
            if key in positions:
                return None, None, None, evidence, "DUPLICATE_POSITION_IN_UNION_SUPPORT"
            positions.add(key)
    ordered = [work.ordered(rows) for rows in unions]
    flip = int(ordered[2] < ordered[1])
    evidence["global_flip"] = flip
    if flip:
        ordered[1], ordered[2] = ordered[2], ordered[1]
        anchors[1], anchors[2] = anchors[2], anchors[1]
    evidence["common_anchor_counts"] = [len(rows) for rows in anchors]
    permutations = {owned[index]["identity"]: [labels[local][1] ^ flip,
                      1 ^ labels[local][1] ^ flip] for local, index in enumerate(members)}
    return ordered, anchors, permutations, evidence, None


def _raw(roles):
    return dict(arc_rows=list(roles[0]), flange_rows=[list(roles[1]), list(roles[2])])


def _bounded(result, cap):
    size = 0
    for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).iterencode(result):
        size += len(fragment.encode("utf-8"))
        _require(size <= cap, "UNION_BUDGET:encoded_plan_bytes")
    return result


def plan_raw_support_unions_v1(points, records, *, budgets=RawSupportUnionBudgetsV1()):
    """Plan every ownership component without fitting, filtering or promotion.

    Required record fields are record_id (unique lowercase SHA256 string),
    raw_support (sorted distinct nonempty arc_rows and two flange_rows lists),
    and the frozen cover helper's seven-field free_circle fit_policy. Other
    fields, including status, associations, nominal values and fitted answers,
    are neither consulted nor attested. A future wrapper must supply ALL valid
    records from its independently validated complete source report.

    COMPLETE means complete structural evaluation, including refused groups.
    Any malformed input or late resource/serialization error makes the whole
    result UNAVAILABLE with no group prefix; attempted counters remain. An
    EXISTING_COVER group never falls back because of a subsequent failed fit.
    UNION_PROPOSED supplies no real cover_record_id or existing context witness.
    """
    result = dict(schema="RAW_SUPPORT_UNION_PLAN_V1", status="UNAVAILABLE", groups=[],
        reasons=[], work={}, configuration=None, production_use_allowed=False,
        measurement_admissible=False, observed_bend_count=None,
        physical_event_identity="NOT_ESTABLISHED",
        policy="EXISTING_COVER_THEN_WHOLE_COMPONENT_RAW_UNION",
        source_binding_scope="CANONICAL_POINTS_AND_STRUCTURAL_RAW_ROLES_POLICY_ONLY",
        downstream_validation_status="NOT_PERFORMED_MANDATORY_BEFORE_MEASUREMENT",
        required_downstream_validations=list(_REQUIRED_DOWNSTREAM),
        counter_semantics="CONSERVATIVE_SET_AND_ORDER_WORK_NOT_LITERAL_CPU",
        memory_semantics="PROSPECTIVE_ACCOUNTED_RESERVATION_NOT_MEASURED_RSS")
    try:
        config = _configuration(budgets)
        result["configuration"] = config
        work = _Work(config)
        result["work"] = work.values
        points, owned = _own(points, records, work)
        structural_sha = _structural_digest(owned, work)
        components, bad = _components(points, owned, work)
        for members in components:
            work.report(2048+256*len(members))
            identities = [owned[index]["identity"] for index in members]
            group = dict(group_id=hashlib.sha256(_encoded(identities)).hexdigest(),
                member_record_ids=identities, status="UNAVAILABLE", proposal_lane=None,
                cover_record_id=None, member_flange_permutations={}, raw_support=None,
                fit_policy=None, common_anchor_raw_support=None, parity_evidence=None,
                measurement_admissible=False, reasons=[])
            result["groups"].append(group)
            if any(index in bad for index in members):
                group["reasons"] = ["CROSS_ROLE_POSITION_CONFLICT"]
                continue
            if len({owned[index]["policy"] for index in members}) != 1:
                group["reasons"] = ["INCOMPATIBLE_MEMBER_FIT_POLICIES"]
                continue
            arc_union, flange_union = set(), set()
            for index in members:
                roles = owned[index]["roles"]
                work.rows(len(roles[0])+2*len(roles[1])+2*len(roles[2]))
                arc_union.update(roles[0])
                flange_union.update(roles[1] | roles[2])
            group["component_raw_union_counts"] = dict(arc_rows=len(arc_union),
                                                       unordered_flange_rows=len(flange_union))
            existing = _existing_cover(members, owned, arc_union, flange_union, work)
            if existing is not None:
                index, permutations = existing
                roles = owned[index]["rows"]
                work.report(8*sum(map(len, roles)))
                group.update(status="EXISTING_COVER", proposal_lane="EXISTING_RAW_SUPPORT_COVER",
                    cover_record_id=owned[index]["identity"], member_flange_permutations=permutations,
                    raw_support=_raw(roles), fit_policy=dict(owned[index]["fit_policy"]),
                    parity_evidence=dict(policy="EXISTING_COVER_UNIQUE_SUBSET_PERMUTATIONS",
                        fallback_evaluated=False, common_anchors_required=False))
                continue
            roles, anchors, permutations, evidence, reason = _aligned_union(members, owned, points, work)
            group["parity_evidence"] = evidence
            if reason is not None:
                group["reasons"] = [reason]
                continue
            work.report(8*(sum(map(len, roles))+sum(map(len, anchors))))
            ordered_anchors = [work.ordered(rows) for rows in anchors]
            group.update(status="UNION_PROPOSED", proposal_lane="WHOLE_COMPONENT_RAW_ROLE_UNION",
                member_flange_permutations=permutations, raw_support=_raw(roles),
                common_anchor_raw_support=_raw(ordered_anchors),
                fit_policy=dict(owned[members[0]]["fit_policy"]))
        result["groups"] = work.ordered(result["groups"], key=lambda row: row["group_id"])
        result.update(status="COMPLETE", input_record_count=len(owned),
            input_record_ids=[row["identity"] for row in owned],
            input_role_row_count=work.values["input_role_rows"],
            component_count=len(result["groups"]),
            raw_points_sha256=hashlib.sha256(memoryview(points)).hexdigest(),
            structural_records_sha256=structural_sha)
        return _bounded(result, config["max_plan_bytes"])
    except Exception as error:
        result.update(status="UNAVAILABLE", groups=[], discarded_plan_prefix=True,
            reasons=["RAW_SUPPORT_UNION_FAILURE:"+type(error).__name__+":"+str(error)[:180]])
        for key in ("input_record_count", "input_record_ids", "input_role_row_count",
                    "component_count", "raw_points_sha256", "structural_records_sha256"):
            result.pop(key, None)
        # Only fixed-schema scalar evidence survives; an invalid requested cap
        # is not a license to return an unbounded exception or partial proposal.
        return result
