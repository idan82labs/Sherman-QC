"""V7: V5 geometry and exact float64-aware JSON metadata, research only.

The 64 MiB number is unchanged; its encoded representation is changed. This is
not unchanged expanded-JSON admission. Arrays stay immutable in the internal
payload; metadata exports their V5 shape/dtype/digest descriptors, not raw data.
Only exact np.float64 values normalize to builtin float at the encoder boundary;
internal analysis types and array bytes remain unchanged. The decoder is strict.
"""

import json
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from backend import cad_guided_noisy_mesh_planes_v5 as _v5


_FORMAT = "noisy_mesh_metadata_schema_v6"
_ENCODED_LIMIT = 67_108_864
_EXPANDED_LIMIT = 268_435_456
_JSON_ENCODER = json.JSONEncoder(allow_nan=False, ensure_ascii=True, separators=(",", ":"))


def _byte_limit(value):
    _v5._require(type(value) is int and 0 <= value <= 67_108_864, "INVALID_METADATA_BYTE_LIMIT")
    return value


def _json_size(value, maximum):
    """Exact compact JSON UTF-8 size, without a complete serialized string."""
    size = 0
    for fragment in _JSON_ENCODER.iterencode(value):
        size += len(fragment.encode("utf-8"))
        _v5._require(size <= maximum, "ENCODED_METADATA_BYTE_LIMIT_EXCEEDED")
    return size


class _Work:
    """One extraction ledger, shared across admission and all final passes."""

    def __init__(self):
        self.visits = 0
        self.hash_bytes = 0

    def visit(self, depth, count=1):
        _v5._require(depth <= 64, "METADATA_DEPTH_LIMIT_EXCEEDED")
        _v5._require(self.visits+count <= 64_000_000, "METADATA_NODE_WORK_LIMIT_EXCEEDED")
        self.visits += count

    def array(self, value):
        _v5._require(value.dtype.kind in "biuf" and value.flags.c_contiguous,
                     "UNSUPPORTED_METADATA_ARRAY")
        _v5._require(self.hash_bytes+value.nbytes <= 4*1024**3, "METADATA_ARRAY_HASH_WORK_LIMIT_EXCEEDED")
        self.hash_bytes += int(value.nbytes)


class _Expanded:
    def __init__(self):
        self.size = 0

    def add(self, size):
        _v5._require(self.size+size <= 268_435_456, "EXPANDED_METADATA_BYTE_LIMIT_EXCEEDED")
        self.size += size


def _scalar_size(value):
    _v5._require(value is None or type(value) in (bool, int, float, str), "UNSUPPORTED_METADATA_VALUE")
    if type(value) is float:
        _v5._require(math.isfinite(value), "NONFINITE_METADATA_VALUE")
    if type(value) is int:
        _v5._require(value.bit_length() <= 4096, "METADATA_INTEGER_LIMIT_EXCEEDED")
    if type(value) is str:
        _v5._require(len(value) <= 65_536, "METADATA_STRING_LIMIT_EXCEEDED")
    # Strings and integers are bounded before their complete scalar encoding;
    # avoid a fresh Python encoder/generator for each repeated number/key.
    return len(_JSON_ENCODER.encode(value).encode("utf-8"))


def _keys(keys, work, depth):
    _v5._require(len(keys) <= 128, "METADATA_SCHEMA_KEY_LIMIT_EXCEEDED")
    work.visit(depth, len(keys))
    _v5._require(all(type(key) is str and len(key) <= 128 for key in keys), "INVALID_METADATA_SCHEMA_KEY")
    _v5._require(len(set(keys)) == len(keys), "DUPLICATE_METADATA_SCHEMA_KEY")


class _Codec:
    def __init__(self, work):
        self.work = work
        self.schemas = []
        self.schema_ids = {}
        self.expanded = _Expanded()

    def encode(self, value, depth=0):
        self.work.visit(depth)
        if type(value) is np.float64:
            # V5 JSON already writes this exact scalar as a binary64 number.
            # Normalize only the export-local value, with no extra node visit.
            value = float(value)
        if type(value) is np.ndarray:
            self.work.array(value)
            return self.encode(_v5._metadata(value), depth)
        if type(value) is dict:
            _v5._require(len(value) <= 128, "METADATA_SCHEMA_KEY_LIMIT_EXCEEDED")
            items = tuple(value.items())
            keys = tuple(key for key, _ in items)
            _keys(keys, self.work, depth)
            schema_id = self.schema_ids.get(keys)
            if schema_id is None:
                _v5._require(len(self.schemas) < 256, "METADATA_SCHEMA_COUNT_LIMIT_EXCEEDED")
                schema_id = len(self.schemas)
                self.schema_ids[keys] = schema_id
                self.schemas.append(list(keys))
            self.expanded.add(2+max(0, len(items)-1)+len(items))  # braces, commas, colons
            for key in keys:
                self.expanded.add(_scalar_size(key))
            return ["D", schema_id, [self.encode(item, depth+1) for _, item in items]]
        if type(value) in (tuple, list):
            self.expanded.add(2+max(0, len(value)-1))
            return ["L", [self.encode(item, depth+1) for item in value]]
        self.expanded.add(_scalar_size(value))
        return value


def _envelope(value, work, max_bytes):
    codec = _Codec(work)
    root = codec.encode(value)
    envelope = dict(format=_FORMAT, schemas=codec.schemas, root=root)
    size = _json_size(envelope, max_bytes)
    return envelope, size, codec.expanded.size


def _metadata(value, *, max_bytes=_ENCODED_LIMIT):
    """Fresh deterministic bounded export; no admission-history schema state."""
    return _envelope(value, _Work(), _byte_limit(max_bytes))[0]


def encode_metadata_v7(value, *, max_bytes=_ENCODED_LIMIT):
    return _metadata(value, max_bytes=max_bytes)


def decode_metadata_v7(envelope, *, max_bytes=_ENCODED_LIMIT):
    """Recover the V5 JSON-wire metadata tree, NOT the original arrays.

    Reject malformed/nonfinite/over-budget input rather than return a prefix.
    Tuple/list distinction is absent in the V5 JSON metadata tree by design.
    """
    limit = _byte_limit(max_bytes)
    _v5._require(type(envelope) is dict and len(envelope) == 3
                 and all(type(key) is str for key in envelope)
                 and set(envelope) == {"format", "schemas", "root"}
                 and type(envelope["format"]) is str and envelope["format"] == _FORMAT,
                 "INVALID_METADATA_ENVELOPE")
    schemas = envelope["schemas"]
    _v5._require(type(schemas) is list and len(schemas) <= 256, "INVALID_METADATA_SCHEMA_TABLE")
    work, expanded = _Work(), _Expanded()
    admitted, seen = [], set()
    for row in schemas:
        _v5._require(type(row) is list, "INVALID_METADATA_SCHEMA")
        _keys(row, work, 0)
        keys = tuple(row)
        _v5._require(keys not in seen, "DUPLICATE_METADATA_SCHEMA")
        seen.add(keys)
        admitted.append(keys)
    used = set()

    def decode(node, depth=0):
        work.visit(depth)
        if type(node) is not list:
            expanded.add(_scalar_size(node))
            return node
        _v5._require(len(node) in (2, 3) and type(node[0]) is str, "INVALID_METADATA_NODE")
        if node[0] == "L":
            _v5._require(len(node) == 2 and type(node[1]) is list, "INVALID_METADATA_LIST_NODE")
            values = node[1]
            expanded.add(2+max(0, len(values)-1))
            return [decode(item, depth+1) for item in values]
        _v5._require(node[0] == "D" and len(node) == 3 and type(node[1]) is int
                     and 0 <= node[1] < len(admitted) and type(node[2]) is list,
                     "INVALID_METADATA_DICTIONARY_NODE")
        keys, values = admitted[node[1]], node[2]
        _v5._require(len(values) == len(keys), "METADATA_SCHEMA_ARITY_MISMATCH")
        work.visit(depth, len(keys))
        used.add(node[1])
        expanded.add(2+max(0, len(keys)-1)+len(keys))
        for key in keys:
            expanded.add(_scalar_size(key))
        return {key: decode(item, depth+1) for key, item in zip(keys, values)}

    result = decode(envelope["root"])
    _v5._require(used == set(range(len(admitted))), "UNUSED_METADATA_SCHEMA")
    _json_size(envelope, limit)
    return result


def _append(target, record, stats, cap, codec):
    """One interner for every incremental append; charge before publication."""
    first_schema, expanded_before = len(codec.schemas), codec.expanded.size
    node = codec.encode(record)
    size = _json_size(node, cap["max_metadata_bytes"])
    for schema in codec.schemas[first_schema:]:
        size += _json_size(schema, cap["max_metadata_bytes"])
    _v5._charge(stats, cap, "metadata_bytes", size, "max_metadata_bytes")
    stats["metadata_expanded_bytes"] += codec.expanded.size-expanded_before
    stats["metadata_node_visits"] = codec.work.visits
    stats["metadata_array_bytes_hashed"] = codec.work.hash_bytes
    target.append(record)


def _finalize(result, cap, work):
    """Exactly three passes; predict stable traversal work, verify actual totals."""
    stats = result["statistics"]
    before_visits, before_hash = work.visits, work.hash_bytes
    envelope, size, expanded = _envelope(result, work, cap["max_metadata_bytes"])
    del envelope  # Do not retain the prior encoded tree during the next pass.
    pass_visits, pass_hash = work.visits-before_visits, work.hash_bytes-before_hash
    stats.update(metadata_node_visits=before_visits+3*pass_visits,
                 metadata_array_bytes_hashed=before_hash+3*pass_hash)
    for _ in range(2):
        stats.update(metadata_bytes=size, metadata_expanded_bytes=expanded)
        envelope, size, expanded = _envelope(result, work, cap["max_metadata_bytes"])
        del envelope
    _v5._require(stats["metadata_bytes"] == size and stats["metadata_expanded_bytes"] == expanded
                 and stats["metadata_node_visits"] == work.visits
                 and stats["metadata_array_bytes_hashed"] == work.hash_bytes,
                 "METADATA_FINAL_ACCOUNTING_UNRESOLVED")


def extract_noisy_scan_plane_proposals_v7(points, triangle_raw_rows, *, position_error_bounds_mm,
        policy=_v5.NoisyMeshPlanePolicyV5(), budgets=_v5.NoisyMeshPlaneBudgetsV5()):
    """V5 computation and internal evidence records with schema-table admission."""
    result = _v5._base()
    result.update(schema_version="noisy_mesh_planes_v7", metadata_encoding=dict(
        format=_FORMAT, logical_tree="V5_JSON_WIRE_METADATA_WITH_ARRAY_DESCRIPTORS",
        encoded_limit_bytes=_ENCODED_LIMIT, expanded_limit_bytes=_EXPANDED_LIMIT,
        final_encoding_passes=3, changed_encoded_representation=True))
    stats, work = result["statistics"], _Work()
    stats.update(metadata_expanded_bytes=0, metadata_node_visits=0, metadata_array_bytes_hashed=0)
    codec = _Codec(work)
    try:
        gate, cap = _v5._configuration(policy, budgets)
        result["metadata_encoding"]["encoded_limit_bytes"] = cap["max_metadata_bytes"]
        result.update(policy=gate, budgets=cap, unit="mm", coordinate_frame="RAW_SCAN_UNREGISTERED",
            unit_assumption="PROVISIONAL_ARRAY_MM_UNVERIFIED",
            closure_schedule_policy="ALTERNATING_TILE_SCALE_NORMAL_NOVELTY_THEN_CENTROID_SPREAD",
            tiling_tie_policy="ORIGINAL_TRIANGLE_ROW_ORDER_NOT_ROW_PERMUTATION_INVARIANT")
        p, t, epsilon = _v5._inputs(points, triangle_raw_rows, position_error_bounds_mm, cap)
        stats.update(input_raw_rows=len(p), input_triangles=len(t))
        result["input_digests"] = dict(points_sha256=_v5._digest(p), triangle_raw_rows_sha256=_v5._digest(t),
                                        position_error_bounds_mm_sha256=_v5._digest(epsilon))
        graph, left, right = _v5._adjacency(t, len(p), stats)
        referenced = np.zeros(len(p), dtype=bool)
        referenced[t.reshape(-1)] = True
        unreferenced = _v5._retained(np.flatnonzero(~referenced), stats, cap)
        result["raw_input_accounting"] = dict(referenced_raw_row_count=int(np.count_nonzero(referenced)),
            unreferenced_raw_vertex_rows=unreferenced,
            unreferenced_raw_vertex_rows_sha256=_v5._digest(unreferenced),
            unreferenced_disposition="INPUT_RETAINED_NOT_TRIANGLE_SUPPORT")
        for size in gate["tile_triangles"]:
            labels, blocks = _v5._tile_partition(graph, size, cap["max_core_records"]-len(result["cores"]))
            _v5._require(len(result["cores"])+len(blocks) <= cap["max_core_records"], "CORE_RECORD_BUDGET_EXCEEDED")
            first_core_id = len(result["cores"])
            label_rows = _v5._retained(labels+first_core_id, stats, cap)
            _append(result["partitions"], dict(tile_triangles=size, triangle_core_labels=label_rows,
                triangle_core_labels_sha256=_v5._digest(label_rows)), stats, cap, codec)
            for base in blocks:
                neighbors = [graph.indices[graph.indptr[row]:graph.indptr[row+1]] for row in base]
                rows = np.unique(np.concatenate([base, *neighbors]))
                _v5._require(len(rows) <= cap["max_core_triangle_rows"], "CORE_TRIANGLE_ROW_BUDGET_EXCEEDED")
                _v5._charge(stats, cap, "triangle_memberships", len(base)+len(rows), "max_triangle_memberships")
                base_rows, rows = _v5._retained(base, stats, cap), _v5._retained(rows, stats, cap)
                raw = _v5._support_rows(t, rows, stats, cap)
                analysis = _v5._fit(p, epsilon, raw, gate, cap, stats)
                core_id = len(result["cores"])
                proposed = analysis["status"] == "PROPOSED"
                record = dict(core_id=core_id, tile_triangles=size, base_triangle_rows=base_rows,
                    triangle_rows=rows, raw_vertex_rows=raw, triangle_rows_sha256=_v5._digest(rows),
                    raw_vertex_rows_sha256=_v5._digest(raw), analysis=analysis,
                    closure_disposition="NOT_SCHEDULED_BY_POLICY" if proposed else "CORE_NOT_QUALIFIED",
                    closure_id=None)
                _append(result["cores"], record, stats, cap, codec)
                stats["cores_evaluated"] += 1
                stats["qualified_core_count"] += int(proposed)
                if proposed:
                    _append(result["proposals"], dict(kind="CORE", record_id=core_id), stats, cap, codec)
        stats.update(core_count=len(result["cores"]), core_construction_complete=True, core_evaluation_complete=True)
        schedule = _v5._schedule(result["cores"], cap["max_closure_schedules"])
        result["scheduled_core_ids"] = schedule
        stats["closures_scheduled"] = len(schedule)
        stats["unscheduled_qualified_core_count"] = stats["qualified_core_count"]-len(schedule)
        seen = {}
        for core_id in schedule:
            core = result["cores"][core_id]
            allowed, cone = _v5._cone_mask(p, epsilon, core["analysis"], cap, stats)
            _v5._charge(stats, cap, "closure_triangle_tests", len(t), "max_closure_triangle_tests")
            eligible = np.all(allowed[t], axis=1)
            _v5._require(np.all(eligible[core["triangle_rows"]]), "NUMERICAL_SEED_CONE_SELF_EXCLUSION")
            _v5._charge(stats, cap, "closure_edge_tests", len(left), "max_closure_edge_tests")
            edge = eligible[left] & eligible[right]
            closure_graph = coo_matrix((np.ones(int(np.count_nonzero(edge)), dtype=np.uint8),
                (left[edge], right[edge])), shape=(len(t), len(t))).tocsr()
            _, labels = connected_components(closure_graph, directed=False, return_labels=True)
            group = labels[int(core["triangle_rows"][0])]
            _v5._require(np.all(labels[core["triangle_rows"]] == group), "CORE_TOPOLOGY_INCONSISTENT")
            rows = np.flatnonzero((labels == group) & eligible)
            key = _v5._digest(np.asarray(rows, dtype="<i8"))
            duplicate = next((index for index in seen.get(key, [])
                              if np.array_equal(result["closures"][index]["triangle_rows"], rows)), None)
            if duplicate is not None:
                closure = result["closures"][duplicate]
                closure["source_core_ids"].append(core_id)
                core.update(closure_disposition="EXACT_SUPPORT_DUPLICATE", closure_id=duplicate)
            else:
                _v5._charge(stats, cap, "triangle_memberships", len(rows), "max_triangle_memberships")
                rows = _v5._retained(rows, stats, cap)
                raw = _v5._support_rows(t, rows, stats, cap)
                analysis = _v5._fit(p, epsilon, raw, gate, cap, stats)
                closure_id = len(result["closures"])
                record = dict(closure_id=closure_id, source_core_ids=[core_id], triangle_rows=rows,
                    raw_vertex_rows=raw, triangle_rows_sha256=_v5._digest(rows), raw_vertex_rows_sha256=_v5._digest(raw),
                    analysis=analysis)
                _append(result["closures"], record, stats, cap, codec)
                seen.setdefault(key, []).append(closure_id)
                proposed = analysis["status"] == "PROPOSED"
                core.update(closure_disposition="PROPOSED" if proposed else "WHOLE_CLOSURE_REJECTED_CORE_RETAINED",
                            closure_id=closure_id)
                stats["qualified_closure_count"] += int(proposed)
                if proposed:
                    _append(result["proposals"], dict(kind="CLOSURE", record_id=closure_id), stats, cap, codec)
            core["cone_proposal"] = cone
            stats["closures_evaluated"] += 1
        stats.update(unique_closure_count=len(result["closures"]), scheduled_closure_evaluation_complete=True,
                     all_qualified_core_closures_evaluated=stats["unscheduled_qualified_core_count"] == 0)
        result.update(status="COMPLETE", reasons=[] if result["proposals"] else ["NO_PROPOSAL_WITHIN_DECLARED_POLICY"])
        _finalize(result, cap, work)
        return result
    except Exception as error:
        stats.update(metadata_node_visits=work.visits, metadata_array_bytes_hashed=work.hash_bytes)
        return _v5._failure(result, error)
