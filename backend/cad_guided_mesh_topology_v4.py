"""Additive byte-bound triangle topology for offline scan plane proposals.

V1 validates every byte/scalar/index and supplies unchanged raw vertex positions.
This adapter additionally preserves triangle vertex-row identities from those
SAME immutable bytes. Nontriangular polygons are not silently triangulated.
Observed mesh adjacency is not verified material topology or scanner calibration.
No files, native runtime, application state or physical inputs are read here.
"""
import hashlib
import io

import numpy as np

from . import cad_guided_mesh_input_v1 as v1


def parse_triangle_mesh_ply_v4(blob):
    """Return immutable raw XYZ/triangle row indices, or withhold both entirely.

    The body is first completely validated by unchanged V1. The second bounded
    pass extracts only its already-validated face lists; source hashing refers
    to the original bytes, while array hashes bind canonical little-endian data.
    No row is removed/reordered, and no face is repaired, rewound or deduplicated.
    """
    base = dict(status="UNAVAILABLE", reasons=[], points=None, triangle_raw_rows=None,
                source_sha256=None, vertex_count=None, triangle_count=None,
                points_sha256=None, triangle_raw_rows_sha256=None,
                raw_row_order="ORIGINAL_PLY_VERTEX_ROWS",
                triangle_row_order="ORIGINAL_PLY_FACE_ROWS",
                topology_status="OBSERVED_MESH_CONNECTIVITY_NOT_MATERIAL_CERTIFICATION",
                units_status="UNVERIFIED", coordinate_frame_status="UNVERIFIED",
                production_use_allowed=False)
    parsed = v1.parse_mesh_ply_v1(blob)
    base["source_sha256"] = parsed["source_sha256"]
    if parsed["status"] != "AVAILABLE":
        return {**base, "reasons": list(parsed["reasons"])}
    try:
        facts = parsed["header"]
        vertex, face = v1._admit_layout(facts)
        v1._require(face is not None and face["count"] > 0, "TRIANGLE_MESH_REQUIRED")
        v1._require(parsed["face_index_count"] == 3 * face["count"],
                    "NONTRIANGULAR_MESH_UNSUPPORTED")
        offset = facts["header_bytes"]
        if facts["format"] == "ascii":
            stream = io.BytesIO(blob)
            stream.seek(offset)
            # V1 already checked exact count/type/range for every scalar row.
            for _ in range(vertex["count"]):
                v1._ascii_row(stream, len(vertex["properties"]))
            indices = np.empty((face["count"], 3), dtype="<i8")
            prop = face["properties"][0]
            for row in range(face["count"]):
                tokens = v1._ascii_row(stream, 4)
                v1._require(len(tokens) == 4 and v1._scalar(
                    tokens[0], v1.PLY_TYPES[prop["count_type"]]) == 3,
                    "NONTRIANGULAR_MESH_UNSUPPORTED")
                indices[row] = [v1._scalar(token, v1.PLY_TYPES[prop["item_type"]])
                                for token in tokens[1:]]
            v1._require(not stream.read(1), "PLY_TRAILING_BYTES")
        else:
            endian = "<" if facts["format"] == "binary_little_endian" else ">"
            vertex_dtype = np.dtype([(row["name"], endian + v1.PLY_TYPES[row["type"]])
                                     for row in vertex["properties"]])
            offset += vertex["count"] * vertex_dtype.itemsize
            prop = face["properties"][0]
            triangle_dtype = np.dtype([
                ("count", endian + v1.PLY_TYPES[prop["count_type"]]),
                ("indices", endian + v1.PLY_TYPES[prop["item_type"]], (3,))])
            v1._require(len(blob)-offset == face["count"] * triangle_dtype.itemsize,
                        "NONTRIANGULAR_MESH_UNSUPPORTED")
            records = np.frombuffer(blob, dtype=triangle_dtype,
                                    count=face["count"], offset=offset)
            v1._require(bool(np.all(records["count"] == 3)), "NONTRIANGULAR_MESH_UNSUPPORTED")
            indices = records["indices"]
        # Recheck the extraction representation, even though V1 verified these
        # exact bytes. The hard V1 vertex cap makes signed int64 lossless here.
        v1._require(indices.shape == (face["count"], 3)
                    and bool(np.all(indices >= 0))
                    and bool(np.all(indices < vertex["count"])), "PLY_FACE_VERTEX_INDEX")
        v1._require(bool(np.all(indices[:, 0] != indices[:, 1])
                         and np.all(indices[:, 0] != indices[:, 2])
                         and np.all(indices[:, 1] != indices[:, 2])),
                    "PLY_FACE_DUPLICATE_VERTEX_INDEX")
        triangles = v1._immutable_array(indices, "<i8")
        points = parsed["points"]
        return {**base, "status": "AVAILABLE", "reasons": [], "points": points,
                "triangle_raw_rows": triangles, "vertex_count": len(points),
                "triangle_count": len(triangles),
                "points_sha256": hashlib.sha256(points.tobytes(order="C")).hexdigest(),
                "triangle_raw_rows_sha256": hashlib.sha256(triangles.tobytes(order="C")).hexdigest()}
    except v1.MeshInputUnavailable as error:
        return {**base, "reasons": [str(error)]}
