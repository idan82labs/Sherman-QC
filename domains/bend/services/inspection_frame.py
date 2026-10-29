"""Explicit geometry-frame contract shared by regular inspection and rendering.

AVAILABLE means coordinates can be related, not qualified metrology.
"""
import copy
import numpy as np


def validate_alignment(alignment):
    if not isinstance(alignment, dict):
        raise ValueError('Shared scan alignment is required')
    if (alignment.get('status') != 'AVAILABLE'
            or alignment.get('geometry_frame') != 'reference_mesh_raw'
            or alignment.get('alignment_abstained') is True):
        raise ValueError('Shared scan alignment is unavailable')
    matrix = np.asarray(alignment.get('raw_scan_to_reference_transform'), dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or np.linalg.det(matrix[:3, :3]) <= 1e-12):
        raise ValueError('Invalid raw scan to reference transform')
    result = copy.deepcopy(alignment)
    result['raw_scan_to_reference_transform'] = matrix.tolist()
    return result


def validate_aligned_points(points, alignment):
    validate_alignment(alignment)
    array = np.asarray(points, dtype=float)
    if (array.ndim != 2 or array.shape[1:] != (3,) or not len(array)
            or not np.isfinite(array).all()):
        raise ValueError('Invalid shared aligned scan points')
    return array
