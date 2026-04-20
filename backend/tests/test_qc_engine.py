import numpy as np

from backend.qc_engine import ScanQCEngine


class _Utility:
    @staticmethod
    def Vector3dVector(values):
        return np.asarray(values, dtype=np.float64)


class _O3D:
    utility = _Utility()


class _PointCloud:
    def __init__(self, points, normals=None, colors=None):
        self.points = np.asarray(points, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64) if normals is not None else np.empty((0, 3))
        self.colors = np.asarray(colors, dtype=np.float64) if colors is not None else np.empty((0, 3))

    def has_normals(self):
        return len(self.normals) == len(self.points)

    def has_colors(self):
        return len(self.colors) == len(self.points)


def test_sort_point_cloud_canonicalizes_points_normals_and_colors():
    engine = ScanQCEngine.__new__(ScanQCEngine)
    engine._o3d = _O3D()

    cloud = _PointCloud(
        points=[[2, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0]],
        normals=[[20, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 0]],
        colors=[[0.2, 0, 0], [0.11, 0, 0], [0.1, 0.1, 0], [0.1, 0, 0]],
    )

    ScanQCEngine._sort_point_cloud(engine, cloud)

    assert cloud.points.tolist() == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    assert cloud.normals.tolist() == [
        [10.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [11.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ]
    assert cloud.colors.tolist() == [
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.11, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ]
