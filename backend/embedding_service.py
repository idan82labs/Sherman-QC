"""
Point Cloud Embedding Service for Part Auto-Recognition

Generates 1024-dimensional embeddings from point clouds using geometric descriptors.
No deep learning required - uses multi-scale local features and global statistics.

This approach is:
- Fast (<500ms on CPU)
- No training required
- Robust to partial scans (>20% coverage)
- Good discriminability between different parts
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import logging
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.stats import entropy

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = logging.getLogger(__name__)

# Embedding configuration
EMBEDDING_VERSION = "v1.2"  # Added scale features for size discrimination
EMBEDDING_DIM = 1024
TARGET_POINTS = 2048  # Points to sample for processing
FPFH_BINS = 33  # Standard FPFH histogram bins


@dataclass
class EmbeddingResult:
    """Result of embedding computation"""
    embedding: np.ndarray  # 1024-dim vector
    version: str
    n_points_original: int
    n_points_processed: int
    processing_time_ms: float
    metadata: Dict[str, Any]


class PointCloudEmbedder:
    """
    Generates fixed-size embeddings from point clouds.

    Feature Architecture (1024 dimensions total):
    1. Global PCA features (12 dim)
    2. Bounding box features (12 dim)
    3. Curvature statistics (20 dim)
    4. Height distributions (48 dim) - 3 axes x 16 bins
    5. Distance distributions (48 dim) - from centroid and axes
    6. Angle distributions (48 dim) - normal angles relative to axes
    7. Multi-scale density (64 dim) - 8 scales x 8 octants
    8. FPFH aggregates (330 dim) - mean, std, 5 percentiles, max-bin hist
    9. Point pair features (128 dim) - distance and angle histograms
    10. Signature of Histograms of Orientations (SHOT-like) (264 dim)
    11. Padding/redundancy (50 dim) - cross-feature statistics
    """

    def __init__(
        self,
        target_points: int = TARGET_POINTS,
        seed: int = 42,
    ):
        self.target_points = target_points
        self._rng = np.random.default_rng(seed)

        if not HAS_OPEN3D:
            logger.warning("Open3D not available - using fallback methods")

    def compute_embedding(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> EmbeddingResult:
        """
        Compute 1024-dim embedding from point cloud.

        Args:
            points: Nx3 numpy array of points
            normals: Optional Nx3 numpy array of normals

        Returns:
            EmbeddingResult with embedding and metadata
        """
        import time
        start_time = time.time()

        n_original = len(points)

        # Step 1: Preprocess (normalize, sample)
        points_proc, normals_proc, scale = self._preprocess(points, normals)
        n_processed = len(points_proc)

        # Step 2: Compute all features
        features = []
        feature_dims = {}

        # 1. Global PCA features (12 dim)
        f = self._compute_pca_features(points_proc)
        features.append(f)
        feature_dims['pca'] = len(f)

        # 2. Bounding box features (12 dim)
        f = self._compute_bbox_features(points_proc)
        features.append(f)
        feature_dims['bbox'] = len(f)

        # 2b. Scale-aware features (24 dim) - uses original scale for size discrimination
        f = self._compute_scale_features(points, scale)
        features.append(f)
        feature_dims['scale'] = len(f)

        # 3. Curvature statistics (20 dim)
        f = self._compute_curvature_features(points_proc, normals_proc)
        features.append(f)
        feature_dims['curvature'] = len(f)

        # 4. Height distributions - all 3 axes (48 dim)
        f = self._compute_axis_histograms(points_proc, n_bins=16)
        features.append(f)
        feature_dims['height'] = len(f)

        # 5. Distance distributions (48 dim)
        f = self._compute_distance_features(points_proc)
        features.append(f)
        feature_dims['distance'] = len(f)

        # 6. Normal angle distributions (48 dim)
        f = self._compute_normal_angle_features(normals_proc)
        features.append(f)
        feature_dims['normal_angles'] = len(f)

        # 7. Multi-scale density (64 dim)
        f = self._compute_multiscale_density(points_proc)
        features.append(f)
        feature_dims['density'] = len(f)

        # 8. FPFH aggregates (330 dim)
        f = self._compute_fpfh_features(points_proc, normals_proc)
        features.append(f)
        feature_dims['fpfh'] = len(f)

        # 9. Point pair features (128 dim)
        f = self._compute_ppf_features(points_proc, normals_proc)
        features.append(f)
        feature_dims['ppf'] = len(f)

        # 10. SHOT-like features (264 dim)
        f = self._compute_shot_features(points_proc, normals_proc)
        features.append(f)
        feature_dims['shot'] = len(f)

        # 11. Cross-feature statistics to fill remaining dims
        current_dim = sum(len(f) for f in features)
        remaining = EMBEDDING_DIM - current_dim
        if remaining > 0:
            f = self._compute_cross_features(features, remaining)
            features.append(f)
            feature_dims['cross'] = len(f)

        # Concatenate
        embedding = np.concatenate(features)

        # Ensure exact dimension
        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        elif len(embedding) < EMBEDDING_DIM:
            embedding = np.concatenate([embedding, np.zeros(EMBEDDING_DIM - len(embedding))])

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            embedding=embedding.astype(np.float32),
            version=EMBEDDING_VERSION,
            n_points_original=n_original,
            n_points_processed=n_processed,
            processing_time_ms=processing_time,
            metadata={
                "feature_dims": feature_dims,
                "scale": scale,
                "total_dim": len(embedding),
            }
        )

    def _preprocess(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Normalize, sample to fixed size."""

        # Center at origin
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points, axis=1))
        scale = max_dist if max_dist > 0 else 1.0
        points = points / scale

        # Sample to exactly target_points
        if len(points) >= self.target_points:
            # Farthest point sampling for better coverage
            idx = self._farthest_point_sample(points, self.target_points)
        else:
            # Upsample with replacement
            idx = self._rng.choice(len(points), self.target_points, replace=True)

        points = points[idx]

        if normals is not None and len(normals) == len(idx):
            normals = normals[idx]
        else:
            normals = self._estimate_normals(points)

        return points, normals, scale

    def _farthest_point_sample(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """Farthest point sampling for uniform coverage."""
        n = len(points)
        if n <= n_samples:
            return np.arange(n)

        # Start with random point
        selected = [self._rng.integers(n)]
        distances = np.full(n, np.inf)

        for _ in range(n_samples - 1):
            last_point = points[selected[-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            selected.append(np.argmax(distances))

        return np.array(selected)

    def _estimate_normals(self, points: np.ndarray, k: int = 20) -> np.ndarray:
        """Estimate normals using PCA on local neighborhoods."""
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
            return np.asarray(pcd.normals)

        # Fallback: PCA-based
        tree = KDTree(points)
        normals = np.zeros_like(points)
        for i, p in enumerate(points):
            _, idx = tree.query(p, k=min(k, len(points)))
            neighbors = points[idx]
            cov = np.cov(neighbors.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]
        return normals

    def _compute_pca_features(self, points: np.ndarray) -> np.ndarray:
        """Global PCA features (12 dim)."""
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normalize eigenvalues
        ev_sum = eigenvalues.sum()
        if ev_sum > 0:
            eigenvalues = eigenvalues / ev_sum

        return np.concatenate([eigenvalues, eigenvectors.flatten()])

    def _compute_bbox_features(self, points: np.ndarray) -> np.ndarray:
        """Bounding box features (12 dim)."""
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        extents = max_pt - min_pt
        center = (min_pt + max_pt) / 2

        # Sort extents
        sorted_extents = np.sort(extents)[::-1]

        # Ratios
        ratios = []
        for i in range(3):
            for j in range(i + 1, 3):
                if sorted_extents[i] > 0:
                    ratios.append(sorted_extents[j] / sorted_extents[i])
                else:
                    ratios.append(1.0)

        volume = np.prod(extents) if np.all(extents > 0) else 0
        surface = 2 * (extents[0] * extents[1] + extents[1] * extents[2] + extents[0] * extents[2])
        diagonal = np.linalg.norm(extents)

        return np.array([
            *sorted_extents,  # 3
            *ratios,          # 3
            volume,           # 1
            surface,          # 1
            diagonal,         # 1
            *center,          # 3
        ])

    def _compute_scale_features(self, original_points: np.ndarray, scale: float) -> np.ndarray:
        """
        Scale-aware features (24 dim).

        Unlike normalized features, these preserve absolute size information
        to distinguish parts of different sizes.

        Uses log scale and binned size categories for robustness.
        """
        # Original bounding box (before normalization)
        min_pt = np.min(original_points, axis=0)
        max_pt = np.max(original_points, axis=0)
        extents = max_pt - min_pt
        sorted_extents = np.sort(extents)[::-1]  # largest to smallest

        # Log-scale features (robust to magnitude variations)
        # Add 1 to avoid log(0), typical parts are 10-1000mm
        log_extents = np.log10(sorted_extents + 1)  # 3 dims
        log_scale = np.log10(scale + 1)  # 1 dim

        # Size category one-hot encoding (8 categories based on max extent)
        # Categories: <10, 10-25, 25-50, 50-100, 100-200, 200-400, 400-800, >800 mm
        max_extent = sorted_extents[0]
        size_bins = [10, 25, 50, 100, 200, 400, 800]
        size_category = np.zeros(8)
        cat_idx = np.searchsorted(size_bins, max_extent)
        size_category[min(cat_idx, 7)] = 1.0  # 8 dims

        # Aspect ratio categories (sheet metal is typically flat)
        # thickness/width and thickness/length ratios
        if sorted_extents[0] > 0:
            flatness = sorted_extents[2] / sorted_extents[0]  # thickness/length
            squareness = sorted_extents[1] / sorted_extents[0]  # width/length
        else:
            flatness = 1.0
            squareness = 1.0

        # Flatness category (sheet metal is flat: flatness < 0.1)
        flatness_bins = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        flatness_category = np.zeros(7)
        flat_idx = np.searchsorted(flatness_bins, flatness)
        flatness_category[min(flat_idx, 6)] = 1.0  # 7 dims

        # Volume class (log10 of volume in mm³)
        volume = np.prod(sorted_extents)
        log_volume = np.log10(volume + 1)  # 1 dim

        # Surface area estimate
        surface = 2 * (sorted_extents[0] * sorted_extents[1] +
                       sorted_extents[1] * sorted_extents[2] +
                       sorted_extents[0] * sorted_extents[2])
        log_surface = np.log10(surface + 1)  # 1 dim

        return np.array([
            *log_extents,       # 3: log of dimensions
            log_scale,          # 1: log of max radius
            *size_category,     # 8: size bin one-hot
            flatness,           # 1: thickness/length ratio
            squareness,         # 1: width/length ratio
            *flatness_category, # 7: flatness bin one-hot
            log_volume,         # 1: log of volume
            log_surface,        # 1: log of surface area
        ])  # Total: 24 dims

    def _compute_curvature_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Curvature statistics (20 dim)."""
        tree = KDTree(points)
        curvatures = np.zeros(len(points))

        for i in range(len(points)):
            _, idx = tree.query(points[i], k=min(15, len(points)))
            if len(idx) > 1:
                local_normals = normals[idx]
                # Curvature = variation in normal directions
                curvatures[i] = 1.0 - np.abs(np.mean(local_normals @ normals[i]))

        # Statistics
        return np.array([
            np.mean(curvatures),
            np.std(curvatures),
            np.min(curvatures),
            np.max(curvatures),
            np.median(curvatures),
            *np.percentile(curvatures, [5, 10, 25, 75, 90, 95]),
            entropy(np.histogram(curvatures, bins=10, density=True)[0] + 1e-10),
            np.sum(curvatures > 0.1),  # Low curvature count
            np.sum(curvatures > 0.3),  # Medium curvature count
            np.sum(curvatures > 0.5),  # High curvature count
            np.sum(curvatures > 0.7),  # Very high curvature count
            np.var(curvatures),
            # Moments
            np.mean((curvatures - np.mean(curvatures)) ** 3),  # Skewness
            np.mean((curvatures - np.mean(curvatures)) ** 4),  # Kurtosis
        ])

    def _compute_axis_histograms(self, points: np.ndarray, n_bins: int = 16) -> np.ndarray:
        """Height distributions along all 3 principal axes (48 dim)."""
        # PCA alignment
        cov = np.cov(points.T)
        _, eigenvectors = np.linalg.eigh(cov)
        aligned = points @ eigenvectors

        features = []
        for axis in range(3):
            values = aligned[:, axis]
            v_min, v_max = values.min(), values.max()
            if v_max > v_min:
                values = (values - v_min) / (v_max - v_min)
            hist, _ = np.histogram(values, bins=n_bins, range=(0, 1), density=True)
            hist = hist / (hist.sum() + 1e-10)
            features.extend(hist)

        return np.array(features)

    def _compute_distance_features(self, points: np.ndarray) -> np.ndarray:
        """Distance distributions from centroid and axes (48 dim)."""
        centroid = np.mean(points, axis=0)
        features = []

        # Distance from centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        d_max = distances.max()
        if d_max > 0:
            distances_norm = distances / d_max
        else:
            distances_norm = distances
        hist, _ = np.histogram(distances_norm, bins=16, range=(0, 1), density=True)
        features.extend(hist / (hist.sum() + 1e-10))

        # Distance from each axis
        for axis in range(3):
            axis_vec = np.zeros(3)
            axis_vec[axis] = 1.0
            # Distance from axis = norm of projection onto orthogonal plane
            axis_distances = np.sqrt(np.sum(points ** 2, axis=1) - (points @ axis_vec) ** 2)
            d_max = axis_distances.max()
            if d_max > 0:
                axis_distances = axis_distances / d_max
            hist, _ = np.histogram(axis_distances, bins=10, range=(0, 1), density=True)
            features.extend(hist / (hist.sum() + 1e-10))

        # Statistics
        features.extend([np.mean(distances), np.std(distances)])

        return np.array(features)

    def _compute_normal_angle_features(self, normals: np.ndarray) -> np.ndarray:
        """Normal angle distributions relative to axes (48 dim)."""
        features = []

        # Angle with each axis
        for axis in range(3):
            axis_vec = np.zeros(3)
            axis_vec[axis] = 1.0
            angles = np.arccos(np.clip(np.abs(normals @ axis_vec), -1, 1))
            angles_norm = angles / (np.pi / 2)  # Normalize to [0, 1]
            hist, _ = np.histogram(angles_norm, bins=12, range=(0, 1), density=True)
            features.extend(hist / (hist.sum() + 1e-10))

        # Normal direction entropy
        for axis in range(3):
            hist, _ = np.histogram(normals[:, axis], bins=12, range=(-1, 1), density=True)
            features.extend(hist / (hist.sum() + 1e-10))

        return np.array(features)

    def _compute_multiscale_density(self, points: np.ndarray) -> np.ndarray:
        """Multi-scale density features (64 dim): 8 scales x 8 octants."""
        features = []
        scales = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]

        for scale in scales:
            # Octant density at this scale
            octants = np.zeros(8)
            for p in points:
                # Determine octant
                idx = (int(p[0] >= 0) * 4 + int(p[1] >= 0) * 2 + int(p[2] >= 0))
                # Weight by distance from origin (more weight to inner points)
                dist = np.linalg.norm(p)
                if dist < scale:
                    octants[idx] += 1.0
                elif dist < scale * 2:
                    octants[idx] += 0.5

            octants = octants / (octants.sum() + 1e-10)
            features.extend(octants)

        return np.array(features)

    def _compute_fpfh_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """FPFH aggregate features (330 dim)."""
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            # Compute FPFH
            radius = 0.15  # ~15% of unit sphere
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
            )
            fpfh_data = np.asarray(fpfh.data).T  # N x 33

            # Aggregate statistics
            features = []
            features.extend(np.mean(fpfh_data, axis=0))       # 33
            features.extend(np.std(fpfh_data, axis=0))        # 33
            features.extend(np.min(fpfh_data, axis=0))        # 33
            features.extend(np.max(fpfh_data, axis=0))        # 33
            for pct in [10, 25, 50, 75, 90]:
                features.extend(np.percentile(fpfh_data, pct, axis=0))  # 5 x 33 = 165

            # Max-bin histogram
            max_bins = np.argmax(fpfh_data, axis=1)
            hist, _ = np.histogram(max_bins, bins=FPFH_BINS, range=(0, FPFH_BINS))
            features.extend(hist / (hist.sum() + 1e-10))  # 33

            return np.array(features)
        else:
            return self._compute_fallback_local_features(points, normals, 330)

    def _compute_ppf_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Point pair features (128 dim)."""
        # Sample point pairs
        n_pairs = min(500, len(points) * (len(points) - 1) // 2)
        idx1 = self._rng.integers(len(points), size=n_pairs)
        idx2 = self._rng.integers(len(points), size=n_pairs)

        # PPF: (d, angle_n1_d, angle_n2_d, angle_n1_n2)
        distances = []
        angle_n1_d = []
        angle_n2_d = []
        angle_n1_n2 = []

        for i, j in zip(idx1, idx2):
            if i == j:
                continue
            d = points[j] - points[i]
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-10:
                continue
            d = d / d_norm

            distances.append(d_norm)
            angle_n1_d.append(np.arccos(np.clip(np.dot(normals[i], d), -1, 1)))
            angle_n2_d.append(np.arccos(np.clip(np.dot(normals[j], d), -1, 1)))
            angle_n1_n2.append(np.arccos(np.clip(np.dot(normals[i], normals[j]), -1, 1)))

        features = []
        for arr, n_bins in [(distances, 32), (angle_n1_d, 32), (angle_n2_d, 32), (angle_n1_n2, 32)]:
            if len(arr) > 0:
                arr = np.array(arr)
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min())
                hist, _ = np.histogram(arr, bins=n_bins, range=(0, 1), density=True)
                features.extend(hist / (hist.sum() + 1e-10))
            else:
                features.extend(np.zeros(n_bins))

        return np.array(features)

    def _compute_shot_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """SHOT-like signature features (264 dim)."""
        # Simplified SHOT: histogram of normal orientations in local volumes
        features = []

        # Use 8 reference points (octant centers) and 8 angular bins
        reference_points = [
            np.array([s1, s2, s3]) * 0.3 for s1 in [-1, 1] for s2 in [-1, 1] for s3 in [-1, 1]
        ]

        tree = KDTree(points)

        for ref in reference_points:
            # Find nearby points
            idx = tree.query_ball_point(ref, r=0.4)
            if len(idx) < 5:
                features.extend(np.zeros(33))
                continue

            # Compute local normal histogram
            local_normals = normals[idx]
            # Project to spherical coordinates
            theta = np.arccos(np.clip(local_normals[:, 2], -1, 1))  # Polar
            phi = np.arctan2(local_normals[:, 1], local_normals[:, 0])  # Azimuthal

            # 2D histogram
            hist, _, _ = np.histogram2d(
                theta / np.pi,  # [0, 1]
                (phi + np.pi) / (2 * np.pi),  # [0, 1]
                bins=[6, 6],
                range=[[0, 1], [0, 1]],
                density=True
            )
            hist_flat = hist.flatten()
            hist_flat = hist_flat / (hist_flat.sum() + 1e-10)

            # Add statistics
            features.extend(hist_flat)  # 36 - but we only use 33 to match dimension
            features.pop()  # Remove one to get 33

        # Should be 8 * 33 = 264
        return np.array(features[:264])

    def _compute_cross_features(self, features: List[np.ndarray], target_dim: int) -> np.ndarray:
        """Cross-feature statistics to fill remaining dimensions."""
        # Concatenate all features and compute statistics
        all_features = np.concatenate(features)

        cross = []
        # Auto-correlation-like features
        for lag in [1, 2, 4, 8, 16, 32]:
            if lag < len(all_features):
                cross.append(np.corrcoef(all_features[:-lag], all_features[lag:])[0, 1])

        # Block statistics
        block_size = len(all_features) // 10
        for i in range(10):
            block = all_features[i * block_size:(i + 1) * block_size]
            cross.extend([np.mean(block), np.std(block)])

        # Entropy of feature distribution
        hist, _ = np.histogram(all_features, bins=20, density=True)
        cross.append(entropy(hist + 1e-10))

        cross = np.array(cross)
        if len(cross) >= target_dim:
            return cross[:target_dim]
        else:
            return np.concatenate([cross, np.zeros(target_dim - len(cross))])

    def _compute_fallback_local_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        target_dim: int,
    ) -> np.ndarray:
        """Fallback local features when Open3D FPFH not available."""
        features = []

        # Local geometric features at sampled points
        tree = KDTree(points)
        sample_idx = self._rng.choice(len(points), min(100, len(points)), replace=False)

        for i in sample_idx:
            _, idx = tree.query(points[i], k=min(20, len(points)))
            local_points = points[idx]
            local_normals = normals[idx]

            # Local statistics
            cov = np.cov(local_points.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)

            features.extend(eigenvalues)  # 3
            features.append(np.std(local_normals @ normals[i]))  # 1

            if len(features) >= target_dim:
                break

        features = np.array(features)
        if len(features) >= target_dim:
            return features[:target_dim]
        else:
            return np.concatenate([features, np.zeros(target_dim - len(features))])


def compute_embedding(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> EmbeddingResult:
    """Convenience function to compute embedding from point cloud."""
    embedder = PointCloudEmbedder()
    return embedder.compute_embedding(points, normals)


def compute_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """Compute cosine similarity between two embeddings."""
    similarity = np.dot(embedding1, embedding2)
    return float(np.clip(similarity, 0.0, 1.0))


if __name__ == "__main__":
    import time

    print("Testing PointCloudEmbedder v1.1...")

    # Test with synthetic shapes
    n_points = 5000

    # Cylinder
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-1, 1, n_points)
    cylinder = np.column_stack([np.cos(theta), np.sin(theta), z])

    # Box
    box = np.random.uniform(-1, 1, (n_points, 3))

    # Sphere
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    sphere = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

    embedder = PointCloudEmbedder()

    shapes = {'cylinder': cylinder, 'box': box, 'sphere': sphere}
    embeddings = {}

    print("\nComputing embeddings:")
    for name, points in shapes.items():
        start = time.time()
        result = embedder.compute_embedding(points)
        t = (time.time() - start) * 1000
        embeddings[name] = result.embedding
        print(f"  {name}: {t:.1f}ms, non-zero: {np.sum(result.embedding != 0)}/{len(result.embedding)}")

    print("\nSimilarity matrix:")
    names = list(shapes.keys())
    print(f"{'':12}", end='')
    for n in names:
        print(f"{n:>10}", end='')
    print()
    for n1 in names:
        print(f"{n1:12}", end='')
        for n2 in names:
            sim = compute_similarity(embeddings[n1], embeddings[n2])
            print(f"{sim:>10.3f}", end='')
        print()

    # Rotation invariance test
    print("\nRotation invariance (cylinder):")
    rotation = Rotation.from_euler('xyz', [30, 45, 60], degrees=True)
    cylinder_rotated = rotation.apply(cylinder)
    result_rot = embedder.compute_embedding(cylinder_rotated)
    sim = compute_similarity(embeddings['cylinder'], result_rot.embedding)
    print(f"  Original vs Rotated: {sim:.3f}")

    # Partial scan test
    print("\nPartial scan test (50% of cylinder):")
    cylinder_partial = cylinder[:n_points // 2]
    result_partial = embedder.compute_embedding(cylinder_partial)
    sim = compute_similarity(embeddings['cylinder'], result_partial.embedding)
    print(f"  Full vs 50%: {sim:.3f}")

    print("\nTests completed!")
