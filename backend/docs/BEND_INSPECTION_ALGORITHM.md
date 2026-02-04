# Bend Inspection Algorithm - Technical Specification

## Overview

This document describes the CAD-driven progressive bend inspection algorithm for sheet metal QC. The system enables inspection at any stage of the bending process without requiring the part to be in its final form.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BEND INSPECTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  CAD File    │───▶│  Bend Spec   │───▶│              │                   │
│  │  (STEP/PLY)  │    │  Extractor   │    │              │                   │
│  └──────────────┘    └──────────────┘    │              │                   │
│                                          │    Bend      │    ┌────────────┐ │
│  ┌──────────────┐    ┌──────────────┐    │   Matcher    │───▶│  Report    │ │
│  │  Scan        │───▶│  Bend        │───▶│              │    │  Generator │ │
│  │  Point Cloud │    │  Detector    │    │              │    └────────────┘ │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Mathematical Foundations

### 1. Plane Segmentation (RANSAC)

**Purpose**: Segment the point cloud into planar regions representing flanges.

**Algorithm**: RANSAC (Random Sample Consensus) for robust plane fitting.

**Mathematical Model**:

A plane in 3D is defined by:
```
ax + by + cz + d = 0
```
Where `n = [a, b, c]` is the unit normal vector.

**RANSAC Procedure**:
1. Randomly sample 3 non-collinear points
2. Fit plane: `n = (p2 - p1) × (p3 - p1)`, normalize
3. Compute `d = -n · p1`
4. Count inliers: points where `|ax + by + cz + d| < threshold`
5. Repeat N iterations, keep best plane
6. Refine with least-squares on inliers

**Iteration count** (for 99% confidence with 50% outliers):
```
N = log(1 - 0.99) / log(1 - 0.5³) ≈ 35 iterations
```

**Industry Reference**: Fischler & Bolles (1981), "Random Sample Consensus: A Paradigm for Model Fitting"

### 2. Dihedral Angle Computation

**Purpose**: Calculate the angle between two adjacent planar surfaces (the bend angle).

**Definition**: The dihedral angle θ between two planes is the angle between their normal vectors.

**Formula**:
```
cos(θ) = |n₁ · n₂| / (|n₁| × |n₂|)

For unit normals:
cos(θ) = |n₁ · n₂|
θ = arccos(|n₁ · n₂|)
```

**Note**: We use absolute value because normal direction is ambiguous. The bend angle is:
```
bend_angle = 180° - θ  (for internal angle)
```

**Sign Convention**:
- Positive bend: material bends toward the viewer
- Negative bend: material bends away from the viewer
- Determined by cross product direction relative to bend line

### 3. Bend Line Detection

**Purpose**: Find the line where two planar surfaces meet (the bend axis).

**Method**: Intersection of two planes.

**Mathematical Derivation**:

Given planes:
```
Plane 1: n₁ · (p - p₁) = 0
Plane 2: n₂ · (p - p₂) = 0
```

The bend line direction is:
```
d = n₁ × n₂ (cross product)
```

A point on the line is found by solving:
```
n₁ · p = n₁ · p₁
n₂ · p = n₂ · p₂
```

Using the formula:
```
p₀ = ((n₁ · p₁)(n₂ × d) + (n₂ · p₂)(d × n₁)) / |d|²
```

### 4. Bend Radius Estimation

**Purpose**: Measure the inner radius of the bend.

**Method**: Cylindrical RANSAC on points in the bend transition zone.

**Cylinder Model**:
```
|p - (c + ((p - c) · a)a)|² = r²
```
Where:
- `c` = point on cylinder axis
- `a` = cylinder axis direction (= bend line direction)
- `r` = cylinder radius

**Procedure**:
1. Identify points in the transition zone between planes
2. Project points onto plane perpendicular to bend line
3. Fit circle to projected points using RANSAC
4. Circle radius = bend radius

**K-factor Consideration**:

The neutral axis offset (K-factor) relates inner radius to material behavior:
```
K = t_neutral / t_material
```
Where K typically ranges from 0.3 to 0.5 for sheet metal.

Bend allowance:
```
BA = π × (R + K × T) × (θ / 180)
```

### 5. Plane Adjacency Detection

**Purpose**: Determine which planar segments are adjacent (share a bend).

**Method**: Boundary analysis and proximity testing.

**Algorithm**:
1. Extract boundary points of each planar segment
2. For each pair of planes:
   - Find closest boundary points
   - If distance < threshold AND boundaries are parallel:
     - Planes are adjacent
   - Compute shared edge length

**Shared Edge Criteria**:
```
adjacent = (min_boundary_distance < 2 × bend_radius) AND
           (edge_parallelism > 0.95) AND
           (shared_length > min_edge_length)
```

### 6. Bend Matching Algorithm

**Purpose**: Match detected bends to CAD specifications without global alignment.

**Feature Vector for Each Bend**:
```python
bend_features = {
    'angle': float,           # Dihedral angle in degrees
    'radius': float,          # Bend radius in mm
    'length': float,          # Bend line length in mm
    'flange1_area': float,    # Area of first adjacent plane
    'flange2_area': float,    # Area of second adjacent plane
    'flange1_aspect': float,  # Aspect ratio of first flange
    'flange2_aspect': float,  # Aspect ratio of second flange
}
```

**Matching Cost Function**:
```
cost(detected, cad) = w₁ × |angle_d - angle_c| / angle_tolerance +
                      w₂ × |radius_d - radius_c| / radius_tolerance +
                      w₃ × |length_d - length_c| / length_tolerance +
                      w₄ × flange_similarity_score
```

Where weights: w₁=0.5, w₂=0.2, w₃=0.2, w₄=0.1

**Hungarian Algorithm**: For optimal assignment when multiple bends have similar characteristics.

## Data Structures

### BendSpecification (from CAD)
```python
@dataclass
class BendSpecification:
    bend_id: str
    target_angle: float        # degrees
    target_radius: float       # mm
    bend_line_start: np.ndarray  # 3D point
    bend_line_end: np.ndarray    # 3D point
    tolerance_angle: float     # ± degrees
    tolerance_radius: float    # ± mm
    flange1_normal: np.ndarray
    flange2_normal: np.ndarray
```

### DetectedBend (from scan)
```python
@dataclass
class DetectedBend:
    measured_angle: float
    measured_radius: float
    bend_line: Tuple[np.ndarray, np.ndarray]
    confidence: float
    inlier_count: int
    flange1_points: np.ndarray
    flange2_points: np.ndarray
```

### BendInspectionResult
```python
@dataclass
class BendInspectionResult:
    bend_id: str
    target_angle: float
    measured_angle: float
    angle_deviation: float
    target_radius: float
    measured_radius: float
    radius_deviation: float
    status: str  # "PASS", "FAIL", "WARNING", "NOT_DETECTED"
    confidence: float
```

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ransac_iterations` | 100 | Iterations for plane RANSAC |
| `plane_threshold` | 0.5 mm | Inlier distance for plane fitting |
| `min_plane_points` | 100 | Minimum points to consider a plane |
| `min_plane_area` | 50 mm² | Minimum area for valid flange |
| `adjacency_threshold` | 5 mm | Max gap between adjacent planes |
| `angle_tolerance` | 1.0° | Default bend angle tolerance |
| `radius_tolerance` | 0.5 mm | Default bend radius tolerance |
| `bend_zone_width` | 3× radius | Width of transition zone for radius fitting |

## Industry Standards Reference

1. **ISO 2768-1**: General tolerances for linear and angular dimensions
2. **DIN 6935**: Cold bending of flat steel products
3. **ASME Y14.5**: Dimensioning and tolerancing standard
4. **VDI/VDE 2634**: Optical 3D measuring systems

## Limitations and Edge Cases

1. **Overlapping bends**: When two bends share a common flange, careful segmentation is required
2. **Small flanges**: Flanges smaller than `min_plane_area` may not be detected
3. **Complex bends**: Joggle bends and offset bends require special handling
4. **Springback**: Material springback should be considered in tolerance allocation
5. **Surface quality**: Reflective or dark surfaces may have sparse point coverage

## Future Extensions

1. **Flat pattern tracking**: Map bends to positions in original flat pattern
2. **Bend sequence optimization**: Suggest optimal bend order based on geometry
3. **Machine learning**: Train classifier on historical bend data for improved matching
4. **Real-time feedback**: Streaming analysis during scanning process

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial specification |
