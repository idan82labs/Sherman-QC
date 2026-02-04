# Sherman QC

**AI-Powered Quality Control System for Sheet Metal Manufacturing**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sherman QC is a comprehensive metrology system that performs automated quality control inspection of sheet metal parts by comparing 3D LiDAR scans against CAD reference models. The system implements industry-standard algorithms for point cloud registration, deviation analysis, and bend inspection with sub-millimeter accuracy.

---

## Table of Contents

- [Features](#features)
- [Mathematical Foundations](#mathematical-foundations)
  - [Point Cloud Registration (ICP)](#1-point-cloud-registration-icp)
  - [Deviation Calculation](#2-deviation-calculation)
  - [Bend Detection & Measurement](#3-bend-detection--measurement)
  - [Statistical Quality Metrics](#4-statistical-quality-metrics)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Academic References](#academic-references)
- [License](#license)

---

## Features

- **3D Scan-to-CAD Alignment**: ISO 5459-compliant datum alignment using ICP with robust M-estimators
- **Deviation Heatmaps**: Color-coded visualization of surface deviations with statistical overlays
- **Automatic Bend Detection**: RANSAC-based surface segmentation with dihedral angle computation
- **Bend Angle Measurement**: Sub-degree accuracy using signed distance gradient analysis
- **Dimension Verification**: XLSX specification parsing with tolerance-based pass/fail determination
- **AI-Powered Analysis**: Multi-model pipeline (Claude, Gemini, GPT) for root cause analysis
- **PDF Report Generation**: Comprehensive QC reports with visualizations and recommendations
- **Real-time Part Recognition**: FAISS-based embedding search for automatic part identification
- **Live Scan Monitoring**: File watcher for automatic scan processing

---

## Mathematical Foundations

Sherman QC implements several sophisticated algorithms from computational geometry, robust statistics, and metrology. This section provides the theoretical basis for each major component.

### 1. Point Cloud Registration (ICP)

The system uses a multi-stage Iterative Closest Point (ICP) pipeline for aligning scan data to CAD reference models.

#### 1.1 PCA-Based Pre-Alignment

Initial orientation is resolved using Principal Component Analysis on point cloud centroids:

```
Given point cloud P = {p_1, p_2, ..., p_n} in R^3

1. Compute centroid: mu = (1/n) * sum(p_i)
2. Center points: P_centered = P - mu
3. Covariance matrix: C = (1/n) * P_centered^T * P_centered
4. Eigendecomposition: C = V * Lambda * V^T where Lambda = diag(lambda_1, lambda_2, lambda_3)
5. Rotation matrix: R = V (ensuring det(R) = +1)
```

The eigenvectors define the principal axes, with eigenvalues indicating variance along each axis.

#### 1.2 Point-to-Plane ICP with Tukey M-Estimator

Fine alignment uses point-to-plane ICP with robust outlier rejection:

```
Objective function:
  E(R, t) = sum_i rho((R*p_i + t - q_i) . n_i)

Where:
  - p_i: source point
  - q_i: closest point on target surface
  - n_i: surface normal at q_i
  - rho: Tukey biweight function
```

**Tukey Biweight Loss Function** (Beaton & Tukey, 1974):

```
rho(r) = {
  (k^2/6) * [1 - (1 - (r/k)^2)^3]   if |r| <= k
  k^2/6                              if |r| > k
}

Influence function:
psi(r) = {
  r * (1 - (r/k)^2)^2   if |r| <= k
  0                      if |r| > k
}
```

The system uses k = 2.0, which completely rejects outliers beyond 6mm (3k), matching industrial metrology software (GOM ATOS, PolyWorks).

#### 1.3 RANSAC Datum Alignment (ISO 5459)

For parts with defined datum surfaces, the system implements 3-2-1 datum alignment:

```
Algorithm:
1. Extract planar surfaces using sequential RANSAC
2. Rank surfaces by spatial extent (bounding box area)
3. Select primary datum A (largest stable surface)
4. Select secondary datum B (largest surface with normal angle > 75 deg from A)
5. Constrain alignment to datum surfaces
6. Accept if: median_datum < median_global * 1.2 AND p90_datum < p90_global * 1.5
```

### 2. Deviation Calculation

#### 2.1 Signed Point-to-Surface Distance

For each scan point p, the signed distance to the CAD surface S is computed:

```
d(p, S) = sign(p, S) * min_{q in S} ||p - q||_2

Where sign(p, S) = {
  +1  if p is on the positive (outward) side of S
  -1  if p is on the negative (inward) side of S
}
```

**Implementation**: Uses Bounding Volume Hierarchy (BVH) for O(log n) closest-point queries on triangulated meshes.

#### 2.2 Regional Analysis

The part is segmented into anatomical regions (Top, Bottom, Front, Back, Left, Right, Center) based on spatial position relative to the bounding box. For each region R:

```
Statistics computed:
  - mu_R = (1/|R|) * sum(d_i)              (mean deviation)
  - sigma_R = sqrt[(1/|R|) * sum((d_i - mu_R)^2)]  (standard deviation)
  - d_max = max(|d_i|)                      (maximum absolute deviation)
  - Pass rate = |{d_i : |d_i| <= tau}| / |R|  (tau = tolerance)
```

### 3. Bend Detection & Measurement

#### 3.1 Surface Segmentation via Curvature Analysis

Surface curvature at each point is estimated using the eigenvalue ratio method:

```
For point p with k-nearest neighbors N(p):

1. Center neighbors: N_centered = N(p) - mean(N(p))
2. Covariance: C = cov(N_centered)
3. Eigenvalues: lambda_0 <= lambda_1 <= lambda_2
4. Curvature estimate: kappa = lambda_0 / (lambda_0 + lambda_1 + lambda_2)

Interpretation:
  - kappa ~ 0: Points lie on a plane (flat surface)
  - kappa > threshold: High curvature (bend transition zone)
```

Low-curvature points are clustered using DBSCAN in a 6D feature space combining surface normal orientation and spatial position.

#### 3.2 Dihedral Angle Computation

For adjacent planar surfaces with normals n_1 and n_2:

```
theta = arccos(|n_1 . n_2|)       (angle between normal directions)
bend_angle = 180 deg - theta      (interior bend angle)

Valid bend criteria:
  - 15 deg < bend_angle < 165 deg
  - Surface separation < 30mm (manufacturing constraint)
```

#### 3.3 Scan-Based Bend Measurement (Signed Distance Gradient Method)

When the scan is ICP-aligned to CAD, angular deviations manifest as systematic signed distance gradients:

```
Model: d(x) = slope * x + intercept

Where:
  - d(x): signed distance at perpendicular distance x from bend line
  - slope: rate of deviation change (radians per mm)

Angular deviation: Delta_theta = arctan(slope_1) + arctan(slope_2)
Measured angle: theta_m = theta_CAD + Delta_theta
```

**Robust Regression**: Uses iterative outlier rejection with 85th percentile residual threshold:

```
For i = 1 to 3:
  1. Fit: [slope, intercept] = (A^T * A)^-1 * A^T * y where A = [x, 1]
  2. Residuals: r = |y - (slope * x + intercept)|
  3. Remove points with r > percentile(r, 85)

Standard error: SE_slope = sqrt(MSE / Var(x))
Confidence: "high" if n >= 50 AND SE_angle < 1.0 deg
```

### 4. Statistical Quality Metrics

#### 4.1 Quality Score (0-100)

```
Quality Score = Score_pass_rate + Score_max_dev + Score_issues

Where:
  Score_pass_rate = pass_rate * 60                    (0-60 points)

  Score_max_dev = {
    20  if |d_max|/tau <= 1.0
    10  if |d_max|/tau <= 2.0
    5   if |d_max|/tau <= 3.0
    0   otherwise
  }

  Score_issues = {
    20  if no issues
    15  if 1-2 minor issues
    5   if 3+ issues
    0   if any critical issue
  }
```

#### 4.2 Verdict Determination

```
PASS:    pass_rate >= 99% AND |d_max| <= 1.5*tau    (confidence: 0.95)
WARNING: pass_rate >= 95% AND |d_max| <= 2.0*tau    (confidence: 0.85)
FAIL:    otherwise                                   (confidence: varies)
```

#### 4.3 Material-Specific Springback Thresholds

| Material | Min Springback | Max Springback | Expected Average |
|----------|----------------|----------------|------------------|
| Mild Steel | 0.5 deg | 2.0 deg | 1.25 deg |
| Aluminum | 1.0 deg | 4.0 deg | 2.5 deg |
| Stainless Steel | 2.0 deg | 5.0 deg | 3.5 deg |
| High-Strength Steel | 3.0 deg | 8.0 deg | 5.5 deg |
| Titanium | 4.0 deg | 10.0 deg | 7.0 deg |

---

## Architecture

```
sherman-qc/
|-- backend/                    # Python FastAPI server
|   |-- server.py              # REST API endpoints
|   |-- qc_engine.py           # Core analysis engine (ICP, deviations)
|   |-- bend_detector.py       # Scan-based bend detection
|   |-- cad_bend_extractor.py  # CAD mesh bend extraction
|   |-- bend_matcher.py        # Specification-to-detection matching
|   |-- dimension_parser.py    # XLSX specification parsing
|   |-- ai_analyzer.py         # Multi-model AI analysis
|   |-- pdf_generator.py       # Report generation
|   |-- part_catalog.py        # Part database management
|   |-- embedding_service.py   # FAISS vector search
|   +-- feature_detection/     # Modular feature detectors
|
|-- frontend/react/            # React + TypeScript UI
|   |-- src/components/        # UI components
|   |   |-- ThreeViewer.tsx   # 3D WebGL visualization
|   |   |-- DeviationHeatmap.tsx
|   |   +-- BendAnalysisPanel.tsx
|   |-- src/pages/            # Application pages
|   +-- src/services/         # API client
|
|-- .claude/commands/          # Claude Code skills
|   +-- test-qc.md            # Testing skill (see Testing section)
|
+-- data/                      # SQLite databases
```

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Open3D dependencies (OpenGL)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend/react
npm install
```

### Environment Variables

Create `.env` in the backend directory:

```env
# AI API Keys (optional - for AI analysis features)
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENAI_API_KEY=sk-...

# Server Configuration
HOST=0.0.0.0
PORT=8080
```

---

## Usage

### Start Backend

```bash
cd backend
python server.py
# Server runs on http://localhost:8080
```

### Start Frontend

```bash
cd frontend/react
npm run dev
# UI available at http://localhost:5173
```

### Basic Analysis Workflow

1. **Upload Files**: Navigate to Upload page, select CAD (.stl/.stp) and scan (.ply) files
2. **Configure Tolerances**: Set tolerance in mm (default: +/-0.1mm)
3. **Run Analysis**: System performs ICP alignment, deviation calculation, bend detection
4. **Review Results**: Interactive 3D viewer with deviation heatmap
5. **Export Report**: Download PDF with full analysis and recommendations

---

## Testing

Sherman QC includes a comprehensive testing skill for Claude Code. Use the `/test-qc` command to run tests across all system components.

### Available Test Data

| Part Number | CAD File | Scan File | Test Focus |
|-------------|----------|-----------|------------|
| 44211000_A | STL (594KB) | PLY (299KB) | Simple part, 7 bends |
| 49125000_A00 | STL (7.9MB) | PLY (4MB) | Complex part, 38+ bends |
| 47266000_F | STP + STL | STL | Dimension measurement |
| 11979003_C | STP + STL | STL | Sheet metal bends |

### Test Categories

1. **Bend Detection Test** - Validates bend detection algorithm accuracy
2. **Dimension Measurement Test** - Tests XLSX dimension parsing
3. **API Integration Test** - Tests full REST API workflow
4. **Hole Detection Test** - Tests hole detection in CAD meshes
5. **Full Accuracy Benchmark** - Comprehensive multi-part testing

### Running Tests

```bash
# All tests
cd backend && pytest tests/ -v

# Fast tests only
pytest tests/ -m "not slow" -v

# With coverage
pytest tests/ --cov=backend --cov-report=term-missing

# Specific test files
pytest tests/test_bend_detector.py -v
pytest tests/test_feature_measurement.py -v
```

### Pass/Fail Criteria

| Metric | PASS | WARNING | FAIL |
|--------|------|---------|------|
| Angle deviation | <= tolerance | <= 2x tolerance | > 2x tolerance |
| Match rate | >= 90% | >= 75% | < 75% |
| Dimension deviation | <= tolerance | <= 1.5x tolerance | > 1.5x tolerance |

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Start QC analysis job |
| GET | `/api/jobs/{id}` | Get job status and results |
| GET | `/api/result/{id}` | Get detailed analysis results |
| GET | `/api/deviations/{id}` | Get deviation point cloud |
| GET | `/api/aligned-scan/{id}.ply` | Download aligned scan |

### Bend Inspection Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/bend-inspection/analyze` | Start bend-specific analysis |
| GET | `/api/bend-inspection/{id}` | Get bend inspection results |
| GET | `/api/bend-inspection/{id}/table` | Get ASCII results table |

### Part Catalog Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/parts` | List all parts |
| POST | `/api/parts` | Create new part |
| POST | `/api/parts/{id}/cad` | Upload CAD file |
| GET | `/api/parts/{id}/bend-specs` | Get bend specifications |

---

## Academic References

### Point Cloud Registration

1. **Besl, P.J. and McKay, N.D.** (1992). "A Method for Registration of 3-D Shapes." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 14(2), 239-256. DOI: [10.1109/34.121791](https://doi.org/10.1109/34.121791)

2. **Chen, Y. and Medioni, G.** (1992). "Object Modeling by Registration of Multiple Range Images." *Image and Vision Computing*, 10(3), 145-155.

3. **Rusinkiewicz, S. and Levoy, M.** (2001). "Efficient Variants of the ICP Algorithm." *Proc. 3rd International Conference on 3-D Digital Imaging and Modeling*, 145-152.

4. **Segal, A., Haehnel, D., and Thrun, S.** (2009). "Generalized-ICP." *Robotics: Science and Systems (RSS)*.

### Robust Statistics

5. **Beaton, A.E. and Tukey, J.W.** (1974). "The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data." *Technometrics*, 16(2), 147-185.

6. **Huber, P.J.** (1981). *Robust Statistics*. New York: John Wiley & Sons.

### RANSAC and Shape Detection

7. **Fischler, M.A. and Bolles, R.C.** (1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." *Communications of the ACM*, 24(6), 381-395. DOI: [10.1145/358669.358692](https://doi.org/10.1145/358669.358692)

8. **Schnabel, R., Wahl, R., and Klein, R.** (2007). "Efficient RANSAC for Point-Cloud Shape Detection." *Computer Graphics Forum*, 26(2), 214-226.

### Metrology Standards

9. **ISO 5459:2024** - *Geometrical product specifications (GPS) - Geometrical tolerancing - Datums and datum systems*. International Organization for Standardization.

10. **ASME Y14.5-2018** - *Dimensioning and Tolerancing*. American Society of Mechanical Engineers.

11. **VDI/VDE 2634** - *Optical 3D measuring systems*. Verein Deutscher Ingenieure.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Open3D** - Point cloud processing library
- **Trimesh** - Mesh processing and signed distance computation
- **React Three Fiber** - 3D visualization in React
- **FastAPI** - High-performance Python web framework

Built for Sherman - Tailoring Integrated Solutions
Braude College - Mechanical Engineering Department
Final Project 2026

---

*Sherman QC - Precision at Scale*
