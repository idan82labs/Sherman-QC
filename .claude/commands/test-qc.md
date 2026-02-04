---
description: Run comprehensive QC tests with real files to measure accuracy
---

# Sherman QC Testing Skill

This skill runs tests across all areas of the Sherman QC system using real CAD and scan files.

## Quick Reference

```bash
# Backend must be running
cd "/Users/idant/82Labs/Sherman QC/Full Project/backend" && python3 server.py &

# Base URL
API_BASE="http://localhost:8080/api"
```

## Available Test Data

### Complete CAD/Scan Pairs (Best for Testing)

| Part Number | CAD File | Scan File | Dimension Spec | Test Focus |
|-------------|----------|-----------|----------------|------------|
| **44211000_A** | Downloads/44211000_A.stl (594KB) | Downloads/44211000_A.ply (299KB) | Downloads/44211000_A מידות.xlsx | Simple part, 7 bends |
| **49125000_A00** | Downloads/49125000_A00.stl (7.9MB) | Downloads/49125000_A00.ply (4MB) | None | Complex part, 38+ bends |
| **47266000_F** | test_files/47266000_F/cad.stp | test_files/47266000_F/scan.stl | test_files/47266000_F/dimensions.xlsx | Dimension measurement |
| **11979003_C** | test_files/11979003_C/cad.stp | test_files/11979003_C/scan.stl | None | Sheet metal bends |

### File Type Classification
- **< 5MB** = Likely CAD (clean mesh)
- **> 5MB** = Likely Scan (dense point cloud)
- **.stl/.stp** = CAD formats
- **.ply** = LiDAR scan format

---

## Test Categories

### 1. Bend Detection Test

Tests the bend detection algorithm accuracy.

```python
import requests
import open3d as o3d
import numpy as np
import sys
sys.path.insert(0, "/Users/idant/82Labs/Sherman QC/Full Project/backend")
from feature_detection import BendDetector, ProgressiveBendMatcher, BendSpecification

def test_bend_detection(cad_path: str, scan_path: str, part_name: str):
    """
    Test bend detection accuracy between CAD and scan.

    Expected: Same number of bends detected, angle deviations < 3°
    """
    print(f"\n{'='*60}")
    print(f"BEND DETECTION TEST: {part_name}")
    print(f"{'='*60}")

    # Load files
    if cad_path.endswith('.stl'):
        cad_mesh = o3d.io.read_triangle_mesh(cad_path)
        cad_pcd = cad_mesh.sample_points_uniformly(15000)
    else:
        cad_pcd = o3d.io.read_point_cloud(cad_path)

    scan_pcd = o3d.io.read_point_cloud(scan_path) if scan_path.endswith('.ply') else \
               o3d.io.read_triangle_mesh(scan_path).sample_points_uniformly(15000)

    cad_points = np.asarray(cad_pcd.points)
    scan_points = np.asarray(scan_pcd.points)

    # Downsample if needed
    if len(scan_points) > 20000:
        idx = np.random.choice(len(scan_points), 20000, replace=False)
        scan_points = scan_points[idx]

    print(f"CAD points:  {len(cad_points):,}")
    print(f"Scan points: {len(scan_points):,}")

    # Detect bends with improved algorithm
    detector = BendDetector(
        adaptive_threshold=True,
        ransac_seed=42,  # Reproducible
        min_plane_points=80,
        min_plane_area=40.0,
    )

    cad_bends = detector.detect_bends(cad_points, preprocess=True)
    scan_bends = detector.detect_bends(scan_points, preprocess=True)

    print(f"\nCAD bends:  {len(cad_bends)}")
    print(f"Scan bends: {len(scan_bends)}")

    # Create specifications from CAD
    cad_specs = [
        BendSpecification(
            bend_id=f"B{i+1}",
            target_angle=b.measured_angle,
            target_radius=b.measured_radius,
            bend_line_start=b.bend_line_start,
            bend_line_end=b.bend_line_end,
            tolerance_angle=3.0,
            tolerance_radius=3.0,
        )
        for i, b in enumerate(cad_bends)
    ]

    # Match and compare
    matcher = ProgressiveBendMatcher(max_angle_diff=15.0)
    report = matcher.match(scan_bends, cad_specs, part_id=part_name)

    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total CAD bends:  {report.total_cad_bends}")
    print(f"Detected in scan: {report.detected_count}")
    print(f"Match rate:       {100*report.detected_count/max(1,report.total_cad_bends):.1f}%")
    print(f"PASS:             {report.pass_count}")
    print(f"WARNINGS:         {report.warning_count}")
    print(f"FAIL:             {report.fail_count}")

    # Deviation stats
    devs = [abs(m.angle_deviation) for m in report.matches if m.detected_bend]
    if devs:
        print(f"\nAngle Deviations:")
        print(f"  Average: {np.mean(devs):.2f}°")
        print(f"  Max:     {np.max(devs):.2f}°")

    # Accuracy score
    accuracy = 100 * report.pass_count / max(1, report.total_cad_bends)
    print(f"\n*** ACCURACY SCORE: {accuracy:.1f}% ***")

    return {
        'part': part_name,
        'cad_bends': len(cad_bends),
        'scan_bends': len(scan_bends),
        'match_rate': 100*report.detected_count/max(1,report.total_cad_bends),
        'pass_count': report.pass_count,
        'accuracy': accuracy,
    }

# Run test
test_bend_detection(
    "/Users/idant/Downloads/44211000_A.stl",  # CAD
    "/Users/idant/Downloads/44211000_A.ply",  # Scan
    "44211000_A"
)
```

### 2. Dimension Measurement Test

Tests XLSX dimension parsing and measurement accuracy.

```python
import sys
sys.path.insert(0, "/Users/idant/82Labs/Sherman QC/Full Project/backend")
from dimension_parser import parse_dimension_file, DimensionType

def test_dimension_parsing(xlsx_path: str):
    """
    Test dimension specification parsing from XLSX.

    Expected: All dimensions parsed with correct types and tolerances.
    """
    print(f"\n{'='*60}")
    print(f"DIMENSION PARSING TEST")
    print(f"{'='*60}")
    print(f"File: {xlsx_path}")

    result = parse_dimension_file(xlsx_path)

    if not result.success:
        print(f"FAILED: {result.error}")
        return None

    print(f"\nParsed {len(result.dimensions)} dimensions:")
    print(f"{'ID':<6} {'Type':<10} {'Value':<12} {'Tolerance':<15} {'Unit':<6}")
    print("-" * 55)

    type_counts = {}
    for dim in result.dimensions:
        type_name = dim.dim_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

        tol_str = f"±{dim.tolerance_plus}" if dim.tolerance_plus == dim.tolerance_minus else \
                  f"+{dim.tolerance_plus}/-{dim.tolerance_minus}"

        print(f"{dim.dim_id:<6} {type_name:<10} {dim.value:<12.3f} {tol_str:<15} {dim.unit:<6}")

    print(f"\nSummary by type:")
    for t, count in type_counts.items():
        print(f"  {t}: {count}")

    return result

# Test with Hebrew XLSX
test_dimension_parsing("/Users/idant/Downloads/44211000_A   מידות.xlsx")
# Or: test_dimension_parsing("/Users/idant/82Labs/Sherman QC/Full Project/test_files/47266000_F/dimensions.xlsx")
```

### 3. API Integration Test

Tests the full API workflow.

```python
import requests
import time

API_BASE = "http://localhost:8080/api"

def test_bend_inspection_api(cad_path: str, scan_path: str, part_id: str):
    """
    Test bend inspection via REST API.
    """
    print(f"\n{'='*60}")
    print(f"API BEND INSPECTION TEST: {part_id}")
    print(f"{'='*60}")

    # Submit job
    with open(cad_path, 'rb') as cad_f, open(scan_path, 'rb') as scan_f:
        files = {
            'cad_file': ('cad.stl', cad_f),
            'scan_file': ('scan.ply', scan_f),
        }
        data = {
            'part_id': part_id,
            'part_name': f'Test {part_id}',
            'default_tolerance_angle': '3.0',
            'default_tolerance_radius': '2.0',
        }

        resp = requests.post(f"{API_BASE}/bend-inspection/analyze", files=files, data=data)

    if resp.status_code != 200:
        print(f"FAILED: {resp.status_code} - {resp.text}")
        return None

    job_id = resp.json()['job_id']
    print(f"Job ID: {job_id}")

    # Poll for completion
    for i in range(60):
        time.sleep(2)
        status_resp = requests.get(f"{API_BASE}/bend-inspection/{job_id}")
        status = status_resp.json()

        print(f"[{(i+1)*2}s] {status.get('status')} - {status.get('progress', 0)}%")

        if status.get('status') == 'completed':
            # Get results table
            table_resp = requests.get(f"{API_BASE}/bend-inspection/{job_id}/table")
            print("\n" + table_resp.json().get('table', ''))
            return status
        elif status.get('status') == 'failed':
            print(f"FAILED: {status.get('error')}")
            return None

    print("TIMEOUT")
    return None

# Run API test (requires server running)
# test_bend_inspection_api(
#     "/Users/idant/Downloads/44211000_A.stl",
#     "/Users/idant/Downloads/44211000_A.ply",
#     "44211000_A"
# )
```

### 4. Hole Detection Test

```python
import sys
sys.path.insert(0, "/Users/idant/82Labs/Sherman QC/Full Project/backend")
from feature_detection import HoleDetector
import open3d as o3d
import numpy as np

def test_hole_detection(mesh_path: str):
    """
    Test hole detection in CAD mesh.
    """
    print(f"\n{'='*60}")
    print(f"HOLE DETECTION TEST")
    print(f"{'='*60}")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    detector = HoleDetector(
        min_diameter=0.5,
        max_diameter=100.0,
        circularity_threshold=0.85,
    )

    holes = detector.detect_holes(vertices, triangles)

    print(f"Found {len(holes)} holes:")
    print(f"{'ID':<6} {'Diameter':<12} {'Circularity':<12} {'Center'}")
    print("-" * 60)

    for h in holes:
        center_str = f"({h.center[0]:.1f}, {h.center[1]:.1f}, {h.center[2]:.1f})"
        print(f"{h.hole_id:<6} {h.diameter:<12.2f} {h.circularity:<12.3f} {center_str}")

    return holes

# test_hole_detection("/Users/idant/82Labs/Sherman QC/Full Project/test_files/47266000_F/cad_mesh.stl")
```

### 5. Full Accuracy Benchmark

```python
def run_full_accuracy_benchmark():
    """
    Run comprehensive accuracy tests across all available test parts.
    """
    test_parts = [
        {
            'name': '44211000_A',
            'cad': '/Users/idant/Downloads/44211000_A.stl',
            'scan': '/Users/idant/Downloads/44211000_A.ply',
            'dims': '/Users/idant/Downloads/44211000_A   מידות.xlsx',
        },
        {
            'name': '49125000_A00',
            'cad': '/Users/idant/Downloads/49125000_A00.stl',
            'scan': '/Users/idant/Downloads/49125000_A00.ply',
            'dims': None,
        },
        {
            'name': '47266000_F',
            'cad': '/Users/idant/82Labs/Sherman QC/Full Project/test_files/47266000_F/cad_mesh.stl',
            'scan': '/Users/idant/82Labs/Sherman QC/Full Project/test_files/47266000_F/scan.stl',
            'dims': '/Users/idant/82Labs/Sherman QC/Full Project/test_files/47266000_F/dimensions.xlsx',
        },
    ]

    results = []
    for part in test_parts:
        print(f"\n\n{'#'*70}")
        print(f"# TESTING: {part['name']}")
        print(f"{'#'*70}")

        # Bend detection
        bend_result = test_bend_detection(part['cad'], part['scan'], part['name'])

        # Dimension parsing (if available)
        if part['dims']:
            dim_result = test_dimension_parsing(part['dims'])

        results.append(bend_result)

    # Summary
    print(f"\n\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Part':<20} {'CAD Bends':<12} {'Scan Bends':<12} {'Match %':<10} {'Accuracy %':<10}")
    print("-" * 70)

    for r in results:
        if r:
            print(f"{r['part']:<20} {r['cad_bends']:<12} {r['scan_bends']:<12} {r['match_rate']:<10.1f} {r['accuracy']:<10.1f}")

    avg_accuracy = np.mean([r['accuracy'] for r in results if r])
    print(f"\n*** OVERALL ACCURACY: {avg_accuracy:.1f}% ***")

# run_full_accuracy_benchmark()
```

---

## Pass/Fail Criteria

### Bend Detection
| Metric | PASS | WARNING | FAIL |
|--------|------|---------|------|
| Angle deviation | ≤ tolerance | ≤ 2× tolerance | > 2× tolerance |
| Match rate | ≥ 90% | ≥ 75% | < 75% |

### Dimension Measurement
| Metric | PASS | WARNING | FAIL |
|--------|------|---------|------|
| Value deviation | ≤ tolerance | ≤ 1.5× tolerance | > 1.5× tolerance |
| Circularity (holes) | ≥ 0.92 | ≥ 0.85 | < 0.85 |

### Quality Score (0-100)
- **Pass rate contribution**: Up to 60 points (pass_rate × 60)
- **Max deviation score**: 20 (≤1×tol), 10 (≤2×tol), 5 (≤3×tol), 0 (>3×tol)
- **Issue penalty**: 20 (none), 15 (1-2 issues), 5 (3+ issues), 0 (critical)

---

## Running pytest Tests

```bash
cd "/Users/idant/82Labs/Sherman QC/Full Project"

# All tests
pytest tests/ -v

# Fast tests only
pytest tests/ -m "not slow" -v

# Specific test files
pytest tests/test_bend_detector.py -v
pytest tests/test_feature_measurement.py -v

# Real part tests
pytest tests/test_part_11979003.py -v
pytest tests/test_part_47266000.py -v

# With coverage
pytest tests/ --cov=backend --cov-report=term-missing
```

---

## Interpreting Results

### Bend Detection Accuracy
- **> 90%**: Excellent - algorithm working well
- **75-90%**: Good - minor tuning may help
- **< 75%**: Needs investigation - check preprocessing, parameters

### Common Issues
1. **Low match rate**: Increase `max_angle_diff` in matcher
2. **False positives**: Increase `min_plane_area` or `min_plane_points`
3. **Inconsistent detection**: Enable `preprocess=True`, use `ransac_seed`
4. **Large radius errors**: Check point density in bend transition zone

### Debugging
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check preprocessing effect
detector = BendDetector(ransac_seed=42)
bends_raw = detector.detect_bends(points, preprocess=False)
bends_clean = detector.detect_bends(points, preprocess=True)
print(f"Raw: {len(bends_raw)}, Preprocessed: {len(bends_clean)}")
```
