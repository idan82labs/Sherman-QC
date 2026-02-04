"""
Quick test script for bend inspection API with synthetic data.
"""
import numpy as np
import open3d as o3d
import tempfile
import os
import requests
import time
import json

# Create a simple L-shaped mesh (two perpendicular planes) as test data
def create_l_shaped_mesh():
    """Create an L-shaped mesh with one 90-degree bend.

    Creates a continuous bent sheet metal shape where the two planes
    share vertices along the bend line for proper adjacency detection.
    """
    vertices = []
    faces = []

    # Parameters
    width = 40  # Width along Y axis (bend line direction)
    flange1_length = 50  # Length of first flange (along X)
    flange2_length = 50  # Length of second flange (along Z)
    step = 2.0  # Grid spacing

    n_width = int(width / step) + 1
    n_f1 = int(flange1_length / step) + 1
    n_f2 = int(flange2_length / step) + 1

    # First flange: horizontal plane (z=0, x from 0 to flange1_length)
    for i in range(n_f1):
        for j in range(n_width):
            x = i * step
            y = -width/2 + j * step
            z = 0
            vertices.append([x, y, z])

    # Create faces for first flange
    for i in range(n_f1 - 1):
        for j in range(n_width - 1):
            v0 = i * n_width + j
            v1 = i * n_width + (j + 1)
            v2 = (i + 1) * n_width + j
            v3 = (i + 1) * n_width + (j + 1)
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Second flange: vertical plane at x=0 (z from 0 to flange2_length)
    # IMPORTANT: Reuse the edge vertices from first flange for continuity
    # The edge at x=0 of first flange (vertices 0, 1, 2, ..., n_width-1)
    # becomes the base of the second flange

    # Add vertices for the rest of second flange (z > 0)
    start_idx = len(vertices)
    for i in range(1, n_f2):  # Start from 1 since z=0 vertices are shared
        for j in range(n_width):
            x = 0
            y = -width/2 + j * step
            z = i * step
            vertices.append([x, y, z])

    # Create faces for second flange
    # First row connects to existing edge vertices
    for j in range(n_width - 1):
        v0 = j  # Shared vertices from first flange edge
        v1 = j + 1
        v2 = start_idx + j
        v3 = start_idx + j + 1
        faces.append([v0, v1, v2])
        faces.append([v1, v3, v2])

    # Remaining rows
    for i in range(1, n_f2 - 1):
        for j in range(n_width - 1):
            v0 = start_idx + (i - 1) * n_width + j
            v1 = start_idx + (i - 1) * n_width + (j + 1)
            v2 = start_idx + i * n_width + j
            v3 = start_idx + i * n_width + (j + 1)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh


def create_scan_with_bend_error(base_mesh, angle_error_deg=0.5):
    """
    Create a 'scan' that has a slight angle error in the bend.
    This simulates a real-world scan where the bend isn't perfect.

    Uses dense point sampling to ensure plane detection works properly.
    """
    np.random.seed(42)

    # Create dense point cloud directly (not from mesh) for reliable detection
    # Horizontal plane (z=0, x from 5 to 50)
    n_h = 1500
    x_h = np.random.uniform(5, 50, n_h)
    y_h = np.random.uniform(-20, 20, n_h)
    z_h = np.random.normal(0, 0.1, n_h)
    points_h = np.column_stack([x_h, y_h, z_h])

    # Vertical plane with angle error
    angle_rad = np.radians(90 + angle_error_deg)
    n_v = 1500
    t_v = np.random.uniform(5, 50, n_v)  # Distance along vertical
    y_v = np.random.uniform(-20, 20, n_v)

    # Apply rotation for the vertical plane
    x_v = t_v * np.cos(angle_rad) + np.random.normal(0, 0.1, n_v)
    z_v = t_v * np.sin(angle_rad) + np.random.normal(0, 0.1, n_v)
    points_v = np.column_stack([x_v, y_v, z_v])

    # Transition zone
    n_t = 500
    x_t = np.random.uniform(0, 5, n_t)
    y_t = np.random.uniform(-20, 20, n_t)
    z_t = np.where(x_t > 2.5, np.random.normal(0, 0.1, n_t), np.random.uniform(0, 5, n_t))
    x_t = np.where(z_t > 2.5, np.random.normal(0, 0.1, n_t), x_t)
    points_t = np.column_stack([x_t, y_t, z_t])

    points = np.vstack([points_h, points_v, points_t])
    points += np.random.normal(0, 0.1, points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_cad_point_cloud():
    """Create a 'CAD' point cloud representing an ideal 90-degree bend."""
    np.random.seed(123)

    # Horizontal plane (z=0, x from 5 to 50)
    n_h = 1500
    x_h = np.random.uniform(5, 50, n_h)
    y_h = np.random.uniform(-20, 20, n_h)
    z_h = np.random.normal(0, 0.1, n_h)
    points_h = np.column_stack([x_h, y_h, z_h])

    # Vertical plane (exact 90 degrees)
    n_v = 1500
    x_v = np.random.normal(0, 0.1, n_v)
    y_v = np.random.uniform(-20, 20, n_v)
    z_v = np.random.uniform(5, 50, n_v)
    points_v = np.column_stack([x_v, y_v, z_v])

    # Transition zone
    n_t = 500
    x_t = np.random.uniform(0, 5, n_t)
    y_t = np.random.uniform(-20, 20, n_t)
    z_t = np.where(x_t > 2.5, np.random.normal(0, 0.1, n_t), np.random.uniform(0, 5, n_t))
    x_t = np.where(z_t > 2.5, np.random.normal(0, 0.1, n_t), x_t)
    points_t = np.column_stack([x_t, y_t, z_t])

    points = np.vstack([points_h, points_v, points_t])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def main():
    print("=" * 60)
    print("BEND INSPECTION API TEST")
    print("=" * 60)

    # Create test data using point clouds for reliable detection
    print("\n1. Creating test point clouds...")
    cad_pcd = create_cad_point_cloud()
    scan_pcd = create_scan_with_bend_error(None, angle_error_deg=2.0)  # 2 degree error

    # Save to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        cad_path = os.path.join(tmpdir, "cad.ply")
        scan_path = os.path.join(tmpdir, "scan.ply")

        o3d.io.write_point_cloud(cad_path, cad_pcd)
        o3d.io.write_point_cloud(scan_path, scan_pcd)

        print(f"   CAD PCD: {len(cad_pcd.points)} points")
        print(f"   Scan PCD: {len(scan_pcd.points)} points")

        # Test the API
        print("\n2. Starting bend inspection...")

        # Check if server is running
        try:
            resp = requests.get("http://localhost:8080/api/health", timeout=5)
            print(f"   Server health: {resp.json()}")
        except Exception as e:
            print(f"   ERROR: Server not reachable: {e}")
            return

        # Start inspection
        with open(cad_path, 'rb') as cad_f, open(scan_path, 'rb') as scan_f:
            files = {
                'cad_file': ('cad.ply', cad_f, 'application/octet-stream'),
                'scan_file': ('scan.ply', scan_f, 'application/octet-stream'),
            }
            data = {
                'part_id': 'L_SHAPED_TEST',
                'part_name': 'Test L-Shaped Part',
                'default_tolerance_angle': '1.0',
                'default_tolerance_radius': '0.5',
            }

            resp = requests.post(
                "http://localhost:8080/api/bend-inspection/analyze",
                files=files,
                data=data,
                timeout=30
            )

        if resp.status_code != 200:
            print(f"   ERROR: {resp.status_code} - {resp.text}")
            return

        result = resp.json()
        job_id = result.get('job_id')
        print(f"   Job started: {job_id}")

        # Poll for results
        print("\n3. Waiting for results...")
        for i in range(30):
            time.sleep(1)

            try:
                resp = requests.get(f"http://localhost:8080/api/bend-inspection/{job_id}", timeout=10)
                status_data = resp.json()

                status = status_data.get('status', 'unknown')
                progress = status_data.get('progress', 0)
                stage = status_data.get('stage', '')

                print(f"   [{i+1}s] Status: {status}, Progress: {progress}%, Stage: {stage}")

                if status == 'completed':
                    print("\n4. RESULTS")
                    print("-" * 50)

                    report = status_data.get('report')
                    if report:
                        summary = report.get('summary', {})
                        print(f"   Total bends in CAD: {summary.get('total_bends', 0)}")
                        print(f"   Detected in scan:   {summary.get('detected', 0)}")
                        print(f"   Passed:             {summary.get('passed', 0)}")
                        print(f"   Failed:             {summary.get('failed', 0)}")
                        print(f"   Warnings:           {summary.get('warnings', 0)}")
                        print(f"   Progress:           {summary.get('progress_pct', 0):.1f}%")

                        print("\n   BEND DETAILS:")
                        for match in report.get('matches', []):
                            bend_id = match.get('bend_id')
                            target = match.get('target_angle')
                            measured = match.get('measured_angle')
                            deviation = match.get('angle_deviation')
                            status = match.get('status')

                            if measured is not None:
                                print(f"   {bend_id}: Target={target:.1f}°, Measured={measured:.1f}°, Deviation={deviation:+.1f}°, Status={status}")
                            else:
                                print(f"   {bend_id}: Target={target:.1f}°, Measured=-, Status={status}")

                    # Get ASCII table
                    table_resp = requests.get(f"http://localhost:8080/api/bend-inspection/{job_id}/table", timeout=10)
                    if table_resp.status_code == 200:
                        print("\n5. ASCII TABLE OUTPUT:")
                        print("-" * 50)
                        print(table_resp.json().get('table', ''))

                    break

                elif status == 'failed':
                    print(f"\n   FAILED: {status_data.get('error', 'Unknown error')}")
                    break

            except requests.exceptions.RequestException as e:
                print(f"   [{i+1}s] Connection error: {e}")
                if i > 5:
                    print("   Server may have crashed, aborting...")
                    break

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
