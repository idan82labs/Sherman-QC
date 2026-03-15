#!/usr/bin/env python3
"""
Scan QC Application - Startup Script

Usage:
    python run.py              # Start server on port 8080
    python run.py --port 3000  # Start on custom port
    python run.py --demo       # Run demo with test files
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))


def maybe_reexec_with_python311():
    """Prefer Python 3.11+ for Open3D stability on macOS."""
    if sys.platform != "darwin":
        return
    if sys.version_info >= (3, 11):
        return

    candidates = [
        os.environ.get("BACKEND_PYTHON", "").strip(),
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.11",
        shutil.which("python3.11") or "",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).exists():
            os.execv(candidate, [candidate, str(Path(__file__).resolve()), *sys.argv[1:]])


def install_dependencies():
    """Install required packages"""
    import subprocess
    
    packages = [
        "fastapi", "uvicorn", "python-multipart",
        "open3d", "trimesh", "numpy", "scipy", "rtree",
        "reportlab", "PyMuPDF"
    ]
    
    print("Installing dependencies...")
    for pkg in packages:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            pkg, "--break-system-packages", "-q"
        ], capture_output=True)
    print("Dependencies installed!")


def start_server(port: int = 8080):
    """Start the FastAPI server"""
    import uvicorn
    from backend.server import app
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           SCAN QC SYSTEM - AI-Powered Quality Control            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Server running at: http://localhost:{port}                      ║
║                                                                  ║
║   API Endpoints:                                                 ║
║     POST /api/analyze     - Start analysis job                   ║
║     GET  /api/progress/{{id}} - Get job progress                   ║
║     GET  /api/result/{{id}}   - Get analysis results               ║
║     GET  /api/download/{{id}}/pdf - Download PDF report            ║
║                                                                  ║
║   Press Ctrl+C to stop                                           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def run_demo():
    """Run demo with synthetic test data"""
    import numpy as np
    import open3d as o3d
    
    print("Creating demo test files...")
    
    # Create directories
    demo_dir = Path(__file__).parent / "demo_files"
    demo_dir.mkdir(exist_ok=True)
    
    # Create reference cube (100x100x20mm)
    mesh = o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=20)
    mesh.compute_vertex_normals()
    ref_path = demo_dir / "reference.stl"
    o3d.io.write_triangle_mesh(str(ref_path), mesh)
    
    # Create scan with deviations
    pcd = mesh.sample_points_uniformly(number_of_points=50000)
    points = np.asarray(pcd.points)
    
    # Add deviations
    noise = np.random.normal(0, 0.03, points.shape)
    corner_mask = (points[:, 0] > 80) & (points[:, 1] > 80)
    noise[corner_mask] += np.array([0, 0, 0.12])
    
    points_deviated = points + noise
    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(points_deviated)
    
    scan_path = demo_dir / "scan.ply"
    o3d.io.write_point_cloud(str(scan_path), scan_pcd)
    
    print(f"""
Demo files created in: {demo_dir}
  - reference.stl (Reference model)
  - scan.ply (Simulated scan with deviations)

To test:
1. Start the server: python run.py
2. Open http://localhost:8080
3. Upload the demo files
    """)


def main():
    maybe_reexec_with_python311()
    parser = argparse.ArgumentParser(description="Scan QC System")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--demo", action="store_true", help="Create demo files")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
        return
    
    if args.demo:
        run_demo()
        return
    
    start_server(args.port)


if __name__ == "__main__":
    main()
