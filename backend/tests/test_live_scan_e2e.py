#!/usr/bin/env python3
"""
End-to-End Integration Test for Live Scan System

Tests the complete workflow:
1. Part catalog with CAD files
2. Embedding computation and FAISS indexing
3. File watcher detecting new scans
4. Session manager lifecycle
5. Part recognition
6. Coverage calculation
7. API endpoints

Run with: python -m pytest tests/test_live_scan_e2e.py -v
Or directly: python tests/test_live_scan_e2e.py
"""

import os
import sys
import time
import json
import shutil
import tempfile
import threading
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import open3d as o3d


def create_test_point_cloud(shape: str = "box", num_points: int = 5000) -> np.ndarray:
    """Create a synthetic point cloud for testing."""
    if shape == "box":
        # Create a box with dimensions similar to sheet metal part
        points = []
        # Top and bottom surfaces
        for z in [0, 2]:
            x = np.random.uniform(0, 50, num_points // 4)
            y = np.random.uniform(0, 30, num_points // 4)
            z_arr = np.full(num_points // 4, z)
            points.append(np.column_stack([x, y, z_arr]))
        # Front and back surfaces
        for y in [0, 30]:
            x = np.random.uniform(0, 50, num_points // 8)
            y_arr = np.full(num_points // 8, y)
            z = np.random.uniform(0, 2, num_points // 8)
            points.append(np.column_stack([x, y_arr, z]))
        # Left and right surfaces
        for x in [0, 50]:
            x_arr = np.full(num_points // 8, x)
            y = np.random.uniform(0, 30, num_points // 8)
            z = np.random.uniform(0, 2, num_points // 8)
            points.append(np.column_stack([x_arr, y, z]))
        return np.vstack(points)

    elif shape == "L":
        # L-shaped bracket
        points = []
        # Horizontal part
        x = np.random.uniform(0, 40, num_points // 2)
        y = np.random.uniform(0, 20, num_points // 2)
        z = np.random.uniform(0, 2, num_points // 2)
        points.append(np.column_stack([x, y, z]))
        # Vertical part (bent up)
        x = np.random.uniform(0, 10, num_points // 2)
        y = np.random.uniform(0, 20, num_points // 2)
        z = np.random.uniform(2, 25, num_points // 2)
        points.append(np.column_stack([x, y, z]))
        return np.vstack(points)

    else:
        # Random points
        return np.random.uniform(-25, 25, (num_points, 3))


def save_point_cloud(points: np.ndarray, path: Path) -> None:
    """Save points to PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), pcd)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = "") -> None:
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


class LiveScanE2ETest:
    """End-to-end test suite for Live Scan system."""

    def __init__(self):
        self.temp_dir = None
        self.watch_dir = None
        self.cad_dir = None
        self.results = []

    def setup(self):
        """Set up test environment."""
        print_header("Setting Up Test Environment")

        # Create temp directories
        self.temp_dir = Path(tempfile.mkdtemp(prefix="livescan_e2e_"))
        self.watch_dir = self.temp_dir / "watch"
        self.cad_dir = self.temp_dir / "cad"
        self.watch_dir.mkdir()
        self.cad_dir.mkdir()

        print(f"  Temp directory: {self.temp_dir}")
        print(f"  Watch directory: {self.watch_dir}")
        print(f"  CAD directory: {self.cad_dir}")

        # Create test CAD files
        print("\n  Creating test CAD files...")

        # Part 1: Simple box
        box_points = create_test_point_cloud("box", 10000)
        save_point_cloud(box_points, self.cad_dir / "TEST_BOX_001.ply")
        print(f"    Created TEST_BOX_001.ply ({len(box_points)} points)")

        # Part 2: L-bracket
        l_points = create_test_point_cloud("L", 10000)
        save_point_cloud(l_points, self.cad_dir / "TEST_LBRACKET_001.ply")
        print(f"    Created TEST_LBRACKET_001.ply ({len(l_points)} points)")

        return True

    def teardown(self):
        """Clean up test environment."""
        print_header("Cleaning Up")

        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"  Removed: {self.temp_dir}")

    def test_part_catalog(self) -> bool:
        """Test 1: Part catalog operations."""
        print_header("Test 1: Part Catalog")

        try:
            from part_catalog import PartCatalogManager, get_catalog

            catalog = get_catalog()

            # Create test parts
            part1 = catalog.create_part(
                part_number="TEST_BOX_001",
                part_name="Test Box Part",
                customer="E2E Test",
                material="Aluminum",
            )
            print_result("Create part 1", part1 is not None, f"ID: {part1.id if part1 else 'None'}")

            part2 = catalog.create_part(
                part_number="TEST_LBRACKET_001",
                part_name="Test L-Bracket",
                customer="E2E Test",
                material="Steel",
            )
            print_result("Create part 2", part2 is not None, f"ID: {part2.id if part2 else 'None'}")

            # Update CAD paths
            if part1:
                catalog.update_part(part1.id, cad_file_path=str(self.cad_dir / "TEST_BOX_001.ply"))
            if part2:
                catalog.update_part(part2.id, cad_file_path=str(self.cad_dir / "TEST_LBRACKET_001.ply"))

            # Verify
            stats = catalog.count_parts()
            print_result("Parts in catalog", stats["total"] >= 2, f"Total: {stats['total']}")

            self.part1_id = part1.id if part1 else None
            self.part2_id = part2.id if part2 else None

            return part1 is not None and part2 is not None

        except Exception as e:
            print_result("Part catalog test", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    def test_embedding_pipeline(self) -> bool:
        """Test 2: Embedding computation and FAISS indexing."""
        print_header("Test 2: Embedding Pipeline")

        try:
            from embedding_service import compute_embedding, EmbeddingResult
            from faiss_index import get_faiss_index
            from part_catalog import get_catalog

            catalog = get_catalog()
            faiss_index = get_faiss_index()

            # Compute embeddings for test parts
            for part_number in ["TEST_BOX_001", "TEST_LBRACKET_001"]:
                cad_path = self.cad_dir / f"{part_number}.ply"

                # Load point cloud
                pcd = o3d.io.read_point_cloud(str(cad_path))
                points = np.asarray(pcd.points)

                # Compute embedding
                start = time.time()
                result = compute_embedding(points)
                elapsed = (time.time() - start) * 1000

                print_result(
                    f"Embedding for {part_number}",
                    result is not None and len(result.embedding) == 1024,
                    f"Time: {elapsed:.0f}ms, Dim: {len(result.embedding) if result else 0}"
                )

                if result:
                    # Get part and add to FAISS
                    part = catalog.get_part_by_number(part_number)
                    if part:
                        faiss_index.add(part.id, result.embedding)
                        # Update catalog
                        catalog.update_part(part.id, has_embedding=True)

            # Verify FAISS index
            count = faiss_index.count()
            print_result("FAISS index count", count >= 2, f"Count: {count}")

            return count >= 2

        except Exception as e:
            print_result("Embedding pipeline test", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    def test_recognition(self) -> bool:
        """Test 3: Part recognition accuracy."""
        print_header("Test 3: Part Recognition")

        try:
            from embedding_service import compute_embedding
            from faiss_index import get_faiss_index
            from part_catalog import get_catalog

            catalog = get_catalog()
            faiss_index = get_faiss_index()

            # Create a "scan" that's similar to TEST_BOX_001
            scan_points = create_test_point_cloud("box", 5000)
            # Add some noise
            scan_points += np.random.normal(0, 0.5, scan_points.shape)

            # Compute embedding
            result = compute_embedding(scan_points)

            # Search FAISS
            matches = faiss_index.search(result.embedding, k=5, threshold=0.0)

            print(f"\n  Recognition results for box-like scan:")
            for i, match in enumerate(matches[:3]):
                part = catalog.get_part(match.part_id)
                part_num = part.part_number if part else "Unknown"
                print(f"    {i+1}. {part_num}: {match.similarity*100:.1f}%")

            # Check if top match is correct
            top_match = matches[0] if matches else None
            if top_match:
                top_part = catalog.get_part(top_match.part_id)
                is_correct = top_part and "BOX" in top_part.part_number
                print_result(
                    "Box scan recognized correctly",
                    is_correct,
                    f"Top: {top_part.part_number if top_part else 'None'} ({top_match.similarity*100:.1f}%)"
                )
            else:
                print_result("Box scan recognized correctly", False, "No matches found")
                return False

            # Now test L-bracket recognition
            l_scan_points = create_test_point_cloud("L", 5000)
            l_scan_points += np.random.normal(0, 0.5, l_scan_points.shape)

            l_result = compute_embedding(l_scan_points)
            l_matches = faiss_index.search(l_result.embedding, k=5, threshold=0.0)

            print(f"\n  Recognition results for L-bracket-like scan:")
            for i, match in enumerate(l_matches[:3]):
                part = catalog.get_part(match.part_id)
                part_num = part.part_number if part else "Unknown"
                print(f"    {i+1}. {part_num}: {match.similarity*100:.1f}%")

            l_top = l_matches[0] if l_matches else None
            if l_top:
                l_part = catalog.get_part(l_top.part_id)
                l_correct = l_part and "LBRACKET" in l_part.part_number
                print_result(
                    "L-bracket scan recognized correctly",
                    l_correct,
                    f"Top: {l_part.part_number if l_part else 'None'} ({l_top.similarity*100:.1f}%)"
                )
                return is_correct and l_correct

            return False

        except Exception as e:
            print_result("Recognition test", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    def test_coverage_calculator(self) -> bool:
        """Test 4: Coverage calculation."""
        print_header("Test 4: Coverage Calculator")

        try:
            from coverage_calculator import CoverageCalculator

            calculator = CoverageCalculator(voxel_size=2.0, tolerance_mm=3.0)

            # CAD reference
            cad_points = create_test_point_cloud("box", 10000)

            # Partial scan (50% coverage - only half the box)
            partial_scan = cad_points[cad_points[:, 0] < 25]  # Left half only
            partial_scan += np.random.normal(0, 0.3, partial_scan.shape)

            # Calculate coverage
            start = time.time()
            result = calculator.compute_coverage(cad_points, partial_scan)
            elapsed = (time.time() - start) * 1000

            print(f"\n  Partial scan coverage:")
            print(f"    Coverage: {result.coverage_percent:.1f}%")
            print(f"    Gaps: {len(result.gap_clusters)}")
            print(f"    Time: {elapsed:.0f}ms")

            # Should be around 40-60% coverage
            coverage_reasonable = 30 < result.coverage_percent < 70
            print_result(
                "Partial coverage calculation",
                coverage_reasonable,
                f"Expected 40-60%, got {result.coverage_percent:.1f}%"
            )

            # Full scan (should be ~100%)
            full_scan = cad_points + np.random.normal(0, 0.3, cad_points.shape)
            full_result = calculator.compute_coverage(cad_points, full_scan)

            print(f"\n  Full scan coverage:")
            print(f"    Coverage: {full_result.coverage_percent:.1f}%")
            print(f"    Gaps: {len(full_result.gap_clusters)}")

            full_coverage_high = full_result.coverage_percent > 90
            print_result(
                "Full coverage calculation",
                full_coverage_high,
                f"Expected >90%, got {full_result.coverage_percent:.1f}%"
            )

            return coverage_reasonable and full_coverage_high

        except Exception as e:
            print_result("Coverage calculator test", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    def test_session_manager(self) -> bool:
        """Test 5: Session manager lifecycle."""
        print_header("Test 5: Session Manager")

        try:
            from live_scan_session import LiveScanSessionManager, SessionState

            # Create session manager
            manager = LiveScanSessionManager(
                str(self.watch_dir),
                session_timeout_seconds=30,
                idle_timeout_seconds=10,
                confidence_threshold=0.85,
            )

            # Track session updates
            updates = []
            def on_update(session):
                updates.append({
                    "state": session.state.value,
                    "time": datetime.now().isoformat(),
                })

            manager.on_session_update = on_update

            # Start manager
            manager.start()
            print_result("Manager started", manager.is_running())

            # Wait a moment for watcher to initialize
            time.sleep(1)

            # Create a scan file
            scan_points = create_test_point_cloud("box", 5000)
            scan_path = self.watch_dir / "SCAN_001.ply"
            save_point_cloud(scan_points, scan_path)
            print(f"  Created scan file: {scan_path.name}")

            # Wait for processing (debounce + recognition)
            print("  Waiting for file detection and recognition...")
            time.sleep(5)

            # Check session
            session = manager.get_current_session()
            has_session = session is not None
            print_result("Session created", has_session)

            if session:
                print(f"    State: {session.state.value}")
                print(f"    Scans: {len(session.scans)}")
                print(f"    Recognition: {session.recognition_result is not None}")

                # Check recognition
                if session.recognition_result:
                    candidates = session.recognition_result.candidates
                    print(f"    Candidates: {len(candidates)}")
                    if candidates:
                        top = candidates[0]
                        print(f"    Top match: {top['part_number']} ({top['similarity']*100:.1f}%)")

            # Test manual confirmation
            if session and session.recognition_result and session.recognition_result.candidates:
                top_candidate = session.recognition_result.candidates[0]
                confirmed = manager.confirm_part(
                    top_candidate["part_id"],
                    top_candidate["part_number"]
                )
                print_result("Part confirmation", confirmed)

                # Check state changed
                session = manager.get_current_session()
                state_correct = session and session.state == SessionState.SCANNING
                print_result("State is SCANNING", state_correct)

            # Test session completion
            if session:
                completed = manager.complete_scan()
                print_result("Scan completion", completed)

                session = manager.get_current_session()
                if session:
                    print(f"    Final state: {session.state.value}")

            # Stop manager
            manager.stop()
            print_result("Manager stopped", not manager.is_running())

            # Summary
            print(f"\n  Session updates received: {len(updates)}")
            for u in updates[:5]:
                print(f"    - {u['state']}")

            return has_session

        except Exception as e:
            print_result("Session manager test", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    def test_api_endpoints(self) -> bool:
        """Test 6: API endpoints (simulated)."""
        print_header("Test 6: API Endpoints (Import Check)")

        try:
            # Import server to verify endpoints are registered
            from server import app

            # Get all routes
            routes = [r.path for r in app.routes]

            required_routes = [
                "/api/live-scan/session",
                "/api/live-scan/session/{session_id}/confirm",
                "/api/live-scan/session/{session_id}/complete",
                "/api/live-scan/session/{session_id}/cancel",
                "/api/live-scan/session/reset",
                "/api/live-scan/session/stream",
                "/api/live-scan/start",
                "/api/live-scan/stop",
                "/api/live-scan/status",
                "/api/recognize/status",
            ]

            all_present = True
            for route in required_routes:
                present = route in routes
                if not present:
                    print_result(f"Route {route}", False, "Not found")
                    all_present = False

            if all_present:
                print_result("All required routes present", True, f"{len(required_routes)} routes")

            return all_present

        except Exception as e:
            print_result("API endpoints test", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    def run_all(self) -> dict:
        """Run all tests and return results."""
        print("\n" + "="*60)
        print("  LIVE SCAN E2E INTEGRATION TEST")
        print("="*60)

        try:
            self.setup()

            tests = [
                ("Part Catalog", self.test_part_catalog),
                ("Embedding Pipeline", self.test_embedding_pipeline),
                ("Recognition", self.test_recognition),
                ("Coverage Calculator", self.test_coverage_calculator),
                ("Session Manager", self.test_session_manager),
                ("API Endpoints", self.test_api_endpoints),
            ]

            results = {}
            for name, test_fn in tests:
                try:
                    results[name] = test_fn()
                except Exception as e:
                    print(f"\n  ERROR in {name}: {e}")
                    import traceback
                    traceback.print_exc()
                    results[name] = False

            # Summary
            print_header("TEST SUMMARY")

            passed = sum(1 for v in results.values() if v)
            total = len(results)

            for name, result in results.items():
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {status}: {name}")

            print(f"\n  Total: {passed}/{total} passed")
            print(f"  Status: {'ALL TESTS PASSED' if passed == total else 'SOME TESTS FAILED'}")

            return results

        finally:
            self.teardown()


if __name__ == "__main__":
    test = LiveScanE2ETest()
    results = test.run_all()

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
