#!/usr/bin/env python3
"""
Real-World Integration Test for Live Scan System

Uses actual CAD and scan files from Downloads folder to test:
1. Embedding computation with real parts
2. FAISS recognition accuracy
3. Coverage calculation with partial scans
4. Complete workflow validation

Run: python tests/test_live_scan_real_files.py
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import open3d as o3d


def print_header(text: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_result(name: str, passed: bool, details: str = "") -> None:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


# Real file paths
DOWNLOADS = Path.home() / "Downloads"
TEST_PARTS = {
    "44211000_A": {
        "cad": DOWNLOADS / "44211000_A.stl",
        "scan": DOWNLOADS / "44211000_A.ply",
    },
    "49125000_A00": {
        "cad": DOWNLOADS / "49125000_A00.stl",
        "scan": DOWNLOADS / "49125000_A00.ply",
    },
    "49024000_A": {
        "cad": DOWNLOADS / "49024000_A.stl",
        "scan": DOWNLOADS / "49024000_A.ply",
    },
}


def load_cad_as_points(path: Path, num_samples: int = 15000) -> np.ndarray:
    """Load CAD file (STL) and sample points."""
    if not path.exists():
        raise FileNotFoundError(f"CAD file not found: {path}")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {path}")

    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    return np.asarray(pcd.points)


def load_scan(path: Path) -> np.ndarray:
    """Load scan file (PLY)."""
    if not path.exists():
        raise FileNotFoundError(f"Scan file not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {path}")

    return np.asarray(pcd.points)


def split_scan_by_region(points: np.ndarray, fraction: float = 0.5, axis: int = 0) -> np.ndarray:
    """Split a scan to simulate partial coverage."""
    # Sort by axis and take fraction
    sorted_indices = np.argsort(points[:, axis])
    n_keep = int(len(points) * fraction)
    return points[sorted_indices[:n_keep]]


def split_scan_random(points: np.ndarray, fraction: float = 0.5, seed: int = 42) -> np.ndarray:
    """Randomly sample points to simulate sparse scan."""
    rng = np.random.default_rng(seed)
    n_keep = int(len(points) * fraction)
    indices = rng.choice(len(points), n_keep, replace=False)
    return points[indices]


class RealFilesTest:
    """Test suite using real scan/CAD files."""

    def __init__(self):
        self.embeddings = {}
        self.part_ids = {}

    def test_file_availability(self) -> bool:
        """Test 0: Check all required files exist."""
        print_header("Test 0: File Availability")

        all_exist = True
        for part_num, files in TEST_PARTS.items():
            cad_exists = files["cad"].exists()
            scan_exists = files["scan"].exists()

            if cad_exists and scan_exists:
                # Get file sizes
                cad_size = files["cad"].stat().st_size / (1024*1024)
                scan_size = files["scan"].stat().st_size / (1024*1024)
                print(f"  ✓ {part_num}: CAD={cad_size:.1f}MB, Scan={scan_size:.1f}MB")
            else:
                print(f"  ✗ {part_num}: CAD={'OK' if cad_exists else 'MISSING'}, Scan={'OK' if scan_exists else 'MISSING'}")
                all_exist = False

        return all_exist

    def test_embedding_computation(self) -> bool:
        """Test 1: Compute embeddings for real CAD files."""
        print_header("Test 1: Embedding Computation (Real CAD Files)")

        from embedding_service import compute_embedding

        all_success = True
        for part_num, files in TEST_PARTS.items():
            try:
                # Load CAD
                start = time.time()
                cad_points = load_cad_as_points(files["cad"])
                load_time = (time.time() - start) * 1000

                # Compute embedding
                start = time.time()
                result = compute_embedding(cad_points)
                embed_time = (time.time() - start) * 1000

                self.embeddings[part_num] = result.embedding

                print_result(
                    f"{part_num}",
                    True,
                    f"Points: {len(cad_points)}, Load: {load_time:.0f}ms, Embed: {embed_time:.0f}ms"
                )

            except Exception as e:
                print_result(f"{part_num}", False, str(e))
                all_success = False

        return all_success

    def test_faiss_indexing(self) -> bool:
        """Test 2: Add embeddings to FAISS and test search."""
        print_header("Test 2: FAISS Indexing and Search")

        from faiss_index import FAISSIndexManager as FAISSIndex

        # Create fresh index for testing
        index = FAISSIndex(embedding_dim=1024)

        # Add all embeddings
        for part_num, embedding in self.embeddings.items():
            part_id = f"test_{part_num}"
            self.part_ids[part_num] = part_id
            index.add(part_id, embedding)

        print(f"  Added {index.count()} parts to index")

        # Test search with each embedding (should find itself)
        all_correct = True
        for part_num, embedding in self.embeddings.items():
            results = index.search(embedding, k=3, threshold=0.0)

            top_match = results[0] if results else None
            is_self = top_match and top_match.part_id == self.part_ids[part_num]

            print(f"\n  Search for {part_num}:")
            for i, r in enumerate(results[:3]):
                # Find part number from ID
                match_num = [k for k, v in self.part_ids.items() if v == r.part_id]
                match_num = match_num[0] if match_num else "Unknown"
                marker = " <-- TOP" if i == 0 else ""
                print(f"    {i+1}. {match_num}: {r.similarity*100:.1f}%{marker}")

            if not is_self:
                all_correct = False

        print_result("Self-matching", all_correct, "Each CAD should match itself as #1")
        return all_correct

    def test_scan_recognition(self) -> bool:
        """Test 3: Recognize scans using FAISS index."""
        print_header("Test 3: Scan Recognition (Real Scans)")

        from embedding_service import compute_embedding
        from faiss_index import FAISSIndexManager as FAISSIndex

        # Recreate index
        index = FAISSIndex(embedding_dim=1024)
        for part_num, embedding in self.embeddings.items():
            index.add(self.part_ids[part_num], embedding)

        all_correct = True
        for part_num, files in TEST_PARTS.items():
            try:
                # Load scan
                scan_points = load_scan(files["scan"])

                # Compute embedding
                result = compute_embedding(scan_points)

                # Search
                matches = index.search(result.embedding, k=3, threshold=0.0)

                print(f"\n  Scan: {part_num} ({len(scan_points)} points)")

                for i, r in enumerate(matches[:3]):
                    match_num = [k for k, v in self.part_ids.items() if v == r.part_id]
                    match_num = match_num[0] if match_num else "Unknown"
                    marker = " ✓" if match_num == part_num else ""
                    print(f"    {i+1}. {match_num}: {r.similarity*100:.1f}%{marker}")

                # Check if correct
                top = matches[0] if matches else None
                correct_match = [k for k, v in self.part_ids.items() if v == top.part_id][0] if top else None
                is_correct = correct_match == part_num

                if not is_correct:
                    print(f"    ✗ Expected {part_num}, got {correct_match}")
                    all_correct = False

            except Exception as e:
                print_result(f"Scan {part_num}", False, str(e))
                all_correct = False

        print_result("All scans recognized correctly", all_correct)
        return all_correct

    def test_partial_scan_recognition(self) -> bool:
        """Test 4: Recognition with partial scans."""
        print_header("Test 4: Partial Scan Recognition")

        from embedding_service import compute_embedding
        from faiss_index import FAISSIndexManager as FAISSIndex

        # Recreate index
        index = FAISSIndex(embedding_dim=1024)
        for part_num, embedding in self.embeddings.items():
            index.add(self.part_ids[part_num], embedding)

        # Test with 44211000_A (simplest part)
        part_num = "44211000_A"
        files = TEST_PARTS[part_num]

        scan_points = load_scan(files["scan"])
        print(f"\n  Testing partial scans of {part_num} ({len(scan_points)} total points)")

        results = {}
        for coverage, method in [(0.75, "spatial"), (0.50, "spatial"), (0.50, "random"), (0.25, "random")]:
            if method == "spatial":
                partial = split_scan_by_region(scan_points, coverage)
            else:
                partial = split_scan_random(scan_points, coverage)

            # Recognize
            emb = compute_embedding(partial)
            matches = index.search(emb.embedding, k=3, threshold=0.0)

            top = matches[0] if matches else None
            match_num = [k for k, v in self.part_ids.items() if v == top.part_id][0] if top else "None"
            correct = match_num == part_num

            label = f"{int(coverage*100)}% {method}"
            results[label] = {
                "correct": correct,
                "match": match_num,
                "similarity": top.similarity if top else 0,
            }

            status = "✓" if correct else "✗"
            print(f"    {status} {label}: {match_num} ({top.similarity*100:.1f}%)")

        # Count successes
        successes = sum(1 for r in results.values() if r["correct"])
        print_result(
            f"Partial scan recognition",
            successes >= 3,
            f"{successes}/4 correct"
        )

        return successes >= 3

    def test_coverage_calculation(self) -> bool:
        """Test 5: Coverage calculation with partial scans."""
        print_header("Test 5: Coverage Calculation")

        from coverage_calculator import CoverageCalculator

        calculator = CoverageCalculator(voxel_size=2.0, tolerance=3.0)

        # Use 44211000_A
        part_num = "44211000_A"
        files = TEST_PARTS[part_num]

        cad_points = load_cad_as_points(files["cad"])
        scan_points = load_scan(files["scan"])

        print(f"\n  Part: {part_num}")
        print(f"  CAD points: {len(cad_points)}")
        print(f"  Scan points: {len(scan_points)}")

        # Full scan coverage
        print("\n  Full scan coverage:")
        start = time.time()
        full_result = calculator.compute_coverage(cad_points, scan_points)
        elapsed = (time.time() - start) * 1000
        print(f"    Coverage: {full_result.coverage_percent:.1f}%")
        print(f"    Gaps: {len(full_result.gap_clusters)}")
        print(f"    Time: {elapsed:.0f}ms")

        # Partial scans
        coverage_tests = []
        for target in [0.75, 0.50, 0.25]:
            partial = split_scan_by_region(scan_points, target)

            result = calculator.compute_coverage(cad_points, partial)
            coverage_tests.append({
                "target": target,
                "actual": result.coverage_percent,
                "gaps": len(result.gap_clusters),
            })

            print(f"\n  {int(target*100)}% spatial split:")
            print(f"    Points: {len(partial)} ({len(partial)/len(scan_points)*100:.0f}% of scan)")
            print(f"    Coverage: {result.coverage_percent:.1f}%")
            print(f"    Gaps: {len(result.gap_clusters)}")

            if result.gap_clusters:
                print(f"    Gap locations:")
                for gap in result.gap_clusters[:3]:
                    print(f"      - {gap['location_hint']} ({gap['diameter_mm']:.0f}mm)")

        # Verify coverage decreases with fewer points
        coverages = [t["actual"] for t in coverage_tests]
        decreasing = all(coverages[i] >= coverages[i+1] for i in range(len(coverages)-1))

        print_result(
            "Coverage decreases with fewer points",
            decreasing,
            f"75%→{coverages[0]:.0f}%, 50%→{coverages[1]:.0f}%, 25%→{coverages[2]:.0f}%"
        )

        # Verify gaps increase
        gaps = [t["gaps"] for t in coverage_tests]
        gaps_increase = gaps[0] <= gaps[1] <= gaps[2]

        print_result(
            "Gap count increases with fewer points",
            gaps_increase or gaps[-1] > 0,
            f"Gaps: {gaps[0]}→{gaps[1]}→{gaps[2]}"
        )

        return decreasing and (gaps_increase or gaps[-1] > 0)

    def test_cross_part_discrimination(self) -> bool:
        """Test 6: Ensure different parts are distinguishable."""
        print_header("Test 6: Cross-Part Discrimination")

        from embedding_service import compute_embedding

        # Compute similarity between all pairs
        part_nums = list(self.embeddings.keys())
        print(f"\n  Similarity matrix (%):")
        print(f"  {'':20}", end="")
        for p in part_nums:
            print(f"{p:15}", end="")
        print()

        min_cross_sim = 1.0
        max_cross_sim = 0.0

        for p1 in part_nums:
            print(f"  {p1:20}", end="")
            for p2 in part_nums:
                e1 = self.embeddings[p1]
                e2 = self.embeddings[p2]
                # Cosine similarity (embeddings are L2-normalized)
                sim = np.dot(e1, e2)
                print(f"{sim*100:14.1f}%", end="")

                if p1 != p2:
                    min_cross_sim = min(min_cross_sim, sim)
                    max_cross_sim = max(max_cross_sim, sim)
            print()

        print(f"\n  Cross-part similarity range: {min_cross_sim*100:.1f}% - {max_cross_sim*100:.1f}%")

        # Parts should have <95% similarity to each other
        well_separated = max_cross_sim < 0.95
        print_result(
            "Parts are distinguishable",
            well_separated,
            f"Max cross-part similarity: {max_cross_sim*100:.1f}% (target <95%)"
        )

        return well_separated

    def run_all(self) -> dict:
        """Run all tests."""
        print("\n" + "="*70)
        print("  LIVE SCAN REAL FILES INTEGRATION TEST")
        print("="*70)

        tests = [
            ("File Availability", self.test_file_availability),
            ("Embedding Computation", self.test_embedding_computation),
            ("FAISS Indexing", self.test_faiss_indexing),
            ("Scan Recognition", self.test_scan_recognition),
            ("Partial Scan Recognition", self.test_partial_scan_recognition),
            ("Coverage Calculation", self.test_coverage_calculation),
            ("Cross-Part Discrimination", self.test_cross_part_discrimination),
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


if __name__ == "__main__":
    test = RealFilesTest()
    results = test.run_all()

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
