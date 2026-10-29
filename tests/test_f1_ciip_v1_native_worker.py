from __future__ import annotations

import dataclasses
import math
import re
import shutil
import struct
import subprocess

import pytest
import numpy as np

from research.f1_ciip_v1.native_worker import (
    DEFAULT_FLAGS,
    NativeBuildError,
    NativeNumericError,
    NativeWorker,
    build_and_load,
    build_native_worker,
)


@pytest.fixture(scope="module")
def worker(tmp_path_factory: pytest.TempPathFactory) -> NativeWorker:
    return build_and_load(tmp_path_factory.mktemp("ciip-native-worker"))


def test_toolchain_source_flags_and_binary_are_hash_bound(worker: NativeWorker) -> None:
    build = worker.build
    assert "Apple clang version" in build.compiler_version
    assert len(build.compiler_sha256) == 64
    assert len(build.source_sha256) == 64
    assert len(build.binary_sha256) == 64
    assert len(build.manifest_sha256) == 64
    assert build.flags[: len(DEFAULT_FLAGS)] == DEFAULT_FLAGS
    assert "-fno-fast-math" in build.flags
    assert "-ffp-contract=off" in build.flags
    assert "-frounding-math" in build.flags
    assert "-fno-associative-math" in build.flags
    assert "-fno-unsafe-math-optimizations" in build.flags


def test_native_identifier_and_numeric_known_answers_gate(worker: NativeWorker) -> None:
    worker.validate_known_answers()


def test_binary_hash_tamper_is_rejected(worker: NativeWorker, tmp_path) -> None:
    copy = tmp_path / worker.build.library_path.name
    shutil.copyfile(worker.build.library_path, copy)
    payload = bytearray(copy.read_bytes())
    payload[-1] ^= 1
    copy.write_bytes(payload)
    forged = dataclasses.replace(worker.build, library_path=copy)
    with pytest.raises(NativeBuildError, match="binary SHA-256"):
        NativeWorker(forged)


def test_identical_toolchain_builds_are_byte_reproducible(worker: NativeWorker, tmp_path) -> None:
    rebuilt = build_native_worker(tmp_path)
    assert rebuilt.source_sha256 == worker.build.source_sha256
    assert rebuilt.binary_sha256 == worker.build.binary_sha256
    assert rebuilt.manifest_sha256 == worker.build.manifest_sha256


def test_native_library_has_no_blas_openmp_or_accelerate_dependency(worker: NativeWorker) -> None:
    result = subprocess.run(
        ["otool", "-L", str(worker.build.library_path)], check=True, text=True, capture_output=True,
    )
    lowered = result.stdout.lower()
    assert "openmp" not in lowered
    assert "blas" not in lowered
    assert "accelerate" not in lowered


def test_native_machine_code_contains_no_fused_multiply_add(worker: NativeWorker) -> None:
    result = subprocess.run(
        ["otool", "-tvV", str(worker.build.library_path)], check=True, text=True, capture_output=True,
    )
    assert re.search(r"\b(?:fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)\b", result.stdout.lower()) is None


def test_signed_int128_checked_arithmetic(worker: NativeWorker) -> None:
    assert worker.i128_add((1 << 100) + 7, -(1 << 100)) == 7
    assert worker.i128_accumulate_product(5, 7, -8) == -51
    assert worker.i128_accumulate_product(-(1 << 126), 1 << 31, 1 << 31) == -(1 << 126) + (1 << 62)
    with pytest.raises(NativeNumericError, match="status 4"):
        worker.i128_add((1 << 127) - 1, 1)
    with pytest.raises(NativeNumericError, match="status 4"):
        worker.i128_add(-(1 << 127), -1)


def test_unsigned_int128_pow4_and_overflow(worker: NativeWorker) -> None:
    assert worker.u128_pow4(729) == 729**4
    assert worker.u128_pow4((1 << 32) - 1) == ((1 << 32) - 1) ** 4
    with pytest.raises(NativeNumericError, match="status 4"):
        worker.u128_pow4(1 << 32)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((9, 21, 37), (0, 0)),
        ((100, 1, 1), (-1, -1)),
        ((1, 100, 100), (1, -1)),
        ((1, 2, 100), (0, 1)),
    ],
)
def test_exact_uint128_growth_inequalities(worker: NativeWorker, counts, expected) -> None:
    assert worker.growth_gate(*counts) == expected


def test_density_and_opposite_balance_integer_gates(worker: NativeWorker) -> None:
    assert worker.density_gate(32, 0)
    assert not worker.density_gate(33, 0)
    assert worker.density_gate(320, 10)
    assert worker.balance_gate(11, 20)
    assert not worker.balance_gate(10, 20)
    with pytest.raises(NativeNumericError, match="status 4"):
        worker.balance_gate(0, 0)


def test_q24_40_uses_round_to_nearest_ties_even(worker: NativeWorker) -> None:
    quantum = 2.0**-40
    assert worker.quantize_q24_40(1.5 * quantum) == 2
    assert worker.quantize_q24_40(2.5 * quantum) == 2
    assert worker.quantize_q24_40(-1.5 * quantum) == -2
    assert worker.quantize_q24_40(-2.5 * quantum) == -2
    with pytest.raises(NativeNumericError, match="status 3"):
        worker.quantize_q24_40(math.inf)


def test_fixed_12_sweep_jacobi_known_answer_is_bit_deterministic(worker: NativeWorker) -> None:
    upper = (3.0, 1.0, 0.0, 2.0, 0.0, 1.0)
    first = worker.jacobi(upper)
    second = worker.jacobi(upper)
    assert struct.pack("<5d", *first.eigenvalues, first.trace, first.max_offdiag) == struct.pack(
        "<5d", *second.eigenvalues, second.trace, second.max_offdiag
    )
    assert first.eigenvalues == pytest.approx(
        (3.618033988749894, 1.3819660112501049, 1.0), rel=0.0, abs=2e-15
    )
    assert first.max_offdiag <= math.ldexp(first.trace, -40)


def test_jacobi_exact_zero_rotation_and_nonfinite_fail_closed(worker: NativeWorker) -> None:
    result = worker.jacobi((3.0, 0.0, 0.0, 2.0, 0.0, 1.0))
    assert result.eigenvalues == (3.0, 2.0, 1.0)
    assert result.max_offdiag == 0.0
    with pytest.raises(NativeNumericError, match="status 3"):
        worker.jacobi((3.0, math.nan, 0.0, 2.0, 0.0, 1.0))


def test_anisotropy_operation_order_known_answer(worker: NativeWorker) -> None:
    result = worker.anisotropy((3.618033988749895, 1.381966011250105, 1.0))
    assert result.scalar == pytest.approx(0.023725054618046786, rel=0.0, abs=1e-18)
    assert result.q == 1555
    assert result.filtration == 63979
    isotropic = worker.anisotropy((2.0, 2.0, 2.0))
    assert isotropic.scalar == 0.0
    assert isotropic.q == 0
    assert isotropic.filtration == 65534


def test_moments_to_tensor_isotropic_cube_known_answer(worker: NativeWorker) -> None:
    result = worker.moments_to_field(1, (0, 0, 0, 0, 0, 0, 0, 0, 0))
    # One occupied cube at the origin has dimensionless covariance I/12.
    assert result.result_code == 0
    assert result.eigenvalue_q == (91_625_968_981,) * 3
    assert result.trace_q == sum(result.eigenvalue_q)
    assert result.anisotropy_q == 0
    assert result.filtration == 65_534


def test_moments_to_tensor_classifies_trace_before_numeric(worker: NativeWorker) -> None:
    nonpositive = worker.moments_to_field(1, (0, 0, 0, -1, -1, -1, 0, 0, 0))
    assert nonpositive.result_code == 6
    assert nonpositive.trace_q == 0
    assert nonpositive.eigenvalue_q == (0, 0, 0)

    numeric = worker.moments_to_field(1, (0, 0, 0, 0, 0, 0, 1 << 62, 0, 0))
    assert numeric.result_code == 7
    assert numeric.trace_q > 0
    assert numeric.eigenvalue_q == (0, 0, 0)
    assert numeric.anisotropy_q == 0
    assert numeric.filtration == 65_534


def _morton(x: int, y: int, z: int) -> int:
    value = 0
    for bit in range(5):
        value |= ((x >> bit) & 1) << (3 * bit)
        value |= ((y >> bit) & 1) << (3 * bit + 1)
        value |= ((z >> bit) & 1) << (3 * bit + 2)
    return value


def _brute_support(points, raw_by_point, center, radius):
    support = []
    plus = minus = 0
    cx, cy, cz = center
    for point in points:
        x, y, z = point
        dx, dy, dz = x - cx, y - cy, z - cz
        if dx * dx + dy * dy + dz * dz > radius * radius:
            continue
        support.append(point)
        if (dx, dy, dz) != (0, 0, 0):
            first = dx if dx else (dy if dy else dz)
            plus += first > 0
            minus += first < 0
    moments = (
        sum(x for x, _, _ in support),
        sum(y for _, y, _ in support),
        sum(z for _, _, z in support),
        sum(x * x for x, _, _ in support),
        sum(y * y for _, y, _ in support),
        sum(z * z for _, _, z in support),
        sum(x * y for x, y, _ in support),
        sum(x * z for x, _, z in support),
        sum(y * z for _, y, z in support),
    )
    counts = sorted(raw_by_point[point] for point in support)
    p10 = counts[(len(counts) + 9) // 10 - 1]
    p90 = counts[(9 * len(counts) + 9) // 10 - 1]
    remaining = set(support)
    components = 0
    while remaining:
        components += 1
        queue = [remaining.pop()]
        while queue:
            x, y, z = queue.pop()
            for neighbor in ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z),
                             (x, y + 1, z), (x, y, z - 1), (x, y, z + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return len(support), moments, p10, p90, min(plus, minus), max(plus, minus), components


@pytest.mark.parametrize("radius", [2, 3, 4])
def test_bulk_fixed_grid_support_evidence_matches_brute_force(worker: NativeWorker, radius: int) -> None:
    points = [
        (0, 0, 0),
        (1, 0, 0),
        (10, 10, 10),
        (10, 11, 10),
        (11, 11, 10),
        (12, 10, 10),
        (13, 10, 10),
        (31, 31, 31),
    ]
    points.sort(key=lambda point: _morton(*point))
    raw_by_point = {point: 3 * index + 1 for index, point in enumerate(points)}
    occupied = np.zeros(32768, dtype=np.uint8)
    raw_counts = np.zeros(32768, dtype="<u4")
    for point in points:
        occupied[_morton(*point)] = 1
        raw_counts[_morton(*point)] = raw_by_point[point]
    leaf_xyz = np.asarray(points, dtype=np.uint8)

    evidence = worker.support_evidence(occupied, raw_counts, leaf_xyz, radius)
    for index, center in enumerate(points):
        expected = _brute_support(points, raw_by_point, center, radius)
        assert int(evidence.support_count[index]) == expected[0]
        assert tuple(map(int, evidence.lattice_sums[index])) == expected[1]
        assert int(evidence.raw_p10[index]) == expected[2]
        assert int(evidence.raw_p90[index]) == expected[3]
        assert int(evidence.balance_numerator[index]) == expected[4]
        assert int(evidence.balance_denominator[index]) == expected[5]
        assert int(evidence.component_count[index]) == expected[6]


def test_bulk_support_input_contract_fails_closed(worker: NativeWorker) -> None:
    occupied = np.zeros(32768, dtype=np.uint8)
    raw_counts = np.zeros(32768, dtype="<u4")
    points = [(2, 2, 2), (3, 2, 2)]
    for index, point in enumerate(points, start=1):
        occupied[_morton(*point)] = 1
        raw_counts[_morton(*point)] = index
    leaf_xyz = np.asarray(sorted(points, key=lambda point: _morton(*point)), dtype=np.uint8)

    with pytest.raises(ValueError, match="multiplier"):
        worker.support_evidence(occupied, raw_counts, leaf_xyz, 5)
    with pytest.raises(ValueError, match="uint8"):
        worker.support_evidence(occupied.astype(np.int8), raw_counts, leaf_xyz, 2)
    with pytest.raises(NativeNumericError, match="status 4"):
        worker.support_evidence(occupied, raw_counts, leaf_xyz[::-1].copy(), 2)
    broken_counts = raw_counts.copy()
    broken_counts[_morton(*points[0])] = 0
    with pytest.raises(NativeNumericError, match="status 4"):
        worker.support_evidence(occupied, broken_counts, leaf_xyz, 2)
