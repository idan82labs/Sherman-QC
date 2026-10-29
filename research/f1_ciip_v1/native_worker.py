"""Reproducible build and ctypes binding for the frozen V7 native worker."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SOURCE_PATH = Path(__file__).with_name("native") / "semantic_worker.cpp"
DEFAULT_FLAGS = (
    "-std=c++17",
    "-O2",
    "-fPIC",
    "-shared",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-frounding-math",
    "-fno-associative-math",
    "-fno-unsafe-math-optimizations",
    "-fno-finite-math-only",
    "-Werror",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
)
DARWIN_LINK_FLAGS = ("-Wl,-install_name,@rpath/libciip_v1_semantic_worker.dylib",)


class NativeBuildError(RuntimeError):
    pass


class NativeNumericError(ArithmeticError):
    def __init__(self, operation: str, status: int):
        super().__init__(f"{operation} failed with native status {status}")
        self.operation = operation
        self.status = status


class I128Parts(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint64), ("hi", ctypes.c_int64)]


class U128Parts(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint64), ("hi", ctypes.c_uint64)]


class JacobiOutput(ctypes.Structure):
    _fields_ = [
        ("lambda1", ctypes.c_double),
        ("lambda2", ctypes.c_double),
        ("lambda3", ctypes.c_double),
        ("trace", ctypes.c_double),
        ("max_offdiag", ctypes.c_double),
    ]


class AnisotropyOutput(ctypes.Structure):
    _fields_ = [("scalar", ctypes.c_double), ("q", ctypes.c_uint16), ("filtration", ctypes.c_uint16)]


@dataclass(frozen=True)
class NativeBuild:
    library_path: Path
    compiler_path: str
    compiler_version: str
    compiler_sha256: str
    flags: tuple[str, ...]
    source_sha256: str
    binary_sha256: str

    def manifest(self) -> dict[str, object]:
        return {
            "compiler_path": self.compiler_path,
            "compiler_version": self.compiler_version,
            "compiler_sha256": self.compiler_sha256,
            "flags": list(self.flags),
            "source_sha256": self.source_sha256,
            "binary_sha256": self.binary_sha256,
        }

    def canonical_manifest_bytes(self) -> bytes:
        return json.dumps(self.manifest(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_manifest_bytes()).hexdigest()


@dataclass(frozen=True)
class JacobiResult:
    eigenvalues: tuple[float, float, float]
    trace: float
    max_offdiag: float


@dataclass(frozen=True)
class AnisotropyResult:
    scalar: float
    q: int
    filtration: int


@dataclass(frozen=True)
class SupportEvidence:
    support_count: object
    lattice_sums: object
    raw_p10: object
    raw_p90: object
    balance_numerator: object
    balance_denominator: object
    component_count: object


@dataclass(frozen=True)
class StagedFieldEvidence:
    """Complete baseline numeric evidence with frozen first-rejection state."""

    support_count: object
    lattice_sums: object
    raw_p10: object
    raw_p90: object
    balance_numerator: object
    balance_denominator: object
    component_count: object
    growth12: object
    growth23: object
    trace_q: object
    lambda1_q: object
    lambda2_q: object
    lambda3_q: object
    anisotropy_q: object
    filtration: object
    stage_mask: object
    first_rejection: object


@dataclass(frozen=True)
class TensorFieldResult:
    trace_q: int
    eigenvalue_q: tuple[int, int, int]
    anisotropy_q: int
    filtration: int
    result_code: int


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apple_clang() -> str:
    xcrun = shutil.which("xcrun")
    if xcrun:
        result = subprocess.run(
            [xcrun, "--find", "clang++"], check=True, text=True, capture_output=True,
        )
        compiler = result.stdout.strip()
    else:
        compiler = shutil.which("clang++") or ""
    if not compiler:
        raise NativeBuildError("Apple clang++ was not found")
    # Preserve the clang++ driver name. Resolving the Xcode symlink to `clang`
    # silently switches driver mode and drops the C++ standard-library paths.
    return str(Path(compiler))


def build_native_worker(output_dir: str | os.PathLike[str]) -> NativeBuild:
    """Compile the worker with the frozen flags and return its hash binding."""
    source = SOURCE_PATH.resolve()
    if not source.is_file():
        raise NativeBuildError(f"native source is missing: {source}")
    compiler = _apple_clang()
    version_result = subprocess.run(
        [compiler, "--version"], check=True, text=True, capture_output=True,
    )
    compiler_version = version_result.stdout.strip()
    if platform.system() == "Darwin" and "Apple clang" not in compiler_version:
        raise NativeBuildError("frozen worker requires Apple clang on Darwin")

    source_sha = _sha256_path(source)
    platform_flags: tuple[str, ...] = ()
    if platform.system() == "Darwin":
        xcrun = shutil.which("xcrun")
        if not xcrun:
            raise NativeBuildError("xcrun is required to bind the macOS SDK")
        sdk = subprocess.run(
            [xcrun, "--sdk", "macosx", "--show-sdk-path"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        platform_flags = ("-isysroot", sdk, *DARWIN_LINK_FLAGS)
    flags = DEFAULT_FLAGS + platform_flags
    flags_string = " ".join(flags)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library_path = destination / f"libciip_v1_semantic_worker_{source_sha[:16]}{suffix}"
    command = [
        compiler,
        *flags,
        f'-DCIIP_SOURCE_SHA256="{source_sha}"',
        f'-DCIIP_BUILD_FLAGS="{flags_string}"',
        str(source),
        "-o",
        str(library_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise NativeBuildError(
            "native worker compilation failed\n"
            f"command: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    build = NativeBuild(
        library_path=library_path,
        compiler_path=compiler,
        compiler_version=compiler_version,
        compiler_sha256=_sha256_path(Path(compiler).resolve()),
        flags=tuple(flags),
        source_sha256=source_sha,
        binary_sha256=_sha256_path(library_path),
    )
    _verify_embedded_build(build)
    return build


def _configure_library(library_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(library_path))
    for name in ("ciip_v1_source_sha256", "ciip_v1_build_flags", "ciip_v1_compiler_version"):
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_char_p
    library.ciip_v1_i128_add.argtypes = [I128Parts, I128Parts, ctypes.POINTER(I128Parts)]
    library.ciip_v1_i128_add.restype = ctypes.c_int
    library.ciip_v1_i128_accumulate_product.argtypes = [
        I128Parts, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(I128Parts),
    ]
    library.ciip_v1_i128_accumulate_product.restype = ctypes.c_int
    library.ciip_v1_u128_pow4.argtypes = [ctypes.c_uint64, ctypes.POINTER(U128Parts)]
    library.ciip_v1_u128_pow4.restype = ctypes.c_int
    library.ciip_v1_growth_gate.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ]
    library.ciip_v1_growth_gate.restype = ctypes.c_int
    library.ciip_v1_density_gate.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8)]
    library.ciip_v1_density_gate.restype = ctypes.c_int
    library.ciip_v1_balance_gate.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8)]
    library.ciip_v1_balance_gate.restype = ctypes.c_int
    library.ciip_v1_quantize_q24_40.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int64)]
    library.ciip_v1_quantize_q24_40.restype = ctypes.c_int
    library.ciip_v1_jacobi_eigenvalues.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(JacobiOutput)]
    library.ciip_v1_jacobi_eigenvalues.restype = ctypes.c_int
    library.ciip_v1_anisotropy.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(AnisotropyOutput)]
    library.ciip_v1_anisotropy.restype = ctypes.c_int
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    library.ciip_v1_support_evidence.argtypes = [
        u8p, u32p, ctypes.c_uint32, u8p, ctypes.c_uint32, ctypes.c_uint8,
        u32p, i64p, i64p, i64p, i64p, i64p, i64p, i64p, i64p, i64p,
        u32p, u32p, u32p, u32p, u32p,
    ]
    library.ciip_v1_support_evidence.restype = ctypes.c_int
    u16p = ctypes.POINTER(ctypes.c_uint16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    library.ciip_v1_support_stage.argtypes = [
        u8p, u32p, ctypes.c_uint32, u8p, ctypes.c_uint32, ctypes.c_uint8,
        ctypes.c_uint8, u8p, u32p, i64p, i64p, i64p, i64p, i64p, i64p,
        i64p, i64p, i64p, u32p, u32p, u32p, u32p, u32p, u8p,
    ]
    library.ciip_v1_support_stage.restype = ctypes.c_int
    library.ciip_v1_growth_gate_masked.argtypes = [
        u32p, u32p, u32p, u8p, ctypes.c_uint32, i32p, i32p, u8p,
    ]
    library.ciip_v1_growth_gate_masked.restype = ctypes.c_int
    library.ciip_v1_moments_to_field_masked.argtypes = [
        u32p, i64p, i64p, i64p, i64p, i64p, i64p, i64p, i64p, i64p,
        u8p, ctypes.c_uint32, i64p, i64p, i64p, i64p, u16p, u16p, u8p,
    ]
    library.ciip_v1_moments_to_field_masked.restype = ctypes.c_int
    library.ciip_v1_build_adjacency.argtypes = [
        u8p, u64p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, u32p,
    ]
    library.ciip_v1_build_adjacency.restype = ctypes.c_int
    library.ciip_v1_h0_persistence.argtypes = [
        u8p, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, u32p,
    ]
    library.ciip_v1_h0_persistence.restype = ctypes.c_int
    library.ciip_v1_validate_persistence_collision_domain.argtypes = [
        u8p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ]
    library.ciip_v1_validate_persistence_collision_domain.restype = ctypes.c_int
    library.ciip_v1_validate_known_answers.argtypes = []
    library.ciip_v1_validate_known_answers.restype = ctypes.c_int
    return library


def _decode_c_string(value: bytes | None, label: str) -> str:
    if value is None:
        raise NativeBuildError(f"native worker omitted {label}")
    return value.decode("utf-8")


def _verify_embedded_build(build: NativeBuild) -> ctypes.CDLL:
    if _sha256_path(build.library_path) != build.binary_sha256:
        raise NativeBuildError("native worker binary SHA-256 mismatch")
    if _sha256_path(SOURCE_PATH) != build.source_sha256:
        raise NativeBuildError("native worker source SHA-256 changed after build")
    if _sha256_path(Path(build.compiler_path).resolve()) != build.compiler_sha256:
        raise NativeBuildError("native worker compiler SHA-256 mismatch")
    library = _configure_library(build.library_path)
    embedded_source = _decode_c_string(library.ciip_v1_source_sha256(), "source hash")
    embedded_flags = _decode_c_string(library.ciip_v1_build_flags(), "build flags")
    embedded_compiler = _decode_c_string(library.ciip_v1_compiler_version(), "compiler version")
    if embedded_source != build.source_sha256:
        raise NativeBuildError("embedded source SHA-256 mismatch")
    if embedded_flags != " ".join(build.flags):
        raise NativeBuildError("embedded build flags mismatch")
    if not embedded_compiler or embedded_compiler not in build.compiler_version:
        raise NativeBuildError("embedded compiler identity mismatch")
    return library


def _i128_parts(value: int) -> I128Parts:
    if not -(1 << 127) <= value < 1 << 127:
        raise OverflowError("value is outside signed int128")
    bits = value & ((1 << 128) - 1)
    high = (bits >> 64) & ((1 << 64) - 1)
    if high >= 1 << 63:
        high -= 1 << 64
    return I128Parts(bits & ((1 << 64) - 1), high)


def _parts_i128(value: I128Parts) -> int:
    bits = (value.hi & ((1 << 64) - 1)) << 64 | value.lo
    return bits - (1 << 128) if bits >= 1 << 127 else bits


def _parts_u128(value: U128Parts) -> int:
    return value.hi << 64 | value.lo


class NativeWorker:
    def __init__(self, build: NativeBuild):
        self.build = build
        self._library = _verify_embedded_build(build)
        self.validate_known_answers()

    @staticmethod
    def _check(operation: str, status: int) -> None:
        if status != 0:
            raise NativeNumericError(operation, status)

    def validate_known_answers(self) -> None:
        """Fail before geometry if the native identifier/numeric KAT drifts."""

        self._check(
            "validate_known_answers",
            self._library.ciip_v1_validate_known_answers(),
        )

    def i128_add(self, left: int, right: int) -> int:
        out = I128Parts()
        status = self._library.ciip_v1_i128_add(_i128_parts(left), _i128_parts(right), ctypes.byref(out))
        self._check("i128_add", status)
        return _parts_i128(out)

    def i128_accumulate_product(self, accumulator: int, left: int, right: int) -> int:
        out = I128Parts()
        status = self._library.ciip_v1_i128_accumulate_product(
            _i128_parts(accumulator), left, right, ctypes.byref(out),
        )
        self._check("i128_accumulate_product", status)
        return _parts_i128(out)

    def u128_pow4(self, value: int) -> int:
        out = U128Parts()
        status = self._library.ciip_v1_u128_pow4(value, ctypes.byref(out))
        self._check("u128_pow4", status)
        return _parts_u128(out)

    def growth_gate(self, n1: int, n2: int, n3: int) -> tuple[int, int]:
        growth12 = ctypes.c_int32()
        growth23 = ctypes.c_int32()
        status = self._library.ciip_v1_growth_gate(n1, n2, n3, ctypes.byref(growth12), ctypes.byref(growth23))
        self._check("growth_gate", status)
        return growth12.value, growth23.value

    def density_gate(self, p90: int, p10: int) -> bool:
        result = ctypes.c_uint8()
        status = self._library.ciip_v1_density_gate(p90, p10, ctypes.byref(result))
        self._check("density_gate", status)
        return bool(result.value)

    def balance_gate(self, numerator: int, denominator: int) -> bool:
        result = ctypes.c_uint8()
        status = self._library.ciip_v1_balance_gate(numerator, denominator, ctypes.byref(result))
        self._check("balance_gate", status)
        return bool(result.value)

    def quantize_q24_40(self, value: float) -> int:
        result = ctypes.c_int64()
        status = self._library.ciip_v1_quantize_q24_40(value, ctypes.byref(result))
        self._check("quantize_q24_40", status)
        return result.value

    def jacobi(self, upper_triangle: Sequence[float]) -> JacobiResult:
        if len(upper_triangle) != 6:
            raise ValueError("upper_triangle must contain a00,a01,a02,a11,a12,a22")
        values = (ctypes.c_double * 6)(*map(float, upper_triangle))
        result = JacobiOutput()
        status = self._library.ciip_v1_jacobi_eigenvalues(values, ctypes.byref(result))
        self._check("jacobi", status)
        return JacobiResult(
            (result.lambda1, result.lambda2, result.lambda3), result.trace, result.max_offdiag,
        )

    def anisotropy(self, eigenvalues: Sequence[float]) -> AnisotropyResult:
        if len(eigenvalues) != 3:
            raise ValueError("eigenvalues must contain lambda1,lambda2,lambda3")
        values = (ctypes.c_double * 3)(*map(float, eigenvalues))
        result = AnisotropyOutput()
        status = self._library.ciip_v1_anisotropy(values, ctypes.byref(result))
        self._check("anisotropy", status)
        return AnisotropyResult(result.scalar, result.q, result.filtration)

    def moments_to_field(self, support_count: int, moments: Sequence[int]) -> TensorFieldResult:
        """Construct one authoritative dimensionless tensor from integer moments."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - project runtime includes NumPy
            raise RuntimeError("NumPy is required for moments_to_field") from exc
        if not 0 < support_count < 1 << 32 or len(moments) != 9:
            raise ValueError("support_count and nine integer moments are required")
        counts = np.asarray([support_count], dtype="<u4")
        moment_arrays = [np.asarray([value], dtype="<i8") for value in moments]
        active = np.ones(1, dtype="u1")
        trace_q = np.zeros(1, dtype="<i8")
        eigen = [np.zeros(1, dtype="<i8") for _ in range(3)]
        anisotropy_q = np.zeros(1, dtype="<u2")
        filtration = np.full(1, 65_534, dtype="<u2")
        result_code = np.zeros(1, dtype="u1")
        u8p = ctypes.POINTER(ctypes.c_uint8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        i64p = ctypes.POINTER(ctypes.c_int64)
        status = self._library.ciip_v1_moments_to_field_masked(
            counts.ctypes.data_as(u32p),
            *(values.ctypes.data_as(i64p) for values in moment_arrays),
            active.ctypes.data_as(u8p),
            1,
            trace_q.ctypes.data_as(i64p),
            *(values.ctypes.data_as(i64p) for values in eigen),
            anisotropy_q.ctypes.data_as(u16p),
            filtration.ctypes.data_as(u16p),
            result_code.ctypes.data_as(u8p),
        )
        self._check("moments_to_field_masked", status)
        return TensorFieldResult(
            int(trace_q[0]),
            tuple(int(values[0]) for values in eigen),
            int(anisotropy_q[0]),
            int(filtration[0]),
            int(result_code[0]),
        )

    def build_adjacency(self, case_digest: bytes, leaf_mortons: object) -> object:
        """Build the complete canonical adjacency table and IDs in native fixed arrays."""
        try:
            import numpy as np
            from .container import numpy_dtype
        except ImportError as exc:  # pragma: no cover - project runtime includes NumPy
            raise RuntimeError("NumPy is required for build_adjacency") from exc
        case = bytes(case_digest)
        if len(case) != 32:
            raise ValueError("case_digest must contain 32 bytes")
        mortons = np.asarray(leaf_mortons)
        if mortons.dtype != np.dtype("<u8") or mortons.ndim != 1 or not mortons.flags.c_contiguous:
            raise ValueError("leaf_mortons must be a C-contiguous little-endian uint64 vector")
        occupied_count = len(mortons)
        if not 0 < occupied_count <= 32768:
            raise ValueError("leaf_mortons count must be in 1..32768")
        rows = np.zeros(11 * occupied_count, dtype=numpy_dtype("space_scale_adjacency"))
        output_count = ctypes.c_uint32()
        case_array = (ctypes.c_uint8 * 32).from_buffer_copy(case)
        status = self._library.ciip_v1_build_adjacency(
            case_array,
            mortons.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            occupied_count,
            ctypes.c_void_p(rows.ctypes.data),
            len(rows),
            ctypes.byref(output_count),
        )
        self._check("build_adjacency", status)
        return rows[: output_count.value]

    def h0_persistence(
        self,
        case_digest: bytes,
        arm_u8: int,
        field_rows_for_arm: object,
        adjacency: object,
    ) -> tuple[object, int]:
        """Run the complete lower-star H0 sweep and identifier emission natively."""
        try:
            import numpy as np
            from .container import numpy_dtype
        except ImportError as exc:  # pragma: no cover - project runtime includes NumPy
            raise RuntimeError("NumPy is required for h0_persistence") from exc
        case = bytes(case_digest)
        if len(case) != 32 or arm_u8 not in (0, 1):
            raise ValueError("case_digest/arm is outside the frozen domain")
        fields = np.asarray(field_rows_for_arm)
        edges = np.asarray(adjacency)
        field_dtype = numpy_dtype("cell_scale_field")
        edge_dtype = numpy_dtype("space_scale_adjacency")
        if fields.dtype != field_dtype or fields.ndim != 1 or not fields.flags.c_contiguous:
            raise ValueError("fields must use the exact contiguous frozen dtype")
        if edges.dtype != edge_dtype or edges.ndim != 1 or not edges.flags.c_contiguous:
            raise ValueError("adjacency must use the exact contiguous frozen dtype")
        output = np.zeros(len(fields), dtype=numpy_dtype("persistence_disposition"))
        attributed = ctypes.c_uint32()
        case_array = (ctypes.c_uint8 * 32).from_buffer_copy(case)
        status = self._library.ciip_v1_h0_persistence(
            case_array,
            arm_u8,
            ctypes.c_void_p(fields.ctypes.data),
            len(fields),
            ctypes.c_void_p(edges.ctypes.data),
            len(edges),
            ctypes.c_void_p(output.ctypes.data),
            ctypes.byref(attributed),
        )
        self._check("h0_persistence", status)
        return output, int(attributed.value)

    def validate_persistence_collision_domain(
        self,
        case_digest: bytes,
        field_rows: object,
        persistence_rows: object,
    ) -> None:
        """Check all arm-bound truncated IDs across both complete H0 arms."""
        try:
            import numpy as np
            from .container import numpy_dtype
        except ImportError as exc:  # pragma: no cover - project runtime includes NumPy
            raise RuntimeError("NumPy is required for persistence collision validation") from exc
        case = bytes(case_digest)
        fields = np.asarray(field_rows)
        persistence = np.asarray(persistence_rows)
        if len(case) != 32 or fields.dtype != numpy_dtype("cell_scale_field") or (
            persistence.dtype != numpy_dtype("persistence_disposition")
        ) or fields.ndim != 1 or persistence.ndim != 1 or len(fields) != len(persistence) or (
            not fields.flags.c_contiguous or not persistence.flags.c_contiguous
        ):
            raise ValueError("complete field/persistence arrays are required")
        case_array = (ctypes.c_uint8 * 32).from_buffer_copy(case)
        status = self._library.ciip_v1_validate_persistence_collision_domain(
            case_array,
            ctypes.c_void_p(fields.ctypes.data),
            ctypes.c_void_p(persistence.ctypes.data),
            len(fields),
        )
        self._check("validate_persistence_collision_domain", status)

    def support_evidence(
        self,
        occupied: object,
        raw_counts: object,
        leaf_xyz: object,
        multiplier: int,
    ) -> SupportEvidence:
        """Compute all support evidence in one native bulk call.

        ``occupied`` and ``raw_counts`` are indexed by the frozen 15-bit Morton
        code. ``leaf_xyz`` must enumerate the same occupied cells in strictly
        increasing Morton order. Lattice sums use absolute integer cell-center
        indices and columns ``x,y,z,xx,yy,zz,xy,xz,yz``.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - project runtime includes NumPy
            raise RuntimeError("NumPy is required for support_evidence") from exc
        occupied_array = np.asarray(occupied)
        raw_count_array = np.asarray(raw_counts)
        leaf_array = np.asarray(leaf_xyz)
        if occupied_array.dtype != np.dtype("u1") or occupied_array.shape != (32768,) or not occupied_array.flags.c_contiguous:
            raise ValueError("occupied must be a C-contiguous uint8[32768] array")
        if raw_count_array.dtype != np.dtype("<u4") or raw_count_array.shape != (32768,) or not raw_count_array.flags.c_contiguous:
            raise ValueError("raw_counts must be a C-contiguous little-endian uint32[32768] array")
        if leaf_array.dtype != np.dtype("u1") or leaf_array.ndim != 2 or leaf_array.shape[1] != 3 or not leaf_array.flags.c_contiguous:
            raise ValueError("leaf_xyz must be a C-contiguous uint8[O,3] array")
        leaf_count = len(leaf_array)
        if not 0 < leaf_count <= 32768:
            raise ValueError("leaf_xyz count must be in 1..32768")
        if multiplier not in (2, 3, 4):
            raise ValueError("multiplier must be one of 2, 3, 4")

        support_count = np.empty(leaf_count, dtype="<u4")
        moment_arrays = [np.empty(leaf_count, dtype="<i8") for _ in range(9)]
        raw_p10 = np.empty(leaf_count, dtype="<u4")
        raw_p90 = np.empty(leaf_count, dtype="<u4")
        balance_numerator = np.empty(leaf_count, dtype="<u4")
        balance_denominator = np.empty(leaf_count, dtype="<u4")
        component_count = np.empty(leaf_count, dtype="<u4")
        u8p = ctypes.POINTER(ctypes.c_uint8)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        i64p = ctypes.POINTER(ctypes.c_int64)
        status = self._library.ciip_v1_support_evidence(
            occupied_array.ctypes.data_as(u8p),
            raw_count_array.ctypes.data_as(u32p),
            32768,
            leaf_array.ctypes.data_as(u8p),
            leaf_count,
            multiplier,
            support_count.ctypes.data_as(u32p),
            *(values.ctypes.data_as(i64p) for values in moment_arrays),
            raw_p10.ctypes.data_as(u32p),
            raw_p90.ctypes.data_as(u32p),
            balance_numerator.ctypes.data_as(u32p),
            balance_denominator.ctypes.data_as(u32p),
            component_count.ctypes.data_as(u32p),
        )
        self._check("support_evidence", status)
        lattice_sums = np.column_stack(moment_arrays)
        return SupportEvidence(
            support_count,
            lattice_sums,
            raw_p10,
            raw_p90,
            balance_numerator,
            balance_denominator,
            component_count,
        )

    def staged_field_evidence(
        self,
        occupied: object,
        raw_counts: object,
        leaf_xyz: object,
    ) -> StagedFieldEvidence:
        """Evaluate the seven frozen stages in order over survivor masks.

        Arrays use scale-major order.  No native call for a later stage receives
        a row rejected by an earlier stage.  Tensor construction, Jacobi,
        quantization and anisotropy all execute in the locked native binary64
        environment.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - project runtime includes NumPy
            raise RuntimeError("NumPy is required for staged_field_evidence") from exc

        occupied_array = np.asarray(occupied)
        raw_count_array = np.asarray(raw_counts)
        leaf_array = np.asarray(leaf_xyz)
        if occupied_array.dtype != np.dtype("u1") or occupied_array.shape != (32768,) or not occupied_array.flags.c_contiguous:
            raise ValueError("occupied must be a C-contiguous uint8[32768] array")
        if raw_count_array.dtype != np.dtype("<u4") or raw_count_array.shape != (32768,) or not raw_count_array.flags.c_contiguous:
            raise ValueError("raw_counts must be a C-contiguous little-endian uint32[32768] array")
        if leaf_array.dtype != np.dtype("u1") or leaf_array.ndim != 2 or leaf_array.shape[1] != 3 or not leaf_array.flags.c_contiguous:
            raise ValueError("leaf_xyz must be a C-contiguous uint8[O,3] array")
        leaf_count = len(leaf_array)
        if not 0 < leaf_count <= 32768:
            raise ValueError("leaf_xyz count must be in 1..32768")

        shape = (3, leaf_count)
        support_count = np.zeros(shape, dtype="<u4")
        moments = [np.zeros(shape, dtype="<i8") for _ in range(9)]
        raw_p10 = np.zeros(shape, dtype="<u4")
        raw_p90 = np.zeros(shape, dtype="<u4")
        balance_numerator = np.zeros(shape, dtype="<u4")
        balance_denominator = np.zeros(shape, dtype="<u4")
        component_count = np.zeros(shape, dtype="<u4")
        growth12 = np.full(shape, -2_147_483_648, dtype="<i4")
        growth23 = np.full(shape, -2_147_483_648, dtype="<i4")
        trace_q = np.zeros(shape, dtype="<i8")
        lambda1_q = np.zeros(shape, dtype="<i8")
        lambda2_q = np.zeros(shape, dtype="<i8")
        lambda3_q = np.zeros(shape, dtype="<i8")
        anisotropy_q = np.zeros(shape, dtype="<u2")
        filtration = np.full(shape, 65_534, dtype="<u2")
        stage_mask = np.zeros(shape, dtype="<u4")
        first_rejection = np.zeros(shape, dtype="<u2")
        active = np.ones(shape, dtype="u1")

        u8p = ctypes.POINTER(ctypes.c_uint8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        i32p = ctypes.POINTER(ctypes.c_int32)
        i64p = ctypes.POINTER(ctypes.c_int64)
        occupied_ptr = occupied_array.ctypes.data_as(u8p)
        raw_ptr = raw_count_array.ctypes.data_as(u32p)
        leaf_ptr = leaf_array.ctypes.data_as(u8p)

        def support_stage(scale: int, stage: int, survivor_mask: object) -> object:
            survivor = np.ascontiguousarray(survivor_mask, dtype="u1")
            passes = np.zeros(leaf_count, dtype="u1")
            status = self._library.ciip_v1_support_stage(
                occupied_ptr,
                raw_ptr,
                32768,
                leaf_ptr,
                leaf_count,
                scale + 2,
                stage,
                survivor.ctypes.data_as(u8p),
                support_count[scale].ctypes.data_as(u32p),
                *(values[scale].ctypes.data_as(i64p) for values in moments),
                raw_p10[scale].ctypes.data_as(u32p),
                raw_p90[scale].ctypes.data_as(u32p),
                balance_numerator[scale].ctypes.data_as(u32p),
                balance_denominator[scale].ctypes.data_as(u32p),
                component_count[scale].ctypes.data_as(u32p),
                passes.ctypes.data_as(u8p),
            )
            self._check(f"support_stage_{stage}", status)
            return passes

        def advance_stage(scale: int, bit: int, rejection: int, passes: object) -> None:
            prior = active[scale].astype(bool, copy=False)
            stage_mask[scale, prior] |= np.uint32(1 << bit)
            failed = prior & ~np.asarray(passes, dtype=bool)
            first_rejection[scale, failed] = np.uint16(rejection)
            active[scale] = np.asarray(passes, dtype=np.uint8)

        # Support, connectivity and opposite-balance stages.
        for scale in range(3):
            passes = support_stage(scale, 0, active[scale])
            advance_stage(scale, 0, 1, passes)
            passes = support_stage(scale, 1, active[scale])
            advance_stage(scale, 1, 2, passes)
            passes = support_stage(scale, 2, active[scale])
            advance_stage(scale, 2, 3, passes)

        # Nested growth is shared evidence but is evaluated only for rows that
        # survived that row's first three stages.
        for scale in range(3):
            prior = np.ascontiguousarray(active[scale], dtype="u1")
            passes = np.zeros(leaf_count, dtype="u1")
            status = self._library.ciip_v1_growth_gate_masked(
                support_count[0].ctypes.data_as(u32p),
                support_count[1].ctypes.data_as(u32p),
                support_count[2].ctypes.data_as(u32p),
                prior.ctypes.data_as(u8p),
                leaf_count,
                growth12[scale].ctypes.data_as(i32p),
                growth23[scale].ctypes.data_as(i32p),
                passes.ctypes.data_as(u8p),
            )
            self._check("growth_gate_masked", status)
            advance_stage(scale, 3, 4, passes)

        # Density and moment construction stages.
        for scale in range(3):
            passes = support_stage(scale, 4, active[scale])
            advance_stage(scale, 4, 5, passes)
            prior = active[scale].astype(bool, copy=False)
            stage_mask[scale, prior] |= np.uint32(1 << 5)
            # Stage 5 computes exact integer moments only. Positive trace and
            # numeric validity are classified together by the tensor batch.
            support_stage(scale, 5, active[scale])

        flat_active = np.ascontiguousarray(active.reshape(-1), dtype="u1")
        result_code = np.zeros(3 * leaf_count, dtype="u1")
        status = self._library.ciip_v1_moments_to_field_masked(
            support_count.reshape(-1).ctypes.data_as(u32p),
            *(values.reshape(-1).ctypes.data_as(i64p) for values in moments),
            flat_active.ctypes.data_as(u8p),
            3 * leaf_count,
            trace_q.reshape(-1).ctypes.data_as(i64p),
            lambda1_q.reshape(-1).ctypes.data_as(i64p),
            lambda2_q.reshape(-1).ctypes.data_as(i64p),
            lambda3_q.reshape(-1).ctypes.data_as(i64p),
            anisotropy_q.reshape(-1).ctypes.data_as(u16p),
            filtration.reshape(-1).ctypes.data_as(u16p),
            result_code.ctypes.data_as(u8p),
        )
        self._check("moments_to_field_masked", status)
        result_code = result_code.reshape(shape)
        for scale in range(3):
            prior = active[scale].astype(bool, copy=False)
            trace_failed = prior & (result_code[scale] == 6)
            numeric_evaluated = prior & ~trace_failed
            stage_mask[scale, numeric_evaluated] |= np.uint32(1 << 6)
            first_rejection[scale, trace_failed] = np.uint16(6)
            numeric_failed = prior & (result_code[scale] == 7)
            first_rejection[scale, numeric_failed] = np.uint16(7)
            active[scale, trace_failed | numeric_failed] = 0

        lattice_sums = np.column_stack([values.reshape(-1) for values in moments])
        return StagedFieldEvidence(
            support_count.reshape(-1),
            lattice_sums,
            raw_p10.reshape(-1),
            raw_p90.reshape(-1),
            balance_numerator.reshape(-1),
            balance_denominator.reshape(-1),
            component_count.reshape(-1),
            growth12.reshape(-1),
            growth23.reshape(-1),
            trace_q.reshape(-1),
            lambda1_q.reshape(-1),
            lambda2_q.reshape(-1),
            lambda3_q.reshape(-1),
            anisotropy_q.reshape(-1),
            filtration.reshape(-1),
            stage_mask.reshape(-1),
            first_rejection.reshape(-1),
        )


def build_and_load(output_dir: str | os.PathLike[str]) -> NativeWorker:
    return NativeWorker(build_native_worker(output_dir))


def manifest_digest(build: NativeBuild) -> str:
    return build.manifest_sha256
