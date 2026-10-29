"""One-shot byte-only CAD corpus inventory; no native/geometry imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
CAD_ROOT = ROOT.parent / "source-assets/02_CAD_QC_3D"
CONFIG = {"entries": 3000, "cad_files": 500, "file_bytes": 256 * 1024**2,
          "total_bytes": 4 * 1024**3, "seconds": 120, "rss_bytes": 1024**3}
CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs",
                  ".dxf", ".dwg", ".sldprt", ".sldasm", ".ipt", ".iam", ".3dm", ".fcstd"}
AMBIGUOUS_MESH_EXTENSIONS = {".ply", ".stl", ".obj"}
SUPPORTED = {".step", ".stp"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_exclusive(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze(attempt, root=CAD_ROOT, config=None):
    # Deliberately lexical: never resolve/stat/list/open the corpus during freeze.
    root = os.path.abspath(os.fspath(root))
    attempt = Path(attempt)
    if os.path.commonpath([root, os.path.abspath(attempt)]) == root:
        raise ValueError("attempt must be outside corpus")
    attempt.mkdir(parents=True, exist_ok=False)
    record = {"schema": "vendor-cad-corpus-inventory-v2", "root": root,
              "config": dict(CONFIG if config is None else config),
              "source_sha256": digest(__file__), "scope": "byte-only; no CAD imports or labels",
              "excluded_names": ["incoming", "EHO", "HOLDOUT10"],
              "scan_name_policy": "case-insensitive SCAN in any relative path component: metadata only",
              "ambiguous_mesh_policy": "metadata only; no provenance grant in V2",
              "ambiguous_mesh_extensions": sorted(AMBIGUOUS_MESH_EXTENSIONS),
              "cad_extensions": sorted(CAD_EXTENSIONS), "supported_extensions": sorted(SUPPORTED)}
    write_exclusive(attempt / "FROZEN.json", record)
    return record


def excluded(name):
    normalized = name.casefold().replace("-", "_")
    return any(token in normalized for token in ("incoming", "eho", "holdout10"))


def open_root(root):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(root).parts[1:]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd
    except BaseException:
        os.close(fd)
        raise


def execute(attempt, expected_freeze_sha256):
    attempt = Path(attempt)
    if digest(attempt / "FROZEN.json") != expected_freeze_sha256:
        raise ValueError("freeze digest mismatch")
    frozen = json.loads((attempt / "FROZEN.json").read_text())
    if frozen["source_sha256"] != digest(__file__):
        raise ValueError("source changed after freeze")
    write_exclusive(attempt / "EXECUTION.json", {"started_unix": time.time(), "freeze_sha256": expected_freeze_sha256})
    started = time.monotonic()
    limits = frozen["config"]
    counters = {"entries": 0, "cad_files": 0, "total_bytes": 0}
    groups = {}
    status, error = "COMPLETE", None
    rows = open(attempt / "FILES.jsonl", "x", encoding="utf-8")

    def check():
        if time.monotonic() - started > limits["seconds"]:
            raise RuntimeError("wall_time_budget")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        if rss_bytes > limits["rss_bytes"]:
            raise RuntimeError("rss_budget")

    def emit(row):
        rows.write(json.dumps(row, sort_keys=True) + "\n")
        rows.flush()

    def walk(fd, prefix=""):
        # Iterator is streamed: no unbounded list/sort of corpus entries.
        with os.scandir(fd) as entries:
            for entry in entries:
                check()
                counters["entries"] += 1
                relative = prefix + entry.name
                row = {"relative_path": relative, "extension": Path(entry.name).suffix.lower()}
                if counters["entries"] > limits["entries"]:
                    emit({**row, "disposition": "entry_budget_refused"})
                    raise RuntimeError("entry_budget")
                if excluded(entry.name):
                    emit({**row, "disposition": "reserved_scope_excluded"})
                    continue
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    emit({**row, "disposition": "symlink_refused"})
                elif stat.S_ISDIR(info.st_mode):
                    if "scan" in relative.casefold():
                        emit({**row, "disposition": "scan_named_directory_metadata_only"})
                        continue
                    emit({**row, "disposition": "directory"})
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        walk(child, relative + "/")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    row["file_bytes"] = info.st_size
                    if "scan" in relative.casefold():
                        emit({**row, "disposition": "scan_named_metadata_only"})
                        continue
                    if row["extension"] in AMBIGUOUS_MESH_EXTENSIONS:
                        emit({**row, "disposition": "ambiguous_mesh_metadata_only"})
                        continue
                    if row["extension"] not in CAD_EXTENSIONS:
                        emit({**row, "disposition": "non_cad_metadata_only"})
                        continue
                    counters["cad_files"] += int(row["extension"] in CAD_EXTENSIONS)
                    if counters["cad_files"] > limits["cad_files"] or info.st_size > limits["file_bytes"] or counters["total_bytes"] + info.st_size > limits["total_bytes"]:
                        emit({**row, "disposition": "byte_or_cad_budget_refused"})
                        raise RuntimeError("byte_or_cad_budget")
                    file_fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                    try:
                        before = os.fstat(file_fd)
                        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
                            raise RuntimeError("file_changed_before_read")
                        sha, count = hashlib.sha256(), 0
                        while True:
                            check()
                            block = os.read(file_fd, min(1024**2, info.st_size - count + 1))
                            if not block:
                                break
                            count += len(block)
                            counters["total_bytes"] += len(block)
                            if count > info.st_size or counters["total_bytes"] > limits["total_bytes"]:
                                raise RuntimeError("file_growth_or_total_budget")
                            sha.update(block)
                        after = os.fstat(file_fd)
                        if count != info.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                            raise RuntimeError("file_changed_during_read")
                        row.update(sha256=sha.hexdigest(), disposition="supported_format" if row["extension"] in SUPPORTED else "unsupported_format")
                        groups.setdefault(row["sha256"], []).append(relative)
                        emit(row)
                    except Exception as exc:
                        emit({**row, "disposition": "file_read_failed", "error": str(exc)})
                        raise
                    finally:
                        os.close(file_fd)
                else:
                    emit({**row, "disposition": "special_file_refused"})

    try:
        fd = open_root(frozen["root"])
        try:
            walk(fd)
        finally:
            os.close(fd)
    except BaseException as exc:
        status, error = "INCOMPLETE_FAIL_CLOSED", f"{type(exc).__name__}: {exc}"
    finally:
        rows.flush()
        os.fsync(rows.fileno())
        rows.close()
        write_exclusive(attempt / "DUPLICATE_GROUPS.json", {k: v for k, v in groups.items() if len(v) > 1})
        write_exclusive(attempt / "TERMINAL.json", {"status": status, "error": error, "counters": counters,
                        "elapsed_seconds": time.monotonic() - started, "files_sha256": digest(attempt / "FILES.jsonl")})
    return 0 if status == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.attempt)
        return 0
    if not args.freeze_sha256:
        parser.error("--execute requires reviewed --freeze-sha256")
    return execute(args.attempt, args.freeze_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
