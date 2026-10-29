"""Additive, stdlib-only terminal publication and single-directory custody.

No repository paths, CLI, imports of runners, directory creation, failure cleanup
or retry are provided. Callers supply an already owned absolute directory, their
frozen byte/entry limits, and an absolute deadline around the entire operation.
These synchronous filesystem operations are not a self-contained hard wall/RSS
supervisor. The literal ceilings below are helper engineering ceilings, not new
permissions or replacements for any runner's smaller frozen limits.

Publication uses an exclusive staging file and exclusive hard link. Only the
normal successful staging link is removed; no failure path deletes a partial.
A failure after successful-stage removal can instead leave only the target. A link
collision after staging is a failure, never a benign preexisting-record result.
Existing-terminal validation does not invoke the new-bytes factory or write.

Inventory covers exactly ONE directory, including dotfiles. No recursion or
symlink following occurs. Unknown entries are metadata-only, always vetoed, and
never read under an invented content limit. Declared regular files are streamed
under per-file/aggregate limits. Two bounded name/identity passes detect ordinary
changes, but do not constitute an atomic snapshot against an adversarial writer.
Callers must stop writers and retain that custody premise through final release.
Same bytes never imply inode aliasing; every name remains a separate entry.

Receipts are observations, not authentication. Callers must retain/hash them and
bind every directory in their own complete campaign inventory. Callback code is
trusted caller code: validate_record(bytes) must return exactly True, must be
read-only, and must not broaden I/O authority. Ordinary failures are structured;
BaseException propagates after descriptor closure, without deleting partials.
"""

import hashlib
import os
import secrets
import stat
import sys


_MAX_FILE_BYTES = 128 * 1024**2
_MAX_TOTAL_BYTES = 8 * 1024**3
_MAX_ENTRIES = 1024
_CHUNK = 65536
_FINGERPRINT = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_mtime_ns", "st_ctime_ns")


class ArtifactFailureV4(RuntimeError):
    """Failure with bounded observations; never embeds dynamic exception text."""

    def __init__(self, code, evidence):
        super().__init__(code)
        self.code = code
        self.evidence = dict(evidence)


def _fail(code, evidence):
    evidence.update(status="UNAVAILABLE", complete=False, reason=code)
    raise ArtifactFailureV4(code, evidence)


def _limit(value, maximum, field, evidence):
    if type(value) is not int or not 0 <= value <= maximum:
        _fail("INVALID_" + field, evidence)
    return value


def _name(value, evidence, *, publishing=False):
    if (type(value) is not str or not value or value in (".", "..")
            or "/" in value or "\x00" in value or len(value) > 255):
        _fail("INVALID_ARTIFACT_NAME", evidence)
    try:
        encoded_length = len(os.fsencode(value))
    except UnicodeError:
        _fail("ARTIFACT_NAME_ENCODING", evidence)
    if encoded_length > (220 if publishing else 255):
        _fail("ARTIFACT_NAME_BYTE_LIMIT", evidence)
    return value


def _fingerprint(row):
    return tuple(getattr(row, field) for field in _FINGERPRINT)


def _identity(row):
    return dict(device=row.st_dev, inode=row.st_ino, links=row.st_nlink,
                bytes=row.st_size, mtime_ns=row.st_mtime_ns, ctime_ns=row.st_ctime_ns)


def _stat(fd, name):
    return os.stat(name, dir_fd=fd, follow_symlinks=False)


def _open_directory(directory, evidence):
    path = os.fspath(directory)
    if (type(path) is not str or not path.startswith("/") or "\x00" in path
            or len(path) > 4096 or len(os.fsencode(path)) > 4096):
        _fail("ABSOLUTE_OWNED_DIRECTORY_REQUIRED", evidence)
    parts = path.split("/")[1:]
    if parts and parts[-1] == "":
        parts.pop()
    if len(parts) > 64 or any(part in ("", ".", "..") for part in parts):
        _fail("DIRECTORY_COMPONENT_POLICY", evidence)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=fd)
            previous, fd = fd, next_fd
            os.close(previous)
        return fd
    except BaseException:
        try:
            os.close(fd)
        except Exception:
            evidence["close_error_types"].append("DIRECTORY_OPEN_CLEANUP_CLOSE_FAILED")
        raise


def _operation(directory, evidence, action):
    fd = None
    try:
        evidence["phase"] = "OPEN_DIRECTORY"
        path = os.fspath(directory)
        fd = _open_directory(path, evidence)
        original = os.fstat(fd)
        evidence["directory_identity"] = dict(device=original.st_dev, inode=original.st_ino)
        result = action(fd)
        # Reopening the captured path also checks every ancestor without following
        # links. A pinned descriptor alone would miss replacement of its pathname.
        final_fd = _open_directory(path, evidence)
        try:
            final = os.fstat(final_fd)
            if (final.st_dev, final.st_ino) != (original.st_dev, original.st_ino):
                _fail("DIRECTORY_PATH_BINDING_CHANGED", evidence)
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                os.close(final_fd)
            except Exception as error:
                evidence["close_error_types"].append(type(error).__name__[:64])
                if not active_error:
                    _fail("DIRECTORY_RECHECK_CLOSE_FAILED", evidence)
        evidence["directory_path_binding_rechecked"] = True
        return result
    except ArtifactFailureV4:
        raise
    except Exception as error:
        evidence["error_type"] = type(error).__name__[:64]
        _fail("FILESYSTEM_OR_CALLBACK_EXCEPTION", evidence)
    finally:
        if fd is not None:
            active_error = sys.exc_info()[0] is not None
            try:
                os.close(fd)
            except Exception as error:
                evidence["close_error_types"].append(type(error).__name__[:64])
                if not active_error:
                    _fail("DIRECTORY_CLOSE_FAILED", evidence)


def _new_evidence(schema):
    return dict(schema=schema, status="UNAVAILABLE", complete=False, reason=None,
                phase="VALIDATION", read_bytes=0, close_error_types=[])


def _read_file(fd, name, cap, evidence, *, collect):
    """Read exactly the admitted size; stable post-stat catches growth/truncation."""
    before_name = _stat(fd, name)
    if not stat.S_ISREG(before_name.st_mode):
        _fail("REGULAR_NON_SYMLINK_FILE_REQUIRED", evidence)
    if not 0 <= before_name.st_size <= cap:
        _fail("FILE_BYTE_LIMIT", evidence)
    child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(child)
        if _fingerprint(before) != _fingerprint(before_name):
            _fail("FILE_CHANGED_BEFORE_READ", evidence)
        remaining = before.st_size
        hasher, chunks = hashlib.sha256(), []
        while remaining:
            block = os.read(child, min(_CHUNK, remaining))
            evidence["read_bytes"] += len(block)
            if not block or len(block) > min(_CHUNK, remaining):
                _fail("SHORT_OR_INVALID_FILE_READ", evidence)
            remaining -= len(block)
            hasher.update(block)
            if collect:
                chunks.append(block)
        after = os.fstat(child)
        if (_fingerprint(before) != _fingerprint(after)
                or _fingerprint(before) != _fingerprint(_stat(fd, name))):
            _fail("FILE_CHANGED_DURING_READ", evidence)
        row = dict(sha256=hasher.hexdigest(), bytes=before.st_size,
                   identity=_identity(before))
        return (b"".join(chunks) if collect else None), row, _fingerprint(before)
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            os.close(child)
        except Exception as error:
            evidence["close_error_types"].append(type(error).__name__[:64])
            if not active_error:
                _fail("FILE_CLOSE_FAILED", evidence)


def _publish(fd, name, blob, cap, evidence):
    if type(blob) is not bytes or len(blob) > cap:
        _fail("PUBLICATION_BYTES_OR_LIMIT", evidence)
    evidence.update(target=name, requested_bytes=len(blob), written_bytes=0,
                    written_bytes_exact=True, write_calls=0,
                    partial_name=None, partial_created=False,
                    target_linked=False, successful_stage_removed=False,
                    target_directory_synced=False, final_directory_synced=False)
    evidence["phase"] = "TARGET_PREFLIGHT"
    try:
        _stat(fd, name)
    except FileNotFoundError:
        pass
    else:
        _fail("TARGET_ALREADY_EXISTS_NO_WRITE", evidence)
    partial = "." + name + ".partial-" + secrets.token_hex(8)
    evidence.update(partial_name=partial, phase="CREATE_STAGE")
    child = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600, dir_fd=fd)
    evidence["partial_created"] = True
    try:
        initial = os.fstat(child)
        evidence["partial_identity"] = _identity(initial)
        evidence["partial_identity_observation_phase"] = "CREATED_BEFORE_WRITE"
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            _fail("STAGING_IDENTITY_INVALID", evidence)
        evidence["phase"] = "WRITE_STAGE"
        view = memoryview(blob)
        for start in range(0, len(blob), _CHUNK):
            block = view[start:start + _CHUNK]
            evidence["write_calls"] += 1
            try:
                written = os.write(child, block)
            except Exception:
                evidence["written_bytes_exact"] = False
                raise
            if type(written) is not int or not 0 <= written <= len(block):
                evidence["written_bytes_exact"] = False
                _fail("INVALID_WRITE_RESULT", evidence)
            evidence["written_bytes"] += written
            if written != len(block):
                _fail("SHORT_STAGE_WRITE_NO_RETRY", evidence)
        evidence["written_bytes_exact"] = True
        evidence["phase"] = "FSYNC_STAGE"
        os.fsync(child)
        ready = os.fstat(child)
        evidence["partial_identity"] = _identity(ready)
        evidence["partial_identity_observation_phase"] = "AFTER_STAGE_FSYNC"
        if ((initial.st_dev, initial.st_ino) != (ready.st_dev, ready.st_ino)
                or ready.st_size != len(blob) or ready.st_nlink != 1
                or _fingerprint(ready) != _fingerprint(_stat(fd, partial))):
            _fail("STAGING_CHANGED_BEFORE_LINK", evidence)
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            os.close(child)
        except Exception as error:
            evidence["close_error_types"].append(type(error).__name__[:64])
            if not active_error:
                _fail("STAGING_CLOSE_FAILED", evidence)
    evidence["phase"] = "EXCLUSIVE_LINK"
    try:
        os.link(partial, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
    except FileExistsError:
        _fail("LINK_TARGET_COLLISION_AFTER_STAGING", evidence)
    evidence["target_linked"] = True
    linked, staged = _stat(fd, name), _stat(fd, partial)
    if ((linked.st_dev, linked.st_ino) != (ready.st_dev, ready.st_ino)
            or _fingerprint(linked) != _fingerprint(staged)
            or linked.st_mode != ready.st_mode or linked.st_size != len(blob)
            or linked.st_mtime_ns != ready.st_mtime_ns or linked.st_nlink != 2):
        _fail("LINKED_TARGET_IDENTITY_MISMATCH", evidence)
    evidence["phase"] = "FSYNC_LINKED_DIRECTORY"
    os.fsync(fd)
    evidence["target_directory_synced"] = True
    evidence["phase"] = "REMOVE_SUCCESSFUL_STAGING_LINK"
    # This is normal successful publication, never cleanup of a failed partial.
    os.unlink(partial, dir_fd=fd)
    evidence["successful_stage_removed"] = True
    evidence["phase"] = "FSYNC_FINAL_DIRECTORY"
    os.fsync(fd)
    evidence["final_directory_synced"] = True
    final = _stat(fd, name)
    if ((final.st_dev, final.st_ino) != (ready.st_dev, ready.st_ino)
            or final.st_mode != ready.st_mode or final.st_size != len(blob)
            or final.st_mtime_ns != ready.st_mtime_ns or final.st_nlink != 1):
        _fail("PUBLISHED_TARGET_CHANGED", evidence)
    evidence.update(status="COMPLETE", complete=True, reason=None, phase="PUBLISHED",
                    disposition="NEW_EXCLUSIVE_PUBLICATION", sha256=hashlib.sha256(blob).hexdigest(),
                    bytes=len(blob), identity=_identity(final))
    return evidence


def publish_bytes_exclusive_v4(directory, name, blob, *, max_bytes):
    """Publish once or raise ArtifactFailureV4; existing names cause no write."""
    evidence = _new_evidence("EXCLUSIVE_ARTIFACT_PUBLICATION_V4")
    name = _name(name, evidence, publishing=True)
    cap = _limit(max_bytes, _MAX_FILE_BYTES, "FILE_BYTE_LIMIT", evidence)
    return _operation(directory, evidence, lambda fd: _publish(fd, name, blob, cap, evidence))


def ensure_terminal_record_v4(directory, name, make_bytes, validate_record, *, max_bytes):
    """Validate an existing stable record, or publish a validated new record.

    make_bytes() must return exact bytes. validate_record(bytes) must return the
    singleton True. Neither callback is invoked on unbounded/unstable input.
    A target appearing after the initial absent check is a failure, not a retry.
    """
    evidence = _new_evidence("ENSURE_TERMINAL_ARTIFACT_V4")
    name = _name(name, evidence, publishing=True)
    cap = _limit(max_bytes, _MAX_FILE_BYTES, "FILE_BYTE_LIMIT", evidence)
    if not callable(make_bytes) or not callable(validate_record):
        _fail("CALLABLE_FACTORY_AND_VALIDATOR_REQUIRED", evidence)

    def action(fd):
        evidence.update(target=name, phase="EXISTING_TARGET_CHECK", publication_attempted=False)
        try:
            _stat(fd, name)
        except FileNotFoundError:
            exists = False
        else:
            exists = True
        if exists:
            evidence["phase"] = "READ_EXISTING_TARGET"
            blob, row, fingerprint = _read_file(fd, name, cap, evidence, collect=True)
            evidence["phase"] = "VALIDATE_EXISTING_TARGET"
            if validate_record(blob) is not True:
                _fail("EXISTING_RECORD_VALIDATION_FAILED", evidence)
            if fingerprint != _fingerprint(_stat(fd, name)):
                _fail("EXISTING_RECORD_CHANGED_DURING_VALIDATION", evidence)
            evidence.update(row, status="COMPLETE", complete=True, reason=None,
                            phase="EXISTING_RECORD_VALIDATED", disposition="PREEXISTING_VALID_NO_WRITE",
                            wrote_new_bytes=False)
            return evidence
        evidence["phase"] = "MAKE_NEW_RECORD"
        blob = make_bytes()
        if type(blob) is not bytes or len(blob) > cap:
            _fail("FACTORY_BYTES_OR_LIMIT", evidence)
        evidence["phase"] = "VALIDATE_NEW_RECORD"
        if validate_record(blob) is not True:
            _fail("NEW_RECORD_VALIDATION_FAILED", evidence)
        evidence["publication_attempted"] = True
        result = _publish(fd, name, blob, cap, evidence)
        result["wrote_new_bytes"] = bool(blob)
        return result

    return _operation(directory, evidence, action)


def _expectations(expected, cap, evidence):
    if type(expected) is not dict or len(expected) > cap:
        _fail("EXPECTED_ENTRY_SET_OR_LIMIT", evidence)
    snapshot = {}
    for name, row in expected.items():
        name = _name(name, evidence)
        if (type(row) is not dict or len(row) != 3
                or any(type(key) is not str for key in row)
                or set(row) != {"max_bytes", "sha256", "bytes"}):
            _fail("EXPECTED_ENTRY_SCHEMA", evidence)
        limit, sha, size = row["max_bytes"], row["sha256"], row["bytes"]
        limit = _limit(limit, _MAX_FILE_BYTES, "EXPECTED_FILE_LIMIT", evidence)
        if sha is not None and (type(sha) is not str or len(sha) != 64
                                or any(char not in "0123456789abcdef" for char in sha)):
            _fail("EXPECTED_HASH_SCHEMA", evidence)
        if size is not None and (type(size) is not int or not 0 <= size <= limit):
            _fail("EXPECTED_SIZE_SCHEMA", evidence)
        snapshot[name] = (limit, sha, size)
    return snapshot


def _scan(fd, cap, evidence, *, first):
    names, seen = [], set()
    # Retain a partial name pass even if enumeration/iterator closure fails.
    evidence["initial_names" if first else "final_names"] = names
    # A fresh open description avoids depending on whether fd-based scandir
    # shares/advances the original descriptor's directory position between passes.
    scan_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        with os.scandir(scan_fd) as entries:
            for entry in entries:
                evidence["enumerated_entries"] += 1
                name = _name(entry.name, evidence)
                if len(names) >= cap:
                    evidence["first_unrecorded_name"] = name
                    _fail("DIRECTORY_ENTRY_LIMIT", evidence)
                if name in seen:
                    _fail("DUPLICATE_DIRECTORY_ENTRY", evidence)
                seen.add(name)
                names.append(name)
                if first:
                    evidence["entries"][name] = dict(status="NAME_OBSERVED_NOT_YET_VALIDATED")
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            os.close(scan_fd)
        except Exception as error:
            evidence["close_error_types"].append(type(error).__name__[:64])
            if not active_error:
                _fail("DIRECTORY_SCAN_CLOSE_FAILED", evidence)
    return sorted(names)


def inventory_directory_v4(directory, expected, *, max_entries, max_total_bytes):
    """Return complete=False on any ordinary failure; never delete an entry.

    expected is exact dict[name, {max_bytes:int, sha256:str|None, bytes:int|None}].
    Hash/length None means newly observed, NOT authenticated. Limits are charged
    before each declared read; failed reads retain their admitted and observed
    byte counts. Unknown entries receive no content read and always veto. At most
    max_entries names per pass plus one overflow name are observed; hashes stream
    at most max_total_bytes total. All discoverable names within the first-pass
    cap are retained even when a particular entry cannot be read or validated.
    """
    evidence = _new_evidence("SINGLE_DIRECTORY_ARTIFACT_CUSTODY_V4")
    evidence.update(entries={}, errors=[], missing_names=[], unexpected_names=[],
                    enumerated_entries=0, charged_read_bytes=0,
                    name_inventory_complete=False, directory_stable=False)
    try:
        cap = _limit(max_entries, _MAX_ENTRIES, "ENTRY_LIMIT", evidence)
        total = _limit(max_total_bytes, _MAX_TOTAL_BYTES, "TOTAL_BYTE_LIMIT", evidence)
        specs = _expectations(expected, cap, evidence)
        evidence["expected_names"] = sorted(specs)

        def issue(code, name=None):
            evidence["errors"].append(dict(code=code, name=name))

        def action(fd):
            evidence["phase"] = "ENUMERATE_INITIAL"
            directory_before = _fingerprint(os.fstat(fd))
            names = _scan(fd, cap, evidence, first=True)
            evidence["initial_names"] = names
            evidence["name_inventory_complete"] = True
            evidence["missing_names"] = sorted(set(specs) - set(names))
            evidence["unexpected_names"] = sorted(set(names) - set(specs))
            if evidence["missing_names"]:
                issue("EXPECTED_NAMES_MISSING")
            if evidence["unexpected_names"]:
                issue("UNDECLARED_NAMES_PRESENT")
            fingerprints = {}
            for name in names:
                row = evidence["entries"][name]
                local = _new_evidence("DECLARED_FILE_READ_V4")
                try:
                    found = _stat(fd, name)
                    fingerprints[name] = _fingerprint(found)
                    row.update(identity=_identity(found), declared=name in specs, sha256=None)
                    if not stat.S_ISREG(found.st_mode):
                        row["status"] = "NON_REGULAR_OR_SYMLINK_NO_READ"
                        issue(row["status"], name)
                        continue
                    if name not in specs:
                        row["status"] = "UNDECLARED_REGULAR_FILE_METADATA_ONLY"
                        continue
                    limit, sha, size = specs[name]
                    if not 0 <= found.st_size <= limit:
                        row["status"] = "FILE_BYTE_LIMIT_NO_READ"
                        issue(row["status"], name)
                        continue
                    if size is not None and found.st_size != size:
                        row["status"] = "EXPECTED_SIZE_MISMATCH_NO_READ"
                        issue(row["status"], name)
                        continue
                    if found.st_size > total - evidence["charged_read_bytes"]:
                        row["status"] = "TOTAL_BYTE_LIMIT_NO_READ"
                        issue(row["status"], name)
                        continue
                    evidence["charged_read_bytes"] += found.st_size
                    evidence["phase"] = "READ_DECLARED_FILES"
                    _, observed, fingerprint = _read_file(fd, name, found.st_size, local, collect=False)
                    if fingerprint != fingerprints[name]:
                        _fail("FILE_CHANGED_AFTER_ADMISSION", local)
                    row.update(observed, status="REGULAR_FILE_BYTES_BOUND")
                    if sha is not None and observed["sha256"] != sha:
                        row["status"] = "EXPECTED_HASH_MISMATCH"
                        issue(row["status"], name)
                except Exception as error:
                    code = error.code if isinstance(error, ArtifactFailureV4) else "ENTRY_STAT_OR_READ_EXCEPTION"
                    row.update(status=code, error_type=type(error).__name__[:64])
                    issue(code, name)
                finally:
                    evidence["read_bytes"] += local["read_bytes"]
                    evidence["close_error_types"].extend(local["close_error_types"])
            evidence["phase"] = "ENUMERATE_FINAL"
            final_names = _scan(fd, cap, evidence, first=False)
            evidence["final_names"] = final_names
            if names != final_names:
                issue("DIRECTORY_NAMES_CHANGED")
            for name in names:
                if name not in fingerprints:
                    continue
                try:
                    if fingerprints[name] != _fingerprint(_stat(fd, name)):
                        issue("ENTRY_CHANGED_BEFORE_INVENTORY_END", name)
                except Exception as error:
                    issue("ENTRY_UNAVAILABLE_AT_INVENTORY_END", name)
                    evidence["entries"][name]["final_error_type"] = type(error).__name__[:64]
            evidence["directory_stable"] = directory_before == _fingerprint(os.fstat(fd))
            if not evidence["directory_stable"]:
                issue("DIRECTORY_IDENTITY_OR_METADATA_CHANGED")
            if evidence["close_error_types"]:
                issue("DESCRIPTOR_CLOSE_FAILURE")
            if evidence["errors"]:
                evidence.update(status="UNAVAILABLE", complete=False, reason="DIRECTORY_CUSTODY_NOT_COMPLETE")
            else:
                evidence.update(status="COMPLETE", complete=True, reason=None)
            evidence["phase"] = "INVENTORY_FINISHED"
            return evidence

        return _operation(directory, evidence, action)
    except ArtifactFailureV4:
        return evidence
    except Exception as error:
        evidence.update(status="UNAVAILABLE", complete=False, reason="INVENTORY_VALIDATION_EXCEPTION",
                        error_type=type(error).__name__[:64])
        return evidence
