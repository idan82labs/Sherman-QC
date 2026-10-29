"""SKU rules and real ZEISS scan-filename parsing (plan §10 amendment 2).

The robotic scan cell names exports after the part. Real local shapes::

    BEFORE BENDING  11979003_C.ply     prefix words, double space, SKU_REV
    23553000_B_FULL_SCAN 1.ply         SKU_REV_<descriptor> <export index>
    38172000_01_SCAN_2 1.ply           numeric revision, SCAN_<n>
    44211000_A_SN0042_20261017T101500.ply   serial + timestamp
    <any of the above> (1).ply         duplicate copy marker
    11979003_C - Copy (2).ply          Windows copy marker
    11979003-REV-D.ply / 11979003_REVD explicit revision marker

Algorithm (never guesses):

1. Strip the directory and one ``.ply``/``.stl`` extension (case-insensitive),
   NFKC-normalise, strip a trailing ``(N)`` duplicate marker and then a
   trailing ``copy`` / ``copy N`` / `` - Copy`` marker (``copy_index`` is the
   ``(N)`` number when present, else N, else 1).
2. Tokenise on every non-alphanumeric run (spaces, ``_``, ``-``, ``.``),
   uppercasing ASCII letters. Known SKUs are tokenised the same way.
3. The SKU token sequence may appear **anywhere** in the name, on token
   boundaries. Matches strictly contained in a longer match are dropped (so
   ``44211000_A`` beats ``44211000``). If more than one distinct SKU remains,
   the result is AMBIGUOUS; if none, UNMATCHED.
4. Revision. Explicit markers win wherever they appear after the SKU:
   ``REV``/``REVISION`` followed by a revision token, or the glued forms
   ``REVD`` / ``REV01``. With a known expected revision, a bare ``R`` right
   after the SKU followed by a revision token is also a marker (``R`` alone
   is ambiguous otherwise and is then read as a revision itself). Without a
   marker, the token right after the SKU is the revision when it equals the
   expected revision or is revision-shaped (``[A-Z]{1,2}`` or two digits); a
   single digit counts too when the expected revision is numeric and the
   digit is not a whitespace-separated export counter. Words that label a
   serial or an order (``SN``, ``ID``, ``NO``, ``NR``, ``PN``, ``PO``, ``WO``,
   ``OK``, ``QC``: :data:`NON_REVISION_WORDS`) are never a bare revision
   unless they equal the expected revision. Numeric revisions compare by value
   (``1`` == ``01``). With a known expected revision, any reading that differs
   from it gives REVISION_MISMATCH (the resolver quarantines it), including a
   name carrying two different readings. Without one nothing can mismatch:
   the status is OK, ``revision_status`` UNVERIFIED, and ``revision`` is null
   when two readings conflict (never a guess).
5. In the rest: ``\\d{8}[T_-]?(\\d{4}|\\d{6})`` is the timestamp; descriptor
   words (FULL, SCAN, SCAN_2, BEFORE, BENDING, ...) form ``scan_label``; a
   whitespace-separated trailing integer is the ``export_index``; the first
   other token is the serial, except that a serial label word (``SN``,
   ``SERIAL``, ``NO``, ``NR``, ``ID``) followed by a token names that token.
6. An optional ``SHERMAN_LINE_FILENAME_REGEX`` (``pattern``) must contain a
   named group ``sku`` and replaces steps 2-3; optional groups ``revision``,
   ``serial`` and ``timestamp`` are honoured.

Every parsed field is informational for the operator and the C05 resolver.
None of it is ever passed to the engine (plan §10 amendment 4).
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from contracts.production.line_job import ScanNameParseV1
from domains.production.recipe.canonical_sku import (
    SkuError,
    canonical_sku,
    find_canonical_collisions,
    is_canonical_sku,
)

__all__ = [
    "SkuError",
    "canonical_sku",
    "is_canonical_sku",
    "find_canonical_collisions",
    "ScanNameParse",
    "parse_scan_filename",
    "compile_filename_pattern",
    "REVISION_TOKEN_RE",
    "DESCRIPTOR_WORDS",
    "NON_REVISION_WORDS",
]

#: A token after the SKU is read as a revision only when it has this shape
#: (or equals the expected revision exactly).
REVISION_TOKEN_RE = re.compile(r"^(?:[A-Z]{1,2}|[0-9]{2})$")

#: Revision-shaped words that label a serial / order number; never a bare
#: revision unless they equal the expected revision (``11979003_SN_0042``).
NON_REVISION_WORDS = frozenset({"SN", "ID", "NO", "NR", "PN", "PO", "WO", "OK", "QC"})
#: Serial label words: ``SN 0042`` / ``NO_0042`` name the following token.
_SERIAL_LABELS = frozenset({"SN", "SERIAL", "NO", "NR", "ID"})

#: Scan-descriptor words; ``SCAN<n>`` / ``STAGE<n>`` glued forms are also descriptors.
DESCRIPTOR_WORDS = frozenset(
    {"FULL", "SCAN", "PARTIAL", "BEFORE", "AFTER", "BENDING", "BEND", "BLANK", "FLAT", "STAGE", "FINAL"}
)
_DESCRIPTOR_GLUED = re.compile(r"^(?:SCAN|STAGE)[0-9]{1,3}$")
_INDEX_FOLLOWERS = frozenset({"SCAN", "STAGE"})

#: Explicit revision markers and their glued form (``REVD``, ``REV01``).
_REV_MARKERS = frozenset({"REV", "REVISION"})
_REV_GLUED = re.compile(r"^REV([A-Z]{1,2}|[0-9]{1,2})$")
_SINGLE_DIGIT = re.compile(r"^[0-9]$")

_EXTENSIONS = (".ply", ".stl")
_COPY_PAREN = re.compile(r"^(?P<stem>.*?)\s*\((?P<n>[0-9]{1,4})\)\s*$")
_COPY_WORD = re.compile(r"^(?P<stem>.*?)\s+copy(?:\s+(?P<n>[0-9]{1,4}))?\s*$", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_TS_ONE = re.compile(r"^(?P<d>[0-9]{8})T?(?P<t>[0-9]{4}|[0-9]{6})$")
_DATE8 = re.compile(r"^[0-9]{8}$")
_TIME = re.compile(r"^(?:[0-9]{4}|[0-9]{6})$")
_SMALL_INT = re.compile(r"^[0-9]{1,3}$")


@dataclass(frozen=True)
class ScanNameParse:
    """Result of :func:`parse_scan_filename`. ``status`` is one of
    OK | UNPARSEABLE | UNMATCHED | AMBIGUOUS | REVISION_MISMATCH."""

    status: str
    sku: str | None = None
    revision: str | None = None
    expected_revision: str | None = None
    revision_status: str | None = None
    serial: str | None = None
    scan_timestamp: str | None = None
    scan_label: str | None = None
    export_index: int | None = None
    copy_index: int | None = None
    prefix: str | None = None
    remainder: str = ""
    candidates: tuple[str, ...] = field(default_factory=tuple)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidates"] = list(self.candidates)
        return data

    def to_contract(self) -> ScanNameParseV1:
        return ScanNameParseV1(
            status=self.status,  # type: ignore[arg-type]
            sku=self.sku,
            revision=self.revision,
            expected_revision=self.expected_revision,
            revision_status=self.revision_status,  # type: ignore[arg-type]
            serial=self.serial,
            scan_timestamp=self.scan_timestamp,
            scan_label=self.scan_label,
            export_index=self.export_index,
            copy_index=self.copy_index,
            candidates=self.candidates,
            reason=self.reason,
        )


@dataclass(frozen=True)
class _Tok:
    text: str  # uppercased
    start: int
    end: int
    sep_before: str  # raw separator text before this token


def _upper_ascii(text: str) -> str:
    return "".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in text)


def _tokenise(text: str) -> list[_Tok]:
    out: list[_Tok] = []
    cursor = 0
    for m in _TOKEN.finditer(text):
        out.append(_Tok(_upper_ascii(m.group()), m.start(), m.end(), text[cursor:m.start()]))
        cursor = m.end()
    return out


def _sku_tokens(sku: str) -> tuple[str, ...]:
    return tuple(t.text for t in _tokenise(sku))


def compile_filename_pattern(pattern: str | re.Pattern[str]) -> re.Pattern[str]:
    """Compile ``SHERMAN_LINE_FILENAME_REGEX``; it must define a named group ``sku``."""
    try:
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    except re.error as exc:
        raise SkuError("FILENAME_REGEX_INVALID", f"regex does not compile: {exc}") from None
    if "sku" not in compiled.groupindex:
        raise SkuError("FILENAME_REGEX_INVALID", "regex must contain a named group 'sku'")
    return compiled


def _known_map(known_skus: Iterable[str] | Mapping[str, Any]) -> dict[str, str | None]:
    """Canonical SKU -> canonical expected revision (None = unknown)."""
    out: dict[str, str | None] = {}
    if isinstance(known_skus, Mapping):
        items = list(known_skus.items())
    else:
        items = [(sku, None) for sku in known_skus]
    for raw, revision in items:
        canon = canonical_sku(raw)
        rev = None
        if revision is not None and str(revision).strip():
            rev = _upper_ascii(unicodedata.normalize("NFKC", str(revision)).strip())
        if canon in out and out[canon] != rev:
            raise SkuError("SKU_INVALID", f"conflicting expected revisions for {canon}")
        out[canon] = rev
    return out


def _parse_timestamp(date8: str, time: str) -> str | None:
    fmt = "%Y%m%d%H%M%S" if len(time) == 6 else "%Y%m%d%H%M"
    try:
        parsed = datetime.strptime(date8 + time, fmt)
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%S")


def _strip_name(filename: str) -> tuple[str, int | None]:
    name = unicodedata.normalize("NFKC", str(filename)).replace("\\", "/").rsplit("/", 1)[-1].strip()
    lower = name.lower()
    for ext in _EXTENSIONS:
        if lower.endswith(ext):
            name = name[: -len(ext)]
            break
    copy_index: int | None = None
    m = _COPY_PAREN.match(name)
    if m:
        name, copy_index = m.group("stem"), int(m.group("n"))
    m = _COPY_WORD.match(name)
    if m and m.group("stem").strip(" -_."):
        word_n = int(m.group("n")) if m.group("n") else 1
        name = m.group("stem").rstrip(" -_.")
        copy_index = copy_index if copy_index is not None else word_n
    return name.strip(), copy_index


def _rev_equal(token: str, expected: str) -> bool:
    """Exact match, or equal numeric value (``1`` == ``01``)."""
    return token == expected or (token.isdigit() and expected.isdigit() and int(token) == int(expected))


def _find_revision(rest: list[_Tok], expected: str | None) -> tuple[str | None, str, set[int]]:
    """Return ``(revision, revision_status, consumed token indices)`` for the
    tokens after the SKU (module docstring, step 4)."""
    numeric_expected = expected is not None and expected.isdigit()

    def shaped(tok: _Tok, *, single_digit_ok: bool, bare: bool = False) -> bool:
        if _SINGLE_DIGIT.match(tok.text) and not single_digit_ok:
            return False
        if expected is not None and _rev_equal(tok.text, expected):
            return True
        if bare and tok.text in NON_REVISION_WORDS:
            return False
        if REVISION_TOKEN_RE.match(tok.text):
            return True
        return single_digit_ok and numeric_expected and bool(_SINGLE_DIGIT.match(tok.text))

    readings: list[tuple[str, set[int]]] = []
    for i, tok in enumerate(rest):
        if tok.text in _REV_MARKERS and i + 1 < len(rest) and shaped(rest[i + 1], single_digit_ok=True):
            readings.append((rest[i + 1].text, {i, i + 1}))
            continue
        glued = _REV_GLUED.match(tok.text)
        if glued:
            readings.append((glued.group(1), {i}))
    used = set().union(*(idx for _, idx in readings)) if readings else set()
    if rest and 0 not in used:
        t0 = rest[0]
        if (
            expected is not None
            and t0.text == "R"
            and not _rev_equal("R", expected)
            and len(rest) > 1
            and 1 not in used
            and shaped(rest[1], single_digit_ok=True)
        ):
            readings.insert(0, (rest[1].text, {0, 1}))
        else:
            # A single digit after whitespace is an export counter, never a revision.
            single_ok = not t0.sep_before.isspace()
            if shaped(t0, single_digit_ok=single_ok, bare=True):
                readings.insert(0, (t0.text, {0}))
    if not readings:
        return None, "NOT_IN_NAME", set()
    consumed = set().union(*(idx for _, idx in readings))
    if expected is None:
        # Nothing to compare against; two different readings stay unresolved.
        values = {text for text, _ in readings}
        return (readings[0][0] if len(values) == 1 else None), "UNVERIFIED", consumed
    mismatched = [text for text, _ in readings if not _rev_equal(text, expected)]
    if mismatched:
        return mismatched[0], "MISMATCH", consumed
    return readings[0][0], "MATCH", consumed


def _classify_rest(rest: list[_Tok], prefix: list[_Tok]) -> dict[str, Any]:
    """Timestamp, descriptors, export index and serial from the remaining tokens."""
    consumed = [False] * len(rest)
    timestamp: str | None = None
    for i, tok in enumerate(rest):
        m = _TS_ONE.match(tok.text)
        if m:
            ts = _parse_timestamp(m.group("d"), m.group("t"))
            if ts:
                timestamp, consumed[i] = ts, True
                break
        if _DATE8.match(tok.text) and i + 1 < len(rest) and _TIME.match(rest[i + 1].text):
            sep = rest[i + 1].sep_before
            if sep in ("_", "-"):
                ts = _parse_timestamp(tok.text, rest[i + 1].text)
                if ts:
                    timestamp = ts
                    consumed[i] = consumed[i + 1] = True
                    break

    export_index: int | None = None
    if rest and not consumed[-1] and _SMALL_INT.match(rest[-1].text) and rest[-1].sep_before.isspace():
        export_index, consumed[-1] = int(rest[-1].text), True

    label_parts: list[str] = []
    for tok in prefix:
        if tok.text in DESCRIPTOR_WORDS or _DESCRIPTOR_GLUED.match(tok.text):
            label_parts.append(tok.text)
    for i, tok in enumerate(rest):
        if consumed[i]:
            continue
        if tok.text in DESCRIPTOR_WORDS or _DESCRIPTOR_GLUED.match(tok.text):
            label_parts.append(tok.text)
            consumed[i] = True
        elif (
            _SMALL_INT.match(tok.text)
            and i > 0
            and rest[i - 1].text in _INDEX_FOLLOWERS
            and consumed[i - 1]
            and not tok.sep_before.isspace()
        ):
            label_parts.append(tok.text)
            consumed[i] = True

    serial: str | None = None
    free = [i for i in range(len(rest)) if not consumed[i]]
    if free:
        first = free[0]
        if rest[first].text in _SERIAL_LABELS and len(free) > 1 and free[1] == first + 1:
            serial = rest[first + 1].text
        else:
            serial = rest[first].text
    return {
        "scan_timestamp": timestamp,
        "export_index": export_index,
        "scan_label": "_".join(label_parts) or None,
        "serial": serial,
    }


def parse_scan_filename(
    filename: str,
    known_skus: Iterable[str] | Mapping[str, Any],
    pattern: str | re.Pattern[str] | None = None,
) -> ScanNameParse:
    """Parse a scan filename against the known SKUs (see module docstring).

    ``known_skus`` is an iterable of SKUs, or a mapping SKU -> expected
    revision (``None`` when the recipe has no revision). Invalid known SKUs
    raise ``SkuError``; an invalid ``pattern`` raises ``SkuError``
    (``FILENAME_REGEX_INVALID``).
    """
    regex = compile_filename_pattern(pattern) if pattern is not None else None
    known = _known_map(known_skus)
    stem, copy_index = _strip_name(filename)
    if not _TOKEN.search(stem):
        return ScanNameParse(status="UNPARSEABLE", copy_index=copy_index, reason="NO_NAME_TOKENS")

    if regex is not None:
        return _parse_with_pattern(stem, copy_index, regex, known)

    tokens = _tokenise(stem)
    texts = [t.text for t in tokens]
    by_seq: dict[tuple[str, ...], list[str]] = {}
    for sku in known:
        by_seq.setdefault(_sku_tokens(sku), []).append(sku)

    matches: list[tuple[int, int, str]] = []
    for seq, skus in by_seq.items():
        n = len(seq)
        for i in range(len(texts) - n + 1):
            if tuple(texts[i:i + n]) == seq:
                matches.extend((i, i + n, sku) for sku in skus)
    maximal = [
        m for m in matches
        if not any(o[0] <= m[0] and m[1] <= o[1] and (o[1] - o[0]) > (m[1] - m[0]) for o in matches)
    ]
    distinct = sorted({m[2] for m in maximal})
    if not distinct:
        return ScanNameParse(status="UNMATCHED", copy_index=copy_index, remainder=stem, reason="NO_KNOWN_SKU_TOKEN")
    if len(distinct) > 1:
        return ScanNameParse(
            status="AMBIGUOUS",
            copy_index=copy_index,
            remainder=stem,
            candidates=tuple(distinct),
            reason="MULTIPLE_KNOWN_SKUS",
        )
    start, end, sku = min(m for m in maximal if m[2] == distinct[0])
    expected = known[sku]
    after = tokens[end:]
    revision, revision_status, consumed = _find_revision(after, expected)
    rest = [tok for i, tok in enumerate(after) if i not in consumed]
    info = _classify_rest(rest, tokens[:start])
    prefix = stem[: tokens[start].start].strip(" _-.") or None
    remainder = stem[tokens[end - 1].end:].strip(" _-.")
    mismatch = revision_status == "MISMATCH"
    return ScanNameParse(
        status="REVISION_MISMATCH" if mismatch else "OK",
        sku=sku,
        revision=revision,
        expected_revision=expected,
        revision_status=revision_status,
        copy_index=copy_index,
        prefix=prefix,
        remainder=remainder,
        reason="REVISION_DIFFERS_FROM_RECIPE" if mismatch else None,
        **info,
    )


def _parse_with_pattern(
    stem: str, copy_index: int | None, regex: re.Pattern[str], known: dict[str, str | None]
) -> ScanNameParse:
    m = regex.fullmatch(stem)
    if not m or not m.group("sku"):
        return ScanNameParse(status="UNPARSEABLE", copy_index=copy_index, remainder=stem, reason="PATTERN_NO_MATCH")
    try:
        sku = canonical_sku(m.group("sku"))
    except SkuError:
        return ScanNameParse(status="UNPARSEABLE", copy_index=copy_index, remainder=stem, reason="PATTERN_SKU_INVALID")
    if sku not in known:
        return ScanNameParse(status="UNMATCHED", copy_index=copy_index, remainder=stem, reason="NO_KNOWN_SKU_TOKEN")
    groups = m.groupdict()
    expected = known[sku]
    raw_rev = groups.get("revision")
    rev_token = _upper_ascii(raw_rev.strip()) if raw_rev else None
    if rev_token is None:
        revision, revision_status = None, "NOT_IN_NAME"
    elif expected is None:
        revision, revision_status = rev_token, "UNVERIFIED"
    else:
        revision, revision_status = rev_token, ("MATCH" if rev_token == expected else "MISMATCH")
    timestamp = None
    if groups.get("timestamp"):
        tm = _TS_ONE.match(re.sub(r"[_-]", "", groups["timestamp"]).upper())
        timestamp = _parse_timestamp(tm.group("d"), tm.group("t")) if tm else None
    mismatch = revision_status == "MISMATCH"
    return ScanNameParse(
        status="REVISION_MISMATCH" if mismatch else "OK",
        sku=sku,
        revision=revision,
        expected_revision=expected,
        revision_status=revision_status,
        serial=(groups.get("serial") or None),
        scan_timestamp=timestamp,
        copy_index=copy_index,
        remainder=stem[m.end("sku"):].strip(" _-.") if m.end("sku") >= 0 else "",
        reason="REVISION_DIFFERS_FROM_RECIPE" if mismatch else None,
    )
