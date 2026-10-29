# CAD corpus byte inventory V1 — proposed 2026-09-30

Status: source implementation; physical inventory NOT executed by its author.
Root must review source/tests, freeze, pin and authorize the one execution.

The latest explicit corpus-expansion request supersedes the earlier five-CAD
scope only for this established corpus. Source declarations in
`scripts/build_vendor_corpus_demo.py` and
`scripts/run_vendor_unfold_five_part_candidate_v4.py` name the same root:
`/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D`.
No additional established root was found in those bounded source seams.
No filesystem discovery, physical stat/hash/read or email retrieval occurred.

## Frozen scope and limits

Inventory is bytes only: paths, extensions, file lengths and streamed SHA-256.
It imports no CAD/native reader and reads no geometry, Gold or labels.
All regular files, including unsupported formats, remain separate records;
only recognized CAD/native/exchange/mesh extensions are opened and hashed.
Non-CAD files (including CSV/JSON/PDF/XYZ supporting records) receive metadata
and `non_cad_metadata_only` disposition, with no content read and no SHA-256.
Duplicate grouping applies only to hashed CAD files.
identical byte content is grouped without deleting duplicates. STEP/STP are
marked supported for the downstream unfolding path; other recognized CAD
formats are retained as unsupported. CAD-extension count includes
mesh/native/exchange formats.
Reserved names containing incoming, EHO or HOLDOUT10 (case-insensitive) are
explicitly recorded and never traversed/read. Symlinks and special files are
recorded refusals, not followed. Excluded descendants are not enumerated.

Hard limits: 3,000 traversed entries, 500 CAD-extension files, 256 MiB/file,
4 GiB total read, 120 seconds wall time and 1 GiB process RSS. Budget exhaustion,
filesystem errors and mutation detected during reading fail closed, preserve
partial JSONL and produce an incomplete terminal record. No retry or raised cap.
The iterator is streamed; ordering follows filesystem enumeration rather than
materializing an unbounded directory list. The time/RSS checks occur at every
entry and every 1 MiB read; an external supervisor should also enforce wall/RSS
limits against blocked kernel I/O. SIGKILL or host failure can prevent terminal
publication; an existing EXECUTION record still prevents re-execution.

## Review and one-shot commands

Run from FullProject. Freeze creates an exclusive new attempt directory and
records lexical absolute scope/config/source hash without touching raw files.
Do not use path resolution against corpus during PRE. Review the frozen JSON,
record its SHA-256, then execute once with that exact pin:

```sh
.venv/bin/python scripts/inventory_vendor_cad_corpus_v1.py --freeze --attempt artifacts/vendor-cad-corpus-inventory-v1-2026-09-30-attempt1
.venv/bin/python scripts/inventory_vendor_cad_corpus_v1.py --execute --attempt artifacts/vendor-cad-corpus-inventory-v1-2026-09-30-attempt1 --freeze-sha256 REVIEWED_SHA256
```

These are proposed commands, not an instruction to bypass root approval.
Execution validates freeze and script hashes, reserves EXECUTION exclusively,
opens root and ancestors without following symlinks, scans only frozen scope,
and writes FILES.jsonl, DUPLICATE_GROUPS.json and TERMINAL.json. Root checks
outer exit and terminal completeness. A complete byte inventory is neither
geometry qualification nor authority to run unfolding; downstream CAD processing
needs its separately frozen candidate/input binding, budgets and independent gate.

Synthetic verification uses only pytest-created temporary fixture bytes:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider -q tests/test_inventory_vendor_cad_corpus_v1.py
```
