# Corpus baseline diagnostic V1 — proposed 2026-09-30

Source-only implementation. No freeze or physical execution by its author.
Root review and independent non-author PRE are required before the one execute.
This separately versioned attempt never resumes or changes old terminal attempts.

## Scope and scientific status

Latest explicit user authority expands CAD processing to the existing corpus.
Released byte-inventory FILES.jsonl is the only input-selection source. Exactly
23 STEP/STP paths bind 20 unique hashes; aliases remain in each case. One exact
representative per hash is processed, not 23 independent parts. DWG remains an
unsupported metadata record. All meshes/scans and other supporting files are
metadata only and never opened by this runner. The earlier inventory's broader
mesh byte hashing does not grant mesh geometry access. Incoming/EHO/HOLDOUT10
and any SCAN-named STEP are excluded. No email fetching, Gold or label access.

Baseline is extractor V2 followed by compatible two-skin candidate adaptation
and V5 selected-skin unfolding. No V3 extractor or ownership candidate is used.
EXTRACTION.json and REJECTIONS.json are written before folded display or
candidate/unfold processing. Folded mesh failure cannot erase extraction;
candidate adaptation refusal is distinct from extraction refusal. No-bend and
unsupported skin outcomes remain explicit. The independent native metric-chart
oracle receipt is exported when V5 supplies it; unavailable oracle stays unknown.
Manufacturing dimensions, units and physical accuracy remain unestablished.
Accuracy is UNSCORED because no known truth is supplied. This is a direct-engine
DIAGNOSTIC, not authenticated HTTP/app-route qualification or generalization.

## Freeze, PRE and exclusive execute

`--freeze --attempt NEW --inventory-sha256 PIN` reads only the released inventory
and source/runtime files, not raw CAD. It exclusively creates SOURCE.tar with
member-hashed SOURCE_MANIFEST.json, inventory copy and FROZEN.json. Runtime/source
closure reuses the audited V4 worker and V2 uploader pins plus runner, tests,
protocol, folded reader and inventory V2 traversal source. Frozen limits,
input/alias bindings and source archive are verified before execute and child.

`--execute --attempt NEW --frozen-sha256 PIN --pre REVIEW --pre-sha256 PIN`
requires a separately pinned review JSON with decision GREEN,
independent_non_author true and the exact frozen_sha256. Root must obtain this
review from a non-author; the JSON fields are declarations, not an identity proof.
EXECUTION.json exclusively reserves the attempt. Each child needs a private nonce
bound to CASE_LAUNCH plus live parent PID. CASE_CLAIM is exclusive and precedes
the sole raw source snapshot read. Root/ancestor/component symlinks are refused.
Snapshot size and SHA-256 must exactly match the inventory binding. Subsequent
native operations use only the private source.step, never aliases or scan files.

## Fixed serial budgets and failures

20 cases, 60 seconds and 4 GiB sampled process-tree RSS per whole case;
1,230 seconds campaign; 5,000,000 source bytes; 40,000,000 bytes per artifact;
1,000,000 combined stdout/stderr bytes. Parent checks limits every 50 ms.
No cap changes, reduced cohort, retries or silent failed-case removal. Samples
are not a continuous memory high-water proof. Kernel I/O, OS scheduling and
cleanup may add bounded supervisor overhead. Campaign reserves a full case
window before launch. Temporary-home/network sandboxing is not claimed.

Parent owns the existing V4-compatible OS admission lock/journal across each
native child, preventing overlap with maintained native workers. ADMITTING and
RUNNING precede native execution; QUIESCENT is written only after observed outer
exit and whole seen-group cleanup. Unknown cleanup leaves admission quarantined,
halts further launches and records all later cases NOT_EXECUTED. Exclusive case
and campaign receipts preserve failures and partial outputs. SIGKILL/host failure
may prevent terminal publication; EXECUTION still bars retry. Root observes the
actual outer exit and checks quiescence before independent POST.

Pure tests use temporary source-generated JSON and stubbed native modules only:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider -q tests/test_evaluate_vendor_cad_corpus_v1.py
```

Next authorized physical step: root-reviewed baseline freeze, independent PRE,
single execution, independent POST and full failure map. A mathematical candidate
covering all 20 unique CADs must be separately frozen after that map, without
reopening this baseline or claiming availability as accuracy.
