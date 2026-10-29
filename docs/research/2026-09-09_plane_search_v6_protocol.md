# V6 repeated-layout fair plane search — prospective mechanism

Date: 2026-09-09. Source-only successor; no execution result is implied.
The terminal V5 search and V6 connected attempts remain frozen. This successor
changes scheduling, not geometry, thresholds, measured answers or scoring.

## Engineering reason and scope

The [retained V6 connected negative](2026-09-09_mesh_connected_planes_v6_result.md)
found repeated inner layouts exposing the same four of sixty compatible CAD
assignments. No generated nominal pose met the unchanged 0.20 mm oracle, including
those omitted by the later finite-prefilter quota. An earlier twelve-of-twenty-one
pair-window selection also omitted relevant combinations. Neither observation
proves that any newly exposed combination will recover the correct pose.

Owned source: `backend/cad_guided_plane_search_v6.py`; exact source SHA-256
`aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676`.
This is a copy of frozen V5 `13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98`
with V6 API/schema names, two scheduling replacements, invocation-local bounded
state and diagnostic fields. All declared-plane typing/admission, normal-angle
intervals, proper Wahba rotation, intrinsic-intersection translation, arithmetic
and conditioning gates, bank/triplet scheduling, signed ledgers and numeric caps
remain as specified by the [V5 mechanism](2026-09-09_plane_search_v5_protocol.md).
There is no physical IO, application startup, learned ranking, case identifier,
Gold, downstream residual or reference-driven schedule selection.

Public names are `PlaneEquationV6`, `ObservedPlaneV6`, `PlaneSearchPolicyV6`,
`PlaneSearchBudgetsV6` and `propose_plane_search_v6`. The explicit millimetre
assumption remains a declaration, not calibration. Supports and normal cones are
unverified here; finite full-raw support verification belongs to the initializer.

## Exact algorithm and limited guarantee

For layout length L and quota q, let m = min(L,q). If m = 0, choose nothing and
make no coverage claim. Otherwise partition ordered ranks into m integer bins:

```text
a[j] = floor(j*L/m), b[j] = floor((j+1)*L/m)
pick[j,k] = a[j] + (k mod (b[j]-a[j]))
```

Here k is the zero-based occurrence of that exact layout in this invocation.
Because L >= m, each bin is nonempty. The bins are disjoint and partition all L
ranks, so every call selects exactly m distinct increasing positions, never more
than its quota. Each bin is a cycle of length at most K = ceil(L/m). Thus every
rank is exposed in any K consecutive occurrences of the same exact layout;
within each bin, exposure counts differ by at most one. This elementary integer
proof does not require a new geometric theorem or empirical fitted parameter.

The first stage keys occurrences by exact `(start, stop)` in the invocation's
single immutable sorted CAD pair table. Its quota remains twelve. The second
stage keys occurrences by the exact populated ordered compatible triple array,
canonically detached as little-endian signed-int32 bytes. Its quota remains four.
Unused allocation tails, row sorting, sets and approximate identities are not
used. A SHA-256 and first-seen layout index are diagnostic only: equality is exact
bytes equality, not digest equality. Identical layouts recur independently of
unrelated interleaving; both registries reset for every public call.

For example, a repeated 21-rank window is fully exposed within two occurrences
at quota twelve, and a repeated 60-rank layout within fifteen at quota four.
These examples are not required physical-case identities or pose guarantees.

The guarantee is deliberately narrower than correspondence completeness:

- Identical pair windows can have different exact compatibility intervals.
  Pair exposure is not proof of interval acceptance.
- A compatible layout can recur on different scan triplets. Covering its CAD
  ranks over those occurrences does not try every rank on each scan triplet.
- Fair pair scheduling can change compatible layouts so some occur only once.
  The two guarantees do not multiply into an exhaustive joint search.
- The bank, scan-triplet subset, later 32-unsigned finite-prefilter and 48-refine
  schedules remain incomplete. Correct correspondences can still be omitted.
- Stable import-local IDs break ties; renumbering invariance, global uniqueness
  and recovery under an adversarial correspondence ordering are not claimed.

## Budgets, diagnostics and failure semantics

All V5 geometric/work limits remain: at most 128 bank planes, 256 scan triplets,
2,096,128 CAD pairs, 32,000,000 join interval tests, twelve selected CAD pairs and
four unsigned assignments per triplet, 49,104 temporary compatible rows,
1,024 unsigned/8,192 signed assignments, and 64 MiB detailed JSON report.

Each new registry has at most `max_scan_triplets` entries (at most 256). Canonical
compatible-key payload has an additional hard 192 MiB cumulative limit. The
single key is size-preflighted before canonical allocation; a new key is also
checked against cumulative payload and registry limits before insertion. Under
the unchanged row/call maxima, total created key payload is additionally bounded
by 49,104 * 3 * 4 * 256 = 150,847,488 bytes. Repeated keys do not consume retained
payload twice; existing keys can be reused at the retained cap. Canonicalization
can transiently allocate another bounded key before identity is determined.
These are payload/work statements, not a Python object, hashing-time or RSS bound;
the external supervisor still enforces whole-process resources.

Each reached triplet records its exact window or compatible-layout identity,
occurrence, selected positions and conditional coverage horizon. Statistics retain
registry sizes, reached schedule occurrences, retained logical key bytes and
bytes newly SHA-hashed for diagnostics. The last counter is not all Python hash
or equality work. Schedules are attempted selections if later work fails.

An ordinary registry, allocation, numeric, join or bookkeeping failure invalidates
all usable poses, including earlier positives. Active triplets become interrupted;
completed and genuinely unstarted triplets remain distinguishable. Reached CAD
usage is still exported. A final report overflow publishes only the pre-existing
bounded refusal envelope, not a usable prefix or a geometric negative. Exception
handling still catches `Exception`, not `BaseException`.

## Qualification and promotion gate

Independent Astra source/math review is GREEN for the exact source above; it
confirms only the stated scheduling contract, unchanged geometry and failure
semantics. It is not execution or connected recovery evidence.

The new independently authored test file must retain all 77 V5 controls (only
API/schema and scheduling-specific oracles change), and add independent integer
bin/identity/cap/failure adversaries. The exact test protocol and root supervisor
closure must be frozen and independently audited before a single first run.
One serial pure job: 60 seconds parent wall, 45 seconds worker CPU, 1 GiB
process-tree RSS, 1 MiB stdout. No native or physical input is permitted here.

Only after this gate may the V7 initializer and connected workflow successor be
tested under their separately frozen protocols. The four-case connected matrix
must retain all three nominal geometry oracles plus the missing-arc abstention;
unchanged quotas do not justify relaxing acceptance. A new paired consumed Gold11
run still requires its complete frozen protocol and independent audit. None of
these synthetic results is a physical F1 score, generalization proof or release.
