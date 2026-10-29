# Extrema V3 / context V4: bounded source-only review closure

## Decision

The specific accounting clarification and canonical-byte assertion findings are
CLOSED AT SOURCE. The efficacy wording/disclosure finding is also CLOSED AT
SOURCE: the author now explicitly treats coordinate/row reductions as partial
observations, discloses the invented 200K-batch regression, and disclaims a
retroactive efficacy gate. Net efficacy itself remains NOT QUALIFIED; this review
does not close that scientific/engineering evidence requirement.

No new concrete blocker was found in these narrow edits. This is a delta review
of the three findings in the original preliminary report, not a frozen PRE,
passed test, full delta certification, wiring/runner audit, integration approval,
physical result or promotion. Fable 5.1 / max remains the implementation owner.

## Exact examined bytes

The response was read completely. Candidate inspection was limited to the changed
accounting description/charges, efficacy observation block and literal-counter/
canonical-byte test assertions. The unchanged context source is identified by
the same hash as the original review, not re-audited in full here.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_raw_extrema_index_v3.py` | 23978 | `83dea6529003f1d5deae592c9d35e566a77e2c9f6f3af8f641928f5465845bc0` |
| `backend/cad_guided_two_plane_context_v4.py` | 26196 | `e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25` |
| `tests/test_cad_guided_raw_extrema_index_v3.py` | 33604 | `f78c18d146d4ca2f86e57a9690daa58251e92b2dee207de1a499a61ece6cc572` |
| `tests/test_cad_guided_two_plane_context_v4.py` | 37955 | `ef57b38f23249c8c1555d0c61fd05913a8068027147ca020b082679154d7a945` |
| `docs/research/2026-09-12_fable51_review_response_v1.md` | 3980 | `99458961d11b507a11a1a33e9640d42b7ca0d8f5669be01c981d664827ad1663` |

The preserved original report is
`docs/research/2026-09-12_fable51_extrema_v3_codex_preliminary_review.md`,
13471 bytes, SHA256
`7549a1e161ef2e97ac053e5df4fd9588178025e987b9b7e11323c3c27804b8af`.
It was not edited. Bindings identify the inspected state, not an atomic freeze;
subsequent edits require rebinding and appropriately scoped reinspection.

## Finding dispositions

### 1. Accounting convention: closed at source

`backend/cad_guided_raw_extrema_index_v3.py:22-32` now states the intended
index-visit mapping and retains the abstract-work, not hardware-operation/RSS,
qualification. At lines 257-259 it charges `S` before creating the new working
permutation copy. At lines 267-274 it charges `2*n` before a partition's key
gather and permutation write, plus the existing coordinate and sort charges.
This closes the specific newly introduced copy/per-partition charge mismatch
identified in the preliminary review under the declared predecessor convention.

The literal standalone tests now expect `4S` for a single leaf and `6S` for one
partition (`tests/test_cad_guided_raw_extrema_index_v3.py:552-574`). The connected
single-leaf cumulative test expects `4*memberships + evaluated_rows`
(`tests/test_cad_guided_two_plane_context_v4.py:527-547`). The unchanged context
delegates inherited index charges to its parent as examined in the first review.
No old predecessor was edited by this reviewer, and this delta needs no such edit.

These are consistent source assertions, not executed validations. They do not
claim a complete count of all native/Python operations or make inherited
unexpanded scalar-bound work a newly introduced defect. Higher attempted index
charges can change capacity refusal points; that is not a changed extremum or
permission to enlarge a started cap.

### 2. Efficacy wording and disclosure: closed at source; efficacy not qualified

The standalone folded-sheet test now times each entire public build/query batch
before oracle checking, prints CPU, index visits, sort and coordinate counters,
and computes the explicitly visible subtotal
`coordinate_visits + sort_work + index_visits`
(`tests/test_cad_guided_raw_extrema_index_v3.py:378-406`). Its comments at
lines 408-412 explicitly say the surviving row/coordinate assertions are NOT a
net-cost or speedup decision and require a separately frozen successor model,
CPU observation and retained regressions/refusals.

The author's response, under “Extrema V3 / context V4,” discloses that its
200K-row/19-query development observation regressed from about 30.8M to 48.6M
in that subtotal and from about 0.24 to 0.34 seconds CPU despite reduced row/
coordinate evaluation. It also labels all explored leaf sizes and the 800K
observations consumed development, not a gate. Those are author-reported
observations read from the released narrative, NOT reproduced or independently
POST-audited numbers in this review. No numerical result is accepted here as
frozen efficacy evidence.

The printed `common_work` field remains a three-ledger subtotal: it does not
include node visits, row visits or hash bytes, nor expand the original rounded
world-bound scalar arithmetic. This is acceptable as a disclosed development
observation, but it must not be silently relabeled a complete operation count.
Any later prospective common-work rule must explicitly map the included costs
and exclusions; the complete public-batch CPU observation is separate evidence.
No particular old V2 efficacy criterion is imposed, and no new gate or execution
is authorized by closing this wording/disclosure finding.

### 3. Full-geometry byte assertion: closed at source

`tests/test_cad_guided_two_plane_context_v4.py:713-714` adds a deterministic JSON
byte encoder with sorted keys, compact separators and nonfinite rejection.
The renamed test at line 730 compares canonical bytes of the work-stripped full
geometry evidence at line 742 and interval bytes at line 745. The existing
stripping helper removes declared work/version/partition fields, not geometry.
Unlike Python float equality, these encoded finite-float values distinguish
negative zero from positive zero. This implements the narrowly requested check
for the compared JSON evidence; it is not a comparison of every internal NumPy
allocation or a universal V3/V4 equivalence proof.

The updated assertion has not been run by this reviewer. The first review's
conditional numerical reasoning and all limits remain; completed-answer
preservation does not imply identical refusal behavior under different work
consumption or declared caps.

## Verification and handoff

Only read-only text/hash/size inspection and creation of this NEW report were
performed. No target import, collection, test, compilation, run, new benchmark,
raw/Gold/artifact/cache/credential access, engine/test edit, or broader wiring/
runner audit occurred. The original preliminary report is retained unchanged.

These specific source findings need no additional reviewer-driven repair before
the owner can prepare its next exact-source gate. Execution, useful net efficacy,
connected workflow behavior and any physical successor still require their own
prospective protocol, independent exact PRE, resource controls and honest result
accounting. No new F1 or physical improvement is asserted.
