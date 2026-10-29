# V7 finalization byte accounting — independent source-only review

Date: 2026-09-12. Reviewer: independent AI technical reviewer
`process_v2_qualification`, not a human approver or execution authority.

## Conclusion and scope

No concrete cap breach was found in the declared owned-producer successful
finalization shape. A conservative, conditional `COMPLETED.json` construction is
259,386 bytes including newline: only 2,758 bytes below its 262,144-byte cap.
The actual contact-scoring output schema plus V7 enrichment has a separate
conservative bound of 12,285,070 bytes including newline, below 33,554,432.
These are source-accounting bounds, not measured runtime artifact sizes,
universal bounds for arbitrary dictionaries, or whole-process memory bounds.

Only named source text, AST/literal data and small synthetic metadata shapes were
read or counted using application Python 3.11 with `-I -S -B`. Canonical JSON was
counted without constructing large payloads: sorted keys, compact separators,
default `ensure_ascii=True`, `allow_nan=False`, and a publication newline where
applicable. No target module was imported, compiled or executed; no tests ran.
No physical input, Gold, attempt/control artifact or private cache was inspected.
The separate six-slot diagnostic remains live and supplies no empirical terminal
size to this review. No frozen request proof or source-gate file was edited.

## Exact source identities

Paths below are repository-relative. The additional projection/request premise
is already pinned in the separate, unchanged V7 transport-accounting review.

| Source | Bytes | SHA-256 |
|---|---:|---|
| `scripts/run_gold11_mesh_development_v7.py` | 169028 | `27737eff034891e8f7a6d09ba63f5bd17a3947c67ffd553ddfb73a931fef5cb3` |
| `scripts/run_gold11_mesh_development_v6.py` | 166049 | `7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80` |
| `scripts/gold11_terminal_artifacts_v4.py` | 24341 | `3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838` |
| `scripts/gold11_owned_process_supervision_v3.py` | 42751 | `9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea` |
| `scripts/gold11_contact_scoring_v2.py` | 50413 | `9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff` |
| `scripts/gold11_mesh_scoring_v1.py` | 13492 | `aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234` |
| `scripts/evaluate_f1_against_manual_gold.py` | 13134 | `1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62` |
| `scripts/f1_physical_evaluation_contract_v2.py` | 8329 | `28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7` |

The V7 design is 8,879 bytes,
`55e905a7bffd25847426826d498624a4ad5b6387d829e66217c03969ba2db403`.

## 119 scientific files is a constrained present-file count

`_campaign_schema`, `verify_parent_artifacts`, `verify_campaign_inventory` and
`execute` must be read together. There are 70 source files, not 70 scientific
outputs. Each of 22 case/arm leaves can have STARTED, COMPLETED, SCORING_INPUT,
OUTPUT and at most one RESULT or RESULT_PARTIAL. The parent rejects an extra
unlisted complete/partial result. Thus a valid maximum leaf shape has 110 files.
Nine coordinator/scorer scientific files may coexist; SCORER_RESULT_PARTIAL is
the alternative to the complete scorer result, not a tenth accepted success
file. Consequently the successful scientific maximum is 110 + 9 = 119.

The schema declares more alternatives: 132 leaf scientific file paths plus 10
root scientific paths. Treating all alternatives as simultaneously admissible
would produce 142, not 119, and would fail the 128-file/custody gates. The nine
spare scientific slots are not permission for unknown entries. Unknown metadata
is counted and vetoed. Source, dispatch and control categories are separate.

| Inventory phase | Schema files | Present files, maximal valid shape | Non-root directory entries | Scientific files | Metadata-entry limit |
|---|---:|---:|---:|---:|---:|
| Before parent COMPLETED | 265 | 240 | 41 | 119 | 434 |
| After COMPLETED | 266 | 241 | 41 | 119 | 435 |
| After FINAL_GATE | 267 | 242 | 41 | 119 | 436 |

There are 42 directories including root and 46 possible issued/claimed dispatch
files (22 inference workers plus scorer). Before parent COMPLETED, the 240
present-file construction consists of 70 source, 119 scientific, 46 dispatch and
five root control files. It conservatively combines all dispatches with the
longer partial-result names even where that exact combination is unreachable.
The final schema read-byte bound, including control re-reads and dispatch
allowances, is 6,566,457,344 bytes, below 8 GiB. This is a quota calculation, not
observed bytes or a synchronous-I/O time guarantee.

## Parent terminal envelope: conditional narrow fit

The terminal does not embed 22 full workflow RESULT/COMPLETED payloads. Its
`terminal_case_records` contain the two terminal-helper publication summaries
per leaf (identity, disposition, size and hash). Artifact evidence contains
119 hash/size records, 24 log summaries, at most 22 required leaf dispatch IDs,
fixed receipt hashes, and one campaign inventory. Source evidence contains two
70-source hash maps, runtime/control equality flags and five control hashes,
not the complete FROZEN/PRE documents. Later inventories are bound by hashes in
FINAL_GATE/stdout, not recursively inserted into parent COMPLETED.

The following assumptions are essential to the constructed upper shape:

- Exact current 70-source paths and schema names; 11 fixed ASCII case tokens,
  two arms, fixed generated IDs, and 64-character ASCII hash values.
- Trusted local filesystem identities and ordinary producer counters have
  decimal representations at most 20 characters. This allowance includes signs
  for signed timestamps; mode/link-count fields are over-allocated to the same
  width. Finite binary64 numeric fields receive 24 characters.
- The owned generated ASCII coordinator-cache path is at most 4,096 bytes;
  this is not a bound on an arbitrary supplied Unicode path or foreign receipt.
- Semantically successful finalization: all source/custody checks complete,
  error/failure lists empty, coordinator exit zero, no budget stop, strict
  quiescence, no unresolved native transition or first transition failure.
  Engine refusals can still be valid scientific negatives within this shape.
- Only fields emitted by this exact owned supervisor/helper chain are present.
  No externally augmented receipt or unbounded exception text is admitted to
  the successful-shape accounting.

The supervisor event-history accounting is important. Each history is capped
at 16. A successful native exit transition records ENTERED followed by RESOLVED;
the RESOLVED event embeds its full ENTERED record. The retained list is a prefix,
not a sliding tail. Sixteen retained successful records therefore contain at
most eight such pairs, not sixteen independent nested RESOLVED records. The
model allocates full 14-scalar before/after identities even for absence, longest
successful outcome strings, and 20/24-character scalar widths. Eight ENTERED
plus eight RESOLVED-with-entry records occupy 26,097 bytes including list
framing. Identity retirement history separately allocates 16 full before/after
events. Successful query errors do not add another error history: those errors
propagate to budget/failure handling, outside the stipulated successful shape.

| Canonically counted component | Bytes |
|---|---:|
| Embedded inventory before COMPLETED | 160288 |
| Artifact evidence including that inventory | 185358 |
| Final source evidence | 16175 |
| 22 terminal-helper record pairs | 17546 |
| Whole COMPLETED except coordinator receipt, including newline | 223904 |
| Dominating clean coordinator receipt | 35482 |
| Combined COMPLETED | **259386** |
| Cap minus combined size | **2758** |

The count is a conservative metadata construction, not a claim that every
allocated extremum occurs together. Its canonical inventory hash is
`57b063a8f58fd496b22396fecb6e731726385772209f126807a3c5d5a481f07f`;
clean-receipt construction hash is
`80013fdfc69a8e4cddcbaafdd98008c336d6be1e039ce755d90474030c8c6110`.
These identify inert accounting witnesses, not runtime evidence. The same
construction gives 160,758 and 161,234 bytes for subsequent inventories.

This is deliberately not a proof that every failure terminal fits. Failure
lists, Unicode exception representations, incomplete/error inventories and
unresolved/error supervisor histories need a different allowance. A cap breach
there remains a publication veto; this review does not promise a full COMPLETED
document for every failure. The real `len(encoded(completed))+1` check and
`PARENT_TERMINAL_CONTROL_BYTE_BUDGET` stop remain necessary and unchanged.

## Actual scorer schema and its V7 enrichment

The scorer's 32 MiB output cap alone would not prove that adding enrichment
fits. The stronger premise here is its actual schema, not an arbitrary accepted
32 MiB JSON blob. `_lane` exposes summaries, six family summaries and 11 case
dispositions; it does not return full prediction/reference geometry arrays or
the aggregate helper's internal per-case data structure.

For two arms, exact source cardinalities give 142 total reference-ID occurrences
across matches/unmatched references, and at most 11,264 prediction-ID occurrences
across matches/unmatched predictions. Each ID appears once in its respective
partition. Reference identifiers have at most 512 Python characters; allowing
12 escaped ASCII bytes per character gives 6,147 bytes including quotes/comma.
Owned prediction identifiers are exactly 64 ASCII hex characters: 67 bytes with
quotes/comma. There are at most 142 matches; 512 bytes per match, excluding the
separately counted IDs, conservatively covers seven finite numeric metrics,
their names and framing. Case reason/phase lengths are bounded at 256/128
characters, diagnostics have fixed keys, binding hashes are 64 hex, and actual
producer `supplied_bytes` is a bounded result-byte count, not an arbitrary
foreign integer.

A conservative component allowance is:

```
142*6147 + 11264*67 + 142*512
  + 22*16384 + 14*2048 + 32768 = 2,122,154 bytes < 3 MiB.
```

The 22 allowances of 16 KiB cover remaining per-case fields/framing; the 14
allowances of 2 KiB cover overall plus six family scalar summaries per arm.
The final 32 KiB covers fixed root, paired/family deltas, request metadata and
claim strings. These deliberately generous allowances do not contain geometry
or an unbounded echoed input. They depend on the exact owned-producer chain and
the current fixed ASCII case/family map, not arbitrary caller dictionaries.

V7 then adds 27 named enrichment fields. `workflow_stage_diagnostics` enforces
32 KiB per returned summary (or emits its small fallback), hence 22 summaries
cost at most 720,896 bytes plus 380 bytes of arm/case map framing. The coordinate
frame map selects `coordinate_frame` subtrees, or null, from the canonical
complete selected-reference map already successfully published under 8 MiB.
The validated row shape contains other fields and bends, so this replacement
does not expand that canonical map. This is a source premise, not a Gold read.

The source-literal enrichment skeleton is 5,161 bytes with four null placeholders.
Removing those four nulls and adding the exact arm requests (14,692) and bindings
(9,620) gives 29,457 bytes excluding stage and coordinate maps. Its canonical
hash is `37d56e6addeb6acfd2754c5e4840a1c46c6513da049f1a37a9d6031d2f55de8c`.
Counting both original/enrichment object braces is conservative when merging.
Thus:

```
3*1048576 + 8*1048576 + 22*32768 + 29457 + 380 + 1
  = 12,285,070 bytes < 33,554,432; slack = 21,269,362 bytes.
```

There is no concrete enriched-report overflow in this conditional actual-schema
bound. The unchanged actual encoded-length/publication checks still govern.
This supplements, rather than edits, the earlier request-only accounting; it
does not turn that request proof into a process-memory or arbitrary-report proof.

## Smallest remaining exact freeze check

At the separately authorized full11 freeze/PRE, reconcile the exact source and
path set, fixed case/family/request/binding lengths and generated ASCII cache-path
policy against this construction. Check the actual newly frozen control-envelope
serialized lengths against their existing caps; in particular, a complete
COORDINATOR_VERIFIED shape includes fixed source/runtime metadata that can be
accounted before dispatch. Confirm the trusted stat-scalar representation
assumption and unchanged event limits. Actual frozen controls settle their own
fixed-byte contribution, not future runtime event/error histories.

No new broad test or physical rehearsal is requested by this accounting. Keep
the actual request, enriched report, manifest and terminal encoded-length vetoes.
If the narrow parent-control assumptions do not hold for the final exact freeze,
re-account that specific envelope before launch. This document itself grants no
freeze, raw access, retry, physical execution, score release or promotion.
