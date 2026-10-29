# Diagnostic V1 — independent two-literal release audit

Date: 2026-09-12. Reviewer: `process_v2_qualification`, independent of the
runner/test/protocol authors and root's release edit. Decision: GREEN FOR THE
EXACT TWO-LITERAL DELTA AND SUBSEQUENT SOURCE-ONLY FREEZE. This is not the
physical frozen PRE, a launch receipt or permission to inspect physical inputs.

Only named source bytes and the accepted source POST/review evidence were read.
Application Python 3.11.15 with `-I -S -B` was used for stdlib AST/literal and
hash inspection; no target compilation, import, collection or execution took
place. No physical path, runtime target, control, attempt, archive or private
cache was accessed, and no process query or signal was issued. Only this new
document was written.

## Exact delta and custody

| Revision | Bytes | SHA256 |
| --- | ---: | --- |
| Qualified unreleased runner | 57923 | 08f9e50b52f0225ca54e1920ef21fed2d0840a19941fb20b338b43e6b141a44f |
| New released runner | 57936 | 47cdd2c716ca82f65c8994a095826a212a9f0c485b62306dc24d9a4355bc50a2 |

Source: `scripts/run_gold11_context_v3_diagnostic_v1.py`. The only edits are:

| Literal | Before | After |
| --- | --- | --- |
| RELEASE_AUTHORIZED | False | True |
| TIMING_RELEASE_STATUS | UNRELEASED_PROSPECTIVE_SIX_SLOT_LIMITS | REVIEWED_EXACT_SIX_SLOT_DIAGNOSTIC_V1_TIMING_RELEASE |

Both exact new assignments occur once. Reversing only these two assignments in
memory reproduces the complete qualified 57923-byte source SHA above. This
establishes byte equality outside the authorized delta, not merely similar ASTs.
The released source parses; no runtime release function was invoked.

The exact 64-file physical source declaration was derived through source AST
literals and hashed. The other 63 files total 1639565 bytes. Replacing only the
runner hash in the current canonical path-to-SHA256 map with the qualified hash
reproduces the accepted unreleased map:
`fecd0786588b1bff72389b4b295b7328c04b3c125217b1d2dc67f4eb27a36d08`.
Thus all 63 other named source bindings are unchanged. The released 64-file set
totals 1697501 bytes; its canonical compact sorted path-to-SHA256 map digest is
`f540758ad5b6d3d3434c508ee80db05e8bb7593e8eb8eae1e8a3e8d008fb0c9e`.

In particular the V6 provider remains166049 bytes/SHA256
`7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80`;
the physical protocol remains12845 bytes/SHA256
`da2db6127916ec2b714fa03c3f04cc901c0ff1263ed35db1b1c8d39d87822ec3`;
the design remains19707 bytes/SHA256
`a79640d7a98ba2e08737bda14004fbbe90d28ff8ff1bb9b207547a840c90e108`.
Historical UNRELEASED/module-draft prose is explicitly superseded only as to
these two literal gates by this audit. It does not conceal a numerical or
scientific change, nor does the new True literal replace the required PRE.

## Qualification inherited, without relabelling its scope

The complete accepted 46-outcome POST was read at
`docs/research/2026-09-12_gold11_context_v3_diagnostic_v1_source_post_run_audit.md`,
9516 bytes, SHA256
`aa5dbf9e4ffc4b45c235b0e3d3cfc51c29c60be84ccb43f8ae4a4005412495d0`.
It records the sole invented gate's46 ordinary passing phase triplets, zero
other dispositions, exit0, complete publication and unchanged source/runtime/
PRE/archive custody. Root supplied actual terminal65db61 and externalps28068f;
this reviewer did not reinspect that attempt or query processes.

Those tests exercised the unreleased source and detached invented profile,
authority, artifact/native-helper seams. This audit does not call their pass a
test execution of the newly True literal or of physical data/native inference.
The narrow release change needs this source comparison, not another unchanged
46/132/sampler/projection/connected/dense suite run. The POST's explicit limits
remain: no qualification of every live cleanup race, all optional dispatch
adversaries or maximal combined receipt capacity, and no actual physical
projection or stage-cadence result from that invented gate.

The accepted source review remains18919 bytes/SHA256
`abb9f698267aa9f10f8fcb21a681e125e02650e548920f5aa7eb13a820153041`.
Its explicit application-3.11 AST recipe and resolved generic input-binding
abort reason are unchanged. Input-scope review remains5144 bytes/SHA256
`c30ae7725a24c4a6c8920d5784f29e6d6d64ae8436e609cb5dced80559da1e70`.
This audit and those review documents are external evidence, not newly added
runner SOURCES or mutable selectors.

## Unchanged limits and scope; exact next boundary

No case selection, arm order/request, engine configuration, threshold, provider
definition, source path, no-score policy, abort/retention rule, private-cache
lifecycle or inventory algorithm changed. Three fixed consumed cases remain
six serial control/challenger slots. The source-declared raw17 map remains
`2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157`;
only the three selected scan/CAD pairs are future diagnostic inputs. No Gold,
reference selector/scorer, subset F1 or all11 launch is introduced.

Case limits remain3540wall/1860CPU seconds,8GiB sampled worker tree,
4GiB coordinator HWM and12GiB outer budget; campaign25200 seconds includes
2400 terminal reserve. The disclosed at-most-one extra one-second emergency
administrative veto is not an engine/input/inventory extension. Fixed input,
result, projection and log byte caps remain unchanged. Scientific negatives
remain six-slot outcomes, not a requirement to obtain all-positive geometry.

After root reads and accepts this exact audit, the next source-only operation
is the existing `--freeze` mode. It must exclusively create the new fixed
diagnostic attempt's64 source files and FROZEN.json without physical input
access. A verified external archive and independently released exact frozen
source/control/runtime inventory must follow. The independent physical PRE
must bind the actual frozen bundle/profile, selected cases, six slots, limits
and no-score policy and reconcile the source-derived input table. Neither a
successful freeze nor these release flags substitutes for that PRE, root's
acceptance or the external no-overlap observation before the sole --execute.

No concrete blocker remains for this exact release-only delta. No actual frozen
attempt has been inspected; physical input access or launch is not authorized
by this document.
