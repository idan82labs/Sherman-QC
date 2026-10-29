# Gold11 A0 provenance trace V5 — 2026-09-10

## Outcome

**Recovered eleven later-use A0 content-digest witnesses**, each tied to an
approved scan SHA in a retained EXP_CAD_0008 semantic-result file. All eleven
semantic-result files' current hashes match the exact summary/visual declarations
recorded in [V4](2026-09-10_gold11_a0_provenance_trace_v4.md). Every projected case
ID and scan SHA agrees with the approved V2 mapping. Each record reports
`status=COMPLETE` and `a0_availability=AVAILABLE`.

The A0 root/path is still **untraced, not established missing**. No A0 prediction
file was opened. Prediction geometry in the eleven EXP_CAD_0008 files was not
displayed, analyzed or scored. These are later-use digest witnesses, not a
recovered August22 A0 execution receipt, original endpoints, or full union.

## Separately authorized access and method

This pass was authorized after V4 completed, for at most eleven exact approved
semantic-result files already named and hashed by the primary summary. Up to
four additional linked metadata/source records were conditional on an explicit
A0 root/path emerging. None emerged, so **zero** additional records were read.

All eleven complete file byte hashes were checked first. Only after all matched
their expected values were JSON projections performed. The projected fields
were case ID, scan SHA, A0 SHA, availability, optional root/path fields, schema,
status, top-level key names and scalar source/config address fields. A separate
projection checked contract key names and address-valued fields; it found no
source/config/root/path address. The prediction, visibility-audit and
ownership-audit bodies were not displayed or analyzed. Complete-file hashing
reads bytes, but does not interpret their geometric content.

Every file uses schema `SHERMAN_QC_CAD_VISIBILITY_EVENT_FUSION_RESULT_V1`.
`a0_root`, `a0_path` and `a0_prediction_path` are **absent keys**, not explicit
null-valued historical declarations; the diagnostic projection rendered absent
keys as null. No address was inferred from that projection.

## Exact A0 digest witnesses

Tokens below are the exact SHA-to-token associations recovered in V2, not IDs
inferred from line counts. The A0 hash column is the retained `a0_sha256` field
inside a byte-verified later result, **not a hash computed from an A0 file in
this pass**. Full approved scan SHAs remain recorded in V2; the scan SHA in each
current metadata projection agrees with that case's approved SHA.

| Approved case | A0 pool token | Retained later-use `a0_sha256` |
|---|---|---|
| CG08506AE97D | C003 | `df42a588cf1747e358c9f8fd8704a5b3c7d01388e5b85cc6df271edf2680d060` |
| CG191DEC7F81 | C021 | `5c0ef9e5743e854550e40dea7808ec11570b897a076dd883dab67e1ae1f9b06a` |
| CG1E0BBA2933 | C011 | `2eac14dd768471a540460f60ac8a696ae04a7d0abfcb4880618bfd4965fb5353` |
| CG4BDD1371B6 | C008 | `bdefea6cfab50cec78907cc997b1ad28b1ac8755a1b448444017f7a6f6216d4a` |
| CG4FAF273093 | C022 | `af8ce6e7827b1adb9fea871ceb5406d28e2871301a145268290eca3708b2dcc9` |
| CG64C412CC73 | C015 | `f464aa39f3a16d1dd6272fc08a735b38c215ca8b7ad5b32dcdfe93101db61176` |
| CG65CD575B1C | C009 | `6160c39dee6998208c9f333855e9e402ed486282dddee7494b43a0104b0837a7` |
| CG84C8FA12D2 | C001 | `3befcc1d963ce9ab70a915e174c3be5503c5a81a46fa16c9f8883c23db9b7914` |
| CGE4B0549E41 | C002 | `eb0b050d970b48108c662f8e943002bfdaf6c3a01a978b4483ca1a6d0c83015f` |
| CGEBA9A869AC | C010 | `9e602684ecd345ace1a840743f0412d14d9106629c1a760dd1b0133d83fa8299` |
| CGF41F49066A | C012 | `02b0e6447e688951294e56a672ff42271ca74ebe8232345e386917aa2ed3b612` |

All eleven digest declarations are distinct lowercase64-hex strings. In the
previously read writer, `a0_sha256` is computed from the actual caller-root/token
file via `_sha(a0_path)`; the writer source's current hash matches the retained
result declaration. This source behavior plus the current-byte-verified later
records supports the description **later-use byte witness**, without proving
when or under which complete environment those records were originally created.

## Exact read ledger and current file hashes

The common path prefix for these eleven and only these eleven files was:

`output/research/autoresearch-loop/exp-cad-0008-visibility-event-fusion-v1/primary/cases/`

Append `<case>/semantic_result.json`. Each current SHA below equals the two
retained declarations in the V4 summary and visual manifest.

| Approved case | Current semantic-result SHA-256 | Verification |
|---|---|---|
| CG08506AE97D | `2c2748c9ba97e7deba6249f023ced5d2ed45c18bd88c4060d7958f2bf9cd200f` | MATCH |
| CG191DEC7F81 | `75691164ea477ecde68123589a09d270d997e2875e1ea4999d5150871968a5c5` | MATCH |
| CG1E0BBA2933 | `fe7376eb8722117b726dfe5c5ca3d88cd121d7366ab2906ca025df937212bed2` | MATCH |
| CG4BDD1371B6 | `f36069c00cc7284a3d52430b5842dc7515164f42bc59889d407fcf6a0f3d56b8` | MATCH |
| CG4FAF273093 | `f913e6f4e970c423891a578756319895179cec318a7afcdd2065db50d784218a` | MATCH |
| CG64C412CC73 | `37a51b2ed92fd1910e48b5195217a8e7b6fecc39d63fdcb5023a79fab2ca7f90` | MATCH |
| CG65CD575B1C | `5a64cc4e2e99f6665ba4081adbeb635bba0ee0c0eb13c18ca2a42d2c1d071a37` | MATCH |
| CG84C8FA12D2 | `23dc541588c03603f4313a876b88d1ad476459239a6ab6599057e88626c6f12e` | MATCH |
| CGE4B0549E41 | `aea50ddb3cea85c78223aa0d262263e8a98b4c70f8dd4eb50aa06ae3f1f1ab9f` | MATCH |
| CGEBA9A869AC | `28519beeb57d06c42829cbd472c4c94c6767be0f5d1f8838be0602a3b3105091` | MATCH |
| CGF41F49066A | `c188073a0eb1bca8d107c01809944a743d5e0e4b36faaa4748bdbfc8f7ecbf44` | MATCH |

No mismatched file or scan identity was observed; absence of an A0 address was
retained rather than repaired. No other-case record, raw scan/CAD, Gold geometry,
email/heldout, original A0 payload, runtime import, inference, scoring or test
execution occurred. Prior traces, frozen source/tests and retained files were
not edited. Only this additive V5 note was written.

## Remaining gap and next bounded action

The trace now has:

1. approved scan SHA to pool token, bound by the V2 manifest;
2. the exact later source-summary/semantic-file byte binding, recovered in V4;
3. per-approved-case A0 digest witnesses, recovered here.

It still lacks an actual A0 root address and the original August22 invocation.
A narrowly identified execution/index metadata record could bind a candidate
root to these tokens and digests. Only after such address authority is established
should a future pass read the eleven original A0 files and preserve any hash or
count mismatch. Do not perform a blind digest search over unrelated predictions,
guess roots from the33 count, substitute later geometry, or promote a recovered
historical metric to a new-engine score. The retained0.458 benchmark remains
distinct from the later rejected EXP_CAD_0008 reported0.46875.
