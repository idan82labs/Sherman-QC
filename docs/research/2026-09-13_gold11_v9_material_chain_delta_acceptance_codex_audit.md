# V9 material chain: bounded delta acceptance

Date: 2026-09-13. Reviewer: Codex/Astra `v9_material_identity_chain`.
Scope: only the previous inner-eligibility P1 and two boolean-publication P2
sites, plus their declared regression tests, at the 05:06 stable checkpoint.
AGENTS/current status and the complete appended response section were read.
Source/test reads and hashes only; no imports, tests, data, Gold, attempts,
controls or caches were accessed. Only this report was written.

## Decision

**Accept the scoped source fixes.** No currently defined unverified material
status can grant an inner-specific claim, and both reported boolean aliases
are rejected at their local publication sites. Generic contact geometry and
its scoring path remain available. This is source-delta acceptance, not a
physical PRE or independently executed test result.

One test-evidence correction remains: two cover mutation branches are present
but missing from the test's parameter list. This does not reopen the correct
source fix; it prevents claiming those two mutation checks actually ran.

## Exact reviewed hashes

The response matched declared SHA256
`94ec97a96d4496548f5870d7cbcfbf6f43872c8e9cbcbd147fec8137aada23d1`.
The following source/test hashes matched its appended delta table:

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `12bb9ce9e595ada6b4cf86947089ef96f2158ddf36009584442e0a414e6ae008` |
| `backend/cad_guided_context_measurement_v2.py` | `9b48c72b4e3eb87899f8e6911929a869c63d0182dd95e254e5deb5671e623f7c` |
| `backend/cad_guided_support_cover_measurement_v3.py` | `6c6bd75813f6e985fd4e841eb401ae330a5be614a6e874b2e5704256e00ca901` |
| `scripts/gold11_contact_scoring_v3.py` | `4f0cbea8cc97056295ab49092465f27230d0707ff53eaa8915f78feea031927b` |
| `scripts/gold11_schedule_projection_v4.py` | `32029cbb813381e360bed791e97701cdfb6735cfa6fa12f9a7f90d2477b95708` |
| `tests/test_cad_guided_measurement_v2.py` | `6573fd2643af819ffc97f403027476a296a3e7b953e6d76db8d2f7afa85db95f` |
| `tests/test_cad_guided_context_measurement_v2.py` | `f001906219048910d0742ff96d327b38efb53537e1b90a26c79ada60f6f11b47` |
| `tests/test_cad_guided_support_cover_measurement_v3.py` | `3716947e2caf8c49e931863a128f810d30f1868ffdec490375cfb6408daf1013` |
| `tests/test_gold11_contact_scoring_v3.py` | `5e84791e2a2547b872a16a2f3b39079e1de95f594b411258829197f5188c18dd` |
| `tests/test_cad_guided_v9_connected_chain_v1.py` | `8b0399202af65ef5e63d92998cee97e999cc437832e2770c796ff6c14962ed46` |

## Reconciled findings

- **Prior P1 closed:** kernel and contact V3 both use an empty
  `INNER_FACE_ELIGIBLE_STATUSES` tuple. Existing status-to-eligibility consumers
  therefore publish false, the generic carrier role, observed surface radius
  rather than inner radius, and unavailable nominal inner-radius deviation.
  The two-layer selection remains explicitly a side hypothesis; provenance
  and measured contact geometry are retained. The connected two-skin test
  now asserts zero inner-eligible carriers, one unresolved generic carrier,
  no canonical inner deviation and the same endpoint transport/numeric
  matcher-arithmetic result. V4 reconstruction still compares against direct
  V3 on the emitted bytes; its source hash is unchanged.
- **Prior context P2 closed:** association eligibility uses `is` with an
  explicitly boolean record value. The context mutation list now actually
  schedules association, record, window and per-context 0-for-false cases.
- **Prior cover P2 closed at source:** finite-instance eligibility uses `is`
  against the derived boolean. Group identity remains checked by the strict
  context helper.

No endpoint or matching-rule change is required by these fixes. This delta
review did not reopen the previously accepted interface audit or the broader
kernel/registration mathematics. Physical observed count, conformance and
historical comparability remain unavailable/unestablished as before.

## Small test-evidence correction

In `tests/test_cad_guided_support_cover_measurement_v3.py`, the branches for
`instance_eligibility_int` and `group_eligibility_int` are at lines 482–483,
but neither is listed in the `fault` parametrization at lines 444–455. They
are therefore not exercised by that test at this exact hash. The appended
review response's statement that these checks ran is unsupported by the test
source; the unchanged reported cover count of 82 is consistent with the
omission.

Add those two strings to the existing matrix and execute that small focused
selection, or correct the execution claim. No engine modification or broader
new audit is needed for this test-only correction. The root was notified
promptly. The author's reported 855-pass result was not independently rerun
by this reviewer and must not be described as such.
