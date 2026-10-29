# V51 new-corpus precommit V1 invalidation

The first source-binding lock was sealed with SHA-256
`19e1ac4ead3dd6b153dc5d5f1042bebd619a08d3615b020049c5d029392e2dc3`.
Its first fail-closed preflight attempt terminated before the audit function,
corpus assembly, truth access, or detector invocation because direct execution
of the preflight script did not place the repository root on `sys.path` and
could not import `research.f1_multifamily_selector_v2.model`.

V1 remains immutable as an invalid protocol artifact. The only correction is
the launcher import path plus the append-only event expectation needed to
record this invalidation. A V2 binding supersedes V1 before any new-corpus
membership exists. Detector/model composition, source-only predicates,
thresholds, gates, matcher, renderer semantics, models, and action token are
unchanged.
