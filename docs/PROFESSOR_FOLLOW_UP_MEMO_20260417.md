# Professor Addendum: Invariant-Fail Tranche

Professor,

We have implemented hard gating for bend-specific claims. The remaining gap is narrower: when can the part fail even if the exact bend attribution is still unresolved?

We want to use a deterministic v1 rule:
- correspondence stays a hard prerequisite for bend-specific claims
- a part may still be `AUTO_FAIL` when every admissible remaining interpretation still fails
- otherwise unresolved attribution stays `HOLD`

Please confirm one thing only:
- Is this deterministic invariant-fail approximation acceptable as the v1 operational rule for the ambiguity-invariant fail tranche, with the current two cases in scope: trusted-frame positional contradiction under ambiguous attribution, and required-bend absence when every admissible interpretation still implies failure?

If not, tell us the exact boundary you want changed.
