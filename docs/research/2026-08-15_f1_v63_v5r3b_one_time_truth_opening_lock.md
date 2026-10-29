# F1 V63 V5R3B one-time truth-opening lock

Status: **LOCKED AFTER PRIMARY/REPLAY FREEZE, BEFORE PRIVATE TRUTH READ**

- V63 execution manifest SHA-256:
  `65d3418e106601a823a2648d8a53eb14abbb8ea4987bd48c3aadcdbd948af886`
- Score-and-render implementation SHA-256:
  `da1078ad31c968f274bf7b0066c826e27f95cfdbcad09cde63163bfae597d24a`
- V63 primary/replay semantic identity: true for all six cases.
- Frozen V52-origin count: 3.
- Frozen V52 deletion count: 3.

The scorer reads each case's private truth exactly once, uses that same
in-memory object to score frozen V62 and V63, renders both rows, then discards
it. No inference code can access truth. Thresholds remain the established
5-degree, 4-mm, 0.60/0.60 one-to-one line-matching contract. Any automatic or
manual retention failure is terminal; it cannot authorize a rerun, replacement
input, or parameter rescue.
