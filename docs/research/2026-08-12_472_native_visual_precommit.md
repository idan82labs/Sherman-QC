# 47251002_02 native-visual line precommit

Status: frozen before V5 inference.

The source-only candidate pool contained only plane-lattice hypotheses: four on H01 and five on H02. Native-resolution inspection grouped them into two long physical bend axes per scan plus parallel aliases.

- H01: select M002 and M003. Reject M001 as a 9.091 mm alias of M003 and M004 as an 8.987 mm alias of M002.
- H02: select M002 and M003. Reject M001 as a 9.152 mm alias of M003; reject M004/M005 as aliases of M002.

This is explicitly an incomplete manual line set. The raw scans visibly contain knee/head/lip/return fold families for which the generator emitted no candidate line. The precommit does not assert that the part has only two bends.

The V/X mapping became visible after the raw multiview inspection but before this exact candidate-line record was written. Therefore this is evaluator-visible holdout evidence, not a pristine role-blind result.
