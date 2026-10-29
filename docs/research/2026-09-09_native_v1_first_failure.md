# Native V1 first constructed execution — retained negative result

The independently reviewed original 16-case matrix ran once with source SHA
`98a33a93c33469309aa97dbc6db3c12286e4e007bcbd44e395a752be49988c2e`
and test SHA `6874941fba9b21d88683e56d882bfa348ce204c054ee87673aa43b0c2a83b029`.
Outcome: 13/16 constructed cases pass; the aggregate test fails. Another24 pure
source/math tests pass. Worker CPU1.130269s, parent1.251620s, sampled peak387,072,000B;
no budget stop under300s wall/240s CPU/2GiB sampled RSS/1MiB output.

The harness printed the full per-case report, but the tool output was truncated
at the requested display budget before root could inspect all three failures.
This was a diagnostic-retention flaw, not permission to discard the negative
result. The supervisor now retains each new V1 synthetic check's bounded output
in a fresh exclusive directory and prints its path/digest; tests additionally
print a compact failure list. No fixture, source mechanism, criterion or resource
limit changes are included in this diagnostic rerun. The initial failure remains
recorded here and in the task's tool output. No physical data was accessed.
