# ADR-004: Research and Runtime Separation

## Decision
Research corpus/evaluation/model code remains inside the repo, but its entrypoints and artifacts must stay explicitly separated from operator/runtime flows.

## Consequence
Runtime may consume research models through explicit service boundaries, but evaluation outputs and experimental artifacts are not treated as production source.
