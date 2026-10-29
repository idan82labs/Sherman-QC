# Sherman QC compatibility entrypoint

Read [AGENTS.md](AGENTS.md) first. It is the canonical shared instruction source
for repository navigation, edit ownership, research custody and verification.
Do not maintain a second copy of those rules here.

Then read [current status](docs/CURRENT_STATUS.md) for research work and use the
[repository map](docs/REPO_MAP.md) to locate the actual runtime seam. Setup and
command tiers are in [local development](docs/operations/local-dev.md).

The old machine-specific limits and simplified algorithm map were removed on
2026-09-07. The operational limit remains one heavy experiment at a time and less
than 16 GiB process-tree RSS, or any stricter frozen protocol limit.

The `.claude/commands/` files are compatibility commands, not research execution
authority. `/test-qc` contains historical real-data recipes: its dated warning and
the current custody gate must be applied before using any of them. `/critique`
must remain a bounded source review unless the user requests implementation.
