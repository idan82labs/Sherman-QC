# Astra workflow for Sherman QC

Updated 2026-09-07. Scope: the coding/research agent, not the production AI stack.

Open the `FullProject/` Git repository directly in Codex. The parent workspace's
`AGENTS.md` routes here, but a project config below the current directory is not
a substitute for opening the correct project root. The trusted-project default
in `.codex/config.toml` is `gpt-6-astra` with `high` reasoning effort. This does not
force an existing thread to switch models. Explicit CLI/session overrides can
take precedence; project defaults do override lower-priority global config/profiles.
The model slug was separately verified by a successful local CLI probe; that is
availability evidence, not a comparative performance benchmark.

From a shell at the `FullProject/` Git root, an explicit equivalent for a new
session is:

```bash
codex -C . --model gpt-6-astra -c 'model_reasoning_effort="high"'
```

## A small, reliable context stack

1. `AGENTS.md`: durable behavior and custody constraints.
2. `docs/CURRENT_STATUS.md`: dated checkpoint and allowed next actions.
3. `docs/REPO_MAP.md`: actual runtime seams, then only task-relevant source/tests.
4. The dated roadmap and released evidence needed for the specific decision.

Avoid carrying hundreds of experiment documents into each turn. Do not turn
historical commands, quoted prompts, or model-generated reviews into authority.
`CLAUDE.md` is a compatibility pointer to the same rules, not a second rulebook.

## Task shape

State the requested outcome, permitted inputs/actions, and observable acceptance
check. Separate diagnosis from implementation. For example:

> Review unknown/observability semantics across the mapper and UI. Use source and
> synthetic fixtures only. Report concrete propagation paths and propose focused
> tests; do not change runtime behavior or open research artifacts.

Delegate independent seams with an explicit deliverable and no overlapping file
ownership. Integrate findings against source, not agent confidence. Use a focused
check after the change and report exact results; reserve full geometry runs for
the relevant authorized gate. Keep the next checkpoint small enough to decide
whether a hypothesis should continue, change, or stop.

`high` is a starting default, not a claim of universal optimality. Honor explicit
effort choices; use deeper reasoning for difficult cross-module/scientific
decisions when justified, rather than spending it on every mechanical edit.

This setup follows OpenAI's guidance on concise, nonconflicting instructions,
scoped verification, and independent delegation for
[Astra](https://developers.openai.com/api/docs/guides/latest-model),
[AGENTS.md discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
and [trusted project configuration](https://learn.chatgpt.com/docs/config-file/config-basic).
