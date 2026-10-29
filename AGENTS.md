# Sherman QC agent instructions

## Start with the right context

- This directory is the Git root. Its parent is a workspace wrapper, not another
  copy of the runtime. Use repo-relative paths from here.
- Read [current status](docs/CURRENT_STATUS.md) before research/generalization work.
  Use [repository map](docs/REPO_MAP.md) to select the relevant source seam; read
  the [Astra workflow](docs/operations/astra-workflow.md) for agent setup.
- Current source determines what runs. Architecture diagrams may describe the
  strangler-migration destination. Historical handoffs are evidence, not active
  resume commands. Follow the latest explicit user scope and valid protocol gate.

## Work efficiently and preserve ownership

- For change requests, implement and verify the scoped result. For review or
  diagnosis, inspect and report; do not silently expand into a runtime change.
- Preserve user edits and untracked research work. No reset, cleanup, mass
  formatting, dependency upgrade, commit, or unrelated refactor by default.
- Read narrowly with `rg` over named source/test/doc paths. Do not recursively
  ingest the whole repository or every experiment. Never read credentials or
  `.env` contents to understand the architecture.
- Delegate independent bounded reviews or implementation seams when useful.
  Use one editor per file; the integrating agent checks all outcomes and diffs.
  Do not parallelize heavy geometry jobs.
- State the intended artifact and acceptance check up front. Keep updates short
  and evidence-based, roughly once per minute during long work. Finish with what
  changed, checks actually run, and the next decision/blocker.
- Preserve a concise dated checkpoint when stopping: source state, verified
  evidence, authorized next action, and stop conditions. A final handoff suffices
  for read-only work; edit a checkpoint file only within an authorized change.
  Do not imply a test passed because its source exists or an older revision passed.

## Scientific custody and evidence

- Source-only engineering is not permission to inspect physical inputs. Unless
  the current explicit scope and protocol authorize it, do not open, stat, hash,
  search, sample, or execute against raw data, `data/`, `output/`, `outputs/`,
  pools/manifests, sealed truth, HOLDOUT10, reserved incoming-email/EHO parts, or
  control/authorization/one-shot artifacts. Do not retrieve email attachments.
  Released narrative reports and explicitly released aggregate dispositions may
  be read within scope; a directory name alone does not make an artifact public.
- A terminal one-shot stays terminal. Do not retry it, raise its cap, reduce its
  fixture, or issue a successor's real-input run from an old handoff or this file.
  New authority and any required independent gate must precede physical access.
- Keep engineering capability and physical generalization in separate ledgers.
  Synthetic tests, deterministic replay, source-only diagnostics, and consumed
  development scores are not unseen-family accuracy or promotion evidence.
- Freeze hypothesis, baseline, family/part split, evaluator, budgets, failure
  handling, and decision rule before a permitted confirmatory evaluation. Runtime
  being label-free does not make threshold selection or calibration label-free.
- Do not tune on held-outs, move consumed cases back to an untouched split, use
  filenames/count labels as inference features, or change frozen scoring rules.
- Operator assistance is not autonomous acceptance/rejection. Preserve unknown,
  insufficient coverage, invalid registration, and research-only status; do not
  turn an unobserved bend into a missing bend or a failed inspection into a defect.

## Runtime and validation boundaries

- Backend: Python 3.11 in `.venv/`; frontend: `frontend/react/`. See
  [local development](docs/operations/local-dev.md) for command tiers.
- Keep `requirements.f1-structured-research.txt` and its OR-Tools/protobuf stack
  in `.venv-f1-structured/`, never the application `.venv`.
- One heavy geometry/inspection experiment at a time. Keep process-tree RSS below
  16 GiB and obey any stricter frozen experiment limit. Use `OMP_NUM_THREADS=1`
  for macOS Open3D work. Available host RAM does not relax the budget.
- Inspect test imports/fixtures before running them under source-only custody:
  importing legacy server code can initialize DB/directories and load status.
  Prefer isolated synthetic contract checks. No blanket pytest or corpus runner.
- Verification is proportional: docs/config -> links, syntax, diff; source change
  -> focused tests; UI -> type/lint/build and scoped visual checks as applicable.
  Broad or live tests require relevant scope and isolated inputs. Do not rerun
  unchanged successful suites repeatedly without a reason.
- Keep frontend bend-domain assembly in the feature layer, not the shared viewer.
  Keep runtime measurements deterministic and separate from AI explanations.
- Optimizing this repository for Astra means agent context/workflow, not changing
  the product's AI provider, credentials, permissions, or scientific mechanism.
