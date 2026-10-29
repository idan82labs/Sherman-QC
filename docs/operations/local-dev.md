# Local development

Run commands from the `FullProject/` Git root, not its workspace wrapper. Read
[current status](../CURRENT_STATUS.md) before research work. Commands here explain
how to perform a permitted task; they do not authorize data access or execution.

## Environments

Use Python 3.11 and the application `.venv/`; reuse the installed environment when
possible. For a new authorized setup only:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The app requirements do not currently provide a complete pinned test bootstrap.
Verify pytest is available before using the smoke command; do not silently install
or upgrade dependencies to make a review pass. Do not use `run.py --install` as
the dependency contract: it uses a partial package list and does not reliably
check installation success.

`requirements.f1-structured-research.txt` belongs in `.venv-f1-structured/`, never
the app `.venv/`: the OR-Tools/protobuf constraints are intentionally isolated.
Do not install that stack into the runtime environment.

## Source-only validation

For documentation/config edits: check local links, config syntax and the scoped
diff. For code changes: inspect imports, fixtures and conftest before choosing
tests. Server/route imports can initialize persistent storage and load research
status; a broad pytest invocation is not a source-only check.

This inspected mapper/contract smoke test constructs two in-memory fixtures:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider backend/tests/test_ui_contracts.py -q
```

It disables project pytest configuration/conftest, plugin autoload, pytest cache
and bytecode writes. It does not test runtime behavior or production readiness.

With existing frontend dependencies, static checks are:

```bash
(cd frontend/react && ./node_modules/.bin/tsc --noEmit)
npm --prefix frontend/react run lint
```

Recheck selected tests/scripts when their source changes; this is not a permanent
blanket safety certification.

## Runtime development and build

Use these only for a task that includes setup, build or runtime work:

```bash
OMP_NUM_THREADS=1 .venv/bin/python run.py --port 9000
```

In a separate terminal at the same Git root:

```bash
npm --prefix frontend/react ci
npm --prefix frontend/react run dev
```

The Vite configuration uses frontend port 3000 and proxies backend requests to
9000. `run.py` otherwise defaults to 8080, so specify `--port 9000`.

```bash
npm --prefix frontend/react run build
```

Build writes generated artifacts. `audit:manual-assistant-ux` is a live UI/API
exercise that logs in, sends chat requests and writes screenshots; it is not a
static lint check. Corpus runners, upload tests, readiness routes and geometry
inspections need their own permitted, isolated inputs and resource scope.

## Resource discipline

One heavy inspection/experiment at a time. Preserve a process-tree RSS budget
below 16 GiB and any stricter frozen experiment limit. Set `OMP_NUM_THREADS=1` for
macOS Open3D work. Host hardware may change; available RAM does not change the
scientific budget or permit simultaneous heavy runs. Do not kill unrelated jobs
or take another project's port to launch this application.
