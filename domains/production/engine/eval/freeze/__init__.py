# Package ``domains.production.engine.eval.freeze`` = E4a's freeze ledger + E4b's pre-A1 freeze tools.
#
# Why this file exists (E4b, 2026-10-17): E4a's hash-chained freeze ledger is the module
# ``domains/production/engine/eval/freeze.py``; the orchestrator gave E4b the subpackage
# ``domains/production/engine/eval/freeze/**``.  A directory with an ``__init__.py`` takes
# precedence over a same-named ``.py`` module, so a plain ``__init__`` here would hide E4a's
# ledger from every importer (runner, CLI, tests).  Instead this file *becomes* E4a's ledger:
#
# - it executes ``eval/freeze.py`` unchanged, in this module's own namespace, after pointing
#   ``__spec__``/``__file__``/``__package__`` at that file (so its relative imports resolve
#   against ``domains.production.engine.eval`` exactly as before, without import warnings);
# - it keeps ``__path__`` = this directory, so E4b's submodules
#   (``domains.production.engine.eval.freeze.gold``, ``.cases``, ``.recipes``, ``.legacy``,
#   ``.physical``, ``.g3_material``, ``.ledger``, ``.prefreeze``) import normally.
#
# Every E4a name (``append_entry``, ``check_against_ledger``, ``ALL_THRESHOLDS`` ...) is a
# global of this module, so ``monkeypatch.setattr(freeze, ...)`` and ``importlib.reload``
# behave exactly as with the bare module; ``__doc__`` is E4a's docstring.  E4b's own
# documentation is the docstring of :mod:`.prefreeze`.  Nothing here may change E4a
# behaviour; ``tests/test_prod_engine_freeze_shim.py`` checks that.
import importlib.util as _ilu
from pathlib import Path as _Path

_E4A_LEDGER_PATH = _Path(__file__).resolve().parent.parent / "freeze.py"
_E4A_LEDGER_SPEC = _ilu.spec_from_file_location(__name__, _E4A_LEDGER_PATH)
if _E4A_LEDGER_SPEC is None or _E4A_LEDGER_SPEC.loader is None:  # pragma: no cover - the file is part of the repo
    raise ImportError(f"E4a freeze ledger module not found at {_E4A_LEDGER_PATH}")
__path__ = [str(_Path(__file__).resolve().parent)]
__spec__ = _E4A_LEDGER_SPEC
__loader__ = _E4A_LEDGER_SPEC.loader
__file__ = str(_E4A_LEDGER_PATH)
__package__ = _E4A_LEDGER_SPEC.parent
__cached__ = _E4A_LEDGER_SPEC.cached
del _ilu, _Path
exec(compile(_E4A_LEDGER_PATH.read_bytes(), str(_E4A_LEDGER_PATH), "exec"), globals())  # noqa: S102 - E4a's own source
