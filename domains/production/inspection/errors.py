"""Error type and codes of the C06 postprocess / render step.

Exit-code mapping (ENGINE_IO §5, runner step 4):

- ``PostprocessError`` with ``exit_code`` 2: invalid input. The CLI writes
  ``result/postprocess_error.json {code, message}`` and an ``ERROR`` record.
- any other exception: exit 3 with code ``INTERNAL_ERROR`` (same file).

The runner maps every non-zero exit to ``POSTPROCESS_ERROR`` carrying the
code and message, so the codes below are diagnostic detail, never a part
verdict.
"""
from __future__ import annotations

__all__ = ["POSTPROCESS_ERROR_CODES", "PostprocessError"]

POSTPROCESS_ERROR_CODES: frozenset[str] = frozenset(
    {
        "INPUT_MISSING",  # a required input file / directory is absent
        "USAGE_ERROR",  # bad command line
        "RECIPE_INVALID",  # recipe.json does not validate (hashes recomputed)
        "COMPILED_PART_INVALID",  # compiled_part.json does not validate
        "ENGINE_OUTPUT_INVALID",  # result.json invalid or inconsistent with compiled/recipe
        "RESULT_INVALID",  # the rendered result failed its own validation (self-check)
        "RESULT_ALTERED",  # self-check: the rendered result changed the engine verdict
        "INTERNAL_ERROR",  # anything unexpected (exit 3)
    }
)


class PostprocessError(Exception):
    """An input problem the step reports with a code (exit 2 by default)."""

    def __init__(self, code: str, message: str, *, exit_code: int = 2) -> None:
        if code not in POSTPROCESS_ERROR_CODES:
            raise ValueError(f"unknown postprocess error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.exit_code = exit_code
