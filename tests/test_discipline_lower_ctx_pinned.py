'''D10 — `LowerCtx` is pinned at 5 fields.

Per `docs/code_discipline.md` D10 and
`docs/phase_zero_prerequisites.md` §3.2: adding a 6th field to
`LowerCtx` requires a doc amendment + owner sign-off, NOT a silent
schema bump.

This is the canonical "god-object guardrail" for the redesign — the
old `LoweringCtx` had ~25 fields and that monolith is what the
redesign exists to dismantle. The pin lives in CI so the new ctx
stays small.
'''

from __future__ import annotations

import dataclasses

from srdatalog.ir.core.lower_ctx import LowerCtx


def test_lower_ctx_field_count_pinned_at_five() -> None:
  fields = dataclasses.fields(LowerCtx)
  assert len(fields) == 5, (
    f'LowerCtx has {len(fields)} fields, must be exactly 5. '
    f'Per docs/code_discipline.md D10 + docs/phase_zero_prerequisites.md §3.2: '
    f'adding a 6th requires a doc amendment + owner sign-off. Existing fields: '
    f'{[f.name for f in fields]}'
  )
