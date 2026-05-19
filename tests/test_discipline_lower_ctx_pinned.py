'''D10 — `LowerCtx` is pinned at 5 fields.

Per `docs/code_discipline.md` D10 and
`docs/phase_zero_prerequisites.md` §3.2: adding a 6th field to
`LowerCtx` requires a doc amendment + owner sign-off, NOT a silent
schema bump.

This is the canonical "god-object guardrail" for the redesign — the
old `LoweringCtx` had ~25 fields and that monolith is what the
redesign exists to dismantle. The pin lives in CI so the new ctx
stays small.

PR-1 (per `docs/phase_decomposition_redesign.md` § 3.2.1) adds a
sibling test that the dialect-level `LoweringCtx` exposes a
`render_ctx` slot of `CudaRenderCtx`. The dialect ctx is the
transition surface every helper sees today; the render_ctx slot is
the canonical home for the 11 CUDA-render-specific identifier
fields. Post-redesign waves migrate every read site to access via
`ctx.render_ctx.<field>`; PR-1 establishes the shape.
'''

from __future__ import annotations

import dataclasses

from srdatalog.ir.codegen.cuda.lower_ctx import CudaRenderCtx
from srdatalog.ir.core.lower_ctx import LowerCtx


def test_lower_ctx_field_count_pinned_at_five() -> None:
  fields = dataclasses.fields(LowerCtx)
  assert len(fields) == 5, (
    f'LowerCtx has {len(fields)} fields, must be exactly 5. '
    f'Per docs/code_discipline.md D10 + docs/phase_zero_prerequisites.md §3.2: '
    f'adding a 6th requires a doc amendment + owner sign-off. Existing fields: '
    f'{[f.name for f in fields]}'
  )


def test_dialect_lowering_ctx_threads_render_ctx() -> None:
  '''The dialect-level `LoweringCtx` (in
  `srdatalog.ir.dialects.relation.sorted_array.lowerings`) carries
  the CUDA-render-specific identifier fields via the `render_ctx`
  slot of type `CudaRenderCtx`. PR-1 (per
  `docs/phase_decomposition_redesign.md` § 3.2.1): this shape is
  the foundation every subsequent target-abstraction PR builds on —
  the 11 CUDA identifier fields belong on the target-private
  `CudaRenderCtx`, not on a target-agnostic ctx.
  '''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  ctx = LoweringCtx()
  assert isinstance(ctx.render_ctx, CudaRenderCtx), (
    f'LoweringCtx.render_ctx must be a CudaRenderCtx; got {type(ctx.render_ctx).__name__!r}'
  )


def test_cuda_render_ctx_field_count_pinned_at_eleven() -> None:
  '''PR-1 pins the CUDA-render field count. New CUDA-identifier-
  bearing state added to `CudaRenderCtx` MUST come with a doc
  amendment (the post-redesign goal is per-op render contexts, not
  one mega-ctx). Per `docs/phase_decomposition_redesign.md` § 3.2.1.
  '''
  fields = dataclasses.fields(CudaRenderCtx)
  assert len(fields) == 11, (
    f'CudaRenderCtx has {len(fields)} fields, must be exactly 11. '
    f'Adding a 12th requires a doc amendment per D10. Existing '
    f'fields: {[f.name for f in fields]}'
  )
