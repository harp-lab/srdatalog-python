'''Public compile entry point for the dialect-based codegen.

`compile_pipeline(ep, target='cuda')` emits a complete C++ JIT batch
file by routing the pipeline through the IIR-sorted-array dialect +
target.cuda emit, wrapped in the dialect's envelope helpers (file
prelude, banner, functor struct, view declarations, footer).

Pipeline shapes the dialect doesn't (yet) handle raise loudly via
`lower_scan_pipeline` — `_supported_pipeline()` is the authoritative
scope statement. Adding coverage for a new shape means adding a
lowering rule, not a fallback.

The byte-equivalence harness in `tests/test_byte_equivalence_jit.py`
diffs the dialect output against the upstream Nim goldens.

See:
  - docs/stage2_emitter_audit.md — the per-milestone migration plan.
  - docs/ir_lowering_semantics.md — the formal lowering rules.
  - docs/design_principles.md — discipline rules for the rewrite.
'''

from __future__ import annotations

from typing import Literal

import srdatalog.mir.types as m

Target = Literal['cuda']


def compile_pipeline(ep: m.ExecutePipeline, *, target: Target = 'cuda') -> str:
  '''Compile an MIR ExecutePipeline to target C++ source via the dialect.

  Raises ValueError on unsupported targets. Raises (via
  `lower_scan_pipeline`) on pipeline shapes the dialect doesn't
  cover — there is no legacy fallback.
  '''
  if target != 'cuda':
    raise ValueError(f'compile_pipeline: unsupported target {target!r}')

  # Delayed imports: keep `compile_pipeline` cheap to import for
  # tests that don't actually invoke it.
  from srdatalog.dialects.relation.sorted_array.lowerings import (
    LoweringCtx,
    lower_scan_pipeline,
  )
  from srdatalog.dialects.target.cuda.emit import EmitCtx, emit
  from srdatalog.dialects.target.cuda.envelope import (
    assign_handle_positions,
    collect_unique_view_specs,
    emit_full_file,
    emit_view_declarations,
  )

  pipeline = list(ep.pipeline)
  assign_handle_positions(pipeline)

  # Inner body: view declarations + dialect-emitted kernel logic.
  view_specs = collect_unique_view_specs(pipeline)
  view_decls, view_vars = emit_view_declarations(view_specs, pipeline)

  lower_ctx = LoweringCtx(
    view_var_names={k: v for k, v in view_vars.items() if k.isdigit()},
  )
  iir = lower_scan_pipeline(pipeline, lower_ctx)
  emit_ctx = EmitCtx(indent_level=4)
  body = view_decls + emit(iir, emit_ctx)

  return emit_full_file(ep, body)


__all__ = ['Target', 'compile_pipeline']
