'''Public compile entry point for the dialect-based codegen.

`compile_pipeline(ep, target='cuda')` emits a complete C++ JIT batch
file. Two paths:

  - **Dialect path**: when `_supported_pipeline(ep.pipeline)` is true,
    routes the inner kernel body through the IIR-sorted-array dialect
    + target.cuda emit, wrapped by the dialect's own envelope helpers
    (file prelude, banner, functor struct, view declarations, footer).

  - **Legacy fallback**: shapes the dialect doesn't yet handle
    (e.g., dedup-hash, multi-view, BG, tiled-cartesian when those
    features start appearing in fixtures) fall through to the
    legacy `gen_jit_file_content_from_execute_pipeline`.

The byte-equivalence harness in tests/test_byte_equivalence_jit.py
compares both paths fixture-by-fixture and fails if they diverge.

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
  '''Compile an MIR ExecutePipeline to target C++ source.

  Routes through the dialect framework when the pipeline shape is
  supported, falls back to the legacy emitter otherwise. The two
  paths produce byte-equivalent output (modulo `_cpp_norm`
  normalization) on every fixture in the byte-equivalence harness.
  '''
  if target != 'cuda':
    raise ValueError(f'compile_pipeline: unsupported target {target!r}')

  # Delayed imports: keep `compile_pipeline` cheap to import for
  # tests that don't actually invoke it.
  from srdatalog.codegen.jit.file import gen_jit_file_content_from_execute_pipeline
  from srdatalog.dialects.relation.sorted_array.lowerings import (
    _supported_pipeline,
  )

  if not _supported_pipeline(list(ep.pipeline)):
    # Legacy fallback for shapes the dialect doesn't yet cover.
    return gen_jit_file_content_from_execute_pipeline(ep)

  return _compile_full_file_via_dialect(ep)


def _compile_full_file_via_dialect(ep: m.ExecutePipeline) -> str:
  '''Emit the complete batch file using the dialect throughout —
  envelope helpers from `dialects/target/cuda/envelope.py`, body
  from the sorted_array → cuda lowering. Byte-equivalent (modulo
  `_cpp_norm`) to `gen_jit_file_content_from_execute_pipeline`.
  '''
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
