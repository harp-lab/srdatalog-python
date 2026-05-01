'''Public compile entry point for the dialect-based codegen.

`compile_pipeline(ep, target='cuda')` emits a complete C++ JIT batch
file. Two paths:

  - **Dialect path**: when `_supported_pipeline(ep.pipeline)` is true,
    routes the inner kernel body through the IIR-sorted-array dialect
    + target.cuda emit. Wraps the result in the standard envelope
    (file prelude, banner, functor struct, view declarations,
    footer) — that envelope is still produced by the legacy helpers
    because it has no feature-flag-conditional logic worth dialecting.

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
  '''Emit the complete batch file using the dialect for the inner
  body and the legacy helpers for the envelope.

  Mirrors `gen_jit_file_content_from_execute_pipeline` →
  `gen_jit_file_content` → `jit_full_kernel` byte-for-byte:

    JIT_COMMON_INCLUDES                         (legacy fixed string)
    + banner(rule_name, num_handles)            (legacy logic)
    + jit_functor_start(rule_name, ...)         (legacy fixed-shape opener)
    + [DedupTable struct]                       (legacy, when dedup_hash)
    + view_declarations + dialect-emitted body  (dialect)
    + jit_functor_end()                         (legacy fixed-shape closer)
    + "\\n"
    + JIT_FILE_FOOTER                           (legacy fixed string)
  '''
  from srdatalog.codegen.jit.context import new_code_gen_context
  from srdatalog.codegen.jit.emit_helpers import assign_handle_positions
  from srdatalog.codegen.jit.file import JIT_COMMON_INCLUDES, JIT_FILE_FOOTER
  from srdatalog.codegen.jit.kernel_functor import (
    _first_dest_arity,
    count_handles_in_pipeline,
    gen_dedup_table_struct,
    jit_functor_end,
    jit_functor_start,
  )
  from srdatalog.codegen.jit.view_management import (
    collect_unique_view_specs,
    jit_emit_view_declarations,
  )
  from srdatalog.dialects.relation.sorted_array.lowerings import (
    LoweringCtx,
    lower_scan_pipeline,
  )
  from srdatalog.dialects.target.cuda.emit import EmitCtx, emit

  pipeline = list(ep.pipeline)
  assign_handle_positions(pipeline)

  # Set up the legacy ctx for envelope/view-decl emission. The
  # dialect path doesn't share this ctx with anything stateful — it
  # only reads view_var_names + a few scalar flags.
  legacy_ctx = new_code_gen_context()
  legacy_ctx.indent = 4
  legacy_ctx.dedup_hash_enabled = ep.dedup_hash
  if legacy_ctx.dedup_hash_enabled:
    for op in pipeline:
      if isinstance(op, m.InsertInto):
        legacy_ctx.dedup_hash_vars = list(op.vars)
        break

  # File prelude (fixed) + banner (computed from pipeline).
  num_handles = count_handles_in_pipeline(pipeline)
  banner = (
    '// =============================================================\n'
    f'// JIT-Generated Kernel Functor: {ep.rule_name}\n'
    f'// Handles: {num_handles}\n'
    '// =============================================================\n\n'
  )
  header = banner + jit_functor_start(
    ep.rule_name,
    legacy_ctx.scalar_mode,
    dedup_hash=legacy_ctx.dedup_hash_enabled,
  )

  # Optional DedupTable struct injection (between constexpr decls
  # and operator()), matching legacy line-for-line. Only fires for
  # dedup_hash_enabled fixtures (none of the M1-M7 corpus uses it,
  # but the wiring is here for when one does).
  if legacy_ctx.dedup_hash_enabled:
    arity = _first_dest_arity(pipeline)
    marker = 'static constexpr int kGroupSize = '
    marker_pos = header.find(marker)
    if marker_pos != -1:
      newline_after = header.find('\n\n', marker_pos)
      if newline_after != -1:
        insert_at = newline_after + 2
        header = (
          header[:insert_at]
          + gen_dedup_table_struct(arity)
          + header[insert_at:]
        )

  # Inner body: view declarations (legacy emitter; populates
  # legacy_ctx.view_vars) + dialect-emitted body.
  view_specs = collect_unique_view_specs(pipeline)
  view_decls = jit_emit_view_declarations(view_specs, pipeline, [], legacy_ctx)

  lower_ctx = LoweringCtx(
    view_var_names={
      k: v for k, v in legacy_ctx.view_vars.items() if k.isdigit()
    },
    is_counting=legacy_ctx.is_counting,
    inside_cartesian=legacy_ctx.inside_cartesian,
    output_var=legacy_ctx.output_var_name,
    tile_var=legacy_ctx.tile_var,
    debug=legacy_ctx.debug,
    output_var_overrides=dict(legacy_ctx.output_vars),
  )
  iir = lower_scan_pipeline(pipeline, lower_ctx)
  emit_ctx = EmitCtx(indent_level=legacy_ctx.indent, tile_var=legacy_ctx.tile_var)
  body = view_decls + emit(iir, emit_ctx)

  full_kernel = header + body + jit_functor_end()
  return JIT_COMMON_INCLUDES + full_kernel + '\n' + JIT_FILE_FOOTER


__all__ = ['Target', 'compile_pipeline']
