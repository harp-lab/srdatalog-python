'''Public compile entry point for the dialect-based codegen.

`compile_pipeline(ep, target='cuda')` emits a complete C++ JIT batch
file by routing the pipeline through the IIR-sorted-array dialect +
target.cuda emit, wrapped in the dialect's envelope helpers (file
prelude, banner, functor struct, view declarations, footer).

`compile_kernel_body(ep, ...)` is the lower-level entry: emits just
the operator() body (view_decls + dialect-emitted kernel logic),
parameterized by phase (count vs materialize) and output-var bindings.
The runner emit (Phase N3) calls into this for each kernel it wraps.

Pipeline shapes the dialect doesn't (yet) handle raise loudly via
`lower_scan_pipeline` — `_supported_pipeline()` is the authoritative
scope statement. Adding coverage for a new shape means adding a
lowering rule, not a fallback.

The byte-equivalence harnesses:
  - tests/test_byte_equivalence_jit.py — materialize-phase kernel
    functor against the upstream Nim goldens.
  - tests/test_count_phase_byte_equivalence.py — count-phase body
    against the legacy `jit_pipeline` count emit (the only
    spec for count-phase shape, since the runner files contain
    count bodies but no isolated count-only goldens exist).

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

  from srdatalog.dialects.target.cuda.envelope import (
    assign_handle_positions,
    emit_full_file,
  )

  pipeline = list(ep.pipeline)
  assign_handle_positions(pipeline)

  body = compile_kernel_body(ep, is_counting=False)
  return emit_full_file(ep, body)


def compile_kernel_body(
  ep: m.ExecutePipeline,
  *,
  is_counting: bool,
  output_var_name: str = 'output',
  output_vars: dict[str, str] | None = None,
) -> str:
  '''Emit the operator() body for one kernel — view_decls followed by
  the dialect-emitted kernel logic. Caller is responsible for the
  envelope (file prelude, kernel signature, OutputContext setup).

  Parameters mirror the legacy `_make_kernel_ctx` knobs the runner
  emit (complete_runner.py) twiddles per kernel:

    is_counting: True selects count-phase emit (`emit_direct()` with
      no args, AddCount-style increments).

    output_var_name: name of the OutputContext variable used by the
      single-output InsertInto path (legacy default 'output';
      runner uses 'output_ctx' in count phase, 'output_ctx_0' in
      materialize).

    output_vars: per-relation output-var override map. Multi-head
      rules use this so each InsertInto resolves to its own dest's
      OutputContext. Pass `{rel_name: '__skip_counting__'}` to
      suppress count-phase emission for secondary outputs.
  '''
  from srdatalog.dialects.relation.sorted_array.lowerings import (
    LoweringCtx,
    lower_scan_pipeline,
  )
  from srdatalog.dialects.target.cuda.emit import EmitCtx, emit
  from srdatalog.dialects.target.cuda.envelope import (
    assign_handle_positions,
    collect_unique_view_specs,
    emit_view_declarations,
  )

  pipeline = list(ep.pipeline)
  assign_handle_positions(pipeline)

  view_specs = collect_unique_view_specs(pipeline)
  view_decls, view_vars = emit_view_declarations(view_specs, pipeline)

  lower_ctx = LoweringCtx(
    view_var_names={k: v for k, v in view_vars.items() if k.isdigit()},
    is_counting=is_counting,
    output_var=output_var_name,
    output_var_overrides=dict(output_vars) if output_vars else {},
  )
  iir = lower_scan_pipeline(pipeline, lower_ctx)
  emit_ctx = EmitCtx(indent_level=4)
  return view_decls + emit(iir, emit_ctx)


__all__ = ['Target', 'compile_kernel_body', 'compile_pipeline']
