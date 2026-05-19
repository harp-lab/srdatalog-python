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

import srdatalog.ir.mir.types as m

Target = Literal['cuda']


def compile_pipeline(ep: m.ExecutePipeline, *, target: Target = 'cuda') -> str:
  '''Compile an MIR ExecutePipeline to target C++ source via the dialect.

  Raises ValueError on unsupported targets. Raises (via
  `lower_scan_pipeline`) on pipeline shapes the dialect doesn't
  cover — there is no legacy fallback.
  '''
  if target != 'cuda':
    raise ValueError(f'compile_pipeline: unsupported target {target!r}')

  from srdatalog.ir.codegen.cuda.envelope import (
    assign_handle_positions_in_ep,
    emit_full_file,
  )

  # Rebuild EP with handle_start filled in on both pipeline ops AND the
  # parallel source_specs references. `emit_full_file` reads handles
  # off `ep.pipeline` via `count_handles`; `compile_kernel_body` walks
  # `ep.pipeline` to assign view slots. Both must see the same handles.
  ep = assign_handle_positions_in_ep(ep)

  # Standalone jit_batch shape uses handle_idx-based view slots; runner
  # contexts (compile_runner / compile_kernel_body) default to positional.
  body = compile_kernel_body(ep, is_counting=False, slot_mode='handle_idx')
  return emit_full_file(ep, body)


def compile_runner(
  ep: m.ExecutePipeline,
  db_type_name: str,
  rel_index_types: dict[str, str] | None = None,
) -> str:
  '''Compile an ExecutePipeline to its full per-rule runner — the
  `JitRunner_<rule>` struct + kernel definitions + out-of-line phase
  methods + execute(). Production output: this is what
  `jit_runner.<rule>.cpp` golden files capture.

  The dialect's `codegen.cuda.runner` module owns the runner emission
  surface. Most pieces (phase methods, execute, BG variants, fused
  kernel) currently delegate to legacy helpers in
  `ir.codegen.cuda.complete_runner`; later milestones (N2/N4/N5/N6/N8)
  collapse them into native dialect emission.

  Kernel *bodies* (count + materialize) already route through
  `compile_kernel_body` when `_dialect_safe_kernel` holds — see the
  swap inside `complete_runner._gen_kernel_count` /
  `_gen_kernel_materialize`.

  The byte-equivalence gate (`tests/test_runner_byte_equivalence.py`)
  anchors this entry point to the upstream goldens throughout the
  migration.
  '''
  from srdatalog.ir.codegen.cuda.runner import emit_runner_full

  return emit_runner_full(ep, db_type_name, rel_index_types=rel_index_types)


def compile_kernel_body(
  ep: m.ExecutePipeline,
  *,
  is_counting: bool,
  output_var_name: str = 'output',
  output_vars: dict[str, str] | None = None,
  slot_mode: str = 'positional',
  rel_index_types: dict[str, str] | None = None,
  tiled_cartesian: bool = False,
  bg_enabled: bool = False,
  target: str = 'cuda',
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

    slot_mode: 'positional' (default, matches `jit_runner.<rule>.cpp`
      production goldens) or 'handle_idx' (matches the standalone
      `jit_batch.<rule>.cpp` test fixtures emitted via
      `compile_pipeline`). See `emit_view_declarations` docstring.

    rel_index_types: per-relation custom index type (e.g.,
      `Device2LevelIndex`). Used to compute per-spec view_counts via
      `relation.d2l` (and any future index dialect) so positional
      slots advance by 2 per FULL_VER D2L source — matching legacy
      `compute_view_slot_offsets`. Pass {} or None for plain DSAI.

  Implementation: F5.3 routes this entry point through
  `Compiler.run(KernelCtx(...), pipeline=DEFAULT_KERNEL_PIPELINE)` —
  the imperative block that previously lived here is now five
  `ProgramPass` shims in `srdatalog.ir.default_pipelines`. Output is
  byte-equivalent (the shim composition mirrors the old block 1-to-1;
  `AssignHandlesShim` is idempotent so re-running it over an EP whose
  handles are already assigned — the `compile_pipeline` path — leaves
  the pipeline byte-identical).
  '''
  from srdatalog.ir.core.dialect import Compiler
  from srdatalog.ir.default_pipelines import DEFAULT_KERNEL_PIPELINE, KernelCtx

  state: KernelCtx = Compiler().run(
    KernelCtx(
      ep=ep,
      is_counting=is_counting,
      output_var_name=output_var_name,
      output_vars=output_vars,
      slot_mode=slot_mode,
      rel_index_types=rel_index_types,
      tiled_cartesian=tiled_cartesian,
      bg_enabled=bg_enabled,
      target=target,
    ),
    pipeline=DEFAULT_KERNEL_PIPELINE,
    target=target,
  )
  body_text = state.body_text
  assert body_text is not None, 'compile_kernel_body: CudaRenderShim left body_text unset'
  return body_text


__all__ = ['Target', 'compile_kernel_body', 'compile_pipeline', 'compile_runner']
