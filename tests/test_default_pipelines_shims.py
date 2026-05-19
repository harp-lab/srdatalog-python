'''F5.1 smoke tests — declarative pipeline shims.

Spec: docs/phase_f5_declarative_pipeline.md.

Each shim runs in isolation against a hand-built minimal state, and
both `DEFAULT_PROGRAM_PIPELINE` + `DEFAULT_KERNEL_PIPELINE` run
end-to-end against a tiny TC-style program. F5.1 is pure addition;
byte-equivalence to the imperative path is anchored separately in
F5.4.
'''

from __future__ import annotations

import srdatalog.ir.mir.types as mir
from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pass
from srdatalog.ir.default_pipelines import (
  DEFAULT_KERNEL_PIPELINE,
  DEFAULT_PROGRAM_PIPELINE,
  AssignHandlesShim,
  CollectViewSpecsShim,
  EmitViewDeclsShim,
  HirPlanningShim,
  HirToMirShim,
  InitialProg,
  KernelCtx,
  LowerKernelBodyShim,
  MirOptShim,
  MirProgramAssembly,
  RenderShim,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _tc_program() -> Program:
  '''Tiny TC-style program: ArcInput -> Edge, Edge -> Path (base + rec).'''
  X, Y, Z = Var('x'), Var('y'), Var('z')
  arc = Relation('ArcInput', 2)
  edge = Relation('Edge', 2)
  path = Relation('Path', 2)
  return Program(
    rules=[
      (edge(X, Y) <= arc(X, Y)).named('EdgeLoad'),
      (path(X, Y) <= edge(X, Y)).named('TCBase'),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named('TCRec'),
    ],
  )


def _find_execute_pipelines(node: object) -> list[mir.ExecutePipeline]:
  '''Walk a MIR plan and collect every ExecutePipeline. Mirrors the
  shape of `_collect_pipelines` in batchfile.py for our test needs.'''
  out: list[mir.ExecutePipeline] = []
  if isinstance(node, mir.ExecutePipeline):
    out.append(node)
  elif isinstance(node, mir.FixpointPlan):
    for inst in node.instructions:
      out.extend(_find_execute_pipelines(inst))
  elif isinstance(node, mir.ParallelGroup):
    for op in node.ops:
      out.extend(_find_execute_pipelines(op))
  return out


def _first_ep(mir_prog: mir.Program) -> mir.ExecutePipeline:
  for step, _is_rec in mir_prog.steps:
    eps = _find_execute_pipelines(step)
    if eps:
      return eps[0]
  raise AssertionError('no ExecutePipeline in mir_prog')


# -----------------------------------------------------------------------------
# Type-erasure check (spec §6: every list entry is a Pass)
# -----------------------------------------------------------------------------


def test_every_program_pipeline_entry_is_a_pass():
  for p in DEFAULT_PROGRAM_PIPELINE:
    assert isinstance(p, Pass), p


def test_every_kernel_pipeline_entry_is_a_pass():
  for p in DEFAULT_KERNEL_PIPELINE:
    assert isinstance(p, Pass), p


def test_program_pipeline_names_unique_and_ordered():
  names = [p.name for p in DEFAULT_PROGRAM_PIPELINE]
  assert names == ['hir_planning', 'hir_to_mir', 'mir_opt', 'mir_program_assembly']
  assert len(set(names)) == len(names)


def test_kernel_pipeline_names_unique_and_ordered():
  names = [p.name for p in DEFAULT_KERNEL_PIPELINE]
  assert names == [
    'assign_handles',
    'collect_view_specs',
    'emit_view_decls',
    'lower_kernel_body',
    'verify_renderability',
    'render',
  ]
  assert len(set(names)) == len(names)


def test_first_shim_in_each_pipeline_consumes_nothing():
  '''Spec note: the FIRST shim must have consumes=() because no
  pseudo-dialects are registered with the Compiler.'''
  assert DEFAULT_PROGRAM_PIPELINE[0].consumes == ()
  assert DEFAULT_KERNEL_PIPELINE[0].consumes == ()


# -----------------------------------------------------------------------------
# Per-shim isolation tests (program pipeline)
# -----------------------------------------------------------------------------


def test_hir_planning_shim_runs_in_isolation():
  shim = HirPlanningShim()
  state = InitialProg(program=_tc_program())
  out = shim.apply(state, None)
  assert out.hir is not None
  assert out.program is state.program  # unchanged


def test_hir_planning_shim_idempotent_when_hir_already_set():
  '''If the caller pre-populated hir, the shim is a no-op (matches
  `compile_to_mir(program, hir=...)` legacy semantics).'''
  from srdatalog.ir.hir import compile_to_hir

  shim = HirPlanningShim()
  hir = compile_to_hir(_tc_program())
  state = InitialProg(program=_tc_program(), hir=hir)
  out = shim.apply(state, None)
  assert out.hir is hir


def test_hir_to_mir_shim_runs_in_isolation():
  from srdatalog.ir.hir import compile_to_hir

  hir = compile_to_hir(_tc_program())
  state = InitialProg(program=_tc_program(), hir=hir)
  out = HirToMirShim().apply(state, None)
  assert out.steps is not None
  assert len(out.steps) == 6
  assert all(isinstance(node, mir.MirNode) for node, _ in out.steps)


def test_mir_opt_shim_preserves_step_count():
  from srdatalog.ir.hir import compile_to_hir
  from srdatalog.ir.hir.lower import lower_hir_to_mir_steps

  hir = compile_to_hir(_tc_program())
  steps = lower_hir_to_mir_steps(hir)
  state = InitialProg(program=_tc_program(), hir=hir, steps=steps)
  out = MirOptShim().apply(state, None)
  assert out.steps is not None
  assert len(out.steps) == len(steps)


def test_mir_program_assembly_shim_returns_mir_program():
  from srdatalog.ir.hir import compile_to_hir
  from srdatalog.ir.hir.lower import lower_hir_to_mir_steps
  from srdatalog.ir.mir.passes import apply_all_mir_passes

  hir = compile_to_hir(_tc_program())
  steps = apply_all_mir_passes(lower_hir_to_mir_steps(hir))
  state = InitialProg(program=_tc_program(), hir=hir, steps=steps)
  out = MirProgramAssembly().apply(state, None)
  assert isinstance(out.mir_program, mir.Program)
  assert len(out.mir_program.steps) == len(steps)


# -----------------------------------------------------------------------------
# Per-shim isolation tests (kernel pipeline)
# -----------------------------------------------------------------------------


def _build_kernel_ctx() -> KernelCtx:
  '''Build a minimal KernelCtx by running the program pipeline first.'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)
  return KernelCtx(ep=ep, is_counting=False)


def test_assign_handles_shim_runs_in_isolation():
  state = _build_kernel_ctx()
  out = AssignHandlesShim().apply(state, None)
  # ep is rebuilt; pipeline list length preserved.
  assert len(out.ep.pipeline) == len(state.ep.pipeline)


def test_collect_view_specs_shim_runs_in_isolation():
  state = AssignHandlesShim().apply(_build_kernel_ctx(), None)
  out = CollectViewSpecsShim().apply(state, None)
  assert out.view_specs is not None
  # TC has at least one Edge view spec.
  assert len(out.view_specs) >= 1


def test_emit_view_decls_shim_populates_maps():
  state = _build_kernel_ctx()
  state = AssignHandlesShim().apply(state, None)
  state = CollectViewSpecsShim().apply(state, None)
  out = EmitViewDeclsShim().apply(state, None)
  assert out.view_decls is not None
  assert isinstance(out.view_decls, str)
  assert out.name_map is not None
  assert out.base_map is not None
  # Every name_map key is a digit string (per the spec's split rule).
  assert all(k.isdigit() for k in out.name_map)


def test_lower_kernel_body_shim_produces_iir():
  state = _build_kernel_ctx()
  state = AssignHandlesShim().apply(state, None)
  state = CollectViewSpecsShim().apply(state, None)
  state = EmitViewDeclsShim().apply(state, None)
  out = LowerKernelBodyShim().apply(state, None)
  assert out.iir is not None


def test_render_shim_returns_string_body():
  state = _build_kernel_ctx()
  state = AssignHandlesShim().apply(state, None)
  state = CollectViewSpecsShim().apply(state, None)
  state = EmitViewDeclsShim().apply(state, None)
  state = LowerKernelBodyShim().apply(state, None)
  out = RenderShim().apply(state, None)
  assert isinstance(out.body_text, str)
  assert len(out.body_text) > 0
  # Should contain the view-decls preamble + emitted body.
  assert 'ViewType' in out.body_text


# -----------------------------------------------------------------------------
# End-to-end pipeline runs via Compiler.run
# -----------------------------------------------------------------------------


def test_default_program_pipeline_end_to_end():
  '''DEFAULT_PROGRAM_PIPELINE runs from a Program to a mir.Program via
  `Compiler.run`. No imperative `compile_to_mir` involved.'''
  compiler = Compiler.with_default_plugins()
  state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert isinstance(state.mir_program, mir.Program)
  # tc -> 3 strata -> 6 steps (FixpointPlan + PostStratumReconstructInternCols each).
  assert len(state.mir_program.steps) == 6


def test_default_kernel_pipeline_end_to_end():
  '''DEFAULT_KERNEL_PIPELINE consumes a single ExecutePipeline and
  emits a non-empty body string via `Compiler.run`.'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  kernel_state = compiler.run(KernelCtx(ep=ep, is_counting=False), pipeline=DEFAULT_KERNEL_PIPELINE)
  assert isinstance(kernel_state.body_text, str)
  assert len(kernel_state.body_text) > 0


def test_kernel_pipeline_count_phase_runs():
  '''Smoke: count-phase emit through the declarative pipeline.'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)
  kc = KernelCtx(ep=ep, is_counting=True, output_var_name='output_ctx')
  out = compiler.run(kc, pipeline=DEFAULT_KERNEL_PIPELINE)
  assert isinstance(out.body_text, str)
  assert len(out.body_text) > 0


# -----------------------------------------------------------------------------
# F5.2: compile_to_mir uses DEFAULT_PROGRAM_PIPELINE under the hood
# -----------------------------------------------------------------------------


def test_compile_to_mir_uses_default_pipeline():
  '''F5.2 cut-over: `compile_to_mir(program)` is now a thin wrapper
  over `Compiler().run(InitialProg(...), pipeline=DEFAULT_PROGRAM_PIPELINE)`.
  The output must match the direct `Compiler.run` invocation byte-for-byte.'''
  from srdatalog.ir.hir import compile_to_mir

  prog = _tc_program()

  # Compare imperative entry point against the declarative pipeline
  # invoked directly (the pipeline IS the source of truth post-F5.2).
  via_entry = compile_to_mir(prog)
  via_pipeline = (
    Compiler()
    .run(
      InitialProg(program=prog),
      pipeline=DEFAULT_PROGRAM_PIPELINE,
    )
    .mir_program
  )

  assert via_entry == via_pipeline


def test_compile_to_mir_apply_mir_passes_false_filters_mir_opt():
  '''F5.2: `apply_mir_passes=False` drops `MirOptShim` from the pipeline.
  The output must match a direct `Compiler.run` over the filtered
  pipeline (i.e. raw HIR -> MIR steps, no optimization).'''
  from srdatalog.ir.hir import compile_to_mir

  prog = _tc_program()
  via_entry = compile_to_mir(prog, apply_mir_passes=False)

  filtered = [p for p in DEFAULT_PROGRAM_PIPELINE if not isinstance(p, MirOptShim)]
  via_pipeline = (
    Compiler()
    .run(
      InitialProg(program=prog),
      pipeline=filtered,
    )
    .mir_program
  )

  assert via_entry == via_pipeline


def test_compile_to_mir_with_precomputed_hir_skips_planning():
  '''F5.2: passing `hir=...` short-circuits `HirPlanningShim` (which is
  idempotent on `state.hir is not None`). Output must match the
  default path.'''
  from srdatalog.ir.hir import compile_to_hir, compile_to_mir

  prog = _tc_program()
  hir = compile_to_hir(prog)
  via_hir = compile_to_mir(prog, hir=hir)
  default = compile_to_mir(prog)
  assert via_hir == default


def test_compile_to_mir_verbose_does_not_break():
  '''F5.2: `verbose=True` is threaded through `InitialProg.verbose`
  into `HirPlanningShim`, which forwards to `compile_to_hir`. Smoke
  test: it doesn't crash and the output is byte-equivalent to the
  non-verbose default (verbose only affects logging side effects).'''
  from srdatalog.ir.hir import compile_to_mir

  prog = _tc_program()
  loud = compile_to_mir(prog, verbose=True)
  quiet = compile_to_mir(prog)
  assert loud == quiet


# -----------------------------------------------------------------------------
# F5.3: compile_kernel_body cut-over
# -----------------------------------------------------------------------------


def test_compile_kernel_body_routes_through_default_pipeline():
  '''F5.3: `compile_kernel_body` is now a thin `Compiler.run` over
  `DEFAULT_KERNEL_PIPELINE`. The returned string must equal the
  direct shim composition for both phases.'''
  from srdatalog.ir.codegen.cuda.api import compile_kernel_body

  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  # Materialize phase.
  via_api = compile_kernel_body(ep, is_counting=False)
  via_pipeline = compiler.run(
    KernelCtx(ep=ep, is_counting=False), pipeline=DEFAULT_KERNEL_PIPELINE
  ).body_text
  assert via_api == via_pipeline

  # Count phase with overridden output var (mirrors runner emit).
  via_api_ct = compile_kernel_body(ep, is_counting=True, output_var_name='output_ctx')
  via_pipeline_ct = compiler.run(
    KernelCtx(ep=ep, is_counting=True, output_var_name='output_ctx'),
    pipeline=DEFAULT_KERNEL_PIPELINE,
  ).body_text
  assert via_api_ct == via_pipeline_ct


def test_compile_kernel_body_threads_all_kwargs():
  '''Sanity check: every kwarg on `compile_kernel_body` survives the
  cut-over. Build a KernelCtx with non-default values for each knob
  the shims read and confirm `compile_kernel_body` reaches the same
  shim output.'''
  from srdatalog.ir.codegen.cuda.api import compile_kernel_body

  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  kwargs = dict(
    is_counting=False,
    output_var_name='output_ctx_0',
    output_vars={'Path': 'output_ctx_0'},
    slot_mode='positional',
    rel_index_types={},
    tiled_cartesian=False,
    bg_enabled=False,
  )
  via_api = compile_kernel_body(ep, **kwargs)
  via_pipeline = compiler.run(
    KernelCtx(ep=ep, **kwargs), pipeline=DEFAULT_KERNEL_PIPELINE
  ).body_text
  assert via_api == via_pipeline


def test_assign_handles_shim_is_idempotent():
  '''Compile_pipeline rebuilds the EP with `assign_handle_positions_in_ep`
  before calling `compile_kernel_body`, which then re-runs the
  AssignHandlesShim. The shim must produce a byte-identical pipeline
  on the second pass, otherwise `compile_pipeline` would shift handle
  indices and break the jit_batch goldens.'''
  from srdatalog.ir.codegen.cuda.envelope import assign_handle_positions_in_ep

  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  ep_once = assign_handle_positions_in_ep(ep)
  state_twice = AssignHandlesShim().apply(KernelCtx(ep=ep_once, is_counting=False), None)
  # Each MIR op's handle_start must be unchanged after re-assignment.
  for before, after in zip(ep_once.pipeline, state_twice.ep.pipeline, strict=True):
    assert getattr(before, 'handle_start', None) == getattr(after, 'handle_start', None)


if __name__ == '__main__':
  test_every_program_pipeline_entry_is_a_pass()
  test_every_kernel_pipeline_entry_is_a_pass()
  test_program_pipeline_names_unique_and_ordered()
  test_kernel_pipeline_names_unique_and_ordered()
  test_first_shim_in_each_pipeline_consumes_nothing()
  test_hir_planning_shim_runs_in_isolation()
  test_hir_planning_shim_idempotent_when_hir_already_set()
  test_hir_to_mir_shim_runs_in_isolation()
  test_mir_opt_shim_preserves_step_count()
  test_mir_program_assembly_shim_returns_mir_program()
  test_assign_handles_shim_runs_in_isolation()
  test_collect_view_specs_shim_runs_in_isolation()
  test_emit_view_decls_shim_populates_maps()
  test_lower_kernel_body_shim_produces_iir()
  test_render_shim_returns_string_body()
  test_default_program_pipeline_end_to_end()
  test_default_kernel_pipeline_end_to_end()
  test_kernel_pipeline_count_phase_runs()
  test_compile_to_mir_uses_default_pipeline()
  test_compile_to_mir_apply_mir_passes_false_filters_mir_opt()
  test_compile_to_mir_with_precomputed_hir_skips_planning()
  test_compile_to_mir_verbose_does_not_break()
  test_compile_kernel_body_routes_through_default_pipeline()
  test_compile_kernel_body_threads_all_kwargs()
  test_assign_handles_shim_is_idempotent()
  print('OK')
