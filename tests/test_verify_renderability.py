'''R3 closure check — `verify_renderability` + `VerifyRenderabilityShim`.

Spec: docs/code_discipline.md R3 + docs/concept_glossary.md section 13.6
(the closure invariant). Topology check #4 in docs/concept_glossary.md
section 13.3.

Four shapes covered:
  1. Smoke: an IIR tree built from production-registered ops passes.
  2. Negative: a synthetic op with neither @register_render nor a
     @rewrite raises `UnrenderableOpError` naming the op + target.
  3. Rewrite escape hatch: registering a @rewrite for an otherwise
     unrenderable op closes the check.
  4. Integration: `Compiler.run(KernelCtx, pipeline=DEFAULT_KERNEL_PIPELINE)`
     on a TC fixture passes (because every op renders); inserting a
     fake shim that produces an unrenderable op raises at the
     `VerifyRenderabilityShim` step, BEFORE `CudaRenderShim` runs.
'''

from __future__ import annotations

from dataclasses import dataclass

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import (
  Compiler,
  Dialect,
  Op,
  ProgramPass,
  UnrenderableOpError,
  verify_renderability,
)
from srdatalog.ir.core.passes import rewrite
from srdatalog.ir.default_pipelines import (
  DEFAULT_KERNEL_PIPELINE,
  DEFAULT_PROGRAM_PIPELINE,
  InitialProg,
  KernelCtx,
  VerifyRenderabilityShim,
)
from srdatalog.ir.dialects.iir.cf.ops import Block, RawString, VarRef

# -----------------------------------------------------------------------------
# Synthetic op fixtures (NOT registered in any production renderer)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SyntheticUnrenderableOp(Op):
  '''An op type that no production renderer claims. Used to drive the
  negative path of `verify_renderability`.

  Carries one child Op so we can also test the walker's recursion into
  nested unrenderable subtrees.'''

  payload: str
  child: Op | None = None


@dataclass(frozen=True, slots=True)
class _SyntheticLeaf(Op):
  '''Companion leaf for `_SyntheticUnrenderableOp`. Also unrenderable
  unless a rewrite is registered.'''

  name: str


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _tc_program() -> Program:
  '''Tiny TC-style program: ArcInput -> Edge, Edge -> Path (base + rec).
  Same fixture as test_default_pipelines_shims.py and test_f5_acceptance.py.'''
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


def _real_iir_tree() -> object:
  '''Build a production IIR tree by running the kernel pipeline up to
  (but not including) the renderability check. Returns the populated
  `iir` field — guaranteed to consist only of registered op types.'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)
  # Run kernel pipeline up through LowerScanPipelineShim (inclusive)
  # — i.e. everything BEFORE the gate we're testing. This avoids
  # recursion (we're literally building the fixture the gate consumes).
  partial_pipeline = []
  for p in DEFAULT_KERNEL_PIPELINE:
    if p.name == 'verify_renderability':
      break
    partial_pipeline.append(p)
  state = compiler.run(KernelCtx(ep=ep, is_counting=False), pipeline=partial_pipeline)
  assert state.iir is not None
  return state.iir


# -----------------------------------------------------------------------------
# 1. Smoke: production IIR tree passes
# -----------------------------------------------------------------------------


def test_verify_renderability_passes_on_production_iir():
  '''An IIR tree built from the TC fixture contains only ops that
  every production renderer registers. The check returns None
  (no exception).'''
  tree = _real_iir_tree()
  # No exception = success. Pass `compiler` to mirror production
  # call site in `VerifyRenderabilityShim`.
  compiler = Compiler.with_default_plugins()
  assert verify_renderability(tree, target='cuda', compiler=compiler) is None


def test_verify_renderability_passes_on_minimal_block():
  '''The simplest renderable tree — a `Block` of a `RawString`. Both
  classes are registered in `codegen/cuda/render/iir_cf.py`. Closure
  check must pass without a Compiler.'''
  tree = Block(stmts=(RawString(text='int x = 0;'),))
  # Without a compiler — only the global render registry gates.
  assert verify_renderability(tree, target='cuda') is None


def test_verify_renderability_passes_on_var_ref_expression():
  '''VarRef is registered in expr mode only — the closure check counts
  expression-mode handlers as renderable even though they don't
  appear in the statement-mode dispatch path.'''
  tree = VarRef(name='x')
  assert verify_renderability(tree, target='cuda') is None


# -----------------------------------------------------------------------------
# 2. Negative: synthetic unrenderable op raises
# -----------------------------------------------------------------------------


def test_verify_renderability_raises_on_synthetic_unrenderable_op():
  '''A tree rooted at a synthetic op with no @register_render and no
  @rewrite raises `UnrenderableOpError` naming both the op type and
  the target.'''
  tree = _SyntheticUnrenderableOp(payload='nope', child=None)
  with pytest.raises(UnrenderableOpError) as excinfo:
    verify_renderability(tree, target='cuda')
  err = excinfo.value
  assert err.op_type is _SyntheticUnrenderableOp
  assert err.target == 'cuda'
  # Message must name the op + target so the failure is actionable.
  msg = str(err)
  assert '_SyntheticUnrenderableOp' in msg
  assert "'cuda'" in msg


def test_verify_renderability_raises_on_nested_unrenderable_op():
  '''A renderable root that contains an unrenderable child still
  raises — the check is closure over the entire tree, not just the
  root. Here `Block` is renderable but `_SyntheticUnrenderableOp`
  (nested two levels deep) is not.'''
  inner = _SyntheticUnrenderableOp(payload='nested', child=None)
  tree = Block(stmts=(RawString(text='ok'), inner))
  with pytest.raises(UnrenderableOpError) as excinfo:
    verify_renderability(tree, target='cuda')
  assert excinfo.value.op_type is _SyntheticUnrenderableOp


def test_verify_renderability_raises_on_unknown_target():
  '''Asking for a target that has no registered renderers fails on the
  first renderable-but-only-via-cuda op. This exercises the "other
  target degrades gracefully" branch — useful for plugin authors
  scaffolding a non-CUDA backend.'''
  tree = Block(stmts=(RawString(text='hi'),))
  with pytest.raises(UnrenderableOpError) as excinfo:
    verify_renderability(tree, target='metal')
  # First reachable op type (deterministic from set iteration order
  # isn't guaranteed; just assert SOMETHING reachable raised).
  assert excinfo.value.target == 'metal'
  assert excinfo.value.op_type in (Block, RawString)


# -----------------------------------------------------------------------------
# 3. Rewrite escape hatch
# -----------------------------------------------------------------------------


def test_rewrite_closes_an_otherwise_unrenderable_op():
  '''If a @rewrite is registered on a Dialect for an op type, the
  closure check treats that op as renderable (the rewrite is presumed
  to decompose it into ops that are themselves closed). Without the
  Compiler, the SAME tree raises — with the Compiler, it passes.
  '''
  # Set up: a synthetic dialect with a rewrite registered for
  # _SyntheticLeaf. The rewrite body is a no-op identity (the
  # closure check doesn't actually invoke rewrites — it only consults
  # the REGISTRATION).
  dialect = Dialect(name='test.synthetic.rewrite_close')

  @rewrite(dialect, _SyntheticLeaf)
  def _identity_rewrite(op, _compiler):
    return op

  compiler = Compiler()
  compiler.register_dialect(dialect)

  tree = _SyntheticLeaf(name='x')

  # Without the compiler: only the render registry counts → raises.
  with pytest.raises(UnrenderableOpError):
    verify_renderability(tree, target='cuda')

  # With the compiler: the rewrite registration closes the op type.
  assert verify_renderability(tree, target='cuda', compiler=compiler) is None


# -----------------------------------------------------------------------------
# 4. Integration: VerifyRenderabilityShim in DEFAULT_KERNEL_PIPELINE
# -----------------------------------------------------------------------------


def test_verify_renderability_shim_is_between_lower_scan_and_cuda_render():
  '''Pipeline shape contract: the gate runs AFTER LowerKernelBodyShim
  (which produces IIR) and BEFORE RenderShim (which consumes it).

  PR-1 rename (per `docs/phase_decomposition_redesign.md` § 3.3.3):
  `lower_scan_pipeline` -> `lower_kernel_body`; `cuda_render` ->
  `render`.
  '''
  names = [p.name for p in DEFAULT_KERNEL_PIPELINE]
  i_lower = names.index('lower_kernel_body')
  i_verify = names.index('verify_renderability')
  i_render = names.index('render')
  assert i_lower < i_verify < i_render


def test_verify_renderability_shim_is_a_pass_through():
  '''The shim is a gate, not a transformation. It returns the
  KernelCtx unchanged on success.'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  partial_pipeline = []
  for p in DEFAULT_KERNEL_PIPELINE:
    if p.name == 'verify_renderability':
      break
    partial_pipeline.append(p)
  state = compiler.run(KernelCtx(ep=ep, is_counting=False), pipeline=partial_pipeline)

  out = VerifyRenderabilityShim().apply(state, compiler)
  assert out is state  # pass-through


def test_default_kernel_pipeline_runs_end_to_end_with_gate():
  '''The full DEFAULT_KERNEL_PIPELINE (now including the gate) runs
  cleanly on a TC fixture. This is the regression guard for "every
  op type in production IIR has a renderer".'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)
  out = compiler.run(KernelCtx(ep=ep, is_counting=False), pipeline=DEFAULT_KERNEL_PIPELINE)
  assert isinstance(out.body_text, str)
  assert len(out.body_text) > 0


def test_kernel_pipeline_with_injected_unrenderable_op_raises_at_gate():
  '''Negative integration: insert a shim that REPLACES the IIR with a
  tree containing an unrenderable synthetic op. The pipeline must
  raise `UnrenderableOpError` at `VerifyRenderabilityShim` — BEFORE
  `CudaRenderShim` runs (which would otherwise hit its own KeyError).

  The injected shim consumes 'iir' and produces 'iir' so the pipeline
  ordering check still passes; only the closure check fires.'''
  import dataclasses

  class _InjectUnrenderable(ProgramPass):
    def __init__(self) -> None:
      super().__init__(
        name='inject_unrenderable',
        consumes=('iir',),
        produces=('iir',),
        fn=self._fn,
      )

    @staticmethod
    def _fn(state: KernelCtx, _compiler):
      return dataclasses.replace(
        state, iir=_SyntheticUnrenderableOp(payload='injected', child=None)
      )

  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  # Build a pipeline that injects the unrenderable op AFTER
  # LowerKernelBodyShim but BEFORE the gate. The gate must catch it.
  inject = _InjectUnrenderable()
  patched: list = []
  for p in DEFAULT_KERNEL_PIPELINE:
    patched.append(p)
    if p.name == 'lower_kernel_body':
      patched.append(inject)

  with pytest.raises(UnrenderableOpError) as excinfo:
    compiler.run(KernelCtx(ep=ep, is_counting=False), pipeline=patched)
  assert excinfo.value.op_type is _SyntheticUnrenderableOp
  assert excinfo.value.target == 'cuda'


def test_verify_renderability_shim_consumes_iir_produces_nothing():
  '''Pipeline-data contract: the gate consumes the iir dialect produced
  by LowerScanPipelineShim. Because it doesn't introduce a new
  pseudo-dialect, `produces=()`.'''
  shim = VerifyRenderabilityShim()
  assert shim.consumes == ('iir',)
  assert shim.produces == ()


if __name__ == '__main__':
  test_verify_renderability_passes_on_production_iir()
  test_verify_renderability_passes_on_minimal_block()
  test_verify_renderability_passes_on_var_ref_expression()
  test_verify_renderability_raises_on_synthetic_unrenderable_op()
  test_verify_renderability_raises_on_nested_unrenderable_op()
  test_verify_renderability_raises_on_unknown_target()
  test_rewrite_closes_an_otherwise_unrenderable_op()
  test_verify_renderability_shim_is_between_lower_scan_and_cuda_render()
  test_verify_renderability_shim_is_a_pass_through()
  test_default_kernel_pipeline_runs_end_to_end_with_gate()
  test_kernel_pipeline_with_injected_unrenderable_op_raises_at_gate()
  test_verify_renderability_shim_consumes_iir_produces_nothing()
  print('OK')
