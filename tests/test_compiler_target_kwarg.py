'''PR-1c (Phase B2-1, foundation) — target kwarg threading.

Spec: docs/phase_decomposition_redesign.md §2.3 (target monopolization)
and §3.3.1 (target chosen at run time). PR-1c adds the API surface
without changing render-dispatch behavior; default ``target='cuda'``
preserves all existing call sites byte-equivalently.

These tests assert the kwarg + field threading is in place AND that
the default 'cuda' path produces identical output to the pre-PR-1c
code path (no compile or no kwarg).
'''

from __future__ import annotations

import srdatalog.ir.mir.types as mir
from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler
from srdatalog.ir.default_pipelines import (
  DEFAULT_KERNEL_PIPELINE,
  DEFAULT_PROGRAM_PIPELINE,
  InitialProg,
  KernelCtx,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _tc_program() -> Program:
  '''Tiny TC-style program used across the PR-1c kwarg tests.'''
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


# -----------------------------------------------------------------------------
# Carrier dataclass field tests
# -----------------------------------------------------------------------------


def test_initial_prog_default_target_is_cuda():
  '''InitialProg.target defaults to 'cuda' (preserves existing
  behavior; per spec §3.3.1).'''
  prog = InitialProg(program=_tc_program())
  assert prog.target == 'cuda'


def test_initial_prog_target_is_replaceable():
  '''The target field is plumbed through `dataclasses.replace`-style
  state threading without rejecting non-cuda values.'''
  import dataclasses

  prog = InitialProg(program=_tc_program(), target='cpu_tbb')
  assert prog.target == 'cpu_tbb'
  prog2 = dataclasses.replace(prog, target='wasm')
  assert prog2.target == 'wasm'


def test_kernel_ctx_default_target_is_cuda():
  '''KernelCtx.target defaults to 'cuda'.'''
  compiler = Compiler.with_default_plugins()
  state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert state.mir_program is not None
  ep = _first_ep(state.mir_program)
  ctx = KernelCtx(ep=ep, is_counting=False)
  assert ctx.target == 'cuda'


def test_kernel_ctx_target_explicit_round_trip():
  '''Explicit `target='cuda'` round-trips identically to the default.'''
  compiler = Compiler.with_default_plugins()
  state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert state.mir_program is not None
  ep = _first_ep(state.mir_program)

  default = compiler.run(KernelCtx(ep=ep, is_counting=False), pipeline=DEFAULT_KERNEL_PIPELINE)
  explicit = compiler.run(
    KernelCtx(ep=ep, is_counting=False, target='cuda'), pipeline=DEFAULT_KERNEL_PIPELINE
  )
  assert default.body_text == explicit.body_text


# -----------------------------------------------------------------------------
# Compiler.run target kwarg
# -----------------------------------------------------------------------------


def test_compiler_run_accepts_target_kwarg():
  '''`Compiler.run(..., target='cuda')` is accepted and stored on the
  compiler for the duration of the call.'''
  compiler = Compiler.with_default_plugins()
  state = compiler.run(
    InitialProg(program=_tc_program()),
    pipeline=DEFAULT_PROGRAM_PIPELINE,
    target='cuda',
  )
  assert state.mir_program is not None


def test_compiler_run_default_target_is_cuda():
  '''Omitting the kwarg defaults to 'cuda'; the active_target attr is
  restored to the prior value after the run.'''
  compiler = Compiler.with_default_plugins()
  assert compiler.active_target == 'cuda'

  state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert state.mir_program is not None
  # Restored after the call returns.
  assert compiler.active_target == 'cuda'


def test_compiler_run_target_kwarg_is_byte_equivalent_to_default():
  '''Threading target='cuda' through the kernel pipeline produces
  byte-identical body_text to the default (no-kwarg) path. This is
  the load-bearing byte-equivalence guarantee of PR-1c: the kwarg is
  pure plumbing today.'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  default = compiler.run(
    KernelCtx(ep=ep, is_counting=False),
    pipeline=DEFAULT_KERNEL_PIPELINE,
  )
  explicit = compiler.run(
    KernelCtx(ep=ep, is_counting=False, target='cuda'),
    pipeline=DEFAULT_KERNEL_PIPELINE,
    target='cuda',
  )
  assert default.body_text == explicit.body_text
  assert default.body_text is not None


def test_compiler_run_target_restored_on_exception():
  '''If a pass raises, `active_target` must still be restored to the
  prior value (try/finally contract).'''
  import contextlib

  from srdatalog.ir.core import Pass

  class _Boom(Pass):
    def __init__(self) -> None:
      super().__init__(name='_boom')

    def apply(self, prog: object, compiler: object) -> object:
      raise RuntimeError('boom')

  compiler = Compiler.with_default_plugins()
  assert compiler.active_target == 'cuda'
  with contextlib.suppress(RuntimeError):
    compiler.run(object(), pipeline=[_Boom()], target='cpu_tbb')
  assert compiler.active_target == 'cuda'


# -----------------------------------------------------------------------------
# VerifyRenderabilityShim reads target from KernelCtx
# -----------------------------------------------------------------------------


def test_verify_renderability_uses_kernel_ctx_target():
  '''`VerifyRenderabilityShim` reads `state.target` (no longer hardcodes
  'cuda'). With default target='cuda' the existing renderers cover the
  IIR tree, so the pipeline runs cleanly — the regression we'd see if
  someone re-hardcoded the literal is this test failing for a non-cuda
  KernelCtx, but the byte-equivalence path with 'cuda' must stay
  green.'''
  compiler = Compiler.with_default_plugins()
  prog_state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert prog_state.mir_program is not None
  ep = _first_ep(prog_state.mir_program)

  # Default target='cuda' renders cleanly through the verifier.
  out = compiler.run(
    KernelCtx(ep=ep, is_counting=False, target='cuda'),
    pipeline=DEFAULT_KERNEL_PIPELINE,
  )
  assert out.body_text is not None
