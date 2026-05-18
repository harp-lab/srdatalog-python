'''F5.4 acceptance — pipeline is data, not control flow.

Anchors `docs/phase_f5_declarative_pipeline.md` section 8: the imperative
entry points (`compile_to_mir`, `compile_kernel_body`) and the
declarative path (`Compiler().run(..., pipeline=DEFAULT_*_PIPELINE)`)
must produce byte-identical outputs across a representative cross
section of DSL fixtures. Future refactors that silently break this
property fail here first.

Fixtures cover the three structural axes the byte-equivalence suite
also targets:

  1. Transitive closure  — single recursive rule.
  2. Negation            — a 4-atom rule with `~knows(...)` + Filter.
  3. Recursive 3-atom    — points-to style join under recursion.

Plus two composability checks: pipeline-swap (drop a shim, get the
`apply_mir_passes=False` shape) and pipeline-extension (append a no-op
ProgramPass; the result still validates).
'''

from __future__ import annotations

import dataclasses

import srdatalog.ir.mir.types as mir
from srdatalog.dsl import Filter, Program, Relation, Var
from srdatalog.ir.codegen.cuda.api import compile_kernel_body
from srdatalog.ir.core import Compiler, ProgramPass
from srdatalog.ir.default_pipelines import (
  DEFAULT_KERNEL_PIPELINE,
  DEFAULT_PROGRAM_PIPELINE,
  InitialProg,
  KernelCtx,
  MirOptShim,
)
from srdatalog.ir.hir import compile_to_mir

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _tc_program() -> Program:
  '''Tiny TC-style program: ArcInput -> Edge, Edge -> Path (base + rec).

  Mirrors `_tc_program` in test_default_pipelines_shims.py — the
  simplest recursive stratum the codebase ships.'''
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


def _negation_program() -> Program:
  '''2-hop path with negation + a binary filter. Exercises the negated-
  scan + Filter lowering paths through the program pipeline.'''
  P1, P2, P3 = Var('p1'), Var('p2'), Var('p3')
  k_in = Relation('KnowsInput', 2)
  knows = Relation('Knows', 2)
  hop2 = Relation('Hop2', 3)
  return Program(
    rules=[
      (knows(P1, P2) <= k_in(P1, P2)).named('KnowsLoad'),
      (
        hop2(P1, P2, P3)
        <= knows(P1, P2)
        & knows(P2, P3)
        & ~knows(P1, P3)
        & Filter(vars=('p1', 'p3'), code='return p1 != p3;')
      ).named('Hop2NoDirect'),
    ],
  )


def _recursive_three_atom_program() -> Program:
  '''Points-to style recursion: 3-atom rule whose head recurses.
  Stresses semi-naive variant generation + the multi-join lowering
  path that single-atom TC never hits.'''
  X, Y, Z, W = Var('x'), Var('y'), Var('z'), Var('w')
  ao = Relation('AddressOf', 2)
  load_r = Relation('Load', 2)
  pt = Relation('PointsTo', 2)
  return Program(
    rules=[
      (pt(Y, X) <= ao(Y, X)).named('Base'),
      (pt(Y, W) <= pt(X, Z) & load_r(Y, X) & pt(Z, W)).named('LoadRec'),
    ],
  )


PROGRAM_FIXTURES = [
  ('tc', _tc_program),
  ('negation', _negation_program),
  ('recursive_three_atom', _recursive_three_atom_program),
]


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


def _all_eps(mir_prog: mir.Program) -> list[mir.ExecutePipeline]:
  out: list[mir.ExecutePipeline] = []
  for step, _is_rec in mir_prog.steps:
    out.extend(_find_execute_pipelines(step))
  return out


# -----------------------------------------------------------------------------
# Test 1: program pipeline byte-equivalence
# -----------------------------------------------------------------------------


def test_default_program_pipeline_matches_compile_to_mir() -> None:
  '''For every fixture, the declarative pipeline reaches the same
  `mir.Program` value as the imperative `compile_to_mir` wrapper.

  This is the foundational property: it locks `compile_to_mir` to
  `DEFAULT_PROGRAM_PIPELINE` so swapping or reordering shims at the
  pipeline level is the ONLY way to change `compile_to_mir`'s
  behavior — no hidden control flow in the wrapper.'''
  for name, build in PROGRAM_FIXTURES:
    prog = build()
    declarative = (
      Compiler().run(InitialProg(program=prog), pipeline=DEFAULT_PROGRAM_PIPELINE).mir_program
    )
    imperative = compile_to_mir(prog)
    assert declarative == imperative, (
      f'fixture {name!r}: declarative pipeline diverged from compile_to_mir'
    )


# -----------------------------------------------------------------------------
# Test 2: kernel pipeline byte-equivalence
# -----------------------------------------------------------------------------


def test_default_kernel_pipeline_matches_compile_kernel_body() -> None:
  '''For every EP extracted from each fixture's MIR, both count and
  materialize phases produce byte-identical body text whether routed
  through `compile_kernel_body` or `DEFAULT_KERNEL_PIPELINE` direct.
  '''
  compiler = Compiler()
  eps_seen = 0
  for name, build in PROGRAM_FIXTURES:
    mir_prog = compile_to_mir(build())
    for ep in _all_eps(mir_prog):
      eps_seen += 1
      for is_counting in (False, True):
        via_api = compile_kernel_body(ep, is_counting=is_counting)
        via_pipeline = compiler.run(
          KernelCtx(ep=ep, is_counting=is_counting),
          pipeline=DEFAULT_KERNEL_PIPELINE,
        ).body_text
        assert via_api == via_pipeline, (
          f'fixture {name!r} ep {ep.rule_name!r} '
          f'is_counting={is_counting}: declarative kernel pipeline '
          f'diverged from compile_kernel_body'
        )
  # Defensive: if fixture discovery regresses to zero EPs, the test
  # would silently pass. Anchor a minimum.
  assert eps_seen >= 3, f'expected >= 3 ExecutePipelines across fixtures, saw {eps_seen}'


# -----------------------------------------------------------------------------
# Test 3: pipeline-as-data — swap a shim
# -----------------------------------------------------------------------------


def test_pipeline_swap_works() -> None:
  '''Demonstrate the "pipeline is data" win: removing `MirOptShim`
  from `DEFAULT_PROGRAM_PIPELINE` at the test site must produce the
  same result as the imperative `apply_mir_passes=False` knob.

  If this passes, downstream callers (alternate targets, debug tools,
  future plugins) can introduce custom pipelines without forking the
  entry-point function.'''
  for name, build in PROGRAM_FIXTURES:
    prog = build()
    filtered = [p for p in DEFAULT_PROGRAM_PIPELINE if not isinstance(p, MirOptShim)]
    via_swap = Compiler().run(InitialProg(program=prog), pipeline=filtered).mir_program
    via_knob = compile_to_mir(prog, apply_mir_passes=False)
    assert via_swap == via_knob, (
      f'fixture {name!r}: pipeline-swap (drop MirOptShim) diverged '
      f'from compile_to_mir(apply_mir_passes=False)'
    )


# -----------------------------------------------------------------------------
# Test 4: pipeline-as-data — append a no-op pass
# -----------------------------------------------------------------------------


def _identity_program_pass() -> ProgramPass:
  '''Build a no-op user `ProgramPass` that returns its state unchanged.
  Declares `consumes=('mir_program',)` so the ordering validator
  places it after `MirProgramAssembly`.'''

  def _identity(state: InitialProg, _compiler: object) -> InitialProg:
    # dataclasses.replace with no overrides — proves the state object
    # round-trips a `dataclasses.replace` call (the shim contract).
    return dataclasses.replace(state)

  return ProgramPass(
    name='user_noop',
    consumes=('mir_program',),
    produces=('mir_program',),
    fn=_identity,
  )


def test_register_extra_pass_doesnt_break_default() -> None:
  '''Append a no-op user `ProgramPass` after `MirProgramAssembly`.
  The result must still be a `mir.Program` byte-equal to the default
  path; this confirms the pipeline list is composable with extra
  steps without touching any imperative wrapper.'''
  extended = [*DEFAULT_PROGRAM_PIPELINE, _identity_program_pass()]
  for name, build in PROGRAM_FIXTURES:
    prog = build()
    via_extended = Compiler().run(InitialProg(program=prog), pipeline=extended).mir_program
    via_default = compile_to_mir(prog)
    assert isinstance(via_extended, mir.Program), (
      f'fixture {name!r}: extended pipeline did not yield a mir.Program'
    )
    assert via_extended == via_default, (
      f'fixture {name!r}: appending a no-op ProgramPass perturbed the result'
    )


if __name__ == '__main__':
  test_default_program_pipeline_matches_compile_to_mir()
  test_default_kernel_pipeline_matches_compile_kernel_body()
  test_pipeline_swap_works()
  test_register_extra_pass_doesnt_break_default()
  print('OK')
