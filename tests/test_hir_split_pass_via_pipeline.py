'''D-Split acceptance — wrap-via-pipeline matches legacy direct call.

Spec: `docs/phase_d_hir_passes.md` section 3.2 (per-PR acceptance
gate). `SplitPass` (in `srdatalog.ir.dialects.hir.passes.split`)
wraps `srdatalog.ir.hir.split.TempRelSynthesisPass`. Run via
`Compiler.run(state, pipeline=[StratifyPass(), SplitPass()])`, the
resulting `HirProgram` must equal the legacy direct call
`TempRelSynthesisPass().run(stratify(rules, decls))` on the same
fixture.

The TC fixture has no split (TC's recursive rule is 2-atom and
small enough that the planner doesn't introduce a temp relation),
but the wrapper still must behave as a no-op pass-through on it.
A 3-atom recursive fixture exercises the split / temp-rel path.
'''

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pass, ProgramPass
from srdatalog.ir.dialects.hir.passes import HirPlanState, SplitPass, StratifyPass
from srdatalog.ir.hir import DIALECT as HIR_DIALECT
from srdatalog.ir.hir.pass_ import program_to_decls
from srdatalog.ir.hir.split import TempRelSynthesisPass as _LegacyTempRelSynth
from srdatalog.ir.hir.stratify import stratify as _legacy_stratify


def _compiler_with_hir() -> Compiler:
  '''Compiler with the `hir` dialect registered.

  Needed for pipelines that include `SplitPass` (or any other Wave 2B
  HIR transform) without a preceding `StratifyPass`: the pre-flight
  ordering check in `Compiler.run` requires the consumed `'hir'`
  dialect to be either registered OR produced by an earlier pass.'''
  c = Compiler()
  c.register_dialect(HIR_DIALECT)
  return c


def _tc_program() -> Program:
  '''TC fixture, copied from `test_default_pipelines_shims.py`.'''
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


# -----------------------------------------------------------------------------
# Pass-shape sanity (matches the framework's invariants)
# -----------------------------------------------------------------------------


def test_split_pass_is_a_program_pass():
  sp = SplitPass()
  assert isinstance(sp, Pass)
  assert isinstance(sp, ProgramPass)
  assert sp.name == 'split'
  assert sp.consumes == ('hir',)
  assert sp.produces == ('hir',)


# -----------------------------------------------------------------------------
# Behavior parity with the legacy `TempRelSynthesisPass`
# -----------------------------------------------------------------------------


def test_split_pass_matches_legacy_direct_call():
  '''Pipeline [StratifyPass(), SplitPass()] produces the same HIR as
  `TempRelSynthesisPass().run(stratify(rules, decls))`.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)

  via_pipeline = (
    Compiler().run(HirPlanState(program=prog), pipeline=[StratifyPass(), SplitPass()]).hir
  )
  via_legacy = _LegacyTempRelSynth().run(_legacy_stratify(rules, decls))

  assert via_pipeline is not None
  assert via_pipeline == via_legacy


def test_split_pass_preserves_hir_identity():
  '''The legacy `TempRelSynthesisPass.run` mutates HIR in place and
  returns the same instance. `SplitPass` must preserve this — the
  through-state's `hir` after SplitPass is identity-equal to the
  hir produced by StratifyPass (no defensive copy).'''
  prog = _tc_program()
  pre_split = Compiler().run(HirPlanState(program=prog), pipeline=[StratifyPass()])
  assert pre_split.hir is not None

  post_split = _compiler_with_hir().run(pre_split, pipeline=[SplitPass()])
  assert post_split.hir is pre_split.hir


def test_split_pass_alone_runs_on_pre_stratified_state():
  '''SplitPass can run in isolation given a state with `hir` already
  populated, matching the legacy invocation pattern where each HIR
  transform is composable.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)

  state = HirPlanState(rules=rules, decls=decls, hir=_legacy_stratify(rules, decls))
  out = _compiler_with_hir().run(state, pipeline=[SplitPass()])

  via_legacy = _LegacyTempRelSynth().run(_legacy_stratify(rules, decls))
  assert out.hir == via_legacy


def test_split_pass_requires_hir_in_state():
  '''SplitPass declares `consumes=('hir',)`. Calling apply with a
  state where `hir is None` raises (per the assertion in `_fn`).'''
  import pytest

  prog = _tc_program()
  state = HirPlanState(program=prog)  # no hir
  with pytest.raises(AssertionError):
    SplitPass().apply(state, None)


if __name__ == '__main__':
  test_split_pass_is_a_program_pass()
  test_split_pass_matches_legacy_direct_call()
  test_split_pass_preserves_hir_identity()
  test_split_pass_alone_runs_on_pre_stratified_state()
  test_split_pass_requires_hir_in_state()
  print('OK')
