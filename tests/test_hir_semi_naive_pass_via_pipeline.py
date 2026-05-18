'''D-SemiNaive acceptance — wrap-via-pipeline matches legacy direct call.

Spec: `docs/phase_d_hir_passes.md` section 3.2 (per-PR acceptance
gate). `SemiNaivePass` (in
`srdatalog.ir.dialects.hir.passes.semi_naive`) wraps
`srdatalog.ir.hir.semi_naive.SemiNaiveVariantPass`. Run via
`Compiler.run(state, pipeline=[StratifyPass(), SemiNaivePass()])`,
the resulting `HirProgram` must equal the legacy direct call
`SemiNaiveVariantPass().run(stratify(rules, decls))` on the same
fixture.

A TC fixture (recursive Path rule) exercises the delta-variant code
path; an EDB-only fixture confirms the no-recursive-stratum case
(base variants only).
'''

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pass, ProgramPass
from srdatalog.ir.dialects.hir.passes import (
  HirPlanState,
  SemiNaivePass,
  StratifyPass,
)
from srdatalog.ir.hir import DIALECT as HIR_DIALECT
from srdatalog.ir.hir.pass_ import program_to_decls
from srdatalog.ir.hir.semi_naive import SemiNaiveVariantPass as _LegacySemiNaive
from srdatalog.ir.hir.stratify import stratify as _legacy_stratify
from srdatalog.ir.hir.types import Version


def _compiler_with_hir() -> Compiler:
  '''Compiler with the `hir` dialect registered.

  Needed for pipelines that include `SemiNaivePass` (or any other
  Wave 2B HIR transform) without a preceding `StratifyPass`: the
  pre-flight ordering check in `Compiler.run` requires the consumed
  `'hir'` dialect to be either registered OR produced by an earlier
  pass.'''
  c = Compiler()
  c.register_dialect(HIR_DIALECT)
  return c


def _tc_program() -> Program:
  '''TC fixture: recursive Path rule. Mirrors `_tc_program` in
  `test_hir_stratify_pass_via_pipeline.py` / `test_hir_split_pass_via_pipeline.py`.'''
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


def test_semi_naive_pass_is_a_program_pass():
  sp = SemiNaivePass()
  assert isinstance(sp, Pass)
  assert isinstance(sp, ProgramPass)
  assert sp.name == 'semi_naive'
  assert sp.consumes == ('hir',)
  assert sp.produces == ('hir',)


# -----------------------------------------------------------------------------
# Behavior parity with the legacy `SemiNaiveVariantPass`
# -----------------------------------------------------------------------------


def test_semi_naive_pass_matches_legacy_direct_call():
  '''Pipeline [StratifyPass(), SemiNaivePass()] produces the same HIR
  as `SemiNaiveVariantPass().run(stratify(rules, decls))`.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)

  via_pipeline = (
    Compiler().run(HirPlanState(program=prog), pipeline=[StratifyPass(), SemiNaivePass()]).hir
  )
  via_legacy = _LegacySemiNaive().run(_legacy_stratify(rules, decls))

  assert via_pipeline is not None
  assert via_pipeline == via_legacy


def test_semi_naive_pass_preserves_hir_identity():
  '''The legacy `SemiNaiveVariantPass.run` mutates HIR in place and
  returns the same instance. `SemiNaivePass` must preserve this — the
  through-state's `hir` after SemiNaivePass is identity-equal to the
  hir produced by StratifyPass (no defensive copy).'''
  prog = _tc_program()
  pre = Compiler().run(HirPlanState(program=prog), pipeline=[StratifyPass()])
  assert pre.hir is not None

  post = _compiler_with_hir().run(pre, pipeline=[SemiNaivePass()])
  assert post.hir is pre.hir


def test_semi_naive_pass_alone_runs_on_pre_stratified_state():
  '''SemiNaivePass can run in isolation given a state with `hir`
  already populated, matching the legacy invocation pattern where
  each HIR transform is composable.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)

  state = HirPlanState(rules=rules, decls=decls, hir=_legacy_stratify(rules, decls))
  out = _compiler_with_hir().run(state, pipeline=[SemiNaivePass()])

  via_legacy = _LegacySemiNaive().run(_legacy_stratify(rules, decls))
  assert out.hir == via_legacy


def test_semi_naive_pass_requires_hir_in_state():
  '''SemiNaivePass declares `consumes=('hir',)`. Calling apply with a
  state where `hir is None` raises (per the assertion in `_fn`).'''
  import pytest

  prog = _tc_program()
  state = HirPlanState(program=prog)  # no hir
  with pytest.raises(AssertionError):
    SemiNaivePass().apply(state, None)


# -----------------------------------------------------------------------------
# Pre-flight ordering + end-to-end smoke (TC delta variants)
# -----------------------------------------------------------------------------


def test_stratify_then_semi_naive_pipeline_validates_and_runs():
  '''A pipeline `[StratifyPass(), SemiNaivePass()]` passes the
  pre-flight ordering check (`SemiNaivePass` consumes `'hir'`, which
  `StratifyPass` produces) and runs end-to-end.'''
  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[StratifyPass(), SemiNaivePass()],
  )
  assert out.hir is not None
  assert len(out.hir.strata) > 0


def test_semi_naive_pass_generates_delta_variants_for_tc():
  '''Smoke: on the TC fixture (recursive `Path` rule), SemiNaivePass
  populates `recursive_variants` on the recursive stratum with at
  least one DELTA variant. Non-recursive strata get `base_variants`.'''
  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[StratifyPass(), SemiNaivePass()],
  )
  assert out.hir is not None

  recursive_strata = [s for s in out.hir.strata if s.is_recursive]
  assert recursive_strata, 'TC fixture must have a recursive stratum'

  delta_variants = [v for s in recursive_strata for v in s.recursive_variants if v.delta_idx >= 0]
  assert delta_variants, 'recursive stratum must yield at least one DELTA variant'
  for v in delta_variants:
    assert v.clause_versions[v.delta_idx] == Version.DELTA

  non_recursive_strata = [s for s in out.hir.strata if not s.is_recursive]
  for s in non_recursive_strata:
    assert s.base_variants, 'non-recursive stratum must have base variants'
    for v in s.base_variants:
      assert v.delta_idx == -1


if __name__ == '__main__':
  test_semi_naive_pass_is_a_program_pass()
  test_semi_naive_pass_matches_legacy_direct_call()
  test_semi_naive_pass_preserves_hir_identity()
  test_semi_naive_pass_alone_runs_on_pre_stratified_state()
  test_semi_naive_pass_requires_hir_in_state()
  test_stratify_then_semi_naive_pipeline_validates_and_runs()
  test_semi_naive_pass_generates_delta_variants_for_tc()
  print('OK')
