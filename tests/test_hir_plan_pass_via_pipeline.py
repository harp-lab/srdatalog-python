'''D-Plan acceptance — wrap-via-pipeline matches legacy direct call.

Spec: `docs/phase_d_hir_passes.md` section 3.2 (per-PR acceptance
gate). `PlanPass` (in `srdatalog.ir.dialects.hir.passes.plan`) wraps
`srdatalog.ir.hir.plan.plan_joins`. Run via `Compiler.run(state,
pipeline=[..., PlanPass()])`, the resulting `HirProgram` must equal
the legacy direct call sequence on the same fixture.

`plan_joins` operates on per-variant fields populated by the
semi-naive variant generation step. The pipeline therefore must run
something that creates variants before `PlanPass`. When the Wave 2B
`SemiNaivePass` (D-SemiNaive) is importable, this file exercises the
spec's ordering `[StratifyPass, SplitPass, SemiNaivePass, PlanPass]`;
when it isn't yet, the variant step is supplied by the legacy
`SemiNaiveVariantPass.run(hir)` invoked between StratifyPass /
SplitPass and PlanPass (both branches share the parity assertion).
'''

from __future__ import annotations

import dataclasses

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pass, ProgramPass
from srdatalog.ir.dialects.hir.passes import HirPlanState, PlanPass, SplitPass, StratifyPass
from srdatalog.ir.hir import DIALECT as HIR_DIALECT
from srdatalog.ir.hir.pass_ import program_to_decls
from srdatalog.ir.hir.plan import plan_joins as _legacy_plan_joins
from srdatalog.ir.hir.semi_naive import SemiNaiveVariantPass as _LegacySemiNaive
from srdatalog.ir.hir.split import TempRelSynthesisPass as _LegacyTempRelSynth
from srdatalog.ir.hir.stratify import stratify as _legacy_stratify

try:
  from srdatalog.ir.dialects.hir.passes import SemiNaivePass  # type: ignore[attr-defined]

  _HAS_SEMI_NAIVE_PASS = True
except ImportError:  # D-SemiNaive in flight in parallel; not yet landed.
  SemiNaivePass = None  # type: ignore[assignment,misc]
  _HAS_SEMI_NAIVE_PASS = False


def _compiler_with_hir() -> Compiler:
  '''Compiler with the `hir` dialect registered. Same helper as
  `test_hir_split_pass_via_pipeline.py` — needed for pipelines that
  start mid-stream (consume `'hir'` without a preceding StratifyPass).'''
  c = Compiler()
  c.register_dialect(HIR_DIALECT)
  return c


def _tc_program() -> Program:
  '''TC fixture, copied from `test_hir_split_pass_via_pipeline.py`.'''
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


def _legacy_hir_through_plan(prog: Program):
  '''Drive the legacy HIR chain up to (and including) plan_joins.
  Mirrors the imperative order from `default_pipeline` in
  `srdatalog.ir.hir.__init__`, restricted to the HIR transforms that
  plan_joins depends on: stratify -> SemiNaive variants -> split
  (temp-rel synthesis is order-independent here, but matching the
  default_pipeline order keeps the comparison honest) -> plan.

  Returns the post-plan HirProgram.'''
  rules = list(prog.rules)
  decls = program_to_decls(prog)
  hir = _legacy_stratify(rules, decls)
  hir = _LegacySemiNaive().run(hir)
  hir = _LegacyTempRelSynth().run(hir)
  hir = _legacy_plan_joins(hir)
  return hir


# -----------------------------------------------------------------------------
# Pass-shape sanity (matches the framework's invariants)
# -----------------------------------------------------------------------------


def test_plan_pass_is_a_program_pass():
  pp = PlanPass()
  assert isinstance(pp, Pass)
  assert isinstance(pp, ProgramPass)
  assert pp.name == 'plan'
  assert pp.consumes == ('hir',)
  assert pp.produces == ('hir',)


# -----------------------------------------------------------------------------
# Pre-flight ordering — the spec's exact pipeline order
# -----------------------------------------------------------------------------


def test_plan_pass_preflight_ordering_with_stratify_and_split():
  '''The minimum pipeline that places PlanPass after a `'hir'`
  producer must pass Compiler.run's pre-flight ordering check.
  StratifyPass produces `'hir'`; SplitPass and PlanPass both consume
  and re-produce it. No PassOrderingError is raised.'''
  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[StratifyPass(), SplitPass(), PlanPass()],
  )
  assert out.hir is not None


def test_plan_pass_spec_pipeline_order_validates():
  '''Spec ordering: [StratifyPass, SplitPass, SemiNaivePass, PlanPass].
  When SemiNaivePass is importable, the exact spec pipeline validates
  end-to-end. When not, the same shape minus SemiNaivePass is
  exercised by `test_plan_pass_preflight_ordering_with_stratify_and_split`
  and this test is a no-op skip.'''
  if not _HAS_SEMI_NAIVE_PASS:
    import pytest

    pytest.skip('SemiNaivePass not importable yet (D-SemiNaive in flight)')

  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[StratifyPass(), SplitPass(), SemiNaivePass(), PlanPass()],
  )
  assert out.hir is not None


# -----------------------------------------------------------------------------
# Behavior parity with the legacy `plan_joins`
# -----------------------------------------------------------------------------


def test_plan_pass_matches_legacy_direct_call():
  '''Run PlanPass end-to-end (with the necessary upstream variant
  generation) and compare to the legacy direct call on the same
  fixture. The two HirPrograms must be `==`.'''
  prog = _tc_program()

  if _HAS_SEMI_NAIVE_PASS:
    pipeline = [StratifyPass(), SplitPass(), SemiNaivePass(), PlanPass()]
    via_pipeline = Compiler().run(HirPlanState(program=prog), pipeline=pipeline).hir
  else:
    # Stand in for the not-yet-landed SemiNaivePass: drive Stratify +
    # Split via the wrapper, then inject the legacy variant pass, then
    # run PlanPass on the resulting state.
    pre = Compiler().run(
      HirPlanState(program=prog),
      pipeline=[StratifyPass(), SplitPass()],
    )
    assert pre.hir is not None
    hir_with_variants = _LegacySemiNaive().run(pre.hir)
    state = dataclasses.replace(pre, hir=hir_with_variants)
    via_pipeline = _compiler_with_hir().run(state, pipeline=[PlanPass()]).hir

  via_legacy = _legacy_hir_through_plan(prog)
  assert via_pipeline is not None
  assert via_pipeline == via_legacy


def test_plan_pass_preserves_hir_identity():
  '''The legacy `plan_joins` mutates HIR in place and returns the
  same instance. `PlanPass` must preserve this — post-PlanPass, the
  through-state's `hir` is identity-equal to its pre-PlanPass `hir`
  (no defensive copy).'''
  prog = _tc_program()
  pre = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[StratifyPass(), SplitPass()],
  )
  assert pre.hir is not None
  hir_with_variants = _LegacySemiNaive().run(pre.hir)
  state = dataclasses.replace(pre, hir=hir_with_variants)

  post = _compiler_with_hir().run(state, pipeline=[PlanPass()])
  assert post.hir is state.hir


def test_plan_pass_alone_runs_on_pre_planned_state():
  '''PlanPass can run in isolation given a state with `hir` already
  populated (and variants generated), matching the legacy composable
  pattern.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)
  hir = _LegacySemiNaive().run(_LegacyTempRelSynth().run(_legacy_stratify(rules, decls)))

  state = HirPlanState(rules=rules, decls=decls, hir=hir)
  out = _compiler_with_hir().run(state, pipeline=[PlanPass()])

  via_legacy = _legacy_hir_through_plan(prog)
  assert out.hir == via_legacy


def test_plan_pass_requires_hir_in_state():
  '''PlanPass declares `consumes=('hir',)`. Calling apply with a
  state where `hir is None` raises (per the assertion in `_fn`).'''
  import pytest

  prog = _tc_program()
  state = HirPlanState(program=prog)  # no hir
  with pytest.raises(AssertionError):
    PlanPass().apply(state, None)


if __name__ == '__main__':
  test_plan_pass_is_a_program_pass()
  test_plan_pass_preflight_ordering_with_stratify_and_split()
  test_plan_pass_spec_pipeline_order_validates()
  test_plan_pass_matches_legacy_direct_call()
  test_plan_pass_preserves_hir_identity()
  test_plan_pass_alone_runs_on_pre_planned_state()
  test_plan_pass_requires_hir_in_state()
  print('OK')
