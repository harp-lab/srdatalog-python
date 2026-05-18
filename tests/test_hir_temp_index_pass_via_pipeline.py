'''D-TempIndex acceptance — wrap-via-pipeline matches legacy direct call.

Spec: `docs/phase_d_hir_passes.md` section 3.2 (per-PR acceptance
gate). `TempIndexPass` (in
`srdatalog.ir.dialects.hir.passes.temp_index`) wraps
`srdatalog.ir.hir.split.TempIndexRegistrationPass` (Pass 5.5: register
the identity index `[0..arity-1]` for each split variant's temp
relation in the enclosing stratum's `required_indices` /
`canonical_index` maps). Run via `Compiler.run(state, pipeline=[...,
TempIndexPass()])`, the resulting `HirProgram` must equal the legacy
direct-call chain on the same fixture.

Naming-drift note: the spec table cites D-Split as wrapping
`split_multihead`, but `split_multihead` was never implemented; the
in-tree `hir/split.py` exports two classes — `TempRelSynthesisPass`
(Pass 4.5, wrapped by D-Split / `SplitPass` in PR #48) and
`TempIndexRegistrationPass` (Pass 5.5, wrapped here as
`TempIndexPass`). This PR fills the remaining D-TempIndex slot.

`TempIndexRegistrationPass` runs AFTER `IndexSelectionPass` in the
legacy default pipeline (the selection pass doesn't see temp
relations, which are synthetic rule heads). The Wave 2B
`IndexSelectionPass` wrapper (D-Index) is in flight in parallel —
where it's importable, this file exercises the full spec pipeline
`[StratifyPass, SplitPass, SemiNaivePass, PlanPass,
IndexSelectionPass, TempIndexPass]`; where it isn't yet, the
selection step is supplied by the legacy `IndexSelectionPass.run(hir)`
invoked between the wrappers and `TempIndexPass` (both branches
share the parity assertion).
'''

from __future__ import annotations

import dataclasses

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pass, ProgramPass
from srdatalog.ir.dialects.hir.passes import (
  HirPlanState,
  PlanPass,
  SemiNaivePass,
  SplitPass,
  StratifyPass,
  TempIndexPass,
)
from srdatalog.ir.hir import DIALECT as HIR_DIALECT
from srdatalog.ir.hir.index import IndexSelectionPass as _LegacyIndexSelection
from srdatalog.ir.hir.pass_ import program_to_decls
from srdatalog.ir.hir.plan import plan_joins as _legacy_plan_joins
from srdatalog.ir.hir.semi_naive import SemiNaiveVariantPass as _LegacySemiNaive
from srdatalog.ir.hir.split import (
  TempIndexRegistrationPass as _LegacyTempIndexReg,
)
from srdatalog.ir.hir.split import (
  TempRelSynthesisPass as _LegacyTempRelSynth,
)
from srdatalog.ir.hir.stratify import stratify as _legacy_stratify

try:
  from srdatalog.ir.dialects.hir.passes import (  # type: ignore[attr-defined]
    IndexSelectionPass as _WrapperIndexSelectionPass,
  )

  _HAS_INDEX_PASS = True
except ImportError:  # D-Index in flight in parallel; not yet landed.
  _WrapperIndexSelectionPass = None  # type: ignore[assignment,misc]
  _HAS_INDEX_PASS = False


def _compiler_with_hir() -> Compiler:
  '''Compiler with the `hir` dialect registered. Same helper as the
  other Wave 2B test files — needed for pipelines that start
  mid-stream (consume `'hir'` without a preceding StratifyPass).'''
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


def _three_atom_program() -> Program:
  '''Recursive 3-atom rule that the planner splits, exercising the
  temp-rel / temp-index path. Mirrors the split-fixture intent from
  `test_hir_split_pass_via_pipeline.py` (which notes TC alone has no
  split because its recursive body is only 2 atoms).'''
  X, Y, Z, W = Var('x'), Var('y'), Var('z'), Var('w')
  arc = Relation('ArcInput', 2)
  edge = Relation('Edge', 2)
  path = Relation('Path', 2)
  return Program(
    rules=[
      (edge(X, Y) <= arc(X, Y)).named('EdgeLoad'),
      (path(X, Y) <= edge(X, Y)).named('PBase'),
      (path(X, W) <= path(X, Y) & edge(Y, Z) & edge(Z, W)).named('P3Rec'),
    ],
  )


def _legacy_hir_through_temp_index(prog: Program):
  '''Drive the legacy HIR chain through TempIndexRegistrationPass.
  Mirrors the imperative order from `default_pipeline` in
  `srdatalog.ir.hir.__init__`:
    stratify -> SemiNaive variants -> TempRelSynth -> plan ->
    IndexSelection -> TempIndexRegistration.
  Returns the post-temp-index HirProgram.'''
  rules = list(prog.rules)
  decls = program_to_decls(prog)
  hir = _legacy_stratify(rules, decls)
  hir = _LegacySemiNaive().run(hir)
  hir = _LegacyTempRelSynth().run(hir)
  hir = _legacy_plan_joins(hir)
  hir = _LegacyIndexSelection().run(hir)
  hir = _LegacyTempIndexReg().run(hir)
  return hir


def _state_pre_temp_index(prog: Program) -> HirPlanState:
  '''Build an HirPlanState driven through every legacy step up to (but
  not including) TempIndexRegistrationPass. Used by the isolation /
  identity tests so they don't depend on the D-Index wrapper.'''
  rules = list(prog.rules)
  decls = program_to_decls(prog)
  hir = _legacy_stratify(rules, decls)
  hir = _LegacySemiNaive().run(hir)
  hir = _LegacyTempRelSynth().run(hir)
  hir = _legacy_plan_joins(hir)
  hir = _LegacyIndexSelection().run(hir)
  return HirPlanState(rules=rules, decls=decls, hir=hir)


# -----------------------------------------------------------------------------
# Pass-shape sanity (matches the framework's invariants)
# -----------------------------------------------------------------------------


def test_temp_index_pass_is_a_program_pass():
  tip = TempIndexPass()
  assert isinstance(tip, Pass)
  assert isinstance(tip, ProgramPass)
  assert tip.name == 'temp_index'
  assert tip.consumes == ('hir',)
  assert tip.produces == ('hir',)


# -----------------------------------------------------------------------------
# Pre-flight ordering — the spec's exact pipeline order
# -----------------------------------------------------------------------------


def test_temp_index_pass_preflight_ordering_minimum():
  '''Minimum pipeline that places TempIndexPass after a `'hir'`
  producer must pass Compiler.run's pre-flight ordering check.
  StratifyPass produces `'hir'`; every later pass consumes and
  re-produces it. No PassOrderingError is raised.'''
  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[
      StratifyPass(),
      SplitPass(),
      SemiNaivePass(),
      PlanPass(),
      TempIndexPass(),
    ],
  )
  assert out.hir is not None


def test_temp_index_pass_spec_pipeline_order_validates():
  '''Spec ordering: [StratifyPass, SplitPass, SemiNaivePass, PlanPass,
  IndexSelectionPass, TempIndexPass]. When the D-Index wrapper is
  importable, the exact spec pipeline validates and runs end-to-end.
  When not, the same shape minus IndexSelectionPass is exercised by
  `test_temp_index_pass_preflight_ordering_minimum` and this test
  is a no-op skip.'''
  if not _HAS_INDEX_PASS:
    import pytest

    pytest.skip('IndexSelectionPass wrapper not importable yet (D-Index in flight)')

  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[
      StratifyPass(),
      SplitPass(),
      SemiNaivePass(),
      PlanPass(),
      _WrapperIndexSelectionPass(),
      TempIndexPass(),
    ],
  )
  assert out.hir is not None


# -----------------------------------------------------------------------------
# Behavior parity with the legacy `TempIndexRegistrationPass`
# -----------------------------------------------------------------------------


def test_temp_index_pass_matches_legacy_direct_call_tc():
  '''Run TempIndexPass end-to-end and compare to the legacy direct
  chain on the TC fixture. Both must produce equal HirPrograms.

  TC's recursive body is 2 atoms (no split), so TempIndexPass is a
  no-op pass-through here — but the wrapper must still preserve the
  legacy contract.'''
  prog = _tc_program()

  if _HAS_INDEX_PASS:
    pipeline = [
      StratifyPass(),
      SplitPass(),
      SemiNaivePass(),
      PlanPass(),
      _WrapperIndexSelectionPass(),
      TempIndexPass(),
    ]
    via_pipeline = Compiler().run(HirPlanState(program=prog), pipeline=pipeline).hir
  else:
    pre = _state_pre_temp_index(prog)
    via_pipeline = _compiler_with_hir().run(pre, pipeline=[TempIndexPass()]).hir

  via_legacy = _legacy_hir_through_temp_index(prog)
  assert via_pipeline is not None
  assert via_pipeline == via_legacy


def test_temp_index_pass_matches_legacy_direct_call_split_fixture():
  '''Same parity assertion on a 3-atom recursive rule, which forces
  the planner to introduce a temp relation — so TempIndexPass
  actually registers an index here (non-no-op path).'''
  prog = _three_atom_program()

  if _HAS_INDEX_PASS:
    pipeline = [
      StratifyPass(),
      SplitPass(),
      SemiNaivePass(),
      PlanPass(),
      _WrapperIndexSelectionPass(),
      TempIndexPass(),
    ]
    via_pipeline = Compiler().run(HirPlanState(program=prog), pipeline=pipeline).hir
  else:
    pre = _state_pre_temp_index(prog)
    via_pipeline = _compiler_with_hir().run(pre, pipeline=[TempIndexPass()]).hir

  via_legacy = _legacy_hir_through_temp_index(prog)
  assert via_pipeline is not None
  assert via_pipeline == via_legacy


def test_temp_index_pass_preserves_hir_identity():
  '''The legacy `TempIndexRegistrationPass.run` mutates HIR in place
  and returns the same instance. `TempIndexPass` must preserve this
  — post-TempIndexPass, the through-state's `hir` is identity-equal
  to its pre-TempIndexPass `hir` (no defensive copy).'''
  prog = _three_atom_program()
  pre = _state_pre_temp_index(prog)

  post = _compiler_with_hir().run(pre, pipeline=[TempIndexPass()])
  assert post.hir is pre.hir


def test_temp_index_pass_alone_runs_on_pre_indexed_state():
  '''TempIndexPass can run in isolation given a state with `hir`
  already populated (and indices selected), matching the legacy
  composable pattern.'''
  prog = _three_atom_program()
  pre = _state_pre_temp_index(prog)

  out = _compiler_with_hir().run(dataclasses.replace(pre), pipeline=[TempIndexPass()])

  via_legacy = _legacy_hir_through_temp_index(prog)
  assert out.hir == via_legacy


def test_temp_index_pass_requires_hir_in_state():
  '''TempIndexPass declares `consumes=('hir',)`. Calling apply with a
  state where `hir is None` raises (per the assertion in `_fn`).'''
  import pytest

  prog = _tc_program()
  state = HirPlanState(program=prog)  # no hir
  with pytest.raises(AssertionError):
    TempIndexPass().apply(state, None)


if __name__ == '__main__':
  test_temp_index_pass_is_a_program_pass()
  test_temp_index_pass_preflight_ordering_minimum()
  test_temp_index_pass_spec_pipeline_order_validates()
  test_temp_index_pass_matches_legacy_direct_call_tc()
  test_temp_index_pass_matches_legacy_direct_call_split_fixture()
  test_temp_index_pass_preserves_hir_identity()
  test_temp_index_pass_alone_runs_on_pre_indexed_state()
  test_temp_index_pass_requires_hir_in_state()
  print('OK')
