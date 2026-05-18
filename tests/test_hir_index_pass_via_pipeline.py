'''D-Index acceptance — wrap-via-pipeline matches legacy direct call.

Spec: `docs/phase_d_hir_passes.md` section 3.2 (per-PR acceptance
gate). `IndexSelectionPass` (in
`srdatalog.ir.dialects.hir.passes.index`) wraps
`srdatalog.ir.hir.index.IndexSelectionPass`. Run via
`Compiler.run(state, pipeline=[StratifyPass(), SplitPass(),
SemiNaivePass(), PlanPass(), IndexSelectionPass()])`, the resulting
`HirProgram` must equal the legacy direct call chain
`IndexSelectionPass().run(plan_joins(temp_rel(semi_naive(stratify(...)))))`
on the same fixture.

Index selection reads per-variant `access_patterns` /
`negation_patterns` populated by `PlanPass` on top of
`SemiNaivePass`-generated variants, so the full HIR pipeline through
`PlanPass` must run first.

A TC fixture (recursive Path rule) exercises the delta-version index
code path; the smoke test confirms that `required_indices`,
`canonical_index`, and `global_index_map` are actually populated.
'''

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pass, ProgramPass
from srdatalog.ir.dialects.hir.passes import (
  HirPlanState,
  IndexSelectionPass,
  PlanPass,
  SemiNaivePass,
  SplitPass,
  StratifyPass,
)
from srdatalog.ir.hir import DIALECT as HIR_DIALECT
from srdatalog.ir.hir import compile_to_hir
from srdatalog.ir.hir.index import IndexSelectionPass as _LegacyIndex
from srdatalog.ir.hir.pass_ import program_to_decls
from srdatalog.ir.hir.plan import plan_joins as _legacy_plan_joins
from srdatalog.ir.hir.semi_naive import SemiNaiveVariantPass as _LegacySemiNaive
from srdatalog.ir.hir.split import TempRelSynthesisPass as _LegacyTempRelSynth
from srdatalog.ir.hir.stratify import stratify as _legacy_stratify


def _compiler_with_hir() -> Compiler:
  '''Compiler with the `hir` dialect registered.

  Needed for pipelines that include `IndexSelectionPass` (or any other
  Wave 2B HIR transform) without a preceding `StratifyPass`: the
  pre-flight ordering check in `Compiler.run` requires the consumed
  `'hir'` dialect to be either registered OR produced by an earlier
  pass.'''
  c = Compiler()
  c.register_dialect(HIR_DIALECT)
  return c


def _tc_program() -> Program:
  '''TC fixture: recursive Path rule. Mirrors `_tc_program` in
  `test_hir_semi_naive_pass_via_pipeline.py` /
  `test_hir_plan_pass_via_pipeline.py`.'''
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


def _legacy_hir_through_index(prog: Program):
  '''Drive the legacy HIR chain up to (and including) index selection.
  Mirrors the imperative order from `default_pipeline` in
  `srdatalog.ir.hir.__init__`, restricted to the HIR transforms that
  index selection depends on: stratify -> SemiNaive variants -> split
  (temp-rel synthesis) -> plan -> index. Returns the post-index
  HirProgram.'''
  rules = list(prog.rules)
  decls = program_to_decls(prog)
  hir = _legacy_stratify(rules, decls)
  hir = _LegacySemiNaive().run(hir)
  hir = _LegacyTempRelSynth().run(hir)
  hir = _legacy_plan_joins(hir)
  hir = _LegacyIndex().run(hir)
  return hir


# -----------------------------------------------------------------------------
# Pass-shape sanity (matches the framework's invariants)
# -----------------------------------------------------------------------------


def test_index_pass_is_a_program_pass():
  ip = IndexSelectionPass()
  assert isinstance(ip, Pass)
  assert isinstance(ip, ProgramPass)
  assert ip.name == 'index_selection'
  assert ip.consumes == ('hir',)
  assert ip.produces == ('hir',)


# -----------------------------------------------------------------------------
# Pre-flight ordering — the spec's exact pipeline order
# -----------------------------------------------------------------------------


def test_index_pass_spec_pipeline_order_validates_and_runs():
  '''Spec ordering: [StratifyPass, SplitPass, SemiNaivePass, PlanPass,
  IndexSelectionPass]. The full HIR pipeline so far validates against
  `Compiler.run`'s pre-flight ordering check and runs end-to-end.'''
  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[
      StratifyPass(),
      SplitPass(),
      SemiNaivePass(),
      PlanPass(),
      IndexSelectionPass(),
    ],
  )
  assert out.hir is not None
  assert len(out.hir.strata) > 0


# -----------------------------------------------------------------------------
# Behavior parity with the legacy `IndexSelectionPass`
# -----------------------------------------------------------------------------


def test_index_pass_matches_legacy_direct_call():
  '''The full HIR pipeline through `IndexSelectionPass` produces the
  same HIR as the legacy direct call chain on the same fixture.'''
  prog = _tc_program()

  via_pipeline = (
    Compiler()
    .run(
      HirPlanState(program=prog),
      pipeline=[
        StratifyPass(),
        SplitPass(),
        SemiNaivePass(),
        PlanPass(),
        IndexSelectionPass(),
      ],
    )
    .hir
  )
  via_legacy = _legacy_hir_through_index(prog)

  assert via_pipeline is not None
  assert via_pipeline == via_legacy


def test_index_pass_matches_compile_to_hir():
  '''The full HIR pipeline through `IndexSelectionPass` produces an
  HIR whose index-selection outputs (required_indices, canonical_index,
  global_index_map) match `compile_to_hir`'s — `compile_to_hir` runs
  the same passes plus `TempIndexRegistrationPass` (which doesn't
  touch the index-selection fields), so the index-selection fields
  must agree.'''
  prog = _tc_program()

  via_pipeline = (
    Compiler()
    .run(
      HirPlanState(program=prog),
      pipeline=[
        StratifyPass(),
        SplitPass(),
        SemiNaivePass(),
        PlanPass(),
        IndexSelectionPass(),
      ],
    )
    .hir
  )
  via_compile_to_hir = compile_to_hir(prog)

  assert via_pipeline is not None
  assert via_pipeline.global_index_map == via_compile_to_hir.global_index_map
  assert len(via_pipeline.strata) == len(via_compile_to_hir.strata)
  for s_pipe, s_ref in zip(via_pipeline.strata, via_compile_to_hir.strata):
    assert s_pipe.required_indices == s_ref.required_indices
    assert s_pipe.canonical_index == s_ref.canonical_index


def test_index_pass_preserves_hir_identity():
  '''The legacy `IndexSelectionPass.run` mutates HIR in place and
  returns the same instance. `IndexSelectionPass` must preserve this —
  post-IndexSelectionPass, the through-state's `hir` is identity-equal
  to its pre-IndexSelectionPass `hir` (no defensive copy).'''
  prog = _tc_program()
  pre = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[StratifyPass(), SplitPass(), SemiNaivePass(), PlanPass()],
  )
  assert pre.hir is not None

  post = _compiler_with_hir().run(pre, pipeline=[IndexSelectionPass()])
  assert post.hir is pre.hir


def test_index_pass_alone_runs_on_pre_planned_state():
  '''IndexSelectionPass can run in isolation given a state with `hir`
  already planned (variants generated + join-planned), matching the
  legacy composable pattern.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)
  hir = _legacy_plan_joins(
    _LegacyTempRelSynth().run(_LegacySemiNaive().run(_legacy_stratify(rules, decls)))
  )

  state = HirPlanState(rules=rules, decls=decls, hir=hir)
  out = _compiler_with_hir().run(state, pipeline=[IndexSelectionPass()])

  via_legacy = _legacy_hir_through_index(prog)
  assert out.hir == via_legacy


def test_index_pass_requires_hir_in_state():
  '''IndexSelectionPass declares `consumes=('hir',)`. Calling apply
  with a state where `hir is None` raises (per the assertion in `_fn`).'''
  import pytest

  prog = _tc_program()
  state = HirPlanState(program=prog)  # no hir
  with pytest.raises(AssertionError):
    IndexSelectionPass().apply(state, None)


# -----------------------------------------------------------------------------
# Smoke: indices actually get populated on the TC fixture
# -----------------------------------------------------------------------------


def test_index_pass_populates_indices_on_tc_fixture():
  '''Smoke: after the spec pipeline, the TC fixture's HIR has
  `global_index_map`, per-stratum `required_indices`, and per-stratum
  `canonical_index` all populated (non-empty for every relation the
  fixture mentions).'''
  prog = _tc_program()
  out = Compiler().run(
    HirPlanState(program=prog),
    pipeline=[
      StratifyPass(),
      SplitPass(),
      SemiNaivePass(),
      PlanPass(),
      IndexSelectionPass(),
    ],
  )
  assert out.hir is not None

  assert out.hir.global_index_map, 'global_index_map must be populated by IndexSelectionPass'
  for rel_name, idxs in out.hir.global_index_map.items():
    assert idxs, f'global_index_map[{rel_name!r}] must contain at least one index'

  for stratum in out.hir.strata:
    assert stratum.required_indices, (
      f'stratum {stratum.scc_members!r} must have required_indices populated'
    )
    assert stratum.canonical_index, (
      f'stratum {stratum.scc_members!r} must have canonical_index populated'
    )
    for rel_name in stratum.scc_members:
      assert rel_name in stratum.required_indices
      assert rel_name in stratum.canonical_index
      assert stratum.required_indices[rel_name], f'required_indices[{rel_name!r}] must be non-empty'
      assert stratum.canonical_index[rel_name], f'canonical_index[{rel_name!r}] must be non-empty'


if __name__ == '__main__':
  test_index_pass_is_a_program_pass()
  test_index_pass_spec_pipeline_order_validates_and_runs()
  test_index_pass_matches_legacy_direct_call()
  test_index_pass_matches_compile_to_hir()
  test_index_pass_preserves_hir_identity()
  test_index_pass_alone_runs_on_pre_planned_state()
  test_index_pass_requires_hir_in_state()
  test_index_pass_populates_indices_on_tc_fixture()
  print('OK')
