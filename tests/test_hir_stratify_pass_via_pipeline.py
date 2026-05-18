'''D-Stratify acceptance — wrap-via-pipeline matches legacy direct call.

Spec: `docs/phase_d_hir_passes.md` section 3.2 (per-PR acceptance
gate). `StratifyPass` (in `srdatalog.ir.dialects.hir.passes.stratify`)
runs via `Compiler.run(state, pipeline=[StratifyPass()])` and the
resulting `HirProgram` must match what the legacy
`srdatalog.ir.hir.stratify.stratify(rules, decls)` produces directly
on the same fixture.

`compile_to_hir` runs the *full* HIR pipeline (stratify + rule
rewrites + variant gen + plan + ...); a single-pass run cannot match
its output. So this test compares against the legacy `stratify`
function directly — the contract the wrapper actually wraps.
'''

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pass, ProgramPass
from srdatalog.ir.dialects.hir.passes import HirPlanState, StratifyPass
from srdatalog.ir.hir.pass_ import program_to_decls
from srdatalog.ir.hir.stratify import stratify as _legacy_stratify
from srdatalog.ir.hir.types import HirProgram


def _tc_program() -> Program:
  '''Tiny TC-style program. Mirrors `_tc_program` in
  `test_default_pipelines_shims.py` / `test_f5_acceptance.py`.'''
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


def test_stratify_pass_is_a_program_pass():
  sp = StratifyPass()
  assert isinstance(sp, Pass)
  assert isinstance(sp, ProgramPass)
  assert sp.name == 'stratify'
  assert sp.consumes == ()
  assert sp.produces == ('hir',)


# -----------------------------------------------------------------------------
# Behavior parity with the legacy `stratify`
# -----------------------------------------------------------------------------


def test_stratify_pass_matches_legacy_direct_call():
  '''StratifyPass run via Compiler.run produces a HirProgram equal to
  `stratify(rules, decls)` on the same fixture.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)

  via_pipeline = Compiler().run(HirPlanState(program=prog), pipeline=[StratifyPass()]).hir
  via_legacy = _legacy_stratify(rules, decls)

  assert isinstance(via_pipeline, HirProgram)
  assert via_pipeline == via_legacy


def test_stratify_pass_accepts_explicit_rules_and_decls():
  '''Callers that already have `(rules, decls)` can pass them
  directly, bypassing the `Program -> (rules, decls)` translation.'''
  prog = _tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)

  state = HirPlanState(rules=rules, decls=decls)
  out = Compiler().run(state, pipeline=[StratifyPass()])
  via_legacy = _legacy_stratify(rules, decls)

  assert out.hir is not None
  assert out.hir == via_legacy


def test_stratify_pass_threads_rules_and_decls_through_state():
  '''Post-stratify, the through-state carries the (rules, decls) that
  were fed to the legacy function — downstream Wave 2B passes that
  want to re-derive HIR can rely on them.'''
  prog = _tc_program()
  out = Compiler().run(HirPlanState(program=prog), pipeline=[StratifyPass()])
  assert out.rules == list(prog.rules)
  assert [d.rel_name for d in out.decls] == [r.name for r in prog.relations]


def test_stratify_pass_idempotent_on_program_with_no_rules():
  '''Edge case: empty program. The legacy `stratify` returns a
  `HirProgram` with empty strata; the wrapper must preserve this.'''
  prog = Program(rules=[])
  out = Compiler().run(HirPlanState(program=prog), pipeline=[StratifyPass()])
  assert out.hir is not None
  assert out.hir.strata == []


if __name__ == '__main__':
  test_stratify_pass_is_a_program_pass()
  test_stratify_pass_matches_legacy_direct_call()
  test_stratify_pass_accepts_explicit_rules_and_decls()
  test_stratify_pass_threads_rules_and_decls_through_state()
  test_stratify_pass_idempotent_on_program_with_no_rules()
  print('OK')
