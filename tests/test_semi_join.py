'''Semi-join optimization tests. Byte-diffs against the Nim fixture and
exercises the pass logic directly.
'''

import json
from pathlib import Path

from srdatalog.dsl import Atom, Program, Relation, Var
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir.emit import hir_to_obj
from srdatalog.mir.emit import print_mir_sexpr
from srdatalog.provenance import ProvenanceKind
from srdatalog.rule_rewrite import _is_semi_join_candidate, optimize_semi_joins

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_semi_join_program() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  r = Relation("R", 3)
  s = Relation("S", 1)
  t = Relation("T", 2)
  main = Relation("Main", 2)
  rule = (main(X, Z) <= r(X, Y, Z) & s(Y) & t(X, Z)).named("SJTest").with_semi_join()
  return Program(relations=[r, s, t, main], rules=[rule])


# -----------------------------------------------------------------------------
# Unit tests on the core helpers
# -----------------------------------------------------------------------------


def test_is_semi_join_candidate_true_for_subset():
  X, Y = Var("x"), Var("y")
  r = Relation("R", 2)
  s = Relation("S", 1)
  assert _is_semi_join_candidate(s(X), r(X, Y)) is True


def test_is_semi_join_candidate_false_for_equal_size():
  X, Y = Var("x"), Var("y")
  r = Relation("R", 2)
  # Same size -> not a filter, even if subset.
  assert _is_semi_join_candidate(r(X, Y), r(X, Y)) is False


def test_is_semi_join_candidate_false_for_disjoint_vars():
  X, Y = Var("x"), Var("y")
  s = Relation("S", 1)
  r = Relation("R", 2)
  assert _is_semi_join_candidate(s(X), r(Y, Y)) is False


def test_optimize_semi_joins_skips_when_semi_join_false():
  '''Rule without .with_semi_join() is left alone.'''
  prog = build_semi_join_program()
  # Build a version without the pragma
  rule_no_sj = prog.rules[0]
  from dataclasses import replace

  rule_no_sj = replace(rule_no_sj, semi_join=False)
  out_rules, out_decls = optimize_semi_joins([rule_no_sj], [])
  assert out_rules == [rule_no_sj]


def test_optimize_semi_joins_skips_rules_with_two_or_fewer_clauses():
  X, Y = Var("x"), Var("y")
  a = Relation("A", 2)
  b = Relation("B", 1)
  p = Relation("P", 2)
  r = (p(X, Y) <= a(X, Y) & b(X)).named("R").with_semi_join()
  out_rules, _ = optimize_semi_joins([r], [])
  assert out_rules == [r]


def test_optimize_semi_joins_generates_expected_rel_and_prov():
  rules, decls = optimize_semi_joins(
    list(build_semi_join_program().rules),
    [],
  )
  # Output: [_SJ_R_S_0_2_Gen, SJTest (rewritten)]
  assert len(rules) == 2
  gen, main = rules
  assert gen.name == "_SJ_R_S_0_2_Gen"
  assert gen.is_generated is True
  assert gen.prov.kind is ProvenanceKind.COMPILER_GEN
  assert gen.prov.parent_rule == "SJTest"
  assert gen.prov.derived_from == "R"
  assert gen.prov.transform_pass == "semi_join"
  # Main's body should now be [_SJ_R_S_0_2(x, z), T(x, z)].
  assert len(main.body) == 2
  first, second = main.body
  assert isinstance(first, Atom)
  assert first.rel == "_SJ_R_S_0_2"
  assert first.prov.kind is ProvenanceKind.COMPILER_GEN
  assert isinstance(second, Atom) and second.rel == "T"


# -----------------------------------------------------------------------------
# End-to-end byte-match against the Nim fixture
# -----------------------------------------------------------------------------


def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_semi_join_hir_byte_match():
  hir = compile_to_hir(build_semi_join_program())
  actual = hir_to_obj(hir)
  golden = json.loads((FIXTURES / "semi_join.hir.json").read_text())
  golden.pop("hirSExpr", None)
  if _canonical(actual) != _canonical(golden):
    import difflib

    diff = "\n".join(
      difflib.unified_diff(
        _canonical(golden).splitlines(),
        _canonical(actual).splitlines(),
        fromfile="nim-golden",
        tofile="python",
        lineterm="",
      )
    )
    raise AssertionError("HIR mismatch:\n" + diff)


def test_semi_join_mir_byte_match():
  mir_prog = compile_to_mir(build_semi_join_program())
  actual = print_mir_sexpr(mir_prog)
  golden = (FIXTURES / "semi_join.mir.sexpr").read_text().rstrip("\n")
  if actual != golden:
    import difflib

    diff = "\n".join(
      difflib.unified_diff(
        golden.splitlines(),
        actual.splitlines(),
        fromfile="nim-golden",
        tofile="python",
        lineterm="",
      )
    )
    raise AssertionError("MIR mismatch:\n" + diff)


if __name__ == "__main__":
  tests = [
    test_is_semi_join_candidate_true_for_subset,
    test_is_semi_join_candidate_false_for_equal_size,
    test_is_semi_join_candidate_false_for_disjoint_vars,
    test_optimize_semi_joins_skips_when_semi_join_false,
    test_optimize_semi_joins_skips_rules_with_two_or_fewer_clauses,
    test_optimize_semi_joins_generates_expected_rel_and_prov,
    test_semi_join_hir_byte_match,
    test_semi_join_mir_byte_match,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
