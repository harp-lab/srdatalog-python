'''Aggregate end-to-end tests.

The Nim HIR pipeline parses AggClause into HIR JSON (kind="aggregation")
but never constructs a moAggregate MirNode from it. Python mirrors that:
DSL `agg(...)` round-trips through HIR but disappears from MIR. Both the
HIR and MIR outputs byte-match the Nim golden.
'''

import json
from pathlib import Path

from srdatalog.dsl import Agg, Program, Relation, Var, agg, count
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir.emit import hir_to_obj
from srdatalog.mir.emit import print_mir_sexpr

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_agg_program() -> Program:
  X, Y, CNT = Var("x"), Var("y"), Var("cnt")
  r = Relation("R", 2)
  counts = Relation("Counts", 1)
  rule = (counts(CNT) <= count(CNT, r(X, Y))).named("CountR")
  return Program(rules=[rule])


# -----------------------------------------------------------------------------
# Unit tests on the DSL helper + analysis plumbing
# -----------------------------------------------------------------------------


def test_count_helper_builds_agg_clause():
  CNT, X, Y = Var("cnt"), Var("x"), Var("y")
  r = Relation("R", 2)
  clause = count(CNT, r(X, Y))
  assert isinstance(clause, Agg)
  assert clause.result_var == "cnt"
  assert clause.func == "count"
  assert clause.rel == "R"
  assert [a.var_name for a in clause.args] == ["x", "y"]
  assert clause.cpp_type == ""


def test_agg_helper_accepts_string_result_var():
  X, Y = Var("x"), Var("y")
  r = Relation("R", 2)
  clause = agg("cnt", "count", r(X, Y))
  assert clause.result_var == "cnt"


def test_agg_helper_cpp_type_threads_through():
  CNT, X = Var("cnt"), Var("x")
  r = Relation("R", 1)
  clause = agg(CNT, "agg", r(X), cpp_type="MyAgg<int>")
  assert clause.func == "agg"
  assert clause.cpp_type == "MyAgg<int>"


def test_analyze_rule_counts_agg_args_and_result_var_as_positive():
  from srdatalog.hir.plan import analyze_rule

  hir = compile_to_hir(build_agg_program())
  # Variant's analysis should treat x, y, cnt all as vars.
  rule = hir.strata[0].base_variants[0].original_rule
  analysis = analyze_rule(rule)
  assert analysis.vars == {"x", "y", "cnt"}
  # No join vars (only one clause).
  assert analysis.join_vars == set()
  # Head vars are {cnt}.
  assert analysis.head_vars == {"cnt"}


def test_var_order_puts_agg_args_then_result_var():
  '''Nim's extractClauseVars for AggClause returns relation args first,
  then the result_var. var_order therefore emerges as [x, y, cnt].
  '''
  hir = compile_to_hir(build_agg_program())
  v = hir.strata[0].base_variants[0]
  assert v.var_order == ["x", "y", "cnt"]
  assert v.clause_order == [0]
  # Agg produces no access pattern (like Nim's else branch).
  assert v.access_patterns == []


# -----------------------------------------------------------------------------
# End-to-end byte-match against Nim fixture
# -----------------------------------------------------------------------------


def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_agg_hir_byte_match():
  hir = compile_to_hir(build_agg_program())
  actual = hir_to_obj(hir)
  golden = json.loads((FIXTURES / "agg_test.hir.json").read_text())
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


def test_agg_mir_byte_match():
  '''MIR loses the aggregate (both Nim and Python drop AggClause during
  lowering). Pipeline is just (insert-into :schema Counts ...).
  '''
  mir_prog = compile_to_mir(build_agg_program())
  actual = print_mir_sexpr(mir_prog)
  golden = (FIXTURES / "agg_test.mir.sexpr").read_text().rstrip("\n")
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
    test_count_helper_builds_agg_clause,
    test_agg_helper_accepts_string_result_var,
    test_agg_helper_cpp_type_threads_through,
    test_analyze_rule_counts_agg_args_and_result_var_as_positive,
    test_var_order_puts_agg_args_then_result_var,
    test_agg_hir_byte_match,
    test_agg_mir_byte_match,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
