'''User-specified plan tests.

Exercises `Rule.with_plan` / `Rule.with_plans`: the planner must honor
user-provided var_order / clause_order / pragma flags rather than
running the default heuristic.
'''
import json
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program, PlanEntry
from srdatalog.hir_types import Version
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir_emit import hir_to_obj
from srdatalog.hir_plan import derive_clause_order_from_var_order
from srdatalog.mir_emit import print_mir_sexpr


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_p_with_plan() -> Program:
  '''Mirror fixtures/tc_with_plan.nim — Base with var_order=[y,x] and a
  dual-delta Compose rule with an explicit plan list.
  '''
  X, Y, Z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2)
  p = Relation("P", 2)
  base = (p(X, Y) <= seed(X, Y)).named("Base").with_plan(var_order=["y", "x"])
  compose = (
    (p(X, Z) <= p(X, Y) & p(Y, Z))
    .named("Compose")
    .with_plans([
      PlanEntry(delta=0, clause_order=(0, 1), var_order=("x", "y", "z")),
      PlanEntry(delta=1, clause_order=(1, 0), var_order=("z", "y", "x")),
    ])
  )
  return Program(relations=[seed, p], rules=[base, compose])


# -----------------------------------------------------------------------------
# Structural checks on the planned variants
# -----------------------------------------------------------------------------

def test_user_var_order_applied_to_base_variant():
  hir = compile_to_hir(build_p_with_plan())
  s0 = hir.strata[0]
  v = s0.base_variants[0]
  assert v.var_order == ["y", "x"]
  assert v.clause_order == [0]              # derived from var_order


def test_user_full_plan_applied_to_both_delta_variants():
  hir = compile_to_hir(build_p_with_plan())
  s1 = hir.strata[1]
  rec = s1.recursive_variants
  assert len(rec) == 2

  v0 = next(v for v in rec if v.delta_idx == 0)
  assert v0.clause_order == [0, 1]
  assert v0.var_order == ["x", "y", "z"]

  v1 = next(v for v in rec if v.delta_idx == 1)
  assert v1.clause_order == [1, 0]
  assert v1.var_order == ["z", "y", "x"]


def test_access_pattern_respects_user_var_order():
  '''With var_order=[y,x], the Seed(x,y) access has accessOrder=[y,x] and
  indexCols=[1,0] (y at col 1, x at col 0).
  '''
  hir = compile_to_hir(build_p_with_plan())
  ap = hir.strata[0].base_variants[0].access_patterns[0]
  assert ap.rel_name == "Seed"
  assert ap.access_order == ["y", "x"]
  assert ap.index_cols == [1, 0]
  assert ap.prefix_len == 0


def test_count_pragma_propagated_to_variants():
  X, Y = Var("x"), Var("y")
  seed = Relation("Seed", 2)
  p = Relation("P", 2)
  prog = Program(
    relations=[seed, p],
    rules=[
      (p(X, Y) <= seed(X, Y)).named("Base").with_count(),
    ],
  )
  hir = compile_to_hir(prog)
  assert hir.strata[0].base_variants[0].count is True


def test_plan_pragma_flags_propagate_to_variant():
  X, Y = Var("x"), Var("y")
  seed = Relation("Seed", 2)
  p = Relation("P", 2)
  prog = Program(
    relations=[seed, p],
    rules=[
      (p(X, Y) <= seed(X, Y))
        .named("Base")
        .with_plan(var_order=["y", "x"], block_group=True, fanout=True),
    ],
  )
  hir = compile_to_hir(prog)
  v = hir.strata[0].base_variants[0]
  assert v.block_group is True
  assert v.fanout is True
  assert v.work_stealing is False           # default
  assert v.dedup_hash is False              # default


# -----------------------------------------------------------------------------
# derive_clause_order_from_var_order unit tests
# -----------------------------------------------------------------------------

def test_derive_clause_order_simple():
  '''For a body with [Seed(x,y)] and var_order=[y,x], clause 0 introduces
  y (and x) -> scheduled first; trivially [0].
  '''
  X, Y = Var("x"), Var("y")
  seed = Relation("Seed", 2)
  p = Relation("P", 2)
  r = p(X, Y) <= seed(X, Y)
  assert derive_clause_order_from_var_order(r, ["y", "x"]) == [0]


def test_derive_clause_order_two_clauses_picks_by_var_order():
  '''Body=[A(x), B(y)]; var_order=[y,x] should schedule B(y) first, then A(x).'''
  X, Y = Var("x"), Var("y")
  a = Relation("A", 1)
  b = Relation("B", 1)
  out = Relation("Out", 2)
  r = (out(X, Y) <= a(X) & b(Y))
  assert derive_clause_order_from_var_order(r, ["y", "x"]) == [1, 0]


# -----------------------------------------------------------------------------
# End-to-end: DSL -> HIR JSON matches Nim fixture; MIR S-expr matches too.
# -----------------------------------------------------------------------------

def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_user_plan_hir_byte_match():
  hir = compile_to_hir(build_p_with_plan())
  actual = hir_to_obj(hir)
  golden = json.loads((FIXTURES / "tc_with_plan.hir.json").read_text())
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


def test_user_plan_mir_byte_match():
  mir_prog = compile_to_mir(build_p_with_plan())
  actual = print_mir_sexpr(mir_prog)
  golden = (FIXTURES / "tc_with_plan.mir.sexpr").read_text().rstrip("\n")
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
    test_user_var_order_applied_to_base_variant,
    test_user_full_plan_applied_to_both_delta_variants,
    test_access_pattern_respects_user_var_order,
    test_count_pragma_propagated_to_variants,
    test_plan_pragma_flags_propagate_to_variant,
    test_derive_clause_order_simple,
    test_derive_clause_order_two_clauses_picks_by_var_order,
    test_user_plan_hir_byte_match,
    test_user_plan_mir_byte_match,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
