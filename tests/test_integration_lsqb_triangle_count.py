'''lsqb_triangle_count.nim -- Triangle count with `count: true`'''
from integration_helpers import diff_hir, diff_mir
from srdatalog.dsl import Var, Relation, Program


def build_lsqb_triangle_count() -> Program:
  X, Y, A, B, C = Var("x"), Var("y"), Var("a"), Var("b"), Var("c")
  k_in = Relation("KnowsInput", 2)
  knows = Relation("Knows", 2)
  tri = Relation("Triangle", 3)
  return Program(
    relations=[k_in, knows, tri],
    rules=[
      (knows(X, Y) <= k_in(X, Y)).named("KnowsLoad"),
      (
        tri(A, B, C) <= knows(A, B) & knows(B, C) & knows(A, C)
      ).named("TriangleJoin").with_plan(var_order=["a", "b", "c"]).with_count(),
    ],
  )


def test_lsqb_triangle_count_hir():
  diff_hir(build_lsqb_triangle_count(), "lsqb_triangle_count")


def test_lsqb_triangle_count_mir():
  diff_mir(build_lsqb_triangle_count(), "lsqb_triangle_count")


if __name__ == "__main__":
  test_lsqb_triangle_count_hir()
  test_lsqb_triangle_count_mir()
  print("lsqb_triangle_count: OK")
