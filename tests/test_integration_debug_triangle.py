'''debug_triangle.nim -- simpler triangle, all arity-2 relations'''
from integration_helpers import diff_hir, diff_mir
from srdatalog.dsl import Var, Relation, Program


def build_debug_triangle() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  rr = Relation("RRel", 2)
  sr = Relation("SRel", 2)
  tr = Relation("TRel", 2)
  zr = Relation("ZRel", 3)
  return Program(
    relations=[rr, sr, tr, zr],
    rules=[
      (zr(X, Y, Z) <= rr(X, Y) & sr(Y, Z) & tr(Z, X))
        .named("TriangleDebug")
        .with_plan(var_order=["x", "y", "z"]),
    ],
  )


def test_debug_triangle_hir():
  diff_hir(build_debug_triangle(), "debug_triangle")


def test_debug_triangle_mir():
  diff_mir(build_debug_triangle(), "debug_triangle")


if __name__ == "__main__":
  test_debug_triangle_hir()
  test_debug_triangle_mir()
  print("debug_triangle: OK")
