'''triangle.nim -- ZRel(x,y,z) :- RRel(x,y), SRel(y,z,h), TRel(z,x,f)'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import Program, Relation, Var


def build_triangle() -> Program:
  X, Y, Z, H, F = Var("x"), Var("y"), Var("z"), Var("h"), Var("f")
  rr = Relation("RRel", 2)
  sr = Relation("SRel", 3)
  tr = Relation("TRel", 3)
  zr = Relation("ZRel", 3)
  return Program(
    rules=[
      (zr(X, Y, Z) <= rr(X, Y) & sr(Y, Z, H) & tr(Z, X, F))
      .named("Triangle")
      .with_plan(var_order=["x", "y", "z"]),
    ],
  )


def test_triangle_hir():
  diff_hir(build_triangle(), "triangle")


def test_triangle_mir():
  diff_mir(build_triangle(), "triangle")


if __name__ == "__main__":
  test_triangle_hir()
  test_triangle_mir()
  print("triangle: OK")
