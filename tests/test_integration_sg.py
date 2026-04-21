'''sg.nim -- same-generation with filter + recursive case'''
from integration_helpers import diff_hir, diff_mir
from srdatalog.dsl import Var, Relation, Program, Filter


def build_sg() -> Program:
  P, Q, X, Y = Var("p"), Var("q"), Var("x"), Var("y")
  arc = Relation("Arc", 2)
  sg = Relation("Sg", 2)
  base = (
    (sg(X, Y) <= arc(P, X) & arc(P, Y) & Filter(vars=("x", "y"), code="return x != y;"))
    .named("SGBase")
    .with_plan(var_order=["p", "x", "y"])
  )
  rec = (
    (sg(X, Y) <= arc(P, X) & sg(P, Q) & arc(Q, Y))
    .named("SGRec")
    .with_plan(var_order=["p", "q", "x", "y"])
  )
  return Program(relations=[arc, sg], rules=[base, rec])


def test_sg_hir():
  diff_hir(build_sg(), "sg")


def test_sg_mir():
  diff_mir(build_sg(), "sg")


if __name__ == "__main__":
  test_sg_hir()
  test_sg_mir()
  print("sg: OK")
