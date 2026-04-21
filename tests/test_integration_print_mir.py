'''print_mir.nim -- same as andersen but no input pragmas, different Load plan'''
from integration_helpers import diff_hir, diff_mir
from srdatalog.dsl import Var, Relation, Program, PlanEntry


def build_print_mir() -> Program:
  X, Y, Z, W = Var("x"), Var("y"), Var("z"), Var("w")
  ao = Relation("AddressOf", 2)
  asgn = Relation("Assign", 2)
  load_r = Relation("Load", 2)
  store_r = Relation("Store", 2)
  pt = Relation("PointsTo", 2)
  return Program(
    relations=[ao, asgn, load_r, store_r, pt],
    rules=[
      (pt(Y, X) <= ao(Y, X)).named("Base").with_plan(var_order=["y", "x"]),
      (pt(Y, X) <= pt(Z, X) & asgn(Y, Z)).named("Assign").with_plan(
        delta=0, clause_order=[0, 1], var_order=["z", "x", "y"],
      ),
      (pt(Y, W) <= pt(X, Z) & load_r(Y, X) & pt(Z, W)).named("Load").with_plans([
        PlanEntry(delta=0, clause_order=(0, 1, 2), var_order=("x", "z", "y", "w")),
        PlanEntry(delta=2, clause_order=(2, 0, 1), var_order=("z", "w", "x", "y")),
      ]),
      (pt(Z, W) <= pt(Y, Z) & store_r(Y, X) & pt(X, W)).named("Store").with_plans([
        PlanEntry(delta=0, clause_order=(0, 1, 2), var_order=("y", "x", "z", "w")),
        PlanEntry(delta=2, clause_order=(2, 1, 0), var_order=("x", "y", "w", "z")),
      ]),
    ],
  )


def test_print_mir_hir():
  diff_hir(build_print_mir(), "print_mir")


def test_print_mir_mir():
  diff_mir(build_print_mir(), "print_mir")


if __name__ == "__main__":
  test_print_mir_hir()
  test_print_mir_mir()
  print("print_mir: OK")
