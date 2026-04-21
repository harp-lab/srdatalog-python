'''andersen.nim -- multi-variant plans, delta/clause_order/var_order triples'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import PlanEntry, Program, Relation, Var


def build_andersen() -> Program:
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
      (pt(Y, X) <= pt(Z, X) & asgn(Y, Z))
      .named("Assign")
      .with_plan(
        delta=0,
        clause_order=[0, 1],
        var_order=["z", "x", "y"],
      ),
      (pt(Y, W) <= pt(X, Z) & load_r(Y, X) & pt(Z, W))
      .named("Load")
      .with_plans(
        [
          PlanEntry(delta=0, clause_order=(0, 1, 2), var_order=("x", "z", "y", "w")),
          PlanEntry(delta=2, clause_order=(2, 0, 1), var_order=("z", "x", "y", "w")),
        ]
      ),
      (pt(Z, W) <= pt(Y, Z) & store_r(Y, X) & pt(X, W))
      .named("Store")
      .with_plans(
        [
          PlanEntry(delta=0, clause_order=(0, 1, 2), var_order=("y", "x", "z", "w")),
          PlanEntry(delta=2, clause_order=(2, 1, 0), var_order=("x", "y", "w", "z")),
        ]
      )
      .with_inject_cpp("""
      auto& points_to_delta = get_relation_by_schema<PointsTo, DELTA_VER>(db);
      auto& points_to_delta_idx = points_to_delta.get_index({{0,1}});
      auto& points_to_full = get_relation_by_schema<PointsTo, FULL_VER>(db);
      auto& points_to_full_idx = points_to_full.get_index({{0,1}});
      std::cout << "  PointsTo delta: " << points_to_delta_idx.root().degree()
                << ", full: " << points_to_full_idx.root().degree() << std::endl;
    """),
    ],
  )


def test_andersen_hir():
  diff_hir(build_andersen(), "andersen")


def test_andersen_mir():
  diff_mir(build_andersen(), "andersen")


if __name__ == "__main__":
  test_andersen_hir()
  test_andersen_mir()
  print("andersen: OK")
