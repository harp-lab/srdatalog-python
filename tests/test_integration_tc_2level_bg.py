'''tc_2level_bg.nim -- tc with block_group pragma on recursive variant'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import Program, Relation, Var


def build_tc_2level_bg() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  arc_in = Relation("ArcInput", 2)
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    relations=[arc_in, edge, path],
    rules=[
      (edge(X, Y) <= arc_in(X, Y)).named("EdgeLoad"),
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z))
      .named("TCRec")
      .with_plan(
        delta=0,
        block_group=True,
        var_order=["y", "x", "z"],
      ),
    ],
  )


def test_tc_2level_bg_hir():
  diff_hir(build_tc_2level_bg(), "tc_2level_bg")


def test_tc_2level_bg_mir():
  diff_mir(build_tc_2level_bg(), "tc_2level_bg")


if __name__ == "__main__":
  test_tc_2level_bg_hir()
  test_tc_2level_bg_mir()
  print("tc_2level_bg: OK")
